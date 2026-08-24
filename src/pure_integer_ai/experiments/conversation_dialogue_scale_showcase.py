"""公开完整句、长问句和广域来源问答的可复跑展示入口。

该入口只消费公开 pack 与 K 盘只读 Wikipedia SQLite，不写训练状态，也不把
检索命中或模板回放命名为自由生成。每一轮同时保留状态、来源身份和有限
hot-history 的整数 turn key；同一输入重复运行必须得到相同规范摘要。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Callable

from pure_integer_ai.experiments.conversation_broad_qa_runtime import (
    BroadDialogueState,
    DialogueTurn,
    answer_broad_dialogue_turn,
)
from pure_integer_ai.experiments.conversation_training_pack import (
    DialogueTrainingPack,
    load_dialogue_training_pack,
)
from pure_integer_ai.experiments.run_conversation_training import (
    default_course_paths,
)
from pure_integer_ai.experiments.conversation_public_sentence_demo import (
    build_public_sentence_demo_catalog,
    run_public_sentence_demo_bytes,
)
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_structure_learner import (
    SurfaceSemantic,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_sparse_qa_snapshot import (
    load_or_rebuild_public_sparse_qa_runtime,
)


_LONG_QUESTIONS = (
    "根据保满铁路条目的工程资料，保满铁路全长多少公里？",
    "从矮寨大桥的工程时间线看，矮寨大桥何时建成通车？",
    "根据维基数据条目中的结构化数据信息，维基数据的结构化数据采用什么许可发布？",
    "请只依据黄山松条目中关于地理分布的公开资料回答：黄山松分布在哪些地区？",
)
_NARROW_QUESTIONS = (
    "什么使得河水上涨？",
    "暴雨使得什么？",
    "河水上涨的原因是什么？",
)
_BROAD_QUESTIONS = (
    "保满铁路设有多少座车站？",
    "大写锁定键有什么作用？",
    "儋州西庆机场距离儋州市区多远？",
    "维基数据由哪个组织托管？",
    "火星上的矮寨大桥何时通车？",
)
_FOLLOWUP_QUESTIONS = (
    "它分布在哪些地区？",
    "请用一个完整句子说明该条目的地理分布。",
)


@dataclass(frozen=True, slots=True)
class DialogueScaleShowcaseTurn:
    """一轮展示的可回读值；不持有连接、索引或宿主对象。"""

    ordinal: int
    question: str
    question_utf8_bytes: int
    is_long_question: bool
    status: str
    answer: str | None
    evidence_answer: str | None
    answer_utf8_bytes: int
    source_title: str | None
    source_url: str | None
    turn_key_u8: tuple[int, ...]

    @classmethod
    def from_turn(cls, turn: DialogueTurn) -> "DialogueScaleShowcaseTurn":
        """把统一 dialogue turn 投影为稳定展示记录。"""
        question_bytes = turn.question.encode("utf-8")
        return cls(
            turn.ordinal,
            turn.question,
            len(question_bytes),
            len(question_bytes) >= 48,
            turn.status,
            turn.display_answer,
            turn.answer,
            len((turn.display_answer or "").encode("utf-8")),
            turn.source_title,
            turn.source_url,
            tuple(turn.turn_key),
        )

    def to_dict(self) -> dict[str, object]:
        """导出不含本机连接信息的规范 JSON 对象。"""
        return {
            "answer": self.answer,
            "evidence_answer": self.evidence_answer,
            "answer_utf8_bytes": self.answer_utf8_bytes,
            "is_long_question": self.is_long_question,
            "ordinal": self.ordinal,
            "question": self.question,
            "question_utf8_bytes": self.question_utf8_bytes,
            "source_title": self.source_title,
            "source_url": self.source_url,
            "status": self.status,
            "turn_key_u8": list(self.turn_key_u8),
        }


@dataclass(frozen=True, slots=True)
class TrainingObservation:
    """一次 K 盘 observe run 的只读整数状态摘要。"""

    run_id: str
    pack_sha256: str
    training_item_count: int
    heldout_probe_count: int
    stages_completed: tuple[int, ...]
    weaning_ready: bool
    graph_size: int
    concept_node_count: int
    edge_count: int
    occurrence_count: int
    database_name: str

    def to_dict(self) -> dict[str, object]:
        """导出不含本机绝对路径的训练观察记录。"""
        return {
            "concept_node_count": self.concept_node_count,
            "database_name": self.database_name,
            "edge_count": self.edge_count,
            "graph_size": self.graph_size,
            "heldout_probe_count": self.heldout_probe_count,
            "pack_sha256": self.pack_sha256,
            "occurrence_count": self.occurrence_count,
            "run_id": self.run_id,
            "stages_completed": list(self.stages_completed),
            "training_item_count": self.training_item_count,
            "weaning_ready": self.weaning_ready,
        }


def load_training_observation(
        run_root: str | Path,
        *,
        expected_pack_sha256: str | None = None,
        ) -> TrainingObservation:
    """只读加载 K 盘训练摘要和 SQLite 计数，并核验两者绑定。"""
    root = Path(run_root).resolve()
    if root.drive.upper() != "K:" or not root.is_dir():
        raise ValueError("training observation run root 必须是存在的 K 盘目录")
    summary_path = root / "training_summary.json"
    if not summary_path.is_file():
        raise ValueError("training observation 缺少 training_summary.json")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("training observation 摘要不可回读") from error
    required = {
        "run_id", "pack_sha256", "training_item_count", "heldout_probe_count",
        "stages_completed", "weaning_ready", "database",
    }
    if not isinstance(summary, dict) or not required <= set(summary):
        raise ValueError("training observation 摘要字段缺失")
    if (expected_pack_sha256 is not None
            and summary["pack_sha256"] != expected_pack_sha256):
        raise ValueError("training observation pack identity 漂移")
    database = Path(str(summary["database"])).resolve()
    try:
        database.relative_to(root)
    except ValueError as error:
        raise ValueError("training observation database 越出 run root") from error
    if not database.is_file():
        raise ValueError("training observation database 缺失")
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        tables = {item[0] for item in connection.execute(
            "select name from sqlite_master where type='table'")}
        required_tables = {"concept_node", "edge", "occurrence"}
        if not required_tables <= tables:
            raise ValueError("training observation SQLite 表缺失")
        concept_node_count = int(connection.execute(
            "select count(*) from concept_node").fetchone()[0])
        edge_count = int(connection.execute(
            "select count(*) from edge").fetchone()[0])
        occurrence_count = int(connection.execute(
            "select count(*) from occurrence").fetchone()[0])
    finally:
        connection.close()
    graph_size = int(summary.get("graph_size", concept_node_count))
    if graph_size != concept_node_count:
        raise ValueError("training observation graph_size 与 SQLite 漂移")
    stages = tuple(int(item) for item in summary["stages_completed"])
    return TrainingObservation(
        str(summary["run_id"]), str(summary["pack_sha256"]),
        int(summary["training_item_count"]), int(summary["heldout_probe_count"]),
        stages, bool(summary["weaning_ready"]), graph_size,
        concept_node_count, edge_count, occurrence_count, database.name,
    )


def showcase_questions() -> tuple[str, ...]:
    """返回固定的完整句、长问句、广域问句和拒答探针集合。"""
    # 追问紧跟黄山松长问句，验证来源焦点会跨轮保留；随后再运行广域拒答
    # 探针，避免 UNKNOWN 被前一轮的来源标题污染。
    return tuple(dict.fromkeys((
        *_NARROW_QUESTIONS,
        *_LONG_QUESTIONS,
        *_FOLLOWUP_QUESTIONS,
        *_BROAD_QUESTIONS,
    )))


def _pack_stats(pack: DialogueTrainingPack) -> dict[str, object]:
    """摘要公开课程真实表层规模，区分自然句与结构载体字节。"""
    surfaces = tuple(item.raw_text for item in pack.cases)
    lengths = tuple(len(value.encode("utf-8")) for value in surfaces)
    return {
        "case_count": len(pack.cases),
        "split_counts": [list(item) for item in pack.split_counts],
        "pack_sha256": pack.pack_sha256,
        "surface_utf8_bytes_max": max(lengths),
        "surface_utf8_bytes_total": sum(lengths),
        "surface_ge_48_bytes_count": sum(value >= 48 for value in lengths),
    }


def _trained_surface_typed_probes(trained_surface) -> tuple[dict[str, object], ...]:
    """用新值验证四类已学表层结构的真实消费。"""
    if trained_surface is None:
        return ()
    cases = (
        ("ANSWER", "polite", ("subject", "predicate", "qualifier", "object"),
         ("新入口", "启用时间", "审计记录", "2030年1月"),
         SurfaceSemantic("probe-answer", "fact", "新入口", "启用时间", "2030年1月")),
        ("UNKNOWN", "neutral", ("source", "scope"),
         ("当前", "青石台的运行预算"),
         SurfaceSemantic("probe-unknown", "unknown", "新对象", "新属性", "未提供")),
        ("CLARIFY", "polite", ("choice", "target"),
         ("甲区还是乙区", "数量"),
         SurfaceSemantic("probe-clarify", "scope", "新对象", "新属性", "待选区域")),
        ("REPAIR", "polite", ("acknowledge", "request"),
         ("前面的条件不够明确", "具体时间"),
         SurfaceSemantic("probe-repair", "repair", "先前问题", "需要", "完整限定")),
    )
    result = []
    for act, register, roles, values, semantic in cases:
        rendered = trained_surface.render_typed(
            semantic, response_act=act, register=register,
            ordered_roles=roles, slot_values=values,
            source_id="showcase-probe-source",
            context_id="showcase-probe-context",
            family_id="showcase-probe-family",
        )
        result.append({
            "response_act": act,
            "used": rendered.used,
            "surface": rendered.surface,
            "pattern_id": rendered.pattern_id,
            "reason": rendered.reason,
        })
    return tuple(result)


def _trained_surface_variant_probes(
        trained_surface,
        inputs: tuple[str, ...],
        ) -> tuple[dict[str, object], ...]:
    """只消费调用方显式提供的 G7 variant probe 输入。"""
    if trained_surface is None:
        if inputs:
            raise ValueError("variant probe input 需要绑定训练表层 runtime")
        return ()
    result = []
    for ordinal, value in enumerate(inputs):
        if not isinstance(value, str) or not value or value.strip() != value:
            raise ValueError("variant probe input 必须是规范非空字符串")
        rendered = trained_surface.render(
            value, response_act="ANSWER",
            source_title="showcase-variant-probe", ordinal=ordinal,
        )
        replay = trained_surface.render(
            value, response_act="ANSWER",
            source_title="showcase-variant-probe", ordinal=ordinal,
        )
        result.append({
            "input_surface": value,
            "output_surface": rendered.surface,
            "used": rendered.used,
            "reason": rendered.reason,
            "pattern_id": rendered.pattern_id,
            "trace_u": list(rendered.trace),
            "replay_bit_identical": rendered == replay,
        })
    return tuple(result)


def _trained_surface_order_probes(
        trained_surface,
        probe_values: tuple[str, ...],
        probe_roles: tuple[str, ...],
        ) -> tuple[dict[str, object], ...]:
    """消费调用方显式提供的 G9 typed role-order probe。"""
    if not probe_values and not probe_roles:
        return ()
    if trained_surface is None:
        raise ValueError("order probe 需要绑定训练表层 runtime")
    if (len(probe_values) != len(probe_roles)
            or not probe_values
            or len(set(probe_roles)) != len(probe_roles)):
        raise ValueError("order probe values/roles 必须一一对应且不重复")
    fields = {"subject": "runtime", "predicate": "runtime", "object": "runtime"}
    for role, value in zip(probe_roles, probe_values):
        if not isinstance(value, str) or not value or value.strip() != value:
            raise ValueError("order probe value 必须是规范非空字符串")
        if role in {"subject", "topic", "cause"}:
            fields["subject"] = value
        elif role in {"predicate", "relation"}:
            fields["predicate"] = value
        elif role in {"object", "claim", "effect"}:
            fields["object"] = value
    semantic = SurfaceSemantic(
        "showcase-order-probe", "state", fields["subject"],
        fields["predicate"], fields["object"],
    )
    rendered = trained_surface.render_order_typed(
        semantic, response_act="ANSWER", register="neutral",
        ordered_roles=probe_roles, slot_values=probe_values,
        source_id="showcase-order-probe", context_id="showcase-order-context",
        family_id="showcase-order-family",
    )
    replay = trained_surface.render_order_typed(
        semantic, response_act="ANSWER", register="neutral",
        ordered_roles=probe_roles, slot_values=probe_values,
        source_id="showcase-order-probe", context_id="showcase-order-context",
        family_id="showcase-order-family",
    )
    return ({
        "input_roles": list(probe_roles),
        "input_values": list(probe_values),
        "output_surface": rendered.surface,
        "used": rendered.used,
        "reason": rendered.reason,
        "pattern_id": rendered.pattern_id,
        "trace_u": list(rendered.trace),
        "replay_bit_identical": rendered == replay,
    },)


def _turns_digest(turns: tuple[DialogueScaleShowcaseTurn, ...]) -> str:
    """对展示轮次做规范摘要，供重复运行逐字节比较。"""
    payload = json.dumps(
        [item.to_dict() for item in turns],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _answer_with_narrow_runtime(
        trained_surface=None,
        ) -> Callable[[str], tuple[str, str] | None]:
    """构造只读窄域完整句消费者，未命中即交给广域来源检索。"""
    runtime = load_or_rebuild_public_sparse_qa_runtime()
    catalog = build_public_sentence_demo_catalog(runtime)
    used_count = [0]

    def answer(question: str) -> tuple[str, str] | None:
        result = run_public_sentence_demo_bytes(
            runtime, catalog, question.encode("utf-8"))
        if result.generated_proposition_surface is None:
            return None
        surface = result.generated_proposition_surface
        if trained_surface is not None:
            rendered = trained_surface.render(surface, response_act="ANSWER")
            if rendered.used:
                used_count[0] += 1
                surface = rendered.surface
        return surface, "ANSWER"

    answer.surface_consumer_used_count = used_count
    return answer

    return answer


def _run_once(database: sqlite3.Connection,
              questions: tuple[str, ...],
              narrow_answer: Callable[[str], tuple[str, str] | None],
              ) -> tuple[DialogueScaleShowcaseTurn, ...]:
    """在一个只读连接上执行固定问题序列。"""
    state = BroadDialogueState((1, 1, 8))
    result: list[DialogueScaleShowcaseTurn] = []
    for question in questions:
        state, turn = answer_broad_dialogue_turn(
            state, question, database, narrow_answer=narrow_answer)
        result.append(DialogueScaleShowcaseTurn.from_turn(turn))
    return tuple(result)


def build_dialogue_scale_showcase(*, project_root: str | Path,
                                  database_path: str | Path,
                                  training_run_root: str | Path | None = None,
                                  extra_training_course_paths: tuple[str | Path, ...] = (),
                                  extra_variant_course_paths: tuple[str | Path, ...] = (),
                                  extra_variant_evidence_paths: tuple[str | Path, ...] = (),
                                  extra_order_course_paths: tuple[str | Path, ...] = (),
                                  extra_order_evidence_paths: tuple[str | Path, ...] = (),
                                  variant_probe_inputs: tuple[str, ...] = (),
                                  order_probe_values: tuple[str, ...] = (),
                                  order_probe_roles: tuple[str, ...] = (),
                                  ) -> dict[str, object]:
    """运行展示并返回可写入 K 盘的摘要；不改变任何持久学习状态。"""
    database = Path(database_path).resolve()
    if database.drive.upper() != "K:" or not database.is_file():
        raise ValueError("dialogue showcase database 必须是存在的 K 盘文件")
    course_paths = (*default_course_paths(project_root), *tuple(
        Path(item).resolve() for item in extra_training_course_paths))
    if len(course_paths) != len(set(course_paths)):
        raise ValueError("extra training course path 与默认课程重复")
    pack = load_dialogue_training_pack(course_paths)
    training_observation = None
    trained_surface = None
    if training_run_root is not None:
        training_observation = load_training_observation(
            training_run_root, expected_pack_sha256=pack.pack_sha256)
        from pure_integer_ai.experiments.conversation_trained_surface_runtime import (
            load_trained_surface_runtime,
        )
        trained_surface = load_trained_surface_runtime(
            project_root=project_root,
            training_run_root=training_run_root,
            expected_pack_sha256=pack.pack_sha256,
            extra_variant_course_paths=tuple(
                Path(item).resolve() for item in extra_variant_course_paths),
            extra_variant_evidence_paths=tuple(
                Path(item).resolve() for item in extra_variant_evidence_paths),
            extra_order_course_paths=tuple(
                Path(item).resolve() for item in extra_order_course_paths),
            extra_order_evidence_paths=tuple(
                Path(item).resolve() for item in extra_order_evidence_paths),
        )
    questions = showcase_questions()
    narrow_answer = _answer_with_narrow_runtime(trained_surface)
    first_connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        first = _run_once(first_connection, questions, narrow_answer)
    finally:
        first_connection.close()
    first_surface_used_count = (
        None if trained_surface is None
        else int(getattr(narrow_answer, "surface_consumer_used_count")[0]))
    typed_surface_probes = _trained_surface_typed_probes(trained_surface)
    variant_surface_probes = _trained_surface_variant_probes(
        trained_surface, variant_probe_inputs)
    order_surface_probes = _trained_surface_order_probes(
        trained_surface, order_probe_values, order_probe_roles)
    replay_connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        replay = _run_once(replay_connection, questions, narrow_answer)
    finally:
        replay_connection.close()
    first_digest = _turns_digest(first)
    replay_digest = _turns_digest(replay)
    status_counts: dict[str, int] = {}
    for item in first:
        status_counts[item.status] = status_counts.get(item.status, 0) + 1
    return {
        "format_version": 1,
        "database_source": database.name,
        "question_count": len(questions),
        "long_question_count": sum(item.is_long_question for item in first),
        "status_counts": dict(sorted(status_counts.items())),
        "source_bound_answer_count": sum(
            item.status == "ANSWER" and item.source_title is not None
            for item in first),
        "replay_bit_identical": first_digest == replay_digest
        and first == replay,
        "turns_sha256": first_digest,
        "pack": _pack_stats(pack),
        "training_observation": (
            None if training_observation is None
            else training_observation.to_dict()),
        "trained_surface_consumer": {
            "bound": trained_surface is not None,
            "first_replay_used_count": first_surface_used_count,
            "typed_probe_count": len(typed_surface_probes),
            "typed_probe_used_count": sum(
                int(item["used"]) for item in typed_surface_probes),
            "typed_probe_results": list(typed_surface_probes),
            "variant_probe_count": len(variant_surface_probes),
            "variant_probe_used_count": sum(
                int(item["used"]) for item in variant_surface_probes),
            "variant_probe_results": list(variant_surface_probes),
            "order_probe_count": len(order_surface_probes),
            "order_probe_used_count": sum(
                int(item["used"]) for item in order_surface_probes),
            "order_probe_results": list(order_surface_probes),
        },
        "turns": [item.to_dict() for item in first],
    }


def write_dialogue_scale_showcase(value: dict[str, object],
                                  output_path: str | Path) -> str:
    """只允许创建 K 盘展示摘要，不覆盖既有产物。"""
    output = Path(output_path).resolve()
    if output.drive.upper() != "K:" or output.exists():
        raise ValueError("showcase output 必须是不存在的 K 盘文件")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")) + "\n"
    output.write_text(payload, encoding="utf-8")
    return str(output)


__all__ = [
    "DialogueScaleShowcaseTurn",
    "TrainingObservation",
    "build_dialogue_scale_showcase",
    "load_training_observation",
    "showcase_questions",
    "write_dialogue_scale_showcase",
]
