"""扩大来源覆盖的多轮问答开发切片。

v1 只验证一条焦点边界；v2 把铁路、桥梁、知识图谱、机场和地理分布五个
来源域放进同一只读运行，并为每个 ANSWER 保存预声明的主证据词。这样
``ANSWER`` 但选错证据的情况会明确失败，而不会被轮次数量掩盖。
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3

from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)
from pure_integer_ai.experiments.conversation_multiturn_scale import (
    MULTITURN_FAIL,
    MULTITURN_PASS,
    MultiturnScaleError,
    MultiturnTurn,
    _narrow_answer_factory,
    _run_once,
    _turns_digest,
)
from pure_integer_ai.experiments.conversation_training_pack import (
    load_dialogue_training_pack,
)
from pure_integer_ai.experiments.conversation_trained_surface_runtime import (
    load_trained_surface_runtime,
)
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_selector import (
    SurfaceSemantic,
)
from pure_integer_ai.experiments.run_conversation_training import (
    default_course_paths,
)
from pure_integer_ai.experiments.conversation_dialogue_scale_showcase import (
    load_training_observation,
)


MULTITURN_V2_PROTOCOL_V1 = 1
_TRACE_DOMAIN = "pure_integer_ai.dialogue.multiturn.scale.v2"


def _u(value: str) -> str:
    """保留源码 UTF-8 字符串，集中校验固定问式不是空值。"""
    if not isinstance(value, str) or not value.strip():
        raise MultiturnScaleError("v2 question 不能为空")
    return value


_QUESTIONS = tuple(map(_u, (
    # railway
    "根据保满铁路的工程资料回答：保满铁路全长多少公里？",
    "它设有多少座车站？",
    "请根据该条目的工程资料，用一个完整句子说明车站数量。",
    # bridge
    "从矮寨大桥的工程时间线看，矮寨大桥何时建成通车？",
    "它何时建成通车？",
    "请用一个完整句子说明该桥的建成时间。",
    # wikidata
    "根据维基数据条目中的结构化数据信息，维基数据的结构化数据采用什么许可发布？",
    "它由哪个组织托管？",
    "它是由哪个组织托管的？",
    # airport
    "儋州西庆机场距离儋州市区多远？",
    "它位于哪里？",
    "请用一个完整句子说明该机场的位置。",
    # a source -> unknown -> pronoun boundary, then narrow recovery
    "从矮寨大桥的工程时间线看，矮寨大桥何时建成通车？",
    "火星上的矮寨大桥何时通车？",
    "它分布在哪些地区？",
    "什么使得河水上涨？",
    # Huangshan pine
    "请只依据黄山松条目中关于地理分布的公开资料回答：黄山松分布在哪些地区？",
    "它分布在哪些地区？",
    "请用一个完整句子说明该条目的地理分布。",
)))

_EXPECTED_STATUS = (
    "ANSWER", "ANSWER", "ANSWER",
    "ANSWER", "ANSWER", "ANSWER",
    "ANSWER", "ANSWER", "ANSWER",
    "ANSWER", "ANSWER", "ANSWER",
    "ANSWER", "UNKNOWN", "CLARIFY", "ANSWER",
    "ANSWER", "ANSWER", "ANSWER",
)

# Every non-empty tuple must be contained in the displayed primary evidence.
_EVIDENCE_TOKENS = (
    ("35公里",), ("5座",), ("5座",),
    ("2012年3月31日",), ("2012年3月31日",), ("2012年3月31日",),
    ("知识共享CC0",), ("维基媒体基金会",), ("维基媒体基金会",),
    ("14公里",), ("中国海南省",), ("中国海南省",),
    ("2012年3月31日",), (), (), ("暴雨使得河水上涨",),
    ("福建", "贵州"), ("福建", "贵州"), ("福建", "贵州"),
)


@dataclass(frozen=True, slots=True)
class ExpandedMultiturnReport:
    status: str
    run_id: str
    pack_sha256: str
    database_name: str
    scenario_count: int
    question_count: int
    answer_count: int
    unknown_count: int
    clarify_count: int
    long_answer_count: int
    evidence_expected_count: int
    evidence_hit_count: int
    trained_surface_used_count: int
    focus_injection_count: int
    focus_not_crossed_unknown: int
    replay_bit_identical: bool
    turns_sha256: str
    turns: tuple[MultiturnTurn, ...]
    trace: tuple[int, ...]
    surface_variant_probe_count: int = 0
    surface_variant_probe_used_count: int = 0
    surface_order_probe_count: int = 0
    surface_order_probe_used_count: int = 0
    surface_probe_results: tuple[dict[str, object], ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "answer_count": self.answer_count,
            "clarify_count": self.clarify_count,
            "database_name": self.database_name,
            "evidence_expected_count": self.evidence_expected_count,
            "evidence_hit_count": self.evidence_hit_count,
            "focus_injection_count": self.focus_injection_count,
            "focus_not_crossed_unknown": bool(self.focus_not_crossed_unknown),
            "format_version": MULTITURN_V2_PROTOCOL_V1,
            "long_answer_count": self.long_answer_count,
            "pack_sha256": self.pack_sha256,
            "question_count": self.question_count,
            "replay_bit_identical": self.replay_bit_identical,
            "run_id": self.run_id,
            "scenario_count": self.scenario_count,
            "status": self.status,
            "surface_order_probe_count": self.surface_order_probe_count,
            "surface_order_probe_used_count": self.surface_order_probe_used_count,
            "surface_probe_results": list(self.surface_probe_results),
            "surface_variant_probe_count": self.surface_variant_probe_count,
            "surface_variant_probe_used_count": self.surface_variant_probe_used_count,
            "trace_u": list(self.trace),
            "trained_surface_used_count": self.trained_surface_used_count,
            "turns": [item.to_dict() for item in self.turns],
            "turns_sha256": self.turns_sha256,
            "unknown_count": self.unknown_count,
        }


def expanded_multiturn_questions() -> tuple[str, ...]:
    return _QUESTIONS


def build_expanded_multiturn_report(*, project_root: str | Path,
                                    database_path: str | Path,
                                    training_run_root: str | Path,
                                    expected_pack_sha256: str | None = None,
                                    extra_training_course_paths: tuple[str | Path, ...] = (),
                                    extra_variant_course_paths: tuple[str | Path, ...] = (),
                                    extra_variant_evidence_paths: tuple[str | Path, ...] = (),
                                    extra_order_course_paths: tuple[str | Path, ...] = (),
                                    extra_order_evidence_paths: tuple[str | Path, ...] = (),
                                    variant_probe_inputs: tuple[str, ...] = (),
                                    order_probe_values: tuple[str, ...] = (),
                                    order_probe_roles: tuple[str, ...] = (),
                                    ) -> ExpandedMultiturnReport:
    """运行 19 轮、6 个来源/边界场景的只读开发评估。"""
    root = Path(project_root).resolve()
    database = Path(database_path).resolve()
    if database.drive.upper() != "K:" or not database.is_file():
        raise MultiturnScaleError("v2 database 必须是存在的 K 盘文件")
    course_paths = (*default_course_paths(root), *tuple(
        Path(item).resolve() for item in extra_training_course_paths))
    if len(course_paths) != len(set(course_paths)):
        raise MultiturnScaleError("extra training course path 重复")
    pack = load_dialogue_training_pack(course_paths)
    expected = pack.pack_sha256 if expected_pack_sha256 is None else expected_pack_sha256
    trained_surface = load_trained_surface_runtime(
        project_root=root, training_run_root=training_run_root,
        expected_pack_sha256=expected,
        extra_variant_course_paths=tuple(
            Path(item).resolve() for item in extra_variant_course_paths),
        extra_variant_evidence_paths=tuple(
            Path(item).resolve() for item in extra_variant_evidence_paths),
        extra_order_course_paths=tuple(
            Path(item).resolve() for item in extra_order_course_paths),
        extra_order_evidence_paths=tuple(
            Path(item).resolve() for item in extra_order_evidence_paths),
    )
    observation = load_training_observation(
        training_run_root, expected_pack_sha256=expected)
    narrow_answer, used_count = _narrow_answer_factory(trained_surface)
    first_connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        first = _run_once(first_connection, _QUESTIONS, narrow_answer)
    finally:
        first_connection.close()
    used_first = used_count[0]
    replay_narrow, _ = _narrow_answer_factory(trained_surface)
    replay_connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        replay = _run_once(replay_connection, _QUESTIONS, replay_narrow)
    finally:
        replay_connection.close()
    statuses = tuple(item.status for item in first)
    evidence_expected = sum(bool(tokens) for tokens in _EVIDENCE_TOKENS)
    evidence_hit = sum(
        bool(tokens) and item.display_answer is not None
        and all(token in item.display_answer for token in tokens)
        for item, tokens in zip(first, _EVIDENCE_TOKENS))
    focus_not_crossed_unknown = int(
        first[13].status == "UNKNOWN"
        and first[14].status == "CLARIFY"
        and first[14].retrieval_question == first[14].question
        and first[14].source_title is None
    )
    focus_injection_count = sum(
        item.retrieval_question is not None
        and item.retrieval_question != item.question
        for item in first)
    answer_count = sum(item.status == "ANSWER" for item in first)
    unknown_count = sum(item.status == "UNKNOWN" for item in first)
    clarify_count = sum(item.status == "CLARIFY" for item in first)
    long_answer_count = sum(
        len((item.display_answer or "").encode("utf-8")) >= 48
        for item in first if item.status == "ANSWER")
    digest = _turns_digest(first)
    replay_digest = _turns_digest(replay)
    replay_identical = first == replay and digest == replay_digest
    passed = (
        statuses == _EXPECTED_STATUS
        and evidence_hit == evidence_expected
        and focus_not_crossed_unknown
        and focus_injection_count == 10
        and used_first == 1
        and replay_identical
    )
    probe_results: list[dict[str, object]] = []
    for input_surface in variant_probe_inputs:
        rendered = trained_surface.render(
            input_surface, response_act="ANSWER",
            source_title="multiturn-surface-probe", ordinal=0)
        replay_rendered = trained_surface.render(
            input_surface, response_act="ANSWER",
            source_title="multiturn-surface-probe", ordinal=0)
        probe_results.append({
            "kind": "variant",
            "input_surface": input_surface,
            "output_surface": rendered.surface,
            "used": rendered.used,
            "reason": rendered.reason,
            "replay_bit_identical": rendered == replay_rendered,
        })
    if order_probe_values or order_probe_roles:
        if (len(order_probe_values) != len(order_probe_roles)
                or not order_probe_values
                or len(set(order_probe_roles)) != len(order_probe_roles)):
            raise MultiturnScaleError("order probe role/value 数量不一致")
        fields = {"subject": "runtime", "predicate": "runtime",
                  "object": "runtime"}
        for role, value in zip(order_probe_roles, order_probe_values):
            if not isinstance(value, str) or not value or value.strip() != value:
                raise MultiturnScaleError("order probe value 必须是规范非空字符串")
            if role in {"subject", "topic", "cause"}:
                fields["subject"] = value
            elif role in {"predicate", "relation"}:
                fields["predicate"] = value
            elif role in {"object", "claim", "effect"}:
                fields["object"] = value
        semantic = SurfaceSemantic(
            "multiturn-order-probe", "state",
            fields["subject"], fields["predicate"], fields["object"],
        )
        rendered = trained_surface.render_order_typed(
            semantic, response_act="ANSWER", register="neutral",
            ordered_roles=order_probe_roles,
            slot_values=order_probe_values,
            source_id="multiturn-order-probe",
            context_id="multiturn-order-context",
            family_id="multiturn-order-family",
        )
        replay_rendered = trained_surface.render_order_typed(
            semantic, response_act="ANSWER", register="neutral",
            ordered_roles=order_probe_roles,
            slot_values=order_probe_values,
            source_id="multiturn-order-probe",
            context_id="multiturn-order-context",
            family_id="multiturn-order-family",
        )
        probe_results.append({
            "kind": "order",
            "input_roles": list(order_probe_roles),
            "input_values": list(order_probe_values),
            "output_surface": rendered.surface,
            "used": rendered.used,
            "reason": rendered.reason,
            "replay_bit_identical": rendered == replay_rendered,
        })
    status = MULTITURN_PASS if passed else MULTITURN_FAIL
    trace_values = [MULTITURN_V2_PROTOCOL_V1, len(first), answer_count,
                    unknown_count, clarify_count, evidence_expected,
                    evidence_hit, focus_injection_count,
                    focus_not_crossed_unknown, used_first]
    for item in first:
        trace_values.extend(item.canonical_record())
    trace = integer_tuple_fingerprint(
        tuple(trace_values), domain=_TRACE_DOMAIN)
    return ExpandedMultiturnReport(
        status, observation.run_id, observation.pack_sha256, database.name,
        6, len(first), answer_count, unknown_count, clarify_count,
        long_answer_count, evidence_expected, evidence_hit, used_first,
        focus_injection_count, focus_not_crossed_unknown, replay_identical,
        digest, first, trace,
        sum(item["kind"] == "variant" for item in probe_results),
        sum(item["kind"] == "variant" and item["used"] is True
            for item in probe_results),
        sum(item["kind"] == "order" for item in probe_results),
        sum(item["kind"] == "order" and item["used"] is True
            for item in probe_results),
        tuple(probe_results),
    )


def write_expanded_multiturn_report(report: ExpandedMultiturnReport,
                                    output_path: str | Path) -> str:
    output = Path(output_path).resolve()
    if output.drive.upper() != "K:" or output.exists():
        raise ValueError("v2 output 必须是不存在的 K 盘文件")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.to_dict(), ensure_ascii=False,
                                 sort_keys=True, separators=(",", ":")) + "\n",
                       encoding="utf-8")
    return str(output)


__all__ = [
    "ExpandedMultiturnReport", "MULTITURN_V2_PROTOCOL_V1",
    "build_expanded_multiturn_report", "expanded_multiturn_questions",
    "write_expanded_multiturn_report",
]
