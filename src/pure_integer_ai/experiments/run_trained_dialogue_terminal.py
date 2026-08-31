"""K 盘训练状态驱动的公开交互终端。

这是面向真实训练结果的轻量 host 入口：QA SQLite 以只读方式打开，训练 run
只读加载，回答仍由既有窄域/广域查询和表层组织 runtime 产生。Core、Runtime
和训练账本不会因交互写入；默认会话热区仅存在当前进程，也可通过独立 K 盘
``session_root`` 以纯整数 checkpoint 持久化最近 8 轮。
"""
from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import sqlite3
import sys
import time
from typing import BinaryIO, Callable
from typing import Any

from pure_integer_ai.experiments.conversation_broad_qa_runtime import (
    BroadDialogueState,
    DialogueCitation,
    DialogueTurn,
    answer_broad_dialogue_turn,
    build_index_evidence_source_followup_resolver,
)
from pure_integer_ai.experiments.ph2_broad_qa_query import BroadQaQueryCache
from pure_integer_ai.experiments.ph2_broad_qa_query import SurfaceVariantProvider
from pure_integer_ai.experiments.conversation_broad_dialogue_persistence import (
    BroadDialoguePersistenceError,
    PersistentBroadDialogueRecovery,
    append_broad_dialogue_checkpoint,
    recover_broad_dialogue_checkpoint,
)
from pure_integer_ai.experiments.conversation_broad_memory_recall import (
    BroadDialogueMemoryRecallIndex,
)
from pure_integer_ai.experiments.conversation_broad_runtime_memory_bridge import (
    BroadDialogueRuntimeMemoryError,
    append_dialogue_turn_to_runtime_memory,
    empty_runtime_memory_for_conversation,
    replay_dialogue_state_to_runtime_memory,
)
from pure_integer_ai.experiments.conversation_dialogue_scale_showcase import (
    load_training_observation,
)
from pure_integer_ai.experiments.conversation_public_sentence_demo import (
    build_public_sentence_demo_catalog,
    run_public_sentence_demo_bytes,
)
from pure_integer_ai.experiments.conversation_training_pack import (
    load_dialogue_training_pack,
)
from pure_integer_ai.experiments.conversation_runtime_material_binding_persistence import (
    RuntimeMaterialBindingPersistenceError,
    load_runtime_material_response_provider,
)
from pure_integer_ai.experiments.conversation_runtime_material_persistence import (
    RuntimeMaterialPersistenceError,
    load_runtime_material_runtime,
    open_runtime_material_sqlite,
    rebuild_runtime_material_observations,
)
from pure_integer_ai.experiments.conversation_trained_surface_runtime import (
    TrainedSurfaceRuntime,
    load_trained_surface_runtime,
)
from pure_integer_ai.experiments.build_learned_dialogue_response_artifact import (
    load_learned_dialogue_response_artifact,
)
from pure_integer_ai.experiments.conversation_learned_dialogue_response import (
    LearnedDialogueResponseRuntime,
    PRODUCTION_MIN_FRAGMENT_OCCURRENCES,
    PRODUCTION_MIN_SIMILARITY_PERMILLE,
)
from pure_integer_ai.experiments.conversation_dialogue_experts import (
    LearnedDialogueExpertRouter,
)
from pure_integer_ai.experiments.scidb_csq_passage_index import (
    ScidbCsqPassageRuntime,
)
from pure_integer_ai.experiments.sqlite_learned_dialogue_intent import (
    SqliteLearnedDialogueIntentRuntime,
)
from pure_integer_ai.experiments.dialogue_expert_routing_artifact import (
    load_dialogue_expert_routing_model,
)
from pure_integer_ai.experiments.dialogue_successor_graph import (
    SqliteDialogueSuccessorRuntime,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_sparse_qa_snapshot import (
    load_or_rebuild_public_sparse_qa_runtime,
)
from pure_integer_ai.experiments.run_conversation_training import (
    default_course_paths,
)
from pure_integer_ai.experiments.ph2_broad_qa_relation_evidence_learning import (
    learn_relation_evidence_model,
)
from pure_integer_ai.experiments.ph2_broad_qa_relation_role_evidence_learning import (
    learn_relation_role_evidence_model,
)
from pure_integer_ai.experiments.ph2_broad_qa_relation_marker_evidence_learning import (
    learn_relation_marker_evidence_model,
)
from pure_integer_ai.experiments.ph2_broad_qa_relation_answer_frame_learning import (
    learn_relation_answer_frame_model,
)
from pure_integer_ai.experiments.ph2_broad_qa_question_slots import (
    load_broad_qa_question_slots,
)
from pure_integer_ai.storage.k_run_boundary import (
    ensure_normal_relative_directory,
    open_existing_run_root,
)
from pure_integer_ai.storage.source_record import SourceRecordRepository


# 正式交互入口的来源段必须有足够完整的正文覆盖；通用 runtime 的较低
# 默认门仅用于离线索引探针，不能让低置信候选进入用户可见回答。
PRODUCTION_SOURCE_MIN_CONFIDENCE_PERMILLE = 500


def _require_k_file(value: str | Path, *, label: str) -> Path:
    """核验只读外部索引位于 K 盘且确实存在。"""
    path = Path(value).resolve()
    if path.drive.upper() != "K:" or not path.is_file():
        raise ValueError(f"{label} 必须是 K 盘已存在文件")
    return path


def _release_bound_artifact_root(
        release: object | None, explicit_root: str | Path | None, *,
        attribute: str, label: str,
        ) -> Path | str | None:
    """选择 release 内 artifact；显式外部路径不得替换已发布身份。"""
    embedded = getattr(release, attribute, None) if release is not None else None
    if embedded is None:
        return explicit_root
    embedded_path = Path(embedded).resolve()
    if (explicit_root is not None
            and Path(explicit_root).resolve() != embedded_path):
        raise ValueError(f"release_root 与 {label} 不得冲突")
    return embedded_path


def _release_bound_artifact_roots(
        release: object | None, explicit_roots: tuple[str | Path, ...], *,
        attribute: str, label: str,
        ) -> tuple[Path | str, ...]:
    """Select one ordered embedded expert family without external replacement."""
    embedded = (tuple(getattr(release, attribute, ()))
                if release is not None else ())
    explicit = tuple(explicit_roots)
    if not embedded:
        return explicit
    embedded_paths = tuple(Path(item).resolve() for item in embedded)
    if (explicit and tuple(Path(item).resolve() for item in explicit)
            != embedded_paths):
        raise ValueError(f"release_root 与 {label} 不得冲突")
    return embedded_paths


def _release_bound_artifact_file(
        release: object | None, explicit: str | Path | None, *,
        attribute: str, label: str,
        ) -> Path | str | None:
    embedded = getattr(release, attribute, None) if release is not None else None
    if embedded is None:
        return explicit
    embedded_path = Path(embedded).resolve()
    if explicit is not None and Path(explicit).resolve() != embedded_path:
        raise ValueError(f"release_root 与 {label} 不得冲突")
    return embedded_path


def _course_paths_for_training_run(
        project_root: Path,
        training_run_root: Path,
        extra_course_paths: tuple[str | Path, ...],
        ) -> tuple[tuple[Path, ...], int | None]:
    """按训练 run 冻结的 source_files 重建完全相同的公开 pack。

    默认课程 glob 会遗漏训练时显式附加的公开课程；终端必须读取 run 自己
    的 manifest，不能靠调用方记忆补参数，也不能放宽 pack SHA 校验。
    """
    manifest_path = training_run_root / "dialogue_pack_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("training run 缺少 dialogue_pack_manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("training run pack manifest 不可回读") from error
    rows = manifest.get("source_files") if isinstance(manifest, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("training run pack manifest 缺少 source_files")
    max_cases = manifest.get("max_cases")
    if max_cases is not None and (type(max_cases) is not int or max_cases <= 0):
        raise ValueError("training run max_cases 非法")
    resolved: list[Path] = []
    for row in rows:
        if not isinstance(row, list) or not row:
            raise ValueError("training run source_files 记录非法")
        candidate = Path(str(row[0]))
        if not candidate.is_absolute():
            project_candidate = (project_root / candidate).resolve()
            if project_candidate.is_file():
                try:
                    project_candidate.relative_to(project_root)
                except ValueError as error:
                    raise ValueError(
                        "training run source_files 越出 project_root") from error
                candidate = project_candidate
            else:
                # Portable training runs may retain their public source copy
                # under the K: run root.  This keeps D: free of expanded
                # training data while allowing an independent terminal to
                # rebuild exactly the recorded pack.
                run_candidate = (training_run_root / candidate).resolve()
                try:
                    run_candidate.relative_to(training_run_root)
                except ValueError as error:
                    raise ValueError(
                        "training run source_files 越出 training_run_root") from error
                candidate = run_candidate
        else:
            candidate = candidate.resolve()
        if not candidate.is_file():
            raise ValueError(f"training run source_files 缺失: {candidate}")
        resolved.append(candidate)
    for value in extra_course_paths:
        candidate = Path(value).resolve()
        try:
            candidate.relative_to(project_root)
        except ValueError as error:
            raise ValueError("extra course 越出 project_root") from error
        if candidate not in resolved:
            resolved.append(candidate)
    if len(resolved) != len(set(resolved)):
        raise ValueError("training run source_files 重复")
    identities = manifest.get("source_identities")
    if identities is not None:
        if not isinstance(identities, list) or len(identities) != len(resolved):
            raise ValueError("training run source_identities 非法")
        source_identity_map = {}
        for ordinal, item in enumerate(identities):
            if (not isinstance(item, list) or len(item) != 2
                    or not isinstance(item[1], str) or not item[1]):
                raise ValueError("training run source identity 记录非法")
            source_identity_map[resolved[ordinal]] = item[1]
    else:
        # 早期 portable run 将 identity 直接写在 source_files[0]，但未
        # 单独写 source_identities；优先从该冻结字段恢复，避免把当前宿主
        # 的绝对路径重新带入 pack SHA。
        portable_rows = all(
            isinstance(row, list) and len(row) == 3
            and isinstance(row[0], str)
            and not Path(row[0]).is_absolute()
            and not Path(row[0]).drive
            for row in rows)
        source_identity_map = (
            {resolved[ordinal]: str(row[0])
             for ordinal, row in enumerate(rows)}
            if portable_rows else None)
    return tuple(resolved), max_cases, source_identity_map


def _frozen_pack_sha256(training_run_root: Path) -> str:
    """读取训练 run 已冻结的 pack identity，不重新解析 JSONL 课程。

    发布运行的课程文件已经由 ``public_model_release`` manifest 绑定并在
    strict 档校验 SHA；pack manifest 同时记录了由训练时 canonical 投影
    得到的 pack SHA。因此 deferred 档可以复用这个整数身份，把启动时
    数百万次字符/整数编码留给训练入口，而不是每次对话进程重复执行。
    """
    manifest_path = training_run_root / "dialogue_pack_manifest.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("training run pack manifest 不可回读") from error
    value = payload.get("pack_sha256") if isinstance(payload, dict) else None
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise ValueError("training run pack_sha256 非法")
    return value


def _narrow_answer(runtime, catalog, trained_surface: TrainedSurfaceRuntime | None):
    """建立只读窄域消费者；未命中交给广域来源查询。"""
    def answer(question: str) -> tuple[str, str] | None:
        result = run_public_sentence_demo_bytes(
            runtime, catalog, question.encode("utf-8"))
        if result.generated_proposition_surface is None:
            return None
        surface = result.generated_proposition_surface
        if trained_surface is not None:
            rendered = trained_surface.render(surface, response_act="ANSWER")
            if rendered.used:
                surface = rendered.surface
        return surface, "ANSWER"
    return answer


def _display(turn) -> str:
    """将既有 turn 的答案载荷投影为人可读的一行，不改动 turn 语义。"""
    if turn.citations:
        blocks = []
        for citation in turn.citations:
            block = citation.surface
            if citation.source_title:
                source = f"来源：{citation.source_title}"
                if citation.source_url:
                    source += f"（{citation.source_url}）"
                block += f"\n{source}"
            blocks.append(block)
        return "\n\n".join(blocks)
    if turn.display_answer:
        answer = turn.display_answer
    elif turn.status == "CLARIFY":
        answer = "请补充问题的范围或限定条件。"
    elif turn.status == "REPAIR":
        answer = "当前问题需要补充信息后才能回答。"
    else:
        answer = "当前公开资料无法确认这个问题。"
    if turn.source_title:
        source = f"来源：{turn.source_title}"
        if turn.source_url:
            source += f"（{turn.source_url}）"
        return f"{answer}\n{source}"
    return answer


def _public_turn_text(turn) -> str:
    """返回正式用户接口唯一可见的自然语言文本。

    ``DialogueTurn.status`` 是理解/评测侧的状态，不属于公开语言协议。
    这里复用终端已有的人类可读投影；来源引用仍由协议的 ``citations``
    字段独立传递。这样 JSONL 与终端共享同一语言表面，不会把内部状态
    枚举泄漏给调用者。
    """
    value = _display(turn)
    if type(value) is not str or not value.strip():
        raise ValueError("公开对话文本不能为空")
    return value


def _protocol_turn_payload(turn) -> dict[str, object]:
    """把 DialogueTurn 投影为稳定的机器可读响应，不改变语义。"""
    return {
        "type": "response",
        "ordinal": turn.ordinal,
        "question": turn.question,
        "retrieval_question": turn.retrieval_question,
        "status": turn.status,
        "answer": turn.answer,
        "display_answer": turn.display_answer,
        "source_title": turn.source_title,
        "source_url": turn.source_url,
        "citations": [
            {
                "surface": citation.surface,
                "source_title": citation.source_title,
                "source_url": citation.source_url,
                "license_id": citation.license_id,
                "attribution": citation.attribution,
                "source_ref": (
                    list(citation.source_ref)
                    if citation.source_ref is not None else None),
            }
            for citation in turn.citations
        ],
        "turn_key": list(turn.turn_key),
    }


def _public_protocol_turn_payload(turn) -> dict[str, object]:
    """把内部 turn 投影为正式 JSONL 用户响应。

    公开协议只提供自然语言 ``text``、来源引用和稳定轮次身份；内部
    ``status``、原始答案槽和检索问式留在评测/审计投影中，不对用户暴露。
    """
    if not hasattr(turn, "ordinal"):
        raise TypeError("公开协议 turn 类型错误")
    payload: dict[str, object] = {
        "type": "response",
        "ordinal": turn.ordinal,
        "text": _public_turn_text(turn),
        "citations": [
            {
                "surface": citation.surface,
                "source_title": citation.source_title,
                "source_url": citation.source_url,
                "license_id": citation.license_id,
                "attribution": citation.attribution,
                "source_ref": (
                    list(citation.source_ref)
                    if citation.source_ref is not None else None),
            }
            for citation in turn.citations
        ],
        "turn_key": list(turn.turn_key),
    }
    if turn.source_title is not None or turn.source_url is not None:
        payload["source"] = {
            "title": turn.source_title,
            "url": turn.source_url,
        }
    return payload


def _write_protocol_payload(stream_out: BinaryIO,
                            payload: dict[str, object]) -> None:
    """以无 BOM UTF-8 JSONL 写出一条协议响应。"""
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    stream_out.write(encoded)
    stream_out.flush()


def _peak_working_set_bytes() -> int:
    """读取宿主进程峰值工作集，仅用于 K 盘性能诊断。"""
    if sys.platform == "win32":
        class _Counters(ctypes.Structure):
            _fields_ = (
                ("cb", ctypes.c_ulong),
                ("page_fault_count", ctypes.c_ulong),
                ("peak_working_set_size", ctypes.c_size_t),
                ("working_set_size", ctypes.c_size_t),
                ("quota_peak_paged_pool_usage", ctypes.c_size_t),
                ("quota_paged_pool_usage", ctypes.c_size_t),
                ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
                ("quota_non_paged_pool_usage", ctypes.c_size_t),
                ("pagefile_usage", ctypes.c_size_t),
                ("peak_pagefile_usage", ctypes.c_size_t),
            )
        counters = _Counters()
        counters.cb = ctypes.sizeof(_Counters)
        get_process = ctypes.windll.kernel32.GetCurrentProcess
        get_process.restype = ctypes.c_void_p
        get_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_info.argtypes = (
            ctypes.c_void_p, ctypes.POINTER(_Counters), ctypes.c_ulong)
        get_info.restype = ctypes.c_int
        process = get_process()
        if get_info(process, ctypes.byref(counters), counters.cb):
            return int(counters.peak_working_set_size)
        return 0
    try:
        import resource
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, AttributeError, OSError):
        return 0
    return value * (1024 if sys.platform != "darwin" else 1)


def _nearest_rank_latency_us(
        ordered_values: list[int], percentile: int,
        ) -> int:
    """用纯整数 nearest-rank 计算延迟分位，短序列不漏掉 p95 尾点。"""
    if (not isinstance(ordered_values, list) or not ordered_values
            or any(type(value) is not int or value < 0
                   for value in ordered_values)
            or ordered_values != sorted(ordered_values)):
        raise ValueError("latency values 必须是非空有序非负整数 list")
    if type(percentile) is not int or not 1 <= percentile <= 100:
        raise ValueError("latency percentile 必须是 1..100 严格整数")
    rank = (len(ordered_values) * percentile + 99) // 100
    return ordered_values[rank - 1]


def run_trained_dialogue_terminal(
        *,
        project_root: str | Path,
        qa_database: str | Path | None = None,
        release_root: str | Path | None = None,
        training_run_root: str | Path | None = None,
        response_organization_artifact_root: str | Path | None = None,
        dialogue_response_artifact_root: str | Path | None = None,
        dialogue_domain_expert_artifact_roots: tuple[str | Path, ...] = (),
        dialogue_expert_routing_artifact_root: str | Path | None = None,
        science_passage_artifact_root: str | Path | None = None,
        session_root: str | Path | None = None,
        extra_course_paths: tuple[str | Path, ...] = (),
        extra_variant_course_paths: tuple[str | Path, ...] = (),
        extra_variant_evidence_paths: tuple[str | Path, ...] = (),
        runtime_material_answer: Callable[
            [str], tuple[str, str | None, str | None] | None] | None = None,
        runtime_material_response: Callable[
            [str], tuple[object, ...] | None]
        | None = None,
        runtime_material_binding_root: str | Path | None = None,
        runtime_material_source_records: SourceRecordRepository | None = None,
        runtime_material_observations: tuple[object, ...] = (),
        surface_variant_provider: SurfaceVariantProvider | None = None,
        source_followup_resolver: Callable[[str, DialogueTurn], bool]
        | None = None,
        runtime_material_runtime_root: str | Path | None = None,
        runtime_material_runtime_database: str | Path | None = None,
        runtime_material_binding_relative_path: str | Path = (
            "runtime_material_response/bindings.int"),
        learned_relation_evidence_model: Any | None = None,
        learned_relation_role_evidence_model: Any | None = None,
        learned_relation_marker_evidence_model: Any | None = None,
        learned_relation_answer_frame_model: Any | None = None,
        memory_recall_response: Callable[[str], str | None] | None = None,
        input_stream: BinaryIO | None = None,
        output_stream: BinaryIO | None = None,
        protocol_stream: bool = False,
        metrics_output: str | Path | None = None,
        performance_tier: str = "strict",
        ) -> int:
    """运行可回放的只读交互会话。

    默认使用人类终端；``protocol_stream=True`` 时读取无 BOM UTF-8 JSONL，
    每行一个 ``{"id": ..., "op": "turn", "text": ...}`` 请求并输出一条
    稳定 JSON 响应。两种入口共享完全相同的查询、证据、拒答和 checkpoint 路径。
    """
    if performance_tier not in {
            "strict", "deferred-narrow", "deferred-narrow-fast"}:
        raise ValueError(
            "performance_tier 必须是 strict、deferred-narrow 或 "
            "deferred-narrow-fast")
    runtime_sqlite_runtime = None
    runtime_material_response_provider = None
    if runtime_material_runtime_database is not None:
        if runtime_material_runtime_root is None:
            raise ValueError("runtime material SQLite 必须同时指定 runtime ledger root")
        if (runtime_material_binding_root is not None
                and Path(runtime_material_binding_root).resolve()
                != Path(runtime_material_runtime_root).resolve()):
            raise ValueError("runtime material binding/runtime ledger root 不一致")
        runtime_sqlite_runtime = open_runtime_material_sqlite(
            runtime_material_runtime_database)
        runtime_material_source_records = runtime_sqlite_runtime.source_records
        runtime_material_runtime_context = runtime_sqlite_runtime.context
        runtime_material_binding_root = runtime_material_runtime_root
    else:
        runtime_material_runtime_context = None

    if runtime_material_runtime_root is not None:
        if runtime_material_binding_root is not None and (
                Path(runtime_material_binding_root).resolve()
                != Path(runtime_material_runtime_root).resolve()):
            raise ValueError("runtime material binding/runtime ledger root 不一致")
        if runtime_material_response is not None or runtime_material_answer is not None:
            raise ValueError(
                "runtime ledger 不得与手工 Runtime provider 同时指定")
        def _runtime_fail_closed(_question: str):
            return "UNKNOWN", None, None, None
        if (runtime_material_runtime_context is None
                or not isinstance(runtime_material_source_records,
                                   SourceRecordRepository)):
            runtime_material_response = _runtime_fail_closed
        else:
            try:
                recovery = load_runtime_material_runtime(
                    runtime_material_runtime_root,
                    source_records=runtime_material_source_records,
                )
                observations = rebuild_runtime_material_observations(
                    runtime_material_runtime_context,
                    recovery,
                    source_records=runtime_material_source_records,
                )
                restored_provider = load_runtime_material_response_provider(
                    runtime_material_runtime_root,
                    source_records=runtime_material_source_records,
                    observations=observations,
                    relative_path=runtime_material_binding_relative_path,
                )
                runtime_material_response_provider = restored_provider
                runtime_material_response = restored_provider.response
            except (RuntimeMaterialPersistenceError,
                    RuntimeMaterialBindingPersistenceError,
                    TypeError, ValueError):
                runtime_material_response = _runtime_fail_closed

    if runtime_material_binding_root is not None and runtime_material_runtime_root is None:
        if runtime_material_response is not None or runtime_material_answer is not None:
            raise ValueError(
                "binding ledger 不得与手工 Runtime provider 同时指定")
        def _binding_fail_closed(_question: str):
            return "UNKNOWN", None, None, None
        if (not isinstance(runtime_material_source_records, SourceRecordRepository)
                or not isinstance(runtime_material_observations, tuple)
                or not runtime_material_observations):
            # ledger 存在但 observation source 缺失时，禁止落入窄域/广域
            # fallback；调用方仍可结束会话，但所有问题都保持未知。
            runtime_material_response = _binding_fail_closed
        else:
            try:
                restored_provider = load_runtime_material_response_provider(
                    runtime_material_binding_root,
                    source_records=runtime_material_source_records,
                    observations=runtime_material_observations,
                    relative_path=runtime_material_binding_relative_path,
                )
                runtime_material_response_provider = restored_provider
                runtime_material_response = restored_provider.response
            except (RuntimeMaterialBindingPersistenceError, TypeError, ValueError):
                runtime_material_response = _binding_fail_closed
    release = None
    if release_root is not None:
        from pure_integer_ai.experiments.public_model_release import (
            load_public_model_release,
        )
        release = load_public_model_release(
            release_root,
            # The fast tier is explicit and opt-in.  It keeps path/size/closed
            # manifest checks but leaves full payload SHA auditing to strict
            # startup and the independent release validator.
            verify_payload_hashes=(performance_tier != "deferred-narrow-fast"),
        )
        root = release.root
        database = release.qa_database
        resolved_training_run_root = release.training_root
        sparse_snapshot = release.sparse_snapshot
    else:
        if qa_database is None:
            raise ValueError("必须指定 release_root 或 qa_database")
        database = _require_k_file(qa_database, label="qa_database")
        root = Path(project_root).resolve()
        resolved_training_run_root = None
        sparse_snapshot = None
    dialogue_response_artifact_root = _release_bound_artifact_root(
        release, dialogue_response_artifact_root,
        attribute="dialogue_response_artifact",
        label="dialogue_response_artifact_root")
    dialogue_domain_expert_artifact_roots = _release_bound_artifact_roots(
        release, dialogue_domain_expert_artifact_roots,
        attribute="dialogue_domain_expert_artifacts",
        label="dialogue_domain_expert_artifact_roots")
    dialogue_expert_routing_artifact_root = _release_bound_artifact_file(
        release, dialogue_expert_routing_artifact_root,
        attribute="dialogue_expert_routing_artifact",
        label="dialogue_expert_routing_artifact_root")
    response_organization_artifact_root = _release_bound_artifact_root(
        release, response_organization_artifact_root,
        attribute="response_organization_artifact",
        label="response_organization_artifact_root")
    science_passage_artifact_root = _release_bound_artifact_root(
        release, science_passage_artifact_root,
        attribute="science_passage_artifact",
        label="science_passage_artifact_root")
    trained_surface = None
    learned_dialogue_answer = None
    learned_dialogue_clarify_answer = None
    core_dialogue_runtime = None
    # session checkpoint 恢复后注入的冷记忆索引；未启用 session 时保持空。
    dialogue_recovery = None
    dialogue_memory_index = None
    dialogue_response_runtime = None
    science_passage_runtime = None
    source_passage_response = None
    if release is not None and training_run_root is not None:
        if Path(training_run_root).resolve() != resolved_training_run_root:
            raise ValueError("release_root 与 training_run_root 不得冲突")
    run_root_value = resolved_training_run_root or training_run_root
    if run_root_value is not None:
        run_root = Path(run_root_value).resolve()
        if run_root.drive.upper() != "K:" or not run_root.is_dir():
            raise ValueError("training_run_root 必须是 K 盘已存在目录")
        course_paths, max_cases, source_identity_map = _course_paths_for_training_run(
            root, run_root, extra_course_paths)
        source_identities = None
        if release is not None:
            source_identities = {
                path: f"data/ph2/{path.name}" for path in course_paths
            }
        # A published release already freezes both source-file identities and
        # the canonical pack SHA. Rebuilding 3k+ JSONL records here was the
        # dominant cold-start cost (~17 s) and contributes nothing to a
        # read-only response once the trained model is loaded. Keep the full
        # parser for strict/non-release callers and use only the frozen
        # identity in deferred release tiers.
        if (release is not None
                and performance_tier in {"deferred-narrow", "deferred-narrow-fast"}
                and not extra_course_paths):
            expected_pack_sha256 = _frozen_pack_sha256(run_root)
        else:
            pack = load_dialogue_training_pack(
                course_paths, max_cases=max_cases,
                source_path_identities=(source_identity_map or source_identities))
            expected_pack_sha256 = pack.pack_sha256
        load_training_observation(
            run_root, expected_pack_sha256=expected_pack_sha256)
        trained_surface = load_trained_surface_runtime(
            project_root=root,
            training_run_root=run_root,
            expected_pack_sha256=expected_pack_sha256,
            response_organization_artifact_root=(
                response_organization_artifact_root),
            extra_variant_course_paths=tuple(
                Path(item).resolve() for item in extra_variant_course_paths),
            extra_variant_evidence_paths=tuple(
                Path(item).resolve() for item in extra_variant_evidence_paths),
        )
        model_database = run_root / "training.sqlite3"
        if model_database.is_file():
            try:
                candidate_runtime = SqliteDialogueSuccessorRuntime(
                    model_database)
                if candidate_runtime.count() > 0:
                    core_dialogue_runtime = candidate_runtime
                else:
                    candidate_runtime.close()
            except ValueError:
                # 旧 release 没有后继表时保持原兼容路径；新 release validator
                # 会单独要求该生产能力非零。
                core_dialogue_runtime = None
    if (dialogue_domain_expert_artifact_roots
            and dialogue_response_artifact_root is None):
        raise ValueError("领域对话专家必须绑定通用对话专家")
    if dialogue_response_artifact_root is not None:
        def _load_dialogue_response_runtime(
                artifact_root: str | Path,
                ) -> LearnedDialogueResponseRuntime:
            artifact = load_learned_dialogue_response_artifact(
                artifact_root,
                verify_payload_hashes=(
                    performance_tier != "deferred-narrow-fast"))
            sqlite_intent = (
                SqliteLearnedDialogueIntentRuntime(
                    artifact.intent_index_path, artifact.model.fragments)
                if artifact.intent_index_path is not None else None)
            return LearnedDialogueResponseRuntime(
                artifact.model, artifact.intent_model,
                intent_runtime=sqlite_intent)

        general_runtime = _load_dialogue_response_runtime(
            dialogue_response_artifact_root)
        if dialogue_expert_routing_artifact_root is not None:
            routing = load_dialogue_expert_routing_model(
                dialogue_expert_routing_artifact_root)
            if (routing.general_course_sha256
                    != general_runtime.model.course_sha256):
                raise ValueError("expert router 与通用模型 course SHA 不一致")
            if len(routing.domain_activation_features) != len(
                    dialogue_domain_expert_artifact_roots):
                raise ValueError("expert router 与领域 artifact 数量不一致")
            def _bound_domain_loader(
                    item: str | Path, expected_sha: tuple[int, ...],
                    ) -> LearnedDialogueResponseRuntime:
                runtime = _load_dialogue_response_runtime(item)
                if runtime.model.course_sha256 != expected_sha:
                    runtime.close()
                    raise ValueError(
                        "expert router 与领域模型 course SHA 不一致")
                return runtime

            lazy_domains = tuple(
                (frozenset(features),
                 lambda item=item, expected_sha=expected_sha:
                     _bound_domain_loader(item, expected_sha))
                for features, item, expected_sha in zip(
                    routing.domain_activation_features,
                    dialogue_domain_expert_artifact_roots,
                    routing.domain_course_sha256s))
            dialogue_response_runtime = LearnedDialogueExpertRouter(
                general_runtime, (), lazy_domains=lazy_domains)
        else:
            domain_runtimes = tuple(
                _load_dialogue_response_runtime(item)
                for item in dialogue_domain_expert_artifact_roots)
            dialogue_response_runtime = LearnedDialogueExpertRouter(
                general_runtime, domain_runtimes)

        def _dialogue_history(value: str) -> tuple[tuple[int, str], ...]:
            history: list[tuple[int, str]] = []
            hot_ordinals = set()
            for prior_turn in state.turns:
                hot_ordinals.add(prior_turn.ordinal)
                history.append((1, prior_turn.question))
                if prior_turn.answer:
                    history.append((2, prior_turn.answer))
            if dialogue_memory_index is not None:
                for prior_turn in dialogue_memory_index.query_relevant_turns(
                        value, limit=4, minimum_similarity_permille=600):
                    if prior_turn.ordinal in hot_ordinals:
                        continue
                    history.append((1, prior_turn.question))
                    if prior_turn.answer:
                        history.append((2, prior_turn.answer))
            return tuple(history)

        def _learned_dialogue_answer(value: str) -> str | None:
            result = dialogue_response_runtime.respond(
                value,
                history=_dialogue_history(value),
                minimum_fragment_occurrences=(
                    PRODUCTION_MIN_FRAGMENT_OCCURRENCES),
                minimum_similarity_permille=(
                    PRODUCTION_MIN_SIMILARITY_PERMILLE),
            )
            return result.surface if result.used else None

        learned_dialogue_answer = _learned_dialogue_answer

        def _learned_dialogue_clarify_answer(value: str) -> str | None:
            result = dialogue_response_runtime.respond(
                value,
                history=_dialogue_history(value),
                minimum_fragment_occurrences=(
                    PRODUCTION_MIN_FRAGMENT_OCCURRENCES),
                minimum_similarity_permille=900,
            )
            return result.surface if result.used else None

        learned_dialogue_clarify_answer = (
            _learned_dialogue_clarify_answer)
    if core_dialogue_runtime is not None:
        legacy_dialogue_answer = learned_dialogue_answer
        legacy_dialogue_clarify = learned_dialogue_clarify_answer

        def _core_dialogue_history(value: str) -> tuple[tuple[int, str], ...]:
            """把既有有界会话状态投影为核心后继查询的分层热区。"""
            history: list[tuple[int, str]] = []
            hot_ordinals = set()
            for prior_turn in state.turns:
                hot_ordinals.add(prior_turn.ordinal)
                history.append((1, prior_turn.question))
                if prior_turn.answer:
                    history.append((2, prior_turn.answer))
            if dialogue_memory_index is not None:
                for prior_turn in dialogue_memory_index.query_relevant_turns(
                        value, limit=4, minimum_similarity_permille=600):
                    if prior_turn.ordinal in hot_ordinals:
                        continue
                    history.append((1, prior_turn.question))
                    if prior_turn.answer:
                        history.append((2, prior_turn.answer))
            return tuple(history)

        def _core_dialogue_answer(value: str) -> str | None:
            result = core_dialogue_runtime.respond(
                value,
                history=_core_dialogue_history(value),
                minimum_similarity_permille=(
                    PRODUCTION_MIN_SIMILARITY_PERMILLE),
            )
            if result is not None:
                return result.surface
            return (None if legacy_dialogue_answer is None
                    else legacy_dialogue_answer(value))

        def _core_dialogue_clarify(value: str) -> str | None:
            result = core_dialogue_runtime.respond(
                value,
                history=_core_dialogue_history(value),
                minimum_similarity_permille=900,
                minimum_margin_permille=100,
            )
            if result is not None:
                return result.surface
            return (None if legacy_dialogue_clarify is None
                    else legacy_dialogue_clarify(value))

        learned_dialogue_answer = _core_dialogue_answer
        learned_dialogue_clarify_answer = _core_dialogue_clarify
    if science_passage_artifact_root is not None:
        science_passage_runtime = ScidbCsqPassageRuntime(
            science_passage_artifact_root,
            verify_database_sha256=(
                performance_tier != "deferred-narrow-fast"),
        )

        def _source_passage_response(value: str) -> tuple[object, ...] | None:
            result = science_passage_runtime.query(
                value,
                minimum_confidence_permille=(
                    PRODUCTION_SOURCE_MIN_CONFIDENCE_PERMILLE),
                surface_variant_provider=surface_variant_provider)
            if result.status != "ANSWER" or result.surface is None:
                return None
            citation = DialogueCitation(
                result.surface,
                result.source_title,
                result.source_url,
                result.license_id,
                result.attribution,
                result.source_ref,
            )
            return (
                "ANSWER", result.surface, result.source_title,
                result.source_url, (citation,),
            )

        source_passage_response = _source_passage_response
    surface_consumer = None
    if trained_surface is not None:
        def _trained_surface_consumer(
                value: str, status: str, source_title: str | None,
                ) -> str | None:
            result = trained_surface.render(
                value, response_act=status, source_title=source_title)
            return result.surface if result.used else None

        surface_consumer = _trained_surface_consumer
    session_capability = None
    if session_root is not None:
        session_path = Path(session_root).resolve()
        if session_path.drive.upper() != "K:" or not session_path.is_dir():
            raise ValueError("session_root 必须是 K 盘已存在目录")
        if run_root_value is not None and session_path == Path(run_root_value).resolve():
            raise ValueError("session_root 不得与 training_run_root 相同")
        session_capability = open_existing_run_root(
            session_path, label="broad dialogue session root")
        ensure_normal_relative_directory(
            session_capability, "broad_dialogue_checkpoints",
            label="broad dialogue checkpoint directory")
    if performance_tier in {"deferred-narrow", "deferred-narrow-fast"}:
        # Broad retrieval is the common path for a published release.  Keep
        # the heavier sparse snapshot construction behind the first narrow
        # miss, while preserving exact strict-mode behavior and answer data.
        deferred_narrow = True
        narrow_runtime = None
        narrow_catalog = None
        narrow_answer = None

        def _lazy_narrow_answer(question: str):
            nonlocal narrow_runtime, narrow_catalog, narrow_answer
            if narrow_answer is None:
                if sparse_snapshot is None:
                    narrow_runtime = load_or_rebuild_public_sparse_qa_runtime()
                else:
                    narrow_runtime = load_or_rebuild_public_sparse_qa_runtime(
                        sparse_snapshot, repository=root)
                narrow_catalog = build_public_sentence_demo_catalog(narrow_runtime)
                narrow_answer = _narrow_answer(
                    narrow_runtime, narrow_catalog, trained_surface)
            return narrow_answer(question)

        narrow = _lazy_narrow_answer
    else:
        deferred_narrow = False
        if sparse_snapshot is None:
            sparse_runtime = load_or_rebuild_public_sparse_qa_runtime()
        else:
            sparse_runtime = load_or_rebuild_public_sparse_qa_runtime(
                sparse_snapshot, repository=root)
        narrow = _narrow_answer(
            sparse_runtime,
            build_public_sentence_demo_catalog(sparse_runtime),
            trained_surface,
        )
    if (release is not None
            and performance_tier == "deferred-narrow-fast"):
        # Public release processes pay the sparse snapshot construction once at
        # startup instead of charging it to the first user turn.  This keeps
        # the measured response path stable for long-lived sessions while the
        # persisted snapshot still bounds memory and work.
        narrow_runtime = load_or_rebuild_public_sparse_qa_runtime(
            sparse_snapshot, repository=root)
        narrow_catalog = build_public_sentence_demo_catalog(narrow_runtime)
        narrow_answer = _narrow_answer(
            narrow_runtime, narrow_catalog, trained_surface)
    stream_in = sys.stdin.buffer if input_stream is None else input_stream
    stream_out = sys.stdout.buffer if output_stream is None else output_stream
    state = BroadDialogueState((1, 1, 8))
    runtime_memory_state = None
    checkpoint_ordinal = 0
    checkpoint_identity: tuple[int, ...] = ()
    if session_capability is not None:
        checkpoint_dir = session_capability.path / "broad_dialogue_checkpoints"
        if any(item.is_file() and item.suffix == ".int"
               and item.name.startswith("checkpoint-")
               for item in checkpoint_dir.iterdir()):
            try:
                recovery = recover_broad_dialogue_checkpoint(
                    session_capability.path,
                    require_k_drive=not session_capability.test_transport,
                )
                dialogue_recovery = recovery
                state = recovery.checkpoint.state
                checkpoint_ordinal = recovery.checkpoint.ordinal
                checkpoint_identity = recovery.checkpoint_identity
                runtime_memory_state = recovery.checkpoint.runtime_memory_state
                if runtime_memory_state is None:
                    runtime_memory_state = replay_dialogue_state_to_runtime_memory(state)
            except (BroadDialoguePersistenceError,
                    BroadDialogueRuntimeMemoryError) as error:
                raise ValueError(f"会话检查点无法恢复: {error}") from error
        dialogue_memory_index = BroadDialogueMemoryRecallIndex(
            () if dialogue_recovery is None else dialogue_recovery.cold_turns)

    # Keep memory policy at the host boundary.  The broad QA core remains
    # language-agnostic; this callback only returns a surface backed by a
    # high-similarity persisted turn or by the learned dialogue runtime.
    if memory_recall_response is None and dialogue_memory_index is not None:
        def _memory_recall_response(value: str) -> str | None:
            return dialogue_memory_index.recall(value)
        memory_recall_response = _memory_recall_response
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    # release/index 连接在整个会话中只读；有界缓存让重复问题直接复用
    # 不可变结果，不再重复执行 SQLite 与排序路径。
    query_cache = BroadQaQueryCache(connection)
    query_cache.prepare()
    if source_followup_resolver is None:
        # Published runs get a data-driven resolver by default.  It only admits
        # a focus rewrite when the broad index itself returns an ANSWER tied to
        # the immediately preceding source; no language/代词 table lives here.
        source_followup_resolver = (
            build_index_evidence_source_followup_resolver(
                connection,
                query_cache=query_cache,
                learned_relation_evidence_model=(
                    learned_relation_evidence_model),
                surface_variant_provider=surface_variant_provider,
                fast_path=(performance_tier == "deferred-narrow-fast"),
            ))
    # 在首个用户请求计时前完成一次性规范问式加载。
    load_broad_qa_question_slots()
    latency_us: list[int] = []
    turn_statuses: list[str] = []
    sqlite_statement_counts: list[int] = []
    sqlite_statement_count = 0
    if metrics_output is not None:
        def _trace_statement(_sql: str) -> None:
            nonlocal sqlite_statement_count
            sqlite_statement_count += 1
        connection.set_trace_callback(_trace_statement)
    try:
        while True:
            if not protocol_stream:
                stream_out.write("你> ".encode("utf-8"))
                stream_out.flush()
            raw = stream_in.readline()
            if raw == b"":
                break
            if protocol_stream:
                request: dict[str, object] = {}
                try:
                    request = json.loads(raw.decode("utf-8"))
                    if not isinstance(request, dict):
                        raise ValueError("请求必须是 JSON 对象")
                    request_id = request.get("id")
                    operation = request.get("op", "turn")
                    if operation in {"quit", "exit"}:
                        _write_protocol_payload(stream_out, {
                            "id": request_id, "type": "bye",
                            "text": "会话已结束。",
                        })
                        break
                    if operation != "turn":
                        raise ValueError("op 必须是 turn、quit 或 exit")
                    question = request.get("text")
                    if type(question) is not str or not question.strip():
                        raise ValueError("turn.text 必须是非空字符串")
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                    _write_protocol_payload(stream_out, {
                        "id": request.get("id") if 'request' in locals()
                        and isinstance(request, dict) else None,
                        "type": "error",
                        "text": f"请求无法处理：{error}",
                    })
                    continue
            else:
                payload = raw.rstrip(b"\r\n")
                if payload in {b":quit", b":exit"}:
                    break
                try:
                    question = payload.decode("utf-8")
                except UnicodeDecodeError:
                    stream_out.write("系统> 输入必须是 UTF-8 文本。\n".encode("utf-8"))
                    stream_out.flush()
                    continue
            if not question.strip():
                continue
            runtime_response = runtime_material_response
            if runtime_material_response_provider is not None:
                provider = runtime_material_response_provider
                prior_state = state

                def _runtime_response_with_focus(
                        value: str,
                        *,
                        _provider=provider,
                        _prior_state=prior_state,
                        ):
                    exact = _provider.response_with_citations(value)
                    if exact is not None:
                        return exact
                    if _prior_state.turns:
                        previous = _prior_state.turns[-1]
                        if (previous.status == "ANSWER"
                                and previous.source_title):
                            followup = _provider.response_followup_with_citations(
                                value, previous.source_title)
                            if followup is not None:
                                return followup
                            related = _provider.response_related_with_citations(
                                value, previous.source_title)
                            if related is not None:
                                return related
                    related = _provider.response_related_with_citations(value)
                    if related is not None:
                        return related
                    return None

                runtime_response = _runtime_response_with_focus
            started_ns = time.perf_counter_ns()
            state, turn = answer_broad_dialogue_turn(
                state, question, connection, narrow_answer=narrow,
                defer_narrow=deferred_narrow,
                runtime_material_answer=runtime_material_answer,
                runtime_material_response=runtime_response,
                learned_relation_evidence_model=(
                    learned_relation_evidence_model),
                learned_relation_role_evidence_model=(
                    learned_relation_role_evidence_model),
                learned_relation_marker_evidence_model=(
                    learned_relation_marker_evidence_model),
                learned_relation_answer_frame_model=(
                    learned_relation_answer_frame_model),
                learned_dialogue_answer=learned_dialogue_answer,
                learned_dialogue_clarify_answer=(
                    learned_dialogue_clarify_answer),
                memory_recall_response=memory_recall_response,
                source_passage_response=source_passage_response,
                prefer_source_passage=(
                    source_passage_response is not None
                    and performance_tier == "deferred-narrow-fast"),
                prefer_learned_dialogue=(
                    learned_dialogue_answer is not None
                    and performance_tier == "deferred-narrow-fast"),
                surface_consumer=surface_consumer,
                query_cache=query_cache,
                surface_variant_provider=surface_variant_provider,
                source_followup_resolver=source_followup_resolver,
                fast_path=(performance_tier == "deferred-narrow-fast"))
            latency_us.append(max(0, (time.perf_counter_ns() - started_ns) // 1000))
            turn_statuses.append(turn.status)
            if metrics_output is not None:
                previous_count = sum(sqlite_statement_counts)
                sqlite_statement_counts.append(
                    sqlite_statement_count - previous_count)
            if session_capability is not None:
                try:
                    if runtime_memory_state is None:
                        runtime_memory_state = empty_runtime_memory_for_conversation(
                            state.conversation_key)
                    runtime_memory_state = append_dialogue_turn_to_runtime_memory(
                        runtime_memory_state, state.conversation_key, turn,
                    ).memory_after
                    _path, checkpoint_identity = append_broad_dialogue_checkpoint(
                        session_capability,
                        state,
                        previous_ordinal=checkpoint_ordinal,
                        previous_identity=checkpoint_identity,
                        runtime_memory_state=runtime_memory_state,
                    )
                    checkpoint_ordinal += 1
                    if dialogue_memory_index is not None:
                        dialogue_memory_index.append(turn)
                except (BroadDialoguePersistenceError,
                        BroadDialogueRuntimeMemoryError) as error:
                    raise ValueError(f"会话记忆写入失败: {error}") from error
            if protocol_stream:
                response = _public_protocol_turn_payload(turn)
                response["id"] = request_id
                _write_protocol_payload(stream_out, response)
            else:
                stream_out.write(("系统> " + _display(turn) + "\n").encode("utf-8"))
                stream_out.flush()
    finally:
        connection.close()
        if runtime_sqlite_runtime is not None:
            runtime_sqlite_runtime.close()
        if dialogue_response_runtime is not None:
            dialogue_response_runtime.close()
        if core_dialogue_runtime is not None:
            core_dialogue_runtime.close()
        if science_passage_runtime is not None:
            science_passage_runtime.close()
        if metrics_output is not None:
            metrics_path = Path(metrics_output).resolve()
            if metrics_path.drive.upper() != "K:":
                raise ValueError("metrics_output 必须是 K 盘路径")
            if not latency_us:
                raise ValueError("metrics_output 没有可记录的对话轮次")
            ordered = sorted(latency_us)
            p50 = _nearest_rank_latency_us(ordered, 50)
            p95 = _nearest_rank_latency_us(ordered, 95)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(
                json.dumps({
                "protocol": "jsonl" if protocol_stream else "terminal",
                "performance_tier": performance_tier,
                "turn_count": len(latency_us),
                    "latency_us": latency_us,
                    "p50_us": p50,
                    "p95_us": p95,
                    "max_us": ordered[-1],
                    "status_count": {
                        status: turn_statuses.count(status)
                        for status in sorted(set(turn_statuses))
                    },
                    "status_per_turn": turn_statuses,
                    "sqlite_statement_count_total": sqlite_statement_count,
                    "sqlite_statement_count_per_turn": sqlite_statement_counts,
                    "peak_working_set_bytes": _peak_working_set_bytes(),
                }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n",
                encoding="utf-8",
                newline="\n",
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="run read-only trained dialogue terminal")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--qa-database", default=None,
                        help="兼容旧入口；独立发布请使用 --release-root")
    parser.add_argument("--release-root", default=None,
                        help="K 盘自包含公开模型 release root")
    parser.add_argument("--training-run-root", default=None)
    parser.add_argument(
        "--response-organization-artifact-root", default=None,
        help="可选 K 盘回答结构后继 artifact root")
    parser.add_argument(
        "--dialogue-response-artifact-root", default=None,
        help="可选 K 盘人工对话聚合片段模型 artifact root")
    parser.add_argument(
        "--dialogue-domain-expert-artifact-root", action="append", default=[],
        help="可重复的 K 盘后置领域对话专家 artifact root")
    parser.add_argument(
        "--dialogue-expert-routing-artifact-root", default=None,
        help="K 盘纯整数领域激活路由文件")
    parser.add_argument(
        "--science-passage-artifact-root", default=None,
        help="可选 K 盘 CSQ train-only 来源知识段 artifact root")
    parser.add_argument("--session-root", default=None,
                        help="可选 K 盘会话根；启用后跨进程恢复最近 8 轮")
    parser.add_argument("--protocol", choices=("terminal", "jsonl"),
                        default="terminal",
                        help="交互入口：人类终端或 UTF-8 JSONL 协议")
    parser.add_argument("--metrics-output", default=None,
                        help="可选 K 盘性能摘要 JSON 路径")
    parser.add_argument(
        "--performance-tier", choices=(
            "strict", "deferred-narrow", "deferred-narrow-fast"),
        default="strict",
        help=("strict 完整校验并窄域优先；deferred-narrow 延迟构建窄域；"
              "deferred-narrow-fast 另跳过启动逐文件 SHA，仍保留 manifest "
              "闭合/大小检查，并使用有界首段证据快速路径"),
    )
    parser.add_argument("--extra-course", action="append", default=[])
    parser.add_argument("--variant-course", action="append", default=[])
    parser.add_argument("--variant-evidence", action="append", default=[])
    parser.add_argument("--runtime-material-ledger-root", default=None)
    parser.add_argument("--runtime-material-sqlite", default=None)
    parser.add_argument("--relation-evidence-course", action="append", default=[],
                        help="可选公开关系/证据课程，可重复")
    parser.add_argument("--relation-role-evidence-course", action="append",
                        default=[], help="可选公开 value/qualifier 课程")
    parser.add_argument("--relation-marker-evidence-course", action="append",
                        default=[], help="可选公开 marker/value 课程")
    parser.add_argument("--relation-answer-frame-course", action="append",
                        default=[], help="可选公开回答句面课程")
    args = parser.parse_args(argv)
    relation_courses = tuple(Path(item).resolve()
                             for item in args.relation_evidence_course)
    relation_role_courses = tuple(Path(item).resolve()
                                  for item in args.relation_role_evidence_course)
    relation_marker_courses = tuple(Path(item).resolve()
                                    for item in args.relation_marker_evidence_course)
    relation_frame_courses = tuple(Path(item).resolve()
                                   for item in args.relation_answer_frame_course)
    return run_trained_dialogue_terminal(
        project_root=args.project_root,
        qa_database=args.qa_database,
        release_root=args.release_root,
        training_run_root=args.training_run_root,
        response_organization_artifact_root=(
            args.response_organization_artifact_root),
        dialogue_response_artifact_root=(
            args.dialogue_response_artifact_root),
        dialogue_domain_expert_artifact_roots=tuple(
            args.dialogue_domain_expert_artifact_root),
        dialogue_expert_routing_artifact_root=(
            args.dialogue_expert_routing_artifact_root),
        science_passage_artifact_root=(
            args.science_passage_artifact_root),
        session_root=args.session_root,
        extra_course_paths=tuple(args.extra_course),
        extra_variant_course_paths=tuple(args.variant_course),
        extra_variant_evidence_paths=tuple(args.variant_evidence),
        runtime_material_runtime_root=args.runtime_material_ledger_root,
        runtime_material_runtime_database=args.runtime_material_sqlite,
        protocol_stream=(args.protocol == "jsonl"),
        metrics_output=args.metrics_output,
        performance_tier=args.performance_tier,
        learned_relation_evidence_model=(
            learn_relation_evidence_model(relation_courses)
            if relation_courses else None),
        learned_relation_role_evidence_model=(
            learn_relation_role_evidence_model(relation_role_courses)
            if relation_role_courses else None),
        learned_relation_marker_evidence_model=(
            learn_relation_marker_evidence_model(relation_marker_courses)
            if relation_marker_courses else None),
        learned_relation_answer_frame_model=(
            learn_relation_answer_frame_model(relation_frame_courses)
            if relation_frame_courses else None),
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "main", "run_trained_dialogue_terminal",
    "PRODUCTION_SOURCE_MIN_CONFIDENCE_PERMILLE",
]
