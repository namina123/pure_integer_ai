"""运行公开对话 pack 的第一条真实 formal_train 切片。

大输入与 checkpoint 只写显式 K 盘 run root。该入口默认只跑 observe/skeleton
阶段，先让真实 dialogue case 改变图状态；后续 reward 阶段由独立课程切片开启。
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
from pathlib import PurePosixPath
import shutil
import sys

from pure_integer_ai.config import gates
from pure_integer_ai.cognition.shared.identity import (
    concept_identity,
    minimal_instruction_identity,
)
from pure_integer_ai.cognition.shared.typed_binding import (
    BindingFailureProtocol,
    SubstitutionProtocol,
)
from pure_integer_ai.cognition.understanding.occurrence_index import (
    OccurrenceProtocol,
)
from pure_integer_ai.cognition.understanding.occurrence_order import (
    OccurrenceOrderProtocol,
)
from pure_integer_ai.cognition.understanding.segmentation_span import (
    SegmentationSpanProtocol,
)
from pure_integer_ai.cognition.understanding.span_index import SpanProtocol
from pure_integer_ai.cognition.understanding.semantic_builder import (
    SemanticBuilderProtocol,
)
from pure_integer_ai.experiments.conversation_training_pack import (
    load_dialogue_training_pack,
)
from pure_integer_ai.experiments.conversation_training_contrast import (
    build_dialogue_training_contrast,
)
from pure_integer_ai.experiments.formal_train import FormalTrainConfig, formal_train
from pure_integer_ai.experiments.conversation_training_cursor import (
    DialogueTrainingCursor,
    write_training_cursor,
)
from pure_integer_ai.experiments.dialogue_training_typed_adapter import (
    TypedDialogueCourseAdapter,
)
from pure_integer_ai.experiments.dialogue_successor_graph import (
    DialogueSuccessorProtocol,
)
from pure_integer_ai.experiments.corpus_identity import assign_corpus_source_refs
from pure_integer_ai.experiments.evaluation_protocol import (
    collected_item_content_identity,
)
from pure_integer_ai.experiments.typed_dialogue_semantic_course import (
    TypedDialogueSemanticMapper,
    TypedDialogueSemanticQueryMapper,
    build_typed_dialogue_semantic_protocol,
    build_typed_dialogue_semantic_query_protocol,
)
from pure_integer_ai.experiments.typed_dialogue_generation_owner import (
    TypedDialogueGenerationRuntimeFactory,
)
from pure_integer_ai.experiments.typed_dialogue_evaluation import (
    build_typed_dialogue_evaluation_bundle,
)
from pure_integer_ai.experiments.ph2_dataset_io import read_record_artifact
from pure_integer_ai.experiments.ph2_w09_contract import (
    make_w09_request,
    open_w09_frozen_contract,
)
from pure_integer_ai.experiments.ph2_w09_firewall import W09PayloadFirewall
from pure_integer_ai.experiments.ph2_w09_weaning import (
    W09DevCalibrationOwner,
    W09FrozenTeacherEvidenceSource,
    W09ShadowErrorAudit,
    W09TypedWeaningRuntime,
    make_w09_typed_weaning_protocol_from_contract,
    w09_commitment,
)
from pure_integer_ai.experiments.ph2_w05_contract import digest_value
from pure_integer_ai.storage.edge_store import EPI_STRUCTURED
from pure_integer_ai.storage.backend import SQLiteBackend
from pure_integer_ai.storage.dialogue_successor import (
    DIALOGUE_SUCCESSOR_FEATURE_TABLE,
    DIALOGUE_SUCCESSOR_TABLE,
)
from pure_integer_ai.storage.k_run_boundary import open_existing_run_root


def _peak_working_set_bytes() -> int:
    """返回训练宿主进程峰值工作集，作为性能记录而非能力信号。"""
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
        if get_info(get_process(), ctypes.byref(counters), counters.cb):
            return int(counters.peak_working_set_size)
        return 0
    try:
        import resource
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, AttributeError, OSError):
        return 0
    return value * (1024 if sys.platform != "darwin" else 1)


def default_course_paths(project_root: str | Path) -> tuple[Path, ...]:
    """返回当前仓库登记的公开 authored/dialogue 课程文件。"""
    root = Path(project_root).resolve() / "data" / "ph2"
    paths = sorted(root.glob("authored_*.jsonl.sample"))
    paths.extend(sorted(root.glob("dlg_raw_public_*course_v1.jsonl.sample")))
    paths.extend(sorted(root.glob("lc16_*_carrier_v1.jsonl.sample")))
    surface = root / "dlg_raw16_surface_organization_v1.jsonl.sample"
    if surface.is_file():
        paths.append(surface)
    return tuple(paths)


def _write_json(path: Path, value: object) -> None:
    """以单次创建写入紧凑、可回读的运行摘要。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                               separators=(",", ":")) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    """返回公开课程辅助 evidence 的文件摘要，绑定训练输入版本。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_CAMPAIGN_REQUIRED_STAGES = (1, 2, 3, 4)


def _resume_completed_stages(
        run_root: Path, resume_from: str | None, *, expected_pack_sha256: str,
        allow_additive_pack: bool = False,
        ) -> tuple[int, ...]:
    """回读同一 pack 的恢复谱系，取得已真实完成的阶段集合。

    ``FormalTrainResult`` 只描述当前进程请求的阶段。若把一个只跑 stage 4
    的恢复切片直接写成全局 ``weaning_ready``，此前未通过的 stage 2 会被
    遮蔽。因此发布级判断必须显式回读每一代已封存摘要，而不是从当前调用的
    ``active_stages`` 推断历史完成度。
    """
    if resume_from is None:
        return ()
    completed: set[int] = set()
    seen: set[Path] = set()
    current = resume_from
    while current is not None:
        if (not isinstance(current, str) or not current
                or Path(current).name != current):
            raise ValueError("resume_from 必须是当前 run root 内的单个 run identity")
        candidate = (run_root / current).resolve()
        try:
            candidate.relative_to(run_root)
        except ValueError as error:
            raise ValueError("resume_from 越出 run root") from error
        if candidate in seen:
            raise ValueError("resume_from 谱系出现循环")
        seen.add(candidate)
        summary_path = candidate / "training_summary.json"
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("resume_from 缺少可回读训练摘要") from error
        if (not isinstance(summary, dict)
                or summary.get("run_id") != current):
            raise ValueError("resume_from 训练身份或 pack SHA 漂移")
        pack_matches = summary.get("pack_sha256") == expected_pack_sha256
        # Additive shard runs may layer several public course packs over a
        # lineage (for example: base dialogue -> CAUSES shard -> full stage
        # pack).  The explicit opt-in permits those lineage SHA changes; all
        # ordinary resumes retain the exact-pack identity requirement.
        if not pack_matches and not allow_additive_pack:
            raise ValueError("resume_from 训练身份或 pack SHA 漂移")
        stages = summary.get("stages_completed")
        if (not isinstance(stages, list)
                or any(type(item) is not int or item < 1 for item in stages)):
            raise ValueError("resume_from stages_completed 非法")
        completed.update(stages)
        parent = summary.get("resume_from")
        if parent is not None and not isinstance(parent, str):
            raise ValueError("resume_from 谱系父节点非法")
        current = parent
    return tuple(sorted(completed))


def _resume_lineage_pack_namespace(
        run_root: Path, resume_from: str | None,
        ) -> str | None:
    """Return the oldest pack namespace in an explicit resume lineage."""
    if resume_from is None:
        return None
    seen: set[Path] = set()
    current = resume_from
    oldest: str | None = None
    while current is not None:
        if (not isinstance(current, str) or not current
                or Path(current).name != current):
            raise ValueError("resume_from 必须是当前 run root 内的单个 run identity")
        candidate = (run_root / current).resolve()
        try:
            candidate.relative_to(run_root)
        except ValueError as error:
            raise ValueError("resume_from 越出 run root") from error
        if candidate in seen:
            raise ValueError("resume_from 谱系出现循环")
        seen.add(candidate)
        try:
            summary = json.loads(
                (candidate / "training_summary.json").read_text(
                    encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("resume_from 缺少可回读训练摘要") from error
        if not isinstance(summary, dict) or summary.get("run_id") != current:
            raise ValueError("resume_from 训练身份非法")
        pack_sha = summary.get("pack_sha256")
        if not isinstance(pack_sha, str) or len(pack_sha) != 64:
            raise ValueError("resume_from pack SHA 非法")
        oldest = pack_sha
        parent = summary.get("resume_from")
        if parent is not None and not isinstance(parent, str):
            raise ValueError("resume_from 谱系父节点非法")
        current = parent
    return oldest


def _campaign_completion(
        *, prior_completed_stages: tuple[int, ...],
        local_completed_stages: tuple[int, ...],
        stage_weaning_ready: bool,
        stage_blockers: tuple[str, ...],
        ) -> tuple[tuple[int, ...], bool, tuple[str, ...]]:
    """把局部 Stage 结果投影为可发布训练 campaign 的就绪结论。"""
    cumulative = tuple(sorted(set(prior_completed_stages)
                              | set(local_completed_stages)))
    missing = tuple(
        stage for stage in _CAMPAIGN_REQUIRED_STAGES if stage not in cumulative)
    blockers = list(stage_blockers)
    blockers.extend(
        f"CAMPAIGN_STAGE_{stage}_INCOMPLETE" for stage in missing)
    return cumulative, bool(stage_weaning_ready and not missing), tuple(blockers)


def _typed_floor_summary(report) -> dict[str, object] | None:
    """将 typed floor 的可审计失败原因压缩进 K 盘运行摘要。

    训练摘要此前只保存阶段完成列表，floor 未通过时无法区分能力失败、
    评测无样本和协议/运行失败。这里仅保存整数计数、稳定协议键和失败原因，
    不写入 surface、label 或 private evaluator 内容。
    """
    if report is None:
        return None
    dimensions = []
    for item in getattr(report, "dimensions", ()):
        requirement = item.requirement
        dimensions.append({
            "dimension": list(requirement.dimension.stable_key()),
            "verifier": list(requirement.verifier.stable_key()),
            "minimum_match_permille": requirement.minimum_match_permille,
            "total": item.total,
            "matched": item.matched,
            "missing": item.missing,
            "operational_failure": item.operational_failure,
            "match_permille": item.match_permille,
            "complete": item.complete,
        })
    cases = []
    for item in getattr(report, "cases", ()):
        episode = item.episode
        cases.append({
            "identity_sha256": item.identity.content.sha256,
            "failure": item.failure,
            "episode_present": episode is not None,
            "signal_count": 0 if episode is None else len(episode.signals),
            "signal_keys": (() if episode is None else [
                {
                    "dimension": list(signal.dimension.stable_key()),
                    "verifier": list(signal.verifier.stable_key()),
                    "applicability": signal.applicability,
                    "verdict": signal.verdict,
                    "operational_failure": signal.operational_failure,
                }
                for signal in episode.signals
            ]),
        })
    return {
        "protocol_version": report.protocol_version,
        "split": list(report.split.stable_key()),
        "measured": report.measured,
        "complete": report.complete,
        "unexpected_dimensions": getattr(report, "unexpected_dimensions", 0),
        "dimensions": dimensions,
        "cases": cases,
    }


def _dialogue_semantic_protocol():
    """建立公开对话 typed 课程使用的最小 S-02/S-03 整数协议。"""
    namespace = (21402, 1)
    predicates = tuple(
        concept_identity((*namespace, 10, ordinal))
        for ordinal in range(1, 12)
    )
    failures = BindingFailureProtocol(*tuple(
        minimal_instruction_identity((*namespace, 20, ordinal))
        for ordinal in range(1, 10)
    ))
    semantic = build_typed_dialogue_semantic_protocol(
        TypedDialogueSemanticMapper((21402, 30), 1),
        builder_identity=SemanticBuilderProtocol(
            minimal_instruction_identity((*namespace, 40, 1)),
            (*namespace, 40, 3)),
        atomic_predicates=predicates[:6],
        trace_predicates=predicates[6:9],
        scope_predicates=predicates[9:],
        substitution=SubstitutionProtocol(
            minimal_instruction_identity((*namespace, 40, 2)), failures),
        provenance_kind=EPI_STRUCTURED,
    )
    occurrence = OccurrenceProtocol(
        (*namespace, 50, 1), (*namespace, 50, 2))
    span = SegmentationSpanProtocol(
        SpanProtocol(
            (*namespace, 51, 1), (*namespace, 51, 2),
            (*namespace, 51, 3), (*namespace, 51, 4)),
        (*namespace, 52, 1), (*namespace, 52, 2), (*namespace, 52, 3),
        (*namespace, 52, 4),
    )
    return semantic, occurrence, span


def dialogue_semantic_protocols():
    """返回公开对话 Runtime 可复用的 S-02/S-03/L-03/L-04 协议。

    训练入口与 Runtime 资料入口必须共享同一组整数协议身份；此公开包装避免
    Runtime 调用方复制私有 namespace，且不携带任何训练状态。
    """
    return _dialogue_semantic_protocol()


def _dialogue_semantic_query_protocol():
    """建立只读评测使用的 typed candidate recovery mapper。"""
    return build_typed_dialogue_semantic_query_protocol(
        TypedDialogueSemanticQueryMapper((21402, 30)))


def _dialogue_successor_protocol() -> DialogueSuccessorProtocol:
    """建立普通结构化对话共享的后继图与 H-00 整数协议。"""
    return DialogueSuccessorProtocol((21402, 60), EPI_STRUCTURED)


def dialogue_semantic_query_protocol():
    """返回公开 Runtime 只读查询使用的 typed semantic protocol。"""
    return _dialogue_semantic_query_protocol()


def _w09_dev_observations(project_root: Path, context) -> tuple[object, ...]:
    """只读载入冻结 W-09 dev Observation，不接触 evaluator labels。"""
    records = []
    for binding in context.dev_bindings:
        if binding.identity.owner_kind != "observation":
            continue
        target = (project_root / Path(*PurePosixPath(
            binding.relative_path).parts)).resolve()
        local_parts = PurePosixPath(binding.identity.relative_path).parts
        artifact_root = target.parents[len(local_parts) - 1]
        records.extend(read_record_artifact(artifact_root, binding.identity))
    if not records:
        raise RuntimeError("W-09 dev Observation 为空")
    return tuple(records)


def _build_w09_builder(project_root: Path):
    """创建动态 W-09 builder，candidate/input 均来自当前真实运行。"""
    context = open_w09_frozen_contract(project_root)
    payload = W09PayloadFirewall.open(
        project_root,
        context,
        make_w09_request(context),
    ).read_training_payload()
    training_source = W09FrozenTeacherEvidenceSource(context, payload)
    dev_owner = W09DevCalibrationOwner(
        context,
        _w09_dev_observations(project_root, context),
    )
    shadow_auditor = W09ShadowErrorAudit()
    input_commitment = w09_commitment(payload.training_evidence)

    def build(_ctx: object, stage4_report: object):
        candidate_identity = w09_commitment(stage4_report)
        protocol = make_w09_typed_weaning_protocol_from_contract(
            context,
            candidate_identity=candidate_identity,
            input_commitment=input_commitment,
            threshold_key=digest_value((
                "W09_TYPED_DIALOGUE_THRESHOLD",
                candidate_identity,
                input_commitment,
            )),
        )
        runtime = W09TypedWeaningRuntime(
            protocol,
            training_material_source=training_source,
            dev_calibrator=dev_owner,
            shadow_auditor=shadow_auditor,
            frozen_contract=context,
        )
        return protocol, runtime

    return build


def run_conversation_training(*, project_root: str | Path,
                              run_root: str | Path,
                              run_id: str = "dialogue-pack-v1",
                              active_stages: tuple[int, ...] = (1,),
                              resume_from: str | None = None,
                              max_cases: int | None = None,
                              with_heldout_probe: bool = False,
                              causal_only: bool = False,
                              typed_semantic: bool = True,
                              extra_course_paths: tuple[str | Path, ...] = (),
                              portable_source_identity: bool = False,
                              include_default_courses: bool = True,
                              allow_additive_resume_pack: bool = False,
                              typed_language_stage_items_only: bool = False,
                              replay_completed_stages: bool = False,
                              storage_performance_mode: str = "durable",
                              sqlite_page_resume: bool = False,
                              typed_language_diagnostic_only: bool = False,
                              ) -> dict[str, object]:
    """消费公开 train split，并产出真实 SQLite graph/checkpoint 摘要。"""
    if max_cases is not None and (type(max_cases) is not int or max_cases <= 0):
        raise ValueError("max_cases 必须是正整数")
    if storage_performance_mode not in {"durable", "bulk"}:
        raise ValueError(
            "storage_performance_mode 必须是 durable 或 bulk")
    if allow_additive_resume_pack:
        if resume_from is None:
            raise ValueError("增量 pack 恢复必须指定 resume_from")
        if not extra_course_paths:
            raise ValueError("增量 pack 恢复必须提供新增公开课程")
    root = Path(run_root).resolve()
    if root.drive.upper() != "K:" or not root.is_dir():
        raise ValueError("run_root 必须是已存在的 K 盘目录")
    default_paths = default_course_paths(project_root) if include_default_courses else ()
    paths = tuple(default_paths) + tuple(
        Path(item).resolve() for item in extra_course_paths)
    if len(paths) != len(set(paths)):
        raise ValueError("extra course path 与默认课程重复")
    source_identities = (
        {path: f"data/ph2/{path.name}" for path in paths}
        if portable_source_identity else None)
    pack = load_dialogue_training_pack(
        paths, max_cases=max_cases, source_path_identities=source_identities)
    source_namespace = pack.pack_sha256
    if allow_additive_resume_pack:
        lineage_namespace = _resume_lineage_pack_namespace(root, resume_from)
        if lineage_namespace is None:
            raise ValueError("增量 pack 恢复缺少基座命名空间")
        # Reuse the oldest namespace so inherited typed items resolve to the
        # graph already present in the checkpoint.  New shard items remain
        # distinguished by their source/content identity and manifest SHA.
        source_namespace = lineage_namespace
    prior_completed_stages = _resume_completed_stages(
        root, resume_from, expected_pack_sha256=pack.pack_sha256,
        allow_additive_pack=allow_additive_resume_pack)
    typed_report = TypedDialogueCourseAdapter().report(pack.cases)
    strict_bundle = None
    if typed_semantic and any(stage >= 3 for stage in active_stages):
        all_items = pack.items_for_split(split=None, causal_only=causal_only)
        assign_corpus_source_refs(all_items, source_namespace=source_namespace)
        by_case = {
            case.case_id: item for case, item in zip(pack.cases, all_items)
        }
        strict_bundle = build_typed_dialogue_evaluation_bundle(pack, by_case)
        assignments = strict_bundle.evaluation_plan.assignments
        training_keys = {
            item.identity.lookup_key()
            for item in assignments
            if item.split == strict_bundle.evaluation_plan.protocol.training_split
        }
        train_items = [
            item for item in strict_bundle.corpus
            if item.source_ref is not None and (
                item.source_ref.stable_key(),
                collected_item_content_identity(item).payload,
            ) in training_keys
        ]
        heldout_items = [
            item for item in strict_bundle.corpus
            if item not in train_items
        ]
    else:
        # Long indexed courses keep only raw_text + integer reference in the
        # corpus; formal_train materializes one item for one round and releases
        # it afterwards. Stage >=3 strict evaluation keeps its eager contract.
        defer_indexed_surface = not any(stage >= 3 for stage in active_stages)
        train_items = pack.training_items(
            causal_only=causal_only,
            defer_indexed_surface=defer_indexed_surface)
        heldout_items = pack.training_items(
            split="heldout", causal_only=causal_only,
            defer_indexed_surface=defer_indexed_surface)
    contrast = build_dialogue_training_contrast(pack)
    semantic_protocol = occurrence_protocol = span_protocol = None
    semantic_query_protocol = None
    dialogue_successor_protocol = None
    if typed_semantic:
        (semantic_protocol, occurrence_protocol,
         span_protocol) = _dialogue_semantic_protocol()
        semantic_query_protocol = _dialogue_semantic_query_protocol()
        if any(case.dialogue_turns for case in pack.cases):
            dialogue_successor_protocol = _dialogue_successor_protocol()
    # A shard may contain only relation/structure items while its SQLite
    # checkpoint base already carries the optional dialogue successor tables.
    # Register the same schema before page-resume validation so an additive
    # recovery does not fail merely because this shard has no dialogue turns.
    if allow_additive_resume_pack and dialogue_successor_protocol is None:
        dialogue_successor_protocol = _dialogue_successor_protocol()
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    project_root_path = Path(project_root).resolve()
    surface_evidence_path = (
        project_root_path / "data" / "ph2"
        / "dlg_raw16_surface_slot_evidence_v1.jsonl.sample")
    if not surface_evidence_path.is_file():
        raise ValueError("DLG-RAW-16 surface evidence 缺失")
    # A portable run records only release-relative identities in its nested
    # manifest.  The loader still reads the resolved host paths above, while
    # the persisted identity remains reproducible after copying the run into
    # an independent release root.
    def manifest_identity(path: str | Path) -> str:
        resolved = Path(path).resolve()
        if not portable_source_identity:
            return resolved.as_posix()
        identity = source_identities.get(resolved) if source_identities else None
        if identity is not None:
            return identity
        if resolved == surface_evidence_path:
            return "data/ph2/" + resolved.name
        return "data/ph2/" + resolved.name

    if portable_source_identity:
        # A portable manifest is only useful when its recorded source files
        # travel with the run.  Copy the immutable public inputs once, then
        # let resume resolve the relative identities from run_dir.  This is
        # intentionally a byte-for-byte stdlib copy; the manifest still
        # carries the source SHA and pack SHA as the semantic identity.
        portable_root = run_dir / "data" / "ph2"
        portable_root.mkdir(parents=True, exist_ok=True)
        portable_sources = tuple(paths) + (surface_evidence_path,)
        copied: dict[Path, Path] = {}
        for source in portable_sources:
            resolved = Path(source).resolve()
            if not resolved.is_file():
                raise ValueError(f"portable training source 缺失: {resolved}")
            destination_identity = manifest_identity(resolved)
            relative = PurePosixPath(destination_identity)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("portable training source identity 非法")
            destination = (run_dir / Path(*relative.parts)).resolve()
            try:
                destination.relative_to(run_dir)
            except ValueError as error:
                raise ValueError("portable training source 越出 run root") from error
            previous = copied.get(destination)
            if previous is not None:
                if previous != resolved:
                    raise ValueError("portable training source basename 冲突")
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(resolved, destination)
            if _sha256_file(destination) != _sha256_file(resolved):
                raise ValueError("portable training source copy SHA 漂移")
            copied[destination] = resolved

    manifest_source_files = [
        [manifest_identity(row[0]), row[1], row[2]]
        for row in pack.source_files
    ]
    _write_json(run_dir / "dialogue_pack_manifest.json", {
        "protocol": 1,
        "pack_sha256": pack.pack_sha256,
        "source_namespace": source_namespace,
        "source_files": manifest_source_files,
        "case_count": len(pack.cases),
        "max_cases": max_cases,
        "split_counts": pack.split_counts,
        "dialogue_structure": pack.dialogue_structure_counts,
        "train_surface_count": len(train_items),
        "heldout_surface_count": len(heldout_items),
        "typed_course": typed_report.to_dict(),
        "causal_only": causal_only,
        "extra_course_paths": tuple(
            manifest_identity(item) for item in extra_course_paths),
        "resume_pack_mode": (
            "additive_shard" if allow_additive_resume_pack else "exact"),
        "typed_language_stage_items_only": bool(
            typed_language_stage_items_only),
        "surface_evidence_files": ((
            manifest_identity(surface_evidence_path),
            _sha256_file(surface_evidence_path),
        ),),
    })
    _write_json(run_dir / "contrast_report.json", contrast.to_dict())
    database_path = run_dir / "training.sqlite3"
    sqlite_resume_binding_sha256 = None
    if sqlite_page_resume:
        if resume_from is None:
            raise ValueError("sqlite_page_resume 必须指定 resume_from")
        from pure_integer_ai.experiments.sqlite_training_resume import (
            prepare_sqlite_page_resume,
        )
        sqlite_binding = prepare_sqlite_page_resume(
            root / resume_from, database_path)
        sqlite_resume_binding_sha256 = sqlite_binding.manifest_sha256
    backend = SQLiteBackend(
        str(database_path), performance_mode=storage_performance_mode)
    corpus = (
        list(strict_bundle.corpus)
        if strict_bundle is not None
        else train_items + heldout_items if with_heldout_probe else train_items
    )
    if typed_semantic:
        # L-03 occurrence identity needs one corpus-wide ordinal pass; doing
        # this lazily per item would collapse repeated surfaces to one source.
        assign_corpus_source_refs(corpus, source_namespace=source_namespace)
    generation_factory = (
        TypedDialogueGenerationRuntimeFactory.from_project_root(project_root)
        if typed_semantic else None
    )
    w09_builder = (
        _build_w09_builder(Path(project_root).resolve())
        if typed_semantic and 4 in active_stages else None
    )
    previous = gates.TRAINING_MODE
    dialogue_successor_counts = (0, 0)
    try:
        gates.TRAINING_MODE = True
        result = formal_train(
            FormalTrainConfig(
                run_dir=str(root),
                run_id=run_id,
                rounds_per_stage=1,
                active_training_stages=active_stages,
                replay_completed_stages=replay_completed_stages,
                resume=resume_from is not None,
                base_run_id=resume_from,
                sqlite_resume_binding_sha256=(
                    sqlite_resume_binding_sha256),
                typed_language_diagnostic_only=(
                    typed_language_diagnostic_only),
                probe_holdout=(
                    0 if strict_bundle is not None
                    else len(heldout_items) if with_heldout_probe else 0),
                probe_version=(
                    0 if strict_bundle is not None
                    else 1 if with_heldout_probe else 0),
                persist_graph_dump=True,
                evaluation_plan=(
                    None if strict_bundle is None
                    else strict_bundle.evaluation_plan),
                language_generation_h2_protocol=(
                    None if strict_bundle is None
                    else strict_bundle.h2_protocol),
                language_generation_floor_protocol=(
                    None if strict_bundle is None
                    else strict_bundle.floor_protocol),
                language_occurrence_protocol=occurrence_protocol,
                language_occurrence_order_protocol=(
                    None if occurrence_protocol is None
                    else OccurrenceOrderProtocol((21402, 1, 50, 3))),
                language_span_protocol=span_protocol,
                language_semantic_course_protocol=semantic_protocol,
                language_semantic_query_protocol=semantic_query_protocol,
                language_dialogue_successor_protocol=(
                    dialogue_successor_protocol),
                language_generation_runtime_factory=generation_factory,
                w09_weaning_builder=w09_builder,
                w09_execute_zero_call_windows=w09_builder is not None,
                typed_language_stage_items_only=(
                    typed_language_stage_items_only),
            ),
            corpus,
            backend=backend,
        )
        if dialogue_successor_protocol is not None:
            dialogue_successor_counts = (
                backend.count(DIALOGUE_SUCCESSOR_TABLE),
                backend.count(DIALOGUE_SUCCESSOR_FEATURE_TABLE),
            )
    finally:
        gates.TRAINING_MODE = previous
        backend.commit()
        backend.close()
    if typed_language_diagnostic_only:
        summary = {
            "run_id": run_id,
            "pack_sha256": pack.pack_sha256,
            "source_namespace": source_namespace,
            "case_count": len(pack.cases),
            "max_cases": max_cases,
            "split_counts": pack.split_counts,
            "training_item_count": len(train_items),
            "heldout_probe_count": len(heldout_items),
            "typed_course": typed_report.to_dict(),
            "typed_language_h2": _typed_floor_summary(
                result.typed_language_h2_report),
            "typed_language_floor": _typed_floor_summary(
                result.typed_language_floor_report),
            "active_stages": active_stages,
            "resume_from": resume_from,
        "resume_pack_mode": (
            "additive_shard" if allow_additive_resume_pack else "exact"),
        "typed_language_stage_items_only": bool(
            typed_language_stage_items_only),
        "diagnostic_only": True,
            "storage_performance_mode": storage_performance_mode,
            "sqlite_page_resume": bool(sqlite_page_resume),
            "sqlite_resume_source_manifest_sha256": (
                sqlite_resume_binding_sha256),
            "peak_working_set_bytes": _peak_working_set_bytes(),
            "database": str(database_path),
            "execution": result.execution.to_json(),
        }
        _write_json(run_dir / "training_summary.json", summary)
        return summary
    local_completed_stages = tuple(sorted(result.stages_completed))
    stage_weaning_ready = bool(result.weaning_ready)
    (cumulative_completed_stages,
     campaign_weaning_ready,
     campaign_blockers) = _campaign_completion(
         prior_completed_stages=prior_completed_stages,
         local_completed_stages=local_completed_stages,
         stage_weaning_ready=stage_weaning_ready,
         stage_blockers=tuple(result.weaning_blockers),
     )
    cursor = DialogueTrainingCursor(
        tuple(bytes.fromhex(pack.pack_sha256)),
        tuple(run_id.encode("utf-8")),
        tuple(sorted(active_stages)),
        local_completed_stages,
        len(train_items),
        len(heldout_items) if with_heldout_probe else 0,
        int(result.final_metrics.graph_size),
        campaign_weaning_ready,
    )
    cursor_path = write_training_cursor(
        open_existing_run_root(run_dir, require_k_drive=True),
        cursor,
    )
    summary = {
        "run_id": run_id,
        "pack_sha256": pack.pack_sha256,
        "source_namespace": source_namespace,
        "case_count": len(pack.cases),
        "max_cases": max_cases,
        "split_counts": pack.split_counts,
        "dialogue_structure": pack.dialogue_structure_counts,
        "training_item_count": len(train_items),
        "heldout_probe_count": len(heldout_items) if with_heldout_probe else 0,
        "typed_course": typed_report.to_dict(),
        "typed_language_floor": _typed_floor_summary(
            result.typed_language_floor_report),
        "causal_only": causal_only,
        "typed_semantic": typed_semantic,
        "lang_generalization": None if result.lang_generalization is None else {
            "total_held_out": result.lang_generalization.total_held_out,
            "recognized": result.lang_generalization.recognized,
            "verified": result.lang_generalization.verified,
            "lang_rate_permille": result.lang_generalization.lang_rate_permille,
        },
        "active_stages": active_stages,
        "resume_from": resume_from,
        "resume_pack_mode": (
            "additive_shard" if allow_additive_resume_pack else "exact"),
        "typed_language_stage_items_only": bool(
            typed_language_stage_items_only),
        "stages_completed": local_completed_stages,
        "cumulative_stages_completed": cumulative_completed_stages,
        "campaign_required_stages": _CAMPAIGN_REQUIRED_STAGES,
        "stage_weaning_ready": stage_weaning_ready,
        "weaning_ready": campaign_weaning_ready,
        "weaning_blockers": campaign_blockers,
        "storage_performance_mode": storage_performance_mode,
        "sqlite_page_resume": bool(sqlite_page_resume),
        "sqlite_resume_source_manifest_sha256": (
            sqlite_resume_binding_sha256),
        "storage_write_calls": getattr(backend, "storage_write_calls", 0),
        "storage_write_rows": getattr(backend, "storage_write_rows", 0),
        "occurrence_count": int(result.occurrence_count),
        "source_record_count": int(result.source_record_count),
        "occurrence_order_fact_count": int(
            result.occurrence_order_fact_count),
        "dialogue_successor_count": dialogue_successor_counts[0],
        "dialogue_successor_feature_count": dialogue_successor_counts[1],
        "peak_working_set_bytes": _peak_working_set_bytes(),
        "database": str(database_path),
        "training_cursor": str(cursor_path),
        "training_cursor_identity": list(cursor.identity()),
    }
    _write_json(run_dir / "training_summary.json", summary)
    from pure_integer_ai.experiments.sqlite_training_resume import (
        publish_sqlite_training_resume,
    )
    publish_sqlite_training_resume(run_dir)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run public dialogue training slice")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--run-id", default="dialogue-pack-v1")
    parser.add_argument("--stages", default="1",
                        help="comma-separated formal_train stages")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--max-cases", type=int, default=None,
                        help="只读取公开课程的确定性前缀，用于有界恢复/关系分片")
    parser.add_argument("--with-heldout-probe", action="store_true")
    parser.add_argument("--causal-only", action="store_true")
    parser.add_argument("--no-typed-semantic", action="store_true",
                        help="仅运行旧 observe 图；默认接入 typed S-02 课程")
    parser.add_argument("--extra-course", action="append", default=[],
                        help="可选的公开课程 JSONL；不改变默认 v6 pack")
    parser.add_argument("--portable-source-identity", action="store_true",
                        help="用 data/ph2 basename 绑定 pack，允许跨机器恢复")
    parser.add_argument("--no-default-courses", action="store_true",
                        help="增量 shard 只消费显式 extra course")
    parser.add_argument(
        "--allow-additive-resume-pack", action="store_true",
        help=("允许基于旧 run 的 SQLite checkpoint 消费新增课程 shard；"
              "必须同时使用 --resume-from 和 --extra-course"),
    )
    parser.add_argument(
        "--typed-language-stage-items-only", action="store_true",
        help="Stage 3/4 仅消费显式 typed 语言课程；Stage 1/2 仍保留全量观察",
    )
    parser.add_argument("--replay-completed-stages", action="store_true",
                        help="E1 恢复后显式重放 active stage，供增量 shard 使用")
    parser.add_argument(
        "--storage-performance-mode", choices=("durable", "bulk"),
        default="durable",
        help=("SQLite 训练存储档位；durable 默认保崩溃恢复，bulk 仅用于可重建 "
              "训练 run 并降低同步写开销"),
    )
    parser.add_argument(
        "--sqlite-page-resume", action="store_true",
        help="从已发布 SQLite checkpoint 做 page-copy 快速续训")
    parser.add_argument(
        "--typed-language-diagnostic-only", action="store_true",
        help="只读执行 typed H2/floor，跳过 boot/discovery/训练/dump")
    args = parser.parse_args(argv)
    summary = run_conversation_training(
        project_root=args.project_root,
        run_root=args.run_root,
        run_id=args.run_id,
        active_stages=tuple(int(item) for item in args.stages.split(",") if item),
        resume_from=args.resume_from,
        max_cases=args.max_cases,
        with_heldout_probe=args.with_heldout_probe,
        causal_only=args.causal_only,
        typed_semantic=not args.no_typed_semantic,
        extra_course_paths=tuple(args.extra_course),
        portable_source_identity=args.portable_source_identity,
        include_default_courses=not args.no_default_courses,
        allow_additive_resume_pack=args.allow_additive_resume_pack,
        typed_language_stage_items_only=args.typed_language_stage_items_only,
        replay_completed_stages=args.replay_completed_stages,
        storage_performance_mode=args.storage_performance_mode,
        sqlite_page_resume=args.sqlite_page_resume,
        typed_language_diagnostic_only=args.typed_language_diagnostic_only,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "default_course_paths",
    "dialogue_semantic_protocols",
    "dialogue_semantic_query_protocol",
    "main",
    "run_conversation_training",
]
