"""W02 V4-first R6 的标准七文件 I/O、评测与 PASS receipt。"""
from __future__ import annotations

import gzip
import hashlib
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator

from pure_integer_ai.experiments.ph2_dataset_contract import (
    EvaluatorLabelRecord,
    ObservationRecord,
    SourceRefRecord,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    read_canonical_object,
    write_immutable_json,
)
from pure_integer_ai.experiments.ph2_d03_v2_blind_private_source_extension_v7 import (
    blind_private_source_specs_v7,
    validate_blind_private_owner_record_v7,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_contract import (
    V2EvaluatorBoundaryContract,
    V2EvaluatorResourceBudget,
    V2PrivateFamilyRegistration,
    validate_v2_safe_report,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_firewall import (
    V2AccessPermit,
    V2AccessRequest,
    V2PhysicalRoots,
    V2WriteAccount,
    authorize_v2_access,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_base_language_family_adapter import (
    W02_BASE_LANGUAGE_FAMILY_ADAPTER_VERSION,
    predict_w02_dev_observation_language_family,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_candidate_store import (
    open_w02_candidate_predictor,
    read_w02_candidate_artifact,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_compiler import _dimension_key
from pure_integer_ai.experiments.ph2_d03_v2_w02_dev_calibration import (
    W02_DEV_DIMENSIONS,
    _dimension_report,
    _evaluate_pair,
    _hash_value,
    _tree_sha256,
    load_w02_dev_candidate_index,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_overlay import (
    load_w02_morphology_overlay_index,
    read_w02_morphology_overlay_artifact,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor import (
    W02MorphologyRankingCache,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_dev_calibration import (
    _requested_spans,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v2 import (
    W02MorphologySuccessorV2Cache,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v2_overlay import (
    load_w02_morphology_successor_v2_overlay_index,
    read_w02_morphology_successor_v2_overlay_artifact,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_evaluator import (
    W02_MORPH_V3_PRIVATE_SUPPORT_KEYS,
    _audit_v3_extension,
    _result_status,
    _support_result,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_evaluator_r3 import (
    _apply_label_fail_closed,
    _prepare_label_stream,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_route import (
    W02MorphologySourceCapability,
    authorize_w02_morphology_source_routes,
    build_w02_morphology_routed_indexes,
    predict_w02_morphology_successor_v3,
    w02_ud_morphology_source_capability,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v4_artifact import (
    read_w02_morphology_successor_v4_artifact,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v4_language_overlay import (
    predict_w02_morphology_successor_v4,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v4_private_r6_protocol import (
    W02_MORPH_V4_PRIVATE_R6_LAYOUTS,
    W02_MORPH_V4_PRIVATE_R6_PATHS,
    W02_MORPH_V4_PRIVATE_R6_PROTOCOL_PATH,
    W02_MORPH_V4_PRIVATE_R6_SPLITS,
    W02MorphologySuccessorV4PrivateR6FileIdentity,
    read_w02_morphology_successor_v4_private_r6_protocol_freeze,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v4_r6_feasibility_receipt import (
    W02_MORPH_V4_R6_SPLIT_COUNTS,
)


W02_MORPH_V4_PRIVATE_R6_EVALUATOR_VERSION = (
    "PH2-D03-V2-W02-MORPHOLOGY-SUCCESSOR-V4-PRIVATE-R6-EVALUATOR-V1"
)
W02_MORPH_V4_PRIVATE_R6_RECEIPT_VERSION = (
    "PH2-D03-V2-W02-MORPHOLOGY-SUCCESSOR-V4-R6-RUNTIME-RECEIPT-V1"
)
W02_MORPH_V4_PRIVATE_R6_RECEIPT_PATH = (
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_morphology_successor_v4_r6_runtime_receipt_v1.json"
)


# object-model: exception
class W02MorphologySuccessorV4PrivateR6RuntimeError(RuntimeError):
    """R6 I/O、V4 推理、资源、aggregate 或 receipt 漂移。"""


def _sha256_file(path: Path) -> tuple[int, str]:
    """流式计算一个文件的长度与 SHA-256。"""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


def _file_by_layout(
        files: tuple[W02MorphologySuccessorV4PrivateR6FileIdentity, ...],
        layout_key: str,
        ) -> W02MorphologySuccessorV4PrivateR6FileIdentity:
    """从冻结 inventory 中取得唯一 layout。"""
    matches = tuple(row for row in files if row.layout_key == layout_key)
    if len(matches) != 1:
        raise W02MorphologySuccessorV4PrivateR6RuntimeError(
            "R6 private layout 非唯一")
    return matches[0]


def _split_layout(split: str, kind: str) -> str:
    """把 split/kind 映射为冻结 layout key。"""
    if split not in W02_MORPH_V4_PRIVATE_R6_SPLITS or kind not in {
            "observation", "label"}:
        raise W02MorphologySuccessorV4PrivateR6RuntimeError(
            "R6 split/kind 未注册")
    suffix = "OBSERVATION" if kind == "observation" else "LABEL"
    return f"PRIVATE_{split.upper()}_{suffix}"


def authorize_w02_morphology_successor_v4_private_r6_files(
        boundary: V2EvaluatorBoundaryContract,
        roots: V2PhysicalRoots,
        registration: V2PrivateFamilyRegistration,
        files: tuple[W02MorphologySuccessorV4PrivateR6FileIdentity, ...],
        ) -> dict[str, V2AccessPermit]:
    """将 R6 七文件的 transport 身份接入冻结 firewall。"""
    if tuple(row.layout_key for row in files) != W02_MORPH_V4_PRIVATE_R6_LAYOUTS:
        raise W02MorphologySuccessorV4PrivateR6RuntimeError(
            "R6 private inventory 漂移")
    permits = {}
    for layout_key in W02_MORPH_V4_PRIVATE_R6_LAYOUTS:
        identity = _file_by_layout(files, layout_key)
        request = V2AccessRequest(
            "W-02", "PH2_V2_PRIVATE_EVALUATOR",
            identity.split or "held_out", identity.record_kind,
            W02_MORPH_V4_PRIVATE_R6_PATHS[layout_key],
            identity.transport_sha256, identity.transport_size_bytes,
            "PRIVATE_EVALUATION", registration.candidate_freeze_sha256,
            registration.code_freeze_sha256, V2WriteAccount())
        permits[layout_key] = authorize_v2_access(
            boundary, roots, request, registration=registration)
    return permits


def iter_w02_morphology_successor_v4_private_r6_records(
        identity: W02MorphologySuccessorV4PrivateR6FileIdentity,
        permit: V2AccessPermit,
        ) -> Iterator[object]:
    """读取一个获准 gzip，并闭合 canonical、内容与 transport 身份。"""
    if (permit.root_key != "PRIVATE_EVALUATOR_ROOT"
            or permit.record_kind != identity.record_kind
            or permit.content_sha256 != identity.transport_sha256
            or permit.content_size_bytes != identity.transport_size_bytes):
        raise W02MorphologySuccessorV4PrivateR6RuntimeError(
            "R6 private permit 与 identity 不匹配")
    digest = hashlib.sha256()
    size = 0
    count = 0
    first = last = previous = None
    try:
        with permit.target_path.open("rb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="rb") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.endswith(b"\n") or line.endswith(b"\n\n"):
                        raise W02MorphologySuccessorV4PrivateR6RuntimeError(
                            f"R6 JSONL 第 {line_number} 行换行漂移")
                    digest.update(line)
                    size += len(line)
                    value = parse_canonical_json_bytes(
                        line[:-1], require_object=True)
                    assert isinstance(value, dict)
                    record = validate_blind_private_owner_record_v7(value)
                    if getattr(record, "RECORD_KIND", None) != identity.record_kind:
                        raise W02MorphologySuccessorV4PrivateR6RuntimeError(
                            "R6 record kind 漂移")
                    if (identity.split
                            and getattr(record, "split", identity.split)
                            != identity.split):
                        raise W02MorphologySuccessorV4PrivateR6RuntimeError(
                            "R6 record split 漂移")
                    key = record.stable_key.components
                    if previous is not None and key <= previous:
                        raise W02MorphologySuccessorV4PrivateR6RuntimeError(
                            "R6 stable key 非严格递增")
                    previous = key
                    first = key if first is None else first
                    last = key
                    count += 1
                    yield record
    except W02MorphologySuccessorV4PrivateR6RuntimeError:
        raise
    except (OSError, EOFError, ValueError) as error:
        raise W02MorphologySuccessorV4PrivateR6RuntimeError(
            "R6 gzip/JSONL 读取失败") from error
    if (count != identity.record_count
            or size != identity.content_size_bytes
            or digest.hexdigest() != identity.content_sha256
            or first != identity.first_record_key
            or last != identity.last_record_key):
        raise W02MorphologySuccessorV4PrivateR6RuntimeError(
            "R6 private content 身份漂移")
    transport_size, transport_sha = _sha256_file(permit.target_path)
    if (transport_size != identity.transport_size_bytes
            or transport_sha != identity.transport_sha256):
        raise W02MorphologySuccessorV4PrivateR6RuntimeError(
            "R6 private transport 在读取中漂移")


def read_and_close_w02_morphology_successor_v4_private_r6_sources(
        files: tuple[W02MorphologySuccessorV4PrivateR6FileIdentity, ...],
        permits: dict[str, V2AccessPermit],
        ) -> tuple[SourceRefRecord, ...]:
    """在任何 observation/label 前完整闭合 500 条 V7 SourceRef。"""
    sources = tuple(iter_w02_morphology_successor_v4_private_r6_records(
        _file_by_layout(files, "PRIVATE_SOURCE"), permits["PRIVATE_SOURCE"]))
    if (len(sources) != 500
            or any(not isinstance(row, SourceRefRecord) for row in sources)):
        raise W02MorphologySuccessorV4PrivateR6RuntimeError(
            "R6 SourceRef 闭合未完成")
    return sources


def iter_w02_morphology_successor_v4_private_r6_pairs(
        files: tuple[W02MorphologySuccessorV4PrivateR6FileIdentity, ...],
        permits: dict[str, V2AccessPermit],
        split: str,
        ) -> Iterator[tuple[ObservationRecord, EvaluatorLabelRecord]]:
    """按 stable key 顺序配对原始 lzh observation 与只读 label。"""
    observation_layout = _split_layout(split, "observation")
    label_layout = _split_layout(split, "label")
    observations = iter_w02_morphology_successor_v4_private_r6_records(
        _file_by_layout(files, observation_layout), permits[observation_layout])
    labels = iter_w02_morphology_successor_v4_private_r6_records(
        _file_by_layout(files, label_layout), permits[label_layout])
    count = 0
    for observation, evaluation in zip(observations, labels, strict=True):
        if (not isinstance(observation, ObservationRecord)
                or not isinstance(evaluation, EvaluatorLabelRecord)
                or observation.language != "lzh"
                or observation.split != split
                or evaluation.observation_key != observation.stable_key
                or evaluation.visible_stage != "W-02"
                or evaluation.owner_mode != "read_only"):
            raise W02MorphologySuccessorV4PrivateR6RuntimeError(
                "R6 private pair 语言或绑定漂移")
        count += 1
        yield observation, evaluation
    if count != _file_by_layout(files, observation_layout).record_count:
        raise W02MorphologySuccessorV4PrivateR6RuntimeError(
            "R6 private pair 数量漂移")


def _evaluate_v4_core(
        candidate_artifact_root: str | Path,
        v1_overlay_artifact_root: str | Path,
        v2_overlay_artifact_root: str | Path,
        v4_overlay_artifact_root: str | Path,
        sources: Iterable[SourceRefRecord],
        capabilities: Iterable[W02MorphologySourceCapability],
        pairs: Iterable[tuple[ObservationRecord, EvaluatorLabelRecord]],
        budget: V2EvaluatorResourceBudget,
        ) -> dict[str, object]:
    """执行 base adapter、V3 route 与 V4 overlay 后完成五维评分。"""
    if not isinstance(budget, V2EvaluatorResourceBudget):
        raise TypeError("R6 resource budget 类型漂移")
    roots = tuple(Path(value).resolve() for value in (
        candidate_artifact_root, v1_overlay_artifact_root,
        v2_overlay_artifact_root, v4_overlay_artifact_root))
    before = tuple(_tree_sha256(root) for root in roots)
    candidate_result = read_w02_candidate_artifact(roots[0])
    v1_result = read_w02_morphology_overlay_artifact(roots[1])
    v2_result = read_w02_morphology_successor_v2_overlay_artifact(roots[2])
    v4_result = read_w02_morphology_successor_v4_artifact(roots[3])
    parent_v1 = load_w02_morphology_overlay_index(roots[1])
    parent_v2 = load_w02_morphology_successor_v2_overlay_index(roots[2])
    if (v1_result.parent_candidate_semantic_sha256
            != candidate_result.candidate_semantic_sha256
            or parent_v1.dataset_keys != parent_v2.dataset_keys
            or v4_result.index.languages != ("lzh",)):
        raise W02MorphologySuccessorV4PrivateR6RuntimeError(
            "R6 Candidate/V1/V2/V4 artifact 链漂移")

    source_rows = tuple(sources)
    routes = authorize_w02_morphology_source_routes(
        source_rows, capabilities, max_sources=budget.max_records)
    indexes = build_w02_morphology_routed_indexes(parent_v1, parent_v2, routes)
    dimensions = {
        _dimension_key(name).components: name for name in W02_DEV_DIMENSIONS
    }
    rows = {name: [] for name in W02_DEV_DIMENSIONS}
    family_counts = {name: 0 for name in (
        "AUTHORED_OOV", "UD_ANNOTATION", "UNICODE_ANNOTATION")}
    v1_cache = W02MorphologyRankingCache.empty()
    v2_cache = W02MorphologySuccessorV2Cache.empty()
    source_digest = hashlib.sha256()
    for source in source_rows:
        source_digest.update(canonical_json_bytes(source.stable_key.to_list()))
    pair_iterator = iter(pairs)
    counts = {
        "adapter": 0, "evaluated": 0, "input": 0, "route": 0,
        "queried": 0, "v4_exact": 0, "v4_backoff": 0,
    }
    operations = routes.logic_operations + parent_v1.logic_operations + parent_v2.logic_operations
    resource_exhausted = False
    clone_probe = None
    try:
        with open_w02_candidate_predictor(roots[0]) as predictor:
            candidate_index = load_w02_dev_candidate_index(predictor)
            for observation, evaluation in pair_iterator:
                counts["input"] += 1
                if observation.language != "lzh":
                    raise W02MorphologySuccessorV4PrivateR6RuntimeError(
                        "R6 observation 非原始 lzh")
                original = observation.to_dict()
                base, base_ops = predict_w02_dev_observation_language_family(
                    candidate_index, observation)
                if observation.to_dict() != original:
                    raise W02MorphologySuccessorV4PrivateR6RuntimeError(
                        "R6 base adapter 修改了原 observation")
                spans = _requested_spans(evaluation)
                v3 = predict_w02_morphology_successor_v3(
                    indexes, observation, base, requested_spans=spans,
                    v1_cache=v1_cache, v2_cache=v2_cache)
                if v3.route_authorized != 1:
                    raise W02MorphologySuccessorV4PrivateR6RuntimeError(
                        "R6 原始 lzh route 未授权")
                _audit_v3_extension(v3.v1, v3.v2, spans)
                v4 = predict_w02_morphology_successor_v4(
                    v4_result.index, observation, v3, requested_spans=spans)
                projected = operations + base_ops + 9 + v3.v1.logic_operations + (
                    v3.v2.logic_operations + v3.route_logic_operations
                    + v4.logic_operations)
                if projected > budget.max_logic_operations:
                    resource_exhausted = True
                    break
                dimension, family, passed, evidence_sha = _evaluate_pair(
                    observation, evaluation, v4.prediction, dimensions)
                if dimension not in rows or family not in family_counts:
                    raise W02MorphologySuccessorV4PrivateR6RuntimeError(
                        "R6 dimension/family 未注册")
                rows[dimension].append((passed, evidence_sha))
                family_counts[family] += 1
                operations = projected
                counts["adapter"] += 1
                counts["evaluated"] += 1
                counts["route"] += 1
                counts["queried"] += len(spans)
                counts["v4_exact"] += v4.exact_candidate_count
                counts["v4_backoff"] += v4.backoff_candidate_count
                if clone_probe is None:
                    clone_probe = (
                        observation, spans, _hash_value(v4.prediction.to_dict()))
            if resource_exhausted:
                for _observation, _evaluation in pair_iterator:
                    counts["input"] += 1
        dimension_results = [
            _dimension_report(name, rows[name]) for name in W02_DEV_DIMENSIONS]
        clone_ok = None
        clone_ops = 0
        if not resource_exhausted and clone_probe is not None:
            observation, spans, expected_sha = clone_probe
            clone_v1_cache = W02MorphologyRankingCache.empty()
            clone_v2_cache = W02MorphologySuccessorV2Cache.empty()
            try:
                with open_w02_candidate_predictor(roots[0]) as predictor:
                    clone_index = load_w02_dev_candidate_index(predictor)
                    clone_base, clone_ops = (
                        predict_w02_dev_observation_language_family(
                            clone_index, observation))
                clone_v1 = load_w02_morphology_overlay_index(roots[1])
                clone_v2 = load_w02_morphology_successor_v2_overlay_index(roots[2])
                clone_v4 = read_w02_morphology_successor_v4_artifact(roots[3])
                clone_indexes = build_w02_morphology_routed_indexes(
                    clone_v1, clone_v2, routes)
                clone_v3 = predict_w02_morphology_successor_v3(
                    clone_indexes, observation, clone_base,
                    requested_spans=spans, v1_cache=clone_v1_cache,
                    v2_cache=clone_v2_cache)
                clone_prediction = predict_w02_morphology_successor_v4(
                    clone_v4.index, observation, clone_v3,
                    requested_spans=spans)
                clone_ops += (
                    clone_v3.v1.logic_operations + clone_v3.v2.logic_operations
                    + clone_v3.route_logic_operations
                    + clone_prediction.logic_operations)
                clone_ok = (
                    clone_v4.semantic_sha256 == v4_result.semantic_sha256
                    and clone_v3.route_authorized == 1
                    and _hash_value(clone_prediction.prediction.to_dict())
                    == expected_sha)
            finally:
                clone_v1_cache.close()
                clone_v2_cache.close()
        operations += clone_ops
        after = tuple(_tree_sha256(root) for root in roots)
        rollback_ok = after == before
        resource_ok = None if (
            resource_exhausted or operations > budget.max_logic_operations
        ) else True
        zero_windows = [
            {"api_calls": 0, "llm_calls": 0, "teacher_calls": 0,
             "window_key": key}
            for key in ("BEFORE_PRIVATE_READ", "DURING_PRIVATE_EVALUATION",
                        "AFTER_PRIVATE_EVALUATION")
        ]
        support = [
            _support_result(W02_MORPH_V3_PRIVATE_SUPPORT_KEYS[0], resource_ok, {
                "logic_operations": operations,
                "max_logic_operations": budget.max_logic_operations,
                "resource_exhausted": int(resource_exhausted),
            }),
            _support_result(W02_MORPH_V3_PRIVATE_SUPPORT_KEYS[1], rollback_ok, {
                "candidate_unchanged": int(after[0] == before[0]),
                "v1_overlay_unchanged": int(after[1] == before[1]),
                "v2_overlay_unchanged": int(after[2] == before[2]),
                "v3_route_writes": 0,
                "v4_overlay_unchanged": int(after[3] == before[3]),
            }),
            _support_result(
                W02_MORPH_V3_PRIVATE_SUPPORT_KEYS[2], True, zero_windows),
            _support_result(W02_MORPH_V3_PRIVATE_SUPPORT_KEYS[3], clone_ok, {
                "clone_probe_present": int(clone_probe is not None),
                "clone_semantic_equal": int(clone_ok is True),
            }),
        ]
        hard = [*dimension_results, *support]
        return {
            "base_language_adapter_count": counts["adapter"],
            "base_language_adapter_version":
                W02_BASE_LANGUAGE_FAMILY_ADAPTER_VERSION,
            "base_language_original_lzh_count": counts["adapter"],
            "base_language_temporary_scope_language": "zh",
            "candidate_artifact_manifest_sha256":
                candidate_result.artifact_manifest_sha256,
            "candidate_semantic_sha256":
                candidate_result.candidate_semantic_sha256,
            "dimension_results": dimension_results,
            "evaluation_count": counts["evaluated"],
            "family_counts": family_counts,
            "hard_conjunct_results": hard,
            "input_pair_count": counts["input"],
            "logic_operations": operations,
            "queried_span_count": counts["queried"],
            "route_authorized_count": counts["route"],
            "route_semantic_sha256": routes.semantic_sha256,
            "source_count": len(source_rows),
            "source_identity_sha256": source_digest.hexdigest(),
            "status": _result_status(hard),
            "support_results": support,
            "v1_overlay_artifact_manifest_sha256":
                v1_result.artifact_manifest_sha256,
            "v1_overlay_semantic_sha256": parent_v1.semantic_sha256,
            "v2_overlay_artifact_manifest_sha256":
                v2_result.artifact_manifest_sha256,
            "v2_overlay_semantic_sha256": parent_v2.semantic_sha256,
            "v4_backoff_candidate_count": counts["v4_backoff"],
            "v4_exact_candidate_count": counts["v4_exact"],
            "v4_overlay_artifact_manifest_sha256": v4_result.manifest_sha256,
            "v4_overlay_semantic_sha256": v4_result.semantic_sha256,
            "zero_call_windows": zero_windows,
        }
    finally:
        close_pairs = getattr(pair_iterator, "close", None)
        if callable(close_pairs):
            close_pairs()
        v1_cache.close()
        v2_cache.close()


def evaluate_w02_morphology_successor_v4_private_r6_pair_stream(
        candidate_artifact_root: str | Path,
        v1_overlay_artifact_root: str | Path,
        v2_overlay_artifact_root: str | Path,
        v4_overlay_artifact_root: str | Path,
        sources: Iterable[SourceRefRecord],
        capabilities: Iterable[W02MorphologySourceCapability],
        pairs: Iterable[tuple[ObservationRecord, EvaluatorLabelRecord]],
        budget: V2EvaluatorResourceBudget,
        ) -> dict[str, object]:
    """在 V4 core 外保持 label unknown/family/state 全部 fail closed。"""
    prepared, audit = _prepare_label_stream(pairs)
    core = _evaluate_v4_core(
        candidate_artifact_root, v1_overlay_artifact_root,
        v2_overlay_artifact_root, v4_overlay_artifact_root,
        sources, capabilities, prepared, budget)
    return _apply_label_fail_closed(core, audit)


def run_w02_morphology_successor_v4_private_r6_evaluation(
        boundary: V2EvaluatorBoundaryContract,
        roots: V2PhysicalRoots,
        registration: V2PrivateFamilyRegistration,
        files: tuple[W02MorphologySuccessorV4PrivateR6FileIdentity, ...],
        candidate_artifact_root: str | Path,
        v1_overlay_artifact_root: str | Path,
        v2_overlay_artifact_root: str | Path,
        v4_overlay_artifact_root: str | Path,
        *, protocol_sha256: str, run_id: int = 1,
        ) -> dict[str, object]:
    """唯一消费 R6 七文件并生成不含 private 内容的安全 aggregate。"""
    if run_id != 1 or registration.formal_run_count != 0:
        raise W02MorphologySuccessorV4PrivateR6RuntimeError(
            "R6 formal registration/run_id 漂移")
    permits = authorize_w02_morphology_successor_v4_private_r6_files(
        boundary, roots, registration, files)
    sources = read_and_close_w02_morphology_successor_v4_private_r6_sources(
        files, permits)
    pairs = (
        pair for split in W02_MORPH_V4_PRIVATE_R6_SPLITS
        for pair in iter_w02_morphology_successor_v4_private_r6_pairs(
            files, permits, split)
    )
    capabilities = tuple(
        w02_ud_morphology_source_capability(spec)
        for spec in blind_private_source_specs_v7())
    core = evaluate_w02_morphology_successor_v4_private_r6_pair_stream(
        candidate_artifact_root, v1_overlay_artifact_root,
        v2_overlay_artifact_root, v4_overlay_artifact_root,
        sources, capabilities, pairs, registration.resource_budget)
    authorize_w02_morphology_successor_v4_private_r6_files(
        boundary, roots, registration, files)
    expected = sum(W02_MORPH_V4_R6_SPLIT_COUNTS.values())
    if core["source_count"] != 500 or core["input_pair_count"] != expected:
        raise W02MorphologySuccessorV4PrivateR6RuntimeError(
            "R6 private 输入未完整消费")
    transport_bytes = sum(row.transport_size_bytes for row in files)
    record_reads = 500 + expected * 2
    report = {
        "artifact_kind": (
            "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_V4_PRIVATE_R6_REPORT"),
        "artifact_version": W02_MORPH_V4_PRIVATE_R6_EVALUATOR_VERSION,
        **core,
        "family_commitment": registration.family_commitment,
        "formal_private_evaluation_runs": 1,
        "label_record_reads": expected,
        "language_capability_mastered": 0,
        "language_readiness": 0,
        "next_action": (
            "W03_COMPILE_FREEZE" if core["status"] == "PASS"
            else "W02_PRIVATE_FAILED_OR_NE_STOP"),
        "observation_reads": expected,
        "private_content_stream_reads": len(files),
        "private_payload_gets": record_reads,
        "private_payload_reads": len(files) * 4,
        "private_record_reads": record_reads,
        "protocol_sha256": protocol_sha256,
        "release_key": "PH2-D03-V2",
        "run_id": 1,
        "run_scope": "FORMAL_BLIND_PRIVATE_EVALUATION",
        "source_ref_closure_before_pair_stream": 1,
        "source_ref_records_closed": 500,
        "stage_key": "W-02",
        "teacher_calls": 0,
        "transport_bytes_read": transport_bytes * 4,
        "validated_layout_count": len(files),
        "zero_write_audit": {
            "assessment_writes": 0, "candidate_writes": 0,
            "clock_writes": 0, "companion_writes": 0, "core_writes": 0,
            "evaluator_label_writes": 0, "evidence_writes": 0,
            "host_writes": 0, "memory_writes": 0, "use_writes": 0,
            "v1_overlay_writes": 0, "v2_overlay_writes": 0,
            "v3_route_writes": 0, "v4_overlay_writes": 0,
        },
    }
    if (report["private_record_reads"] > registration.resource_budget.max_records
            or report["transport_bytes_read"]
            > registration.resource_budget.max_payload_bytes
            or report["private_payload_gets"]
            > registration.resource_budget.max_payload_gets):
        raise W02MorphologySuccessorV4PrivateR6RuntimeError(
            "R6 非 logic 资源预算超限")
    validate_v2_safe_report(report)
    return report


def _repository_file(root: Path, relative: str) -> Path:
    """解析一个仓内普通文件。"""
    pure = PurePosixPath(relative)
    target = (root / Path(*pure.parts)).resolve()
    if (pure.is_absolute() or "\\" in relative or target.is_symlink()
            or not target.is_relative_to(root) or not target.is_file()):
        raise W02MorphologySuccessorV4PrivateR6RuntimeError(
            "R6 receipt 公开路径非法")
    return target


def build_w02_morphology_successor_v4_private_r6_receipt(
        repository_root: str | Path,
        report: dict[str, object],
        *, report_sha256: str,
        ) -> dict[str, object]:
    """仅在五维与四个 support 全 PASS 时构造 runtime receipt。"""
    repository = Path(repository_root).resolve()
    protocol = read_w02_morphology_successor_v4_private_r6_protocol_freeze(
        repository)
    _require = lambda value: (
        isinstance(value, str) and len(value) == 64
        and all(char in "0123456789abcdef" for char in value))
    if not _require(report_sha256):
        raise W02MorphologySuccessorV4PrivateR6RuntimeError(
            "R6 report SHA 非法")
    dimensions = report.get("dimension_results")
    support = report.get("support_results")
    if (report.get("status") != "PASS"
            or report.get("formal_private_evaluation_runs") != 1
            or not isinstance(dimensions, list) or len(dimensions) != 5
            or not isinstance(support, list) or len(support) != 4
            or any(row.get("status") != "PASS"
                   or row.get("numerator") != 100
                   or row.get("denominator") != 100 for row in dimensions)
            or any(row.get("status") != "PASS" for row in support)):
        raise W02MorphologySuccessorV4PrivateR6RuntimeError(
            "R6 FAIL/NE 不得发布 receipt")
    protocol_path = _repository_file(repository, W02_MORPH_V4_PRIVATE_R6_PROTOCOL_PATH)
    protocol_size, protocol_sha = _sha256_file(protocol_path)
    return {
        "artifact_kind": (
            "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_V4_R6_RUNTIME_RECEIPT"),
        "artifact_version": W02_MORPH_V4_PRIVATE_R6_RECEIPT_VERSION,
        "dimension_results": dimensions,
        "family_commitment": report["family_commitment"],
        "formal_private_evaluation_runs": 1,
        "language_capability_mastered": 1,
        "language_readiness": 1,
        "next_action": "W03_COMPILE_FREEZE_AND_GENERIC_EVALUATION_KERNEL",
        "private_payload_reads": report["private_payload_reads"],
        "protocol_artifact_chain_sha256": protocol["artifact_chain_sha256"],
        "protocol_file_sha256": protocol_sha,
        "protocol_file_size_bytes": protocol_size,
        "report_sha256": report_sha256,
        "source_ref_records_closed": 500,
        "status": "W02_V4_R6_RUNTIME_EVIDENCE_PASS",
        "support_results": support,
        "teacher_calls": 0,
        "w02_blocked_failed": 0,
        "w02_runtime_evidenced": 1,
        "w03_started": 0,
    }


def publish_w02_morphology_successor_v4_private_r6_receipt(
        repository_root: str | Path,
        report: dict[str, object],
        *, report_sha256: str,
        path: str | Path | None = None,
        ) -> Path:
    """排他或幂等发布唯一 PASS runtime receipt。"""
    repository = Path(repository_root).resolve()
    target = (
        repository / Path(*PurePosixPath(
            W02_MORPH_V4_PRIVATE_R6_RECEIPT_PATH).parts)
        if path is None else Path(path).resolve())
    write_immutable_json(
        build_w02_morphology_successor_v4_private_r6_receipt(
            repository, report, report_sha256=report_sha256), target)
    return target


__all__ = [
    "W02_MORPH_V4_PRIVATE_R6_EVALUATOR_VERSION",
    "W02_MORPH_V4_PRIVATE_R6_RECEIPT_PATH",
    "W02_MORPH_V4_PRIVATE_R6_RECEIPT_VERSION",
    "W02MorphologySuccessorV4PrivateR6RuntimeError",
    "authorize_w02_morphology_successor_v4_private_r6_files",
    "build_w02_morphology_successor_v4_private_r6_receipt",
    "evaluate_w02_morphology_successor_v4_private_r6_pair_stream",
    "iter_w02_morphology_successor_v4_private_r6_pairs",
    "iter_w02_morphology_successor_v4_private_r6_records",
    "publish_w02_morphology_successor_v4_private_r6_receipt",
    "read_and_close_w02_morphology_successor_v4_private_r6_sources",
    "run_w02_morphology_successor_v4_private_r6_evaluation",
]
