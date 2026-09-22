"""在独立 K 盘会话中运行 carrier projection 到三图库 QueryState 的接线切片。"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from pure_integer_ai.cognition.shared.artifact_envelope import (
    PROJECTION_GENERATION,
    PROJECTION_REASONING,
    PROJECTION_UNDERSTANDING,
)
from pure_integer_ai.cognition.shared.candidate_verifier import (
    RevealedObjectObservation,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    SourceRef,
    VersionBundle,
    concept_identity,
)
from pure_integer_ai.cognition.shared.memory_event import MemoryObjectRef
from pure_integer_ai.cognition.understanding.artifact_query_input import (
    ArtifactQueryInput,
)
from pure_integer_ai.experiments.ph2_carrier_projection_mapper import (
    CarrierProjectionMapper,
)
from pure_integer_ai.experiments.ph2_carrier_projection_mapper_catalog import (
    PARENT_PACK_PATH,
    build_carrier_projection_mapper_manifest,
)
from pure_integer_ai.experiments.ph2_carrier_projection_runtime import (
    CarrierProjectionRuntime,
    CarrierProjectionSpec,
)
from pure_integer_ai.experiments.ph2_document_container_carrier_adapter import (
    adapt_document_container_carrier_record,
)
from pure_integer_ai.experiments.ph2_document_container_carrier_contract import (
    read_document_container_carrier_records,
)
from pure_integer_ai.experiments.ph2_reference_link_embed_carrier_adapter import (
    adapt_reference_link_embed_carrier_record,
)
from pure_integer_ai.experiments.ph2_reference_link_embed_carrier_contract import (
    read_reference_link_embed_carrier_records,
)
from pure_integer_ai.experiments.ph2_typed_carrier_pack_contract import (
    read_typed_carrier_pack_manifest,
)
from pure_integer_ai.experiments.trained_graph_query_bridge import (
    TrainedGraphQueryBridge,
)
from pure_integer_ai.storage.backend import DictBackend


_CARRIER_SOURCES = {
    "DOCUMENT_CONTAINER": (
        "data/ph2/lc16_document_container_carrier_v1.jsonl.sample",
        read_document_container_carrier_records,
        adapt_document_container_carrier_record,
    ),
    "REFERENCE_LINK_EMBED": (
        "data/ph2/lc16_reference_link_embed_carrier_v1.jsonl.sample",
        read_reference_link_embed_carrier_records,
        adapt_reference_link_embed_carrier_record,
    ),
}
_VERIFIER_SOURCE_KIND = 91545
_CANDIDATE_DOMAIN = 91546
_SLICE_FORMAT = "PURE_INTEGER_ARTIFACT_QUERY_SLICE_V1"


def _json_bytes(value: object) -> bytes:
    """以冻结 ASCII JSON 写 receipt；图语义本体仍保留完整整数键。"""
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("ascii")


def _sha256(path: Path) -> str:
    """流式计算输出 artifact SHA-256，不把 trace 另复制到内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _write_trace(trace: dict[str, object], target: Path) -> str:
    """规范 gzip 写入 trace，使重跑时内容身份只依赖整数状态。"""
    encoder = json.JSONEncoder(
        ensure_ascii=True, sort_keys=True,
        separators=(",", ":"), allow_nan=False)
    with target.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
            for piece in encoder.iterencode(trace):
                stream.write(piece.encode("ascii"))
    return _sha256(target)


def _require_new_k_run_root(run_root: Path) -> Path:
    """严格限制可写 session/receipt 到一个此前不存在的 K 盘子目录。"""
    root = run_root.resolve()
    if root.drive.upper() != "K:" or root == Path(root.anchor):
        raise ValueError("artifact query run root 必须是 K 盘非根目录")
    if root.exists():
        raise ValueError("artifact query run root 已存在，禁止覆盖")
    root.mkdir(parents=True)
    return root


def _source(source_id: int) -> SourceRef:
    """返回 carrier lifecycle 独立核验来源，不能充当 Core 训练来源。"""
    return SourceRef(
        _VERIFIER_SOURCE_KIND, source_id, 0,
        GLOBAL_OWNER_SCOPE, VersionBundle())


def _artifact_input(
        repository_root: Path, bridge: TrainedGraphQueryBridge,
        carrier_key: str,
        ) -> tuple[ArtifactQueryInput, dict[str, object]]:
    """由冻结 DOCUMENT_CONTAINER 与已有投影生命周期形成可查询 carrier。"""
    try:
        sample_path, reader, adapter = _CARRIER_SOURCES[carrier_key]
    except KeyError as error:
        raise ValueError("artifact slice carrier 未登记") from error
    records = reader(repository_root / sample_path)
    materialization = adapter(records[0])
    parent = read_typed_carrier_pack_manifest(repository_root / PARENT_PACK_PATH)
    mapper_manifest = build_carrier_projection_mapper_manifest(repository_root)
    mapper = CarrierProjectionMapper(parent)
    rule = next(item for item in mapper_manifest.rules
                if item.carrier_key == carrier_key)
    mapped = mapper.map(
        carrier_key, materialization, rule,
        item_indices=(0,), input_key=(91547, 1))
    target = next(
        binding.filler for fact in bridge.facts for binding in fact.bindings)
    spec = CarrierProjectionSpec(
        concept_identity((_CANDIDATE_DOMAIN, 1)),
        (_CANDIDATE_DOMAIN, 2),
        concept_identity((_CANDIDATE_DOMAIN, 3)),
        target,
        (PROJECTION_UNDERSTANDING, PROJECTION_REASONING, PROJECTION_GENERATION),
        mapped.feature_identities,
        (_source(1), _source(2)),
    )
    backend = DictBackend()
    try:
        runtime = CarrierProjectionRuntime(backend)
        lifecycle = runtime.learn(
            spec, mapped,
            revealed=RevealedObjectObservation(
                mapped.source, mapped.scope, mapped.input_key, _source(3),
                supported_targets=(target,), trace=(91547, 2)),
        )
    finally:
        backend.close()
    if lifecycle.projection is None:
        raise RuntimeError("carrier lifecycle 没有物化 ArtifactSemanticProjection")
    envelope = mapped.envelope
    artifact_input = ArtifactQueryInput(
        envelope,
        tuple(sorted(
            (item for item in materialization.anchors
             if item.envelope_identity == envelope.identity),
            key=lambda item: item.stable_key())),
        tuple(sorted(
            (item for item in materialization.structure_nodes
             if item.envelope_identity == envelope.identity),
            key=lambda item: item.stable_key())),
        (lifecycle.projection,),
        tuple(sorted(
            (item for item in getattr(materialization, "references", ())
             if item.envelope_identity == envelope.identity),
            key=lambda item: item.stable_key())),
    )
    return artifact_input, {
        "carrier_key": mapped.carrier_key,
        "sample_case_key": list(mapped.case_key.stable_key()),
        "projection_lifecycle_state": list(
            lifecycle.projection.lifecycle_state.stable_key()),
        "semantic_object": list(target.stable_key()),
        "projection_training_performed": 0,
        "independent_verifier_source": list(_source(3).stable_key()),
        "reference_count": len(artifact_input.references),
        "reference_states": [item.target_state for item in artifact_input.references],
    }


def _snapshot(memory) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """读取权威 O/H/E 对象键，以检验 append-only 历史未丢失。"""
    state = memory.active_structures()
    return (
        tuple(sorted(item.observation_key for item in state.observations)),
        tuple(sorted(item.hypothesis_key for item in state.hypotheses)),
        tuple(sorted(item.evidence_key for item in state.evidence)),
    )


def run_slice(
        database: Path, run_root: Path, *, tenant_id: int = 1,
        user_id: int = 1, session_id: int = 1,
        carrier_key: str = "DOCUMENT_CONTAINER",
        ) -> dict[str, object]:
    """执行一次真实 carrier->Memory->QueryState 接线，不写模型或重训 Core。"""
    model = database.resolve(strict=True)
    root = _require_new_k_run_root(run_root)
    session = root / "session.sqlite3"
    repository_root = Path(__file__).resolve().parents[3]
    scope = dict(tenant_id=tenant_id, user_id=user_id, session_id=session_id)
    with TrainedGraphQueryBridge(
            model, memory_database=session, **scope) as bridge:
        before = _snapshot(bridge.memory)
        artifact_input, lifecycle_receipt = _artifact_input(
            repository_root, bridge, carrier_key)
        if carrier_key == "REFERENCE_LINK_EMBED" and not artifact_input.references:
            raise RuntimeError("REFERENCE_LINK_EMBED 切片没有可消费的 ArtifactReferenceBinding")
        appended = bridge.memory.append_artifact(artifact_input, speaker_kind=1)
        observation_ref = bridge.memory.intake.result_for_source(
            appended.source).observation_ref
        if not isinstance(observation_ref, MemoryObjectRef):
            raise RuntimeError("artifact append 没有返回权威 Memory Observation 引用")
        trace = bridge.query_artifact(
            artifact_input, max_depth=64,
            input_observation_ref=observation_ref)
        after = _snapshot(bridge.memory)
        expected_artifact_routes = set(artifact_input.memory_candidate_keys())
        selected_before_close = bridge.memory.active_structures(
            candidate_keys=artifact_input.memory_candidate_keys(),
            observation_refs=(observation_ref,))
        matched_artifact_routes = {
            item.candidate_key for item in selected_before_close.hypotheses
            if item.candidate_key in expected_artifact_routes
        }
        root_spaces = {item["space"] for item in trace["roots"]}
        expanded_spaces = {
            item["space"] for item in trace["roots"]
            if item["status"] == 2
        }
        evidence_spaces = {item["space"] for item in trace["evidence"]}
        if root_spaces != {1, 2, 3} or expanded_spaces != {1, 2, 3}:
            raise AssertionError("artifact query 没有完成同一次 QueryState 三图 roots")
        if evidence_spaces != {1, 2, 3}:
            raise AssertionError("artifact query 没有三图 Evidence 参与")
        if not matched_artifact_routes:
            raise AssertionError("artifact projection 没有通过精确 Memory route 被消费")
        if trace["artifact_input"] != artifact_input.integer_trace():
            raise AssertionError("artifact integer trace 与实际输入不一致")
        reference_payloads = {
            tuple(item["payload_key"]) for item in trace["evidence"]
            if item["space"] == 1 and item["polarity"] == 3
        }
        if not {
                item.stable_key() for item in artifact_input.references
        } <= reference_payloads:
            raise AssertionError("artifact reference 没有作为 UNKNOWN Core Evidence 保留")
        if any(not set(old) <= set(new) for old, new in zip(before, after)):
            raise AssertionError("artifact append 破坏了既有 Memory 历史")
        trace_sha256 = _write_trace(trace, root / "query-001.json.gz")
    with TrainedGraphQueryBridge(model, memory_database=session, **scope) as bridge:
        restored_snapshot = _snapshot(bridge.memory)
        restored_selected = bridge.memory.active_structures(
            candidate_keys=artifact_input.memory_candidate_keys(),
            observation_refs=(observation_ref,))
        if restored_snapshot != after or restored_selected != selected_before_close:
            raise AssertionError("artifact session 冷恢复与运行期 O/H/E 不一致")
    receipt = {
        "format": _SLICE_FORMAT,
        "schema_version": 1,
        "model_read_only": 1,
        "core_training_performed": 0,
        "projection_training_performed": 0,
        "free_dialogue_complete": 0,
        "model_path": str(model),
        "model_sha256": _sha256(model),
        "session_path": str(session.resolve()),
        "owner_scope": [tenant_id, user_id, session_id],
        "before_ohe": list(map(len, before)),
        "after_ohe": list(map(len, after)),
        "history_retained": 1,
        "cold_restore_equal": 1,
        "artifact_projection_count": len(artifact_input.understanding_projections),
        "artifact_reference_count": len(artifact_input.references),
        "artifact_memory_candidate_route_count": len(matched_artifact_routes),
        "roots_by_space": [
            sum(item["space"] == space for item in trace["roots"])
            for space in (1, 2, 3)],
        "evidence_by_space": [
            sum(item["space"] == space for item in trace["evidence"])
            for space in (1, 2, 3)],
        "depth": trace["depth"],
        "termination": trace["termination_reason"],
        "response_surface": trace["response_surface"],
        "trace": "query-001.json.gz",
        "trace_sha256": trace_sha256,
        "lifecycle": lifecycle_receipt,
    }
    (root / "receipt.json").write_bytes(_json_bytes(receipt))
    return receipt


def main() -> int:
    """显式接收只读模型和新的 K 盘根，不提供隐藏默认路径。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--tenant-id", type=int, default=1)
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--session-id", type=int, required=True)
    parser.add_argument("--carrier", choices=tuple(sorted(_CARRIER_SOURCES)),
                        default="DOCUMENT_CONTAINER")
    args = parser.parse_args()
    receipt = run_slice(
        args.database, args.run_root,
        tenant_id=args.tenant_id, user_id=args.user_id,
        session_id=args.session_id, carrier_key=args.carrier)
    print(json.dumps(receipt, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
