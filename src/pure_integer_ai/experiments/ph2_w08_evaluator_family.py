"""W08-09 private family 的 metadata-only 预注册、发布与唯一 guard。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.ph2_w08_authority import (
    W08_ABLATION_KEYS,
    W08_DIMENSION_KEYS,
    W08_VISIBLE_PACK_KEYS,
)
from pure_integer_ai.experiments.ph2_w08_contract import (
    W08_RESOURCE_BUDGET,
    open_w08_frozen_contract,
)
from pure_integer_ai.experiments.ph2_w08_evaluator_contract import (
    W08_EVALUATOR_FAILURE_PHASES,
    W08_EVALUATOR_PHASES,
    W08_EVALUATOR_THRESHOLD,
    W08_PRIVATE_CASE_NAME,
    W08_PRIVATE_CLUSTER_NAME,
    W08_PRIVATE_FAMILY_FREEZE_NAME,
    W08_PRIVATE_FIRST_RUN_GUARD_NAME,
    W08_PRIVATE_LABEL_NAME,
    W08_PRIVATE_INFERENCE_INTERFACE_VERSION,
    W08_PRIVATE_OWNER_KEY,
    W08_PRIVATE_SCHEMA_NAME,
    W08_PRIVATE_SOURCE_NAME,
    W08PrivateEvaluationError,
    evidence_commitment,
    strict_sha256,
)
from pure_integer_ai.experiments.ph2_w08_external_package import (
    W08ExternalPrivatePackageManifest,
)
from pure_integer_ai.experiments.ph2_w08_lc16 import (
    W08_LC16_CELL_STATES,
    W08_LC16_EVALUATOR_KEY,
    W08_LC16_SCOPE_KEY,
)
from pure_integer_ai.experiments.ph2_w08_open_generation_contract import (
    W08_OPEN_GENERATION_COVERAGE_KEYS,
    W08_OPEN_GENERATION_LAYER_KEYS,
)


@dataclass(frozen=True)
class W08PrivateFamilyDocuments:
    """五份 metadata-only private 文档及其不可变 commitment。"""

    source_bytes: bytes
    schema_bytes: bytes
    case_bytes: bytes
    label_bytes: bytes
    cluster_bytes: bytes
    family_key: str
    payload_commitment: str
    case_commitment: str
    label_commitment: str
    cluster_commitment: str
    candidate_contract_sha256: str
    candidate_guard_sha256: str
    candidate_host_sha256: str
    candidate_seal_sha256: str
    evaluator_public_head_commit_sha1: str
    external_package_manifest_sha256: str | None = None
    external_package_commitment: str | None = None

    def files(self) -> tuple[tuple[str, bytes], ...]:
        return (
            (W08_PRIVATE_SOURCE_NAME, self.source_bytes),
            (W08_PRIVATE_SCHEMA_NAME, self.schema_bytes),
            (W08_PRIVATE_CASE_NAME, self.case_bytes),
            (W08_PRIVATE_LABEL_NAME, self.label_bytes),
            (W08_PRIVATE_CLUSTER_NAME, self.cluster_bytes),
        )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strict_sha1(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise W08PrivateEvaluationError("W08 evaluator public HEAD 非法")
    return value


def _git_sha(repository: Path) -> str:
    try:
        value = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise W08PrivateEvaluationError("W08 evaluator 无法读取 public HEAD") from error
    return _strict_sha1(value)


def _nonce_commitment(nonce: tuple[int, ...]) -> str:
    if (
        not isinstance(nonce, tuple)
        or not nonce
        or any(type(item) is not int or not 0 <= item <= 255 for item in nonce)
    ):
        raise W08PrivateEvaluationError("W08 private nonce 必须是 byte tuple")
    return evidence_commitment({"nonce": list(nonce)})


def _identity(binding) -> dict[str, object]:
    return {
        "access_phase": binding.access_phase,
        "identity": binding.identity.to_dict(),
        "pack_key": binding.pack_key,
        "relative_path": binding.relative_path,
    }


def build_w08_private_family_documents(
    repository_root: str | Path,
    *,
    candidate_contract_sha256: str,
    candidate_guard_sha256: str,
    candidate_host_sha256: str,
    candidate_seal_sha256: str,
    evaluator_public_head_commit_sha1: str | None = None,
    nonce: tuple[int, ...] = (8, 19, 37, 61),
    external_package_manifest: W08ExternalPrivatePackageManifest | None = None,
) -> W08PrivateFamilyDocuments:
    """只从 public manifest identity 预注册 family，不读取 held-out/label。"""
    repository = Path(repository_root).resolve()
    head = _git_sha(repository)
    supplied_head = _strict_sha1(evaluator_public_head_commit_sha1 or head)
    if supplied_head != head:
        raise W08PrivateEvaluationError("W08 evaluator public HEAD 参数漂移")
    candidate_contract = strict_sha256(
        candidate_contract_sha256, label="Candidate contract"
    )
    candidate_guard = strict_sha256(candidate_guard_sha256, label="Candidate guard")
    candidate_host = strict_sha256(candidate_host_sha256, label="Candidate host")
    candidate_seal = strict_sha256(candidate_seal_sha256, label="Candidate seal")
    if external_package_manifest is None:
        context = open_w08_frozen_contract(repository)
        observations = tuple(
            item
            for item in context.evaluator_bindings
            if (
                item.pack_key in W08_VISIBLE_PACK_KEYS
                and item.identity.owner_kind == "observation"
            )
        )
        labels = tuple(
            item
            for item in context.evaluator_bindings
            if (
                item.pack_key in W08_VISIBLE_PACK_KEYS
                and item.identity.owner_kind == "evaluator"
            )
        )
    else:
        if not isinstance(
            external_package_manifest, W08ExternalPrivatePackageManifest
        ):
            raise W08PrivateEvaluationError("W08 external package manifest 类型非法")
        observations = tuple(
            item
            for item in external_package_manifest.bindings
            if item.identity.owner_kind == "observation"
        )
        labels = tuple(
            item
            for item in external_package_manifest.bindings
            if item.identity.owner_kind == "evaluator"
        )
    if not observations or len(observations) != len(labels):
        raise W08PrivateEvaluationError("W08 evaluator observation/label inventory 不闭合")
    nonce_key = _nonce_commitment(nonce)
    family_identity = {
        "candidate_contract": candidate_contract,
        "candidate_guard": candidate_guard,
        "candidate_host": candidate_host,
        "candidate_seal": candidate_seal,
        "evaluator_head": supplied_head,
        "nonce": nonce_key,
        "owner": W08_PRIVATE_OWNER_KEY,
    }
    if external_package_manifest is not None:
        family_identity["external_package_commitment"] = (
            external_package_manifest.package_commitment
        )
        family_identity["external_package_manifest_sha256"] = (
            external_package_manifest.sha256()
        )
    family_key = evidence_commitment(family_identity)
    source = {
        "artifact_kind": "PH2_W08_PRIVATE_SOURCE",
        "candidate_contract_sha256": candidate_contract,
        "candidate_first_run_guard_sha256": candidate_guard,
        "candidate_host_freeze_sha256": candidate_host,
        "candidate_terminal_seal_sha256": candidate_seal,
        "evaluator_public_head_commit_sha1": supplied_head,
        "family_key": family_key,
        "format_version": 1,
        "nonce_commitment": nonce_key,
        "owner_key": W08_PRIVATE_OWNER_KEY,
    }
    if external_package_manifest is not None:
        source["external_private_package"] = {
            "case_commitment": external_package_manifest.case_commitment,
            "cluster_commitment": external_package_manifest.cluster_commitment,
            "label_commitment": external_package_manifest.label_commitment,
            "manifest_sha256": external_package_manifest.sha256(),
            "package_commitment": external_package_manifest.package_commitment,
            "package_version": external_package_manifest.package_version,
            "payload_commitment": external_package_manifest.payload_commitment,
            "payload_kind_inventory": list(
                external_package_manifest.payload_kind_inventory
            ),
        }
    schema = {
        "ablation_order": list(W08_ABLATION_KEYS),
        "artifact_kind": "PH2_W08_PRIVATE_SCHEMA",
        "dimension_order": list(W08_DIMENSION_KEYS),
        "failure_phases": list(W08_EVALUATOR_FAILURE_PHASES),
        "fault_registry": list(W08_EVALUATOR_PHASES),
        "format_version": 1,
        "lc16_contract": {
            "cell_states": list(W08_LC16_CELL_STATES),
            "evaluator_key": W08_LC16_EVALUATOR_KEY,
            "scope_key": W08_LC16_SCOPE_KEY,
        },
        "candidate_inference_interface": {
            "required": 1,
            "version": W08_PRIVATE_INFERENCE_INTERFACE_VERSION,
        },
        "open_generation_contract": {
            "coverage_keys": list(W08_OPEN_GENERATION_COVERAGE_KEYS),
            "layer_keys": list(W08_OPEN_GENERATION_LAYER_KEYS),
        },
        "resource_limits": dict(sorted(W08_RESOURCE_BUDGET.items())),
        "schema_key": evidence_commitment({
            "ablations": list(W08_ABLATION_KEYS),
            "dimensions": list(W08_DIMENSION_KEYS),
            "family": family_key,
            "phases": list(W08_EVALUATOR_PHASES),
        }),
        "thresholds": [
            {"dimension_key": key, **W08_EVALUATOR_THRESHOLD}
            for key in W08_DIMENSION_KEYS
        ],
    }
    cases = {
        "artifact_kind": "PH2_W08_PRIVATE_CASE_INVENTORY",
        "binding_count": len(observations),
        "bindings": [_identity(item) for item in observations],
        "format_version": 1,
        "formal_run_count": 0,
    }
    label_doc = {
        "artifact_kind": "PH2_W08_PRIVATE_LABEL_INVENTORY",
        "binding_count": len(labels),
        "bindings": [_identity(item) for item in labels],
        "format_version": 1,
        "formal_run_count": 0,
    }
    by_pack: dict[str, dict[str, object]] = {}
    for binding in (*observations, *labels):
        item = by_pack.setdefault(binding.pack_key, {
            "case_record_count": 0,
            "label_record_count": 0,
            "pack_key": binding.pack_key,
            "source_cluster_keys": [],
        })
        count_key = (
            "case_record_count"
            if binding.identity.owner_kind == "observation"
            else "label_record_count"
        )
        item[count_key] = int(item[count_key]) + binding.identity.record_count
        cluster_keys = {
            tuple(key)
            for key in item["source_cluster_keys"]
        }
        cluster_keys.update(
            key.components for key in binding.identity.source_cluster_keys
        )
        item["source_cluster_keys"] = [
            list(key) for key in sorted(cluster_keys)
        ]
    clusters = {
        "artifact_kind": "PH2_W08_PRIVATE_CLUSTER_INVENTORY",
        "clusters": [by_pack[key] for key in sorted(by_pack)],
        "format_version": 1,
    }
    case_bytes = canonical_json_bytes(cases)
    label_bytes = canonical_json_bytes(label_doc)
    cluster_bytes = canonical_json_bytes(clusters)
    if external_package_manifest is not None:
        external_source = source["external_private_package"]
        assert isinstance(external_source, dict)
        external_source["case_inventory_sha256"] = _sha256(case_bytes)
        external_source["cluster_inventory_sha256"] = _sha256(cluster_bytes)
        external_source["label_inventory_sha256"] = _sha256(label_bytes)
    source_bytes = canonical_json_bytes(source)
    schema_bytes = canonical_json_bytes(schema)
    document_payload_commitment = evidence_commitment({
        "source": _sha256(source_bytes),
        "schema": _sha256(schema_bytes),
        "cases": _sha256(case_bytes),
        "labels": _sha256(label_bytes),
        "clusters": _sha256(cluster_bytes),
    })
    return W08PrivateFamilyDocuments(
        source_bytes,
        schema_bytes,
        case_bytes,
        label_bytes,
        cluster_bytes,
        family_key,
        (
            external_package_manifest.payload_commitment
            if external_package_manifest is not None
            else document_payload_commitment
        ),
        (
            external_package_manifest.case_commitment
            if external_package_manifest is not None
            else _sha256(case_bytes)
        ),
        (
            external_package_manifest.label_commitment
            if external_package_manifest is not None
            else _sha256(label_bytes)
        ),
        (
            external_package_manifest.cluster_commitment
            if external_package_manifest is not None
            else _sha256(cluster_bytes)
        ),
        candidate_contract,
        candidate_guard,
        candidate_host,
        candidate_seal,
        supplied_head,
        (
            external_package_manifest.sha256()
            if external_package_manifest is not None
            else None
        ),
        (
            external_package_manifest.package_commitment
            if external_package_manifest is not None
            else None
        ),
    )


def _decode_document(payload: bytes, *, kind: str) -> dict[str, object]:
    try:
        value = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise W08PrivateEvaluationError("W08 private family document 无法解析") from error
    if (
        not isinstance(value, dict)
        or canonical_json_bytes(value) != payload
        or value.get("artifact_kind") != kind
        or value.get("format_version") != 1
    ):
        raise W08PrivateEvaluationError("W08 private family document identity 漂移")
    return value


def validate_w08_private_family_documents(
    documents: W08PrivateFamilyDocuments,
) -> None:
    if not isinstance(documents, W08PrivateFamilyDocuments):
        raise W08PrivateEvaluationError("W08 private family documents 类型非法")
    source = _decode_document(documents.source_bytes, kind="PH2_W08_PRIVATE_SOURCE")
    schema = _decode_document(documents.schema_bytes, kind="PH2_W08_PRIVATE_SCHEMA")
    cases = _decode_document(
        documents.case_bytes, kind="PH2_W08_PRIVATE_CASE_INVENTORY"
    )
    labels = _decode_document(
        documents.label_bytes, kind="PH2_W08_PRIVATE_LABEL_INVENTORY"
    )
    _decode_document(
        documents.cluster_bytes, kind="PH2_W08_PRIVATE_CLUSTER_INVENTORY"
    )
    external = source.get("external_private_package")
    if documents.external_package_manifest_sha256 is None:
        commitment_drift = (
            external is not None
            or documents.external_package_commitment is not None
            or documents.case_commitment != _sha256(documents.case_bytes)
            or documents.label_commitment != _sha256(documents.label_bytes)
            or documents.cluster_commitment != _sha256(documents.cluster_bytes)
            or documents.payload_commitment != evidence_commitment({
                "source": _sha256(documents.source_bytes),
                "schema": _sha256(documents.schema_bytes),
                "cases": _sha256(documents.case_bytes),
                "labels": _sha256(documents.label_bytes),
                "clusters": _sha256(documents.cluster_bytes),
            })
        )
    else:
        commitment_drift = (
            not isinstance(external, dict)
            or external.get("manifest_sha256")
            != documents.external_package_manifest_sha256
            or external.get("package_commitment")
            != documents.external_package_commitment
            or external.get("case_commitment") != documents.case_commitment
            or external.get("label_commitment") != documents.label_commitment
            or external.get("cluster_commitment") != documents.cluster_commitment
            or external.get("payload_commitment") != documents.payload_commitment
            or external.get("case_inventory_sha256")
            != _sha256(documents.case_bytes)
            or external.get("label_inventory_sha256")
            != _sha256(documents.label_bytes)
            or external.get("cluster_inventory_sha256")
            != _sha256(documents.cluster_bytes)
        )
    if (
        source.get("family_key") != documents.family_key
        or source.get("candidate_contract_sha256")
        != documents.candidate_contract_sha256
        or source.get("candidate_first_run_guard_sha256")
        != documents.candidate_guard_sha256
        or source.get("candidate_host_freeze_sha256")
        != documents.candidate_host_sha256
        or source.get("candidate_terminal_seal_sha256")
        != documents.candidate_seal_sha256
        or source.get("evaluator_public_head_commit_sha1")
        != documents.evaluator_public_head_commit_sha1
        or tuple(schema.get("dimension_order", ())) != W08_DIMENSION_KEYS
        or tuple(schema.get("ablation_order", ())) != W08_ABLATION_KEYS
        or schema.get("candidate_inference_interface") != {
            "required": 1,
            "version": W08_PRIVATE_INFERENCE_INTERFACE_VERSION,
        }
        or schema.get("resource_limits")
        != dict(sorted(W08_RESOURCE_BUDGET.items()))
        or cases.get("formal_run_count") != 0
        or labels.get("formal_run_count") != 0
        or cases.get("binding_count") != labels.get("binding_count")
        or commitment_drift
    ):
        raise W08PrivateEvaluationError("W08 private family cross-reference 漂移")


def _validate_root(root: Path, forbidden_roots: tuple[str | Path, ...]) -> None:
    for forbidden in forbidden_roots:
        other = Path(forbidden).resolve()
        if root == other or root.is_relative_to(other) or other.is_relative_to(root):
            raise W08PrivateEvaluationError("W08 private root 与 public/Candidate root 重叠")


def publish_w08_private_family(
    artifact_root: str | Path,
    documents: W08PrivateFamilyDocuments,
    *,
    forbidden_roots: tuple[str | Path, ...],
) -> tuple[Path, str]:
    """在全新 Git/Candidate 外 root 排他发布五文档及 family freeze。"""
    validate_w08_private_family_documents(documents)
    root = Path(artifact_root).resolve()
    _validate_root(root, forbidden_roots)
    root.parent.mkdir(parents=True, exist_ok=True)
    try:
        root.mkdir()
    except FileExistsError as error:
        raise W08PrivateEvaluationError("W08 private root 必须全新且不可复用") from error
    identities = []
    for name, payload in documents.files():
        target = root / name
        with target.open("xb") as handle:
            handle.write(payload)
        identities.append({
            "name": name,
            "sha256": _sha256(payload),
            "size_bytes": len(payload),
        })
    freeze_value = {
        "ablation_order": list(W08_ABLATION_KEYS),
        "artifact_kind": "PH2_W08_PRIVATE_FAMILY_FREEZE",
        "case_commitment": documents.case_commitment,
        "candidate_contract_sha256": documents.candidate_contract_sha256,
        "candidate_first_run_guard_sha256": documents.candidate_guard_sha256,
        "candidate_host_freeze_sha256": documents.candidate_host_sha256,
        "candidate_terminal_seal_sha256": documents.candidate_seal_sha256,
        "cluster_commitment": documents.cluster_commitment,
        "dimension_order": list(W08_DIMENSION_KEYS),
        "documents": identities,
        "evaluator_public_head_commit_sha1": documents.evaluator_public_head_commit_sha1,
        "failure_phases": list(W08_EVALUATOR_FAILURE_PHASES),
        "family_key": documents.family_key,
        "formal_run_count": 0,
        "format_version": 1,
        "label_commitment": documents.label_commitment,
        "payload_commitment": documents.payload_commitment,
        "private_payload_reads": 0,
        "resource_limits": dict(sorted(W08_RESOURCE_BUDGET.items())),
    }
    if documents.external_package_manifest_sha256 is not None:
        freeze_value["external_private_package"] = {
            "manifest_sha256": documents.external_package_manifest_sha256,
            "package_commitment": documents.external_package_commitment,
        }
    freeze = canonical_json_bytes(freeze_value)
    target = root / W08_PRIVATE_FAMILY_FREEZE_NAME
    with target.open("xb") as handle:
        handle.write(freeze)
    return target, _sha256(freeze)


def consume_w08_private_first_run_guard(
    artifact_root: str | Path,
    *,
    family_freeze_sha256: str,
) -> tuple[Path, str]:
    """在首次 held-out/label read 前排他消费唯一 private guard。"""
    root = Path(artifact_root).resolve()
    expected = strict_sha256(family_freeze_sha256, label="family freeze")
    freeze = root / W08_PRIVATE_FAMILY_FREEZE_NAME
    if (
        not freeze.is_file()
        or freeze.is_symlink()
        or _sha256(freeze.read_bytes()) != expected
    ):
        raise W08PrivateEvaluationError("W08 private family freeze SHA 漂移")
    try:
        value = json.loads(freeze.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise W08PrivateEvaluationError("W08 private family freeze 无法读取") from error
    if value.get("formal_run_count") != 0 or value.get("private_payload_reads") != 0:
        raise W08PrivateEvaluationError("W08 private family 已运行或提前读取 payload")
    payload = canonical_json_bytes({
        "artifact_kind": "PH2_W08_PRIVATE_FIRST_RUN_GUARD",
        "family_freeze_sha256": expected,
        "formal_run_count_after": 1,
        "formal_run_count_before": 0,
        "format_version": 1,
    })
    target = root / W08_PRIVATE_FIRST_RUN_GUARD_NAME
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as error:
        raise W08PrivateEvaluationError(
            "W08 private family guard 已消费，不可重跑"
        ) from error
    return target, _sha256(payload)


__all__ = [
    "W08PrivateFamilyDocuments",
    "build_w08_private_family_documents",
    "consume_w08_private_first_run_guard",
    "publish_w08_private_family",
    "validate_w08_private_family_documents",
]
