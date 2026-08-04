"""W-08 只追加权威基线、现场构建器与严格规范回读器。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
from typing import Any


W08_AUTHORITY_KIND = "PH2_W08_AUTHORITY_BASELINE"
W08_AUTHORITY_VERSION = "PH2-W08-AUTHORITY-BASELINE-V1"
W08_AUTHORITY_RELATIVE_PATH = (
    "data/ph2/manifests/d03_v1/w08_authority_baseline_v1.json"
)
W08_STAGE_MANIFEST_PATH = (
    "data/ph2/manifests/d03_v1/stages/w08_stage_manifest_v1.json"
)
W08_GLOBAL_MANIFEST_PATH = (
    "data/ph2/manifests/d03_v1/ph2_global_course_manifest_v1.json"
)
W08_INVALIDATION_GRAPH_PATH = (
    "data/ph2/manifests/d03_v1/stage_invalidation_graph_v1.json"
)
W08_LC16_OVERLAY_PATH = "data/ph2/manifests/d03_lc16_successor_overlay_v1.json"
W08_W07_RECEIPT_PATH = (
    "data/ph2/manifests/d03_v1/w07_runtime_evidence_receipt_v1.json"
)

W08_DIMENSION_KEYS = (
    "W-08-CHINESE_VARIATION",
    "W-08-DISCOURSE",
    "W-08-LOCAL_RECOMPUTE",
    "W-08-LONG_CONTEXT",
    "W-08-P3IA",
)
W08_ABLATION_KEYS = tuple(f"{key}-ABLATION" for key in W08_DIMENSION_KEYS)
W08_SUBTASK_ORDER = (
    "CHINESE_VARIATION",
    "DISCOURSE",
    "LOCAL_RECOMPUTE",
    "LONG_CONTEXT",
    "P3IA",
)
W08_FUTURE_PACK_KEYS = (
    "AUTHORED_CC0_V1--CC0-1.0--free-text-hierarchy-recall-v1",
    "AUTHORED_CC0_V1--CC0-1.0--generation-postcheck-v1",
    "AUTHORED_CC0_V1--CC0-1.0--gg03-generation-generalization-v1",
    "AUTHORED_CC0_V1--CC0-1.0--question-answer-v1",
)
W08_VISIBLE_PACK_KEYS = (
    "AUTHORED_CC0_V1--CC0-1.0--discourse-revision-v1",
    "AUTHORED_CC0_V1--CC0-1.0--lc07-discourse-information-v1",
    "AUTHORED_CC0_V1--CC0-1.0--lc08-open-set-clarification-v1",
    "AUTHORED_CC0_V1--CC0-1.0--lc14-attribution-quotation-v1",
    "ZHWIKIPEDIA_20260701--CC-BY-SA-4.0--source-pack-v1",
    "ZHWIKTIONARY_20260701--CC-BY-SA-4.0--source-pack-v1",
    "UD_ZH_GSDSIMP_R2_18--CC-BY-SA-4.0--source-pack-v1",
)
W08_PARENT_PATHS = (
    W08_STAGE_MANIFEST_PATH,
    W08_GLOBAL_MANIFEST_PATH,
    W08_INVALIDATION_GRAPH_PATH,
    W08_LC16_OVERLAY_PATH,
    W08_W07_RECEIPT_PATH,
    "data/ph2/manifests/p3ia_free_text_hierarchy_recall_course_v2.json",
    "data/ph2/manifests/language_capability_baseline_v41.json",
    "data/ph2/manifests/md05_center_diffusion_decision_v1.json",
    "data/ph2/manifests/r03_correction_recovery_absorption_v1.json",
    "data/ph2/manifests/r04_authorized_center_generation_absorption_v1.json",
    "data/ph2/manifests/r06_long_context_absorption_v1.json",
)
W08_RETENTION_PATHS = (
    "data/ph2/manifests/w02_lc16_supplemental_runtime_receipt_v1.json",
    "data/ph2/manifests/d03_v1/w03_runtime_evidence_receipt_v1.json",
    "data/ph2/manifests/d03_v1/w04_runtime_evidence_receipt_v1.json",
    "data/ph2/manifests/d03_v1/w05_runtime_evidence_receipt_v1.json",
    "data/ph2/manifests/d03_v1/w06_runtime_evidence_receipt_v1.json",
    W08_W07_RECEIPT_PATH,
)
W08_ZERO_EXECUTION_STATE = {
    "LANGUAGE_CAPABILITY_MASTERED": 0,
    "LANGUAGE_READINESS": 0,
    "W07_RUNTIME_EVIDENCED": 1,
    "W08_STARTED": 0,
    "W09_STARTED": 0,
    "companion_writes": 0,
    "formal_w08_training_runs": 0,
    "llm_calls": 0,
    "memory_learning_writes": 0,
    "teacher_calls": 0,
}
_ROOT_FIELDS = {
    "ablation_keys",
    "aggregation_policy",
    "artifact_kind",
    "artifact_version",
    "baseline_public_head_commit_sha1",
    "dimension_keys",
    "execution_state",
    "format_version",
    "generation_account",
    "lc16_scope",
    "parent_identities",
    "p3ia_boundary",
    "retention_identities",
    "stage_inventory",
    "subtask_order",
    "visible_pack_identities",
}


class W08AuthorityError(RuntimeError):
    """W-08 权威边界或绑定身份发生漂移。"""


def _safe_relative(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise W08AuthorityError("W-08 authority path is not canonical")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise W08AuthorityError("W-08 authority path escapes the repository")
    return pure.as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
    except OSError as error:
        raise W08AuthorityError(f"missing W-08 authority parent: {path.name}") from error
    return digest.hexdigest()


def _canonical_object(path: Path) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise W08AuthorityError(f"invalid W-08 authority JSON: {path.name}") from error
    if not isinstance(value, dict):
        raise W08AuthorityError("W-08 authority JSON must be an object")
    return value


def _identity(repository: Path, relative_path: str) -> dict[str, Any]:
    relative = _safe_relative(relative_path)
    path = repository / relative
    try:
        size = path.stat().st_size
    except OSError as error:
        raise W08AuthorityError(f"missing W-08 authority parent: {relative}") from error
    return {"relative_path": relative, "sha256": _sha256(path), "size_bytes": size}


def _git_sha(repository: Path, revision: str) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", revision],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise W08AuthorityError(f"cannot resolve Git revision: {revision}") from error
    if len(result) != 40 or any(char not in "0123456789abcdef" for char in result):
        raise W08AuthorityError(f"invalid Git revision: {revision}")
    return result


def _git_is_ancestor(repository: Path, ancestor: str, descendant: str) -> bool:
    try:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=repository,
            check=False,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise W08AuthorityError("cannot verify W-08 baseline ancestry") from error
    if result.returncode not in {0, 1}:
        raise W08AuthorityError("cannot verify W-08 baseline ancestry")
    return result.returncode == 0


def _baseline_public_head(repository: Path) -> str:
    """首次发布冻结现场 HEAD；后续提交只允许在该基线上追加。"""
    authority_path = repository / W08_AUTHORITY_RELATIVE_PATH
    if authority_path.is_file():
        value = _canonical_object(authority_path)
        baseline = value.get("baseline_public_head_commit_sha1")
        if not isinstance(baseline, str) or _git_sha(repository, baseline) != baseline:
            raise W08AuthorityError("W-08 baseline commit is missing")
        current = _git_sha(repository, "HEAD")
        if not _git_is_ancestor(repository, baseline, current):
            raise W08AuthorityError("current HEAD is not descended from W-08 baseline")
        return baseline
    local_head = _git_sha(repository, "HEAD")
    tracking_head = _git_sha(repository, "origin/master")
    if local_head != tracking_head:
        raise W08AuthorityError("local and origin/master drifted before W-08")
    return local_head


def _pack_manifest_identities(
    repository: Path,
    global_manifest: dict[str, Any],
    pack_keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    bindings = global_manifest.get("pack_bindings")
    if not isinstance(bindings, list):
        raise W08AuthorityError("global pack inventory is missing")
    by_key = {item.get("pack_key"): item for item in bindings if isinstance(item, dict)}
    result: list[dict[str, Any]] = []
    for pack_key in pack_keys:
        item = by_key.get(pack_key)
        if not isinstance(item, dict) or not isinstance(item.get("manifest_identity"), dict):
            raise W08AuthorityError(f"missing W-08 pack binding: {pack_key}")
        declared = item["manifest_identity"]
        actual = _identity(repository, declared.get("relative_path"))
        if actual != declared:
            raise W08AuthorityError(f"W-08 pack identity drift: {pack_key}")
        result.append({"earliest_stage": item.get("earliest_stage"), "pack_key": pack_key, **actual})
    return result


def _validate_source_authorities(repository: Path) -> None:
    p3ia = _canonical_object(repository / W08_PARENT_PATHS[5])
    if (
        p3ia.get("stage") != "W-09"
        or p3ia.get("runtime_status") != "NOT_STARTED"
        or p3ia.get("pack_manifest_relative_path")
        != "ph2_p3ia_dataset_artifacts/d02_language_courses_v1/packs/"
        "AUTHORED_CC0_V1--CC0-1.0--free-text-hierarchy-recall-v1/manifest.json"
    ):
        raise W08AuthorityError("P3-Ia future course was moved into W-08")
    baseline = _canonical_object(repository / W08_PARENT_PATHS[6])
    facts = baseline.get("capability_facts", {})
    if (
        baseline.get("artifact_status") != "BASELINE_FROZEN"
        or facts.get("p3ia_production_contract_status") != "CONTRACT_READY"
        or facts.get("r03_recovery_status") != "PRODUCTION_EVIDENCED"
        or facts.get("r04_authorized_generation_status") != "PRODUCTION_EVIDENCED"
        or facts.get("r06_long_context_status") != "PRODUCTION_EVIDENCED"
    ):
        raise W08AuthorityError("W-08 public production parent is not evidenced")
    for relative in W08_PARENT_PATHS[7:]:
        value = _canonical_object(repository / relative)
        status = value.get("artifact_status", value.get("verdict"))
        if status not in {"PRODUCTION_EVIDENCED", "PASS"}:
            raise W08AuthorityError(f"W-08 production parent is not ready: {relative}")


def build_w08_authority(repository_root: str | Path) -> dict[str, Any]:
    """现场重算完整 W-08 权威，且不打开任何 future payload。"""
    repository = Path(repository_root).resolve()
    stage = _canonical_object(repository / W08_STAGE_MANIFEST_PATH)
    global_manifest = _canonical_object(repository / W08_GLOBAL_MANIFEST_PATH)
    invalidation = _canonical_object(repository / W08_INVALIDATION_GRAPH_PATH)
    overlay = _canonical_object(repository / W08_LC16_OVERLAY_PATH)
    receipt = _canonical_object(repository / W08_W07_RECEIPT_PATH)

    evaluation = stage.get("evaluation_binding", {})
    visibility = stage.get("data_visibility", {})
    thresholds = evaluation.get("thresholds", [])
    dimension_keys = tuple(item.get("dimension_key") for item in thresholds)
    ablation_keys = tuple(evaluation.get("ablation_keys", ()))
    if (
        stage.get("stage_identity", {}).get("stage_key") != "W-08"
        or stage.get("stage_identity", {}).get("substage_keys") != []
        or dimension_keys != W08_DIMENSION_KEYS
        or ablation_keys != W08_ABLATION_KEYS
        or evaluation.get("aggregation_policy") != "ALL_BEARING_DIMENSIONS_MUST_PASS"
        or any(item.get("ne_policy") != "BLOCK" for item in thresholds)
    ):
        raise W08AuthorityError("W-08 bearing contract drifted")
    future = tuple(visibility.get("future_pack_keys", ()))
    train = tuple(visibility.get("train_pack_keys", ()))
    if future != W08_FUTURE_PACK_KEYS or set(future) & set(train):
        raise W08AuthorityError("W-09 future pack entered W-08 train inventory")
    if invalidation.get("stage_edges", [])[-2:] != [
        {"consumer_stage": "W-08", "prerequisite_stage": "W-07"},
        {"consumer_stage": "W-09", "prerequisite_stage": "W-08"},
    ]:
        raise W08AuthorityError("W-08 invalidation chain drifted")
    accounts = {item.get("account_key"): item for item in overlay.get("generation_accounts", [])}
    open_account = accounts.get("OPEN_GENERATION", {})
    replay_account = accounts.get("SOURCE_GROUNDED_SURFACE_REPLAY", {})
    if (
        open_account.get("aggregate_with_other_account") != 0
        or replay_account.get("aggregate_with_other_account") != 0
        or open_account.get("current_status") != "NE_NOT_YET_EVALUABLE"
    ):
        raise W08AuthorityError("OPEN_GENERATION account is merged or already promoted")
    scopes = {item.get("scope_key"): item for item in overlay.get("scope_records", [])}
    if scopes.get("DISCOURSE_REFERENCE_GENERATION", {}).get("earliest_stage") != "W-08":
        raise W08AuthorityError("LC-16 W-08 scope drifted")
    state = receipt.get("execution_state", {})
    if (
        receipt.get("status") != "RUNTIME_EVIDENCED"
        or state.get("W07_RUNTIME_EVIDENCED") != 1
        or state.get("W08_STARTED") != 0
        or receipt.get("open_generation_state") != "NE_NOT_YET_EVALUABLE"
    ):
        raise W08AuthorityError("W07 receipt does not release W-08")
    _validate_source_authorities(repository)

    parent_identities = [_identity(repository, path) for path in W08_PARENT_PATHS]
    retention = [_identity(repository, path) for path in W08_RETENTION_PATHS]
    visible = _pack_manifest_identities(repository, global_manifest, W08_VISIBLE_PACK_KEYS)
    baseline_head = _baseline_public_head(repository)
    return {
        "ablation_keys": list(W08_ABLATION_KEYS),
        "aggregation_policy": "ALL_BEARING_DIMENSIONS_MUST_PASS",
        "artifact_kind": W08_AUTHORITY_KIND,
        "artifact_version": W08_AUTHORITY_VERSION,
        "baseline_public_head_commit_sha1": baseline_head,
        "dimension_keys": list(W08_DIMENSION_KEYS),
        "execution_state": dict(W08_ZERO_EXECUTION_STATE),
        "format_version": 1,
        "generation_account": {
            "aggregate_with_source_replay": 0,
            "allowed_transition": ["NE_NOT_YET_EVALUABLE", "RUNTIME_EVIDENCED_BOUNDED"],
            "failure_or_ne_rolls_back_to": "NE_NOT_YET_EVALUABLE",
        },
        "lc16_scope": {
            "carrier_count": 9,
            "directions": ["GENERATION", "REASONING", "UNDERSTANDING"],
            "scope_key": "DISCOURSE_REFERENCE_GENERATION",
        },
        "parent_identities": parent_identities,
        "p3ia_boundary": {
            "allowed_w08_case_sources": [
                "W08_VISIBLE_DISCOURSE_REVISION",
                "LC07_LC08_LC14",
                "WIKIPEDIA_WIKTIONARY_UD",
            ],
            "future_pack_key": W08_FUTURE_PACK_KEYS[0],
            "future_payload_reads": 0,
            "independent_course_stage": "W-09",
            "w08_failure_policy": "NE_BLOCKED",
        },
        "retention_identities": retention,
        "stage_inventory": {
            "candidate_allowed_splits": list(visibility.get("candidate_allowed_splits", [])),
            "candidate_forbidden_splits": list(visibility.get("candidate_forbidden_splits", [])),
            "dev_pack_keys": list(visibility.get("dev_pack_keys", [])),
            "evaluator_pack_keys": list(visibility.get("evaluator_pack_keys", [])),
            "future_pack_keys": list(future),
            "held_out_pack_keys": list(visibility.get("held_out_pack_keys", [])),
            "train_pack_keys": list(train),
        },
        "subtask_order": list(W08_SUBTASK_ORDER),
        "visible_pack_identities": visible,
    }


def canonical_w08_authority_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def validate_w08_authority(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _ROOT_FIELDS:
        raise W08AuthorityError("W-08 authority fields drifted")
    if (
        value.get("artifact_kind") != W08_AUTHORITY_KIND
        or value.get("artifact_version") != W08_AUTHORITY_VERSION
        or value.get("format_version") != 1
        or tuple(value.get("dimension_keys", ())) != W08_DIMENSION_KEYS
        or tuple(value.get("ablation_keys", ())) != W08_ABLATION_KEYS
        or tuple(value.get("subtask_order", ())) != W08_SUBTASK_ORDER
        or value.get("aggregation_policy") != "ALL_BEARING_DIMENSIONS_MUST_PASS"
        or value.get("execution_state") != W08_ZERO_EXECUTION_STATE
    ):
        raise W08AuthorityError("W-08 authority contract drifted")
    inventory = value.get("stage_inventory", {})
    if (
        tuple(inventory.get("future_pack_keys", ())) != W08_FUTURE_PACK_KEYS
        or set(inventory.get("future_pack_keys", ())) & set(inventory.get("train_pack_keys", ()))
    ):
        raise W08AuthorityError("W-08 future inventory drifted")
    generation = value.get("generation_account", {})
    if generation.get("aggregate_with_source_replay") != 0:
        raise W08AuthorityError("OPEN_GENERATION was merged")
    p3ia = value.get("p3ia_boundary", {})
    if p3ia.get("independent_course_stage") != "W-09" or p3ia.get("future_payload_reads") != 0:
        raise W08AuthorityError("P3-Ia future boundary drifted")
    for group in ("parent_identities", "retention_identities", "visible_pack_identities"):
        entries = value.get(group)
        if not isinstance(entries, list) or not entries:
            raise W08AuthorityError(f"W-08 authority {group} is empty")
        for item in entries:
            _safe_relative(item.get("relative_path") if isinstance(item, dict) else None)
            if (
                not isinstance(item, dict)
                or len(item.get("sha256", "")) != 64
                or type(item.get("size_bytes")) is not int
                or item["size_bytes"] <= 0
            ):
                raise W08AuthorityError(f"W-08 authority {group} identity is invalid")
    return value


def publish_w08_authority(repository_root: str | Path) -> str:
    repository = Path(repository_root).resolve()
    value = validate_w08_authority(build_w08_authority(repository))
    payload = canonical_w08_authority_bytes(value)
    destination = repository / W08_AUTHORITY_RELATIVE_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as error:
        raise W08AuthorityError("W-08 authority is append-only and already exists") from error
    return hashlib.sha256(payload).hexdigest()


def read_w08_authority(repository_root: str | Path) -> dict[str, Any]:
    repository = Path(repository_root).resolve()
    path = repository / W08_AUTHORITY_RELATIVE_PATH
    value = _canonical_object(path)
    validate_w08_authority(value)
    if canonical_w08_authority_bytes(value) != path.read_bytes():
        raise W08AuthorityError("W-08 authority is not canonical JSON")
    rebuilt = build_w08_authority(repository)
    if value != rebuilt:
        raise W08AuthorityError("W-08 authority no longer matches live parents")
    return value


__all__ = [
    "W08_ABLATION_KEYS",
    "W08_AUTHORITY_RELATIVE_PATH",
    "W08AuthorityError",
    "W08_DIMENSION_KEYS",
    "W08_FUTURE_PACK_KEYS",
    "W08_SUBTASK_ORDER",
    "build_w08_authority",
    "canonical_w08_authority_bytes",
    "publish_w08_authority",
    "read_w08_authority",
    "validate_w08_authority",
]
