"""从冻结 v32 基线构建 LC-13 三向 consumer 账。"""
from __future__ import annotations

import hashlib
from pathlib import Path

from pure_integer_ai.experiments.ph2_dataset_contract import CanonicalJsonObject
from pure_integer_ai.experiments.ph2_directional_consumer_contract import (
    ARTIFACT_STATUS,
    DIRECTIONS,
    EXECUTION_STATE,
    FORMAT_VERSION,
    POSTCHECK_DIMENSIONS,
    RUNTIME_STATUS,
    VERIFIER_DIMENSIONS,
    VERIFIER_NE_CONDITIONS,
    DirectionalConsumerEvidenceFile,
    DirectionalConsumerManifest,
    DirectionalConsumerRoute,
)
from pure_integer_ai.experiments.ph2_language_baseline_manifest import (
    read_language_baseline_manifest,
)


LC13_MANIFEST_PATH = Path(
    "data/ph2/manifests/lc13_directional_consumer_manifest_v1.json")
LC13_ARTIFACT_VERSION = "LC-13-directional-consumer-manifest-v1"
BASELINE_MANIFEST_PATH = Path(
    "data/ph2/manifests/language_capability_baseline_v32.json")
_POSTCHECK_KEYS = {
    "GENERATION": "LC13_GENERATION_LAYERED_POSTCHECK_V1",
    "REASONING": "LC13_REASONING_PROOF_SCOPE_POSTCHECK_V1",
    "UNDERSTANDING": "LC13_UNDERSTANDING_OBJECT_SCOPE_POSTCHECK_V1",
}


class DirectionalConsumerCatalogError(RuntimeError):
    """冻结基线、consumer 文件身份或方向映射无法闭合。"""


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _route(capability_key: str, direction: str, value: dict) -> DirectionalConsumerRoute:
    """把 LC-00 一个方向槽投影成不夸大连接状态的 LC-13 路由。"""
    applicability = value["applicability"]
    fact_state = value["fact_state"]
    refs = tuple(value["consumer_refs"])
    permissions = tuple(value["write_permissions"])
    if applicability == "N_A":
        return DirectionalConsumerRoute(
            capability_key,
            direction,
            applicability,
            fact_state,
            "OUT_OF_SCOPE",
            (),
            "OUT_OF_SCOPE",
            (),
            "OUT_OF_SCOPE",
            _POSTCHECK_KEYS[direction],
            "OUT_OF_SCOPE",
            POSTCHECK_DIMENSIONS[direction],
            "OUT_OF_SCOPE",
            ("NON_TEXT_WALL_OUT_OF_SCOPE",),
            0,
        )
    if refs:
        exact_state = (
            "CONTRACT_FROZEN_NOT_CONSUMED"
            if (capability_key == "LAYERED_GENERATION"
                and direction == "GENERATION")
            else "REQUIRED_NOT_CONNECTED")
        ne = {
            "FORMAL_W_RUNTIME_NOT_STARTED",
            "LC13_ROUTE_NOT_EXECUTED",
            "POSTCHECK_NOT_EXECUTED",
        }
        if exact_state == "CONTRACT_FROZEN_NOT_CONSUMED":
            ne.add("ASSESSMENT_CONSUMER_NOT_CONNECTED")
        return DirectionalConsumerRoute(
            capability_key,
            direction,
            applicability,
            fact_state,
            "AVAILABLE_NOT_EXECUTED",
            refs,
            f"LC13_{capability_key}_{direction}_OWNER",
            permissions,
            exact_state,
            _POSTCHECK_KEYS[direction],
            "AVAILABLE_NOT_EXECUTED",
            POSTCHECK_DIMENSIONS[direction],
            "NE",
            tuple(sorted(ne)),
            0,
        )
    return DirectionalConsumerRoute(
        capability_key,
        direction,
        applicability,
        fact_state,
        "MISSING_NE",
        (),
        "UNASSIGNED_NE",
        (),
        "REQUIRED_NOT_CONNECTED",
        _POSTCHECK_KEYS[direction],
        "REQUIRED_NOT_CONNECTED",
        POSTCHECK_DIMENSIONS[direction],
        "NE",
        (
            "DIRECTIONAL_CONSUMER_NOT_CONNECTED",
            "FORMAL_W_RUNTIME_NOT_STARTED",
        ),
        0,
    )


def _evidence_inventory(
        repository_root: Path,
        routes: tuple[DirectionalConsumerRoute, ...],
        ) -> tuple[DirectionalConsumerEvidenceFile, ...]:
    result = []
    for relative_path in sorted({
            path for route in routes for path in route.consumer_refs}):
        path = repository_root / Path(*relative_path.split("/"))
        if not path.is_file():
            raise DirectionalConsumerCatalogError(
                f"LC-13 consumer 文件缺失: {relative_path}")
        result.append(DirectionalConsumerEvidenceFile(
            relative_path, path.stat().st_size, _sha256_path(path)))
    return tuple(result)


def build_directional_consumer_manifest(
        repository_root: str | Path,
        ) -> DirectionalConsumerManifest:
    """回读 v32 能力账并冻结完整 20×3 consumer map。"""
    repository = Path(repository_root).resolve()
    baseline_path = repository / BASELINE_MANIFEST_PATH
    if not baseline_path.is_file():
        raise DirectionalConsumerCatalogError("LC-13 基线文件缺失")
    baseline = read_language_baseline_manifest(baseline_path)
    routes = tuple(
        _route(entry.capability_key, direction, values[direction])
        for entry in baseline.capability_ledger.entries
        for values in (entry.directional_consumption.to_value(),)
        for direction in DIRECTIONS
    )
    available = sum(
        item.consumer_state == "AVAILABLE_NOT_EXECUTED" for item in routes)
    missing = sum(item.consumer_state == "MISSING_NE" for item in routes)
    out_of_scope = sum(
        item.consumer_state == "OUT_OF_SCOPE" for item in routes)
    exact = sum(
        item.exact_use_outcome_state == "CONTRACT_FROZEN_NOT_CONSUMED"
        for item in routes)
    return DirectionalConsumerManifest(
        FORMAT_VERSION,
        LC13_ARTIFACT_VERSION,
        ARTIFACT_STATUS,
        RUNTIME_STATUS,
        "LC-13",
        BASELINE_MANIFEST_PATH.as_posix(),
        _sha256_path(baseline_path),
        routes,
        _evidence_inventory(repository, routes),
        len(routes),
        available,
        missing,
        out_of_scope,
        exact,
        0,
        VERIFIER_DIMENSIONS,
        VERIFIER_NE_CONDITIONS,
        CanonicalJsonObject.from_value(EXECUTION_STATE),
    )


__all__ = [
    "BASELINE_MANIFEST_PATH",
    "LC13_ARTIFACT_VERSION",
    "LC13_MANIFEST_PATH",
    "DirectionalConsumerCatalogError",
    "build_directional_consumer_manifest",
]
