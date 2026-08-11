"""在两条来源绑定三 Role 事实上学习两种问题构造。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question import (
    build_raw_question_catalog,
    compile_raw_question_pattern,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_contract import (
    RawQuestionConstruction,
    RawQuestionPattern,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_vertical import (
    run_w03_w04_w05_vertical_query,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_vertical_contract import (
    W03W04W05VerticalQuery,
    W03W04W05VerticalResult,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_vertical_three_role import (
    THREE_ROLE_VERTICAL_TARGETS,
    W03W04W05ThreeRoleVerticalOverlay,
)


THREE_ROLE_ACTOR_QUESTION_SAMPLE_SHA256 = (
    "9b3e8fc8fbd3687d95857f7e69bff49be370df0456c6a7d98454a7dba2c14ed0")
THREE_ROLE_LOCATION_QUESTION_SAMPLE_SHA256 = (
    "5a9b094da5202733998562f83e59c1aa14fbea751f20c09a01cf386e1ff85c95")
THREE_ROLE_QUESTION_VERTICAL_SHA256S = (
    "f31d50dde1a760a57f8ffe09833ae7d7d0c2609885c3cbc5ee29f49d8283e246",
    "9b713d819b98ed1a8cafaea76b8260df5354fbcef4437bb4ee9b1f4d4598c90f",
)
THREE_ROLE_QUESTION_BUNDLE_SHA256 = (
    "9ac37ef16b5a1bc3bc142dd771980c419446d1782b41ec897c4f3d103ef1d590")
THREE_ROLE_QUESTION_EXPRESSION_BOUNDARY = (
    (
        "construction_replacement",
        "SUPPORTED_BY_TWO_LEARNED_CONSTRUCTIONS",
    ),
    (
        "content_replacement",
        "SUPPORTED_BY_TWO_SOURCE_BOUND_PROPOSITIONS",
    ),
    ("role_inventory", "PROVEN_FOR_THREE_ROLE_PROPOSITIONS"),
    ("target_role_count", "TWO_DISTINCT_TARGET_ROLES"),
    (
        "four_role_or_more",
        "UNKNOWN_UNTIL_PUBLIC_VERTICAL_DEPENDENCIES_EXIST",
    ),
)


# object-model: exception
class W03W04W05ThreeRoleQuestionError(ValueError):
    """已学三 Role 问题课程或交叉应用目录发生漂移。"""


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _vertical_results(
        overlay: W03W04W05ThreeRoleVerticalOverlay,
        ) -> tuple[W03W04W05VerticalResult, ...]:
    results = tuple(
        run_w03_w04_w05_vertical_query(
            overlay.w03_batch,
            overlay.w04_batch,
            overlay.w05_batch,
            W03W04W05VerticalQuery(
                spec.surface,
                spec.context,
                spec.proposition_surface,
            ),
            overlay_validation_sha256=overlay.validation_sha256,
        )
        for spec in THREE_ROLE_VERTICAL_TARGETS
    )
    if (tuple(item.sha256() for item in results)
            != THREE_ROLE_QUESTION_VERTICAL_SHA256S
            or any(item.status != "BRIDGED" or item.link is None
                   for item in results)):
        raise W03W04W05ThreeRoleQuestionError(
            "three-Role vertical results drifted")
    return results


def _identity_payload(
        overlay: W03W04W05ThreeRoleVerticalOverlay,
        vertical_results: tuple[W03W04W05VerticalResult, ...],
        patterns: tuple[RawQuestionPattern, ...],
        catalog: tuple[RawQuestionConstruction, ...],
        ) -> dict[str, object]:
    return {
        "catalog": [item.to_dict() for item in catalog],
        "expression_boundary": [
            {"capability": key, "status": value}
            for key, value in THREE_ROLE_QUESTION_EXPRESSION_BOUNDARY
        ],
        "overlay_validation_sha256": overlay.validation_sha256,
        "patterns": [item.to_dict() for item in patterns],
        "vertical_results": [
            item.to_dict() for item in vertical_results
        ],
    }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W03W04W05ThreeRoleQuestionBundle:
    """保存两个构造与两个三 Role 命题的完整交叉应用。"""

    overlay: W03W04W05ThreeRoleVerticalOverlay
    vertical_results: tuple[W03W04W05VerticalResult, ...]
    patterns: tuple[RawQuestionPattern, ...]
    catalog: tuple[RawQuestionConstruction, ...]
    identity_sha256: str

    def __post_init__(self) -> None:
        if (not isinstance(
                self.overlay, W03W04W05ThreeRoleVerticalOverlay)
                or len(self.vertical_results) != 2
                or any(not isinstance(item, W03W04W05VerticalResult)
                       for item in self.vertical_results)
                or len(self.patterns) != 2
                or any(not isinstance(item, RawQuestionPattern)
                       for item in self.patterns)
                or len(self.catalog) != 4
                or any(not isinstance(item, RawQuestionConstruction)
                       for item in self.catalog)
                or not isinstance(self.identity_sha256, str)
                or len(self.identity_sha256) != 64):
            raise W03W04W05ThreeRoleQuestionError(
                "three-Role question bundle inventory drifted")
        pattern_ids = {item.pattern.sha256() for item in self.catalog}
        source_ids = {item.source_record_key for item in self.catalog}
        target_roles = {item.target_role_key for item in self.catalog}
        pairs = {
            (item.pattern.sha256(), item.source_record_key)
            for item in self.catalog
        }
        if (len(pattern_ids) != 2 or len(source_ids) != 2
                or len(target_roles) != 2 or len(pairs) != 4
                or self.identity_sha256 != self.sha256()):
            raise W03W04W05ThreeRoleQuestionError(
                "three-Role construction/content cross product drifted")

    def to_dict(self) -> dict[str, object]:
        return _identity_payload(
            self.overlay,
            self.vertical_results,
            self.patterns,
            self.catalog,
        )

    def sha256(self) -> str:
        return _sha(self.to_dict())


def build_three_role_question_bundle(
        overlay: W03W04W05ThreeRoleVerticalOverlay,
        actor_sample_path: str | Path,
        location_sample_path: str | Path,
        ) -> W03W04W05ThreeRoleQuestionBundle:
    """学习两种 target Role 段计划，并应用到两条已学事实。"""
    if not isinstance(overlay, W03W04W05ThreeRoleVerticalOverlay):
        raise TypeError("three-Role question overlay is invalid")
    verticals = _vertical_results(overlay)
    patterns = tuple(sorted(
        (
            compile_raw_question_pattern(
                actor_sample_path,
                verticals[0],
                expected_sample_sha256=(
                    THREE_ROLE_ACTOR_QUESTION_SAMPLE_SHA256),
            ),
            compile_raw_question_pattern(
                location_sample_path,
                verticals[0],
                expected_sample_sha256=(
                    THREE_ROLE_LOCATION_QUESTION_SAMPLE_SHA256),
            ),
        ),
        key=lambda item: item.sha256(),
    ))
    catalog = build_raw_question_catalog(patterns, verticals)
    identity = _sha(_identity_payload(
        overlay,
        verticals,
        patterns,
        catalog,
    ))
    value = W03W04W05ThreeRoleQuestionBundle(
        overlay,
        verticals,
        patterns,
        catalog,
        identity,
    )
    if identity != THREE_ROLE_QUESTION_BUNDLE_SHA256:
        raise W03W04W05ThreeRoleQuestionError(
            "three-Role question bundle identity drifted")
    return value


__all__ = [
    "THREE_ROLE_ACTOR_QUESTION_SAMPLE_SHA256",
    "THREE_ROLE_LOCATION_QUESTION_SAMPLE_SHA256",
    "THREE_ROLE_QUESTION_BUNDLE_SHA256",
    "THREE_ROLE_QUESTION_EXPRESSION_BOUNDARY",
    "THREE_ROLE_QUESTION_VERTICAL_SHA256S",
    "W03W04W05ThreeRoleQuestionBundle",
    "W03W04W05ThreeRoleQuestionError",
    "build_three_role_question_bundle",
]
