"""FT15 显式、alias 与隐式已学问题特征组合。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_feature_catalog import (
    RawQuestionFeatureCatalog,
    raw_question_feature_catalog,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_alias import (
    build_learned_predicate_alias_bridge,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_alias_contract import (
    LearnedPredicateAliasBridge,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_contract import (
    RawQuestionRequest,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_implicit import (
    RawQuestionImplicitPredicateAnswerResult,
    W03W04W05ImplicitQuestionBundle,
    build_implicit_question_bundle,
    run_implicit_predicate_question_answer,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_three_role import (
    W03W04W05ThreeRoleQuestionBundle,
)


THREE_ROLE_IMPLICIT_ACTOR_SAMPLE_SHA256 = (
    "266b2a273442e0e896bc9cea270ada2c66565d9bc85dacc2d9202c598be94679")
THREE_ROLE_IMPLICIT_LOCATION_SAMPLE_SHA256 = (
    "d6113523b89a62941ee76d7d832e64c7076afe0eee02d0380883634ef6179837")
THREE_ROLE_FEATURE_CATALOG_SHA256 = (
    "93703535b5f765b3cf93d28fe0948b880cae38b0f2264741a67499b5bb3ae904")
THREE_ROLE_PREDICATE_ALIAS_BRIDGE_SHA256 = (
    "eb022365817c7c0c13427d47837981b19908de29f8ac6909df894b977f5b20ed")
THREE_ROLE_IMPLICIT_QUESTION_BUNDLE_SHA256 = (
    "aeca55301ae814ab32d7d8fb44d1ccf3a10af5e5a6a0ae2fb0e1913ae3ecf59c")
THREE_ROLE_QUESTION_FEATURE_COMPOSITION_SHA256 = (
    "45a354706d5f5c8fb05f59aaa1d54ad8709d95531ef2b3449bd5ad64351a04be")
THREE_ROLE_QUESTION_FEATURE_ANSWER_SHA256S = (
    "5c0ca6ec24a62a5a7c216cd40dbd87c12b1221d2eaaa196d41e95d46ed39f9c1",
    "dedec830a9c1ed5bbf440851612e1fbede9b32b65e982340ffcca845ba627c98",
    "f6083fbc4362b972a71d08a9ad5d4377dcab3c77c6225f13debe307475f6f7dc",
    "1cf427eda6e4f92e500b96545d1802ec98c1c841007129bb600b0a54df9a9397",
    "703ba79d08905c9a2c6509221011b954794b7ab87ef48b425fab0f8f8e3c0cc1",
    "ce68de0f323b7e56fb43ab8e7f143fe77ad71850857f9f551382236513d80d50",
    "66cb30e334e60b09f792d558b7ca4833a35d146b5d6c912154679e1f6c054dc8",
    "f4e6f2af22e0c6e0fc2c54ee773198da3b25ec064482e07541fc1ecfe5edbd15",
    "2deb106d2f27c35d688ff314d2ad7b1ce7ffe24bcabe59005a0c039ba6fab263",
    "d38da0946cc2af36e8ddfe00eb195c0adb39ab35a88ebf3e9c266a7ce5ce6513",
    "7c24a3d48ea7a1e15a13260eae1a13232bebff4ff6e86bf859bc94331b6eeba5",
    "2fa38bec0dd08fc5a984c916ef50f52a16eade335bb07159d4bf5f93b651511e",
)
THREE_ROLE_QUESTION_FEATURE_EXPRESSION_BOUNDARY = (
    ("explicit_predicate", "THREE_ROLE_TWO_TARGETS_TWO_CONTENTS"),
    ("predicate_alias", "PUBLIC_SUPERSEDE_ROUTES"),
    ("implicit_predicate", "TWO_ANSWER_FREE_LEARNED_CONSTRUCTIONS"),
    ("missing_learned_feature", "UNKNOWN"),
    ("non_equivalent_interpretations", "CLARIFY"),
    ("role_inventory", "PROVEN_FOR_THREE_ROLE_PROPOSITIONS"),
)


# object-model: exception
class W03W04W05QuestionFeatureCompositionError(ValueError):
    """FT15 特征组合目录或身份发生漂移。"""


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _identity_payload(
        feature_catalog: RawQuestionFeatureCatalog,
        alias_bridge: LearnedPredicateAliasBridge,
        implicit_bundle: W03W04W05ImplicitQuestionBundle,
        ) -> dict[str, object]:
    return {
        "alias_bridge_sha256": alias_bridge.identity_sha256,
        "explicit_feature_catalog_sha256": feature_catalog.sha256(),
        "expression_boundary": [
            {"capability": key, "status": status}
            for key, status in THREE_ROLE_QUESTION_FEATURE_EXPRESSION_BOUNDARY
        ],
        "implicit_bundle_sha256": implicit_bundle.identity_sha256,
    }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W03W04W05QuestionFeatureComposition:
    """同一共享目录上的显式、alias 与隐式已学特征。"""

    feature_catalog: RawQuestionFeatureCatalog
    alias_bridge: LearnedPredicateAliasBridge
    implicit_bundle: W03W04W05ImplicitQuestionBundle
    identity_sha256: str

    def __post_init__(self) -> None:
        if (not isinstance(self.feature_catalog, RawQuestionFeatureCatalog)
                or not isinstance(
                    self.alias_bridge, LearnedPredicateAliasBridge)
                or not isinstance(
                    self.implicit_bundle, W03W04W05ImplicitQuestionBundle)
                or self.implicit_bundle.explicit_catalog
                != self.feature_catalog
                or self.alias_bridge.raw_question_bundle_sha256
                != self.feature_catalog.bundle_identity_sha256
                or self.identity_sha256 != self.sha256()):
            raise W03W04W05QuestionFeatureCompositionError(
                "question feature composition drifted")

    def to_dict(self) -> dict[str, object]:
        return _identity_payload(
            self.feature_catalog,
            self.alias_bridge,
            self.implicit_bundle,
        )

    def sha256(self) -> str:
        return _sha(self.to_dict())


def build_three_role_question_feature_composition(
        explicit_bundle: W03W04W05ThreeRoleQuestionBundle,
        implicit_actor_sample_path: str | Path,
        implicit_location_sample_path: str | Path,
    ) -> W03W04W05QuestionFeatureComposition:
    """在 FT14 三 Role 事实之上组合全部已学问题特征。"""
    if not isinstance(explicit_bundle, W03W04W05ThreeRoleQuestionBundle):
        raise TypeError("question feature composition parent is invalid")
    feature_catalog = raw_question_feature_catalog(explicit_bundle)
    if feature_catalog.sha256() != THREE_ROLE_FEATURE_CATALOG_SHA256:
        raise W03W04W05QuestionFeatureCompositionError(
            "three-Role feature catalog commitment drifted")
    alias_bridge = build_learned_predicate_alias_bridge(
        feature_catalog,
        expected_identity_sha256=(
            THREE_ROLE_PREDICATE_ALIAS_BRIDGE_SHA256),
    )
    implicit_bundle = build_implicit_question_bundle(
        feature_catalog,
        implicit_actor_sample_path,
        implicit_location_sample_path,
        expected_reason_sample_sha256=(
            THREE_ROLE_IMPLICIT_ACTOR_SAMPLE_SHA256),
        expected_result_sample_sha256=(
            THREE_ROLE_IMPLICIT_LOCATION_SAMPLE_SHA256),
        expected_identity_sha256=(
            THREE_ROLE_IMPLICIT_QUESTION_BUNDLE_SHA256),
    )
    identity = _sha(_identity_payload(
        feature_catalog,
        alias_bridge,
        implicit_bundle,
    ))
    value = W03W04W05QuestionFeatureComposition(
        feature_catalog,
        alias_bridge,
        implicit_bundle,
        identity,
    )
    if identity != THREE_ROLE_QUESTION_FEATURE_COMPOSITION_SHA256:
        raise W03W04W05QuestionFeatureCompositionError(
            "question feature composition commitment drifted")
    return value


def run_three_role_question_feature_answer(
        composition: W03W04W05QuestionFeatureComposition,
        request: RawQuestionRequest,
    ) -> RawQuestionImplicitPredicateAnswerResult:
    """依次分派显式构造、已学 alias 与已学隐式结构。"""
    if (not isinstance(composition, W03W04W05QuestionFeatureComposition)
            or not isinstance(request, RawQuestionRequest)):
        raise TypeError("question feature composition runtime inputs are invalid")
    catalog = composition.feature_catalog
    return run_implicit_predicate_question_answer(
        composition.alias_bridge,
        composition.implicit_bundle,
        catalog.w03_batch,
        catalog.w04_batch,
        catalog.w05_batch,
        request,
        overlay_validation_sha256=catalog.overlay_validation_sha256,
    )


__all__ = [
    "THREE_ROLE_FEATURE_CATALOG_SHA256",
    "THREE_ROLE_IMPLICIT_ACTOR_SAMPLE_SHA256",
    "THREE_ROLE_IMPLICIT_LOCATION_SAMPLE_SHA256",
    "THREE_ROLE_IMPLICIT_QUESTION_BUNDLE_SHA256",
    "THREE_ROLE_PREDICATE_ALIAS_BRIDGE_SHA256",
    "THREE_ROLE_QUESTION_FEATURE_ANSWER_SHA256S",
    "THREE_ROLE_QUESTION_FEATURE_COMPOSITION_SHA256",
    "THREE_ROLE_QUESTION_FEATURE_EXPRESSION_BOUNDARY",
    "W03W04W05QuestionFeatureComposition",
    "W03W04W05QuestionFeatureCompositionError",
    "build_three_role_question_feature_composition",
    "run_three_role_question_feature_answer",
]
