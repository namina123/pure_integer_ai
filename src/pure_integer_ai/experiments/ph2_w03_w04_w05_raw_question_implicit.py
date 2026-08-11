"""学习没有谓词表层、但仍绑定来源命题的问题构造。"""
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
    run_raw_question_feature_answer,
    run_raw_question_feature_candidate_answer,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question import (
    build_raw_question_catalog,
    compile_raw_question_pattern,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_alias import (
    run_question_feature_predicate_alias_answer,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_alias_contract import (
    LearnedPredicateAliasBridge,
    RawQuestionPredicateAliasAnswerResult,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_contract import (
    RAW_QUESTION_STATUSES,
    RawQuestionAnswerResult,
    RawQuestionConstruction,
    RawQuestionPattern,
    RawQuestionRequest,
)


IMPLICIT_QUESTION_REASON_SAMPLE_SHA256 = (
    "be32062c4ba221d2c1f81dae540fa31425fa7ab55e3c3765c6d0aec1b09e0fd8")
IMPLICIT_QUESTION_RESULT_SAMPLE_SHA256 = (
    "378bdc58a4b57d2a31e4d35a60e299cbc4a1f174229454efb60fbe7bb1f60f0d")
IMPLICIT_QUESTION_BUNDLE_SHA256 = (
    "5472fe7dab4eb077029cae5db1162da73c2bb6ac98411571c89e2d431c4df82a")
IMPLICIT_QUESTION_ANSWER_SHA256S = (
    "00016d006ed583ac01971791a5e91276f14715e9191329c66fbabec98e675c16",
    "6a16e6af54341585d7b27bd2c5ddceb21ef866d6627dacb9d9f1eabd9862fedb",
    "85e52553582ef93f412522f523bd3311146fe675f4df550977fd701f0d6d3382",
    "bfa679b0e4a137b315d16821b4db8b36675d251a3e709621ec482b5223600992",
)
IMPLICIT_QUESTION_EXPRESSION_BOUNDARY = (
    ("explicit_or_alias_predicate", "PRESERVED_FROM_FT12"),
    ("implicit_predicate", "SUPPORTED_BY_LEARNED_CONSTRUCTION"),
    ("construction_replacement", "TWO_INDEPENDENT_PATTERNS"),
    ("content_replacement", "TWO_SOURCE_BOUND_PROPOSITIONS"),
    ("missing_structure", "UNKNOWN"),
    ("non_equivalent_interpretations", "CLARIFY"),
    ("role_inventory", "CURRENTLY_PROVEN_FOR_TWO_ROLE_PROPOSITIONS"),
)
IMPLICIT_QUESTION_INTERPRETATION_STATUSES = {
    "AMBIGUOUS", "MISSING", "SELECTED"}


# object-model: exception
class W03W04W05ImplicitQuestionError(ValueError):
    """隐式问题构造、来源绑定目录或回答投影发生漂移。"""


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _strict_key(value: tuple[int, ...], *, where: str) -> tuple[int, ...]:
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int for item in value)):
        raise W03W04W05ImplicitQuestionError(
            f"{where} is not a strict integer key")
    return value


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ImplicitQuestionInterpretationKey:
    """隐式构造恢复出的 primitive、Proposition 与目标 Role 解释。"""

    primitive_registry: str
    primitive_kind: int
    proposition_key: tuple[int, ...]
    target_role_key: tuple[int, ...]

    def __post_init__(self) -> None:
        if (not isinstance(self.primitive_registry, str)
                or not self.primitive_registry
                or self.primitive_registry.strip()
                != self.primitive_registry
                or type(self.primitive_kind) is not int
                or self.primitive_kind <= 0):
            raise W03W04W05ImplicitQuestionError(
                "implicit interpretation primitive drifted")
        _strict_key(
            self.proposition_key,
            where="implicit interpretation Proposition",
        )
        _strict_key(
            self.target_role_key,
            where="implicit interpretation target Role",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "primitive_kind": self.primitive_kind,
            "primitive_registry": self.primitive_registry,
            "proposition_key": list(self.proposition_key),
            "target_role_key": list(self.target_role_key),
        }

    def sha256(self) -> str:
        return _sha(self.to_dict())


def _interpretation_key(
        construction: RawQuestionConstruction,
        ) -> ImplicitQuestionInterpretationKey:
    link = construction.vertical_result.link
    if link is None:
        raise W03W04W05ImplicitQuestionError(
            "implicit construction lacks a vertical link")
    return ImplicitQuestionInterpretationKey(
        link.primitive_registry,
        link.primitive_kind,
        link.proposition_key,
        construction.target_role_key,
    )


def resolve_implicit_question_interpretations(
        keys: tuple[ImplicitQuestionInterpretationKey, ...],
        ) -> str:
    """等价来源轨迹收敛；primitive、命题或目标 Role 分歧则歧义。"""
    if (not isinstance(keys, tuple)
            or any(not isinstance(item, ImplicitQuestionInterpretationKey)
                   for item in keys)):
        raise TypeError("implicit interpretation keys are invalid")
    identities = {item.sha256() for item in keys}
    if not identities:
        return "MISSING"
    if len(identities) == 1:
        return "SELECTED"
    return "AMBIGUOUS"


def _identity_payload(
        explicit_catalog: RawQuestionFeatureCatalog,
        patterns: tuple[RawQuestionPattern, ...],
        catalog: tuple[RawQuestionConstruction, ...],
        ) -> dict[str, object]:
    return {
        "catalog": [item.to_dict() for item in catalog],
        "explicit_bundle_sha256": (
            explicit_catalog.bundle_identity_sha256),
        "expression_boundary": [
            {"capability": key, "status": status}
            for key, status in IMPLICIT_QUESTION_EXPRESSION_BOUNDARY
        ],
        "overlay_validation_sha256": (
            explicit_catalog.overlay_validation_sha256),
        "patterns": [item.to_dict() for item in patterns],
        "vertical_result_sha256s": [
            item.sha256() for item in explicit_catalog.vertical_results
        ],
    }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W03W04W05ImplicitQuestionBundle:
    """两个隐式构造交叉应用到两个来源绑定 Proposition 的目录。"""

    explicit_catalog: RawQuestionFeatureCatalog
    patterns: tuple[RawQuestionPattern, ...]
    catalog: tuple[RawQuestionConstruction, ...]
    identity_sha256: str

    def __post_init__(self) -> None:
        if (not isinstance(self.explicit_catalog, RawQuestionFeatureCatalog)
                or len(self.patterns) != 2
                or any(not isinstance(item, RawQuestionPattern)
                       for item in self.patterns)
                or len(self.catalog) != 4
                or any(not isinstance(item, RawQuestionConstruction)
                       for item in self.catalog)
                or not isinstance(self.identity_sha256, str)
                or len(self.identity_sha256) != 64):
            raise W03W04W05ImplicitQuestionError(
                "implicit question inventory drifted")
        if any(
                any(segment.kind == "PREDICATE"
                    for segment in pattern.segments)
                for pattern in self.patterns):
            raise W03W04W05ImplicitQuestionError(
                "implicit question pattern published a predicate segment")
        pattern_ids = {item.pattern.sha256() for item in self.catalog}
        source_ids = {item.source_record_key for item in self.catalog}
        pairs = {
            (item.pattern.sha256(), item.source_record_key)
            for item in self.catalog
        }
        identities = tuple(item.sha256() for item in self.catalog)
        if (len(pattern_ids) != 2 or len(source_ids) != 2
                or len(pairs) != 4
                or identities != tuple(sorted(identities))
                or len(set(identities)) != len(identities)
                or self.identity_sha256 != self.sha256()):
            raise W03W04W05ImplicitQuestionError(
                "implicit construction/content cross product drifted")

    def to_dict(self) -> dict[str, object]:
        return _identity_payload(
            self.explicit_catalog,
            self.patterns,
            self.catalog,
        )

    def sha256(self) -> str:
        return _sha(self.to_dict())


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class RawQuestionImplicitPredicateAnswerResult:
    """FT12 的原结果以及可选的隐式构造 FT09 执行结果。"""

    request: RawQuestionRequest
    status: str
    answer_surface: str | None
    predicate_result: RawQuestionPredicateAliasAnswerResult
    implicit_result: RawQuestionAnswerResult | None
    experimental: int = 1
    formal_mastery_claim: int = 0
    w03_started: int = 0
    w04_started: int = 0
    w05_started: int = 0

    def __post_init__(self) -> None:
        if (not isinstance(self.request, RawQuestionRequest)
                or self.status not in RAW_QUESTION_STATUSES
                or not isinstance(
                    self.predicate_result,
                    RawQuestionPredicateAliasAnswerResult)
                or self.predicate_result.request != self.request):
            raise W03W04W05ImplicitQuestionError(
                "implicit question answer projection drifted")
        if self.predicate_result.status != "UNKNOWN":
            if (self.implicit_result is not None
                    or self.status != self.predicate_result.status
                    or self.answer_surface
                    != self.predicate_result.answer_surface):
                raise W03W04W05ImplicitQuestionError(
                    "FT12 result was not preserved")
        else:
            if (not isinstance(self.implicit_result, RawQuestionAnswerResult)
                    or self.implicit_result.request != self.request
                    or self.status != self.implicit_result.status
                    or self.answer_surface
                    != self.implicit_result.answer_surface):
                raise W03W04W05ImplicitQuestionError(
                    "implicit result escaped the learned construction")
        if (self.experimental, self.formal_mastery_claim, self.w03_started,
                self.w04_started, self.w05_started) != (1, 0, 0, 0, 0):
            raise W03W04W05ImplicitQuestionError(
                "implicit question boundary flags drifted")

    def to_dict(self) -> dict[str, object]:
        return {
            "answer_surface": self.answer_surface,
            "experimental": self.experimental,
            "formal_mastery_claim": self.formal_mastery_claim,
            "implicit_result": (
                None if self.implicit_result is None
                else self.implicit_result.to_dict()
            ),
            "predicate_result": self.predicate_result.to_dict(),
            "request": self.request.to_dict(),
            "status": self.status,
            "w03_started": self.w03_started,
            "w04_started": self.w04_started,
            "w05_started": self.w05_started,
        }

    def sha256(self) -> str:
        return _sha(self.to_dict())


def build_implicit_question_bundle(
        explicit_source: object,
        reason_sample_path: str | Path,
        result_sample_path: str | Path,
        *,
        expected_reason_sample_sha256: str = (
            IMPLICIT_QUESTION_REASON_SAMPLE_SHA256),
        expected_result_sample_sha256: str = (
            IMPLICIT_QUESTION_RESULT_SAMPLE_SHA256),
        expected_identity_sha256: str = IMPLICIT_QUESTION_BUNDLE_SHA256,
        ) -> W03W04W05ImplicitQuestionBundle:
    """学习两个无谓词表层的结构并形成双构造、双内容目录。"""
    explicit_catalog = raw_question_feature_catalog(explicit_source)
    verticals = explicit_catalog.vertical_results
    patterns = tuple(sorted(
        (
            compile_raw_question_pattern(
                reason_sample_path,
                verticals[0],
                expected_sample_sha256=expected_reason_sample_sha256,
            ),
            compile_raw_question_pattern(
                result_sample_path,
                verticals[1],
                expected_sample_sha256=expected_result_sample_sha256,
            ),
        ),
        key=RawQuestionPattern.sha256,
    ))
    if any(
            any(segment.kind == "PREDICATE"
                for segment in pattern.segments)
            for pattern in patterns):
        raise W03W04W05ImplicitQuestionError(
            "implicit sample compiled an explicit predicate")
    catalog = build_raw_question_catalog(patterns, verticals)
    identity = _sha(_identity_payload(explicit_catalog, patterns, catalog))
    value = W03W04W05ImplicitQuestionBundle(
        explicit_catalog,
        patterns,
        catalog,
        identity,
    )
    if identity != expected_identity_sha256:
        raise W03W04W05ImplicitQuestionError(
            "implicit question bundle identity drifted")
    return value


def run_implicit_question_catalog_answer(
        bundle: W03W04W05ImplicitQuestionBundle,
        request: RawQuestionRequest,
        ) -> RawQuestionAnswerResult:
    """前序阶段均未知时，只运行已学隐式构造目录。"""
    return _run_implicit_question_catalog_answer(
        bundle,
        request,
        bundle.catalog,
    )


def run_implicit_question_candidate_answer(
        bundle: W03W04W05ImplicitQuestionBundle,
        request: RawQuestionRequest,
        candidate_constructions: tuple[RawQuestionConstruction, ...],
        *,
        state_sha256: str | None = None,
        ) -> RawQuestionAnswerResult:
    """只在已由不可变索引验证的隐式构式候选中执行 FT13。"""
    if (not isinstance(candidate_constructions, tuple)
            or not candidate_constructions
            or any(not isinstance(item, RawQuestionConstruction)
                   for item in candidate_constructions)):
        raise TypeError("implicit question candidate inputs are invalid")
    return _run_implicit_question_catalog_answer(
        bundle,
        request,
        candidate_constructions,
        state_sha256=state_sha256,
    )


def _run_implicit_question_catalog_answer(
        bundle: W03W04W05ImplicitQuestionBundle,
        request: RawQuestionRequest,
        candidate_constructions: tuple[RawQuestionConstruction, ...],
        *,
        state_sha256: str | None = None,
        ) -> RawQuestionAnswerResult:
    if (not isinstance(bundle, W03W04W05ImplicitQuestionBundle)
            or not isinstance(request, RawQuestionRequest)):
        raise TypeError("implicit catalog inputs are invalid")
    matches = tuple(
        item for item in candidate_constructions
        if item.question_surface == request.question_surface
        and (request.source_record_key is None
             or item.source_record_key == request.source_record_key)
    )
    resolution = resolve_implicit_question_interpretations(tuple(
        _interpretation_key(item) for item in matches))
    if resolution == "MISSING":
        implicit_catalog = RawQuestionFeatureCatalog(
            bundle.explicit_catalog.bundle_identity_sha256,
            bundle.explicit_catalog.overlay_validation_sha256,
            bundle.explicit_catalog.vertical_results,
            bundle.patterns,
            bundle.catalog,
            bundle.explicit_catalog.w03_batch,
            bundle.explicit_catalog.w04_batch,
            bundle.explicit_catalog.w05_batch,
        )
        return run_raw_question_feature_candidate_answer(
            implicit_catalog,
            request,
            candidate_constructions,
            state_sha256=state_sha256,
        )
    if resolution == "AMBIGUOUS":
        return RawQuestionAnswerResult(
            request,
            "CLARIFY",
            None,
            tuple(sorted(item.sha256() for item in matches)),
            None,
            None,
        )
    selected = min(matches, key=RawQuestionConstruction.sha256)
    selected_catalog = RawQuestionFeatureCatalog(
        bundle.explicit_catalog.bundle_identity_sha256,
        bundle.explicit_catalog.overlay_validation_sha256,
        (selected.vertical_result,),
        (selected.pattern,),
        (selected,),
        bundle.explicit_catalog.w03_batch,
        bundle.explicit_catalog.w04_batch,
        bundle.explicit_catalog.w05_batch,
    )
    return run_raw_question_feature_answer(
        selected_catalog,
        request,
        state_sha256=state_sha256,
    )


def run_implicit_predicate_question_answer(
        alias_bridge: LearnedPredicateAliasBridge,
        implicit_bundle: W03W04W05ImplicitQuestionBundle,
        w03_batch,
        w04_batch,
        w05_batch,
        request: RawQuestionRequest,
        *,
        overlay_validation_sha256: str,
        ) -> RawQuestionImplicitPredicateAnswerResult:
    """保留 FT12；只有未知时才查询来源绑定的隐式构造目录。"""
    if (not isinstance(alias_bridge, LearnedPredicateAliasBridge)
            or not isinstance(
                implicit_bundle, W03W04W05ImplicitQuestionBundle)
            or not isinstance(request, RawQuestionRequest)):
        raise TypeError("implicit predicate question inputs are invalid")
    explicit = implicit_bundle.explicit_catalog
    if ((w03_batch, w04_batch, w05_batch)
            != (explicit.w03_batch, explicit.w04_batch, explicit.w05_batch)
            or overlay_validation_sha256
            != explicit.overlay_validation_sha256):
        raise W03W04W05ImplicitQuestionError(
            "implicit runtime escaped its feature catalog")
    predicate_result = run_question_feature_predicate_alias_answer(
        alias_bridge,
        explicit,
        request,
    )
    return continue_implicit_predicate_question_answer(
        alias_bridge,
        implicit_bundle,
        request,
        predicate_result,
    )


def continue_implicit_predicate_question_answer(
        alias_bridge: LearnedPredicateAliasBridge,
        implicit_bundle: W03W04W05ImplicitQuestionBundle,
        request: RawQuestionRequest,
        predicate_result: RawQuestionPredicateAliasAnswerResult,
        ) -> RawQuestionImplicitPredicateAnswerResult:
    """从保留的 FT12 结果继续 FT13，不重复执行前序阶段。"""
    return _continue_implicit_predicate_question_answer(
        alias_bridge,
        implicit_bundle,
        request,
        predicate_result,
        None,
    )


def continue_implicit_predicate_question_candidate_answer(
        alias_bridge: LearnedPredicateAliasBridge,
        implicit_bundle: W03W04W05ImplicitQuestionBundle,
        request: RawQuestionRequest,
        predicate_result: RawQuestionPredicateAliasAnswerResult,
        candidate_constructions: tuple[RawQuestionConstruction, ...],
        *,
        state_sha256: str | None = None,
        ) -> RawQuestionImplicitPredicateAnswerResult:
    """从 FT12 结果继续，只执行索引选出的隐式构式。"""
    if (not isinstance(candidate_constructions, tuple)
            or not candidate_constructions
            or any(not isinstance(item, RawQuestionConstruction)
                   for item in candidate_constructions)):
        raise TypeError("implicit candidate continuation inputs are invalid")
    return _continue_implicit_predicate_question_answer(
        alias_bridge,
        implicit_bundle,
        request,
        predicate_result,
        candidate_constructions,
        state_sha256=state_sha256,
    )


def _continue_implicit_predicate_question_answer(
        alias_bridge: LearnedPredicateAliasBridge,
        implicit_bundle: W03W04W05ImplicitQuestionBundle,
        request: RawQuestionRequest,
        predicate_result: RawQuestionPredicateAliasAnswerResult,
        candidate_constructions: tuple[RawQuestionConstruction, ...] | None,
        *,
        state_sha256: str | None = None,
        ) -> RawQuestionImplicitPredicateAnswerResult:
    if (not isinstance(alias_bridge, LearnedPredicateAliasBridge)
            or not isinstance(
                implicit_bundle, W03W04W05ImplicitQuestionBundle)
            or not isinstance(request, RawQuestionRequest)
            or not isinstance(
                predicate_result, RawQuestionPredicateAliasAnswerResult)):
        raise TypeError("implicit continuation inputs are invalid")
    if predicate_result.request != request:
        raise W03W04W05ImplicitQuestionError(
            "implicit continuation escaped its predicate request")
    explicit = implicit_bundle.explicit_catalog
    if (alias_bridge.overlay_validation_sha256
            != explicit.overlay_validation_sha256
            or alias_bridge.raw_question_bundle_sha256
            != explicit.bundle_identity_sha256):
        raise W03W04W05ImplicitQuestionError(
            "implicit continuation escaped its learned catalog")
    if predicate_result.status != "UNKNOWN":
        return RawQuestionImplicitPredicateAnswerResult(
            request,
            predicate_result.status,
            predicate_result.answer_surface,
            predicate_result,
            None,
        )
    implicit_result = (
        run_implicit_question_catalog_answer(implicit_bundle, request)
        if candidate_constructions is None
        else run_implicit_question_candidate_answer(
            implicit_bundle,
            request,
            candidate_constructions,
            state_sha256=state_sha256,
        )
    )
    return RawQuestionImplicitPredicateAnswerResult(
        request,
        implicit_result.status,
        implicit_result.answer_surface,
        predicate_result,
        implicit_result,
    )


__all__ = [
    "IMPLICIT_QUESTION_ANSWER_SHA256S",
    "IMPLICIT_QUESTION_BUNDLE_SHA256",
    "IMPLICIT_QUESTION_EXPRESSION_BOUNDARY",
    "IMPLICIT_QUESTION_INTERPRETATION_STATUSES",
    "IMPLICIT_QUESTION_REASON_SAMPLE_SHA256",
    "IMPLICIT_QUESTION_RESULT_SAMPLE_SHA256",
    "ImplicitQuestionInterpretationKey",
    "RawQuestionImplicitPredicateAnswerResult",
    "W03W04W05ImplicitQuestionBundle",
    "W03W04W05ImplicitQuestionError",
    "build_implicit_question_bundle",
    "continue_implicit_predicate_question_candidate_answer",
    "continue_implicit_predicate_question_answer",
    "resolve_implicit_question_interpretations",
    "run_implicit_question_catalog_answer",
    "run_implicit_question_candidate_answer",
    "run_implicit_predicate_question_answer",
]
