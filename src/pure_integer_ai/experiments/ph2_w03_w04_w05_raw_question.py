"""从公开问题构造恢复 target Role，并原样进入 FT09。"""
from __future__ import annotations

import hashlib
from pathlib import Path

from pure_integer_ai.experiments.ph2_dataset_contract import (
    DatasetContractError,
    canonical_json_line,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_evaluation_public_source import (
    EvaluationPublicBatch,
)
from pure_integer_ai.experiments.ph2_w03_v2_public_source import (
    W03V2PublicEvaluationBatch,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_answer import (
    run_w03_w04_w05_question_answer,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_answer_contract import (
    W03W04W05QuestionRequest,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_contract import (
    RAW_QUESTION_SEGMENT_KINDS,
    RawQuestionAnswerResult,
    RawQuestionConstruction,
    RawQuestionConstructionSegment,
    RawQuestionPattern,
    RawQuestionPatternSegment,
    RawQuestionRequest,
    W03W04W05RawQuestionError,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_vertical_contract import (
    W03W04W05VerticalResult,
)
from pure_integer_ai.experiments.ph2_w04_v2_public_source import (
    W04V2PublicEvaluationBatch,
)
from pure_integer_ai.experiments.ph2_w05_v2_public_source import (
    W05V2PublicEvaluationBatch,
)


RAW_QUESTION_SAMPLE_SHA256 = (
    "f8acc8a2eafe0e79e4ceb78030ad724b1ec62b7fbbb2db0d8cad039077eb05c6")
RAW_QUESTION_PATTERN_SHA256 = (
    "e37e82b201617a9fdba618715c3e5bdc1e95a0bdaa4f1566b6176f156aafd9ef")
RAW_QUESTION_CONSTRUCTION_SHA256 = (
    "021750219a9afe6fd309efa9d9ac03578398304bdd12d7461052f9833325da3c")
RAW_QUESTION_REQUEST_SHA256 = (
    "390b42a35a2c6da82212003cf86ee13ca42850ad0ec585d0e8220522ae49e79a")
RAW_QUESTION_ANSWER_RESULT_SHA256 = (
    "8c2c0b0291005e81de47e2bfa51ee63e30d2add9ea69b3229154aae8c3e69e52")
_SOURCE_KEY = "AUTHORED_CC0_VERTICAL_QUESTION_V1"
_LICENSE_ID = "CC0-1.0"
_SEED_FIELDS = {
    "construction_id", "exemplar_vertical_sha256", "language", "license_id",
    "logical_order", "question_surface", "redistribution_policy", "segments",
    "source_key",
}
_SEGMENT_FIELDS = {"end", "kind", "ordinal", "role_ordinal", "start"}


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _text(value: object, *, where: str) -> str:
    if (not isinstance(value, str) or not value
            or value.strip() != value):
        raise W03W04W05RawQuestionError(
            f"{where} is not canonical text")
    return value


def _integer(value: object, *, where: str, positive: bool = False) -> int:
    if (type(value) is not int or value < int(positive)
            or (positive and value == 0)):
        raise W03W04W05RawQuestionError(f"{where} is not a strict integer")
    return value


def _candidate(vertical: W03W04W05VerticalResult):
    if vertical.status != "BRIDGED" or vertical.link is None:
        raise W03W04W05RawQuestionError(
            "raw question construction requires a bridged result")
    matches = tuple(
        item for item in vertical.w04_w05.w05_result.candidates
        if item.proposition_key == vertical.link.proposition_key
        and item.active == 1
        and item.reasoning_status == "AUTHORIZED")
    if len(matches) != 1:
        raise W03W04W05RawQuestionError(
            "raw question construction Proposition is not unique")
    return matches[0]


def instantiate_raw_question_construction(
        pattern: RawQuestionPattern,
        vertical: W03W04W05VerticalResult,
        ) -> RawQuestionConstruction:
    """把通用段计划应用到一个来源绑定已学 Proposition。"""
    if (not isinstance(pattern, RawQuestionPattern)
            or not isinstance(vertical, W03W04W05VerticalResult)):
        raise TypeError("raw question construction inputs are invalid")
    if (vertical.link is None
            or (vertical.link.primitive_registry, vertical.link.primitive_kind)
            != (pattern.primitive_registry, pattern.primitive_kind)):
        raise W03W04W05RawQuestionError(
            "raw question pattern does not apply to this primitive")
    candidate = _candidate(vertical)
    bindings = candidate.role_bindings
    role_count = sum(
        item.role_ordinal is not None for item in pattern.segments)
    if len(bindings) != role_count:
        raise W03W04W05RawQuestionError(
            "raw question pattern role count does not apply")
    predicate = tuple(
        item for item in candidate.occurrences
        if item.identity_key == vertical.link.predicate_occurrence_key)
    if len(predicate) != 1:
        raise W03W04W05RawQuestionError(
            "raw question predicate occurrence is unavailable")
    segments = []
    target_role = None
    for item in pattern.segments:
        if item.kind == "VARIABLE":
            binding = bindings[item.role_ordinal]
            target_role = binding.role_key
            segment = RawQuestionConstructionSegment(
                item.kind, item.literal_surface, binding.role_key, None)
        elif item.kind == "BOUNDARY":
            segment = RawQuestionConstructionSegment(
                item.kind, item.literal_surface, None, None)
        elif item.kind == "PREDICATE":
            segment = RawQuestionConstructionSegment(
                item.kind,
                predicate[0].surface_fragment,
                None,
                predicate[0].identity_key,
            )
        else:
            binding = bindings[item.role_ordinal]
            occurrences = tuple(
                occurrence for occurrence in candidate.occurrences
                if occurrence.semantic_object_key == binding.filler_key)
            if len(occurrences) != 1:
                raise W03W04W05RawQuestionError(
                    "raw question role filler occurrence is unavailable")
            segment = RawQuestionConstructionSegment(
                item.kind,
                occurrences[0].surface_fragment,
                binding.role_key,
                occurrences[0].identity_key,
            )
        segments.append(segment)
    if target_role is None:
        raise W03W04W05RawQuestionError(
            "raw question pattern lacks a target role")
    values = tuple(segments)
    return RawQuestionConstruction(
        pattern,
        vertical,
        values,
        "".join(item.surface for item in values),
        target_role,
    )


def compile_raw_question_pattern(
        sample_path: str | Path,
        exemplar: W03W04W05VerticalResult,
        *,
        expected_sample_sha256: str = RAW_QUESTION_SAMPLE_SHA256,
        ) -> RawQuestionPattern:
    """从无答案 CC0 样本学习段计划，并用已学命题验证一次。"""
    if not isinstance(exemplar, W03W04W05VerticalResult):
        raise TypeError("raw question exemplar type is invalid")
    try:
        payload = Path(sample_path).read_bytes()
    except OSError as error:
        raise W03W04W05RawQuestionError(
            "raw question sample cannot be read") from error
    if (not payload or not payload.endswith(b"\n")
            or payload.endswith(b"\n\n")
            or len(payload.splitlines()) != 1):
        raise W03W04W05RawQuestionError(
            "raw question sample must be one canonical JSON line")
    sample_sha = _sha_bytes(payload)
    if sample_sha != expected_sample_sha256:
        raise W03W04W05RawQuestionError(
            "raw question sample SHA drifted")
    try:
        value = parse_canonical_json_bytes(
            payload[:-1], require_object=True)
    except DatasetContractError as error:
        raise W03W04W05RawQuestionError(
            "raw question sample is not canonical JSON") from error
    if not isinstance(value, dict) or set(value) != _SEED_FIELDS:
        raise W03W04W05RawQuestionError(
            "raw question sample fields drifted")
    if canonical_json_line(value) != payload:
        raise W03W04W05RawQuestionError(
            "raw question sample bytes are not canonical")
    if (value["source_key"] != _SOURCE_KEY
            or value["license_id"] != _LICENSE_ID
            or value["redistribution_policy"] != "PUBLIC"):
        raise W03W04W05RawQuestionError(
            "raw question sample is not public CC0 data")
    question_surface = _text(
        value["question_surface"], where="raw question exemplar surface")
    exemplar_sha = exemplar.sha256()
    if (value["exemplar_vertical_sha256"] != exemplar_sha
            or value["language"] != exemplar.query.language):
        raise W03W04W05RawQuestionError(
            "raw question exemplar does not bind the vertical query")
    raw_segments = value["segments"]
    if not isinstance(raw_segments, list) or not raw_segments:
        raise W03W04W05RawQuestionError(
            "raw question exemplar segments are empty")
    patterns = []
    expected_start = 0
    roles = []
    for expected_ordinal, raw in enumerate(raw_segments):
        if not isinstance(raw, dict) or set(raw) != _SEGMENT_FIELDS:
            raise W03W04W05RawQuestionError(
                "raw question exemplar segment fields drifted")
        ordinal = _integer(raw["ordinal"], where="question segment ordinal")
        start = _integer(raw["start"], where="question segment start")
        end = _integer(raw["end"], where="question segment end")
        kind = _text(raw["kind"], where="question segment kind")
        if (ordinal != expected_ordinal or start != expected_start
                or end <= start or end > len(question_surface)
                or kind not in RAW_QUESTION_SEGMENT_KINDS):
            raise W03W04W05RawQuestionError(
                "raw question exemplar segment order or span drifted")
        role_ordinal = raw["role_ordinal"]
        if role_ordinal is not None:
            role_ordinal = _integer(
                role_ordinal,
                where="question segment role ordinal",
            )
            roles.append(role_ordinal)
        surface = question_surface[start:end]
        literal = surface if kind in {"VARIABLE", "BOUNDARY"} else None
        patterns.append(RawQuestionPatternSegment(
            kind, role_ordinal, literal))
        expected_start = end
    if expected_start != len(question_surface):
        raise W03W04W05RawQuestionError(
            "raw question exemplar segments do not cover the surface")
    if len(set(roles)) != len(roles):
        raise W03W04W05RawQuestionError(
            "raw question exemplar repeats role ordinals")
    if exemplar.link is None:
        raise W03W04W05RawQuestionError(
            "raw question exemplar lacks a vertical link")
    pattern = RawQuestionPattern(
        _text(value["construction_id"], where="question construction id"),
        value["source_key"],
        value["license_id"],
        sample_sha,
        _integer(
            value["logical_order"],
            where="question pattern logical order",
            positive=True,
        ),
        value["language"],
        exemplar.link.primitive_registry,
        exemplar.link.primitive_kind,
        tuple(patterns),
        exemplar_sha,
    )
    construction = instantiate_raw_question_construction(pattern, exemplar)
    if construction.question_surface != question_surface:
        raise W03W04W05RawQuestionError(
            "raw question exemplar is not reproduced by learned structure")
    return pattern


def build_raw_question_catalog(
        patterns: tuple[RawQuestionPattern, ...],
        vertical_results: tuple[W03W04W05VerticalResult, ...],
        ) -> tuple[RawQuestionConstruction, ...]:
    """把问题模式应用到兼容已学命题并形成确定性查询目录。"""
    if (not isinstance(patterns, tuple) or not patterns
            or any(not isinstance(item, RawQuestionPattern)
                   for item in patterns)
            or not isinstance(vertical_results, tuple)
            or not vertical_results
            or any(not isinstance(item, W03W04W05VerticalResult)
                   for item in vertical_results)):
        raise TypeError("raw question catalog inputs are invalid")
    values = []
    for pattern in patterns:
        for vertical in vertical_results:
            if (vertical.link is not None
                    and (vertical.link.primitive_registry,
                         vertical.link.primitive_kind)
                    == (pattern.primitive_registry, pattern.primitive_kind)):
                values.append(instantiate_raw_question_construction(
                    pattern, vertical))
    ordered = tuple(sorted(values, key=lambda item: item.sha256()))
    identities = tuple(item.sha256() for item in ordered)
    if not ordered or len(set(identities)) != len(identities):
        raise W03W04W05RawQuestionError(
            "raw question catalog is empty or duplicated")
    return ordered


def run_raw_question_answer(
        catalog: tuple[RawQuestionConstruction, ...],
        w03_batch: W03V2PublicEvaluationBatch,
        w04_batch: W04V2PublicEvaluationBatch,
        w05_batch: W05V2PublicEvaluationBatch,
        request: RawQuestionRequest,
        *,
        overlay_validation_sha256: str,
        ) -> RawQuestionAnswerResult:
    """匹配已学问题构造；唯一时只调用 FT09，绝不复制回答逻辑。"""
    if (not isinstance(catalog, tuple) or not catalog
            or any(not isinstance(item, RawQuestionConstruction)
                   for item in catalog)
            or not isinstance(w03_batch, EvaluationPublicBatch)
            or not isinstance(w04_batch, EvaluationPublicBatch)
            or not isinstance(w05_batch, EvaluationPublicBatch)
            or not isinstance(request, RawQuestionRequest)):
        raise TypeError("raw question run inputs are invalid")
    identities = tuple(item.sha256() for item in catalog)
    if identities != tuple(sorted(identities)) or len(set(identities)) != len(
            identities):
        raise W03W04W05RawQuestionError(
            "raw question catalog is not canonical")
    matches = tuple(
        item for item in catalog
        if item.question_surface == request.question_surface
        and (request.source_record_key is None
             or item.source_record_key == request.source_record_key)
    )
    matching_sha = tuple(item.sha256() for item in matches)
    if not matches:
        return RawQuestionAnswerResult(
            request, "UNKNOWN", None, (), None, None)
    if len(matches) > 1:
        return RawQuestionAnswerResult(
            request,
            "CLARIFY",
            None,
            tuple(sorted(matching_sha)),
            None,
            None,
        )
    selected = matches[0]
    typed = run_w03_w04_w05_question_answer(
        w03_batch,
        w04_batch,
        w05_batch,
        W03W04W05QuestionRequest(
            request.question_surface,
            selected.vertical_result.query,
            (selected.target_role_key,),
            request.source_record_key,
        ),
        overlay_validation_sha256=overlay_validation_sha256,
    )
    return RawQuestionAnswerResult(
        request,
        typed.status,
        typed.answer_surface,
        (selected.sha256(),),
        selected,
        typed,
    )


__all__ = [
    "RAW_QUESTION_ANSWER_RESULT_SHA256",
    "RAW_QUESTION_CONSTRUCTION_SHA256",
    "RAW_QUESTION_PATTERN_SHA256",
    "RAW_QUESTION_REQUEST_SHA256",
    "RAW_QUESTION_SAMPLE_SHA256",
    "build_raw_question_catalog",
    "compile_raw_question_pattern",
    "instantiate_raw_question_construction",
    "run_raw_question_answer",
]
