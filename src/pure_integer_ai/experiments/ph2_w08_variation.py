"""W-08 中文表层变体的薄 typed adapter 与 assessment 合同。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import unicodedata
from typing import Any, Iterable

from pure_integer_ai.cognition.shared.identity import (
    ObjectIdentity,
    SourceRef,
    language_atom_identity,
    language_branch_identity,
    representation_identity,
    sense_identity,
    span_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.semantic_object import proposition_identity
from pure_integer_ai.crosscut.integer.unicode_codec import decode, encode
from pure_integer_ai.experiments.ph2_dataset_contract import (
    ObservationRecord,
    StableRecordKey,
    TeacherEvidenceRecord,
)
from pure_integer_ai.experiments.ph2_w05_contract import digest_value
from pure_integer_ai.experiments.ph2_w08_authority import W08_DIMENSION_KEYS
from pure_integer_ai.experiments.ph2_w08_contract import (
    W08_CONSUMER_KEYS,
    W08_STOP_STATES,
)
from pure_integer_ai.experiments.ph2_w08_payload import W08TrainingPayload
from pure_integer_ai.experiments.ph2_w08_registry import (
    W08RegistryError,
    audit_w08_registry_payload,
)


W08_VARIATION_NAMESPACE = 80802
W08_REFERENCE_MODES = (
    "ELLIPSIS",
    "NOUN_PHRASE",
    "PRONOUN",
    "ZERO_REFERENCE",
)
W08_VARIATION_FAMILIES = (
    "SCRIPT_ORTHOGRAPHY",
    "PUNCTUATION_WHITESPACE_WIDTH",
    "WORD_ORDER",
    "PARAPHRASE_DESCRIPTION_QUESTION",
    "ELLIPSIS_ZERO_REFERENCE",
    "PRONOUN_NOUN_PHRASE",
    "POLYSEMY",
    "MULTIPLE_PARSE",
    "SOURCE_GROUNDED_SURFACE",
)
_SURFACE_FIELDS = {
    "DiscourseRevisionQuery": ("surface",),
    "DiscourseInformationCandidateV1": ("observed_surface", "text"),
    "OpenSetClarificationCandidateV1": ("observed_surface", "text"),
    "AttributionQuotationCandidateV1": ("observed_surface", "text"),
    "RAW_SOURCE_OBSERVATION_V1": ("raw_observation", "text"),
}
_EVIDENCE_KIND_BY_PAYLOAD = {
    "DiscourseRevisionQuery": "DISCOURSE_REVISION_LABEL",
    "DiscourseInformationCandidateV1": "DISCOURSE_INFORMATION_LABEL",
    "OpenSetClarificationCandidateV1": "OPEN_SET_CLARIFICATION_LABEL",
    "AttributionQuotationCandidateV1": "ATTRIBUTION_QUOTATION_LABEL",
    "RAW_SOURCE_OBSERVATION_V1": "SOURCE_PARSER_RECEIPT_V1",
}
_OPERATION_FIELDS = frozenset({
    "candidate_kind",
    "direction",
    "isolation_axis",
    "query_kind",
    "transfer_kind",
    "variant_kind",
})
_FORBIDDEN_SCHEMA_FIELDS = frozenset({
    "evaluator_label",
    "expected",
    "expected_payload",
    "expected_state",
    "label",
})


class W08VariationError(ValueError):
    """W-08 变体输入、Evidence、identity 或消融证据无法闭合。"""


def _strict_key(value: object, *, where: str) -> tuple[int, ...]:
    if (
        not isinstance(value, tuple)
        or not value
        or any(type(item) is not int for item in value)
    ):
        raise W08VariationError(f"{where} must be a non-empty strict integer tuple")
    return value


def _domain_key(domain: str, value: tuple[int, ...]) -> tuple[int, ...]:
    return digest_value({"domain": domain, "value": list(_strict_key(value, where=domain))})


def _scalar_codepoints(text: str) -> tuple[int, ...]:
    if not isinstance(text, str):
        raise W08VariationError("surface must be text")
    codepoints = encode(text)
    if any(0xD800 <= item <= 0xDFFF for item in codepoints):
        raise W08VariationError("surface contains a surrogate code point")
    return codepoints


def _normalized_surface(text: str) -> str:
    compatible = unicodedata.normalize("NFKC", text)
    return "".join(" " if character.isspace() else character for character in compatible)


def _unicode_categories(text: str) -> tuple[str, ...]:
    return tuple(unicodedata.category(character) for character in text)


def _parser_candidates(codepoints: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """按 Unicode 类别边界形成候选，不把任何中文词或句式写入规则。"""
    if not codepoints:
        return (digest_value({"parser": "EMPTY", "span": []}),)
    text = decode(codepoints)
    runs: list[tuple[int, ...]] = []
    current: list[int] = []
    current_family = ""
    for character, codepoint in zip(text, codepoints, strict=True):
        category = unicodedata.category(character)
        family = category[:1]
        if current and (family != current_family or family in {"P", "Z"}):
            runs.append(tuple(current))
            current = []
        current.append(codepoint)
        current_family = family
        if family in {"P", "Z"}:
            runs.append(tuple(current))
            current = []
            current_family = ""
    if current:
        runs.append(tuple(current))
    whole = digest_value({"parser": "WHOLE", "span": list(codepoints)})
    run_key = digest_value({"parser": "UNICODE_CATEGORY_RUNS", "runs": [list(item) for item in runs]})
    return tuple(dict.fromkeys((whole, run_key)))


@dataclass(frozen=True)
class W08SurfaceReceipt:
    """逐码点 raw Observation 和只追加的规范化/parser receipts。"""

    raw_codepoints: tuple[int, ...]
    normalized_codepoints: tuple[int, ...]
    unicode_categories: tuple[str, ...]
    normalization_receipt_key: tuple[int, ...]
    parser_candidate_keys: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if any(type(item) is not int for item in self.raw_codepoints):
            raise W08VariationError("raw surface is not a strict integer tuple")
        if any(type(item) is not int for item in self.normalized_codepoints):
            raise W08VariationError("normalized surface is not a strict integer tuple")
        _strict_key(self.normalization_receipt_key, where="normalization_receipt_key")
        if (
            len(self.unicode_categories) != len(self.normalized_codepoints)
            or not self.parser_candidate_keys
        ):
            raise W08VariationError("surface receipts are incomplete")
        for key in self.parser_candidate_keys:
            _strict_key(key, where="parser_candidate_key")

    def recover_raw(self) -> str:
        return decode(self.raw_codepoints)


def make_w08_surface_receipt(surface: str) -> W08SurfaceReceipt:
    """在 Unicode I/O 边缘保留 raw，并追加兼容规范化和 parser 候选。"""
    raw = _scalar_codepoints(surface)
    normalized_text = _normalized_surface(surface)
    normalized = _scalar_codepoints(normalized_text)
    receipt = digest_value({
        "kind": "W08_UNICODE_NORMALIZATION_RECEIPT_V1",
        "normalization": "NFKC_AND_UNICODE_WHITESPACE_CLASSIFICATION",
        "raw_sha256": hashlib.sha256(surface.encode("utf-8")).hexdigest(),
        "normalized_codepoints": list(normalized),
    })
    return W08SurfaceReceipt(
        raw,
        normalized,
        _unicode_categories(normalized_text),
        receipt,
        _parser_candidates(normalized),
    )


@dataclass(frozen=True)
class W08VariationKeys:
    """表层族、内容、结构、来源、文档页和完整组合的互不折叠键。"""

    surface_family_key: tuple[int, ...]
    content_key: tuple[int, ...]
    structure_key: tuple[int, ...]
    source_key: tuple[int, ...]
    document_key: tuple[int, ...]
    combination_key: tuple[int, ...]

    def __post_init__(self) -> None:
        fields = (
            self.surface_family_key,
            self.content_key,
            self.structure_key,
            self.source_key,
            self.document_key,
            self.combination_key,
        )
        for index, item in enumerate(fields):
            _strict_key(item, where=f"variation_key[{index}]")
        if len(set(fields)) != len(fields):
            raise W08VariationError("variation axes collapsed into one key")


def make_w08_variation_keys(
    *,
    surface_family_key: tuple[int, ...],
    content_key: tuple[int, ...],
    structure_key: tuple[int, ...],
    source_key: tuple[int, ...],
    document_key: tuple[int, ...],
) -> W08VariationKeys:
    axes = {
        "surface_family": _domain_key("surface_family", surface_family_key),
        "content": _domain_key("content", content_key),
        "structure": _domain_key("structure", structure_key),
        "source": _domain_key("source", source_key),
        "document": _domain_key("document", document_key),
    }
    combination = digest_value({
        "domain": "full_combination",
        "axes": {name: list(key) for name, key in axes.items()},
    })
    return W08VariationKeys(
        axes["surface_family"],
        axes["content"],
        axes["structure"],
        axes["source"],
        axes["document"],
        combination,
    )


@dataclass(frozen=True)
class W08VariationBundle:
    """U/R/G 共用的 LanguageAtom/Span/Sense/Structure/Proposition bundle。"""

    representation: ObjectIdentity
    language_atom: ObjectIdentity
    span: ObjectIdentity
    senses: tuple[ObjectIdentity, ...]
    structures: tuple[ObjectIdentity, ...]
    propositions: tuple[ObjectIdentity, ...]

    def __post_init__(self) -> None:
        if not self.senses or not self.structures or not self.propositions:
            raise W08VariationError("typed semantic candidates must remain non-empty")
        identities = (
            self.representation,
            self.language_atom,
            self.span,
            *self.senses,
            *self.structures,
            *self.propositions,
        )
        if any(not isinstance(item, ObjectIdentity) for item in identities):
            raise W08VariationError("variation bundle contains an untyped identity")

    def candidate_keys(self) -> tuple[tuple[int, ...], ...]:
        return tuple(
            digest_value({
                "sense": list(sense.stable_key()),
                "structure": list(structure.stable_key()),
                "proposition": list(proposition.stable_key()),
            })
            for sense in self.senses
            for structure in self.structures
            for proposition in self.propositions
        )


@dataclass(frozen=True)
class W08VariationIntake:
    receipt: W08SurfaceReceipt
    keys: W08VariationKeys
    bundle: W08VariationBundle
    reference_mode: str

    def __post_init__(self) -> None:
        if self.reference_mode not in W08_REFERENCE_MODES:
            raise W08VariationError("reference mode must be typed, not inferred from a cue")


def make_w08_variation_intake(
    *,
    surface: str,
    source: SourceRef,
    surface_family_key: tuple[int, ...],
    content_key: tuple[int, ...],
    structure_keys: tuple[tuple[int, ...], ...],
    sense_keys: tuple[tuple[int, ...], ...],
    proposition_keys: tuple[tuple[int, ...], ...],
    source_key: tuple[int, ...],
    document_key: tuple[int, ...],
    reference_mode: str,
) -> W08VariationIntake:
    """用注入的 typed keys 构造身份；surface 只定义 Representation/Span。"""
    if not isinstance(source, SourceRef):
        raise W08VariationError("variation intake requires SourceRef")
    receipt = make_w08_surface_receipt(surface)
    if not receipt.raw_codepoints:
        raise W08VariationError("variation surface must not be empty")
    keys = make_w08_variation_keys(
        surface_family_key=surface_family_key,
        content_key=content_key,
        structure_key=digest_value({"members": [list(item) for item in structure_keys]}),
        source_key=source_key,
        document_key=document_key,
    )
    branch = language_branch_identity((W08_VARIATION_NAMESPACE, 1))
    bundle = W08VariationBundle(
        representation_identity(keys.surface_family_key, receipt.raw_codepoints),
        language_atom_identity(branch, _domain_key("language_atom_content", content_key)),
        span_identity(source, members=((0, len(receipt.raw_codepoints)),)),
        tuple(sense_identity(source, sense_key=_strict_key(item, where="sense_key"))
              for item in sense_keys),
        tuple(structure_concept_identity(_strict_key(item, where="structure_key"))
              for item in structure_keys),
        tuple(proposition_identity(source, _strict_key(item, where="proposition_key"))
              for item in proposition_keys),
    )
    return W08VariationIntake(receipt, keys, bundle, reference_mode)


@dataclass(frozen=True)
class W08VariationUse:
    consumer_key: str
    request_key: tuple[int, ...]
    selected_candidate_key: tuple[int, ...] | None
    evidence_keys: tuple[tuple[int, ...], ...]
    directional_choice_key: tuple[int, ...]
    outcome_state: str
    outcome_key: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.consumer_key not in W08_CONSUMER_KEYS:
            raise W08VariationError("variation consumer is not registered")
        if self.outcome_state not in W08_STOP_STATES:
            raise W08VariationError("variation outcome is not registered")
        _strict_key(self.request_key, where="request_key")
        _strict_key(self.directional_choice_key, where="directional_choice_key")
        _strict_key(self.outcome_key, where="outcome_key")
        for key in self.evidence_keys:
            _strict_key(key, where="evidence_key")
        if self.selected_candidate_key is not None:
            _strict_key(self.selected_candidate_key, where="selected_candidate_key")


@dataclass(frozen=True)
class W08VariationResult:
    intake: W08VariationIntake
    candidate_keys: tuple[tuple[int, ...], ...]
    selected_candidate_key: tuple[int, ...] | None
    uses: tuple[W08VariationUse, ...]

    def __post_init__(self) -> None:
        if tuple(item.consumer_key for item in self.uses) != W08_CONSUMER_KEYS:
            raise W08VariationError("U/R/G uses are incomplete or reordered")
        if len({item.directional_choice_key for item in self.uses}) != 3:
            raise W08VariationError("U/R/G directional choices were collapsed")


def resolve_w08_variation(
    intake: W08VariationIntake,
    *,
    evidence_keys: tuple[tuple[int, ...], ...],
    clarification_candidate_key: tuple[int, ...] | None = None,
) -> W08VariationResult:
    """只允许 singleton 或显式 clarification 采用，其他多候选返回 CLARIFY。"""
    if not isinstance(intake, W08VariationIntake):
        raise W08VariationError("variation resolver requires typed intake")
    candidates = intake.bundle.candidate_keys()
    for key in evidence_keys:
        _strict_key(key, where="evidence_key")
    selected: tuple[int, ...] | None
    if clarification_candidate_key is not None:
        selected = _strict_key(clarification_candidate_key, where="clarification_candidate_key")
        if selected not in candidates:
            raise W08VariationError("clarification selected an unknown candidate")
    elif len(candidates) == 1:
        selected = candidates[0]
    else:
        selected = None
    outcome = "RESOLVED" if selected is not None else "CLARIFY"
    uses = []
    for consumer in W08_CONSUMER_KEYS:
        request = digest_value({
            "consumer": consumer,
            "combination": list(intake.keys.combination_key),
        })
        direction = digest_value({
            "consumer": consumer,
            "request": list(request),
            "selected": None if selected is None else list(selected),
        })
        outcome_key = digest_value({
            "consumer": consumer,
            "directional_choice": list(direction),
            "outcome": outcome,
        })
        uses.append(W08VariationUse(
            consumer,
            request,
            selected,
            evidence_keys,
            direction,
            outcome,
            outcome_key,
        ))
    return W08VariationResult(intake, candidates, selected, tuple(uses))


def _schema_shape(value: Any) -> Any:
    if isinstance(value, dict):
        lowered = {str(key).lower() for key in value}
        if lowered & _FORBIDDEN_SCHEMA_FIELDS:
            raise W08VariationError("expected/label field entered variation schema")
        return {str(key): _schema_shape(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        shapes = {_shape_token(_schema_shape(item)) for item in value}
        return [item for item in sorted(shapes)]
    if value is None:
        return "NULL"
    if type(value) is bool:
        return "BOOL"
    if type(value) is int:
        return "INT"
    if isinstance(value, str):
        return "TEXT"
    raise W08VariationError("typed payload contains an unsupported schema value")


def _shape_token(value: Any) -> str:
    return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()


def _typed_operations(value: Any, *, path: tuple[str, ...] = ()) -> set[str]:
    operations: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            next_path = (*path, key_text)
            if key_text in _OPERATION_FIELDS and isinstance(item, str) and item:
                operations.add(".".join(next_path) + "=" + item)
            operations.update(_typed_operations(item, path=next_path))
    elif isinstance(value, list):
        for item in value:
            operations.update(_typed_operations(item, path=path))
    return operations


def _record_key(value: StableRecordKey) -> tuple[int, ...]:
    return value.components


@dataclass(frozen=True)
class W08VariationLearning:
    """只保存 schema、typed operation 和 Evidence identity，不保存答案/surface。"""

    payload_kinds: tuple[str, ...]
    schema_fingerprints: tuple[tuple[str, tuple[int, ...]], ...]
    allowed_typed_operations: tuple[str, ...]
    evidence_bindings: tuple[
        tuple[tuple[int, ...], tuple[int, ...], str], ...
    ]
    source_parser_receipt_count: int

    def __post_init__(self) -> None:
        if tuple(sorted(self.payload_kinds)) != self.payload_kinds:
            raise W08VariationError("variation payload kinds are not canonical")
        if len(self.evidence_bindings) != 63:
            raise W08VariationError("variation Evidence identity coverage is incomplete")
        if self.source_parser_receipt_count != 4:
            raise W08VariationError("source parser receipt coverage is incomplete")


def _validate_source_parser_receipt(
    observation: ObservationRecord,
    evidence: TeacherEvidenceRecord,
) -> None:
    value = observation.typed_payload.to_value()
    receipt = evidence.typed_evidence.to_value()
    if set(receipt) != {
        "definitive_truth_authoritative",
        "parser_version",
        "raw_observation_sha256",
        "source_ref_key",
    }:
        raise W08VariationError("source parser receipt fields drifted")
    if (
        receipt["definitive_truth_authoritative"] != 0
        or receipt["raw_observation_sha256"] != value.get("raw_observation_sha256")
        or receipt["source_ref_key"] != observation.source_ref_key.to_list()
        or type(receipt["parser_version"]) is not int
        or receipt["parser_version"] <= 0
    ):
        raise W08VariationError("source parser receipt identity/hash mismatch")


def learn_w08_variation(payload: W08TrainingPayload) -> W08VariationLearning:
    """从六个 train pack 学 schema/Evidence binding，不读取 authored 答案。"""
    try:
        audit_w08_registry_payload(payload)
    except W08RegistryError as error:
        raise W08VariationError("variation payload registry audit failed") from error
    evidence_by_observation = {
        evidence.observation_key: evidence for evidence in payload.teacher_evidence
    }
    schema_variants: dict[str, set[tuple[int, ...]]] = {}
    operations: set[str] = set()
    bindings = []
    source_receipts = 0
    for observation in payload.observations:
        value = observation.typed_payload.to_value()
        shape = _schema_shape(value)
        fingerprint = digest_value({
            "payload_kind": observation.payload_kind,
            "shape": shape,
        })
        schema_variants.setdefault(observation.payload_kind, set()).add(fingerprint)
        operations.update(_typed_operations(value))
        evidence = evidence_by_observation.get(observation.stable_key)
        if evidence is None:
            raise W08VariationError("variation Observation has no bound Evidence")
        expected_kind = _EVIDENCE_KIND_BY_PAYLOAD.get(observation.payload_kind)
        if evidence.evidence_kind != expected_kind:
            raise W08VariationError("variation Evidence kind mismatch")
        if expected_kind == "SOURCE_PARSER_RECEIPT_V1":
            _validate_source_parser_receipt(observation, evidence)
            source_receipts += 1
        bindings.append((
            _record_key(observation.stable_key),
            _record_key(evidence.stable_key),
            evidence.evidence_kind,
        ))
    schemas = {
        kind: digest_value({
            "payload_kind": kind,
            "schema_variants": [list(item) for item in sorted(variants)],
        })
        for kind, variants in schema_variants.items()
    }
    return W08VariationLearning(
        tuple(sorted(schemas)),
        tuple(sorted(schemas.items())),
        tuple(sorted(operations)),
        tuple(sorted(bindings)),
        source_receipts,
    )


def surface_from_w08_observation(observation: ObservationRecord) -> str:
    """按 payload kind 的公开 schema 取 raw surface，不查看 Evidence。"""
    if not isinstance(observation, ObservationRecord):
        raise W08VariationError("surface adapter requires ObservationRecord")
    path = _SURFACE_FIELDS.get(observation.payload_kind)
    if path is None:
        raise W08VariationError("surface payload kind is not registered")
    value: Any = observation.typed_payload.to_value()
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise W08VariationError("surface field is missing")
        value = value[key]
    if not isinstance(value, str):
        raise W08VariationError("surface field is not text")
    return value


@dataclass(frozen=True)
class W08VariationAblationReport:
    affected_dimensions: tuple[str, ...]
    unaffected_dimensions: tuple[str, ...]


def assess_w08_variation_ablation(
    *,
    full_dimension_outcomes: dict[str, str],
    ablated_dimension_outcomes: dict[str, str],
) -> W08VariationAblationReport:
    """证明删除 variation adapter/Evidence 只击穿本维，其他维必须不变。"""
    expected = set(W08_DIMENSION_KEYS)
    if set(full_dimension_outcomes) != expected or set(ablated_dimension_outcomes) != expected:
        raise W08VariationError("ablation dimension inventory drifted")
    variation = "W-08-CHINESE_VARIATION"
    if full_dimension_outcomes[variation] != "PASS":
        raise W08VariationError("full variation assessment did not pass")
    changed = tuple(
        key for key in W08_DIMENSION_KEYS
        if full_dimension_outcomes[key] != ablated_dimension_outcomes[key]
    )
    if changed != (variation,) or ablated_dimension_outcomes[variation] == "PASS":
        raise W08VariationError("variation ablation is not orthogonal")
    return W08VariationAblationReport(
        changed,
        tuple(key for key in W08_DIMENSION_KEYS if key != variation),
    )


def require_w08_variation_family_coverage(families: Iterable[str]) -> tuple[str, ...]:
    """要求 assessment 覆盖全部公开变体族；不做 token/sentence 去重判分。"""
    actual = tuple(dict.fromkeys(families))
    if set(actual) != set(W08_VARIATION_FAMILIES):
        raise W08VariationError("variation family coverage is incomplete")
    return actual


__all__ = [
    "W08_REFERENCE_MODES",
    "W08_VARIATION_FAMILIES",
    "W08SurfaceReceipt",
    "W08VariationAblationReport",
    "W08VariationBundle",
    "W08VariationError",
    "W08VariationIntake",
    "W08VariationKeys",
    "W08VariationLearning",
    "W08VariationResult",
    "W08VariationUse",
    "assess_w08_variation_ablation",
    "learn_w08_variation",
    "make_w08_surface_receipt",
    "make_w08_variation_intake",
    "make_w08_variation_keys",
    "require_w08_variation_family_coverage",
    "resolve_w08_variation",
    "surface_from_w08_observation",
]
