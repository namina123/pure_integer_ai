"""PH2-D03-V2 W-02 Candidate 的纯学习与预测模型。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import unicodedata
from typing import Any

from pure_integer_ai.experiments.ph2_d03_contract_core import canonical_json_bytes
from pure_integer_ai.experiments.ph2_dataset_contract import (
    ObservationRecord,
    TeacherEvidenceRecord,
)


W02_FORMAL_EVIDENCE_KIND = "W02_FORMAL_FOUNDATION_EVIDENCE_V2"
W02_CAPABILITY_CARRIER_RECONSTRUCTION = "CARRIER_RECONSTRUCTION"
W02_CAPABILITY_OOV_BOUNDARY_LATTICE = "OOV_BOUNDARY_LATTICE"
W02_CAPABILITY_UD_MORPHOLOGY = "UD_MORPHOLOGY"
W02_CAPABILITY_UNICODE_ANALYSIS = "UNICODE_ANALYSIS"
W02_EVIDENCE_AUTHORED = "AUTHORED_OOV"
W02_EVIDENCE_UD = "UD_ANNOTATION"
W02_EVIDENCE_UNICODE = "UNICODE_ANNOTATION"


# object-model: exception
class W02CandidateModelError(ValueError):
    """W-02 Candidate 输入、Evidence 或纯模型结果不满足冻结语义。"""


def _hash_value(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _strict_text(value: object, *, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise W02CandidateModelError(f"{where} 必须是非空文本")
    return value


def _strict_int(value: object, *, where: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise W02CandidateModelError(f"{where} 必须是不小于 {minimum} 的严格整数")
    return value


def _strict_sha256(value: object, *, where: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise W02CandidateModelError(f"{where} 必须是小写 SHA-256")
    return value


def _exact(value: object, fields: set[str], *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise W02CandidateModelError(f"{where} 字段不精确")
    return value


def _key(value: object, *, where: str) -> tuple[int, ...]:
    components = getattr(value, "components", value)
    if (not isinstance(components, tuple) or not components
            or any(type(item) is not int or item <= 0 for item in components)):
        raise W02CandidateModelError(f"{where} 不是正整数稳定键")
    return components


def _coarse_char_kind(char: str) -> str:
    category = unicodedata.category(char)
    code_point = ord(char)
    if char in "0123456789abcdef":
        return "HEX_ASCII"
    if "a" <= char <= "z" or "A" <= char <= "Z":
        return "LATIN_ASCII"
    if 0x3400 <= code_point <= 0x9FFF or 0x20000 <= code_point <= 0x3134F:
        return "CJK"
    if category.startswith("M"):
        return "COMBINING"
    return category


def _unit_signature(surface: str) -> str:
    return "/".join(_coarse_char_kind(char) for char in surface)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W02CarrierRule:
    """由 train Observation/Evidence 归纳的一种载体内容模板。"""

    carrier_kind: str
    prefix: str
    suffix: str
    root_node_kind: str
    content_node_kind: str

    def __post_init__(self) -> None:
        _strict_text(self.carrier_kind, where="carrier kind")
        _strict_text(self.root_node_kind, where="carrier root node kind")
        _strict_text(self.content_node_kind, where="carrier content node kind")
        if not isinstance(self.prefix, str) or not isinstance(self.suffix, str):
            raise W02CandidateModelError("carrier prefix/suffix 必须是文本")

    def key(self) -> tuple[str, str, str, str, str]:
        return (
            self.carrier_kind,
            self.prefix,
            self.suffix,
            self.root_node_kind,
            self.content_node_kind,
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "carrier_kind": self.carrier_kind,
            "content_node_kind": self.content_node_kind,
            "prefix": self.prefix,
            "root_node_kind": self.root_node_kind,
            "suffix": self.suffix,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W02UnicodeUnit:
    """一个由 train Evidence 支持的 Unicode 整数单位。"""

    code_point: int
    category: str
    combining_class: int

    def __post_init__(self) -> None:
        if type(self.code_point) is not int or not 0 <= self.code_point <= 0x10FFFF:
            raise W02CandidateModelError("Unicode code point 越界")
        _strict_text(self.category, where="Unicode category")
        _strict_int(self.combining_class, where="Unicode combining class")

    def to_dict(self) -> dict[str, object]:
        return {
            "category": self.category,
            "code_point": self.code_point,
            "combining_class": self.combining_class,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W02LexemeEvidence:
    """一个 UD 词形候选及其来源化形态 Evidence。"""

    form: str
    lemma: str
    upos: str
    feats_json: str

    def __post_init__(self) -> None:
        for name in ("form", "lemma", "upos", "feats_json"):
            if not isinstance(getattr(self, name), str):
                raise W02CandidateModelError(f"lexeme {name} 必须是文本")
        _strict_text(self.form, where="lexeme form")
        _strict_text(self.upos, where="lexeme upos")

    def key(self) -> tuple[str, str, str, str]:
        return self.form, self.lemma, self.upos, self.feats_json

    def to_dict(self) -> dict[str, str]:
        return {
            "feats_json": self.feats_json,
            "form": self.form,
            "lemma": self.lemma,
            "upos": self.upos,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W02OovUnitEvidence:
    """一个未知连续串中的可撤回单位候选。"""

    surface: str
    class_signature: str
    length: int

    def __post_init__(self) -> None:
        _strict_text(self.surface, where="OOV unit surface")
        _strict_text(self.class_signature, where="OOV unit signature")
        if self.class_signature != _unit_signature(self.surface):
            raise W02CandidateModelError("OOV unit signature 漂移")
        if type(self.length) is not int or self.length != len(self.surface):
            raise W02CandidateModelError("OOV unit length 漂移")

    def key(self) -> tuple[str, str, int]:
        return self.surface, self.class_signature, self.length

    def to_dict(self) -> dict[str, object]:
        return {
            "class_signature": self.class_signature,
            "length": self.length,
            "surface": self.surface,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W02LearningDelta:
    """单个 Observation/Evidence 对产生的可交换、可合并学习增量。"""

    observation_key: tuple[int, ...]
    source_ref_key: tuple[int, ...]
    evidence_sha256: str
    delta_sha256: str
    evidence_mode: str
    carrier_rule: W02CarrierRule
    unicode_units: tuple[W02UnicodeUnit, ...]
    lexemes: tuple[W02LexemeEvidence, ...]
    oov_units: tuple[W02OovUnitEvidence, ...]
    capabilities: tuple[str, ...]
    boundary_points: tuple[int, ...]
    use_outcome_sha256: str
    logic_operations: int

    def __post_init__(self) -> None:
        _key(self.observation_key, where="learning observation key")
        _key(self.source_ref_key, where="learning source key")
        _strict_sha256(self.evidence_sha256, where="learning evidence sha")
        _strict_sha256(self.delta_sha256, where="learning delta sha")
        if self.evidence_mode not in {
                W02_EVIDENCE_AUTHORED, W02_EVIDENCE_UD, W02_EVIDENCE_UNICODE}:
            raise W02CandidateModelError("learning evidence mode 未注册")
        if not isinstance(self.carrier_rule, W02CarrierRule):
            raise W02CandidateModelError("learning carrier rule 类型错误")
        if (not isinstance(self.unicode_units, tuple)
                or any(not isinstance(item, W02UnicodeUnit) for item in self.unicode_units)):
            raise W02CandidateModelError("learning unicode units 类型错误")
        if (not isinstance(self.lexemes, tuple)
                or any(not isinstance(item, W02LexemeEvidence) for item in self.lexemes)):
            raise W02CandidateModelError("learning lexemes 类型错误")
        if (not isinstance(self.oov_units, tuple)
                or any(not isinstance(item, W02OovUnitEvidence) for item in self.oov_units)):
            raise W02CandidateModelError("learning OOV units 类型错误")
        if (not isinstance(self.capabilities, tuple) or not self.capabilities
                or tuple(sorted(set(self.capabilities))) != self.capabilities):
            raise W02CandidateModelError("learning capabilities 必须稳定去重排序")
        if (not isinstance(self.boundary_points, tuple) or not self.boundary_points
                or tuple(sorted(set(self.boundary_points))) != self.boundary_points
                or self.boundary_points[0] != 0):
            raise W02CandidateModelError("learning boundary points 非法")
        _strict_sha256(self.use_outcome_sha256, where="learning use outcome sha")
        _strict_int(self.logic_operations, where="learning logic operations", minimum=1)

    def semantic_dict(self) -> dict[str, object]:
        return {
            "boundary_points": list(self.boundary_points),
            "capabilities": list(self.capabilities),
            "carrier_rule": self.carrier_rule.to_dict(),
            "evidence_mode": self.evidence_mode,
            "lexemes": [item.to_dict() for item in self.lexemes],
            "observation_key": list(self.observation_key),
            "oov_units": [item.to_dict() for item in self.oov_units],
            "source_ref_key": list(self.source_ref_key),
            "unicode_units": [item.to_dict() for item in self.unicode_units],
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W02CarrierGeneration:
    """Candidate 对未见 surface 的载体生成结果。"""

    status: str
    carrier_kind: str
    surface: str
    carrier_serialization: str
    content_span_start: int
    content_span_end: int

    def __post_init__(self) -> None:
        if self.status not in {"GENERATED", "AMBIGUOUS", "UNKNOWN"}:
            raise W02CandidateModelError("carrier generation status 未注册")
        _strict_text(self.carrier_kind, where="generation carrier kind")
        if not isinstance(self.surface, str) or not isinstance(self.carrier_serialization, str):
            raise W02CandidateModelError("generation surface/raw 必须是文本")
        _strict_int(self.content_span_start, where="generation span start")
        _strict_int(self.content_span_end, where="generation span end")
        if self.content_span_end < self.content_span_start:
            raise W02CandidateModelError("generation span 逆序")
        if (self.status == "GENERATED"
                and self.carrier_serialization[
                    self.content_span_start:self.content_span_end] != self.surface):
            raise W02CandidateModelError("generated content span 不能恢复 surface")

    def to_dict(self) -> dict[str, object]:
        return {
            "carrier_kind": self.carrier_kind,
            "carrier_serialization": self.carrier_serialization,
            "content_span_end": self.content_span_end,
            "content_span_start": self.content_span_start,
            "status": self.status,
            "surface": self.surface,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W02ObservedCarrier:
    """从无 label Observation 提取的 Candidate 可见载体请求。"""

    observation_key: tuple[int, ...]
    carrier_kind: str
    surface: str
    carrier_serialization: str
    content_span_start: int
    content_span_end: int

    def __post_init__(self) -> None:
        _key(self.observation_key, where="observed carrier observation key")
        _strict_text(self.carrier_kind, where="observed carrier kind")
        if not isinstance(self.surface, str) or not isinstance(self.carrier_serialization, str):
            raise W02CandidateModelError("observed carrier surface/raw 类型错误")
        _strict_int(self.content_span_start, where="observed carrier start")
        _strict_int(self.content_span_end, where="observed carrier end")
        if (self.content_span_end < self.content_span_start
                or self.carrier_serialization[
                    self.content_span_start:self.content_span_end] != self.surface):
            raise W02CandidateModelError("observed carrier span 不能恢复 surface")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W02MorphologyCandidate:
    """Candidate lexicon 对当前新内容提出的来源无关形态候选。"""

    start: int
    end: int
    form: str
    lemma: str
    upos: str
    feats_json: str
    support_count: int

    def __post_init__(self) -> None:
        _strict_int(self.start, where="morphology candidate start")
        _strict_int(self.end, where="morphology candidate end")
        if self.end <= self.start:
            raise W02CandidateModelError("morphology candidate span 非法")
        for name in ("form", "lemma", "upos", "feats_json"):
            if not isinstance(getattr(self, name), str):
                raise W02CandidateModelError(f"morphology candidate {name} 类型错误")
        _strict_text(self.form, where="morphology candidate form")
        _strict_text(self.upos, where="morphology candidate upos")
        _strict_int(self.support_count, where="morphology candidate support", minimum=1)

    def to_dict(self) -> dict[str, object]:
        return {
            "end": self.end,
            "feats_json": self.feats_json,
            "form": self.form,
            "lemma": self.lemma,
            "start": self.start,
            "support_count": self.support_count,
            "upos": self.upos,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W02CandidatePrediction:
    """Candidate 对单个无 label Observation 的只读预测。"""

    observation_key: tuple[int, ...]
    status: str
    generation: W02CarrierGeneration
    boundary_lattice: tuple[int, ...]
    unicode_units: tuple[W02UnicodeUnit, ...]
    morphology_candidates: tuple[W02MorphologyCandidate, ...]
    capabilities: tuple[str, ...]

    def __post_init__(self) -> None:
        _key(self.observation_key, where="prediction observation key")
        if self.status not in {"PREDICTED", "AMBIGUOUS", "UNKNOWN"}:
            raise W02CandidateModelError("prediction status 未注册")
        if not isinstance(self.generation, W02CarrierGeneration):
            raise W02CandidateModelError("prediction generation 类型错误")
        if (not isinstance(self.boundary_lattice, tuple) or not self.boundary_lattice
                or tuple(sorted(set(self.boundary_lattice))) != self.boundary_lattice
                or self.boundary_lattice[0] != 0):
            raise W02CandidateModelError("prediction boundary lattice 非法")
        if (not isinstance(self.unicode_units, tuple)
                or any(not isinstance(item, W02UnicodeUnit) for item in self.unicode_units)):
            raise W02CandidateModelError("prediction unicode units 类型错误")
        if (not isinstance(self.morphology_candidates, tuple)
                or any(not isinstance(item, W02MorphologyCandidate)
                       for item in self.morphology_candidates)):
            raise W02CandidateModelError("prediction morphology candidates 类型错误")
        if tuple(sorted(set(self.capabilities))) != self.capabilities:
            raise W02CandidateModelError("prediction capabilities 未稳定排序")

    def to_dict(self) -> dict[str, object]:
        return {
            "boundary_lattice": list(self.boundary_lattice),
            "capabilities": list(self.capabilities),
            "generation": self.generation.to_dict(),
            "morphology_candidates": [
                item.to_dict() for item in self.morphology_candidates],
            "observation_key": list(self.observation_key),
            "status": self.status,
            "unicode_units": [item.to_dict() for item in self.unicode_units],
        }


def _carrier_rule(observation: ObservationRecord) -> tuple[W02CarrierRule, str, str, int, int]:
    value = observation.typed_payload.to_value()
    raw = _exact(value, {"carrier", "language_payload"}, where="W-02 typed payload")
    language = _exact(raw["language_payload"], {
        "carrier_serialization", "content_span_end", "content_span_start",
        "source_identity", "surface", "surface_sha256",
    }, where="W-02 language payload")
    carrier = _exact(raw["carrier"], {
        "carrier_kind", "edges", "nodes", "raw_text_sha256", "root_node_keys",
    }, where="W-02 carrier")
    serialization = _strict_text(
        language["carrier_serialization"], where="carrier serialization")
    surface = _strict_text(language["surface"], where="carrier surface")
    start = _strict_int(language["content_span_start"], where="carrier span start")
    end = _strict_int(language["content_span_end"], where="carrier span end")
    if end < start or serialization[start:end] != surface:
        raise W02CandidateModelError("carrier content span 不能精确恢复 surface")
    if hashlib.sha256(serialization.encode("utf-8")).hexdigest() != carrier["raw_text_sha256"]:
        raise W02CandidateModelError("carrier raw SHA 漂移")
    if hashlib.sha256(surface.encode("utf-8")).hexdigest() != language["surface_sha256"]:
        raise W02CandidateModelError("carrier surface SHA 漂移")
    nodes = carrier["nodes"]
    if not isinstance(nodes, list) or len(nodes) != 2:
        raise W02CandidateModelError("W-02 carrier 必须含 root/content 两个节点")
    roots = [item for item in nodes if item.get("parent_node_key") is None]
    contents = [item for item in nodes if item.get("attributes", {}).get("language_content") == 1]
    if len(roots) != 1 or len(contents) != 1:
        raise W02CandidateModelError("W-02 carrier root/content 节点不唯一")
    kind = _strict_text(carrier["carrier_kind"], where="carrier kind")
    if kind != observation.representation:
        raise W02CandidateModelError("carrier kind 与 Observation representation 漂移")
    rule = W02CarrierRule(
        kind,
        serialization[:start],
        serialization[end:],
        _strict_text(roots[0].get("node_kind"), where="root node kind"),
        _strict_text(contents[0].get("node_kind"), where="content node kind"),
    )
    return rule, surface, serialization, start, end


def observe_w02_carrier(observation: ObservationRecord) -> W02ObservedCarrier:
    """从 W-02 typed Observation 提取无 label 的载体请求。"""
    if not isinstance(observation, ObservationRecord):
        raise TypeError("W-02 observed carrier 必须来自 ObservationRecord")
    if (observation.w_stage != "W-02" or observation.payload_kind != "typed_carrier"
            or observation.language != "zh"):
        raise W02CandidateModelError("W-02 observed carrier 作用域非法")
    rule, surface, serialization, start, end = _carrier_rule(observation)
    return W02ObservedCarrier(
        _key(observation.stable_key, where="observed carrier key"),
        rule.carrier_kind, surface, serialization, start, end)


def _validate_pair(
        observation: ObservationRecord,
        evidence: TeacherEvidenceRecord,
        ) -> dict[str, Any]:
    if not isinstance(observation, ObservationRecord):
        raise TypeError("W-02 Candidate observation 类型错误")
    if not isinstance(evidence, TeacherEvidenceRecord):
        raise TypeError("W-02 Candidate Evidence 类型错误")
    if (observation.w_stage != "W-02" or observation.split != "train"
            or observation.payload_kind != "typed_carrier" or observation.language != "zh"):
        raise W02CandidateModelError("W-02 Candidate 只接受 train typed carrier")
    if (evidence.evidence_kind != W02_FORMAL_EVIDENCE_KIND
            or evidence.visible_from_stage != "W-02" or evidence.withdrawal_level != 3):
        raise W02CandidateModelError("W-02 Teacher Evidence 身份漂移")
    if (_key(evidence.observation_key, where="Evidence observation key")
            != _key(observation.stable_key, where="Observation stable key")):
        raise W02CandidateModelError("W-02 Evidence 未绑定当前 Observation")
    if (_key(evidence.source_ref_key, where="Evidence source key")
            != _key(observation.source_ref_key, where="Observation source key")):
        raise W02CandidateModelError("W-02 Evidence source 绑定漂移")
    value = evidence.typed_evidence.to_value()
    if not isinstance(value, dict) or value.get("definitive_truth_authoritative") != 0:
        raise W02CandidateModelError("W-02 Evidence 不得冒充 definitive truth")
    return value


def _learn_authored(
        expected: dict[str, Any],
        *,
        rule: W02CarrierRule,
        surface: str,
        serialization: str,
        start: int,
        end: int,
        ) -> tuple[tuple[W02OovUnitEvidence, ...], tuple[int, ...], int]:
    raw = _exact(expected, {
        "carrier_content_span", "carrier_kind", "definitive_truth_authoritative",
        "generation_target", "oov_boundaries", "oov_units", "surface",
    }, where="W-02 authored Evidence")
    if (raw["carrier_kind"] != rule.carrier_kind or raw["surface"] != surface
            or raw["generation_target"] != serialization
            or raw["carrier_content_span"] != [start, end]):
        raise W02CandidateModelError("authored Evidence 与 carrier 不一致")
    units = raw["oov_units"]
    points = raw["oov_boundaries"]
    if (not isinstance(units, list) or not units
            or any(not isinstance(item, str) or not item for item in units)
            or not isinstance(points, list)
            or any(type(item) is not int for item in points)):
        raise W02CandidateModelError("authored OOV units/boundaries 类型非法")
    if ("".join(units) != surface or points[0] != 0 or points[-1] != len(surface)
            or points != sorted(set(points)) or len(points) != len(units) + 1):
        raise W02CandidateModelError("authored OOV units/boundaries 不闭合")
    cursor = 0
    for unit, boundary in zip(units, points[1:]):
        cursor += len(unit)
        if cursor != boundary:
            raise W02CandidateModelError("authored OOV boundary 与 unit 长度漂移")
    learned = tuple(sorted(
        (W02OovUnitEvidence(item, _unit_signature(item), len(item)) for item in units),
        key=lambda item: item.key(),
    ))
    return learned, tuple(points), len(surface) + len(units) * 4 + 12


def _learn_ud(
        expected: dict[str, Any],
        *,
        rule: W02CarrierRule,
        surface: str,
        ) -> tuple[tuple[W02LexemeEvidence, ...], tuple[int, ...], int]:
    raw = _exact(expected, {
        "boundary_spans", "carrier_kind", "definitive_truth_authoritative",
        "dimension_scope", "morphology", "source_annotation",
    }, where="W-02 UD Evidence")
    if (raw["carrier_kind"] != rule.carrier_kind
            or raw["dimension_scope"] != "TOKEN_BOUNDARY_AND_ANNOTATED_MORPHOLOGY"
            or raw["source_annotation"] != "UD_CHINESE_GSDSIMP_R2_18"):
        raise W02CandidateModelError("UD Evidence 身份漂移")
    spans = raw["boundary_spans"]
    morphology = raw["morphology"]
    if not isinstance(spans, list) or not spans or not isinstance(morphology, list):
        raise W02CandidateModelError("UD boundary/morphology 类型非法")
    points = {0, len(surface)}
    for item in spans:
        row = _exact(item, {"end", "form", "start"}, where="UD boundary span")
        start = _strict_int(row["start"], where="UD span start")
        end = _strict_int(row["end"], where="UD span end")
        form = _strict_text(row["form"], where="UD span form")
        if end < start or surface[start:end] != form:
            raise W02CandidateModelError("UD boundary span 不能恢复 form")
        points.update((start, end))
    lexemes = []
    for item in morphology:
        row = _exact(item, {"feats", "form", "lemma", "node_id", "upos"},
                     where="UD morphology")
        feats = row["feats"]
        if (not isinstance(feats, list)
                or any(not isinstance(pair, list) or len(pair) != 2
                       or any(not isinstance(value, str) for value in pair)
                       for pair in feats)):
            raise W02CandidateModelError("UD feats 类型非法")
        lexemes.append(W02LexemeEvidence(
            _strict_text(row["form"], where="UD form"),
            str(row["lemma"]),
            _strict_text(row["upos"], where="UD upos"),
            canonical_json_bytes(feats).decode("utf-8"),
        ))
    return tuple(sorted(lexemes, key=lambda item: item.key())), tuple(sorted(points)), (
        len(surface) + len(spans) * 5 + len(lexemes) * 8 + 16)


def _learn_unicode(
        expected: dict[str, Any],
        *,
        rule: W02CarrierRule,
        surface: str,
        ) -> tuple[tuple[W02UnicodeUnit, ...], tuple[int, ...], int]:
    allowed = {
        "carrier_kind", "code_point_units", "definitive_truth_authoritative",
        "grapheme_candidate_boundaries", "nfc", "nfkc", "source_annotation",
    }
    if "page_title_sha256" in expected:
        allowed.add("page_title_sha256")
    raw = _exact(expected, allowed, where="W-02 Unicode Evidence")
    if (raw["carrier_kind"] != rule.carrier_kind
            or raw["source_annotation"] != "UNICODE_STANDARD_LIBRARY_DETERMINISTIC"):
        raise W02CandidateModelError("Unicode Evidence 身份漂移")
    if raw["nfc"] != unicodedata.normalize("NFC", surface):
        raise W02CandidateModelError("Unicode NFC Evidence 漂移")
    if raw["nfkc"] != unicodedata.normalize("NFKC", surface):
        raise W02CandidateModelError("Unicode NFKC Evidence 漂移")
    if "page_title_sha256" in raw:
        _strict_sha256(raw["page_title_sha256"], where="Wiktionary title sha")
        if raw["page_title_sha256"] != hashlib.sha256(surface.encode("utf-8")).hexdigest():
            raise W02CandidateModelError("Wiktionary title SHA 漂移")
    units = raw["code_point_units"]
    if not isinstance(units, list) or len(units) != len(surface):
        raise W02CandidateModelError("Unicode unit 数量漂移")
    learned = []
    for char, item in zip(surface, units):
        row = _exact(item, {"category", "code_point", "combining_class", "surface"},
                     where="Unicode unit")
        unit = W02UnicodeUnit(
            _strict_int(row["code_point"], where="Unicode code point"),
            _strict_text(row["category"], where="Unicode category"),
            _strict_int(row["combining_class"], where="Unicode combining class"),
        )
        if (row["surface"] != char or unit.code_point != ord(char)
                or unit.category != unicodedata.category(char)
                or unit.combining_class != unicodedata.combining(char)):
            raise W02CandidateModelError("Unicode unit Evidence 漂移")
        learned.append(unit)
    points = raw["grapheme_candidate_boundaries"]
    if (not isinstance(points, list) or any(type(item) is not int for item in points)
            or points != sorted(set(points)) or not points
            or points[0] != 0 or points[-1] != len(surface)):
        raise W02CandidateModelError("Unicode grapheme boundaries 非法")
    return tuple(learned), tuple(points), len(surface) * 7 + 16


def learn_w02_training_pair(
        observation: ObservationRecord,
        evidence: TeacherEvidenceRecord,
        ) -> W02LearningDelta:
    """把一个严格 train pair 转成与 worker 次序无关的学习增量。"""
    expected = _validate_pair(observation, evidence)
    rule, surface, serialization, start, end = _carrier_rule(observation)
    unicode_units: tuple[W02UnicodeUnit, ...] = ()
    lexemes: tuple[W02LexemeEvidence, ...] = ()
    oov_units: tuple[W02OovUnitEvidence, ...] = ()
    capabilities = {W02_CAPABILITY_CARRIER_RECONSTRUCTION}
    keys = set(expected)
    if "oov_units" in keys:
        evidence_mode = W02_EVIDENCE_AUTHORED
        oov_units, points, operations = _learn_authored(
            expected, rule=rule, surface=surface, serialization=serialization,
            start=start, end=end)
        capabilities.add(W02_CAPABILITY_OOV_BOUNDARY_LATTICE)
    elif "morphology" in keys:
        evidence_mode = W02_EVIDENCE_UD
        lexemes, points, operations = _learn_ud(
            expected, rule=rule, surface=surface)
        capabilities.add(W02_CAPABILITY_UD_MORPHOLOGY)
    elif "code_point_units" in keys:
        evidence_mode = W02_EVIDENCE_UNICODE
        unicode_units, points, operations = _learn_unicode(
            expected, rule=rule, surface=surface)
        capabilities.add(W02_CAPABILITY_UNICODE_ANALYSIS)
    else:
        raise W02CandidateModelError("W-02 Teacher Evidence family 未注册")
    evidence_value = evidence.to_dict()
    evidence_sha = _hash_value(evidence_value)
    semantic = {
        "boundary_points": list(points),
        "capabilities": sorted(capabilities),
        "carrier_rule": rule.to_dict(),
        "evidence_mode": evidence_mode,
        "lexemes": [item.to_dict() for item in lexemes],
        "observation_key": list(_key(observation.stable_key, where="observation key")),
        "oov_units": [item.to_dict() for item in oov_units],
        "source_ref_key": list(_key(observation.source_ref_key, where="source key")),
        "unicode_units": [item.to_dict() for item in unicode_units],
    }
    delta_sha = _hash_value(semantic)
    use_outcome_sha = _hash_value({
        "carrier_reconstruction": serialization
            == rule.prefix + surface + rule.suffix,
        "delta_sha256": delta_sha,
        "evidence_sha256": evidence_sha,
        "generation_target_matches": (
            expected.get("generation_target", serialization) == serialization),
        "observation_key": semantic["observation_key"],
        "status": "SUCCESS",
    })
    return W02LearningDelta(
        tuple(semantic["observation_key"]),
        tuple(semantic["source_ref_key"]),
        evidence_sha,
        delta_sha,
        evidence_mode,
        rule,
        unicode_units,
        lexemes,
        oov_units,
        tuple(sorted(capabilities)),
        points,
        use_outcome_sha,
        operations + len(serialization) + 24,
    )


def generate_with_carrier_rules(
        rules: tuple[tuple[W02CarrierRule, int], ...],
        *,
        carrier_kind: str,
        surface: str,
        ) -> W02CarrierGeneration:
    """按学得支持计数生成载体；并列冲突保持 ambiguous。"""
    _strict_text(carrier_kind, where="generation carrier kind")
    if not isinstance(surface, str):
        raise W02CandidateModelError("generation surface 必须是文本")
    candidates = tuple(
        (rule, count) for rule, count in rules
        if rule.carrier_kind == carrier_kind and type(count) is int and count > 0)
    if not candidates:
        return W02CarrierGeneration("UNKNOWN", carrier_kind, surface, "", 0, 0)
    best = max(count for _, count in candidates)
    winners = tuple(rule for rule, count in candidates if count == best)
    if len(winners) != 1:
        return W02CarrierGeneration("AMBIGUOUS", carrier_kind, surface, "", 0, 0)
    rule = winners[0]
    serialization = rule.prefix + surface + rule.suffix
    start = len(rule.prefix)
    return W02CarrierGeneration(
        "GENERATED", carrier_kind, surface, serialization,
        start, start + len(surface))


def boundary_lattice(
        surface: str,
        *,
        observed_unit_lengths: tuple[int, ...],
        ) -> tuple[int, ...]:
    """形成可撤回候选边界格；不在无 Evidence 时臆选唯一分段。"""
    if not isinstance(surface, str):
        raise W02CandidateModelError("boundary surface 必须是文本")
    if (not isinstance(observed_unit_lengths, tuple)
            or any(type(item) is not int or item <= 0 for item in observed_unit_lengths)):
        raise W02CandidateModelError("observed unit lengths 非法")
    points = {0, len(surface)}
    for index, char in enumerate(surface):
        if index and unicodedata.combining(char) == 0:
            points.add(index)
        if index and _coarse_char_kind(surface[index - 1]) != _coarse_char_kind(char):
            points.add(index)
    for start in range(len(surface)):
        for length in observed_unit_lengths:
            end = start + length
            if end <= len(surface):
                points.add(end)
    return tuple(sorted(points))


__all__ = [
    "W02CandidateModelError",
    "W02CandidatePrediction",
    "W02CarrierGeneration",
    "W02CarrierRule",
    "W02LearningDelta",
    "W02LexemeEvidence",
    "W02MorphologyCandidate",
    "W02ObservedCarrier",
    "W02OovUnitEvidence",
    "W02UnicodeUnit",
    "boundary_lattice",
    "generate_with_carrier_rules",
    "learn_w02_training_pair",
    "observe_w02_carrier",
]
