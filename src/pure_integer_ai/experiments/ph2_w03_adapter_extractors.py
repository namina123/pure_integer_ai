"""W-03 四类 train Observation 的纯解析与候选抽取。"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from pure_integer_ai.experiments.ph2_authored_construction_course import (
    AuthoredConstructionCourseError,
    validate_construction_payload,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    ObservationRecord,
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_mediawiki_multistream_adapter import (
    MediaWikiPageError,
    extract_balanced_templates,
)
from pure_integer_ai.experiments.ph2_source_pack_compiler import (
    SourcePackCompilerError,
    validate_source_pack_payloads,
)
from pure_integer_ai.experiments.ph2_wikidata_adapter import (
    WikidataAdapterError,
    parse_wikidata_entity_terms_bytes,
)


W03_PAYLOAD_KINDS = frozenset({
    "ConstructionCandidateV1",
    "RAW_SOURCE_OBSERVATION_V1",
    "SenseBoundaryQuery",
})

_SENSE_QUERY_KEYS = frozenset({
    "candidate_sense", "context", "query_kind", "surface",
})
_RAW_SOURCE_KEYS = frozenset({
    "combination_axes",
    "combination_cluster_key",
    "definitive_truth_authoritative",
    "raw_observation",
    "raw_observation_append_only",
    "raw_observation_sha256",
    "source_pack_contract_version",
})
_WIKIDATA_RAW_KEYS = frozenset({
    "entity_json_utf8", "qid", "revision",
})
_WIKTIONARY_RAW_KEYS = frozenset({
    "contributor", "page_id", "redirect_title", "revision_id",
    "text", "timestamp", "title",
})
_HEADING_RE = re.compile(r"^(=+)([^=\r\n]+)\1\s*$")
_DEFINITION_RE = re.compile(r"^#(?![#*:])\s*(\S.*?)(?:\r?\n)?$")


class W03AdapterExtractionError(ValueError):
    """W-03 typed payload 的 schema、来源结构或候选边界非法。"""


def _text(value: Any, *, where: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or value.strip() != value:
        raise W03AdapterExtractionError(f"{where} 必须是无首尾空白文本")
    if not value and not allow_empty:
        raise W03AdapterExtractionError(f"{where} 不能为空")
    return value


def _exact(value: Any, keys: frozenset[str], *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or frozenset(value) != keys:
        raise W03AdapterExtractionError(f"{where} 字段集合非法")
    return value


def _text_components(*values: str) -> tuple[int, ...]:
    result: list[int] = [1, len(values)]
    for value in values:
        text = _text(value, where="identity text", allow_empty=True)
        result.extend((len(text), *(ord(character) for character in text)))
    return tuple(result)


def _json_components(value: dict[str, Any]) -> tuple[int, ...]:
    payload = canonical_json_bytes(value)
    return 1, len(payload), *payload


@dataclass(frozen=True)
class W03SurfaceAnchor:
    """一个来源内表层或结构字段锚点。"""

    branch_language: str
    surface: str
    members: tuple[tuple[int, int], ...]
    ordinal: int
    provenance: CanonicalJsonObject


@dataclass(frozen=True)
class W03ExtractedCandidate:
    """尚未进入 lifecycle 的 Sense/Concept/context 候选输入。"""

    candidate_kind: str
    anchor_ordinal: int
    sense_key: tuple[int, ...]
    concept_key: tuple[int, ...]
    context_key: tuple[int, ...]
    competition_context_key: tuple[int, ...]
    external_nondefinitive: bool
    lexicalized_multiword: bool
    provenance: CanonicalJsonObject


@dataclass(frozen=True)
class W03ExtractedObservation:
    """保留原 Observation 的纯解析结果，不含 teacher label。"""

    observation: ObservationRecord
    anchors: tuple[W03SurfaceAnchor, ...]
    candidates: tuple[W03ExtractedCandidate, ...]
    parser_provenance: CanonicalJsonObject
    external_nondefinitive: bool


def _surface_members(context: str, surface: str) -> tuple[tuple[int, int], ...]:
    members: list[tuple[int, int]] = []
    cursor = 0
    while True:
        start = context.find(surface, cursor)
        if start < 0:
            break
        members.append((start, start + len(surface)))
        cursor = start + len(surface)
    if not members:
        raise W03AdapterExtractionError("SenseBoundaryQuery surface 不在 context 中")
    return tuple(members)


def _extract_sense_query(
        observation: ObservationRecord,
        ) -> W03ExtractedObservation:
    raw = _exact(
        observation.typed_payload.to_value(),
        _SENSE_QUERY_KEYS,
        where="SenseBoundaryQuery",
    )
    if raw["query_kind"] != "sense_boundary":
        raise W03AdapterExtractionError("SenseBoundaryQuery query_kind 漂移")
    surface = _text(raw["surface"], where="sense surface")
    context = _text(raw["context"], where="sense context")
    candidate_sense = _text(
        raw["candidate_sense"], where="candidate_sense")
    anchor = W03SurfaceAnchor(
        observation.language,
        surface,
        _surface_members(context, surface),
        0,
        CanonicalJsonObject.from_value({
            "context": context,
            "query_kind": "sense_boundary",
        }),
    )
    candidate = W03ExtractedCandidate(
        "AUTHORED_SENSE",
        0,
        (1, *observation.stable_key.stable_key(),
         *_text_components(candidate_sense)),
        (1, *_text_components(candidate_sense)),
        (1, *_text_components(context)),
        (1, *_text_components(context)),
        False,
        False,
        CanonicalJsonObject.from_value({
            "candidate_sense": candidate_sense,
            "sample_role": observation.sample_role,
        }),
    )
    return W03ExtractedObservation(
        observation,
        (anchor,),
        (candidate,),
        CanonicalJsonObject.from_value({
            "extractor": "SENSE_BOUNDARY_QUERY_V1",
            "query_kind": "sense_boundary",
        }),
        False,
    )


def _extract_construction(
        observation: ObservationRecord,
        ) -> W03ExtractedObservation:
    validate_construction_payload(observation.typed_payload)
    raw = observation.typed_payload.to_value()
    observed = raw["observed_surface"]
    surface = _text(observed["text"], where="construction surface")
    scope = raw["surface_scope"]
    anchors: list[W03SurfaceAnchor] = []
    construction_anchor = -1
    for ordinal, span in enumerate(raw["spans"]):
        members = tuple((item[0], item[1]) for item in span["members"])
        anchors.append(W03SurfaceAnchor(
            _text(scope["language"], where="construction language"),
            surface,
            members,
            ordinal,
            CanonicalJsonObject.from_value({
                "fixed": span["fixed"],
                "span_id": span["span_id"],
                "span_kind": span["span_kind"],
            }),
        ))
        if span["span_id"] == "construction":
            construction_anchor = ordinal
    identity = raw["construction_identity"]
    candidates: tuple[W03ExtractedCandidate, ...] = ()
    if identity["present"] == 1:
        if construction_anchor < 0:
            raise W03AdapterExtractionError("LC-03 构式对象缺 construction span")
        construction_key = _text(
            identity["construction_key"], where="construction_key")
        proposition_group = _text(
            raw["split_identity"]["proposition_group"],
            where="proposition_group",
        )
        candidate_group = _text(
            raw["candidate_group"], where="candidate_group")
        lexicalization = _text(
            identity["lexicalization_state"], where="lexicalization_state")
        context_value = {
            "event_core_mapping": raw["event_core_mapping"],
            "surface_scope": scope,
        }
        candidates = (W03ExtractedCandidate(
            "LC03_CONSTRUCTION_SENSE",
            construction_anchor,
            (2, *observation.stable_key.stable_key(),
             *_text_components(construction_key)),
            (2, *_text_components(proposition_group)),
            (2, *_json_components(context_value)),
            (2, *_text_components(candidate_group)),
            False,
            lexicalization in {
                "PARTIAL_LEXICALIZED", "WHOLE_LEXICALIZED"},
            CanonicalJsonObject.from_value({
                "candidate_kind": raw["candidate_kind"],
                "construction_key": construction_key,
                "lexicalization_state": lexicalization,
                "selection_state": raw["selection_state"],
            }),
        ),)
    return W03ExtractedObservation(
        observation,
        tuple(anchors),
        candidates,
        CanonicalJsonObject.from_value({
            "construction_present": identity["present"],
            "extractor": "LC03_CONSTRUCTION_V1",
            "selection_state": raw["selection_state"],
        }),
        False,
    )


def _language_matches(candidate: str, requested: str) -> bool:
    return candidate == requested or candidate.startswith(requested + "-")


def _extract_wikidata(
        observation: ObservationRecord,
        raw_source: dict[str, Any],
        axes: dict[str, Any],
        ) -> W03ExtractedObservation:
    raw = _exact(raw_source, _WIKIDATA_RAW_KEYS, where="Wikidata raw")
    qid = _text(raw["qid"], where="Wikidata qid")
    revision = raw["revision"]
    if type(revision) is not int or revision <= 0:
        raise W03AdapterExtractionError("Wikidata revision 非法")
    payload = _text(
        raw["entity_json_utf8"], where="Wikidata entity_json_utf8",
    ).encode("utf-8")
    terms = parse_wikidata_entity_terms_bytes(
        payload,
        expected_qid=qid,
        expected_revision=revision,
    )
    labels = terms.labels.to_value()
    descriptions = terms.descriptions.to_value()
    aliases = terms.aliases.to_value()
    grouped: dict[tuple[str, str], set[str]] = {}
    for language, entry in labels.items():
        if _language_matches(language, observation.language):
            grouped.setdefault((language, entry["value"]), set()).add("label")
    for language, entries in aliases.items():
        if not _language_matches(language, observation.language):
            continue
        for entry in entries:
            grouped.setdefault((language, entry["value"]), set()).add("alias")
    if not grouped:
        raise W03AdapterExtractionError("Wikidata 没有目标语言 term")
    anchors: list[W03SurfaceAnchor] = []
    candidates: list[W03ExtractedCandidate] = []
    for ordinal, ((language, surface), roles) in enumerate(sorted(grouped.items())):
        description = descriptions.get(language)
        if description is None:
            description = descriptions.get(observation.language)
        context_value = (
            {"description": description["value"], "language": language}
            if description is not None
            else {"combination_axes": axes, "language": language}
        )
        anchors.append(W03SurfaceAnchor(
            language,
            surface,
            ((0, 0),),
            ordinal,
            CanonicalJsonObject.from_value({
                "field_roles": sorted(roles),
                "qid": qid,
                "structured_field_ordinal": ordinal,
            }),
        ))
        candidates.append(W03ExtractedCandidate(
            "WIKIDATA_TERM",
            ordinal,
            (3, *observation.stable_key.stable_key(), ordinal,
             *_text_components(language, surface)),
            (3, *_text_components(qid)),
            (3, *_json_components(context_value)),
            (3, *_json_components(context_value)),
            True,
            len(surface) > 1,
            CanonicalJsonObject.from_value({
                "field_roles": sorted(roles),
                "qid": qid,
                "revision": revision,
            }),
        ))
    return W03ExtractedObservation(
        observation,
        tuple(anchors),
        tuple(candidates),
        CanonicalJsonObject.from_value({
            "alias_language_count": len(aliases),
            "description_language_count": len(descriptions),
            "extractor": "WIKIDATA_ENTITY_TERMS_V1",
            "label_language_count": len(labels),
            "qid": qid,
            "revision": revision,
        }),
        True,
    )


@dataclass(frozen=True)
class _WiktionaryDefinition:
    section_path: tuple[str, ...]
    start: int
    end: int
    text: str


def _wiktionary_definitions(text: str) -> tuple[_WiktionaryDefinition, ...]:
    headings: dict[int, str] = {}
    definitions: list[_WiktionaryDefinition] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\r\n")
        heading = _HEADING_RE.fullmatch(stripped)
        if heading is not None:
            level = len(heading.group(1))
            headings = {
                key: value for key, value in headings.items() if key < level}
            headings[level] = _text(
                heading.group(2).strip(), where="Wiktionary heading")
        else:
            definition = _DEFINITION_RE.fullmatch(line)
            if definition is not None:
                content = definition.group(1)
                start = offset + line.index(content)
                definitions.append(_WiktionaryDefinition(
                    tuple(headings[key] for key in sorted(headings)),
                    start,
                    start + len(content),
                    content,
                ))
        offset += len(line)
    return tuple(definitions)


def _extract_wiktionary(
        observation: ObservationRecord,
        raw_source: dict[str, Any],
        ) -> W03ExtractedObservation:
    raw = _exact(raw_source, _WIKTIONARY_RAW_KEYS, where="Wiktionary raw")
    title = _text(raw["title"], where="Wiktionary title")
    redirect = _text(
        raw["redirect_title"], where="Wiktionary redirect", allow_empty=True)
    text = _text(raw["text"], where="Wiktionary text")
    page_id = raw["page_id"]
    revision = raw["revision_id"]
    if (type(page_id) is not int or page_id <= 0
            or type(revision) is not int or revision <= 0):
        raise W03AdapterExtractionError("Wiktionary page/revision 非法")
    templates = extract_balanced_templates(
        text, max_templates=4096, max_depth=64)
    definitions = () if redirect else _wiktionary_definitions(text)
    if not redirect and not definitions:
        raise W03AdapterExtractionError("Wiktionary 正文没有可审计 definition")
    anchors: list[W03SurfaceAnchor] = []
    candidates: list[W03ExtractedCandidate] = []
    if redirect:
        anchors.append(W03SurfaceAnchor(
            observation.language,
            title,
            ((0, 0),),
            0,
            CanonicalJsonObject.from_value({
                "redirect_title": redirect,
                "structured_field": "page_title",
            }),
        ))
    else:
        for ordinal, definition in enumerate(definitions):
            context_value = {"section_path": list(definition.section_path)}
            anchors.append(W03SurfaceAnchor(
                observation.language,
                title,
                ((definition.start, definition.end),),
                ordinal,
                CanonicalJsonObject.from_value({
                    "definition_text": definition.text,
                    "section_path": list(definition.section_path),
                }),
            ))
            candidates.append(W03ExtractedCandidate(
                "WIKTIONARY_SENSE",
                ordinal,
                (4, *observation.stable_key.stable_key(), ordinal,
                 *_text_components(definition.text)),
                (4, page_id, ordinal + 1,
                 *_text_components(definition.text)),
                (4, *_json_components(context_value)),
                (4, *_json_components(context_value)),
                True,
                len(title) > 1,
                CanonicalJsonObject.from_value({
                    "definition_text": definition.text,
                    "page_id": page_id,
                    "revision_id": revision,
                    "section_path": list(definition.section_path),
                }),
            ))
    return W03ExtractedObservation(
        observation,
        tuple(anchors),
        tuple(candidates),
        CanonicalJsonObject.from_value({
            "definition_count": len(definitions),
            "extractor": "WIKTIONARY_WIKITEXT_SENSE_V1",
            "page_id": page_id,
            "redirect": int(bool(redirect)),
            "revision_id": revision,
            "template_spans": [item.to_dict() for item in templates],
        }),
        True,
    )


def _extract_raw_source(
        observation: ObservationRecord,
        ) -> W03ExtractedObservation:
    payload = _exact(
        observation.typed_payload.to_value(),
        _RAW_SOURCE_KEYS,
        where="RAW_SOURCE_OBSERVATION_V1",
    )
    raw = payload["raw_observation"]
    axes = payload["combination_axes"]
    if not isinstance(axes, dict) or not axes:
        raise W03AdapterExtractionError("source combination_axes 非法")
    if isinstance(raw, dict) and frozenset(raw) == _WIKIDATA_RAW_KEYS:
        return _extract_wikidata(observation, raw, axes)
    if isinstance(raw, dict) and frozenset(raw) == _WIKTIONARY_RAW_KEYS:
        return _extract_wiktionary(observation, raw)
    raise W03AdapterExtractionError("W-03 raw source 类型未注册")


def extract_w03_observations(
        observations: tuple[ObservationRecord, ...],
        ) -> tuple[W03ExtractedObservation, ...]:
    """验证并抽取全部 W-03 train Observation，未知类型立即拒绝。"""
    if (not isinstance(observations, tuple)
            or any(not isinstance(item, ObservationRecord)
                   for item in observations)):
        raise TypeError("W-03 observations 必须是 ObservationRecord tuple")
    selected = tuple(item for item in observations if item.w_stage == "W-03")
    if not selected or any(item.split != "train" for item in selected):
        raise W03AdapterExtractionError("W-03 adapter 只接受非空 train 集")
    raw_sources = tuple(
        item for item in selected
        if item.payload_kind == "RAW_SOURCE_OBSERVATION_V1")
    try:
        if raw_sources:
            validate_source_pack_payloads(raw_sources)
    except SourcePackCompilerError as exc:
        raise W03AdapterExtractionError(
            f"W-03 source pack validation failed: {exc}") from exc
    extracted: list[W03ExtractedObservation] = []
    for observation in selected:
        if observation.payload_kind not in W03_PAYLOAD_KINDS:
            raise W03AdapterExtractionError(
                f"W-03 payload kind 未注册: {observation.payload_kind}")
        try:
            if observation.payload_kind == "SenseBoundaryQuery":
                value = _extract_sense_query(observation)
            elif observation.payload_kind == "ConstructionCandidateV1":
                value = _extract_construction(observation)
            else:
                value = _extract_raw_source(observation)
        except W03AdapterExtractionError:
            raise
        except (
                AuthoredConstructionCourseError,
                MediaWikiPageError,
                WikidataAdapterError,
                ) as exc:
            raise W03AdapterExtractionError(
                f"W-03 structured parser rejected Observation: {exc}") from exc
        extracted.append(value)
    return tuple(sorted(
        extracted,
        key=lambda item: item.observation.stable_key.stable_key(),
    ))


__all__ = [
    "W03AdapterExtractionError",
    "W03ExtractedCandidate",
    "W03ExtractedObservation",
    "W03SurfaceAnchor",
    "extract_w03_observations",
]
