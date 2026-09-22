"""Source-wide LC-16 projection course with an integer-only artifact boundary.

The legacy carrier adapters are intentionally not imported here.  This course
consumes the already published carrier payload envelope, retains only Unicode
scalar units and structure coordinates, and leaves semantic selection to the
graph/query path.  HTML and Markdown therefore use the same opaque structural
scanner as every other carrier; no host parser or vocabulary is part of the
training boundary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from pure_integer_ai.cognition.shared.artifact_envelope import (
    ANCHOR_DOCUMENT_REGION,
    ANCHOR_TEXT_RANGE,
    ANCHOR_TREE_PATH,
    PROJECTION_GENERATION,
    PROJECTION_REASONING,
    PROJECTION_UNDERSTANDING,
    RAW_UNIT_UNICODE_SCALAR,
    ArtifactAnchor,
    ArtifactEnvelope,
    ArtifactReferenceBinding,
    ArtifactSemanticProjection,
    ArtifactStructureNode,
    make_artifact_anchor,
    make_artifact_envelope,
    make_artifact_semantic_projection,
    make_artifact_structure_node,
)
from pure_integer_ai.cognition.shared.formal_artifact import ArtifactAuthority
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_UNKNOWN,
    EvidenceRecord,
    HypothesisKey,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    CorpusVersion,
    CurriculumVersion,
    ObjectIdentity,
    ParserVersion,
    PrimitiveVersion,
    SourceRef,
    VersionBundle,
    concept_identity,
    language_branch_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    ScopeIdentity,
    document_scope,
)
from pure_integer_ai.cognition.understanding.artifact_query_input import (
    ArtifactQueryInput,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.experiments.trained_graph_query_bridge import (
    TrainedGraphQueryBridge,
)
from pure_integer_ai.storage.backend import SQLiteBackend
from pure_integer_ai.storage.edge_store import EPI_STRUCTURED, SOURCE_DERIVED


FORMAT_VERSION = 1
RECORD_KIND = 16619001
COURSE_KIND = 16619002
GRAPH_KIND = 16619003
QUERY_KIND = 16619004
MODEL_BOUND_COURSE_KIND = 16619920
MODEL_BOUND_STATE_UNBOUND = 0
MODEL_BOUND_STATE_BOUND = 1
TRAINED_FACT_CARRIER_RECORD_KIND = 16619926
TRAINED_FACT_CARRIER_COURSE_KIND = 16619927

CARRIER_KEYS = (
    "DOCUMENT_CONTAINER",
    "HTML",
    "MARKDOWN",
    "MATH_NOTATION",
    "PLAIN_TEXT",
    "REFERENCE_LINK_EMBED",
    "SOURCE_CODE",
    "TABLE_GRID",
    "TRANSCRIBED_OCR_ASR",
)
CARRIER_CODES = {name: index for index, name in enumerate(CARRIER_KEYS, 1)}
SAMPLE_CODES = {
    "POSITIVE": 1,
    "NEGATIVE": 2,
    "AMBIGUOUS": 3,
    "UNKNOWN": 4,
    "REVISION": 5,
    "GENERATION": 6,
    "RETENTION": 7,
}
SPLIT_CODES = {
    "train": 1,
    "dev": 2,
    "held_out": 3,
    "adversarial": 4,
}
SAMPLE_PATHS = tuple(
    Path("data") / "ph2" / f"lc16_{name.lower()}_carrier_v1.jsonl.sample"
    for name in CARRIER_KEYS
)

# The source files use a few historical names which are not a carrier identity.
SAMPLE_PATHS = (
    Path("data/ph2/lc16_document_container_carrier_v1.jsonl.sample"),
    Path("data/ph2/lc16_html_carrier_v1.jsonl.sample"),
    Path("data/ph2/lc16_markdown_carrier_v1.jsonl.sample"),
    Path("data/ph2/lc16_math_notation_carrier_v1.jsonl.sample"),
    Path("data/ph2/lc16_plain_text_carrier_v1.jsonl.sample"),
    Path("data/ph2/lc16_reference_link_embed_carrier_v1.jsonl.sample"),
    Path("data/ph2/lc16_source_code_carrier_v1.jsonl.sample"),
    Path("data/ph2/lc16_table_grid_carrier_v1.jsonl.sample"),
    Path("data/ph2/lc16_transcribed_ocr_asr_carrier_v1.jsonl.sample"),
)

LICENSE_CODES = {
    "CC0-1.0": 1,
}
SEMANTIC_BINDING_STRUCTURE_ONLY = 0
SEMANTIC_BINDING_MODEL_ACTIVE = 1


class SourceWideProjectionError(RuntimeError):
    """The source-wide projection contract is not closed."""


def _pack(values: Iterable[int]) -> tuple[int, ...]:
    values = tuple(values)
    return (len(values), *values)


def _digest_ints(value: bytes) -> tuple[int, ...]:
    """Return a bounded integer digest; hexadecimal text never enters the course."""
    return tuple(hashlib.sha256(value).digest()[:16])


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _integer_tree(value: object, *, where: str) -> None:
    """Reject textual payload leaves while allowing JSON object field names."""
    if type(value) is int:
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _integer_tree(item, where=f"{where}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise SourceWideProjectionError(f"{where} field key 非文本")
            _integer_tree(item, where=f"{where}.{key}")
        return
    raise SourceWideProjectionError(f"{where} 含非整数 payload")


def _read_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise SourceWideProjectionError(
                f"carrier source JSON 损坏: {path}:{line_number}") from error
        if not isinstance(value, dict):
            raise SourceWideProjectionError(f"carrier source record 非 object: {path}:{line_number}")
        rows.append(value)
    if not rows:
        raise SourceWideProjectionError(f"carrier source 为空: {path}")
    return tuple(rows)


def _strict_case_key(value: object, *, where: str) -> tuple[int, ...]:
    if (not isinstance(value, list) or not value
            or any(type(item) is not int for item in value)):
        raise SourceWideProjectionError(f"{where} 必须是整数列表")
    return tuple(value)


def _shape(raw_units: tuple[int, ...]) -> tuple[int, ...]:
    """A carrier-neutral structural signature, never a lexical or character key."""
    if not raw_units:
        raise SourceWideProjectionError("空 raw envelope 不可进入查询课程")
    return (
        len(raw_units),
        sum(value == 10 for value in raw_units),
        sum(value == 9 for value in raw_units),
        sum(value == 32 for value in raw_units),
        sum(value in (123, 125, 91, 93, 40, 41) for value in raw_units),
        sum(value in (60, 62, 35, 42, 92) for value in raw_units),
        min(raw_units),
        max(raw_units),
    )


def _source(
        carrier_code: int,
        sample_index: int,
        ) -> SourceRef:
    owner = GLOBAL_OWNER_SCOPE
    versions = VersionBundle(
        CorpusVersion(2), ParserVersion(2), PrimitiveVersion(2),
        CurriculumVersion(2),
    )
    return SourceRef(
        16619500,
        16619500 + carrier_code,
        sample_index,
        owner,
        versions,
    )


def _concept(source: SourceRef, *key: int) -> ObjectIdentity:
    return concept_identity(key, owner=source.owner, versions=source.versions)


def _authority(source: SourceRef, domain: int) -> ArtifactAuthority:
    return ArtifactAuthority(_concept(source, domain), _concept(source, domain + 1))


def _metadata_code(value: object, *, where: str) -> tuple[int, ...]:
    if not isinstance(value, str) or not value:
        raise SourceWideProjectionError(f"{where} 必须是非空文本")
    return _digest_ints(value.encode("utf-8"))


def _verify_raw_identity(row: dict[str, object], raw_text: str) -> None:
    """Bind the integer envelope to the published source bytes before parsing."""
    declared_count = row.get("raw_unit_count")
    declared_sha256 = row.get("raw_utf8_sha256")
    payload = raw_text.encode("utf-8")
    if type(declared_count) is not int or declared_count != len(raw_text):
        raise SourceWideProjectionError("raw_unit_count 与 source 内容不一致")
    if (not isinstance(declared_sha256, str)
            or hashlib.sha256(payload).hexdigest() != declared_sha256):
        raise SourceWideProjectionError("raw_utf8_sha256 与 source 内容不一致")


def _split_for_sample(sample_kind: str) -> int:
    return {
        "POSITIVE": SPLIT_CODES["train"],
        "NEGATIVE": SPLIT_CODES["train"],
        "AMBIGUOUS": SPLIT_CODES["dev"],
        "UNKNOWN": SPLIT_CODES["held_out"],
        "REVISION": SPLIT_CODES["held_out"],
        "GENERATION": SPLIT_CODES["adversarial"],
        "RETENTION": SPLIT_CODES["adversarial"],
    }[sample_kind]


def _license_code(value: object) -> int:
    if not isinstance(value, str) or value not in LICENSE_CODES:
        raise SourceWideProjectionError("license_id 未登记")
    return LICENSE_CODES[value]


def _artifact_input(
        carrier_code: int,
        sample_index: int,
        row: dict[str, object],
        source_path: Path,
        ) -> tuple[ArtifactQueryInput, dict[str, object]]:
    sample_kind = row.get("sample_kind")
    if not isinstance(sample_kind, str) or sample_kind not in SAMPLE_CODES:
        raise SourceWideProjectionError("sample_kind 未登记")
    raw_text = row.get("raw_text")
    if not isinstance(raw_text, str) or not raw_text:
        raise SourceWideProjectionError("raw_text 必须是非空文本")
    _verify_raw_identity(row, raw_text)
    case_key = _strict_case_key(row.get("case_key"), where="case_key")
    raw_units = tuple(ord(item) for item in raw_text)
    shape = _shape(raw_units)
    source = _source(carrier_code, sample_index)
    scope = document_scope(source)
    parser = _authority(source, 16619600)
    renderer = _authority(source, 16619602)
    envelope = make_artifact_envelope(
        source=source,
        scope=scope,
        carrier_family=structure_concept_identity(
            (16619610, carrier_code), owner=source.owner, versions=source.versions),
        raw_unit_kind=RAW_UNIT_UNICODE_SCALAR,
        raw_units=raw_units,
        media_profile=_concept(source, 16619611, carrier_code),
        language_branch=language_branch_identity(
            (16619612, 1), owner=source.owner, versions=source.versions),
        parser=parser,
        renderer=renderer,
        envelope_key=(16619620, carrier_code, sample_index),
    )
    text_anchor = make_artifact_anchor(
        envelope_identity=envelope.identity, source=source, scope=scope,
        anchor_kind=ANCHOR_TEXT_RANGE,
        coordinates=(0, len(raw_units)), parser=parser,
        linked_text_anchor=None,
        anchor_key=(16619621, carrier_code, sample_index, 1),
    )
    region_anchor = make_artifact_anchor(
        envelope_identity=envelope.identity, source=source, scope=scope,
        anchor_kind=ANCHOR_DOCUMENT_REGION,
        coordinates=(0, len(raw_units)), parser=parser,
        linked_text_anchor=None,
        anchor_key=(16619621, carrier_code, sample_index, 2),
    )
    tree_anchor = make_artifact_anchor(
        envelope_identity=envelope.identity, source=source, scope=scope,
        anchor_kind=ANCHOR_TREE_PATH,
        coordinates=(0,), parser=parser,
        linked_text_anchor=None,
        anchor_key=(16619621, carrier_code, sample_index, 3),
    )
    anchors = tuple(sorted((text_anchor, region_anchor, tree_anchor),
                          key=lambda item: item.stable_key()))
    structure_family = structure_concept_identity(
        (16619630, carrier_code), owner=source.owner, versions=source.versions)
    node_kind = structure_concept_identity(
        (16619631, carrier_code, *shape), owner=source.owner, versions=source.versions)
    node = make_artifact_structure_node(
        envelope_identity=envelope.identity,
        source=source,
        scope=scope,
        anchor_identity=tree_anchor.identity,
        structure_family=structure_family,
        node_kind=node_kind,
        role=None,
        parent_identity=None,
        ordinal=0,
        qualifiers=(
            FORMAT_VERSION, carrier_code, SAMPLE_CODES[sample_kind],
            len(raw_units), *shape,
        ),
        node_key=(16619632, carrier_code, sample_index),
    )
    semantic_object = concept_identity((16619640, *shape))
    hypothesis = HypothesisKey(
        (16619641, 1),
        semantic_object.stable_key(),
        (16619642, *shape),
        scope,
        source,
    )
    evidence_source = SourceRef(
        16619501,
        16619700 + carrier_code,
        sample_index,
        GLOBAL_OWNER_SCOPE,
        source.versions,
    )
    evidence = EvidenceRecord(
        16619800 + carrier_code * 100 + sample_index,
        hypothesis,
        EVIDENCE_UNKNOWN,
        (16619643, carrier_code, sample_index),
        evidence_source,
        1,
        (carrier_code, sample_index, *shape),
    )
    projection = make_artifact_semantic_projection(
        envelope_identity=envelope.identity,
        source=source,
        scope=scope,
        anchor_identities=tuple(sorted(
            (text_anchor.identity, tree_anchor.identity),
            key=ObjectIdentity.stable_key,
        )),
        structure_node_identities=(node.identity,),
        projection_kind=concept_identity((16619650, 1)),
        semantic_object=semantic_object,
        lifecycle_state=concept_identity((16619651, 1)),
        hypothesis=hypothesis,
        evidence=(evidence,),
        directions=(
            PROJECTION_UNDERSTANDING,
            PROJECTION_REASONING,
            PROJECTION_GENERATION,
        ),
        projection_key=(16619652, carrier_code, sample_index),
    )
    artifact = ArtifactQueryInput(
        envelope,
        tuple(sorted(anchors, key=lambda item: item.stable_key())),
        (node,),
        (projection,),
    )
    metadata = {
        "attribution_code": list(_digest_ints(source_path.as_posix().encode("utf-8"))),
        "case_key": list(case_key),
        "carrier_code": carrier_code,
        "license_code": _license_code(row.get("license_id")),
        "projection_directions": [
            PROJECTION_UNDERSTANDING,
            PROJECTION_REASONING,
            PROJECTION_GENERATION,
        ],
        "semantic_binding_state": SEMANTIC_BINDING_STRUCTURE_ONLY,
        "semantic_object": list(semantic_object.stable_key()),
        "source_ref": list(source.stable_key()),
        "sample_code": SAMPLE_CODES[sample_kind],
        "raw_content_code": list(_digest_ints(raw_text.encode("utf-8"))),
        "source_path_code": list(_digest_ints(source_path.as_posix().encode("utf-8"))),
        "split_code": _split_for_sample(sample_kind),
        "version": list(source.versions.stable_key()),
    }
    return artifact, metadata


def _natural_key(artifact: ArtifactQueryInput) -> tuple[int, ...]:
    projection = artifact.projections[0]
    return (
        *_pack(artifact.envelope.source.stable_key()),
        *_pack(projection.semantic_object.stable_key()),
        *_pack(projection.projection_kind.stable_key()),
        *_pack(projection.directions),
        *_pack(projection.projection_key),
    )


def _artifact_from_integer_trace(
        payload: dict[str, object], *, allow_references: bool = False,
        ) -> ArtifactQueryInput:
    """Restore an ArtifactQueryInput from its integer-only trace."""
    if not isinstance(payload, dict):
        raise SourceWideProjectionError("artifact_input 缺失")
    references = payload.get("references", [])
    if references not in ([], None) and not allow_references:
        raise SourceWideProjectionError("当前 artifact course 不支持 references")
    try:
        return ArtifactQueryInput(
            ArtifactEnvelope.from_stable_key(tuple(payload["envelope"])),
            tuple(ArtifactAnchor.from_stable_key(tuple(item))
                  for item in payload["anchors"]),
            tuple(ArtifactStructureNode.from_stable_key(tuple(item))
                  for item in payload["structure_nodes"]),
            tuple(ArtifactSemanticProjection.from_stable_key(tuple(item))
                  for item in payload["projections"]),
            tuple(ArtifactReferenceBinding.from_stable_key(tuple(item))
                  for item in (references or ())),
        )
    except Exception as error:
        raise SourceWideProjectionError("artifact_input stable key 损坏") from error


@dataclass(frozen=True)
class IntegerProjectionRecord:
    """One source-wide row whose leaves are all strict integers."""

    carrier_code: int
    sample_code: int
    metadata: dict[str, object]
    artifact: ArtifactQueryInput

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "artifact_input": self.artifact.integer_trace(),
            "carrier_code": self.carrier_code,
            "format_version": FORMAT_VERSION,
            "metadata": self.metadata,
            "natural_key": list(_natural_key(self.artifact)),
            "record_kind": RECORD_KIND,
            "sample_code": self.sample_code,
        }
        _integer_tree(value, where="IntegerProjectionRecord")
        return value

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "IntegerProjectionRecord":
        if value.get("record_kind") != RECORD_KIND or value.get("format_version") != FORMAT_VERSION:
            raise SourceWideProjectionError("projection record kind/version 漂移")
        carrier_code = value.get("carrier_code")
        sample_code = value.get("sample_code")
        if type(carrier_code) is not int or carrier_code not in range(1, 10):
            raise SourceWideProjectionError("carrier_code 非法")
        if type(sample_code) is not int or sample_code not in range(1, 8):
            raise SourceWideProjectionError("sample_code 非法")
        payload = value.get("artifact_input")
        if not isinstance(payload, dict):
            raise SourceWideProjectionError("artifact_input 缺失")
        if payload.get("references") not in ([], None):
            raise SourceWideProjectionError(
                "当前 source-wide 课程不支持 references，必须显式留空")
        try:
            artifact = ArtifactQueryInput(
                ArtifactEnvelope.from_stable_key(tuple(payload["envelope"])),
                tuple(ArtifactAnchor.from_stable_key(tuple(item)) for item in payload["anchors"]),
                tuple(ArtifactStructureNode.from_stable_key(tuple(item)) for item in payload["structure_nodes"]),
                tuple(ArtifactSemanticProjection.from_stable_key(tuple(item)) for item in payload["projections"]),
                tuple(  # references are optional in this first course
                    from_reference for from_reference in ()
                ),
            )
        except Exception as error:
            raise SourceWideProjectionError("artifact_input stable key 损坏") from error
        metadata = value.get("metadata")
        if not isinstance(metadata, dict):
            raise SourceWideProjectionError("metadata 缺失")
        _integer_tree(value, where="IntegerProjectionRecord")
        if tuple(value.get("natural_key", ())) != _natural_key(artifact):
            raise SourceWideProjectionError("natural_key 漂移")
        if metadata.get("source_ref") != list(artifact.envelope.source.stable_key()):
            raise SourceWideProjectionError("metadata source_ref 漂移")
        if metadata.get("semantic_object") != list(
                artifact.projections[0].semantic_object.stable_key()):
            raise SourceWideProjectionError("metadata semantic_object 漂移")
        if metadata.get("semantic_binding_state") != SEMANTIC_BINDING_STRUCTURE_ONLY:
            raise SourceWideProjectionError("未知 semantic binding state")
        return cls(carrier_code, sample_code, metadata, artifact)


def compile_source_wide_course(
        repository_root: str | Path,
        run_root: str | Path,
        *,
        max_cases_per_carrier: int | None = None,
        ) -> dict[str, object]:
    """Compile all nine carriers into one zero-duplicate integer course."""
    repository = Path(repository_root).resolve()
    root = Path(run_root).resolve()
    if root.drive.upper() != "K:" or root.exists():
        raise SourceWideProjectionError("course run root 必须是新的 K 盘目录")
    root.mkdir(parents=True)
    records: list[IntegerProjectionRecord] = []
    natural_keys: set[tuple[int, ...]] = set()
    source_semantics: set[tuple[int, ...]] = set()
    for carrier_code, relative in enumerate(SAMPLE_PATHS, 1):
        path = (repository / relative).resolve()
        if not path.is_file():
            raise SourceWideProjectionError(f"carrier source 缺失: {relative}")
        rows = _read_jsonl(path)
        if max_cases_per_carrier is not None:
            if type(max_cases_per_carrier) is not int or max_cases_per_carrier <= 0:
                raise SourceWideProjectionError("max_cases_per_carrier 非法")
            rows = rows[:max_cases_per_carrier]
        for sample_index, row in enumerate(rows, 1):
            artifact, metadata = _artifact_input(carrier_code, sample_index, row, relative)
            natural = _natural_key(artifact)
            semantic = (
                *_pack(artifact.envelope.source.stable_key()),
                *_pack(artifact.projections[0].semantic_object.stable_key()),
            )
            if natural in natural_keys:
                raise SourceWideProjectionError("projection natural key 重复")
            if semantic in source_semantics:
                raise SourceWideProjectionError("同一 source semantic object 重复物化")
            natural_keys.add(natural)
            source_semantics.add(semantic)
            records.append(IntegerProjectionRecord(
                carrier_code,
                SAMPLE_CODES[str(row["sample_kind"])],
                metadata,
                artifact,
            ))
    records.sort(key=lambda item: (item.carrier_code, item.sample_code, item.artifact.envelope.source.stable_key()))
    course_path = root / "source_wide_projection.course.jsonl"
    with course_path.open("x", encoding="ascii", newline="\n") as stream:
        for record in records:
            stream.write(_canonical_bytes(record.to_dict()).decode("ascii"))
            stream.write("\n")
    course_bytes = course_path.read_bytes()
    manifest = {
        "carrier_codes": list(range(1, 10)),
        "course_kind": COURSE_KIND,
        "course_sha256": list(_digest_ints(course_bytes)),
        "format_version": FORMAT_VERSION,
        "natural_duplicate_count": 0,
        "record_count": len(records),
        "semantic_duplicate_count": 0,
        "source_semantic_count": len(source_semantics),
        "projection_count": len(records),
    }
    _integer_tree(manifest, where="course manifest")
    (root / "source_wide_projection.manifest.json").write_bytes(
        _canonical_bytes(manifest) + b"\n")
    return {
        "course": str(course_path),
        "manifest": str(root / "source_wide_projection.manifest.json"),
        "record_count": len(records),
        "carrier_count": 9,
        "natural_duplicate_count": 0,
        "semantic_duplicate_count": 0,
        "source_semantic_count": len(source_semantics),
    }


def read_source_wide_course(path: str | Path) -> tuple[IntegerProjectionRecord, ...]:
    """Read and rehydrate the integer course without reading source正文."""
    result = []
    for line in Path(path).read_text(encoding="ascii").splitlines():
        if line:
            result.append(IntegerProjectionRecord.from_dict(json.loads(line)))
    if not result:
        raise SourceWideProjectionError("projection course 为空")
    records = tuple(result)
    if tuple(sorted({item.carrier_code for item in records})) != tuple(range(1, 10)):
        raise SourceWideProjectionError("projection course carrier 覆盖不完整")
    natural_keys = tuple(_natural_key(item.artifact) for item in records)
    source_semantics = tuple(
        (*_pack(item.artifact.envelope.source.stable_key()),
         *_pack(item.artifact.projections[0].semantic_object.stable_key()))
        for item in records)
    if len(set(natural_keys)) != len(natural_keys):
        raise SourceWideProjectionError("projection course natural key 重复")
    if len(set(source_semantics)) != len(source_semantics):
        raise SourceWideProjectionError("projection course source semantic 重复")
    return records


@dataclass(frozen=True)
class TrainedFactCarrierRecord:
    """One active Core fact viewed through one lossless carrier envelope."""

    carrier_code: int
    fact_ordinal: int
    metadata: dict[str, object]
    artifact: ArtifactQueryInput

    def to_dict(self) -> dict[str, object]:
        value = {
            "artifact_input": self.artifact.integer_trace(),
            "carrier_code": self.carrier_code,
            "fact_ordinal": self.fact_ordinal,
            "format_version": FORMAT_VERSION,
            "metadata": self.metadata,
            "record_kind": TRAINED_FACT_CARRIER_RECORD_KIND,
        }
        _integer_tree(value, where="TrainedFactCarrierRecord")
        return value


def _remap_fact_evidence(
        history: tuple[EvidenceRecord, ...], hypothesis: HypothesisKey,
        ) -> tuple[EvidenceRecord, ...]:
    """Keep the original Evidence sources while binding them to the view."""
    return tuple(sorted((EvidenceRecord(
        item.evidence_id,
        hypothesis,
        item.stance,
        item.reason_key,
        item.source,
        item.timestamp_seq,
        item.payload,
        item.supersedes_evidence_id,
    ) for item in history), key=EvidenceRecord.stable_key))


def _trained_fact_artifact(
        carrier_code: int,
        fact_ordinal: int,
        fact: object,
        history: tuple[EvidenceRecord, ...],
        ) -> ArtifactQueryInput:
    """Build a carrier view whose members are exact active-graph identities."""
    raw_text = getattr(fact, "evidence_surface")
    if type(raw_text) is not str or not raw_text:
        raise SourceWideProjectionError("active fact evidence surface 缺失")
    raw_units = tuple(ord(item) for item in raw_text)
    source = _source(carrier_code, fact_ordinal)
    scope = document_scope(source)
    parser = _authority(source, 16619600)
    renderer = _authority(source, 16619602)
    envelope = make_artifact_envelope(
        source=source,
        scope=scope,
        carrier_family=structure_concept_identity(
            (16619610, carrier_code), owner=source.owner, versions=source.versions),
        raw_unit_kind=RAW_UNIT_UNICODE_SCALAR,
        raw_units=raw_units,
        media_profile=_concept(source, 16619611, carrier_code),
        language_branch=language_branch_identity(
            (16619612, 1), owner=source.owner, versions=source.versions),
        parser=parser,
        renderer=renderer,
        envelope_key=(16619928, carrier_code, fact_ordinal),
    )
    text_anchor = make_artifact_anchor(
        envelope_identity=envelope.identity, source=source, scope=scope,
        anchor_kind=ANCHOR_TEXT_RANGE, coordinates=(0, len(raw_units)),
        parser=parser, linked_text_anchor=None,
        anchor_key=(16619929, carrier_code, fact_ordinal, 0),
    )
    tree_anchor = make_artifact_anchor(
        envelope_identity=envelope.identity, source=source, scope=scope,
        anchor_kind=ANCHOR_TREE_PATH, coordinates=(0,), parser=parser,
        linked_text_anchor=None,
        anchor_key=(16619929, carrier_code, fact_ordinal, 1),
    )
    member_anchors = []
    member_values = [
        tuple(ord(item) for item in getattr(fact, "cue")),
        *tuple(tuple(ord(item) for item in binding.surface)
               for binding in getattr(fact, "bindings")),
    ]
    member_ranges = [
        (getattr(fact, "cue_start"), getattr(fact, "cue_end")),
        *tuple((binding.start, binding.end)
               for binding in getattr(fact, "bindings")),
    ]
    for ordinal, (start, end) in enumerate(member_ranges, 1):
        if not (0 <= start < end <= len(raw_units)):
            raise SourceWideProjectionError("active fact member Span 越界")
        member_anchors.append(make_artifact_anchor(
            envelope_identity=envelope.identity, source=source, scope=scope,
            anchor_kind=ANCHOR_TEXT_RANGE, coordinates=(start, end),
            parser=parser, linked_text_anchor=None,
            anchor_key=(16619929, carrier_code, fact_ordinal, ordinal + 1),
        ))
    anchors = tuple(sorted((text_anchor, tree_anchor, *member_anchors),
                           key=lambda item: item.stable_key()))
    shape = _shape(raw_units)
    node = make_artifact_structure_node(
        envelope_identity=envelope.identity, source=source, scope=scope,
        anchor_identity=tree_anchor.identity,
        structure_family=structure_concept_identity(
            (16619630, carrier_code), owner=source.owner, versions=source.versions),
        node_kind=structure_concept_identity(
            (16619631, carrier_code, *shape), owner=source.owner, versions=source.versions),
        role=None, parent_identity=None, ordinal=0,
        qualifiers=(FORMAT_VERSION, carrier_code, fact_ordinal, len(raw_units), *shape),
        node_key=(16619930, carrier_code, fact_ordinal),
    )
    proposition = getattr(fact, "proposition")
    predicate = getattr(fact, "predicate")
    projections = []
    targets = (predicate, *(binding.filler for binding in fact.bindings))
    for ordinal, (target, anchor, values) in enumerate(
            zip(targets, member_anchors, member_values, strict=True), 1):
        hypothesis = HypothesisKey(
            (16619931, 1), proposition.stable_key(),
            (16619932, carrier_code, fact_ordinal, ordinal), scope, source)
        evidence = _remap_fact_evidence(history, hypothesis)
        if not evidence:
            raise SourceWideProjectionError("trained fact carrier 缺少 Evidence")
        projections.append(make_artifact_semantic_projection(
            envelope_identity=envelope.identity, source=source, scope=scope,
            anchor_identities=tuple(sorted(
                (anchor.identity, tree_anchor.identity),
                key=ObjectIdentity.stable_key)),
            structure_node_identities=(node.identity,),
            projection_kind=concept_identity((16619933, ordinal), owner=source.owner,
                                             versions=source.versions),
            semantic_object=target,
            lifecycle_state=concept_identity((16619934, 1), owner=source.owner,
                                             versions=source.versions),
            hypothesis=hypothesis, evidence=evidence,
            directions=(PROJECTION_UNDERSTANDING, PROJECTION_REASONING,
                        PROJECTION_GENERATION),
            projection_key=(16619935, carrier_code, fact_ordinal, ordinal),
        ))
    return ArtifactQueryInput(
        envelope, anchors, (node,),
        tuple(sorted(projections, key=lambda item: item.stable_key())))


def read_trained_fact_carrier_course(
        path: str | Path,
        ) -> tuple[TrainedFactCarrierRecord, ...]:
    """Read and validate the integer-only trained-fact carrier course."""
    result = []
    for line in Path(path).read_text(encoding="ascii").splitlines():
        if not line:
            continue
        value = json.loads(line)
        if (value.get("record_kind") != TRAINED_FACT_CARRIER_RECORD_KIND
                or value.get("format_version") != FORMAT_VERSION):
            raise SourceWideProjectionError("trained fact carrier kind/version 漂移")
        carrier_code = value.get("carrier_code")
        fact_ordinal = value.get("fact_ordinal")
        if (type(carrier_code) is not int or carrier_code not in range(1, 10)
                or type(fact_ordinal) is not int or fact_ordinal <= 0):
            raise SourceWideProjectionError("trained fact carrier ordinal 非法")
        artifact = _artifact_from_integer_trace(value.get("artifact_input"))
        metadata = value.get("metadata")
        if not isinstance(metadata, dict):
            raise SourceWideProjectionError("trained fact carrier metadata 缺失")
        _integer_tree(value, where="TrainedFactCarrierRecord")
        if metadata.get("carrier_code") != carrier_code:
            raise SourceWideProjectionError("trained fact carrier metadata 漂移")
        if metadata.get("fact_ordinal") != fact_ordinal:
            raise SourceWideProjectionError("trained fact carrier fact ordinal 漂移")
        result.append(TrainedFactCarrierRecord(
            carrier_code, fact_ordinal, metadata, artifact))
    if not result or {item.carrier_code for item in result} != set(range(1, 10)):
        raise SourceWideProjectionError("trained fact carrier 未覆盖九类 carrier")
    keys = tuple((item.carrier_code, item.fact_ordinal,
                  item.artifact.stable_key()) for item in result)
    if len(set(keys)) != len(keys):
        raise SourceWideProjectionError("trained fact carrier natural key 重复")
    return tuple(sorted(result, key=lambda item: (item.carrier_code, item.fact_ordinal)))


def compile_trained_fact_carrier_course(
        model_database: str | Path,
        run_root: str | Path,
        *, max_facts: int | None = None,
        ) -> dict[str, object]:
    """Compile active Core facts into non-duplicated, all-carrier view records."""
    root = Path(run_root).resolve()
    if root.drive.upper() != "K:" or root.exists():
        raise SourceWideProjectionError("trained fact carrier run root 必须是新的 K 盘目录")
    if max_facts is not None and (type(max_facts) is not int or max_facts <= 0):
        raise SourceWideProjectionError("max_facts 必须为正严格整数")
    root.mkdir(parents=True)
    records: list[TrainedFactCarrierRecord] = []
    semantic_keys: set[tuple[int, ...]] = set()
    with TrainedGraphQueryBridge(Path(model_database).resolve()) as bridge:
        facts = tuple(bridge.facts[:max_facts] if max_facts is not None else bridge.facts)
        for fact_ordinal, fact in enumerate(facts, 1):
            carrier_code = (fact_ordinal - 1) % len(CARRIER_KEYS) + 1
            history = bridge.core_runtime.evidence_history(fact.proposition)
            artifact = _trained_fact_artifact(
                carrier_code, fact_ordinal, fact, history)
            metadata = {
                "carrier_code": carrier_code,
                "fact_ordinal": fact_ordinal,
                "semantic_binding_state": SEMANTIC_BINDING_MODEL_ACTIVE,
                "source_ref": list(artifact.envelope.source.stable_key()),
                "proposition": list(fact.proposition.stable_key()),
                "predicate": list(fact.predicate.stable_key()),
                "member_count": len(artifact.projections),
            }
            _integer_tree(metadata, where="trained fact carrier metadata")
            for projection in artifact.projections:
                key = (*_pack(artifact.envelope.source.stable_key()),
                       *_pack(projection.semantic_object.stable_key()))
                if key in semantic_keys:
                    raise SourceWideProjectionError(
                        "trained fact carrier source semantic object 重复")
                semantic_keys.add(key)
            records.append(TrainedFactCarrierRecord(
                carrier_code, fact_ordinal, metadata, artifact))
    if not records:
        raise SourceWideProjectionError("active Core facts 为空")
    course_path = root / "trained_fact_carrier.course.jsonl"
    with course_path.open("x", encoding="ascii", newline="\n") as stream:
        for record in records:
            stream.write(_canonical_bytes(record.to_dict()).decode("ascii"))
            stream.write("\n")
    manifest = {
        "carrier_codes": list(range(1, 10)),
        "course_kind": TRAINED_FACT_CARRIER_COURSE_KIND,
        "course_sha256": list(_digest_ints(course_path.read_bytes())),
        "fact_count": len(records),
        "format_version": FORMAT_VERSION,
        "projection_count": sum(len(item.artifact.projections) for item in records),
        "semantic_duplicate_count": 0,
        "source_semantic_count": len(semantic_keys),
    }
    _integer_tree(manifest, where="trained fact carrier manifest")
    manifest_path = root / "trained_fact_carrier.manifest.json"
    manifest_path.write_bytes(_canonical_bytes(manifest) + b"\n")
    graph = materialize_projection_graph(
        tuple(records), root / "trained_fact_carrier_graph.sqlite3")
    receipt = {
        "course": str(course_path),
        "manifest": str(manifest_path),
        "graph": graph,
        "fact_count": len(records),
        "projection_count": manifest["projection_count"],
        "free_dialogue_complete": 0,
    }
    (root / "receipt.json").write_bytes(_canonical_bytes(receipt) + b"\n")
    return receipt


def audit_projection_graph(database: str | Path) -> dict[str, int]:
    """Fail closed on physical duplicate keys or an oversized graph object."""
    path = Path(database).resolve(strict=True)
    checks = {
        "graph_object": ("identity_hash",),
        "graph_object_component": ("identity_hash", "component_ordinal"),
        "graph_statement": ("assertion_hash",),
    }
    report: dict[str, int] = {}
    with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) as conn:
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        for table, columns in checks.items():
            if table not in tables:
                raise SourceWideProjectionError(f"projection graph 缺少 {table}")
            projection = ",".join(f'"{column}"' for column in columns)
            groups, excess = conn.execute(
                f"SELECT COUNT(*), COALESCE(SUM(n - 1), 0) FROM ("
                f"SELECT {projection}, COUNT(*) AS n FROM \"{table}\" "
                f"GROUP BY {projection} HAVING COUNT(*) > 1)"
            ).fetchone()
            report[f"{table}_duplicate_groups"] = int(groups or 0)
            report[f"{table}_duplicate_excess"] = int(excess or 0)
            if groups or excess:
                raise SourceWideProjectionError(f"projection graph {table} natural key 重复")
        max_components = conn.execute(
            "SELECT COALESCE(MAX(n), 0) FROM ("
            "SELECT COUNT(*) AS n FROM graph_object_component "
            "GROUP BY identity_hash)"
        ).fetchone()[0]
        report["max_components_per_object"] = int(max_components or 0)
        if report["max_components_per_object"] > 262144:
            raise SourceWideProjectionError("projection graph 单对象 component 超预算")
    return report


def materialize_projection_graph(
        records: tuple[IntegerProjectionRecord, ...],
        database: str | Path,
        ) -> dict[str, int]:
    """Persist the shared carrier relation ontology into a fresh SQLite graph."""
    backend = SQLiteBackend(str(Path(database)), performance_mode="bulk")
    try:
        context = make_train_context(backend, companion=False)
        ontology = context.graph_ontology
        ontology.enable_physical_statement_projection()
        predicates = {
            index: ontology.materialize(concept_identity((16619100, index)))
            for index in range(1, 8)
        }
        statement_count = 0
        object_keys: set[tuple[int, ...]] = set()
        for record in records:
            artifact = record.artifact
            envelope = ontology.materialize(artifact.envelope.identity)
            anchor_refs = [ontology.materialize(item.identity) for item in artifact.anchors]
            node_refs = [ontology.materialize(item.identity) for item in artifact.structure_nodes]
            object_keys.add(artifact.envelope.identity.stable_key())
            scope = artifact.envelope.scope
            for anchor_ref in anchor_refs:
                ontology.relate(predicates[1], envelope, anchor_ref, scope=scope, provenance_kind=EPI_STRUCTURED, content_version=FORMAT_VERSION)
                statement_count += 1
            for node_ref in node_refs:
                ontology.relate(predicates[2], envelope, node_ref, scope=scope, provenance_kind=EPI_STRUCTURED, content_version=FORMAT_VERSION)
                statement_count += 1
            for projection in artifact.projections:
                projection_ref = ontology.materialize(projection.identity)
                semantic_ref = ontology.materialize(projection.semantic_object)
                hypothesis_ref = ontology.materialize(projection.hypothesis.object_identity())
                object_keys.update({
                    projection.identity.stable_key(),
                    projection.semantic_object.stable_key(),
                    projection.hypothesis.object_identity().stable_key(),
                })
                for node_ref in node_refs:
                    ontology.relate(predicates[3], node_ref, projection_ref, scope=scope, provenance_kind=EPI_STRUCTURED, content_version=FORMAT_VERSION)
                    statement_count += 1
                ontology.relate(predicates[4], projection_ref, semantic_ref, scope=scope, provenance_kind=EPI_STRUCTURED, content_version=FORMAT_VERSION)
                ontology.relate(predicates[5], projection_ref, hypothesis_ref, scope=scope, provenance_kind=EPI_STRUCTURED, content_version=FORMAT_VERSION)
                ontology.relate(predicates[6], projection_ref, ontology.materialize(projection.projection_kind), scope=scope, provenance_kind=EPI_STRUCTURED, content_version=FORMAT_VERSION)
                statement_count += 3
        backend.commit()
        audit = audit_projection_graph(database)
        with sqlite3.connect(f"file:{Path(database).resolve().as_posix()}?mode=ro", uri=True) as conn:
            object_count = int(conn.execute("SELECT COUNT(*) FROM graph_object").fetchone()[0])
            physical_statement_count = int(conn.execute("SELECT COUNT(*) FROM graph_statement").fetchone()[0])
        return {
            "object_count": object_count,
            "statement_count": physical_statement_count,
            "logical_relation_count": statement_count,
            "tracked_object_count": len(object_keys),
            **audit,
        }
    finally:
        backend.close()


def _model_bound_projection(
        record: IntegerProjectionRecord,
        semantic_object: ObjectIdentity,
        proposition: ObjectIdentity,
        evidence_history: tuple[EvidenceRecord, ...],
        ) -> ArtifactQueryInput:
    """Replace only the semantic target; carrier local objects remain shared."""
    base = record.artifact
    source = base.envelope.source
    scope = base.envelope.scope
    hypothesis = HypothesisKey(
        (16619921, 1),
        proposition.stable_key(),
        (16619922, record.carrier_code, record.sample_code),
        scope,
        source,
    )
    evidence = tuple(sorted((EvidenceRecord(
        item.evidence_id,
        hypothesis,
        item.stance,
        item.reason_key,
        item.source,
        item.timestamp_seq,
        item.payload,
        item.supersedes_evidence_id,
    ) for item in evidence_history), key=EvidenceRecord.stable_key))
    if not evidence:
        raise SourceWideProjectionError("MODEL_BOUND projection 缺少 Evidence")
    old = base.projections[0]
    projection = make_artifact_semantic_projection(
        envelope_identity=base.envelope.identity,
        source=source,
        scope=scope,
        anchor_identities=old.anchor_identities,
        structure_node_identities=old.structure_node_identities,
        projection_kind=concept_identity((16619923, 1)),
        semantic_object=semantic_object,
        lifecycle_state=concept_identity((16619924, 1)),
        hypothesis=hypothesis,
        evidence=evidence,
        directions=old.directions,
        projection_key=(16619925, record.carrier_code, record.sample_code),
    )
    return ArtifactQueryInput(
        base.envelope,
        base.anchors,
        base.structure_nodes,
        (projection,),
        base.references,
    )


def compile_model_bound_course(
        repository_root: str | Path,
        source_course: str | Path,
        model_database: str | Path,
        run_root: str | Path,
        ) -> dict[str, object]:
    """Bind only source-wide rows proven by the read-only trained graph."""
    repository = Path(repository_root).resolve()
    root = Path(run_root).resolve()
    if root.drive.upper() != "K:" or root.exists():
        raise SourceWideProjectionError("model-bound run root 必须是新的 K 盘目录")
    root.mkdir(parents=True)
    base_records = read_source_wide_course(source_course)
    source_rows = {
        carrier: _read_jsonl((repository / SAMPLE_PATHS[carrier - 1]).resolve())
        for carrier in range(1, 10)
    }
    bound_rows: list[dict[str, object]] = []
    bound_records: list[IntegerProjectionRecord] = []
    reason_counts = {1: 0, 2: 0, 3: 0}
    with TrainedGraphQueryBridge(Path(model_database).resolve()) as bridge:
        for record in base_records:
            row = source_rows[record.carrier_code][record.sample_code - 1]
            raw_text = row.get("raw_text")
            if not isinstance(raw_text, str) or not raw_text:
                raise SourceWideProjectionError("MODEL_BOUND source raw_text 缺失")
            _verify_raw_identity(row, raw_text)
            structure = bridge.input_projector.project(tuple(ord(item) for item in raw_text))
            fillers = tuple(sorted({
                tuple(item.candidate_key)
                for item in structure.semantic_candidates
                if item.projection_kind == 1 and not item.concept_route_key
            }))
            candidates: list[tuple[ObjectIdentity, ObjectIdentity]] = []
            for filler_key in fillers:
                filler = ObjectIdentity.from_stable_key(filler_key)
                for relation in structure.relation_candidates:
                    fact = bridge.core_fact_index.get(tuple(relation.proposition_key))
                    if fact is None or not any(
                            binding.filler == filler for binding in fact.bindings):
                        continue
                    candidates.append((filler, fact.proposition))
            candidates = sorted(set(candidates), key=lambda item: (item[0].stable_key(), item[1].stable_key()))
            binding_state = MODEL_BOUND_STATE_UNBOUND
            reason = 1
            bound_input = record.artifact
            proposition_key: tuple[int, ...] = ()
            evidence_keys: tuple[tuple[int, ...], ...] = ()
            if len(candidates) == 1:
                semantic_object, proposition = candidates[0]
                history = bridge.core_runtime.evidence_history(proposition)
                support_sources = {item.source.stable_key() for item in history if item.stance == 1}
                frame_present = proposition.stable_key() in bridge.dialogue_frame_index
                if len(support_sources) >= 2 and frame_present:
                    bound_input = _model_bound_projection(
                        record, semantic_object, proposition, history)
                    binding_state = MODEL_BOUND_STATE_BOUND
                    reason = 0
                    proposition_key = proposition.stable_key()
                    evidence_keys = tuple(item.stable_key() for item in history)
                    bound_records.append(IntegerProjectionRecord(
                        record.carrier_code,
                        record.sample_code,
                        {
                            **record.metadata,
                            "binding_reason": 0,
                            "binding_state": MODEL_BOUND_STATE_BOUND,
                            "bound_proposition": list(proposition_key),
                            "semantic_binding_state": SEMANTIC_BINDING_MODEL_ACTIVE,
                        },
                        bound_input,
                    ))
                else:
                    reason = 3
            elif len(candidates) > 1:
                reason = 2
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            bound_rows.append({
                "artifact_input": bound_input.integer_trace(),
                "base_natural_key": list(_natural_key(record.artifact)),
                "binding_reason": reason,
                "binding_state": binding_state,
                "bound_proposition": list(proposition_key),
                "carrier_code": record.carrier_code,
                "evidence_keys": [list(item) for item in evidence_keys],
                "format_version": FORMAT_VERSION,
                "record_kind": MODEL_BOUND_COURSE_KIND,
                "sample_code": record.sample_code,
            })
    course_path = root / "model_bound_projection.course.jsonl"
    with course_path.open("x", encoding="ascii", newline="\n") as stream:
        for row in bound_rows:
            _integer_tree(row, where="model-bound row")
            stream.write(_canonical_bytes(row).decode("ascii"))
            stream.write("\n")
    graph = None
    if bound_records:
        graph = materialize_projection_graph(
            tuple(bound_records), root / "model_bound_projection_graph.sqlite3")
    manifest = {
        "bound_count": len(bound_records),
        "carrier_codes": list(range(1, 10)),
        "course_kind": MODEL_BOUND_COURSE_KIND,
        "format_version": FORMAT_VERSION,
        "record_count": len(bound_rows),
        "reason_counts": {str(key): value for key, value in sorted(reason_counts.items())},
        "unbound_count": len(bound_rows) - len(bound_records),
    }
    _integer_tree(manifest, where="model-bound manifest")
    (root / "model_bound_projection.manifest.json").write_bytes(
        _canonical_bytes(manifest) + b"\n")
    receipt = {
        "course": str(course_path),
        "bound_count": len(bound_records),
        "unbound_count": len(bound_rows) - len(bound_records),
        "graph": graph,
        "manifest": str(root / "model_bound_projection.manifest.json"),
        "free_dialogue_complete": 0,
    }
    (root / "receipt.json").write_bytes(_canonical_bytes(receipt) + b"\n")
    return receipt


def consume_course_query(
        database: str | Path,
        records: tuple[IntegerProjectionRecord, ...],
        run_root: str | Path,
        *,
        session_id: int,
        ) -> dict[str, object]:
    """Consume one representative of every carrier through the existing bridge."""
    root = Path(run_root).resolve()
    session = root / "interaction_memory.sqlite3"
    selected = tuple(
        next(item for item in records if item.carrier_code == carrier)
        for carrier in range(1, 10)
    )
    traces: list[dict[str, object]] = []
    with TrainedGraphQueryBridge(
            Path(database).resolve(), memory_database=session,
            tenant_id=1, user_id=1, session_id=session_id) as bridge:
        for ordinal, record in enumerate(selected, 1):
            appended = bridge.memory.append_artifact(record.artifact, speaker_kind=1)
            observation_ref = bridge.memory.intake.result_for_source(appended.source).observation_ref
            trace = bridge.query_artifact(
                record.artifact,
                max_depth=64,
                input_observation_ref=observation_ref,
            )
            spaces = tuple(sorted({int(item["space"]) for item in trace["roots"]}))
            evidence_spaces = tuple(sorted({int(item["space"]) for item in trace["evidence"]}))
            if spaces != (1, 2, 3) or evidence_spaces != (1, 2, 3):
                raise SourceWideProjectionError("同一 QueryState 未恢复三图 roots/Evidence")
            bridge_trace = trace.get("artifact_bridge")
            if not isinstance(bridge_trace, dict):
                raise SourceWideProjectionError("QueryState trace 缺少 artifact semantic bridge")
            matched = bridge_trace.get("matched")
            projection_keys = bridge_trace.get("projection_keys")
            missing = bridge_trace.get("missing")
            expected = tuple(item.identity.stable_key()
                             for item in record.artifact.understanding_projections)
            if (not isinstance(matched, list)
                    or not isinstance(projection_keys, list)
                    or not isinstance(missing, list)
                    or len(matched) != len(expected)
                    or {tuple(item) for item in projection_keys} != set(expected)
                    or missing):
                raise SourceWideProjectionError(
                    "carrier projection 没有通过完整 bridge identity 连接既有 Core")
            traces.append({
                "carrier_code": record.carrier_code,
                "depth": int(trace["depth"]),
                "evidence_spaces": list(evidence_spaces),
                "root_spaces": list(spaces),
                "termination": trace["termination_reason"],
                "artifact_bridge_match_count": len(matched),
                "artifact_bridge_missing_count": len(missing),
            })
    query_path = root / "source_wide_query.receipt.json"
    receipt = {
        "format_version": FORMAT_VERSION,
        "query_kind": QUERY_KIND,
        "carrier_query_count": len(traces),
        "three_graph_query_count": len(traces),
        "traces": traces,
        "free_dialogue_complete": 0,
    }
    query_path.write_bytes(_canonical_bytes(receipt) + b"\n")
    return receipt


def run_source_wide_course(
        repository_root: str | Path,
        database: str | Path,
        run_root: str | Path,
        *,
        session_id: int,
        max_cases_per_carrier: int | None = None,
        ) -> dict[str, object]:
    """Compile, materialize and consume the source-wide course in one K run."""
    compiled = compile_source_wide_course(
        repository_root, run_root,
        max_cases_per_carrier=max_cases_per_carrier,
    )
    records = read_source_wide_course(compiled["course"])
    graph = materialize_projection_graph(records, Path(run_root) / "projection_graph.sqlite3")
    query = consume_course_query(database, records, run_root, session_id=session_id)
    receipt = {
        "course": compiled,
        "graph": graph,
        "query": query,
        "format_version": FORMAT_VERSION,
        "free_dialogue_complete": 0,
    }
    (Path(run_root) / "receipt.json").write_bytes(_canonical_bytes(receipt) + b"\n")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--session-id", type=int, required=True)
    parser.add_argument("--max-cases-per-carrier", type=int, default=None)
    parser.add_argument(
        "--source-course",
        type=Path,
        default=None,
        help="compile only MODEL_BOUND records from an existing integer course",
    )
    args = parser.parse_args(argv)
    result = (
        compile_model_bound_course(
            args.repository_root,
            args.source_course,
            args.database,
            args.run_root,
        )
        if args.source_course is not None
        else run_source_wide_course(
            args.repository_root,
            args.database,
            args.run_root,
            session_id=args.session_id,
            max_cases_per_carrier=args.max_cases_per_carrier,
        )
    )
    print(json.dumps(
        result, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CARRIER_CODES",
    "CARRIER_KEYS",
    "IntegerProjectionRecord",
    "LICENSE_CODES",
    "SEMANTIC_BINDING_STRUCTURE_ONLY",
    "SourceWideProjectionError",
    "audit_projection_graph",
    "compile_source_wide_course",
    "compile_model_bound_course",
    "compile_trained_fact_carrier_course",
    "consume_course_query",
    "materialize_projection_graph",
    "read_source_wide_course",
    "read_trained_fact_carrier_course",
    "run_source_wide_course",
    "TrainedFactCarrierRecord",
]
