"""从 FT16 已学注册表派生 FT17 不可变索引。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_feature_registry_contract import (
    QUESTION_FEATURE_DISPATCH_PHASES,
    RawQuestionFeatureRegistry,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_contract import (
    RawQuestionConstruction,
)


QUESTION_FEATURE_INDEX_SHA256 = (
    "908786f4da77414a5c7728d01f8c2c1528c23f4787a1efcc068b166f70be095b")
QUESTION_FEATURE_INDEX_EXPRESSION_BOUNDARY = (
    ("index_source", "LEARNED_CONSTRUCTIONS_AND_ALIAS_ROUTES"),
    ("candidate_effect", "NARROW_ONLY"),
    ("global_priority", "EXACT_THEN_ALIAS_THEN_IMPLICIT"),
    ("result_projection", "BYTE_IDENTICAL_TO_FT16_SCAN_RUNTIME"),
    ("source_binding", "OPTIONAL_SOURCE_REF_RETAINS_CATALOG_OWNERSHIP"),
    ("handwritten_dispatch", "FORBIDDEN"),
)


# object-model: exception
class W03W04W05QuestionFeatureIndexError(ValueError):
    """已学索引清单或查询边界发生漂移。"""


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256(value: object, *, where: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise W03W04W05QuestionFeatureIndexError(
            f"{where} is not a canonical SHA-256")
    return value


def _text(value: object, *, where: str) -> str:
    if (not isinstance(value, str) or not value
            or value.strip() != value):
        raise W03W04W05QuestionFeatureIndexError(
            f"{where} is not canonical text")
    return value


def _source_key(value: tuple[int, ...], *, where: str) -> tuple[int, ...]:
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int for item in value)):
        raise W03W04W05QuestionFeatureIndexError(
            f"{where} is not a strict integer source key")
    return value


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class RawQuestionFeatureIndexPosting:
    """一个已学表层和来源引用可到达的注册项。"""

    source_record_key: tuple[int, ...]
    entry_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        _source_key(
            self.source_record_key,
            where="question feature index posting SourceRef",
        )
        if (not isinstance(self.entry_sha256s, tuple)
                or not self.entry_sha256s):
            raise W03W04W05QuestionFeatureIndexError(
                "question feature index posting has no registry entry")
        for item in self.entry_sha256s:
            _sha256(item, where="question feature index posting entry")
        if (self.entry_sha256s != tuple(sorted(self.entry_sha256s))
                or len(set(self.entry_sha256s))
                != len(self.entry_sha256s)):
            raise W03W04W05QuestionFeatureIndexError(
                "question feature index posting entries are not canonical")

    def to_dict(self) -> dict[str, object]:
        return {
            "entry_sha256s": list(self.entry_sha256s),
            "source_record_key": list(self.source_record_key),
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class RawQuestionFeatureIndexRow:
    """一个已学问题表层的全部来源绑定投递项。"""

    question_surface: str
    postings: tuple[RawQuestionFeatureIndexPosting, ...]

    def __post_init__(self) -> None:
        _text(
            self.question_surface,
            where="question feature index row surface",
        )
        if (not isinstance(self.postings, tuple) or not self.postings
                or any(not isinstance(
                    item, RawQuestionFeatureIndexPosting)
                    for item in self.postings)):
            raise W03W04W05QuestionFeatureIndexError(
                "question feature index row has no canonical postings")
        if (self.postings != tuple(sorted(
                self.postings,
                key=lambda item: item.source_record_key,
                ))
                or len({item.source_record_key for item in self.postings})
                != len(self.postings)):
            raise W03W04W05QuestionFeatureIndexError(
                "question feature index row SourceRefs are not canonical")

    def to_dict(self) -> dict[str, object]:
        return {
            "postings": [item.to_dict() for item in self.postings],
            "question_surface": self.question_surface,
        }


def _frame_surface(value: object, *, where: str) -> str:
    if not isinstance(value, str) or value.strip() != value:
        raise W03W04W05QuestionFeatureIndexError(
            f"{where} is not a canonical frame surface")
    return value


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class RawQuestionFeatureAliasFrameRow:
    """一个谓词槽结构帧及其已学路由清单。"""

    prefix_surface: str
    suffix_surface: str
    predicate_surface: str
    learned_alias_surfaces: tuple[str, ...]
    postings: tuple[RawQuestionFeatureIndexPosting, ...]

    def __post_init__(self) -> None:
        _frame_surface(
            self.prefix_surface,
            where="question feature alias frame prefix",
        )
        _frame_surface(
            self.suffix_surface,
            where="question feature alias frame suffix",
        )
        _text(
            self.predicate_surface,
            where="question feature alias frame predicate",
        )
        if (not self.prefix_surface and not self.suffix_surface
                or not isinstance(self.learned_alias_surfaces, tuple)
                or not self.learned_alias_surfaces):
            raise W03W04W05QuestionFeatureIndexError(
                "question feature alias frame boundary is empty")
        for item in self.learned_alias_surfaces:
            _text(item, where="question feature alias frame learned route")
        if (self.learned_alias_surfaces
                != tuple(sorted(set(self.learned_alias_surfaces)))
                or self.predicate_surface in self.learned_alias_surfaces):
            raise W03W04W05QuestionFeatureIndexError(
                "question feature alias routes are not canonical")
        if (not isinstance(self.postings, tuple) or not self.postings
                or any(not isinstance(
                    item, RawQuestionFeatureIndexPosting)
                    for item in self.postings)
                or self.postings != tuple(sorted(
                    self.postings,
                    key=lambda item: item.source_record_key,
                ))
                or len({item.source_record_key for item in self.postings})
                != len(self.postings)):
            raise W03W04W05QuestionFeatureIndexError(
                "question feature alias frame postings are not canonical")

    def sort_key(self) -> tuple[object, ...]:
        return (
            self.prefix_surface,
            self.suffix_surface,
            self.predicate_surface,
            self.learned_alias_surfaces,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "learned_alias_surfaces": list(self.learned_alias_surfaces),
            "postings": [item.to_dict() for item in self.postings],
            "predicate_surface": self.predicate_surface,
            "prefix_surface": self.prefix_surface,
            "suffix_surface": self.suffix_surface,
        }


def _validate_rows(
        rows: tuple[RawQuestionFeatureIndexRow, ...],
        *,
        phase: str,
        entry_sha256s: set[str],
        ) -> None:
    if (phase not in QUESTION_FEATURE_DISPATCH_PHASES
            or not isinstance(rows, tuple) or not rows
            or any(not isinstance(item, RawQuestionFeatureIndexRow)
                   for item in rows)):
        raise W03W04W05QuestionFeatureIndexError(
            f"question feature {phase} index rows are invalid")
    if (rows != tuple(sorted(rows, key=lambda item: item.question_surface))
            or len({item.question_surface for item in rows}) != len(rows)):
        raise W03W04W05QuestionFeatureIndexError(
            f"question feature {phase} index rows are not canonical")
    published = {
        entry_sha256
        for row in rows
        for posting in row.postings
        for entry_sha256 in posting.entry_sha256s
    }
    if not published.issubset(entry_sha256s):
        raise W03W04W05QuestionFeatureIndexError(
            f"question feature {phase} index escaped its registry")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class RawQuestionFeatureIndex:
    """一个冻结 FT16 注册表的精确、别名与隐式查询行。"""

    registry: RawQuestionFeatureRegistry
    exact_rows: tuple[RawQuestionFeatureIndexRow, ...]
    alias_rows: tuple[RawQuestionFeatureIndexRow, ...]
    alias_frame_rows: tuple[RawQuestionFeatureAliasFrameRow, ...]
    implicit_rows: tuple[RawQuestionFeatureIndexRow, ...]
    identity_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.registry, RawQuestionFeatureRegistry):
            raise TypeError("question feature index registry is invalid")
        entry_ids = {item.sha256() for item in self.registry.entries}
        for phase, rows in zip(
                QUESTION_FEATURE_DISPATCH_PHASES,
                (self.exact_rows, self.alias_rows, self.implicit_rows)):
            _validate_rows(rows, phase=phase, entry_sha256s=entry_ids)
        if (not isinstance(self.alias_frame_rows, tuple)
                or not self.alias_frame_rows
                or any(not isinstance(
                    item, RawQuestionFeatureAliasFrameRow)
                    for item in self.alias_frame_rows)
                or self.alias_frame_rows != tuple(sorted(
                    self.alias_frame_rows,
                    key=RawQuestionFeatureAliasFrameRow.sort_key,
                ))
                or len({item.sort_key() for item in self.alias_frame_rows})
                != len(self.alias_frame_rows)):
            raise W03W04W05QuestionFeatureIndexError(
                "question feature alias frames are not canonical")
        frame_entries = {
            entry_sha
            for row in self.alias_frame_rows
            for posting in row.postings
            for entry_sha in posting.entry_sha256s
        }
        if not frame_entries.issubset(entry_ids):
            raise W03W04W05QuestionFeatureIndexError(
                "question feature alias frames escaped their registry")
        _sha256(self.identity_sha256, where="question feature index")
        if self.identity_sha256 != self.sha256():
            raise W03W04W05QuestionFeatureIndexError(
                "question feature index identity drifted")

    def rows_at(
            self,
            phase: str,
            ) -> tuple[RawQuestionFeatureIndexRow, ...]:
        if phase == "EXACT":
            return self.exact_rows
        if phase == "ALIAS":
            return self.alias_rows
        if phase == "IMPLICIT":
            return self.implicit_rows
        raise W03W04W05QuestionFeatureIndexError(
            "question feature index phase is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "alias_frame_rows": [
                item.to_dict() for item in self.alias_frame_rows],
            "alias_rows": [item.to_dict() for item in self.alias_rows],
            "exact_rows": [item.to_dict() for item in self.exact_rows],
            "expression_boundary": [
                {"capability": key, "status": status}
                for key, status in QUESTION_FEATURE_INDEX_EXPRESSION_BOUNDARY
            ],
            "implicit_rows": [item.to_dict() for item in self.implicit_rows],
            "phase_priority": list(QUESTION_FEATURE_DISPATCH_PHASES),
            "registry_identity_sha256": self.registry.identity_sha256,
        }

    def sha256(self) -> str:
        return _sha(self.to_dict())


def _construction_surface(construction: RawQuestionConstruction) -> str:
    surface = "".join(item.surface for item in construction.segments)
    if surface != construction.question_surface:
        raise W03W04W05QuestionFeatureIndexError(
            "question feature index construction surface drifted")
    return surface


def _record(
        values: dict[str, dict[tuple[int, ...], set[str]]],
        *,
        surface: str,
        source_record_key: tuple[int, ...],
        entry_sha256: str,
        ) -> None:
    values.setdefault(surface, {}).setdefault(
        source_record_key, set()).add(entry_sha256)


def _rows(
        values: dict[str, dict[tuple[int, ...], set[str]]],
        ) -> tuple[RawQuestionFeatureIndexRow, ...]:
    return tuple(
        RawQuestionFeatureIndexRow(
            surface,
            tuple(
                RawQuestionFeatureIndexPosting(
                    source,
                    tuple(sorted(entry_ids)),
                )
                for source, entry_ids in sorted(by_source.items())
            ),
        )
        for surface, by_source in sorted(values.items())
    )


def _alias_frame_rows(
        values: dict[
            tuple[str, str, str, tuple[str, ...]],
            dict[tuple[int, ...], set[str]],
        ],
        ) -> tuple[RawQuestionFeatureAliasFrameRow, ...]:
    return tuple(
        RawQuestionFeatureAliasFrameRow(
            frame[0],
            frame[1],
            frame[2],
            frame[3],
            tuple(
                RawQuestionFeatureIndexPosting(
                    source,
                    tuple(sorted(entry_ids)),
                )
                for source, entry_ids in sorted(by_source.items())
            ),
        )
        for frame, by_source in sorted(values.items())
    )


def build_raw_question_feature_index(
        registry: RawQuestionFeatureRegistry,
        *,
        expected_identity_sha256: str | None = None,
        ) -> RawQuestionFeatureIndex:
    """从冻结的已学结构与路由派生全部查询行。"""
    if not isinstance(registry, RawQuestionFeatureRegistry):
        raise TypeError("question feature index input is invalid")
    exact: dict[str, dict[tuple[int, ...], set[str]]] = {}
    alias: dict[str, dict[tuple[int, ...], set[str]]] = {}
    alias_frames: dict[
        tuple[str, str, str, tuple[str, ...]],
        dict[tuple[int, ...], set[str]],
    ] = {}
    implicit: dict[str, dict[tuple[int, ...], set[str]]] = {}
    for entry in registry.entries:
        entry_sha = entry.sha256()
        for construction in entry.feature_catalog.catalog:
            surface = _construction_surface(construction)
            _record(
                exact,
                surface=surface,
                source_record_key=construction.source_record_key,
                entry_sha256=entry_sha,
            )
            predicate_ordinals = tuple(
                ordinal for ordinal, segment in enumerate(
                    construction.segments)
                if segment.kind == "PREDICATE"
            )
            if len(predicate_ordinals) != 1:
                raise W03W04W05QuestionFeatureIndexError(
                    "alias index requires one learned predicate segment")
            predicate_ordinal = predicate_ordinals[0]
            learned_aliases = tuple(sorted({
                route.alias_surface for route in entry.alias_bridge.routes
                if route.alias_surface
                != construction.segments[predicate_ordinal].surface
            }))
            prefix = "".join(
                item.surface
                for item in construction.segments[:predicate_ordinal])
            suffix = "".join(
                item.surface
                for item in construction.segments[predicate_ordinal + 1:])
            frame = (
                prefix,
                suffix,
                construction.segments[predicate_ordinal].surface,
                learned_aliases,
            )
            alias_frames.setdefault(frame, {}).setdefault(
                construction.source_record_key, set()).add(entry_sha)
            for alias_surface in learned_aliases:
                learned_surface = "".join(
                    alias_surface if ordinal == predicate_ordinal
                    else segment.surface
                    for ordinal, segment in enumerate(construction.segments)
                )
                if learned_surface == surface:
                    continue
                _record(
                    alias,
                    surface=learned_surface,
                    source_record_key=construction.source_record_key,
                    entry_sha256=entry_sha,
                )
        for construction in entry.implicit_bundle.catalog:
            _record(
                implicit,
                surface=_construction_surface(construction),
                source_record_key=construction.source_record_key,
                entry_sha256=entry_sha,
            )
    exact_rows = _rows(exact)
    alias_rows = _rows(alias)
    frame_rows = _alias_frame_rows(alias_frames)
    implicit_rows = _rows(implicit)
    payload = {
        "alias_frame_rows": [item.to_dict() for item in frame_rows],
        "alias_rows": [item.to_dict() for item in alias_rows],
        "exact_rows": [item.to_dict() for item in exact_rows],
        "expression_boundary": [
            {"capability": key, "status": status}
            for key, status in QUESTION_FEATURE_INDEX_EXPRESSION_BOUNDARY
        ],
        "implicit_rows": [item.to_dict() for item in implicit_rows],
        "phase_priority": list(QUESTION_FEATURE_DISPATCH_PHASES),
        "registry_identity_sha256": registry.identity_sha256,
    }
    identity = _sha(payload)
    value = RawQuestionFeatureIndex(
        registry,
        exact_rows,
        alias_rows,
        frame_rows,
        implicit_rows,
        identity,
    )
    if (expected_identity_sha256 is not None
            and identity != _sha256(
                expected_identity_sha256,
                where="expected question feature index")):
        raise W03W04W05QuestionFeatureIndexError(
            "question feature index commitment drifted")
    return value


__all__ = [
    "QUESTION_FEATURE_INDEX_EXPRESSION_BOUNDARY",
    "QUESTION_FEATURE_INDEX_SHA256",
    "RawQuestionFeatureAliasFrameRow",
    "RawQuestionFeatureIndex",
    "RawQuestionFeatureIndexPosting",
    "RawQuestionFeatureIndexRow",
    "W03W04W05QuestionFeatureIndexError",
    "build_raw_question_feature_index",
]
