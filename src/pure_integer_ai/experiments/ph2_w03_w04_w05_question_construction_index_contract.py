"""从 FT17 注册项索引派生 FT18 构式候选索引。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_feature_index_contract import (
    RawQuestionFeatureIndex,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_feature_registry_contract import (
    QUESTION_FEATURE_DISPATCH_PHASES,
    RawQuestionFeatureRegistry,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_contract import (
    RawQuestionConstruction,
)


QUESTION_CONSTRUCTION_INDEX_SHA256 = (
    "0d4495b612e5d6e54fa47f1066ad48543fd5bbb00ffc08d04a67de2fc2067ab8")
QUESTION_CONSTRUCTION_INDEX_EXPRESSION_BOUNDARY = (
    ("index_source", "FT17_INDEX_AND_LEARNED_CONSTRUCTIONS"),
    ("posting_identity", "REGISTRY_ENTRY_AND_CONSTRUCTION_SHA256"),
    ("candidate_effect", "NARROW_ENTRY_AND_INTERNAL_CONSTRUCTION"),
    ("global_priority", "EXACT_THEN_ALIAS_THEN_IMPLICIT"),
    ("result_projection", "BYTE_IDENTICAL_TO_FT16_SCAN_RUNTIME"),
    ("normalization", "ALIAS_REUSES_INDEXED_EXACT_CONSTRUCTIONS"),
    ("handwritten_dispatch", "FORBIDDEN"),
)


# object-model: exception
class W03W04W05QuestionConstructionIndexError(ValueError):
    """构式候选索引或其目录所有权发生漂移。"""


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256(value: object, *, where: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise W03W04W05QuestionConstructionIndexError(
            f"{where} is not a canonical SHA-256")
    return value


def _text(value: object, *, where: str) -> str:
    if (not isinstance(value, str) or not value
            or value.strip() != value):
        raise W03W04W05QuestionConstructionIndexError(
            f"{where} is not canonical text")
    return value


def _frame_surface(value: object, *, where: str) -> str:
    if not isinstance(value, str) or value.strip() != value:
        raise W03W04W05QuestionConstructionIndexError(
            f"{where} is not a canonical frame surface")
    return value


def _source_key(value: tuple[int, ...], *, where: str) -> tuple[int, ...]:
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int for item in value)):
        raise W03W04W05QuestionConstructionIndexError(
            f"{where} is not a strict integer source key")
    return value


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class RawQuestionConstructionCandidateSet:
    """一个注册项内的规范构式候选。"""

    entry_sha256: str
    constructions: tuple[RawQuestionConstruction, ...]

    def __post_init__(self) -> None:
        _sha256(self.entry_sha256, where="construction candidate entry")
        if (not isinstance(self.constructions, tuple)
                or not self.constructions
                or any(not isinstance(item, RawQuestionConstruction)
                       for item in self.constructions)):
            raise W03W04W05QuestionConstructionIndexError(
                "construction candidate set is empty or invalid")
        identities = self.construction_sha256s
        if (identities != tuple(sorted(identities))
                or len(set(identities)) != len(identities)):
            raise W03W04W05QuestionConstructionIndexError(
                "construction candidate identities are not canonical")

    @property
    def construction_sha256s(self) -> tuple[str, ...]:
        return tuple(item.sha256() for item in self.constructions)

    def to_dict(self) -> dict[str, object]:
        return {
            "construction_sha256s": list(self.construction_sha256s),
            "entry_sha256": self.entry_sha256,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class RawQuestionConstructionIndexPosting:
    """一个来源引用下的注册项与构式候选。"""

    source_record_key: tuple[int, ...]
    candidates: tuple[RawQuestionConstructionCandidateSet, ...]

    def __post_init__(self) -> None:
        _source_key(
            self.source_record_key,
            where="construction index posting SourceRef",
        )
        if (not isinstance(self.candidates, tuple)
                or not self.candidates
                or any(not isinstance(
                    item, RawQuestionConstructionCandidateSet)
                    for item in self.candidates)
                or self.candidates != tuple(sorted(
                    self.candidates,
                    key=lambda item: item.entry_sha256,
                ))
                or len({item.entry_sha256 for item in self.candidates})
                != len(self.candidates)):
            raise W03W04W05QuestionConstructionIndexError(
                "construction index candidates are not canonical")

    def to_dict(self) -> dict[str, object]:
        return {
            "candidates": [item.to_dict() for item in self.candidates],
            "source_record_key": list(self.source_record_key),
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class RawQuestionConstructionIndexRow:
    """一个问题表层的全部来源绑定构式投递项。"""

    question_surface: str
    postings: tuple[RawQuestionConstructionIndexPosting, ...]

    def __post_init__(self) -> None:
        _text(self.question_surface, where="construction index row surface")
        if (not isinstance(self.postings, tuple) or not self.postings
                or any(not isinstance(
                    item, RawQuestionConstructionIndexPosting)
                    for item in self.postings)
                or self.postings != tuple(sorted(
                    self.postings,
                    key=lambda item: item.source_record_key,
                ))
                or len({item.source_record_key for item in self.postings})
                != len(self.postings)):
            raise W03W04W05QuestionConstructionIndexError(
                "construction index row postings are not canonical")

    def to_dict(self) -> dict[str, object]:
        return {
            "postings": [item.to_dict() for item in self.postings],
            "question_surface": self.question_surface,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class RawQuestionConstructionAliasFrameRow:
    """一个谓词槽结构帧及其构式候选。"""

    prefix_surface: str
    suffix_surface: str
    predicate_surface: str
    learned_alias_surfaces: tuple[str, ...]
    postings: tuple[RawQuestionConstructionIndexPosting, ...]

    def __post_init__(self) -> None:
        _frame_surface(
            self.prefix_surface,
            where="construction alias frame prefix",
        )
        _frame_surface(
            self.suffix_surface,
            where="construction alias frame suffix",
        )
        _text(
            self.predicate_surface,
            where="construction alias frame predicate",
        )
        if (not self.prefix_surface and not self.suffix_surface
                or not isinstance(self.learned_alias_surfaces, tuple)
                or not self.learned_alias_surfaces
                or self.learned_alias_surfaces
                != tuple(sorted(set(self.learned_alias_surfaces)))
                or self.predicate_surface in self.learned_alias_surfaces):
            raise W03W04W05QuestionConstructionIndexError(
                "construction alias frame routes are not canonical")
        for item in self.learned_alias_surfaces:
            _text(item, where="construction alias learned route")
        if (not isinstance(self.postings, tuple) or not self.postings
                or any(not isinstance(
                    item, RawQuestionConstructionIndexPosting)
                    for item in self.postings)
                or self.postings != tuple(sorted(
                    self.postings,
                    key=lambda item: item.source_record_key,
                ))
                or len({item.source_record_key for item in self.postings})
                != len(self.postings)):
            raise W03W04W05QuestionConstructionIndexError(
                "construction alias frame postings are not canonical")

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


def _candidate_sets(rows) -> tuple[RawQuestionConstructionCandidateSet, ...]:
    return tuple(
        candidates
        for row in rows
        for posting in row.postings
        for candidates in posting.candidates
    )


def _validate_rows(
        rows: tuple[RawQuestionConstructionIndexRow, ...],
        *,
        phase: str,
        allowed: dict[str, set[str]],
        ) -> None:
    if (phase not in QUESTION_FEATURE_DISPATCH_PHASES
            or not isinstance(rows, tuple) or not rows
            or any(not isinstance(item, RawQuestionConstructionIndexRow)
                   for item in rows)
            or rows != tuple(sorted(
                rows, key=lambda item: item.question_surface))
            or len({item.question_surface for item in rows}) != len(rows)):
        raise W03W04W05QuestionConstructionIndexError(
            f"construction {phase} index rows are not canonical")
    for item in _candidate_sets(rows):
        if (item.entry_sha256 not in allowed
                or not set(item.construction_sha256s).issubset(
                    allowed[item.entry_sha256])):
            raise W03W04W05QuestionConstructionIndexError(
                f"construction {phase} index escaped its learned catalog")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class RawQuestionConstructionIndex:
    """FT17 注册项索引上的精确、别名与隐式构式候选。"""

    feature_index: RawQuestionFeatureIndex
    exact_rows: tuple[RawQuestionConstructionIndexRow, ...]
    alias_rows: tuple[RawQuestionConstructionIndexRow, ...]
    alias_frame_rows: tuple[RawQuestionConstructionAliasFrameRow, ...]
    implicit_rows: tuple[RawQuestionConstructionIndexRow, ...]
    identity_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.feature_index, RawQuestionFeatureIndex):
            raise TypeError("construction index feature index is invalid")
        explicit = {
            entry.sha256(): {
                item.sha256() for item in entry.feature_catalog.catalog}
            for entry in self.registry.entries
        }
        implicit = {
            entry.sha256(): {
                item.sha256() for item in entry.implicit_bundle.catalog}
            for entry in self.registry.entries
        }
        _validate_rows(self.exact_rows, phase="EXACT", allowed=explicit)
        _validate_rows(self.alias_rows, phase="ALIAS", allowed=explicit)
        _validate_rows(self.implicit_rows, phase="IMPLICIT", allowed=implicit)
        if (not isinstance(self.alias_frame_rows, tuple)
                or not self.alias_frame_rows
                or any(not isinstance(
                    item, RawQuestionConstructionAliasFrameRow)
                    for item in self.alias_frame_rows)
                or self.alias_frame_rows != tuple(sorted(
                    self.alias_frame_rows,
                    key=RawQuestionConstructionAliasFrameRow.sort_key,
                ))
                or len({item.sort_key() for item in self.alias_frame_rows})
                != len(self.alias_frame_rows)):
            raise W03W04W05QuestionConstructionIndexError(
                "construction alias frames are not canonical")
        for item in _candidate_sets(self.alias_frame_rows):
            if (item.entry_sha256 not in explicit
                    or not set(item.construction_sha256s).issubset(
                        explicit[item.entry_sha256])):
                raise W03W04W05QuestionConstructionIndexError(
                    "construction alias frame escaped its learned catalog")
        _sha256(self.identity_sha256, where="question construction index")
        if self.identity_sha256 != self.sha256():
            raise W03W04W05QuestionConstructionIndexError(
                "question construction index identity drifted")

    @property
    def registry(self) -> RawQuestionFeatureRegistry:
        return self.feature_index.registry

    def rows_at(
            self,
            phase: str,
            ) -> tuple[RawQuestionConstructionIndexRow, ...]:
        if phase == "EXACT":
            return self.exact_rows
        if phase == "ALIAS":
            return self.alias_rows
        if phase == "IMPLICIT":
            return self.implicit_rows
        raise W03W04W05QuestionConstructionIndexError(
            "question construction index phase is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "alias_frame_rows": [
                item.to_dict() for item in self.alias_frame_rows],
            "alias_rows": [item.to_dict() for item in self.alias_rows],
            "exact_rows": [item.to_dict() for item in self.exact_rows],
            "expression_boundary": [
                {"capability": key, "status": status}
                for key, status in (
                    QUESTION_CONSTRUCTION_INDEX_EXPRESSION_BOUNDARY)
            ],
            "feature_index_identity_sha256": (
                self.feature_index.identity_sha256),
            "implicit_rows": [item.to_dict() for item in self.implicit_rows],
            "phase_priority": list(QUESTION_FEATURE_DISPATCH_PHASES),
            "registry_identity_sha256": self.registry.identity_sha256,
        }

    def sha256(self) -> str:
        return _sha(self.to_dict())


def _record(
        values: dict[
            str,
            dict[
                tuple[int, ...],
                dict[str, dict[str, RawQuestionConstruction]],
            ],
        ],
        *,
        surface: str,
        source_record_key: tuple[int, ...],
        entry_sha256: str,
        construction: RawQuestionConstruction,
        ) -> None:
    values.setdefault(surface, {}).setdefault(
        source_record_key, {}).setdefault(entry_sha256, {})[
            construction.sha256()] = construction


def _candidate_tuple(
        by_entry: dict[str, dict[str, RawQuestionConstruction]],
        ) -> tuple[RawQuestionConstructionCandidateSet, ...]:
    return tuple(
        RawQuestionConstructionCandidateSet(
            entry_sha,
            tuple(value[key] for key in sorted(value)),
        )
        for entry_sha, value in sorted(by_entry.items())
    )


def _postings(
        by_source: dict[
            tuple[int, ...],
            dict[str, dict[str, RawQuestionConstruction]],
        ],
        ) -> tuple[RawQuestionConstructionIndexPosting, ...]:
    return tuple(
        RawQuestionConstructionIndexPosting(
            source,
            _candidate_tuple(by_entry),
        )
        for source, by_entry in sorted(by_source.items())
    )


def _rows(values) -> tuple[RawQuestionConstructionIndexRow, ...]:
    return tuple(
        RawQuestionConstructionIndexRow(surface, _postings(by_source))
        for surface, by_source in sorted(values.items())
    )


def _alias_frame_rows(values) -> tuple[RawQuestionConstructionAliasFrameRow, ...]:
    return tuple(
        RawQuestionConstructionAliasFrameRow(
            frame[0],
            frame[1],
            frame[2],
            frame[3],
            _postings(by_source),
        )
        for frame, by_source in sorted(values.items())
    )


def _construction_surface(construction: RawQuestionConstruction) -> str:
    surface = "".join(item.surface for item in construction.segments)
    if surface != construction.question_surface:
        raise W03W04W05QuestionConstructionIndexError(
            "construction index surface drifted")
    return surface


def build_raw_question_construction_index(
        feature_index: RawQuestionFeatureIndex,
        *,
        expected_identity_sha256: str | None = None,
        ) -> RawQuestionConstructionIndex:
    """从 FT17 及其冻结已学目录派生构式级投递项。"""
    if not isinstance(feature_index, RawQuestionFeatureIndex):
        raise TypeError("question construction index input is invalid")
    exact = {}
    alias = {}
    alias_frames = {}
    implicit = {}
    for entry in feature_index.registry.entries:
        entry_sha = entry.sha256()
        for construction in entry.feature_catalog.catalog:
            surface = _construction_surface(construction)
            _record(
                exact,
                surface=surface,
                source_record_key=construction.source_record_key,
                entry_sha256=entry_sha,
                construction=construction,
            )
            predicate_ordinals = tuple(
                ordinal for ordinal, segment in enumerate(
                    construction.segments)
                if segment.kind == "PREDICATE"
            )
            if len(predicate_ordinals) != 1:
                raise W03W04W05QuestionConstructionIndexError(
                    "alias construction index requires one predicate segment")
            predicate_ordinal = predicate_ordinals[0]
            predicate_surface = (
                construction.segments[predicate_ordinal].surface)
            learned_aliases = tuple(sorted({
                route.alias_surface for route in entry.alias_bridge.routes
                if route.alias_surface != predicate_surface
            }))
            prefix = "".join(
                item.surface
                for item in construction.segments[:predicate_ordinal])
            suffix = "".join(
                item.surface
                for item in construction.segments[predicate_ordinal + 1:])
            frame = (prefix, suffix, predicate_surface, learned_aliases)
            alias_frames.setdefault(frame, {}).setdefault(
                construction.source_record_key, {}).setdefault(
                    entry_sha, {})[construction.sha256()] = construction
            for alias_surface in learned_aliases:
                learned_surface = "".join(
                    alias_surface if ordinal == predicate_ordinal
                    else segment.surface
                    for ordinal, segment in enumerate(construction.segments)
                )
                if learned_surface != surface:
                    _record(
                        alias,
                        surface=learned_surface,
                        source_record_key=construction.source_record_key,
                        entry_sha256=entry_sha,
                        construction=construction,
                    )
        for construction in entry.implicit_bundle.catalog:
            _record(
                implicit,
                surface=_construction_surface(construction),
                source_record_key=construction.source_record_key,
                entry_sha256=entry_sha,
                construction=construction,
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
            for key, status in QUESTION_CONSTRUCTION_INDEX_EXPRESSION_BOUNDARY
        ],
        "feature_index_identity_sha256": feature_index.identity_sha256,
        "implicit_rows": [item.to_dict() for item in implicit_rows],
        "phase_priority": list(QUESTION_FEATURE_DISPATCH_PHASES),
        "registry_identity_sha256": feature_index.registry.identity_sha256,
    }
    identity = _sha(payload)
    value = RawQuestionConstructionIndex(
        feature_index,
        exact_rows,
        alias_rows,
        frame_rows,
        implicit_rows,
        identity,
    )
    if (expected_identity_sha256 is not None
            and identity != _sha256(
                expected_identity_sha256,
                where="expected question construction index")):
        raise W03W04W05QuestionConstructionIndexError(
            "question construction index commitment drifted")
    return value


__all__ = [
    "QUESTION_CONSTRUCTION_INDEX_EXPRESSION_BOUNDARY",
    "QUESTION_CONSTRUCTION_INDEX_SHA256",
    "RawQuestionConstructionAliasFrameRow",
    "RawQuestionConstructionCandidateSet",
    "RawQuestionConstructionIndex",
    "RawQuestionConstructionIndexPosting",
    "RawQuestionConstructionIndexRow",
    "W03W04W05QuestionConstructionIndexError",
    "build_raw_question_construction_index",
]
