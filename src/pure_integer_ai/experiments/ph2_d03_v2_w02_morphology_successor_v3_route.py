"""Append-only source-capability routing for frozen W-02 morphology overlays."""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
import hashlib
from typing import Iterable

from pure_integer_ai.experiments.ph2_d03_contract_core import canonical_json_bytes
from pure_integer_ai.experiments.ph2_d03_v2_w02_candidate_model import (
    W02CandidatePrediction,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor import (
    W02MorphologyRankingCache,
    W02MorphologySuccessorIndex,
    W02MorphologySuccessorPrediction,
    predict_w02_morphology_successor,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v2 import (
    W02MorphologySuccessorV2Cache,
    W02MorphologySuccessorV2Index,
    W02MorphologySuccessorV2Prediction,
    predict_w02_morphology_successor_v2,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    ObservationRecord,
    SourceRefRecord,
)


W02_MORPH_SUCCESSOR_V3_ROUTE_VERSION = (
    "PH2-D03-V2-W02-MORPHOLOGY-SUCCESSOR-V3-SOURCE-ROUTE-V1"
)
W02_MORPH_V3_ANNOTATION_STANDARD = "UNIVERSAL_DEPENDENCIES"
W02_MORPH_V3_ANNOTATION_SCOPE = (
    "FEATS", "LEMMA", "TOKEN_BOUNDARY", "UPOS",
)

RouteIdentity = tuple[tuple[int, ...], tuple[int, ...], str]


# object-model: exception
class W02MorphologySuccessorV3RouteError(RuntimeError):
    """A source capability, route identity, or frozen parent drifted."""


def _hash_value(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _nonempty(value: object, *, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise W02MorphologySuccessorV3RouteError(f"{where} must be non-empty text")
    return value


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W02MorphologySourceCapability:
    """Student-visible provenance declaring a source's annotation capability."""

    source_key: str
    snapshot_id: str
    revision_id: str
    official_url: str
    upstream_checksum: str
    license_id: str
    language: str
    annotation_standard: str
    annotation_scope: tuple[str, ...]
    annotation_provenance: str

    def __post_init__(self) -> None:
        for name in (
                "source_key", "official_url", "upstream_checksum", "license_id", "language",
                "annotation_standard", "annotation_provenance"):
            _nonempty(getattr(self, name), where=f"source capability {name}")
        if not self.snapshot_id and not self.revision_id:
            raise W02MorphologySuccessorV3RouteError(
                "source capability snapshot/revision is empty")
        if self.annotation_standard != W02_MORPH_V3_ANNOTATION_STANDARD:
            raise W02MorphologySuccessorV3RouteError(
                "source capability annotation standard is not UD")
        if self.annotation_scope != W02_MORPH_V3_ANNOTATION_SCOPE:
            raise W02MorphologySuccessorV3RouteError(
                "source capability annotation scope drifted")
        if not self.official_url.startswith(
                "https://github.com/UniversalDependencies/UD_"):
            raise W02MorphologySuccessorV3RouteError(
                "UD source capability is not grounded in an official repository")

    def to_dict(self) -> dict[str, object]:
        return {
            "annotation_provenance": self.annotation_provenance,
            "annotation_scope": list(self.annotation_scope),
            "annotation_standard": self.annotation_standard,
            "language": self.language,
            "license_id": self.license_id,
            "official_url": self.official_url,
            "revision_id": self.revision_id,
            "snapshot_id": self.snapshot_id,
            "source_key": self.source_key,
            "upstream_checksum": self.upstream_checksum,
        }

    def sha256(self) -> str:
        return _hash_value(self.to_dict())


def w02_ud_morphology_source_capability(
        source_spec: dict[str, object],
        ) -> W02MorphologySourceCapability:
    """Build a generic UD capability from payload-free public source metadata."""
    if not isinstance(source_spec, dict):
        raise TypeError("source spec must be a dictionary")
    data_file = source_spec.get("data_file")
    if isinstance(data_file, dict) and data_file.get("git_blob_sha1"):
        upstream_checksum = "sha1:" + _nonempty(
            data_file.get("git_blob_sha1"), where="source spec git_blob_sha1")
    else:
        upstream_checksum = _nonempty(
            source_spec.get("upstream_checksum"),
            where="source spec upstream_checksum")
    return W02MorphologySourceCapability(
        _nonempty(source_spec.get("source_key"), where="source spec source_key"),
        _nonempty(source_spec.get("snapshot_id"), where="source spec snapshot_id"),
        _nonempty(source_spec.get("commit_sha1"), where="source spec commit_sha1"),
        _nonempty(source_spec.get("repository_url"), where="source spec repository_url"),
        upstream_checksum,
        _nonempty(source_spec.get("license_id"), where="source spec license_id"),
        _nonempty(source_spec.get("language"), where="source spec language"),
        W02_MORPH_V3_ANNOTATION_STANDARD,
        W02_MORPH_V3_ANNOTATION_SCOPE,
        _nonempty(
            source_spec.get("annotation_provenance"),
            where="source spec annotation_provenance"),
    )


def _capability_matches_source(
        capability: W02MorphologySourceCapability,
        source: SourceRefRecord,
        ) -> bool:
    return (
        source.source_key == capability.source_key
        and source.snapshot_id == capability.snapshot_id
        and source.revision_id == capability.revision_id
        and source.official_url == capability.official_url
        and source.upstream_checksum == capability.upstream_checksum
        and source.license_id == capability.license_id
        and source.redistribution_policy == "PUBLIC"
    )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W02MorphologyRouteIndex:
    """Validated runtime routes, independent of training dataset identities."""

    route_identities: tuple[RouteIdentity, ...]
    dataset_keys: tuple[tuple[int, ...], ...]
    capability_sha256s: tuple[str, ...]
    source_count: int
    logic_operations: int
    semantic_sha256: str

    def __post_init__(self) -> None:
        if (not self.route_identities
                or tuple(sorted(set(self.route_identities))) != self.route_identities
                or not self.dataset_keys
                or tuple(sorted(set(self.dataset_keys))) != self.dataset_keys
                or not self.capability_sha256s
                or tuple(sorted(set(self.capability_sha256s)))
                != self.capability_sha256s):
            raise W02MorphologySuccessorV3RouteError(
                "runtime morphology routes are empty or non-canonical")
        if (type(self.source_count) is not int or self.source_count <= 0
                or self.source_count != len(self.route_identities)
                or type(self.logic_operations) is not int
                or self.logic_operations <= 0):
            raise W02MorphologySuccessorV3RouteError(
                "runtime morphology route counts drifted")
        if self.semantic_sha256 != _hash_value(self.semantic_rows()):
            raise W02MorphologySuccessorV3RouteError(
                "runtime morphology route semantic identity drifted")

    def semantic_rows(self) -> tuple[dict[str, object], ...]:
        rows: list[dict[str, object]] = [
            {"capability_sha256": value, "row_kind": "SOURCE_CAPABILITY"}
            for value in self.capability_sha256s
        ]
        rows.extend({
            "dataset_key": list(dataset_key),
            "language": language,
            "row_kind": "SOURCE_ROUTE",
            "source_ref_key": list(source_ref_key),
        } for dataset_key, source_ref_key, language in self.route_identities)
        return tuple(rows)

    def permits(self, observation: ObservationRecord) -> bool:
        if not isinstance(observation, ObservationRecord):
            raise TypeError("route observation type drifted")
        identity = (
            observation.dataset_key.components,
            observation.source_ref_key.components,
            observation.language,
        )
        offset = bisect_left(self.route_identities, identity)
        return (offset < len(self.route_identities)
                and self.route_identities[offset] == identity)


def authorize_w02_morphology_source_routes(
        sources: Iterable[SourceRefRecord],
        capabilities: Iterable[W02MorphologySourceCapability],
        *,
        max_sources: int = 100_000,
        ) -> W02MorphologyRouteIndex:
    """Authorize UD routes from SourceRef provenance without reading labels."""
    if type(max_sources) is not int or max_sources <= 0:
        raise W02MorphologySuccessorV3RouteError("route source budget is invalid")
    capability_rows = tuple(capabilities)
    if (not capability_rows
            or any(not isinstance(row, W02MorphologySourceCapability)
                   for row in capability_rows)):
        raise W02MorphologySuccessorV3RouteError("route capabilities are empty")
    by_source_key = {row.source_key: row for row in capability_rows}
    if len(by_source_key) != len(capability_rows):
        raise W02MorphologySuccessorV3RouteError("route capability key is duplicated")
    routes: list[RouteIdentity] = []
    operations = len(capability_rows)
    for source in sources:
        if not isinstance(source, SourceRefRecord):
            raise W02MorphologySuccessorV3RouteError("route source type drifted")
        capability = by_source_key.get(source.source_key)
        operations += 1
        if capability is None:
            continue
        if len(routes) >= max_sources:
            raise W02MorphologySuccessorV3RouteError("route source resource stop")
        if not _capability_matches_source(capability, source):
            raise W02MorphologySuccessorV3RouteError(
                "route source provenance does not match its capability")
        routes.append((
            source.dataset_key.components,
            source.stable_key.components,
            capability.language,
        ))
        operations += 7
    canonical_routes = tuple(sorted(set(routes)))
    if not canonical_routes or len(canonical_routes) != len(routes):
        raise W02MorphologySuccessorV3RouteError(
            "route source identities are empty or duplicated")
    capability_sha256s = tuple(sorted(row.sha256() for row in capability_rows))
    dataset_keys = tuple(sorted({row[0] for row in canonical_routes}))
    semantic_rows: list[dict[str, object]] = [
        {"capability_sha256": value, "row_kind": "SOURCE_CAPABILITY"}
        for value in capability_sha256s
    ]
    semantic_rows.extend({
        "dataset_key": list(dataset_key),
        "language": language,
        "row_kind": "SOURCE_ROUTE",
        "source_ref_key": list(source_ref_key),
    } for dataset_key, source_ref_key, language in canonical_routes)
    return W02MorphologyRouteIndex(
        canonical_routes,
        dataset_keys,
        capability_sha256s,
        len(canonical_routes),
        operations + len(canonical_routes),
        _hash_value(semantic_rows),
    )


def _extend_v1_routes(
        index: W02MorphologySuccessorIndex,
        dataset_keys: tuple[tuple[int, ...], ...],
        ) -> W02MorphologySuccessorIndex:
    routes = tuple(sorted(set(index.dataset_keys).union(dataset_keys)))
    if routes == index.dataset_keys:
        return index
    rows = [
        {"dataset_key": list(key), "row_kind": "DATASET_ROUTE"}
        for key in routes
    ]
    rows.extend(
        row for row in index.semantic_rows()
        if row["row_kind"] != "DATASET_ROUTE")
    return W02MorphologySuccessorIndex(
        routes,
        index.global_counts,
        index.feature_counts,
        index.max_form_length,
        index.training_pair_count,
        index.morphology_observation_count,
        index.morphology_token_count,
        index.logic_operations,
        _hash_value(rows),
        len(rows),
    )


def _extend_v2_routes(
        index: W02MorphologySuccessorV2Index,
        dataset_keys: tuple[tuple[int, ...], ...],
        ) -> W02MorphologySuccessorV2Index:
    routes = tuple(sorted(set(index.dataset_keys).union(dataset_keys)))
    if routes == index.dataset_keys:
        return index
    rows = [
        {"dataset_key": list(key), "row_kind": "DATASET_ROUTE"}
        for key in routes
    ]
    rows.extend(
        row for row in index.semantic_rows()
        if row["row_kind"] != "DATASET_ROUTE")
    return W02MorphologySuccessorV2Index(
        routes,
        index.global_counts,
        index.feature_counts,
        index.max_form_length,
        index.accepted_lexeme_rows,
        index.accepted_support_count,
        index.unsupported_lexeme_rows,
        index.unsupported_support_count,
        index.logic_operations,
        _hash_value(rows),
        len(rows),
    )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W02MorphologyRoutedIndexes:
    """Frozen parents plus ephemeral route-extended read-only views."""

    parent_v1: W02MorphologySuccessorIndex
    parent_v2: W02MorphologySuccessorV2Index
    routed_v1: W02MorphologySuccessorIndex
    routed_v2: W02MorphologySuccessorV2Index
    routes: W02MorphologyRouteIndex
    semantic_sha256: str

    def __post_init__(self) -> None:
        if (self.parent_v1.dataset_keys != self.parent_v2.dataset_keys
                or self.routed_v1.dataset_keys != self.routed_v2.dataset_keys
                or not set(self.parent_v1.dataset_keys).issubset(
                    self.routed_v1.dataset_keys)
                or not set(self.routes.dataset_keys).issubset(
                    self.routed_v1.dataset_keys)):
            raise W02MorphologySuccessorV3RouteError(
                "routed morphology parent identity drifted")
        expected = _hash_value({
            "parent_v1_semantic_sha256": self.parent_v1.semantic_sha256,
            "parent_v2_semantic_sha256": self.parent_v2.semantic_sha256,
            "route_semantic_sha256": self.routes.semantic_sha256,
            "routed_v1_semantic_sha256": self.routed_v1.semantic_sha256,
            "routed_v2_semantic_sha256": self.routed_v2.semantic_sha256,
        })
        if self.semantic_sha256 != expected:
            raise W02MorphologySuccessorV3RouteError(
                "routed morphology semantic identity drifted")


def build_w02_morphology_routed_indexes(
        v1_index: W02MorphologySuccessorIndex,
        v2_index: W02MorphologySuccessorV2Index,
        routes: W02MorphologyRouteIndex,
        ) -> W02MorphologyRoutedIndexes:
    """Create append-only route views without mutating frozen V1/V2 indexes."""
    if (not isinstance(v1_index, W02MorphologySuccessorIndex)
            or not isinstance(v2_index, W02MorphologySuccessorV2Index)
            or not isinstance(routes, W02MorphologyRouteIndex)):
        raise TypeError("routed morphology index type drifted")
    if v1_index.dataset_keys != v2_index.dataset_keys:
        raise W02MorphologySuccessorV3RouteError(
            "frozen V1/V2 route identity drifted")
    routed_v1 = _extend_v1_routes(v1_index, routes.dataset_keys)
    routed_v2 = _extend_v2_routes(v2_index, routes.dataset_keys)
    semantic = _hash_value({
        "parent_v1_semantic_sha256": v1_index.semantic_sha256,
        "parent_v2_semantic_sha256": v2_index.semantic_sha256,
        "route_semantic_sha256": routes.semantic_sha256,
        "routed_v1_semantic_sha256": routed_v1.semantic_sha256,
        "routed_v2_semantic_sha256": routed_v2.semantic_sha256,
    })
    return W02MorphologyRoutedIndexes(
        v1_index, v2_index, routed_v1, routed_v2, routes, semantic)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W02MorphologySuccessorV3Prediction:
    """V1/V2 prediction with explicit runtime-route evidence."""

    v1: W02MorphologySuccessorPrediction
    v2: W02MorphologySuccessorV2Prediction
    route_authorized: int
    route_logic_operations: int
    route_semantic_sha256: str

    def __post_init__(self) -> None:
        if (not isinstance(self.v1, W02MorphologySuccessorPrediction)
                or not isinstance(self.v2, W02MorphologySuccessorV2Prediction)
                or self.v2.prediction.observation_key
                != self.v1.prediction.observation_key
                or self.route_authorized not in (0, 1)
                or type(self.route_logic_operations) is not int
                or self.route_logic_operations <= 0
                or len(self.route_semantic_sha256) != 64):
            raise W02MorphologySuccessorV3RouteError(
                "V3 route prediction identity drifted")


def predict_w02_morphology_successor_v3(
        indexes: W02MorphologyRoutedIndexes,
        observation: ObservationRecord,
        base: W02CandidatePrediction,
        *,
        requested_spans: tuple[tuple[int, int], ...],
        v1_cache: W02MorphologyRankingCache | None = None,
        v2_cache: W02MorphologySuccessorV2Cache | None = None,
        ) -> W02MorphologySuccessorV3Prediction:
    """Run frozen overlays through a source-authorized route or their old routes."""
    if not isinstance(indexes, W02MorphologyRoutedIndexes):
        raise TypeError("V3 routed indexes type drifted")
    authorized = indexes.routes.permits(observation)
    v1_index = indexes.routed_v1 if authorized else indexes.parent_v1
    v2_index = indexes.routed_v2 if authorized else indexes.parent_v2
    v1 = predict_w02_morphology_successor(
        v1_index,
        observation,
        base,
        requested_spans=requested_spans,
        ranking_cache=v1_cache,
    )
    v2 = predict_w02_morphology_successor_v2(
        v2_index,
        observation,
        v1,
        requested_spans=requested_spans,
        cache=v2_cache,
    )
    return W02MorphologySuccessorV3Prediction(
        v1,
        v2,
        int(authorized),
        len(indexes.routes.route_identities).bit_length() + 4,
        indexes.semantic_sha256,
    )


__all__ = [
    "W02_MORPH_SUCCESSOR_V3_ROUTE_VERSION",
    "W02_MORPH_V3_ANNOTATION_SCOPE",
    "W02_MORPH_V3_ANNOTATION_STANDARD",
    "W02MorphologyRouteIndex",
    "W02MorphologyRoutedIndexes",
    "W02MorphologySourceCapability",
    "W02MorphologySuccessorV3Prediction",
    "W02MorphologySuccessorV3RouteError",
    "authorize_w02_morphology_source_routes",
    "build_w02_morphology_routed_indexes",
    "predict_w02_morphology_successor_v3",
    "w02_ud_morphology_source_capability",
]
