"""LC-16 ArtifactEnvelope、局部 anchor 与共享 identity 基础合同。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.formal_artifact import (
    ArtifactAuthority,
    artifact_authority_from_stable_key,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_ARTIFACT,
    OBJECT_CONCEPT,
    OBJECT_LANGUAGE_BRANCH,
    OBJECT_OCCURRENCE,
    OBJECT_ROLE,
    OBJECT_SPAN,
    OBJECT_STRUCTURE_CONCEPT,
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.semantic_object import (
    validate_semantic_identity,
)
from pure_integer_ai.cognition.shared.unicode_representation import (
    validate_unicode_scalars,
)
from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


ARTIFACT_CARRIER_IDENTITY_MAGIC = 16001
ARTIFACT_CARRIER_IDENTITY_ENVELOPE = 1
ARTIFACT_CARRIER_IDENTITY_ANCHOR = 2
ARTIFACT_CARRIER_IDENTITY_STRUCTURE_NODE = 3
ARTIFACT_CARRIER_IDENTITY_REFERENCE = 4
ARTIFACT_CARRIER_IDENTITY_PROJECTION = 5
ARTIFACT_CARRIER_IDENTITY_REVISION = 6

RAW_UNIT_UNICODE_SCALAR = 1
RAW_UNIT_OCTET = 2

ANCHOR_TEXT_RANGE = 1
ANCHOR_TREE_PATH = 2
ANCHOR_GRID_RECT = 3
ANCHOR_DOCUMENT_REGION = 4
ANCHOR_REFERENCE_SLOT = 5
ANCHOR_TRANSCRIPT_ALIGNMENT = 6

REFERENCE_RESOLVED = 1
REFERENCE_UNRESOLVED = 2
REFERENCE_WITHDRAWN = 3
REFERENCE_ACCESS_BLOCKED = 4

PROJECTION_UNDERSTANDING = 1
PROJECTION_REASONING = 2
PROJECTION_GENERATION = 3

REVISION_MAP_ANCHOR = 1
REVISION_MAP_STRUCTURE_NODE = 2
REVISION_MAP_REFERENCE = 3
REVISION_MAP_PROJECTION = 4

_SOURCE_KEY_SIZE = 11
_FINGERPRINT_SIZE = 34
_LOCAL_IDENTITY_KINDS = frozenset({
    ARTIFACT_CARRIER_IDENTITY_ENVELOPE,
    ARTIFACT_CARRIER_IDENTITY_ANCHOR,
    ARTIFACT_CARRIER_IDENTITY_STRUCTURE_NODE,
    ARTIFACT_CARRIER_IDENTITY_REFERENCE,
    ARTIFACT_CARRIER_IDENTITY_PROJECTION,
    ARTIFACT_CARRIER_IDENTITY_REVISION,
})
_CLASSIFIER_KINDS = frozenset({
    OBJECT_CONCEPT,
    OBJECT_ROLE,
    OBJECT_STRUCTURE_CONCEPT,
})
_ANCHOR_KINDS = frozenset({
    ANCHOR_TEXT_RANGE,
    ANCHOR_TREE_PATH,
    ANCHOR_GRID_RECT,
    ANCHOR_DOCUMENT_REGION,
    ANCHOR_REFERENCE_SLOT,
    ANCHOR_TRANSCRIPT_ALIGNMENT,
})
_REFERENCE_STATES = frozenset({
    REFERENCE_RESOLVED,
    REFERENCE_UNRESOLVED,
    REFERENCE_WITHDRAWN,
    REFERENCE_ACCESS_BLOCKED,
})
_PROJECTION_DIRECTIONS = frozenset({
    PROJECTION_UNDERSTANDING,
    PROJECTION_REASONING,
    PROJECTION_GENERATION,
})
_REVISION_MAPPING_LOCAL_KINDS = {
    REVISION_MAP_ANCHOR: ARTIFACT_CARRIER_IDENTITY_ANCHOR,
    REVISION_MAP_STRUCTURE_NODE: ARTIFACT_CARRIER_IDENTITY_STRUCTURE_NODE,
    REVISION_MAP_REFERENCE: ARTIFACT_CARRIER_IDENTITY_REFERENCE,
    REVISION_MAP_PROJECTION: ARTIFACT_CARRIER_IDENTITY_PROJECTION,
}
_LOCAL_CONTEXT_DOMAINS = {
    ARTIFACT_CARRIER_IDENTITY_ANCHOR: (
        "lc16.artifact-anchor.envelope.v1",
        "lc16.artifact-anchor.scope.v1",
    ),
    ARTIFACT_CARRIER_IDENTITY_STRUCTURE_NODE: (
        "lc16.artifact-node.envelope.v1",
        "lc16.artifact-node.scope.v1",
    ),
    ARTIFACT_CARRIER_IDENTITY_REFERENCE: (
        "lc16.artifact-reference.envelope.v1",
        "lc16.artifact-reference.scope.v1",
    ),
    ARTIFACT_CARRIER_IDENTITY_PROJECTION: (
        "lc16.artifact-projection.envelope.v1",
        "lc16.artifact-projection.scope.v1",
    ),
}


def _ints(
        value: tuple[int, ...],
        *,
        where: str,
        allow_empty: bool = False,
        nonnegative: bool = False,
        ) -> tuple[int, ...]:
    if not isinstance(value, tuple) or (not allow_empty and not value):
        raise ValueError(f"{where} 必须是整数 tuple")
    assert_int(*value, _where=where)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{where} 必须使用严格整数")
    if nonnegative and any(item < 0 for item in value):
        raise ValueError(f"{where} 必须使用非负整数")
    return value


def _nonnegative(value: int, *, where: str) -> int:
    assert_int(value, _where=where)
    if type(value) is not int or value < 0:
        raise ValueError(f"{where} 必须是非负严格整数")
    return value


def _positive(value: int, *, where: str) -> int:
    assert_int(value, _where=where)
    if type(value) is not int or value <= 0:
        raise ValueError(f"{where} 必须是正严格整数")
    return value


def _packed(value: tuple[int, ...]) -> tuple[int, ...]:
    return len(value), *value


def _optional_key(value: tuple[int, ...] | None) -> tuple[int, ...]:
    return () if value is None else value


def _object(value: ObjectIdentity, *, where: str) -> ObjectIdentity:
    if not isinstance(value, ObjectIdentity):
        raise TypeError(f"{where} 必须是 ObjectIdentity")
    if ObjectIdentity.from_stable_key(value.stable_key()) != value:
        raise ValueError(f"{where} 无法稳定 round-trip")
    return value


def _classifier(
        value: ObjectIdentity,
        *,
        where: str,
        object_kind: int | None = None,
        ) -> ObjectIdentity:
    _object(value, where=where)
    if object_kind is not None:
        if value.object_kind != object_kind:
            raise ValueError(f"{where} 对象类型非法")
        if object_kind == OBJECT_LANGUAGE_BRANCH:
            return value
    elif value.object_kind not in _CLASSIFIER_KINDS:
        raise ValueError(f"{where} 必须是 Concept/Role/StructureConcept")
    validate_semantic_identity(value)
    return value


def _scope_for_source(
        scope: ScopeIdentity,
        source: SourceRef,
        *,
        where: str,
        ) -> ScopeIdentity:
    if not isinstance(scope, ScopeIdentity):
        raise TypeError(f"{where} 必须是 ScopeIdentity")
    if (scope.owner != source.owner or scope.versions != source.versions
            or (scope.source is not None and scope.source != source)):
        raise ValueError(f"{where} 与 SourceRef 不一致")
    if ScopeIdentity.from_stable_key(scope.stable_key()) != scope:
        raise ValueError(f"{where} 无法稳定 round-trip")
    return scope


def _authority(value: ArtifactAuthority, *, where: str) -> ArtifactAuthority:
    if not isinstance(value, ArtifactAuthority):
        raise TypeError(f"{where} 必须是 ArtifactAuthority")
    if artifact_authority_from_stable_key(value.stable_key()) != value:
        raise ValueError(f"{where} 无法稳定 round-trip")
    return value


def _fingerprint(value: tuple[int, ...], *, domain: str) -> tuple[int, ...]:
    result = integer_tuple_fingerprint(value, domain=domain)
    if len(result) != _FINGERPRINT_SIZE:
        raise ValueError("整数内容指纹长度漂移")
    return result


def _optional_fingerprint(
        value: tuple[int, ...] | None,
        *,
        domain: str,
        ) -> tuple[int, ...]:
    if value is None:
        return (0, *([0] * _FINGERPRINT_SIZE))
    return (1, *_fingerprint(value, domain=domain))


def _carrier_identity(
        local_kind: int,
        source: SourceRef,
        components: tuple[int, ...],
        ) -> ObjectIdentity:
    if local_kind not in _LOCAL_IDENTITY_KINDS:
        raise ValueError("LC-16 local identity kind 未登记")
    if not isinstance(source, SourceRef):
        raise TypeError("LC-16 local identity source 必须是 SourceRef")
    _ints(components, where="LC-16 identity components")
    return ObjectIdentity(
        OBJECT_ARTIFACT,
        (
            ARTIFACT_CARRIER_IDENTITY_MAGIC,
            local_kind,
            *source.stable_key(),
            *components,
        ),
        source.owner,
        source.versions,
    )


def _validate_fingerprint(values: tuple[int, ...], *, where: str) -> None:
    if (len(values) != _FINGERPRINT_SIZE or values[0] != 1
            or values[1] < 0
            or any(type(item) is not int or item < 0 or item > 255
                   for item in values[2:])):
        raise ValueError(f"{where} 内容指纹非法")


def _validate_optional_fingerprint(
        values: tuple[int, ...], *, where: str,
        ) -> None:
    if len(values) != 1 + _FINGERPRINT_SIZE or values[0] not in {0, 1}:
        raise ValueError(f"{where} optional fingerprint 非法")
    if values[0] == 0:
        if any(values[1:]):
            raise ValueError(f"{where} 空 optional fingerprint 含数据")
    else:
        _validate_fingerprint(values[1:], where=where)


def _take_identity_fingerprint(
        components: tuple[int, ...], cursor: int, *, where: str,
        ) -> int:
    end = cursor + _FINGERPRINT_SIZE
    if end > len(components):
        raise ValueError(f"{where} 被截断")
    _validate_fingerprint(components[cursor:end], where=where)
    return end


def _take_optional_identity_fingerprint(
        components: tuple[int, ...], cursor: int, *, where: str,
        ) -> int:
    end = cursor + 1 + _FINGERPRINT_SIZE
    if end > len(components):
        raise ValueError(f"{where} 被截断")
    _validate_optional_fingerprint(components[cursor:end], where=where)
    return end


def _validate_local_identity_components(identity: ObjectIdentity) -> None:
    components = identity.components
    kind = components[1]
    cursor = 2 + _SOURCE_KEY_SIZE

    if kind == ARTIFACT_CARRIER_IDENTITY_ENVELOPE:
        if cursor >= len(components) or components[cursor] not in {
                RAW_UNIT_UNICODE_SCALAR, RAW_UNIT_OCTET}:
            raise ValueError("ArtifactEnvelope identity raw unit kind 非法")
        cursor += 1
        for label in ("raw", "scope", "carrier", "media"):
            cursor = _take_identity_fingerprint(
                components, cursor, where=f"ArtifactEnvelope {label}")
        cursor = _take_optional_identity_fingerprint(
            components, cursor, where="ArtifactEnvelope language")
        cursor = _take_identity_fingerprint(
            components, cursor, where="ArtifactEnvelope parser")
        cursor = _take_optional_identity_fingerprint(
            components, cursor, where="ArtifactEnvelope renderer")
        cursor = _take_identity_fingerprint(
            components, cursor, where="ArtifactEnvelope key")
    elif kind == ARTIFACT_CARRIER_IDENTITY_ANCHOR:
        cursor = _take_identity_fingerprint(
            components, cursor, where="ArtifactAnchor envelope")
        cursor = _take_identity_fingerprint(
            components, cursor, where="ArtifactAnchor scope")
        if cursor >= len(components) or components[cursor] not in _ANCHOR_KINDS:
            raise ValueError("ArtifactAnchor identity anchor kind 非法")
        anchor_kind = components[cursor]
        cursor += 1
        if cursor >= len(components):
            raise ValueError("ArtifactAnchor identity 缺少 coordinates 长度")
        size = components[cursor]
        cursor += 1
        if size <= 0 or cursor + size > len(components):
            raise ValueError("ArtifactAnchor identity coordinates 长度非法")
        coordinates = components[cursor:cursor + size]
        _validate_anchor_coordinates(anchor_kind, coordinates)
        cursor += size
        cursor = _take_identity_fingerprint(
            components, cursor, where="ArtifactAnchor parser")
        cursor = _take_optional_identity_fingerprint(
            components, cursor, where="ArtifactAnchor linked text")
        cursor = _take_identity_fingerprint(
            components, cursor, where="ArtifactAnchor key")
    elif kind == ARTIFACT_CARRIER_IDENTITY_STRUCTURE_NODE:
        for label in ("envelope", "scope", "anchor", "family", "kind"):
            cursor = _take_identity_fingerprint(
                components, cursor, where=f"ArtifactStructureNode {label}")
        cursor = _take_optional_identity_fingerprint(
            components, cursor, where="ArtifactStructureNode role")
        cursor = _take_optional_identity_fingerprint(
            components, cursor, where="ArtifactStructureNode parent")
        if cursor >= len(components) or components[cursor] < 0:
            raise ValueError("ArtifactStructureNode ordinal 非法")
        cursor += 1
        cursor = _take_identity_fingerprint(
            components, cursor, where="ArtifactStructureNode qualifiers")
        cursor = _take_identity_fingerprint(
            components, cursor, where="ArtifactStructureNode key")
    elif kind == ARTIFACT_CARRIER_IDENTITY_REFERENCE:
        for label in ("envelope", "scope", "anchor", "relation"):
            cursor = _take_identity_fingerprint(
                components, cursor, where=f"ArtifactReference {label}")
        if cursor >= len(components) or components[cursor] not in _REFERENCE_STATES:
            raise ValueError("ArtifactReference target state 非法")
        cursor += 1
        for label in ("target source", "target anchor", "target fingerprint"):
            cursor = _take_optional_identity_fingerprint(
                components, cursor, where=f"ArtifactReference {label}")
        cursor = _take_identity_fingerprint(
            components, cursor, where="ArtifactReference key")
    elif kind == ARTIFACT_CARRIER_IDENTITY_PROJECTION:
        cursor = _take_identity_fingerprint(
            components, cursor, where="ArtifactProjection envelope")
        cursor = _take_identity_fingerprint(
            components, cursor, where="ArtifactProjection scope")
        if cursor >= len(components) or components[cursor] < 0:
            raise ValueError("ArtifactProjection anchor count 非法")
        anchor_count = components[cursor]
        cursor += 1
        for _ in range(anchor_count):
            cursor = _take_identity_fingerprint(
                components, cursor, where="ArtifactProjection anchor")
        if cursor >= len(components) or components[cursor] < 0:
            raise ValueError("ArtifactProjection node count 非法")
        node_count = components[cursor]
        cursor += 1
        if anchor_count + node_count <= 0:
            raise ValueError("ArtifactProjection subject count 非法")
        for _ in range(node_count):
            cursor = _take_identity_fingerprint(
                components, cursor, where="ArtifactProjection node")
        for label in (
                "kind", "semantic object", "lifecycle", "hypothesis",
                "directions", "key"):
            cursor = _take_identity_fingerprint(
                components, cursor, where=f"ArtifactProjection {label}")
    elif kind == ARTIFACT_CARRIER_IDENTITY_REVISION:
        for label in (
                "old envelope", "new envelope", "reason", "hypothesis",
                "mappings", "key"):
            cursor = _take_identity_fingerprint(
                components, cursor, where=f"ArtifactRevision {label}")
    if cursor != len(components):
        raise ValueError("artifact carrier identity 含尾随或缺失字段")


def artifact_carrier_local_kind(identity: ObjectIdentity) -> int:
    """返回 LC-16 Artifact 局部对象类型，并拒绝 FormalArtifact 或伪身份。"""
    _object(identity, where="artifact carrier identity")
    if identity.object_kind != OBJECT_ARTIFACT:
        raise ValueError("artifact carrier identity 对象类型非法")
    components = identity.components
    if (len(components) < 2 + _SOURCE_KEY_SIZE
            or components[0] != ARTIFACT_CARRIER_IDENTITY_MAGIC
            or components[1] not in _LOCAL_IDENTITY_KINDS):
        raise ValueError("artifact carrier identity magic/kind 非法")
    source = SourceRef.from_stable_key(
        components[2:2 + _SOURCE_KEY_SIZE])
    if source.owner != identity.owner or source.versions != identity.versions:
        raise ValueError("artifact carrier identity 与来源 owner/version 不一致")
    _validate_local_identity_components(identity)
    return components[1]


def artifact_carrier_source(identity: ObjectIdentity) -> SourceRef:
    """从任一 LC-16 局部身份恢复固定长度 SourceRef。"""
    artifact_carrier_local_kind(identity)
    return SourceRef.from_stable_key(
        identity.components[2:2 + _SOURCE_KEY_SIZE])


class _KeyReader:
    def __init__(self, key: tuple[int, ...], *, where: str) -> None:
        self.key = _ints(key, where=where)
        self.where = where
        self.cursor = 0

    def integer(self, label: str) -> int:
        if self.cursor >= len(self.key):
            raise ValueError(f"{self.where} 缺少 {label}")
        value = self.key[self.cursor]
        self.cursor += 1
        return value

    def part(self, label: str, *, allow_empty: bool = False) -> tuple[int, ...]:
        size = self.integer(f"{label} 长度")
        if size < 0 or (size == 0 and not allow_empty):
            raise ValueError(f"{self.where} {label} 长度非法")
        end = self.cursor + size
        if end > len(self.key):
            raise ValueError(f"{self.where} {label} 被截断")
        result = self.key[self.cursor:end]
        self.cursor = end
        return result

    def finish(self) -> None:
        if self.cursor != len(self.key):
            raise ValueError(f"{self.where} 含尾随字段")


def _raw_units(raw_unit_kind: int, values: tuple[int, ...]) -> tuple[int, ...]:
    _positive(raw_unit_kind, where="raw_unit_kind")
    values = _ints(
        values, where="ArtifactEnvelope.raw_units", allow_empty=True,
        nonnegative=True)
    if raw_unit_kind == RAW_UNIT_UNICODE_SCALAR:
        validate_unicode_scalars(values)
    elif raw_unit_kind == RAW_UNIT_OCTET:
        if any(value > 255 for value in values):
            raise ValueError("OCTET raw unit 必须在 0..255")
    else:
        raise ValueError("raw_unit_kind 未登记")
    return values


def artifact_envelope_identity(
        *,
        source: SourceRef,
        scope: ScopeIdentity,
        carrier_family: ObjectIdentity,
        raw_unit_kind: int,
        raw_units: tuple[int, ...],
        media_profile: ObjectIdentity,
        language_branch: ObjectIdentity | None,
        parser: ArtifactAuthority,
        renderer: ArtifactAuthority | None,
        envelope_key: tuple[int, ...],
        ) -> ObjectIdentity:
    """构造只含固定长度引用的 envelope 身份，不把完整 raw 内嵌进图身份。"""
    _scope_for_source(scope, source, where="ArtifactEnvelope.scope")
    _classifier(carrier_family, where="ArtifactEnvelope.carrier_family")
    raw_units = _raw_units(raw_unit_kind, raw_units)
    _classifier(media_profile, where="ArtifactEnvelope.media_profile")
    if language_branch is not None:
        _classifier(
            language_branch,
            where="ArtifactEnvelope.language_branch",
            object_kind=OBJECT_LANGUAGE_BRANCH,
        )
        if (language_branch.owner != source.owner
                or language_branch.versions != source.versions):
            raise ValueError("language branch 与 envelope 来源不一致")
    _authority(parser, where="ArtifactEnvelope.parser")
    if renderer is not None:
        _authority(renderer, where="ArtifactEnvelope.renderer")
    envelope_key = _ints(envelope_key, where="ArtifactEnvelope.envelope_key")
    components = (
        raw_unit_kind,
        *_fingerprint(
            raw_units, domain="lc16.artifact-envelope.raw-units.v1"),
        *_fingerprint(
            scope.stable_key(), domain="lc16.artifact-envelope.scope.v1"),
        *_fingerprint(
            carrier_family.stable_key(),
            domain="lc16.artifact-envelope.carrier-family.v1"),
        *_fingerprint(
            media_profile.stable_key(),
            domain="lc16.artifact-envelope.media-profile.v1"),
        *_optional_fingerprint(
            None if language_branch is None else language_branch.stable_key(),
            domain="lc16.artifact-envelope.language-branch.v1"),
        *_fingerprint(
            parser.stable_key(), domain="lc16.artifact-envelope.parser.v1"),
        *_optional_fingerprint(
            None if renderer is None else renderer.stable_key(),
            domain="lc16.artifact-envelope.renderer.v1"),
        *_fingerprint(
            envelope_key, domain="lc16.artifact-envelope.key.v1"),
    )
    return _carrier_identity(
        ARTIFACT_CARRIER_IDENTITY_ENVELOPE, source, components)


def _envelope_scope_fingerprint(identity: ObjectIdentity) -> tuple[int, ...]:
    if artifact_carrier_local_kind(identity) != ARTIFACT_CARRIER_IDENTITY_ENVELOPE:
        raise ValueError("identity 不是 ArtifactEnvelope")
    start = 2 + _SOURCE_KEY_SIZE + 1 + _FINGERPRINT_SIZE
    end = start + _FINGERPRINT_SIZE
    if len(identity.components) < end:
        raise ValueError("ArtifactEnvelope identity 被截断")
    return identity.components[start:end]


def _require_envelope_context(
        envelope_identity: ObjectIdentity,
        source: SourceRef,
        scope: ScopeIdentity,
        ) -> None:
    if artifact_carrier_local_kind(
            envelope_identity) != ARTIFACT_CARRIER_IDENTITY_ENVELOPE:
        raise ValueError("envelope_identity 类型非法")
    if artifact_carrier_source(envelope_identity) != source:
        raise ValueError("局部对象与 envelope SourceRef 不一致")
    _scope_for_source(scope, source, where="artifact carrier local scope")
    expected = _fingerprint(
        scope.stable_key(), domain="lc16.artifact-envelope.scope.v1")
    if _envelope_scope_fingerprint(envelope_identity) != expected:
        raise ValueError("局部对象 scope 与 envelope scope 不一致")


def _require_local_envelope(
        identity: ObjectIdentity,
        *,
        expected_kind: int,
        envelope_identity: ObjectIdentity,
        where: str,
        ) -> None:
    """核验局部身份确实由指定 envelope 派生，而不只共享 SourceRef。"""
    domains = _LOCAL_CONTEXT_DOMAINS.get(expected_kind)
    if domains is None:
        raise ValueError(f"{where} local kind 不支持 envelope 绑定")
    if artifact_carrier_local_kind(identity) != expected_kind:
        raise ValueError(f"{where} 类型非法")
    if artifact_carrier_local_kind(
            envelope_identity) != ARTIFACT_CARRIER_IDENTITY_ENVELOPE:
        raise ValueError(f"{where} envelope 类型非法")
    if artifact_carrier_source(identity) != artifact_carrier_source(
            envelope_identity):
        raise ValueError(f"{where} 来源不一致")
    cursor = 2 + _SOURCE_KEY_SIZE
    actual = identity.components[cursor:cursor + _FINGERPRINT_SIZE]
    expected = _fingerprint(envelope_identity.stable_key(), domain=domains[0])
    if actual != expected:
        raise ValueError(f"{where} 不属于指定 envelope")


def _require_local_context(
        identity: ObjectIdentity,
        *,
        expected_kind: int,
        envelope_identity: ObjectIdentity,
        source: SourceRef,
        scope: ScopeIdentity,
        where: str,
        ) -> None:
    """核验局部身份的 envelope、source 与 scope 三项上下文。"""
    _require_envelope_context(envelope_identity, source, scope)
    _require_local_envelope(
        identity,
        expected_kind=expected_kind,
        envelope_identity=envelope_identity,
        where=where,
    )
    scope_domain = _LOCAL_CONTEXT_DOMAINS[expected_kind][1]
    cursor = 2 + _SOURCE_KEY_SIZE + _FINGERPRINT_SIZE
    actual = identity.components[cursor:cursor + _FINGERPRINT_SIZE]
    expected = _fingerprint(scope.stable_key(), domain=scope_domain)
    if actual != expected:
        raise ValueError(f"{where} scope 与指定 envelope 不一致")


@dataclass(frozen=True)
class ArtifactEnvelope:
    """保留 raw units、来源、scope、parser/renderer 与 carrier 身份的顶层对象。"""

    identity: ObjectIdentity
    source: SourceRef
    scope: ScopeIdentity
    carrier_family: ObjectIdentity
    raw_unit_kind: int
    raw_units: tuple[int, ...]
    media_profile: ObjectIdentity
    language_branch: ObjectIdentity | None
    parser: ArtifactAuthority
    renderer: ArtifactAuthority | None
    envelope_key: tuple[int, ...]

    def __post_init__(self) -> None:
        expected = artifact_envelope_identity(
            source=self.source,
            scope=self.scope,
            carrier_family=self.carrier_family,
            raw_unit_kind=self.raw_unit_kind,
            raw_units=self.raw_units,
            media_profile=self.media_profile,
            language_branch=self.language_branch,
            parser=self.parser,
            renderer=self.renderer,
            envelope_key=self.envelope_key,
        )
        if self.identity != expected:
            raise ValueError("ArtifactEnvelope identity 与字段不一致")

    def stable_key(self) -> tuple[int, ...]:
        return (
            1,
            *_packed(self.identity.stable_key()),
            *_packed(self.source.stable_key()),
            *_packed(self.scope.stable_key()),
            *_packed(self.carrier_family.stable_key()),
            self.raw_unit_kind,
            *_packed(self.raw_units),
            *_packed(self.media_profile.stable_key()),
            *_packed(_optional_key(
                None if self.language_branch is None
                else self.language_branch.stable_key())),
            *_packed(self.parser.stable_key()),
            *_packed(_optional_key(
                None if self.renderer is None else self.renderer.stable_key())),
            *_packed(self.envelope_key),
        )

    @classmethod
    def from_stable_key(cls, key: tuple[int, ...]) -> "ArtifactEnvelope":
        reader = _KeyReader(key, where="ArtifactEnvelope.stable_key")
        if reader.integer("version") != 1:
            raise ValueError("ArtifactEnvelope stable key version 非法")
        identity = ObjectIdentity.from_stable_key(reader.part("identity"))
        source = SourceRef.from_stable_key(reader.part("source"))
        scope = ScopeIdentity.from_stable_key(reader.part("scope"))
        carrier_family = ObjectIdentity.from_stable_key(
            reader.part("carrier_family"))
        raw_unit_kind = reader.integer("raw_unit_kind")
        raw_units = reader.part("raw_units", allow_empty=True)
        media_profile = ObjectIdentity.from_stable_key(
            reader.part("media_profile"))
        language_key = reader.part("language_branch", allow_empty=True)
        parser = artifact_authority_from_stable_key(reader.part("parser"))
        renderer_key = reader.part("renderer", allow_empty=True)
        envelope_key = reader.part("envelope_key")
        reader.finish()
        return cls(
            identity,
            source,
            scope,
            carrier_family,
            raw_unit_kind,
            raw_units,
            media_profile,
            None if not language_key else ObjectIdentity.from_stable_key(
                language_key),
            parser,
            None if not renderer_key else artifact_authority_from_stable_key(
                renderer_key),
            envelope_key,
        )


def make_artifact_envelope(**kwargs) -> ArtifactEnvelope:
    """从字段构造 identity 与完整 ArtifactEnvelope。"""
    identity = artifact_envelope_identity(**kwargs)
    return ArtifactEnvelope(identity=identity, **kwargs)


def _validate_anchor_coordinates(
        anchor_kind: int,
        coordinates: tuple[int, ...],
        ) -> tuple[int, ...]:
    if anchor_kind not in _ANCHOR_KINDS:
        raise ValueError("anchor_kind 未登记")
    coordinates = _ints(
        coordinates, where="ArtifactAnchor.coordinates", nonnegative=True)
    if anchor_kind in {ANCHOR_TEXT_RANGE, ANCHOR_TRANSCRIPT_ALIGNMENT}:
        if len(coordinates) != 2 or coordinates[0] > coordinates[1]:
            raise ValueError("range/alignment anchor 必须是有序二元范围")
    elif anchor_kind == ANCHOR_GRID_RECT:
        if (len(coordinates) != 4 or coordinates[0] > coordinates[1]
                or coordinates[2] > coordinates[3]):
            raise ValueError("grid anchor 必须是有序四元矩形")
    return coordinates


def artifact_anchor_identity(
        *,
        envelope_identity: ObjectIdentity,
        source: SourceRef,
        scope: ScopeIdentity,
        anchor_kind: int,
        coordinates: tuple[int, ...],
        parser: ArtifactAuthority,
        linked_text_anchor: ObjectIdentity | None,
        anchor_key: tuple[int, ...],
        ) -> ObjectIdentity:
    _require_envelope_context(envelope_identity, source, scope)
    coordinates = _validate_anchor_coordinates(anchor_kind, coordinates)
    _authority(parser, where="ArtifactAnchor.parser")
    if linked_text_anchor is not None:
        _object(linked_text_anchor, where="ArtifactAnchor.linked_text_anchor")
        if linked_text_anchor.object_kind not in {OBJECT_SPAN, OBJECT_OCCURRENCE}:
            raise ValueError("linked_text_anchor 必须是 Span 或 Occurrence")
        if (linked_text_anchor.owner != source.owner
                or linked_text_anchor.versions != source.versions):
            raise ValueError("linked_text_anchor 与来源 owner/version 不一致")
    anchor_key = _ints(anchor_key, where="ArtifactAnchor.anchor_key")
    components = (
        *_fingerprint(
            envelope_identity.stable_key(),
            domain="lc16.artifact-anchor.envelope.v1"),
        *_fingerprint(
            scope.stable_key(), domain="lc16.artifact-anchor.scope.v1"),
        anchor_kind,
        *_packed(coordinates),
        *_fingerprint(
            parser.stable_key(), domain="lc16.artifact-anchor.parser.v1"),
        *_optional_fingerprint(
            None if linked_text_anchor is None
            else linked_text_anchor.stable_key(),
            domain="lc16.artifact-anchor.linked-text.v1"),
        *_fingerprint(anchor_key, domain="lc16.artifact-anchor.key.v1"),
    )
    return _carrier_identity(
        ARTIFACT_CARRIER_IDENTITY_ANCHOR, source, components)


@dataclass(frozen=True)
class ArtifactAnchor:
    """不把 tree/grid/region/timecode 压成字符 offset 的 carrier 局部锚。"""

    identity: ObjectIdentity
    envelope_identity: ObjectIdentity
    source: SourceRef
    scope: ScopeIdentity
    anchor_kind: int
    coordinates: tuple[int, ...]
    parser: ArtifactAuthority
    linked_text_anchor: ObjectIdentity | None
    anchor_key: tuple[int, ...]

    def __post_init__(self) -> None:
        expected = artifact_anchor_identity(
            envelope_identity=self.envelope_identity,
            source=self.source,
            scope=self.scope,
            anchor_kind=self.anchor_kind,
            coordinates=self.coordinates,
            parser=self.parser,
            linked_text_anchor=self.linked_text_anchor,
            anchor_key=self.anchor_key,
        )
        if self.identity != expected:
            raise ValueError("ArtifactAnchor identity 与字段不一致")

    def stable_key(self) -> tuple[int, ...]:
        return (
            1,
            *_packed(self.identity.stable_key()),
            *_packed(self.envelope_identity.stable_key()),
            *_packed(self.source.stable_key()),
            *_packed(self.scope.stable_key()),
            self.anchor_kind,
            *_packed(self.coordinates),
            *_packed(self.parser.stable_key()),
            *_packed(_optional_key(
                None if self.linked_text_anchor is None
                else self.linked_text_anchor.stable_key())),
            *_packed(self.anchor_key),
        )

    @classmethod
    def from_stable_key(cls, key: tuple[int, ...]) -> "ArtifactAnchor":
        reader = _KeyReader(key, where="ArtifactAnchor.stable_key")
        if reader.integer("version") != 1:
            raise ValueError("ArtifactAnchor stable key version 非法")
        identity = ObjectIdentity.from_stable_key(reader.part("identity"))
        envelope = ObjectIdentity.from_stable_key(reader.part("envelope"))
        source = SourceRef.from_stable_key(reader.part("source"))
        scope = ScopeIdentity.from_stable_key(reader.part("scope"))
        anchor_kind = reader.integer("anchor_kind")
        coordinates = reader.part("coordinates")
        parser = artifact_authority_from_stable_key(reader.part("parser"))
        linked_key = reader.part("linked_text_anchor", allow_empty=True)
        anchor_key = reader.part("anchor_key")
        reader.finish()
        return cls(
            identity,
            envelope,
            source,
            scope,
            anchor_kind,
            coordinates,
            parser,
            None if not linked_key else ObjectIdentity.from_stable_key(
                linked_key),
            anchor_key,
        )


def make_artifact_anchor(**kwargs) -> ArtifactAnchor:
    identity = artifact_anchor_identity(**kwargs)
    return ArtifactAnchor(identity=identity, **kwargs)

__all__ = [
    "ANCHOR_DOCUMENT_REGION",
    "ANCHOR_GRID_RECT",
    "ANCHOR_REFERENCE_SLOT",
    "ANCHOR_TEXT_RANGE",
    "ANCHOR_TRANSCRIPT_ALIGNMENT",
    "ANCHOR_TREE_PATH",
    "ARTIFACT_CARRIER_IDENTITY_ANCHOR",
    "ARTIFACT_CARRIER_IDENTITY_ENVELOPE",
    "ARTIFACT_CARRIER_IDENTITY_MAGIC",
    "ARTIFACT_CARRIER_IDENTITY_PROJECTION",
    "ARTIFACT_CARRIER_IDENTITY_REFERENCE",
    "ARTIFACT_CARRIER_IDENTITY_REVISION",
    "ARTIFACT_CARRIER_IDENTITY_STRUCTURE_NODE",
    "ArtifactAnchor",
    "ArtifactEnvelope",
    "PROJECTION_GENERATION",
    "PROJECTION_REASONING",
    "PROJECTION_UNDERSTANDING",
    "RAW_UNIT_OCTET",
    "RAW_UNIT_UNICODE_SCALAR",
    "REFERENCE_ACCESS_BLOCKED",
    "REFERENCE_RESOLVED",
    "REFERENCE_UNRESOLVED",
    "REFERENCE_WITHDRAWN",
    "REVISION_MAP_ANCHOR",
    "REVISION_MAP_PROJECTION",
    "REVISION_MAP_REFERENCE",
    "REVISION_MAP_STRUCTURE_NODE",
    "artifact_anchor_identity",
    "artifact_carrier_local_kind",
    "artifact_carrier_source",
    "artifact_envelope_identity",
    "make_artifact_anchor",
    "make_artifact_envelope",
]
