"""generation candidate alias 请求到正式 R-01 course manifest 的纯构造层。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.alias_resolution import (
    AliasResolutionProtocol,
)
from pure_integer_ai.cognition.shared.candidate_projection import (
    CandidateProjectionProtocol,
)
from pure_integer_ai.cognition.shared.candidate_runtime import (
    CandidateProjectionMetadata,
)
from pure_integer_ai.cognition.shared.candidate_verifier import (
    IndependentVerifierProtocol,
    RevealedObjectObservation,
)
from pure_integer_ai.cognition.shared.evidence_candidate import (
    EvidenceCandidateProtocol,
)
from pure_integer_ai.cognition.shared.graph_ontology import (
    relation_concept_identity,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_LANGUAGE_BRANCH,
    OBJECT_PROPOSITION,
    OBJECT_REPRESENTATION,
    ObjectIdentity,
    SourceRef,
    concept_identity,
    minimal_instruction_identity,
    occurrence_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.relation_closure import (
    RelationClosureCandidateSpec,
    RelationClosureField,
    RelationClosureProtocol,
)
from pure_integer_ai.cognition.shared.relation_use import (
    RelationUseGraphProtocol,
    RelationUseWriteMetadata,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    ScopeIdentity,
    document_scope,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicPropositionDefinition,
    AtomicRoleBinding,
    context_scope_identity,
    proposition_identity,
    role_identity,
)
from pure_integer_ai.cognition.shared.typed_relation import (
    RelationSchema,
    RelationSlotSchema,
)
from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)
from pure_integer_ai.experiments.alias_relation_course import (
    AliasRelationCourseEntry,
    AliasRelationCourseManifest,
    AliasRelationCourseRecognition,
    AliasRelationStatementMetadata,
)
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.experiments.ph2_generation_candidate_alias_contract import (
    GenerationCandidateAliasCourseRequest,
    GenerationCandidateAliasRuntimeError,
)
from pure_integer_ai.experiments.ph2_generation_candidate_pack import (
    GenerationCandidatePack,
)
from pure_integer_ai.experiments.relation_closure_runtime import (
    RelationClosureRecognitionInput,
)
from pure_integer_ai.storage.edge_store import EPI_STRUCTURED, SOURCE_BARE_TEXT


_NAMESPACE = 22020


def _packed(value: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长稳定键增加长度边界。"""
    return len(value), *value


def _stable_positive_int(values: tuple[int, ...], *, domain: str) -> int:
    """把完整整数键压缩为稳定正整数 SourceRef id。"""
    fingerprint = integer_tuple_fingerprint(values, domain=domain)
    value = int.from_bytes(bytes(fingerprint[2:10]), "big")
    value &= (1 << 63) - 1
    return value if value > 0 else 1


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class _Protocols:
    """保存同一 candidate pack 派生的全部 R-01 开放协议。"""

    semantic_predicates: tuple[ObjectIdentity, ...]
    candidate_projection: CandidateProjectionProtocol
    learning: EvidenceCandidateProtocol
    verifier: IndependentVerifierProtocol
    relation: RelationClosureProtocol
    alias: AliasResolutionProtocol
    use: RelationUseGraphProtocol
    schemas: tuple[RelationSchema, ...]


# object-model: value; representation=struct; interop=R-01-manifest-profile-v1
@dataclass(frozen=True, slots=True)
class AliasRelationManifestProfile:
    """R-01 manifest 的语言无关输入身份，不携带训练 surface 或课程标签。

    该 profile 只定义协议 namespace、来源/作用域归属、形成来源下限和内容摘要。
    ``GenerationCandidatePack`` 与公开 V3 source projection 都可各自构造一个 profile，
    但两者必须在此后的 manifest 路径共享同一整数/字节语义。
    """

    candidate_version: tuple[int, ...]
    content_sha256: tuple[int, ...]
    owner_source: SourceRef
    owner_scope: ScopeIdentity
    minimum_forming_sources: int = 1

    def __post_init__(self) -> None:
        """冻结所有可观察 R-01 profile 输入，并拒绝 bool 伪整数。"""
        if (not isinstance(self.candidate_version, tuple)
                or not self.candidate_version
                or any(type(item) is not int or item <= 0
                       for item in self.candidate_version)):
            raise GenerationCandidateAliasRuntimeError(
                "alias manifest profile version 非法")
        if (not isinstance(self.content_sha256, tuple)
                or len(self.content_sha256) != 32
                or any(type(item) is not int or item < 0 or item > 255
                       for item in self.content_sha256)):
            raise GenerationCandidateAliasRuntimeError(
                "alias manifest profile content SHA-256 非法")
        if not isinstance(self.owner_source, SourceRef):
            raise TypeError("alias manifest profile owner source 类型错误")
        if (not isinstance(self.owner_scope, ScopeIdentity)
                or self.owner_scope.source != self.owner_source):
            raise GenerationCandidateAliasRuntimeError(
                "alias manifest profile owner scope 未绑定 owner source")
        if (type(self.minimum_forming_sources) is not int
                or self.minimum_forming_sources <= 0):
            raise GenerationCandidateAliasRuntimeError(
                "alias manifest profile forming source 下限非法")

    def sha256(self) -> str:
        """返回 profile 已冻结的 raw SHA-256 十六进制表示。"""
        return bytes(self.content_sha256).hex()

    def canonical_record(self) -> tuple[int, ...]:
        """导出完整有序 profile record，供其他语言重放同一 manifest。"""
        result = [1, len(self.candidate_version), *self.candidate_version]
        for value in (
                self.content_sha256,
                self.owner_source.stable_key(),
                self.owner_scope.stable_key()):
            result.extend((len(value), *value))
        result.append(self.minimum_forming_sources)
        return tuple(result)


def _profile_for_pack(
        pack: GenerationCandidatePack,
        ) -> AliasRelationManifestProfile:
    """把旧 candidate pack 逐字段投影为中性 R-01 manifest profile。"""
    if not isinstance(pack, GenerationCandidatePack):
        raise TypeError("generation alias manifest pack 类型错误")
    return AliasRelationManifestProfile(
        pack.candidate_version,
        tuple(bytes.fromhex(pack.sha256())),
        pack.owner_source,
        pack.owner_scope,
        pack.minimum_forming_sources,
    )


def _prefix(profile: AliasRelationManifestProfile) -> tuple[int, ...]:
    """从 profile 内容锁和候选版本建立协议身份前缀。"""
    if not isinstance(profile, AliasRelationManifestProfile):
        raise TypeError("alias manifest profile 类型错误")
    digest = profile.content_sha256
    return (
        _NAMESPACE,
        len(profile.candidate_version),
        *profile.candidate_version,
        len(digest),
        *digest,
    )


def _concept(pack: GenerationCandidatePack, *suffix: int) -> ObjectIdentity:
    """建立继承 pack owner/version 的一等 Concept。"""
    return concept_identity(
        (*_prefix(pack), *suffix),
        owner=pack.owner_source.owner,
        versions=pack.owner_source.versions,
    )


def _structure(pack: GenerationCandidatePack, *suffix: int) -> ObjectIdentity:
    """建立继承 pack owner/version 的一等 StructureConcept。"""
    return structure_concept_identity(
        (*_prefix(pack), *suffix),
        owner=pack.owner_source.owner,
        versions=pack.owner_source.versions,
    )


def _role(pack: GenerationCandidatePack, *suffix: int) -> ObjectIdentity:
    """建立继承 pack owner/version 的一等 Role。"""
    return role_identity(
        (*_prefix(pack), *suffix),
        owner=pack.owner_source.owner,
        versions=pack.owner_source.versions,
    )


def _instruction(
        pack: GenerationCandidatePack, *suffix: int,
        ) -> ObjectIdentity:
    """建立继承 pack owner/version 的一等 MinimalInstruction。"""
    return minimal_instruction_identity(
        (*_prefix(pack), *suffix),
        owner=pack.owner_source.owner,
        versions=pack.owner_source.versions,
    )


def _protocols(
        pack: GenerationCandidatePack,
        request: GenerationCandidateAliasCourseRequest,
        ) -> _Protocols:
    """从 pack policy 与本次 origin kinds 构造稳定 R-01 协议和 schema。"""
    semantic = tuple(
        relation_concept_identity(
            (*_prefix(pack), 1, ordinal),
            owner=pack.owner_source.owner,
            versions=pack.owner_source.versions,
        )
        for ordinal in range(1, 7)
    )
    projection_values = tuple(
        _concept(pack, 2, ordinal) for ordinal in range(1, 14))
    projection = CandidateProjectionProtocol(
        *projection_values, (*_prefix(pack), 2, 20))
    learning = EvidenceCandidateProtocol(
        (*_prefix(pack), 3, 1),
        (*_prefix(pack), 3, 2),
        pack.owner_source,
        pack.owner_scope,
        pack.minimum_forming_sources,
    )
    verifier = IndependentVerifierProtocol(
        _concept(pack, 4, 1),
        (*_prefix(pack), 4, 2),
        (*_prefix(pack), 4, 3),
        (*_prefix(pack), 4, 4),
        (*_prefix(pack), 4, 5),
    )
    relation = RelationClosureProtocol(
        RelationClosureField(_concept(pack, 5, 1)),
        RelationClosureField(_concept(pack, 5, 2)),
    )
    alias_rel = _concept(pack, 6, 1)
    refers_rel = _concept(pack, 6, 2)
    realizes_rel = _concept(pack, 6, 3)
    alias_roles = (_role(pack, 7, 1), _role(pack, 7, 2))
    refers_roles = (_role(pack, 7, 3), _role(pack, 7, 4))
    realizes_roles = (
        _role(pack, 7, 5), _role(pack, 7, 6), _role(pack, 7, 7))
    alias_schema_id = _structure(pack, 8, 1)
    refers_schema_id = _structure(pack, 8, 2)
    realizes_schema_id = _structure(pack, 8, 3)
    alias_schema = RelationSchema(
        alias_schema_id,
        alias_rel,
        (
            RelationSlotSchema(
                alias_roles[0], frozenset({OBJECT_PROPOSITION}), 1, 1),
            RelationSlotSchema(
                alias_roles[1], frozenset({OBJECT_PROPOSITION}), 1, 1),
        ),
    )
    refers_schema = RelationSchema(
        refers_schema_id,
        refers_rel,
        (
            RelationSlotSchema(
                refers_roles[0], frozenset({OBJECT_PROPOSITION}), 1, 1),
            RelationSlotSchema(
                refers_roles[1], frozenset({OBJECT_PROPOSITION}), 1, 1),
        ),
    )
    origin_kinds = frozenset(
        item.origin.object_kind for item in request.realizations)
    realizes_schema = RelationSchema(
        realizes_schema_id,
        realizes_rel,
        (
            RelationSlotSchema(
                realizes_roles[0], origin_kinds, 1, 1),
            RelationSlotSchema(
                realizes_roles[1], frozenset({OBJECT_REPRESENTATION}), 1, 1),
            RelationSlotSchema(
                realizes_roles[2], frozenset({OBJECT_LANGUAGE_BRANCH}), 1, 1),
        ),
    )
    alias = AliasResolutionProtocol(
        alias_rel,
        (alias_schema_id,),
        *alias_roles,
        _instruction(pack, 9, 1),
        refers_rel,
        (refers_schema_id,),
        *refers_roles,
        _instruction(pack, 9, 2),
        realizes_rel,
        (realizes_schema_id,),
        *realizes_roles,
        _instruction(pack, 9, 3),
        _instruction(pack, 9, 4),
        _instruction(pack, 9, 5),
        _instruction(pack, 9, 6),
    )
    use_values = tuple(
        _concept(pack, 10, ordinal) for ordinal in range(1, 9))
    use = RelationUseGraphProtocol(
        *use_values, (*_prefix(pack), 10, 20))
    return _Protocols(
        semantic,
        projection,
        learning,
        verifier,
        relation,
        alias,
        use,
        (alias_schema, refers_schema, realizes_schema),
    )


def _source(
        pack: GenerationCandidatePack,
        purpose: int,
        values: tuple[int, ...],
        ) -> SourceRef:
    """从 pack owner、用途和完整输入键导出独立来源身份。"""
    source_id = _stable_positive_int(
        (purpose, *_packed(values)),
        domain="gg03.generation.candidate.alias.source.v1",
    )
    return SourceRef(
        _NAMESPACE + purpose,
        source_id,
        0,
        pack.owner_source.owner,
        pack.owner_source.versions,
    )


def _forming_sources(
        pack: GenerationCandidatePack,
        keys: tuple[tuple[int, ...], ...],
        ) -> tuple[SourceRef, ...]:
    """把外部 Evidence stable key 映射为互异 pack-owned SourceRef。"""
    sources = tuple(sorted(
        (_source(pack, 20, key) for key in keys),
        key=SourceRef.stable_key,
    ))
    if len(set(sources)) != len(keys):
        raise GenerationCandidateAliasRuntimeError(
            "forming Evidence SourceRef 映射发生碰撞")
    return sources


def _entry(
        pack: GenerationCandidatePack,
        protocols: _Protocols,
        request_key: tuple[int, ...],
        ordinal: int,
        relation: ObjectIdentity,
        schema: RelationSchema,
        role_fillers: tuple[tuple[ObjectIdentity, ObjectIdentity], ...],
        forming_evidence_keys: tuple[tuple[int, ...], ...],
        ) -> AliasRelationCourseEntry:
    """构造一条来源化 relation spec、独立 reveal 和可重放逻辑序。"""
    relation_key = (
        *_packed(request_key),
        ordinal,
        *_packed(relation.stable_key()),
    )
    statement_source = _source(pack, 21, relation_key)
    proposition = AtomicPropositionDefinition(
        proposition_identity(
            statement_source,
            (*_prefix(pack), 30, ordinal)),
        relation,
        occurrence_identity(
            statement_source, start=0, end=1, ordinal=0),
        context_scope_identity(
            statement_source, (*_prefix(pack), 31, ordinal)),
        tuple(
            AtomicRoleBinding(role, filler, index)
            for index, (role, filler) in enumerate(role_fillers)
        ),
    )
    spec = RelationClosureCandidateSpec(
        proposition,
        schema,
        (*_prefix(pack), 32, ordinal, *_packed(request_key)),
        _forming_sources(pack, forming_evidence_keys),
    )
    observation = _source(pack, 22, relation_key)
    scope = document_scope(observation)
    event_key = (*_prefix(pack), 33, ordinal)
    anchor = occurrence_identity(observation, start=0, end=1, ordinal=0)
    recognition = RelationClosureRecognitionInput(
        proposition.proposition,
        observation,
        scope,
        ProtocolKey((*_prefix(pack), 34, ordinal)),
        event_key,
        anchor,
        (anchor,),
        RevealedObjectObservation(
            observation,
            scope,
            event_key,
            _source(pack, 23, relation_key),
            supported_targets=(proposition.proposition,),
            trace=(*_prefix(pack), 35, ordinal),
        ),
    )
    timestamp_base = ordinal * 10
    return AliasRelationCourseEntry(
        spec,
        document_scope(statement_source),
        timestamp_base,
        (AliasRelationCourseRecognition(
            recognition,
            timestamp_base + 3,
            timestamp_base + 4,
            timestamp_base + 5,
        ),),
    )


def build_alias_relation_manifest(
        profile: AliasRelationManifestProfile,
        request: GenerationCandidateAliasCourseRequest,
        ) -> AliasRelationCourseManifest:
    """将中性 profile 与来源化请求物化为可由 R-01 Loader 恢复的 manifest。"""
    if not isinstance(profile, AliasRelationManifestProfile):
        raise TypeError("alias relation manifest profile 类型错误")
    if not isinstance(request, GenerationCandidateAliasCourseRequest):
        raise TypeError("generation alias manifest request 类型错误")
    protocols = _protocols(profile, request)
    request_key = request.stable_key()
    schema_by_relation = {
        protocols.alias.alias_relation: protocols.schemas[0],
        protocols.alias.refers_relation: protocols.schemas[1],
        protocols.alias.realizes_relation: protocols.schemas[2],
    }
    entries = []
    ordinal = 0
    for item in request.references:
        ordinal += 1
        entries.append(_entry(
            profile,
            protocols,
            request_key,
            ordinal,
            protocols.alias.refers_relation,
            schema_by_relation[protocols.alias.refers_relation],
            (
                (protocols.alias.refers_from_role, item.origin),
                (protocols.alias.refers_to_role, item.target),
            ),
            item.forming_evidence_keys,
        ))
    for item in request.realizations:
        ordinal += 1
        entries.append(_entry(
            profile,
            protocols,
            request_key,
            ordinal,
            protocols.alias.realizes_relation,
            schema_by_relation[protocols.alias.realizes_relation],
            (
                (protocols.alias.realizes_bearer_role, item.origin),
                (protocols.alias.realizes_representation_role,
                 item.representation),
                (protocols.alias.realizes_branch_role, request.branch),
            ),
            item.forming_evidence_keys,
        ))
    course_fingerprint = integer_tuple_fingerprint(
        request_key,
        domain="gg03.generation.candidate.alias.course.v1",
    )
    content_version = profile.candidate_version[-1]
    return AliasRelationCourseManifest(
        1,
        (*profile.candidate_version, *course_fingerprint),
        protocols.semantic_predicates,
        protocols.candidate_projection,
        (*_prefix(profile), 40),
        protocols.learning,
        protocols.verifier,
        CandidateProjectionMetadata(
            SOURCE_BARE_TEXT,
            EPI_STRUCTURED,
            content_version=content_version,
        ),
        AliasRelationStatementMetadata(
            SOURCE_BARE_TEXT,
            EPI_STRUCTURED,
            content_version=content_version,
        ),
        protocols.relation,
        protocols.schemas,
        protocols.alias,
        protocols.use,
        RelationUseWriteMetadata(
            SOURCE_BARE_TEXT,
            EPI_STRUCTURED,
            content_version=content_version,
        ),
        tuple(entries),
    )


def build_generation_candidate_alias_manifest(
        pack: GenerationCandidatePack,
        request: GenerationCandidateAliasCourseRequest,
        ) -> AliasRelationCourseManifest:
    """保持旧 pack 入口的内容等价 wrapper，不改变既有 R-01 manifest。"""
    return build_alias_relation_manifest(_profile_for_pack(pack), request)


__all__ = [
    "AliasRelationManifestProfile",
    "build_alias_relation_manifest",
    "build_generation_candidate_alias_manifest",
]
