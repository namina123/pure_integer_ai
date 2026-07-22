"""跨语言观察、语义对象和 Memory 的共享身份契约。

本模块只定义纯整数身份、owner、版本和逻辑时序，不负责具体对象行为或持久化迁移。
``SourceRef`` 与图节点引用分型，禁止把 Companion 记录伪装成 ``ConceptRef``。旧字符和
词形键只作为兼容投影保留，不能替代后续 LanguageAtom/Representation 权威身份。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.crosscut.guards.int_blocker import assert_int


def _require_nonnegative(value: int, *, where: str) -> int:
    if type(value) is not int:
        assert_int(value, _where=where)
    if value < 0:
        raise ValueError(f"{where} 必须为非负整数")
    return value


@dataclass(frozen=True, order=True)
class CorpusVersion:
    """训练或摄入语料版本。0 表示尚未声明的旧数据。"""

    value: int = 0

    def __post_init__(self) -> None:
        _require_nonnegative(self.value, where="CorpusVersion.value")


@dataclass(frozen=True, order=True)
class ParserVersion:
    """分词、解析和 occurrence/span 生成协议版本。"""

    value: int = 0

    def __post_init__(self) -> None:
        _require_nonnegative(self.value, where="ParserVersion.value")


@dataclass(frozen=True, order=True)
class PrimitiveVersion:
    """冻结逻辑原语和类型坐标版本。"""

    value: int = 0

    def __post_init__(self) -> None:
        _require_nonnegative(self.value, where="PrimitiveVersion.value")


@dataclass(frozen=True, order=True)
class CurriculumVersion:
    """断奶前课程顺序、阶段条件和阈值版本。"""

    value: int = 0

    def __post_init__(self) -> None:
        _require_nonnegative(self.value, where="CurriculumVersion.value")


@dataclass(frozen=True, order=True)
class VersionBundle:
    """对象身份和恢复 manifest 共用的四类版本。"""

    corpus: CorpusVersion = CorpusVersion()
    parser: ParserVersion = ParserVersion()
    primitive: PrimitiveVersion = PrimitiveVersion()
    curriculum: CurriculumVersion = CurriculumVersion()

    def stable_key(self) -> tuple[int, int, int, int]:
        """返回纯整数版本键。"""
        return (
            self.corpus.value,
            self.parser.value,
            self.primitive.value,
            self.curriculum.value,
        )


VISIBILITY_GLOBAL = 1
VISIBILITY_TENANT = 2
VISIBILITY_USER = 3
VISIBILITY_SESSION = 4
_VALID_VISIBILITIES = frozenset({
    VISIBILITY_GLOBAL,
    VISIBILITY_TENANT,
    VISIBILITY_USER,
    VISIBILITY_SESSION,
})


@dataclass(frozen=True, order=True)
class OwnerScope:
    """tenant、user、session 和 visibility 的统一隔离键。"""

    tenant_id: int = 0
    user_id: int = 0
    session_id: int = 0
    visibility: int = VISIBILITY_GLOBAL

    def __post_init__(self) -> None:
        _require_nonnegative(self.tenant_id, where="OwnerScope.tenant_id")
        _require_nonnegative(self.user_id, where="OwnerScope.user_id")
        _require_nonnegative(self.session_id, where="OwnerScope.session_id")
        assert_int(self.visibility, _where="OwnerScope.visibility")
        if self.visibility not in _VALID_VISIBILITIES:
            raise ValueError("OwnerScope.visibility 未注册")
        if self.visibility == VISIBILITY_GLOBAL:
            if self.tenant_id or self.user_id or self.session_id:
                raise ValueError("GLOBAL owner 不得携带 tenant/user/session")
        elif self.visibility == VISIBILITY_TENANT:
            if self.tenant_id <= 0 or self.user_id or self.session_id:
                raise ValueError("TENANT owner 只允许 tenant_id")
        elif self.visibility == VISIBILITY_USER:
            if self.tenant_id <= 0 or self.user_id <= 0 or self.session_id:
                raise ValueError("USER owner 必须携带 tenant_id 和 user_id")
        elif self.visibility == VISIBILITY_SESSION:
            if self.tenant_id <= 0 or self.user_id <= 0 or self.session_id <= 0:
                raise ValueError("SESSION owner 必须携带 tenant/user/session")

    def stable_key(self) -> tuple[int, int, int, int]:
        """返回 owner 的纯整数稳定键。"""
        return self.tenant_id, self.user_id, self.session_id, self.visibility


GLOBAL_OWNER_SCOPE = OwnerScope()


@dataclass(frozen=True, order=True)
class LogicalTime:
    """created、observed、used 三类无墙钟逻辑序。"""

    created_seq: int
    observed_seq: int = 0
    used_seq: int = 0

    def __post_init__(self) -> None:
        _require_nonnegative(self.created_seq, where="LogicalTime.created_seq")
        _require_nonnegative(self.observed_seq, where="LogicalTime.observed_seq")
        _require_nonnegative(self.used_seq, where="LogicalTime.used_seq")
        if self.observed_seq and self.observed_seq < self.created_seq:
            raise ValueError("observed_seq 不得早于 created_seq")
        if self.used_seq and self.used_seq < self.created_seq:
            raise ValueError("used_seq 不得早于 created_seq")

    def stable_key(self) -> tuple[int, int, int]:
        """返回逻辑时序键。"""
        return self.created_seq, self.observed_seq, self.used_seq


OBJECT_LEGACY_CHARACTER = 1
OBJECT_LEGACY_WORD_FORM = 2
# 旧常量只为直接导入兼容保留，新代码应使用带 LEGACY 的名称。
OBJECT_CHARACTER = OBJECT_LEGACY_CHARACTER
OBJECT_WORD_FORM = OBJECT_LEGACY_WORD_FORM
OBJECT_SENSE = 3
OBJECT_CONCEPT = 4
OBJECT_OCCURRENCE = 5
OBJECT_SPAN = 6
OBJECT_PROPOSITION = 7
OBJECT_HYPOTHESIS = 8
OBJECT_ARTIFACT = 9
OBJECT_SOURCE_RECORD = 10
OBJECT_LANGUAGE_BRANCH = 11
OBJECT_LANGUAGE_ATOM = 12
OBJECT_REPRESENTATION = 13
OBJECT_STRUCTURE_CONCEPT = 14
OBJECT_MINIMAL_INSTRUCTION = 15
OBJECT_ENTITY = 16
OBJECT_EVENT = 17
OBJECT_SET_EXPR = 18
OBJECT_VARIABLE = 19
OBJECT_ROLE = 20
OBJECT_BINDER = 21
OBJECT_CONTEXT_SCOPE = 22
OBJECT_ROLE_BINDING = 23

_TYPED_REF_KINDS = frozenset({
    OBJECT_CHARACTER,
    OBJECT_WORD_FORM,
    OBJECT_SENSE,
    OBJECT_CONCEPT,
    OBJECT_OCCURRENCE,
    OBJECT_SPAN,
    OBJECT_PROPOSITION,
    OBJECT_HYPOTHESIS,
    OBJECT_ARTIFACT,
    OBJECT_LANGUAGE_BRANCH,
    OBJECT_LANGUAGE_ATOM,
    OBJECT_REPRESENTATION,
    OBJECT_STRUCTURE_CONCEPT,
    OBJECT_MINIMAL_INSTRUCTION,
    OBJECT_ENTITY,
    OBJECT_EVENT,
    OBJECT_SET_EXPR,
    OBJECT_VARIABLE,
    OBJECT_ROLE,
    OBJECT_BINDER,
    OBJECT_CONTEXT_SCOPE,
    OBJECT_ROLE_BINDING,
})


@dataclass(frozen=True, order=True)
class TypedRef:
    """带对象类型、owner 和版本的整数引用。"""

    object_kind: int
    space_id: int
    local_id: int
    owner: OwnerScope = GLOBAL_OWNER_SCOPE
    versions: VersionBundle = VersionBundle()

    def __post_init__(self) -> None:
        assert_int(self.object_kind, self.space_id, self.local_id,
                   _where="TypedRef")
        if self.object_kind not in _TYPED_REF_KINDS:
            raise ValueError("TypedRef.object_kind 未注册或必须使用 SourceRef")
        if self.space_id <= 0 or self.local_id <= 0:
            raise ValueError("TypedRef 编址必须为正整数")

    def node_ref(self) -> tuple[int, int]:
        """显式降级为旧图节点引用。"""
        return self.space_id, self.local_id

    def stable_key(self) -> tuple[int, ...]:
        """返回包含类型、owner 和版本的稳定键。"""
        return (
            self.object_kind,
            self.space_id,
            self.local_id,
            *self.owner.stable_key(),
            *self.versions.stable_key(),
        )

    @classmethod
    def from_stable_key(cls, key: tuple[int, ...]) -> "TypedRef":
        """从固定长度完整整数键恢复分型引用，拒绝截断、尾随和布尔值。"""
        if not isinstance(key, tuple) or len(key) != 11:
            raise ValueError("TypedRef 稳定键长度非法")
        assert_int(*key, _where="TypedRef.stable_key")
        if any(type(value) is not int for value in key):
            raise ValueError("TypedRef 稳定键必须使用严格整数")
        return cls(
            key[0],
            key[1],
            key[2],
            OwnerScope(key[3], key[4], key[5], key[6]),
            VersionBundle(
                CorpusVersion(key[7]),
                ParserVersion(key[8]),
                PrimitiveVersion(key[9]),
                CurriculumVersion(key[10]),
            ),
        )


@dataclass(frozen=True, order=True)
class SourceRef:
    """Companion SourceRecord 的稳定引用，不是图节点端点。"""

    source_kind: int
    source_id: int
    document_id: int
    owner: OwnerScope
    versions: VersionBundle

    def __post_init__(self) -> None:
        assert_int(self.source_kind, self.source_id, self.document_id,
                   _where="SourceRef")
        if self.source_kind <= 0 or self.source_id <= 0 or self.document_id < 0:
            raise ValueError("SourceRef 的 kind/id 必须稳定且非负")

    def stable_key(self) -> tuple[int, ...]:
        """返回可用于 manifest 和派生对象的纯整数键。"""
        return (
            self.source_kind,
            self.source_id,
            self.document_id,
            *self.owner.stable_key(),
            *self.versions.stable_key(),
        )

    @classmethod
    def from_stable_key(cls, key: tuple[int, ...]) -> "SourceRef":
        """从固定长度完整整数键恢复来源引用，拒绝截断、尾随和布尔值。"""
        if not isinstance(key, tuple) or len(key) != 11:
            raise ValueError("SourceRef 稳定键长度非法")
        assert_int(*key, _where="SourceRef.stable_key")
        if any(type(value) is not int for value in key):
            raise ValueError("SourceRef 稳定键必须使用严格整数")
        return cls(
            key[0],
            key[1],
            key[2],
            OwnerScope(key[3], key[4], key[5], key[6]),
            VersionBundle(
                CorpusVersion(key[7]),
                ParserVersion(key[8]),
                PrimitiveVersion(key[9]),
                CurriculumVersion(key[10]),
            ),
        )


@dataclass(frozen=True, order=True)
class ObjectIdentity:
    """非编址对象的稳定身份键，components 只允许整数。"""

    object_kind: int
    components: tuple[int, ...]
    owner: OwnerScope = GLOBAL_OWNER_SCOPE
    versions: VersionBundle = VersionBundle()

    def __post_init__(self) -> None:
        assert_int(self.object_kind, *self.components, _where="ObjectIdentity")
        if self.object_kind not in _TYPED_REF_KINDS:
            raise ValueError("ObjectIdentity.object_kind 未注册")
        if not self.components:
            raise ValueError("ObjectIdentity.components 不能为空")

    def stable_key(self) -> tuple[int, ...]:
        """返回包含类型、owner、版本和组成部分的稳定键。"""
        return (
            self.object_kind,
            *self.owner.stable_key(),
            *self.versions.stable_key(),
            len(self.components),
            *self.components,
        )

    @classmethod
    def from_stable_key(cls, key: tuple[int, ...]) -> "ObjectIdentity":
        """从完整整数键恢复对象身份，拒绝截断、尾随和布尔值。"""
        if not isinstance(key, tuple) or len(key) < 11:
            raise ValueError("ObjectIdentity 稳定键长度非法")
        assert_int(*key, _where="ObjectIdentity.stable_key")
        if any(type(value) is not int for value in key):
            raise ValueError("ObjectIdentity 稳定键必须使用严格整数")
        component_size = key[9]
        if component_size <= 0 or len(key) != 10 + component_size:
            raise ValueError("ObjectIdentity components 长度非法")
        owner = OwnerScope(key[1], key[2], key[3], key[4])
        versions = VersionBundle(
            CorpusVersion(key[5]),
            ParserVersion(key[6]),
            PrimitiveVersion(key[7]),
            CurriculumVersion(key[8]),
        )
        return cls(key[0], key[10:], owner, versions)


def _identity_components(values: tuple[int, ...], *,
                         where: str) -> tuple[int, ...]:
    """校验外部注入的身份组成键，禁止空键和非整数值进入权威身份。"""
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{where} 必须是非空整数元组")
    assert_int(*values, _where=where)
    if any(type(value) is not int for value in values):
        raise ValueError(f"{where} 不得使用布尔值或整数子类")
    return values


def language_branch_identity(
        branch_key: tuple[int, ...], *,
        owner: OwnerScope = GLOBAL_OWNER_SCOPE,
        versions: VersionBundle = VersionBundle()) -> ObjectIdentity:
    """构造语言分支身份；branch_key 由课程或来源注册，不包含表层形式。"""
    components = _identity_components(
        branch_key, where="language_branch_identity.branch_key")
    return ObjectIdentity(
        OBJECT_LANGUAGE_BRANCH, components, owner, versions)


def language_atom_identity(
        branch: ObjectIdentity, atom_key: tuple[int, ...]) -> ObjectIdentity:
    """在既有语言分支内构造纯概念原子身份，并继承 branch 的 owner 和版本。"""
    if branch.object_kind != OBJECT_LANGUAGE_BRANCH:
        raise ValueError("language_atom_identity.branch 必须是语言分支身份")
    components = _identity_components(
        atom_key, where="language_atom_identity.atom_key")
    return ObjectIdentity(
        OBJECT_LANGUAGE_ATOM,
        (len(branch.components), *branch.components,
         len(components), *components),
        branch.owner,
        branch.versions,
    )


def representation_identity(
        family_key: tuple[int, ...], representation_key: tuple[int, ...], *,
        owner: OwnerScope = GLOBAL_OWNER_SCOPE,
        versions: VersionBundle = VersionBundle()) -> ObjectIdentity:
    """构造表示概念身份；表示族和内容均为注入键，身份不接受语言参数。"""
    family = _identity_components(
        family_key, where="representation_identity.family_key")
    representation = _identity_components(
        representation_key, where="representation_identity.representation_key")
    return ObjectIdentity(
        OBJECT_REPRESENTATION,
        (len(family), *family, len(representation), *representation),
        owner,
        versions,
    )


def concept_identity(
        concept_key: tuple[int, ...], *,
        owner: OwnerScope = GLOBAL_OWNER_SCOPE,
        versions: VersionBundle = VersionBundle()) -> ObjectIdentity:
    """构造不依赖 surface 的通用概念身份；具体含义由来源或图关系给出。"""
    components = _identity_components(
        concept_key, where="concept_identity.concept_key")
    return ObjectIdentity(OBJECT_CONCEPT, components, owner, versions)


def sense_identity(source: SourceRef, *,
                   sense_key: tuple[int, ...]) -> ObjectIdentity:
    """构造来源化词义身份；词义键不得退化为 token surface。"""
    components = _identity_components(
        sense_key, where="sense_identity.sense_key")
    return ObjectIdentity(
        OBJECT_SENSE,
        (*source.stable_key(), len(components), *components),
        source.owner,
        source.versions,
    )


def structure_concept_identity(
        structure_key: tuple[int, ...], *,
        owner: OwnerScope = GLOBAL_OWNER_SCOPE,
        versions: VersionBundle = VersionBundle()) -> ObjectIdentity:
    """构造可整体引用的结构概念身份，不绑定某次 occurrence 或派生签名。"""
    components = _identity_components(
        structure_key, where="structure_concept_identity.structure_key")
    return ObjectIdentity(
        OBJECT_STRUCTURE_CONCEPT, components, owner, versions)


def minimal_instruction_identity(
        instruction_key: tuple[int, ...], *,
        owner: OwnerScope = GLOBAL_OWNER_SCOPE,
        versions: VersionBundle = VersionBundle()) -> ObjectIdentity:
    """构造可选最小执行指令身份；具体操作含义必须由图定义或课程注入。"""
    components = _identity_components(
        instruction_key, where="minimal_instruction_identity.instruction_key")
    return ObjectIdentity(
        OBJECT_MINIMAL_INSTRUCTION, components, owner, versions)


def legacy_character_identity(
        codepoint: int, *, language: int,
        owner: OwnerScope = GLOBAL_OWNER_SCOPE,
        versions: VersionBundle = VersionBundle()) -> ObjectIdentity:
    """构造旧字符投影键；不得作为语言原子或 Unicode 表示的权威身份。"""
    assert_int(codepoint, language, _where="legacy_character_identity")
    return ObjectIdentity(
        OBJECT_LEGACY_CHARACTER, (language, codepoint), owner, versions)


def legacy_word_form_identity(
        codepoints: tuple[int, ...], *, language: int,
        owner: OwnerScope = GLOBAL_OWNER_SCOPE,
        versions: VersionBundle = VersionBundle()) -> ObjectIdentity:
    """构造旧词形投影键；有序码点完整保留但不定义语言原子本体。"""
    assert_int(language, *codepoints, _where="legacy_word_form_identity")
    if not codepoints:
        raise ValueError("词形码点不能为空")
    return ObjectIdentity(
        OBJECT_LEGACY_WORD_FORM,
        (language, len(codepoints), *codepoints),
        owner,
        versions,
    )


def character_identity(
        codepoint: int, *, language: int,
        owner: OwnerScope = GLOBAL_OWNER_SCOPE,
        versions: VersionBundle = VersionBundle()) -> ObjectIdentity:
    """兼容旧调用；新代码必须显式使用 legacy API 或等待 U-00 分型身份。"""
    return legacy_character_identity(
        codepoint, language=language, owner=owner, versions=versions)


def word_form_identity(
        codepoints: tuple[int, ...], *, language: int,
        owner: OwnerScope = GLOBAL_OWNER_SCOPE,
        versions: VersionBundle = VersionBundle()) -> ObjectIdentity:
    """兼容旧调用；返回值不是 LanguageAtom 或 Representation 权威身份。"""
    return legacy_word_form_identity(
        codepoints, language=language, owner=owner, versions=versions)


def occurrence_identity(source: SourceRef, *, start: int, end: int,
                        ordinal: int) -> ObjectIdentity:
    """构造来源内 occurrence 身份；span 和同位序号共同消歧。"""
    assert_int(start, end, ordinal, _where="occurrence_identity")
    if start < 0 or end < start or ordinal < 0:
        raise ValueError("occurrence span/ordinal 非法")
    return ObjectIdentity(
        OBJECT_OCCURRENCE,
        (*source.stable_key(), start, end, ordinal),
        source.owner,
        source.versions,
    )


def normalize_span_members(
        members: tuple[tuple[int, int], ...]
        ) -> tuple[tuple[int, int], ...]:
    """把 Span 的来源区间规范化为有序、不重叠的最小成员序列。"""
    if not isinstance(members, tuple) or not members:
        raise ValueError("span members 必须是非空区间 tuple")
    checked: list[tuple[int, int]] = []
    for index, member in enumerate(members):
        if not isinstance(member, tuple) or len(member) != 2:
            raise ValueError("span member 必须是二元区间")
        start, end = member
        assert_int(start, end, _where=f"span member[{index}]")
        if type(start) is not int or type(end) is not int:
            raise ValueError("span member 必须使用严格整数")
        if start < 0 or end < start:
            raise ValueError("span member 边界非法")
        checked.append((start, end))
    if any(start == end for start, end in checked):
        if len(checked) != 1:
            raise ValueError("零宽 span 不得与其他成员混合")
        return (checked[0],)
    merged: list[tuple[int, int]] = []
    for start, end in sorted(checked):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue
        previous_start, previous_end = merged[-1]
        merged[-1] = (previous_start, max(previous_end, end))
    return tuple(merged)


def span_identity(
        source: SourceRef, *, members: tuple[tuple[int, int], ...],
        ordinal: int = 0) -> ObjectIdentity:
    """构造来源内 Span 身份；结构类型和候选状态不得进入本体键。"""
    assert_int(ordinal, _where="span_identity.ordinal")
    if type(ordinal) is not int or ordinal < 0:
        raise ValueError("span ordinal 必须为非负严格整数")
    normalized = normalize_span_members(members)
    return ObjectIdentity(
        OBJECT_SPAN,
        (
            *source.stable_key(),
            ordinal,
            len(normalized),
            *(value for member in normalized for value in member),
        ),
        source.owner,
        source.versions,
    )


@dataclass(frozen=True)
class ObjectContract:
    """一种对象的唯一身份和持久化责任。"""

    object_kind: int
    persistence_owner: str
    authority: str
    source_required: bool
    owner_required: bool
    version_required: bool
    authoritative_identity: bool = True


OBJECT_CONTRACTS: tuple[ObjectContract, ...] = (
    ObjectContract(
        OBJECT_LEGACY_CHARACTER, "storage.concept_correspondence",
        "compatibility_character_projection", False, True, True,
        authoritative_identity=False),
    ObjectContract(
        OBJECT_LEGACY_WORD_FORM, "storage.word_form_index",
        "compatibility_word_form_projection", True, True, True,
        authoritative_identity=False),
    ObjectContract(OBJECT_SENSE, "storage.sense_candidates",
                   "sense_candidates", True, True, True),
    ObjectContract(OBJECT_CONCEPT, "storage.node_store",
                   "concept_node", False, True, True),
    ObjectContract(OBJECT_OCCURRENCE, "storage.occurrence",
                   "occurrence", True, True, True),
    ObjectContract(OBJECT_SPAN, "storage.span",
                   "span", True, True, True),
    ObjectContract(OBJECT_PROPOSITION, "storage.graph_object",
                   "semantic_proposition", True, True, True),
    ObjectContract(OBJECT_HYPOTHESIS, "storage.memory_event",
                   "memory_event", True, True, True),
    ObjectContract(OBJECT_ARTIFACT, "storage.artifact",
                   "artifact", True, True, True),
    ObjectContract(OBJECT_SOURCE_RECORD, "storage.source_record",
                   "source_record", False, True, True),
    ObjectContract(OBJECT_LANGUAGE_BRANCH, "storage.graph_object",
                   "language_branch", False, True, True),
    ObjectContract(OBJECT_LANGUAGE_ATOM, "storage.graph_object",
                   "language_atom", False, True, True),
    ObjectContract(OBJECT_REPRESENTATION, "storage.graph_object",
                   "representation_concept", False, True, True),
    ObjectContract(OBJECT_STRUCTURE_CONCEPT, "storage.graph_object",
                   "structure_concept", False, True, True),
    ObjectContract(OBJECT_MINIMAL_INSTRUCTION, "storage.graph_object",
                   "minimal_instruction", False, True, True),
    ObjectContract(OBJECT_ENTITY, "storage.graph_object",
                   "semantic_entity", True, True, True),
    ObjectContract(OBJECT_EVENT, "storage.graph_object",
                   "semantic_event", True, True, True),
    ObjectContract(OBJECT_SET_EXPR, "storage.graph_object",
                   "semantic_set_expression", True, True, True),
    ObjectContract(OBJECT_VARIABLE, "storage.graph_object",
                   "typed_variable", True, True, True),
    ObjectContract(OBJECT_ROLE, "storage.graph_object",
                   "semantic_role", False, True, True),
    ObjectContract(OBJECT_BINDER, "storage.graph_object",
                   "semantic_binder", True, True, True),
    ObjectContract(OBJECT_CONTEXT_SCOPE, "storage.graph_object",
                   "semantic_context_scope", True, True, True),
    ObjectContract(OBJECT_ROLE_BINDING, "storage.graph_object",
                   "semantic_role_binding", True, True, True),
)


def object_contracts_by_kind() -> dict[int, ObjectContract]:
    """按对象类型返回唯一持久化责任。"""
    return {contract.object_kind: contract for contract in OBJECT_CONTRACTS}


def validate_object_contracts() -> tuple[str, ...]:
    """检查对象类型和持久化责任没有重复或空缺。"""
    errors: list[str] = []
    seen: set[int] = set()
    for contract in OBJECT_CONTRACTS:
        if contract.object_kind in seen:
            errors.append(f"重复对象契约: {contract.object_kind}")
        seen.add(contract.object_kind)
        if not contract.persistence_owner or not contract.authority:
            errors.append(f"对象契约缺持久化责任: {contract.object_kind}")
        if contract.object_kind in {
                OBJECT_LEGACY_CHARACTER, OBJECT_LEGACY_WORD_FORM}:
            if contract.authoritative_identity:
                errors.append(f"兼容对象不得声明权威身份: {contract.object_kind}")
    expected = set(_TYPED_REF_KINDS) | {OBJECT_SOURCE_RECORD}
    if seen != expected:
        errors.append("对象契约未完整覆盖共享对象类型")
    return tuple(errors)


__all__ = [
    "CorpusVersion",
    "CurriculumVersion",
    "GLOBAL_OWNER_SCOPE",
    "LogicalTime",
    "OBJECT_ARTIFACT",
    "OBJECT_BINDER",
    "OBJECT_CONCEPT",
    "OBJECT_CONTRACTS",
    "OBJECT_CONTEXT_SCOPE",
    "OBJECT_ENTITY",
    "OBJECT_EVENT",
    "OBJECT_HYPOTHESIS",
    "OBJECT_LANGUAGE_ATOM",
    "OBJECT_LANGUAGE_BRANCH",
    "OBJECT_LEGACY_CHARACTER",
    "OBJECT_LEGACY_WORD_FORM",
    "OBJECT_MINIMAL_INSTRUCTION",
    "OBJECT_OCCURRENCE",
    "OBJECT_PROPOSITION",
    "OBJECT_REPRESENTATION",
    "OBJECT_ROLE",
    "OBJECT_ROLE_BINDING",
    "OBJECT_SENSE",
    "OBJECT_SET_EXPR",
    "OBJECT_SOURCE_RECORD",
    "OBJECT_SPAN",
    "OBJECT_STRUCTURE_CONCEPT",
    "OBJECT_VARIABLE",
    "ObjectContract",
    "ObjectIdentity",
    "OwnerScope",
    "ParserVersion",
    "PrimitiveVersion",
    "SourceRef",
    "TypedRef",
    "VersionBundle",
    "VISIBILITY_GLOBAL",
    "VISIBILITY_SESSION",
    "VISIBILITY_TENANT",
    "VISIBILITY_USER",
    "legacy_character_identity",
    "legacy_word_form_identity",
    "concept_identity",
    "language_atom_identity",
    "language_branch_identity",
    "minimal_instruction_identity",
    "normalize_span_members",
    "object_contracts_by_kind",
    "occurrence_identity",
    "representation_identity",
    "sense_identity",
    "span_identity",
    "structure_concept_identity",
    "validate_object_contracts",
]
