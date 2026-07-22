"""S-06 形式域 Artifact 的一等身份、类型和运行期调用协议。

本模块不解释算术、代码、单位名称或 VM opcode。Artifact kind、值类型、单位、
executor、verifier 及其版本都由调用方注入为一等图身份；payload 只是对应 adapter
负责解释的纯整数载荷。当前值和 invocation 带显式 scope，不在此处持久化。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_ARTIFACT,
    OBJECT_CONCEPT,
    OBJECT_PROPOSITION,
    OBJECT_ROLE,
    OBJECT_STRUCTURE_CONCEPT,
    OBJECT_VARIABLE,
    ObjectIdentity,
    SourceRef,
    object_contracts_by_kind,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.semantic_object import (
    describe_variable,
    semantic_source,
    validate_semantic_identity,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


_ARTIFACT_IDENTITY_VERSION = 1
_SOURCE_KEY_SIZE = 11
_CLASSIFIER_KINDS = frozenset({
    OBJECT_CONCEPT,
    OBJECT_STRUCTURE_CONCEPT,
    OBJECT_ROLE,
})


def _strict_tuple(
        value: tuple[int, ...], *, label: str, allow_empty: bool = False,
        ) -> tuple[int, ...]:
    """校验开放整数载荷或键，不允许 bool、浮点和整数子类混入。"""
    if not isinstance(value, tuple) or (not allow_empty and not value):
        qualifier = "整数 tuple" if allow_empty else "非空整数 tuple"
        raise ValueError(f"{label} 必须是{qualifier}")
    assert_int(*value, _where=label)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{label} 必须使用严格整数")
    return value


def _packed(key: tuple[int, ...]) -> tuple[int, ...]:
    """给可变长稳定键增加长度前缀，防止相邻身份段拼接歧义。"""
    return len(key), *key


def _take_packed(
        values: tuple[int, ...], cursor: int, *, label: str,
        allow_empty: bool = False,
        ) -> tuple[tuple[int, ...], int]:
    """读取一个长度前缀段，并拒绝截断、负长度和非法空段。"""
    if cursor >= len(values):
        raise ValueError(f"Artifact 身份缺少 {label} 长度")
    size = values[cursor]
    cursor += 1
    if size < 0 or (size == 0 and not allow_empty):
        raise ValueError(f"Artifact 身份 {label} 长度非法")
    if cursor + size > len(values):
        raise ValueError(f"Artifact 身份 {label} 被截断")
    return values[cursor:cursor + size], cursor + size


def _require_classifier(
        identity: ObjectIdentity, *, label: str,
        ) -> ObjectIdentity:
    """核验可充当 kind、type 或 unit 的一等概念层身份。"""
    if not isinstance(identity, ObjectIdentity):
        raise TypeError(f"{label} 必须是 ObjectIdentity")
    if identity.object_kind not in _CLASSIFIER_KINDS:
        raise ValueError(f"{label} 必须属于概念或结构坐标层")
    validate_semantic_identity(identity)
    if ObjectIdentity.from_stable_key(identity.stable_key()) != identity:
        raise ValueError(f"{label} 完整身份不能稳定 round-trip")
    return identity


def _require_authoritative(
        identity: ObjectIdentity, *, label: str,
        ) -> ObjectIdentity:
    """拒绝兼容投影或不完整身份充当 executor/verifier 权威点。"""
    if not isinstance(identity, ObjectIdentity):
        raise TypeError(f"{label} 必须是 ObjectIdentity")
    contract = object_contracts_by_kind().get(identity.object_kind)
    if contract is None or not contract.authoritative_identity:
        raise ValueError(f"{label} 必须是一等权威对象")
    if ObjectIdentity.from_stable_key(identity.stable_key()) != identity:
        raise ValueError(f"{label} 完整身份不能稳定 round-trip")
    return identity


@dataclass(frozen=True)
class ArtifactIdentityDescriptor:
    """从 Artifact 身份恢复的来源、kind、schema、scope、声明键和载荷。"""

    source: SourceRef
    artifact_kind: ObjectIdentity
    schema: "ArtifactSchema"
    scope: ScopeIdentity | None
    declaration_key: tuple[int, ...]
    payload: tuple[int, ...]


def artifact_identity(
        source: SourceRef, artifact_kind: ObjectIdentity,
        schema: "ArtifactSchema", declaration_key: tuple[int, ...],
        payload: tuple[int, ...] = (),
        scope: ScopeIdentity | None = None,
        ) -> ObjectIdentity:
    """构造来源化 Artifact 身份，完整保留 schema、scope、声明键和 payload。"""
    if not isinstance(source, SourceRef):
        raise TypeError("artifact source 必须是 SourceRef")
    _require_classifier(artifact_kind, label="artifact kind")
    if not isinstance(schema, ArtifactSchema):
        raise TypeError("artifact schema 必须是 ArtifactSchema")
    key = _strict_tuple(declaration_key, label="artifact declaration_key")
    payload = _strict_tuple(
        payload, label="artifact payload", allow_empty=True)
    if scope is not None:
        if not isinstance(scope, ScopeIdentity):
            raise TypeError("artifact scope 必须是 ScopeIdentity 或 None")
        if (scope.owner != source.owner
                or scope.versions != source.versions):
            raise ValueError("Artifact scope 与 source owner/version 不一致")
        if scope.source is not None and scope.source != source:
            raise ValueError("来源化 Artifact scope 必须指向同一 source")
    kind_key = artifact_kind.stable_key()
    type_key = schema.value_type.stable_key()
    unit_key = schema.unit.stable_key()
    scope_key = () if scope is None else scope.stable_key()
    return ObjectIdentity(
        OBJECT_ARTIFACT,
        (
            _ARTIFACT_IDENTITY_VERSION,
            *source.stable_key(),
            *_packed(kind_key),
            *_packed(type_key),
            *_packed(unit_key),
            *_packed(scope_key),
            *_packed(key),
            *_packed(payload),
        ),
        source.owner,
        source.versions,
    )


def describe_artifact_identity(
        identity: ObjectIdentity,
        ) -> ArtifactIdentityDescriptor:
    """完整解析 Artifact 身份并重建核验，拒绝只贴 object kind 的伪对象。"""
    if not isinstance(identity, ObjectIdentity):
        raise TypeError("artifact identity 必须是 ObjectIdentity")
    if identity.object_kind != OBJECT_ARTIFACT:
        raise ValueError("artifact identity 对象类型不匹配")
    values = identity.components
    if (len(values) <= 1 + _SOURCE_KEY_SIZE
            or values[0] != _ARTIFACT_IDENTITY_VERSION):
        raise ValueError("Artifact 身份版本或长度非法")
    source = SourceRef.from_stable_key(values[1:1 + _SOURCE_KEY_SIZE])
    cursor = 1 + _SOURCE_KEY_SIZE
    kind_key, cursor = _take_packed(values, cursor, label="kind")
    type_key, cursor = _take_packed(values, cursor, label="value type")
    unit_key, cursor = _take_packed(values, cursor, label="unit")
    scope_key, cursor = _take_packed(
        values, cursor, label="scope", allow_empty=True)
    declaration_key, cursor = _take_packed(
        values, cursor, label="declaration key")
    payload, cursor = _take_packed(
        values, cursor, label="payload", allow_empty=True)
    if cursor != len(values):
        raise ValueError("Artifact 身份含尾随数据")
    artifact_kind = ObjectIdentity.from_stable_key(kind_key)
    _require_classifier(artifact_kind, label="artifact kind")
    schema = ArtifactSchema(
        ObjectIdentity.from_stable_key(type_key),
        ObjectIdentity.from_stable_key(unit_key),
    )
    scope = None if not scope_key else ScopeIdentity.from_stable_key(scope_key)
    _strict_tuple(declaration_key, label="artifact declaration_key")
    _strict_tuple(payload, label="artifact payload", allow_empty=True)
    expected = artifact_identity(
        source, artifact_kind, schema, declaration_key, payload, scope)
    if expected != identity:
        raise ValueError("Artifact 嵌套身份与外层 owner/version 不一致")
    return ArtifactIdentityDescriptor(
        source, artifact_kind, schema, scope, declaration_key, payload)


@dataclass(frozen=True)
class ArtifactSchema:
    """Artifact 值的图中类型与单位；无量纲也必须传入明确单位身份。"""

    value_type: ObjectIdentity
    unit: ObjectIdentity

    def __post_init__(self) -> None:
        _require_classifier(self.value_type, label="artifact value_type")
        _require_classifier(self.unit, label="artifact unit")

    def stable_key(self) -> tuple[int, ...]:
        """返回包含完整类型和单位身份的无歧义稳定键。"""
        return (
            *_packed(self.value_type.stable_key()),
            *_packed(self.unit.stable_key()),
        )


@dataclass(frozen=True)
class ArtifactAuthority:
    """executor 或 verifier 的一等身份及其独立版本身份。"""

    identity: ObjectIdentity
    version: ObjectIdentity

    def __post_init__(self) -> None:
        _require_authoritative(self.identity, label="artifact authority")
        _require_authoritative(
            self.version, label="artifact authority version")

    def stable_key(self) -> tuple[int, ...]:
        """返回 authority 与版本的完整稳定键。"""
        return (
            *_packed(self.identity.stable_key()),
            *_packed(self.version.stable_key()),
        )


@dataclass(frozen=True)
class FormalArtifact:
    """带完整身份、类型、单位、来源、scope 和整数载荷的 Artifact。"""

    identity: ObjectIdentity
    artifact_kind: ObjectIdentity
    schema: ArtifactSchema
    source: SourceRef
    payload: tuple[int, ...]
    scope: ScopeIdentity | None = None

    def __post_init__(self) -> None:
        descriptor = describe_artifact_identity(self.identity)
        _require_classifier(self.artifact_kind, label="artifact kind")
        if descriptor.source != self.source:
            raise ValueError("Artifact identity 与 source 不一致")
        if descriptor.artifact_kind != self.artifact_kind:
            raise ValueError("Artifact identity 与 artifact_kind 不一致")
        if not isinstance(self.schema, ArtifactSchema):
            raise TypeError("Artifact schema 必须是 ArtifactSchema")
        if descriptor.schema != self.schema:
            raise ValueError("Artifact identity 与 schema 不一致")
        _strict_tuple(self.payload, label="artifact payload", allow_empty=True)
        if descriptor.payload != self.payload:
            raise ValueError("Artifact identity 与 payload 不一致")
        if descriptor.scope != self.scope:
            raise ValueError("Artifact identity 与 scope 不一致")
        if self.scope is not None:
            if not isinstance(self.scope, ScopeIdentity):
                raise TypeError("Artifact scope 必须是 ScopeIdentity 或 None")
            if (self.scope.owner != self.source.owner
                    or self.scope.versions != self.source.versions):
                raise ValueError("Artifact scope 与 source owner/version 不一致")
            if (self.scope.source is not None
                    and self.scope.source != self.source):
                raise ValueError("来源化 Artifact scope 必须指向同一 source")

    @property
    def declaration_key(self) -> tuple[int, ...]:
        """返回身份内经完整核验的开放声明键。"""
        return describe_artifact_identity(self.identity).declaration_key

    def stable_key(self) -> tuple[int, ...]:
        """返回已包含 schema、scope 和 payload 的完整 Artifact 身份键。"""
        return self.identity.stable_key()


@dataclass(frozen=True)
class ArtifactParameter:
    """程序参数的 Variable、类型/单位和 executor 私有绑定键。"""

    variable: ObjectIdentity
    schema: ArtifactSchema
    executor_binding: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.schema, ArtifactSchema):
            raise TypeError("Artifact parameter schema 必须是 ArtifactSchema")
        if (not isinstance(self.variable, ObjectIdentity)
                or self.variable.object_kind != OBJECT_VARIABLE):
            raise ValueError("Artifact parameter 必须是一等 Variable")
        descriptor = describe_variable(self.variable)
        if descriptor.value_type != self.schema.value_type:
            raise ValueError("Artifact parameter schema 与 Variable 类型不一致")
        _strict_tuple(
            self.executor_binding, label="ArtifactParameter.executor_binding")

    def stable_key(self) -> tuple[int, ...]:
        """返回参数身份、schema 和 adapter 绑定键。"""
        return (
            *_packed(self.variable.stable_key()),
            *_packed(self.schema.stable_key()),
            *_packed(self.executor_binding),
        )


@dataclass(frozen=True)
class FormalArtifactDefinition:
    """可调用程序 Artifact 的参数、结果、proof 和 authority 契约。"""

    program: FormalArtifact
    parameters: tuple[ArtifactParameter, ...]
    result_kind: ObjectIdentity
    result_schema: ArtifactSchema
    proof_kind: ObjectIdentity
    proof_schema: ArtifactSchema
    executor: ArtifactAuthority
    verifier: ArtifactAuthority

    def __post_init__(self) -> None:
        if not isinstance(self.program, FormalArtifact):
            raise TypeError("program 必须是 FormalArtifact")
        if not isinstance(self.parameters, tuple):
            raise TypeError("parameters 必须是 ArtifactParameter tuple")
        variables: set[ObjectIdentity] = set()
        bindings: set[tuple[int, ...]] = set()
        for parameter in self.parameters:
            if not isinstance(parameter, ArtifactParameter):
                raise TypeError("parameters 必须只含 ArtifactParameter")
            if parameter.variable in variables:
                raise ValueError("Artifact definition 参数 Variable 重复")
            if parameter.executor_binding in bindings:
                raise ValueError("Artifact definition executor_binding 重复")
            variables.add(parameter.variable)
            bindings.add(parameter.executor_binding)
        _require_classifier(self.result_kind, label="artifact result kind")
        _require_classifier(self.proof_kind, label="artifact proof kind")
        if not isinstance(self.result_schema, ArtifactSchema):
            raise TypeError("result_schema 必须是 ArtifactSchema")
        if not isinstance(self.proof_schema, ArtifactSchema):
            raise TypeError("proof_schema 必须是 ArtifactSchema")
        if not isinstance(self.executor, ArtifactAuthority):
            raise TypeError("executor 必须是 ArtifactAuthority")
        if not isinstance(self.verifier, ArtifactAuthority):
            raise TypeError("verifier 必须是 ArtifactAuthority")

    def stable_key(self) -> tuple[int, ...]:
        """返回程序、参数、结果和 authority 的完整调用契约键。"""
        parts: list[int] = [*_packed(self.program.stable_key())]
        parts.append(len(self.parameters))
        for parameter in self.parameters:
            parts.extend(_packed(parameter.stable_key()))
        for key in (
                self.result_kind.stable_key(),
                self.result_schema.stable_key(),
                self.proof_kind.stable_key(),
                self.proof_schema.stable_key(),
                self.executor.stable_key(),
                self.verifier.stable_key()):
            parts.extend(_packed(key))
        return tuple(parts)


@dataclass(frozen=True)
class ArtifactArgument:
    """一次 invocation 中显式指向参数 Variable 的当前 Artifact 值。"""

    parameter: ObjectIdentity
    value: FormalArtifact

    def __post_init__(self) -> None:
        if (not isinstance(self.parameter, ObjectIdentity)
                or self.parameter.object_kind != OBJECT_VARIABLE):
            raise ValueError("ArtifactArgument.parameter 必须是一等 Variable")
        describe_variable(self.parameter)
        if not isinstance(self.value, FormalArtifact):
            raise TypeError("ArtifactArgument.value 必须是 FormalArtifact")


@dataclass(frozen=True)
class ArtifactInvocation:
    """语言命题对形式程序的一次来源化、scoped 调用请求。"""

    proposition: ObjectIdentity
    definition: FormalArtifactDefinition
    arguments: tuple[ArtifactArgument, ...]
    source: SourceRef
    scope: ScopeIdentity
    invocation_key: tuple[int, ...]
    expected: FormalArtifact | None = None

    def __post_init__(self) -> None:
        if (not isinstance(self.proposition, ObjectIdentity)
                or self.proposition.object_kind != OBJECT_PROPOSITION):
            raise ValueError("Artifact invocation proposition 类型不匹配")
        if semantic_source(self.proposition) != self.source:
            raise ValueError("Artifact invocation proposition 与 source 不一致")
        if not isinstance(self.definition, FormalArtifactDefinition):
            raise TypeError("definition 必须是 FormalArtifactDefinition")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("Artifact invocation scope 必须是 ScopeIdentity")
        if (self.scope.owner != self.source.owner
                or self.scope.versions != self.source.versions):
            raise ValueError("Artifact invocation scope 与 source 不一致")
        if self.scope.source is not None and self.scope.source != self.source:
            raise ValueError("Artifact invocation scope 来源不一致")
        if not isinstance(self.arguments, tuple):
            raise TypeError("arguments 必须是 ArtifactArgument tuple")
        for argument in self.arguments:
            if not isinstance(argument, ArtifactArgument):
                raise TypeError("arguments 必须只含 ArtifactArgument")
            if argument.value.scope != self.scope:
                raise ValueError("Artifact argument 必须属于当前 invocation scope")
        _strict_tuple(self.invocation_key, label="ArtifactInvocation.invocation_key")
        if self.expected is not None:
            if not isinstance(self.expected, FormalArtifact):
                raise TypeError("expected 必须是 FormalArtifact 或 None")
            if self.expected.scope != self.scope:
                raise ValueError("expected Artifact 必须属于当前 invocation scope")

    def stable_key(self) -> tuple[int, ...]:
        """返回命题、程序契约、参数、来源和 scope 的完整审计键。"""
        values: list[int] = []
        for key in (
                self.proposition.stable_key(),
                self.definition.stable_key()):
            values.extend(_packed(key))
        values.append(len(self.arguments))
        for argument in self.arguments:
            values.extend(_packed(argument.parameter.stable_key()))
            values.extend(_packed(argument.value.stable_key()))
        values.extend(_packed(self.source.stable_key()))
        values.extend(_packed(self.scope.stable_key()))
        values.extend(_packed(self.invocation_key))
        expected_key = () if self.expected is None else self.expected.stable_key()
        values.extend(_packed(expected_key))
        return tuple(values)


@dataclass(frozen=True)
class ArtifactCompatibilityResult:
    """type/unit resolver 的定向三态结论、依据及 adapter 换算载荷。"""

    expected: ObjectIdentity
    actual: ObjectIdentity
    compatible: bool | None
    support: tuple[ObjectIdentity, ...] = ()
    adapter_payload: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        _require_classifier(self.expected, label="compatibility expected")
        _require_classifier(self.actual, label="compatibility actual")
        if self.compatible is not None and type(self.compatible) is not bool:
            raise TypeError("compatible 必须是严格 bool 或 None")
        if not isinstance(self.support, tuple):
            raise TypeError("compatibility support 必须是 ObjectIdentity tuple")
        for identity in self.support:
            _require_authoritative(identity, label="compatibility support")
        _strict_tuple(
            self.adapter_payload,
            label="compatibility adapter_payload",
            allow_empty=True,
        )


class ArtifactCompatibilityResolver(Protocol):
    """由图关系或受限 verifier 实现的 type/unit 三态兼容协议。"""

    def resolve(
            self, expected: ObjectIdentity, actual: ObjectIdentity,
            ) -> ArtifactCompatibilityResult:
        """判断 actual 能否填充 expected；无依据时必须返回 unknown。"""
        ...


class ExactArtifactCompatibilityResolver:
    """只承认完整身份相等，其他组合保持 unknown 的保守 resolver。"""

    def resolve(
            self, expected: ObjectIdentity, actual: ObjectIdentity,
            ) -> ArtifactCompatibilityResult:
        """比较完整 ObjectIdentity，不从名称或 payload 推断兼容关系。"""
        _require_classifier(expected, label="compatibility expected")
        _require_classifier(actual, label="compatibility actual")
        return ArtifactCompatibilityResult(
            expected, actual, True if expected == actual else None)


__all__ = [
    "ArtifactArgument",
    "ArtifactAuthority",
    "ArtifactCompatibilityResolver",
    "ArtifactCompatibilityResult",
    "ArtifactIdentityDescriptor",
    "ArtifactInvocation",
    "ArtifactParameter",
    "ArtifactSchema",
    "ExactArtifactCompatibilityResolver",
    "FormalArtifact",
    "FormalArtifactDefinition",
    "artifact_identity",
    "describe_artifact_identity",
]
