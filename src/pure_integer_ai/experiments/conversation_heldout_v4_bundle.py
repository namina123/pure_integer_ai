"""DLG-05 v4 的无标签 source bundle 整数合同。

v4 bundle 从公开构造点正向携带完整 ``QuestionRequest``、Representation、
候选、Evidence 和 ``SourceRef`` 原文。它是供冻结和人类阅读投影使用的只读
输入，不是 Core、Memory、会话数据库或答案标签存储。所有文本在合同内都以
Unicode scalar tuple 保存；``text`` 属性只是临时阅读投影。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Callable, Iterable

from pure_integer_ai.cognition.shared.generation_plan import GenerationCandidate
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_REPRESENTATION,
    SourceRef,
    ObjectIdentity,
)
from pure_integer_ai.cognition.shared.question_answer import (
    QuestionExecutionResult,
    QuestionRequest,
)
from pure_integer_ai.cognition.shared.representation_rendering import (
    RenderedSurface,
    representation_parts,
)
from pure_integer_ai.cognition.shared.hypothesis import EvidenceRecord
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


class ConversationHeldOutV4BundleError(ValueError):
    """v4 source bundle 的完整性、来源链或规范载荷不闭合。"""


def _strict_ints(value: tuple[int, ...], *, label: str,
                 allow_empty: bool = False) -> tuple[int, ...]:
    """核验严格整数 tuple，拒绝 bool、浮点和隐式转换。"""
    if not isinstance(value, tuple) or (not allow_empty and not value):
        raise ConversationHeldOutV4BundleError(
            f"{label} 必须是{'非空' if not allow_empty else ''}整数 tuple")
    assert_int(*value, _where=label)
    if any(type(item) is not int for item in value):
        raise ConversationHeldOutV4BundleError(f"{label} 必须使用严格整数")
    return value


def unicode_scalars(value: str, *, allow_empty: bool = True) -> tuple[int, ...]:
    """把文本转换为未经归一化的 Unicode scalar 序列。"""
    if not isinstance(value, str):
        raise TypeError("文本必须是字符串")
    result = tuple(ord(item) for item in value)
    return _validate_scalars(result, label="文本 Unicode scalar", allow_empty=allow_empty)


def scalars_text(value: tuple[int, ...], *, allow_empty: bool = True) -> str:
    """把已核验 scalar 序列转换为临时宿主字符串，不写回 bundle。"""
    return "".join(chr(item) for item in _validate_scalars(
        value, label="Unicode scalar", allow_empty=allow_empty))


def _validate_scalars(value: tuple[int, ...], *, label: str,
                      allow_empty: bool) -> tuple[int, ...]:
    """校验 Unicode scalar 范围，不执行 NFC/NFKC 或其它隐式变换。"""
    _strict_ints(value, label=label, allow_empty=allow_empty)
    for item in value:
        if (item < 0 or item > 0x10FFFF
                or 0xD800 <= item <= 0xDFFF):
            raise ConversationHeldOutV4BundleError(
                f"{label} 含非 Unicode scalar value")
    return value


def _digest_bytes(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """校验 SHA-256 字节序列，以整数 tuple 保存而非十六进制字符串。"""
    if not isinstance(value, tuple) or len(value) != 32:
        raise ConversationHeldOutV4BundleError(f"{label} 必须是 32 字节 tuple")
    _strict_ints(value, label=label)
    if any(item < 0 or item > 255 for item in value):
        raise ConversationHeldOutV4BundleError(f"{label} 含非法字节")
    return value


def digest_from_hex(value: str, *, label: str = "sha256") -> tuple[int, ...]:
    """把外部 manifest 的 SHA-256 十六进制值转换为整数 tuple。"""
    if not isinstance(value, str):
        raise TypeError(f"{label} 必须是十六进制字符串")
    try:
        raw = bytes.fromhex(value)
    except ValueError as exc:
        raise ConversationHeldOutV4BundleError(f"{label} 十六进制非法") from exc
    if len(raw) != 32:
        raise ConversationHeldOutV4BundleError(f"{label} 必须是 SHA-256")
    return tuple(raw)


def digest_hex(value: tuple[int, ...]) -> str:
    """仅为审计/阅读把整数 digest 转成十六进制，不作为权威身份。"""
    return bytes(_digest_bytes(value, label="digest")).hex()


def _canonical_bytes(payload: tuple[int, ...]) -> bytes:
    """用确定性有符号整数 framing 生成规范 hash 输入。"""
    values = _strict_ints(payload, label="canonical payload")
    result = bytearray(b"PIA-DLG05-V4\x00")
    result.extend(len(values).to_bytes(8, "big", signed=False))
    for item in values:
        sign = 1 if item < 0 else 0
        magnitude = abs(item)
        size = max(1, (magnitude.bit_length() + 7) // 8)
        result.extend((sign,))
        result.extend(size.to_bytes(8, "big", signed=False))
        result.extend(magnitude.to_bytes(size, "big", signed=False))
    return bytes(result)


def _payload_identity(payload: tuple[int, ...]) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    """计算 payload size、SHA-256 和正整数 index。"""
    raw = hashlib.sha256(_canonical_bytes(payload)).digest()
    digest = tuple(raw)
    index = int.from_bytes(raw[:8], "big", signed=False) or 1
    return len(payload), digest, (index,)


def _pack(result: list[int], value: tuple[int, ...]) -> None:
    """给可变长度整数段加边界，避免规范载荷发生拼接碰撞。"""
    result.extend((len(value), *value))


def _key(value: ProtocolKey, *, label: str) -> ProtocolKey:
    """核验非空协议键。"""
    if not isinstance(value, ProtocolKey):
        raise ConversationHeldOutV4BundleError(f"{label} 必须是 ProtocolKey")
    return value


def _identity(value: ObjectIdentity, *, label: str) -> ObjectIdentity:
    """核验一等对象身份。"""
    if not isinstance(value, ObjectIdentity):
        raise ConversationHeldOutV4BundleError(f"{label} 必须是 ObjectIdentity")
    return value


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4DependencyBinding:
    """v4 bundle 依赖的 artifact、inventory、document 三个完整 SHA。"""

    artifact_sha256: tuple[int, ...]
    inventory_sha256: tuple[int, ...]
    document_sha256: tuple[int, ...]

    def __post_init__(self) -> None:
        """拒绝缺失或截断的 freeze 依赖绑定。"""
        for name in ("artifact_sha256", "inventory_sha256", "document_sha256"):
            _digest_bytes(getattr(self, name), label=name)

    def stable_key(self) -> tuple[int, ...]:
        """返回三个依赖 digest 的完整整数键。"""
        result = [1]
        for value in (self.artifact_sha256, self.inventory_sha256,
                      self.document_sha256):
            _pack(result, value)
        return tuple(result)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4Representation:
    """一个 Representation 到原序 Unicode scalar 的可逆映射。"""

    representation: ObjectIdentity
    ordinal: int
    scalars: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验 Representation 类型、序号和其权威 content payload。"""
        if (not isinstance(self.representation, ObjectIdentity)
                or self.representation.object_kind != OBJECT_REPRESENTATION):
            raise ConversationHeldOutV4BundleError(
                "representation 必须是一等 Representation")
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ConversationHeldOutV4BundleError("representation ordinal 非法")
        _validate_scalars(self.scalars, label="representation scalars", allow_empty=False)
        try:
            _family, content = representation_parts(self.representation)
        except (TypeError, ValueError) as exc:
            raise ConversationHeldOutV4BundleError(
                "Representation identity 无法拆出完整 content") from exc
        if content != self.scalars:
            raise ConversationHeldOutV4BundleError(
                "Representation content 与 Unicode scalar 映射不一致")

    @property
    def text(self) -> str:
        """返回临时阅读文本；调用方不得把它作为 bundle 权威字段。"""
        return scalars_text(self.scalars, allow_empty=False)

    def stable_key(self) -> tuple[int, ...]:
        """返回 Representation、顺序和完整 scalar 的整数键。"""
        result = [1, self.ordinal]
        _pack(result, self.representation.stable_key())
        _pack(result, self.scalars)
        return tuple(result)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4SourceRecord:
    """一个完整 SourceRef、原文 scalar、内容 hash 和许可归属记录。"""

    source: SourceRef
    raw_text_scalars: tuple[int, ...]
    content_sha256: tuple[int, ...]
    license_scalars: tuple[int, ...]
    attribution_scalars: tuple[int, ...]
    source_uri_scalars: tuple[int, ...] = ()
    batch_id: int = 0
    companion_type_hash: int = 0
    companion_name_hash: int = 0
    companion_assoc_id: int = 0

    def __post_init__(self) -> None:
        """核验 SourceRef、原文 hash 和许可/归属元数据完整性。"""
        if not isinstance(self.source, SourceRef):
            raise ConversationHeldOutV4BundleError("source 必须是 SourceRef")
        _validate_scalars(self.raw_text_scalars, label="raw text scalars", allow_empty=True)
        for name in ("license_scalars", "attribution_scalars", "source_uri_scalars"):
            _validate_scalars(getattr(self, name), label=name, allow_empty=(name == "source_uri_scalars"))
        if not self.license_scalars or not self.attribution_scalars:
            raise ConversationHeldOutV4BundleError(
                "SourceRecord 必须携带许可和归属说明")
        _digest_bytes(self.content_sha256, label="content_sha256")
        raw = scalars_text(self.raw_text_scalars).encode("utf-8")
        expected = tuple(hashlib.sha256(raw).digest())
        if expected != self.content_sha256:
            raise ConversationHeldOutV4BundleError("SourceRecord 内容 hash 不一致")
        values = (self.batch_id, self.companion_type_hash,
                  self.companion_name_hash, self.companion_assoc_id)
        assert_int(*values, _where="v4 source metadata")
        if any(type(value) is not int or value < 0 for value in values):
            raise ConversationHeldOutV4BundleError("source metadata 必须是非负整数")
        if self.companion_assoc_id and not self.license_scalars:
            raise ConversationHeldOutV4BundleError("Companion assoc 缺少许可说明")

    @property
    def source_key(self) -> tuple[int, ...]:
        """返回 SourceRef 的完整整数键。"""
        return self.source.stable_key()

    @property
    def raw_text(self) -> str:
        """返回临时阅读原文；运行时不得从该属性回读 bundle。"""
        return scalars_text(self.raw_text_scalars)

    def stable_key(self) -> tuple[int, ...]:
        """返回来源、原文、hash、许可和归属的完整整数键。"""
        result = [1]
        _pack(result, self.source.stable_key())
        for value in (self.raw_text_scalars, self.content_sha256,
                      self.license_scalars, self.attribution_scalars,
                      self.source_uri_scalars):
            _pack(result, value)
        result.extend((self.batch_id, self.companion_type_hash,
                       self.companion_name_hash, self.companion_assoc_id))
        return tuple(result)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4Candidate:
    """完整候选及其可逆 surface、Evidence 和 SourceRef 链。"""

    candidate: GenerationCandidate
    surface_scalars: tuple[int, ...]
    surface_representations: tuple[ObjectIdentity, ...]
    evidence: tuple[EvidenceRecord, ...]
    source_chain: tuple[SourceRef, ...]

    def __post_init__(self) -> None:
        """核验候选未被摘要替代，且全部证据来源均显式留链。"""
        if not isinstance(self.candidate, GenerationCandidate):
            raise ConversationHeldOutV4BundleError("candidate 类型错误")
        _validate_scalars(self.surface_scalars, label="candidate surface scalars", allow_empty=False)
        if (not isinstance(self.surface_representations, tuple)
                or not self.surface_representations
                or any(not isinstance(item, ObjectIdentity)
                       or item.object_kind != OBJECT_REPRESENTATION
                       for item in self.surface_representations)):
            raise ConversationHeldOutV4BundleError(
                "candidate surface Representation 序列非法")
        rendered = []
        for representation in self.surface_representations:
            try:
                _family, content = representation_parts(representation)
            except (TypeError, ValueError) as exc:
                raise ConversationHeldOutV4BundleError(
                    "candidate surface Representation 无法拆解") from exc
            rendered.extend(content)
        if tuple(rendered) != self.surface_scalars:
            raise ConversationHeldOutV4BundleError(
                "candidate surface scalar 与 Representation 映射不一致")
        if self.evidence != self.candidate.evidence:
            raise ConversationHeldOutV4BundleError(
                "candidate evidence 必须与 GenerationCandidate 完整一致")
        if (not isinstance(self.source_chain, tuple)
                or any(not isinstance(item, SourceRef) for item in self.source_chain)):
            raise ConversationHeldOutV4BundleError("candidate source_chain 类型错误")
        expected = {
            self.candidate.source,
            *(item.source for item in self.evidence),
            *self.candidate.citation_sources,
        }
        actual = set(self.source_chain)
        if actual != expected or len(actual) != len(self.source_chain):
            raise ConversationHeldOutV4BundleError(
                "candidate SourceRef 链未完整覆盖 Evidence 来源")
        object.__setattr__(self, "source_chain", tuple(sorted(
            self.source_chain, key=lambda item: item.stable_key())))

    @classmethod
    def from_candidate(cls, candidate: GenerationCandidate,
                       surface_scalars: tuple[int, ...],
                       surface_representations: tuple[ObjectIdentity, ...],
                       ) -> "ConversationHeldOutV4Candidate":
        """从真实候选构造完整来源链，不接受摘要或人为候选键。"""
        if not isinstance(candidate, GenerationCandidate):
            raise TypeError("candidate 必须是 GenerationCandidate")
        chain = {
            candidate.source,
            *(item.source for item in candidate.evidence),
            *candidate.citation_sources,
        }
        return cls(candidate, surface_scalars, surface_representations,
                   candidate.evidence,
                   tuple(sorted(chain, key=lambda item: item.stable_key())))

    @property
    def candidate_key(self) -> tuple[int, ...]:
        """返回候选完整 stable integer identity。"""
        return self.candidate.stable_key()

    def stable_key(self) -> tuple[int, ...]:
        """返回候选、surface、Evidence 和来源链的完整整数键。"""
        result = [1]
        for value in (self.candidate_key, self.surface_scalars):
            _pack(result, value)
        result.append(len(self.surface_representations))
        for item in self.surface_representations:
            _pack(result, item.stable_key())
        result.append(len(self.evidence))
        for item in self.evidence:
            _pack(result, item.stable_key())
        result.append(len(self.source_chain))
        for item in self.source_chain:
            _pack(result, item.stable_key())
        return tuple(result)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4Turn:
    """一个完整无标签 turn：请求、Representation、候选集合和来源键。"""

    case_key: ProtocolKey
    turn_key: ProtocolKey
    ordinal: int
    request: QuestionRequest
    representations: tuple[ConversationHeldOutV4Representation, ...]
    surface_representations: tuple[ConversationHeldOutV4Representation, ...]
    candidates: tuple[ConversationHeldOutV4Candidate, ...]
    source_keys: tuple[SourceRef, ...]
    dependencies: ConversationHeldOutV4DependencyBinding
    _payload: tuple[int, ...] = field(init=False, repr=False, compare=False)
    payload_size: int = field(init=False)
    payload_sha256: tuple[int, ...] = field(init=False, compare=False)
    index: int = field(init=False, compare=False)

    def __post_init__(self) -> None:
        """核验请求、表示、候选和来源键闭合；不允许任何答案字段。"""
        _key(self.case_key, label="turn case_key")
        _key(self.turn_key, label="turn turn_key")
        if type(self.ordinal) is not int or self.ordinal <= 0:
            raise ConversationHeldOutV4BundleError("turn ordinal 必须为正整数")
        if not isinstance(self.request, QuestionRequest):
            raise ConversationHeldOutV4BundleError("turn request 类型错误")
        if (not isinstance(self.representations, tuple)
                or not self.representations
                or any(not isinstance(item, ConversationHeldOutV4Representation)
                       for item in self.representations)):
            raise ConversationHeldOutV4BundleError("turn representations 必须非空")
        ordinals = tuple(item.ordinal for item in self.representations)
        if ordinals != tuple(range(len(ordinals))):
            raise ConversationHeldOutV4BundleError("turn Representation ordinal 不连续")
        if (not isinstance(self.surface_representations, tuple)
                or any(not isinstance(item, ConversationHeldOutV4Representation)
                       for item in self.surface_representations)):
            raise ConversationHeldOutV4BundleError(
                "turn surface representations 类型错误")
        surface_ordinals = tuple(item.ordinal for item in self.surface_representations)
        if surface_ordinals != tuple(range(len(surface_ordinals))):
            raise ConversationHeldOutV4BundleError(
                "turn surface Representation ordinal 不连续")
        if (not isinstance(self.candidates, tuple)
                or any(not isinstance(item, ConversationHeldOutV4Candidate)
                       for item in self.candidates)):
            raise ConversationHeldOutV4BundleError("turn candidates 类型错误")
        candidate_keys = tuple(item.candidate_key for item in self.candidates)
        if len(set(candidate_keys)) != len(candidate_keys):
            raise ConversationHeldOutV4BundleError("turn candidate 集合不得重复")
        if self.candidates and not self.surface_representations:
            raise ConversationHeldOutV4BundleError(
                "存在候选时必须提供 surface Representation")
        mapped = {item.representation for item in self.surface_representations}
        if any(set(item.surface_representations) - mapped
               for item in self.candidates):
            raise ConversationHeldOutV4BundleError(
                "turn Representation 映射未覆盖候选 surface")
        if (not isinstance(self.source_keys, tuple)
                or any(not isinstance(item, SourceRef) for item in self.source_keys)):
            raise ConversationHeldOutV4BundleError("turn source_keys 类型错误")
        needed = {self.request.source}
        for candidate in self.candidates:
            needed.update(candidate.source_chain)
        if set(self.source_keys) != needed or len(self.source_keys) != len(needed):
            raise ConversationHeldOutV4BundleError(
                "turn SourceRef source_keys 未覆盖请求和候选来源")
        object.__setattr__(self, "source_keys", tuple(sorted(
            self.source_keys, key=lambda item: item.stable_key())))
        if not isinstance(self.dependencies, ConversationHeldOutV4DependencyBinding):
            raise ConversationHeldOutV4BundleError("turn dependencies 类型错误")
        payload = self._build_payload()
        size, digest, index_tuple = _payload_identity(payload)
        object.__setattr__(self, "_payload", payload)
        object.__setattr__(self, "payload_size", size)
        object.__setattr__(self, "payload_sha256", digest)
        object.__setattr__(self, "index", index_tuple[0])

    def _build_payload(self) -> tuple[int, ...]:
        """构造不含标签的完整 canonical integer payload。"""
        result = [4, 0]
        for value in (self.case_key.components, self.turn_key.components,
                      (self.ordinal,), self.request.stable_key(),
                      self.dependencies.stable_key()):
            _pack(result, value)
        result.append(len(self.representations))
        for item in self.representations:
            _pack(result, item.stable_key())
        result.append(len(self.surface_representations))
        for item in self.surface_representations:
            _pack(result, item.stable_key())
        result.append(len(self.candidates))
        for item in self.candidates:
            _pack(result, item.stable_key())
        result.append(len(self.source_keys))
        for item in self.source_keys:
            _pack(result, item.stable_key())
        return tuple(result)

    @property
    def canonical_payload(self) -> tuple[int, ...]:
        """返回完整 canonical integer payload；不是摘要。"""
        return self._payload


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4SourceBundle:
    """v4 family 的完整无标签 source bundle 和规范身份。"""

    version: int
    family_key: ProtocolKey
    dependencies: ConversationHeldOutV4DependencyBinding
    turns: tuple[ConversationHeldOutV4Turn, ...]
    sources: tuple[ConversationHeldOutV4SourceRecord, ...]
    _payload: tuple[int, ...] = field(init=False, repr=False, compare=False)
    payload_size: int = field(init=False)
    payload_sha256: tuple[int, ...] = field(init=False, compare=False)
    index: int = field(init=False, compare=False)

    def __post_init__(self) -> None:
        """核验 family、turn/source 完整绑定和不可覆盖的规范载荷。"""
        if type(self.version) is not int or self.version <= 0:
            raise ConversationHeldOutV4BundleError("bundle version 必须为正整数")
        _key(self.family_key, label="bundle family_key")
        if not isinstance(self.dependencies, ConversationHeldOutV4DependencyBinding):
            raise ConversationHeldOutV4BundleError("bundle dependencies 类型错误")
        if (not isinstance(self.turns, tuple) or not self.turns
                or any(not isinstance(item, ConversationHeldOutV4Turn)
                       for item in self.turns)):
            raise ConversationHeldOutV4BundleError("bundle turns 必须非空")
        turn_keys = tuple((item.case_key, item.turn_key) for item in self.turns)
        if len(set(turn_keys)) != len(turn_keys):
            raise ConversationHeldOutV4BundleError("bundle turn identity 不得重复")
        if any(item.dependencies != self.dependencies for item in self.turns):
            raise ConversationHeldOutV4BundleError("turn dependency binding 漂移")
        if (not isinstance(self.sources, tuple) or not self.sources
                or any(not isinstance(item, ConversationHeldOutV4SourceRecord)
                       for item in self.sources)):
            raise ConversationHeldOutV4BundleError("bundle sources 必须非空")
        source_keys = tuple(item.source for item in self.sources)
        if len(set(source_keys)) != len(source_keys):
            raise ConversationHeldOutV4BundleError("bundle SourceRef 不得重复")
        source_by_key = {item.source: item for item in self.sources}
        required = {source for turn in self.turns for source in turn.source_keys}
        if set(source_by_key) != required:
            raise ConversationHeldOutV4BundleError(
                "bundle sources 未逐一覆盖 turn 的 SourceRef 链")
        payload = self._build_payload()
        size, digest, index_tuple = _payload_identity(payload)
        object.__setattr__(self, "_payload", payload)
        object.__setattr__(self, "payload_size", size)
        object.__setattr__(self, "payload_sha256", digest)
        object.__setattr__(self, "index", index_tuple[0])

    def _build_payload(self) -> tuple[int, ...]:
        """以固定顺序组合 family、依赖、source 和所有 turn 完整载荷。"""
        result = [4, 1, self.version]
        _pack(result, self.family_key.components)
        _pack(result, self.dependencies.stable_key())
        result.append(len(self.sources))
        for item in sorted(self.sources, key=lambda value: value.source.stable_key()):
            _pack(result, item.stable_key())
        result.append(len(self.turns))
        for item in sorted(self.turns, key=lambda value: (
                value.case_key.components, value.turn_key.components)):
            _pack(result, item.canonical_payload)
        return tuple(result)

    @property
    def canonical_payload(self) -> tuple[int, ...]:
        """返回 bundle 的完整 canonical integer payload。"""
        return self._payload

    def source_for(self, source: SourceRef) -> ConversationHeldOutV4SourceRecord:
        """按完整 SourceRef 只读取 bundle 内的来源记录。"""
        if not isinstance(source, SourceRef):
            raise TypeError("source 必须是 SourceRef")
        for item in self.sources:
            if item.source == source:
                return item
        raise ConversationHeldOutV4BundleError("bundle 中不存在指定 SourceRef")

    def turn_for(self, case_key: ProtocolKey,
                 turn_key: ProtocolKey) -> ConversationHeldOutV4Turn:
        """按完整 case/turn identity 取回单一输入。"""
        _key(case_key, label="lookup case_key")
        _key(turn_key, label="lookup turn_key")
        matches = tuple(item for item in self.turns
                        if item.case_key == case_key and item.turn_key == turn_key)
        if len(matches) != 1:
            raise ConversationHeldOutV4BundleError(
                "bundle case/turn 不存在或不唯一")
        return matches[0]


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4ExecutionInput:
    """一个真实 query execution 在 v4 聚合器中的完整输入帧。"""

    case_key: ProtocolKey
    turn_key: ProtocolKey
    ordinal: int
    request: QuestionRequest
    execution: QuestionExecutionResult
    representations: tuple[ConversationHeldOutV4Representation, ...]
    surface_representations: tuple[ConversationHeldOutV4Representation, ...]
    source_records: tuple[ConversationHeldOutV4SourceRecord, ...]
    dependencies: ConversationHeldOutV4DependencyBinding

    def __post_init__(self) -> None:
        """核验聚合帧只携带真实执行和无标签来源材料。"""
        _key(self.case_key, label="execution input case_key")
        _key(self.turn_key, label="execution input turn_key")
        if type(self.ordinal) is not int or self.ordinal <= 0:
            raise ConversationHeldOutV4BundleError(
                "execution input ordinal 必须为正整数")
        if not isinstance(self.request, QuestionRequest):
            raise TypeError("execution input request 类型错误")
        if not isinstance(self.execution, QuestionExecutionResult):
            raise TypeError("execution input execution 类型错误")
        if self.execution.query.request != self.request:
            raise ConversationHeldOutV4BundleError(
                "execution input request 与 query 漂移")
        if (not isinstance(self.representations, tuple)
                or not self.representations
                or any(not isinstance(item, ConversationHeldOutV4Representation)
                       for item in self.representations)):
            raise ConversationHeldOutV4BundleError(
                "execution input representations 非法")
        if (not isinstance(self.surface_representations, tuple)
                or any(not isinstance(item, ConversationHeldOutV4Representation)
                       for item in self.surface_representations)):
            raise ConversationHeldOutV4BundleError(
                "execution input surface representations 非法")
        if (not isinstance(self.source_records, tuple)
                or any(not isinstance(item, ConversationHeldOutV4SourceRecord)
                       for item in self.source_records)):
            raise ConversationHeldOutV4BundleError(
                "execution input source_records 非法")
        if not isinstance(self.dependencies, ConversationHeldOutV4DependencyBinding):
            raise TypeError("execution input dependencies 类型错误")
def build_v4_source_bundle(
        *, version: int, family_key: ProtocolKey,
        dependencies: ConversationHeldOutV4DependencyBinding,
        turns: Iterable[ConversationHeldOutV4Turn],
        sources: Iterable[ConversationHeldOutV4SourceRecord],
        ) -> ConversationHeldOutV4SourceBundle:
    """从构造点提供的完整对象建立一次不可变 v4 bundle。"""
    return ConversationHeldOutV4SourceBundle(
        version, family_key, dependencies, tuple(turns), tuple(sources))


def build_v4_source_bundle_from_executions(
        *, version: int, family_key: ProtocolKey,
        inputs: tuple[ConversationHeldOutV4ExecutionInput, ...],
        render_candidate: Callable[[GenerationCandidate], RenderedSurface],
        ) -> ConversationHeldOutV4SourceBundle:
    """聚合多个真实 execution，正向生成一个无标签 v4 bundle。"""
    if (not isinstance(inputs, tuple) or not inputs
            or any(not isinstance(item, ConversationHeldOutV4ExecutionInput)
                   for item in inputs)):
        raise TypeError("inputs 必须是非空 V4 execution input tuple")
    dependencies = inputs[0].dependencies
    identities = {(item.case_key, item.turn_key) for item in inputs}
    if len(identities) != len(inputs):
        raise ConversationHeldOutV4BundleError(
            "v4 execution input case/turn 不得重复")
    if any(item.dependencies != dependencies for item in inputs):
        raise ConversationHeldOutV4BundleError(
            "v4 execution input dependency 漂移")
    turns = tuple(
        build_v4_turn_from_execution(
            case_key=item.case_key,
            turn_key=item.turn_key,
            ordinal=item.ordinal,
            request=item.request,
            execution=item.execution,
            representations=item.representations,
            surface_representations=item.surface_representations,
            render_candidate=render_candidate,
            source_records=item.source_records,
            dependencies=dependencies,
        )
        for item in inputs
    )
    sources_by_key: dict[SourceRef, ConversationHeldOutV4SourceRecord] = {}
    for item in inputs:
        for source in item.source_records:
            previous = sources_by_key.get(source.source)
            if previous is not None and previous != source:
                raise ConversationHeldOutV4BundleError(
                    "同一 SourceRef 在 execution input 中绑定了不同原文")
            sources_by_key[source.source] = source
    return build_v4_source_bundle(
        version=version,
        family_key=family_key,
        dependencies=dependencies,
        turns=turns,
        sources=tuple(sorted(
            sources_by_key.values(), key=lambda item: item.source_key)),
    )


def export_v4_candidate_set(
        candidates: tuple[GenerationCandidate, ...],
        render_candidate: Callable[[GenerationCandidate], RenderedSurface],
        ) -> tuple[ConversationHeldOutV4Candidate, ...]:
    """从真实候选构造点正向导出完整 candidate 集合。

    ``render_candidate`` 只能返回已经由生产 renderer 校验的
    ``RenderedSurface``；本函数不接受 stable key、答案标签或人工 surface，
    也不执行选择。候选顺序按完整 stable key 规范化，集合身份不会由 ordinal
    或摘要替代。
    """
    if (not isinstance(candidates, tuple)
            or any(not isinstance(item, GenerationCandidate)
                   for item in candidates)):
        raise TypeError("candidates 必须是 GenerationCandidate tuple")
    if not callable(render_candidate):
        raise TypeError("render_candidate 必须可调用")
    source = []
    seen = set()
    for candidate in candidates:
        key = candidate.stable_key()
        if key in seen:
            raise ConversationHeldOutV4BundleError(
                "candidate 构造点输出了重复候选")
        seen.add(key)
        rendered = render_candidate(candidate)
        if not isinstance(rendered, RenderedSurface):
            raise ConversationHeldOutV4BundleError(
                "candidate renderer 必须返回 RenderedSurface")
        source.append(ConversationHeldOutV4Candidate.from_candidate(
            candidate, rendered.units, rendered.representations))
    return tuple(sorted(source, key=lambda item: item.candidate_key))


def build_v4_turn_from_candidates(
        *, case_key: ProtocolKey, turn_key: ProtocolKey, ordinal: int,
        request: QuestionRequest,
        representations: tuple[ConversationHeldOutV4Representation, ...],
        surface_representations: tuple[ConversationHeldOutV4Representation, ...],
        candidates: tuple[GenerationCandidate, ...],
        render_candidate: Callable[[GenerationCandidate], RenderedSurface],
        source_records: tuple[ConversationHeldOutV4SourceRecord, ...],
        dependencies: ConversationHeldOutV4DependencyBinding,
        ) -> ConversationHeldOutV4Turn:
    """从真实 query 的完整候选集合正向生成一个无标签 v4 turn。"""
    if (not isinstance(source_records, tuple)
            or any(not isinstance(item, ConversationHeldOutV4SourceRecord)
                   for item in source_records)):
        raise TypeError("source_records 必须是 V4 SourceRecord tuple")
    candidate_exports = export_v4_candidate_set(candidates, render_candidate)
    needed = {request.source}
    for item in candidate_exports:
        needed.update(item.source_chain)
    available = {item.source for item in source_records}
    if available != needed or len(available) != len(source_records):
        raise ConversationHeldOutV4BundleError(
            "构造点 SourceRecord 未完整覆盖 query/candidate SourceRef 链")
    return ConversationHeldOutV4Turn(
        case_key, turn_key, ordinal, request, representations,
        surface_representations,
        candidate_exports,
        tuple(sorted(available, key=lambda item: item.stable_key())),
        dependencies,
    )


def build_v4_turn_from_execution(
        *, case_key: ProtocolKey, turn_key: ProtocolKey, ordinal: int,
        request: QuestionRequest,
        execution: QuestionExecutionResult,
        representations: tuple[ConversationHeldOutV4Representation, ...],
        surface_representations: tuple[ConversationHeldOutV4Representation, ...],
        render_candidate: Callable[[GenerationCandidate], RenderedSurface],
        source_records: tuple[ConversationHeldOutV4SourceRecord, ...],
        dependencies: ConversationHeldOutV4DependencyBinding,
        ) -> ConversationHeldOutV4Turn:
    """从同次 QuestionExecutionResult 正向导出 turn，拒绝 request/candidate 替换。"""
    if not isinstance(execution, QuestionExecutionResult):
        raise TypeError("execution 必须是 QuestionExecutionResult")
    if execution.query.request != request:
        raise ConversationHeldOutV4BundleError(
            "QuestionExecutionResult request 与 v4 turn request 漂移")
    return build_v4_turn_from_candidates(
        case_key=case_key,
        turn_key=turn_key,
        ordinal=ordinal,
        request=request,
        representations=representations,
        surface_representations=surface_representations,
        candidates=execution.candidates,
        render_candidate=render_candidate,
        source_records=source_records,
        dependencies=dependencies,
    )


__all__ = [
    "ConversationHeldOutV4BundleError",
    "ConversationHeldOutV4Candidate",
    "ConversationHeldOutV4DependencyBinding",
    "ConversationHeldOutV4ExecutionInput",
    "ConversationHeldOutV4Representation",
    "ConversationHeldOutV4SourceBundle",
    "ConversationHeldOutV4SourceRecord",
    "ConversationHeldOutV4Turn",
    "build_v4_source_bundle",
    "build_v4_source_bundle_from_executions",
    "build_v4_turn_from_candidates",
    "build_v4_turn_from_execution",
    "digest_from_hex",
    "digest_hex",
    "export_v4_candidate_set",
    "scalars_text",
    "unicode_scalars",
]
