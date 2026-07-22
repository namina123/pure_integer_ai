"""独立评测的数据身份、切分隔离和分维探针协议。

本模块只承认调用方注入的整数协议键，不内置语言、关系、探针种类或评测维度。
完整规范内容和完整来源簇身份始终保留；摘要只用于索引和报告，不替代身份核验。
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, TYPE_CHECKING

from pure_integer_ai.cognition.shared.identity import SourceRef
from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.crosscut.guards.int_blocker import assert_int

if TYPE_CHECKING:
    from pure_integer_ai.experiments.collection import CollectedItem


_IDENTITY_INDEX_HASHER = Hasher("evaluation.canonical_identity.v1")


class EvaluationProtocolError(RuntimeError):
    """评测协议键、记录覆盖或探针结果不完整。"""


class EvaluationLeakageError(EvaluationProtocolError):
    """训练与评测之间存在内容、来源或去重簇泄漏。"""


class EvaluationStatePollutionError(EvaluationProtocolError):
    """探针执行改变了调用方要求保持不变的状态。"""


def _canonical_value(value: Any) -> Any:
    """把受支持对象转成带类型标签的规范 JSON 值。"""
    if value is None:
        return [0]
    if isinstance(value, bool):
        return [1, 1 if value else 0]
    if type(value) is int:
        return [2, value]
    if isinstance(value, str):
        return [3, value]
    if isinstance(value, bytes):
        return [4, value.hex()]
    if isinstance(value, tuple):
        return [5, [_canonical_value(item) for item in value]]
    if isinstance(value, list):
        return [6, [_canonical_value(item) for item in value]]
    if isinstance(value, (set, frozenset)):
        members = [_canonical_value(item) for item in value]
        members.sort(key=lambda item: json.dumps(
            item, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        return [7, members]
    if isinstance(value, dict):
        pairs = [
            (_canonical_value(key), _canonical_value(item))
            for key, item in value.items()
        ]
        pairs.sort(key=lambda pair: json.dumps(
            pair[0], ensure_ascii=False,
            sort_keys=True, separators=(",", ":")))
        return [8, [[key, item] for key, item in pairs]]
    if dataclasses.is_dataclass(value):
        fields = [
            [field.name, _canonical_value(getattr(value, field.name))]
            for field in dataclasses.fields(value)
        ]
        return [9, type(value).__qualname__, fields]
    raise TypeError(f"评测规范身份不支持类型 {type(value).__name__}")


def canonical_payload(value: Any) -> bytes:
    """生成保留完整值和类型边界的稳定 UTF-8 规范载荷。"""
    canonical = _canonical_value(value)
    text = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return text.encode("utf-8")


@dataclass(frozen=True, order=True)
class ProtocolKey:
    """由 manifest 或调用方注入的非空严格整数协议键。"""

    components: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.components, tuple) or not self.components:
            raise ValueError("ProtocolKey.components 必须是非空整数 tuple")
        assert_int(*self.components, _where="ProtocolKey.components")
        if any(type(value) is not int or value < 0
               for value in self.components):
            raise ValueError("ProtocolKey.components 必须是非负严格整数")

    def stable_key(self) -> tuple[int, ...]:
        """返回可写入 manifest 或图协议的完整整数键。"""
        return self.components


@dataclass(frozen=True, order=True)
class CanonicalIdentity:
    """保留完整规范载荷，并提供只作索引的 SHA-256 和整数摘要。"""

    payload: bytes
    sha256: str = dataclasses.field(init=False, compare=False)
    index: int = dataclasses.field(init=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.payload, bytes) or not self.payload:
            raise ValueError("CanonicalIdentity.payload 必须是非空 bytes")
        digest = hashlib.sha256(self.payload).hexdigest()
        index = _IDENTITY_INDEX_HASHER.h63(self.payload)
        object.__setattr__(self, "sha256", digest)
        object.__setattr__(self, "index", index if index > 0 else 1)

    @classmethod
    def from_value(cls, value: Any) -> "CanonicalIdentity":
        """从结构化值建立保留完整载荷的身份。"""
        return cls(canonical_payload(value))

    def to_dict(self) -> dict[str, str]:
        """导出完整载荷及其可核验摘要，摘要不替代载荷。"""
        return {
            "payload_hex": self.payload.hex(),
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CanonicalIdentity":
        """从 manifest 恢复完整载荷，并拒绝摘要或十六进制损坏。"""
        try:
            payload = bytes.fromhex(str(value["payload_hex"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise EvaluationProtocolError("规范身份载荷十六进制非法") from exc
        identity = cls(payload)
        if identity.sha256 != value.get("sha256"):
            raise EvaluationProtocolError("规范身份 SHA-256 不匹配")
        return identity


@dataclass(frozen=True)
class EvaluationProtocol:
    """声明五类切分、证据分账以及必须覆盖的维度和对抗种类。"""

    version: int
    training_split: ProtocolKey
    development_split: ProtocolKey
    held_out_split: ProtocolKey
    adversarial_split: ProtocolKey
    external_split: ProtocolKey
    statistical_evidence: ProtocolKey
    external_evidence: ProtocolKey
    required_dimensions: tuple[ProtocolKey, ...]
    required_adversarial_kinds: tuple[ProtocolKey, ...]

    def __post_init__(self) -> None:
        assert_int(self.version, _where="EvaluationProtocol.version")
        if type(self.version) is not int or self.version <= 0:
            raise ValueError("EvaluationProtocol.version 必须是正严格整数")
        splits = self.split_keys()
        if len(set(splits)) != len(splits):
            raise ValueError("评测五类 split 协议键必须互不相同")
        if self.statistical_evidence == self.external_evidence:
            raise ValueError("统计证据与 EXTERNAL 证据协议键必须不同")
        for name, keys in (
                ("required_dimensions", self.required_dimensions),
                ("required_adversarial_kinds",
                 self.required_adversarial_kinds)):
            if not isinstance(keys, tuple) or not keys:
                raise ValueError(f"EvaluationProtocol.{name} 不能为空")
            if len(set(keys)) != len(keys):
                raise ValueError(f"EvaluationProtocol.{name} 不得重复")

    def split_keys(self) -> tuple[ProtocolKey, ...]:
        """按训练、开发、留出、对抗和 EXTERNAL 返回五类切分键。"""
        return (
            self.training_split,
            self.development_split,
            self.held_out_split,
            self.adversarial_split,
            self.external_split,
        )

    def evidence_for(self, split: ProtocolKey) -> ProtocolKey:
        """按切分返回统计或 EXTERNAL 证据账本键。"""
        if split not in self.split_keys():
            raise EvaluationProtocolError("记录使用了未注册 split")
        return (self.external_evidence
                if split == self.external_split
                else self.statistical_evidence)

    def to_dict(self) -> dict[str, Any]:
        """导出不写死具体整数意义的协议 manifest 对象。"""
        return {
            "adversarial_split": list(self.adversarial_split.stable_key()),
            "development_split": list(self.development_split.stable_key()),
            "external_evidence": list(self.external_evidence.stable_key()),
            "external_split": list(self.external_split.stable_key()),
            "held_out_split": list(self.held_out_split.stable_key()),
            "required_adversarial_kinds": [
                list(key.stable_key())
                for key in self.required_adversarial_kinds
            ],
            "required_dimensions": [
                list(key.stable_key())
                for key in self.required_dimensions
            ],
            "statistical_evidence": list(
                self.statistical_evidence.stable_key()),
            "training_split": list(self.training_split.stable_key()),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvaluationProtocol":
        """从 manifest 恢复全部注入键并重新执行协议约束。"""
        def key(name: str) -> ProtocolKey:
            """严格读取一个整数协议键字段。"""
            raw = value[name]
            if not isinstance(raw, list):
                raise EvaluationProtocolError(f"协议字段 {name} 必须是整数列表")
            return ProtocolKey(tuple(raw))

        def keys(name: str) -> tuple[ProtocolKey, ...]:
            """严格读取一个协议键列表字段。"""
            raw = value[name]
            if not isinstance(raw, list):
                raise EvaluationProtocolError(f"协议字段 {name} 必须是键列表")
            out: list[ProtocolKey] = []
            for item in raw:
                if not isinstance(item, list):
                    raise EvaluationProtocolError(
                        f"协议字段 {name} 包含非列表键")
                out.append(ProtocolKey(tuple(item)))
            return tuple(out)

        return cls(
            version=value["version"],
            training_split=key("training_split"),
            development_split=key("development_split"),
            held_out_split=key("held_out_split"),
            adversarial_split=key("adversarial_split"),
            external_split=key("external_split"),
            statistical_evidence=key("statistical_evidence"),
            external_evidence=key("external_evidence"),
            required_dimensions=keys("required_dimensions"),
            required_adversarial_kinds=keys(
                "required_adversarial_kinds"),
        )


def collected_item_content_identity(
        item: "CollectedItem") -> CanonicalIdentity:
    """提取学习者实际接收的输入内容，不混入来源、标签或运行期缓存。"""
    if item.arith_source is not None:
        primary = (1, item.arith_source)
    elif item.code_source is not None:
        primary = (2, item.code_source)
    elif item.raw_text is not None:
        primary = (3, item.raw_text)
    else:
        primary = (4, tuple(item.tokens))
    return CanonicalIdentity.from_value((
        item.modality,
        item.lang,
        item.domain,
        primary,
        (None if item.speaker_identity is None
         else item.speaker_identity.stable_key()),
    ))


def source_cluster_identity(source_ref: SourceRef) -> CanonicalIdentity:
    """从完整 SourceRef 提取排除 document_id 的来源簇身份。"""
    return CanonicalIdentity.from_value((
        source_ref.source_kind,
        source_ref.source_id,
        source_ref.owner.stable_key(),
        source_ref.versions.stable_key(),
    ))


@dataclass(frozen=True)
class EvaluationDataIdentity:
    """一个评测输入的来源、完整内容、去重簇和 provenance 簇身份。"""

    source_ref: SourceRef
    content: CanonicalIdentity
    dedup_cluster: CanonicalIdentity
    provenance_cluster: CanonicalIdentity

    def lookup_key(self) -> tuple[tuple[int, ...], bytes]:
        """返回查找键；完整载荷参与比较，摘要不替代身份。"""
        return self.source_ref.stable_key(), self.content.payload


def make_evaluation_data_identity(
        item: "CollectedItem", *,
        dedup_cluster: Any,
        provenance_cluster: Any | None = None,
        ) -> EvaluationDataIdentity:
    """从显式来源和调用方簇键建立可核验评测数据身份。"""
    if item.source_ref is None:
        raise EvaluationProtocolError(
            "严格评测项必须在建计划前携带显式 SourceRef")
    provenance = (
        source_cluster_identity(item.source_ref)
        if provenance_cluster is None
        else CanonicalIdentity.from_value(provenance_cluster)
    )
    return EvaluationDataIdentity(
        source_ref=item.source_ref,
        content=collected_item_content_identity(item),
        dedup_cluster=CanonicalIdentity.from_value(dedup_cluster),
        provenance_cluster=provenance,
    )


@dataclass(frozen=True)
class EvaluationAssignment:
    """给一个完整数据身份分配 split、探针种类、维度和可选预期。"""

    identity: EvaluationDataIdentity
    split: ProtocolKey
    probe_kind: ProtocolKey | None = None
    dimensions: tuple[ProtocolKey, ...] = ()
    expected_outcome: CanonicalIdentity | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.dimensions, tuple):
            object.__setattr__(self, "dimensions", tuple(self.dimensions))
        if len(set(self.dimensions)) != len(self.dimensions):
            raise ValueError("同一评测项的 dimensions 不得重复")


def _data_identity_to_dict(
        identity: EvaluationDataIdentity) -> dict[str, Any]:
    """导出包含完整 SourceRef 和三类规范身份的 manifest 对象。"""
    return {
        "content": identity.content.to_dict(),
        "dedup_cluster": identity.dedup_cluster.to_dict(),
        "provenance_cluster": identity.provenance_cluster.to_dict(),
        "source_ref": list(identity.source_ref.stable_key()),
    }


def _data_identity_from_dict(
        value: dict[str, Any]) -> EvaluationDataIdentity:
    """从 manifest 恢复完整数据身份并拒绝截断 SourceRef。"""
    source_key = value.get("source_ref")
    if not isinstance(source_key, list):
        raise EvaluationProtocolError("评测记录 SourceRef 必须是整数列表")
    try:
        source_ref = SourceRef.from_stable_key(tuple(source_key))
    except (TypeError, ValueError) as exc:
        raise EvaluationProtocolError("评测记录 SourceRef 非法") from exc
    return EvaluationDataIdentity(
        source_ref=source_ref,
        content=CanonicalIdentity.from_dict(value["content"]),
        dedup_cluster=CanonicalIdentity.from_dict(value["dedup_cluster"]),
        provenance_cluster=CanonicalIdentity.from_dict(
            value["provenance_cluster"]),
    )


def _assignment_to_dict(
        assignment: EvaluationAssignment) -> dict[str, Any]:
    """导出一条包含注入键和完整身份的 split ledger 记录。"""
    return {
        "dimensions": [
            list(key.stable_key()) for key in assignment.dimensions
        ],
        "expected_outcome": (
            None if assignment.expected_outcome is None
            else assignment.expected_outcome.to_dict()
        ),
        "identity": _data_identity_to_dict(assignment.identity),
        "probe_kind": (
            None if assignment.probe_kind is None
            else list(assignment.probe_kind.stable_key())
        ),
        "split": list(assignment.split.stable_key()),
    }


def _assignment_from_dict(value: dict[str, Any]) -> EvaluationAssignment:
    """从 manifest 恢复一条记录并重新核验所有严格整数键。"""
    split = value.get("split")
    dimensions = value.get("dimensions")
    probe_kind = value.get("probe_kind")
    if not isinstance(split, list) or not isinstance(dimensions, list):
        raise EvaluationProtocolError("评测记录 split/dimensions 格式非法")
    if any(not isinstance(item, list) for item in dimensions):
        raise EvaluationProtocolError("评测记录 dimensions 包含非列表键")
    if probe_kind is not None and not isinstance(probe_kind, list):
        raise EvaluationProtocolError("评测记录 probe_kind 格式非法")
    expected = value.get("expected_outcome")
    if expected is not None and not isinstance(expected, dict):
        raise EvaluationProtocolError("评测记录 expected_outcome 格式非法")
    identity_value = value.get("identity")
    if not isinstance(identity_value, dict):
        raise EvaluationProtocolError("评测记录 identity 格式非法")
    return EvaluationAssignment(
        identity=_data_identity_from_dict(identity_value),
        split=ProtocolKey(tuple(split)),
        probe_kind=(None if probe_kind is None
                    else ProtocolKey(tuple(probe_kind))),
        dimensions=tuple(
            ProtocolKey(tuple(item)) for item in dimensions),
        expected_outcome=(
            None if expected is None
            else CanonicalIdentity.from_dict(expected)
        ),
    )


@dataclass(frozen=True)
class EvaluationPartition:
    """按注入 split 键保存原语料对象，保持调用方输入顺序。"""

    protocol: EvaluationProtocol
    groups: tuple[tuple[ProtocolKey, tuple[Any, ...]], ...]

    def items(self, split: ProtocolKey) -> tuple[Any, ...]:
        """读取一个已注册 split 的原语料对象。"""
        for key, items in self.groups:
            if key == split:
                return items
        raise EvaluationProtocolError("读取了未注册评测 split")

    def as_dict(self) -> dict[ProtocolKey, list[Any]]:
        """返回供运行时上下文持有的独立可变列表副本。"""
        return {key: list(items) for key, items in self.groups}

    def non_training_items(self) -> tuple[Any, ...]:
        """按协议顺序合并所有禁止进入训练消费者的评测项。"""
        out: list[Any] = []
        for split in self.protocol.split_keys():
            if split != self.protocol.training_split:
                out.extend(self.items(split))
        return tuple(out)


@dataclass(frozen=True)
class EvaluationPlan:
    """保存完整 split ledger，并在构造时 fail closed 核验泄漏与覆盖。"""

    protocol: EvaluationProtocol
    assignments: tuple[EvaluationAssignment, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.assignments, tuple) or not self.assignments:
            raise ValueError("EvaluationPlan.assignments 不能为空")
        self._validate_assignments()
        self._validate_cross_split_isolation()
        self._validate_required_coverage()

    def _validate_assignments(self) -> None:
        """核验 split 注册、训练/探针字段边界和完整记录唯一性。"""
        split_keys = set(self.protocol.split_keys())
        seen: set[tuple[tuple[int, ...], bytes]] = set()
        present_splits: set[ProtocolKey] = set()
        expected_by_content: dict[
            tuple[ProtocolKey, CanonicalIdentity],
            CanonicalIdentity | None,
        ] = {}
        for assignment in self.assignments:
            if assignment.split not in split_keys:
                raise EvaluationProtocolError("评测记录使用了未注册 split")
            present_splits.add(assignment.split)
            lookup_key = assignment.identity.lookup_key()
            if lookup_key in seen:
                raise EvaluationProtocolError("同一来源文档和完整内容被重复分配")
            seen.add(lookup_key)
            is_training = assignment.split == self.protocol.training_split
            if is_training and (
                    assignment.probe_kind is not None
                    or assignment.dimensions):
                raise EvaluationProtocolError("训练记录不得伪装成评测探针")
            if not is_training and (
                    assignment.probe_kind is None
                    or not assignment.dimensions):
                raise EvaluationProtocolError(
                    "非训练记录必须声明注入式 probe kind 和评测维度")
            content_key = (assignment.split, assignment.identity.content)
            if (content_key in expected_by_content
                    and expected_by_content[content_key]
                    != assignment.expected_outcome):
                raise EvaluationProtocolError(
                    "同一 split 的相同完整输入携带冲突预期")
            expected_by_content[content_key] = assignment.expected_outcome
        missing = split_keys - present_splits
        if missing:
            raise EvaluationProtocolError("严格评测计划必须覆盖全部五类 split")

    @staticmethod
    def _split_sets(
            assignments: Iterable[EvaluationAssignment],
            attribute: str,
            ) -> dict[ProtocolKey, set[Any]]:
        """按 split 收集一个完整身份字段，供交叉隔离核验。"""
        result: dict[ProtocolKey, set[Any]] = {}
        for assignment in assignments:
            result.setdefault(assignment.split, set()).add(
                getattr(assignment.identity, attribute))
        return result

    def _validate_cross_split_isolation(self) -> None:
        """拒绝内容、文档、去重簇及训练/EXTERNAL 来源簇泄漏。"""
        by_content = self._split_sets(self.assignments, "content")
        by_dedup = self._split_sets(self.assignments, "dedup_cluster")
        by_provenance = self._split_sets(
            self.assignments, "provenance_cluster")
        by_source_ref: dict[ProtocolKey, set[tuple[int, ...]]] = {}
        for assignment in self.assignments:
            by_source_ref.setdefault(assignment.split, set()).add(
                assignment.identity.source_ref.stable_key())

        splits = self.protocol.split_keys()
        for index, left in enumerate(splits):
            for right in splits[index + 1:]:
                if by_content.get(left, set()) & by_content.get(right, set()):
                    raise EvaluationLeakageError("完整内容跨 split 泄漏")
                if by_dedup.get(left, set()) & by_dedup.get(right, set()):
                    raise EvaluationLeakageError("dedup cluster 跨 split 泄漏")
                if (by_source_ref.get(left, set())
                        & by_source_ref.get(right, set())):
                    raise EvaluationLeakageError("同一 SourceRef 跨 split 泄漏")

        training = self.protocol.training_split
        evaluation_splits = set(splits) - {training}
        training_provenance = by_provenance.get(training, set())
        for split in evaluation_splits:
            if training_provenance & by_provenance.get(split, set()):
                raise EvaluationLeakageError(
                    "训练与评测共享 provenance cluster，包含同源改写泄漏")

        external = self.protocol.external_split
        external_provenance = by_provenance.get(external, set())
        for split in set(splits) - {external}:
            if external_provenance & by_provenance.get(split, set()):
                raise EvaluationLeakageError(
                    "EXTERNAL 与非 EXTERNAL 共享 provenance cluster")

    def _validate_required_coverage(self) -> None:
        """核验所有注入维度和对抗种类至少有一个真实探针。"""
        dimensions: set[ProtocolKey] = set()
        adversarial_kinds: set[ProtocolKey] = set()
        for assignment in self.assignments:
            if assignment.split == self.protocol.training_split:
                continue
            dimensions.update(assignment.dimensions)
            if assignment.split == self.protocol.adversarial_split:
                if assignment.probe_kind is None:
                    raise EvaluationProtocolError("对抗记录缺少 probe kind")
                adversarial_kinds.add(assignment.probe_kind)
        missing_dimensions = set(
            self.protocol.required_dimensions) - dimensions
        if missing_dimensions:
            raise EvaluationProtocolError("评测计划缺少必需分维探针")
        missing_kinds = set(
            self.protocol.required_adversarial_kinds) - adversarial_kinds
        if missing_kinds:
            raise EvaluationProtocolError("评测计划缺少必需对抗种类")

    def assignment_for(
            self, identity: EvaluationDataIdentity) -> EvaluationAssignment:
        """按完整来源和内容身份读取唯一计划记录。"""
        matches = [
            assignment for assignment in self.assignments
            if assignment.identity.lookup_key() == identity.lookup_key()
        ]
        if len(matches) != 1:
            raise EvaluationProtocolError("评测身份在计划中缺失或不唯一")
        return matches[0]

    def partition(self, items: Sequence["CollectedItem"]) -> EvaluationPartition:
        """按完整来源和内容把输入语料分区，并拒绝遗漏或计划外记录。"""
        assignments = {
            assignment.identity.lookup_key(): assignment
            for assignment in self.assignments
        }
        groups: dict[ProtocolKey, list[Any]] = {
            split: [] for split in self.protocol.split_keys()
        }
        consumed: set[tuple[tuple[int, ...], bytes]] = set()
        for item in items:
            if item.source_ref is None:
                raise EvaluationProtocolError(
                    "严格评测分区要求每个语料项携带显式 SourceRef")
            key = (
                item.source_ref.stable_key(),
                collected_item_content_identity(item).payload,
            )
            assignment = assignments.get(key)
            if assignment is None:
                raise EvaluationProtocolError("语料包含评测计划外记录")
            if key in consumed:
                raise EvaluationProtocolError("同一计划记录在语料中重复出现")
            consumed.add(key)
            groups[assignment.split].append(item)
        if consumed != set(assignments):
            raise EvaluationProtocolError("评测计划包含未出现在语料中的记录")
        return EvaluationPartition(
            self.protocol,
            tuple(
                (split, tuple(groups[split]))
                for split in self.protocol.split_keys()
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """导出完整协议和按输入顺序保存的 split ledger。"""
        return {
            "assignments": [
                _assignment_to_dict(assignment)
                for assignment in self.assignments
            ],
            "protocol": self.protocol.to_dict(),
        }

    def canonical_bytes(self) -> bytes:
        """返回可跨 run 逐字节复现的规范评测计划。"""
        text = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return (text + "\n").encode("utf-8")

    def sha256(self) -> str:
        """返回整个协议和 ledger 的内容摘要。"""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvaluationPlan":
        """从 manifest 对象恢复计划并重新执行全部泄漏与覆盖守卫。"""
        protocol_value = value.get("protocol")
        assignments_value = value.get("assignments")
        if (not isinstance(protocol_value, dict)
                or not isinstance(assignments_value, list)
                or any(not isinstance(item, dict)
                       for item in assignments_value)):
            raise EvaluationProtocolError("评测计划 protocol/assignments 格式非法")
        return cls(
            EvaluationProtocol.from_dict(protocol_value),
            tuple(
                _assignment_from_dict(item)
                for item in assignments_value
            ),
        )


@dataclass(frozen=True)
class ProbeOutcome:
    """单次探针的三态结论和纯整数测量值。"""

    passed: bool | None
    value: int = 0
    sample_count: int = 1

    def __post_init__(self) -> None:
        if self.passed is not None and not isinstance(self.passed, bool):
            raise TypeError("ProbeOutcome.passed 必须是 bool 或 None")
        assert_int(self.value, self.sample_count, _where="ProbeOutcome")
        if type(self.value) is not int or type(self.sample_count) is not int:
            raise TypeError("ProbeOutcome 测量必须使用严格整数")
        if self.sample_count < 0:
            raise ValueError("ProbeOutcome.sample_count 不得为负")


@dataclass(frozen=True)
class ProbeObservation:
    """一个身份、维度和证据账本下的不可混计探针结果。"""

    identity: EvaluationDataIdentity
    split: ProtocolKey
    probe_kind: ProtocolKey
    dimension: ProtocolKey
    evidence: ProtocolKey
    outcome: ProbeOutcome


def evaluate_probe(
        plan: EvaluationPlan,
        assignment: EvaluationAssignment,
        dimension: ProtocolKey,
        evaluator: Callable[[], ProbeOutcome], *,
        state_reader: Callable[[], Any] | None = None,
        ) -> ProbeObservation:
    """执行一个注册探针，并可核验执行前后宿主状态 bit-identical。"""
    registered = plan.assignment_for(assignment.identity)
    if registered != assignment:
        raise EvaluationProtocolError("探针记录与权威计划不一致")
    if assignment.split == plan.protocol.training_split:
        raise EvaluationProtocolError("训练记录不得进入 probe API")
    if dimension not in assignment.dimensions:
        raise EvaluationProtocolError("探针请求了记录未声明的维度")
    if assignment.probe_kind is None:
        raise EvaluationProtocolError("探针记录缺少注入式 probe kind")
    before = (None if state_reader is None
              else CanonicalIdentity.from_value(state_reader()))
    outcome = evaluator()
    if not isinstance(outcome, ProbeOutcome):
        raise TypeError("probe evaluator 必须返回 ProbeOutcome")
    after = (None if state_reader is None
             else CanonicalIdentity.from_value(state_reader()))
    if before != after:
        raise EvaluationStatePollutionError("探针改变了宿主状态")
    return ProbeObservation(
        identity=assignment.identity,
        split=assignment.split,
        probe_kind=assignment.probe_kind,
        dimension=dimension,
        evidence=plan.protocol.evidence_for(assignment.split),
        outcome=outcome,
    )


@dataclass(frozen=True, order=True)
class DimensionMeasurement:
    """一个维度在一个证据账本中的 PASS、FAIL 和 NE 独立计数。"""

    dimension: ProtocolKey
    evidence: ProtocolKey
    planned: int
    passed: int
    failed: int
    not_evaluated: int

    def __post_init__(self) -> None:
        assert_int(
            self.planned,
            self.passed,
            self.failed,
            self.not_evaluated,
            _where="DimensionMeasurement",
        )
        if any(type(value) is not int or value < 0 for value in (
                self.planned, self.passed, self.failed,
                self.not_evaluated)):
            raise ValueError("DimensionMeasurement 计数必须是非负严格整数")
        if self.passed + self.failed + self.not_evaluated != self.planned:
            raise ValueError("DimensionMeasurement 分项之和必须等于计划数")


@dataclass(frozen=True)
class EvaluationReport:
    """只保存分维分证据计数，不提供可掩盖 FAIL/NE 的综合分数。"""

    measurements: tuple[DimensionMeasurement, ...]


def build_evaluation_report(
        plan: EvaluationPlan,
        observations: Iterable[ProbeObservation],
        ) -> EvaluationReport:
    """按维度和证据账本汇总探针，EXTERNAL 永不混入统计 episode。"""
    planned: dict[tuple[ProtocolKey, ProtocolKey], int] = {}
    assignments_by_key = {
        assignment.identity.lookup_key(): assignment
        for assignment in plan.assignments
    }
    for assignment in plan.assignments:
        if assignment.split == plan.protocol.training_split:
            continue
        evidence = plan.protocol.evidence_for(assignment.split)
        for dimension in assignment.dimensions:
            key = (dimension, evidence)
            planned[key] = planned.get(key, 0) + 1

    seen: set[tuple[tuple[tuple[int, ...], bytes], ProtocolKey]] = set()
    counts: dict[tuple[ProtocolKey, ProtocolKey], list[int]] = {}
    for observation in observations:
        identity_key = observation.identity.lookup_key()
        assignment = assignments_by_key.get(identity_key)
        if assignment is None:
            raise EvaluationProtocolError("报告包含计划外探针结果")
        if (observation.split != assignment.split
                or observation.probe_kind != assignment.probe_kind
                or observation.dimension not in assignment.dimensions):
            raise EvaluationProtocolError("探针结果与计划字段不一致")
        expected_evidence = plan.protocol.evidence_for(assignment.split)
        if observation.evidence != expected_evidence:
            raise EvaluationProtocolError(
                "统计 episode 与 EXTERNAL 证据发生混账")
        observation_key = (identity_key, observation.dimension)
        if observation_key in seen:
            raise EvaluationProtocolError("同一探针维度结果重复")
        seen.add(observation_key)
        key = (observation.dimension, observation.evidence)
        bucket = counts.setdefault(key, [0, 0, 0])
        if observation.outcome.passed is True:
            bucket[0] += 1
        elif observation.outcome.passed is False:
            bucket[1] += 1
        else:
            bucket[2] += 1

    measurements: list[DimensionMeasurement] = []
    for key in sorted(planned, key=lambda item: (
            item[0].stable_key(), item[1].stable_key())):
        pass_count, fail_count, explicit_ne = counts.get(key, [0, 0, 0])
        measured = pass_count + fail_count + explicit_ne
        if measured > planned[key]:
            raise EvaluationProtocolError("探针结果数量超过计划数")
        measurements.append(DimensionMeasurement(
            dimension=key[0],
            evidence=key[1],
            planned=planned[key],
            passed=pass_count,
            failed=fail_count,
            not_evaluated=explicit_ne + planned[key] - measured,
        ))
    return EvaluationReport(tuple(measurements))


def _manifest_bytes(plan: EvaluationPlan) -> bytes:
    """生成同时保留完整计划和计划摘要的规范 manifest 包装。"""
    envelope = {
        "plan": plan.to_dict(),
        "sha256": plan.sha256(),
    }
    text = json.dumps(
        envelope,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (text + "\n").encode("utf-8")


def write_evaluation_plan(
        plan: EvaluationPlan, path: str | Path) -> Path:
    """幂等写不可变评测 manifest，既有不同内容必须换新路径。"""
    target = Path(path)
    payload = _manifest_bytes(plan)
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise EvaluationProtocolError(
                "既有评测 manifest 内容不同，必须使用新版本路径")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(payload)
    return target


def read_evaluation_plan(path: str | Path) -> EvaluationPlan:
    """严格读取评测 manifest，核验完整计划摘要并重跑全部守卫。"""
    try:
        raw = Path(path).read_text(encoding="utf-8")
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationProtocolError("评测 manifest 无法严格读取") from exc
    if (not isinstance(value, dict)
            or set(value) != {"plan", "sha256"}
            or not isinstance(value["plan"], dict)
            or not isinstance(value["sha256"], str)):
        raise EvaluationProtocolError("评测 manifest 包装字段非法")
    plan = EvaluationPlan.from_dict(value["plan"])
    if plan.sha256() != value["sha256"]:
        raise EvaluationProtocolError("评测 manifest 计划 SHA-256 不匹配")
    if _manifest_bytes(plan) != Path(path).read_bytes():
        raise EvaluationProtocolError("评测 manifest 不是规范编码")
    return plan


__all__ = [
    "CanonicalIdentity",
    "DimensionMeasurement",
    "EvaluationAssignment",
    "EvaluationDataIdentity",
    "EvaluationLeakageError",
    "EvaluationPartition",
    "EvaluationPlan",
    "EvaluationProtocol",
    "EvaluationProtocolError",
    "EvaluationReport",
    "EvaluationStatePollutionError",
    "ProbeObservation",
    "ProbeOutcome",
    "ProtocolKey",
    "build_evaluation_report",
    "canonical_payload",
    "collected_item_content_identity",
    "evaluate_probe",
    "make_evaluation_data_identity",
    "read_evaluation_plan",
    "source_cluster_identity",
    "write_evaluation_plan",
]
