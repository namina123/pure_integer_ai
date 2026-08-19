"""DLG-05 独立对话输入 catalog 与 manifest builder。

catalog 保存可执行的 typed ``QuestionRequest`` 和回合轴；manifest 只保存
其完整整数身份，不保存 expected answer、evaluator label 或 surface。这样
selection-first runner 可以从同一份公开、label-free catalog 重建请求，并在
正式运行前核验 content/source/scope 没有被替换。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.question_answer import QuestionRequest
from pure_integer_ai.experiments.conversation_heldout_language_input import (
    ConversationQuestionLanguageInput,
)
from pure_integer_ai.experiments.conversation_heldout_protocol import (
    ConversationHeldOutCase,
    ConversationHeldOutManifest,
    ConversationHeldOutTurn,
    MEMORY_OFF,
    MEMORY_ON,
)
from pure_integer_ai.experiments.conversation_heldout_runtime import (
    conversation_turn_content_identity,
    conversation_turn_scope_key,
    conversation_turn_source_key,
)
from pure_integer_ai.experiments.evaluation_protocol import (
    CanonicalIdentity,
    ProtocolKey,
)


class ConversationHeldOutCatalogError(RuntimeError):
    """typed catalog、manifest 或请求重建不闭合。"""


def _pack(result: list[int], key: tuple[int, ...]) -> None:
    """把可变长度整数键以边界帧写入 catalog 稳定键。"""
    result.extend((len(key), *key))


def _positive(value: int, *, label: str) -> None:
    """核验正严格整数。"""
    if type(value) is not int or value <= 0:
        raise ConversationHeldOutCatalogError(f"{label} 必须是正严格整数")


def _key(value: ProtocolKey, *, label: str) -> None:
    """核验非空协议键。"""
    if not isinstance(value, ProtocolKey) or not value.components:
        raise ConversationHeldOutCatalogError(f"{label} 必须是非空 ProtocolKey")


def _identity(value: CanonicalIdentity, *, label: str) -> None:
    """核验保留完整载荷的规范身份。"""
    if not isinstance(value, CanonicalIdentity) or not value.payload:
        raise ConversationHeldOutCatalogError(
            f"{label} 必须是非空 CanonicalIdentity")


@dataclass(frozen=True, slots=True)
class ConversationHeldOutCatalogTurn:
    """一个无标签、可执行且可重建的 typed 对话回合输入。"""

    turn_key: ProtocolKey
    ordinal: int
    request: QuestionRequest
    context_mode: ProtocolKey
    memory_mode: ProtocolKey
    reference_mode: ProtocolKey
    rollback_mode: ProtocolKey
    language_input: ConversationQuestionLanguageInput | None = None

    def __post_init__(self) -> None:
        """核验回合输入完整保留 QuestionRequest 和所有运行轴。"""
        _key(self.turn_key, label="catalog turn key")
        _positive(self.ordinal, label="catalog turn ordinal")
        if not isinstance(self.request, QuestionRequest):
            raise TypeError("catalog turn request 必须是 QuestionRequest")
        for name, value in (
                ("context_mode", self.context_mode),
                ("memory_mode", self.memory_mode),
                ("reference_mode", self.reference_mode),
                ("rollback_mode", self.rollback_mode)):
            _key(value, label=f"catalog turn {name}")
        if self.memory_mode not in {MEMORY_OFF, MEMORY_ON}:
            raise ConversationHeldOutCatalogError(
                "catalog turn memory_mode 未注册")
        if self.language_input is not None and not isinstance(
                self.language_input, ConversationQuestionLanguageInput):
            raise TypeError("catalog turn language_input 类型错误")

    @property
    def content(self) -> CanonicalIdentity:
        """返回完整基础 QuestionRequest 的规范内容身份。"""
        return conversation_turn_content_identity(
            self.request,
            language_input_key=(
                () if self.language_input is None
                else self.language_input.stable_key()
            ),
        )

    @property
    def source_key(self) -> ProtocolKey:
        """返回请求目标的完整 SourceRef 身份。"""
        return conversation_turn_source_key(self.request)

    @property
    def scope_key(self) -> ProtocolKey:
        """返回请求实际回答 scope 的完整身份。"""
        return conversation_turn_scope_key(self.request)

    def to_manifest_turn(self) -> ConversationHeldOutTurn:
        """把 typed 请求投影为不携带请求对象的 manifest turn。"""
        return ConversationHeldOutTurn(
            self.turn_key,
            self.ordinal,
            self.content,
            self.source_key,
            self.scope_key,
            self.context_mode,
            self.memory_mode,
            self.reference_mode,
            self.rollback_mode,
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回包含完整 QuestionRequest 的纯整数 catalog 身份。"""
        result = [1]
        for value in (
                self.turn_key.components,
                (self.ordinal,),
                self.request.stable_key(),
                self.context_mode.components,
                self.memory_mode.components,
                self.reference_mode.components,
                self.rollback_mode.components,
                (() if self.language_input is None
                 else self.language_input.stable_key())):
            _pack(result, value)
        return tuple(result)


@dataclass(frozen=True, slots=True)
class ConversationHeldOutCatalogCase:
    """一个独立 case 的 typed 输入和去重/来源簇身份。"""

    case_key: ProtocolKey
    family_key: ProtocolKey
    axis_keys: tuple[ProtocolKey, ...]
    dedup_cluster: CanonicalIdentity
    provenance_cluster: CanonicalIdentity
    turns: tuple[ConversationHeldOutCatalogTurn, ...]

    def __post_init__(self) -> None:
        """核验 case 回合有序、唯一且不混入 evaluator 字段。"""
        _key(self.case_key, label="catalog case key")
        _key(self.family_key, label="catalog family key")
        if (not isinstance(self.axis_keys, tuple) or not self.axis_keys
                or any(not isinstance(item, ProtocolKey)
                       for item in self.axis_keys)):
            raise ConversationHeldOutCatalogError("catalog case axis_keys 非法")
        if len(set(self.axis_keys)) != len(self.axis_keys):
            raise ConversationHeldOutCatalogError("catalog case axis_keys 不得重复")
        _identity(self.dedup_cluster, label="catalog case dedup cluster")
        _identity(self.provenance_cluster, label="catalog case provenance cluster")
        if (not isinstance(self.turns, tuple) or not self.turns
                or any(not isinstance(item, ConversationHeldOutCatalogTurn)
                       for item in self.turns)):
            raise ConversationHeldOutCatalogError("catalog case turns 非空且 typed")
        ordinals = tuple(item.ordinal for item in self.turns)
        if ordinals != tuple(sorted(ordinals)):
            raise ConversationHeldOutCatalogError("catalog case turns 必须按 ordinal 排序")
        if len(set(item.turn_key for item in self.turns)) != len(self.turns):
            raise ConversationHeldOutCatalogError("catalog case turn_key 不得重复")
        if any(item.ordinal != index for index, item in enumerate(self.turns, 1)):
            raise ConversationHeldOutCatalogError(
                "catalog case ordinal 必须从 1 连续递增")
        if any(item.request.source != item.request.response_scope.source
               for item in self.turns):
            raise ConversationHeldOutCatalogError(
                "catalog case request source/scope 不一致")

    def to_manifest_case(self) -> ConversationHeldOutCase:
        """把 case 的 typed turns 投影为无标签 manifest case。"""
        return ConversationHeldOutCase(
            self.case_key,
            self.family_key,
            self.axis_keys,
            self.dedup_cluster,
            self.provenance_cluster,
            tuple(item.to_manifest_turn() for item in self.turns),
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回包含全部 typed turn 的纯整数 case 身份。"""
        result = [1]
        for value in (
                self.case_key.components,
                self.family_key.components,
                self.dedup_cluster.payload,
                self.provenance_cluster.payload):
            _pack(result, value)
        result.append(len(self.axis_keys))
        for axis in self.axis_keys:
            _pack(result, axis.components)
        result.append(len(self.turns))
        for turn in self.turns:
            _pack(result, turn.stable_key())
        return tuple(result)


@dataclass(frozen=True, slots=True)
class ConversationHeldOutInputCatalog:
    """公开、label-free 的可重建 DLG-05 输入 catalog。"""

    version: int
    family_key: ProtocolKey
    train_contents: tuple[CanonicalIdentity, ...]
    train_dedup_clusters: tuple[CanonicalIdentity, ...]
    train_provenance_clusters: tuple[CanonicalIdentity, ...]
    cases: tuple[ConversationHeldOutCatalogCase, ...]

    def __post_init__(self) -> None:
        """核验 catalog 与 TRAIN 的 content/dedup/provenance 边界。"""
        _positive(self.version, label="catalog version")
        _key(self.family_key, label="catalog family key")
        for name, values in (
                ("train_contents", self.train_contents),
                ("train_dedup_clusters", self.train_dedup_clusters),
                ("train_provenance_clusters", self.train_provenance_clusters)):
            if not isinstance(values, tuple):
                raise TypeError(f"catalog {name} 必须是 tuple")
            for item in values:
                _identity(item, label=f"catalog {name}")
            if len({item.payload for item in values}) != len(values):
                raise ConversationHeldOutCatalogError(f"catalog {name} 不得重复")
        if (not isinstance(self.cases, tuple) or not self.cases
                or any(not isinstance(item, ConversationHeldOutCatalogCase)
                       for item in self.cases)):
            raise ConversationHeldOutCatalogError("catalog cases 必须非空 typed tuple")
        if any(item.family_key != self.family_key for item in self.cases):
            raise ConversationHeldOutCatalogError("catalog case family key 漂移")
        case_keys = tuple(item.case_key for item in self.cases)
        if len(set(case_keys)) != len(case_keys):
            raise ConversationHeldOutCatalogError("catalog case_key 不得重复")
        train_contents = {item.payload for item in self.train_contents}
        train_dedup = {item.payload for item in self.train_dedup_clusters}
        train_provenance = {item.payload for item in self.train_provenance_clusters}
        content_owner: dict[tuple[int, ...], ProtocolKey] = {}
        for case in self.cases:
            if case.dedup_cluster.payload in train_dedup:
                raise ConversationHeldOutCatalogError(
                    "held-out dedup cluster 泄漏到 train")
            if case.provenance_cluster.payload in train_provenance:
                raise ConversationHeldOutCatalogError(
                    "held-out provenance cluster 泄漏到 train")
            for turn in case.turns:
                if turn.content.payload in train_contents:
                    raise ConversationHeldOutCatalogError(
                        "held-out content 泄漏到 train")
                owner = content_owner.setdefault(turn.content.payload, case.case_key)
                if owner != case.case_key:
                    raise ConversationHeldOutCatalogError(
                        "同一完整 QuestionRequest content 不得跨 case 重用")

    def to_manifest(self) -> ConversationHeldOutManifest:
        """从 typed catalog 构造仅含身份的 manifest。"""
        axes = []
        for case in self.cases:
            for axis in case.axis_keys:
                if axis not in axes:
                    axes.append(axis)
        return ConversationHeldOutManifest(
            self.version,
            self.family_key,
            self.train_contents,
            self.train_dedup_clusters,
            self.train_provenance_clusters,
            tuple(item.to_manifest_case() for item in self.cases),
            tuple(axes),
            (MEMORY_OFF, MEMORY_ON),
        )

    def manifest_with_axes(
            self,
            required_axes: tuple[ProtocolKey, ...],
            required_memory_modes: tuple[ProtocolKey, ...] =
            (MEMORY_OFF, MEMORY_ON),
            ) -> ConversationHeldOutManifest:
        """按冻结顺序显式声明 required axes，避免隐式改变评测分母。"""
        if not isinstance(required_axes, tuple) or not required_axes:
            raise ConversationHeldOutCatalogError("required_axes 必须非空 tuple")
        if not isinstance(required_memory_modes, tuple) or not required_memory_modes:
            raise ConversationHeldOutCatalogError(
                "required_memory_modes 必须非空 tuple")
        return ConversationHeldOutManifest(
            self.version,
            self.family_key,
            self.train_contents,
            self.train_dedup_clusters,
            self.train_provenance_clusters,
            tuple(item.to_manifest_case() for item in self.cases),
            required_axes,
            required_memory_modes,
        )

    def request_for(
            self,
            case_key: ProtocolKey,
            turn_key: ProtocolKey,
            ) -> QuestionRequest:
        """按完整 case/turn 身份恢复原始 typed QuestionRequest。"""
        return self.turn_for(case_key, turn_key).request

    def turn_for(
            self,
            case_key: ProtocolKey,
            turn_key: ProtocolKey,
            ) -> ConversationHeldOutCatalogTurn:
        """按完整 case/turn 身份恢复 typed turn，供真实 factory 绑定。"""
        _key(case_key, label="request lookup case key")
        _key(turn_key, label="request lookup turn key")
        for case in self.cases:
            if case.case_key != case_key:
                continue
            for turn in case.turns:
                if turn.turn_key == turn_key:
                    return turn
            break
        raise ConversationHeldOutCatalogError(
            "catalog 中不存在指定 case/turn typed input")

    def assert_manifest_rebuildable(
            self,
            manifest: ConversationHeldOutManifest,
            ) -> None:
        """逐回合证明 manifest 身份可由 catalog 的真实 request 重建。"""
        if not isinstance(manifest, ConversationHeldOutManifest):
            raise TypeError("manifest 必须是 ConversationHeldOutManifest")
        expected = tuple(item.to_manifest_case() for item in self.cases)
        if manifest.cases != expected:
            raise ConversationHeldOutCatalogError(
                "manifest case/turn 不是由当前 typed catalog 重建")
        for case in self.cases:
            for turn in case.turns:
                rebuilt = manifest_case_turn(manifest, case.case_key, turn.turn_key)
                if (rebuilt.content != turn.content
                        or rebuilt.source_key != turn.source_key
                        or rebuilt.scope_key != turn.scope_key):
                    raise ConversationHeldOutCatalogError(
                        "manifest content/source/scope 与 typed request 漂移")

    def stable_key(self) -> tuple[int, ...]:
        """返回不含 label 的完整 catalog 整数身份。"""
        result = [self.version]
        _pack(result, self.family_key.components)
        for values in (
                self.train_contents,
                self.train_dedup_clusters,
                self.train_provenance_clusters):
            result.append(len(values))
            for value in values:
                _pack(result, value.payload)
        result.append(len(self.cases))
        for case in self.cases:
            _pack(result, case.stable_key())
        return tuple(result)


def manifest_case_turn(
        manifest: ConversationHeldOutManifest,
        case_key: ProtocolKey,
        turn_key: ProtocolKey,
        ) -> ConversationHeldOutTurn:
    """从 manifest 中按完整身份取回单一 turn，拒绝歧义。"""
    if not isinstance(manifest, ConversationHeldOutManifest):
        raise TypeError("manifest 类型错误")
    matches = [
        turn
        for case in manifest.cases
        if case.case_key == case_key
        for turn in case.turns
        if turn.turn_key == turn_key
    ]
    if len(matches) != 1:
        raise ConversationHeldOutCatalogError(
            "manifest case/turn 不存在或不唯一")
    return matches[0]


def build_conversation_heldout_manifest(
        catalog: ConversationHeldOutInputCatalog,
        required_axes: tuple[ProtocolKey, ...],
        required_memory_modes: tuple[ProtocolKey, ...] =
        (MEMORY_OFF, MEMORY_ON),
        ) -> ConversationHeldOutManifest:
    """构造 manifest 并立即执行 catalog/request 可重建性证明。"""
    if not isinstance(catalog, ConversationHeldOutInputCatalog):
        raise TypeError("catalog 类型错误")
    manifest = catalog.manifest_with_axes(required_axes, required_memory_modes)
    catalog.assert_manifest_rebuildable(manifest)
    return manifest


__all__ = [
    "ConversationHeldOutCatalogCase",
    "ConversationHeldOutCatalogError",
    "ConversationHeldOutCatalogTurn",
    "ConversationHeldOutInputCatalog",
    "build_conversation_heldout_manifest",
    "manifest_case_turn",
]
