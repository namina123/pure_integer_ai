"""DLG-05 公开 preflight 到唯一 label-late formal 的资格闸。

该模块只合取已经发生的无标签证据，不执行 runtime、不读取 evaluator label，
也不把资格闸结果称为泛化或问答通过。所有 replay/storage/fault 字段必须由
外部真实 harness 在运行后提供；缺任何一项都 fail closed。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)
from pure_integer_ai.experiments.conversation_heldout_family import (
    ConversationHeldOutInputCatalog,
)
from pure_integer_ai.experiments.conversation_heldout_preflight import (
    ConversationHeldOutAxisInputAudit,
)
from pure_integer_ai.experiments.conversation_heldout_protocol import (
    AXIS_CONFLICT,
    AXIS_MEMORY_MISS,
    ConversationHeldOutManifest,
)
from pure_integer_ai.experiments.conversation_heldout_runtime import (
    ConversationHeldOutSelectionReceipt,
    SELECTION_FIRST_LABEL_FREE_CONTRACT_KEY,
)


# object-model: exception
class ConversationHeldOutQualificationError(RuntimeError):
    """DLG-05 formal qualification 的硬闸未全部闭合。"""


def _key(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验非空纯整数摘要键。"""
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int or item < 0 for item in value)):
        raise ConversationHeldOutQualificationError(
            f"{label} 必须是非空非负整数 tuple")
    return value


ROLLBACK_FAULT_CONTRACT_PREFIX = (31005, 1301, 1)


def conversation_heldout_rollback_fault_key(
        error: BaseException,
        ) -> tuple[int, ...]:
    """从真实捕获的故障类型与消息生成审计键。"""
    if not isinstance(error, BaseException):
        raise TypeError("rollback fault key 需要真实异常")
    error_type = tuple(ord(character) for character in type(error).__qualname__)
    error_message = tuple(ord(character) for character in str(error))
    return (
        *ROLLBACK_FAULT_CONTRACT_PREFIX,
        *integer_tuple_fingerprint(
            (len(error_type), *error_type,
             len(error_message), *error_message),
            domain="conversation.heldout.rollback.fault.v1",
        ),
    )


def _selection_receipt(
        value: ConversationHeldOutSelectionReceipt,
        *,
        label: str,
        ) -> ConversationHeldOutSelectionReceipt:
    """核验一次由 selection-first runner 产生的无标签 receipt。"""
    if not isinstance(value, ConversationHeldOutSelectionReceipt):
        raise ConversationHeldOutQualificationError(
            f"{label} 必须是 selection-first receipt")
    if value.contract_key != SELECTION_FIRST_LABEL_FREE_CONTRACT_KEY:
        raise ConversationHeldOutQualificationError(
            f"{label} label-free contract 漂移")
    return value


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutRollbackRecoveryReceipt:
    """一次真实故障注入、快照恢复和确定性重放的无标签 receipt。"""

    fault_key: tuple[int, ...]
    before_snapshot_key: tuple[int, ...]
    after_fault_snapshot_key: tuple[int, ...]
    after_recovery_snapshot_key: tuple[int, ...]
    recovered_execution: ConversationHeldOutSelectionReceipt

    def __post_init__(self) -> None:
        """核验故障与恢复 receipt 的纯整数边界。"""
        for label, value in (
                ("rollback fault", self.fault_key),
                ("rollback before snapshot", self.before_snapshot_key),
                ("rollback after-fault snapshot", self.after_fault_snapshot_key),
                ("rollback after-recovery snapshot", self.after_recovery_snapshot_key)):
            _key(value, label=label)
        if self.fault_key[:len(ROLLBACK_FAULT_CONTRACT_PREFIX)] \
                != ROLLBACK_FAULT_CONTRACT_PREFIX:
            raise ConversationHeldOutQualificationError(
                "rollback fault receipt 不是由真实异常键构造")
        _selection_receipt(self.recovered_execution,
                           label="rollback recovered execution")

    @property
    def recovered_clean(self) -> bool:
        """返回故障前、故障后和重放后的持久状态是否逐字节稳定。"""
        return (
            self.before_snapshot_key == self.after_fault_snapshot_key
            == self.after_recovery_snapshot_key
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回 rollback recovery receipt 的纯整数键。"""
        result = [1]
        for value in (
                self.fault_key,
                self.before_snapshot_key,
                self.after_fault_snapshot_key,
                self.after_recovery_snapshot_key,
                self.recovered_execution.stable_key()):
            result.extend((len(value), *value))
        return tuple(result)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutQualificationReceipt:
    """一次公开 preflight qualification 的不可变合取证据。"""

    manifest_key: tuple[int, ...]
    catalog_key: tuple[int, ...]
    execution: ConversationHeldOutSelectionReceipt
    fresh_execution: ConversationHeldOutSelectionReceipt
    clone_execution: ConversationHeldOutSelectionReceipt
    resumed_execution: ConversationHeldOutSelectionReceipt
    rollback_recovery: ConversationHeldOutRollbackRecoveryReceipt
    storage_snapshot_keys: tuple[tuple[int, ...], ...]
    axis_audit: tuple[ConversationHeldOutAxisInputAudit, ...]

    def __post_init__(self) -> None:
        """逐项核验资格输入，不读取或承载任何 evaluator label。"""
        _key(self.manifest_key, label="qualification manifest key")
        _key(self.catalog_key, label="qualification catalog key")
        for label, value in (
                ("execution", self.execution),
                ("fresh execution", self.fresh_execution),
                ("clone execution", self.clone_execution),
                ("resumed execution", self.resumed_execution)):
            _selection_receipt(value, label=label)
        if not isinstance(self.rollback_recovery,
                          ConversationHeldOutRollbackRecoveryReceipt):
            raise ConversationHeldOutQualificationError(
                "qualification rollback recovery receipt 类型错误")
        if not isinstance(self.storage_snapshot_keys, tuple) \
                or len(self.storage_snapshot_keys) < 4:
            raise ConversationHeldOutQualificationError(
                "qualification storage snapshot keys 必须覆盖 host/fresh/clone/resume")
        for index, value in enumerate(self.storage_snapshot_keys):
            _key(value, label=f"qualification storage snapshot[{index}]")
        if (not isinstance(self.axis_audit, tuple)
                or any(not isinstance(item, ConversationHeldOutAxisInputAudit)
                       for item in self.axis_audit)):
            raise ConversationHeldOutQualificationError(
                "qualification axis audit 类型错误")
        if len({item.axis for item in self.axis_audit}) != len(self.axis_audit):
            raise ConversationHeldOutQualificationError(
                "qualification axis audit 不得重复轴")

    @property
    def replay_stable(self) -> bool:
        """返回五次无标签 observations 是否逐项 bit-identical。"""
        return (
            self.fresh_execution.observations == self.execution.observations
            and self.clone_execution.observations == self.execution.observations
            and self.resumed_execution.observations == self.execution.observations
            and (self.rollback_recovery.recovered_execution.observations
                 == self.execution.observations)
        )

    @property
    def storage_stable(self) -> bool:
        """返回所有 supplied backend snapshot keys 是否相等。"""
        return len(set(self.storage_snapshot_keys)) == 1

    def stable_key(self) -> tuple[int, ...]:
        """返回资格证据的完整整数键，不包含 labels。"""
        result = [1]
        for value in (self.manifest_key, self.catalog_key):
            result.extend((len(value), *value))
        for execution in (
                self.execution,
                self.fresh_execution,
                self.clone_execution,
                self.resumed_execution):
            key = execution.stable_key()
            result.extend((len(key), *key))
        key = self.rollback_recovery.stable_key()
        result.extend((len(key), *key))
        result.append(len(self.storage_snapshot_keys))
        for value in self.storage_snapshot_keys:
            result.extend((len(value), *value))
        result.append(len(self.axis_audit))
        for item in self.axis_audit:
            result.extend((
                len(item.axis.components),
                *item.axis.components,
                item.typed_input_bound,
                item.semantic_runtime_bound,
                item.manifest_only,
                len(item.case_keys),
            ))
            for case_key in item.case_keys:
                result.extend((
                    len(case_key.components),
                    *case_key.components,
                ))
        return tuple(result)


def qualify_dlg05_preflight(
        catalog: ConversationHeldOutInputCatalog,
        manifest: ConversationHeldOutManifest,
        receipt: ConversationHeldOutQualificationReceipt,
        ) -> ConversationHeldOutQualificationReceipt:
    """合取公开 preflight 证据并返回可审计资格 receipt。"""
    if not isinstance(catalog, ConversationHeldOutInputCatalog):
        raise TypeError("qualification catalog 类型错误")
    if not isinstance(manifest, ConversationHeldOutManifest):
        raise TypeError("qualification manifest 类型错误")
    if not isinstance(receipt, ConversationHeldOutQualificationReceipt):
        raise TypeError("qualification receipt 类型错误")
    if receipt.manifest_key != manifest.stable_key():
        raise ConversationHeldOutQualificationError(
            "qualification manifest key 漂移")
    if receipt.catalog_key != catalog.stable_key():
        raise ConversationHeldOutQualificationError(
            "qualification catalog key 漂移")
    catalog.assert_manifest_rebuildable(manifest)
    expected_cases = tuple(case.case_key for case in manifest.cases)
    for label, observations in (
            ("observations", receipt.execution.observations),
            ("fresh observations", receipt.fresh_execution.observations),
            ("clone observations", receipt.clone_execution.observations),
            ("resumed observations", receipt.resumed_execution.observations),
            ("rollback recovered observations",
             receipt.rollback_recovery.recovered_execution.observations)):
        if tuple(item.case_key for item in observations) != expected_cases:
            raise ConversationHeldOutQualificationError(
                f"{label} 未逐 case 覆盖 manifest")
    for label, execution in (
            ("execution", receipt.execution),
            ("fresh execution", receipt.fresh_execution),
            ("clone execution", receipt.clone_execution),
            ("resumed execution", receipt.resumed_execution),
            ("rollback recovered execution",
             receipt.rollback_recovery.recovered_execution)):
        if execution.manifest_key != receipt.manifest_key:
            raise ConversationHeldOutQualificationError(
                f"{label} manifest key 漂移")
    audited = {item.axis: item for item in receipt.axis_audit}
    input_axes = set(manifest.required_axes) - {
        AXIS_CONFLICT, AXIS_MEMORY_MISS}
    if set(audited) != set(manifest.required_axes):
        raise ConversationHeldOutQualificationError(
            "qualification axis audit 未覆盖 manifest required axes")
    for axis in input_axes:
        item = audited[axis]
        if item.typed_input_bound != 1 or item.manifest_only != 0:
            raise ConversationHeldOutQualificationError(
                "qualification 存在未闭合或 manifest-only typed axis")
    proven = {
        axis for observation in receipt.execution.observations
        for axis in observation.proven_axis_keys
    }
    if set(manifest.required_axes) - proven:
        raise ConversationHeldOutQualificationError(
            "qualification observations 未逐轴提供真实 runtime proof")
    if not receipt.replay_stable:
        raise ConversationHeldOutQualificationError(
            "qualification fresh/clone/resume/rollback observations 漂移")
    if not receipt.storage_stable:
        raise ConversationHeldOutQualificationError(
            "qualification backend snapshot 漂移")
    if not receipt.rollback_recovery.recovered_clean:
        raise ConversationHeldOutQualificationError(
            "qualification rollback fault recovery snapshot 未闭合")
    train_contents = {item.payload for item in catalog.train_contents}
    train_dedup = {item.payload for item in catalog.train_dedup_clusters}
    train_provenance = {item.payload for item in catalog.train_provenance_clusters}
    heldout_contents = {
        turn.content.payload
        for case in catalog.cases for turn in case.turns
    }
    heldout_dedup = {case.dedup_cluster.payload for case in catalog.cases}
    heldout_provenance = {case.provenance_cluster.payload for case in catalog.cases}
    if (heldout_contents & train_contents
            or heldout_dedup & train_dedup
            or heldout_provenance & train_provenance):
        raise ConversationHeldOutQualificationError(
            "qualification leakage boundary 未闭合")
    if any(
            execution.contract_key != SELECTION_FIRST_LABEL_FREE_CONTRACT_KEY
            for execution in (
                receipt.execution,
                receipt.fresh_execution,
                receipt.clone_execution,
                receipt.resumed_execution,
                receipt.rollback_recovery.recovered_execution)):
        raise ConversationHeldOutQualificationError(
            "qualification execution 未保持 label-free")
    return receipt


__all__ = [
    "conversation_heldout_rollback_fault_key",
    "ConversationHeldOutQualificationError",
    "ConversationHeldOutQualificationReceipt",
    "ConversationHeldOutRollbackRecoveryReceipt",
    "qualify_dlg05_preflight",
    "ROLLBACK_FAULT_CONTRACT_PREFIX",
]
