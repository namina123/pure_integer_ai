"""W-07 train-only payload transport firewall。"""
from __future__ import annotations

from pathlib import Path

from pure_integer_ai.experiments.ph2_dataset_contract import (
    ObservationRecord,
    SourceRefRecord,
    TeacherEvidenceRecord,
)
from pure_integer_ai.experiments.ph2_dataset_io import (
    DatasetArtifactIOError,
    read_record_artifact,
)
from pure_integer_ai.experiments.ph2_w07_contract import (
    W07ContractError,
    W07FrozenContext,
    W07PayloadAudit,
    W07PayloadBinding,
    W07RunRequest,
    validate_w07_request,
)
from pure_integer_ai.experiments.ph2_w07_payload import W07TrainingPayload


def _has_symlink_component(root: Path, relative: str) -> bool:
    """拒绝 repository root 以下任意 symlink 路径分量。"""
    current = root
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _payload_file(root: Path, relative: str) -> Path:
    """解析并限制 payload 文件必须位于 repository root 内。"""
    if _has_symlink_component(root, relative):
        raise W07ContractError("W-07 payload path 含 symlink")
    target = (root / relative).resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise W07ContractError("W-07 payload path 越界") from error
    if not target.is_file():
        raise W07ContractError("W-07 payload 文件不存在")
    return target


class W07PayloadFirewall:
    """在完整 public preflight 后一次性交付 train-only payload。"""

    def __init__(
            self,
            root: Path,
            context: W07FrozenContext,
            request: W07RunRequest,
            audit: W07PayloadAudit,
            ) -> None:
        self._root = root
        self._context = context
        self._request = request
        self.audit = audit
        self._consumed = False

    @classmethod
    def open(
            cls,
            repository_root: str | Path,
            context: W07FrozenContext,
            request: W07RunRequest,
            *,
            audit: W07PayloadAudit | None = None,
            ) -> "W07PayloadFirewall":
        """在首次 transport 前验证请求和全部精确 whitelist。"""
        if audit is not None and not isinstance(audit, W07PayloadAudit):
            raise W07ContractError("W-07 payload audit 类型非法")
        validate_w07_request(context, request)
        return cls(
            Path(repository_root).resolve(),
            context,
            request,
            audit if audit is not None else W07PayloadAudit(),
        )

    def _read(self, binding: W07PayloadBinding) -> tuple[object, ...]:
        """按 ArtifactFileIdentity 验 gzip、transport/content SHA 后读取记录。"""
        target = _payload_file(self._root, binding.relative_path)
        local_parts = Path(binding.file_identity.relative_path).parts
        artifact_root = target.parents[len(local_parts) - 1]
        self.audit.transport_attempts += 1
        try:
            self.audit.transport_bytes += target.stat().st_size
            return read_record_artifact(artifact_root, binding.file_identity)
        except (DatasetArtifactIOError, OSError) as error:
            raise W07ContractError(
                f"W-07 payload transport/SHA 校验失败：{binding.relative_path}"
            ) from error

    @staticmethod
    def _validate(
            source_refs: list[SourceRefRecord],
            observations: list[ObservationRecord],
            teachers: list[TeacherEvidenceRecord],
            ) -> None:
        """确认交付只含闭合 W07 train 引用，不含 evaluator/future owner。"""
        source_keys = {item.stable_key for item in source_refs}
        observation_keys = {item.stable_key for item in observations}
        if len(source_keys) != len(source_refs):
            raise W07ContractError("W-07 SourceRef key 重复")
        if len(observation_keys) != len(observations):
            raise W07ContractError("W-07 Observation key 重复")
        if any(
                item.w_stage != "W-07"
                or item.split != "train"
                or item.source_ref_key not in source_keys
                for item in observations):
            raise W07ContractError("W-07 Observation 非闭合 train record")
        if any(
                item.visible_from_stage != "W-07"
                or item.observation_key not in observation_keys
                or item.source_ref_key not in source_keys
                for item in teachers):
            raise W07ContractError("W-07 TeacherEvidence 非闭合 train record")

    def read_training_payload(self) -> W07TrainingPayload:
        """完整校验后原子交付一次，不允许同 firewall replay。"""
        if self._consumed:
            raise W07ContractError("同一 W-07 payload firewall 禁止重放")
        self._consumed = True
        source_refs: list[SourceRefRecord] = []
        observations: list[ObservationRecord] = []
        teachers: list[TeacherEvidenceRecord] = []
        for binding in self._context.candidate_payload_bindings:
            records = self._read(binding)
            if binding.owner_kind == "source":
                if any(not isinstance(item, SourceRefRecord) for item in records):
                    raise W07ContractError("W-07 source 文件混入其他 record")
                source_refs.extend(records)
            elif binding.owner_kind == "observation":
                if any(not isinstance(item, ObservationRecord) for item in records):
                    raise W07ContractError("W-07 observation 文件混入其他 record")
                observations.extend(records)
            else:
                raise W07ContractError("W-07 candidate whitelist 混入 forbidden owner")
        for binding in self._context.teacher_evidence_bindings:
            records = self._read(binding)
            if any(not isinstance(item, TeacherEvidenceRecord) for item in records):
                raise W07ContractError("W-07 teacher 文件混入其他 record")
            teachers.extend(records)
        self._validate(source_refs, observations, teachers)
        bindings = (
            *self._context.candidate_payload_bindings,
            *self._context.teacher_evidence_bindings,
        )
        payload_bytes = sum(
            item.file_identity.transport_size_bytes for item in bindings)
        budget = dict(self._context.resource_budget)
        if (len(bindings) > budget["max_payload_gets"]
                or payload_bytes > budget["max_payload_bytes"]
                or len(observations) > budget["max_records"]):
            raise W07ContractError("W-07 train payload 超出冻结资源预算")
        self.audit.payload_gets += len(bindings)
        self.audit.payload_bytes += payload_bytes
        self.audit.source_ref_reads += len(source_refs)
        self.audit.observation_reads += len(observations)
        self.audit.teacher_evidence_reads += len(teachers)
        return W07TrainingPayload(
            tuple(source_refs), tuple(observations), tuple(teachers))


__all__ = ["W07PayloadFirewall", "W07TrainingPayload"]
