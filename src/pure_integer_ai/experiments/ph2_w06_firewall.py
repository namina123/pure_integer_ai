"""W-06 有效 18-pack 的 train-only 一次性交付防火墙。"""
from __future__ import annotations

from pathlib import Path, PurePosixPath

from pure_integer_ai.experiments.ph2_dataset_contract import (
    ObservationRecord,
    SourceRefRecord,
    TeacherEvidenceRecord,
)
from pure_integer_ai.experiments.ph2_dataset_io import (
    DatasetArtifactIOError,
    read_record_artifact,
)
from pure_integer_ai.experiments.ph2_w06_contract import (
    W06ContractError,
    W06FrozenContext,
    W06PayloadAudit,
    W06PayloadBinding,
    W06RunRequest,
    validate_w06_request,
)
from pure_integer_ai.experiments.ph2_w06_payload import W06TrainingPayload


def _has_symlink_component(root: Path, relative: str) -> bool:
    """检查冻结相对路径任一组件是否被符号链接替换。"""
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _payload_file(root: Path, relative: str) -> Path:
    """解析 repository-relative 文件并拒绝越界、symlink 和非文件。"""
    if _has_symlink_component(root, relative):
        raise W06ContractError("W-06 payload path 含 symlink")
    target = (root / Path(*PurePosixPath(relative).parts)).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise W06ContractError("W-06 payload path 越界或文件缺失")
    return target


class W06PayloadFirewall:
    """最多一次读取有效 pack 的 source/train observation/train Evidence。"""

    def __init__(
            self,
            root: Path,
            context: W06FrozenContext,
            request: W06RunRequest,
            audit: W06PayloadAudit,
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
            context: W06FrozenContext,
            request: W06RunRequest,
            *,
            audit: W06PayloadAudit | None = None,
            ) -> "W06PayloadFirewall":
        """在首次 transport 前验证请求和全部精确 whitelist。"""
        if audit is not None and not isinstance(audit, W06PayloadAudit):
            raise W06ContractError("W-06 payload audit 类型非法")
        validate_w06_request(context, request)
        return cls(
            Path(repository_root).resolve(),
            context,
            request,
            audit if audit is not None else W06PayloadAudit(),
        )

    def _read(self, binding: W06PayloadBinding) -> tuple[object, ...]:
        """按 ArtifactFileIdentity 验 gzip、transport/content SHA 后读取记录。"""
        target = _payload_file(self._root, binding.relative_path)
        local_parts = Path(binding.file_identity.relative_path).parts
        artifact_root = target.parents[len(local_parts) - 1]
        self.audit.transport_attempts += 1
        try:
            self.audit.transport_bytes += target.stat().st_size
            return read_record_artifact(artifact_root, binding.file_identity)
        except (DatasetArtifactIOError, OSError) as error:
            raise W06ContractError(
                f"W-06 payload transport/SHA 校验失败：{binding.relative_path}"
            ) from error

    @staticmethod
    def _validate(
            source_refs: list[SourceRefRecord],
            observations: list[ObservationRecord],
            teachers: list[TeacherEvidenceRecord],
            ) -> None:
        """确认交付只含闭合 train 引用，不含 evaluator/future owner。"""
        source_keys = {item.stable_key for item in source_refs}
        observation_keys = {item.stable_key for item in observations}
        if len(source_keys) != len(source_refs):
            raise W06ContractError("W-06 SourceRef key 重复")
        if len(observation_keys) != len(observations):
            raise W06ContractError("W-06 Observation key 重复")
        if any(
                item.split != "train"
                or item.source_ref_key not in source_keys
                for item in observations):
            raise W06ContractError("W-06 Observation 非闭合 train record")
        allowed_stages = {"W-02", "W-03", "W-04", "W-05", "W-06"}
        if any(
                item.visible_from_stage not in allowed_stages
                or item.observation_key not in observation_keys
                or item.source_ref_key not in source_keys
                for item in teachers):
            raise W06ContractError("W-06 teacher Evidence 非 train 或引用泄漏")

    def read_training_payload(self) -> W06TrainingPayload:
        """完整校验后原子交付一次，不允许同 firewall replay。"""
        if self._consumed:
            raise W06ContractError("同一 W-06 payload firewall 禁止重放")
        self._consumed = True
        source_refs: list[SourceRefRecord] = []
        observations: list[ObservationRecord] = []
        teachers: list[TeacherEvidenceRecord] = []
        for binding in self._context.candidate_payload_bindings:
            records = self._read(binding)
            if binding.owner_kind == "source":
                if any(not isinstance(item, SourceRefRecord) for item in records):
                    raise W06ContractError("W-06 source 文件混入其他 record")
                source_refs.extend(records)
            elif binding.owner_kind == "observation":
                if any(not isinstance(item, ObservationRecord) for item in records):
                    raise W06ContractError("W-06 observation 文件混入其他 record")
                observations.extend(records)
            else:
                raise W06ContractError("W-06 candidate whitelist 混入 private owner")
        for binding in self._context.teacher_evidence_bindings:
            records = self._read(binding)
            if any(not isinstance(item, TeacherEvidenceRecord) for item in records):
                raise W06ContractError("W-06 teacher 文件混入其他 record")
            teachers.extend(records)
        self._validate(source_refs, observations, teachers)
        bindings = (
            *self._context.candidate_payload_bindings,
            *self._context.teacher_evidence_bindings,
        )
        self.audit.payload_gets += len(bindings)
        self.audit.payload_bytes += sum(
            item.file_identity.transport_size_bytes for item in bindings)
        self.audit.source_ref_reads += len(source_refs)
        self.audit.observation_reads += len(observations)
        self.audit.teacher_evidence_reads += len(teachers)
        return W06TrainingPayload(
            tuple(source_refs), tuple(observations), tuple(teachers))


__all__ = ["W06PayloadFirewall", "W06TrainingPayload"]
