"""W-04 train-only payload transport 与交付防火墙。"""
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
from pure_integer_ai.experiments.ph2_w04_contract import (
    W04ContractError,
    W04FrozenContext,
    W04PayloadAudit,
    W04PayloadBinding,
    W04RunRequest,
    validate_w04_request,
)
from pure_integer_ai.experiments.ph2_w04_payload import W04TrainingPayload


def _has_symlink_component(root: Path, relative: str) -> bool:
    """检查 POSIX 相对路径任一组件是否被 symlink 替换。"""
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _overlay_file(primary: Path, dependency: Path, relative: str) -> Path:
    """解析精确文件，同时拒绝路径和 symlink 替换。"""
    parts = Path(*PurePosixPath(relative).parts)
    for root in (primary, dependency):
        lexical = root / parts
        if not lexical.exists() and not lexical.is_symlink():
            continue
        if _has_symlink_component(root, relative):
            raise W04ContractError("W-04 payload path contains a symlink")
        target = lexical.resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise W04ContractError("W-04 payload path escaped or is not a file")
        return target
    raise W04ContractError(f"W-04 frozen payload file is missing: {relative}")


class W04PayloadFirewall:
    """最多一次交付 W-04 candidate train 文件与 teacher train Evidence。"""

    def __init__(
            self,
            primary: Path,
            dependency: Path,
            context: W04FrozenContext,
            request: W04RunRequest,
            audit: W04PayloadAudit,
            ) -> None:
        self._primary = primary
        self._dependency = dependency
        self._context = context
        self._request = request
        self.audit = audit
        self._consumed = False

    @classmethod
    def open(
            cls,
            repository_root: str | Path,
            context: W04FrozenContext,
            request: W04RunRequest,
            *,
            dependency_root: str | Path | None = None,
            audit: W04PayloadAudit | None = None,
            ) -> "W04PayloadFirewall":
        """在任何 transport attempt 前闭合 request、owner 和 split 门。"""
        if audit is not None and not isinstance(audit, W04PayloadAudit):
            raise W04ContractError("W-04 payload audit type is invalid")
        actual_audit = audit if audit is not None else W04PayloadAudit()
        validate_w04_request(context, request)
        primary = Path(repository_root).resolve()
        dependency = (Path(dependency_root).resolve()
                      if dependency_root is not None else primary)
        return cls(primary, dependency, context, request, actual_audit)

    def _read_binding(self, binding: W04PayloadBinding):
        target = _overlay_file(
            self._primary, self._dependency, binding.relative_path)
        local_parts = Path(binding.file_identity.relative_path).parts
        artifact_root = target.parents[len(local_parts) - 1]
        self.audit.transport_attempts += 1
        try:
            self.audit.transport_bytes += target.stat().st_size
            return read_record_artifact(artifact_root, binding.file_identity)
        except (DatasetArtifactIOError, OSError) as exc:
            raise W04ContractError(
                f"payload transport/gzip/SHA-256 verification failed: "
                f"{binding.relative_path}"
            ) from exc

    @staticmethod
    def _validate_records(
            source_refs: list[SourceRefRecord],
            observations: list[ObservationRecord],
            teacher: list[TeacherEvidenceRecord],
            ) -> None:
        """确认交付内容只有 train owner/split，且 teacher 只引用可见 observation。"""
        source_keys = {item.stable_key for item in source_refs}
        observation_keys = {item.stable_key for item in observations}
        if (len(source_keys) != len(source_refs)
                or len(observation_keys) != len(observations)):
            raise W04ContractError("W-04 train SourceRef/Observation keys are duplicated")
        if any(
                item.split != "train"
                or item.source_ref_key not in source_keys
                for item in observations):
            raise W04ContractError("Observation is not a closed W-04 train record")
        if any(
                item.visible_from_stage not in {"W-02", "W-03", "W-04"}
                or item.observation_key not in observation_keys
                or item.source_ref_key not in source_keys
                for item in teacher):
            raise W04ContractError("teacher Evidence is non-train, future, or unreferenced")

    def read_training_payload(self) -> W04TrainingPayload:
        """校验全部文件和引用后，记账一次原子交付。"""
        if self._consumed:
            raise W04ContractError("the same W-04 payload firewall cannot be replayed")
        self._consumed = True
        source_refs: list[SourceRefRecord] = []
        observations: list[ObservationRecord] = []
        teacher: list[TeacherEvidenceRecord] = []
        try:
            for binding in self._context.candidate_payload_bindings:
                records = self._read_binding(binding)
                if binding.owner_kind == "source":
                    if any(not isinstance(item, SourceRefRecord) for item in records):
                        raise W04ContractError("candidate SourceRef file contains other records")
                    source_refs.extend(records)
                elif binding.owner_kind == "observation":
                    if any(not isinstance(item, ObservationRecord) for item in records):
                        raise W04ContractError("candidate Observation file contains other records")
                    observations.extend(records)
                else:
                    raise W04ContractError("candidate whitelist contains a private owner")
            for binding in self._context.teacher_evidence_bindings:
                records = self._read_binding(binding)
                if any(not isinstance(item, TeacherEvidenceRecord) for item in records):
                    raise W04ContractError("teacher Evidence file contains other records")
                teacher.extend(records)
            self._validate_records(source_refs, observations, teacher)
        except W04ContractError:
            raise
        except (TypeError, ValueError, KeyError) as exc:
            raise W04ContractError(
                "W-04 train payload references or owner are invalid") from exc

        bindings = (
            *self._context.candidate_payload_bindings,
            *self._context.teacher_evidence_bindings,
        )
        self.audit.payload_gets += len(bindings)
        self.audit.payload_bytes += sum(item.transport_size_bytes for item in bindings)
        self.audit.source_ref_reads += len(source_refs)
        self.audit.observation_reads += len(observations)
        self.audit.teacher_evidence_reads += len(teacher)
        return W04TrainingPayload(
            tuple(source_refs), tuple(observations), tuple(teacher))


__all__ = ["W04PayloadFirewall", "W04TrainingPayload"]
