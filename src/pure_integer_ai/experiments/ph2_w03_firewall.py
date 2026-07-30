"""W-03 train-only payload transport 与交付防火墙。"""
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
from pure_integer_ai.experiments.ph2_w03_continuity import (
    W03PublicationObservation,
    validate_w03_publication_observation,
)
from pure_integer_ai.experiments.ph2_w03_payload import W03TrainingPayload
from pure_integer_ai.experiments.ph2_w03_contract import (
    W03ContractError,
    W03FrozenContext,
    W03PayloadAudit,
    W03PayloadBinding,
    W03RunRequest,
    validate_w03_request,
)


def _has_symlink_component(root: Path, relative: str) -> bool:
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
            raise W03ContractError("W-03 payload path contains a symlink")
        target = lexical.resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise W03ContractError("W-03 payload path escaped or is not a file")
        return target
    raise W03ContractError(f"W-03 frozen payload file is missing: {relative}")


class W03PayloadFirewall:
    """最多一次交付精确 12 个 candidate 加 6 个 teacher 文件。"""

    def __init__(
            self,
            primary: Path,
            dependency: Path,
            context: W03FrozenContext,
            request: W03RunRequest,
            audit: W03PayloadAudit,
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
            context: W03FrozenContext,
            request: W03RunRequest,
            *,
            publication_observation: W03PublicationObservation,
            dependency_root: str | Path | None = None,
            audit: W03PayloadAudit | None = None,
            ) -> "W03PayloadFirewall":
        """在任何 transport attempt 前闭合 request 和 publication 门。"""
        if audit is not None and not isinstance(audit, W03PayloadAudit):
            raise W03ContractError("W-03 payload audit type is invalid")
        actual_audit = audit if audit is not None else W03PayloadAudit()
        validate_w03_request(context, request)
        validate_w03_publication_observation(
            context.publication_baseline, publication_observation)
        primary = Path(repository_root).resolve()
        dependency = (Path(dependency_root).resolve()
                      if dependency_root is not None else primary)
        return cls(primary, dependency, context, request, actual_audit)

    def _read_binding(self, binding: W03PayloadBinding):
        target = _overlay_file(
            self._primary, self._dependency, binding.relative_path)
        local_parts = Path(binding.file_identity.relative_path).parts
        artifact_root = target.parents[len(local_parts) - 1]
        self.audit.transport_attempts += 1
        try:
            self.audit.transport_bytes += target.stat().st_size
            return read_record_artifact(artifact_root, binding.file_identity)
        except (DatasetArtifactIOError, OSError) as exc:
            raise W03ContractError(
                f"payload transport/gzip/SHA-256 verification failed: "
                f"{binding.relative_path}"
            ) from exc

    @staticmethod
    def _validate_records(
            source_refs: list[SourceRefRecord],
            observations: list[ObservationRecord],
            teacher: list[TeacherEvidenceRecord],
            ) -> None:
        source_keys = {item.stable_key for item in source_refs}
        observation_keys = {item.stable_key for item in observations}
        if (len(source_keys) != len(source_refs)
                or len(observation_keys) != len(observations)):
            raise W03ContractError("W-03 train SourceRef/Observation keys are duplicated")
        if any(
                item.w_stage not in {"W-02", "W-03"}
                or item.split != "train"
                or item.source_ref_key not in source_keys
                for item in observations):
            raise W03ContractError("Observation is not a closed W-02/W-03 train record")
        if any(
                item.visible_from_stage not in {"W-02", "W-03"}
                or item.observation_key not in observation_keys
                or item.source_ref_key not in source_keys
                for item in teacher):
            raise W03ContractError("teacher Evidence is non-train, future, or unreferenced")

    def read_training_payload(self) -> W03TrainingPayload:
        """校验全部文件和引用后，记账一次原子交付。"""
        if self._consumed:
            raise W03ContractError("the same W-03 payload firewall cannot be replayed")
        self._consumed = True
        source_refs: list[SourceRefRecord] = []
        observations: list[ObservationRecord] = []
        teacher: list[TeacherEvidenceRecord] = []
        try:
            for binding in self._context.candidate_payload_bindings:
                records = self._read_binding(binding)
                if binding.owner_kind == "source":
                    if any(not isinstance(item, SourceRefRecord) for item in records):
                        raise W03ContractError("candidate SourceRef file contains other records")
                    source_refs.extend(records)
                elif binding.owner_kind == "observation":
                    if any(not isinstance(item, ObservationRecord) for item in records):
                        raise W03ContractError("candidate Observation file contains other records")
                    observations.extend(records)
                else:
                    raise W03ContractError("candidate whitelist contains a private owner")
            for binding in self._context.teacher_evidence_bindings:
                records = self._read_binding(binding)
                if any(not isinstance(item, TeacherEvidenceRecord) for item in records):
                    raise W03ContractError("teacher Evidence file contains other records")
                teacher.extend(records)
            self._validate_records(source_refs, observations, teacher)
        except W03ContractError:
            raise
        except (TypeError, ValueError, KeyError) as exc:
            raise W03ContractError("W-03 train payload references or owner are invalid") from exc

        bindings = (
            *self._context.candidate_payload_bindings,
            *self._context.teacher_evidence_bindings,
        )
        self.audit.payload_gets += len(bindings)
        self.audit.payload_bytes += sum(item.transport_size_bytes for item in bindings)
        self.audit.source_ref_reads += len(source_refs)
        self.audit.observation_reads += len(observations)
        self.audit.teacher_evidence_reads += len(teacher)
        return W03TrainingPayload(
            tuple(source_refs), tuple(observations), tuple(teacher))


__all__ = ["W03PayloadFirewall", "W03TrainingPayload"]
