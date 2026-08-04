"""W-08 candidate/evaluator 分相可见性与 train payload 防火墙。"""
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
from pure_integer_ai.experiments.ph2_w08_contract import (
    W08ContractError,
    W08FileBinding,
    W08FrozenContract,
    W08PayloadAudit,
    W08RunRequest,
    validate_w08_request,
)
from pure_integer_ai.experiments.ph2_w08_payload import W08TrainingPayload


def _has_link_component(root: Path, relative: str) -> bool:
    """拒绝 root 以下任何 symlink 或 Windows junction 分量。"""
    current = root
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink():
            return True
        is_junction = getattr(current, "is_junction", None)
        if is_junction is not None and is_junction():
            return True
    return False


def _payload_file(root: Path, relative: str) -> Path:
    """要求 payload 是 root 内无链接的普通文件。"""
    if _has_link_component(root, relative):
        raise W08ContractError("W-08 payload path contains a link")
    target = (root / relative).resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise W08ContractError("W-08 payload path escapes repository") from error
    if not target.is_file():
        raise W08ContractError("W-08 payload file is missing")
    return target


class W08VisibilityFirewall:
    """只做路径授权，不替 evaluator 提前读取 held-out/label。"""

    def __init__(self, context: W08FrozenContract, audit: W08PayloadAudit) -> None:
        self.context = context
        self.audit = audit

    def authorize_candidate(self, relative_path: str) -> W08FileBinding:
        """Candidate 仅能命中 source/train Observation/train Evidence。"""
        allowed = (*self.context.candidate_bindings, *self.context.teacher_bindings)
        for item in allowed:
            if item.relative_path == relative_path:
                return item
        if relative_path in self.context.future_forbidden_paths:
            raise W08ContractError("W-08 future pack is forbidden to candidate")
        raise W08ContractError("W-08 candidate path is outside exact whitelist")

    def authorize_evaluator(
        self, relative_path: str, *, candidate_sealed: int
    ) -> W08FileBinding:
        """Evaluator 只有在 Candidate 封存后才获得 held-out/label 路径。"""
        if candidate_sealed != 1:
            raise W08ContractError("W-08 evaluator requires sealed candidate")
        if relative_path in self.context.future_forbidden_paths:
            raise W08ContractError("W-08 future pack is forbidden to evaluator")
        for item in self.context.evaluator_bindings:
            if item.relative_path == relative_path:
                return item
        raise W08ContractError("W-08 evaluator path is outside exact whitelist")


class W08PayloadFirewall:
    """在完整 preflight 后一次性交付六个允许 pack 的 train payload。"""

    def __init__(
        self,
        root: Path,
        context: W08FrozenContract,
        request: W08RunRequest,
        audit: W08PayloadAudit,
    ) -> None:
        self._root = root
        self.context = context
        self.request = request
        self.audit = audit
        self.visibility = W08VisibilityFirewall(context, audit)
        self._consumed = False

    @classmethod
    def open(
        cls,
        repository_root: str | Path,
        context: W08FrozenContract,
        request: W08RunRequest,
        *,
        audit: W08PayloadAudit | None = None,
    ) -> "W08PayloadFirewall":
        """所有 owner/path/resource 检查必须先于首次 transport。"""
        if audit is not None and not isinstance(audit, W08PayloadAudit):
            raise W08ContractError("W-08 payload audit type is invalid")
        validate_w08_request(context, request)
        return cls(
            Path(repository_root).resolve(),
            context,
            request,
            audit if audit is not None else W08PayloadAudit(),
        )

    def _read(self, binding: W08FileBinding) -> tuple[object, ...]:
        self.visibility.authorize_candidate(binding.relative_path)
        target = _payload_file(self._root, binding.relative_path)
        local_parts = Path(binding.identity.relative_path).parts
        artifact_root = target.parents[len(local_parts) - 1]
        self.audit.transport_attempts += 1
        try:
            self.audit.transport_bytes += target.stat().st_size
            return read_record_artifact(artifact_root, binding.identity)
        except (DatasetArtifactIOError, OSError) as error:
            raise W08ContractError(
                f"W-08 payload transport/SHA failed: {binding.relative_path}"
            ) from error

    @staticmethod
    def _validate(
        source_refs: list[SourceRefRecord],
        observations: list[ObservationRecord],
        teachers: list[TeacherEvidenceRecord],
    ) -> None:
        source_keys = {item.stable_key for item in source_refs}
        observation_keys = {item.stable_key for item in observations}
        if len(source_keys) != len(source_refs) or len(observation_keys) != len(observations):
            raise W08ContractError("W-08 train payload contains duplicate stable key")
        if any(
            item.split != "train"
            or item.w_stage not in {"W-03", "W-08"}
            or item.source_ref_key not in source_keys
            for item in observations
        ):
            raise W08ContractError("W-08 payload contains non-train Observation")
        if any(
            item.visible_from_stage not in {"W-03", "W-08"}
            or item.observation_key not in observation_keys
            or item.source_ref_key not in source_keys
            for item in teachers
        ):
            raise W08ContractError("W-08 train Evidence is not closed over Observation")

    def read_training_payload(self) -> W08TrainingPayload:
        """校验双摘要后原子交付一次，禁止同实例 replay。"""
        if self._consumed:
            raise W08ContractError("W-08 payload firewall forbids replay")
        self._consumed = True
        source_refs: list[SourceRefRecord] = []
        observations: list[ObservationRecord] = []
        teachers: list[TeacherEvidenceRecord] = []
        for binding in self.context.candidate_bindings:
            records = self._read(binding)
            if binding.identity.owner_kind == "source":
                if any(not isinstance(item, SourceRefRecord) for item in records):
                    raise W08ContractError("W-08 source file contains another record kind")
                source_refs.extend(records)
            elif binding.identity.owner_kind == "observation":
                if any(not isinstance(item, ObservationRecord) for item in records):
                    raise W08ContractError("W-08 observation file contains another record kind")
                observations.extend(records)
            else:
                raise W08ContractError("W-08 candidate whitelist contains forbidden owner")
        for binding in self.context.teacher_bindings:
            records = self._read(binding)
            if any(not isinstance(item, TeacherEvidenceRecord) for item in records):
                raise W08ContractError("W-08 teacher file contains another record kind")
            teachers.extend(records)
        self._validate(source_refs, observations, teachers)
        bindings = (*self.context.candidate_bindings, *self.context.teacher_bindings)
        payload_bytes = sum(item.identity.transport_size_bytes for item in bindings)
        budget = dict(self.context.resource_budget)
        if (
            len(bindings) > budget["max_payload_gets"]
            or payload_bytes > budget["max_payload_bytes"]
            or len(observations) > budget["max_records"]
        ):
            raise W08ContractError("W-08 train payload exceeds resource budget")
        self.audit.payload_gets += len(bindings)
        self.audit.payload_bytes += payload_bytes
        self.audit.source_ref_reads += len(source_refs)
        self.audit.observation_reads += len(observations)
        self.audit.teacher_evidence_reads += len(teachers)
        return W08TrainingPayload(
            tuple(source_refs), tuple(observations), tuple(teachers)
        )


__all__ = [
    "W08PayloadFirewall",
    "W08TrainingPayload",
    "W08VisibilityFirewall",
]
