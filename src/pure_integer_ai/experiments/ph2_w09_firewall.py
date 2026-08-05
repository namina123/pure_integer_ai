"""W-09 分相可见性、safe failure 与 train-only payload 防火墙。"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import hashlib
from pathlib import Path, PurePosixPath

from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    ObservationRecord,
    SourceRefRecord,
    TeacherEvidenceRecord,
    W_STAGES,
    canonical_json_line,
)
from pure_integer_ai.experiments.ph2_dataset_io import (
    DatasetArtifactIOError,
    read_record_artifact,
)
from pure_integer_ai.experiments.ph2_w09_contract import (
    W09_ACCESS_PHASES,
    W09_CANDIDATE_OWNER,
    W09_DEV_OWNER,
    W09_EVALUATOR_OWNER,
    W09ContractError,
    W09FileBinding,
    W09FrozenContract,
    W09HostWriteSnapshot,
    W09PayloadAudit,
    W09RunRequest,
    validate_w09_request,
)


W09_FAILURE_OPERATIONS = (
    "PREFLIGHT",
    "AUTHORIZE",
    "TRANSPORT",
    "SCHEMA",
    "DELIVER",
)
W09_LEGACY_CANDIDATE_REDACTION_FIELDS = (
    "expected_authoritative",
    "expected_eliminated_branches",
)


class W09FailureKind(str, Enum):
    INVALID_REQUEST = "INVALID_REQUEST"
    PATH_NOT_REGISTERED = "PATH_NOT_REGISTERED"
    PATH_TRAVERSAL = "PATH_TRAVERSAL"
    LINK_COMPONENT = "LINK_COMPONENT"
    OWNER_SPOOF = "OWNER_SPOOF"
    CANDIDATE_NOT_SEALED = "CANDIDATE_NOT_SEALED"
    CODE_NOT_FROZEN = "CODE_NOT_FROZEN"
    HOST_WRITE_INTENT = "HOST_WRITE_INTENT"
    TRANSPORT_IDENTITY = "TRANSPORT_IDENTITY"
    CONTENT_SCHEMA = "CONTENT_SCHEMA"
    PAYLOAD_REPLAY = "PAYLOAD_REPLAY"
    LABEL_LEAK = "LABEL_LEAK"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"


@dataclass(frozen=True)
class W09SafeFailure:
    """只包含枚举位置、失败种类和计数的异常报告。"""

    phase: str
    operation: str
    dimension: str
    failure_kind: W09FailureKind
    transport_attempts: int
    payload_gets: int
    delivery_count: int

    def __post_init__(self) -> None:
        if self.phase not in {"preflight", *W09_ACCESS_PHASES}:
            raise W09ContractError("W-09 safe failure phase is invalid")
        if self.operation not in W09_FAILURE_OPERATIONS:
            raise W09ContractError("W-09 safe failure operation is invalid")
        if self.dimension != "BOUNDARY":
            raise W09ContractError("W-09 safe failure dimension is invalid")
        if not isinstance(self.failure_kind, W09FailureKind):
            raise W09ContractError("W-09 safe failure kind is invalid")
        if any(
            type(value) is not int or value < 0
            for value in (
                self.transport_attempts,
                self.payload_gets,
                self.delivery_count,
            )
        ):
            raise W09ContractError("W-09 safe failure count is invalid")

    def safe_code(self) -> str:
        return ":".join((
            "W09_BOUNDARY",
            self.phase.upper(),
            self.operation,
            self.failure_kind.value,
            str(self.transport_attempts),
            str(self.payload_gets),
            str(self.delivery_count),
        ))


class W09FirewallError(W09ContractError):
    """携带固定枚举报告且不回显 payload/path/message 的边界失败。"""

    def __init__(self, report: W09SafeFailure) -> None:
        self.report = report
        super().__init__(report.safe_code())


def _failure(
    audit: W09PayloadAudit,
    *,
    phase: str,
    operation: str,
    kind: W09FailureKind,
    delivery_count: int = 0,
) -> W09FirewallError:
    return W09FirewallError(W09SafeFailure(
        phase,
        operation,
        "BOUNDARY",
        kind,
        audit.transport_attempts,
        audit.payload_gets,
        delivery_count,
    ))


def _path_parts(relative: object) -> tuple[str, ...]:
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise ValueError
    path = PurePosixPath(relative)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != relative
        or "//" in relative
        or "!" in relative
    ):
        raise ValueError
    return path.parts


def _has_link_component(root: Path, parts: tuple[str, ...]) -> bool:
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            return True
        is_junction = getattr(current, "is_junction", None)
        if is_junction is not None and is_junction():
            return True
    return False


def _payload_file(
    root: Path,
    relative: object,
    audit: W09PayloadAudit,
    *,
    phase: str,
) -> Path:
    try:
        parts = _path_parts(relative)
    except ValueError:
        raise _failure(
            audit,
            phase=phase,
            operation="AUTHORIZE",
            kind=W09FailureKind.PATH_TRAVERSAL,
        ) from None
    if _has_link_component(root, parts):
        raise _failure(
            audit,
            phase=phase,
            operation="AUTHORIZE",
            kind=W09FailureKind.LINK_COMPONENT,
        ) from None
    target = (root / Path(*parts)).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise _failure(
            audit,
            phase=phase,
            operation="AUTHORIZE",
            kind=W09FailureKind.PATH_TRAVERSAL,
        ) from None
    if not target.is_file():
        raise _failure(
            audit,
            phase=phase,
            operation="TRANSPORT",
            kind=W09FailureKind.TRANSPORT_IDENTITY,
        ) from None
    return target


def _assert_exact_record_schema(
    records: tuple[object, ...],
    binding: W09FileBinding,
    audit: W09PayloadAudit,
) -> None:
    """重建规范 records；raw 额外字段会使 identity 无法相等。"""
    digest = hashlib.sha256()
    size_bytes = 0
    try:
        for item in records:
            payload = canonical_json_line(item.to_dict())
            digest.update(payload)
            size_bytes += len(payload)
    except (AttributeError, TypeError, ValueError):
        raise _failure(
            audit,
            phase=binding.access_phase,
            operation="SCHEMA",
            kind=W09FailureKind.CONTENT_SCHEMA,
        ) from None
    if (
        digest.hexdigest() != binding.identity.content_sha256
        or size_bytes != binding.identity.content_size_bytes
    ):
        raise _failure(
            audit,
            phase=binding.access_phase,
            operation="SCHEMA",
            kind=W09FailureKind.CONTENT_SCHEMA,
        ) from None


def _contains_label_key(value: object) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in ("expected", "label", "evaluator")):
                return True
            if _contains_label_key(item):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_label_key(item) for item in value)
    return False


def _sanitize_candidate_observation(
    observation: ObservationRecord,
    audit: W09PayloadAudit,
) -> ObservationRecord:
    """只删除两个冻结 legacy 非答案字段；其他可疑键按 poison 拒绝。"""
    redacted = 0

    def sanitize(value: object) -> object:
        nonlocal redacted
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                lowered = str(key).lower()
                if str(key) in W09_LEGACY_CANDIDATE_REDACTION_FIELDS:
                    redacted += 1
                    continue
                if any(token in lowered for token in ("expected", "label", "evaluator")):
                    raise _failure(
                        audit,
                        phase="candidate",
                        operation="SCHEMA",
                        kind=W09FailureKind.LABEL_LEAK,
                    ) from None
                result[key] = sanitize(item)
            return result
        if isinstance(value, list):
            return [sanitize(item) for item in value]
        return value

    payload = sanitize(observation.typed_payload.to_value())
    assert isinstance(payload, dict)
    audit.redacted_candidate_fields += redacted
    if not redacted:
        return observation
    return replace(
        observation,
        typed_payload=CanonicalJsonObject.from_value(payload),
    )


@dataclass(frozen=True)
class W09TrainingPayload:
    """经双摘要、schema、owner 和引用闭合后交付的 34-pack train payload。"""

    source_refs: tuple[SourceRefRecord, ...]
    observations: tuple[ObservationRecord, ...]
    training_evidence: tuple[TeacherEvidenceRecord, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source_refs, tuple)
            or not isinstance(self.observations, tuple)
            or not isinstance(self.training_evidence, tuple)
            or any(not isinstance(item, SourceRefRecord) for item in self.source_refs)
            or any(not isinstance(item, ObservationRecord) for item in self.observations)
            or any(
                not isinstance(item, TeacherEvidenceRecord)
                for item in self.training_evidence
            )
        ):
            raise W09ContractError("W-09 training payload typed members are invalid")


class W09VisibilityFirewall:
    """冻结 candidate/dev/evaluator 的路径、owner、freeze 与 host-write 边界。"""

    def __init__(self, context: W09FrozenContract, audit: W09PayloadAudit) -> None:
        self.context = context
        self.audit = audit

    def authorize_candidate(self, relative_path: object) -> W09FileBinding:
        for item in (
            *self.context.candidate_bindings,
            *self.context.training_material_bindings,
        ):
            if item.relative_path == relative_path:
                return item
        kind = W09FailureKind.PATH_NOT_REGISTERED
        try:
            _path_parts(relative_path)
        except ValueError:
            kind = W09FailureKind.PATH_TRAVERSAL
        raise _failure(
            self.audit,
            phase="candidate",
            operation="AUTHORIZE",
            kind=kind,
        ) from None

    def authorize_dev(
        self,
        relative_path: object,
        *,
        owner_key: str,
        host_writes: W09HostWriteSnapshot,
    ) -> W09FileBinding:
        if owner_key != W09_DEV_OWNER:
            raise _failure(
                self.audit,
                phase="dev",
                operation="AUTHORIZE",
                kind=W09FailureKind.OWNER_SPOOF,
            ) from None
        if not isinstance(host_writes, W09HostWriteSnapshot) or not host_writes.is_zero:
            raise _failure(
                self.audit,
                phase="dev",
                operation="AUTHORIZE",
                kind=W09FailureKind.HOST_WRITE_INTENT,
            ) from None
        for item in self.context.dev_bindings:
            if item.relative_path == relative_path:
                return item
        raise _failure(
            self.audit,
            phase="dev",
            operation="AUTHORIZE",
            kind=W09FailureKind.PATH_NOT_REGISTERED,
        ) from None

    def authorize_evaluator(
        self,
        relative_path: object,
        *,
        owner_key: str,
        candidate_sealed: int,
        code_frozen: int,
        host_writes: W09HostWriteSnapshot,
    ) -> W09FileBinding:
        if owner_key != W09_EVALUATOR_OWNER:
            raise _failure(
                self.audit,
                phase="evaluator",
                operation="AUTHORIZE",
                kind=W09FailureKind.OWNER_SPOOF,
            ) from None
        if candidate_sealed != 1:
            raise _failure(
                self.audit,
                phase="evaluator",
                operation="AUTHORIZE",
                kind=W09FailureKind.CANDIDATE_NOT_SEALED,
            ) from None
        if code_frozen != 1:
            raise _failure(
                self.audit,
                phase="evaluator",
                operation="AUTHORIZE",
                kind=W09FailureKind.CODE_NOT_FROZEN,
            ) from None
        if not isinstance(host_writes, W09HostWriteSnapshot) or not host_writes.is_zero:
            raise _failure(
                self.audit,
                phase="evaluator",
                operation="AUTHORIZE",
                kind=W09FailureKind.HOST_WRITE_INTENT,
            ) from None
        for item in self.context.evaluator_bindings:
            if item.relative_path == relative_path:
                return item
        raise _failure(
            self.audit,
            phase="evaluator",
            operation="AUTHORIZE",
            kind=W09FailureKind.PATH_NOT_REGISTERED,
        ) from None


class W09PayloadFirewall:
    """完整 preflight 后一次性交付 34 train pack，不触碰 dev/held-out/label。"""

    def __init__(
        self,
        root: Path,
        context: W09FrozenContract,
        request: W09RunRequest,
        audit: W09PayloadAudit,
    ) -> None:
        self._root = root
        self.context = context
        self.request = request
        self.audit = audit
        self.visibility = W09VisibilityFirewall(context, audit)
        self._consumed = False

    @classmethod
    def open(
        cls,
        repository_root: str | Path,
        context: W09FrozenContract,
        request: W09RunRequest,
        *,
        audit: W09PayloadAudit | None = None,
    ) -> "W09PayloadFirewall":
        actual_audit = audit if isinstance(audit, W09PayloadAudit) else W09PayloadAudit()
        if audit is not None and not isinstance(audit, W09PayloadAudit):
            raise _failure(
                actual_audit,
                phase="preflight",
                operation="PREFLIGHT",
                kind=W09FailureKind.INVALID_REQUEST,
            ) from None
        try:
            validate_w09_request(context, request)
        except W09ContractError:
            raise _failure(
                actual_audit,
                phase="preflight",
                operation="PREFLIGHT",
                kind=W09FailureKind.INVALID_REQUEST,
            ) from None
        return cls(Path(repository_root).resolve(), context, request, actual_audit)

    def _read(self, binding: W09FileBinding) -> tuple[object, ...]:
        authorized = self.visibility.authorize_candidate(binding.relative_path)
        if authorized != binding:
            raise _failure(
                self.audit,
                phase=binding.access_phase,
                operation="AUTHORIZE",
                kind=W09FailureKind.OWNER_SPOOF,
            ) from None
        target = _payload_file(
            self._root,
            binding.relative_path,
            self.audit,
            phase=binding.access_phase,
        )
        local_parts = PurePosixPath(binding.identity.relative_path).parts
        artifact_root = target.parents[len(local_parts) - 1]
        self.audit.transport_attempts += 1
        try:
            size_bytes = target.stat().st_size
            records = read_record_artifact(artifact_root, binding.identity)
        except (DatasetArtifactIOError, OSError):
            raise _failure(
                self.audit,
                phase=binding.access_phase,
                operation="TRANSPORT",
                kind=W09FailureKind.TRANSPORT_IDENTITY,
            ) from None
        self.audit.transport_bytes += size_bytes
        _assert_exact_record_schema(records, binding, self.audit)
        return records

    @staticmethod
    def _validate(
        source_refs: list[SourceRefRecord],
        observations: list[ObservationRecord],
        evidence: list[TeacherEvidenceRecord],
        audit: W09PayloadAudit,
    ) -> None:
        source_keys = {item.stable_key for item in source_refs}
        observation_keys = {item.stable_key for item in observations}
        if (
            len(source_keys) != len(source_refs)
            or len(observation_keys) != len(observations)
            or len({item.stable_key for item in evidence}) != len(evidence)
        ):
            raise _failure(
                audit,
                phase="candidate",
                operation="SCHEMA",
                kind=W09FailureKind.CONTENT_SCHEMA,
            ) from None
        if any(
            item.split != "train"
            or item.source_ref_key not in source_keys
            or item.w_stage not in W_STAGES
            or W_STAGES.index(item.w_stage) > W_STAGES.index("W-09")
            for item in observations
        ):
            raise _failure(
                audit,
                phase="candidate",
                operation="SCHEMA",
                kind=W09FailureKind.CONTENT_SCHEMA,
            ) from None
        if any(
            item.observation_key not in observation_keys
            or item.source_ref_key not in source_keys
            or item.visible_from_stage not in W_STAGES
            or W_STAGES.index(item.visible_from_stage) > W_STAGES.index("W-09")
            for item in evidence
        ):
            raise _failure(
                audit,
                phase="training_material",
                operation="SCHEMA",
                kind=W09FailureKind.CONTENT_SCHEMA,
            ) from None
        if len(evidence) != len(observations):
            raise _failure(
                audit,
                phase="training_material",
                operation="SCHEMA",
                kind=W09FailureKind.CONTENT_SCHEMA,
            ) from None
        if any(_contains_label_key(item.typed_payload.to_value()) for item in observations):
            raise _failure(
                audit,
                phase="candidate",
                operation="SCHEMA",
                kind=W09FailureKind.LABEL_LEAK,
            ) from None

    def read_training_payload(self) -> W09TrainingPayload:
        if self._consumed:
            raise _failure(
                self.audit,
                phase="candidate",
                operation="DELIVER",
                kind=W09FailureKind.PAYLOAD_REPLAY,
            ) from None
        self._consumed = True
        source_refs: list[SourceRefRecord] = []
        observations: list[ObservationRecord] = []
        evidence: list[TeacherEvidenceRecord] = []
        for binding in self.context.candidate_bindings:
            records = self._read(binding)
            if binding.identity.owner_kind == "source":
                if any(not isinstance(item, SourceRefRecord) for item in records):
                    raise _failure(
                        self.audit,
                        phase="candidate",
                        operation="SCHEMA",
                        kind=W09FailureKind.OWNER_SPOOF,
                    ) from None
                source_refs.extend(records)
            elif binding.identity.owner_kind == "observation":
                if any(not isinstance(item, ObservationRecord) for item in records):
                    raise _failure(
                        self.audit,
                        phase="candidate",
                        operation="SCHEMA",
                        kind=W09FailureKind.OWNER_SPOOF,
                    ) from None
                observations.extend(
                    _sanitize_candidate_observation(item, self.audit)
                    for item in records
                )
            else:
                raise _failure(
                    self.audit,
                    phase="candidate",
                    operation="SCHEMA",
                    kind=W09FailureKind.OWNER_SPOOF,
                ) from None
        for binding in self.context.training_material_bindings:
            records = self._read(binding)
            if any(not isinstance(item, TeacherEvidenceRecord) for item in records):
                raise _failure(
                    self.audit,
                    phase="training_material",
                    operation="SCHEMA",
                    kind=W09FailureKind.OWNER_SPOOF,
                ) from None
            evidence.extend(records)
        self._validate(source_refs, observations, evidence, self.audit)
        bindings = (
            *self.context.candidate_bindings,
            *self.context.training_material_bindings,
        )
        payload_bytes = sum(item.identity.transport_size_bytes for item in bindings)
        budget = dict(self.context.resource_budget)
        if (
            len(bindings) > budget["max_payload_gets"]
            or payload_bytes > budget["max_payload_bytes"]
            or len(observations) > budget["max_records"]
        ):
            raise _failure(
                self.audit,
                phase="candidate",
                operation="DELIVER",
                kind=W09FailureKind.RESOURCE_LIMIT,
            ) from None
        self.audit.payload_gets += len(bindings)
        self.audit.payload_bytes += payload_bytes
        self.audit.source_ref_reads += len(source_refs)
        self.audit.observation_reads += len(observations)
        self.audit.training_evidence_reads += len(evidence)
        return W09TrainingPayload(
            tuple(source_refs),
            tuple(observations),
            tuple(evidence),
        )


__all__ = [
    "W09FailureKind",
    "W09FirewallError",
    "W09PayloadFirewall",
    "W09SafeFailure",
    "W09TrainingPayload",
    "W09VisibilityFirewall",
    "W09_LEGACY_CANDIDATE_REDACTION_FIELDS",
]
