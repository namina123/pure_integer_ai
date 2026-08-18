"""独立 ``CONFLICT_SET`` family 的 prediction-first 语义协议。

本协议与现有单命题 ``CONFLICT`` evaluator 隔离；private label 读取前只提交
typed projection identity，不发布表面文本或逐条 verdict。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    exact_dict,
)
from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_contract import (
    ConflictSetSemanticProjection,
)


CONFLICT_SET_SEMANTIC_LABEL_ARTIFACT_KIND = (
    "PH2_GG03_CONFLICT_SET_SEMANTIC_LABEL_V1")
CONFLICT_SET_SEMANTIC_PROJECTION_KIND = (
    "PH2_GG03_CONFLICT_SET_SEMANTIC_PROJECTION_V1")
CONFLICT_SET_PREDICTION_SEAL_ARTIFACT_KIND = (
    "PH2_GG03_CONFLICT_SET_PREDICTION_SEAL_V1")
CONFLICT_SET_FORMAL_AGGREGATE_ARTIFACT_KIND = (
    "PH2_GG03_CONFLICT_SET_FORMAL_AGGREGATE_V1")
CONFLICT_SET_PREDICTION_SEAL_STATUS = (
    "CONFLICT_SET_PREDICTIONS_SEALED_LABELS_UNREAD")
CONFLICT_SET_FORMAL_LABEL_STATUSES = ("PASS", "FAIL", "NE")
CONFLICT_SET_FORMAL_REQUIREMENTS = (
    "CLAIM_ORDER",
    "CLAIM_SOURCE_CLOSURE",
    "CLAIM_STANCE_CLOSURE",
    "SCOPE_CLOSURE",
)


# object-model: exception
class ConflictSetFormalProtocolError(ValueError):
    """prediction、label 或 aggregate 违反 typed formal 协议。"""


def _sha(value: object, *, where: str) -> str:
    if not isinstance(value, str):
        raise ConflictSetFormalProtocolError(f"{where} must be a SHA-256 text")
    try:
        raw = bytes.fromhex(value)
    except ValueError as error:
        raise ConflictSetFormalProtocolError(
            f"{where} must be a SHA-256 text") from error
    if len(raw) != 32 or value != value.lower():
        raise ConflictSetFormalProtocolError(
            f"{where} must be a lowercase SHA-256")
    return value


def _digest(value: object) -> str:
    import hashlib
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _projection_dimension_values(
        projection: ConflictSetSemanticProjection,
        ) -> dict[str, object]:
    if not isinstance(projection, ConflictSetSemanticProjection):
        raise TypeError("conflict-set projection type is invalid")
    return {
        "CLAIM_ORDER": {
            "claim_ids": list(projection.claim_ids),
        },
        "CLAIM_SOURCE_CLOSURE": {
            "claim_source_ids": [
                {"claim_id": claim_id, "source_ids": list(source_ids)}
                for claim_id, source_ids in projection.claim_source_ids
            ],
            "cited_source_ids": list(projection.cited_source_ids),
        },
        "CLAIM_STANCE_CLOSURE": {
            "claim_states": [
                {"claim_id": claim_id, "support": support, "refute": refute}
                for claim_id, support, refute in projection.claim_states
            ],
        },
        "SCOPE_CLOSURE": {
            "carrier_kind": projection.carrier_kind,
            "response_act": projection.response_act,
            "scope_id": projection.scope_id,
        },
    }


def conflict_set_projection_dimension_sha256(
        projection: ConflictSetSemanticProjection,
        requirement: str,
        ) -> str:
    """返回一个可审计维度的确定性 commitment。"""
    if requirement not in CONFLICT_SET_FORMAL_REQUIREMENTS:
        raise ConflictSetFormalProtocolError("unknown conflict-set requirement")
    return _digest({
        "artifact_kind": CONFLICT_SET_SEMANTIC_PROJECTION_KIND,
        "requirement": requirement,
        "value": _projection_dimension_values(projection)[requirement],
        "version": 1,
    })


def conflict_set_projection_sha256(
        projection: ConflictSetSemanticProjection,
        ) -> str:
    """返回完整 typed projection commitment。"""
    return _digest({
        "artifact_kind": CONFLICT_SET_SEMANTIC_PROJECTION_KIND,
        "projection": projection.to_dict(),
        "version": 1,
    })


def _status_for_digest(
        expected: str,
        actual: str | None,
        ) -> str:
    _sha(expected, where="expected dimension SHA")
    if actual is None:
        return "NE"
    _sha(actual, where="predicted dimension SHA")
    return "PASS" if actual == expected else "FAIL"


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True, order=True)
class ConflictSetSemanticLabelRecord:
    """private expected commitment；不含 surface 或 Evidence 文本。"""

    observation_stable_key_sha256: str
    expected_projection_sha256: str
    expected_dimensions: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _sha(self.observation_stable_key_sha256,
             where="label observation stable key")
        _sha(self.expected_projection_sha256,
             where="label expected projection SHA")
        if tuple(item[0] for item in self.expected_dimensions) != (
                CONFLICT_SET_FORMAL_REQUIREMENTS):
            raise ConflictSetFormalProtocolError(
                "label requirement order is invalid")
        for requirement, value in self.expected_dimensions:
            if requirement not in CONFLICT_SET_FORMAL_REQUIREMENTS:
                raise ConflictSetFormalProtocolError(
                    "label requirement is not registered")
            _sha(value, where=f"label {requirement} SHA")

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_kind": CONFLICT_SET_SEMANTIC_LABEL_ARTIFACT_KIND,
            "expected_dimensions": [
                {"requirement": requirement, "sha256": value}
                for requirement, value in self.expected_dimensions
            ],
            "expected_projection_sha256": self.expected_projection_sha256,
            "format_version": 1,
            "observation_stable_key_sha256": (
                self.observation_stable_key_sha256),
            "split": "held_out",
        }

    def status_for(
            self, requirement: str, actual: str | None,
            ) -> str:
        expected = dict(self.expected_dimensions).get(requirement)
        if expected is None:
            raise ConflictSetFormalProtocolError(
                "label requirement is not available")
        return _status_for_digest(expected, actual)

    @classmethod
    def from_dict(cls, value: object) -> "ConflictSetSemanticLabelRecord":
        try:
            raw = exact_dict(value, {
                "artifact_kind", "expected_dimensions",
                "expected_projection_sha256", "format_version",
                "observation_stable_key_sha256", "split",
            }, where="conflict-set semantic label")
        except (TypeError, ValueError) as error:
            raise ConflictSetFormalProtocolError(
                "semantic label field set is invalid") from error
        if (raw["artifact_kind"] != CONFLICT_SET_SEMANTIC_LABEL_ARTIFACT_KIND
                or raw["format_version"] != 1
                or raw["split"] != "held_out"
                or not isinstance(raw["expected_dimensions"], list)):
            raise ConflictSetFormalProtocolError("semantic label kind/version drift")
        dimensions = tuple(
            (item["requirement"], item["sha256"])
            for item in raw["expected_dimensions"]
            if isinstance(item, dict)
            and set(item) == {"requirement", "sha256"}
        )
        if len(dimensions) != len(raw["expected_dimensions"]):
            raise ConflictSetFormalProtocolError("semantic label dimensions malformed")
        return cls(
            raw["observation_stable_key_sha256"],
            raw["expected_projection_sha256"],
            dimensions,
        )


def build_conflict_set_semantic_label_record(
        observation_stable_key_sha256: str,
        expected: ConflictSetSemanticProjection,
        ) -> ConflictSetSemanticLabelRecord:
    """只从 owner typed expected meaning 构造 commitment。"""
    _sha(observation_stable_key_sha256, where="label observation stable key")
    if not isinstance(expected, ConflictSetSemanticProjection):
        raise TypeError("expected conflict-set projection type is invalid")
    return ConflictSetSemanticLabelRecord(
        observation_stable_key_sha256,
        conflict_set_projection_sha256(expected),
        tuple((
            requirement,
            conflict_set_projection_dimension_sha256(expected, requirement),
        ) for requirement in CONFLICT_SET_FORMAL_REQUIREMENTS),
    )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True, order=True)
class ConflictSetSemanticPredictionRecord:
    """private label 读取前封存的一条 label-free prediction。"""

    observation_stable_key_sha256: str
    candidate_identity_sha256: str
    actual_status: str
    projection_sha256: str | None
    dimension_sha256: tuple[tuple[str, str | None], ...]
    run_sha256: str

    def __post_init__(self) -> None:
        _sha(self.observation_stable_key_sha256,
             where="prediction observation stable key")
        _sha(self.candidate_identity_sha256,
             where="prediction candidate identity")
        if self.actual_status not in CONFLICT_SET_FORMAL_LABEL_STATUSES:
            raise ConflictSetFormalProtocolError(
                "prediction actual status is invalid")
        if self.projection_sha256 is not None:
            _sha(self.projection_sha256, where="prediction projection SHA")
        if ((self.actual_status == "NE")
                != (self.projection_sha256 is None)):
            raise ConflictSetFormalProtocolError(
                "prediction actual status/projection availability drift")
        if tuple(item[0] for item in self.dimension_sha256) != (
                CONFLICT_SET_FORMAL_REQUIREMENTS):
            raise ConflictSetFormalProtocolError(
                "prediction requirement order is invalid")
        for requirement, value in self.dimension_sha256:
            if value is not None:
                _sha(value, where=f"prediction {requirement} SHA")
        _sha(self.run_sha256, where="prediction run SHA")

    def to_dict(self) -> dict[str, object]:
        return {
            "actual_status": self.actual_status,
            "candidate_identity_sha256": self.candidate_identity_sha256,
            "dimension_sha256": [
                {"requirement": requirement, "sha256": value}
                for requirement, value in self.dimension_sha256
            ],
            "observation_stable_key_sha256": (
                self.observation_stable_key_sha256),
            "projection_sha256": self.projection_sha256,
            "run_sha256": self.run_sha256,
        }


def build_conflict_set_semantic_prediction_record(
        observation_stable_key_sha256: str,
        candidate_identity_sha256: str,
        actual_status: str,
        projection: ConflictSetSemanticProjection | None,
        ) -> ConflictSetSemanticPredictionRecord:
    """不访问 label，把实际 runtime 输出投影为安全记录。"""
    _sha(observation_stable_key_sha256,
         where="prediction observation stable key")
    _sha(candidate_identity_sha256,
         where="prediction candidate identity")
    if actual_status not in CONFLICT_SET_FORMAL_LABEL_STATUSES:
        raise ConflictSetFormalProtocolError(
            "prediction actual status is invalid")
    projection_sha = None if projection is None else conflict_set_projection_sha256(
        projection)
    dimensions = tuple((
        requirement,
        None if projection is None else
        conflict_set_projection_dimension_sha256(projection, requirement),
    ) for requirement in CONFLICT_SET_FORMAL_REQUIREMENTS)
    return ConflictSetSemanticPredictionRecord(
        observation_stable_key_sha256,
        candidate_identity_sha256,
        actual_status,
        projection_sha,
        dimensions,
        _digest({
            "observation_stable_key_sha256": observation_stable_key_sha256,
            "candidate_identity_sha256": candidate_identity_sha256,
            "actual_status": actual_status,
            "dimension_sha256": [
                {"requirement": requirement, "sha256": value}
                for requirement, value in dimensions
            ],
            "projection_sha256": projection_sha,
            "version": 1,
        }),
    )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConflictSetSemanticPredictionSeal:
    """不可变的 pre-label prediction inventory。"""

    family_manifest_sha256: str
    family_commitment_sha256: str
    candidate_manifest_sha256: str
    records: tuple[ConflictSetSemanticPredictionRecord, ...]
    teacher_call_count: int = 0
    label_read_count: int = 0
    host_learning_write_count: int = 0
    status: str = CONFLICT_SET_PREDICTION_SEAL_STATUS

    def __post_init__(self) -> None:
        for name in (
                "family_manifest_sha256", "family_commitment_sha256",
                "candidate_manifest_sha256"):
            _sha(getattr(self, name), where=f"prediction seal {name}")
        if (not self.records
                or tuple(item.observation_stable_key_sha256
                         for item in self.records)
                != tuple(sorted({
                    item.observation_stable_key_sha256
                    for item in self.records
                }))):
            raise ConflictSetFormalProtocolError(
                "prediction seal records must be sorted and unique")
        if any(type(getattr(self, name)) is not int
               or getattr(self, name) != 0 for name in (
                   "teacher_call_count", "label_read_count",
                   "host_learning_write_count")):
            raise ConflictSetFormalProtocolError(
                "prediction seal pre-label counters must be zero")
        if self.status != CONFLICT_SET_PREDICTION_SEAL_STATUS:
            raise ConflictSetFormalProtocolError("prediction seal status drift")

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_kind": CONFLICT_SET_PREDICTION_SEAL_ARTIFACT_KIND,
            "candidate_manifest_sha256": self.candidate_manifest_sha256,
            "family_commitment_sha256": self.family_commitment_sha256,
            "family_manifest_sha256": self.family_manifest_sha256,
            "format_version": 1,
            "host_learning_write_count": self.host_learning_write_count,
            "label_read_count": self.label_read_count,
            "records": [item.to_dict() for item in self.records],
            "status": self.status,
            "teacher_call_count": self.teacher_call_count,
        }

    def sha256(self) -> str:
        return _digest(self.to_dict())


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConflictSetDimensionResult:
    """一个 conflict-set requirement 的安全计数结果。"""

    requirement: str
    status: str
    denominator: int
    pass_count: int
    fail_count: int
    ne_count: int
    detail_sha256: str

    def __post_init__(self) -> None:
        if self.requirement not in CONFLICT_SET_FORMAL_REQUIREMENTS:
            raise ConflictSetFormalProtocolError("dimension is not registered")
        if self.status not in CONFLICT_SET_FORMAL_LABEL_STATUSES:
            raise ConflictSetFormalProtocolError("dimension status is invalid")
        counts = (self.pass_count, self.fail_count, self.ne_count)
        if (type(self.denominator) is not int or self.denominator <= 0
                or any(type(item) is not int or item < 0 for item in counts)
                or sum(counts) != self.denominator):
            raise ConflictSetFormalProtocolError("dimension counts are invalid")
        expected = (
            "FAIL" if self.fail_count else
            "NE" if self.ne_count else "PASS")
        if self.status != expected:
            raise ConflictSetFormalProtocolError("dimension status/count drift")
        _sha(self.detail_sha256, where="dimension detail SHA")

    def to_dict(self) -> dict[str, object]:
        return {
            "detail_sha256": self.detail_sha256,
            "denominator": self.denominator,
            "fail_count": self.fail_count,
            "ne_count": self.ne_count,
            "pass_count": self.pass_count,
            "requirement": self.requirement,
            "status": self.status,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConflictSetSemanticFormalAggregate:
    """四个独立语义维度的只含计数 aggregate。"""

    family_manifest_sha256: str
    family_commitment_sha256: str
    prediction_seal_sha256: str
    label_commitment_sha256: str
    dimensions: tuple[ConflictSetDimensionResult, ...]
    label_read_count: int
    label_record_count: int
    label_transport_bytes: int
    status: str
    failure_phase: str = "NONE"

    def __post_init__(self) -> None:
        for name in (
                "family_manifest_sha256", "family_commitment_sha256",
                "prediction_seal_sha256", "label_commitment_sha256"):
            _sha(getattr(self, name), where=f"aggregate {name}")
        if tuple(item.requirement for item in self.dimensions) != (
                CONFLICT_SET_FORMAL_REQUIREMENTS):
            raise ConflictSetFormalProtocolError("aggregate dimension order drift")
        expected = (
            "FAIL" if any(item.status == "FAIL" for item in self.dimensions)
            else "NE" if any(item.status == "NE" for item in self.dimensions)
            else "PASS")
        if self.status != expected:
            raise ConflictSetFormalProtocolError("aggregate status drift")
        if any(type(item) is not int or item < 0 for item in (
                self.label_read_count, self.label_record_count,
                self.label_transport_bytes)):
            raise ConflictSetFormalProtocolError("aggregate label counters invalid")
        if self.failure_phase == "NONE" and (
                self.label_read_count != 1 or self.label_record_count <= 0):
            raise ConflictSetFormalProtocolError("aggregate label read audit incomplete")
        if not isinstance(self.failure_phase, str) or not self.failure_phase:
            raise ConflictSetFormalProtocolError("aggregate failure phase invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_kind": CONFLICT_SET_FORMAL_AGGREGATE_ARTIFACT_KIND,
            "dimensions": [item.to_dict() for item in self.dimensions],
            "failure_phase": self.failure_phase,
            "family_commitment_sha256": self.family_commitment_sha256,
            "family_manifest_sha256": self.family_manifest_sha256,
            "format_version": 1,
            "label_commitment_sha256": self.label_commitment_sha256,
            "label_read_count": self.label_read_count,
            "label_record_count": self.label_record_count,
            "label_transport_bytes": self.label_transport_bytes,
            "prediction_seal_sha256": self.prediction_seal_sha256,
            "status": self.status,
            "write_counts": {
                "candidate_writes": 0,
                "evaluator_label_writes": 0,
                "host_learning_writes": 0,
                "memory_learning_writes": 0,
                "teacher_calls": 0,
            },
        }

    def sha256(self) -> str:
        return _digest(self.to_dict())


def build_conflict_set_semantic_prediction_seal(
        records: tuple[ConflictSetSemanticPredictionRecord, ...],
        *,
        family_manifest_sha256: str,
        family_commitment_sha256: str,
        candidate_manifest_sha256: str,
        ) -> ConflictSetSemanticPredictionSeal:
    if not isinstance(records, tuple) or not records:
        raise TypeError("prediction records must be a non-empty tuple")
    if any(not isinstance(item, ConflictSetSemanticPredictionRecord)
           for item in records):
        raise TypeError("prediction records contain an invalid item")
    return ConflictSetSemanticPredictionSeal(
        family_manifest_sha256, family_commitment_sha256,
        candidate_manifest_sha256, records,
    )


def build_conflict_set_semantic_formal_aggregate(
        prediction: ConflictSetSemanticPredictionSeal,
        labels: tuple[ConflictSetSemanticLabelRecord, ...],
        *,
        label_commitment_sha256: str,
        label_transport_bytes: int,
        ) -> ConflictSetSemanticFormalAggregate:
    """只在 prediction seal 存在且 label 已物化后评分。"""
    if (not isinstance(prediction, ConflictSetSemanticPredictionSeal)
            or not isinstance(labels, tuple)
            or not labels
            or len(labels) != len(prediction.records)
            or any(not isinstance(item, ConflictSetSemanticLabelRecord)
                   for item in labels)):
        raise TypeError("prediction/label aggregate inputs are invalid")
    _sha(label_commitment_sha256, where="aggregate label commitment SHA")
    prediction_keys = tuple(
        item.observation_stable_key_sha256 for item in prediction.records)
    label_keys = tuple(
        item.observation_stable_key_sha256 for item in labels)
    if prediction_keys != label_keys:
        raise ConflictSetFormalProtocolError(
            "prediction and label observation identity drifted")
    statuses: dict[str, list[str]] = {
        requirement: [] for requirement in CONFLICT_SET_FORMAL_REQUIREMENTS}
    for predicted, label in zip(prediction.records, labels, strict=True):
        identity_status = (
            "PASS"
            if predicted.projection_sha256 == label.expected_projection_sha256
            else "NE" if predicted.projection_sha256 is None else "FAIL"
        )
        for requirement, actual in predicted.dimension_sha256:
            dimension_status = label.status_for(requirement, actual)
            if "FAIL" in {predicted.actual_status, identity_status}:
                dimension_status = "FAIL"
            elif "NE" in {predicted.actual_status, identity_status}:
                dimension_status = "NE"
            statuses[requirement].append(dimension_status)
    dimensions = tuple(
        ConflictSetDimensionResult(
            requirement,
            (
                "FAIL" if "FAIL" in statuses[requirement]
                else "NE" if "NE" in statuses[requirement]
                else "PASS"
            ),
            len(statuses[requirement]),
            statuses[requirement].count("PASS"),
            statuses[requirement].count("FAIL"),
            statuses[requirement].count("NE"),
            _digest({
                "requirement": requirement,
                "statuses": statuses[requirement],
                "version": 1,
            }),
        )
        for requirement in CONFLICT_SET_FORMAL_REQUIREMENTS
    )
    return ConflictSetSemanticFormalAggregate(
        prediction.family_manifest_sha256,
        prediction.family_commitment_sha256,
        prediction.sha256(),
        label_commitment_sha256,
        dimensions,
        1,
        len(labels),
        label_transport_bytes,
        (
            "FAIL" if any(item.status == "FAIL" for item in dimensions)
            else "NE" if any(item.status == "NE" for item in dimensions)
            else "PASS"
        ),
    )


def build_conflict_set_semantic_unavailable_aggregate(
        prediction: ConflictSetSemanticPredictionSeal,
        *,
        label_commitment_sha256: str,
        prediction_seal_sha256: str | None = None,
        failure_phase: str,
        label_read_count: int,
        label_record_count: int,
        label_transport_bytes: int,
        ) -> ConflictSetSemanticFormalAggregate:
    """在 guard 后运行不可判定时只发布四维 NE 计数。"""
    if not isinstance(prediction, ConflictSetSemanticPredictionSeal):
        raise TypeError("prediction seal 类型错误")
    if not isinstance(failure_phase, str) or not failure_phase:
        raise ConflictSetFormalProtocolError("failure phase 非法")
    _sha(label_commitment_sha256, where="unavailable label commitment SHA")
    if prediction_seal_sha256 is None:
        prediction_seal_sha256 = prediction.sha256()
    _sha(prediction_seal_sha256, where="unavailable prediction seal SHA")
    dimensions = tuple(
        ConflictSetDimensionResult(
            requirement,
            "NE",
            len(prediction.records),
            0,
            0,
            len(prediction.records),
            _digest({
                "failure_phase": failure_phase,
                "requirement": requirement,
                "version": 1,
            }),
        )
        for requirement in CONFLICT_SET_FORMAL_REQUIREMENTS
    )
    return ConflictSetSemanticFormalAggregate(
        prediction.family_manifest_sha256,
        prediction.family_commitment_sha256,
        prediction_seal_sha256,
        label_commitment_sha256,
        dimensions,
        label_read_count,
        label_record_count,
        label_transport_bytes,
        "NE",
        failure_phase,
    )


def conflict_set_semantic_verdict_contract_sha256() -> str:
    """冻结独立四维 PASS/FAIL/NE 合同。"""
    return _digest({
        "artifact_kind": CONFLICT_SET_SEMANTIC_LABEL_ARTIFACT_KIND,
        "dimensions": list(CONFLICT_SET_FORMAL_REQUIREMENTS),
        "fail_condition": "PREDICTED_DIMENSION_COMMITMENT_DIFFERS",
        "ne_condition": "PREDICTED_PROJECTION_UNAVAILABLE",
        "pass_condition": "PREDICTED_DIMENSION_COMMITMENT_EQUALS_EXPECTED",
        "status_precedence": ["FAIL", "NE", "PASS"],
        "version": 1,
    })


__all__ = [
    "CONFLICT_SET_FORMAL_AGGREGATE_ARTIFACT_KIND",
    "CONFLICT_SET_FORMAL_LABEL_STATUSES",
    "CONFLICT_SET_FORMAL_REQUIREMENTS",
    "CONFLICT_SET_PREDICTION_SEAL_ARTIFACT_KIND",
    "CONFLICT_SET_PREDICTION_SEAL_STATUS",
    "CONFLICT_SET_SEMANTIC_LABEL_ARTIFACT_KIND",
    "CONFLICT_SET_SEMANTIC_PROJECTION_KIND",
    "ConflictSetDimensionResult",
    "ConflictSetFormalProtocolError",
    "ConflictSetSemanticFormalAggregate",
    "ConflictSetSemanticLabelRecord",
    "ConflictSetSemanticPredictionRecord",
    "ConflictSetSemanticPredictionSeal",
    "build_conflict_set_semantic_formal_aggregate",
    "build_conflict_set_semantic_label_record",
    "build_conflict_set_semantic_prediction_record",
    "build_conflict_set_semantic_prediction_seal",
    "build_conflict_set_semantic_unavailable_aggregate",
    "conflict_set_projection_dimension_sha256",
    "conflict_set_projection_sha256",
    "conflict_set_semantic_verdict_contract_sha256",
]
