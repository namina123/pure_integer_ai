"""LC-01 文本观察保真与 LC-15 初版目标的原创 CC0 课程编译器。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pure_integer_ai.experiments.ph2_authored_course_common import (
    AuthoredCompiledSeed,
    AuthoredCourseBuild,
    AuthoredCourseSpec,
    publish_authored_course,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    EXPECTED_STATES,
    SAMPLE_ROLES,
    CanonicalJsonObject,
    canonical_json_line,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_language_course_contract import (
    COURSE_CAPABILITY_KEYS,
    COURSE_INVARIANTS,
    COURSE_RETENTION_PROTOCOLS,
    COURSE_SPLIT_AXES,
    COURSE_TASK_KEYS,
    COURSE_VERIFIER_NE_CONDITIONS,
    LANGUAGE_OBJECTIVE_KEYS,
    TEXT_FIDELITY_EVALUATOR_DIMENSIONS,
    TEXT_FIDELITY_PAYLOAD_KEYS,
    LanguageCourseContractError,
    LanguageCourseManifest,
    LearningObjectiveSpec,
)
from pure_integer_ai.experiments.ph2_language_coverage_contract import (
    SAMPLE_FAMILIES,
)


SOURCE_KEY = "AUTHORED_CC0_V1"
LICENSE_ID = "CC0-1.0"
COURSE_VERSION = 1
ARTIFACT_VERSION = 1
ADAPTER_VERSION = 1
GENERATOR_VERSION = 1
PARSER_VERSION = 1
PACK_NAME = "AUTHORED_CC0_V1--CC0-1.0--lc01-text-fidelity-v1"
STAGE = "W-02"
SUBSTAGE = "LC01_TEXT_FIDELITY"
PAYLOAD_KIND = "TextFidelityCandidateV1"
OBJECTIVE_TAXONOMY_VERSION = "LC-15-initial-objective-taxonomy-v1"
COURSE_MANIFEST_ARTIFACT_VERSION = "LC-01-LC-15-initial-course-v1"
COURSE_MANIFEST_PATH = Path(
    "data/ph2/manifests/lc01_lc15_initial_course_v1.json")
FORMAL_ARTIFACT_RELATIVE_ROOT = (
    "ph2_dataset_artifacts/d02_language_courses_v1")

CANDIDATE_KINDS = (
    "GENERATION",
    "IDENTITY",
    "NORMALIZATION",
    "REPAIR",
    "RETENTION",
    "SEGMENTATION",
    "TOKENIZATION",
)
LOSS_KINDS = (
    "BOUNDARY_UNCERTAINTY",
    "LEXICAL_UNCERTAINTY",
    "NONE",
    "PUNCTUATION_DISTINCTION",
    "SCRIPT_VARIANT_IDENTITY",
    "WHITESPACE_MULTIPLICITY",
    "WIDTH_DISTINCTION",
)
OPERATION_KINDS = (
    "FULLWIDTH_TO_HALFWIDTH",
    "GENERATION_CANDIDATE",
    "IDENTITY",
    "PUNCTUATION_DROP",
    "RETAINED_REVERIFY",
    "SEGMENTATION_CANDIDATE",
    "TOKENIZATION_CANDIDATE",
    "TRADITIONAL_TO_SIMPLIFIED",
    "TYPO_REPAIR_CANDIDATE",
    "WHITESPACE_COLLAPSE",
)
_SEED_FIELDS = {
    "candidate_group", "candidate_id", "candidate_kind", "derived_text",
    "evaluation_dimension", "expected_payload", "expected_state", "family",
    "information_loss", "label_owner", "license_id", "logical_order",
    "loss_kind", "objective_keys", "operations", "perturbation_kind",
    "raw_text", "retention_anchor_id", "sample_family", "sample_role",
    "seed_id", "segments", "split", "supersedes_seed_id",
    "template_family", "tokens",
}
_OPERATION_FIELDS = {
    "input_end", "input_start", "operation_kind", "output_end",
    "output_start", "reversible",
}
_ROLE_BY_FAMILY = {
    "AMBIGUOUS": "conflict",
    "NEGATIVE": "refute",
    "POSITIVE": "support",
    "RETENTION": "read_only_probe",
    "REVISION": "supersede",
    "UNKNOWN": "anomaly",
}
_STATE_BY_FAMILY = {
    "AMBIGUOUS": "CONFLICT",
    "NEGATIVE": "FALSE",
    "POSITIVE": "TRUE",
    "RETENTION": "TRUE",
    "REVISION": "TRUE",
    "UNKNOWN": "UNKNOWN",
}


class AuthoredTextFidelityCourseError(RuntimeError):
    """文本保真 seed、receipt、目标或课程边界不完整。"""


def _exact_keys(value: Any, keys: set[str], *, where: str) -> dict[str, Any]:
    """要求 JSON object 字段集合精确相等。"""
    if not isinstance(value, dict) or set(value) != keys:
        raise AuthoredTextFidelityCourseError(f"{where} 字段不精确")
    return value


def _text(value: Any, *, where: str, allow_empty: bool = False) -> str:
    """要求文本类型；raw/derived 可保留首尾空白。"""
    if not isinstance(value, str):
        raise AuthoredTextFidelityCourseError(f"{where} 必须是字符串")
    if not allow_empty and not value:
        raise AuthoredTextFidelityCourseError(f"{where} 不能为空")
    return value


def _clean_text(value: Any, *, where: str, allow_empty: bool = False) -> str:
    """要求控制字段无首尾空白。"""
    text = _text(value, where=where, allow_empty=allow_empty)
    if text.strip() != text:
        raise AuthoredTextFidelityCourseError(f"{where} 不得含首尾空白")
    return text


def _integer(value: Any, *, where: str, nonnegative: bool = True) -> int:
    """要求严格整数并按需限制非负。"""
    if type(value) is not int or (nonnegative and value < 0):
        raise AuthoredTextFidelityCourseError(f"{where} 必须是合法严格整数")
    return value


def _flag(value: Any, *, where: str) -> int:
    """要求严格整数布尔标志。"""
    if type(value) is not int or value not in (0, 1):
        raise AuthoredTextFidelityCourseError(f"{where} 必须为 0/1")
    return value


def _string_tuple(
        value: Any,
        *,
        where: str,
        allow_empty: bool = False,
        sorted_unique: bool = False) -> tuple[str, ...]:
    """把 JSON string array 严格恢复为 tuple。"""
    if not isinstance(value, list) or (not value and not allow_empty):
        raise AuthoredTextFidelityCourseError(f"{where} 必须是字符串数组")
    result = tuple(_clean_text(item, where=where) for item in value)
    if len(result) != len(set(result)):
        raise AuthoredTextFidelityCourseError(f"{where} 含重复项")
    if sorted_unique and result != tuple(sorted(result)):
        raise AuthoredTextFidelityCourseError(f"{where} 必须排序")
    return result


def _sha256_text(value: str) -> str:
    """计算 UTF-8 文本 SHA-256。"""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_operation(
        operation: CanonicalJsonObject,
        *,
        raw_length: int,
        derived_length: int) -> dict[str, Any]:
    """校验 receipt operation 的类型、span 和可逆标志。"""
    raw = _exact_keys(
        operation.to_value(), _OPERATION_FIELDS,
        where="normalization operation",
    )
    kind = _clean_text(raw["operation_kind"], where="operation_kind")
    if kind not in OPERATION_KINDS:
        raise AuthoredTextFidelityCourseError("operation_kind 未登记")
    input_start = _integer(raw["input_start"], where="input_start")
    input_end = _integer(raw["input_end"], where="input_end")
    output_start = _integer(raw["output_start"], where="output_start")
    output_end = _integer(raw["output_end"], where="output_end")
    if not (input_start <= input_end <= raw_length):
        raise AuthoredTextFidelityCourseError("operation input span 越界")
    if not (output_start <= output_end <= derived_length):
        raise AuthoredTextFidelityCourseError("operation output span 越界")
    _flag(raw["reversible"], where="operation reversible")
    return raw


@dataclass(frozen=True)
class TextFidelityPayloadAudit:
    """返回 payload 重算后的原文/派生 hash 和不可逆操作数。"""

    raw_sha256: str
    derived_sha256: str
    irreversible_operation_count: int
    objective_count: int


def validate_text_fidelity_payload(
        payload: CanonicalJsonObject | dict[str, Any],
        ) -> TextFidelityPayloadAudit:
    """重算 raw/derived/receipt/长度，拒绝覆盖原文或静默信息损失。"""
    value = payload.to_value() if isinstance(payload, CanonicalJsonObject) else payload
    raw = _exact_keys(value, set(TEXT_FIDELITY_PAYLOAD_KEYS), where="payload")
    raw_observation = _exact_keys(raw["raw_observation"], {
        "append_only", "sha256", "text",
    }, where="raw_observation")
    raw_text = _text(raw_observation["text"], where="raw text")
    if _flag(raw_observation["append_only"], where="raw append_only") != 1:
        raise AuthoredTextFidelityCourseError("raw Observation 必须 append-only")
    raw_sha256 = _sha256_text(raw_text)
    if raw_observation["sha256"] != raw_sha256:
        raise AuthoredTextFidelityCourseError("raw Observation SHA-256 漂移")

    derived = _exact_keys(raw["derived_candidate"], {
        "segments", "sha256", "text", "tokens",
    }, where="derived_candidate")
    derived_text = _text(derived["text"], where="derived text")
    derived_sha256 = _sha256_text(derived_text)
    if derived["sha256"] != derived_sha256:
        raise AuthoredTextFidelityCourseError("derived candidate SHA-256 漂移")
    segments = _string_tuple(
        derived["segments"], where="segments", allow_empty=True)
    tokens = _string_tuple(derived["tokens"], where="tokens", allow_empty=True)

    candidate_kind = _clean_text(raw["candidate_kind"], where="candidate_kind")
    if candidate_kind not in CANDIDATE_KINDS:
        raise AuthoredTextFidelityCourseError("candidate_kind 未登记")
    if candidate_kind == "SEGMENTATION" and len(segments) < 2:
        raise AuthoredTextFidelityCourseError("分段候选必须保留多个 span")
    if candidate_kind == "TOKENIZATION" and len(tokens) < 2:
        raise AuthoredTextFidelityCourseError("分词候选必须保留多个 token")

    receipt = _exact_keys(raw["normalization_receipt"], {
        "derived_sha256", "operations", "raw_sha256", "receipt_version",
    }, where="normalization_receipt")
    if receipt["receipt_version"] != 1:
        raise AuthoredTextFidelityCourseError("receipt_version 非法")
    if receipt["raw_sha256"] != raw_sha256:
        raise AuthoredTextFidelityCourseError("receipt raw SHA-256 漂移")
    if receipt["derived_sha256"] != derived_sha256:
        raise AuthoredTextFidelityCourseError("receipt derived SHA-256 漂移")
    if not isinstance(receipt["operations"], list) or not receipt["operations"]:
        raise AuthoredTextFidelityCourseError("receipt operations 不能为空")
    operations = tuple(CanonicalJsonObject.from_value(item)
                       for item in receipt["operations"])
    checked = tuple(_validate_operation(
        item, raw_length=len(raw_text), derived_length=len(derived_text))
        for item in operations)
    irreversible_count = sum(1 for item in checked if item["reversible"] == 0)
    information_loss = _flag(raw["information_loss"], where="information_loss")
    loss_kind = _clean_text(raw["loss_kind"], where="loss_kind")
    if loss_kind not in LOSS_KINDS:
        raise AuthoredTextFidelityCourseError("loss_kind 未登记")
    if information_loss == 0 and (irreversible_count or loss_kind != "NONE"):
        raise AuthoredTextFidelityCourseError("可逆候选不得伪报信息损失")
    if information_loss == 1 and (not irreversible_count or loss_kind == "NONE"):
        raise AuthoredTextFidelityCourseError("不可逆候选必须显式声明损失")

    description = _exact_keys(raw["description_length"], {
        "derived_unit_count", "longer_by_units", "raw_unit_count",
        "shorter_by_units", "unit_kind",
    }, where="description_length")
    if description["unit_kind"] != "UNICODE_CODE_POINT":
        raise AuthoredTextFidelityCourseError("description unit_kind 非法")
    raw_units = _integer(description["raw_unit_count"], where="raw_unit_count")
    derived_units = _integer(
        description["derived_unit_count"], where="derived_unit_count")
    if raw_units != len(raw_text) or derived_units != len(derived_text):
        raise AuthoredTextFidelityCourseError("description length 未重算")
    if description["shorter_by_units"] != max(0, raw_units - derived_units):
        raise AuthoredTextFidelityCourseError("shorter_by_units 漂移")
    if description["longer_by_units"] != max(0, derived_units - raw_units):
        raise AuthoredTextFidelityCourseError("longer_by_units 漂移")

    objective_keys = tuple(raw["objective_keys"])
    if objective_keys != tuple(sorted(set(objective_keys))):
        raise AuthoredTextFidelityCourseError("objective_keys 必须排序去重")
    if not objective_keys or any(
            key not in LANGUAGE_OBJECTIVE_KEYS for key in objective_keys):
        raise AuthoredTextFidelityCourseError("objective_keys 未登记")
    if raw["sample_family"] not in SAMPLE_FAMILIES:
        raise AuthoredTextFidelityCourseError("sample_family 未登记")
    if raw["selection_state"] != "UNSELECTED":
        raise AuthoredTextFidelityCourseError("候选不得在 Observation 私选")
    _clean_text(raw["candidate_group"], where="candidate_group")
    _clean_text(raw["candidate_id"], where="candidate_id")
    anchor = _clean_text(
        raw["retention_anchor_id"], where="retention_anchor_id",
        allow_empty=True,
    )
    if raw["sample_family"] == "RETENTION" and not anchor:
        raise AuthoredTextFidelityCourseError("retention 必须绑定重验 anchor")
    if raw["sample_family"] != "RETENTION" and anchor:
        raise AuthoredTextFidelityCourseError("非 retention 不得伪造 anchor")
    return TextFidelityPayloadAudit(
        raw_sha256,
        derived_sha256,
        irreversible_count,
        len(objective_keys),
    )


@dataclass(frozen=True)
class AuthoredTextFidelitySeed:
    """一个 raw 不可覆盖、派生候选可审计的 LC-01 seed。"""

    seed_id: str
    family: str
    template_family: str
    label_owner: str
    split: str
    sample_family: str
    sample_role: str
    candidate_id: str
    candidate_group: str
    candidate_kind: str
    raw_text: str
    derived_text: str
    segments: tuple[str, ...]
    tokens: tuple[str, ...]
    operations: tuple[CanonicalJsonObject, ...]
    information_loss: int
    loss_kind: str
    objective_keys: tuple[str, ...]
    expected_state: str
    expected_payload: CanonicalJsonObject
    evaluation_dimension: str
    perturbation_kind: str
    supersedes_seed_id: str
    retention_anchor_id: str
    logical_order: int
    license_id: str

    def __post_init__(self) -> None:
        for name, value in (
                ("seed_id", self.seed_id),
                ("family", self.family),
                ("template_family", self.template_family),
                ("candidate_id", self.candidate_id),
                ("candidate_group", self.candidate_group),
                ("perturbation_kind", self.perturbation_kind)):
            _clean_text(value, where=name)
        if self.label_owner not in {"teacher", "evaluator"}:
            raise AuthoredTextFidelityCourseError("label_owner 非法")
        expected_split = "train" if self.label_owner == "teacher" else "held_out"
        if self.split != expected_split:
            raise AuthoredTextFidelityCourseError("label_owner 与 split 不一致")
        if self.sample_family not in SAMPLE_FAMILIES:
            raise AuthoredTextFidelityCourseError("sample_family 未登记")
        if self.sample_role not in SAMPLE_ROLES:
            raise AuthoredTextFidelityCourseError("sample_role 未登记")
        if self.candidate_kind not in CANDIDATE_KINDS:
            raise AuthoredTextFidelityCourseError("candidate_kind 未登记")
        _text(self.raw_text, where="raw_text")
        _text(self.derived_text, where="derived_text")
        if not isinstance(self.segments, tuple) or not isinstance(self.tokens, tuple):
            raise AuthoredTextFidelityCourseError("segments/tokens 类型非法")
        if not isinstance(self.operations, tuple) or not self.operations:
            raise AuthoredTextFidelityCourseError("operations 不能为空")
        if any(not isinstance(item, CanonicalJsonObject) for item in self.operations):
            raise AuthoredTextFidelityCourseError("operation 类型非法")
        _flag(self.information_loss, where="information_loss")
        if self.loss_kind not in LOSS_KINDS:
            raise AuthoredTextFidelityCourseError("loss_kind 未登记")
        if self.objective_keys != tuple(sorted(set(self.objective_keys))):
            raise AuthoredTextFidelityCourseError("objective_keys 必须排序去重")
        if not self.objective_keys or any(
                key not in LANGUAGE_OBJECTIVE_KEYS for key in self.objective_keys):
            raise AuthoredTextFidelityCourseError("objective_keys 未登记")
        if self.expected_state not in EXPECTED_STATES:
            raise AuthoredTextFidelityCourseError("expected_state 非四态")
        if not isinstance(self.expected_payload, CanonicalJsonObject):
            raise AuthoredTextFidelityCourseError("expected_payload 类型非法")
        if self.evaluation_dimension not in TEXT_FIDELITY_EVALUATOR_DIMENSIONS:
            raise AuthoredTextFidelityCourseError("evaluation_dimension 未登记")
        _clean_text(
            self.supersedes_seed_id,
            where="supersedes_seed_id",
            allow_empty=True,
        )
        _clean_text(
            self.retention_anchor_id,
            where="retention_anchor_id",
            allow_empty=True,
        )
        if type(self.logical_order) is not int or self.logical_order <= 0:
            raise AuthoredTextFidelityCourseError("logical_order 必须为正严格整数")
        if self.license_id != LICENSE_ID:
            raise AuthoredTextFidelityCourseError("原创文本保真课程必须为 CC0-1.0")
        validate_text_fidelity_payload(self.observation_payload())

    def observation_payload(self) -> CanonicalJsonObject:
        """构造学生可见 raw/derived 双轨 payload，不含 expected。"""
        raw_sha256 = _sha256_text(self.raw_text)
        derived_sha256 = _sha256_text(self.derived_text)
        raw_units = len(self.raw_text)
        derived_units = len(self.derived_text)
        return CanonicalJsonObject.from_value({
            "candidate_group": self.candidate_group,
            "candidate_id": self.candidate_id,
            "candidate_kind": self.candidate_kind,
            "derived_candidate": {
                "segments": list(self.segments),
                "sha256": derived_sha256,
                "text": self.derived_text,
                "tokens": list(self.tokens),
            },
            "description_length": {
                "derived_unit_count": derived_units,
                "longer_by_units": max(0, derived_units - raw_units),
                "raw_unit_count": raw_units,
                "shorter_by_units": max(0, raw_units - derived_units),
                "unit_kind": "UNICODE_CODE_POINT",
            },
            "information_loss": self.information_loss,
            "loss_kind": self.loss_kind,
            "normalization_receipt": {
                "derived_sha256": derived_sha256,
                "operations": [item.to_value() for item in self.operations],
                "raw_sha256": raw_sha256,
                "receipt_version": 1,
            },
            "objective_keys": list(self.objective_keys),
            "raw_observation": {
                "append_only": 1,
                "sha256": raw_sha256,
                "text": self.raw_text,
            },
            "retention_anchor_id": self.retention_anchor_id,
            "sample_family": self.sample_family,
            "selection_state": "UNSELECTED",
        })

    def compiled_seed(self) -> AuthoredCompiledSeed:
        """映射到既有四 owner Dataset 发布器，不新建 record schema。"""
        return AuthoredCompiledSeed(
            self.seed_id,
            self.family,
            self.template_family,
            self.label_owner,
            self.split,
            self.sample_role,
            PAYLOAD_KIND,
            self.observation_payload(),
            self.expected_state,
            self.expected_payload,
            self.perturbation_kind,
            self.supersedes_seed_id,
            self.logical_order,
            (self.candidate_group, self.candidate_id, self.raw_text, self.derived_text),
            (self.candidate_group, self.raw_text),
            (self.candidate_kind, self.sample_family, *self.objective_keys),
            self.evaluation_dimension,
        )


def _seed_from_value(value: dict[str, Any]) -> AuthoredTextFidelitySeed:
    """从精确规范 JSON object 恢复一个课程 seed。"""
    raw = _exact_keys(value, _SEED_FIELDS, where="text fidelity seed")
    operations_value = raw["operations"]
    if not isinstance(operations_value, list) or not operations_value:
        raise AuthoredTextFidelityCourseError("operations 必须是非空数组")
    operations = tuple(CanonicalJsonObject.from_value(
        _exact_keys(item, _OPERATION_FIELDS, where="operation"))
        for item in operations_value)
    expected_payload = raw["expected_payload"]
    if not isinstance(expected_payload, dict):
        raise AuthoredTextFidelityCourseError("expected_payload 必须是 object")
    return AuthoredTextFidelitySeed(
        _clean_text(raw["seed_id"], where="seed_id"),
        _clean_text(raw["family"], where="family"),
        _clean_text(raw["template_family"], where="template_family"),
        _clean_text(raw["label_owner"], where="label_owner"),
        _clean_text(raw["split"], where="split"),
        _clean_text(raw["sample_family"], where="sample_family"),
        _clean_text(raw["sample_role"], where="sample_role"),
        _clean_text(raw["candidate_id"], where="candidate_id"),
        _clean_text(raw["candidate_group"], where="candidate_group"),
        _clean_text(raw["candidate_kind"], where="candidate_kind"),
        _text(raw["raw_text"], where="raw_text"),
        _text(raw["derived_text"], where="derived_text"),
        _string_tuple(raw["segments"], where="segments", allow_empty=True),
        _string_tuple(raw["tokens"], where="tokens", allow_empty=True),
        operations,
        _flag(raw["information_loss"], where="information_loss"),
        _clean_text(raw["loss_kind"], where="loss_kind"),
        _string_tuple(
            raw["objective_keys"], where="objective_keys", sorted_unique=True),
        _clean_text(raw["expected_state"], where="expected_state"),
        CanonicalJsonObject.from_value(expected_payload),
        _clean_text(raw["evaluation_dimension"], where="evaluation_dimension"),
        _clean_text(raw["perturbation_kind"], where="perturbation_kind"),
        _clean_text(
            raw["supersedes_seed_id"], where="supersedes_seed_id",
            allow_empty=True,
        ),
        _clean_text(
            raw["retention_anchor_id"], where="retention_anchor_id",
            allow_empty=True,
        ),
        _integer(raw["logical_order"], where="logical_order"),
        _clean_text(raw["license_id"], where="license_id"),
    )


def read_authored_text_fidelity_seeds(
        path: str | Path) -> tuple[AuthoredTextFidelitySeed, ...]:
    """严格读取规范 JSONL，并核 owner、七类 family、lattice 和 revision。"""
    try:
        payload = Path(path).read_bytes()
    except OSError as error:
        raise AuthoredTextFidelityCourseError("text fidelity sample 无法读取") from error
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise AuthoredTextFidelityCourseError("text fidelity sample 换行非法")
    seeds: list[AuthoredTextFidelitySeed] = []
    try:
        for line in payload.splitlines(keepends=True):
            value = parse_canonical_json_bytes(line[:-1], require_object=True)
            assert isinstance(value, dict)
            if canonical_json_line(value) != line:
                raise AuthoredTextFidelityCourseError("text fidelity seed 非规范 JSON")
            seeds.append(_seed_from_value(value))
    except AuthoredTextFidelityCourseError:
        raise
    except Exception as error:
        raise AuthoredTextFidelityCourseError("text fidelity seed 损坏") from error
    if not seeds:
        raise AuthoredTextFidelityCourseError("text fidelity seeds 不能为空")
    identifiers = [seed.seed_id for seed in seeds]
    orders = [seed.logical_order for seed in seeds]
    if len(identifiers) != len(set(identifiers)):
        raise AuthoredTextFidelityCourseError("text fidelity seed_id 重复")
    if orders != sorted(orders) or len(orders) != len(set(orders)):
        raise AuthoredTextFidelityCourseError("logical_order 必须严格递增")
    candidate_ids = [seed.candidate_id for seed in seeds]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise AuthoredTextFidelityCourseError("candidate_id 重复")
    index = {seed.seed_id: seed for seed in seeds}
    for seed in seeds:
        if seed.sample_family in _ROLE_BY_FAMILY:
            if seed.sample_role != _ROLE_BY_FAMILY[seed.sample_family]:
                raise AuthoredTextFidelityCourseError("sample family/role 不一致")
        elif seed.sample_family == "GENERATION":
            if seed.sample_role not in {"support", "read_only_probe", "refute"}:
                raise AuthoredTextFidelityCourseError("generation sample_role 非法")
        if seed.sample_family in _STATE_BY_FAMILY:
            if seed.expected_state != _STATE_BY_FAMILY[seed.sample_family]:
                raise AuthoredTextFidelityCourseError("sample family/state 不一致")
        elif seed.sample_family == "GENERATION":
            if seed.expected_state not in {"TRUE", "FALSE"}:
                raise AuthoredTextFidelityCourseError("generation state 非法")
        if seed.information_loss and seed.expected_state == "TRUE":
            raise AuthoredTextFidelityCourseError("不可逆候选不得直接标 TRUE")
        if seed.sample_family == "REVISION":
            target = index.get(seed.supersedes_seed_id)
            if (target is None or target.logical_order >= seed.logical_order
                    or target.family != seed.family or target.split != seed.split):
                raise AuthoredTextFidelityCourseError("revision 必须 supersede 同族更早 seed")
        elif seed.supersedes_seed_id:
            raise AuthoredTextFidelityCourseError("非 revision 不得 supersede")
        if seed.sample_family == "RETENTION":
            anchor = index.get(seed.retention_anchor_id)
            if (anchor is None or anchor.logical_order >= seed.logical_order
                    or anchor.split != seed.split):
                raise AuthoredTextFidelityCourseError("retention anchor 必须是同 split 更早 seed")
        elif seed.retention_anchor_id:
            raise AuthoredTextFidelityCourseError("非 retention 不得绑定 anchor")

    teacher = tuple(seed for seed in seeds if seed.label_owner == "teacher")
    evaluator = tuple(seed for seed in seeds if seed.label_owner == "evaluator")
    if ({seed.sample_family for seed in teacher} != set(SAMPLE_FAMILIES)
            or {seed.sample_family for seed in evaluator} != set(SAMPLE_FAMILIES)):
        raise AuthoredTextFidelityCourseError("teacher/evaluator 均须覆盖七类 sample family")
    if ({seed.family for seed in teacher} & {seed.family for seed in evaluator}
            or {seed.template_family for seed in teacher}
            & {seed.template_family for seed in evaluator}):
        raise AuthoredTextFidelityCourseError("teacher/evaluator family/template 必须互斥")
    if {key for seed in seeds for key in seed.objective_keys} != set(
            LANGUAGE_OBJECTIVE_KEYS):
        raise AuthoredTextFidelityCourseError("LC-15 初版目标未列全")
    for owner_seeds in (teacher, evaluator):
        ambiguous_groups: dict[str, int] = {}
        for seed in owner_seeds:
            if seed.sample_family == "AMBIGUOUS":
                ambiguous_groups[seed.candidate_group] = (
                    ambiguous_groups.get(seed.candidate_group, 0) + 1)
        if not ambiguous_groups or max(ambiguous_groups.values()) < 2:
            raise AuthoredTextFidelityCourseError("分段/分词 lattice 必须保留至少两个候选")
    return tuple(seeds)


def _course_spec() -> AuthoredCourseSpec:
    """返回复用既有 Dataset publisher 的 LC-01 课程 spec。"""
    return AuthoredCourseSpec(
        SOURCE_KEY,
        LICENSE_ID,
        COURSE_VERSION,
        ARTIFACT_VERSION,
        ADAPTER_VERSION,
        GENERATOR_VERSION,
        PARSER_VERSION,
        PACK_NAME,
        STAGE,
        SUBSTAGE,
        "authored-text-fidelity-seed-v1",
        "urn:pure-integer-ai:ph2:authored-text-fidelity-v1",
        "Pure Integer AI PH2 authored text fidelity seed",
        "TEXT_FIDELITY_LABEL",
        "lc01-text-fidelity",
        200,
    )


def compile_authored_text_fidelity_course(
        sample_path: str | Path,
        release_root: str | Path) -> AuthoredCourseBuild:
    """编译 LC-01/LC-15 初版 pack，不调用 teacher 或写学习状态。"""
    seeds = read_authored_text_fidelity_seeds(sample_path)
    compiled = tuple(seed.compiled_seed() for seed in seeds)
    return publish_authored_course(
        compiled, sample_path, release_root, _course_spec())


def initial_learning_objectives() -> tuple[LearningObjectiveSpec, ...]:
    """冻结 LC-15 初版十一类目标及其独立淘汰信号。"""
    mapping = {
        "CONTROLLED_PERTURBATION": (
            "PERTURBATION_DISCRIMINATION",
            "IRREVERSIBLE_LOSS_DISCLOSURE"),
        "CROSS_CONTEXT_CONSISTENCY": (
            "REVERIFY_CONFLICT", "RETENTION_REVERIFY"),
        "GENERATION_ADOPTION": (
            "GENERATION_POSTCHECK_ACCEPT", "GENERATION_SURFACE_FIDELITY"),
        "GENERATION_FAILURE": (
            "GENERATION_POSTCHECK_FAILURE", "GENERATION_SURFACE_FIDELITY"),
        "INTEGER_DESCRIPTION_LENGTH": (
            "DESCRIPTION_LENGTH_DELTA", "LEARNING_OBJECTIVE_BINDING"),
        "MASKED_SPAN": (
            "SPAN_RECONSTRUCTION_ERROR", "CANDIDATE_LATTICE"),
        "MASKED_TOKEN": (
            "TOKEN_RECONSTRUCTION_ERROR", "NORMALIZATION_RECEIPT"),
        "NEXT_DISCOURSE_UNIT": (
            "DISCOURSE_UNIT_PREDICTION_ERROR", "RETENTION_REVERIFY"),
        "NEXT_SPAN": (
            "SPAN_PREDICTION_ERROR", "CANDIDATE_LATTICE"),
        "NEXT_TOKEN": (
            "TOKEN_PREDICTION_ERROR", "NORMALIZATION_RECEIPT"),
        "ORDER_RECOVERY": (
            "ORDER_PREDICTION_ERROR", "CANDIDATE_LATTICE"),
    }
    return tuple(LearningObjectiveSpec(
        key,
        COURSE_CAPABILITY_KEYS,
        mapping[key][0],
        "TEACHER_EVIDENCE",
        "EVALUATOR_LABEL",
        mapping[key][1],
        1,
        0,
        0,
    ) for key in LANGUAGE_OBJECTIVE_KEYS)


def build_text_fidelity_course_manifest(
        sample_path: str | Path,
        build: AuthoredCourseBuild,
        *,
        artifact_relative_root: str = FORMAL_ARTIFACT_RELATIVE_ROOT,
        ) -> LanguageCourseManifest:
    """把 sample、pack、目标、evaluator、retention 和零执行状态汇合。"""
    sample = Path(sample_path)
    seeds = read_authored_text_fidelity_seeds(sample)
    sample_sha256 = hashlib.sha256(sample.read_bytes()).hexdigest()
    teacher = tuple(seed for seed in seeds if seed.label_owner == "teacher")
    evaluator = tuple(seed for seed in seeds if seed.label_owner == "evaluator")
    return LanguageCourseManifest(
        1,
        COURSE_MANIFEST_ARTIFACT_VERSION,
        "COURSE_FROZEN",
        "INITIAL_FROZEN",
        COURSE_TASK_KEYS,
        CanonicalJsonObject.from_value({
            "RAW_TEXT_NOISE": "COURSE_FROZEN",
            "TYPED_LEARNING_OBJECTIVES": "PARTIAL_COURSE",
        }),
        STAGE,
        SUBSTAGE,
        SOURCE_KEY,
        LICENSE_ID,
        f"data/ph2/{sample.name}",
        sample_sha256,
        len(seeds),
        SAMPLE_FAMILIES,
        COURSE_SPLIT_AXES,
        tuple(sorted({seed.family for seed in teacher})),
        tuple(sorted({seed.family for seed in evaluator})),
        tuple(sorted({seed.template_family for seed in teacher})),
        tuple(sorted({seed.template_family for seed in evaluator})),
        PAYLOAD_KIND,
        TEXT_FIDELITY_PAYLOAD_KEYS,
        OBJECTIVE_TAXONOMY_VERSION,
        initial_learning_objectives(),
        TEXT_FIDELITY_EVALUATOR_DIMENSIONS,
        COURSE_VERIFIER_NE_CONDITIONS,
        COURSE_RETENTION_PROTOCOLS,
        f"{artifact_relative_root}/packs/{PACK_NAME}/manifest.json",
        build.manifest.sha256(),
        build.manifest.record_count,
        build.manifest.splits,
        CanonicalJsonObject.from_value(COURSE_INVARIANTS),
    )


__all__ = [
    "ARTIFACT_VERSION",
    "AuthoredTextFidelityCourseError",
    "AuthoredTextFidelitySeed",
    "CANDIDATE_KINDS",
    "COURSE_MANIFEST_ARTIFACT_VERSION",
    "COURSE_MANIFEST_PATH",
    "COURSE_VERSION",
    "FORMAL_ARTIFACT_RELATIVE_ROOT",
    "LICENSE_ID",
    "LOSS_KINDS",
    "OBJECTIVE_TAXONOMY_VERSION",
    "OPERATION_KINDS",
    "PACK_NAME",
    "PAYLOAD_KIND",
    "SOURCE_KEY",
    "STAGE",
    "SUBSTAGE",
    "TextFidelityPayloadAudit",
    "build_text_fidelity_course_manifest",
    "compile_authored_text_fidelity_course",
    "initial_learning_objectives",
    "read_authored_text_fidelity_seeds",
    "validate_text_fidelity_payload",
]
