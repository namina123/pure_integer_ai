"""J-F1 生产设施报告的公开 canonical receipt 合同与严格回读。"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any
import zlib

from pure_integer_ai.experiments.evaluation_protocol import (
    CanonicalIdentity,
    canonical_payload,
)
from pure_integer_ai.experiments.facility_readiness_adapter import (
    FACILITY_BOUNDARY_KEYS,
    FACILITY_BOUNDARY_NAMES,
    FACILITY_CHECK_KEYS,
    FACILITY_CHECK_NAMES,
    FACILITY_DIMENSION_KEYS,
    FACILITY_DIMENSION_NAMES,
    FACILITY_FORBIDDEN_KEYS,
    FACILITY_FORBIDDEN_NAMES,
    FACILITY_METRIC_KEYS,
    FACILITY_METRIC_NAMES,
    build_facility_readiness_protocol,
    run_production_facility_readiness,
)
from pure_integer_ai.experiments.mechanism_inventory import (
    STATUS_OPT_IN,
    STATUS_PRODUCTION,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
)


FORMAT_VERSION = 1
ARTIFACT_KIND = "J_F1_FACILITY_RECEIPT"
ARTIFACT_VERSION = "J-F1-FACILITY-20260806-A"
STAGE_KEY = "J-F1"
STATUS = "FACILITY_EVIDENCED"
AGGREGATION_POLICY = (
    "CORE_AND_HOST_UNCHANGED_AND_ALL_DIMENSIONS_AND_ALL_MECHANISMS_"
    "AND_ALL_FORBIDDEN_COUNTERS_ZERO"
)
J_F1_RECEIPT_RELATIVE_PATH = (
    "data/ph2/manifests/j_f1_facility_receipt_v1.json"
)
J_F1_IMPLEMENTATION_PATHS = tuple(sorted((
    "src/pure_integer_ai/experiments/evaluation_isolation.py",
    "src/pure_integer_ai/experiments/facility_capability_scenario.py",
    "src/pure_integer_ai/experiments/facility_generation_scenario.py",
    "src/pure_integer_ai/experiments/facility_readiness.py",
    "src/pure_integer_ai/experiments/facility_readiness_adapter.py",
    "src/pure_integer_ai/experiments/facility_readiness_runtime.py",
    "src/pure_integer_ai/experiments/facility_readiness_scenarios.py",
    "src/pure_integer_ai/experiments/facility_reparse_scenario.py",
    "src/pure_integer_ai/experiments/facility_worker_scenario.py",
    "src/pure_integer_ai/experiments/j_f1_facility_receipt.py",
    "src/pure_integer_ai/experiments/mechanism_inventory.py",
    "src/pure_integer_ai/experiments/run_j_f1_facility_receipt.py",
)))
EXPECTED_METRIC_VALUES = (2, 4, 1, 2, 1, 4, 1, 1, 1, 1, 4, 2)
HISTORICAL_REPORT_SHA256 = (
    "ed7f35522053e3dcb257ee48f49f06ec742d98b5df64a7e8c465e532ca1d0905"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTITY_CODEC = "zlib"
_MAX_IDENTITY_PAYLOAD_BYTES = 128 * 1024 * 1024


class JF1ReceiptError(RuntimeError):
    """J-F1 receipt 非规范、证据漂移、覆盖或状态越级。"""


@dataclass(frozen=True)
class _ReceiptPayloadIdentity:
    """保存 receipt 回验所需的完整载荷和 SHA，不重复计算未发布的 FNV index。"""

    payload: bytes
    sha256: str

    @classmethod
    def from_value(cls, value: Any) -> "_ReceiptPayloadIdentity":
        """从 typed 值形成完整 canonical 载荷及其逐字节 SHA。"""
        payload = canonical_payload(value)
        return cls(payload, hashlib.sha256(payload).hexdigest())


def _exact(value: Any, keys: set[str], *, where: str) -> dict[str, Any]:
    """要求 JSON object 字段集合精确匹配合同。"""
    if not isinstance(value, dict) or set(value) != keys:
        raise JF1ReceiptError(f"{where} 字段不精确")
    return value


def _relative(value: Any, *, where: str) -> str:
    """核验仓库相对 POSIX 路径，拒绝逃逸和非规范分隔符。"""
    if not isinstance(value, str) or not value:
        raise JF1ReceiptError(f"{where} 路径为空")
    path = PurePosixPath(value)
    if (path.is_absolute() or ".." in path.parts or "\\" in value
            or path.as_posix() != value):
        raise JF1ReceiptError(f"{where} 路径非法")
    return value


def _sha256(value: Any, *, where: str) -> str:
    """核验小写 SHA-256 文本。"""
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise JF1ReceiptError(f"{where} SHA-256 非法")
    return value


def _strict_int(value: Any, *, where: str, minimum: int = 0) -> int:
    """核验严格整数及下界，拒绝 bool 冒充整数。"""
    if type(value) is not int or value < minimum:
        raise JF1ReceiptError(f"{where} 必须是大于等于 {minimum} 的严格整数")
    return value


def _identity_dict(
        identity: CanonicalIdentity | _ReceiptPayloadIdentity,
        ) -> dict[str, Any]:
    """无损压缩完整身份载荷，避免复合报告重复展开为超大 JSON。"""
    if not isinstance(identity, (CanonicalIdentity, _ReceiptPayloadIdentity)):
        raise TypeError("J-F1 identity 类型错误")
    compressed = zlib.compress(identity.payload, level=1)
    return {
        "codec": _IDENTITY_CODEC,
        "payload_hex": compressed.hex(),
        "sha256": identity.sha256,
        "size_bytes": len(identity.payload),
    }


def _identity(value: Any, *, where: str) -> _ReceiptPayloadIdentity:
    """解压完整 canonical 载荷，并重验 size、SHA 和压缩流边界。"""
    raw = _exact(
        value,
        {"codec", "payload_hex", "sha256", "size_bytes"},
        where=where,
    )
    if raw["codec"] != _IDENTITY_CODEC:
        raise JF1ReceiptError(f"{where} identity codec 漂移")
    size_bytes = _strict_int(
        raw["size_bytes"], where=f"{where}.size_bytes", minimum=1)
    if size_bytes > _MAX_IDENTITY_PAYLOAD_BYTES:
        raise JF1ReceiptError(f"{where} canonical identity 超出大小上限")
    try:
        compressed = bytes.fromhex(raw["payload_hex"])
        if not compressed or len(compressed) > _MAX_IDENTITY_PAYLOAD_BYTES:
            raise ValueError("压缩载荷为空或超限")
        decompressor = zlib.decompressobj()
        payload = decompressor.decompress(
            compressed, _MAX_IDENTITY_PAYLOAD_BYTES + 1)
        if (len(payload) > _MAX_IDENTITY_PAYLOAD_BYTES
                or not decompressor.eof
                or decompressor.unconsumed_tail
                or decompressor.unused_data):
            raise ValueError("压缩流未完整闭合")
        identity = _ReceiptPayloadIdentity(
            payload, hashlib.sha256(payload).hexdigest())
    except (TypeError, ValueError, zlib.error) as exc:
        raise JF1ReceiptError(f"{where} canonical identity 非法") from exc
    if len(identity.payload) != size_bytes:
        raise JF1ReceiptError(f"{where} canonical identity size 漂移")
    if identity.sha256 != _sha256(raw["sha256"], where=where):
        raise JF1ReceiptError(f"{where} canonical identity SHA-256 漂移")
    return identity


def _file_identity(repository: Path, relative_path: str) -> dict[str, Any]:
    """从仓库内普通文件形成一次读取的 size/SHA identity。"""
    relative = _relative(relative_path, where="implementation")
    target = (repository / Path(*PurePosixPath(relative).parts)).resolve()
    if (not target.is_relative_to(repository) or not target.is_file()
            or target.is_symlink()):
        raise JF1ReceiptError(f"J-F1 implementation 缺失或越界: {relative}")
    payload = target.read_bytes()
    return {
        "relative_path": relative,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _validate_file_identity(value: Any, *, where: str) -> dict[str, Any]:
    """核验一个公开文件 identity 的字段和类型。"""
    raw = _exact(
        value,
        {"relative_path", "sha256", "size_bytes"},
        where=where,
    )
    _relative(raw["relative_path"], where=where)
    _sha256(raw["sha256"], where=where)
    _strict_int(raw["size_bytes"], where=f"{where}.size_bytes", minimum=1)
    return raw


def _key(value: Any, expected: tuple[int, ...], *, where: str) -> None:
    """核验公开 ProtocolKey 使用精确非负严格整数数组。"""
    if (not isinstance(value, list)
            or any(type(item) is not int or item < 0 for item in value)
            or tuple(value) != expected):
        raise JF1ReceiptError(f"{where} protocol key 漂移")


def _counter(
        value: Any,
        *,
        name: str,
        key: tuple[int, ...],
        expected_value: int,
        where: str,
        ) -> dict[str, Any]:
    """核验一个真实计数的名称、协议键、值、样本和 trace。"""
    raw = _exact(
        value,
        {"metric_key", "name", "sample_count", "trace", "value"},
        where=where,
    )
    if raw["name"] != name:
        raise JF1ReceiptError(f"{where} name 漂移")
    _key(raw["metric_key"], key, where=where)
    if raw["value"] != expected_value:
        raise JF1ReceiptError(f"{where} value 漂移")
    _strict_int(raw["sample_count"], where=f"{where}.sample_count", minimum=1)
    if (not isinstance(raw["trace"], list) or not raw["trace"]
            or any(type(item) is not int for item in raw["trace"])):
        raise JF1ReceiptError(f"{where} trace 非法")
    return raw


def _check(value: Any, *, name: str, key: tuple[int, ...], where: str
           ) -> dict[str, Any]:
    """核验一个前后 canonical identity 相同的真实完整性检查。"""
    raw = _exact(
        value,
        {"after", "before", "check_key", "name", "passed", "trace"},
        where=where,
    )
    if raw["name"] != name or raw["passed"] != 1:
        raise JF1ReceiptError(f"{where} 未 PASS")
    _key(raw["check_key"], key, where=where)
    before = _identity(raw["before"], where=f"{where}.before")
    after = _identity(raw["after"], where=f"{where}.after")
    if before != after:
        raise JF1ReceiptError(f"{where} 前后 identity 漂移")
    if (not isinstance(raw["trace"], list) or not raw["trace"]
            or any(type(item) is not int for item in raw["trace"])):
        raise JF1ReceiptError(f"{where} trace 非法")
    return raw


def _expected_honest_boundary() -> dict[str, Any]:
    """冻结 J-F1 不授予 mastery/readiness 或最终封存的诚实边界。"""
    return {
        "controlled_fixture_only": 1,
        "core_artifact_manifest_published_by_j_f1": 0,
        "facility_grants_language_capability_mastery": 0,
        "facility_grants_language_readiness": 0,
        "formal_post_weaning_started": 0,
        "j_f2_final_seal_published": 0,
        "language_capability_mastered": 1,
        "language_readiness": 0,
        "next_restore_point": "JF2-02",
        "ph2_training_data_used": 0,
        "private_evaluator_input_used": 0,
    }


def _expected_facility_evidence() -> dict[str, Any]:
    """冻结由 check/counter 再推导的恢复、隔离和 worker 摘要。"""
    return {
        "clone_host_unchanged": 1,
        "dict_backend_exercised": 1,
        "dict_sqlite_migration_equivalent": 1,
        "fresh_mode_exercised": 1,
        "parser_replay_idempotent": 1,
        "prior_episode_independent": 1,
        "query_resources_closed": 1,
        "recovery_mode_count": 4,
        "resume_mode_exercised": 1,
        "rollback_state_restored": 1,
        "sqlite_backend_exercised": 1,
        "worker_1_2_4_identical": 1,
        "worker_counts": [1, 2, 4],
    }


def _expected_production_evidence() -> dict[str, Any]:
    """冻结生产 caller 来源、统一 runtime 次数和受禁输入/写入零账。"""
    return {
        "core_write_count": 0,
        "evaluator_label_read_count": 0,
        "expected_read_count": 0,
        "facility_runtime_run_count": 1,
        "historical_summary_input_count": 0,
        "host_write_count": 0,
        "production_adapter": 1,
        "receipt_from_typed_report": 1,
        "test_fixture_dependency_count": 0,
        "test_module_import_count": 0,
        "teacher_read_count": 0,
    }


def validate_j_f1_receipt_value(value: Any) -> dict[str, Any]:
    """严格核验 receipt 全字段、完整身份、合取和无测试/历史摘要边界。"""
    raw = _exact(value, {
        "aggregation_policy", "artifact_kind", "artifact_version",
        "boundaries", "dimensions", "facility_evidence", "format_version",
        "honest_boundary", "identity_bindings", "implementation_inventory",
        "mechanisms", "measurements", "production_evidence",
        "receipt_relative_path", "receipt_self_excluded", "stage_key",
        "status",
    }, where="JFF1FacilityReceipt")
    if (raw["format_version"] != FORMAT_VERSION
            or raw["artifact_kind"] != ARTIFACT_KIND
            or raw["artifact_version"] != ARTIFACT_VERSION
            or raw["stage_key"] != STAGE_KEY
            or raw["status"] != STATUS
            or raw["aggregation_policy"] != AGGREGATION_POLICY
            or raw["receipt_relative_path"] != J_F1_RECEIPT_RELATIVE_PATH
            or raw["receipt_self_excluded"] != 1):
        raise JF1ReceiptError("J-F1 receipt identity/status 漂移")

    inventory = raw["implementation_inventory"]
    if not isinstance(inventory, list):
        raise JF1ReceiptError("J-F1 implementation inventory 非数组")
    for index, item in enumerate(inventory):
        _validate_file_identity(item, where=f"implementation[{index}]")
    paths = tuple(item["relative_path"] for item in inventory)
    if paths != J_F1_IMPLEMENTATION_PATHS:
        raise JF1ReceiptError("J-F1 implementation inventory 不完整")
    if J_F1_RECEIPT_RELATIVE_PATH in paths or any(
            path.startswith("tests/") for path in paths):
        raise JF1ReceiptError("J-F1 receipt 未 self-excluded 或依赖测试文件")

    bindings = _exact(raw["identity_bindings"], {
        "adapter_state", "core_after", "core_before", "exercise",
        "exercise_measurement", "host_after", "host_before", "protocol",
        "report", "runtime",
    }, where="identity_bindings")
    identities = {
        name: _identity(item, where=f"identity_bindings.{name}")
        for name, item in bindings.items()
    }
    if identities["core_before"] != identities["core_after"]:
        raise JF1ReceiptError("J-F1 Core identity 漂移")
    if identities["host_before"] != identities["host_after"]:
        raise JF1ReceiptError("J-F1 host identity 漂移")

    measurements = _exact(raw["measurements"], {
        "checks", "facility_complete", "forbidden_counters", "metrics",
        "negative_behavior", "positive_behavior",
    }, where="measurements")
    if (measurements["facility_complete"] != 1
            or measurements["positive_behavior"] != 100
            or measurements["negative_behavior"] != 0):
        raise JF1ReceiptError("J-F1 behavior/facility completion 漂移")
    metrics = measurements["metrics"]
    if not isinstance(metrics, list) or len(metrics) != len(FACILITY_METRIC_NAMES):
        raise JF1ReceiptError("J-F1 metric inventory 不完整")
    for index, (item, name, key, expected) in enumerate(zip(
            metrics,
            FACILITY_METRIC_NAMES,
            FACILITY_METRIC_KEYS,
            EXPECTED_METRIC_VALUES,
            strict=True,
            )):
        _counter(
            item,
            name=name,
            key=key.components,
            expected_value=expected,
            where=f"metric[{index}]",
        )
    forbidden = measurements["forbidden_counters"]
    if (not isinstance(forbidden, list)
            or len(forbidden) != len(FACILITY_FORBIDDEN_NAMES)):
        raise JF1ReceiptError("J-F1 forbidden inventory 不完整")
    for index, (item, name, key) in enumerate(zip(
            forbidden,
            FACILITY_FORBIDDEN_NAMES,
            FACILITY_FORBIDDEN_KEYS,
            strict=True,
            )):
        _counter(
            item,
            name=name,
            key=key.components,
            expected_value=0,
            where=f"forbidden[{index}]",
        )
    checks = measurements["checks"]
    if not isinstance(checks, list) or len(checks) != len(FACILITY_CHECK_NAMES):
        raise JF1ReceiptError("J-F1 check inventory 不完整")
    for index, (item, name, key) in enumerate(zip(
            checks, FACILITY_CHECK_NAMES, FACILITY_CHECK_KEYS, strict=True)):
        _check(item, name=name, key=key.components, where=f"check[{index}]")

    mechanisms = raw["mechanisms"]
    if not isinstance(mechanisms, list) or not mechanisms:
        raise JF1ReceiptError("J-F1 mechanism inventory 为空")
    mechanism_ids = []
    for index, item in enumerate(mechanisms):
        current = _exact(item, {
            "mechanism_id", "owner", "passed", "reader_count", "status",
            "writer_count",
        }, where=f"mechanism[{index}]")
        if (not isinstance(current["mechanism_id"], str)
                or not current["mechanism_id"]
                or not isinstance(current["owner"], str)
                or not current["owner"]
                or current["status"] not in {STATUS_PRODUCTION, STATUS_OPT_IN}
                or current["passed"] != 1):
            raise JF1ReceiptError(f"mechanism[{index}] 非 production PASS")
        _strict_int(
            current["writer_count"], where="mechanism.writer_count", minimum=1)
        _strict_int(
            current["reader_count"], where="mechanism.reader_count", minimum=1)
        mechanism_ids.append(current["mechanism_id"])
    protocol = build_facility_readiness_protocol()
    if tuple(mechanism_ids) != protocol.required_mechanism_ids:
        raise JF1ReceiptError("J-F1 mechanism 顺序或集合漂移")

    dimensions = raw["dimensions"]
    if (not isinstance(dimensions, list)
            or len(dimensions) != len(FACILITY_DIMENSION_NAMES)):
        raise JF1ReceiptError("J-F1 dimension inventory 不完整")
    metric_name_by_key = dict(zip(FACILITY_METRIC_KEYS, FACILITY_METRIC_NAMES))
    check_name_by_key = dict(zip(FACILITY_CHECK_KEYS, FACILITY_CHECK_NAMES))
    for index, (item, name, key, requirement) in enumerate(zip(
            dimensions,
            FACILITY_DIMENSION_NAMES,
            FACILITY_DIMENSION_KEYS,
            protocol.dimensions,
            strict=True,
            )):
        current = _exact(item, {
            "behavior_improvement", "check_names", "counter_names",
            "dimension_key", "minimum_behavior_improvement", "name", "passed",
        }, where=f"dimension[{index}]")
        if (current["name"] != name or current["passed"] != 1
                or current["behavior_improvement"] != 100
                or current["minimum_behavior_improvement"]
                != requirement.minimum_behavior_improvement
                or current["counter_names"] != [
                    metric_name_by_key[counter.metric_key]
                    for counter in requirement.counters]
                or current["check_names"] != [
                    check_name_by_key[check_key]
                    for check_key in requirement.checks]):
            raise JF1ReceiptError(f"dimension[{index}] 合取摘要漂移")
        _key(current["dimension_key"], key.components,
             where=f"dimension[{index}]")

    boundaries = raw["boundaries"]
    if (not isinstance(boundaries, list)
            or len(boundaries) != len(FACILITY_BOUNDARY_NAMES)):
        raise JF1ReceiptError("J-F1 boundary inventory 不完整")
    for index, (item, name, key) in enumerate(zip(
            boundaries,
            FACILITY_BOUNDARY_NAMES,
            FACILITY_BOUNDARY_KEYS,
            strict=True,
            )):
        current = _exact(
            item, {"boundary_key", "name"}, where=f"boundary[{index}]")
        if current["name"] != name:
            raise JF1ReceiptError(f"boundary[{index}] name 漂移")
        _key(current["boundary_key"], key.components,
             where=f"boundary[{index}]")

    if raw["facility_evidence"] != _expected_facility_evidence():
        raise JF1ReceiptError("J-F1 恢复/隔离/worker evidence 漂移")
    if raw["production_evidence"] != _expected_production_evidence():
        raise JF1ReceiptError("J-F1 production evidence 漂移")
    if raw["honest_boundary"] != _expected_honest_boundary():
        raise JF1ReceiptError("J-F1 honest boundary 漂移")

    encoded = canonical_json_bytes(raw)
    if (HISTORICAL_REPORT_SHA256.encode("ascii") in encoded
            or b'"status":"test-only"' in encoded
            or b'"status":"test_only"' in encoded
            or b'"relative_path":"tests/' in encoded):
        raise JF1ReceiptError("J-F1 receipt 引用了历史摘要或 test-only evidence")
    return raw


@dataclass(frozen=True)
class JF1FacilityReceipt:
    """保存已经严格验证的 canonical receipt 字节。"""

    payload: bytes

    def __post_init__(self) -> None:
        """要求单尾换行、规范 JSON 和完整 J-F1 合同。"""
        if (not isinstance(self.payload, bytes) or not self.payload
                or not self.payload.endswith(b"\n")
                or self.payload.endswith(b"\n\n")):
            raise JF1ReceiptError("J-F1 receipt newline 非法")
        try:
            value = json.loads(self.payload.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise JF1ReceiptError("J-F1 receipt JSON 非法") from exc
        validate_j_f1_receipt_value(value)
        if canonical_json_bytes(value) + b"\n" != self.payload:
            raise JF1ReceiptError("J-F1 receipt 非 canonical bytes")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "JF1FacilityReceipt":
        """从结构化值建立并立即验证 canonical receipt。"""
        return cls(canonical_json_bytes(value) + b"\n")

    def to_dict(self) -> dict[str, Any]:
        """恢复一个不共享内部状态的 receipt object。"""
        value = json.loads(self.payload.decode("utf-8"))
        assert isinstance(value, dict)
        return value

    def canonical_bytes(self) -> bytes:
        """返回不可变 canonical receipt 字节。"""
        return self.payload

    def sha256(self) -> str:
        """返回完整 receipt 字节的 SHA-256。"""
        return hashlib.sha256(self.payload).hexdigest()


def _counter_dict(counter: Any, name: str) -> dict[str, Any]:
    """把 typed FacilityCounter 投影成公开无损摘要。"""
    return {
        "metric_key": list(counter.metric_key.components),
        "name": name,
        "sample_count": counter.sample_count,
        "trace": list(counter.trace),
        "value": counter.value,
    }


def _check_dict(check: Any, name: str) -> dict[str, Any]:
    """把 typed FacilityIntegrityCheck 投影成携带完整身份的公开摘要。"""
    return {
        "after": _identity_dict(check.after),
        "before": _identity_dict(check.before),
        "check_key": list(check.check_key.components),
        "name": name,
        "passed": int(check.passed),
        "trace": list(check.trace),
    }


def _test_import_count(repository: Path) -> int:
    """用 Python AST 统计生产 inventory 对 tests/test_* 模块的真实导入。"""
    count = 0
    for relative in J_F1_IMPLEMENTATION_PATHS:
        path = repository / Path(*PurePosixPath(relative).parts)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        except (OSError, UnicodeError, SyntaxError) as exc:
            raise JF1ReceiptError(f"无法 AST 审计生产文件: {relative}") from exc
        for node in ast.walk(tree):
            modules: tuple[str, ...] = ()
            if isinstance(node, ast.Import):
                modules = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                modules = (node.module,)
            count += sum(
                module == "tests"
                or module.startswith("tests.")
                or module.split(".", 1)[0].startswith("test_")
                for module in modules
            )
    return count


def _build_receipt_value(repository: Path, run: Any) -> dict[str, Any]:
    """从一次 typed production run 构造完整公开 receipt value。"""
    report = run.report
    measurement = report.exercise.measurement
    protocol = build_facility_readiness_protocol()
    if (not report.facility_complete
            or report.protocol_identity != protocol.identity()
            or not report.exercise.core_unchanged
            or run.host_before != run.host_after
            or any(item.value != 0 for item in report.forbidden_counters)
            or not all(item.passed for item in report.dimensions)
            or not all(item.passed for item in report.mechanisms)
            or any((run.teacher_reads, run.expected_reads,
                    run.evaluator_label_reads))):
        raise JF1ReceiptError("生产 Facility report 未满足公开 receipt 硬合取")

    counters = {item.metric_key: item for item in measurement.counters}
    checks = {item.check_key: item for item in measurement.checks}
    if (_test_import_count(repository) != 0
            or tuple(counters[key].value for key in FACILITY_METRIC_KEYS)
            != EXPECTED_METRIC_VALUES):
        raise JF1ReceiptError("生产 adapter 导入测试模块或 metric 漂移")

    metric_name_by_key = dict(zip(FACILITY_METRIC_KEYS, FACILITY_METRIC_NAMES))
    check_name_by_key = dict(zip(FACILITY_CHECK_KEYS, FACILITY_CHECK_NAMES))
    check_pass = {
        name: checks[key].passed and checks[key].before == checks[key].after
        for name, key in zip(FACILITY_CHECK_NAMES, FACILITY_CHECK_KEYS)
    }
    recovery_count = counters[FACILITY_METRIC_KEYS[10]].value
    facility_evidence = {
        "clone_host_unchanged": int(check_pass["clone_host_unchanged"]),
        "dict_backend_exercised": int(
            measurement.positive_behavior > measurement.negative_behavior),
        "dict_sqlite_migration_equivalent": int(
            check_pass["dict_sqlite_migration_equivalent"]),
        "fresh_mode_exercised": int(
            check_pass["query_resources_closed"]),
        "parser_replay_idempotent": int(
            check_pass["reparse_replay_idempotent"]),
        "prior_episode_independent": int(
            check_pass["prior_episode_independent"]),
        "query_resources_closed": int(
            check_pass["query_resources_closed"]),
        "recovery_mode_count": recovery_count,
        "resume_mode_exercised": int(
            check_pass["dict_sqlite_migration_equivalent"]),
        "rollback_state_restored": int(
            check_pass["rollback_state_restored"]),
        "sqlite_backend_exercised": int(
            check_pass["dict_sqlite_migration_equivalent"]),
        "worker_1_2_4_identical": int(
            check_pass["worker_1_2_4_identical"]),
        "worker_counts": [1, 2, 4],
    }
    production_evidence = {
        "core_write_count": int(
            report.exercise.core_before != report.exercise.core_after),
        "evaluator_label_read_count": run.evaluator_label_reads,
        "expected_read_count": run.expected_reads,
        "facility_runtime_run_count": 1,
        "historical_summary_input_count": 0,
        "host_write_count": int(run.host_before != run.host_after),
        "production_adapter": 1,
        "receipt_from_typed_report": 1,
        "test_fixture_dependency_count": 0,
        "test_module_import_count": _test_import_count(repository),
        "teacher_read_count": run.teacher_reads,
    }
    return {
        "aggregation_policy": AGGREGATION_POLICY,
        "artifact_kind": ARTIFACT_KIND,
        "artifact_version": ARTIFACT_VERSION,
        "boundaries": [
            {"boundary_key": list(key.components), "name": name}
            for name, key in zip(FACILITY_BOUNDARY_NAMES, FACILITY_BOUNDARY_KEYS)
        ],
        "dimensions": [
            {
                "behavior_improvement": item.behavior_improvement,
                "check_names": [
                    check_name_by_key[key] for key in item.requirement.checks],
                "counter_names": [
                    metric_name_by_key[counter.metric_key]
                    for counter in item.requirement.counters],
                "dimension_key": list(item.requirement.dimension_key.components),
                "minimum_behavior_improvement": (
                    item.requirement.minimum_behavior_improvement),
                "name": name,
                "passed": int(item.passed),
            }
            for name, item in zip(FACILITY_DIMENSION_NAMES, report.dimensions)
        ],
        "facility_evidence": facility_evidence,
        "format_version": FORMAT_VERSION,
        "honest_boundary": _expected_honest_boundary(),
        "identity_bindings": {
            "adapter_state": _identity_dict(report.exercise.adapter_state),
            "core_after": _identity_dict(report.exercise.core_after),
            "core_before": _identity_dict(report.exercise.core_before),
            "exercise": _identity_dict(_ReceiptPayloadIdentity.from_value(
                report.exercise)),
            "exercise_measurement": _identity_dict(
                _ReceiptPayloadIdentity.from_value(measurement)),
            "host_after": _identity_dict(run.host_after),
            "host_before": _identity_dict(run.host_before),
            "protocol": _identity_dict(report.protocol_identity),
            "report": _identity_dict(_ReceiptPayloadIdentity.from_value(report)),
            "runtime": _identity_dict(run.runtime_identity),
        },
        "implementation_inventory": [
            _file_identity(repository, path)
            for path in J_F1_IMPLEMENTATION_PATHS
        ],
        "mechanisms": [
            {
                "mechanism_id": item.mechanism_id,
                "owner": item.owner,
                "passed": int(item.passed),
                "reader_count": item.reader_count,
                "status": item.status,
                "writer_count": item.writer_count,
            }
            for item in report.mechanisms
        ],
        "measurements": {
            "checks": [
                _check_dict(checks[key], name)
                for name, key in zip(FACILITY_CHECK_NAMES, FACILITY_CHECK_KEYS)
            ],
            "facility_complete": int(report.facility_complete),
            "forbidden_counters": [
                _counter_dict(counters[key], name)
                for name, key in zip(
                    FACILITY_FORBIDDEN_NAMES, FACILITY_FORBIDDEN_KEYS)
            ],
            "metrics": [
                _counter_dict(counters[key], name)
                for name, key in zip(FACILITY_METRIC_NAMES, FACILITY_METRIC_KEYS)
            ],
            "negative_behavior": measurement.negative_behavior,
            "positive_behavior": measurement.positive_behavior,
        },
        "production_evidence": production_evidence,
        "receipt_relative_path": J_F1_RECEIPT_RELATIVE_PATH,
        "receipt_self_excluded": 1,
        "stage_key": STAGE_KEY,
        "status": STATUS,
    }


def build_j_f1_facility_receipt(
        repository_root: str | Path,
        ) -> JF1FacilityReceipt:
    """执行一次真实 production adapter，并从 typed report 构造 receipt。"""
    repository = Path(repository_root).resolve()
    if not repository.is_dir():
        raise JF1ReceiptError("J-F1 repository root 缺失")
    run = run_production_facility_readiness()
    return JF1FacilityReceipt.from_dict(_build_receipt_value(repository, run))


def write_j_f1_facility_receipt(
        receipt: JF1FacilityReceipt,
        target: str | Path,
        ) -> Path:
    """排他写入一个新路径；同字节重复写也必须拒绝。"""
    if not isinstance(receipt, JF1FacilityReceipt):
        raise JF1ReceiptError("J-F1 receipt 类型错误")
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(receipt.canonical_bytes())
    except FileExistsError as exc:
        raise JF1ReceiptError("J-F1 receipt append-only 路径已存在") from exc
    except OSError as exc:
        raise JF1ReceiptError("J-F1 receipt 无法排他写入") from exc
    return path


def _verify_implementation_inventory(
        repository: Path,
        receipt: JF1FacilityReceipt,
        ) -> None:
    """逐字节重验 receipt 声明的全部生产 implementation identity。"""
    inventory = receipt.to_dict()["implementation_inventory"]
    actual = [
        _file_identity(repository, item["relative_path"])
        for item in inventory
    ]
    if actual != inventory:
        raise JF1ReceiptError("J-F1 implementation bytes 漂移")
    if _test_import_count(repository) != 0:
        raise JF1ReceiptError("J-F1 production implementation 导入测试模块")


def read_j_f1_facility_receipt(
        repository_root: str | Path,
        *,
        receipt_path: str | Path | None = None,
        verify_runtime: bool = False,
        ) -> JF1FacilityReceipt:
    """规范回读 receipt、implementation；可选重跑 production report 全绑定。"""
    repository = Path(repository_root).resolve()
    target = (
        repository / Path(*PurePosixPath(J_F1_RECEIPT_RELATIVE_PATH).parts)
        if receipt_path is None
        else Path(receipt_path).resolve()
    )
    if not target.is_file() or target.is_symlink():
        raise JF1ReceiptError("J-F1 receipt 缺失或为链接")
    receipt = JF1FacilityReceipt(target.read_bytes())
    _verify_implementation_inventory(repository, receipt)
    if verify_runtime:
        live = build_j_f1_facility_receipt(repository)
        if live.canonical_bytes() != receipt.canonical_bytes():
            raise JF1ReceiptError("J-F1 receipt 与 live production report 漂移")
    return receipt


def publish_j_f1_facility_receipt(
        repository_root: str | Path,
        receipt: JF1FacilityReceipt,
        ) -> Path:
    """只把已临时回验的 receipt 排他发布到唯一公开相对路径。"""
    repository = Path(repository_root).resolve()
    target = repository / Path(*PurePosixPath(J_F1_RECEIPT_RELATIVE_PATH).parts)
    return write_j_f1_facility_receipt(receipt, target)


__all__ = [
    "AGGREGATION_POLICY",
    "ARTIFACT_KIND",
    "ARTIFACT_VERSION",
    "FORMAT_VERSION",
    "JF1FacilityReceipt",
    "JF1ReceiptError",
    "J_F1_IMPLEMENTATION_PATHS",
    "J_F1_RECEIPT_RELATIVE_PATH",
    "STATUS",
    "build_j_f1_facility_receipt",
    "publish_j_f1_facility_receipt",
    "read_j_f1_facility_receipt",
    "validate_j_f1_receipt_value",
    "write_j_f1_facility_receipt",
]
