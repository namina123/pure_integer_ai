"""发布 normalization successor 双运行等价后的禁用态 rule pack。

publisher 只消费两个完整 learner run 和同一 TRAIN protocol；reader 以外部冻结
pack/protocol SHA 及 protocol 重派生结果严格回读，不接触任何 evaluation。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_successor_learner import (
    read_normalization_successor_learner,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_successor_learning_records import (
    NORMALIZATION_SUCCESSOR_OUTPUT_FILE_ROLES,
    derive_normalization_successor_learning_outputs,
    normalization_successor_output_payloads,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_successor_training_protocol import (
    read_normalization_successor_learner_input,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    canonical_json_line,
)


NORMALIZATION_SUCCESSOR_RULE_PACK_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_SUCCESSOR_RULE_PACK_V1")
NORMALIZATION_SUCCESSOR_RULE_PACK_STATUS = (
    "FROZEN_NOT_EVALUATED_NOT_DEPLOYED")


def _sha256(payload: bytes) -> str:
    """返回规范 artifact、文件或结果的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _sha_value(value: object, *, label: str) -> str:
    """核验并返回小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise BroadQaExternalDataError(f"{label} 非法")
    return value


def _strict_equal(value: object, expected: object) -> bool:
    """递归比较 JSON 值并区分 bool 与 int。"""
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return (set(value) == set(expected)
                and all(_strict_equal(value[key], expected[key])
                        for key in expected))
    if isinstance(expected, list):
        return (len(value) == len(expected)
                and all(_strict_equal(item, expected_item)
                        for item, expected_item in zip(value, expected)))
    return value == expected


def _require_k_run_root(value: str | Path) -> Path:
    """要求 pack 发布根是已存在 K 盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "normalization successor pack run root 必须是 K 盘目录")
    return root


def _within(root: Path, value: str | Path, *, label: str) -> Path:
    """解析 publisher 输入输出并拒绝逃出 K 盘 run root。"""
    path = Path(value).resolve()
    if not path.is_relative_to(root):
        raise BroadQaExternalDataError(f"{label} 必须位于 run root 内")
    return path


def _files(
        *,
        outputs: dict[str, tuple[dict[str, object], ...]],
        payloads: dict[str, bytes],
        ) -> list[dict[str, object]]:
    """构造 pack 内六份语义输出的物理承诺。"""
    return [{
        "bytes": len(payloads[name]),
        "record_count": len(outputs[name]),
        "relative_path": name,
        "role": role,
        "sha256": _sha256(payloads[name]),
    } for name, role, _identity in NORMALIZATION_SUCCESSOR_OUTPUT_FILE_ROLES]


def _semantic_result_sha256(
        *,
        protocol_manifest_sha256: str,
        files: list[dict[str, object]],
        summary: dict[str, object],
        ) -> str:
    """重算与两个 learner run identity 无关的唯一语义结果。"""
    return _sha256(canonical_json_bytes({
        "files": files,
        "protocol_manifest_sha256": protocol_manifest_sha256,
        "summary": summary,
    }))


def _lineage(manifest: dict[str, object], *, role: str) -> dict[str, object]:
    """从一个完整 learner manifest 投影 pack 所需运行血缘。"""
    return {
        "checkpoint_chain_sha256": manifest["checkpoint_chain_sha256"],
        "checkpoint_terminal_sha256": manifest["checkpoint_terminal_sha256"],
        "learner_manifest_sha256": manifest["manifest_sha256"],
        "resume_marker_count": manifest["resume_markers"]["record_count"],
        "role": role,
        "run_id": manifest["run_id"],
    }


def _manifest(
        *,
        protocol_manifest_sha256: str,
        files: list[dict[str, object]],
        summary: dict[str, object],
        lineages: list[dict[str, object]],
        ) -> dict[str, object]:
    """构造未评测、未部署、生产禁用的 successor pack manifest。"""
    return {
        "artifact_kind": NORMALIZATION_SUCCESSOR_RULE_PACK_KIND,
        "evaluation_or_reserve_read_count": 0,
        "failed_icu_v2_artifact_read_count": 0,
        "files": files,
        "format_version": 1,
        "fresh_resume_output_bytes_equal": 1,
        "learner_lineages": lineages,
        "mastery_claimed": 0,
        "production_enabled": 0,
        "protocol_manifest_sha256": protocol_manifest_sha256,
        "runtime_state": "LEARNED_PACK_DISABLED",
        "semantic_result_sha256": _semantic_result_sha256(
            protocol_manifest_sha256=protocol_manifest_sha256,
            files=files,
            summary=summary,
        ),
        "source_pack_read_count": 0,
        "status": NORMALIZATION_SUCCESSOR_RULE_PACK_STATUS,
        "summary": summary,
        "teacher_api_llm_call_count": 0,
    }


def _validate_lineages(lineages: object) -> list[dict[str, object]]:
    """核验 fresh/resumed 两条独立运行血缘及 resume 证据。"""
    expected_fields = {
        "checkpoint_chain_sha256", "checkpoint_terminal_sha256",
        "learner_manifest_sha256", "resume_marker_count", "role", "run_id",
    }
    if (not isinstance(lineages, list) or len(lineages) != 2
            or any(not isinstance(item, dict) or set(item) != expected_fields
                   for item in lineages)
            or [item["role"] for item in lineages] != ["FRESH", "RESUMED"]
            or lineages[0]["run_id"] == lineages[1]["run_id"]
            or type(lineages[0]["resume_marker_count"]) is not int
            or lineages[0]["resume_marker_count"] != 0
            or type(lineages[1]["resume_marker_count"]) is not int
            or lineages[1]["resume_marker_count"] < 1):
        raise BroadQaExternalDataError(
            "successor rule pack fresh/resumed lineage 漂移")
    for lineage in lineages:
        for name in (
                "checkpoint_chain_sha256", "checkpoint_terminal_sha256",
                "learner_manifest_sha256", "run_id"):
            _sha_value(lineage[name], label=f"successor lineage {name}")
    return lineages


def publish_normalization_successor_rule_pack(
        *,
        run_root: str | Path,
        protocol_dir: str | Path,
        expected_protocol_manifest_sha256: str,
        fresh_run_dir: str | Path,
        resumed_run_dir: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """验证双运行语义字节相等后，不可覆盖发布禁用态 successor pack。"""
    root = _require_k_run_root(run_root)
    protocol_root = _within(root, protocol_dir, label="protocol_dir")
    fresh_root = _within(root, fresh_run_dir, label="fresh_run_dir")
    resumed_root = _within(root, resumed_run_dir, label="resumed_run_dir")
    target = _within(root, target_dir, label="target_dir")
    protocol_sha = _sha_value(
        expected_protocol_manifest_sha256,
        label="successor pack protocol manifest")
    if (fresh_root == resumed_root or target.exists()
            or not fresh_root.is_dir() or not resumed_root.is_dir()):
        raise BroadQaExternalDataError(
            "successor pack run 混淆、缺失或 target 已存在")
    fresh_manifest, fresh_outputs = read_normalization_successor_learner(
        fresh_root,
        protocol_dir=protocol_root,
        expected_protocol_manifest_sha256=protocol_sha,
    )
    resumed_manifest, resumed_outputs = read_normalization_successor_learner(
        resumed_root,
        protocol_dir=protocol_root,
        expected_protocol_manifest_sha256=protocol_sha,
    )
    fresh_payloads = normalization_successor_output_payloads(fresh_outputs)
    resumed_payloads = normalization_successor_output_payloads(resumed_outputs)
    if (fresh_payloads != resumed_payloads
            or fresh_manifest["semantic_result_sha256"]
            != resumed_manifest["semantic_result_sha256"]
            or not _strict_equal(
                fresh_manifest["summary"], resumed_manifest["summary"])):
        raise BroadQaExternalDataError(
            "successor learner fresh/resume 语义输出不等价")
    lineages = _validate_lineages([
        _lineage(fresh_manifest, role="FRESH"),
        _lineage(resumed_manifest, role="RESUMED"),
    ])
    files = _files(outputs=fresh_outputs, payloads=fresh_payloads)
    pack_manifest = _manifest(
        protocol_manifest_sha256=protocol_sha,
        files=files,
        summary=fresh_manifest["summary"],
        lineages=lineages,
    )
    if (pack_manifest["semantic_result_sha256"]
            != fresh_manifest["semantic_result_sha256"]):
        raise BroadQaExternalDataError(
            "successor pack/learner semantic result 漂移")
    target.mkdir(parents=True)
    for name, _role, _identity in NORMALIZATION_SUCCESSOR_OUTPUT_FILE_ROLES:
        with (target / name).open("xb") as handle:
            handle.write(fresh_payloads[name])
    manifest_path = target / "manifest.json"
    with manifest_path.open("xb") as handle:
        handle.write(canonical_json_line(pack_manifest))
    return {
        **pack_manifest,
        "manifest_sha256": _sha256(manifest_path.read_bytes()),
    }


def read_normalization_successor_rule_pack(
        pack_dir: str | Path,
        *,
        protocol_dir: str | Path,
        expected_protocol_manifest_sha256: str,
        expected_pack_manifest_sha256: str,
        ) -> tuple[
            dict[str, object],
            dict[str, tuple[dict[str, object], ...]],
        ]:
    """以双外部 SHA 和 protocol 重派生严格回读禁用态 pack。"""
    root = Path(pack_dir).resolve()
    protocol_root = Path(protocol_dir).resolve()
    protocol_sha = _sha_value(
        expected_protocol_manifest_sha256,
        label="successor pack expected protocol manifest")
    pack_sha = _sha_value(
        expected_pack_manifest_sha256,
        label="successor pack expected manifest")
    values = read_normalization_successor_learner_input(
        protocol_root,
        expected_manifest_sha256=protocol_sha,
    )
    outputs, summary = derive_normalization_successor_learning_outputs(
        protocol_manifest=values[0],
        observations=values[1],
        groups=values[2],
        contexts=values[3],
        work=values[4],
    )
    payloads = normalization_successor_output_payloads(outputs)
    try:
        encoded_manifest = (root / "manifest.json").read_bytes()
        manifest = json.loads(encoded_manifest)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "successor rule pack manifest 不可读") from error
    if (_sha256(encoded_manifest) != pack_sha
            or not isinstance(manifest, dict)
            or canonical_json_line(manifest) != encoded_manifest):
        raise BroadQaExternalDataError(
            "successor rule pack manifest identity/encoding 漂移")
    lineages = _validate_lineages(manifest.get("learner_lineages"))
    for name, _role, _identity in NORMALIZATION_SUCCESSOR_OUTPUT_FILE_ROLES:
        try:
            stored = (root / name).read_bytes()
        except OSError as error:
            raise BroadQaExternalDataError(
                f"successor rule pack {name} 不可读") from error
        if stored != payloads[name]:
            raise BroadQaExternalDataError(
                f"successor rule pack {name} 与 protocol 派生漂移")
    expected = _manifest(
        protocol_manifest_sha256=protocol_sha,
        files=_files(outputs=outputs, payloads=payloads),
        summary=summary,
        lineages=lineages,
    )
    if not _strict_equal(manifest, expected):
        raise BroadQaExternalDataError(
            "successor rule pack manifest 漂移")
    return ({**manifest, "manifest_sha256": pack_sha}, outputs)


__all__ = [
    "NORMALIZATION_SUCCESSOR_RULE_PACK_KIND",
    "NORMALIZATION_SUCCESSOR_RULE_PACK_STATUS",
    "publish_normalization_successor_rule_pack",
    "read_normalization_successor_rule_pack",
]
