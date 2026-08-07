"""重建、发布和回读 PW-00A 可装载的 W09 推理状态 artifact。"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w09_contract import (
    make_w09_request,
    open_w09_frozen_contract,
)
from pure_integer_ai.experiments.ph2_w09_firewall import W09PayloadFirewall
from pure_integer_ai.experiments.ph2_w09_inference import (
    W09InferenceState,
    compile_w09_inference_state,
)


FORMAT_VERSION = 1
ARTIFACT_KIND = "PURE_INTEGER_AI_PW00A_W09_INFERENCE_ARTIFACT"
ARTIFACT_VERSION = "PW00A-W09-INFERENCE-20260807-A"
ARTIFACT_PATH = "data/ph2/manifests/pw00a_w09_inference_state_v1.json"
STATUS = "PW00A_LOADABLE_INFERENCE_EVIDENCED"
W09_RECEIPT_PATH = "data/ph2/manifests/d03_v1/w09_runtime_evidence_receipt_v1.json"
W09_RECEIPT_SHA256 = (
    "c3f2f8c4791d09cb7b7644cac78a68c2fff4069ac5f5249950f30ede18cb2ef0"
)
CORE_MANIFEST_PATH = "data/ph2/manifests/j_f2_core_artifact_manifest_v1.json"
CORE_MANIFEST_SHA256 = (
    "d68e8e27f3d0cfe0632f3d51ff56adfe0087b1546a1a58fc5fe6f5062e5e6759"
)
EXPECTED_INFERENCE_SHA256 = (
    "4df2369120d91512c503a6b0977adcf9b25ef78ade6605d88ce3232bed02ccc8"
)
EXPECTED_DUMP_SHA256 = (
    "9dc0862bea1aa536997235c158a914c8049c4f9e1cd13847617fe2a7825fef2e"
)
EXPECTED_HOST_SHA256 = (
    "b176656f2af248b4a7c1132ceaff4498bd773dbf79e6dd9d4fda8817fad1897f"
)
EXPECTED_TERMINAL_SEAL_SHA256 = (
    "426b4d0828e133cf2e3de51799bf33881336056db038b7dd8df939d3ed14f1ef"
)
_FORBIDDEN_ARTIFACT_TOKENS = (
    b'"expected', b'"label', b'"surface', b'"raw_observation',
    b'"private_path', b'"candidate_root":', b'"rotation_root":',
)


def _canonical_bytes(value: object) -> bytes:
    """把 artifact 编码为带单个末尾换行的规范 JSON 字节。"""
    return canonical_json_bytes(value) + b"\n"


def _file_sha256(root: Path, relative_path: str) -> str:
    """回读仓库内固定依赖并返回 SHA-256。"""
    target = (root / Path(*relative_path.split("/"))).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise ValueError(f"PW-00A inference 依赖缺失: {relative_path}")
    return hashlib.sha256(target.read_bytes()).hexdigest()


def _commitment(
        relative_path: str,
        sha256: str,
        status: str,
        ) -> dict[str, str]:
    """形成一个只含公开固定身份的依赖承诺。"""
    return {
        "relative_path": relative_path,
        "sha256": sha256,
        "status": status,
    }


def build_pw00a_w09_inference_artifact(
        repository_root: str | Path,
        ) -> dict[str, Any]:
    """从公开 train-only firewall 重建并核对封存承诺，不读取 private 数据。"""
    root = Path(repository_root).resolve()
    for path, expected in (
            (W09_RECEIPT_PATH, W09_RECEIPT_SHA256),
            (CORE_MANIFEST_PATH, CORE_MANIFEST_SHA256)):
        if _file_sha256(root, path) != expected:
            raise ValueError(f"PW-00A inference 公开依赖漂移: {path}")
    context = open_w09_frozen_contract(root)
    request = make_w09_request(context)
    payload = W09PayloadFirewall.open(root, context, request).read_training_payload()
    state = compile_w09_inference_state(payload)
    if state.sha256() != EXPECTED_INFERENCE_SHA256:
        raise ValueError("PW-00A inference 重建结果不等于 W09 封存承诺")
    value = {
        "artifact_kind": ARTIFACT_KIND,
        "artifact_version": ARTIFACT_VERSION,
        "commitments": {
            "candidate_host_sha256": EXPECTED_HOST_SHA256,
            "candidate_terminal_seal_sha256": EXPECTED_TERMINAL_SEAL_SHA256,
            "core_manifest": _commitment(
                CORE_MANIFEST_PATH, CORE_MANIFEST_SHA256, "HISTORICAL_BASE"),
            "runtime_dump_sha256": EXPECTED_DUMP_SHA256,
            "w09_receipt": _commitment(
                W09_RECEIPT_PATH, W09_RECEIPT_SHA256, "RUNTIME_EVIDENCED"),
        },
        "format_version": FORMAT_VERSION,
        "inference_state": state.to_dict(),
        "inference_state_sha256": EXPECTED_INFERENCE_SHA256,
        "readiness_transition": {
            "LANGUAGE_READINESS_REPUBLISHED": 0,
            "PW00A_STARTED": 0,
        },
        "reconstruction": {
            "candidate_root_reads": 0,
            "evaluator_label_reads": 0,
            "private_root_reads": 0,
            "rule_count": len(state.rules),
            "teacher_api_calls": 0,
            "training_evidence_count": state.training_evidence_count,
            "training_observation_count": state.training_record_count,
        },
        "status": STATUS,
    }
    encoded = canonical_json_bytes(value)
    if any(token in encoded for token in _FORBIDDEN_ARTIFACT_TOKENS):
        raise ValueError("PW-00A inference artifact 含禁止字段")
    return value


def _validate(value: dict[str, Any]) -> W09InferenceState:
    """严格核验 artifact 字段、承诺、零读取边界和可恢复状态。"""
    if set(value) != {
            "artifact_kind", "artifact_version", "commitments",
            "format_version", "inference_state", "inference_state_sha256",
            "readiness_transition", "reconstruction", "status"}:
        raise ValueError("PW-00A inference artifact 字段不精确")
    if (value["artifact_kind"] != ARTIFACT_KIND
            or value["artifact_version"] != ARTIFACT_VERSION
            or value["format_version"] != FORMAT_VERSION
            or value["status"] != STATUS
            or value["inference_state_sha256"] != EXPECTED_INFERENCE_SHA256):
        raise ValueError("PW-00A inference artifact 固定身份漂移")
    if value["readiness_transition"] != {
            "LANGUAGE_READINESS_REPUBLISHED": 0,
            "PW00A_STARTED": 0}:
        raise ValueError("PW-00A inference artifact 不得转移 readiness")
    if value["commitments"] != {
            "candidate_host_sha256": EXPECTED_HOST_SHA256,
            "candidate_terminal_seal_sha256": EXPECTED_TERMINAL_SEAL_SHA256,
            "core_manifest": _commitment(
                CORE_MANIFEST_PATH, CORE_MANIFEST_SHA256, "HISTORICAL_BASE"),
            "runtime_dump_sha256": EXPECTED_DUMP_SHA256,
            "w09_receipt": _commitment(
                W09_RECEIPT_PATH, W09_RECEIPT_SHA256, "RUNTIME_EVIDENCED")}:
        raise ValueError("PW-00A inference artifact 承诺漂移")
    state = W09InferenceState.from_dict(value["inference_state"])
    if state.sha256() != EXPECTED_INFERENCE_SHA256:
        raise ValueError("PW-00A inference state SHA 漂移")
    if value["reconstruction"] != {
            "candidate_root_reads": 0,
            "evaluator_label_reads": 0,
            "private_root_reads": 0,
            "rule_count": 299,
            "teacher_api_calls": 0,
            "training_evidence_count": 309,
            "training_observation_count": 309}:
        raise ValueError("PW-00A inference 重建审计漂移")
    if any(token in canonical_json_bytes(value)
           for token in _FORBIDDEN_ARTIFACT_TOKENS):
        raise ValueError("PW-00A inference artifact 含禁止字段")
    return state


def read_pw00a_w09_inference_artifact(
        repository_root: str | Path,
        path: str | Path = ARTIFACT_PATH,
        *,
        verify_dependencies: bool = True,
        ) -> tuple[dict[str, Any], W09InferenceState]:
    """规范回读可装载 artifact，并可严格复核当前公开依赖。"""
    root = Path(repository_root).resolve()
    target = Path(path)
    if not target.is_absolute():
        target = root / Path(*str(target).replace("\\", "/").split("/"))
    payload = target.read_bytes()
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise ValueError("PW-00A inference artifact 末尾换行非法")
    value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    if _canonical_bytes(value) != payload:
        raise ValueError("PW-00A inference artifact 非规范字节")
    state = _validate(value)
    if verify_dependencies:
        for dependency in value["commitments"].values():
            if (isinstance(dependency, dict)
                    and _file_sha256(root, dependency["relative_path"])
                    != dependency["sha256"]):
                raise ValueError(
                    "PW-00A inference 当前公开依赖漂移: "
                    f"{dependency['relative_path']}")
    return value, state


def publish_pw00a_w09_inference_artifact(
        repository_root: str | Path,
        *,
        target: str | Path = ARTIFACT_PATH,
        ) -> tuple[dict[str, Any], W09InferenceState]:
    """独占发布可装载 artifact，目标已存在时禁止重建或覆盖。"""
    root = Path(repository_root).resolve()
    destination = Path(target)
    if not destination.is_absolute():
        destination = root / Path(*str(destination).replace("\\", "/").split("/"))
    if destination.exists():
        raise ValueError("PW-00A inference artifact 已发布，禁止覆盖")
    value = build_pw00a_w09_inference_artifact(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as stream:
            stream.write(_canonical_bytes(value))
    except FileExistsError as error:
        raise ValueError("PW-00A inference artifact 已发布，禁止覆盖") from error
    restored, state = read_pw00a_w09_inference_artifact(root, destination)
    if restored != value:
        raise ValueError("PW-00A inference artifact 发布回读漂移")
    return restored, state


__all__ = [
    "ARTIFACT_KIND",
    "ARTIFACT_PATH",
    "ARTIFACT_VERSION",
    "EXPECTED_INFERENCE_SHA256",
    "STATUS",
    "build_pw00a_w09_inference_artifact",
    "publish_pw00a_w09_inference_artifact",
    "read_pw00a_w09_inference_artifact",
]
