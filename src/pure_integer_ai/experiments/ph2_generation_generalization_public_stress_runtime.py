"""运行 GG-03 public stress，并不可覆盖发布 K 盘整数性能证据。"""
from __future__ import annotations

from pathlib import Path, PurePosixPath
import time

from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_generation_candidate_pack import (
    LoadedGenerationCandidatePack,
)
from pure_integer_ai.experiments.ph2_generation_generalization_evaluation_family_identity import (
    build_generation_generalization_code_identity,
    double_scan_generation_generalization_observation_inventory,
    generation_generalization_sha256_bytes,
)
from pure_integer_ai.experiments.ph2_generation_generalization_evaluation_observation import (
    read_generation_generalization_evaluation_observations,
)
from pure_integer_ai.experiments.ph2_generation_generalization_evaluation_runner import (
    GenerationGeneralizationEvaluationPolicy,
    generation_generalization_evaluation_requirements,
    run_generation_generalization_evaluation_actual,
)
from pure_integer_ai.experiments.ph2_generation_generalization_public_stress import (
    PUBLIC_STRESS_BUDGET,
    PUBLIC_STRESS_CASE_IDS,
)
from pure_integer_ai.experiments.train_context import TrainContext


ARTIFACT_KIND = "PH2_GG03_PUBLIC_STRESS_PERFORMANCE_RECEIPT_V1"
DEFAULT_RECEIPT_RELATIVE_PATH = (
    "publication/generation-generalization-public-stress-performance.json")


class GenerationGeneralizationPublicStressRuntimeError(RuntimeError):
    """public stress 性能运行、宿主隔离或 K 盘发布失败。"""


def _require_k_run_root(path: str | Path) -> Path:
    """要求性能运行根是显式、已存在的 K 盘目录。"""
    root = Path(path).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise GenerationGeneralizationPublicStressRuntimeError(
            "public stress run root 必须是已存在的 K 盘目录")
    return root


def _target_in_run_root(root: Path, relative_path: str) -> Path:
    """解析规范相对路径，并拒绝目标逃出显式 K 盘 run root。"""
    relative = PurePosixPath(relative_path)
    if (relative.is_absolute() or not relative.parts
            or any(part in {"", ".", ".."} for part in relative.parts)):
        raise GenerationGeneralizationPublicStressRuntimeError(
            "public stress receipt relative path 非法")
    target = (root / Path(*relative.parts)).resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise GenerationGeneralizationPublicStressRuntimeError(
            "public stress receipt 逃出 run root") from error
    return target


def _sha256_stable_key(value: tuple[int, ...]) -> str:
    """把完整纯整数 stable key 转为固定长度公开内容锁。"""
    return generation_generalization_sha256_bytes(
        canonical_json_line(list(value))[:-1])


def run_and_publish_generation_generalization_public_stress(
        host_ctx: TrainContext,
        loaded: LoadedGenerationCandidatePack,
        observation_path: str | Path,
        repository_root: str | Path,
        run_root: str | Path,
        *,
        receipt_relative_path: str = DEFAULT_RECEIPT_RELATIVE_PATH,
        policy: GenerationGeneralizationEvaluationPolicy | None = None,
        clock_ns=time.perf_counter_ns,
        ) -> tuple[Path, dict[str, object], str]:
    """运行全部 public stress，并发布无 surface 正文的性能 receipt。"""
    if not isinstance(host_ctx, TrainContext):
        raise TypeError("public stress host context 类型错误")
    if not isinstance(loaded, LoadedGenerationCandidatePack):
        raise TypeError("public stress candidate pack 类型错误")
    if not callable(clock_ns):
        raise TypeError("public stress clock_ns 必须可调用")
    policy = policy or GenerationGeneralizationEvaluationPolicy()
    if not isinstance(policy, GenerationGeneralizationEvaluationPolicy):
        raise TypeError("public stress policy 类型错误")
    root = _require_k_run_root(run_root)
    target = _target_in_run_root(root, receipt_relative_path)
    if target.exists():
        raise GenerationGeneralizationPublicStressRuntimeError(
            "public stress performance receipt 已存在")

    observations = read_generation_generalization_evaluation_observations(
        observation_path)
    if (len(observations) != len(PUBLIC_STRESS_CASE_IDS)
            or {item.episode_id for item in observations}
            != set(PUBLIC_STRESS_CASE_IDS)):
        raise GenerationGeneralizationPublicStressRuntimeError(
            "public stress Observation inventory 漂移")
    inventory = double_scan_generation_generalization_observation_inventory(
        observation_path, resource_ceiling=PUBLIC_STRESS_BUDGET)
    code_identity = build_generation_generalization_code_identity(
        repository_root)
    policy_sha256 = generation_generalization_sha256_bytes(
        canonical_json_line(policy.to_dict()))
    before = host_ctx.backend.snapshot()
    batch_started = clock_ns()
    cases = []
    total_elapsed_ns = 0
    for ordinal, observation in enumerate(observations, start=1):
        started = clock_ns()
        actual = run_generation_generalization_evaluation_actual(
            host_ctx, loaded, observation, policy)
        elapsed_ns = max(1, clock_ns() - started)
        if actual.runtime_status != "PASS_EVALUATION_ACTUAL_CONJUNCTION":
            raise GenerationGeneralizationPublicStressRuntimeError(
                f"public stress case 未 PASS: {observation.episode_id}")
        total_elapsed_ns += elapsed_ns
        cases.append({
            "actual_stable_key_sha256": _sha256_stable_key(
                actual.stable_key()),
            "elapsed_ns": elapsed_ns,
            "episode_id": observation.episode_id,
            "ordinal": ordinal,
            "output_surface_bytes": len(
                actual.surface_text.encode("utf-8")),
            "path": actual.path,
            "reference_option_count": (
                0 if observation.reference_course is None
                else len(observation.reference_course.options)),
            "requirements": list(
                generation_generalization_evaluation_requirements(
                    observation)),
            "runtime_status": actual.runtime_status,
            "surface_representation_units": len(
                actual.execution.representations),
            "verifier_dimension_count": actual.verifier_dimension_count,
        })
    batch_wall_elapsed_ns = max(1, clock_ns() - batch_started)
    after = host_ctx.backend.snapshot()
    if before != after:
        raise GenerationGeneralizationPublicStressRuntimeError(
            "public stress 修改了 host learning state")
    receipt: dict[str, object] = {
        "artifact_kind": ARTIFACT_KIND,
        "batch_wall_elapsed_ns": batch_wall_elapsed_ns,
        "candidate_payload_sha256": loaded.pack.sha256(),
        "cases": cases,
        "code_identity_sha256": code_identity.aggregate_sha256,
        "format_version": 1,
        "host_snapshot_equal": 1,
        "label_read_count": 0,
        "observation_inventory_sha256": inventory.record_inventory_sha256,
        "observation_transport_sha256": inventory.transport_sha256,
        "observation_transport_size_bytes": inventory.transport_size_bytes,
        "policy_sha256": policy_sha256,
        "record_count": len(cases),
        "status": "PASS",
        "teacher_call_count": 0,
        "total_elapsed_ns": total_elapsed_ns,
    }
    payload = canonical_json_line(receipt)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except OSError as error:
        raise GenerationGeneralizationPublicStressRuntimeError(
            "public stress performance receipt 发布失败") from error
    readback = target.read_bytes()
    if (readback != payload
            or parse_canonical_json_bytes(
                readback[:-1], require_object=True) != receipt):
        raise GenerationGeneralizationPublicStressRuntimeError(
            "public stress performance receipt 回读漂移")
    return target, receipt, generation_generalization_sha256_bytes(payload)


__all__ = [
    "ARTIFACT_KIND",
    "DEFAULT_RECEIPT_RELATIVE_PATH",
    "GenerationGeneralizationPublicStressRuntimeError",
    "run_and_publish_generation_generalization_public_stress",
]
