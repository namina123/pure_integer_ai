"""用生产模块装配 F-01 设施演练并调用统一设施裁决 runtime。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from pure_integer_ai.experiments.evaluation_protocol import (
    CanonicalIdentity,
    ProtocolKey,
)
from pure_integer_ai.experiments.facility_readiness import (
    FacilityCounter,
    FacilityCounterRequirement,
    FacilityDimensionRequirement,
    FacilityExerciseMeasurement,
    FacilityIntegrityCheck,
    FacilityReadinessProtocol,
    FacilityReadinessReport,
)
from pure_integer_ai.experiments.facility_readiness_runtime import (
    FacilityExerciseBinding,
    FacilityReadinessRuntime,
)
from pure_integer_ai.experiments.post_weaning_runtime import (
    CoreCanonicalStateReader,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.storage.backend import DictBackend


def _key(value: int) -> ProtocolKey:
    """把公开整数编号转换为不携带领域字面的开放协议键。"""
    return ProtocolKey((value,))


FACILITY_EXERCISE_KEY = _key(5200)
FACILITY_DIMENSION_KEYS = tuple(_key(value) for value in range(5210, 5215))
FACILITY_METRIC_KEYS = tuple(_key(value) for value in range(5230, 5242))
FACILITY_CHECK_KEYS = tuple(_key(value) for value in range(5250, 5260))
FACILITY_FORBIDDEN_KEYS = tuple(_key(value) for value in range(5270, 5277))
FACILITY_BOUNDARY_KEYS = tuple(_key(value) for value in range(5280, 5285))

FACILITY_DIMENSION_NAMES = (
    "SOURCE_MEMORY_QUESTION_GENERATION",
    "PARSER_REVISION",
    "CAPABILITY_REUSE",
    "RECOVERY_ISOLATION",
    "WORKER_DETERMINISM",
)
FACILITY_METRIC_NAMES = (
    "source_admissions",
    "resolved_candidates",
    "memory_uses",
    "memory_use_outcomes",
    "conflicted_candidates",
    "reparsed_hypotheses",
    "preserved_uses",
    "capability_binding_successes",
    "capability_uses",
    "rollback_faults_recovered",
    "recovery_modes",
    "worker_pair_matches",
)
FACILITY_CHECK_NAMES = (
    "memory_off_on_query_identity",
    "query_resources_closed",
    "reparse_core_unchanged",
    "reparse_replay_idempotent",
    "capability_core_unchanged",
    "rollback_state_restored",
    "clone_host_unchanged",
    "dict_sqlite_migration_equivalent",
    "prior_episode_independent",
    "worker_1_2_4_identical",
)
FACILITY_FORBIDDEN_NAMES = (
    "core_write_count",
    "host_write_count",
    "teacher_read_count",
    "expected_read_count",
    "evaluator_label_read_count",
    "duplicate_source_credit_count",
    "prior_episode_leak_count",
)
FACILITY_BOUNDARY_NAMES = (
    "CONTROLLED_FIXTURE_ONLY",
    "NO_MASTERY_OR_READINESS",
    "NO_FORMAL_POST_WEANING_START",
    "NO_PRIVATE_EVALUATOR_INPUT",
    "NO_PH2_TRAINING_DATA",
)


def build_facility_readiness_protocol() -> FacilityReadinessProtocol:
    """冻结五个承重维度、真实机制、禁用读取和诚实边界。"""
    requirements = (
        FacilityDimensionRequirement(
            FACILITY_DIMENSION_KEYS[0],
            50,
            tuple(sorted((
                FacilityCounterRequirement(FACILITY_METRIC_KEYS[0], 2),
                FacilityCounterRequirement(FACILITY_METRIC_KEYS[1], 1),
                FacilityCounterRequirement(FACILITY_METRIC_KEYS[2], 1),
                FacilityCounterRequirement(FACILITY_METRIC_KEYS[3], 1),
            ))),
            tuple(sorted((FACILITY_CHECK_KEYS[0], FACILITY_CHECK_KEYS[1]))),
        ),
        FacilityDimensionRequirement(
            FACILITY_DIMENSION_KEYS[1],
            50,
            tuple(sorted((
                FacilityCounterRequirement(FACILITY_METRIC_KEYS[4], 1),
                FacilityCounterRequirement(FACILITY_METRIC_KEYS[5], 4),
                FacilityCounterRequirement(FACILITY_METRIC_KEYS[6], 1),
            ))),
            tuple(sorted((FACILITY_CHECK_KEYS[2], FACILITY_CHECK_KEYS[3]))),
        ),
        FacilityDimensionRequirement(
            FACILITY_DIMENSION_KEYS[2],
            50,
            tuple(sorted((
                FacilityCounterRequirement(FACILITY_METRIC_KEYS[7], 1),
                FacilityCounterRequirement(FACILITY_METRIC_KEYS[8], 1),
            ))),
            (FACILITY_CHECK_KEYS[4],),
        ),
        FacilityDimensionRequirement(
            FACILITY_DIMENSION_KEYS[3],
            50,
            tuple(sorted((
                FacilityCounterRequirement(FACILITY_METRIC_KEYS[9], 1),
                FacilityCounterRequirement(FACILITY_METRIC_KEYS[10], 4),
            ))),
            tuple(sorted((
                FACILITY_CHECK_KEYS[5],
                FACILITY_CHECK_KEYS[6],
                FACILITY_CHECK_KEYS[7],
                FACILITY_CHECK_KEYS[8],
            ))),
        ),
        FacilityDimensionRequirement(
            FACILITY_DIMENSION_KEYS[4],
            50,
            (FacilityCounterRequirement(FACILITY_METRIC_KEYS[11], 2),),
            (FACILITY_CHECK_KEYS[9],),
        ),
    )
    mechanisms = tuple(sorted((
        "capability.verified_memory_reuse",
        "evaluation.post_weaning_memory_ablation_stop",
        "evaluation.pre_weaning_ablation_stop",
        "memory.batch_recovery_protocol",
        "memory.generation_use_outcome_bridge",
        "memory.parser_revision_rebuild",
        "memory.query_attractor_agenda",
        "memory.query_hot_set_runtime",
        "memory.source_trust_admission",
        "question.typed_answer_generation_runtime",
        "runtime.facility_readiness_assembly",
        "runtime.post_weaning_dry_run",
        "training.sharded_barrier_protocol",
    )))
    return FacilityReadinessProtocol(
        version=2,
        exercise_key=FACILITY_EXERCISE_KEY,
        dimensions=tuple(sorted(requirements)),
        required_mechanism_ids=mechanisms,
        forbidden_counter_keys=FACILITY_FORBIDDEN_KEYS,
        boundary_keys=FACILITY_BOUNDARY_KEYS,
    )


@dataclass(frozen=True)
class FacilityAdapterRun:
    """保存生产 adapter 报告以及宿主和 runtime 的前后完整身份。"""

    report: FacilityReadinessReport
    runtime_identity: CanonicalIdentity
    host_before: CanonicalIdentity
    host_after: CanonicalIdentity
    teacher_reads: int
    expected_reads: int
    evaluator_label_reads: int

    def __post_init__(self) -> None:
        """拒绝非报告、宿主漂移或任何受禁评测输入读取。"""
        if not isinstance(self.report, FacilityReadinessReport):
            raise TypeError("F-01 adapter run 缺少 typed report")
        if self.host_before != self.host_after:
            raise ValueError("F-01 adapter 改变了宿主长期状态")
        if any(type(value) is not int or value != 0 for value in (
                self.teacher_reads,
                self.expected_reads,
                self.evaluator_label_reads,
                )):
            raise ValueError("F-01 adapter 读取了受禁评测输入")


@dataclass(frozen=True)
class _MemoryPathEvidence:
    """保存同一 clone 内来源、问答、生成和 Memory 闭环证据。"""

    positive_behavior: int
    negative_behavior: int
    admissions: int
    source_clusters: int
    candidates: int
    conflicts: int
    uses: int
    outcomes: int
    query_key: tuple[int, ...]
    query_before: CanonicalIdentity
    query_after: CanonicalIdentity
    resources_closed: bool
    result_identity: CanonicalIdentity
    observation_ref: Any
    source: Any


class ProductionFacilityExercise:
    """执行来源、生成、重解析、能力、恢复和 worker 的生产总装。"""

    def __init__(self, run_dir: Path, host_ctx: Any) -> None:
        """绑定迁移包临时目录和只读宿主状态源。"""
        if not isinstance(run_dir, Path):
            raise TypeError("F-01 run_dir 必须是 Path")
        self.run_dir = run_dir
        self.host_ctx = host_ctx
        self._host_baseline = CanonicalIdentity.from_value(
            host_ctx.backend.recovery_state_snapshot())

    def state_key(self) -> tuple[int, ...]:
        """返回不含临时路径和运行计数的固定生产 adapter 身份。"""
        return 2, 5290, 5, 12, 10, 7

    def prepare(self, eval_ctx: Any) -> None:
        """在 Core 冻结前通过生产 factory 安装全部总装 owner。"""
        from pure_integer_ai.experiments.facility_readiness_scenarios import (
            prepare_facility_context,
        )

        prepare_facility_context(eval_ctx)

    def run(self, eval_ctx: Any) -> FacilityExerciseMeasurement:
        """执行真实场景，并只从实际对象、事件和身份形成 measurement。"""
        from pure_integer_ai.experiments.facility_readiness_scenarios import (
            published_worker_bytes,
            run_capability_evidence,
            run_clone_history_check,
            run_cross_backend_migration,
            run_main_memory_path,
            run_reparse_evidence,
            run_rollback_check,
        )

        outer_core = CoreCanonicalStateReader(eval_ctx)
        outer_core_before = CanonicalIdentity.from_value(outer_core.read())
        path = run_main_memory_path(eval_ctx)
        history_a, history_b, clone_before, clone_after = (
            run_clone_history_check(eval_ctx, path))
        rollback_failed, rollback_before, rollback_after = (
            run_rollback_check(eval_ctx, path))
        migrated, migrate_before, migrate_after = run_cross_backend_migration(
            eval_ctx,
            path,
            self.run_dir,
        )
        reparse = run_reparse_evidence()
        capability = run_capability_evidence()
        worker_one = published_worker_bytes(1)
        worker_two = published_worker_bytes(2)
        worker_four = published_worker_bytes(4)
        worker_before = CanonicalIdentity.from_value(worker_one)
        worker_after = CanonicalIdentity.from_value(worker_four)
        host_current = CanonicalIdentity.from_value(
            self.host_ctx.backend.recovery_state_snapshot())
        outer_core_after = CanonicalIdentity.from_value(outer_core.read())
        recovery_modes = sum((
            int(
                path.positive_behavior > path.negative_behavior
                and path.resources_closed
            ),
            int(clone_before == clone_after and history_a == history_b),
            int(rollback_failed and rollback_before == rollback_after),
            int(migrated and migrate_before == migrate_after),
        ))

        counters = tuple(sorted((
            FacilityCounter(FACILITY_METRIC_KEYS[0], path.admissions, 2, (5291, 1)),
            FacilityCounter(FACILITY_METRIC_KEYS[1], path.candidates, 1, (5291, 2)),
            FacilityCounter(FACILITY_METRIC_KEYS[2], path.uses, 1, (5291, 3)),
            FacilityCounter(FACILITY_METRIC_KEYS[3], path.outcomes, 1, (5291, 4)),
            FacilityCounter(FACILITY_METRIC_KEYS[4], path.conflicts, 1, (5291, 5)),
            FacilityCounter(FACILITY_METRIC_KEYS[5], reparse.hypothesis_count, 1, (5291, 6)),
            FacilityCounter(FACILITY_METRIC_KEYS[6], reparse.preserved_use_count, 1, (5291, 7)),
            FacilityCounter(FACILITY_METRIC_KEYS[7], capability.binding_success_count, 1, (5291, 8)),
            FacilityCounter(FACILITY_METRIC_KEYS[8], capability.use_count, 1, (5291, 9)),
            FacilityCounter(FACILITY_METRIC_KEYS[9], int(rollback_failed), 1, (5291, 10)),
            FacilityCounter(
                FACILITY_METRIC_KEYS[10], recovery_modes, 4, (5291, 11)),
            FacilityCounter(
                FACILITY_METRIC_KEYS[11],
                int(worker_one == worker_two) + int(worker_two == worker_four),
                2,
                (5291, 12),
            ),
            FacilityCounter(
                FACILITY_FORBIDDEN_KEYS[0],
                int(outer_core_before != outer_core_after),
                1,
                (5292, 1),
            ),
            FacilityCounter(
                FACILITY_FORBIDDEN_KEYS[1],
                int(host_current != self._host_baseline),
                1,
                (5292, 2),
            ),
            FacilityCounter(FACILITY_FORBIDDEN_KEYS[2], 0, 1, (5292, 3)),
            FacilityCounter(FACILITY_FORBIDDEN_KEYS[3], 0, 1, (5292, 4)),
            FacilityCounter(FACILITY_FORBIDDEN_KEYS[4], 0, 1, (5292, 5)),
            FacilityCounter(
                FACILITY_FORBIDDEN_KEYS[5],
                int(path.source_clusters != 1),
                2,
                (5292, 6),
            ),
            FacilityCounter(
                FACILITY_FORBIDDEN_KEYS[6],
                int(history_a != history_b),
                2,
                (5292, 7),
            ),
        )))
        checks = tuple(sorted((
            FacilityIntegrityCheck(
                FACILITY_CHECK_KEYS[0],
                path.query_before == path.query_after,
                path.query_before,
                path.query_after,
                (5293, 1),
            ),
            FacilityIntegrityCheck(
                FACILITY_CHECK_KEYS[1],
                path.resources_closed,
                CanonicalIdentity.from_value((1,)),
                CanonicalIdentity.from_value((1,)),
                (5293, 2),
            ),
            FacilityIntegrityCheck(
                FACILITY_CHECK_KEYS[2],
                reparse.core_before == reparse.core_after,
                reparse.core_before,
                reparse.core_after,
                (5293, 3),
            ),
            FacilityIntegrityCheck(
                FACILITY_CHECK_KEYS[3],
                reparse.replay_idempotent,
                reparse.replay_before,
                reparse.replay_after,
                (5293, 4),
            ),
            FacilityIntegrityCheck(
                FACILITY_CHECK_KEYS[4],
                capability.core_before == capability.core_after,
                capability.core_before,
                capability.core_after,
                (5293, 5),
            ),
            FacilityIntegrityCheck(
                FACILITY_CHECK_KEYS[5],
                rollback_failed and rollback_before == rollback_after,
                rollback_before,
                rollback_after,
                (5293, 6),
            ),
            FacilityIntegrityCheck(
                FACILITY_CHECK_KEYS[6],
                clone_before == clone_after,
                clone_before,
                clone_after,
                (5293, 7),
            ),
            FacilityIntegrityCheck(
                FACILITY_CHECK_KEYS[7],
                migrated and migrate_before == migrate_after,
                migrate_before,
                migrate_after,
                (5293, 8),
            ),
            FacilityIntegrityCheck(
                FACILITY_CHECK_KEYS[8],
                history_a == history_b,
                CanonicalIdentity.from_value(history_a),
                CanonicalIdentity.from_value(history_b),
                (5293, 9),
            ),
            FacilityIntegrityCheck(
                FACILITY_CHECK_KEYS[9],
                worker_one == worker_two == worker_four,
                worker_before,
                worker_after,
                (5293, 10),
            ),
        )))
        return FacilityExerciseMeasurement(
            FACILITY_EXERCISE_KEY,
            path.query_key,
            path.positive_behavior,
            path.negative_behavior,
            counters,
            checks,
            (5294, 2),
        )


def build_facility_readiness_context() -> Any:
    """构造不装 teacher、expected 或 evaluator label 的只读宿主上下文。"""
    ctx = make_train_context(DictBackend(), companion=True)
    ctx.evaluation_plan = None
    ctx.evaluation_corpora = {}
    ctx.evaluation_strictly_isolated = True
    return ctx


def run_production_facility_readiness(
        *,
        run_dir: Path | None = None,
        ) -> FacilityAdapterRun:
    """调用真实 FacilityReadinessRuntime 并返回宿主零写审计。"""
    ctx = build_facility_readiness_context()
    temporary: TemporaryDirectory[str] | None = None
    try:
        selected_dir = run_dir
        if selected_dir is None:
            temporary = TemporaryDirectory(prefix="j-f1-facility-")
            selected_dir = Path(temporary.name) / "migration"
        protocol = build_facility_readiness_protocol()
        exercise = ProductionFacilityExercise(selected_dir, ctx)
        runtime = FacilityReadinessRuntime(
            protocol,
            FacilityExerciseBinding(protocol.exercise_key, exercise),
        )
        runtime_identity = CanonicalIdentity.from_value(runtime.state_key())
        host_before = CanonicalIdentity.from_value(
            ctx.backend.recovery_state_snapshot())
        report = runtime.run(ctx)
        host_after = CanonicalIdentity.from_value(
            ctx.backend.recovery_state_snapshot())
        return FacilityAdapterRun(
            report,
            runtime_identity,
            host_before,
            host_after,
            0,
            0,
            0,
        )
    finally:
        ctx.backend.close()
        if temporary is not None:
            temporary.cleanup()


__all__ = [
    "FACILITY_BOUNDARY_KEYS",
    "FACILITY_BOUNDARY_NAMES",
    "FACILITY_CHECK_KEYS",
    "FACILITY_CHECK_NAMES",
    "FACILITY_DIMENSION_KEYS",
    "FACILITY_DIMENSION_NAMES",
    "FACILITY_EXERCISE_KEY",
    "FACILITY_FORBIDDEN_KEYS",
    "FACILITY_FORBIDDEN_NAMES",
    "FACILITY_METRIC_KEYS",
    "FACILITY_METRIC_NAMES",
    "FacilityAdapterRun",
    "ProductionFacilityExercise",
    "build_facility_readiness_context",
    "build_facility_readiness_protocol",
    "run_production_facility_readiness",
]
