"""W-00 版本化课程 hard gate、恢复和 formal_train 接线验收。"""
from __future__ import annotations

from dataclasses import replace

import pytest

from pure_integer_ai.experiments.curriculum_mastery_runtime import (
    CurriculumArtifactVersions,
    CurriculumEvaluatorDriftError,
    CurriculumGateCheck,
    CurriculumHardGateError,
    CurriculumMasteryProtocol,
    CurriculumMasteryRuntime,
    CurriculumStageEvaluation,
    CurriculumStageEvaluationRequest,
    CurriculumStagePlan,
)
from pure_integer_ai.experiments.formal_train import (
    FormalTrainConfig,
    formal_train,
)
from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.curriculum_mastery import (
    CURRICULUM_STAGE_REPORT_PART_TABLE,
    CURRICULUM_STAGE_REPORT_TABLE,
    FAULT_CURRICULUM_REPORT_AFTER_HEADER,
    FAULT_CURRICULUM_REPORT_AFTER_PARTS,
)
from pure_integer_ai.training.stages import (
    STAGE1_SKELETON,
    STAGE2_CAUSES_ABS,
    STAGE3_REWARD,
    stage_floor_overrides,
)


_STAGE_KEYS = ((101,), (102,), (103,))


class _Evaluator:
    """按注入结果返回稳定检查的测试评测器。"""

    def __init__(self, outcomes=None, *, state_key=(7001,)) -> None:
        """保存每阶段结论和可变版本键，便于对抗漂移。"""
        self.outcomes = dict(outcomes or {})
        self.key = state_key
        self.requests: list[CurriculumStageEvaluationRequest] = []

    def state_key(self) -> tuple[int, ...]:
        """返回当前评测器版本键。"""
        return self.key

    def evaluate(
            self,
            request: CurriculumStageEvaluationRequest,
            ) -> CurriculumStageEvaluation:
        """按阶段注入结论，并留下不依赖调用次数的证据。"""
        self.requests.append(request)
        passed = self.outcomes.get(request.stage_key, True)
        return CurriculumStageEvaluation((CurriculumGateCheck(
            (7100, *request.stage_key),
            passed,
            (7200, int(passed), *request.stage_key),
        ),))


class _RaiseAt:
    """在指定持久化边界抛错一次。"""

    def __init__(self, point: int) -> None:
        """记录目标故障点。"""
        self.point = point
        self.raised = False

    def hit(self, point: int, context: dict[str, int]) -> None:
        """首次命中目标边界时中断。"""
        if point == self.point and not self.raised:
            self.raised = True
            raise RuntimeError("injected curriculum report fault")


class _CountingDictBackend(DictBackend):
    """统计 mastery 读取次数的内存后端。"""

    def __init__(self) -> None:
        """初始化后端和读取计数。"""
        super().__init__()
        self.select_calls = 0

    def select(self, *args, **kwargs):
        """累计读取次数后转发到标准后端。"""
        self.select_calls += 1
        return super().select(*args, **kwargs)


def _versions(value: int = 1) -> CurriculumArtifactVersions:
    """构造四维完整测试版本。"""
    return CurriculumArtifactVersions(
        (11, value),
        (12, value),
        (13, value),
        (14, value),
    )


def _runtime(
        backend,
        evaluator: _Evaluator,
        *,
        versions: CurriculumArtifactVersions | None = None,
        stages: tuple[tuple[int, ...], ...] = _STAGE_KEYS,
        skippable: frozenset[tuple[int, ...]] | None = None,
        ) -> CurriculumMasteryRuntime:
    """注册表并构造测试 mastery 运行时。"""
    bootstrap(backend)
    return CurriculumMasteryRuntime(
        backend,
        CurriculumStagePlan(
            stages,
            frozenset(stages) if skippable is None else skippable,
        ),
        versions or _versions(),
        evaluator,
    )


@pytest.mark.parametrize("backend_factory", (DictBackend, SQLiteBackend))
def test_stage_failure_blocks_later_stage_on_both_backends(backend_factory):
    """第二阶段失败必须留下报告，并阻断第三阶段。"""
    backend = backend_factory()
    evaluator = _Evaluator({_STAGE_KEYS[1]: False})
    runtime = _runtime(backend, evaluator)
    try:
        first = runtime.evaluate_and_record(_STAGE_KEYS[0], evidence=None)
        second = runtime.evaluate_and_record(_STAGE_KEYS[1], evidence=None)

        assert first.passed == 1
        assert second.passed == 0
        assert runtime.is_mastered(_STAGE_KEYS[0]) is True
        assert runtime.is_mastered(_STAGE_KEYS[1]) is False
        with pytest.raises(CurriculumHardGateError, match="前置阶段"):
            runtime.evaluate_and_record(_STAGE_KEYS[2], evidence=None)
        assert len(runtime.store.all_reports()) == 2
    finally:
        backend.close()


@pytest.mark.parametrize(
    "changed",
    (
        {"data_key": (11, 2)},
        {"code_key": (12, 2)},
        {"primitive_key": (13, 2)},
        {"curriculum_key": (14, 2)},
    ),
)
def test_each_artifact_version_change_invalidates_mastery(changed):
    """数据、代码、原语或课程任一版本变化都不得复用旧 mastery。"""
    backend = DictBackend()
    evaluator = _Evaluator()
    original = _versions()
    runtime = _runtime(
        backend,
        evaluator,
        versions=original,
        stages=(_STAGE_KEYS[0],),
    )
    runtime.evaluate_and_record(_STAGE_KEYS[0], evidence=None)

    changed_runtime = _runtime(
        backend,
        evaluator,
        versions=replace(original, **changed),
        stages=(_STAGE_KEYS[0],),
    )

    assert changed_runtime.is_mastered(_STAGE_KEYS[0]) is False
    assert changed_runtime.prepare((_STAGE_KEYS[0],)) == (_STAGE_KEYS[0],)


def test_evaluator_state_key_drift_fails_closed():
    """绑定后的评测器版本漂移必须报错，不能降级成未命中。"""
    evaluator = _Evaluator()
    runtime = _runtime(
        DictBackend(),
        evaluator,
        stages=(_STAGE_KEYS[0],),
    )
    runtime.evaluate_and_record(_STAGE_KEYS[0], evidence=None)

    evaluator.key = (7002,)

    with pytest.raises(CurriculumEvaluatorDriftError, match="漂移"):
        runtime.is_mastered(_STAGE_KEYS[0])


def test_upstream_reevaluation_invalidates_downstream_report():
    """上游新报告即使仍通过，也必须使绑定旧前置报告的下游失效。"""
    evaluator = _Evaluator()
    runtime = _runtime(DictBackend(), evaluator, stages=_STAGE_KEYS[:2])
    runtime.evaluate_and_record(_STAGE_KEYS[0], evidence=None)
    runtime.evaluate_and_record(_STAGE_KEYS[1], evidence=None)
    assert runtime.is_mastered(_STAGE_KEYS[1]) is True

    runtime.evaluate_and_record(_STAGE_KEYS[0], evidence=None)

    assert runtime.is_mastered(_STAGE_KEYS[0]) is True
    assert runtime.is_mastered(_STAGE_KEYS[1]) is False
    assert runtime.prepare(_STAGE_KEYS[:2]) == (_STAGE_KEYS[1],)


def test_prepare_rejects_missing_stage_inside_pending_suffix():
    """待重跑后缀不能越过中间阶段，避免训练到一半才发现前置失效。"""
    runtime = _runtime(
        DictBackend(),
        _Evaluator(),
        skippable=frozenset({_STAGE_KEYS[1]}),
    )
    runtime.evaluate_and_record(_STAGE_KEYS[0], evidence=None)
    runtime.evaluate_and_record(_STAGE_KEYS[1], evidence=None)

    with pytest.raises(CurriculumHardGateError, match="不得跳过"):
        runtime.prepare((_STAGE_KEYS[0], _STAGE_KEYS[2]))


def test_current_report_reads_each_prerequisite_once():
    """长严格课程的单次 mastery 判定必须线性读取，不能重复递归。"""
    stages = tuple((200 + index,) for index in range(8))
    backend = _CountingDictBackend()
    runtime = _runtime(backend, _Evaluator(), stages=stages)
    for stage_key in stages:
        runtime.evaluate_and_record(stage_key, evidence=None)
    backend.select_calls = 0

    assert runtime.is_mastered(stages[-1]) is True
    assert backend.select_calls == len(stages) * 2


def test_parts_fault_keeps_old_view_and_retry_reuses_orphan_parts():
    """parts 后故障不发布报告，同证据重试补 header。"""
    backend = DictBackend()
    runtime = _runtime(
        backend,
        _Evaluator(),
        stages=(_STAGE_KEYS[0],),
    )
    injector = _RaiseAt(FAULT_CURRICULUM_REPORT_AFTER_PARTS)

    with pytest.raises(RuntimeError, match="injected"):
        runtime.evaluate_and_record(
            _STAGE_KEYS[0],
            evidence=None,
            fault_injector=injector,
        )

    orphan_count = backend.count(CURRICULUM_STAGE_REPORT_PART_TABLE)
    assert orphan_count > 0
    assert backend.count(CURRICULUM_STAGE_REPORT_TABLE) == 0
    assert runtime.is_mastered(_STAGE_KEYS[0]) is False

    report = runtime.evaluate_and_record(_STAGE_KEYS[0], evidence=None)

    assert report.report_seq == 1
    assert backend.count(CURRICULUM_STAGE_REPORT_PART_TABLE) == orphan_count
    assert backend.count(CURRICULUM_STAGE_REPORT_TABLE) == 1
    assert runtime.is_mastered(_STAGE_KEYS[0]) is True


def test_header_fault_leaves_complete_new_view():
    """header 后故障已经越过唯一可见点，重启读取必须看到完整报告。"""
    backend = DictBackend()
    runtime = _runtime(
        backend,
        _Evaluator(),
        stages=(_STAGE_KEYS[0],),
    )

    with pytest.raises(RuntimeError, match="injected"):
        runtime.evaluate_and_record(
            _STAGE_KEYS[0],
            evidence=None,
            fault_injector=_RaiseAt(FAULT_CURRICULUM_REPORT_AFTER_HEADER),
        )

    assert backend.count(CURRICULUM_STAGE_REPORT_TABLE) == 1
    assert runtime.is_mastered(_STAGE_KEYS[0]) is True


def test_sqlite_file_restart_restores_mastery(tmp_path):
    """文件 SQLite 关闭重开后必须恢复同一正式报告。"""
    path = tmp_path / "curriculum.sqlite"
    evaluator = _Evaluator()
    first_backend = SQLiteBackend(str(path))
    first_runtime = _runtime(
        first_backend,
        evaluator,
        stages=(_STAGE_KEYS[0],),
    )
    report = first_runtime.evaluate_and_record(_STAGE_KEYS[0], evidence=None)
    first_backend.commit()
    first_backend.close()

    second_backend = SQLiteBackend(str(path))
    try:
        second_runtime = _runtime(
            second_backend,
            _Evaluator(),
            stages=(_STAGE_KEYS[0],),
        )
        restored = second_runtime.current_report(_STAGE_KEYS[0])
        assert restored is not None
        assert restored.report_hash == report.report_hash
    finally:
        second_backend.close()


def test_no_direct_mastered_setter_and_failed_evidence_is_not_mastery():
    """运行时不暴露 set_mastered，失败检查也不能形成 mastery。"""
    runtime = _runtime(
        DictBackend(),
        _Evaluator({_STAGE_KEYS[0]: False}),
        stages=(_STAGE_KEYS[0],),
    )

    report = runtime.evaluate_and_record(_STAGE_KEYS[0], evidence=None)

    assert not hasattr(runtime, "set_mastered")
    assert not hasattr(runtime.store, "set_mastered")
    assert report.passed == 0
    assert runtime.is_mastered(_STAGE_KEYS[0]) is False


def _formal_protocol(evaluator: _Evaluator) -> CurriculumMasteryProtocol:
    """构造 legacy 三阶段到开放课程键的一一映射。"""
    return CurriculumMasteryProtocol(
        CurriculumStagePlan(
            _STAGE_KEYS,
            frozenset({_STAGE_KEYS[0], _STAGE_KEYS[1]}),
        ),
        _versions(),
        evaluator,
        (
            (STAGE1_SKELETON, _STAGE_KEYS[0]),
            (STAGE2_CAUSES_ABS, _STAGE_KEYS[1]),
            (STAGE3_REWARD, _STAGE_KEYS[2]),
        ),
        (7300,),
    )


def test_formal_train_blocks_missing_prerequisite_before_graph_bootstrap(tmp_path):
    """只请求第二阶段且缺第一阶段 mastery 时，不得先写训练图。"""
    backend = DictBackend()
    config = FormalTrainConfig(
        run_dir=str(tmp_path / "runs"),
        run_id="missing-prerequisite",
        rounds_per_stage=1,
        active_training_stages=(STAGE2_CAUSES_ABS,),
        curriculum_active_relations=frozenset(),
        curriculum_boot_relations=frozenset(),
        persist_graph_dump=False,
        curriculum_mastery_protocol=_formal_protocol(_Evaluator()),
    )

    with pytest.raises(CurriculumHardGateError, match="前置阶段"):
        formal_train(config, [], backend=backend)

    assert backend.count("concept_node") == 0
    assert backend.count("edge") == 0
    assert backend.count(CURRICULUM_STAGE_REPORT_TABLE) == 0


def test_formal_train_failed_gate_records_report_without_completed(tmp_path):
    """宿主 gate 失败必须写失败报告，且不能进入 stages_completed。"""
    backend = DictBackend()
    config = FormalTrainConfig(
        run_dir=str(tmp_path / "runs"),
        run_id="failed-stage",
        rounds_per_stage=1,
        active_training_stages=(STAGE1_SKELETON,),
        curriculum_active_relations=frozenset(),
        curriculum_boot_relations=frozenset(),
        persist_graph_dump=False,
        curriculum_mastery_protocol=_formal_protocol(_Evaluator()),
    )

    with stage_floor_overrides({"FLOOR_GRAPH_SIZE_S1": 1_000_000_000}):
        result = formal_train(config, [], backend=backend)

    assert result.stages_requested == [STAGE1_SKELETON]
    assert result.stages_completed == []
    assert len(result.curriculum_stage_reports) == 1
    assert result.curriculum_stage_reports[0].passed == 0


def test_formal_train_dump_resume_skips_same_version_mastered_stage(tmp_path):
    """真实终 dump/load 后，只跳过同版本且声明 skippable 的 mastered 阶段。"""
    run_dir = str(tmp_path / "runs")
    first_backend = DictBackend()
    first_config = FormalTrainConfig(
        run_dir=run_dir,
        run_id="mastery-base",
        rounds_per_stage=1,
        active_training_stages=(STAGE1_SKELETON,),
        curriculum_active_relations=frozenset(),
        curriculum_boot_relations=frozenset(),
        curriculum_mastery_protocol=_formal_protocol(_Evaluator()),
    )
    with stage_floor_overrides({"FLOOR_GRAPH_SIZE_S1": 0}):
        first = formal_train(first_config, [], backend=first_backend)
    assert first.stages_completed == [STAGE1_SKELETON]

    resumed_backend = DictBackend()
    resumed_config = FormalTrainConfig(
        run_dir=run_dir,
        run_id="mastery-resumed",
        resume=True,
        base_run_id="mastery-base",
        rounds_per_stage=1,
        active_training_stages=(STAGE1_SKELETON,),
        curriculum_active_relations=frozenset(),
        curriculum_boot_relations=frozenset(),
        persist_graph_dump=False,
        curriculum_mastery_protocol=_formal_protocol(_Evaluator()),
    )
    with stage_floor_overrides({"FLOOR_GRAPH_SIZE_S1": 0}):
        resumed = formal_train(resumed_config, [], backend=resumed_backend)

    assert resumed.stages_completed == []
    assert resumed.stages_skipped == [STAGE1_SKELETON]
    assert resumed.curriculum_stage_reports == []
    restored_runtime = resumed_config.curriculum_mastery_protocol.bind(
        resumed_backend)
    assert restored_runtime.is_mastered(_STAGE_KEYS[0]) is True
