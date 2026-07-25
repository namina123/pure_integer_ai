"""L-05B2B typed language H2 的分维只读校准测试。"""
from __future__ import annotations

from dataclasses import replace
import importlib

import pytest

from pure_integer_ai.cognition.shared.identity import (
    minimal_instruction_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    document_scope,
    episode_scope,
)
from pure_integer_ai.cognition.shared.types import MODALITY_LANGUAGE
from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.config import gates
from pure_integer_ai.experiments.collection import CollectedItem
from pure_integer_ai.experiments.evaluation_protocol import EvaluationPlan
from pure_integer_ai.experiments.language_generation_h2 import (
    TypedLanguageH2Case,
    TypedLanguageH2Expectation,
    TypedLanguageH2Protocol,
    run_typed_language_h2,
)
from pure_integer_ai.experiments.language_generation_floor import (
    TypedLanguageFloorProtocol,
    TypedLanguageFloorRequirement,
    run_typed_language_floor,
)
from pure_integer_ai.experiments.language_semantic_course import (
    LanguageSemanticCourseDecision,
)
from pure_integer_ai.experiments.round_runtime import DefaultRoundRunner
from pure_integer_ai.experiments.formal_train import (
    FormalTrainConfig,
    formal_train,
)
from pure_integer_ai.experiments.generation_production_runtime import (
    ProductionGenerationInstallation,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.experiments.verification_orchestration import (
    VERDICT_REFUTE,
    VERDICT_SUPPORT,
)
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.training.stages import (
    STAGE3_REWARD,
    STAGE4_PROMOTE_WEAN,
)
from tests.test_l05b2b_semantic_course_runtime import (
    _SemanticNoRequestFactory,
    _dual_lesson,
    _fixture,
    _full_generation_runtime,
)
from tests.test_experiments import (
    MODE_RECORD,
    _teacher,
    register_recording_table,
)
from tests.test_v00_evaluation_protocol import (
    _assignment,
    _item,
    _protocol,
)

_EPISODE_HASHER = Hasher("formal_train.episode_scope.v1")
_BASE_STAGE4_TEST_OWNER = 970_100


class _FixedTypedRunner:
    """在 H2 沙箱中返回预先形成的只读 typed episode。"""

    def __init__(self, episode) -> None:
        self.episode = episode
        self.calls = 0

    def run_round_many(self, *_args):
        """记录调度次数并返回唯一 typed episode。"""
        self.calls += 1
        return (self.episode,)


class _RoundTypedRunner:
    """按 formal 逻辑轮次返回内容相同但事件身份互异的 typed episode。"""

    def __init__(self, episode) -> None:
        self.episode = episode
        self.episodes = []

    def run_round_many(self, *_args):
        """使用调用方 round id 建立本轮 episode，并保存实际提交顺序。"""
        current = replace(self.episode, round_id=_args[-1])
        self.episodes.append(current)
        return (current,)


class _UnreachedStage4Owner:
    """阶段3未通过时绝不应收到 episode。"""

    def apply(self, _episodes):
        """调用即说明 H2/floor 没有阻断阶段推进。"""
        raise AssertionError("阶段3未通过时不得执行 stage4")

    def state_key(self):
        """返回无状态测试 owner 的稳定配置。"""
        return (_BASE_STAGE4_TEST_OWNER,)


class _StageAwareNoRequestFactory(_SemanticNoRequestFactory):
    """为阶段3门控测试补齐不可达的 stage4 安装协议。"""

    def build_installation(self, ctx):
        """安装无请求 generation runtime 和不可达 stage4 owner。"""
        return ProductionGenerationInstallation(
            super().build(ctx),
            _UnreachedStage4Owner(),
        )

    def clone_for_evaluation(self):
        """返回配置相同且不共享对象的测试 factory。"""
        return _StageAwareNoRequestFactory(self.key)


def _typed_episode_and_item():
    """通过正式 semantic course 和 generation runtime 形成完整六维 episode。"""
    backend, ctx, semantic_runtime, mapper, _, payload, observed = _fixture()
    source = payload.source_ref
    mapper.decision = LanguageSemanticCourseDecision(
        minimal_instruction_identity((970_001, 1)),
        (970_001, 2),
        _dual_lesson(source, observed.occurrence_refs[0]),
    )
    episode_id = _EPISODE_HASHER.h63((STAGE3_REWARD, 1)) or 1
    formal_scope = episode_scope(
        episode_id,
        parent=document_scope(source),
    )
    item = CollectedItem(
        tokens=["甲", "乙"],
        raw_text="甲乙",
        source=source.source_kind,
        source_ref=source,
        modality=MODALITY_LANGUAGE,
    )
    prepared = semantic_runtime.process(
        ctx,
        item,
        replace(payload, scope_identity=formal_scope),
        observed,
    )
    runtime, _, alias, _, _, _ = _full_generation_runtime(prepared.request)
    ctx.language_generation_runtime = runtime
    result = DefaultRoundRunner().run_round_full(
        ctx,
        item,
        STAGE3_REWARD,
        1,
    )
    episode = replace(result.typed_episode, read_only=True)
    return backend, alias, episode, item


def _evaluation_plan(
        development: CollectedItem,
        *,
        held_out_item: CollectedItem | None = None,
        ):
    """构造只把目标样本放入 development 的完整五类 V-00 计划。"""
    protocol = _protocol(full_coverage=False)
    training = _item("训练输入", source_id=81)
    held_out = (
        _item("留出输入", source_id=83)
        if held_out_item is None else held_out_item)
    adversarial = _item("对抗输入", source_id=84)
    external = _item("外部输入", source_id=85)
    assignments = (
        _assignment(
            training,
            protocol=protocol,
            split=protocol.training_split,
            probe_kind=None,
            provenance=("h2", "train"),
        ),
        _assignment(
            development,
            protocol=protocol,
            split=protocol.development_split,
            probe_kind=protocol.required_adversarial_kinds[0],
            provenance=("h2", "development"),
            expected=("typed", "dimensions"),
        ),
        _assignment(
            held_out,
            protocol=protocol,
            split=protocol.held_out_split,
            probe_kind=protocol.required_adversarial_kinds[0],
            provenance=("h2", "held-out"),
        ),
        _assignment(
            adversarial,
            protocol=protocol,
            split=protocol.adversarial_split,
            probe_kind=protocol.required_adversarial_kinds[0],
            provenance=("h2", "adversarial"),
        ),
        _assignment(
            external,
            protocol=protocol,
            split=protocol.external_split,
            probe_kind=protocol.required_adversarial_kinds[0],
            provenance=("h2", "external"),
        ),
    )
    plan = EvaluationPlan(protocol, assignments)
    return plan, [training, development, held_out, adversarial, external]


def test_typed_language_h2_uses_development_and_keeps_dimensions_separate():
    """H2 只跑 development，逐维比较且不写宿主或产生综合 reward。"""
    backend, alias, episode, development = _typed_episode_and_item()
    h2_backend = DictBackend()
    try:
        plan, items = _evaluation_plan(development)
        ctx = make_train_context(h2_backend)
        ctx.evaluation_plan = plan
        ctx.evaluation_corpora = plan.partition(items).as_dict()
        ctx.evaluation_strictly_isolated = True
        expectations = tuple(
            TypedLanguageH2Expectation(
                item.dimension,
                item.verifier,
                item.applicability,
                item.verdict,
            )
            for item in episode.signals
        )
        h2_protocol = TypedLanguageH2Protocol(
            1,
            (TypedLanguageH2Case(
                plan.assignments[1].identity,
                expectations,
            ),),
        )
        runner = _FixedTypedRunner(episode)
        host_before = h2_backend.snapshot()

        report = run_typed_language_h2(ctx, runner, h2_protocol)

        assert runner.calls == 1
        assert report.split == plan.protocol.development_split
        assert report.measured is True
        assert report.complete is True
        assert len(report.cases) == 1
        assert len(report.cases[0].dimensions) == 6
        assert all(item.matched for item in report.cases[0].dimensions)
        assert h2_backend.snapshot() == host_before
        assert not hasattr(report, "score")
        assert not hasattr(report, "reward")
        assert not hasattr(report, "weights")

        first = expectations[0]
        opposite = (
            VERDICT_REFUTE
            if first.verdict == VERDICT_SUPPORT
            else VERDICT_SUPPORT
        )
        changed = replace(first, verdict=opposite)
        mismatched_protocol = TypedLanguageH2Protocol(
            2,
            (TypedLanguageH2Case(
                plan.assignments[1].identity,
                (changed, *expectations[1:]),
            ),),
        )
        mismatched = run_typed_language_h2(
            ctx,
            _FixedTypedRunner(episode),
            mismatched_protocol,
        )
        assert mismatched.complete is False
        assert mismatched.cases[0].dimensions[0].matched is False
        assert h2_backend.snapshot() == host_before
    finally:
        alias.close()
        backend.close()
        h2_backend.close()


def test_formal_train_typed_owner_never_calls_legacy_h2(monkeypatch, tmp_path):
    """正式 typed owner 只调分维 H2，教师在位也不得校准旧 JudgeWeights。"""
    development = _item("开发输入", source_id=92)
    plan, items = _evaluation_plan(development)
    backend = DictBackend()
    register_recording_table(backend)
    class _CompleteReport:
        """模拟已逐维通过的 typed H2 报告。"""

        complete = True

    sentinel = _CompleteReport()
    calls = []
    formal_module = importlib.import_module(
        "pure_integer_ai.experiments.formal_train")

    def typed_h2(ctx, runner, protocol):
        """记录正式 typed H2 调度并返回哨兵报告。"""
        calls.append((ctx, runner, protocol))
        return sentinel

    def forbidden_legacy_h2(*_args, **_kwargs):
        """typed owner 下调用旧标量 H2 即让测试失败。"""
        raise AssertionError("typed owner 不得调用 legacy H2")

    monkeypatch.setattr(formal_module, "run_typed_language_h2", typed_h2)
    monkeypatch.setattr(
        formal_module,
        "run_typed_language_floor",
        lambda *_args: sentinel,
    )
    monkeypatch.setattr(formal_module, "_h2_calibrate", forbidden_legacy_h2)
    monkeypatch.setattr(gates, "TRAINING_MODE", True)
    marker_protocol = object()

    class _NoEpisodeRunner:
        """让编排测试避开无关的 semantic mapper 内容构造。"""

        def run_round_many(self, *_args):
            """返回空结果；本测试只核验 H2 分派。"""
            return ()

    try:
        result = formal_train(
            FormalTrainConfig(
                run_dir=str(tmp_path),
                run_id="typed-h2-dispatch",
                rounds_per_stage=1,
                active_training_stages=(STAGE3_REWARD,),
                persist_graph_dump=False,
                evaluation_plan=plan,
                language_generation_runtime_factory=(
                    _StageAwareNoRequestFactory()),
                language_generation_h2_protocol=marker_protocol,
                language_generation_floor_protocol=marker_protocol,
            ),
            items,
            backend=backend,
            teacher=_teacher(backend, MODE_RECORD),
            runner=_NoEpisodeRunner(),
        )
        assert result.typed_language_h2_report is sentinel
        assert result.typed_language_floor_report is sentinel
        assert len(calls) == 1
        assert calls[0][2] is marker_protocol
    finally:
        backend.close()


@pytest.mark.parametrize("missing_protocol", ["h2", "floor"])
def test_formal_typed_stage3_rejects_missing_protocol(
        monkeypatch, tmp_path, missing_protocol):
    """正式 typed 阶段3缺任一分维协议时必须在训练前拒绝。"""
    development = _item("开发输入", source_id=93)
    plan, items = _evaluation_plan(development)
    backend = DictBackend()
    register_recording_table(backend)
    monkeypatch.setattr(gates, "TRAINING_MODE", True)
    marker_protocol = object()
    config = FormalTrainConfig(
        run_dir=str(tmp_path),
        run_id=f"typed-missing-{missing_protocol}",
        rounds_per_stage=1,
        active_training_stages=(STAGE3_REWARD,),
        persist_graph_dump=False,
        evaluation_plan=plan,
        language_generation_runtime_factory=_SemanticNoRequestFactory(),
        language_generation_h2_protocol=(
            None if missing_protocol == "h2" else marker_protocol),
        language_generation_floor_protocol=(
            None if missing_protocol == "floor" else marker_protocol),
    )
    try:
        with pytest.raises(ValueError, match=(
                "分维 H2" if missing_protocol == "h2" else "分维 floor")):
            formal_train(
                config,
                items,
                backend=backend,
                teacher=_teacher(backend, MODE_RECORD),
            )
    finally:
        backend.close()


@pytest.mark.parametrize("incomplete_report", ["h2", "floor"])
def test_formal_typed_incomplete_reports_block_training_progression(
        monkeypatch, tmp_path, incomplete_report):
    """H2 或 floor 未完整通过时不得让正式 typed 训练越过阶段3。"""
    development = _item("开发输入", source_id=94)
    plan, items = _evaluation_plan(development)
    backend = DictBackend()
    register_recording_table(backend)

    class _Report:
        """提供正式编排只需要的 complete 协议位。"""

        def __init__(self, complete):
            self.complete = complete

    complete = _Report(True)
    incomplete = _Report(False)
    formal_module = importlib.import_module(
        "pure_integer_ai.experiments.formal_train")
    monkeypatch.setattr(
        formal_module,
        "run_typed_language_h2",
        lambda *_args: incomplete if incomplete_report == "h2" else complete,
    )
    monkeypatch.setattr(
        formal_module,
        "run_typed_language_floor",
        lambda *_args: incomplete if incomplete_report == "floor" else complete,
    )
    monkeypatch.setattr(gates, "TRAINING_MODE", True)
    marker_protocol = object()

    class _NoEpisodeRunner:
        """隔离本测试与无关的 generation episode 内容。"""

        def run_round_many(self, *_args):
            """返回空批次，让测试只观察阶段推进。"""
            return ()

    try:
        run = lambda: formal_train(
            FormalTrainConfig(
                run_dir=str(tmp_path),
                run_id=f"typed-incomplete-{incomplete_report}",
                rounds_per_stage=1,
                active_training_stages=(
                    STAGE3_REWARD,
                    STAGE4_PROMOTE_WEAN,
                ),
                persist_graph_dump=False,
                evaluation_plan=plan,
                language_generation_runtime_factory=(
                    _StageAwareNoRequestFactory()),
                language_generation_h2_protocol=marker_protocol,
                language_generation_floor_protocol=marker_protocol,
            ),
            items,
            backend=backend,
            teacher=_teacher(backend, MODE_RECORD),
            runner=_NoEpisodeRunner(),
        )
        if incomplete_report == "h2":
            with pytest.raises(RuntimeError, match="H2 分维校准未通过"):
                run()
        else:
            result = run()
            assert result.typed_language_floor_report is incomplete
            assert result.stages_completed == []
            assert STAGE4_PROMOTE_WEAN not in result.stages_completed
    finally:
        backend.close()


def test_typed_stage4_fails_before_legacy_promotion(monkeypatch, tmp_path):
    """typed 候选生命周期未建前，阶段4不得扫描或晋升 legacy SHADOW edge。"""
    backend = DictBackend()
    register_recording_table(backend)
    formal_module = importlib.import_module(
        "pure_integer_ai.experiments.formal_train")

    def forbidden_legacy_promote(*_args, **_kwargs):
        """任何 legacy 晋升调用都说明 typed stage4 错误串线。"""
        raise AssertionError("typed stage4 不得调用 legacy promote")

    class _NoEpisodeRunner:
        """阶段4入口应在 round 调度前失败。"""

        def run_round_many(self, *_args):
            """若被调用则说明 fail-closed 边界过晚。"""
            raise AssertionError("typed stage4 不得先执行 round")

    monkeypatch.setattr(
        formal_module, "_promote_eligible", forbidden_legacy_promote)
    monkeypatch.setattr(gates, "TRAINING_MODE", True)
    try:
        with pytest.raises(
                RuntimeError,
                match="typed language 阶段4.*禁止回退 legacy promote"):
            formal_train(
                FormalTrainConfig(
                    run_dir=str(tmp_path),
                    run_id="typed-stage4-fail-closed",
                    rounds_per_stage=1,
                    active_training_stages=(STAGE4_PROMOTE_WEAN,),
                    persist_graph_dump=False,
                    language_generation_runtime_factory=(
                        _SemanticNoRequestFactory()),
                ),
                [_item("阶段四输入", source_id=95)],
                backend=backend,
                teacher=_teacher(backend, MODE_RECORD),
                runner=_NoEpisodeRunner(),
            )
    finally:
        backend.close()


@pytest.mark.parametrize(
    "report_complete,round_count,expected_completed",
    ((True, 2, [STAGE4_PROMOTE_WEAN]), (False, 1, [])),
)
def test_formal_typed_stage4_consumes_current_typed_episodes(
        monkeypatch, tmp_path, report_complete, round_count,
        expected_completed):
    """正式阶段4只批量提交本阶段 episode，失败报告不得完成课程游标。"""
    episode_backend, alias, episode, item = _typed_episode_and_item()
    backend = DictBackend()
    register_recording_table(backend)
    formal_module = importlib.import_module(
        "pure_integer_ai.experiments.formal_train")

    class _Report:
        """表示 stage4 已形成完整 typed lifecycle 报告。"""

        changed_count = 1

        def __init__(self, complete):
            self.complete = complete

    report = _Report(report_complete)

    class _Stage4Owner:
        """记录 formal 提交的完整 typed episode 批次。"""

        def __init__(self):
            self.batches = []

        def apply(self, episodes):
            """保存本阶段批次并返回完整报告。"""
            self.batches.append(episodes)
            return report

        def state_key(self):
            """返回已处理批次数和 episode 数量。"""
            return tuple(len(batch) for batch in self.batches)

    owner = _Stage4Owner()

    class _Stage4Factory(_SemanticNoRequestFactory):
        """在同一次 installation 中提供 generation 和 stage4 owner。"""

        def build_installation(self, ctx):
            """复用合法 typed runtime，并附带当前 stage4 owner。"""
            return ProductionGenerationInstallation(
                super().build(ctx),
                owner,
            )

        def clone_for_evaluation(self):
            """本测试不运行评测，仍返回同配置独立 factory。"""
            return _Stage4Factory(self.key)

    def forbidden_legacy_promote(*_args, **_kwargs):
        """typed stage4 触达 legacy promote 即失败。"""
        raise AssertionError("typed stage4 不得调用 legacy promote")

    monkeypatch.setattr(
        formal_module,
        "_promote_eligible",
        forbidden_legacy_promote,
    )
    monkeypatch.setattr(
        formal_module,
        "_discover_and_recognize_lang_structures",
        forbidden_legacy_promote,
    )

    def forbidden_legacy_weaning(*_args, **_kwargs):
        """typed stage4 完成后不得进入旧标量断奶、floor 或退场模拟。"""
        raise AssertionError("typed stage4 不得调用 legacy weaning tail")

    for name in (
            "weaning_check",
            "_run_calibration_phase",
            "_run_simulated_offline_eval",
            "_measure_floor_pass",
            ):
        monkeypatch.setattr(formal_module, name, forbidden_legacy_weaning)
    completed_calls = []

    def record_completed(*args, **kwargs):
        """记录真正越过阶段门的游标提交。"""
        completed_calls.append((args, kwargs))

    monkeypatch.setattr(formal_module, "mark_completed", record_completed)
    monkeypatch.setattr(gates, "TRAINING_MODE", True)
    try:
        writable_episode = replace(episode, read_only=False)
        runner = _RoundTypedRunner(writable_episode)
        result = formal_train(
            FormalTrainConfig(
                run_dir=str(tmp_path),
                run_id=f"typed-stage4-dispatch-{int(report_complete)}",
                rounds_per_stage=round_count,
                active_training_stages=(STAGE4_PROMOTE_WEAN,),
                persist_graph_dump=False,
                calibrate_mode_b=True,
                simulate_offline_eval=True,
                language_generation_runtime_factory=_Stage4Factory(),
            ),
            [item],
            backend=backend,
            teacher=_teacher(backend, MODE_RECORD),
            runner=runner,
        )

        assert owner.batches == [tuple(runner.episodes)]
        assert len(owner.batches) == 1
        assert result.typed_language_stage4_report is report
        assert result.stages_completed == expected_completed
        assert len(completed_calls) == int(report_complete)
        assert result.weaning_ready is False
        assert result.weaning_blockers == (
            ["W-09_typed_weaning_protocol_missing"]
            if report_complete else []
        )
    finally:
        alias.close()
        episode_backend.close()
        backend.close()


def test_typed_language_floor_uses_held_out_and_keeps_per_dimension_rates():
    """typed floor 只跑 held-out，并按维度独立阈值形成合取门。"""
    backend, alias, episode, held_out = _typed_episode_and_item()
    floor_backend = DictBackend()
    try:
        development = _item("开发输入", source_id=96)
        plan, items = _evaluation_plan(
            development,
            held_out_item=held_out,
        )
        ctx = make_train_context(floor_backend)
        ctx.evaluation_plan = plan
        ctx.evaluation_corpora = plan.partition(items).as_dict()
        ctx.evaluation_strictly_isolated = True
        expectations = tuple(
            TypedLanguageH2Expectation(
                item.dimension,
                item.verifier,
                item.applicability,
                item.verdict,
            )
            for item in episode.signals
        )
        protocol = TypedLanguageFloorProtocol(
            1,
            (TypedLanguageH2Case(plan.assignments[2].identity, expectations),),
            tuple(TypedLanguageFloorRequirement(
                item.dimension,
                item.verifier,
                1000,
            ) for item in expectations),
        )
        runner = _FixedTypedRunner(episode)
        before = floor_backend.snapshot()

        report = run_typed_language_floor(ctx, runner, protocol)

        assert runner.calls == 1
        assert report.split == plan.protocol.held_out_split
        assert report.complete is True
        assert len(report.dimensions) == 6
        assert all(item.match_permille == 1000 for item in report.dimensions)
        assert floor_backend.snapshot() == before
        assert not hasattr(report, "score")
        assert not hasattr(report, "reward")
    finally:
        alias.close()
        backend.close()
        floor_backend.close()
