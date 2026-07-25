"""R-09 多维 verifier 编排、隔离和 legacy runtime 接线测试。"""
from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from pure_integer_ai.cognition.shared.types import (
    DOMAIN_TEXT,
    LANG_ZH,
    MODALITY_LANGUAGE,
    Episode,
    InputPayload,
    Segment,
)
from pure_integer_ai.config import gates
from pure_integer_ai.experiments.collection import (
    COLLECT_CAUSES,
    CollectedItem,
)
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.experiments.evaluation_isolation import (
    isolated_evaluation,
)
from pure_integer_ai.experiments.formal_train import (
    FormalTrainConfig,
    formal_train,
)
from pure_integer_ai.experiments.round_runtime import (
    DefaultRoundRunner,
    MultipleVerificationEpisodesError,
    RoundResult,
)
from pure_integer_ai.experiments.verification_dispatch import (
    VERIFY_ROUTE_COMPARISON,
    VERIFY_ROUTE_EXISTENTIAL,
    VERIFY_ROUTE_NUMERIC,
    VERIFY_ROUTE_OCCURRENCE_ORDER,
    VERIFY_ROUTE_UNIVERSAL,
)
from pure_integer_ai.experiments.verification_orchestration import (
    APPLICABILITY_APPLICABLE,
    APPLICABILITY_NOT_APPLICABLE,
    MultiVerifierOrchestrator,
    VERDICT_CONFLICTED,
    VERDICT_REFUTE,
    VERDICT_SUPPORT,
    VERDICT_UNKNOWN,
    VerificationEffect,
    VerificationEvaluation,
    VerificationReport,
    VerificationResult,
    VerifierRegistration,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import SOURCE_BARE_TEXT
from pure_integer_ai.training.stages import (
    STAGE3_REWARD,
    stage_floor_overrides,
)


def _registration(
        dimension: int,
        verdict: int,
        calls: list[int],
        ) -> VerifierRegistration:
    """构造会记录执行顺序的无副作用 verifier。"""
    return VerifierRegistration(
        dimension=ProtocolKey((dimension,)),
        verifier=ProtocolKey((1, dimension)),
        applies=lambda _request: True,
        evaluate=lambda _request: (
            calls.append(dimension)
            or VerificationEvaluation(
                verdict=verdict,
                claim_keys=((dimension, 0),),
                detail=(dimension,),
            )
        ),
    )


def test_registration_order_does_not_change_execution_or_result_set():
    """注册输入顺序变化后仍按完整键执行全部适用 verifier。"""
    calls_a: list[int] = []
    calls_b: list[int] = []
    report_a = MultiVerifierOrchestrator().run(
        None,
        (
            _registration(20, VERDICT_REFUTE, calls_a),
            _registration(10, VERDICT_SUPPORT, calls_a),
        ),
        read_only=True,
    )
    report_b = MultiVerifierOrchestrator().run(
        None,
        (
            _registration(10, VERDICT_SUPPORT, calls_b),
            _registration(20, VERDICT_REFUTE, calls_b),
        ),
        read_only=True,
    )
    assert calls_a == [10, 20]
    assert calls_b == [10, 20]
    assert report_a.verdict_key() == report_b.verdict_key()
    assert [result.verdict for result in report_a.results] == [
        VERDICT_SUPPORT,
        VERDICT_REFUTE,
    ]


def test_one_verifier_failure_does_not_swallow_other_dimensions():
    """一个 evaluator 异常只留下本维 unknown，其他维继续执行。"""
    good_calls = []
    failed = VerifierRegistration(
        dimension=ProtocolKey((10,)),
        verifier=ProtocolKey((1, 10)),
        applies=lambda _request: True,
        evaluate=lambda _request: (_ for _ in ()).throw(
            RuntimeError("dimension failed")),
    )
    good = _registration(20, VERDICT_SUPPORT, good_calls)
    report = MultiVerifierOrchestrator().run(
        None, (good, failed), read_only=True)

    assert good_calls == [20]
    assert report.results[0].applicability == APPLICABILITY_APPLICABLE
    assert report.results[0].verdict == VERDICT_UNKNOWN
    assert report.results[0].operational_failure == "RuntimeError"
    assert report.results[1].verdict == VERDICT_SUPPORT


def test_not_applicable_and_conflicted_are_first_class_results():
    """not-applicable 与 conflicted 均须独立保留，不能压成 false。"""
    not_applicable = VerifierRegistration(
        dimension=ProtocolKey((50,)),
        verifier=ProtocolKey((1, 50)),
        applies=lambda _request: False,
        evaluate=lambda _request: VerificationEvaluation(
            verdict=VERDICT_SUPPORT),
    )
    conflicted = VerifierRegistration(
        dimension=ProtocolKey((60,)),
        verifier=ProtocolKey((1, 60)),
        applies=lambda _request: True,
        evaluate=lambda _request: VerificationEvaluation(
            verdict=VERDICT_CONFLICTED,
            claim_keys=((60, 0),),
        ),
    )
    report = MultiVerifierOrchestrator().run(
        None, (conflicted, not_applicable), read_only=True)
    assert report.results[0].applicability == APPLICABILITY_NOT_APPLICABLE
    assert report.results[0].verdict == VERDICT_UNKNOWN
    assert report.results[1].verdict == VERDICT_CONFLICTED


def test_variable_length_protocol_keys_use_one_canonical_order():
    """不同长度开放键在执行和 report 校验中必须使用同一排序规则。"""
    calls = []
    long_key = VerifierRegistration(
        dimension=ProtocolKey((9, 9)),
        verifier=ProtocolKey((1, 9, 9)),
        applies=lambda _request: True,
        evaluate=lambda _request: (
            calls.append((9, 9))
            or VerificationEvaluation(
                verdict=VERDICT_SUPPORT,
                claim_keys=((9, 9, 0),),
            )
        ),
    )
    short_key = VerifierRegistration(
        dimension=ProtocolKey((10,)),
        verifier=ProtocolKey((1, 10)),
        applies=lambda _request: True,
        evaluate=lambda _request: (
            calls.append((10,))
            or VerificationEvaluation(
                verdict=VERDICT_SUPPORT,
                claim_keys=((10, 0),),
            )
        ),
    )
    report = MultiVerifierOrchestrator().run(
        None, (long_key, short_key), read_only=True)
    assert calls == [(10,), (9, 9)]
    assert len(report.results) == 2


def test_read_only_preserves_verdict_and_blocks_commit():
    """只读评测必须得到相同 verdict 集，但不得提交任何 effect。"""
    dimension = ProtocolKey((30,))
    evidence_kind = ProtocolKey((300,))
    effect = VerificationEffect(dimension, evidence_kind, (30, 1))
    commits = []
    registration = VerifierRegistration(
        dimension=dimension,
        verifier=ProtocolKey((1, 30)),
        applies=lambda _request: True,
        evaluate=lambda _request: VerificationEvaluation(
            verdict=VERDICT_SUPPORT,
            claim_keys=((30, 0),),
            proposed_effects=(effect,),
        ),
        allowed_target_kinds=(evidence_kind,),
        commit=lambda evaluation: (
            commits.append(evaluation.verdict) or (effect,)
        ),
    )
    writable = MultiVerifierOrchestrator().run(
        None, (registration,), read_only=False)
    assert commits == [VERDICT_SUPPORT]
    assert writable.results[0].committed_effects == (effect,)

    commits.clear()
    read_only = MultiVerifierOrchestrator().run(
        None, (registration,), read_only=True)
    assert commits == []
    assert read_only.results[0].committed_effects == ()
    assert writable.verdict_key() == read_only.verdict_key()


def test_result_schema_rejects_untyped_keys_effects_and_report_entries():
    """结果边界须主动拒绝错类型，不能靠后续 AttributeError 偶发失败。"""
    key = ProtocolKey((30,))
    with pytest.raises(TypeError, match="result.dimension"):
        VerificationResult(
            dimension=(30,),
            verifier=key,
            applicability=APPLICABILITY_APPLICABLE,
            verdict=VERDICT_UNKNOWN,
        )
    with pytest.raises(TypeError, match="VerificationEffect"):
        VerificationEvaluation(
            verdict=VERDICT_UNKNOWN,
            proposed_effects=(key,),
        )
    with pytest.raises(TypeError, match="results 必须是 tuple"):
        VerificationReport(read_only=True, results=[])


def test_occurrence_order_effect_cannot_target_causal_namespace():
    """位置序 verifier 未获因果目标授权时必须拒绝更新 CAUSES。"""
    order_dimension = ProtocolKey((40,))
    order_evidence = ProtocolKey((400,))
    causal_evidence = ProtocolKey((401,))
    bad_effect = VerificationEffect(
        order_dimension, causal_evidence, (40, 1))
    registration = VerifierRegistration(
        dimension=order_dimension,
        verifier=ProtocolKey((1, 40)),
        applies=lambda _request: True,
        evaluate=lambda _request: VerificationEvaluation(
            verdict=VERDICT_SUPPORT,
            claim_keys=((40, 0),),
            proposed_effects=(bad_effect,),
        ),
        allowed_target_kinds=(order_evidence,),
    )
    report = MultiVerifierOrchestrator().run(
        None, (registration,), read_only=False)
    result = report.results[0]
    assert result.verdict == VERDICT_UNKNOWN
    assert result.operational_failure == "VerificationProtocolError"
    assert result.committed_effects == ()


def _mixed_segment() -> Segment:
    """构造五个 legacy 维度同时适用的整数声明段。"""
    return Segment(
        seg_id=0,
        modality=MODALITY_LANGUAGE,
        domain=DOMAIN_TEXT,
        lang=LANG_ZH,
        numeric_claims=[(1, 1, 1, 2)],
        comparison_claims=[(2, 1, 1)],
        universal_claims=[(0, 1)],
        existential_claims=[(1, 2)],
        precedes_pairs=[(0, 2)],
    )


def _real_mixed_item() -> CollectedItem:
    """构造可由现有 parser 同时提取五类声明的真实语言 item。"""
    tokens = [
        "3", "加", "5", "等于", "8",
        "5", "大于", "3",
        "鸟", "都是", "动物",
        "有的", "鸟", "是", "动物",
        "甲", "然后", "乙",
    ]
    return CollectedItem(
        tokens=tokens,
        role_seq=[1] * len(tokens),
        collect_type=COLLECT_CAUSES,
        source=SOURCE_BARE_TEXT,
        lang=LANG_ZH,
    )


def test_round_runtime_executes_all_legacy_dimensions_without_scalar_merge(
        monkeypatch):
    """混合样本的五个 adapter 必须全部执行并保留各自 episode。"""
    runner = DefaultRoundRunner()
    calls = []
    reward_by_route = {
        VERIFY_ROUTE_NUMERIC: 1,
        VERIFY_ROUTE_COMPARISON: 0,
        VERIFY_ROUTE_UNIVERSAL: None,
        VERIFY_ROUTE_EXISTENTIAL: 1,
        VERIFY_ROUTE_OCCURRENCE_ORDER: 1,
    }
    method_by_route = {
        VERIFY_ROUTE_NUMERIC: "_run_numeric_verify_round",
        VERIFY_ROUTE_COMPARISON: "_run_comparison_verify_round",
        VERIFY_ROUTE_UNIVERSAL: "_run_universal_verify_round",
        VERIFY_ROUTE_EXISTENTIAL: "_run_existential_verify_round",
        VERIFY_ROUTE_OCCURRENCE_ORDER: (
            "_run_occurrence_order_verify_round"),
    }
    for route, method_name in method_by_route.items():
        reward = reward_by_route[route]

        def stub(_ctx, _item, _raw, _obs, round_id, *,
                 selected_route=route, selected_reward=reward):
            """记录该维调用，并返回独立的测试 episode。"""
            calls.append((selected_route, round_id))
            if selected_reward is None:
                return RoundResult()
            return RoundResult(episode=Episode(
                episode_id=round_id,
                run_id=round_id,
                reward=selected_reward,
            ))

        monkeypatch.setattr(runner, method_name, stub)

    segment = _mixed_segment()
    item = CollectedItem(
        modality=MODALITY_LANGUAGE,
        domain=DOMAIN_TEXT,
        lang=LANG_ZH,
        source=1,
    )
    raw = InputPayload(segments=[segment], source=1, stage=3)
    ctx = SimpleNamespace(scope_owner=None, verification_reports=[])
    routes = (
        VERIFY_ROUTE_OCCURRENCE_ORDER,
        VERIFY_ROUTE_EXISTENTIAL,
        VERIFY_ROUTE_UNIVERSAL,
        VERIFY_ROUTE_COMPARISON,
        VERIFY_ROUTE_NUMERIC,
    )
    result = runner._run_verification_routes(
        ctx, item, raw, SimpleNamespace(), 7, routes)

    assert [route for route, _round_id in calls] == [
        VERIFY_ROUTE_NUMERIC,
        VERIFY_ROUTE_COMPARISON,
        VERIFY_ROUTE_UNIVERSAL,
        VERIFY_ROUTE_EXISTENTIAL,
        VERIFY_ROUTE_OCCURRENCE_ORDER,
    ]
    assert result.episode is None
    assert len(result.verification_report.results) == 5
    assert [entry.verdict for entry in result.verification_report.results] == [
        VERDICT_SUPPORT,
        VERDICT_REFUTE,
        VERDICT_UNKNOWN,
        VERDICT_SUPPORT,
        VERDICT_SUPPORT,
    ]
    assert len(result.episodes()) == 4
    assert len(ctx.verification_reports) == 1
    assert ctx.verification_reports[0].verdict_key() == (
        result.verification_report.verdict_key())
    assert all(
        entry.artifact is None
        for entry in ctx.verification_reports[0].results
    )


def test_real_round_records_all_parser_applicable_dimensions():
    """真实 observe/parser 路径必须在一个 report 中保留五个适用维度。"""
    ctx = make_train_context(DictBackend())
    runner = DefaultRoundRunner()
    with gates.gate_overrides({
            "CUE_EXTRACTOR_MODE": True,
            "NUMERIC_PROOF_MODE": True,
            "COMPARISON_PROOF_MODE": True,
            "UNIVERSAL_PROOF_MODE": True,
            "EXISTENTIAL_PROOF_MODE": True,
            "TIME_SEQ_PROOF_MODE": True,
            }):
        result = runner.run_round_full(
            ctx, _real_mixed_item(), STAGE3_REWARD, 11)
    report = result.verification_report
    assert report is not None
    assert [
        entry.dimension.stable_key()
        for entry in report.applicable_results()
    ] == [(2,), (3,), (4,), (5,), (6,)]
    assert [entry.verdict for entry in report.results] == [
        VERDICT_SUPPORT,
        VERDICT_SUPPORT,
        VERDICT_UNKNOWN,
        VERDICT_UNKNOWN,
        VERDICT_UNKNOWN,
    ]
    assert len(result.episodes()) == 2


def test_formal_train_returns_detached_multi_dimension_reports(tmp_path):
    """顶层正式训练结果必须携带轻量分维报告，而非只在 round 内可见。"""
    config = FormalTrainConfig(
        run_dir=str(tmp_path),
        run_id="r09-formal-train",
        rounds_per_stage=1,
        active_training_stages=(STAGE3_REWARD,),
        persist_graph_dump=False,
    )
    with stage_floor_overrides({"FLOOR_CONDUCTION_S3": 0}):
        with gates.gate_overrides({"TRAINING_MODE": True}):
            result = formal_train(
                config,
                [_real_mixed_item()],
                backend=DictBackend(),
                runner=DefaultRoundRunner(),
            )

    assert len(result.verification_reports) == 1
    report = result.verification_reports[0]
    assert [entry.dimension.stable_key() for entry in report.results] == [
        (2,), (3,), (4,), (5,), (6,),
    ]
    assert report.read_only is False
    assert all(entry.artifact is None for entry in report.results)


def test_isolated_evaluation_keeps_same_verdicts_and_host_reports_unchanged():
    """V-06 沙箱应得到同一分维 verdict，且不回写宿主 report 列表。"""
    ctx = make_train_context(DictBackend())
    runner = DefaultRoundRunner()
    item = _real_mixed_item()
    overrides = {
        "CUE_EXTRACTOR_MODE": True,
        "NUMERIC_PROOF_MODE": True,
        "COMPARISON_PROOF_MODE": True,
        "UNIVERSAL_PROOF_MODE": True,
        "EXISTENTIAL_PROOF_MODE": True,
        "TIME_SEQ_PROOF_MODE": True,
    }
    with gates.gate_overrides(overrides):
        training = runner.run_round_full(
            ctx, item, STAGE3_REWARD, 12)
    training_key = training.verification_report.verdict_key()
    ctx.verification_reports.clear()

    with isolated_evaluation(ctx, label="r09-multi-verifier") as eval_ctx:
        with gates.gate_overrides(overrides):
            evaluated = runner.run_round_full(
                eval_ctx,
                copy.deepcopy(item),
                STAGE3_REWARD,
                12,
            )
        assert evaluated.verification_report.read_only is True
        assert evaluated.verification_report.verdict_key() == training_key
        assert len(eval_ctx.verification_reports) == 1
    assert ctx.verification_reports == []


def test_old_single_episode_api_rejects_multiple_verification_episodes(
        monkeypatch):
    """旧接口不得从多个 verifier episode 中选择首项返回。"""
    runner = DefaultRoundRunner()
    monkeypatch.setattr(
        runner,
        "run_round_many",
        lambda *_args: (Episode(reward=1), Episode(reward=0)),
    )
    with pytest.raises(MultipleVerificationEpisodesError):
        runner.run_round(
            SimpleNamespace(),
            CollectedItem(),
            3,
            1,
        )
