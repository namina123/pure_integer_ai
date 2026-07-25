"""H-03 前缀描述长度、递归复用、反例成本和稳定排名对抗测试。"""
from __future__ import annotations

import pytest

from pure_integer_ai.cognition.shared.description_length import (
    DescriptionCandidate,
    DescriptionEncoding,
    DescriptionFragment,
    DescriptionLengthEngine,
    DescriptionLengthProblem,
    DescriptionModel,
    DescriptionObservation,
    DescriptionTerm,
    integer_tuple_bit_cost,
    nonnegative_prefix_bit_cost,
    signed_prefix_bit_cost,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_UNKNOWN,
    EvidenceRecord,
    HypothesisKey,
    HypothesisLedger,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    OBJECT_REPRESENTATION,
    OBJECT_STRUCTURE_CONCEPT,
    ObjectIdentity,
    SourceRef,
    VersionBundle,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope


def _source(document_id: int = 1) -> SourceRef:
    """构造描述长度问题使用的稳定裸观察来源。"""
    return SourceRef(
        1,
        1300,
        document_id,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(),
    )


def _unit(value: int) -> ObjectIdentity:
    """构造不绑定语言或 Unicode 的一等被编码对象。"""
    return ObjectIdentity(OBJECT_REPRESENTATION, (13000, value))


def _fragment(value: int) -> ObjectIdentity:
    """构造由候选模型定义的一等结构 fragment。"""
    return ObjectIdentity(OBJECT_STRUCTURE_CONCEPT, (13100, value))


def _hypothesis(source: SourceRef, marker: int) -> HypothesisKey:
    """构造共享竞争边界且完整来源化的 H-00 模型候选。"""
    return HypothesisKey(
        (13200, 1),
        (13201, marker),
        (13202, 1),
        document_scope(source),
        source,
    )


def _problem(
        source: SourceRef,
        unit_sequences: tuple[tuple[ObjectIdentity, ...], ...],
        ) -> DescriptionLengthProblem:
    """把多条对象序列组成必须被每个候选完整编码的问题。"""
    return DescriptionLengthProblem(tuple(
        DescriptionObservation(
            source,
            document_scope(source),
            (13300, index + 1),
            units,
        )
        for index, units in enumerate(unit_sequences)
    ))


def _literal_candidate(
        hypothesis: HypothesisKey,
        problem: DescriptionLengthProblem,
        ) -> DescriptionCandidate:
    """构造零 fragment、逐对象 literal 编码的可解码基线候选。"""
    encodings = tuple(
        DescriptionEncoding(
            observation.source,
            observation.scope,
            observation.event_key,
            tuple(DescriptionTerm.literal(unit) for unit in observation.units),
        )
        for observation in problem.observations
    )
    return DescriptionCandidate(DescriptionModel(hypothesis), encodings)


def _ledger(*hypotheses: HypothesisKey) -> HypothesisLedger:
    """建立真实拥有全部测试候选的 H-00 ledger。"""
    ledger = HypothesisLedger()
    for hypothesis in hypotheses:
        ledger.register(hypothesis)
    return ledger


def test_prefix_costs_are_integer_prefix_free_units_and_reject_bool():
    """gamma/zigzag/tuple 成本使用统一 bit 单位且拒绝 bool 冒充整数。"""
    assert tuple(nonnegative_prefix_bit_cost(value) for value in range(4)) == (
        1, 3, 3, 5)
    assert signed_prefix_bit_cost(0) == 1
    assert signed_prefix_bit_cost(-1) == signed_prefix_bit_cost(1) == 3
    assert integer_tuple_bit_cost(()) == 1
    assert integer_tuple_bit_cost((0, -1, 1)) == 12
    with pytest.raises((TypeError, ValueError)):
        nonnegative_prefix_bit_cost(True)
    with pytest.raises((TypeError, ValueError)):
        integer_tuple_bit_cost((1, True))


def test_recursive_fragments_beat_literal_after_real_reuse():
    """重复对象足够多时，递归 fragment 的净描述长度真实低于 literal。"""
    source = _source(1)
    units = (_unit(1), _unit(2)) * 64
    problem = _problem(source, (units,))
    hypothesis = _hypothesis(source, 2)
    first = _fragment(1)
    second = _fragment(2)
    model = DescriptionModel(hypothesis, (
        DescriptionFragment(first, (
            DescriptionTerm.literal(_unit(1)),
            DescriptionTerm.literal(_unit(2)),
        )),
        DescriptionFragment(second, (
            DescriptionTerm.fragment(first),
            DescriptionTerm.fragment(first),
        )),
    ))
    observation = problem.observations[0]
    candidate = DescriptionCandidate(model, (
        DescriptionEncoding(
            source,
            observation.scope,
            observation.event_key,
            tuple(DescriptionTerm.fragment(second) for _ in range(32)),
        ),
    ))

    score = DescriptionLengthEngine(_ledger(hypothesis)).score(
        problem, candidate)
    assert score.total_cost < score.literal_baseline_cost
    assert score.recursive_reuse_gain > 0
    assert score.fragment_count == 2
    assert score.recursive_fragment_reference_count == 2
    assert score.fragment_reference_count == 34


def test_one_fragment_per_observation_loses_to_literal_memorization_baseline():
    """每句单独建 fragment 只重复记忆定义，模型成本使其败给逐对象 literal。"""
    source = _source(2)
    problem = _problem(source, tuple(
        tuple(_unit(index * 4 + offset) for offset in range(4))
        for index in range(3)
    ))
    literal_hypothesis = _hypothesis(source, 4)
    memorized_hypothesis = _hypothesis(source, 5)
    literal = _literal_candidate(literal_hypothesis, problem)
    fragments = tuple(
        DescriptionFragment(
            _fragment(index + 10),
            tuple(DescriptionTerm.literal(unit) for unit in observation.units),
        )
        for index, observation in enumerate(problem.observations)
    )
    memorized = DescriptionCandidate(
        DescriptionModel(memorized_hypothesis, fragments),
        tuple(
            DescriptionEncoding(
                observation.source,
                observation.scope,
                observation.event_key,
                (DescriptionTerm.fragment(fragments[index].fragment),),
            )
            for index, observation in enumerate(problem.observations)
        ),
    )
    engine = DescriptionLengthEngine(_ledger(
        literal_hypothesis, memorized_hypothesis))
    literal_score = engine.score(problem, literal)
    memorized_score = engine.score(problem, memorized)

    assert literal_score.recursive_reuse_gain == 0
    assert memorized_score.recursive_reuse_gain < 0
    assert memorized_score.total_cost > literal_score.total_cost
    assert engine.rank(problem, (memorized, literal))[0].hypothesis == (
        literal_hypothesis)


def test_active_refute_adds_deduplicated_exception_cost_and_changes_rank():
    """过宽模型的 active 负 Evidence 增加例外成本，同内容重放不重复收费。"""
    source = _source(3)
    problem = _problem(source, ((_unit(1), _unit(2)),))
    broad_hypothesis = _hypothesis(source, 6)
    clean_hypothesis = _hypothesis(source, 7)
    ledger = _ledger(broad_hypothesis, clean_hypothesis)
    for evidence_id, timestamp in ((14001, 1), (14002, 2)):
        ledger.append_evidence(EvidenceRecord(
            evidence_id,
            broad_hypothesis,
            EVIDENCE_REFUTE,
            (13400, 1),
            source,
            timestamp,
            payload=(13400, 2, 3),
        ))
    broad = _literal_candidate(broad_hypothesis, problem)
    clean = _literal_candidate(clean_hypothesis, problem)
    engine = DescriptionLengthEngine(ledger)
    ledger_state = ledger.state_key()
    broad_score = engine.score(problem, broad)
    clean_score = engine.score(problem, clean)

    assert broad_score.exception_count == 1
    assert broad_score.exception_cost > 0
    assert clean_score.exception_cost == 0
    assert broad_score.total_cost > clean_score.total_cost
    assert engine.rank(problem, (broad, clean))[0].hypothesis == clean_hypothesis
    assert ledger.state_key() == ledger_state


def test_superseded_refute_no_longer_contributes_exception_cost():
    """只读取 snapshot 当前 active refute，被 unknown 替代的旧反例保留历史但不重复计费。"""
    source = _source(4)
    problem = _problem(source, ((_unit(1),),))
    hypothesis = _hypothesis(source, 8)
    ledger = _ledger(hypothesis)
    ledger.append_evidence(EvidenceRecord(
        14101,
        hypothesis,
        EVIDENCE_REFUTE,
        (13500, 1),
        source,
        1,
        payload=(13500, 2),
    ))
    before = DescriptionLengthEngine(ledger).score(
        problem, _literal_candidate(hypothesis, problem))
    ledger.append_evidence(EvidenceRecord(
        14102,
        hypothesis,
        EVIDENCE_UNKNOWN,
        (13500, 3),
        source,
        2,
        payload=(13500, 4),
        supersedes_evidence_id=14101,
    ))
    after = DescriptionLengthEngine(ledger).score(
        problem, _literal_candidate(hypothesis, problem))

    assert before.exception_count == 1
    assert after.exception_count == 0
    assert after.exception_cost == 0
    assert len(ledger.evidence_history(hypothesis)) == 2


def test_candidate_must_exactly_cover_problem_and_use_closed_acyclic_fragments():
    """漏槽、重复槽、错误展开、模型外引用和递归环全部 fail closed。"""
    source = _source(5)
    problem = _problem(source, ((_unit(1), _unit(2)),))
    hypothesis = _hypothesis(source, 10)
    engine = DescriptionLengthEngine(_ledger(hypothesis))
    observation = problem.observations[0]

    with pytest.raises(ValueError, match="全部观察槽"):
        engine.score(problem, DescriptionCandidate(
            DescriptionModel(hypothesis), ()))
    duplicate = DescriptionEncoding(
        source,
        observation.scope,
        observation.event_key,
        (DescriptionTerm.literal(_unit(1)),
         DescriptionTerm.literal(_unit(2))),
    )
    with pytest.raises(ValueError, match="重复编码"):
        engine.score(problem, DescriptionCandidate(
            DescriptionModel(hypothesis), (duplicate, duplicate)))
    wrong = DescriptionEncoding(
        source,
        observation.scope,
        observation.event_key,
        (DescriptionTerm.literal(_unit(2)),
         DescriptionTerm.literal(_unit(1))),
    )
    with pytest.raises(ValueError, match="无损展开"):
        engine.score(problem, DescriptionCandidate(
            DescriptionModel(hypothesis), (wrong,)))
    unknown_fragment = DescriptionEncoding(
        source,
        observation.scope,
        observation.event_key,
        (DescriptionTerm.fragment(_fragment(99)),),
    )
    with pytest.raises(ValueError, match="模型外"):
        engine.score(problem, DescriptionCandidate(
            DescriptionModel(hypothesis), (unknown_fragment,)))

    first = _fragment(20)
    second = _fragment(21)
    cyclic_model = DescriptionModel(hypothesis, (
        DescriptionFragment(first, (DescriptionTerm.fragment(second),)),
        DescriptionFragment(second, (DescriptionTerm.fragment(first),)),
    ))
    with pytest.raises(ValueError, match="可终止 DAG"):
        engine.score(problem, DescriptionCandidate(
            cyclic_model,
            (DescriptionEncoding(
                source,
                observation.scope,
                observation.event_key,
                (DescriptionTerm.fragment(first),),
            ),),
        ))


def test_empty_observation_is_losslessly_encoded_without_special_language_case():
    """空对象观察由空 term 序编码，成本只保留统一边界而无需语言特判。"""
    source = _source(6)
    problem = _problem(source, ((),))
    hypothesis = _hypothesis(source, 12)
    observation = problem.observations[0]
    candidate = DescriptionCandidate(
        DescriptionModel(hypothesis),
        (DescriptionEncoding(
            source,
            observation.scope,
            observation.event_key,
            (),
        ),),
    )
    score = DescriptionLengthEngine(_ledger(hypothesis)).score(
        problem, candidate)
    assert score.recursive_reuse_gain == 0
    assert score.total_cost == score.literal_baseline_cost


def test_breakdown_ablations_are_arithmetic_and_ties_use_full_identity():
    """四项消融逐项可审计，总成本同分时只按完整 Hypothesis 键决胜。"""
    source = _source(7)
    problem = _problem(source, ((_unit(1), _unit(2)),))
    first_hypothesis = _hypothesis(source, 14)
    second_hypothesis = _hypothesis(source, 15)
    first = _literal_candidate(first_hypothesis, problem)
    second = _literal_candidate(second_hypothesis, problem)
    engine = DescriptionLengthEngine(_ledger(
        first_hypothesis, second_hypothesis))
    first_score = engine.score(problem, first)
    second_score = engine.score(problem, second)

    assert first_score.total_cost == (
        first_score.model_cost
        + first_score.encoded_data_cost
        + first_score.boundary_cost
        + first_score.exception_cost)
    assert first_score.without_model_cost == (
        first_score.total_cost - first_score.model_cost)
    assert first_score.without_exception_cost == first_score.total_cost
    assert first_score.without_boundary_cost == (
        first_score.total_cost - first_score.boundary_cost)
    assert first_score.without_reuse_cost == (
        first_score.literal_baseline_cost + first_score.exception_cost)
    assert first_score.total_cost == second_score.total_cost
    ranked = engine.rank(problem, (second, first))
    assert tuple(item.hypothesis for item in ranked) == (
        first_hypothesis,
        second_hypothesis,
    )
