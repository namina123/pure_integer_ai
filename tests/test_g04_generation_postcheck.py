"""G-04 surface 反解析、多维复核和失败分型对抗测试。"""
from __future__ import annotations

from dataclasses import replace

import pytest

from pure_integer_ai.cognition.shared.formal_artifact_bridge import (
    ArtifactVerificationObservation,
)
from pure_integer_ai.cognition.shared.generation_verification import (
    GenerationPostcheckRequest,
    GenerationSourceRequirement,
    GenerationSurfaceObservation,
    GenerationSurfaceParseRequest,
    GenerationSurfaceParseResult,
    GenerationTaskObservation,
    GenerationTaskRequirement,
    RecoveredGenerationProposition,
)
from pure_integer_ai.cognition.shared.identity import (
    concept_identity,
    minimal_instruction_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import generation_scope
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.experiments.generation_production_runtime import (
    ProductionGenerationRuntime,
)
from pure_integer_ai.experiments.generation_verification_runtime import (
    GenerationPostcheckProtocol,
    GenerationPostcheckRuntime,
)
from pure_integer_ai.experiments.verification_orchestration import (
    APPLICABILITY_APPLICABLE,
    APPLICABILITY_NOT_APPLICABLE,
    VERDICT_REFUTE,
    VERDICT_SUPPORT,
    VERDICT_UNKNOWN,
    VerificationEvaluation,
)
from pure_integer_ai.experiments.round_runtime import DefaultRoundRunner
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.training.stages import STAGE3_REWARD
from tests.test_l05b2_typed_production_generation import (
    _forbid_legacy,
    _language_item,
    _production_fixture,
)


_BASE = 13600


def _protocol() -> GenerationPostcheckProtocol:
    """构造六维互异协议键和全部内置分型 reason。"""
    keys = tuple(ProtocolKey((_BASE + 1, index)) for index in range(1, 13))
    reasons = tuple(
        minimal_instruction_identity((_BASE + 2, index))
        for index in range(1, 16)
    )
    return GenerationPostcheckProtocol(*keys, *reasons)


def _planned(execution):
    """返回当前 execution 中 G-02 逐点保存的 planned Proposition。"""
    return execution.surface.preview.request.structure.propositions.propositions


def _source_requirements(execution):
    """为每个 planned Proposition 要求 citation 和独立 trust 核验。"""
    return tuple(
        GenerationSourceRequirement(
            item.candidate_key,
            item.source,
            item.scope,
            True,
            True,
            (_BASE + 3, index),
        )
        for index, item in enumerate(_planned(execution), start=1)
    )


def _recovered(execution, *, scope=None):
    """按 planned candidate 构造 parser 恢复结果，可注入 scope 漂移。"""
    return tuple(
        RecoveredGenerationProposition(
            item.candidate_key,
            item.proposition,
            item.source,
            item.scope if scope is None else scope,
            (_BASE + 4, index),
        )
        for index, item in enumerate(_planned(execution), start=1)
    )


def _observation(
        execution,
        *,
        propositions=None,
        scope=None,
        artifact_keys=(),
        task_observations=(),
        cited_sources=None,
        ) -> GenerationSurfaceObservation:
    """构造绑定同次 surface/renderer 的 typed 反解析观察。"""
    goal = execution.plan.request.goal
    structure = execution.surface.preview.request.structure
    recovered = _recovered(execution) if propositions is None else propositions
    cited = (
        tuple(sorted(
            {item.source for item in _planned(execution)},
            key=lambda item: item.stable_key(),
        ))
        if cited_sources is None else tuple(cited_sources)
    )
    return GenerationSurfaceObservation(
        GenerationSurfaceParseRequest.from_execution(execution).stable_key(),
        execution.representations,
        goal.target_branch,
        structure.selection.stance,
        goal.source,
        goal.scope if scope is None else scope,
        recovered,
        artifact_keys,
        cited,
        structure.syntax.stable_key(),
        task_observations,
        (_BASE + 5, 1),
    )


class _Parser:
    """返回测试注入的成功观察或 typed parse failure。"""

    def __init__(self, parsed: GenerationSurfaceParseResult) -> None:
        self.parsed = parsed
        self.calls = 0

    def parse(self, request):
        """记录调用并返回固定 parse 结果。"""
        assert request.units
        self.calls += 1
        return self.parsed


class _ExecutionParser:
    """从 mapper 预登记的 fixture 观察返回结果，不读取 generation plan。"""

    def __init__(self) -> None:
        self.calls = 0
        self._observations = {}

    def record(self, execution, *, artifact_keys=(), cited_sources=None) -> None:
        """由测试 mapper 按受限请求键显式登记预期观察。"""
        request_key = GenerationSurfaceParseRequest.from_execution(
            execution).stable_key()
        self._observations[request_key] = _observation(
            execution,
            artifact_keys=artifact_keys,
            cited_sources=cited_sources,
        )

    def parse(self, request):
        """只按受限请求键查找 fixture 观察并返回 typed 结果。"""
        self.calls += 1
        observation = self._observations.get(request.stable_key())
        if observation is None:
            return GenerationSurfaceParseResult(
                minimal_instruction_identity((_BASE + 16, 3)),
                (_BASE + 16, 4),
            )
        return GenerationSurfaceParseResult(
            minimal_instruction_identity((_BASE + 16, 1)),
            (_BASE + 16, 2),
            observation,
        )


class _StaticVerifier:
    """为结构或来源维度返回固定 verdict 的独立 verifier。"""

    def __init__(self, verdict: int, marker: int) -> None:
        self.verdict = verdict
        self.marker = marker
        self.calls = 0
        self.requests = []

    def verify(self, request):
        """把当前 execution 作为 claim，并绑定 generation goal 归属。"""
        self.calls += 1
        self.requests.append(request)
        goal = request.postcheck.execution.plan.request.goal
        return VerificationEvaluation(
            self.verdict,
            (request.postcheck.execution.stable_key(),),
            detail=(_BASE + 6, self.marker),
            source=goal.source,
            scope=goal.scope,
        )


class _EmptyClaimVerifier:
    """模拟遗漏 claim 的外部 verifier 契约漂移。"""

    def verify(self, request):
        """返回其余字段合法但无归因 claim 的结果。"""
        goal = request.postcheck.execution.plan.request.goal
        return VerificationEvaluation(
            VERDICT_SUPPORT,
            detail=(_BASE + 6, 9),
            source=goal.source,
            scope=goal.scope,
        )


class _TaskVerifier:
    """逐任务比较显式 expected_result_key 和实际 result_key。"""

    def verify(self, request):
        """结果逐点相同则 support，否则 refute。"""
        expected = {
            item.task: item.expected_result_key for item in request.requirements}
        actual = {
            item.task: item.result_key
            for item in request.observation.task_observations
        }
        goal = request.postcheck.execution.plan.request.goal
        return VerificationEvaluation(
            VERDICT_SUPPORT if expected == actual else VERDICT_REFUTE,
            tuple(item.task.stable_key() for item in request.requirements),
            detail=(_BASE + 7, 1 if expected == actual else 2),
            source=goal.source,
            scope=goal.scope,
        )


class _ArtifactVerifier:
    """返回注入的三态 S-06 复核结果，或模拟 verifier 运行异常。"""

    def __init__(self, accepted, *, raises: bool = False) -> None:
        self.accepted = accepted
        self.raises = raises
        self.calls = 0

    def verify(self, request):
        """保持 invocation 归属并返回三态观察；异常仅供分型对抗。"""
        self.calls += 1
        if self.raises:
            raise RuntimeError("artifact verifier failed")
        return ArtifactVerificationObservation(
            request.invocation.definition.verifier,
            request.invocation.source,
            request.invocation.scope,
            self.accepted,
            (_BASE + 14, 1),
            (_BASE + 14, 2),
        )


class _ProductionPostcheckMapper:
    """把 formal owner 持有的 attachment 和来源要求显式绑定到同次执行。"""

    def __init__(self, artifacts=(), *, parser=None) -> None:
        self.artifacts = tuple(artifacts)
        self.parser = parser
        self.calls = 0

    def build(self, ctx, item, input_payload, observation, execution):
        """在 generation scope 内建立请求，不从 item token 或旧 path 猜语义。"""
        del item, observation
        self.calls += 1
        assert ctx.work_memory.active_generation_scope is not None
        assert input_payload.scope_identity == ctx.work_memory.active_episode_scope
        if self.parser is not None:
            self.parser.record(
                execution,
                artifact_keys=execution.surface.preview.request.structure
                .selection.selected_artifact_keys,
            )
        return GenerationPostcheckRequest(
            execution,
            self.artifacts,
            _source_requirements(execution),
        )


def _runtime(
        parsed,
        *,
        source_verdict=VERDICT_SUPPORT,
        source_verifier=None,
        artifact_verifier=None,
        task_verifier=None,
        ):
    """装配无副作用的 G-04 parser、结构、来源和可选任务 verifier。"""
    protocol = _protocol()
    runtime = GenerationPostcheckRuntime(
        protocol,
        _Parser(parsed),
        _StaticVerifier(VERDICT_SUPPORT, 1),
        (_StaticVerifier(source_verdict, 2)
         if source_verifier is None else source_verifier),
        artifact_verifier=artifact_verifier,
        task_verifier=task_verifier,
    )
    return protocol, runtime


def _result(run, dimension):
    """返回指定 dimension 的唯一 VerificationResult。"""
    matches = run.report.dimension_results(dimension)
    assert len(matches) == 1
    return matches[0]


def test_g04_complete_surface_passes_independent_dimensions_without_reward():
    """完整反解析应通过结构、命题、scope 和来源，Artifact/task 保持 N/A。"""
    fixture = _production_fixture()
    try:
        execution = fixture.runtime._executor.execute(fixture.request)
        parsed = GenerationSurfaceParseResult(
            minimal_instruction_identity((_BASE + 8, 1)),
            (_BASE + 8, 2),
            _observation(execution),
        )
        protocol, runtime = _runtime(parsed)
        request = GenerationPostcheckRequest(
            execution, (), _source_requirements(execution))

        run = runtime.run(request)

        assert run.complete
        assert len(run.report.results) == 6
        assert _result(run, protocol.structure_dimension).verdict == VERDICT_SUPPORT
        assert _result(run, protocol.proposition_dimension).verdict == VERDICT_SUPPORT
        assert _result(run, protocol.scope_dimension).verdict == VERDICT_SUPPORT
        assert _result(run, protocol.source_dimension).verdict == VERDICT_SUPPORT
        assert _result(
            run, protocol.artifact_dimension).applicability == (
                APPLICABILITY_NOT_APPLICABLE)
        assert _result(
            run, protocol.task_dimension).applicability == (
                APPLICABILITY_NOT_APPLICABLE)
        assert not any(
            item.proposed_effects or item.committed_effects
            for item in run.report.results)
        assert runtime.run(request).stable_key() == run.stable_key()
    finally:
        fixture.close()


def test_g04_fluent_surface_with_missing_core_proposition_is_refuted():
    """renderer 输出和结构均可通过，但少一个核心 Proposition 时命题维必须失败。"""
    fixture = _production_fixture()
    try:
        execution = fixture.runtime._executor.execute(fixture.request)
        recovered = _recovered(execution)
        assert len(recovered) >= 2
        parsed = GenerationSurfaceParseResult(
            minimal_instruction_identity((_BASE + 9, 1)),
            (_BASE + 9, 2),
            _observation(execution, propositions=recovered[:-1]),
        )
        protocol, runtime = _runtime(parsed)

        run = runtime.run(GenerationPostcheckRequest(
            execution, (), _source_requirements(execution)))

        assert not run.complete
        assert _result(run, protocol.structure_dimension).verdict == VERDICT_SUPPORT
        assert _result(run, protocol.proposition_dimension).verdict == VERDICT_REFUTE
    finally:
        fixture.close()


def test_g04_scope_drift_is_independent_from_proposition_identity():
    """命题身份未变而运行 scope 漂移时，命题维通过但 scope 维失败。"""
    fixture = _production_fixture()
    try:
        execution = fixture.runtime._executor.execute(fixture.request)
        goal = execution.plan.request.goal
        drift_scope = generation_scope(99, parent=goal.scope)
        parsed = GenerationSurfaceParseResult(
            minimal_instruction_identity((_BASE + 10, 1)),
            (_BASE + 10, 2),
            _observation(
                execution,
                propositions=_recovered(execution, scope=drift_scope),
                scope=drift_scope,
            ),
        )
        protocol, runtime = _runtime(parsed)

        run = runtime.run(GenerationPostcheckRequest(
            execution, (), _source_requirements(execution)))

        assert not run.complete
        assert _result(run, protocol.proposition_dimension).verdict == VERDICT_SUPPORT
        assert _result(run, protocol.scope_dimension).verdict == VERDICT_REFUTE
    finally:
        fixture.close()


def test_g04_citation_does_not_override_source_verifier_refutation():
    """所有 citation 都存在时，独立来源 verifier 反驳仍必须保留为失败。"""
    fixture = _production_fixture()
    try:
        execution = fixture.runtime._executor.execute(fixture.request)
        parsed = GenerationSurfaceParseResult(
            minimal_instruction_identity((_BASE + 11, 1)),
            (_BASE + 11, 2),
            _observation(execution),
        )
        protocol, runtime = _runtime(
            parsed, source_verdict=VERDICT_REFUTE)

        run = runtime.run(GenerationPostcheckRequest(
            execution, (), _source_requirements(execution)))

        assert not run.complete
        assert _result(run, protocol.source_dimension).verdict == VERDICT_REFUTE
    finally:
        fixture.close()


def test_g04_citation_only_requirement_does_not_invoke_trust_verifier():
    """只要求 citation 时，引用满足即可 support，不得伪造额外可信度前提。"""
    fixture = _production_fixture()
    try:
        execution = fixture.runtime._executor.execute(fixture.request)
        parsed = GenerationSurfaceParseResult(
            minimal_instruction_identity((_BASE + 11, 3)),
            (_BASE + 11, 4),
            _observation(execution),
        )
        protocol, runtime = _runtime(parsed)
        requirements = tuple(
            replace(item, trust_required=False)
            for item in _source_requirements(execution)
        )

        run = runtime.run(GenerationPostcheckRequest(
            execution, (), requirements))

        assert run.complete
        assert _result(
            run, protocol.source_dimension).verdict == VERDICT_SUPPORT
        assert runtime.source_verifier.calls == 0
    finally:
        fixture.close()


def test_g04_citation_covers_actual_evidence_sources_not_only_owner_source():
    """命题归属来源存在但缺一条实际 Evidence 来源时，来源维仍须拒绝。"""
    fixture = _production_fixture()
    try:
        execution = fixture.runtime._executor.execute(fixture.request)
        requirements = list(_source_requirements(execution))
        owner_source = requirements[0].source
        evidence_source = replace(
            owner_source,
            source_id=owner_source.source_id + _BASE,
        )
        requirements[0] = replace(
            requirements[0],
            evidence_sources=(owner_source, evidence_source),
        )
        parsed = GenerationSurfaceParseResult(
            minimal_instruction_identity((_BASE + 11, 7)),
            (_BASE + 11, 8),
            _observation(execution),
        )
        protocol, runtime = _runtime(parsed)

        run = runtime.run(GenerationPostcheckRequest(
            execution, (), tuple(requirements)))

        assert requirements[0].source == owner_source
        assert evidence_source in requirements[0].evidence_sources
        assert _result(run, protocol.source_dimension).verdict == VERDICT_REFUTE
        assert runtime.source_verifier.calls == 0
    finally:
        fixture.close()


def test_g04_passes_all_actual_evidence_sources_to_trust_verifier():
    """全部实际 Evidence 来源被引用后，独立 trust verifier 应收到完整集合。"""
    fixture = _production_fixture()
    try:
        execution = fixture.runtime._executor.execute(fixture.request)
        requirements = list(_source_requirements(execution))
        owner_source = requirements[0].source
        evidence_source = replace(
            owner_source,
            source_id=owner_source.source_id + _BASE,
        )
        requirements[0] = replace(
            requirements[0],
            evidence_sources=(owner_source, evidence_source),
        )
        cited_sources = tuple({
            *(item.source for item in _planned(execution)),
            evidence_source,
        })
        parsed = GenerationSurfaceParseResult(
            minimal_instruction_identity((_BASE + 11, 9)),
            (_BASE + 11, 10),
            _observation(execution, cited_sources=cited_sources),
        )
        verifier = _StaticVerifier(VERDICT_SUPPORT, 3)
        protocol, runtime = _runtime(parsed, source_verifier=verifier)

        run = runtime.run(GenerationPostcheckRequest(
            execution, (), tuple(requirements)))

        assert run.complete
        assert _result(run, protocol.source_dimension).verdict == VERDICT_SUPPORT
        assert verifier.calls == 1
        received = verifier.requests[0].requirements
        assert received[0].evidence_sources == tuple(sorted(
            (owner_source, evidence_source),
            key=lambda item: item.stable_key(),
        ))
    finally:
        fixture.close()


def test_g04_external_verifier_empty_claim_becomes_typed_unknown():
    """外部 verifier 漏 claim 属契约漂移，应分型 unknown 而非中止整份报告。"""
    fixture = _production_fixture()
    try:
        execution = fixture.runtime._executor.execute(fixture.request)
        parsed = GenerationSurfaceParseResult(
            minimal_instruction_identity((_BASE + 11, 5)),
            (_BASE + 11, 6),
            _observation(execution),
        )
        protocol, runtime = _runtime(
            parsed, source_verifier=_EmptyClaimVerifier())

        run = runtime.run(GenerationPostcheckRequest(
            execution, (), _source_requirements(execution)))

        result = _result(run, protocol.source_dimension)
        assert result.verdict == VERDICT_UNKNOWN
        assert result.claim_keys
        assert result.operational_failure is None
    finally:
        fixture.close()


def test_g04_parse_failure_returns_typed_multidimensional_feedback():
    """parser 无法恢复时必须返回分维 refute/unknown，而非旧 judge 或统一 reward。"""
    fixture = _production_fixture()
    try:
        execution = fixture.runtime._executor.execute(fixture.request)
        parsed = GenerationSurfaceParseResult(
            minimal_instruction_identity((_BASE + 12, 1)),
            (_BASE + 12, 2),
        )
        protocol, runtime = _runtime(parsed)

        run = runtime.run(GenerationPostcheckRequest(
            execution, (), _source_requirements(execution)))

        assert not run.complete
        assert _result(run, protocol.structure_dimension).verdict == VERDICT_REFUTE
        assert _result(run, protocol.proposition_dimension).verdict == VERDICT_UNKNOWN
        assert _result(run, protocol.scope_dimension).verdict == VERDICT_UNKNOWN
        assert not any(item.operational_failure for item in run.report.results)
    finally:
        fixture.close()


def test_g04_task_result_uses_explicit_requirement_and_actual_observation():
    """task dimension 只比较 postcheck request 的要求和实际观察，不解释 goal_kind。"""
    fixture = _production_fixture()
    try:
        execution = fixture.runtime._executor.execute(fixture.request)
        goal = execution.plan.request.goal
        task = concept_identity((_BASE + 13, 1))
        requirement = GenerationTaskRequirement(
            task,
            minimal_instruction_identity((_BASE + 13, 2)),
            (_BASE + 13, 3),
            goal.source,
            goal.scope,
            (_BASE + 13, 4),
        )
        observation = GenerationTaskObservation(
            task,
            requirement.expected_result_key,
            goal.source,
            goal.scope,
            (_BASE + 13, 5),
        )
        parsed = GenerationSurfaceParseResult(
            minimal_instruction_identity((_BASE + 13, 6)),
            (_BASE + 13, 7),
            _observation(execution, task_observations=(observation,)),
        )
        protocol, runtime = _runtime(parsed, task_verifier=_TaskVerifier())
        request = GenerationPostcheckRequest(
            execution,
            (),
            _source_requirements(execution),
            (requirement,),
        )

        passed = runtime.run(request)
        failed_parse = replace(
            parsed,
            observation=replace(
                parsed.observation,
                task_observations=(replace(observation, result_key=(_BASE + 13, 8)),),
            ),
        )
        _, failed_runtime = _runtime(
            failed_parse, task_verifier=_TaskVerifier())
        failed = failed_runtime.run(request)

        assert passed.complete
        assert _result(passed, protocol.task_dimension).verdict == VERDICT_SUPPORT
        assert not failed.complete
        assert _result(failed, protocol.task_dimension).verdict == VERDICT_REFUTE
    finally:
        fixture.close()


@pytest.mark.parametrize(
    ("accepted", "raises", "expected_verdict", "expected_complete"),
    (
        (True, False, VERDICT_SUPPORT, True),
        (False, False, VERDICT_REFUTE, False),
        (None, False, VERDICT_UNKNOWN, False),
        (None, True, VERDICT_UNKNOWN, False),
    ),
)
def test_g04_artifact_reverification_keeps_domain_failures_typed(
        accepted, raises, expected_verdict, expected_complete):
    """真实采用的 Artifact 必须独立复核，拒绝、未知和异常不得变成运行失败。"""
    fixture = _production_fixture(with_artifact=True)
    try:
        execution = fixture.runtime._executor.execute(fixture.request)
        selected_keys = execution.surface.preview.request.structure.selection \
            .selected_artifact_keys
        parsed = GenerationSurfaceParseResult(
            minimal_instruction_identity((_BASE + 15, 1)),
            (_BASE + 15, 2),
            _observation(execution, artifact_keys=selected_keys),
        )
        verifier = _ArtifactVerifier(accepted, raises=raises)
        protocol, runtime = _runtime(
            parsed, artifact_verifier=verifier)

        run = runtime.run(GenerationPostcheckRequest(
            execution,
            fixture.artifacts,
            _source_requirements(execution),
        ))

        result = _result(run, protocol.artifact_dimension)
        assert result.applicability == APPLICABILITY_APPLICABLE
        assert result.verdict == expected_verdict
        assert run.complete is expected_complete
        assert result.operational_failure is None
        assert not any(item.operational_failure for item in run.report.results)
        assert verifier.calls == 1
    finally:
        fixture.close()


@pytest.mark.parametrize(
    ("source_verdict", "expected_complete"),
    (
        (VERDICT_SUPPORT, True),
        (VERDICT_REFUTE, False),
    ),
)
def test_g04_production_round_runs_postcheck_after_committed_surface_use(
        monkeypatch, source_verdict, expected_complete):
    """正式 round 在真实 Use 提交后运行 G-04，复核失败不得回滚生成事实。"""
    fixture = _production_fixture()
    backend = DictBackend()
    try:
        protocol = _protocol()
        parser = _ExecutionParser()
        postcheck_mapper = _ProductionPostcheckMapper(
            fixture.artifacts,
            parser=parser,
        )
        postchecker = GenerationPostcheckRuntime(
            protocol,
            parser,
            _StaticVerifier(VERDICT_SUPPORT, 1),
            _StaticVerifier(source_verdict, 2),
        )
        runtime = ProductionGenerationRuntime(
            fixture.runtime._mapper,
            fixture.runtime._executor,
            postcheck_mapper=postcheck_mapper,
            postchecker=postchecker,
        )
        ctx = make_train_context(backend)
        ctx.language_generation_runtime = runtime
        _forbid_legacy(monkeypatch)
        before_alias = fixture.alias.runtime.state_key()

        result = DefaultRoundRunner().run_round_full(
            ctx,
                _language_item(fixture.request.goal.source),
            STAGE3_REWARD,
            1,
        )

        assert result.output is not None
        assert result.output.complete
        assert result.output.postcheck is not None
        assert result.output.postcheck_complete is expected_complete
        assert _result(
            result.output.postcheck,
            protocol.source_dimension,
        ).verdict == source_verdict
        assert fixture.alias.runtime.state_key() != before_alias
        assert postcheck_mapper.calls == 1
        assert parser.calls == 1
        assert ctx.work_memory.active_query_scope is None
        assert ctx.work_memory.active_generation_scope is None
    finally:
        backend.close()
        fixture.close()
