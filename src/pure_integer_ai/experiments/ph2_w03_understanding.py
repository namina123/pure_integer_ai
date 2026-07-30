"""PH2 W-03 typed Sense 候选的 H-00/H-04 理解闭环。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.candidate_projection import (
    CandidateProjectionGraph,
)
from pure_integer_ai.cognition.shared.candidate_runtime import (
    CandidateLearningOutcome,
    CandidateLearningRuntime,
    CandidateProjectionMetadata,
)
from pure_integer_ai.cognition.shared.candidate_verifier import (
    RevealedObjectObservation,
)
from pure_integer_ai.cognition.shared.graph_ontology import GraphOntology
from pure_integer_ai.cognition.shared.hypothesis import (
    EPISTEMIC_CONFLICTED,
    EPISTEMIC_REFUTED,
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EVIDENCE_UNKNOWN,
    LIFECYCLE_ACTIVE,
    HypothesisLedger,
)
from pure_integer_ai.cognition.shared.hypothesis_resolution import (
    HypothesisResolver,
)
from pure_integer_ai.cognition.shared.identity import (
    ObjectIdentity,
    VersionBundle,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.understanding.language_candidate import (
    ActiveSenseConsumer,
)
from pure_integer_ai.cognition.shared.training_hypothesis import (
    TrainingCandidateHistoryLog,
    TrainingHypothesisEventSink,
)
from pure_integer_ai.experiments.ph2_dataset_contract import StableRecordKey
from pure_integer_ai.experiments.ph2_w03_adapter import (
    W03EvidenceBinding,
    W03SenseCandidateEnvelope,
    W03TypedAdapterOutput,
)
from pure_integer_ai.experiments.ph2_w03_understanding_contract import (
    W03_UNDERSTANDING_AMBIGUOUS,
    W03_UNDERSTANDING_CLARIFY,
    W03_UNDERSTANDING_UNIQUE,
    W03_UNDERSTANDING_UNKNOWN,
    W03EvidenceAccount,
    W03EvidenceApplication,
    W03SenseResolution,
    W03UnderstandingError,
    W03UnderstandingReport,
    _W03_NAMESPACE,
    _W03_STATUS_CODES,
    _candidate_engine,
    _event_key,
    _history_protocol,
    _projection_protocol,
    _teacher_source,
    _verifier,
)
from pure_integer_ai.storage.edge_store import EPI_STRUCTURED, SOURCE_BARE_TEXT


class W03UnderstandingRuntime:
    """消费 W03-02 输出并编排现有候选 runtime 与 active Sense consumer。"""

    def __init__(
            self,
            output: W03TypedAdapterOutput,
            ontology: GraphOntology,
            *,
            history: TrainingCandidateHistoryLog | None = None,
            restore: bool = False,
            ) -> None:
        if not isinstance(output, W03TypedAdapterOutput):
            raise TypeError("understanding 只接受 W03TypedAdapterOutput")
        if not isinstance(ontology, GraphOntology):
            raise TypeError("ontology 必须是 GraphOntology")
        if history is not None and not isinstance(
                history, TrainingCandidateHistoryLog):
            raise TypeError("understanding history 类型非法")
        if type(restore) is not bool:
            raise TypeError("understanding restore 必须是严格 bool")
        if restore and history is None:
            raise W03UnderstandingError("恢复 understanding 必须提供持久 history")
        self._validate_output(output)
        self.output = output
        self.graph = CandidateProjectionGraph(
            ontology, _projection_protocol())
        self.consumer = ActiveSenseConsumer(
            self.graph, output.sense_protocol)
        self._candidate_by_sense = {
            item.sense: item for item in output.candidates}
        self._candidates_by_observation: dict[
            StableRecordKey, tuple[W03SenseCandidateEnvelope, ...]
        ] = {
            item.observation.stable_key: tuple(sorted(
                item.candidates,
                key=lambda candidate: candidate.sense.stable_key(),
            ))
            for item in output.observations
        }
        self._binding_by_key = {
            item.teacher_record.stable_key: item for item in output.evidence}
        self._applications: dict[StableRecordKey, W03EvidenceApplication] = {}
        self._accounts: dict[ObjectIdentity, list[W03EvidenceAccount]] = {
            item.sense: [] for item in output.candidates}
        self._unbound_evidence: list[W03EvidenceBinding] = []
        self._supersedes: dict[ObjectIdentity, StableRecordKey] = {}
        by_versions: dict[
            VersionBundle, list[W03SenseCandidateEnvelope]
        ] = {}
        for candidate in output.candidates:
            by_versions.setdefault(
                candidate.sense.versions, []).append(candidate)
        runtimes: list[CandidateLearningRuntime] = []
        self._runtime_by_sense: dict[
            ObjectIdentity, CandidateLearningRuntime] = {}
        for versions in sorted(
                by_versions, key=VersionBundle.stable_key):
            candidates = tuple(sorted(
                by_versions[versions],
                key=lambda item: item.definition.stable_key(),
            ))
            engine = _candidate_engine(versions)
            metadata = CandidateProjectionMetadata(
                SOURCE_BARE_TEXT,
                EPI_STRUCTURED,
            )
            if history is None:
                runtime = CandidateLearningRuntime(
                    engine, self.graph, _verifier(), metadata)
            else:
                history_protocol = _history_protocol(versions)
                sink = TrainingHypothesisEventSink(
                    history, history_protocol)
                existing = sink.hypotheses()
                if restore:
                    if not existing:
                        raise W03UnderstandingError(
                            "恢复 understanding 缺持久候选历史")
                    runtime = CandidateLearningRuntime.restore_for_training_graph(
                        engine.protocol,
                        self.graph,
                        _verifier(),
                        metadata,
                        history,
                        history_protocol,
                    )
                else:
                    if existing:
                        raise W03UnderstandingError(
                            "fresh understanding 已存在候选历史")
                    ledger = HypothesisLedger(sink)
                    resolver = HypothesisResolver(ledger, sink=sink)
                    runtime = CandidateLearningRuntime(
                        type(engine)(
                            engine.protocol,
                            ledger=ledger,
                            resolver=resolver,
                        ),
                        self.graph,
                        _verifier(),
                        metadata,
                    )
            if restore:
                restored_definitions = tuple(sorted(
                    runtime.engine.definitions(),
                    key=lambda item: item.stable_key(),
                ))
                if restored_definitions != tuple(
                        item.definition for item in candidates):
                    raise W03UnderstandingError(
                        "恢复 candidate definition 与 typed output 漂移")
            else:
                runtime.register_many(tuple(
                    (item.definition, ordinal)
                    for ordinal, item in enumerate(candidates, start=1)
                ))
            runtimes.append(runtime)
            for candidate in candidates:
                self._runtime_by_sense[candidate.sense] = runtime
        self.candidate_runtimes = tuple(runtimes)
        if restore:
            self._restore_domain_accounts()

    def _restore_domain_accounts(self) -> None:
        """从无损 H-00/H-04 历史重建 W-03 Evidence 分账和依赖索引。"""
        for binding in sorted(
                self.output.evidence,
                key=lambda item: (
                    item.logical_order,
                    item.observation.stable_key.stable_key(),
                    item.teacher_record.stable_key.stable_key(),
                )):
            candidates = tuple(
                self._candidate_by_sense[sense]
                for sense in sorted(
                    binding.candidates,
                    key=ObjectIdentity.stable_key,
                )
            )
            expected: list[tuple[
                W03SenseCandidateEnvelope, int, int, bool
            ]] = []
            for candidate in candidates:
                expected.extend(
                    (candidate, stance, ordinal, False)
                    for ordinal, stance in enumerate(
                        binding.stances, start=1)
                )
            superseded: list[ObjectIdentity] = []
            if binding.supersedes_observation_key is not None:
                old_candidates = tuple(sorted(
                    self.candidate_for_observation(
                        binding.supersedes_observation_key),
                    key=lambda item: item.sense.stable_key(),
                ))
                expected.extend(
                    (candidate, EVIDENCE_REFUTE, ordinal, True)
                    for ordinal, candidate in enumerate(
                        old_candidates, start=1)
                )
                superseded.extend(item.sense for item in old_candidates)
                for candidate in candidates:
                    self._supersedes[candidate.sense] = (
                        binding.supersedes_observation_key)
            accounts = []
            for candidate, stance, stance_ordinal, derived in expected:
                runtime = self.candidate_runtime_for(candidate.sense)
                hypothesis = runtime.hypothesis_for_candidate(candidate.sense)
                event_key = _event_key(
                    binding,
                    candidate,
                    stance=stance,
                    stance_ordinal=stance_ordinal,
                    derived_supersede=derived,
                )
                matches = tuple(
                    item for item in runtime.engine.recognition_history(
                        hypothesis)
                    if item.prediction.event_key == event_key
                )
                if len(matches) != 1:
                    raise W03UnderstandingError(
                        "恢复 Evidence account 缺唯一 recognition")
                record = matches[0]
                decisions = tuple(
                    item for item in runtime.engine.resolver.decision_history(
                        hypothesis)
                    if item.timestamp_seq == record.evidence.timestamp_seq + 1
                )
                if len(decisions) != 1:
                    raise W03UnderstandingError(
                        "恢复 Evidence account 缺唯一 H-04 decision")
                candidate_ref = runtime.graph.ontology.resolve(candidate.sense)
                projection = None
                if candidate_ref is not None:
                    matching_events = tuple(
                        item for item in runtime.graph.history(candidate_ref)
                        if (item.definition.decision_key
                            == decisions[0].stable_key()
                            and item.definition.timestamp_seq
                            == record.evidence.timestamp_seq + 2)
                    )
                    if matching_events:
                        projection = runtime.graph.project(candidate_ref)
                outcome = CandidateLearningOutcome(
                    record.prediction,
                    record.verification,
                    record.evidence,
                    decisions[0],
                    projection,
                )
                account = W03EvidenceAccount(
                    binding.teacher_record,
                    candidate.sense,
                    stance,
                    record.prediction.observation,
                    record.prediction.scope,
                    event_key,
                    outcome,
                    derived,
                )
                accounts.append(account)
                self._accounts[candidate.sense].append(account)
            if not candidates:
                self._unbound_evidence.append(binding)
            self._applications[binding.teacher_record.stable_key] = (
                W03EvidenceApplication(
                    binding,
                    tuple(accounts),
                    None,
                    tuple(sorted(
                        superseded,
                        key=ObjectIdentity.stable_key,
                    )),
                )
            )

    @staticmethod
    def _validate_output(output: W03TypedAdapterOutput) -> None:
        """在任何图写前核验 adapter 输出的 candidate/Evidence 闭包。"""
        if dict(output.execution_state) != {
                "LANGUAGE_CAPABILITY_MASTERED": 0,
                "LANGUAGE_READINESS": 0,
                "W03_STARTED": 0,
                "W04_STARTED": 0,
                "formal_w03_training_runs": 0,
                "learning_writes": 0,
                "teacher_calls": 0,
                }:
            raise W03UnderstandingError("adapter execution state 非零或不完整")
        by_sense = {item.sense: item for item in output.candidates}
        if len(by_sense) != len(output.candidates):
            raise W03UnderstandingError("adapter candidate Sense identity 重复")
        flattened = tuple(
            candidate
            for observation in output.observations
            for candidate in observation.candidates
        )
        if ({item.sense for item in flattened} != set(by_sense)
                or len(flattened) != len(output.candidates)):
            raise W03UnderstandingError("Observation 与 candidate inventory 不闭合")
        competition_versions: dict[tuple[int, ...], VersionBundle] = {}
        for candidate in output.candidates:
            prior = competition_versions.get(candidate.competition_key)
            if prior is not None and prior != candidate.sense.versions:
                raise W03UnderstandingError("同一 competition 跨 candidate version")
            competition_versions[candidate.competition_key] = (
                candidate.sense.versions)
        observation_keys = {
            item.observation.stable_key for item in output.observations}
        if (len(observation_keys) != len(output.observations)
                or len({item.teacher_record.stable_key
                        for item in output.evidence}) != len(output.evidence)):
            raise W03UnderstandingError("Observation 或 teacher Evidence identity 重复")
        if {item.observation.stable_key for item in output.evidence} != (
                observation_keys):
            raise W03UnderstandingError("Observation 与 Evidence binding 不闭合")
        for binding in output.evidence:
            expected_candidates = tuple(
                item.sense for item in flattened
                if item.observation.stable_key
                == binding.observation.stable_key
            )
            if set(binding.candidates) != set(expected_candidates):
                raise W03UnderstandingError("Evidence candidate binding 漂移")
            if binding.supersedes_observation_key is None:
                continue
            if (binding.observation.supersedes_key
                    != binding.supersedes_observation_key):
                raise W03UnderstandingError("supersede Observation identity 漂移")
            old_candidates = tuple(
                item for item in flattened
                if item.observation.stable_key
                == binding.supersedes_observation_key
            )
            if not expected_candidates or not old_candidates:
                raise W03UnderstandingError("supersede dependency 缺新旧 Sense")
            if (len({by_sense[item].anchor.atom
                     for item in expected_candidates}) != 1
                    or len({item.anchor.atom for item in old_candidates}) != 1):
                raise W03UnderstandingError("supersede dependency 含多个 LanguageAtom")

    @property
    def unbound_evidence(self) -> tuple[W03EvidenceBinding, ...]:
        """返回没有伪造 Sense 的 anti-literal/redirect Evidence。"""
        return tuple(self._unbound_evidence)

    def candidate_for_observation(
            self,
            observation_key: StableRecordKey,
            ) -> tuple[W03SenseCandidateEnvelope, ...]:
        """按完整 Observation stable key 回读全部 Sense 候选。"""
        if not isinstance(observation_key, StableRecordKey):
            raise TypeError("observation_key 必须是 StableRecordKey")
        try:
            return self._candidates_by_observation[observation_key]
        except KeyError as exc:
            raise KeyError("未知 W-03 Observation") from exc

    def candidate_runtime_for(
            self,
            candidate: ObjectIdentity,
            ) -> CandidateLearningRuntime:
        """按完整 Sense identity 返回其 version-isolated 候选 owner。"""
        if not isinstance(candidate, ObjectIdentity):
            raise TypeError("candidate 必须是 ObjectIdentity")
        try:
            return self._runtime_by_sense[candidate]
        except KeyError as exc:
            raise KeyError("未知 W-03 Sense candidate") from exc

    def candidate_runtime_state_key(self) -> tuple:
        """返回全部 version bucket runtime 的规范隔离状态。"""
        return tuple(
            runtime.state_key() for runtime in self.candidate_runtimes)

    def _recognize(
            self,
            candidate: W03SenseCandidateEnvelope,
            binding: W03EvidenceBinding,
            *,
            stance: int,
            stance_ordinal: int,
            derived_supersede: bool,
            archive_refuted: bool,
            ) -> W03EvidenceAccount:
        """把一个显式 stance 送入既有 prediction/reveal/resolve/projection 链。"""
        source = _teacher_source(binding)
        scope = document_scope(source)
        event_key = _event_key(
            binding,
            candidate,
            stance=stance,
            stance_ordinal=stance_ordinal,
            derived_supersede=derived_supersede,
        )
        supported = (
            (candidate.concept,) if stance == EVIDENCE_SUPPORT else ())
        refuted = (
            (candidate.concept,) if stance == EVIDENCE_REFUTE else ())
        if stance not in {
                EVIDENCE_SUPPORT, EVIDENCE_REFUTE, EVIDENCE_UNKNOWN}:
            raise W03UnderstandingError("Evidence stance 非法")
        runtime = self.candidate_runtime_for(candidate.sense)
        evidence_seq, decision_seq, projection_seq = (
            runtime.next_timestamps(3))
        hypothesis = runtime.hypothesis_for_candidate(candidate.sense)
        outcome = runtime.recognize(
            hypothesis,
            observation=source,
            scope=scope,
            event_key=event_key,
            visible_inputs=(
                candidate.anchor.atom,
                candidate.anchor.span,
                candidate.context,
            ),
            predicted=candidate.concept,
            revealed=RevealedObjectObservation(
                source,
                scope,
                event_key,
                source,
                supported,
                refuted,
                (
                    _W03_NAMESPACE,
                    binding.logical_order,
                    binding.withdrawal_level,
                ),
            ),
            timestamp_seq=evidence_seq,
            resolve_timestamp_seq=decision_seq,
            projection_timestamp_seq=projection_seq,
            archive_refuted=archive_refuted,
        )
        account = W03EvidenceAccount(
            binding.teacher_record,
            candidate.sense,
            stance,
            source,
            scope,
            event_key,
            outcome,
            derived_supersede,
        )
        self._accounts[candidate.sense].append(account)
        return account

    def apply_evidence(
            self,
            binding: W03EvidenceBinding,
            ) -> W03EvidenceApplication:
        """消费一个正式 binding，并让跨 context revision 定向退出旧 Sense。"""
        if not isinstance(binding, W03EvidenceBinding):
            raise TypeError("binding 必须是 W03EvidenceBinding")
        key = binding.teacher_record.stable_key
        canonical = self._binding_by_key.get(key)
        if canonical != binding:
            raise W03UnderstandingError("Evidence binding 不属于当前 adapter output")
        if key in self._applications:
            raise W03UnderstandingError("同一 Evidence binding 不得重复消费")
        candidates = tuple(
            self._candidate_by_sense[sense]
            for sense in sorted(
                binding.candidates,
                key=ObjectIdentity.stable_key,
            )
        )
        accounts: list[W03EvidenceAccount] = []
        for candidate in candidates:
            for stance_ordinal, stance in enumerate(binding.stances, start=1):
                accounts.append(self._recognize(
                    candidate,
                    binding,
                    stance=stance,
                    stance_ordinal=stance_ordinal,
                    derived_supersede=False,
                    archive_refuted=False,
                ))
        if not candidates:
            self._unbound_evidence.append(binding)

        before_supersede = None
        superseded: list[ObjectIdentity] = []
        if binding.supersedes_observation_key is not None:
            old_candidates = self.candidate_for_observation(
                binding.supersedes_observation_key)
            if not candidates or not old_candidates:
                raise W03UnderstandingError("supersede dependency 缺新旧 Sense")
            new_atoms = {item.anchor.atom for item in candidates}
            old_atoms = {item.anchor.atom for item in old_candidates}
            if len(new_atoms) != 1 or len(old_atoms) != 1:
                raise W03UnderstandingError("supersede dependency 含多个 LanguageAtom")
            if old_atoms == new_atoms:
                before_supersede = self.resolve(next(iter(new_atoms)))
            for stance_ordinal, candidate in enumerate(
                    sorted(
                        old_candidates,
                        key=lambda item: item.sense.stable_key(),
                    ), start=1):
                accounts.append(self._recognize(
                    candidate,
                    binding,
                    stance=EVIDENCE_REFUTE,
                    stance_ordinal=stance_ordinal,
                    derived_supersede=True,
                    archive_refuted=True,
                ))
                superseded.append(candidate.sense)
            for candidate in candidates:
                self._supersedes[candidate.sense] = (
                    binding.supersedes_observation_key)

        application = W03EvidenceApplication(
            binding,
            tuple(accounts),
            before_supersede,
            tuple(sorted(superseded, key=ObjectIdentity.stable_key)),
        )
        self._applications[key] = application
        return application

    def apply_all_evidence(self) -> tuple[W03EvidenceApplication, ...]:
        """按 logical order 和完整 stable key 消费全部正式 train Evidence。"""
        if self._applications:
            raise W03UnderstandingError("批量 Evidence 只能从未消费状态开始")
        ordered = tuple(sorted(
            self.output.evidence,
            key=lambda item: (
                item.logical_order,
                item.observation.stable_key.stable_key(),
                item.teacher_record.stable_key.stable_key(),
            ),
        ))
        return tuple(self.apply_evidence(item) for item in ordered)

    def evidence_accounts(
            self,
            candidate: ObjectIdentity,
            ) -> tuple[W03EvidenceAccount, ...]:
        """按 Sense identity 回读全部 teacher/derived Evidence 分账。"""
        if not isinstance(candidate, ObjectIdentity):
            raise TypeError("candidate 必须是 ObjectIdentity")
        try:
            return tuple(self._accounts[candidate])
        except KeyError as exc:
            raise KeyError("未知 W-03 Sense candidate") from exc

    def supersedes_observation(
            self,
            candidate: ObjectIdentity,
            ) -> StableRecordKey | None:
        """回读新 Sense 保留的 observation-level supersede 依赖。"""
        if candidate not in self._candidate_by_sense:
            raise KeyError("未知 W-03 Sense candidate")
        return self._supersedes.get(candidate)

    def resolve(
            self,
            atom: ObjectIdentity,
            *,
            context: ObjectIdentity | None = None,
            ) -> W03SenseResolution:
        """从 active typed 图决断 unique/ambiguous/clarify/unknown。"""
        if not isinstance(atom, ObjectIdentity):
            raise TypeError("atom 必须是 ObjectIdentity")
        matching = tuple(sorted(
            (
                item for item in self.output.candidates
                if (item.anchor.atom == atom
                    and (context is None or item.context == context))
            ),
            key=lambda item: item.sense.stable_key(),
        ))
        viable = []
        for item in matching:
            runtime = self.candidate_runtime_for(item.sense)
            hypothesis = runtime.hypothesis_for_candidate(item.sense)
            snapshot = runtime.engine.ledger.snapshot(hypothesis)
            if (snapshot.lifecycle == LIFECYCLE_ACTIVE
                    and snapshot.epistemic_status != EPISTEMIC_REFUTED):
                viable.append(item)
        candidates = tuple(viable)
        active = self.consumer.lookup(atom, context=context)
        if len(active) == 1:
            status = W03_UNDERSTANDING_UNIQUE
            selected = active[0]
        elif len(active) > 1:
            contexts = {item.context for item in active}
            status = (
                W03_UNDERSTANDING_CLARIFY
                if context is None and len(contexts) > 1
                else W03_UNDERSTANDING_AMBIGUOUS
            )
            selected = None
        elif len(candidates) > 1:
            contexts = {item.context for item in candidates}
            status = (
                W03_UNDERSTANDING_CLARIFY
                if context is None and len(contexts) > 1
                else W03_UNDERSTANDING_AMBIGUOUS
            )
            selected = None
        else:
            status = W03_UNDERSTANDING_UNKNOWN
            selected = None
        return W03SenseResolution(
            status,
            atom,
            context,
            candidates,
            active,
            selected,
            status in {
                W03_UNDERSTANDING_AMBIGUOUS,
                W03_UNDERSTANDING_CLARIFY,
            },
            (_W03_NAMESPACE, 300 + _W03_STATUS_CODES[status]),
        )

    def report(self) -> W03UnderstandingReport:
        """从 runtime 历史派生闭环计数，同时保留全零阶段状态。"""
        runtime_reports = tuple(
            runtime.report() for runtime in self.candidate_runtimes)
        conflicts = 0
        for candidate in self.output.candidates:
            runtime = self.candidate_runtime_for(candidate.sense)
            hypothesis = runtime.hypothesis_for_candidate(candidate.sense)
            if runtime.engine.ledger.snapshot(
                    hypothesis).epistemic_status == EPISTEMIC_CONFLICTED:
                conflicts += 1
        return W03UnderstandingReport(
            sum(item.candidate_count for item in runtime_reports),
            len(self._applications),
            sum(item.prediction_count for item in runtime_reports),
            len(self._unbound_evidence),
            sum(item.active_projection_count for item in runtime_reports),
            conflicts,
            self.output.execution_state,
        )


def build_w03_understanding_runtime(
        output: W03TypedAdapterOutput,
        ontology: GraphOntology,
        *,
        history: TrainingCandidateHistoryLog | None = None,
        restore: bool = False,
        ) -> W03UnderstandingRuntime:
    """建立独立 W03-03 runtime；调用本身只形成候选，不采用 Sense。"""
    return W03UnderstandingRuntime(
        output,
        ontology,
        history=history,
        restore=restore,
    )


__all__ = [
    "W03_UNDERSTANDING_AMBIGUOUS",
    "W03_UNDERSTANDING_CLARIFY",
    "W03_UNDERSTANDING_UNIQUE",
    "W03_UNDERSTANDING_UNKNOWN",
    "W03EvidenceAccount",
    "W03EvidenceApplication",
    "W03SenseResolution",
    "W03UnderstandingError",
    "W03UnderstandingReport",
    "W03UnderstandingRuntime",
    "build_w03_understanding_runtime",
]
