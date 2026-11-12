"""Bind trained integer correction cues to one preceding Memory target."""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EVIDENCE_UNKNOWN,
)
from pure_integer_ai.cognition.shared.identity import SourceRef
from pure_integer_ai.cognition.understanding.query_input_structure import (
    QueryInputStructure,
    STRUCTURE_ORDER_CONFLICT,
)
from pure_integer_ai.cognition.understanding.query_memory_candidates import (
    MEMORY_CATEGORY_DISCOURSE,
    memory_candidate_category,
    memory_input_candidate_keys,
    memory_topic_candidate,
)
from pure_integer_ai.experiments.trained_dialogue_memory_graph import (
    MemoryActiveStructures,
    MemoryEvidenceStructure,
    MemoryHypothesisStructure,
    MemoryObservationStructure,
)
from pure_integer_ai.experiments.trained_revision_cue_graph import (
    RevisionCueGraphRecord,
)
from pure_integer_ai.cognition.understanding.query_assertion_scope import TrainedAssertionScopeFrame
from pure_integer_ai.experiments.natural_revision_scope import (
    read_revision_scope, require_mounted_revision_scope, revision_scope_stance,
)


REVISION_REASON_NAMESPACE = (91692, 1)
REVISION_DETAIL_NAMESPACE = (91692, 2)


class NaturalRevisionBindingError(ValueError):
    """更正拒绝的规范整数原因：1结构，2缺目标，4不支持的作用域。"""

    def __init__(self, message: str, *, reason_code: int = 1) -> None:
        """保留原异常接口，以有界整数记录可跨语言恢复的失败类别。"""
        if type(reason_code) is not int or reason_code not in (1, 2, 4):
            raise ValueError("invalid revision rejection code")
        super().__init__(message)
        self.reason_code = reason_code


def _pack(key: tuple[int, ...]) -> tuple[int, ...]:
    return (len(key), *key)


def _strict_key(value: tuple[int, ...], *, label: str) -> None:
    if (type(value) is not tuple or not value
            or any(type(item) is not int or item < 0 for item in value)):
        raise ValueError(f"{label} must be a nonempty nonnegative integer tuple")


@dataclass(frozen=True, slots=True)
class NaturalRevisionInputBinding:
    cue_identity: tuple[int, ...]
    action_key: tuple[int, ...]
    proposition_key: tuple[int, ...]
    relation_key: tuple[int, ...]
    input_source_ref: tuple[int, ...]
    cue_start: int
    cue_end: int
    source_identity: tuple[int, ...]
    license_identity: tuple[int, ...]
    replacement_stance: int = EVIDENCE_SUPPORT
    scope_binding_key: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for label in (
                "cue_identity", "action_key", "proposition_key", "relation_key",
                "input_source_ref", "source_identity", "license_identity"):
            _strict_key(getattr(self, label), label=label)
        if (type(self.cue_start) is not int or type(self.cue_end) is not int
                or self.cue_start < 0 or self.cue_end <= self.cue_start):
            raise ValueError("revision cue interval is invalid")
        if type(self.replacement_stance) is not int:
            raise ValueError("revision stance must be a strict integer")
        if self.replacement_stance == EVIDENCE_REFUTE:
            read_revision_scope(self.scope_binding_key, relation_key=self.relation_key,
                                source_ref=self.input_source_ref)
        elif self.replacement_stance != EVIDENCE_SUPPORT or self.scope_binding_key:
            raise ValueError("revision stance lacks a supported scope contract")

    def stable_key(self) -> tuple[int, ...]:
        original = (
            1,
            *_pack(self.cue_identity),
            *_pack(self.action_key),
            *_pack(self.proposition_key),
            *_pack(self.relation_key),
            *_pack(self.input_source_ref),
            self.cue_start,
            self.cue_end,
            *_pack(self.source_identity),
            *_pack(self.license_identity),
        )
        if self.replacement_stance == EVIDENCE_SUPPORT:
            return original
        return (4, *_pack(original), self.replacement_stance,
                *_pack(self.scope_binding_key))


@dataclass(frozen=True, slots=True)
class NaturalRevisionTargetBinding:
    input_binding: NaturalRevisionInputBinding
    prior_observation_key: tuple[int, ...]
    prior_hypothesis_key: tuple[int, ...]
    prior_evidence_key: tuple[int, ...]
    prior_source_ref: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.input_binding, NaturalRevisionInputBinding):
            raise TypeError("input_binding must be a NaturalRevisionInputBinding")
        for label in (
                "prior_observation_key", "prior_hypothesis_key",
                "prior_evidence_key", "prior_source_ref"):
            _strict_key(getattr(self, label), label=label)

    def stable_key(self) -> tuple[int, ...]:
        return (
            2,
            *_pack(self.input_binding.stable_key()),
            *_pack(self.prior_observation_key),
            *_pack(self.prior_hypothesis_key),
            *_pack(self.prior_evidence_key),
            *_pack(self.prior_source_ref),
        )


@dataclass(frozen=True, slots=True)
class NaturalRevisionCommit:
    target_binding: NaturalRevisionTargetBinding
    correction_source_ref: tuple[int, ...]
    replacement_evidence_key: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.target_binding, NaturalRevisionTargetBinding):
            raise TypeError("target_binding must be a NaturalRevisionTargetBinding")
        _strict_key(self.correction_source_ref, label="correction_source_ref")
        _strict_key(self.replacement_evidence_key,
                    label="replacement_evidence_key")

    def stable_key(self) -> tuple[int, ...]:
        return (
            3,
            *_pack(self.target_binding.stable_key()),
            *_pack(self.correction_source_ref),
            *_pack(self.replacement_evidence_key),
        )


class NaturalRevisionRuntime:
    """Pure projection plus bounded target resolution over Interaction Memory."""

    def __init__(self, records: tuple[RevisionCueGraphRecord, ...], *,
                 scope_frames: tuple[TrainedAssertionScopeFrame, ...] = ()) -> None:
        if (type(records) is not tuple or not records
                or any(not isinstance(item, RevisionCueGraphRecord)
                       for item in records)
                or records != tuple(sorted(
                    set(records), key=lambda item: item.stable_key()))):
            raise ValueError("revision cue records must be sorted unique graph records")
        sources = {item.source_identity for item in records}
        licenses = {item.license_identity for item in records}
        if len(sources) != 1 or len(licenses) != 1:
            raise ValueError("revision cue graph provenance must be uniform")
        self.records = records
        self.source_identity = next(iter(sources))
        self.license_identity = next(iter(licenses))
        if (type(scope_frames) is not tuple
                or any(not isinstance(item, TrainedAssertionScopeFrame) for item in scope_frames)
                or len(set(scope_frames)) != len(scope_frames)):
            raise ValueError("revision scopes must be unique trained graph records")
        self.scope_frames = scope_frames

    def bind_input(
            self,
            input_structure: QueryInputStructure,
            *,
            memory_input_structure: QueryInputStructure | None = None,
            ) -> NaturalRevisionInputBinding | None:
        """Return no binding for ordinary input; reject malformed cue inputs."""
        if not isinstance(input_structure, QueryInputStructure):
            raise TypeError("input_structure must be QueryInputStructure")
        if memory_input_structure is not None and not isinstance(
                memory_input_structure, QueryInputStructure):
            raise TypeError("memory_input_structure must be QueryInputStructure")
        values = input_structure.token_values
        matches = []
        for record in self.records:
            cue = record.cue.surface_values
            width = len(cue)
            matches.extend(
                (record, start, start + width)
                for start in range(len(values) - width + 1)
                if values[start:start + width] == cue
            )
        if not matches:
            return None
        if len(matches) != 1:
            raise NaturalRevisionBindingError(
                "revision input must contain exactly one trained cue")

        record, cue_start, cue_end = matches[0]
        try:
            stance, scope_key = revision_scope_stance(self.scope_frames, input_structure)
        except ValueError as error:
            raise NaturalRevisionBindingError(str(error), reason_code=4) from error
        relations = input_structure.relation_candidates
        if scope_key:
            scope = read_revision_scope(scope_key, relation_key=input_structure.assertion_scope_bindings[0].relation_key,
                                        source_ref=input_structure.source_ref)
            relations = tuple(item for item in relations if item.stable_key() == scope.relation_key)
        if len(relations) != 1:
            raise NaturalRevisionBindingError(
                "revision cue must bind exactly one proposition candidate")
        relation = relations[0]
        if (relation.state == STRUCTURE_ORDER_CONFLICT
                or not relation.proposition_key or not relation.span_refs):
            raise NaturalRevisionBindingError(
                "revision proposition is incomplete or conflicting")
        spans = {item.ref_key: item for item in input_structure.spans}
        relation_spans = tuple(spans.get(key) for key in relation.span_refs)
        if any(item is None for item in relation_spans):
            raise NaturalRevisionBindingError(
                "revision proposition references a missing input span")
        if cue_end > min(item.start for item in relation_spans):
            raise NaturalRevisionBindingError(
                "revision cue is not structurally before its proposition")
        if memory_input_structure is not None:
            memory_relations = tuple(
                item for item in memory_input_structure.relation_candidates
                if item.proposition_key == relation.proposition_key
                and item.stable_key() == relation.stable_key())
            if len(memory_relations) != 1:
                raise NaturalRevisionBindingError(
                    "revision proposition is outside the authorized Memory view")
        return NaturalRevisionInputBinding(
            record.cue.cue_identity,
            record.cue.action_key,
            relation.proposition_key,
            relation.stable_key(),
            input_structure.source_ref,
            cue_start,
            cue_end,
            record.source_identity,
            record.license_identity,
            stance,
            scope_key,
        )

    def resolve_target(
            self,
            memory,
            input_structure: QueryInputStructure,
            input_binding: NaturalRevisionInputBinding,
            ) -> NaturalRevisionTargetBinding:
        """Resolve one latest matching user Hypothesis and its active Evidence."""
        if not isinstance(input_binding, NaturalRevisionInputBinding):
            raise TypeError("input_binding must be NaturalRevisionInputBinding")
        if self.bind_input(input_structure) != input_binding:
            raise NaturalRevisionBindingError("revision binding differs from its actual input")
        if memory is None or not hasattr(memory, "active_structures"):
            raise NaturalRevisionBindingError("revision requires Interaction Memory")
        candidate_keys = memory_input_candidate_keys(
            input_structure,
            session_id=memory.session_id,
            speaker_kind=1,
        )
        snapshot = memory.active_structures(candidate_keys=candidate_keys)
        target = self._target_from_snapshot(snapshot, input_binding,
                                            next_turn=memory._next_turn_seq)
        return target

    @staticmethod
    def _target_from_snapshot(
            snapshot: MemoryActiveStructures,
            input_binding: NaturalRevisionInputBinding,
            *, next_turn: int,
            ) -> NaturalRevisionTargetBinding:
        if not isinstance(snapshot, MemoryActiveStructures):
            raise TypeError("snapshot must be MemoryActiveStructures")
        observations = {item.observation_key: item
                       for item in snapshot.observations}
        matching = []
        for hypothesis in snapshot.hypotheses:
            observation = observations.get(hypothesis.observation_key)
            if observation is None or observation.speaker_kind != 1:
                continue
            if observation.turn_seq >= next_turn:
                continue
            if memory_candidate_category(hypothesis.candidate_key) != MEMORY_CATEGORY_DISCOURSE:
                continue
            topic = memory_topic_candidate(hypothesis.candidate_key)
            if topic is None or topic.proposition_key != input_binding.proposition_key:
                continue
            linked = {item.evidence_key: item for item in snapshot.evidence
                      if item.hypothesis_key == hypothesis.hypothesis_key}
            if not linked or set(hypothesis.evidence_keys) != set(linked):
                raise NaturalRevisionBindingError("revision target Evidence closure is incomplete")
            # 只有来源一致、从未断言且无替代边的初始UNKNOWN不构成用户断言。
            # 反驳、竞争或替代后的UNKNOWN仍进入原最新目标裁决，不得跳过。
            if all(item.stance == EVIDENCE_UNKNOWN and not item.supersedes_key
                   and item.source_ref == observation.source_ref for item in linked.values()):
                continue
            matching.append((observation, hypothesis))
        if not matching:
            raise NaturalRevisionBindingError(
                "revision has no preceding Memory proposition target", reason_code=2)
        latest_turn = max(item[0].turn_seq for item in matching)
        latest = tuple(item for item in matching
                       if item[0].turn_seq == latest_turn)
        if len(latest) != 1:
            raise NaturalRevisionBindingError(
                "revision target is ambiguous in the preceding Memory context")
        observation, hypothesis = latest[0]
        evidence_by_key = {
            item.evidence_key: item for item in snapshot.evidence
            if item.hypothesis_key == hypothesis.hypothesis_key
        }
        if set(hypothesis.evidence_keys) != set(evidence_by_key):
            raise NaturalRevisionBindingError(
                "revision target Evidence closure is incomplete")
        superseded = {
            item.supersedes_key for item in evidence_by_key.values()
            if item.supersedes_key
        }
        leaves = tuple(
            item for item in evidence_by_key.values()
            if item.evidence_key not in superseded)
        if (len(leaves) != 1 or leaves[0].stance != EVIDENCE_SUPPORT
                or any(item.stance == EVIDENCE_REFUTE for item in leaves)):
            raise NaturalRevisionBindingError(
                "revision target has competing or unsupported Evidence")
        target_evidence = leaves[0]
        return NaturalRevisionTargetBinding(
            input_binding,
            observation.observation_key,
            hypothesis.hypothesis_key,
            target_evidence.evidence_key,
            target_evidence.source_ref,
        )

    def commit(
            self,
            memory,
            target_binding: NaturalRevisionTargetBinding,
            correction_source: SourceRef,
            ) -> NaturalRevisionCommit:
        """Append one source-qualified superseder after all gates have passed."""
        if not isinstance(target_binding, NaturalRevisionTargetBinding):
            raise TypeError("target_binding must be NaturalRevisionTargetBinding")
        if not isinstance(correction_source, SourceRef):
            raise TypeError("correction_source must be SourceRef")
        binding = target_binding.input_binding
        if binding.replacement_stance == EVIDENCE_REFUTE:
            try:
                require_mounted_revision_scope(self.scope_frames, binding.scope_binding_key,
                    relation_key=binding.relation_key, source_ref=binding.input_source_ref)
            except ValueError as error:
                raise NaturalRevisionBindingError(str(error), reason_code=4) from error
        if correction_source.owner != memory.owner:
            raise NaturalRevisionBindingError(
                "correction source does not belong to the current Memory owner")
        prior_source = SourceRef.from_stable_key(
            target_binding.prior_source_ref[1:])
        if (prior_source.owner != correction_source.owner
                or prior_source.source_id != correction_source.source_id
                or prior_source.document_id >= correction_source.document_id):
            raise NaturalRevisionBindingError(
                "revision target is outside the preceding session source")
        evidence = memory.append_evidence(
            target_binding.prior_hypothesis_key,
            source=correction_source,
            stance=binding.replacement_stance,
            reason_key=(
                *REVISION_REASON_NAMESPACE,
                *target_binding.input_binding.cue_identity),
            detail=(
                *REVISION_DETAIL_NAMESPACE,
                *target_binding.input_binding.proposition_key),
            supersedes_key=target_binding.prior_evidence_key,
        )
        if (evidence.source_ref[1:] != correction_source.stable_key()
                or evidence.supersedes_key != target_binding.prior_evidence_key):
            raise RuntimeError("revision Evidence lost its source or target binding")
        return NaturalRevisionCommit(
            target_binding,
            correction_source.stable_key(),
            evidence.evidence_key,
        )


__all__ = [
    "NaturalRevisionBindingError",
    "NaturalRevisionCommit",
    "NaturalRevisionInputBinding",
    "NaturalRevisionRuntime",
    "NaturalRevisionTargetBinding",
]
