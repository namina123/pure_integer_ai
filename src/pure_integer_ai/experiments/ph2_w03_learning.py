"""W-03 typed Evidence、projection 与 generation ledger 的持久编排。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from pure_integer_ai.cognition.shared.candidate_projection import (
    CandidateProjectionError,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.experiments.formal_train import make_train_context
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_w03_adapter import (
    W03SenseCandidateEnvelope,
    adapt_w03_training_payload,
)
from pure_integer_ai.experiments.ph2_w03_artifacts import (
    ARTIFACT_EVIDENCE_ACCOUNT,
    ARTIFACT_GENERATION_CHOICE,
    ARTIFACT_GENERATION_DECISION,
    ARTIFACT_GENERATION_OUTCOME,
    ARTIFACT_GENERATION_USE,
    ARTIFACT_PROJECTION,
    ARTIFACT_W02_RETENTION,
    W03ArtifactStore,
    persist_training_payload,
    restore_training_payload,
)
from pure_integer_ai.experiments.ph2_w03_contract import (
    W03FrozenContext,
    digest_value,
)
from pure_integer_ai.experiments.ph2_w03_generation import (
    W03_GENERATION_READY,
    W03ExpressionConstraints,
    W03GenerationRequest,
    W03GenerationRuntime,
    build_w03_generation_runtime,
)
from pure_integer_ai.experiments.ph2_w03_payload import W03TrainingPayload
from pure_integer_ai.experiments.ph2_w03_understanding import (
    W03EvidenceApplication,
    W03UnderstandingRuntime,
    build_w03_understanding_runtime,
)
from pure_integer_ai.storage.backend import StorageBackend
from pure_integer_ai.storage.training_candidate_event import (
    TRAINING_CANDIDATE_EVENT_PART_TABLE,
    TRAINING_CANDIDATE_EVENT_TABLE,
)


_EXPECTED_ARTIFACT_COUNTS = {
    ARTIFACT_EVIDENCE_ACCOUNT: 64,
    ARTIFACT_GENERATION_CHOICE: 2,
    ARTIFACT_GENERATION_DECISION: 3,
    ARTIFACT_GENERATION_OUTCOME: 4,
    ARTIFACT_GENERATION_USE: 3,
    ARTIFACT_PROJECTION: 59,
    "TRAIN_ENVELOPE": 163,
    ARTIFACT_W02_RETENTION: 1,
}


class W03LearningError(RuntimeError):
    """W-03 持久学习、结构 probe 或 artifact 回读不闭合。"""


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _key(value: object) -> list[int]:
    stable_key = getattr(value, "stable_key", None)
    if not callable(stable_key):
        raise W03LearningError("W-03 artifact 对象缺 stable_key")
    result = stable_key()
    if (not isinstance(result, tuple) or not result
            or any(type(item) is not int for item in result)):
        raise W03LearningError("W-03 artifact stable_key 非法")
    return list(result)


def _optional_key(value: object | None) -> list[int] | None:
    return None if value is None else _key(value)


def _request(
        candidate: W03SenseCandidateEnvelope,
        *,
        purpose: str,
        ) -> W03GenerationRequest:
    """只从 typed target identity 派生无答案 probe 请求。"""
    request_key = LosslessIntegerKey((
        30304,
        900,
        *digest_value({
            "purpose": purpose,
            "sense": list(candidate.sense.stable_key()),
            "source": list(candidate.source_ref.stable_key()),
        }),
    ))
    return W03GenerationRequest(
        request_key,
        candidate.sense,
        candidate.concept,
        candidate.context,
        candidate.anchor.branch,
        W03ExpressionConstraints(True, True, 64),
        candidate.source_ref,
        document_scope(candidate.source_ref),
    )


def _adopt_target(
        generation: W03GenerationRuntime,
        candidate: W03SenseCandidateEnvelope,
        *,
        purpose: str,
        ):
    choice = generation.choose(_request(candidate, purpose=purpose))
    if choice.status != W03_GENERATION_READY:
        raise W03LearningError("W-03 structural generation probe 未 READY")
    matches = tuple(
        item for item in choice.options
        if (item.sense == candidate.sense
            and item.concept == candidate.concept
            and item.context == candidate.context
            and item.branch == candidate.anchor.branch)
    )
    if len(matches) != 1:
        raise W03LearningError("W-03 target Sense 未映射到唯一精确 option")
    uses = generation.adopt(choice, (matches[0].stable_key(),))
    return choice, uses


def _run_generation_probes(
        understanding: W03UnderstandingRuntime,
        ordered_bindings,
        ) -> tuple[
            tuple[W03EvidenceApplication, ...], W03GenerationRuntime
        ]:
    """在唯一 revision 前后和唯一多 surface 组上形成真实 ledger。"""
    revisions = []
    for item in ordered_bindings:
        if item.supersedes_observation_key is None:
            continue
        old = understanding.candidate_for_observation(
            item.supersedes_observation_key)
        new = understanding.candidate_for_observation(
            item.observation.stable_key)
        if (len(old) == len(new) == 1
                and old[0].anchor.extracted.surface
                == new[0].anchor.extracted.surface):
            revisions.append(item)
    revisions = tuple(revisions)
    if len(revisions) != 1:
        raise W03LearningError(
            "W-03 train 必须有唯一同 surface Sense supersede chain")
    revision = revisions[0]
    old_candidates = understanding.candidate_for_observation(
        revision.supersedes_observation_key)
    if len(old_candidates) != 1:
        raise W03LearningError("W-03 supersede probe 旧 Sense 不唯一")

    applications = []
    generation = build_w03_generation_runtime(understanding)
    old_use = None
    for binding in ordered_bindings:
        if binding is revision:
            _choice, uses = _adopt_target(
                generation,
                old_candidates[0],
                purpose="BEFORE_SUPERSEDE",
            )
            if len(uses) != 1:
                raise W03LearningError("W-03 supersede 前 choice 必须只有一个 Use")
            old_use = uses[0]
            generation.verify_use(old_use)
        applications.append(understanding.apply_evidence(binding))
        if binding is revision:
            assert old_use is not None
            generation.verify_use(old_use)

    active_groups: dict[tuple, list[W03SenseCandidateEnvelope]] = {}
    for candidate in understanding.output.candidates:
        active = tuple(
            item for item in understanding.consumer.lookup(
                candidate.anchor.atom,
                context=candidate.context,
            )
            if item.sense == candidate.sense
        )
        if len(active) > 1:
            raise W03LearningError("W-03 active consumer 返回重复 Sense")
        if not active:
            continue
        group_key = (
            candidate.concept.stable_key(),
            candidate.context.stable_key(),
            candidate.anchor.branch,
        )
        active_groups.setdefault(group_key, []).append(candidate)
    multi = tuple(
        tuple(sorted(items, key=lambda item: item.sense.stable_key()))
        for items in active_groups.values()
        if len(items) > 1
    )
    if len(multi) != 1 or len(multi[0]) != 2:
        raise W03LearningError("W-03 train 必须产生唯一双 surface active 组")
    choice, uses = _adopt_target(
        generation,
        multi[0][0],
        purpose="MULTI_SURFACE",
    )
    if len(choice.options) != 2 or len(uses) != 2:
        raise W03LearningError("W-03 多 surface probe 未保留两个独立 option/Use")
    for use in uses:
        generation.verify_use(use)
    return tuple(applications), generation


def _account_payload(
        application: W03EvidenceApplication,
        account,
        ) -> dict[str, Any]:
    outcome = account.outcome
    projection = outcome.projection
    return {
        "before_supersede": (
            None if application.before_supersede is None
            else list(application.before_supersede.stable_key())
        ),
        "candidate": _key(account.candidate),
        "decision": _key(outcome.decision),
        "derived_supersede": int(account.derived_supersede),
        "event_key": list(account.event_key),
        "evidence": _key(outcome.evidence),
        "observation_source": _key(account.observation_source),
        "prediction": _key(outcome.prediction),
        "projection_state": (
            None if projection is None else _key(projection.state)
        ),
        "scope": _key(account.scope),
        "stance": account.stance,
        "superseded_candidates": [
            _key(item) for item in application.superseded_candidates
        ],
        "teacher_record": account.teacher_record.to_dict(),
        "verification": {
            "authority": _key(outcome.verification.authority),
            "authority_version": list(
                outcome.verification.authority_version),
            "payload_for_prediction": list(
                outcome.verification.payload_for(outcome.prediction)),
            "reason_key": list(outcome.verification.reason_key),
            "source": _key(outcome.verification.source),
            "stance": outcome.verification.stance,
            "trace": list(outcome.verification.trace),
        },
    }


def _projection_payload(
        understanding: W03UnderstandingRuntime,
        candidate: W03SenseCandidateEnvelope,
        ) -> dict[str, Any]:
    owner = understanding.candidate_runtime_for(candidate.sense)
    hypothesis = owner.hypothesis_for_candidate(candidate.sense)
    snapshot = owner.engine.ledger.snapshot(hypothesis)
    ref = understanding.graph.ontology.resolve(candidate.sense)
    projection = None
    if ref is not None:
        try:
            projection = understanding.graph.project(ref)
        except CandidateProjectionError:
            projection = None
    return {
        "candidate": _key(candidate.sense),
        "concept": _key(candidate.concept),
        "context": _key(candidate.context),
        "definition": _key(candidate.definition),
        "epistemic_status": snapshot.epistemic_status,
        "history": (
            [] if projection is None
            else [_key(item.definition.event) for item in projection.history]
        ),
        "lifecycle": snapshot.lifecycle,
        "projection_state": (
            None if projection is None else _key(projection.state)
        ),
        "refute_evidence_ids": list(snapshot.refute_evidence_ids),
        "replacement": (
            None if projection is None
            else _optional_key(projection.replacement)
        ),
        "source": _key(candidate.source_ref),
        "supersedes_observation": (
            None
            if understanding.supersedes_observation(candidate.sense) is None
            else list(understanding.supersedes_observation(
                candidate.sense).stable_key())
        ),
        "support_evidence_ids": list(snapshot.support_evidence_ids),
        "unknown_evidence_ids": list(snapshot.unknown_evidence_ids),
    }


def _option_payload(option) -> dict[str, Any]:
    return {
        "atom": _key(option.atom),
        "authorization_key": list(option.authorization_key.components),
        "branch": _key(option.branch),
        "concept": _key(option.concept),
        "context": _key(option.context),
        "lexicalized_multiword": int(option.lexicalized_multiword),
        "representation": _key(option.representation),
        "sense": _key(option.sense),
        "source": _key(option.forming_source),
        "span": _key(option.span),
        "stable_key": list(option.stable_key()),
        "surface": option.surface,
    }


def _choice_payload(choice) -> dict[str, Any]:
    request = choice.request
    return {
        "options": [_option_payload(item) for item in choice.options],
        "reason_key": list(choice.reason_key.components),
        "request": {
            "branch": _key(request.branch),
            "constraints": list(request.constraints.stable_key()),
            "context": _optional_key(request.context),
            "request_key": list(request.request_key.components),
            "scope": _key(request.scope),
            "source": _key(request.source),
            "target_concept": _key(request.target_concept),
            "target_sense": _key(request.target_sense),
        },
        "selected": (
            None if choice.selected is None else _option_payload(choice.selected)
        ),
        "stable_key": list(choice.stable_key()),
        "status": choice.status,
    }


def _persist_domain_artifacts(
        store: W03ArtifactStore,
        understanding: W03UnderstandingRuntime,
        applications: tuple[W03EvidenceApplication, ...],
        generation: W03GenerationRuntime,
        context: W03FrozenContext,
        ) -> None:
    ordinal = 0
    for application in applications:
        for account in application.accounts:
            ordinal += 1
            store.put(
                ARTIFACT_EVIDENCE_ACCOUNT,
                ordinal,
                _account_payload(application, account),
            )
    for ordinal, candidate in enumerate(
            sorted(
                understanding.output.candidates,
                key=lambda item: item.sense.stable_key(),
            ), start=1):
        store.put(
            ARTIFACT_PROJECTION,
            ordinal,
            _projection_payload(understanding, candidate),
        )
    for ordinal, choice in enumerate(generation.choices, start=1):
        store.put(
            ARTIFACT_GENERATION_CHOICE,
            ordinal,
            _choice_payload(choice),
        )
    for ordinal, decision in enumerate(generation.decisions, start=1):
        store.put(ARTIFACT_GENERATION_DECISION, ordinal, {
            "action": decision.action,
            "choice_key": list(decision.choice.stable_key()),
            "decision_key": list(decision.decision_key.components),
            "option": _option_payload(decision.option),
            "stable_key": list(decision.stable_key()),
        })
    for ordinal, use in enumerate(generation.uses, start=1):
        store.put(ARTIFACT_GENERATION_USE, ordinal, {
            "decision_key": list(use.decision.stable_key()),
            "scope": _key(use.ref.scope),
            "selection_key": list(use.ref.selection_key.components),
            "stable_key": list(use.stable_key()),
            "use_key": list(use.ref.use_key.components),
            "use_kind": use.ref.use_kind,
        })
    for ordinal, outcome in enumerate(generation.outcomes, start=1):
        store.put(ARTIFACT_GENERATION_OUTCOME, ordinal, {
            "current_authorization_key": (
                None
                if outcome.current_authorization_key is None
                else list(outcome.current_authorization_key.components)
            ),
            "dimension_key": list(outcome.ref.dimension_key.components),
            "outcome_key": list(outcome.ref.outcome_key.components),
            "result_key": list(outcome.ref.result_key.components),
            "stable_key": list(outcome.stable_key()),
            "use_key": list(outcome.ref.use_key.components),
            "verdict": outcome.verdict,
            "verifier_key": list(outcome.ref.verifier_key.components),
        })
    continuity = context.w02_continuity
    store.put(ARTIFACT_W02_RETENTION, 1, {
        "candidate_run_id": continuity.candidate_run_id,
        "continuity_key": list(continuity.stable_key()),
        "dimension_pass_counts": list(continuity.dimension_pass_counts),
        "dimension_statuses": list(continuity.dimension_statuses),
        "execution_state": dict(continuity.execution_state),
        "fail_count": continuity.fail_count,
        "formal_training_runs": continuity.formal_training_runs,
        "host_digests": dict(continuity.host_digests),
        "ne_count": continuity.ne_count,
        "receipt_identity": continuity.receipt_identity.to_dict(),
    })


def _artifact_digests(store: W03ArtifactStore) -> tuple[str, str, str]:
    projection = store.payloads(ARTIFACT_PROJECTION)
    generation = tuple(
        (kind, store.payloads(kind))
        for kind in (
            ARTIFACT_GENERATION_CHOICE,
            ARTIFACT_GENERATION_DECISION,
            ARTIFACT_GENERATION_USE,
            ARTIFACT_GENERATION_OUTCOME,
        )
    )
    retention = store.payloads(ARTIFACT_W02_RETENTION)
    return _digest(projection), _digest(generation), _digest(retention)


def _history_digest(backend: StorageBackend) -> str:
    return _digest(tuple(
        (table, tuple(tuple(sorted(row.items())) for row in backend.select(table)))
        for table in (
            TRAINING_CANDIDATE_EVENT_TABLE,
            TRAINING_CANDIDATE_EVENT_PART_TABLE,
        )
    ))


@dataclass(frozen=True)
class W03LearningResult:
    understanding: W03UnderstandingRuntime
    artifact_store: W03ArtifactStore
    candidate_history_digest: str
    projection_digest: str
    generation_digest: str
    retention_digest: str
    artifact_counts: tuple[tuple[str, int], ...]
    new_learning_write_count: int


def run_w03_learning(
        backend: StorageBackend,
        payload: W03TrainingPayload,
        context: W03FrozenContext,
        *,
        restore: bool,
        ) -> W03LearningResult:
    """fresh 时写一次；restore 时只从 Core history/artifact 恢复。"""
    if not isinstance(backend, StorageBackend):
        raise TypeError("W-03 learning backend 类型非法")
    if not isinstance(payload, W03TrainingPayload):
        raise TypeError("W-03 learning payload 类型非法")
    if not isinstance(context, W03FrozenContext):
        raise TypeError("W-03 learning context 类型非法")
    if type(restore) is not bool:
        raise TypeError("W-03 learning restore 必须是严格 bool")

    train_context = make_train_context(backend)
    history = train_context.training_candidate_history
    if history is None:
        raise W03LearningError("W-03 缺 Core training candidate history")
    store = W03ArtifactStore(backend)
    # SQLite reopen 时先注册完整 schema，再以全部已存在行作为零写基线。
    before_rows = sum(len(rows) for rows in backend.snapshot().values())
    if restore:
        stored_payload = restore_training_payload(store)
        if tuple(sorted(
                canonical_json_bytes(item.to_dict())
                for item in (
                    *stored_payload.source_refs,
                    *stored_payload.observations,
                    *stored_payload.teacher_evidence,
                ))) != tuple(sorted(
                    canonical_json_bytes(item.to_dict())
                    for item in (
                        *payload.source_refs,
                        *payload.observations,
                        *payload.teacher_evidence,
                    ))):
            raise W03LearningError("W-03 恢复 envelope 与调用 payload 漂移")
        output = adapt_w03_training_payload(stored_payload)
        understanding = build_w03_understanding_runtime(
            output,
            train_context.graph_ontology,
            history=history,
            restore=True,
        )
        new_writes = sum(
            len(rows) for rows in backend.snapshot().values()) - before_rows
        if new_writes != 0:
            raise W03LearningError("W-03 restore 产生了新的 learning write")
    else:
        persist_training_payload(store, payload)
        stored_payload = restore_training_payload(store)
        output = adapt_w03_training_payload(stored_payload)
        understanding = build_w03_understanding_runtime(
            output,
            train_context.graph_ontology,
            history=history,
        )
        ordered = tuple(sorted(
            output.evidence,
            key=lambda item: (
                item.logical_order,
                item.observation.stable_key.stable_key(),
                item.teacher_record.stable_key.stable_key(),
            ),
        ))
        applications, generation = _run_generation_probes(
            understanding, ordered)
        _persist_domain_artifacts(
            store, understanding, applications, generation, context)
        new_writes = sum(
            len(rows) for rows in backend.snapshot().values()) - before_rows
        if new_writes <= 0:
            raise W03LearningError("W-03 fresh 未产生 learning write")

    counts = store.counts()
    if dict(counts) != _EXPECTED_ARTIFACT_COUNTS:
        raise W03LearningError(
            f"W-03 artifact count 漂移: {dict(counts)!r}")
    report = understanding.report()
    if (report.candidate_count != 59
            or report.applied_observation_evidence_count != 21
            or report.unbound_evidence_count != 2
            or sum(len(understanding.evidence_accounts(item.sense))
                   for item in understanding.output.candidates) != 64):
        raise W03LearningError("W-03 restored understanding closure 漂移")
    projection, generation_digest, retention = _artifact_digests(store)
    return W03LearningResult(
        understanding,
        store,
        _history_digest(backend),
        projection,
        generation_digest,
        retention,
        counts,
        new_writes,
    )


__all__ = [
    "W03LearningError",
    "W03LearningResult",
    "run_w03_learning",
]
