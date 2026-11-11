from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
)
from pure_integer_ai.cognition.shared.memory_event import EvidencePayload
from pure_integer_ai.experiments.trained_dialogue_memory_graph import (
    TrainedDialogueMemoryGraph,
)


def test_memory_supersedes_from_a_new_turn_keeps_its_source_and_advances_target_clock(
        tmp_path):
    path = tmp_path / "cross-source-revision.sqlite3"
    surface = "\u9752\u7ed2\u8bb0\u53f7\u8868\u793a\u96e8\u5e18\u5323"

    with TrainedDialogueMemoryGraph(path, session_id=20261025) as memory:
        memory.append(surface, speaker_kind=1, stance=EVIDENCE_SUPPORT)
        refuted = memory.retract(surface, speaker_kind=1)
        prior_snapshot = memory.active_structures()
        prior_refutations = tuple(
            item for item in prior_snapshot.evidence
            if item.stance == EVIDENCE_REFUTE
            and item.source_ref[1:] == refuted.source.stable_key())
        assert prior_refutations

        correction = memory.append(
            "\u6211\u66f4\u6b63\u4e00\u4e0b\uff0c\u9752\u7ed2\u8bb0\u53f7\u8868\u793a\u96e8\u5e18\u5323\u3002",
            speaker_kind=1,
            stance=EVIDENCE_SUPPORT,
        )
        replacements = tuple(
            memory.append_evidence(
                old.hypothesis_key,
                source=correction.source,
                stance=EVIDENCE_SUPPORT,
                reason_key=(20261025, 24, 1, ordinal),
                detail=(20261025, 24, 2, ordinal),
                supersedes_key=old.evidence_key,
            )
            for ordinal, old in enumerate(prior_refutations, 1)
        )

        assert len(replacements) == len(prior_refutations)
        for old, new in zip(prior_refutations, replacements):
            old_payload = EvidencePayload.from_stable_key(old.payload_key)
            new_payload = EvidencePayload.from_stable_key(new.payload_key)
            assert new.hypothesis_key == old.hypothesis_key
            assert new.stance == EVIDENCE_SUPPORT
            assert new.supersedes_key == old.evidence_key
            assert new.source_ref[1:] == correction.source.stable_key()
            assert new.scope_key != old.scope_key
            assert new_payload.source == correction.source
            assert new_payload.observed_at.clock == old_payload.observed_at.clock
            assert new_payload.observed_at.seq > old_payload.observed_at.seq

        restored = memory.active_structures()
        restored_keys = {item.evidence_key for item in restored.evidence}
        assert {item.evidence_key for item in prior_refutations} <= restored_keys
        assert {item.evidence_key for item in replacements} <= restored_keys

    with TrainedDialogueMemoryGraph(path, session_id=20261025) as restored_memory:
        cold = restored_memory.active_structures()
        cold_by_key = {item.evidence_key: item for item in cold.evidence}
        assert {item.evidence_key for item in prior_refutations} <= set(cold_by_key)
        assert {item.evidence_key for item in replacements} <= set(cold_by_key)
        for old, new in zip(prior_refutations, replacements):
            restored_payload = EvidencePayload.from_stable_key(
                cold_by_key[new.evidence_key].payload_key)
            assert cold_by_key[new.evidence_key].supersedes_key == old.evidence_key
            assert restored_payload.source == correction.source
