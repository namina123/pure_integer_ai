"""Join stored admission witnesses to cold active Core and recognition history."""
from __future__ import annotations

from pathlib import Path

from pure_integer_ai.experiments.independent_relation_fact_admission import ADMISSION_READY
from pure_integer_ai.experiments.trained_relation_graph_runtime import TrainedRelationGraphRuntime


def verify_cold_admission_history(database: Path, admissions: list[dict]) -> dict:
    rows = []
    with TrainedRelationGraphRuntime(database) as owner:
        active = {item.definition.proposition.stable_key(): item
                  for item in owner.active_propositions()}
        for admission in admissions:
            claims = tuple(tuple(key) for key in admission["proposition_keys"])
            if admission["status"] != ADMISSION_READY:
                if claims or admission.get("witness"):
                    raise ValueError("rejected admission contains a learnable claim or witness")
                continue
            witness = tuple(admission["witness"])
            for key in claims:
                if key not in active:
                    raise ValueError("admitted fact is absent from cold active Core")
                proposition = active[key].definition.proposition
                hypothesis = owner.owner.learning.hypothesis_for_candidate(proposition)
                recognition = owner.owner.learning.engine.recognition_history(hypothesis)
                evidence = owner.owner.learning.engine.ledger.evidence_history(hypothesis)
                if not recognition or not evidence:
                    raise ValueError("admitted fact has no cold prediction/reveal or Evidence history")
                traces = tuple(item.verification.trace for item in recognition)
                if any(len(trace) < 2 or trace[1] != len(witness)
                       or trace[2:2 + trace[1]] != witness for trace in traces):
                    raise ValueError("cold Core verifier did not consume the full admission witness")
                rows.append({"seed_id": admission["seed_id"], "proposition_key": list(key),
                             "recognition_count": len(recognition), "evidence_count": len(evidence),
                             "full_witness_matches": 1})
    return {"status": "PASS_COLD_ADMISSION_CORE_HISTORY", "claims": rows,
            "query_reexecution_count": 0, "core_write_count": 0, "free_dialogue_complete": 0}


__all__ = ["verify_cold_admission_history"]
