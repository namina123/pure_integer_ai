"""Probe one live graph input in a fresh process-local Memory session."""
from __future__ import annotations
import json
from pathlib import Path
from pure_integer_ai.experiments.heldout_response_runtime import HeldoutResponseRuntime
from pure_integer_ai.experiments.trained_generation_connector_runtime import TrainedGenerationConnectorRuntime
from pure_integer_ai.experiments.trained_graph_query_bridge import TrainedGraphQueryBridge

DB = Path(r"K:/pure_integer_ai_work/current_runs/discourse-relation-response-graph-20260917b/training.sqlite3")
COURSE = Path(r"K:/pure_integer_ai_work/current_runs/broad-structural-heldout-course-20260923b/heldout_response_course.int.json")
MANIFEST = Path(r"K:/pure_integer_ai_work/current_runs/broad-structural-heldout-course-20260923b/heldout_manifest.json")
ROOT = Path(r"K:/pure_integer_ai_work/current_runs/independent-graph-input-20260923c")
SURFACE = "".join(map(chr, (23545, 35937)))
GRAPH = (7, 0, 0, 0, 1, 1, 1, 1, 1, 15, 1, 206, 903069331810179240,
         0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 1, 1)

def main() -> int:
    if ROOT.exists(): raise ValueError(f"probe root exists: {ROOT}")
    ROOT.mkdir(parents=True)
    report = {"format": "INDEPENDENT_GRAPH_INPUT_PROBE_V1", "integer_payload": 1}
    with TrainedGenerationConnectorRuntime(DB) as canonical:
        with HeldoutResponseRuntime(canonical, course_path=COURSE, manifest_path=MANIFEST) as runtime:
            with TrainedGraphQueryBridge(DB, memory_database=ROOT / "session.sqlite3", session_id=92097, surface_generator=runtime) as bridge:
                appended = bridge.memory.append(SURFACE, speaker_kind=1)
                observation = bridge.memory.intake.result_for_source(appended.source).observation_ref
                result = bridge.query(SURFACE, max_depth=64, graph_object_keys=(GRAPH,), input_observation_ref=observation)
                generation = result.get("generation")
                heldout = bool(
                    generation
                    and runtime.is_heldout_connector(
                        tuple(generation.get("connector", ())))
                )
                response_surface = result.get("response_surface", "")
                report.update({
                    "termination": result.get("termination"),
                    "heldout_connector_selected": int(heldout),
                    "generation": int(bool(generation)),
                    "response_surface_units": list(map(ord, response_surface)),
                    "generic_generation_count": len(result.get("generic_response_generations", ())),
                    "graph_input_consumed": list(result.get("generic_graph_input_keys", ())),
                })
    (ROOT / "report.int.json").write_text(json.dumps(report, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n", encoding="ascii")
    print(json.dumps(report, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
    return 0

if __name__ == "__main__": raise SystemExit(main())
