from __future__ import annotations

from pure_integer_ai.experiments.portable_core_witness import (
    _relation_fact_rows,
    _response_plan_keys,
)


def test_relation_fact_rows_keep_each_identitys_root_overlap() -> None:
    facts = {
        ((101,), (201,)): {(1,), (2,)},
        ((102,), (202,)): {(3,)},
    }

    assert _relation_fact_rows(facts) == [
        {"proposition": [101], "predicate": [201],
         "root_overlap_count": 2},
        {"proposition": [102], "predicate": [202],
         "root_overlap_count": 1},
    ]


def test_response_plan_keys_include_compact_integer_key_references() -> None:
    trace = {
        "integer_key_table": [[1, 2, 3], [4, 5, 6]],
        "response_plan": {
            "claim_refs": [{"integer_key_ref": 1}],
            "event_refs": [[7, 8]],
            "discourse_links": [],
            "slot_sequence": [{"filler": {"integer_key_ref": 2}}],
        },
    }

    assert _response_plan_keys(trace) == {
        (1, 2, 3), (4, 5, 6), (7, 8),
    }
