import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_relation_evidence_learning import (
    learn_relation_evidence_model,
)
from pure_integer_ai.experiments.ph2_broad_qa_relation_marker_evidence_learning import (
    learn_relation_marker_evidence_model,
)
from pure_integer_ai.experiments.ph2_broad_qa_relation_marker_negative import (
    MarkerNegativeProtocolError,
    audit_marker_negative_projection,
    load_marker_negative_jsonl,
)


_ROOT = Path(__file__).resolve().parents[1]
_NEGATIVE = _ROOT / (
    "data/ph2/dlg_raw_public_relation_marker_negative_v1.jsonl.sample")
_RELATION = (
    _ROOT / "data/ph2/dlg_raw_public_relation_evidence_v1.jsonl.sample",
    _ROOT / "data/ph2/dlg_raw_public_relation_evidence_v2.jsonl.sample",
)
_MARKER = (
    _ROOT / "data/ph2/dlg_raw_public_relation_marker_evidence_v1.jsonl.sample",
    _ROOT / "data/ph2/dlg_raw_public_relation_marker_evidence_v2.jsonl.sample",
)


def test_negative_protocol_has_no_answer_and_fails_closed() -> None:
    cases = load_marker_negative_jsonl((_NEGATIVE,))
    relation = learn_relation_evidence_model(_RELATION)
    marker = learn_relation_marker_evidence_model(_MARKER)
    assert len(cases) == 2
    assert all(len(item.candidates) == 2 for item in cases)
    assert all(type(value) is int
               for item in cases for value in item.canonical_record())
    assert audit_marker_negative_projection(cases, relation, marker)[:2] == (1, 2)


def test_negative_protocol_rejects_answer_field(tmp_path: Path) -> None:
    payload = _NEGATIVE.read_text(encoding="utf-8").splitlines()[0]
    value = json.loads(payload)
    value["answer"] = "forbidden"
    path = tmp_path / "answer.jsonl"
    path.write_text(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8")
    with pytest.raises(MarkerNegativeProtocolError):
        load_marker_negative_jsonl((path,))
