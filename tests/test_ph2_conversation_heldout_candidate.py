"""DLG-05 独立生产候选与公开 observation 的专项。"""
from __future__ import annotations

import inspect

import pytest

from pure_integer_ai.experiments.conversation_heldout_candidate_runtime import (
    ConversationHeldOutCandidateError,
    build_dlg05_candidate_observation_document,
    qualify_dlg05_public_candidate,
    run_dlg05_public_candidate,
    verify_dlg05_candidate_observation,
    write_dlg05_candidate_observation,
)


def test_dlg05_production_candidate_runs_without_label_owner(tmp_path):
    """生产入口只接 SQLite 路径，并闭合六案例、重放、存储和 rollback。"""
    assert tuple(inspect.signature(
        run_dlg05_public_candidate).parameters) == ("database_path",)
    result = run_dlg05_public_candidate(tmp_path / "candidate.sqlite3")

    assert result.execution.contract_key
    assert len(result.execution.observations) == 6
    document = build_dlg05_candidate_observation_document(result)
    assert document["labels_included"] == 0
    assert document["formal_run"] == 0
    assert "label" not in " ".join(document).lower().replace(
        "labels_included", "")

    root = tmp_path / "public-root"
    target = (
        root / "data" / "ph2" / "manifests"
        / "dlg05_public_candidate_observation_v3.json"
    )
    write_dlg05_candidate_observation(target, root, result)
    first = target.read_bytes()
    write_dlg05_candidate_observation(target, root, result)
    assert target.read_bytes() == first
    verified = verify_dlg05_candidate_observation(target)
    assert verified["path"] == target.name
    assert verified["document_sha256"] == document["document_sha256"]
    assert verified["observation_count"] == 6
    assert verified["verified"] == 1
    target.write_bytes(b"{}\n")
    with pytest.raises(
            ConversationHeldOutCandidateError, match="不允许覆盖"):
        write_dlg05_candidate_observation(target, root, result)


def test_dlg05_public_qualification_is_separate_from_single_selection(tmp_path):
    """qualification 的重放证据不被一次性 candidate selection 假充。"""
    result = qualify_dlg05_public_candidate(tmp_path / "qualification.sqlite3")
    assert result.qualification.replay_stable
    assert result.qualification.storage_stable
    assert result.qualification.rollback_recovery.recovered_clean
