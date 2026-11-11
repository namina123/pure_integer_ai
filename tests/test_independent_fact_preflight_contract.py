"""Physical cold restore, pre-training carrier admission and split/permission drift."""
from pathlib import Path
import shutil

import pytest

from pure_integer_ai.experiments.connected_relation_training_course import compile_relation_training_course
from pure_integer_ai.experiments.independent_fact_preflight_contract import (
    INPUTS, LABELS, PERMISSION, QUESTIONS, SOURCE_FILES, SYNTAX, TEACHER,
    audit_prepared_question_carriers, authorized_fact_inputs,
    fact_input_source, inspect_independent_fact_contract, read_rows,
)
from pure_integer_ai.experiments.independent_relation_fact_source import (
    compile_independent_relation_fact_source, read_independent_relation_fact_source,
)
from pure_integer_ai.experiments.natural_relation_preflight import file_sha256
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


@pytest.fixture
def frozen(tmp_path):
    project = Path(__file__).resolve().parents[1]
    for relative, original in SOURCE_FILES.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project / original, target)
    return tmp_path


def _write(root, relative, rows):
    (root / relative).write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))


def test_contract_freezes_narrow_scope_and_explicit_input_permissions(frozen):
    report = inspect_independent_fact_contract(frozen)
    assert report["evaluation_case_counts"] == {
        "held_out": 5, "negative": 2, "conflict": 1, "unauthorized": 1}
    assert report["operation_family_count"] == 1
    assert report["teacher_factual_truth_held_out"] == report["free_dialogue_complete"] == 0
    assert authorized_fact_inputs(frozen) == tuple(fact_input_source(i) for i in range(1, 9))
    assert fact_input_source(9) not in authorized_fact_inputs(frozen)


@pytest.mark.parametrize("drift", ["permission", "question_overlap", "label_owner", "entity_overlap"])
def test_pretraining_contract_rejects_partition_or_authority_drift(frozen, drift):
    if drift == "permission":
        rows = read_rows(frozen / PERMISSION)
        rows[0]["production_authorized"] = 1
        _write(frozen, PERMISSION, rows)
    elif drift == "question_overlap":
        rows = read_rows(frozen / INPUTS)
        rows[0]["input"] = read_rows(frozen / QUESTIONS)[0]["input"]
        _write(frozen, INPUTS, rows)
    elif drift == "label_owner":
        rows = read_rows(frozen / LABELS)
        rows[0]["policy"] = "NO_FACTUAL_ANSWER"
        _write(frozen, LABELS, rows)
    else:
        rows = read_rows(frozen / TEACHER)
        rows[0]["terms"][0][2] = read_rows(frozen / SYNTAX)[0]["terms"][0][2]
        _write(frozen, TEACHER, rows)
    with pytest.raises(ValueError):
        inspect_independent_fact_contract(frozen)


def test_carriers_compose_from_train_sources_and_missing_slot_stays_unbound(frozen):
    compiled = compile_relation_training_course(
        frozen / SYNTAX, frozen / QUESTIONS, frozen / "syntax_pack")
    teacher = compile_independent_relation_fact_source(frozen / TEACHER, frozen / "teacher_pack")
    inputs = read_rows(frozen / INPUTS)
    report = audit_prepared_question_carriers(compiled.question_frames, inputs, teacher_course=teacher.course)
    assert all(row["composed_meaning_count"] == 1 for row in report["cases"][:5])
    assert all(row["unbound_composed_meaning_count"] > 1 for row in report["cases"][:5])
    assert report["cases"][5]["complete_meaning_count"] == 0
    inputs[0]["input"] = "未训练前缀" + inputs[0]["input"]
    with pytest.raises(ValueError, match="carrier"):
        audit_prepared_question_carriers(compiled.question_frames, inputs, teacher_course=teacher.course)


def test_teacher_pack_restores_without_republishing_or_writing(frozen):
    original = compile_independent_relation_fact_source(frozen / TEACHER, frozen / "teacher_pack")
    before = {path: file_sha256(path) for path in original.pack.pack_root.rglob("*") if path.is_file()}
    restored = read_independent_relation_fact_source(frozen / TEACHER, original.pack.pack_root)
    assert restored == original
    assert restored.pack.validation.evaluator_label_count == 0
    assert restored.pack.validation.teacher_evidence_count == 9
    assert before == {path: file_sha256(path) for path in before}
