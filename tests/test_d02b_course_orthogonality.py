"""D-02B W-03~W-05 三类原创课程的正交差分与跨 pack 隔离 T0/T1。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from pure_integer_ai.experiments.ph2_authored_atomic_compile import (
    compile_atomic_seed,
)
from pure_integer_ai.experiments.ph2_authored_atomic_course import (
    compile_authored_atomic_course,
    read_authored_atomic_seeds,
)
from pure_integer_ai.experiments.ph2_authored_primitive_course import (
    compile_authored_primitive_course,
    read_authored_primitive_seeds,
)
from pure_integer_ai.experiments.ph2_authored_sense_course import (
    compile_authored_sense_course,
    read_authored_sense_seeds,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    RECORD_EVALUATOR_LABEL,
    RECORD_OBSERVATION,
    RECORD_TEACHER_EVIDENCE,
)
from pure_integer_ai.experiments.ph2_dataset_io import read_record_artifact


SENSE_SAMPLE = Path("data/ph2/authored_sense_seed_v1.jsonl.sample")
PRIMITIVE_SAMPLE = Path("data/ph2/authored_primitive_seed_v1.jsonl.sample")
ATOMIC_SAMPLE = Path("data/ph2/authored_atomic_seed_v1.jsonl.sample")


def _by_id(seeds):
    """按 seed id 建唯一索引，便于直接审计成对反向破坏。"""
    return {seed.seed_id: seed for seed in seeds}


def _records(build, kind: str):
    """读取一个 pack 中指定 owner/Observation record kind。"""
    out = []
    for identity in build.manifest.files:
        if identity.record_kind == kind:
            out.extend(read_record_artifact(build.pack_root, identity))
    return tuple(out)


def test_sense_content_replacement_changes_only_candidate_dimension():
    """同 surface/context 的内容替换只改变 sense 候选，不伪造新模板。"""
    seeds = _by_id(read_authored_sense_seeds(SENSE_SAMPLE))
    supported = seeds["teacher-bank-finance-v1"]
    refuted = seeds["teacher-bank-river-refute-v1"]
    assert supported.surface == refuted.surface == "银行"
    assert supported.context == refuted.context
    assert supported.family == refuted.family
    assert supported.template_family == refuted.template_family
    assert supported.candidate_sense != refuted.candidate_sense
    assert supported.perturbation_kind == "NONE"
    assert refuted.perturbation_kind == "CONTENT_REPLACEMENT"


def test_primitive_mismatch_and_surface_replacement_keep_orthogonal_axes():
    """primitive mismatch 只换整数坐标，surface replacement 保持同一原语。"""
    seeds = _by_id(read_authored_primitive_seeds(PRIMITIVE_SAMPLE))
    supported = seeds["teacher-causes-v1"]
    mismatch = seeds["teacher-precedes-mismatch-v1"]
    replacement = seeds["teacher-causes-v2"]
    assert supported.surface_form == mismatch.surface_form
    assert supported.context == mismatch.context
    assert supported.family == mismatch.family
    assert supported.template_family == mismatch.template_family
    assert supported.primitive_registry == mismatch.primitive_registry
    assert supported.primitive_kind != mismatch.primitive_kind
    assert mismatch.perturbation_kind == "PRIMITIVE_MISMATCH"

    assert replacement.primitive_registry == supported.primitive_registry
    assert replacement.primitive_kind == supported.primitive_kind
    assert replacement.surface_form != supported.surface_form
    assert replacement.context != supported.context
    assert replacement.supersedes_seed_id == supported.seed_id
    assert replacement.perturbation_kind == "CUE_REPLACEMENT"


def test_atomic_role_scope_and_occurrence_changes_are_independently_visible():
    """Role、ContextScope 和 occurrence 覆盖各自可变，其他候选轴保持不变。"""
    seeds = _by_id(read_authored_atomic_seeds(ATOMIC_SAMPLE))
    supported = seeds["teacher-chase-valid-v1"]
    role_swap = seeds["teacher-chase-role-swap-v1"]
    assert supported.surface == role_swap.surface
    assert supported.occurrences == role_swap.occurrences
    assert supported.occurrence_order == role_swap.occurrence_order
    assert supported.predicate_kind == role_swap.predicate_kind
    assert supported.context_local_id == role_swap.context_local_id
    supported_roles = {
        item.role_kind: item.filler_occurrence_id for item in supported.bindings
    }
    swapped_roles = {
        item.role_kind: item.filler_occurrence_id for item in role_swap.bindings
    }
    assert set(supported_roles) == set(swapped_roles)
    assert supported_roles != swapped_roles

    shifted = seeds["teacher-finish-scope-shift-v1"]
    unshifted = replace(shifted, context_local_id=1)
    shifted_payload = compile_atomic_seed(shifted).observation_payload.to_value()
    unshifted_payload = compile_atomic_seed(unshifted).observation_payload.to_value()
    assert shifted_payload["surface"] == unshifted_payload["surface"]
    assert shifted_payload["occurrences"] == unshifted_payload["occurrences"]
    assert shifted_payload["occurrence_order"] == unshifted_payload[
        "occurrence_order"]
    shifted_definition = shifted_payload["candidate_definition"]
    unshifted_definition = unshifted_payload["candidate_definition"]
    assert shifted_definition["context_key"] != unshifted_definition["context_key"]
    for field in {
            "predicate_key", "proposition_key", "role_bindings",
            "source_anchor_key"}:
        assert shifted_definition[field] == unshifted_definition[field]

    omitted = seeds["teacher-stop-omission-v1"]
    restored = seeds["teacher-stop-restored-v2"]
    assert omitted.surface == restored.surface
    assert omitted.predicate_kind == restored.predicate_kind
    assert omitted.context_local_id == restored.context_local_id
    assert tuple(item.occurrence_id for item in omitted.occurrences) == (
        "bird", "stop")
    assert tuple(item.occurrence_id for item in restored.occurrences) == (
        "bird", "stop", "branch")
    assert restored.supersedes_seed_id == omitted.seed_id


def test_three_course_packs_have_disjoint_artifacts_records_and_owner_files(
        tmp_path):
    """W-03/W-04/W-05 pack 身份互斥，Observation 与双 owner 继续物理隔离。"""
    builds = (
        compile_authored_sense_course(SENSE_SAMPLE, tmp_path / "sense"),
        compile_authored_primitive_course(
            PRIMITIVE_SAMPLE, tmp_path / "primitive"),
        compile_authored_atomic_course(ATOMIC_SAMPLE, tmp_path / "atomic"),
    )
    assert {build.manifest.w_stages for build in builds} == {
        ("W-03",), ("W-04",), ("W-05",)}
    assert len({build.manifest.stable_key for build in builds}) == 3
    observation_key_sets = []
    owner_key_sets = []
    for build in builds:
        observations = _records(build, RECORD_OBSERVATION)
        teachers = _records(build, RECORD_TEACHER_EVIDENCE)
        evaluators = _records(build, RECORD_EVALUATOR_LABEL)
        assert observations and teachers and evaluators
        for observation in observations:
            payload = observation.typed_payload.to_value()
            assert "expected_state" not in payload
            assert "expected_payload" not in payload
        observation_key_sets.append({item.stable_key for item in observations})
        teacher_owners = {item.owner_key for item in teachers}
        evaluator_owners = {item.owner_key for item in evaluators}
        assert teacher_owners.isdisjoint(evaluator_owners)
        owner_key_sets.append(teacher_owners | evaluator_owners)
        paths = {identity.relative_path for identity in build.manifest.files}
        assert "owners/teacher/train.evidence.jsonl.gz" in paths
        assert "owners/evaluator/held_out.labels.jsonl.gz" in paths
    for index, keys in enumerate(observation_key_sets):
        for other in observation_key_sets[index + 1:]:
            assert keys.isdisjoint(other)
    for index, keys in enumerate(owner_key_sets):
        for other in owner_key_sets[index + 1:]:
            assert keys.isdisjoint(other)
