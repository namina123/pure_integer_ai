"""D-02B AUTHORED_CC0_V1 W-05 occurrence/角色/原子命题 T0。"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONTEXT_SCOPE,
    OBJECT_ENTITY,
    OBJECT_EVENT,
    OBJECT_OCCURRENCE,
    OBJECT_PROPOSITION,
    OBJECT_ROLE,
    OBJECT_ROLE_BINDING,
    ObjectIdentity,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicPropositionDefinition,
    AtomicRoleBinding,
)
from pure_integer_ai.experiments.ph2_authored_atomic_course import (
    ALLOWED_ROLE_KINDS,
    LICENSE_ID,
    PACK_NAME,
    PREDICATE_REGISTRY,
    ROLE_REGISTRY,
    SOURCE_KEY,
    AuthoredAtomicCourseError,
    compile_authored_atomic_course,
    read_authored_atomic_seeds,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    RECORD_EVALUATOR_LABEL,
    RECORD_OBSERVATION,
    RECORD_SOURCE_REF,
    RECORD_TEACHER_EVIDENCE,
    EvaluatorLabelRecord,
    ObservationRecord,
    SourceRefRecord,
    TeacherEvidenceRecord,
    canonical_json_line,
)
from pure_integer_ai.experiments.ph2_dataset_io import (
    read_artifact_manifest,
    read_record_artifact,
)
from pure_integer_ai.experiments.ph2_dataset_validation import (
    validate_stage_visibility,
)


SAMPLE_PATH = Path("data/ph2/authored_atomic_seed_v1.jsonl.sample")
COMPILER_PATHS = (
    Path("src/pure_integer_ai/experiments/ph2_authored_atomic_course.py"),
    Path("src/pure_integer_ai/experiments/ph2_authored_atomic_schema.py"),
    Path("src/pure_integer_ai/experiments/ph2_authored_atomic_compile.py"),
)


def _sample_values() -> list[dict]:
    """读取仓库极小 sample 为独立可修改 JSON object 列表。"""
    return [json.loads(line) for line in SAMPLE_PATH.read_text(
        encoding="utf-8").splitlines()]


def _write_values(path: Path, values: list[dict]) -> None:
    """按统一规范 JSONL 写测试 seed，不复用课程 parser。"""
    path.write_bytes(b"".join(canonical_json_line(value) for value in values))


def _write_json_with_float(path: Path, values: list[dict]) -> None:
    """绕过合同 writer 落一个只因 float 非法的 parser 负例。"""
    path.write_bytes(b"".join(
        (json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ) + "\n").encode("utf-8")
        for row in values
    ))


def _records_by_kind(build, kind: str):
    """读取 manifest 中指定 record kind 的全部物理文件。"""
    records = []
    for identity in build.manifest.files:
        if identity.record_kind == kind:
            records.extend(read_record_artifact(build.pack_root, identity))
    return tuple(records)


def _identity(value) -> ObjectIdentity:
    """从规范 payload 的整数列表恢复并核验一等对象身份。"""
    assert isinstance(value, list)
    assert all(type(item) is int for item in value)
    return ObjectIdentity.from_stable_key(tuple(value))


def test_sample_has_typed_coordinates_required_breakage_and_owner_isolation():
    """sample 覆盖四类 W-05 破坏，全部 kind/role 均为严格整数坐标。"""
    seeds = read_authored_atomic_seeds(SAMPLE_PATH)
    assert len(seeds) == 10
    assert LICENSE_ID == "CC0-1.0"
    assert {seed.sample_role for seed in seeds} == {
        "support", "refute", "conflict", "supersede"}
    assert {
        "ROLE_SWAP",
        "ORDER_REVERSAL",
        "SCOPE_SHIFT",
        "OCCURRENCE_OMISSION",
    }.issubset({seed.perturbation_kind for seed in seeds})
    teacher_families = {seed.family for seed in seeds if seed.label_owner == "teacher"}
    evaluator_families = {
        seed.family for seed in seeds if seed.label_owner == "evaluator"}
    teacher_templates = {
        seed.template_family for seed in seeds if seed.label_owner == "teacher"}
    evaluator_templates = {
        seed.template_family for seed in seeds if seed.label_owner == "evaluator"}
    assert teacher_families.isdisjoint(evaluator_families)
    assert teacher_templates.isdisjoint(evaluator_templates)
    for seed in seeds:
        assert seed.predicate_registry == PREDICATE_REGISTRY
        assert type(seed.predicate_kind) is int
        for occurrence in seed.occurrences:
            assert type(occurrence.semantic_kind) is int
            assert occurrence.semantic_kind in {OBJECT_ENTITY, OBJECT_EVENT}
        for binding in seed.bindings:
            assert binding.role_registry == ROLE_REGISTRY
            assert type(binding.role_kind) is int
            assert binding.role_kind in ALLOWED_ROLE_KINDS


def test_compiler_writes_bit_identical_pack_and_private_owner_records(tmp_path):
    """两目录产物 bit-identical，10/10/6/4 分账且 expected 不进 Observation。"""
    first = compile_authored_atomic_course(SAMPLE_PATH, tmp_path / "first")
    second = compile_authored_atomic_course(SAMPLE_PATH, tmp_path / "second")
    assert first.pack_root.name == PACK_NAME
    assert first.manifest == second.manifest
    assert first.manifest.source_key == SOURCE_KEY
    assert first.manifest.license_partition == LICENSE_ID
    assert first.manifest.splits == ("train", "held_out")
    assert first.manifest.w_stages == ("W-05",)
    assert first.validation.source_ref_count == 10
    assert first.validation.observation_count == 10
    assert first.validation.teacher_evidence_count == 6
    assert first.validation.evaluator_label_count == 4
    assert first.validation.source_cluster_count == 2
    assert read_artifact_manifest(first.pack_root / "manifest.json") == (
        first.manifest)
    assert (first.pack_root / "manifest.json").read_bytes() == (
        second.pack_root / "manifest.json").read_bytes()
    for first_file, second_file in zip(first.manifest.files, second.manifest.files):
        assert first_file == second_file
        assert (first.pack_root / first_file.relative_path).read_bytes() == (
            second.pack_root / second_file.relative_path).read_bytes()

    sources = _records_by_kind(first, RECORD_SOURCE_REF)
    observations = _records_by_kind(first, RECORD_OBSERVATION)
    teachers = _records_by_kind(first, RECORD_TEACHER_EVIDENCE)
    evaluators = _records_by_kind(first, RECORD_EVALUATOR_LABEL)
    assert all(isinstance(item, SourceRefRecord) for item in sources)
    assert all(isinstance(item, ObservationRecord) for item in observations)
    assert all(isinstance(item, TeacherEvidenceRecord) for item in teachers)
    assert all(isinstance(item, EvaluatorLabelRecord) for item in evaluators)
    assert (len(sources), len(observations), len(teachers), len(evaluators)) == (
        10, 10, 6, 4)
    for item in observations:
        payload = item.typed_payload.to_value()
        assert set(payload) == {
            "candidate_definition",
            "occurrence_order",
            "occurrences",
            "query_kind",
            "surface",
        }
        assert "expected_state" not in payload
        assert "expected_payload" not in payload


def test_observation_payload_roundtrips_current_semantic_object_contract(tmp_path):
    """每条候选都可重建现役 Occurrence/Context/RoleBinding/Proposition 对象。"""
    build = compile_authored_atomic_course(SAMPLE_PATH, tmp_path)
    observations = _records_by_kind(build, RECORD_OBSERVATION)
    for item in observations:
        payload = item.typed_payload.to_value()
        candidate = payload["candidate_definition"]
        proposition = _identity(candidate["proposition_key"])
        predicate = _identity(candidate["predicate_key"])
        source_anchor = _identity(candidate["source_anchor_key"])
        context = _identity(candidate["context_key"])
        assert proposition.object_kind == OBJECT_PROPOSITION
        assert source_anchor.object_kind == OBJECT_OCCURRENCE
        assert context.object_kind == OBJECT_CONTEXT_SCOPE

        bindings = []
        for value in candidate["role_bindings"]:
            role = _identity(value["role_key"])
            filler = _identity(value["filler_key"])
            binding_identity = _identity(value["binding_key"])
            binding = AtomicRoleBinding(role, filler, value["ordinal"])
            assert role.object_kind == OBJECT_ROLE
            assert binding_identity.object_kind == OBJECT_ROLE_BINDING
            assert binding.identity_for(proposition) == binding_identity
            bindings.append(binding)
        restored = AtomicPropositionDefinition(
            proposition,
            predicate,
            source_anchor,
            context,
            tuple(bindings),
        )
        assert restored.proposition == proposition

        occurrence_keys = []
        for occurrence in payload["occurrences"]:
            identity = _identity(occurrence["identity_key"])
            semantic = _identity(occurrence["semantic_key"])
            assert identity.object_kind == OBJECT_OCCURRENCE
            assert semantic.object_kind == occurrence["semantic_kind"]
            assert semantic.object_kind in {OBJECT_ENTITY, OBJECT_EVENT}
            assert payload["surface"][occurrence["start"]:occurrence["end"]] == (
                occurrence["surface_fragment"])
            occurrence_keys.append(occurrence["identity_key"])
        assert payload["occurrence_order"] == occurrence_keys


def test_split_clusters_stage_views_supersede_and_omission_are_auditable(
        tmp_path):
    """split 来源簇/owner 隔离，漏 occurrence 与恢复 supersede 有直接证据。"""
    seeds = read_authored_atomic_seeds(SAMPLE_PATH)
    omitted = next(seed for seed in seeds if seed.seed_id == "teacher-stop-omission-v1")
    restored = next(seed for seed in seeds if seed.seed_id == "teacher-stop-restored-v2")
    held_omission = next(
        seed for seed in seeds if seed.seed_id == "evaluator-inspect-omission-v1")
    assert "树枝" not in {item.surface_fragment for item in omitted.occurrences}
    assert "树枝" in {item.surface_fragment for item in restored.occurrences}
    assert "病人" not in {item.surface_fragment for item in held_omission.occurrences}
    assert restored.supersedes_seed_id == omitted.seed_id
    assert omitted.logical_order < restored.logical_order

    build = compile_authored_atomic_course(SAMPLE_PATH, tmp_path)
    sources = _records_by_kind(build, RECORD_SOURCE_REF)
    observations = _records_by_kind(build, RECORD_OBSERVATION)
    teachers = _records_by_kind(build, RECORD_TEACHER_EVIDENCE)
    evaluators = _records_by_kind(build, RECORD_EVALUATOR_LABEL)
    source_index = {item.stable_key: item for item in sources}
    train = tuple(item for item in observations if item.split == "train")
    held_out = tuple(item for item in observations if item.split == "held_out")
    train_clusters = {
        source_index[item.source_ref_key].source_cluster_key for item in train
    }
    held_clusters = {
        source_index[item.source_ref_key].source_cluster_key for item in held_out
    }
    assert train_clusters.isdisjoint(held_clusters)
    assert {item.owner_key for item in teachers}.isdisjoint(
        {item.owner_key for item in evaluators})
    validate_stage_visibility(
        train, teachers, (), current_stage="W-05", view_kind="training")
    validate_stage_visibility(
        held_out, (), evaluators, current_stage="W-05", view_kind="evaluation")
    superseder = next(
        item for item in observations if item.sample_role == "supersede")
    target = next(
        item for item in observations if item.stable_key == superseder.supersedes_key)
    assert target.logical_order < superseder.logical_order
    assert target.split == superseder.split == "train"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: rows[0].__setitem__("license_id", "UNKNOWN"), "CC0-1.0"),
        (lambda rows: rows[1].__setitem__("seed_id", rows[0]["seed_id"]), "重复"),
        (lambda rows: rows[6].__setitem__("split", "train"), "split"),
        (lambda rows: rows[0]["occurrences"][0].__setitem__(
            "semantic_kind", OBJECT_OCCURRENCE), "Entity/Event"),
        (lambda rows: rows[0]["occurrences"][0].__setitem__(
            "surface_fragment", "错误"), "surface 不一致"),
        (lambda rows: rows[0]["occurrence_order"].pop(), "恰好覆盖"),
        (lambda rows: rows[0].__setitem__("predicate_occurrence_id", "cat"),
         "Event occurrence"),
        (lambda rows: rows[0]["bindings"][0].__setitem__(
            "role_registry", "LEGACY_ROLE"), "registry"),
        (lambda rows: rows[0]["bindings"][0].__setitem__("role_kind", 999),
         "role kind"),
        (lambda rows: rows[0]["bindings"][1].__setitem__(
            "filler_occurrence_id", "missing"), "未知 occurrence"),
        (lambda rows: rows[0]["bindings"][1].__setitem__("role_kind", 1),
         "slot 重复"),
        (lambda rows: (
            rows[0].__setitem__("sample_role", "supersede"),
            rows[0].__setitem__("supersedes_seed_id", rows[1]["seed_id"]),
        ), "更早"),
        (lambda rows: rows[5].__setitem__(
            "supersedes_seed_id", rows[0]["seed_id"]), "修正 occurrence omission"),
    ],
)
def test_bad_license_identity_span_binding_owner_and_supersede_fail_closed(
        tmp_path, mutate, message):
    """坏许可、类型/span/绑定、owner 串线和错误替代均 fail closed。"""
    rows = _sample_values()
    mutate(rows)
    bad = tmp_path / "bad.sample"
    _write_values(bad, rows)
    with pytest.raises(AuthoredAtomicCourseError, match=message):
        read_authored_atomic_seeds(bad)


def test_float_noncanonical_json_and_existing_pack_fail_closed(tmp_path):
    """float、非规范 JSON 和覆盖既有版本均不能进入正式 pack。"""
    rows = _sample_values()
    rows[0]["predicate_kind"] = 1.0
    bad_float = tmp_path / "float.sample"
    _write_json_with_float(bad_float, rows)
    with pytest.raises(AuthoredAtomicCourseError, match="规范 JSON"):
        read_authored_atomic_seeds(bad_float)

    bad_json = tmp_path / "noncanonical.sample"
    bad_json.write_bytes(b'{"surface": "x"}\n')
    with pytest.raises(AuthoredAtomicCourseError, match="规范 JSON"):
        read_authored_atomic_seeds(bad_json)

    build = compile_authored_atomic_course(
        SAMPLE_PATH, tmp_path / "release")
    assert build.pack_root.exists()
    with pytest.raises(AuthoredAtomicCourseError, match="发布失败"):
        compile_authored_atomic_course(SAMPLE_PATH, tmp_path / "release")


def test_compiler_source_has_no_sample_surface_to_structure_lookup_table():
    """编译器只消费 typed seed，不固化 sample 中文 surface 到结构标签映射。"""
    string_constants = set()
    for path in COMPILER_PATHS:
        source = path.read_text(encoding="utf-8")
        string_constants.update(
            node.value for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        )
    fragments = {
        occurrence["surface_fragment"]
        for row in _sample_values()
        for occurrence in row["occurrences"]
    }
    assert fragments.isdisjoint(string_constants)
