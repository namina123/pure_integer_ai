"""D-02B AUTHORED_CC0_V1 原语/surface 对应编译器 T0。"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.operator_primitives import (
    OP_ADD,
    OP_GE,
    OP_GT,
    OP_LE,
    OP_LT,
    OP_MUL,
    OP_SUB,
)
from pure_integer_ai.cognition.shared.relation_primitives import (
    REL_CAUSES,
    REL_EQUAL,
    REL_MEMBER,
    REL_MEREOLOGY,
    REL_PRECEDES,
    REL_PROPERTY,
    REL_SIMILAR,
    REL_SUBSET,
)
from pure_integer_ai.cognition.shared.symbol_types import (
    TYPE_ATTR_MARKER,
    TYPE_CAUSES,
    TYPE_CMP,
    TYPE_COPULA,
    TYPE_NEGATION,
)
from pure_integer_ai.experiments.ph2_authored_primitive_course import (
    LICENSE_ID,
    PACK_NAME,
    SOURCE_KEY,
    AuthoredPrimitiveCourseError,
    compile_authored_primitive_course,
    read_authored_primitive_seeds,
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


SAMPLE_PATH = Path("data/ph2/authored_primitive_seed_v1.jsonl.sample")
COMPILER_PATH = Path(
    "src/pure_integer_ai/experiments/ph2_authored_primitive_course.py")
ACTIVE_PRIMITIVES = {
    "relation": {
        REL_SUBSET, REL_MEMBER, REL_EQUAL, REL_CAUSES, REL_PRECEDES,
        REL_MEREOLOGY, REL_PROPERTY, REL_SIMILAR,
    },
    "operator": {OP_ADD, OP_SUB, OP_MUL, OP_GT, OP_LT, OP_GE, OP_LE},
    "symbol_type": {
        TYPE_NEGATION, TYPE_COPULA, TYPE_CMP, TYPE_CAUSES, TYPE_ATTR_MARKER,
    },
}


def _sample_values() -> list[dict]:
    """读取仓库极小 sample 为独立可修改 JSON object 列表。"""
    return [json.loads(line) for line in SAMPLE_PATH.read_text(
        encoding="utf-8").splitlines()]


def _write_values(path: Path, values: list[dict]) -> None:
    """按统一规范 JSONL 写测试 seed，不复用编译器 parser。"""
    path.write_bytes(b"".join(canonical_json_line(value) for value in values))


def _records_by_kind(build, kind: str):
    """读取 manifest 中指定 record kind 的全部物理文件。"""
    records = []
    for identity in build.manifest.files:
        if identity.record_kind == kind:
            records.extend(read_record_artifact(build.pack_root, identity))
    return tuple(records)


def test_sample_uses_frozen_integer_coordinates_and_required_perturbations():
    """sample 只引用冻结整数坐标，并覆盖 W-04 必需破坏和修正角色。"""
    seeds = read_authored_primitive_seeds(SAMPLE_PATH)
    assert len(seeds) == 7
    assert LICENSE_ID == "CC0-1.0"
    assert {seed.sample_role for seed in seeds} == {
        "support", "refute", "conflict", "supersede"}
    assert {
        "PRIMITIVE_MISMATCH",
        "SAME_SURFACE_AMBIGUITY",
        "CUE_REPLACEMENT",
        "CUE_DELETION",
    }.issubset({seed.perturbation_kind for seed in seeds})
    for seed in seeds:
        assert type(seed.primitive_kind) is int
        assert seed.primitive_kind in ACTIVE_PRIMITIVES[seed.primitive_registry]
    same_surface = [seed for seed in seeds if seed.surface_form == "导致"]
    assert len(same_surface) == 2
    assert len({seed.primitive_kind for seed in same_surface}) == 2


def test_compiler_writes_bit_identical_pack_and_keeps_expected_private(tmp_path):
    """两目录编译 bit-identical，Observation 不携带 teacher/evaluator expected。"""
    first = compile_authored_primitive_course(SAMPLE_PATH, tmp_path / "first")
    second = compile_authored_primitive_course(SAMPLE_PATH, tmp_path / "second")
    assert first.pack_root.name == PACK_NAME
    assert first.manifest == second.manifest
    assert first.manifest.source_key == SOURCE_KEY
    assert first.manifest.license_partition == LICENSE_ID
    assert first.manifest.splits == ("train", "held_out")
    assert first.manifest.w_stages == ("W-04",)
    assert first.validation.source_ref_count == 7
    assert first.validation.observation_count == 7
    assert first.validation.teacher_evidence_count == 4
    assert first.validation.evaluator_label_count == 3
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
        7, 7, 4, 3)
    for item in observations:
        payload = item.typed_payload.to_value()
        assert set(payload) == {
            "candidate_primitive", "context", "query_kind", "surface_form"}
        coordinate = payload["candidate_primitive"]
        assert type(coordinate["kind"]) is int
        assert coordinate["kind"] in ACTIVE_PRIMITIVES[coordinate["registry"]]
        assert "expected_state" not in payload
        assert "expected_payload" not in payload


def test_split_clusters_owner_views_and_supersede_are_directly_auditable(
        tmp_path):
    """来源簇、模板、owner 视图隔离，supersede 只指向更早训练记录。"""
    seeds = read_authored_primitive_seeds(SAMPLE_PATH)
    teacher_families = {seed.family for seed in seeds if seed.label_owner == "teacher"}
    evaluator_families = {
        seed.family for seed in seeds if seed.label_owner == "evaluator"}
    teacher_templates = {
        seed.template_family for seed in seeds if seed.label_owner == "teacher"}
    evaluator_templates = {
        seed.template_family for seed in seeds if seed.label_owner == "evaluator"}
    assert teacher_families.isdisjoint(evaluator_families)
    assert teacher_templates.isdisjoint(evaluator_templates)

    build = compile_authored_primitive_course(SAMPLE_PATH, tmp_path)
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
        train, teachers, (), current_stage="W-04", view_kind="training")
    validate_stage_visibility(
        held_out, (), evaluators, current_stage="W-04", view_kind="evaluation")
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
        (lambda rows: rows[4].__setitem__("split", "train"), "split"),
        (lambda rows: rows[0].__setitem__("primitive_registry", "legacy"),
         "active 冻结坐标"),
        (lambda rows: rows[0].__setitem__("primitive_kind", 999),
         "active 冻结坐标"),
        (lambda rows: (
            rows[0].__setitem__("sample_role", "supersede"),
            rows[0].__setitem__("supersedes_seed_id", rows[1]["seed_id"]),
        ), "更早"),
    ],
)
def test_bad_license_duplicate_split_registry_kind_and_future_supersede_fail(
        tmp_path, mutate, message):
    """坏许可、重复、串线、伪坐标和未来 supersede 均 fail closed。"""
    rows = _sample_values()
    mutate(rows)
    bad = tmp_path / "bad.sample"
    _write_values(bad, rows)
    with pytest.raises(AuthoredPrimitiveCourseError, match=message):
        read_authored_primitive_seeds(bad)


def test_float_kind_noncanonical_json_and_existing_pack_fail_closed(tmp_path):
    """浮点 kind、非规范 JSON 和覆盖既有 pack 均被拒绝。"""
    rows = _sample_values()
    rows[0]["primitive_kind"] = 4.0
    bad_float = tmp_path / "float.sample"
    bad_float.write_bytes(b"".join(
        (json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ) + "\n").encode("utf-8")
        for row in rows
    ))
    with pytest.raises(AuthoredPrimitiveCourseError, match="规范 JSON"):
        read_authored_primitive_seeds(bad_float)

    bad_json = tmp_path / "noncanonical.sample"
    bad_json.write_bytes(b'{"surface_form": "x"}\n')
    with pytest.raises(AuthoredPrimitiveCourseError, match="规范 JSON"):
        read_authored_primitive_seeds(bad_json)

    build = compile_authored_primitive_course(
        SAMPLE_PATH, tmp_path / "release")
    assert build.pack_root.exists()
    with pytest.raises(AuthoredPrimitiveCourseError, match="发布失败"):
        compile_authored_primitive_course(SAMPLE_PATH, tmp_path / "release")


def test_compiler_source_has_no_sample_surface_to_label_lookup_table():
    """编译器只解析 typed seed，不把 sample 中文 surface 固化成标签词表。"""
    source = COMPILER_PATH.read_text(encoding="utf-8")
    string_constants = {
        node.value for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    for cue in {row["surface_form"] for row in _sample_values()}:
        assert cue not in string_constants
