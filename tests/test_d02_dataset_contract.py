"""D-02A 统一资料合同、许可、split、阶段和 supersede T0。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import (
    RECORD_EVALUATOR_LABEL,
    RECORD_OBSERVATION,
    RECORD_SOURCE_REF,
    RECORD_TEACHER_EVIDENCE,
    ArtifactFileIdentity,
    ArtifactManifest,
    CanonicalJsonObject,
    DatasetContractError,
    EvaluatorLabelRecord,
    ObservationRecord,
    SourceRefRecord,
    StableRecordKey,
    TeacherEvidenceRecord,
    record_from_dict,
)
from pure_integer_ai.experiments.ph2_dataset_io import (
    ArtifactWriteSpec,
    DatasetArtifactIOError,
    read_artifact_manifest,
    read_record_artifact,
    write_artifact_manifest,
    write_record_artifact,
)
from pure_integer_ai.experiments.ph2_dataset_validation import (
    DatasetValidationError,
    validate_artifact_manifest,
    validate_dataset_bundle,
    validate_stage_visibility,
    validate_supersede_and_prerequisite_graph,
)


def _key(value: int) -> StableRecordKey:
    """构造测试使用的单分量正整数稳定键。"""
    return StableRecordKey((value,))


def _payload(value: dict) -> CanonicalJsonObject:
    """把测试 object 转为不可变规范 JSON 载荷。"""
    return CanonicalJsonObject.from_value(value)


def _source(
        record_id: int,
        cluster_id: int,
        *,
        source_key: str = "AUTHORED_CC0_V1",
        license_id: str = "CC0-1.0",
        policy: str = "PUBLIC") -> SourceRefRecord:
    """构造具备完整来源、许可、checksum 和 span 的 SourceRefRecord。"""
    digit = format(record_id % 16, "x")
    return SourceRefRecord(
        1,
        1,
        1,
        _key(60001),
        _key(60002),
        _key(record_id),
        source_key,
        "seed-snapshot-v1",
        "",
        "https://example.invalid/ph2/source",
        f"seed/{record_id}.json",
        "sha256:" + digit * 64,
        digit * 64,
        license_id,
        policy,
        "Pure Integer AI PH2 authored seed",
        1,
        _payload({
            "byte_end": record_id + 1,
            "byte_start": record_id,
            "document": f"seed-{record_id}",
        }),
        0,
        _key(cluster_id),
    )


def _observation(
        record_id: int,
        source_ref_key: StableRecordKey,
        *,
        split: str,
        cluster_base: int,
        stage: str = "W-03",
        logical_order: int = 1,
        supersedes_key: StableRecordKey | None = None,
        prerequisite_keys: tuple[StableRecordKey, ...] = (),
        license_partition: str = "CC0-1.0") -> ObservationRecord:
    """构造不含 expected/teacher/evaluator 私有字段的 ObservationRecord。"""
    return ObservationRecord(
        1,
        1,
        1,
        _key(60001),
        _key(60002),
        _key(record_id),
        stage,
        "sense-boundary",
        split,
        "zh",
        "typed-proposition",
        source_ref_key,
        license_partition,
        _key(cluster_base + 1),
        _key(cluster_base + 2),
        _key(cluster_base + 3),
        _key(cluster_base + 4),
        "forming",
        "support",
        "Proposition",
        _payload({
            "kind": "Proposition",
            "roles": [
                {"role": "SUBJECT", "value": record_id},
                {"role": "PREDICATE", "value": cluster_base},
            ],
        }),
        "NONE",
        supersedes_key,
        prerequisite_keys,
        logical_order,
    )


def _teacher(
        record_id: int,
        observation: ObservationRecord,
        source: SourceRefRecord) -> TeacherEvidenceRecord:
    """构造独立 teacher owner 的 typed Evidence。"""
    return TeacherEvidenceRecord(
        1,
        1,
        1,
        _key(60001),
        _key(60002),
        _key(record_id),
        observation.stable_key,
        "FORM_REVEAL",
        _payload({"evidence_kind": "FORM_REVEAL", "predicate": 701}),
        source.stable_key,
        observation.w_stage,
        0,
        _key(9001),
    )


def _evaluator(record_id: int, observation: ObservationRecord) -> EvaluatorLabelRecord:
    """构造只读 evaluator owner 的四态和结构标签。"""
    return EvaluatorLabelRecord(
        1,
        1,
        1,
        _key(60001),
        _key(60002),
        _key(record_id),
        observation.stable_key,
        _key(9101),
        "TRUE",
        _payload({"expected_structure": {"kind": "Proposition"}}),
        100,
        1,
        observation.w_stage,
        _key(9002),
    )


def _bundle():
    """返回来源簇、split 和 owner 均物理独立的最小 D-02A bundle。"""
    train_source = _source(101, 1001)
    held_source = _source(102, 1002)
    train = _observation(
        201, train_source.stable_key,
        split="train", cluster_base=2000,
    )
    held = _observation(
        202, held_source.stable_key,
        split="held_out", cluster_base=3000,
    )
    teacher = _teacher(301, train, train_source)
    evaluator = _evaluator(401, held)
    return (train_source, held_source), (train, held), (teacher,), (evaluator,)


def _dummy_file(
        record_kind: str,
        owner_kind: str,
        relative_path: str,
        split: str | None,
        key_value: int,
        clusters: tuple[StableRecordKey, ...],
        *,
        license_partition: str = "CC0-1.0") -> ArtifactFileIdentity:
    """构造只用于合同 round-trip 的完整文件身份。"""
    key = _key(key_value)
    return ArtifactFileIdentity(
        record_kind,
        owner_kind,
        relative_path,
        split,
        license_partition,
        1,
        "0" * 64,
        "1" * 64,
        1,
        1,
        key,
        key,
        clusters,
    )


def test_five_record_kinds_round_trip_without_float_or_mutable_payload():
    """五类最低对象均能规范往返，typed payload 不暴露内部可变状态。"""
    sources, observations, teachers, evaluators = _bundle()
    records = sources + observations + teachers + evaluators
    for record in records:
        assert record_from_dict(record.to_dict()) == record

    clusters = tuple(item.source_cluster_key for item in sources)
    files = (
        _dummy_file(RECORD_SOURCE_REF, "source", "source_refs.jsonl.gz", None, 1, clusters),
        _dummy_file(RECORD_OBSERVATION, "observation", "observations/train.jsonl.gz", "train", 2, clusters),
        _dummy_file(RECORD_TEACHER_EVIDENCE, "teacher", "owners/teacher/train.evidence.jsonl.gz", "train", 3, clusters),
        _dummy_file(RECORD_EVALUATOR_LABEL, "evaluator", "owners/evaluator/held_out.labels.jsonl.gz", "held_out", 4, clusters),
    )
    manifest = ArtifactManifest(
        1, 1, 1, 1, _key(60001), _key(60002),
        "AUTHORED_CC0_V1", "CC0-1.0", "PUBLIC",
        1, 1, 1, files, ("train", "held_out"), ("W-03",), clusters, (), "W-03",
    )
    assert record_from_dict(manifest.to_dict()) == manifest
    restored = observations[0].typed_payload.to_value()
    restored["kind"] = "mutated"
    assert observations[0].typed_payload.to_value()["kind"] == "Proposition"
    with pytest.raises(DatasetContractError, match="浮点"):
        _payload({"bad": 1.5})


@pytest.mark.parametrize(
    ("factory", "expected"),
    [
        (lambda: replace(_source(1, 11), source_key=""), "source_key"),
        (lambda: replace(_source(1, 11), license_id="UNKNOWN"), "允许许可"),
        (lambda: replace(_observation(2, _key(1), split="train", cluster_base=20),
                         schema_version=0), "正严格整数"),
        (lambda: replace(_observation(2, _key(1), split="train", cluster_base=20),
                         split="future"), "允许集合"),
    ],
)
def test_bad_version_license_empty_source_and_bad_split_fail_closed(factory, expected):
    """坏版本、坏许可、空来源和坏 split 均在对象入口失败。"""
    with pytest.raises(DatasetContractError, match=expected):
        factory()


def test_valid_bundle_has_direct_partition_and_owner_counts():
    """有效 bundle 返回来源、记录、owner、split 和阶段直接计数。"""
    sources, observations, teachers, evaluators = _bundle()
    report = validate_dataset_bundle(
        sources,
        observations,
        teachers,
        evaluators,
        source_key="AUTHORED_CC0_V1",
        license_partition="CC0-1.0",
        public_release=True,
    )
    assert report.source_ref_count == 2
    assert report.observation_count == 2
    assert report.teacher_evidence_count == 1
    assert report.evaluator_label_count == 1
    assert report.source_cluster_count == 2
    assert report.splits == ("train", "held_out")
    assert report.stages == ("W-03",)
    assert report.format_version == 1
    assert report.schema_version == 1
    assert report.course_version == 1
    assert report.dataset_key == _key(60001)
    assert report.artifact_key == _key(60002)


def test_bundle_and_manifest_reject_dataset_artifact_or_version_drift():
    """每条记录与 manifest 的 dataset/artifact/course 直接绑定不得漂移。"""
    sources, observations, teachers, evaluators = _bundle()
    drifted = replace(evaluators[0], artifact_key=_key(60003))
    with pytest.raises(DatasetValidationError, match="绑定漂移"):
        validate_dataset_bundle(
            sources, observations, teachers, (drifted,),
            source_key="AUTHORED_CC0_V1",
            license_partition="CC0-1.0",
            public_release=True,
        )

    clusters = tuple(item.source_cluster_key for item in sources)
    files = (
        _dummy_file(RECORD_SOURCE_REF, "source", "source_refs.jsonl.gz", None, 1, clusters),
        _dummy_file(RECORD_OBSERVATION, "observation", "observations/train.jsonl.gz", "train", 2, clusters),
        _dummy_file(RECORD_TEACHER_EVIDENCE, "teacher", "owners/teacher/train.evidence.jsonl.gz", "train", 3, clusters),
        _dummy_file(RECORD_EVALUATOR_LABEL, "evaluator", "owners/evaluator/held_out.labels.jsonl.gz", "held_out", 4, clusters),
    )
    wrong_manifest = ArtifactManifest(
        1, 1, 1, 1, _key(60001), _key(60003),
        "AUTHORED_CC0_V1", "CC0-1.0", "PUBLIC", 1, 1, 1,
        files, ("train", "held_out"), ("W-03",), clusters, (), "W-03",
    )
    with pytest.raises(DatasetValidationError, match="身份绑定"):
        validate_artifact_manifest(wrong_manifest, sources, observations)


def test_duplicate_stable_key_fails_across_record_kinds():
    """Source/Observation/teacher/evaluator 共用全局键空间，跨类型重复也拒绝。"""
    sources, observations, teachers, evaluators = _bundle()
    duplicate = replace(teachers[0], stable_key=observations[0].stable_key)
    with pytest.raises(DatasetValidationError, match="重复 stable key"):
        validate_dataset_bundle(
            sources, observations, (duplicate,), evaluators,
            source_key="AUTHORED_CC0_V1",
            license_partition="CC0-1.0",
            public_release=True,
        )


def test_mixed_license_pack_and_noassertion_public_release_fail_closed():
    """同一 pack 混许可和 NOASSERTION 公开发布均不能通过。"""
    sources, observations, teachers, evaluators = _bundle()
    mixed_source = replace(sources[1], license_id="CC-BY-4.0")
    mixed_observation = replace(observations[1], license_partition="CC-BY-4.0")
    with pytest.raises(DatasetValidationError, match="许可"):
        validate_dataset_bundle(
            (sources[0], mixed_source),
            (observations[0], mixed_observation),
            teachers,
            evaluators,
            source_key="AUTHORED_CC0_V1",
            license_partition="CC0-1.0",
            public_release=True,
        )

    local_source = _source(
        501,
        5001,
        source_key="CHINESE_SEMANTIC_KB_LOCAL",
        license_id="NOASSERTION-README-SHARING-NOT-RECOMMENDED",
        policy="LOCAL_ONLY",
    )
    local_observation = _observation(
        502,
        local_source.stable_key,
        split="train",
        cluster_base=5100,
        license_partition="NOASSERTION-README-SHARING-NOT-RECOMMENDED",
    )
    with pytest.raises(DatasetValidationError, match="公开 release"):
        validate_dataset_bundle(
            (local_source,),
            (local_observation,),
            (),
            (),
            source_key="CHINESE_SEMANTIC_KB_LOCAL",
            license_partition="NOASSERTION-README-SHARING-NOT-RECOMMENDED",
            public_release=True,
        )


def test_dedup_or_source_cluster_cross_split_fails_closed():
    """去重簇或真实来源簇跨 train/held-out 时不得伪装独立。"""
    sources, observations, teachers, evaluators = _bundle()
    leaked = replace(
        observations[1], dedup_cluster_key=observations[0].dedup_cluster_key)
    with pytest.raises(DatasetValidationError, match="跨 split"):
        validate_dataset_bundle(
            sources, (observations[0], leaked), teachers, evaluators,
            source_key="AUTHORED_CC0_V1",
            license_partition="CC0-1.0",
            public_release=True,
        )

    same_source_cluster = replace(
        sources[1], source_cluster_key=sources[0].source_cluster_key)
    with pytest.raises(DatasetValidationError, match="跨 split"):
        validate_dataset_bundle(
            (sources[0], same_source_cluster), observations, teachers, evaluators,
            source_key="AUTHORED_CC0_V1",
            license_partition="CC0-1.0",
            public_release=True,
        )


def test_future_stage_visibility_and_future_prerequisite_fail_closed():
    """当前阶段不能读取未来 Observation，也不能前置引用未来阶段。"""
    sources, observations, _, _ = _bundle()
    future = replace(observations[0], w_stage="W-04")
    with pytest.raises(DatasetValidationError, match="未来阶段"):
        validate_stage_visibility(
            (future,), (), (), current_stage="W-03", view_kind="training")

    future_target = replace(
        observations[0], stable_key=_key(601), w_stage="W-04", logical_order=1)
    current = replace(
        observations[0],
        stable_key=_key(602),
        logical_order=2,
        prerequisite_keys=(future_target.stable_key,),
    )
    with pytest.raises(DatasetValidationError, match="未来阶段"):
        validate_supersede_and_prerequisite_graph((future_target, current))
    assert sources


def test_supersede_cycle_fails_before_temporal_shortcut():
    """两节点 supersede 环必须被显式环检测拒绝，不能靠偶然顺序掩盖。"""
    base = _observation(701, _key(101), split="train", cluster_base=7000)
    first = replace(
        base,
        stable_key=_key(701),
        logical_order=1,
        supersedes_key=_key(702),
    )
    second = replace(
        base,
        stable_key=_key(702),
        logical_order=2,
        supersedes_key=_key(701),
    )
    with pytest.raises(DatasetValidationError, match="存在环"):
        validate_supersede_and_prerequisite_graph((first, second))


def test_observation_private_expected_or_teacher_fields_fail_closed():
    """expected/teacher/evaluator 私有字段不得塞入学生可见 typed payload。"""
    sources, observations, teachers, evaluators = _bundle()
    leaked = replace(
        observations[0],
        typed_payload=_payload({
            "kind": "Proposition",
            "nested": {"expected_output": "答案"},
        }),
    )
    with pytest.raises(DatasetValidationError, match="私有字段"):
        validate_dataset_bundle(
            sources, (leaked, observations[1]), teachers, evaluators,
            source_key="AUTHORED_CC0_V1",
            license_partition="CC0-1.0",
            public_release=True,
        )


def _write_minimal_pack(root: Path):
    """写覆盖四类 JSONL record 的最小单来源单许可 pack。"""
    sources, observations, teachers, evaluators = _bundle()
    cluster_train = (sources[0].source_cluster_key,)
    cluster_held = (sources[1].source_cluster_key,)
    files = (
        write_record_artifact(
            reversed(sources),
            root,
            ArtifactWriteSpec(
                RECORD_SOURCE_REF, "source", "source_refs.jsonl.gz", None,
                "CC0-1.0", cluster_train + cluster_held,
            ),
        ),
        write_record_artifact(
            (observations[0],),
            root,
            ArtifactWriteSpec(
                RECORD_OBSERVATION, "observation", "observations/train.jsonl.gz",
                "train", "CC0-1.0", cluster_train,
            ),
        ),
        write_record_artifact(
            (observations[1],),
            root,
            ArtifactWriteSpec(
                RECORD_OBSERVATION, "observation", "observations/held_out.jsonl.gz",
                "held_out", "CC0-1.0", cluster_held,
            ),
        ),
        write_record_artifact(
            teachers,
            root,
            ArtifactWriteSpec(
                RECORD_TEACHER_EVIDENCE, "teacher",
                "owners/teacher/train.evidence.jsonl.gz", "train",
                "CC0-1.0", cluster_train,
            ),
        ),
        write_record_artifact(
            evaluators,
            root,
            ArtifactWriteSpec(
                RECORD_EVALUATOR_LABEL, "evaluator",
                "owners/evaluator/held_out.labels.jsonl.gz", "held_out",
                "CC0-1.0", cluster_held,
            ),
        ),
    )
    manifest = ArtifactManifest(
        1,
        1,
        1,
        1,
        _key(60001),
        _key(60002),
        "AUTHORED_CC0_V1",
        "CC0-1.0",
        "PUBLIC",
        1,
        1,
        1,
        files,
        ("held_out", "train"),
        ("W-03",),
        cluster_train + cluster_held,
        (),
        "W-03",
    )
    validate_artifact_manifest(manifest, sources, observations)
    write_artifact_manifest(manifest, root)
    return sources, observations, teachers, evaluators, manifest


def test_jsonl_gzip_and_manifest_are_bit_identical_and_fully_readable(tmp_path):
    """两目录规范输出 bit-identical，并按双 hash、计数和键范围完整读取。"""
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first = _write_minimal_pack(first_root)
    second = _write_minimal_pack(second_root)
    first_manifest = first[-1]
    second_manifest = second[-1]
    assert first_manifest.canonical_bytes() == second_manifest.canonical_bytes()
    assert (first_root / "manifest.json").read_bytes() == (
        second_root / "manifest.json").read_bytes()
    loaded_manifest = read_artifact_manifest(first_root / "manifest.json")
    assert loaded_manifest == first_manifest
    for first_file, second_file in zip(first_manifest.files, second_manifest.files):
        assert first_file == second_file
        assert (first_root / first_file.relative_path).read_bytes() == (
            second_root / second_file.relative_path).read_bytes()
        loaded = read_record_artifact(first_root, first_file)
        assert len(loaded) == first_file.record_count
    source_file = next(
        item for item in first_manifest.files if item.record_kind == RECORD_SOURCE_REF)
    loaded_sources = read_record_artifact(first_root, source_file)
    assert tuple(item.stable_key for item in loaded_sources) == tuple(sorted(
        item.stable_key for item in first[0]))
    assert source_file.content_sha256 != source_file.transport_sha256


def test_manifest_content_identity_excludes_only_gzip_transport(tmp_path):
    """压缩 backend 可改变 transport，但不得改变规范内容身份。"""
    *_, manifest = _write_minimal_pack(tmp_path)
    first = manifest.files[0]
    changed = replace(
        first,
        transport_sha256="f" * 64,
        transport_size_bytes=first.transport_size_bytes + 1,
    )
    variant = replace(
        manifest,
        files=(changed, *manifest.files[1:]),
    )
    assert variant.sha256() != manifest.sha256()
    assert variant.content_identity_bytes() == manifest.content_identity_bytes()
    assert variant.content_sha256() == manifest.content_sha256()


def test_writer_rejects_duplicate_key_and_wrong_owner_path(tmp_path):
    """重复键以及 teacher/evaluator 写进 Observation 路径均在发布前失败。"""
    sources, observations, teachers, _ = _bundle()
    with pytest.raises(DatasetArtifactIOError, match="重复"):
        write_record_artifact(
            (sources[0], sources[0]),
            tmp_path,
            ArtifactWriteSpec(
                RECORD_SOURCE_REF, "source", "source_refs.jsonl.gz", None,
                "CC0-1.0", (sources[0].source_cluster_key,),
            ),
        )
    with pytest.raises(DatasetArtifactIOError, match="TeacherEvidence"):
        ArtifactWriteSpec(
            RECORD_TEACHER_EVIDENCE,
            "observation",
            "observations/train.jsonl.gz",
            "train",
            "CC0-1.0",
            (sources[0].source_cluster_key,),
        )
    assert observations and teachers


def test_reader_rejects_transport_mutation(tmp_path):
    """gzip transport 任一字节变化都不能被内容 reader 忽略。"""
    _, _, _, _, manifest = _write_minimal_pack(tmp_path)
    identity = next(
        item for item in manifest.files if item.record_kind == RECORD_SOURCE_REF)
    path = tmp_path / identity.relative_path
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(DatasetArtifactIOError, match="transport size"):
        read_record_artifact(tmp_path, identity)


def test_manifest_rejects_mixed_license_file_identity():
    """ArtifactManifest 本体也必须拒绝混许可文件。"""
    sources, _, _, _ = _bundle()
    clusters = tuple(item.source_cluster_key for item in sources)
    cc0 = _dummy_file(
        RECORD_SOURCE_REF, "source", "source_refs.jsonl.gz", None, 1, clusters)
    by = _dummy_file(
        RECORD_OBSERVATION, "observation", "observations/train.jsonl.gz",
        "train", 2, clusters, license_partition="CC-BY-4.0")
    with pytest.raises(DatasetContractError, match="混许可"):
        ArtifactManifest(
            1, 1, 1, 1, _key(60001), _key(60002),
            "AUTHORED_CC0_V1", "CC0-1.0", "PUBLIC",
            1, 1, 1, (cc0, by), ("train",), ("W-03",), clusters, (), "W-03",
        )
