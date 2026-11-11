"""R45 -> R46 relation surface artifact identity rebinding checks."""
from __future__ import annotations

import sqlite3
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT,
    OBJECT_ENTITY,
    OBJECT_PROPOSITION,
    OBJECT_ROLE,
    ObjectIdentity,
)
from pure_integer_ai.experiments import trained_relation_surface_artifact as artifact
from pure_integer_ai.experiments.trained_relation_graph_runtime import (
    ActiveRelationSurface,
    RelationSurfaceBinding,
    RelationSurfaceFrame,
)


def _fixture_records():
    proposition = ObjectIdentity(OBJECT_PROPOSITION, (81001,))
    predicate = ObjectIdentity(OBJECT_CONCEPT, (81002,))
    role = ObjectIdentity(OBJECT_ROLE, (81003,))
    filler = ObjectIdentity(OBJECT_ENTITY, (81004,))
    fact = ActiveRelationSurface(
        proposition, predicate, "属于",
        (RelationSurfaceBinding(role, filler, "猫", 501, 0, 1),),
        "猫属于动物", 501, 1, 3)
    frame = RelationSurfaceFrame(
        proposition, predicate, (role,), ("", "属于动物"), 501, 0, 6)
    return fact, frame


def _write_artifact(path, core_sha256, fact, frame, *, schema_version=1):
    connection = sqlite3.connect(path)
    try:
        artifact._create_schema(connection)
        connection.execute(
            "INSERT INTO artifact_meta(key_id, payload) VALUES (?, ?)",
            (1, artifact.encode_integer_tuple(
                artifact._header(core_sha256, 1, 1, schema_version=schema_version))),
        )
        connection.execute(
            "INSERT INTO relation_surface_row "
            "(row_kind, ordinal, payload) VALUES (?, ?, ?)",
            (artifact._FACT_ROW, 0,
             artifact.encode_integer_tuple(artifact._pack_fact(fact, schema_version=schema_version))),
        )
        connection.execute(
            "INSERT INTO relation_surface_row "
            "(row_kind, ordinal, payload) VALUES (?, ?, ?)",
            (artifact._FRAME_ROW, 0,
             artifact.encode_integer_tuple(artifact._pack_frame(frame, schema_version=schema_version))),
        )
        connection.commit()
    finally:
        connection.close()


def _active_prop(proposition):
    return SimpleNamespace(
        definition=SimpleNamespace(proposition=proposition))


@pytest.mark.parametrize('schema_version', (1, 2, 3))
def test_rebind_preserves_integer_rows_and_changes_only_core_binding(
        tmp_path, monkeypatch, schema_version):
    source_core = tmp_path / "source.sqlite3"
    target_core = tmp_path / "target.sqlite3"
    source_artifact = tmp_path / "source-surfaces.sqlite3"
    output = tmp_path / "rebound-surfaces.sqlite3"
    source_core.write_bytes(b"source core")
    target_core.write_bytes(b"target core")
    fact, frame = _fixture_records()
    if schema_version == 3:
        from pure_integer_ai.experiments.relation_language_carrier import learn_carrier_frame
        frame = learn_carrier_frame(fact, replace(frame, envelope_end=len(fact.evidence_surface)))
    if schema_version != 1:
        fact = replace(fact, evidence_surface='')
    _write_artifact(
        source_artifact, artifact.sha256_path(source_core), fact, frame, schema_version=schema_version)

    active_by_path = {
        source_core.resolve(): (fact.proposition,),
        target_core.resolve(): (fact.proposition,),
    }

    class FakeRuntime:
        def __init__(self, database, *, surface_artifact_database):
            self.database = database

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def active_propositions(self):
            return tuple(_active_prop(item)
                         for item in active_by_path[self.database])

    monkeypatch.setattr(artifact, "TrainedRelationGraphRuntime", FakeRuntime)
    source_rows_sha256 = artifact._artifact_row_payload_sha256(source_artifact)
    report = artifact.rebind_relation_surface_artifact(
        source_core, source_artifact, target_core, output)

    rebound_facts, rebound_frames, metadata = artifact.read_relation_surface_artifact(
        output, expected_core_sha256=artifact.sha256_path(target_core))
    assert rebound_facts == (fact,)
    assert rebound_frames == (frame,)
    assert metadata["core_sha256"] == artifact.sha256_path(target_core)
    assert metadata['schema_version'] == schema_version
    assert artifact._artifact_row_payload_sha256(output) == source_rows_sha256
    assert report["record_payload_sha256"] == source_rows_sha256
    assert report["active_proposition_count"] == 1
    assert not tuple(tmp_path.glob(".rebound-surfaces.sqlite3.*.tmp"))


def test_rebind_rejects_target_active_set_mismatch_without_committing(
        tmp_path, monkeypatch):
    source_core = tmp_path / "source.sqlite3"
    target_core = tmp_path / "target.sqlite3"
    source_artifact = tmp_path / "source-surfaces.sqlite3"
    output = tmp_path / "rebound-surfaces.sqlite3"
    source_core.write_bytes(b"source core")
    target_core.write_bytes(b"target core")
    fact, frame = _fixture_records()
    _write_artifact(
        source_artifact, artifact.sha256_path(source_core), fact, frame)
    other = ObjectIdentity(OBJECT_PROPOSITION, (81005,))
    active_by_path = {
        source_core.resolve(): (fact.proposition,),
        target_core.resolve(): (other,),
    }

    class FakeRuntime:
        def __init__(self, database, *, surface_artifact_database):
            self.database = database

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def active_propositions(self):
            return tuple(_active_prop(item)
                         for item in active_by_path[self.database])

    monkeypatch.setattr(artifact, "TrainedRelationGraphRuntime", FakeRuntime)
    with pytest.raises(
            artifact.RelationSurfaceArtifactError,
            match="target active Core"):
        artifact.rebind_relation_surface_artifact(
            source_core, source_artifact, target_core, output)

    assert not output.exists()
    assert not tuple(tmp_path.glob(".rebound-surfaces.sqlite3.*.tmp"))
