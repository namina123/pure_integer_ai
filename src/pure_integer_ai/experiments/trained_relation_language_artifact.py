"""构建不保存完整来源句的V3语言成员组件，保留真实Core及载体边缘。"""
from dataclasses import replace
from pathlib import Path
from contextlib import closing
import sqlite3

from pure_integer_ai.experiments.trained_relation_graph_runtime import TrainedRelationGraphRuntime
from pure_integer_ai.experiments.trained_relation_surface_artifact import (
    CARRIER_LANGUAGE_ARTIFACT_SCHEMA_VERSION, _FACT_ROW, _FRAME_ROW, _create_schema,
    _header, _pack_fact, _pack_frame, read_relation_surface_artifact,
    sha256_path, validate_language_members,
)
from pure_integer_ai.storage.integer_codec import encode_integer_tuple
from pure_integer_ai.experiments.relation_language_carrier import learn_carrier_frame


LANGUAGE_VERSION = CARRIER_LANGUAGE_ARTIFACT_SCHEMA_VERSION


def materialize_relation_language_artifact(source_core, source_artifact, target_core, database):
    """从原模型的角色/语言成员创建新组件，不复制或重构完整来源句。

    原模型只读保留。先核对完整父命题，再验证新组件与最终Core的全部
    active命题相等。写入仅发生在显式K根，新文件验证后才公开最终路径。
    """
    source = Path(source_core).resolve(strict=True)
    language = Path(source_artifact).resolve(strict=True)
    target = Path(target_core).resolve(strict=True)
    source_sha, target_sha = sha256_path(source), sha256_path(target)
    with TrainedRelationGraphRuntime(source, surface_artifact_database=language) as runtime:
        parent = runtime.active_propositions()
        original = runtime.active_surface_facts()
        by_key = {fact.proposition.stable_key(): fact for fact in original}
        frames = tuple(learn_carrier_frame(by_key[frame.proposition.stable_key()], frame)
                       for frame in runtime.active_surface_frames())
        facts = tuple(replace(fact, evidence_surface='') for fact in original)
    result = materialize_relation_language_members(target, facts, frames, database,
        expected_propositions=parent)
    if sha256_path(source) != source_sha or sha256_path(target) != target_sha:
        raise ValueError('read-only Core changed during language materialization')
    return result


def materialize_relation_language_members(target_core, facts, frames, database, *,
                                        expected_propositions=None):
    """首次落盘即保存V3无正文成员，并由普通消费者冷恢复全部当前Core。

    调用方负责训练期来源核验。这里不读取来源正文、不从空正文恢复父框架，
    只有完整语言成员与当前active集合共同闭合后才公布最终文件。
    """
    target = Path(target_core).resolve(strict=True)
    output = Path(database).resolve()
    staging = output.with_name(output.name + '.building')
    if (target.drive.upper() != 'K:' or output.drive.upper() != 'K:'
            or output.exists() or staging.exists()):
        raise ValueError('language component needs a new explicit K identity')
    target_sha = sha256_path(target)
    validate_language_members(facts, frames, require_carrier=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(staging)) as db, db:
        _create_schema(db)
        header = _header(target_sha, len(facts), len(frames), schema_version=LANGUAGE_VERSION)
        db.execute('INSERT INTO artifact_meta VALUES(1,?)', (encode_integer_tuple(header),))
        db.executemany('INSERT INTO relation_surface_row VALUES(?,?,?)', (
            (_FACT_ROW, index, encode_integer_tuple(_pack_fact(fact,
                schema_version=LANGUAGE_VERSION))) for index, fact in enumerate(facts)))
        db.executemany('INSERT INTO relation_surface_row VALUES(?,?,?)', (
            (_FRAME_ROW, index, encode_integer_tuple(_pack_frame(frame, schema_version=LANGUAGE_VERSION)))
            for index, frame in enumerate(frames)))
    restored_facts, restored_frames, _metadata = read_relation_surface_artifact(staging,
        expected_core_sha256=target_sha)
    if restored_facts != facts or restored_frames != frames:
        raise ValueError('language members changed during typed storage')
    with TrainedRelationGraphRuntime(target, surface_artifact_database=staging) as runtime:
        if (expected_propositions is not None
                and runtime.active_propositions() != expected_propositions):
            raise ValueError('target Core differs from the complete parent propositions')
        if (runtime.active_surface_facts() != facts or runtime.active_surface_frames() != frames):
            raise ValueError('language members differ in the current production runtime')
    if sha256_path(target) != target_sha:
        raise ValueError('read-only Core changed during language materialization')
    staging.rename(output)
    return dict(schema_version=LANGUAGE_VERSION, fact_count=len(facts),
        frame_count=len(frames), full_source_sentence_count=0, core_sha256=target_sha,
        artifact_sha256=sha256_path(output), free_dialogue_complete=0, production_authorized=0)
