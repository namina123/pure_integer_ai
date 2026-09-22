"""在 K 盘父图副本追加来源化对话结构，支持逐课程和 R-01 阶段恢复。"""
from __future__ import annotations

import argparse
from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3

from pure_integer_ai.cognition.shared.identity import OBJECT_REPRESENTATION
from pure_integer_ai.cognition.shared.representation_rendering import representation_parts
from pure_integer_ai.experiments.alias_relation_course import AliasRelationCourseLoader
from pure_integer_ai.experiments.materialize_trained_relation_generation import (
    _advance_graph_id_pool, _canonical_json, _profile_from_prefix, _read_object, _sha256,
)
from pure_integer_ai.experiments.ph2_generation_candidate_alias_contract import GenerationCandidateAliasCourseRequest
from pure_integer_ai.experiments.ph2_generation_candidate_alias_course import build_alias_relation_manifest
from pure_integer_ai.experiments.ph2_grounded_answer_order import install_source_generation_order
from pure_integer_ai.experiments.response_generation_course import compile_response_course
from pure_integer_ai.experiments.response_generation_graph import materialize_response_connector
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.experiments.trained_generation_connector_runtime import (
    TrainedGenerationConnectorRuntime, _ALIAS_BUDGET, _definition_graph, _generation_protocols,
    _lifecycle, _surface_protocol,
)
from pure_integer_ai.experiments.trained_relation_graph_runtime import TrainedRelationGraphRuntime
from pure_integer_ai.storage.assertion_identity import (
    IDENTITY_SOURCE_RECORD, IntegerIdentityRegistry,
)
from pure_integer_ai.storage.backend import SQLiteBackend


def _integers(value):
    """交换载体只能含严格整数及有序数组；转换不解释任何文字。"""
    if type(value) is int:
        return value
    if type(value) is list:
        return tuple(_integers(item) for item in value)
    raise ValueError("对话结构课程不得含字符串、浮点或布尔语义值")


def _explicit_frames(value, facts):
    """Resolve manifest-pinned old frames by complete proposition integer identity."""
    if value in (None, []):
        return {}
    entries = _integers(value)
    by_proposition = {fact.proposition.stable_key(): fact for fact in facts}
    result = {}
    for entry in entries:
        if (type(entry) is not tuple or len(entry) != 2
                or type(entry[0]) is not int or entry[0] <= 0
                or type(entry[1]) is not tuple or not entry[1]
                or entry[0] in result):
            raise ValueError("父框架显式映射不是完整纯整数协议")
        fact = by_proposition.get(entry[1])
        if fact is None:
            raise ValueError("父框架显式映射的命题不在当前 active 图")
        result[entry[0]] = fact
    return result


def _role_mapping(dynamic, candidate):
    """Return an exact role-carrier bijection, or None when it does not close."""
    if len(candidate.bindings) != len(dynamic):
        return None
    available = list(range(len(candidate.bindings)))
    mapping = []
    for part in dynamic:
        matches = [
            index for index in available
            if tuple(map(ord, candidate.bindings[index].surface)) == part[-1]
        ]
        if len(matches) != 1:
            return None
        index = matches[0]
        mapping.append(index)
        available.remove(index)
    return tuple(mapping) if not available else None


def _compile(database: Path, payload: bytes, parent_frame_map=()):
    """只读父图全部被标注的角色结构，复用原 R-01 profile 与表示族。"""
    record = _integers(json.loads(payload))
    if type(record) is not tuple or len(record) != 3 or record[:2] != (91525, 1):
        raise ValueError("未知对话观察协议")
    observations = record[2]
    if (not observations or len({item[0] for item in observations}) != len(observations)
            or tuple(item[0] for item in observations) != tuple(range(1, len(observations) + 1))):
        raise ValueError("对话观察身份必须完整连续且不重复")
    digest = tuple(hashlib.sha256(payload).digest())
    with TrainedGenerationConnectorRuntime(database) as runtime, TrainedRelationGraphRuntime(database) as core:
        if len(runtime._branches) != 1:
            raise ValueError("此阶段课程要求明确唯一语言分支")
        owner = runtime._branches[0]
        profiles = runtime._discover_alias_protocols()
        if len(profiles) != 1:
            raise ValueError("父图存在多个 R-01 profile，课程必须进一步声明目标")
        profile = _profile_from_prefix(*profiles[0])
        alias = runtime.alias_runtime(owner.branch)
        families = {representation_parts(binding.filler)[0]
                    for fact in alias.closure.consumer.lookup_relation(alias.selector.protocol.realizes_relation)
                    for binding in fact.proposition.bindings if binding.filler.object_kind == OBJECT_REPRESENTATION}
        if len(families) != 1:
            raise ValueError("课程目标表示族必须唯一")
        family = next(iter(families))
        facts = core.active_surface_facts()
        explicit_frames = _explicit_frames(parent_frame_map, facts)
        values = owner.definition_graph.value_protocol
        content = _generation_protocols(runtime.context, owner.branch).content
        courses = []
        for observation in observations:
            if type(observation[1]) is not int or observation[1] <= 0:
                raise ValueError("观察父图框架坐标非法")
            # 旧课程只携带完整整数角色载体，父图 ordinal 不是稳定身份。
            # 迁移必须由图内角色载体的双射唯一闭合，不能按 ordinal 或表层近邻猜测。
            dynamic = tuple(
                part for part in observation[3]
                if type(part) is tuple and part and part[0] == 2
            )
            pinned = explicit_frames.get(observation[1])
            if pinned is not None:
                mapping = _role_mapping(dynamic, pinned)
                if mapping is None:
                    raise ValueError("显式父命题没有保持完整角色整数双射")
                fact = pinned
                fact_index = facts.index(fact) + 1
            else:
                candidates = []
                for fact_index, candidate in enumerate(facts, 1):
                    mapping = _role_mapping(dynamic, candidate)
                    if mapping is not None:
                        candidates.append((fact_index, mapping, candidate))
                if len(candidates) != 1:
                    raise ValueError("对话角色整数载体无法唯一闭合到当前父图")
                fact_index, mapping, fact = candidates[0]
            remapped = []
            dynamic_index = 0
            for part in observation[3]:
                if part[0] == 2:
                    remapped.append((2, mapping[dynamic_index], part[-1]))
                    dynamic_index += 1
                else:
                    remapped.append(part)
            observation = (observation[0], fact_index, observation[2], tuple(remapped))
            course = compile_response_course(
                observation, digest=digest, fact=fact, parent_template=runtime._role_template(fact),
                values=values, family=family, content=content, surface=_surface_protocol(owner.branch))
            values = course.values
            courses.append(course)
        courses = tuple(replace(course, values=values) for course in courses)
        request = GenerationCandidateAliasCourseRequest(owner.branch, tuple(
            binding for course in courses for binding in course.aliases))
        manifest = build_alias_relation_manifest(profile, request)
    return courses, manifest


def materialize(*, source_database: Path, observations: Path, source_manifest: Path,
                parent_source_manifest: Path, target_root: Path,
                resume: bool = False, in_place: bool = False) -> dict:
    """追加真实顺序、动作拓扑和 R-01；失败保留全部历史与恢复点。"""
    if type(in_place) is not bool:
        raise TypeError("in_place 必须是严格 bool")
    source = source_database.resolve(strict=True)
    observation_path = observations.resolve(strict=True)
    ledger_path = source_manifest.resolve(strict=True)
    parent_ledger_path = parent_source_manifest.resolve(strict=True)
    payload = observation_path.read_bytes()
    ledger_bytes = ledger_path.read_bytes()
    parent_ledger_bytes = parent_ledger_path.read_bytes()
    ledger = json.loads(ledger_bytes)
    root = target_root.resolve()
    if (root.drive.upper() != "K:" or root == Path(root.anchor)
            or any(root == path or root in path.parents for path in (
                source, observation_path, ledger_path, parent_ledger_path))):
        raise ValueError("对话训练必须使用显式独立 K 盘目录")
    source_sha = _sha256(source)
    database = source if in_place else root / "training.sqlite3"
    course_sha = hashlib.sha256(payload).hexdigest()
    parent_ledger_sha = hashlib.sha256(parent_ledger_bytes).hexdigest()
    if (ledger.get("format") != "RESPONSE_STRUCTURE_SOURCE_LEDGER_V1"
            or not ledger.get("license_ids") or not ledger.get("attribution")
            or not ledger.get("rights_basis") or not ledger.get("selection_rule")
            or ledger.get("source_sha256") != course_sha
            or ledger.get("parent_database_sha256") != source_sha
            or ledger.get("parent_source_manifest_sha256") != parent_ledger_sha):
        raise ValueError("对话课程来源、父图或许可清单未闭合")
    identities = dict(parent_database_sha256=source_sha, source_sha256=course_sha,
                      source_manifest_sha256=hashlib.sha256(ledger_bytes).hexdigest(),
                      parent_source_manifest_sha256=parent_ledger_sha)
    cursor_path = root / "response_generation_cursor.json"
    if resume:
        cursor = _read_object(cursor_path, label="response generation cursor")
        identity_keys = tuple(
            key for key in identities
            if not (in_place and key == "parent_database_sha256"))
        if (cursor.get("in_place") != int(in_place)
                or cursor.get("database_path") != str(database)
                or any(cursor.get(key) != identities[key]
                       for key in identity_keys)):
            raise ValueError("恢复时输入身份发生变化")
        for name, expected in (("observations.int.json", payload), ("source_manifest.json", ledger_bytes),
                               ("parent_source_manifest.json", parent_ledger_bytes)):
            if (root / name).read_bytes() != expected:
                raise ValueError("恢复目录中的来源副本发生变化")
    else:
        if root.exists():
            raise ValueError("训练目录已存在，必须指定 resume 或新的 run identity")
        root.mkdir(parents=True)
        cursor = dict(
            schema_version=1, stage=0, course_cursor=0, order_count=0,
            in_place=int(in_place), database_path=str(database),
            database_bytes_before=source.stat().st_size,
            database_copy_count=0 if in_place else 1, **identities)
        for name, contents in (("observations.int.json", payload), ("source_manifest.json", ledger_bytes),
                               ("parent_source_manifest.json", parent_ledger_bytes)):
            (root / name).write_bytes(contents)

    def checkpoint():
        """同目录原子提交单调阶段，课程原文与 append-only 图保留不动。"""
        pending = root / "response_generation_cursor.pending"
        pending.write_bytes(_canonical_json(cursor))
        pending.replace(cursor_path)

    code_paths = (Path(__file__), Path(__file__).with_name("response_generation_course.py"),
                  Path(__file__).with_name("response_generation_graph.py"),
                  Path(__file__).with_name("ph2_grounded_answer_order.py"),
                  Path(__file__).with_name("trained_generation_connector_runtime.py"),
                  Path(__file__).parents[1] / "cognition/shared/generation_response.py")
    code_identity = {path.name: _sha256(path) for path in code_paths}
    versions = cursor.setdefault("code_versions", [])
    if not versions or versions[-1] != code_identity:
        versions.append(code_identity)
    checkpoint()
    if cursor["stage"] == 0:
        if not in_place:
            origin = sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)
            copied = sqlite3.connect(str(database))
            try:
                origin.backup(copied)
            finally:
                copied.close()
                origin.close()
        cursor["stage"] = 1
        checkpoint()
    if cursor["stage"] < 3:
        courses, manifest = _compile(
            database, payload, ledger.get("parent_frame_map", ()))
        manifest_sha = manifest.sha256()
        if cursor.get("alias_manifest_sha256") not in (None, manifest_sha):
            raise ValueError("恢复时生成课程身份漂移")
        if len(courses) != ledger.get("observation_count"):
            raise ValueError("冻结课程数量不闭合")
        cursor.update(alias_manifest_sha256=manifest_sha, connector_count=len(courses),
                      realization_count=len(manifest.entries))
        checkpoint()
        backend = SQLiteBackend(str(database))
        try:
            context = make_train_context(backend)
            _advance_graph_id_pool(backend)
            source_identities = IntegerIdentityRegistry(backend)
            for index, course in enumerate(courses):
                if index < cursor["course_cursor"]:
                    continue
                definition = course.response
                lifecycle = _lifecycle(context, definition.template.language_branch)
                graph = _definition_graph(context, definition.template.language_branch, lifecycle)
                graph.value_protocol = course.values
                source_identities.register(
                    IDENTITY_SOURCE_RECORD, definition.source.stable_key())
                count = install_source_generation_order(course.order, lifecycle)
                materialize_response_connector(graph, definition)
                backend.commit()
                artifact = gzip.compress(_canonical_json(list(definition.stable_key())), mtime=0)
                name = f"response-structure-{index + 1:03d}.int.json.gz"
                (root / name).write_bytes(artifact)
                cursor.update(course_cursor=index + 1, order_count=cursor["order_count"] + count)
                checkpoint()
            cursor["stage"] = 2
            checkpoint()
            AliasRelationCourseLoader(manifest, manifest_sha).load(context)
            backend.commit()
            cursor["stage"] = 3
            checkpoint()
        finally:
            backend.close()
    if cursor["stage"] < 4:
        with TrainedGenerationConnectorRuntime(database) as runtime:
            recovered = runtime.response_connectors()
            expected = tuple(tuple(json.loads(gzip.decompress((root / f"response-structure-{index:03d}.int.json.gz").read_bytes())))
                             for index in range(1, cursor["connector_count"] + 1))
            current = tuple(item for item in recovered if item.stable_key() in set(expected))
            if {item.stable_key() for item in current} != set(expected):
                raise ValueError("对话图只读恢复未完整保留课程成员及证据")
            atom_count = 0
            role_count = 0
            for item in current:
                owner = next(owner for owner in runtime._branches if owner.branch == item.template.language_branch)
                values = owner.definition_graph.value_protocol
                item.response_template(values)
                for binding in item.template.bindings:
                    if binding.source == values.role_filler_source:
                        role_count += 1
                    elif binding.source == values.constant_source:
                        proposal = runtime.alias_runtime(owner.branch).preview_surface(
                            binding.constant, owner.branch, budget=_ALIAS_BUDGET, allowed_prefix_steps=())
                        if proposal.result.selected is None:
                            raise ValueError("已训练语言原子的 R-01 未闭合")
                        atom_count += 1
            database_bytes = database.stat().st_size
            receipt = dict(
                cursor, stage=4, recovered_connector_count=len(current),
                recovered_role_slot_count=role_count,
                recovered_atom_route_count=atom_count,
                free_dialogue_complete=0, generated_response_count=0,
                model_sha256=_sha256(database), database_bytes=database_bytes,
                database_bytes_added=(
                    database_bytes - int(cursor["database_bytes_before"])),
                semantic_object_copy_count=0)
            (root / "response_generation_receipt.json").write_bytes(_canonical_json(receipt))
        cursor["stage"] = 4
        checkpoint()
    return _read_object(root / "response_generation_receipt.json", label="response generation receipt")


def main(argv=None) -> int:
    """只接受明确的父图、来源清单与 K 盘目录，不写包内默认文件。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-database", required=True, type=Path)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--parent-source-manifest", required=True, type=Path)
    parser.add_argument("--target-root", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--in-place", action="store_true",
        help="在 source-database 原地追加；target-root 只保存 cursor/receipt")
    arguments = parser.parse_args(argv)
    receipt = materialize(**vars(arguments))
    print(json.dumps({key: receipt[key] for key in (
        "stage", "connector_count", "order_count", "realization_count", "recovered_role_slot_count",
        "recovered_atom_route_count", "model_sha256", "free_dialogue_complete")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
