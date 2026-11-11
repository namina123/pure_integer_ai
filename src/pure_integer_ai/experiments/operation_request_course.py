"""物理许可请求问式的编译和纯整数图存储；发布运行只读图。"""
from __future__ import annotations

import hashlib
from pathlib import Path

from pure_integer_ai.cognition.shared.identity import GLOBAL_OWNER_SCOPE, SourceRef
from pure_integer_ai.cognition.understanding.query_operation_requests import FRAME, OperationRequestFrame
from pure_integer_ai.experiments.operation_fact_source import verify_operation_fact_source
from pure_integer_ai.experiments.ph2_authored_relation_compile import authored_relation_identity, authored_relation_role_identity
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line, parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w06_identity_protocol import W06_IDENTITY_VERSIONS
from pure_integer_ai.experiments.trained_relation_question_graph import _hash
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import TYPE_INT, SQLiteBackend, register_extension_table


def compile_train_operation_request(path, facts, program_witness):
    """teacher 只授权 gap 和 predicate/role/type reader；示例 target 不进入恢复图。"""
    _adapter, seeds = verify_operation_fact_source(facts)
    raw = Path(path).resolve(strict=True).read_bytes()
    if len(raw) > 1024 * 1024 or not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise ValueError("operation request needs one bounded physical train record")
    row = parse_canonical_json_bytes(raw[:-1], require_object=True)
    return _compile_operation_request_record(row, raw, facts, program_witness, seeds)


def compile_train_operation_request_record(row, facts, program_witness):
    """编译 typed 来源重建的内存记录，不将 JSONL 用作持久课程或整数存储。

    原交换入口与本入口共享同一来源、reader 和变量校验；规范行仅在内存中
    计算既有来源 digest，因此相同记录的完整 frame 身份保持逐字节一致。
    """
    _adapter, seeds = verify_operation_fact_source(facts)
    if type(row) is not dict:
        raise ValueError("operation request needs one bounded physical train record")
    raw = canonical_json_line(row)
    if len(raw) > 1024 * 1024:
        raise ValueError("operation request needs one bounded physical train record")
    return _compile_operation_request_record(row, raw, facts, program_witness, seeds)


def _compile_operation_request_record(row, raw, facts, program_witness, source_seeds):
    """共用原请求编译算法；只授权实际 train 来源与完整外部变量 reader。"""
    if (set(row) != {"question", "readers", "owner", "split", "license_id"}
            or (row["owner"], row["split"], row["license_id"]) != ("teacher", "train", "CC0-1.0")
            or type(row["question"]) is not str or not row["question"] or len(row["question"]) > 2048
            or type(row["readers"]) is not list or not 1 <= len(row["readers"]) <= 8
            or program_witness.source_identity != facts.course.source_identity
            or program_witness.license_identity != facts.course.license_identity):
        raise ValueError("operation request needs a licensed train reader instruction")
    seeds = {seed.seed_id: seed for seed in source_seeds}
    program, readers, positions = program_witness.program, [], []
    for instruction in row["readers"]:
        if not isinstance(instruction, dict) or set(instruction) != {"variable", "seed_id", "role_key"}:
            raise ValueError("operation request reader instruction is incomplete")
        variable, role = instruction["variable"], tuple(instruction["role_key"])
        seed = seeds.get(instruction["seed_id"])
        if seed is None or seed.split != "train" or seed.label_owner != "teacher":
            raise ValueError("operation request reader has no independent train fact")
        predicate = authored_relation_identity(seed.relation_kind).stable_key()
        allowed = tuple(slot for atom in program.atoms if atom.schema.relation.stable_key() == predicate
                        for semantic_role, coordinate in atom.role_variables if coordinate == variable
                        and semantic_role.stable_key() == role for slot in atom.schema.slots if slot.role == semantic_role)
        if not allowed or variable not in program.external_variables:
            raise ValueError("request reader does not bind an actual external program role")
        bindings = tuple(binding for binding in seed.bindings
                         if authored_relation_role_identity(binding.role_kind).stable_key() == role)
        if len(bindings) != 1:
            raise ValueError("request teacher does not contain one exact typed semantic reader")
        endpoints = tuple(endpoint for endpoint in seed.endpoints if endpoint.endpoint_id == bindings[0].endpoint_id)
        if len(endpoints) != 1:
            raise ValueError("request teacher reader has ambiguous endpoints")
        fragment = endpoints[0].surface_fragment
        start = row["question"].find(fragment)
        if start < 0 or row["question"].find(fragment, start + 1) >= 0:
            raise ValueError("request teacher term needs one complete input occurrence")
        kinds = tuple(sorted(set.intersection(*(set(slot.allowed_object_kinds) for slot in allowed))))
        if endpoints[0].object_kind not in kinds:
            raise ValueError("request exemplar kind is outside the full program reader contract")
        positions.append((start, start + len(fragment)))
        readers.append((variable, predicate, role, kinds))
    ordered_readers = sorted(zip(positions, readers), key=lambda item: item[0])
    positions = [position for position, _reader in ordered_readers]
    readers = [reader for _position, reader in ordered_readers]
    if (set(row[0] for row in readers) != set(program.external_variables)
            or len(readers) != len(program.external_variables)
            or any(left[1] > right[0] for left, right in zip(positions, positions[1:]))):
        raise ValueError("request source must bind every external variable once in input order")
    gaps, cursor = [], 0
    for start, end in positions:
        gaps.append(tuple(map(ord, row["question"][cursor:start])))
        cursor = end
    gaps.append(tuple(map(ord, row["question"][cursor:])))
    digest = hashlib.sha256(raw).digest()
    source = SourceRef(FRAME, (int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)) or 1, 0,
                       GLOBAL_OWNER_SCOPE, W06_IDENTITY_VERSIONS)
    return OperationRequestFrame(program.operation_key, source, tuple(digest), program_witness.source_identity,
                                 program_witness.license_identity, tuple(readers), tuple(gaps))


def _register(backend):
    """存储完整 frame 本体和有序成员，索引不定义其身份。"""
    register_extension_table(backend, "operation_request_frame", [("frame_hash", TYPE_INT), ("frame_version", TYPE_INT)],
        discipline=disc.DISC_APPEND_ONLY, recovery_key=("frame_hash",))
    register_extension_table(backend, "operation_request_part",
        [("frame_hash", TYPE_INT), ("ordinal", TYPE_INT), ("value", TYPE_INT)],
        discipline=disc.DISC_APPEND_ONLY, indexes=[("frame_hash", "ordinal")], recovery_key=("frame_hash", "ordinal"))


def materialize_operation_request_graph(frames, database):
    """写入唯一新 K identity；禁止覆盖旧图或以句子/答案做运行入口。"""
    target = Path(database).resolve()
    if target.drive.upper() != "K:" or target.exists():
        raise ValueError("operation request graph needs a new explicit K database")
    if type(frames) is not tuple or not frames or len(set(frames)) != len(frames):
        raise ValueError("operation request graph needs unique licensed frames")
    backend = SQLiteBackend(str(target))
    try:
        _register(backend)
        for frame in sorted(frames, key=lambda item: item.stable_key()):
            key, identity = frame.stable_key(), _hash(frame.stable_key())
            if backend.select("operation_request_frame", where={"frame_hash": identity}):
                raise ValueError("operation request frame identity collision")
            backend.insert("operation_request_frame", {"frame_hash": identity, "frame_version": FRAME})
            backend.insert_many("operation_request_part", [{"frame_hash": identity, "ordinal": ordinal, "value": value}
                for ordinal, value in enumerate(key)])
        backend.commit()
    finally:
        backend.close()


def read_operation_request_graph(database):
    """只读冷恢复全部来源及 reader，拒绝孤儿或截断成员。"""
    backend = SQLiteBackend(str(Path(database).resolve(strict=True)), read_only=True)
    try:
        _register(backend)
        rows = backend.select("operation_request_frame", order_by="frame_hash")
        parts = backend.select("operation_request_part", order_by="frame_hash, ordinal")
        if not rows or any(part["frame_hash"] not in {row["frame_hash"] for row in rows} for part in parts):
            raise ValueError("request graph is empty or has orphan members")
        result = []
        for row in rows:
            members = tuple(part for part in parts if part["frame_hash"] == row["frame_hash"])
            key = tuple(part["value"] for part in members)
            if (row["frame_version"] != FRAME or _hash(key) != row["frame_hash"]
                    or tuple(part["ordinal"] for part in members) != tuple(range(len(members)))):
                raise ValueError("request graph full frame identity drifted")
            result.append(OperationRequestFrame.from_stable_key(key))
        return tuple(sorted(result, key=lambda item: item.stable_key()))
    finally:
        backend.close()
