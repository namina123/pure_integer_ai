"""Licensed train output obligations and a read-only integer graph codec."""
from __future__ import annotations

import hashlib
from pathlib import Path

from pure_integer_ai.cognition.shared.identity import GLOBAL_OWNER_SCOPE, SourceRef
from pure_integer_ai.cognition.shared.integer_operation_output import (
    OUTPUT_WITNESS, OperationOutputContract, OperationOutputWitness,
)
from pure_integer_ai.cognition.shared.integer_operation_output_verifier import validate_output_permission
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line, parse_canonical_json_bytes
from pure_integer_ai.experiments.ph2_w06_identity_protocol import W06_IDENTITY_VERSIONS
from pure_integer_ai.experiments.trained_relation_question_graph import _hash
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import TYPE_INT, SQLiteBackend, register_extension_table


TABLE, PARTS = "operation_output_obligation", "operation_output_obligation_part"


def compile_train_operation_output(path, programs, *, rules=()):
    """从旧物理交换记录恢复合同，来源与权限交由同一个编译核验证。"""
    raw = Path(path).resolve(strict=True).read_bytes()
    if len(raw) > 4 * 1024 * 1024 or not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise ValueError("output obligation needs one bounded physical train record")
    row = parse_canonical_json_bytes(raw[:-1], require_object=True)
    return _compile_operation_output_record(row, raw, programs, rules=rules)


def compile_train_operation_output_record(row, programs, *, rules=()):
    """消费typed课程恢复的完整记录；规范字节仅在内存计算既有来源身份。"""
    if type(row) is not dict:
        raise ValueError("output obligation requires an integer-only CC0 train instruction")
    raw = canonical_json_line(row)
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError("output obligation needs one bounded physical train record")
    return _compile_operation_output_record(row, raw, programs, rules=rules)


def _compile_operation_output_record(row, raw, programs, *, rules):
    """两个入口共享完整合同、独立规则及实际程序权限，不按输入载体放宽。"""
    if (set(row) != {"contract_key", "owner", "split", "license_id"}
            or (row["owner"], row["split"], row["license_id"]) != ("teacher", "train", "CC0-1.0")
            or type(row["contract_key"]) is not list):
        raise ValueError("output obligation requires an integer-only CC0 train instruction")
    contract = OperationOutputContract.from_stable_key(tuple(row["contract_key"]))
    if any(path.rule_key and path.rule_key not in {rule.stable_key() for rule in rules}
           for path in contract.paths):
        raise ValueError("output explanation rule lacks independent physical train permission")
    digest = hashlib.sha256(raw).digest()
    source = SourceRef(OUTPUT_WITNESS, (int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)) or 1,
                       0, GLOBAL_OWNER_SCOPE, W06_IDENTITY_VERSIONS)
    witness = OperationOutputWitness(contract, source, tuple(digest))
    validate_output_permission(witness, programs)
    return witness


def _register(backend):
    register_extension_table(backend, TABLE, [("obligation_hash", TYPE_INT)],
        discipline=disc.DISC_APPEND_ONLY, indexes=[("obligation_hash",)], recovery_key=("obligation_hash",))
    register_extension_table(backend, PARTS,
        [("obligation_hash", TYPE_INT), ("ordinal", TYPE_INT), ("value", TYPE_INT)],
        discipline=disc.DISC_APPEND_ONLY, indexes=[("obligation_hash", "ordinal")],
        recovery_key=("obligation_hash", "ordinal"))


def validate_operation_output_witnesses(witnesses, programs):
    if (type(witnesses) is not tuple or any(not isinstance(item, OperationOutputWitness) for item in witnesses)
            or len({item.contract.program_witness_key for item in witnesses}) != len(witnesses)):
        raise ValueError("output graph has competing or incomplete obligations")
    for witness in witnesses:
        validate_output_permission(witness, programs)


def materialize_operation_output_graph(witnesses, programs, database):
    validate_operation_output_witnesses(witnesses, programs)
    target = Path(database).resolve()
    if target.exists():
        raise FileExistsError("output obligations require a new graph identity")
    backend = SQLiteBackend(str(target), performance_mode="bulk")
    try:
        _register(backend)
        for witness in sorted(witnesses, key=OperationOutputWitness.stable_key):
            key = witness.stable_key()
            identity = _hash(key)
            if backend.select(TABLE, where={"obligation_hash": identity}):
                raise ValueError("output obligation hash collision")
            backend.insert(TABLE, {"obligation_hash": identity})
            backend.insert_many(PARTS, [{"obligation_hash": identity, "ordinal": ordinal, "value": value}
                                      for ordinal, value in enumerate(key)])
        backend.commit()
    finally:
        backend.close()


def read_operation_output_graph(database, programs):
    backend = SQLiteBackend(str(Path(database).resolve(strict=True)), read_only=True)
    try:
        _register(backend)
        rows, parts = backend.select(TABLE, order_by="obligation_hash"), backend.select(PARTS,
            order_by="obligation_hash, ordinal")
        ids = {row["obligation_hash"] for row in rows}
        if len(ids) != len(rows) or any(part["obligation_hash"] not in ids for part in parts):
            raise ValueError("output graph has duplicate identities or orphan parts")
        result = []
        for row in rows:
            members = tuple(part for part in parts if part["obligation_hash"] == row["obligation_hash"])
            if not members or tuple(part["ordinal"] for part in members) != tuple(range(len(members))):
                raise ValueError("output graph full integer witness is incomplete")
            key = tuple(part["value"] for part in members)
            if _hash(key) != row["obligation_hash"]:
                raise ValueError("output graph full witness identity drifted")
            result.append(OperationOutputWitness.from_stable_key(key))
        witnesses = tuple(sorted(result, key=OperationOutputWitness.stable_key))
        validate_operation_output_witnesses(witnesses, programs)
        return witnesses
    finally:
        backend.close()
