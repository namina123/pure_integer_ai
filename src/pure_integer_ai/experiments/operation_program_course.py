"""Compile a physical CC0 train program against independently audited facts."""
from __future__ import annotations

import hashlib
from pathlib import Path

from pure_integer_ai.cognition.shared.identity import GLOBAL_OWNER_SCOPE, SourceRef
from pure_integer_ai.cognition.shared.integer_operation_program import (
    IntegerOperationProgram, OperationProgramWitness,
)
from pure_integer_ai.cognition.shared.relation_composition import schema_key
from pure_integer_ai.experiments.operation_fact_source import verify_operation_fact_source
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line, parse_canonical_json_bytes
from pure_integer_ai.experiments.ph2_w06_identity_protocol import W06_IDENTITY_VERSIONS


def compile_train_operation_program(source_path, facts):
    """读取原规范交换记录，复用完整来源与 schema 编译核。"""
    adapter, _seeds = verify_operation_fact_source(facts)
    raw = Path(source_path).resolve(strict=True).read_bytes()
    if len(raw) > 4 * 1024 * 1024 or not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise ValueError("operation source needs one bounded canonical physical record")
    row = parse_canonical_json_bytes(raw[:-1], require_object=True)
    return _compile_operation_program_record(row, raw, facts, adapter)


def compile_train_operation_program_record(row, facts):
    """编译 typed 课程恢复的内存记录；规范字节只计算既有来源身份。

    程序、许可、schema 和完整来源与交换入口完全相同；不保存 JSONL
    或课程正文到模型，external variable 仍由原整数程序合同验证。
    """
    adapter, _seeds = verify_operation_fact_source(facts)
    if type(row) is not dict:
        raise ValueError("operation source requires one canonical object record")
    raw = canonical_json_line(row)
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError("operation source needs one bounded canonical physical record")
    return _compile_operation_program_record(row, raw, facts, adapter)


def _compile_operation_program_record(row, raw, facts, adapter):
    """共享完整纯整数程序、许可及当前教师 schema 校验，不提供事实或表面。"""
    if (set(row) != {"program_key", "license_id", "owner", "split"}
            or row["split"] != "train" or row["owner"] != "teacher"
            or row["license_id"] != "CC0-1.0" or type(row["program_key"]) is not list):
        raise ValueError("operation source requires an integer-only CC0 train teacher")
    program = IntegerOperationProgram.from_stable_key(tuple(row["program_key"]))
    schemas = {schema.relation: schema_key(schema) for schema in adapter.schemas}
    if any(schemas.get(atom.schema.relation) != schema_key(atom.schema) for atom in program.atoms):
        raise ValueError("operation source schema differs from the actual teacher Core schema")
    digest = hashlib.sha256(raw).digest()
    source = SourceRef(92221, (int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)) or 1,
                       0, GLOBAL_OWNER_SCOPE, W06_IDENTITY_VERSIONS)
    return OperationProgramWitness(program, source, tuple(digest), facts.course.source_identity,
        facts.course.license_identity, tuple(sorted({candidate.source_ref for candidate in adapter.candidates},
                                                  key=SourceRef.stable_key)))
