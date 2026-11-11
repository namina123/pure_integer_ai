"""从显式许可的整数组合课程学习请求间隔，运行时仍只消费原生图框架。"""
from __future__ import annotations

import hashlib

from pure_integer_ai.cognition.shared.identity import SourceRef
from pure_integer_ai.cognition.understanding.query_operation_requests import FRAME, OperationRequestFrame
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


def compile_train_operation_request_gap_record(record):
    """严格消费完整父框架和逐 gap 来源指令，不推断可互换词或扩大操作权限。"""
    required = {"owner", "split", "license_id", "parent_frame_keys", "base_parent_ordinal",
                "authorized_gap_ordinals", "gap_parent_ordinals"}
    if (type(record) is not dict or set(record) != required
            or (record["owner"], record["split"], record["license_id"]) != ("teacher", "train", "CC0-1.0")):
        raise ValueError("request gap learning requires a licensed teacher train instruction")
    keys = record["parent_frame_keys"]
    if (type(keys) is not list or not 2 <= len(keys) <= 8
            or any(type(key) is not list for key in keys)):
        raise ValueError("request gap learning requires bounded complete parent frames")
    parents = tuple(OperationRequestFrame.from_stable_key(tuple(key)) for key in keys)
    if len(set(parents)) != len(parents):
        raise ValueError("request gap learning parent frames must be unique")
    first = parents[0]
    contract = lambda frame: (frame.operation_key, frame.readers, frame.source_identity, frame.license_identity,
                              frame.source.owner, frame.source.versions)
    if any(contract(parent) != contract(first) for parent in parents):
        raise ValueError("request gap learning cannot mix operation/reader/source/license contracts")
    base, authorized, choices = (record[name] for name in (
        "base_parent_ordinal", "authorized_gap_ordinals", "gap_parent_ordinals"))
    if (type(base) is not int or not 0 <= base < len(parents)
            or type(authorized) is not list or not authorized
            or any(type(index) is not int or not 0 <= index < len(first.gaps) for index in authorized)
            or authorized != sorted(set(authorized))
            or type(choices) is not list or len(choices) != len(first.gaps)
            or any(type(index) is not int or not 0 <= index < len(parents) for index in choices)
            or any(index not in authorized and parent != base for index, parent in enumerate(choices))):
        raise ValueError("request gap learning has an unauthorized or incomplete gap selection")
    gaps = tuple(parents[parent].gaps[index] for index, parent in enumerate(choices))
    if any(gaps == parent.gaps for parent in parents):
        raise ValueError("request gap sequence already exists; preserve its original source")
    raw = canonical_json_line(record)
    if len(raw) > 1024 * 1024:
        raise ValueError("request gap instruction exceeds its source budget")
    digest = hashlib.sha256(raw).digest()
    source = SourceRef(FRAME, (int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)) or 1,
                       0, first.source.owner, first.source.versions)
    return OperationRequestFrame(first.operation_key, source, tuple(digest), first.source_identity,
                                 first.license_identity, first.readers, gaps)
