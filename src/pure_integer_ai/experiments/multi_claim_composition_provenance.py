"""多来源生成索引的完整整数权限成员与规范图级封套。"""
from __future__ import annotations

from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import TYPE_INT, register_extension_table


TABLE = "multi_claim_composition_variant_identity_part"
FORMAT = "PURE_INTEGER_MULTI_CLAIM_COMPOSITION_GRAPH_V3"
GRAPH_KIND = 91640


def permission_envelopes(permissions):
    """按variant顺序保留每个来源和许可；封套只标识集合，不授予执行权限。"""
    if (not permissions or tuple(row[0] for row in permissions)
            != tuple(sorted({row[0] for row in permissions}))):
        raise ValueError("composition permissions require unique ordered variants")
    envelopes = []
    for field in (1, 2):
        values = [GRAPH_KIND, 3, field, len(permissions)]
        for row in permissions:
            key = row[field]
            if (type(key) is not tuple or not key
                    or any(type(value) is not int or value < 0 for value in key)):
                raise ValueError("composition permission identity is incomplete")
            values.extend((row[0], len(key), *key))
        envelopes.append(tuple(values))
    return tuple(envelopes)


def _register(backend):
    """权限为有类型append-only成员，字段与序位均参与恢复身份。"""
    register_extension_table(backend, TABLE,
        [("variant_id", TYPE_INT), ("field_code", TYPE_INT),
         ("part_ordinal", TYPE_INT), ("part_value", TYPE_INT)],
        discipline=disc.DISC_APPEND_ONLY,
        indexes=[("variant_id", "field_code")],
        recovery_key=("variant_id", "field_code", "part_ordinal"))


def write_variant_permissions(backend, permissions):
    """逐成员写入完整来源和许可，不用摘要代替整数本体。"""
    permission_envelopes(permissions)
    _register(backend)
    backend.insert_many(TABLE, [
        {"variant_id": row[0], "field_code": field,
         "part_ordinal": ordinal, "part_value": value}
        for row in permissions for field in (1, 2)
        for ordinal, value in enumerate(row[field])])


def read_variant_permissions(backend, variant_ids):
    """完整覆盖且规范恢复每项权限；未知字段、孤儿、缺项与重复序位均拒绝。"""
    _register(backend)
    rows = backend.select(TABLE, order_by="variant_id,field_code,part_ordinal")
    groups = {}
    for row in rows:
        if row["variant_id"] not in variant_ids or row["field_code"] not in (1, 2):
            raise ValueError("composition permission has orphan or unknown members")
        groups.setdefault((row["variant_id"], row["field_code"]), []).append(row)
    permissions = []
    for identity in variant_ids:
        keys = []
        for field in (1, 2):
            members = groups.get((identity, field), ())
            if (not members or tuple(row["part_ordinal"] for row in members)
                    != tuple(range(len(members)))):
                raise ValueError("composition permission members are incomplete")
            keys.append(tuple(row["part_value"] for row in members))
        permissions.append((identity, *keys))
    result = tuple(permissions)
    permission_envelopes(result)
    return result
