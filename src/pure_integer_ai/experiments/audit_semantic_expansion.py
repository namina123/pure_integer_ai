"""Read-only semantic expansion audit for the canonical integer graph.

The physical duplicate audit is necessary but cannot detect two different
identity hashes carrying the same integer semantic payload.  This module
groups only non-text integer fields and reports alias candidates, provenance
mix, qualifier fan-out, and connector variant expansion.  It never writes to
the database and never selects source-record raw text.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


FORMAT = "PURE_INTEGER_SEMANTIC_EXPANSION_AUDIT_V1"
_OBJECT_CODEC_COLUMNS = (
    "object_kind", "identity_codec", "component_size", "codec_ref_hash",
    "codec_value_0", "codec_value_1", "codec_value_2", "codec_value_3",
    "codec_value_4", "codec_value_5", "codec_value_6", "codec_value_7",
)
_OBJECT_FULL_COLUMNS = (
    "object_kind", "space_id", "local_id", "space_type", "space_type_hash",
    "space_name_hash", "identity_codec", "owner_tenant_id", "owner_user_id",
    "owner_session_id", "owner_visibility", "corpus_version",
    "parser_version", "primitive_version", "curriculum_version",
    "component_size", "codec_ref_hash", "codec_value_0", "codec_value_1",
    "codec_value_2", "codec_value_3", "codec_value_4", "codec_value_5",
    "codec_value_6", "codec_value_7",
)
_ASSERTION_COLUMNS = (
    "assertion_role", "key_version", "relation_kind", "subject_object_kind",
    "subject_space_id", "subject_local_id", "subject_tenant_id",
    "subject_user_id", "subject_session_id", "subject_visibility",
    "subject_corpus_version", "subject_parser_version",
    "subject_primitive_version", "subject_curriculum_version",
    "object_object_kind", "object_space_id", "object_local_id",
    "object_tenant_id", "object_user_id", "object_session_id",
    "object_visibility", "object_corpus_version", "object_parser_version",
    "object_primitive_version", "object_curriculum_version", "scope_hash",
    "provenance_kind", "epistemic_origin", "content_version", "qualifier_0",
    "qualifier_1", "qualifier_2", "qualifier_size",
)


def _quote(name: str) -> str:
    if not name or any(ch not in "abcdefghijklmnopqrstuvwxyz"
                       "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_" for ch in name):
        raise ValueError(f"unsafe identifier: {name!r}")
    return '"' + name + '"'


def _count(conn: sqlite3.Connection, table: str) -> int:
    return int(conn.execute(f"SELECT COUNT(*) FROM {_quote(table)}").fetchone()[0])


def _distribution(conn: sqlite3.Connection, table: str, column: str) -> list[dict[str, int]]:
    rows = conn.execute(
        f"SELECT {_quote(column)}, COUNT(*) FROM {_quote(table)} "
        f"GROUP BY {_quote(column)} ORDER BY {_quote(column)}"
    )
    return [{"value": (None if value is None else int(value)), "count": int(count)}
            for value, count in rows]


def _alias_groups(conn: sqlite3.Connection, table: str, identity: str,
                  columns: tuple[str, ...]) -> dict[str, int]:
    projection = ", ".join(_quote(item) for item in columns)
    sql = (
        "SELECT COUNT(*), COALESCE(SUM(distinct_ids - 1), 0) FROM ("
        f"SELECT {projection}, COUNT(DISTINCT {_quote(identity)}) AS distinct_ids "
        f"FROM {_quote(table)} GROUP BY {projection} "
        "HAVING COUNT(DISTINCT " + _quote(identity) + ") > 1)"
    )
    groups, excess = conn.execute(sql).fetchone()
    return {"groups": int(groups or 0), "distinct_identity_excess": int(excess or 0)}


def audit_database(path: Path) -> dict[str, Any]:
    conn = sqlite3.connect("file:" + str(path) + "?mode=ro", uri=True)
    try:
        object_count = _count(conn, "graph_object")
        object_components = _count(conn, "graph_object_component")
        identity_headers = _count(conn, "identity_header")
        identity_parts = _count(conn, "identity_part")
        assertions = _count(conn, "assertion_record")
        qualifiers = _count(conn, "assertion_qualifier")
        sources = _count(conn, "source_record")
        bindings = _count(conn, "artifact_semantic_binding")
        binding_parts = _count(conn, "artifact_semantic_binding_part")
        return {
            "database": str(path),
            "bytes": path.stat().st_size,
            "read_only": 1,
            "source_text_selected": 0,
            "row_counts": {
                "graph_object": object_count,
                "graph_object_component": object_components,
                "identity_header": identity_headers,
                "identity_part": identity_parts,
                "assertion_record": assertions,
                "assertion_qualifier": qualifiers,
                "source_record": sources,
                "artifact_semantic_binding": bindings,
                "artifact_semantic_binding_part": binding_parts,
            },
            "expansion": {
                "graph_object_components_per_object": (
                    object_components / object_count if object_count else 0.0),
                "identity_parts_per_header": (
                    identity_parts / identity_headers if identity_headers else 0.0),
                "qualifiers_per_assertion": (
                    qualifiers / assertions if assertions else 0.0),
                "binding_parts_per_binding": (
                    binding_parts / bindings if bindings else 0.0),
            },
            "semantic_alias_candidates": {
                "graph_object_codec": _alias_groups(
                    conn, "graph_object", "identity_hash", _OBJECT_CODEC_COLUMNS),
                "graph_object_full_payload": _alias_groups(
                    conn, "graph_object", "identity_hash", _OBJECT_FULL_COLUMNS),
                "assertion_record_payload": _alias_groups(
                    conn, "assertion_record", "identity_hash", _ASSERTION_COLUMNS),
            },
            "provenance_distribution": {
                "assertion_provenance_kind": _distribution(
                    conn, "assertion_record", "provenance_kind"),
                "assertion_epistemic_origin": _distribution(
                    conn, "assertion_record", "epistemic_origin"),
                "assertion_content_version": _distribution(
                    conn, "assertion_record", "content_version"),
            },
            "qualifier_size_distribution": _distribution(
                conn, "assertion_record", "qualifier_size"),
            "connector_variant_counts": {
                "by_carrier_code": _distribution(
                    conn, "artifact_semantic_binding", "carrier_code"),
                "by_binding_state": _distribution(
                    conn, "artifact_semantic_binding", "binding_state"),
            },
        }
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("databases", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    missing = [str(path) for path in args.databases if not path.is_file()]
    if missing:
        parser.error("database not found: " + ", ".join(missing))
    report = {
        "format": FORMAT,
        "schema_version": 1,
        "read_only": 1,
        "source_text_selected": 0,
        "databases": [audit_database(path) for path in args.databases],
    }
    target = args.output.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=True, sort_keys=True,
                                 separators=(",", ":")) + "\n", encoding="ascii")
    print(json.dumps({
        "database_count": len(report["databases"]),
        "format": FORMAT,
    }, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["audit_database", "main"]
