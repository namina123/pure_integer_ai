"""Wave 1 批量课程与准入合同的纯整数、只读审计工具。

本模块只检查物化课程和已经保存的证据，不执行 Core、Memory、Query 或生成。
课程中的 operation/family 标签是声明，只有带有实际整数图编辑前后证据的 case
才可以进入行为资格；缺证据统一保持 UNKNOWN，结构冲突则 FAIL。
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


SPLITS = ("train", "held_out", "negative", "conflict")
MIN_CASES_PER_SPLIT = 20
MIN_RELATION_FAMILIES = 5
MIN_OPERATION_FAMILIES = 8
MIN_NOVEL_HELD_OUT = 20
MIN_SEMANTIC_BEHAVIORS = 8
PASS = "PASS"
FAIL = "FAIL"
UNKNOWN = "UNKNOWN"

_FORMAT = "PURE_INTEGER_WAVE1_SPLIT_V1"
_ROW_FORMAT = "PURE_INTEGER_WAVE1_CASE_V1"
_REQUIRED_ROW_KEYS = {
    "case_id", "split_code", "source_identity", "license_identity",
    "source_ref_key", "source_record_key", "semantic_key", "document_key",
    "dialogue_chain_key", "concept_cluster_key", "shape_group_key",
    "template_group_key", "relation_key", "relation_combo_key",
    "relation_family_code", "operation_key", "topology_record", "input_values",
    "free_dialogue_complete",
}
_PROVENANCE_FIELDS = (
    "source_ref_key", "source_record_key", "semantic_key", "document_key",
    "dialogue_chain_key", "concept_cluster_key", "shape_group_key",
    "template_group_key",
)
_FORBIDDEN_KEYS = {
    "answer", "answer_table", "knowledge", "knowledge_text", "raw_text",
    "source_text", "whole_sentence", "whole_sentence_or_answer_table",
    "replay", "old_answer", "opencc", "jsonl_publish_entry",
}
_ALLOWED_NON_INTEGER_KEYS = {"format"}


def sha256(path: Path) -> str:
    """读取文件摘要；调用方必须保证目标是已结束的只读 artifact。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _integer_key(value: object, *, allow_empty: bool = False) -> tuple[int, ...] | None:
    if not isinstance(value, list) or (not value and not allow_empty):
        return None
    if any(type(item) is not int or item < 0 for item in value):
        return None
    return tuple(value)


def _unicode_values(value: object) -> tuple[int, ...] | None:
    key = _integer_key(value)
    if key is None or any(
        item > 0x10FFFF or 0xD800 <= item <= 0xDFFF for item in key
    ):
        return None
    return key


def _code_key(value: object) -> tuple[int, ...] | None:
    """把单整数关系族代码也规范成一个整数 key。"""
    if type(value) is int and value >= 0:
        return (value,)
    return _integer_key(value, allow_empty=True)


def _find_forbidden(value: object, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            forbidden_key = lowered in _FORBIDDEN_KEYS or any(token in lowered for token in (
                "answer", "knowledge", "raw_text", "whole_sentence", "replay",
                "opencc", "jsonl",
            ))
            # Receipt contracts may carry an explicit zero-valued prohibition
            # flag; that flag is evidence of the ban, not a forbidden payload.
            if forbidden_key and child not in (0, False, None, [], {}):
                found.append(f"{path}.{key}" if path else str(key))
            found.extend(_find_forbidden(child, f"{path}.{key}" if path else str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_find_forbidden(child, f"{path}[{index}]"))
    return found


def _non_integer_payload(value: object, path: str = "") -> list[str]:
    """只允许协议 format 字符串；课程语义载荷必须由整数记录组成。"""
    errors: list[str] = []
    if type(value) is bool:
        errors.append(path or "<root>")
        return errors
    if type(value) is int or value is None:
        return errors
    if isinstance(value, str):
        if path.rsplit(".", 1)[-1] not in _ALLOWED_NON_INTEGER_KEYS:
            errors.append(path or "<root>")
        return errors
    if isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(_non_integer_payload(child, f"{path}[{index}]"))
        return errors
    if isinstance(value, dict):
        for key, child in value.items():
            errors.extend(_non_integer_payload(child, f"{path}.{key}" if path else str(key)))
        return errors
    errors.append(path or "<root>")
    return errors


def _status(errors: Iterable[str], missing: Iterable[str]) -> str:
    if tuple(errors):
        return FAIL
    if tuple(missing):
        return UNKNOWN
    return PASS


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"artifact is not an object: {path}")
    return value


def _split_rows(root: Path, name: str) -> tuple[dict[str, Any], ...]:
    path = root / f"{name}.int.json"
    payload = _read_json(path)
    if payload.get("format") != _FORMAT or payload.get("split") != name:
        raise ValueError(f"invalid {name} split format")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{name} split has no row cases")
    if payload.get("case_count") != len(rows):
        raise ValueError(f"{name} split case count drift")
    if payload.get("free_dialogue_complete") != 0:
        raise ValueError(f"{name} split changes free_dialogue_complete")
    return tuple(row for row in rows if isinstance(row, dict))


def _behavior_evidence(row: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    evidence = row.get("semantic_behavior_evidence")
    if not isinstance(evidence, dict):
        return "missing_semantic_behavior_evidence", None
    required = (
        "receipt_key", "before_graph_sha256", "after_graph_sha256",
        "changed_fields", "baseline_trace_sha256", "counterfactual_trace_sha256",
        "selection_changed", "delivery_or_refusal_changed", "counterfactual_fail_closed",
    )
    missing = [field for field in required if field not in evidence]
    if missing:
        return "missing_behavior_fields:" + ",".join(missing), None
    for field in ("receipt_key", "changed_fields"):
        if _integer_key(evidence[field], allow_empty=field == "receipt_key") is None:
            return f"invalid_behavior_integer_key:{field}", None
    for field in ("before_graph_sha256", "after_graph_sha256",
                  "baseline_trace_sha256", "counterfactual_trace_sha256"):
        value = _integer_key(evidence[field])
        if value is None or len(value) != 32 or any(item > 255 for item in value):
            return f"invalid_behavior_digest:{field}", None
    for field in ("selection_changed", "delivery_or_refusal_changed", "counterfactual_fail_closed"):
        if type(evidence[field]) is not int or evidence[field] not in (0, 1):
            return f"invalid_behavior_flag:{field}", None
    if evidence["before_graph_sha256"] == evidence["after_graph_sha256"]:
        return "behavior_graph_identity_unchanged", None
    if evidence["baseline_trace_sha256"] == evidence["counterfactual_trace_sha256"]:
        return "behavior_trace_identity_unchanged", None
    if not evidence["selection_changed"] or not evidence["delivery_or_refusal_changed"]:
        return "behavior_delta_not_observed", None
    if not evidence["counterfactual_fail_closed"]:
        return "behavior_counterfactual_not_fail_closed", None
    return "", evidence


def _topology_record(value: object) -> tuple[tuple[tuple[int, ...], ...], ...] | None:
    """读取有序整数边记录；不接受单独的 fresh topology label。"""
    if not isinstance(value, list) or not value:
        return None
    edges: list[tuple[tuple[int, ...], ...]] = []
    for edge in value:
        if not isinstance(edge, list) or len(edge) != 3:
            return None
        parts = tuple(_integer_key(part) for part in edge)
        if any(part is None for part in parts):
            return None
        edges.append(parts)  # type: ignore[arg-type]
    result = tuple(edges)
    if result != tuple(sorted(set(result))) or len(result) != len(set(result)):
        return None
    return result


def audit_course(root: Path) -> dict[str, Any]:
    """审计四路整数课程，标签不参与语义行为 PASS。"""
    root = root.resolve(strict=True)
    errors: list[str] = []
    missing: list[str] = []
    manifest_path = root / "manifest.json"
    try:
        manifest = _read_json(manifest_path)
    except (OSError, ValueError, UnicodeError) as exc:
        return {
            "status": FAIL, "errors": [f"manifest:{type(exc).__name__}:{exc}"],
            "missing_evidence": [], "free_dialogue_complete": 0,
        }
    if manifest.get("free_dialogue_complete") != 0:
        errors.append("manifest_free_dialogue_complete_not_zero")
    if manifest.get("source_body_rows_added") != 0:
        errors.append("manifest_source_body_rows_added")
    if manifest.get("database_copy_count") != 0:
        errors.append("manifest_database_copy_count")
    forbidden = _find_forbidden(manifest)
    if forbidden:
        errors.extend(f"manifest_forbidden:{item}" for item in forbidden)
    for key in ("source_identity", "license_identity"):
        if _integer_key(manifest.get(key)) is None:
            missing.append(f"manifest.{key}")
    if not isinstance(manifest.get("license"), str) or not manifest["license"]:
        missing.append("manifest.license")
    if not isinstance(manifest.get("source_manifest_sha256"), str):
        missing.append("manifest.source_manifest_sha256")
    if not isinstance(manifest.get("license_manifest_sha256"), str):
        missing.append("manifest.license_manifest_sha256")

    split_rows: dict[str, tuple[dict[str, Any], ...]] = {}
    split_counts: dict[str, int] = {}
    split_hashes: dict[str, str] = {}
    for name in SPLITS:
        path = root / f"{name}.int.json"
        if not path.is_file():
            missing.append(f"missing:{name}.int.json")
            continue
        split_hashes[path.name] = sha256(path)
        try:
            rows = _split_rows(root, name)
        except (OSError, ValueError, UnicodeError) as exc:
            errors.append(f"{name}:{type(exc).__name__}:{exc}")
            continue
        split_rows[name] = rows
        split_counts[name] = len(rows)
        if len(rows) < MIN_CASES_PER_SPLIT:
            errors.append(f"{name}_case_count_below_{MIN_CASES_PER_SPLIT}")

    all_case_ids: set[tuple[int, ...]] = set()
    field_values: dict[str, dict[str, set[tuple[int, ...]]]] = {
        name: {field: set() for field in _PROVENANCE_FIELDS} for name in SPLITS
    }
    train_families: set[tuple[int, ...]] = set()
    train_operations: set[tuple[int, ...]] = set()
    held_families: set[tuple[int, ...]] = set()
    held_operations: set[tuple[int, ...]] = set()
    train_combos: set[tuple[int, ...]] = set()
    train_topologies: set[tuple[tuple[tuple[int, ...], ...], ...]] = set()
    held_topologies: set[tuple[tuple[tuple[int, ...], ...], ...]] = set()
    held_novel_combos: set[tuple[int, ...]] = set()
    behavior_signatures: set[tuple[Any, ...]] = set()
    behavior_missing: list[str] = []
    derived_cases: list[str] = []
    topology_missing: list[str] = []

    for split, rows in split_rows.items():
        expected_code = SPLITS.index(split) + 1
        for ordinal, row in enumerate(rows, 1):
            label = f"{split}:{ordinal}"
            if row.get("format") != _ROW_FORMAT:
                errors.append(f"{label}:invalid_row_format")
            if row.get("free_dialogue_complete") != 0:
                errors.append(f"{label}:free_dialogue_complete_not_zero")
            if row.get("split_code") != expected_code:
                errors.append(f"{label}:split_code_drift")
            forbidden = _find_forbidden(row)
            errors.extend(f"{label}:forbidden:{item}" for item in forbidden)
            errors.extend(f"{label}:non_integer:{item}" for item in _non_integer_payload(row))
            missing_keys = sorted(_REQUIRED_ROW_KEYS - set(row))
            missing.extend(f"{label}:{key}" for key in missing_keys)
            case_id = _integer_key(row.get("case_id"))
            if case_id is None:
                errors.append(f"{label}:invalid_case_id")
            elif case_id in all_case_ids:
                errors.append(f"{label}:duplicate_case_id")
            else:
                all_case_ids.add(case_id)
            values = _unicode_values(row.get("input_values"))
            if values is None:
                errors.append(f"{label}:invalid_unicode_input")
            for field in _PROVENANCE_FIELDS:
                key = _integer_key(row.get(field))
                if key is None:
                    missing.append(f"{label}:{field}")
                else:
                    field_values[split][field].add(key)
            family = _code_key(row.get("relation_family_code"))
            operation = _integer_key(row.get("operation_key"), allow_empty=True)
            combo = _integer_key(row.get("relation_combo_key"), allow_empty=True)
            topology = _topology_record(row.get("topology_record"))
            if family is None or not family:
                errors.append(f"{label}:invalid_relation_family")
            if operation is None or not operation:
                errors.append(f"{label}:invalid_operation_key")
            if combo is None or not combo:
                errors.append(f"{label}:invalid_relation_combo")
            if topology is None:
                topology_missing.append(label)
            if split == "train":
                if topology:
                    train_topologies.add(topology)
                    train_families.update(edge[0] for edge in topology)
                if operation:
                    train_operations.add(operation)
                if combo:
                    train_combos.add(combo)
            elif split == "held_out":
                if topology:
                    held_topologies.add(topology)
                    held_families.update(edge[0] for edge in topology)
                if operation:
                    held_operations.add(operation)
                if (topology and topology not in train_topologies
                        and combo and combo not in train_combos):
                    held_novel_combos.add(combo)
            if int(row.get("source_pack_index", 1) or 0) == 0:
                derived_cases.append(label)
            reason, evidence = _behavior_evidence(row)
            if reason:
                behavior_missing.append(f"{label}:{reason}")
            elif evidence is not None:
                behavior_signatures.add((
                    tuple(evidence["changed_fields"]),
                    int(evidence["selection_changed"]),
                    int(evidence["delivery_or_refusal_changed"]),
                    int(evidence["counterfactual_fail_closed"]),
                ))

    pair_overlaps: list[dict[str, Any]] = []
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1:]:
            for field in _PROVENANCE_FIELDS:
                count = len(field_values[left][field] & field_values[right][field])
                if count:
                    pair_overlaps.append({"left": left, "right": right,
                                          "field": field, "overlap_count": count})
                    errors.append(f"provenance_overlap:{left}:{right}:{field}")

    if not topology_missing:
        if len(train_families) < MIN_RELATION_FAMILIES:
            errors.append(f"train_relation_family_below_{MIN_RELATION_FAMILIES}")
        if len(held_families) < MIN_RELATION_FAMILIES:
            errors.append(f"held_out_relation_family_below_{MIN_RELATION_FAMILIES}")
    if len(train_operations) < MIN_OPERATION_FAMILIES:
        errors.append(f"declared_train_operation_below_{MIN_OPERATION_FAMILIES}")

    structural_status = _status(errors, missing)
    source_status = _status(
        (), [item for item in missing if item.startswith("manifest.")])
    split_status = _status(
        [item for item in errors if "split" in item or "provenance" in item
         or "case_count" in item or "duplicate_case" in item],
        [item for item in missing if not item.startswith("manifest.")],
    )
    novelty_missing = topology_missing
    novelty_errors = []
    if not topology_missing and len(held_novel_combos) < MIN_NOVEL_HELD_OUT:
        novelty_errors.append(f"held_out_novel_relation_combo_below_{MIN_NOVEL_HELD_OUT}")
    distinct_novel_topologies = len(held_topologies - train_topologies)
    if not topology_missing and distinct_novel_topologies < MIN_NOVEL_HELD_OUT:
        novelty_errors.append(f"held_out_novel_topology_below_{MIN_NOVEL_HELD_OUT}")
    novelty_status = _status(novelty_errors, novelty_missing)
    behavior_errors = [item for item in behavior_missing if not any(
        marker in item for marker in ("missing_semantic_behavior_evidence", "missing_behavior_fields"))]
    if not behavior_missing and len(behavior_signatures) < MIN_SEMANTIC_BEHAVIORS:
        behavior_errors.append(
            f"claimed_semantic_behavior_signature_below_{MIN_SEMANTIC_BEHAVIORS}")
    # These fields are row-authored claims. Distinct hashes and flags do not
    # independently prove that a graph was edited or a live QueryState used it.
    # A separate physical graph/trace verifier is required before any family
    # can become PASS; this course-only audit is intentionally fail-closed.
    behavior_status = FAIL if behavior_errors else UNKNOWN
    if not behavior_errors:
        behavior_missing.append("independent_physical_graph_and_trace_verifier")
    if derived_cases:
        structural_status = FAIL
        errors.append(f"derived_cases_in_required_splits:{len(derived_cases)}")
    operation_status = FAIL if len(train_operations) < MIN_OPERATION_FAMILIES else UNKNOWN

    overall_missing = [*missing,
                       *(f"semantic_behavior:{item}" for item in behavior_missing),
                       *(f"novelty:{item}" for item in topology_missing)]
    if operation_status == UNKNOWN:
        overall_missing.append("operation_coverage:independent_behavior_proof")
    return {
        "status": _status(errors, overall_missing),
        "structural_status": structural_status,
        "source_license_status": source_status,
        "split_status": split_status,
        "novelty_status": novelty_status,
        "semantic_behavior_status": behavior_status,
        "operation_coverage_status": operation_status,
        "errors": sorted(set(errors)),
        "missing_evidence": sorted(set(overall_missing)),
        "behavior_missing": sorted(set(behavior_missing)),
        "topology_missing": sorted(set(topology_missing)),
        "derived_cases": derived_cases,
        "pair_provenance_overlaps": pair_overlaps,
        "split_case_counts": split_counts,
        "split_sha256": split_hashes,
        "train_relation_family_count": len(train_families),
        "declared_train_operation_count": len(train_operations),
        "train_operation_count": 0,
        "held_out_relation_family_count": len(held_families),
        "held_out_operation_count": len(held_operations),
        "held_out_novel_relation_combo_count": len(held_novel_combos),
        "held_out_novel_topology_count": distinct_novel_topologies,
        "train_topology_count": len(train_topologies),
        "held_out_topology_count": len(held_topologies),
        "behavior_signature_count": 0,
        "claimed_behavior_signature_count": len(behavior_signatures),
        "required": {
            "min_cases_per_split": MIN_CASES_PER_SPLIT,
            "min_relation_families": MIN_RELATION_FAMILIES,
            "min_operation_families": MIN_OPERATION_FAMILIES,
            "min_novel_held_out": MIN_NOVEL_HELD_OUT,
            "min_semantic_behaviors": MIN_SEMANTIC_BEHAVIORS,
        },
        "free_dialogue_complete": 0,
    }


def audit_capability_matrix(path: Path | None) -> dict[str, Any]:
    """检查八个端到端能力门；没有 matrix 只能 UNKNOWN。"""
    if path is None or not path.is_file():
        return {"status": UNKNOWN, "missing_evidence": ["capability_matrix"],
                "errors": [], "free_dialogue_complete": 0}
    try:
        value = _read_json(path)
    except (OSError, ValueError, UnicodeError) as exc:
        return {"status": FAIL, "errors": [f"matrix:{type(exc).__name__}:{exc}"],
                "missing_evidence": [], "free_dialogue_complete": 0}
    rows = value.get("rows")
    if not isinstance(rows, list):
        return {"status": FAIL, "errors": ["matrix.rows_missing"],
                "missing_evidence": [], "free_dialogue_complete": 0}
    expected = {"surface", "context", "core", "multi_relation", "memory_ohe",
                "companion", "response_plan", "generation_delivery"}
    seen = {row.get("capability") for row in rows if isinstance(row, dict)}
    missing = sorted(expected - seen)
    errors = ["matrix.duplicate_capability"] if len(seen) != len(rows) else []
    if value.get("free_dialogue_complete") != 0:
        errors.append("matrix.free_dialogue_complete_not_zero")
    bad = [str(row.get("capability")) for row in rows
           if isinstance(row, dict) and row.get("status") != PASS]
    errors.extend(f"matrix.not_pass:{name}" for name in bad)
    return {"status": _status(errors, missing), "errors": errors,
            "missing_evidence": missing, "row_count": len(rows),
            "non_pass_capabilities": bad, "free_dialogue_complete": 0}


def audit_route_files(paths: Iterable[Path]) -> dict[str, Any]:
    """静态审计生产入口禁区；不把静态通过当成真实能力。"""
    files = tuple(Path(path).resolve(strict=True) for path in paths)
    if not files:
        return {"status": UNKNOWN, "errors": [], "missing_evidence": ["route_manifest"],
                "files_checked": [], "free_dialogue_complete": 0}
    errors: list[str] = []
    checked: list[str] = []
    for path in files:
        checked.append(str(path))
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeError) as exc:
            errors.append(f"{path}:{type(exc).__name__}:{exc}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = (node.names if isinstance(node, ast.Import) else
                         (ast.alias(name=node.module or "", asname=None),))
                for alias in names:
                    if alias.name.split(".", 1)[0].lower() == "opencc":
                        errors.append(f"{path}:{node.lineno}:opencc_import")
            if isinstance(node, ast.Name) and node.id.lower() in {
                "answer_table", "knowledge_text", "old_answer", "replay",
            }:
                errors.append(f"{path}:{node.lineno}:{node.id}")
            if isinstance(node, ast.Attribute) and node.attr.lower() in {
                "raw_text", "whole_sentence", "answer_table", "opencc",
            }:
                errors.append(f"{path}:{node.lineno}:{node.attr}")
    return {"status": FAIL if errors else PASS, "errors": sorted(set(errors)),
            "missing_evidence": [], "files_checked": checked,
            "free_dialogue_complete": 0}


def build_production_start_readiness(
        course: dict[str, Any], routes: dict[str, Any], *,
        execution_contract: dict[str, Any], bounded_preflight: dict[str, Any],
        resource_budget: dict[str, Any]) -> dict[str, Any]:
    """只判断能否安全启动；不把运行后的能力证据倒灌为准入条件。

    开发预检必须明确证明没有读取正式 held-out 行或标签。能力资格由
    build_post_run_capability_qualification 在真实批次和独立冷审之后判定。
    """
    held_out_reads = bounded_preflight.get("formal_held_out_rows_read")
    held_out_label_reads = bounded_preflight.get("formal_held_out_labels_read")
    isolation_status = bounded_preflight.get("status", UNKNOWN)
    if (type(held_out_reads) is int and type(held_out_label_reads) is int
            and (held_out_reads != 0 or held_out_label_reads != 0)):
        isolation_status = FAIL
    elif type(held_out_reads) is not int or type(held_out_label_reads) is not int:
        isolation_status = UNKNOWN
    elif isolation_status == PASS:
        isolation_status = PASS

    components = {
        "course_structure": course.get("structural_status", UNKNOWN),
        "source_license": course.get("source_license_status", UNKNOWN),
        "split_isolation": course.get("split_status", UNKNOWN),
        "novelty_design": course.get("novelty_status", UNKNOWN),
        "execution_contract": execution_contract.get("status", UNKNOWN),
        "bounded_preflight": isolation_status,
        "resource_budget": resource_budget.get("status", UNKNOWN),
        "route_prohibition": routes.get("status", UNKNOWN),
    }
    status = PASS if all(value == PASS for value in components.values()) else "BLOCKED"
    return {
        "status": status, "components": components,
        "required_all_pass": 1, "production_start_authorized": int(status == PASS),
        "production_not_started": 1, "qualified_semantic_operation_families": 0,
        "free_dialogue_complete": 0,
    }


def build_post_run_capability_qualification(
        *, behavior_verifier: dict[str, Any], capability: dict[str, Any],
        production_receipt: dict[str, Any], independent_cold_audit: dict[str, Any]) -> dict[str, Any]:
    """只合并真实消费者与独立冷审结果；缺少任一项就不授予资格。"""
    families = behavior_verifier.get("qualified_semantic_operation_families")
    behavior_status = behavior_verifier.get("status", UNKNOWN)
    if type(families) is not int or families < MIN_OPERATION_FAMILIES:
        behavior_status = FAIL if type(families) is int else UNKNOWN
    components = {
        "semantic_behavior": behavior_status,
        "operation_coverage": (PASS if type(families) is int
                                and families >= MIN_OPERATION_FAMILIES else behavior_status),
        "capability_matrix": capability.get("status", UNKNOWN),
        "production_receipt": production_receipt.get("status", UNKNOWN),
        "independent_cold_audit": independent_cold_audit.get("status", UNKNOWN),
    }
    status = PASS if all(value == PASS for value in components.values()) else "NOT_QUALIFIED"
    return {
        "status": status, "components": components,
        "qualified_semantic_operation_families": families if status == PASS else 0,
        "production_not_started": 0, "free_dialogue_complete": 0,
    }


__all__ = [
    "FAIL", "MIN_CASES_PER_SPLIT", "MIN_NOVEL_HELD_OUT", "MIN_OPERATION_FAMILIES",
    "MIN_RELATION_FAMILIES", "MIN_SEMANTIC_BEHAVIORS", "PASS", "SPLITS", "UNKNOWN",
    "audit_capability_matrix", "audit_course", "audit_route_files",
    "build_post_run_capability_qualification", "build_production_start_readiness", "sha256",
]
