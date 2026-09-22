"""Audit historical facilities against the current pure-integer mainline.

This is a read-only source audit.  It follows Python imports from the formal
runtime/training roots, then reports whether each feature has an observable
path to the shared QueryState and generation/training contracts.  It never
imports the target package and never writes inside the package.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
import tokenize
from io import StringIO


MAINLINE_ROOTS = (
    "pure_integer_ai.experiments.trained_graph_query_bridge",
    "pure_integer_ai.experiments.trained_generation_connector_runtime",
    "pure_integer_ai.experiments.trained_dialogue_session_projection",
    "pure_integer_ai.experiments.run_unknown_input_structure_slice",
    "pure_integer_ai.experiments.run_open_role_memory_slice",
    "pure_integer_ai.experiments.formal_train",
    "pure_integer_ai.experiments.materialize_generic_response_generation",
    "pure_integer_ai.experiments.trained_graph_release",
)

ENTRY_SCOPES = {
    "query": ("pure_integer_ai.experiments.trained_graph_query_bridge",),
    "generation": ("pure_integer_ai.experiments.trained_generation_connector_runtime",),
    "terminal": (
        "pure_integer_ai.experiments.trained_dialogue_session_projection",
        "pure_integer_ai.experiments.run_unknown_input_structure_slice",
        "pure_integer_ai.experiments.run_open_role_memory_slice",
    ),
    "training": (
        "pure_integer_ai.experiments.formal_train",
        "pure_integer_ai.experiments.materialize_generic_response_generation",
    ),
    "release": ("pure_integer_ai.experiments.trained_graph_release",),
}

FEATURES = {
    "entity": ("entity", "is_a", "semantic_object", "concept_index"),
    "event": ("event", "causes", "event_time", "episode"),
    "property": ("property", "degree_intensity", "mereology"),
    "time": ("time", "preced", "occurrence_order", "event_time"),
    "reference": ("reference", "refers", "pronoun", "occurrence"),
    "discourse_topic": ("discourse", "topic", "focus", "work_memory"),
    "memory": ("memory", "hypothesis", "evidence", "hot_set", "query_memory"),
    "generation": ("generation", "response", "connector", "surface"),
    "reasoning": ("reason", "causal", "closure", "relation"),
    "legacy_result": ("cognition.result",),
}

FORBIDDEN = {
    "successor_surface_replay": (
        # ``dialogue_successor`` and ``successor_graph`` are legitimate
        # integer evidence/topology modules.  Only explicit surface replay
        # routes are forbidden here; broad module-name matching created false
        # positives for the production QueryState bridge.
        "successor_surface", "replay_successor", "successor_occurrence",
        "successor_sentence", "surface_replay",
    ),
    "character_nearest": ("char_nearest", "character_nearest", "nearest_char", "levenshtein"),
    "opencc": ("opencc",),
    # A compact course sidecar may call its integer atom dictionary a
    # ``vocabulary``.  That is a storage deduplication codec, not a language
    # word-list route.  Only explicit language-vocabulary mechanisms are
    # forbidden here; otherwise the audit pressures training back toward
    # repeated full-surface storage.
    "language_vocabulary": (
        "language_vocab", "language_vocabulary", "word_list",
    ),
    "source_text_answer": ("source_text_answer", "answer_from_source_text", "raw_source_answer"),
}

CONTRACTS = {
    "query_state": ("QueryState", "QueryRoot", "FrontierEntry"),
    "binding": ("BindingEntry", "bindings", "query_bindings"),
    "evidence": ("EvidenceEntry", "evidence", "query_evidence"),
    "visited": ("VisitedKey", "visited", "unvisited_frontier"),
    "termination": ("termination", "TERMINATION_", "evidence_closed"),
    "response_plan": ("ResponsePlan", "ResponseActGeneration", "response_plan"),
    "training_publish": ("formal_train", "build_trained_graph_release", "publish", "training"),
}


def _module_name(path: Path, source_root: Path) -> str:
    rel = path.relative_to(source_root).with_suffix("")
    return ".".join(rel.parts)


def _imports(tree: ast.AST, module: str) -> set[str]:
    result: set[str] = set()
    package = module.split(".")[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - node.level + 1]
                prefix = ".".join(base + ([node.module] if node.module else []))
            else:
                prefix = node.module or ""
            if prefix:
                result.add(prefix)
    return result


class _EagerImportVisitor(ast.NodeVisitor):
    """Collect imports executed while a module is imported.

    Imports nested below functions/classes are callable edges, not import-time
    dependencies.  This distinction is what exposes training-layer leakage
    into a read-only production runtime without discarding real lazy hooks.
    """

    def __init__(self, module: str) -> None:
        self.module = module
        self.package = module.split(".")[:-1]
        self.result: set[str] = set()
        self._scope_depth = 0

    def _add(self, node: ast.Import | ast.ImportFrom) -> None:
        if isinstance(node, ast.Import):
            self.result.update(alias.name for alias in node.names)
            return
        if node.level:
            base = self.package[:len(self.package) - node.level + 1]
            prefix = ".".join(base + ([node.module] if node.module else []))
        else:
            prefix = node.module or ""
        if prefix:
            self.result.add(prefix)

    def visit_Import(self, node: ast.Import) -> None:
        if self._scope_depth == 0:
            self._add(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if self._scope_depth == 0:
            self._add(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._scope_depth += 1
        self.generic_visit(node)
        self._scope_depth -= 1

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scope_depth += 1
        self.generic_visit(node)
        self._scope_depth -= 1


def _eager_imports(tree: ast.AST, module: str) -> set[str]:
    visitor = _EagerImportVisitor(module)
    visitor.visit(tree)
    return visitor.result


def _code_lines(source: str) -> tuple[tuple[int, str], ...]:
    """Return executable token lines, excluding comments/docstrings.

    Historical modules contain extensive design notes mentioning forbidden
    paths.  Those notes must remain auditable, but they are not an executable
    import/call edge.  Tokenizing keeps this read-only audit conservative and
    avoids treating a docstring as a production dependency.
    """
    lines: dict[int, list[str]] = {}
    try:
        tokens = tokenize.generate_tokens(StringIO(source).readline)
        for token in tokens:
            if token.type in {tokenize.COMMENT, tokenize.NL,
                              tokenize.ENCODING, tokenize.ENDMARKER}:
                continue
            if token.type == tokenize.STRING:
                continue
            lines.setdefault(token.start[0], []).append(token.string)
    except (IndentationError, tokenize.TokenError):
        return tuple((index, line) for index, line in enumerate(
            source.splitlines(), 1))
    return tuple((line, " ".join(values)) for line, values in sorted(lines.items()))


def _line_hits(source: str, markers: tuple[str, ...]) -> tuple[int, ...]:
    hits = []
    for line_number, line in _code_lines(source):
        folded = line.casefold()
        if any(marker.casefold() in folded for marker in markers):
            hits.append(line_number)
    return tuple(hits)


def _ast_marker_hits(source: str, markers: tuple[str, ...]) -> tuple[int, ...]:
    """Find executable imports/calls/attributes, excluding prose and locals."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _line_hits(source, markers)
    hits: set[int] = set()
    for node in ast.walk(tree):
        values: list[str] = []
        if isinstance(node, ast.Import):
            values.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            values.append(node.module or "")
            values.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                values.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                values.append(node.func.attr)
        elif isinstance(node, ast.Attribute):
            values.append(node.attr)
        if any(any(marker.casefold() in value.casefold() for marker in markers)
               for value in values):
            hits.add(node.lineno)
    return tuple(sorted(hits))


def _contract_hits(source: str) -> dict[str, tuple[int, ...]]:
    return {name: _line_hits(source, markers) for name, markers in CONTRACTS.items()}


def _feature_marker_matches_module(module: str, marker: str) -> bool:
    """Match complete dotted/underscore module-name components.

    Plain substring matching classified ``runtime`` as ``time`` and
    ``identity`` as ``entity``.  Those false positives made the historical
    facility inventory unusable for deciding what should enter the dialogue
    mainline.  Multi-component markers such as ``event_time`` remain valid.
    """
    value = module.casefold().replace("-", "_")
    token = marker.casefold().replace("-", "_")
    return re.search(
        rf"(?:^|[._]){re.escape(token)}(?:$|[._])",
        value,
    ) is not None


def _root_modules(source_root: Path) -> tuple[Path, str]:
    """Resolve either ``src`` or a package directory without prefix drift."""
    if (source_root / "__init__.py").is_file():
        return source_root, source_root.name
    packages = tuple(sorted(
        item for item in source_root.iterdir()
        if item.is_dir() and (item / "__init__.py").is_file()
    ))
    if len(packages) == 1:
        return source_root, packages[0].name
    return source_root, ""


def _qualified_module(path: Path, root: Path, package_prefix: str) -> str:
    raw = _module_name(path, root)
    if raw == "__init__":
        return package_prefix or raw
    if raw.endswith(".__init__"):
        raw = raw[:-len(".__init__")]
    if package_prefix and (raw == package_prefix
                           or raw.startswith(package_prefix + ".")):
        return raw
    return package_prefix + "." + raw if package_prefix else raw


def audit(source_root: str | Path) -> dict[str, object]:
    root = Path(source_root).resolve()
    if not root.is_dir():
        raise ValueError("source root does not exist")
    root, package_prefix = _root_modules(root)
    files = {
        _qualified_module(path, root, package_prefix): path
        for path in sorted(root.rglob("*.py"))
    }
    sources: dict[str, str] = {}
    graph: dict[str, set[str]] = {}
    eager_graph: dict[str, set[str]] = {}
    parse_errors = {}
    for module, path in files.items():
        source = path.read_text(encoding="utf-8")
        sources[module] = source
        try:
            tree = ast.parse(source, filename=str(path))
            imports = _imports(tree, module)
            eager_imports = _eager_imports(tree, module)
            graph[module] = {
                item if item in files else item.rsplit(".", 1)[0]
                for item in imports
                if item in files or item.rsplit(".", 1)[0] in files
            }
            eager_graph[module] = {
                item if item in files else item.rsplit(".", 1)[0]
                for item in eager_imports
                if item in files or item.rsplit(".", 1)[0] in files
            }
        except SyntaxError as error:
            graph[module] = set()
            eager_graph[module] = set()
            parse_errors[module] = f"{error.msg}:{error.lineno}"

    def reachable_from(roots: tuple[str, ...]) -> set[str]:
        reachable: set[str] = set()
        frontier = [item for item in roots if item in files]
        while frontier:
            current = frontier.pop()
            if current in reachable:
                continue
            reachable.add(current)
            frontier.extend(sorted(graph.get(current, ()), reverse=True))
        return reachable

    scope_reachable = {
        scope: reachable_from(roots)
        for scope, roots in ENTRY_SCOPES.items()
    }
    def eager_reachable_from(roots: tuple[str, ...]) -> set[str]:
        reachable: set[str] = set()
        frontier = [item for item in roots if item in files]
        while frontier:
            current = frontier.pop()
            if current in reachable:
                continue
            reachable.add(current)
            frontier.extend(sorted(eager_graph.get(current, ()), reverse=True))
        return reachable

    eager_scope_reachable = {
        scope: eager_reachable_from(roots)
        for scope, roots in ENTRY_SCOPES.items()
    }
    reachable = set().union(*scope_reachable.values()) if scope_reachable else set()

    modules = []
    production_scopes = ("query", "generation", "terminal")
    for module in sorted(files):
        source = sources[module]
        contracts = _contract_hits(source)
        forbidden = {
            name: _ast_marker_hits(source, markers)
            for name, markers in FORBIDDEN.items()
            if _ast_marker_hits(source, markers)
        }
        feature_names = tuple(sorted(
            name for name, markers in FEATURES.items()
            if any(_feature_marker_matches_module(module, marker)
                   for marker in markers)
        ))
        consumed = tuple(sorted(name for name, hits in contracts.items() if hits))
        production_reachable = any(
            module in scope_reachable[scope] for scope in production_scopes)
        production_eager = any(
            module in eager_scope_reachable[scope] for scope in production_scopes)
        training_only = module in scope_reachable["training"] and not production_reachable
        modules.append({
            "module": module,
            "path": str(files[module]),
            "reachable_from_mainline": int(module in reachable),
            "reachable_scopes": [scope for scope, items in scope_reachable.items()
                                 if module in items],
            "eager_reachable_scopes": [
                scope for scope, items in eager_scope_reachable.items()
                if module in items
            ],
            "production_reachable": int(production_reachable),
            "production_eager_reachable": int(production_eager),
            "training_only_reachable": int(training_only),
            "features": list(feature_names),
            "contracts_observed": list(consumed),
            "contract_lines": {name: list(hits) for name, hits in contracts.items() if hits},
            "forbidden_markers": {name: list(hits) for name, hits in forbidden.items()},
            "imports": sorted(graph.get(module, ())),
            "eager_imports": sorted(eager_graph.get(module, ())),
            "status": (
                "forbidden_isolated" if forbidden and not reachable else
                "forbidden_eager_reachable" if forbidden and production_eager else
                "forbidden_lazy_reachable" if forbidden and production_reachable else
                "forbidden_training_only" if forbidden and training_only else
                "integrated_candidate" if production_reachable and consumed else
                "training_candidate" if training_only and consumed else
                "reachable_without_contract" if module in reachable else
                "isolated"
            ),
        })

    feature_summary = {}
    feature_routes = {}
    for feature in FEATURES:
        rows = [item for item in modules if feature in item["features"]]
        feature_summary[feature] = {
            "module_count": len(rows),
            "reachable_count": sum(item["reachable_from_mainline"] for item in rows),
            "integrated_candidate_count": sum(item["status"] == "integrated_candidate" for item in rows),
            "isolated_count": sum(item["status"] == "isolated" for item in rows),
            "forbidden_count": sum(bool(item["forbidden_markers"]) for item in rows),
        }
        feature_routes[feature] = {
            "production_consumers": [
                item["module"] for item in rows
                if item["production_reachable"] and item["contracts_observed"]
            ],
            "training_producers": [
                item["module"] for item in rows
                if item["training_only_reachable"]
            ],
            "isolated_facilities": [
                item["module"] for item in rows
                if not item["reachable_from_mainline"]
            ],
        }
    return {
        "schema_version": 2,
        "source_root": str(root),
        "package_prefix": package_prefix,
        "mainline_roots": list(MAINLINE_ROOTS),
        "entry_scopes": {name: list(roots) for name, roots in ENTRY_SCOPES.items()},
        "module_count": len(files),
        "reachable_module_count": len(reachable),
        "reachable_scope_counts": {
            scope: len(items) for scope, items in scope_reachable.items()
        },
        "eager_reachable_scope_counts": {
            scope: len(items) for scope, items in eager_scope_reachable.items()
        },
        "parse_error_count": len(parse_errors),
        "parse_errors": parse_errors,
        "feature_summary": feature_summary,
        "feature_routes": feature_routes,
        "modules": modules,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.source_root)
    target = args.output.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=True, sort_keys=True, separators=(",", ":")), encoding="ascii")
    print(json.dumps({
        "module_count": report["module_count"],
        "reachable_module_count": report["reachable_module_count"],
        "parse_error_count": report["parse_error_count"],
        "feature_summary": report["feature_summary"],
    }, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
