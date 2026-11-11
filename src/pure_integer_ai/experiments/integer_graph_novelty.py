"""Bounded, read-only novelty admission for registered integer graph courses.

Identity equality establishes links before IDs are removed. Exact canonical
incidence trees compare topology independently of provenance, surface text,
operation labels and caller-supplied shape/combo group keys. This is admission
evidence only: it does not prove parsing, inference or operation behavior.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from itertools import permutations, product
import json
from math import factorial, prod
from typing import Mapping, Protocol, Sequence, runtime_checkable

from pure_integer_ai.cognition.shared.identity import ObjectIdentity
from pure_integer_ai.experiments.ph2_authored_relation_compile import (
    authored_relation_role_identity,
)
from pure_integer_ai.experiments.ph2_w06_source_semantic import W06_RELATION_PROFILES


class IntegerGraphNoveltyError(ValueError):
    """Evidence is absent, unsupported or outside the exact comparison budget."""


_PROFILES = {profile.relation_kind: profile for profile in W06_RELATION_PROFILES.values()}
_SPLITS = ("train", "held_out", "negative", "conflict")


@runtime_checkable
class IntegerGraphTopologyRecord(Protocol):
    """Label-free structural interface shared by train and evaluation records."""

    source_ref_key: tuple[int, ...]
    observation_key: tuple[int, ...]
    source_cluster_key: tuple[int, ...]
    content_group_key: tuple[int, ...]
    dedup_cluster_key: tuple[int, ...]
    proposition_key: tuple[int, ...]
    relation_kind: int
    directionality: int
    role_bindings: tuple
    slots: tuple
    relation_constraints: tuple


def _key(value: object, field: str) -> tuple[int, ...]:
    if (not isinstance(value, (tuple, list)) or not value
            or any(type(item) is not int or item < 0 for item in value)):
        raise IntegerGraphNoveltyError(f"{field} requires a full nonempty integer key")
    return tuple(value)


@dataclass(frozen=True, slots=True)
class _Edge:
    proposition: tuple[int, ...]
    fillers: tuple[tuple[int, ...], ...]
    shape: tuple
    relation: tuple


def _edge(record: IntegerGraphTopologyRecord) -> _Edge:
    profile = _PROFILES[record.relation_kind]
    roles = tuple(authored_relation_role_identity(kind).stable_key()
                  for kind, _ in profile.role_object_kinds)
    bindings = {tuple(row["role_key"]): row for item in record.role_bindings
                for row in (item.to_value(),)}
    slots = {tuple(row["role_key"]): row for item in record.slots
             for row in (item.to_value(),)}
    if set(bindings) != set(roles) or set(slots) != set(roles):
        raise IntegerGraphNoveltyError("unregistered schema/role identities; novelty is unknown")
    role_indices = {key: index + 1 for index, key in enumerate(roles)}
    constraints = []
    for item in record.relation_constraints:
        keys = tuple(tuple(key) for key in item.to_value()["role_keys"])
        if not set(keys).issubset(role_indices):
            raise IntegerGraphNoveltyError("constraint refers to an absent schema role")
        # This course's constraint adapter supports SameKind only. Its opaque
        # declaration ID is not a new structure; the participating roles are.
        constraints.append(tuple(sorted(role_indices[key] for key in keys)))
    layout = []
    fillers = []
    for index, key in enumerate(roles, 1):
        binding, slot = bindings[key], slots[key]
        filler = tuple(binding["filler_key"])
        kind = ObjectIdentity.from_stable_key(filler).object_kind
        maximum = slot["max_count"]
        layout.append((index, binding["ordinal"], kind, slot["min_count"],
                       0 if maximum is None else maximum + 1))
        fillers.append(filler)
    shape = (record.directionality, tuple(layout), tuple(sorted(constraints)))
    relation = (record.relation_kind,
                tuple(kind for kind, _ in profile.role_object_kinds), shape)
    return _Edge(record.proposition_key, tuple(fillers), shape, relation)


def _canonical(edges: tuple[_Edge, ...], *, include_relations: bool,
               max_labelings: int) -> str:
    groups: dict[tuple, list[_Edge]] = {}
    for edge in edges:
        label = edge.relation if include_relations else edge.shape
        groups.setdefault(label, []).append(edge)
    group_rows = tuple(tuple(groups[label]) for label in sorted(groups))
    labelings = prod(factorial(len(group)) for group in group_rows)
    if labelings > max_labelings:
        raise IntegerGraphNoveltyError("exact topology labeling budget exceeded; novelty is unknown")
    best = None
    for choice in product(*(permutations(group) for group in group_rows)):
        ids: dict[tuple[int, ...], int] = {}
        kinds = []

        def node(key: tuple[int, ...]) -> int:
            if key not in ids:
                ids[key] = len(ids) + 1
                kinds.append(key[0])
            return ids[key]

        rows = tuple(
            (edge.relation if include_relations else edge.shape,
             node(edge.proposition), tuple(node(filler) for filler in edge.fillers))
            for group in choice for edge in group)
        tree = (1, tuple(kinds), rows)
        if best is None or tree < best:
            best = tree
    payload = json.dumps(best, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class IntegerGraphTopology:
    structure_sha256: str
    relation_combo_sha256: str
    relation_count: int
    endpoint_count: int
    component_count: int
    relation_kinds: tuple[int, ...]


def consumed_integer_graph_topology(records: Sequence[IntegerGraphTopologyRecord],
                                    proposition_keys: Sequence[tuple[int, ...]]) -> IntegerGraphTopology:
    """只对已验证 execution 实际匹配的前提规范化，不用输出身份冒充新颖性。"""
    by_key = {record.proposition_key: record for record in records}
    keys = tuple(sorted(set(tuple(key) for key in proposition_keys)))
    if len(by_key) != len(records) or not keys or any(key not in by_key for key in keys):
        raise IntegerGraphNoveltyError("consumed premises are missing from the physical course")
    return integer_graph_topology(tuple(by_key[key] for key in keys))


def integer_graph_topology(records: Sequence[IntegerGraphTopologyRecord], *,
                           max_labelings: int = 100_000) -> IntegerGraphTopology:
    """Canonicalize at most eight semantic edges; do not count evidence copies."""
    if (not isinstance(records, (tuple, list)) or not 1 <= len(records) <= 8
            or any(not isinstance(row, IntegerGraphTopologyRecord) for row in records)):
        raise IntegerGraphNoveltyError("topology requires 1..8 actual integer graph records")
    if type(max_labelings) is not int or max_labelings <= 0:
        raise IntegerGraphNoveltyError("exact labeling budget must be a positive integer")
    edges = tuple(_edge(row) for row in records)
    if len({edge.proposition for edge in edges}) != len(edges):
        raise IntegerGraphNoveltyError("repeated proposition definitions are not novel topology")
    if len({(edge.relation, edge.fillers) for edge in edges}) != len(edges):
        raise IntegerGraphNoveltyError("duplicated semantic edges are evidence copies, not topology")
    adjacency: dict[tuple[int, ...], set[tuple[int, ...]]] = {}
    for edge in edges:
        adjacency.setdefault(edge.proposition, set()).update(edge.fillers)
        for filler in edge.fillers:
            adjacency.setdefault(filler, set()).add(edge.proposition)
    remaining = set(adjacency)
    components = 0
    while remaining:
        components += 1
        pending = [remaining.pop()]
        while pending:
            for neighbor in adjacency[pending.pop()] & remaining:
                remaining.remove(neighbor)
                pending.append(neighbor)
    return IntegerGraphTopology(
        _canonical(edges, include_relations=False, max_labelings=max_labelings),
        _canonical(edges, include_relations=True, max_labelings=max_labelings),
        len(edges), len({key for edge in edges for key in edge.fillers}),
        components, tuple(sorted({row.relation_kind for row in records})),
    )


@dataclass(frozen=True, slots=True)
class IntegerGraphNoveltyCase:
    """A frozen case joins real records and an explicit dialogue-chain owner."""

    case_key: tuple[int, ...]
    input_values: tuple[int, ...]
    records: tuple[IntegerGraphTopologyRecord, ...]
    dialogue_chain_keys: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_key", _key(self.case_key, "case_key"))
        values = _key(self.input_values, "input_values")
        if any(value > 0x10FFFF or 0xD800 <= value <= 0xDFFF for value in values):
            raise IntegerGraphNoveltyError("input values are not Unicode scalar values")
        object.__setattr__(self, "input_values", values)
        if (not isinstance(self.records, (tuple, list)) or not self.records
                or any(not isinstance(row, IntegerGraphTopologyRecord) for row in self.records)):
            raise IntegerGraphNoveltyError("case has no actual graph records; novelty is unknown")
        object.__setattr__(self, "records", tuple(self.records))
        if not isinstance(self.dialogue_chain_keys, (tuple, list)) or not self.dialogue_chain_keys:
            raise IntegerGraphNoveltyError("dialogue-chain isolation evidence is missing")
        chains = tuple(_key(key, "dialogue_chain_key") for key in self.dialogue_chain_keys)
        if len(set(chains)) != len(chains):
            raise IntegerGraphNoveltyError("dialogue-chain identities are duplicated")
        object.__setattr__(self, "dialogue_chain_keys", chains)

    def isolation_keys(self) -> dict[str, frozenset[tuple[int, ...]]]:
        """Use actual course provenance, never shape or operation group labels.

        Document/content and concept/dedup ownership use the current compiler's
        declared group mapping. Their upstream correctness still needs a source
        audit; source-cluster checks additionally stop provenance relabeling.
        """
        fields = {
            "source": "source_ref_key", "source_cluster": "source_cluster_key",
            "document": "content_group_key", "concept_cluster": "dedup_cluster_key",
            "observation": "observation_key",
        }
        result = {label: frozenset(getattr(row, field) for row in self.records)
                  for label, field in fields.items()}
        result["dialogue_chain"] = frozenset(self.dialogue_chain_keys)
        return result


def validate_integer_graph_novelty_splits(
        splits: Mapping[str, Sequence[IntegerGraphNoveltyCase]], *,
        minimum_novel_held_out: int = 20,
        minimum_relation_families: int = 5) -> dict:
    """Return a reviewable admission matrix, excluding behavioral qualification."""
    if not isinstance(splits, Mapping) or set(splits) != set(_SPLITS):
        raise IntegerGraphNoveltyError("all four isolated splits are required")
    if any(type(value) is not int or value <= 0 for value in (
            minimum_novel_held_out, minimum_relation_families)):
        raise IntegerGraphNoveltyError("admission minimums must be positive integers")
    rows = []
    keys = set()
    owners: dict[tuple[str, tuple[int, ...]], str] = {}
    issues: set[str] = set()
    topology_by_case = {}
    for split in _SPLITS:
        cases = splits[split]
        if not isinstance(cases, (tuple, list)) or not cases:
            raise IntegerGraphNoveltyError(f"{split} has no frozen cases")
        for case in cases:
            if not isinstance(case, IntegerGraphNoveltyCase) or case.case_key in keys:
                raise IntegerGraphNoveltyError("missing graph case or duplicate case identity")
            keys.add(case.case_key)
            topology = integer_graph_topology(case.records)
            topology_by_case[case.case_key] = topology
            for label, values in case.isolation_keys().items():
                for value in values:
                    prior = owners.setdefault((label, value), split)
                    if prior != split:
                        issues.add(f"SPLIT_OVERLAP:{label}:{prior}:{split}")
    train = splits["train"]
    train_surfaces = {case.input_values for case in train}
    train_shapes = {topology_by_case[case.case_key].structure_sha256 for case in train}
    train_combos = {topology_by_case[case.case_key].relation_combo_sha256 for case in train}
    novel_combos = set()
    novel_case_count = 0
    held_kinds = set()
    for split in _SPLITS:
        for case in splits[split]:
            topology = topology_by_case[case.case_key]
            surface_novel = case.input_values not in train_surfaces
            structure_novel = topology.structure_sha256 not in train_shapes
            combo_novel = topology.relation_combo_sha256 not in train_combos
            if split == "held_out":
                held_kinds.update(topology.relation_kinds)
                if surface_novel and structure_novel and combo_novel and topology.component_count == 1:
                    novel_case_count += 1
                    novel_combos.add(topology.relation_combo_sha256)
                else:
                    issues.add("HELD_OUT_NOT_NOVEL_OR_CONNECTED")
            rows.append({
                "case_key": list(case.case_key), "split": split,
                "structure_sha256": topology.structure_sha256,
                "relation_combo_sha256": topology.relation_combo_sha256,
                "relation_count": topology.relation_count,
                "component_count": topology.component_count,
                "surface_novel": int(surface_novel),
                "structure_novel": int(structure_novel),
                "relation_combo_novel": int(combo_novel),
            })
    family_count = len({_PROFILES[kind].substage_key for kind in held_kinds})
    if len(novel_combos) < minimum_novel_held_out:
        issues.add("INSUFFICIENT_INDEPENDENT_NOVEL_HELD_OUT_GRAPHS")
    if family_count < minimum_relation_families:
        issues.add("INSUFFICIENT_HELD_OUT_RELATION_FAMILIES")
    return {
        "format": "PURE_INTEGER_GRAPH_NOVELTY_ADMISSION_V1",
        "status": "BLOCKED" if issues else "PASS_STRUCTURAL_ADMISSION_ONLY",
        "issues": sorted(issues), "rows": rows,
        "novel_held_out_case_count": novel_case_count,
        "independent_novel_held_out_graph_count": len(novel_combos),
        "held_out_relation_family_count": family_count,
        "operation_qualification": "UNKNOWN_requires_actual_behavior_evidence",
        "upstream_cluster_qualification": "UNKNOWN_requires_source_group_audit",
        "free_dialogue_complete": 0,
    }
