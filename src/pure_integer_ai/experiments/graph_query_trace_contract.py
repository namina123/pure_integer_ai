"""Decode graph evidence from either authoritative QueryBridge trace format.

Version 4 stores keys in an integer table; version 3 stores expanded keys.
The explicit compact artifact profile accepts the integer-only projection
written by the legacy Wave 1 trace writer, which removes string headers.
Missing evidence fields are contract errors, never observed zero counts. This
adapter does not query graphs, reconstruct old evidence, or generate answers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


COMPACT_INTEGER_TRACE_ARTIFACT_V1 = "COMPACT_INTEGER_TRACE_ARTIFACT_V1"


class GraphQueryTraceContractError(ValueError):
    """The supplied trace does not provide the evidence being requested."""


def _integer_key(value: object, field: str) -> tuple[int, ...]:
    """逐项验证完整非负整数键，避免为每个整数创建生成器恢复点。"""
    if not isinstance(value, (list, tuple)):
        raise GraphQueryTraceContractError(f"{field} must be a nonnegative integer key")
    for item in value:
        if type(item) is not int or item < 0:
            raise GraphQueryTraceContractError(f"{field} must be a nonnegative integer key")
    return tuple(value)


def _sequence(value: object, field: str) -> list | tuple:
    if not isinstance(value, (list, tuple)):
        raise GraphQueryTraceContractError(f"{field} must be a sequence")
    return value


def _integer_artifact_tree(value: object) -> None:
    """完整校验整数树；叶子就地核验，遍历栈只持有复合或待拒绝成员。"""
    pending = [value]
    while pending:
        item = pending.pop()
        if item is None or type(item) is int:
            continue
        if isinstance(item, (list, tuple)):
            children = item
        elif isinstance(item, Mapping):
            if any(not isinstance(key, str) or not key for key in item):
                raise GraphQueryTraceContractError("integer artifact object key is invalid")
            children = item.values()
        else:
            raise GraphQueryTraceContractError("integer artifact contains a noninteger value")
        for child in children:
            if child is not None and type(child) is not int:
                pending.append(child)


@dataclass(frozen=True, slots=True)
class GraphQueryHopEvidence:
    space: int
    proposition_key: tuple[int, ...]
    predicate_key: tuple[int, ...]
    source_hash: int


class GraphQueryTraceEvidence:
    """A strict, read-only view of an already-produced query trace."""

    def __init__(self, trace: Mapping[str, Any], *,
                 artifact_profile: str | None = None) -> None:
        if not isinstance(trace, Mapping):
            raise GraphQueryTraceContractError("query trace must be an object")
        if artifact_profile not in {None, COMPACT_INTEGER_TRACE_ARTIFACT_V1}:
            raise GraphQueryTraceContractError("unsupported query trace artifact profile")
        self.trace = trace
        self.artifact_profile = artifact_profile
        self.version = trace.get("schema_version")
        if type(self.version) is not int or self.version not in {3, 4}:
            raise GraphQueryTraceContractError("unsupported QueryBridge trace schema")
        self.keys: tuple[tuple[int, ...], ...] = ()
        if artifact_profile is not None:
            if self.version != 4:
                raise GraphQueryTraceContractError("integer artifact profile requires schema 4")
            _integer_artifact_tree(trace)
            if "integer_key_encoding" in trace or "format" in trace:
                raise GraphQueryTraceContractError("integer artifact retains a stripped header")
        if self.version == 4:
            if (artifact_profile is None
                    and trace.get("integer_key_encoding") != "INTEGER_KEY_TABLE_V1"):
                raise GraphQueryTraceContractError("compact integer key encoding is missing")
            table = _sequence(self._required("integer_key_table"), "integer_key_table")
            self.keys = tuple(_integer_key(key, "integer_key_table") for key in table)
            if any(not key for key in self.keys) or len(set(self.keys)) != len(self.keys):
                raise GraphQueryTraceContractError("compact key table is empty or duplicated")
        if artifact_profile is not None:
            self._validate_compact_artifact_envelope()

    @classmethod
    def from_integer_artifact(cls, trace: Mapping[str, Any]) -> GraphQueryTraceEvidence:
        """Read a known legacy integer projection, never infer its origin."""
        return cls(trace, artifact_profile=COMPACT_INTEGER_TRACE_ARTIFACT_V1)

    def _validate_compact_artifact_envelope(self) -> None:
        query = self.resolve_ref(self._required("query_ref"), "query_ref")
        state = self._required("state_key")
        frontier = self._required("frontier_trace")
        if not query or not isinstance(state, Mapping) or not isinstance(frontier, Mapping):
            raise GraphQueryTraceContractError("integer artifact query/state/frontier is invalid")
        if state.get("query_ref") != self.trace["query_ref"]:
            raise GraphQueryTraceContractError("integer artifact query/state reference drift")
        if not isinstance(self._required("input_structure"), Mapping):
            raise GraphQueryTraceContractError("integer artifact input structure is missing")
        _sequence(frontier.get("steps"), "frontier_trace.steps")
        scalars = _sequence(state.get("scalars"), "state_key.scalars")
        if len(scalars) != 17 or any(type(item) is not int for item in scalars):
            raise GraphQueryTraceContractError("integer artifact state scalar contract is invalid")

    def _required(self, field: str) -> Any:
        if field not in self.trace:
            raise GraphQueryTraceContractError(f"missing trace evidence field: {field}")
        return self.trace[field]

    def resolve_ref(self, value: object, field: str = "ref") -> tuple[int, ...]:
        if type(value) is not int or not 0 <= value <= len(self.keys):
            raise GraphQueryTraceContractError(f"{field} is not a valid compact key reference")
        return () if value == 0 else self.keys[value - 1]

    def _keys(self, expanded: str, compact: str) -> tuple[tuple[int, ...], ...]:
        field = compact if self.version == 4 else expanded
        values = _sequence(self._required(field), field)
        keys = tuple(
            self.resolve_ref(value, field) if self.version == 4
            else _integer_key(value, field)
            for value in values)
        if any(not key for key in keys):
            raise GraphQueryTraceContractError(f"{field} contains an empty evidence key")
        # If an expanded compatibility field exists, it must agree. Do not
        # select whichever representation happens to produce a nicer count.
        if self.version == 4 and expanded in self.trace:
            other = tuple(_integer_key(value, expanded) for value in _sequence(
                self.trace[expanded], expanded))
            if other != keys:
                raise GraphQueryTraceContractError(f"{expanded}/{compact} evidence drift")
        return keys

    def memory_routes(self) -> tuple[tuple[int, ...], ...]:
        return self._keys("memory_routes", "memory_route_refs")

    def expanded_edges(self) -> tuple[tuple[int, ...], ...]:
        return self._keys("expanded_edges", "expanded_edge_refs")

    def hops(self) -> tuple[GraphQueryHopEvidence, ...]:
        rows = _sequence(self._required("hops"), "hops")
        result = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise GraphQueryTraceContractError("hop evidence must be an object")
            space, source = row.get("space"), row.get("source_hash")
            if type(space) is not int or space not in {1, 2, 3}:
                raise GraphQueryTraceContractError("hop space is invalid")
            if type(source) is not int or source < 0:
                raise GraphQueryTraceContractError("hop source hash is invalid")
            if self.version == 4:
                proposition = self.resolve_ref(row.get("proposition_ref"), "proposition_ref")
                predicate = self.resolve_ref(row.get("predicate_ref"), "predicate_ref")
            else:
                proposition = _integer_key(row.get("proposition"), "proposition")
                predicate = _integer_key(row.get("predicate"), "predicate")
            result.append(GraphQueryHopEvidence(space, proposition, predicate, source))
        return tuple(result)

    def operation_contracts(
            self) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
        contracts = []
        for row in _sequence(self._required("operation_contracts"), "operation_contracts"):
            pair = _sequence(row, "operation_contract")
            if len(pair) != 2:
                raise GraphQueryTraceContractError("operation contract must have key and mode")
            key = _integer_key(pair[0], "operation_key")
            mode = _integer_key(pair[1], "operation_mode")
            if not key or not mode:
                raise GraphQueryTraceContractError("operation contract key/mode is empty")
            contracts.append((key, mode))
        if len({key for key, _ in contracts}) != len(contracts):
            raise GraphQueryTraceContractError("operation contract key is duplicated")
        return tuple(contracts)

    def counts(self) -> dict[str, int]:
        """Count only evidence fields that are present and valid in this trace."""
        hops = self.hops()
        return {
            "core_hop_count": sum(item.space == 1 for item in hops),
            "memory_hop_count": sum(item.space == 2 for item in hops),
            "companion_hop_count": sum(item.space == 3 for item in hops),
            "all_space_hop_count": len(hops),
            "memory_route_count": len(self.memory_routes()),
            "operation_contract_count": len(self.operation_contracts()),
            "operation_mode_count": len({mode for _, mode in self.operation_contracts()}),
        }
