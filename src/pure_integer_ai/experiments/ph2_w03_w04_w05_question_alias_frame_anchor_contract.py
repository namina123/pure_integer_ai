"""从 FT18 alias frame 派生 FT19 前后缀 anchor trie。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_construction_index_contract import (
    RawQuestionConstructionIndex,
)


QUESTION_ALIAS_FRAME_ANCHOR_SHA256 = (
    "af4dcb9afb8ee0a96ff5883e057afed9527fd118e21bd5f93dea8cae9583dad2")
QUESTION_ALIAS_FRAME_ANCHOR_EXPRESSION_BOUNDARY = (
    ("index_source", "FT18_ALIAS_FRAMES_ONLY"),
    ("anchor_structure", "PREFIX_TRIE_AND_REVERSED_SUFFIX_TRIE"),
    ("candidate_operation", "PREFIX_SUFFIX_FRAME_INTERSECTION"),
    ("source_binding", "FILTERED_AT_TERMINAL_POSTING"),
    ("structural_unknown", "PRESERVED_FOR_UNLEARNED_ALIAS_SURFACES"),
    ("result_projection", "BYTE_IDENTICAL_TO_FT16_SCAN_RUNTIME"),
    ("handwritten_dispatch", "FORBIDDEN"),
)


# object-model: exception
class W03W04W05QuestionAliasFrameAnchorError(ValueError):
    """alias frame anchor trie 或其 FT18 所有权发生漂移。"""


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256(value: object, *, where: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise W03W04W05QuestionAliasFrameAnchorError(
            f"{where} is not a canonical SHA-256")
    return value


def _source_key(value: tuple[int, ...], *, where: str) -> tuple[int, ...]:
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int for item in value)):
        raise W03W04W05QuestionAliasFrameAnchorError(
            f"{where} is not a strict integer source key")
    return value


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class RawQuestionAliasFrameAnchorTerminal:
    """一个 anchor 终点可达的 frame ordinal 与来源域。"""

    frame_ordinal: int
    source_record_keys: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if type(self.frame_ordinal) is not int or self.frame_ordinal < 0:
            raise W03W04W05QuestionAliasFrameAnchorError(
                "alias frame terminal ordinal is invalid")
        if (not isinstance(self.source_record_keys, tuple)
                or not self.source_record_keys
                or self.source_record_keys
                != tuple(sorted(set(self.source_record_keys)))):
            raise W03W04W05QuestionAliasFrameAnchorError(
                "alias frame terminal SourceRefs are not canonical")
        for item in self.source_record_keys:
            _source_key(item, where="alias frame terminal SourceRef")

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_ordinal": self.frame_ordinal,
            "source_record_keys": [
                list(item) for item in self.source_record_keys],
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class RawQuestionAliasFrameAnchorEdge:
    """一个 Unicode 字符到下一个 trie node ordinal 的边。"""

    surface: str
    child_ordinal: int

    def __post_init__(self) -> None:
        if not isinstance(self.surface, str) or len(self.surface) != 1:
            raise W03W04W05QuestionAliasFrameAnchorError(
                "alias frame anchor edge is not one character")
        if type(self.child_ordinal) is not int or self.child_ordinal <= 0:
            raise W03W04W05QuestionAliasFrameAnchorError(
                "alias frame anchor child ordinal is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "child_ordinal": self.child_ordinal,
            "surface": self.surface,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class RawQuestionAliasFrameAnchorNode:
    """一个不可变 trie node。"""

    terminals: tuple[RawQuestionAliasFrameAnchorTerminal, ...]
    edges: tuple[RawQuestionAliasFrameAnchorEdge, ...]

    def __post_init__(self) -> None:
        if (not isinstance(self.terminals, tuple)
                or any(not isinstance(
                    item, RawQuestionAliasFrameAnchorTerminal)
                    for item in self.terminals)
                or self.terminals != tuple(sorted(
                    self.terminals,
                    key=lambda item: item.frame_ordinal,
                ))
                or len({item.frame_ordinal for item in self.terminals})
                != len(self.terminals)):
            raise W03W04W05QuestionAliasFrameAnchorError(
                "alias frame anchor terminals are not canonical")
        if (not isinstance(self.edges, tuple)
                or any(not isinstance(
                    item, RawQuestionAliasFrameAnchorEdge)
                    for item in self.edges)
                or self.edges != tuple(sorted(
                    self.edges,
                    key=lambda item: item.surface,
                ))
                or len({item.surface for item in self.edges})
                != len(self.edges)):
            raise W03W04W05QuestionAliasFrameAnchorError(
                "alias frame anchor edges are not canonical")

    def to_dict(self) -> dict[str, object]:
        return {
            "edges": [item.to_dict() for item in self.edges],
            "terminals": [item.to_dict() for item in self.terminals],
        }


def _validate_trie(
        nodes: tuple[RawQuestionAliasFrameAnchorNode, ...],
        construction_index: RawQuestionConstructionIndex,
        *,
        where: str,
        ) -> None:
    if (not isinstance(nodes, tuple) or not nodes
            or any(not isinstance(item, RawQuestionAliasFrameAnchorNode)
                   for item in nodes)):
        raise W03W04W05QuestionAliasFrameAnchorError(
            f"{where} alias frame anchor trie is invalid")
    parent_counts = [0 for _ in nodes]
    paths: list[str | None] = [None for _ in nodes]
    paths[0] = ""
    published: dict[int, tuple[tuple[int, ...], ...]] = {}
    for ordinal, node in enumerate(nodes):
        path = paths[ordinal]
        if path is None:
            raise W03W04W05QuestionAliasFrameAnchorError(
                f"{where} alias frame anchor node is unreachable")
        for edge in node.edges:
            if (edge.child_ordinal <= ordinal
                    or edge.child_ordinal >= len(nodes)):
                raise W03W04W05QuestionAliasFrameAnchorError(
                    f"{where} alias frame anchor edge escaped its trie")
            parent_counts[edge.child_ordinal] += 1
            child_path = path + edge.surface
            if (paths[edge.child_ordinal] is not None
                    and paths[edge.child_ordinal] != child_path):
                raise W03W04W05QuestionAliasFrameAnchorError(
                    f"{where} alias frame anchor path is ambiguous")
            paths[edge.child_ordinal] = child_path
        for terminal in node.terminals:
            if terminal.frame_ordinal in published:
                raise W03W04W05QuestionAliasFrameAnchorError(
                    f"{where} alias frame was published more than once")
            if terminal.frame_ordinal >= len(
                    construction_index.alias_frame_rows):
                raise W03W04W05QuestionAliasFrameAnchorError(
                    f"{where} alias frame terminal escaped FT18 rows")
            row = construction_index.alias_frame_rows[
                terminal.frame_ordinal]
            expected_path = (
                row.prefix_surface
                if where == "prefix" else row.suffix_surface[::-1]
            )
            if path != expected_path:
                raise W03W04W05QuestionAliasFrameAnchorError(
                    f"{where} alias frame terminal path drifted")
            published[terminal.frame_ordinal] = terminal.source_record_keys
    if parent_counts[0] != 0 or any(item != 1 for item in parent_counts[1:]):
        raise W03W04W05QuestionAliasFrameAnchorError(
            f"{where} alias frame anchor trie is not one rooted tree")
    expected = {
        ordinal: tuple(
            posting.source_record_key for posting in row.postings)
        for ordinal, row in enumerate(construction_index.alias_frame_rows)
    }
    if published != expected:
        raise W03W04W05QuestionAliasFrameAnchorError(
            f"{where} alias frame terminals escaped FT18 ownership")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class RawQuestionAliasFrameAnchorIndex:
    """FT18 alias frame 的前缀与反向后缀不可变 trie。"""

    construction_index: RawQuestionConstructionIndex
    prefix_nodes: tuple[RawQuestionAliasFrameAnchorNode, ...]
    suffix_nodes: tuple[RawQuestionAliasFrameAnchorNode, ...]
    identity_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.construction_index, RawQuestionConstructionIndex):
            raise TypeError("alias frame anchor construction index is invalid")
        _validate_trie(
            self.prefix_nodes,
            self.construction_index,
            where="prefix",
        )
        _validate_trie(
            self.suffix_nodes,
            self.construction_index,
            where="suffix",
        )
        _sha256(self.identity_sha256, where="alias frame anchor index")
        if self.identity_sha256 != self.sha256():
            raise W03W04W05QuestionAliasFrameAnchorError(
                "alias frame anchor identity drifted")

    def to_dict(self) -> dict[str, object]:
        return {
            "construction_index_identity_sha256": (
                self.construction_index.identity_sha256),
            "expression_boundary": [
                {"capability": key, "status": status}
                for key, status in (
                    QUESTION_ALIAS_FRAME_ANCHOR_EXPRESSION_BOUNDARY)
            ],
            "prefix_nodes": [item.to_dict() for item in self.prefix_nodes],
            "suffix_nodes": [item.to_dict() for item in self.suffix_nodes],
        }

    def sha256(self) -> str:
        return _sha(self.to_dict())


def _insert_anchor(
        root: dict[str, object],
        surface: str,
        frame_ordinal: int,
        source_record_keys: tuple[tuple[int, ...], ...],
        ) -> None:
    node = root
    for character in surface:
        children = node["children"]
        assert isinstance(children, dict)
        node = children.setdefault(
            character,
            {"children": {}, "terminals": {}},
        )
    terminals = node["terminals"]
    assert isinstance(terminals, dict)
    terminals[frame_ordinal] = source_record_keys


def _freeze_trie(root: dict[str, object]) -> tuple[RawQuestionAliasFrameAnchorNode, ...]:
    nodes: list[RawQuestionAliasFrameAnchorNode | None] = []

    def visit(value: dict[str, object]) -> int:
        ordinal = len(nodes)
        nodes.append(None)
        children = value["children"]
        terminals = value["terminals"]
        assert isinstance(children, dict) and isinstance(terminals, dict)
        edges = tuple(
            RawQuestionAliasFrameAnchorEdge(character, visit(child))
            for character, child in sorted(children.items())
        )
        node = RawQuestionAliasFrameAnchorNode(
            tuple(
                RawQuestionAliasFrameAnchorTerminal(
                    frame_ordinal,
                    source_record_keys,
                )
                for frame_ordinal, source_record_keys in sorted(
                    terminals.items())
            ),
            edges,
        )
        nodes[ordinal] = node
        return ordinal

    if visit(root) != 0 or any(item is None for item in nodes):
        raise W03W04W05QuestionAliasFrameAnchorError(
            "alias frame anchor trie freeze failed")
    return tuple(item for item in nodes if item is not None)


def build_raw_question_alias_frame_anchor_index(
        construction_index: RawQuestionConstructionIndex,
        *,
        expected_identity_sha256: str | None = None,
        ) -> RawQuestionAliasFrameAnchorIndex:
    """从 FT18 frame prefix/suffix 与真实 SourceRef 派生双 trie。"""
    if not isinstance(construction_index, RawQuestionConstructionIndex):
        raise TypeError("alias frame anchor index input is invalid")
    prefix_root = {"children": {}, "terminals": {}}
    suffix_root = {"children": {}, "terminals": {}}
    for ordinal, row in enumerate(construction_index.alias_frame_rows):
        source_record_keys = tuple(
            posting.source_record_key for posting in row.postings)
        _insert_anchor(
            prefix_root,
            row.prefix_surface,
            ordinal,
            source_record_keys,
        )
        _insert_anchor(
            suffix_root,
            row.suffix_surface[::-1],
            ordinal,
            source_record_keys,
        )
    prefix_nodes = _freeze_trie(prefix_root)
    suffix_nodes = _freeze_trie(suffix_root)
    payload = {
        "construction_index_identity_sha256": (
            construction_index.identity_sha256),
        "expression_boundary": [
            {"capability": key, "status": status}
            for key, status in QUESTION_ALIAS_FRAME_ANCHOR_EXPRESSION_BOUNDARY
        ],
        "prefix_nodes": [item.to_dict() for item in prefix_nodes],
        "suffix_nodes": [item.to_dict() for item in suffix_nodes],
    }
    identity = _sha(payload)
    value = RawQuestionAliasFrameAnchorIndex(
        construction_index,
        prefix_nodes,
        suffix_nodes,
        identity,
    )
    if (expected_identity_sha256 is not None
            and identity != _sha256(
                expected_identity_sha256,
                where="expected alias frame anchor index")):
        raise W03W04W05QuestionAliasFrameAnchorError(
            "alias frame anchor commitment drifted")
    return value


__all__ = [
    "QUESTION_ALIAS_FRAME_ANCHOR_EXPRESSION_BOUNDARY",
    "QUESTION_ALIAS_FRAME_ANCHOR_SHA256",
    "RawQuestionAliasFrameAnchorEdge",
    "RawQuestionAliasFrameAnchorIndex",
    "RawQuestionAliasFrameAnchorNode",
    "RawQuestionAliasFrameAnchorTerminal",
    "W03W04W05QuestionAliasFrameAnchorError",
    "build_raw_question_alias_frame_anchor_index",
]
