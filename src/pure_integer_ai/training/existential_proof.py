"""开放世界下的集合非空交声明验证器。

``A SUBSET_EQ B`` 本身不蕴含 A 非空，两个集合间也可能在不存在子集关系时
相交。因此存在声明的正证只能来自显式 overlap，或某个已知非空集合同时是
两侧的子集；反证只能来自显式 DISJOINT。普通缺边始终保持未知。
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pure_integer_ai.cognition.result.judge import SelfProofFn
from pure_integer_ai.cognition.shared.types import ConceptRef


Claim = tuple[ConceptRef, ConceptRef]


def _symmetric_pairs(pairs: Iterable[Claim]) -> frozenset[Claim]:
    """冻结无向集合关系，使参数顺序不改变 overlap 或 DISJOINT 语义。"""
    expanded: set[Claim] = set()
    for left, right in pairs:
        expanded.add((left, right))
        expanded.add((right, left))
    return frozenset(expanded)


def existential_proof_fn_factory(
        *, ancestor_map: dict[ConceptRef, set[ConceptRef]],
        claims: list[Claim],
        known_nonempty: Iterable[ConceptRef] = (),
        overlap_witnesses: Iterable[Claim] = (),
        disjoint_pairs: Iterable[Claim] = (),
        ) -> SelfProofFn:
    """构造集合相交验证函数，并冻结调用方注入的完整证据视图。

    ``overlap_witnesses`` 表示已由共同 MEMBER 见证或显式 overlap 规则归约出的
    集合对。``known_nonempty`` 中任一集合若同时为声明两侧的子集，也构成正证。
    不变量：祖先图缺边不产生反证；正证与 DISJOINT 冲突时返回未知。
    """
    frozen_ancestors = {child: set(parents)
                        for child, parents in ancestor_map.items()}
    frozen_claims = tuple(claims)
    frozen_nonempty = frozenset(known_nonempty)
    frozen_overlap = _symmetric_pairs(overlap_witnesses)
    frozen_disjoint = _symmetric_pairs(disjoint_pairs)

    def is_subset_eq(child: ConceptRef, parent: ConceptRef) -> bool:
        """查询注入的子集闭包，并显式保留集合自反性。"""
        return child == parent or parent in frozen_ancestors.get(child, set())

    def has_nonempty_subclass(left: ConceptRef, right: ConceptRef) -> bool:
        """判断是否存在已知非空集合同时包含于声明的两侧。"""
        return any(
            is_subset_eq(candidate, left)
            and is_subset_eq(candidate, right)
            for candidate in frozen_nonempty
        )

    def existential_proof_fn(
            output: Any, dag_path: Any, graph: Any,
            ) -> int | None:
        """按冻结证据返回 1、0 或未知，不从分类图缺边猜测空交集。"""
        del output, dag_path, graph
        if not frozen_claims:
            return None

        saw_unknown = False
        saw_disjoint = False
        for left, right in frozen_claims:
            supported = (
                (left, right) in frozen_overlap
                or has_nonempty_subclass(left, right)
            )
            refuted = (left, right) in frozen_disjoint
            if supported and refuted:
                return None
            if refuted:
                saw_disjoint = True
            elif not supported:
                saw_unknown = True
        if saw_disjoint:
            return 0
        if saw_unknown:
            return None
        return 1

    return existential_proof_fn
