"""开放世界下的全称子集声明验证器。

祖先路径只能支持 ``child SUBSET_EQ parent``。外部知识图中缺少路径不是
反例；只有调用方注入的独立反驳证据才能返回 0。同一声明同时得到支持和
反驳时返回 ``None``，避免从冲突证据中任意选择一侧。
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pure_integer_ai.cognition.result.judge import SelfProofFn
from pure_integer_ai.cognition.shared.types import ConceptRef


Claim = tuple[ConceptRef, ConceptRef]


def universal_proof_fn_factory(
        *, ancestor_map: dict[ConceptRef, set[ConceptRef]],
        claims: list[Claim],
        refuted_claims: Iterable[Claim] = (),
        ) -> SelfProofFn:
    """构造全称子集验证函数，并冻结调用方提供的支持与反驳证据。

    不变量：祖先路径只产生支持，缺边保持未知；``refuted_claims`` 必须来自
    独立的显式反例或否定证据，不能由祖先图缺边反推。多声明按合取处理，
    但任一声明自身证据冲突时整体返回未知。
    """
    frozen_ancestors = {child: set(parents)
                        for child, parents in ancestor_map.items()}
    frozen_claims = tuple(claims)
    frozen_refutations = frozenset(refuted_claims)

    def universal_proof_fn(
            output: Any, dag_path: Any, graph: Any,
            ) -> int | None:
        """按冻结证据返回 1、0 或未知，不读取运行期图的隐含缺边。"""
        del output, dag_path, graph
        if not frozen_claims:
            return None

        saw_unknown = False
        saw_refuted = False
        for child, parent in frozen_claims:
            supported = parent in frozen_ancestors.get(child, set())
            refuted = (child, parent) in frozen_refutations
            if supported and refuted:
                return None
            if refuted:
                saw_refuted = True
            elif not supported:
                saw_unknown = True
        if saw_refuted:
            return 0
        if saw_unknown:
            return None
        return 1

    return universal_proof_fn
