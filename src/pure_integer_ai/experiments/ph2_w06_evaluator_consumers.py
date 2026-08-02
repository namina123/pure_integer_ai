"""W-06 private evaluator 对七类公开 U/R/G facade 的隔离复用。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pure_integer_ai.cognition.shared.alias_resolution import (
    AliasRouteSearchBudget,
)
from pure_integer_ai.cognition.shared.property_relation import (
    PropertyQueryBudget,
)
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    GenerationExpressionConstraints,
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_w06_adapter import W06TypedAdapterOutput
from pure_integer_ai.experiments.ph2_w06_candidate import W06_CASE_FAMILIES
from pure_integer_ai.experiments.ph2_w06_evaluator_contract import (
    W06PrivateEvaluationError,
)
from pure_integer_ai.experiments.ph2_w06_learning import (
    build_w06_learning_runtime,
)
from pure_integer_ai.experiments.ph2_w06_r01 import (
    W06R01Runtime,
    generation_request_for_candidate as r01_generation_request,
    slice_w06_r01_adapter,
)
from pure_integer_ai.experiments.ph2_w06_r01_contract import (
    W06R01ConsumerProtocol,
    W06R01ReasoningRequest,
    W06R01UnderstandingRequest,
)
from pure_integer_ai.experiments.ph2_w06_r01_shared import (
    candidate_endpoints as r01_candidate_endpoints,
    w06_r01_language_branch,
)
from pure_integer_ai.experiments.ph2_w06_r02 import (
    W06R02Runtime,
    generation_request_for_candidate as r02_generation_request,
    query_for_candidate as r02_query,
    slice_w06_r02_adapter,
)
from pure_integer_ai.experiments.ph2_w06_r02_contract import (
    W06R02ConsumerProtocol,
)
from pure_integer_ai.experiments.ph2_w06_r02_endpoint_projection import (
    W06_R02_ENDPOINT_PROJECTION_PATH,
    read_w06_r02_endpoint_projection,
)
from pure_integer_ai.experiments.ph2_w06_r02_shared import (
    w06_r02_language_branch,
)
from pure_integer_ai.experiments.ph2_w06_r03 import (
    W06R03Runtime,
    generation_request_for_candidate as r03_generation_request,
    slice_w06_r03_adapter,
)
from pure_integer_ai.experiments.ph2_w06_r03_contract import (
    W06R03ConsumerProtocol,
    W06R03ReasoningRequest,
    W06R03UnderstandingRequest,
)
from pure_integer_ai.experiments.ph2_w06_r03_shared import (
    w06_r03_language_branch,
)
from pure_integer_ai.experiments.ph2_w06_r04 import (
    W06R04Runtime,
    generation_request_for_candidate as r04_generation_request,
    query_for_candidate as r04_query,
    slice_w06_r04_adapter,
)
from pure_integer_ai.experiments.ph2_w06_r04_contract import (
    W06R04ConsumerProtocol,
)
from pure_integer_ai.experiments.ph2_w06_r04_endpoint_projection import (
    W06_R04_ENDPOINT_PROJECTION_PATH,
    read_w06_r04_endpoint_projection,
)
from pure_integer_ai.experiments.ph2_w06_r04_shared import (
    w06_r04_language_branch,
)
from pure_integer_ai.experiments.ph2_w06_r05 import (
    W06R05Runtime,
    generation_request_for_candidate as r05_generation_request,
    query_for_candidate as r05_query,
    slice_w06_r05_adapter,
)
from pure_integer_ai.experiments.ph2_w06_r05_contract import (
    W06R05ConsumerProtocol,
)
from pure_integer_ai.experiments.ph2_w06_r05_shared import (
    candidate_endpoints as r05_candidate_endpoints,
    w06_r05_language_branch,
)
from pure_integer_ai.experiments.ph2_w06_r06 import (
    W06R06Runtime,
    generation_request_for_candidate as r06_generation_request,
    slice_w06_r06_adapter,
)
from pure_integer_ai.experiments.ph2_w06_r06_contract import (
    W06R06ConsumerProtocol,
)
from pure_integer_ai.experiments.ph2_w06_r06_endpoint_projection import (
    W06_R06_ENDPOINT_PROJECTION_PATH,
    read_w06_r06_endpoint_projection,
)
from pure_integer_ai.experiments.ph2_w06_r06_shared import (
    w06_r06_language_branch,
)
from pure_integer_ai.experiments.ph2_w06_r07 import (
    W06R07Runtime,
    generation_request_for_candidate as r07_generation_request,
    slice_w06_r07_adapter,
)
from pure_integer_ai.experiments.ph2_w06_r07_contract import (
    W06R07ConsumerProtocol,
)
from pure_integer_ai.experiments.ph2_w06_r07_endpoint_projection import (
    W06_R07_ENDPOINT_PROJECTION_PATH,
    read_w06_r07_endpoint_projection,
)
from pure_integer_ai.experiments.ph2_w06_r07_shared import (
    w06_r07_language_branch,
)
from pure_integer_ai.experiments.ph2_w06_source_semantic import (
    W06_RELATION_SUBSTAGE_ORDER,
)
from pure_integer_ai.storage.backend import StorageBackend


_EVALUATOR_NAMESPACE = 50661
_EXPECTED_BY_SUBSTAGE = {
    key: (candidate_count, active_count)
    for key, candidate_count, active_count in W06_CASE_FAMILIES
}


@dataclass(frozen=True)
class _RelationBundle:
    """一个 relation substage 的隔离 adapter、learning owner 和 projection。"""

    substage: str
    backend: StorageBackend
    adapter: W06TypedAdapterOutput
    learning: object
    endpoint_resolver: object | None = None


class W06EvaluatorConsumerSuite:
    """持有七个隔离 relation owner，并只通过公开 facade 执行评测。"""

    def __init__(self, bundles: tuple[_RelationBundle, ...]) -> None:
        if (not isinstance(bundles, tuple)
                or tuple(item.substage for item in bundles)
                != W06_RELATION_SUBSTAGE_ORDER):
            raise W06PrivateEvaluationError("W-06 evaluator suite 顺序漂移")
        self._bundles = {item.substage: item for item in bundles}
        self._audit = {
            "generation_choices": 0,
            "generation_outcomes": 0,
            "generation_uses": 0,
            "reasoning_outcomes": 0,
            "reasoning_uses": 0,
            "understanding_outcomes": 0,
            "understanding_uses": 0,
        }

    def close(self) -> None:
        """关闭七个 evaluator-local backend。"""
        for bundle in self._bundles.values():
            bundle.backend.close()

    def audit(self) -> dict[str, int]:
        """返回由实际 facade 返回对象累计出的安全计数。"""
        return dict(self._audit)

    def _bundle(self, substage: str) -> _RelationBundle:
        """按冻结 substage 返回唯一隔离 bundle。"""
        try:
            return self._bundles[substage]
        except KeyError as error:
            raise W06PrivateEvaluationError(
                "W-06 evaluator substage 未注册") from error

    def _runtime(
            self,
            substage: str,
            *,
            target_connected: bool,
            generation_connected: bool,
            ):
        """用真实 relation protocol 开关建立一次无共享历史的 facade。"""
        bundle = self._bundle(substage)
        if substage == "PURE_ALIAS_REFERS":
            return W06R01Runtime(
                bundle.learning,
                bundle.adapter,
                protocol=W06R01ConsumerProtocol(
                    alias_refers_bridge_connected=target_connected,
                    generation_connected=generation_connected,
                ),
            )
        if substage == "SUBSET_MEMBER":
            return W06R02Runtime(
                bundle.learning,
                bundle.adapter,
                bundle.endpoint_resolver,
                protocol=W06R02ConsumerProtocol(
                    set_relation_bridge_connected=target_connected,
                    generation_connected=generation_connected,
                ),
            )
        if substage == "PROPERTY":
            return W06R03Runtime(
                bundle.learning,
                bundle.adapter,
                protocol=W06R03ConsumerProtocol(
                    property_bridge_connected=target_connected,
                    generation_connected=generation_connected,
                ),
            )
        if substage == "MEREOLOGY":
            return W06R04Runtime(
                bundle.learning,
                bundle.adapter,
                bundle.endpoint_resolver,
                protocol=W06R04ConsumerProtocol(
                    mereology_bridge_connected=target_connected,
                    generation_connected=generation_connected,
                ),
            )
        if substage == "SIMILAR_ANTONYM":
            return W06R05Runtime(
                bundle.learning,
                bundle.adapter,
                protocol=W06R05ConsumerProtocol(
                    similar_connected=target_connected,
                    antonym_connected=target_connected,
                    generation_connected=generation_connected,
                ),
            )
        if substage == "PRECEDES":
            return W06R06Runtime(
                bundle.learning,
                bundle.adapter,
                bundle.endpoint_resolver,
                protocol=W06R06ConsumerProtocol(
                    before_connected=target_connected,
                    after_connected=target_connected,
                    same_connected=target_connected,
                    unknown_connected=target_connected,
                    generation_connected=generation_connected,
                ),
            )
        if substage == "CAUSES":
            return W06R07Runtime(
                bundle.learning,
                bundle.adapter,
                bundle.endpoint_resolver,
                protocol=W06R07ConsumerProtocol(
                    causes_connected=target_connected,
                    generation_connected=generation_connected,
                ),
            )
        raise W06PrivateEvaluationError("W-06 evaluator runtime 未注册")

    def _active_candidates(self, substage: str) -> tuple:
        """返回一个隔离 owner 的 current active candidate。"""
        bundle = self._bundle(substage)
        return tuple(sorted(
            (
                item for item in bundle.adapter.candidates
                if bundle.learning.snapshot_for(
                    item.proposition.proposition).active_fact is not None
            ),
            key=lambda item: item.proposition.proposition.stable_key(),
        ))

    @staticmethod
    def _query_candidates(substage: str, runtime, active: tuple) -> tuple:
        """对对称修订和 CAUSES 修订只保留 canonical 查询代表。"""
        if substage == "SIMILAR_ANTONYM":
            representatives = {}
            for candidate in active:
                pair = tuple(sorted(
                    r05_candidate_endpoints(candidate),
                    key=lambda item: item.stable_key(),
                ))
                representatives.setdefault(
                    (candidate.relation_family, pair), candidate)
            return tuple(representatives[key] for key in sorted(
                representatives,
                key=lambda item: (
                    item[0],
                    *(value.stable_key() for value in item[1]),
                ),
            ))
        if substage == "CAUSES":
            representatives = {}
            for candidate in active:
                representatives.setdefault(
                    runtime.view.endpoints_for(candidate), candidate)
            return tuple(representatives.values())
        return active

    @staticmethod
    def _key(
            challenge: tuple[int, ...],
            evaluation_ordinal: int,
            consumer_ordinal: int,
            candidate_ordinal: int,
            ) -> LosslessIntegerKey:
        """把 private challenge 映射为不泄漏内容的 consumer request key。"""
        return LosslessIntegerKey((
            _EVALUATOR_NAMESPACE,
            evaluation_ordinal,
            consumer_ordinal,
            candidate_ordinal,
            len(challenge),
            *challenge,
        ))

    @staticmethod
    def _constraints(substage: str, candidate) -> GenerationExpressionConstraints:
        """按各 relation 已发布语言分支建立无 expected surface 约束。"""
        branches = {
            "PURE_ALIAS_REFERS": w06_r01_language_branch,
            "SUBSET_MEMBER": w06_r02_language_branch,
            "PROPERTY": w06_r03_language_branch,
            "MEREOLOGY": w06_r04_language_branch,
            "SIMILAR_ANTONYM": w06_r05_language_branch,
            "PRECEDES": w06_r06_language_branch,
            "CAUSES": w06_r07_language_branch,
        }
        branch = branches[substage](candidate)
        return GenerationExpressionConstraints(
            branch, (), (branch,), 0, 0, 0, 128)

    def _queries(
            self,
            substage: str,
            runtime,
            candidate,
            challenge: tuple[int, ...],
            evaluation_ordinal: int,
            candidate_ordinal: int,
            ) -> tuple[object, object]:
        """从 typed candidate 构造各 facade 的 Understanding/Reasoning 请求。"""
        understanding_key = self._key(
            challenge, evaluation_ordinal, 1, candidate_ordinal)
        reasoning_key = self._key(
            challenge, evaluation_ordinal, 2, candidate_ordinal)
        if substage == "PURE_ALIAS_REFERS":
            source, target = r01_candidate_endpoints(candidate)
            return (
                W06R01UnderstandingRequest(
                    understanding_key,
                    source,
                    (target.object_kind,),
                    AliasRouteSearchBudget(16, 16, 16),
                    False,
                ),
                W06R01ReasoningRequest(
                    reasoning_key,
                    candidate.relation_family,
                    source,
                    target,
                ),
            )
        if substage == "SUBSET_MEMBER":
            resolver = runtime.view.endpoint_resolver
            return (
                r02_query(candidate, request_key=understanding_key,
                          endpoint_resolver=resolver),
                r02_query(candidate, request_key=reasoning_key,
                          endpoint_resolver=resolver),
            )
        if substage == "PROPERTY":
            claim = runtime.view.claim_for(candidate)
            budget = PropertyQueryBudget(32, 32)
            return (
                W06R03UnderstandingRequest(
                    understanding_key, claim.subject, claim.attribute, budget),
                W06R03ReasoningRequest(reasoning_key, claim, budget),
            )
        if substage == "MEREOLOGY":
            resolver = runtime.view.endpoint_resolver
            return (
                r04_query(candidate, request_key=understanding_key,
                          endpoint_resolver=resolver),
                r04_query(candidate, request_key=reasoning_key,
                          endpoint_resolver=resolver),
            )
        if substage == "SIMILAR_ANTONYM":
            return (
                r05_query(candidate, request_key=understanding_key),
                r05_query(candidate, request_key=reasoning_key),
            )
        if substage == "PRECEDES":
            return (
                runtime.query_for_candidate(
                    candidate, request_key=understanding_key),
                runtime.query_for_candidate(
                    candidate, request_key=reasoning_key),
            )
        if substage == "CAUSES":
            return (
                runtime.query_for_candidate(
                    candidate, request_key=understanding_key),
                runtime.query_for_candidate(
                    candidate, request_key=reasoning_key),
            )
        raise W06PrivateEvaluationError("W-06 evaluator query 未注册")

    def _generation_request(
            self,
            substage: str,
            runtime,
            candidate,
            challenge: tuple[int, ...],
            evaluation_ordinal: int,
            candidate_ordinal: int,
            ):
        """从 typed active fact 构造各 relation 的真实 Generation 请求。"""
        key = self._key(
            challenge, evaluation_ordinal, 3, candidate_ordinal)
        constraints = self._constraints(substage, candidate)
        if substage == "PURE_ALIAS_REFERS":
            return r01_generation_request(
                candidate, request_key=key, constraints=constraints)
        if substage == "SUBSET_MEMBER":
            return r02_generation_request(
                candidate, request_key=key, constraints=constraints)
        if substage == "PROPERTY":
            return r03_generation_request(
                candidate,
                claim=runtime.view.claim_for(candidate),
                request_key=key,
                constraints=constraints,
            )
        if substage == "MEREOLOGY":
            return r04_generation_request(
                candidate, request_key=key, constraints=constraints)
        if substage == "SIMILAR_ANTONYM":
            return r05_generation_request(
                candidate, request_key=key, constraints=constraints)
        if substage == "PRECEDES":
            return r06_generation_request(
                candidate, request_key=key, constraints=constraints)
        if substage == "CAUSES":
            return r07_generation_request(
                candidate, request_key=key, constraints=constraints)
        raise W06PrivateEvaluationError("W-06 evaluator generation 未注册")

    def _consume_query(self, runtime, request, *, consumer: str) -> bool:
        """执行一次真实 resolve/adopt/verify，并只保留 SUPPORT 布尔值。"""
        if consumer == "UNDERSTANDING":
            resolution = runtime.resolve_understanding(request)
            if resolution.status not in {"UNIQUE", "SUPPORTED"}:
                return False
            use = runtime.adopt_understanding(resolution)
            self._audit["understanding_uses"] += 1
            outcome = runtime.verify_understanding(use)
            self._audit["understanding_outcomes"] += 1
            return outcome.verdict == "SUPPORT"
        if consumer == "REASONING":
            resolution = runtime.resolve_reasoning(request)
            if resolution.status != "SUPPORTED":
                return False
            use = runtime.adopt_reasoning(resolution)
            self._audit["reasoning_uses"] += 1
            outcome = runtime.verify_reasoning(use)
            self._audit["reasoning_outcomes"] += 1
            return outcome.verdict == "SUPPORT"
        raise W06PrivateEvaluationError("W-06 evaluator query consumer 非法")

    def _consume_generation(self, runtime, request) -> bool:
        """执行真实 choice/adopt/postcheck，并只保留 SUPPORT 布尔值。"""
        choice = runtime.choose_generation(request)
        self._audit["generation_choices"] += 1
        if choice.status != "READY" or not choice.options:
            return False
        use = runtime.adopt_generation(
            choice, choice.options[0].stable_key())
        self._audit["generation_uses"] += 1
        outcome = runtime.verify_generation(use)
        self._audit["generation_outcomes"] += 1
        return outcome.verdict == "SUPPORT"

    def evaluate_relation(
            self,
            substage: str,
            challenge: tuple[int, ...],
            *,
            target_connected: bool,
            evaluation_ordinal: int,
            ) -> tuple[bool, dict[str, object]]:
        """对一个 relation bearing 执行真实 U/R/G 和独立 postcheck。"""
        runtime = self._runtime(
            substage,
            target_connected=target_connected,
            generation_connected=True,
        )
        bundle = self._bundle(substage)
        active = self._active_candidates(substage)
        query_candidates = self._query_candidates(substage, runtime, active)
        understanding = []
        reasoning = []
        for ordinal, candidate in enumerate(query_candidates, start=1):
            requests = self._queries(
                substage,
                runtime,
                candidate,
                challenge,
                evaluation_ordinal,
                ordinal,
            )
            understanding.append(self._consume_query(
                runtime, requests[0], consumer="UNDERSTANDING"))
            reasoning.append(self._consume_query(
                runtime, requests[1], consumer="REASONING"))
        generation = [
            self._consume_generation(
                runtime,
                self._generation_request(
                    substage,
                    runtime,
                    candidate,
                    challenge,
                    evaluation_ordinal,
                    ordinal,
                ),
            )
            for ordinal, candidate in enumerate(active, start=1)
        ]
        expected_candidate, expected_active = _EXPECTED_BY_SUBSTAGE[substage]
        passed = all((
            len(bundle.adapter.candidates) == expected_candidate,
            len(active) == expected_active,
            bool(query_candidates),
            len(understanding) == len(query_candidates),
            all(understanding),
            len(reasoning) == len(query_candidates),
            all(reasoning),
            len(generation) == expected_active,
            all(generation),
        ))
        return passed, {
            "active_count": len(active),
            "candidate_count": len(bundle.adapter.candidates),
            "generation_postcheck_support_count": sum(generation),
            "generation_use_count": len(generation),
            "query_case_count": len(query_candidates),
            "reasoning_support_count": sum(reasoning),
            "substage": substage,
            "target_connected": int(target_connected),
            "understanding_support_count": sum(understanding),
        }

    def evaluate_generation_hard_conjunct(
            self,
            challenge: tuple[int, ...],
            *,
            generation_connected: bool,
            evaluation_ordinal: int,
            ) -> tuple[bool, dict[str, object]]:
        """逐 relation 执行一个真实 Generation choice/use/postcheck。"""
        support = []
        for substage_ordinal, substage in enumerate(
                W06_RELATION_SUBSTAGE_ORDER, start=1):
            runtime = self._runtime(
                substage,
                target_connected=True,
                generation_connected=generation_connected,
            )
            active = self._active_candidates(substage)
            if not active:
                support.append(False)
                continue
            request = self._generation_request(
                substage,
                runtime,
                active[0],
                challenge,
                evaluation_ordinal,
                100 + substage_ordinal,
            )
            support.append(self._consume_generation(runtime, request))
        passed = (
            len(support) == len(W06_RELATION_SUBSTAGE_ORDER)
            and all(support)
        )
        return passed, {
            "generation_connected": int(generation_connected),
            "postcheck_support_count": sum(support),
            "relation_substage_count": len(support),
        }


def build_w06_evaluator_consumer_suite(
        repository_root: str | Path,
        adapter: W06TypedAdapterOutput,
        *,
        backend_factory: Callable[[str], StorageBackend],
        ) -> W06EvaluatorConsumerSuite:
    """为七个 relation 建立物理分账的 evaluator-local learning owner。"""
    if not isinstance(adapter, W06TypedAdapterOutput):
        raise TypeError("W-06 evaluator suite 需要完整 adapter")
    if not callable(backend_factory):
        raise TypeError("W-06 evaluator backend factory 非法")
    root = Path(repository_root).resolve()
    slice_builders = {
        "PURE_ALIAS_REFERS": slice_w06_r01_adapter,
        "SUBSET_MEMBER": slice_w06_r02_adapter,
        "PROPERTY": slice_w06_r03_adapter,
        "MEREOLOGY": slice_w06_r04_adapter,
        "SIMILAR_ANTONYM": slice_w06_r05_adapter,
        "PRECEDES": slice_w06_r06_adapter,
        "CAUSES": slice_w06_r07_adapter,
    }
    resolvers = {
        "SUBSET_MEMBER": read_w06_r02_endpoint_projection(
            root / W06_R02_ENDPOINT_PROJECTION_PATH),
        "MEREOLOGY": read_w06_r04_endpoint_projection(
            root / W06_R04_ENDPOINT_PROJECTION_PATH),
        "PRECEDES": read_w06_r06_endpoint_projection(
            root / W06_R06_ENDPOINT_PROJECTION_PATH),
        "CAUSES": read_w06_r07_endpoint_projection(
            root / W06_R07_ENDPOINT_PROJECTION_PATH),
    }
    bundles = []
    try:
        for substage in W06_RELATION_SUBSTAGE_ORDER:
            sliced = slice_builders[substage](adapter)
            backend = backend_factory(substage)
            bundles.append(_RelationBundle(
                substage,
                backend,
                sliced,
                build_w06_learning_runtime(backend, sliced),
                resolvers.get(substage),
            ))
    except Exception:
        for bundle in bundles:
            bundle.backend.close()
        raise
    return W06EvaluatorConsumerSuite(tuple(bundles))


__all__ = [
    "W06EvaluatorConsumerSuite",
    "build_w06_evaluator_consumer_suite",
]
