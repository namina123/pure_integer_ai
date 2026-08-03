"""W-07 train slice、learned profile 与共享 S-04 执行边界。"""
from __future__ import annotations

import hashlib

from pure_integer_ai.cognition.shared.identity import (
    ObjectIdentity,
    SourceRef,
    language_branch_identity,
    minimal_instruction_identity,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EPISTEMIC_REFUTED,
    LIFECYCLE_ARCHIVED,
    LIFECYCLE_SUPERSEDED,
)
from pure_integer_ai.cognition.shared.logic_candidate import (
    LogicOperatorAdoption,
)
from pure_integer_ai.cognition.shared.logic_executor import (
    LogicAtomEvidence,
    LogicEvaluation,
    LogicEvidenceState,
    LogicExecutor,
    LogicFailureProtocol,
    LogicOperatorDefinition,
    LogicOperatorRegistry,
    ModalResolution,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicPropositionDefinition,
    AtomicRoleBinding,
)
from pure_integer_ai.cognition.shared.typed_binding import (
    BindingEnvironment,
    BindingFailureProtocol,
    BoundProposition,
    ExactTypeCompatibilityResolver,
    PropositionTemplateGraph,
    ScopedPropositionTemplate,
    SubstitutionProtocol,
)
from pure_integer_ai.experiments.ph2_w07_adapter import (
    W07LogicProposal,
    W07ModalResolutionPlan,
    W07QuantifierBinding,
    W07TypedAdapterOutput,
    W07_IDENTITY_VERSIONS,
    W07_NAMESPACE,
)
from pure_integer_ai.experiments.ph2_w07_contract import W07_SUBSTAGE_ORDER
from pure_integer_ai.experiments.ph2_w07_learning import W07LogicLearningRuntime
from pure_integer_ai.experiments.ph2_w07_logic_contract import (
    W07LogicConsumerProtocol,
    W07LogicContractError,
    W07LogicExecution,
    W07LogicRequest,
)


W07_LOGIC_RUNTIME_NAMESPACE = 70731


def slice_w07_adapter(
        adapter: W07TypedAdapterOutput,
        enabled_substages: tuple[str, ...],
        ) -> W07TypedAdapterOutput:
    """保留严格前缀 train 记录；后续 substage 不进入当前 learning owner。"""
    if not isinstance(adapter, W07TypedAdapterOutput):
        raise TypeError("W-07 logic slice 需要 W07TypedAdapterOutput")
    if (not isinstance(enabled_substages, tuple) or not enabled_substages
            or enabled_substages != W07_SUBSTAGE_ORDER[:len(enabled_substages)]):
        raise W07LogicContractError("W-07 logic slice 必须是冻结子序前缀")
    enabled = set(enabled_substages)
    proposals = tuple(
        item for item in adapter.proposals
        if item.observation.substage in enabled)
    proposal_ids = {item.observation.stable_key for item in proposals}
    candidate_ids = {
        spec.candidate for item in proposals for spec in item.specs}
    specs = tuple(
        item for item in adapter.specs if item.candidate in candidate_ids)
    evidence = tuple(
        item for item in adapter.evidence
        if item.proposal.observation.stable_key in proposal_ids)
    source_keys = {
        item.source_binding.record.stable_key for item in proposals}
    rejections = tuple(
        item for item in adapter.rejections
        if item.observation.substage in enabled)
    source_keys.update(item.source_record.stable_key for item in rejections)
    sources = tuple(
        item for item in adapter.source_bindings
        if item.record.stable_key in source_keys)
    return W07TypedAdapterOutput(
        adapter.protocol,
        sources,
        proposals,
        specs,
        evidence,
        rejections,
        len(sources) + len(proposals) + len(evidence) + len(rejections),
        sum(len(item.bound_root.stable_key()) for item in proposals),
    )


def w07_logic_language_branch(proposal: W07LogicProposal) -> ObjectIdentity:
    """从 Observation.language 建一等 branch，不从 surface 猜语言。"""
    if not isinstance(proposal, W07LogicProposal):
        raise TypeError("W-07 language branch 需要 W07LogicProposal")
    value = proposal.observation.language
    return language_branch_identity(
        (W07_NAMESPACE, 731, len(value), *(ord(item) for item in value)),
        versions=W07_IDENTITY_VERSIONS,
    )


def _walk_bound(root: BoundProposition) -> tuple[BoundProposition, ...]:
    result = []

    def visit(item: BoundProposition) -> None:
        result.append(item)
        for binding in item.bindings:
            if isinstance(binding.filler, BoundProposition):
                visit(binding.filler)

    visit(root)
    return tuple(result)


def structure_tree_key(root: BoundProposition) -> tuple[int, ...]:
    """递归保存 operator tree，postcheck 不只检查根节点。"""
    values = [*root.structure.stable_key(), len(root.bindings)]
    for item in root.bindings:
        values.extend((
            len(item.role.stable_key()), *item.role.stable_key(),
            item.ordinal,
        ))
        if isinstance(item.filler, BoundProposition):
            child = structure_tree_key(item.filler)
            values.extend((2, len(child), *child))
        else:
            filler = item.filler.stable_key()
            values.extend((1, len(filler), *filler))
    return tuple(values)


def role_tree_key(
        root: BoundProposition,
        *,
        include_bound_provenance: bool = False,
        ) -> tuple[int, ...]:
    """递归保存 Role/ordinal；L02 起可冻结完整 bound provenance。"""
    if type(include_bound_provenance) is not bool:
        raise TypeError("include_bound_provenance 必须是严格 bool")
    if not include_bound_provenance:
        values = [len(root.bindings)]
        for item in root.bindings:
            role = item.role.stable_key()
            values.extend((len(role), *role, item.ordinal))
            filler = (
                role_tree_key(item.filler)
                if isinstance(item.filler, BoundProposition)
                else item.filler.stable_key()
            )
            values.extend((len(filler), *filler))
        return tuple(values)

    values = [W07_LOGIC_RUNTIME_NAMESPACE, 2]
    for identity in (root.template, root.source_anchor, root.context):
        key = identity.stable_key()
        values.extend((len(key), *key))
    values.append(len(root.introduced_binders))
    for binder in root.introduced_binders:
        key = binder.stable_key()
        values.extend((len(key), *key))
    values.append(len(root.applied_variables))
    for variable in root.applied_variables:
        key = variable.stable_key()
        values.extend((len(key), *key))
    values.append(len(root.bindings))
    for item in root.bindings:
        role = item.role.stable_key()
        values.extend((len(role), *role, item.ordinal))
        if isinstance(item.filler, BoundProposition):
            filler = role_tree_key(
                item.filler, include_bound_provenance=True)
            values.extend((2, len(filler), *filler))
        else:
            filler = item.filler.stable_key()
            values.extend((1, len(filler), *filler))
    return tuple(values)


def _template_graph(root: BoundProposition) -> PropositionTemplateGraph:
    """把 adapter 的 bound tree 恢复为真实 S-03 template graph。"""
    templates = []
    for item in _walk_bound(root):
        bindings = tuple(AtomicRoleBinding(
            binding.role,
            (binding.filler.template
             if isinstance(binding.filler, BoundProposition)
             else binding.filler),
            binding.ordinal,
        ) for binding in item.bindings)
        definition = AtomicPropositionDefinition(
            item.template,
            item.predicate,
            item.source_anchor,
            item.context,
            bindings,
        )
        templates.append(ScopedPropositionTemplate(
            definition, item.structure, item.introduced_binders))
    return PropositionTemplateGraph(tuple(templates))


def _evidence_id(tag: int, *keys: tuple[int, ...]) -> int:
    payload = bytearray()
    payload.extend(tag.to_bytes(4, "big", signed=False))
    for key in keys:
        payload.extend(len(key).to_bytes(4, "big", signed=False))
        for item in key:
            payload.extend(item.to_bytes(16, "big", signed=True))
    result = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    result &= (1 << 63) - 1
    return result or 1


def _state(value: dict, *, where: str) -> LogicEvidenceState:
    if (not isinstance(value, dict) or set(value) < {"support", "refute"}
            or type(value["support"]) is not int
            or type(value["refute"]) is not int
            or value["support"] not in {0, 1}
            or value["refute"] not in {0, 1}):
        raise W07LogicContractError(f"{where} 四态位非法")
    return LogicEvidenceState(
        bool(value["support"]), bool(value["refute"]))


class _ProposalAtomResolver:
    """只按公开 train operand/value Evidence 解析内容 premise。"""

    def __init__(self, proposal: W07LogicProposal) -> None:
        self.proposal = proposal
        raw = proposal.observation.typed_payload.to_value()
        self._direct: dict[ObjectIdentity, LogicEvidenceState] = {}
        for item in raw.get("operand_evidence", ()):
            key = ObjectIdentity.from_stable_key(tuple(item["template_key"]))
            self._direct[key] = _state(item, where="operand Evidence")
        leaf = raw.get("leaf_evidence")
        if leaf is not None:
            key = ObjectIdentity.from_stable_key(tuple(leaf["template_key"]))
            self._direct[key] = _state(leaf, where="leaf Evidence")

    def _quantifier_state(
            self, proposition: BoundProposition,
            ) -> LogicEvidenceState | None:
        for quantifier in self.proposal.quantifiers:
            matches = tuple(
                item.filler for item in proposition.bindings
                if item.role == quantifier.value_role)
            if len(matches) != 1 or not isinstance(matches[0], ObjectIdentity):
                continue
            for evidence in quantifier.value_evidence:
                if evidence.value == matches[0]:
                    return evidence.state
        return None

    def resolve(self, proposition, *, source, scope):
        state = self._direct.get(proposition.template)
        if state is None:
            state = self._quantifier_state(proposition)
        if state is None:
            return None
        base = _evidence_id(
            1, proposition.template.stable_key(), source.stable_key(),
            scope.stable_key())
        support = (base,) if state.support else ()
        refute = (_evidence_id(2, (base,)),) if state.refute else ()
        unknown = (_evidence_id(3, (base,)),) if not support and not refute else ()
        return LogicAtomEvidence(
            proposition.template,
            state,
            source,
            scope,
            None,
            support,
            refute,
            unknown,
        )


class _ProposalQuantifierResolver:
    def __init__(self, bindings: tuple[W07QuantifierBinding, ...]) -> None:
        self._bindings = {
            item.operator_candidate: item.definition for item in bindings}

    def resolve(self, operator, proposition, context):
        resolved = self._bindings.get(proposition.template)
        if resolved is not None and operator.structure != proposition.structure:
            raise W07LogicContractError("quantifier operator/structure 漂移")
        return resolved


class _ProposalModalResolver:
    def __init__(self, plans: tuple[W07ModalResolutionPlan, ...]) -> None:
        self._plans = plans

    def resolve(self, operator, child, context):
        candidates = tuple(
            item for item in self._plans
            if item.source == context.source
            and item.input_scope == context.scope)
        if len(candidates) != 1:
            return None
        plan = candidates[0]
        if plan.status != "RESOLVED" or plan.output_scope is None:
            return None
        return ModalResolution(
            plan.state,
            plan.source,
            plan.output_scope,
            plan.evidence_ids,
        )


def _binding_failures() -> BindingFailureProtocol:
    return BindingFailureProtocol(*tuple(
        minimal_instruction_identity((W07_LOGIC_RUNTIME_NAMESPACE, 10, index))
        for index in range(1, 10)
    ))


def _logic_failures() -> LogicFailureProtocol:
    return LogicFailureProtocol(*tuple(
        minimal_instruction_identity((W07_LOGIC_RUNTIME_NAMESPACE, 20, index))
        for index in range(1, 10)
    ))


def _course_budget(proposal: W07LogicProposal) -> dict[str, int]:
    raw = proposal.observation.typed_payload.to_value()
    request = raw.get("consumer_request")
    budget = None if not isinstance(request, dict) else request.get("budget")
    if not isinstance(budget, dict):
        raise W07LogicContractError("W-07 proposal 缺 consumer budget")
    return budget


def _verify_budget(
        proposal: W07LogicProposal,
        request: W07LogicRequest,
        ) -> None:
    nodes = _walk_bound(proposal.bound_root)
    depth = 0

    def visit(item: BoundProposition, level: int) -> None:
        nonlocal depth
        depth = max(depth, level)
        for binding in item.bindings:
            if isinstance(binding.filler, BoundProposition):
                visit(binding.filler, level + 1)

    visit(proposal.bound_root, 1)
    branches = sum(
        len(item.definition.domain.values) for item in proposal.quantifiers)
    resolvers = len(proposal.quantifiers) + len(proposal.modal_plans)
    actual = {
        "max_depth": depth,
        "max_branches": max(1, branches),
        "max_steps": len(nodes) + branches,
        "max_resolver_calls": max(1, resolvers),
    }
    requested = {
        "max_depth": request.budget.max_depth,
        "max_branches": request.budget.max_branches,
        "max_steps": request.budget.max_steps,
        "max_resolver_calls": request.budget.max_resolver_calls,
    }
    frozen = _course_budget(proposal)
    for key, value in actual.items():
        if value > requested[key]:
            raise W07LogicContractError(f"W-07 {key} resource 超限")
        if key in frozen and value > frozen[key]:
            raise W07LogicContractError(f"W-07 {key} 超出课程冻结预算")


class W07LogicView:
    """共享一个 W07 learning owner，并聚合同定义的多来源 adoption。"""

    def __init__(
            self,
            learning: W07LogicLearningRuntime,
            adapter: W07TypedAdapterOutput,
            protocol: W07LogicConsumerProtocol,
            ) -> None:
        if not isinstance(learning, W07LogicLearningRuntime):
            raise TypeError("W-07 logic view learning 类型非法")
        if not isinstance(adapter, W07TypedAdapterOutput):
            raise TypeError("W-07 logic view adapter 类型非法")
        if not isinstance(protocol, W07LogicConsumerProtocol):
            raise TypeError("W-07 logic view protocol 类型非法")
        registered = {item.candidate for item in learning.registered_specs()}
        expected = {item.candidate for item in adapter.specs}
        if registered != expected:
            raise W07LogicContractError(
                "W-07 logic view 必须绑定同一 prefix learning owner")
        present = tuple(
            item for item in W07_SUBSTAGE_ORDER
            if any(proposal.observation.substage == item
                   for proposal in adapter.proposals))
        if present != protocol.enabled_substages:
            raise W07LogicContractError("W-07 view/protocol substage 漂移")
        self.learning = learning
        self.adapter = adapter
        self.protocol = protocol
        self.proposals = adapter.proposals

    def proposal_for(self, request: W07LogicRequest) -> W07LogicProposal | None:
        matches = tuple(
            item for item in self.proposals
            if item.observation.substage == request.substage
            and item.bound_root.template == request.target_proposition
            and item.source_binding.source_ref == request.source
            and item.request_scope == request.scope)
        if len(matches) > 1:
            raise W07LogicContractError("同一 logic request 命中多个 proposal")
        return matches[0] if matches else None

    def active_adoptions(
            self, proposal: W07LogicProposal,
            ) -> tuple[LogicOperatorAdoption, ...]:
        """用共享 learned profile 执行内容；reject/superseded proposal 不借用。"""
        if set(proposal.operator_families).intersection(
                self.protocol.disabled_operator_families):
            return ()
        snapshots = tuple(
            self.learning.snapshot_for(item.candidate)
            for item in proposal.specs)
        if any(
                item.epistemic_status == EPISTEMIC_REFUTED
                or item.lifecycle in {LIFECYCLE_ARCHIVED, LIFECYCLE_SUPERSEDED}
                for item in snapshots):
            return ()
        _definitions, profiles = self._registry_profiles()
        required = {item.definition.structure for item in proposal.specs}
        values = tuple(
            item for item in profiles
            if item.spec.definition.structure in required)
        if {item.spec.definition.structure for item in values} != required:
            return ()
        return values

    def executable_proposals(self, substage: str) -> tuple[W07LogicProposal, ...]:
        return tuple(
            item for item in self.proposals
            if item.observation.substage == substage
            and self.active_adoptions(item))

    def _registry_profiles(
            self,
            ) -> tuple[
                tuple[LogicOperatorDefinition, ...],
                tuple[LogicOperatorAdoption, ...],
            ]:
        grouped: dict[ObjectIdentity, list[LogicOperatorAdoption]] = {}
        for proposal in self.proposals:
            if proposal.observation.substage in self.protocol.disabled_substages:
                continue
            for family, spec in zip(
                    proposal.operator_families, proposal.specs, strict=True):
                if family in self.protocol.disabled_operator_families:
                    continue
                adoption = self.learning.logic.adoption(spec)
                if adoption is not None:
                    grouped.setdefault(
                        spec.definition.structure, []).append(adoption)
        definitions = []
        adoptions = []
        for structure in sorted(grouped, key=ObjectIdentity.stable_key):
            current = grouped[structure]
            unique = {item.spec.definition.stable_key() for item in current}
            if len(unique) != 1:
                raise W07LogicContractError(
                    "同一 StructureConcept 出现不同 active operator profile")
            definitions.append(current[0].spec.definition)
            adoptions.extend(sorted(
                current, key=lambda item: item.spec.candidate.stable_key()))
        return tuple(definitions), tuple(adoptions)

    def _evaluate_bound(
            self,
            proposal: W07LogicProposal,
            request: W07LogicRequest,
            root: BoundProposition,
            *,
            definitions: tuple[LogicOperatorDefinition, ...] | None = None,
            ) -> LogicEvaluation:
        if definitions is None:
            definitions, _adoptions = self._registry_profiles()
        graph = _template_graph(proposal.bound_root)
        binding_failures = _binding_failures()
        substitution = SubstitutionProtocol(root.instruction, binding_failures)
        executor = LogicExecutor(
            LogicOperatorRegistry(definitions),
            _ProposalAtomResolver(proposal),
            _logic_failures(),
            substitution,
            ExactTypeCompatibilityResolver(),
            binding_failures,
        )
        return executor.evaluate(
            root,
            source=request.source,
            scope=request.scope,
            graph=graph,
            environment=BindingEnvironment(),
            quantifier_resolver=(
                _ProposalQuantifierResolver(proposal.quantifiers)
                if proposal.quantifiers else None),
            modal_resolver=(
                _ProposalModalResolver(proposal.modal_plans)
                if any(item.status == "RESOLVED"
                       for item in proposal.modal_plans) else None),
        )

    def evaluate_bound(
            self,
            request: W07LogicRequest,
            root: BoundProposition,
            ) -> LogicEvaluation | None:
        """用同一 learned registry 重算 proposal 内一个来源化 bound 子树。"""
        if not isinstance(request, W07LogicRequest):
            raise TypeError("W-07 child evaluation request 类型非法")
        if not isinstance(root, BoundProposition):
            raise TypeError("W-07 child evaluation root 类型非法")
        if (request.substage not in self.protocol.enabled_substages
                or request.substage in self.protocol.disabled_substages):
            return None
        proposal = self.proposal_for(request)
        if proposal is None or not self.active_adoptions(proposal):
            return None
        if root not in _walk_bound(proposal.bound_root):
            raise W07LogicContractError("child evaluation root 不属于 proposal")
        _verify_budget(proposal, request)
        return self._evaluate_bound(proposal, request, root)

    def execute(self, request: W07LogicRequest) -> W07LogicExecution | None:
        if not isinstance(request, W07LogicRequest):
            raise TypeError("W-07 execute request 类型非法")
        if (request.substage not in self.protocol.enabled_substages
                or request.substage in self.protocol.disabled_substages):
            return None
        proposal = self.proposal_for(request)
        if proposal is None or not self.active_adoptions(proposal):
            return None
        _verify_budget(proposal, request)
        definitions, all_adoptions = self._registry_profiles()
        evaluation = self._evaluate_bound(
            proposal,
            request,
            proposal.bound_root,
            definitions=definitions,
        )
        structures = tuple(sorted(
            {item.operator for item in evaluation.derivation},
            key=ObjectIdentity.stable_key))
        # Resolver/绑定失败没有 derivation step，但 active 根 operator 仍应
        # 归因到本次结构化 UNKNOWN execution。
        if not structures and evaluation.failures:
            root_definition = next(
                (item for item in definitions
                 if item.structure == proposal.bound_root.structure),
                None,
            )
            if root_definition is not None:
                structures = (root_definition.structure,)
        adoptions = tuple(
            item for item in all_adoptions
            if item.spec.definition.structure in set(structures))
        if not adoptions:
            return None
        operator_keys = tuple(sorted({
            (1, *evidence.stable_key())
            for adoption in adoptions for evidence in adoption.evidence
        }))
        content_keys = tuple(sorted({
            (2, item) for item in evaluation.evidence_ids
        }))
        return W07LogicExecution(
            request,
            evaluation,
            adoptions,
            operator_keys,
            content_keys,
            structures,
        )


__all__ = [
    "W07LogicView",
    "W07_LOGIC_RUNTIME_NAMESPACE",
    "role_tree_key",
    "slice_w07_adapter",
    "structure_tree_key",
    "w07_logic_language_branch",
]
