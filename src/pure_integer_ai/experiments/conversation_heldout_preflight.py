"""DLG-05 六 case typed 输入 preflight。

这里仅构造无标签的 QuestionRequest 输入与 family 轴，供 catalog/manifest
边界、fresh/resume/clone 和真实 query executor 接入前使用。每个 case 使用
独立 SourceRef、Scope 和 Proposition 身份；本模块不声明 response-act、答案、
surface 或评测结果，也不承担正式 held-out label owner。
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    OBJECT_EVENT,
    OBJECT_MINIMAL_INSTRUCTION,
    ObjectIdentity,
    SourceRef,
    VersionBundle,
    concept_identity,
    language_atom_identity,
    language_branch_identity,
    minimal_instruction_identity,
    occurrence_identity,
    structure_concept_identity,
    representation_identity,
)
from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
from pure_integer_ai.cognition.shared.generation_plan import GenerationCandidate
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EvidenceRecord,
    HypothesisKey,
)
from pure_integer_ai.cognition.shared.question_answer import (
    QuestionExecutionResult,
    QuestionQuery,
)
from pure_integer_ai.cognition.shared.question_answer import QuestionRequest
from pure_integer_ai.cognition.shared.scope_identity import (
    document_scope,
    episode_scope,
    query_scope,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicRoleBinding,
    AtomicPropositionDefinition,
    context_scope_identity,
    entity_identity,
    event_identity,
    proposition_identity,
    role_identity,
)
from pure_integer_ai.cognition.shared.typed_binding import (
    BindingEnvironment,
    BindingFailureProtocol,
    PropositionSubstituter,
    PropositionTemplateGraph,
    ScopedPropositionTemplate,
    SubstitutionProtocol,
    BoundProposition,
)
from pure_integer_ai.experiments.conversation_heldout_family import (
    ConversationHeldOutCatalogCase,
    ConversationHeldOutCatalogTurn,
    ConversationHeldOutInputCatalog,
    build_conversation_heldout_manifest,
)
from pure_integer_ai.experiments.conversation_heldout_protocol import (
    AXIS_CONFLICT,
    AXIS_EVENT_REFERENCE,
    AXIS_EXPLICIT_REPEAT,
    AXIS_MEMORY_MISS,
    AXIS_OMISSION,
    AXIS_ORDER,
    AXIS_PROPOSITION_REFERENCE,
    AXIS_ROLLBACK,
    AXIS_SCOPE_DRIFT,
    AXIS_SYNONYM,
    AXIS_UNSEEN_RELATION,
    AXIS_UNSEEN_SOURCE,
    CONTEXT_CARRY,
    CONTEXT_EXPLICIT_REPEAT,
    CONTEXT_FRESH,
    CONTEXT_SCOPE_CHANGE,
    MEMORY_OFF,
    MEMORY_ON,
    REFERENCE_NONE,
    REFERENCE_PROPOSITION,
    REFERENCE_EVENT,
    ROLLBACK_NONE,
    ROLLBACK_READ_ONLY,
    ConversationHeldOutManifest,
)
from pure_integer_ai.experiments.conversation_heldout_language_input import (
    ConversationLexicalEvidence,
    ConversationLexicalNormalizer,
    ConversationLexicalRoute,
    ConversationQuestionInputCompiler,
    ConversationQuestionLanguageInput,
    ConversationQuestionRequestFrame,
    ConversationTypedUtterance,
)
from pure_integer_ai.experiments.conversation_heldout_relation_input import (
    ConversationRelationStructure,
    ConversationRelationTrainingInventory,
    ConversationUnseenRelationCompiler,
)
from pure_integer_ai.experiments.conversation_heldout_source_input import (
    ConversationSourceTrainingInventory,
    ConversationUnseenSourceCompiler,
)
from pure_integer_ai.experiments.evaluation_protocol import (
    CanonicalIdentity,
    ProtocolKey,
)
from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)


_NAMESPACE = 31005
_FAMILY = ProtocolKey((_NAMESPACE, 1, 1))
_REQUIRED_AXES = (
    AXIS_SYNONYM,
    AXIS_ORDER,
    AXIS_OMISSION,
    AXIS_EXPLICIT_REPEAT,
    AXIS_PROPOSITION_REFERENCE,
    AXIS_EVENT_REFERENCE,
    AXIS_UNSEEN_SOURCE,
    AXIS_UNSEEN_RELATION,
    AXIS_CONFLICT,
    AXIS_MEMORY_MISS,
    AXIS_SCOPE_DRIFT,
    AXIS_ROLLBACK,
)


@dataclass(frozen=True, slots=True)
class ConversationHeldOutAxisInputAudit:
    """记录某个轴是否改变了 typed runtime input，而非只改变 manifest 键。"""

    axis: ProtocolKey
    case_keys: tuple[ProtocolKey, ...]
    typed_input_bound: int
    semantic_runtime_bound: int
    manifest_only: int
    note: str

    def __post_init__(self) -> None:
        """核验审计结果只使用明确的 0/1 状态。"""
        if not isinstance(self.axis, ProtocolKey) or not self.axis.components:
            raise TypeError("axis audit axis 非法")
        if (not isinstance(self.case_keys, tuple)
                or any(not isinstance(item, ProtocolKey)
                       for item in self.case_keys)):
            raise TypeError("axis audit case keys 非法")
        for name, value in (
                ("typed_input_bound", self.typed_input_bound),
                ("semantic_runtime_bound", self.semantic_runtime_bound),
                ("manifest_only", self.manifest_only)):
            if type(value) is not int or value not in (0, 1):
                raise ValueError(f"axis audit {name} 必须为 0/1")
        if not isinstance(self.note, str) or not self.note:
            raise ValueError("axis audit note 不得为空")


def _bound_event_keys(
        proposition: BoundProposition,
        ) -> tuple[tuple[int, ...], ...]:
    """递归读取 BoundProposition 中实际绑定的 Event identity。"""
    if not isinstance(proposition, BoundProposition):
        raise TypeError("event reference audit 需要 BoundProposition")
    found: list[tuple[int, ...]] = []
    for binding in proposition.bindings:
        filler = binding.filler
        if isinstance(filler, BoundProposition):
            found.extend(_bound_event_keys(filler))
        elif filler.object_kind == OBJECT_EVENT:
            found.append(filler.stable_key())
    return tuple(sorted(set(found)))


def audit_dlg05_preflight_axis_inputs(
        catalog: ConversationHeldOutInputCatalog,
        *,
        compiler: ConversationQuestionInputCompiler | None = None,
        relation_compiler: ConversationUnseenRelationCompiler | None = None,
        source_compiler: ConversationUnseenSourceCompiler | None = None,
        ) -> tuple[ConversationHeldOutAxisInputAudit, ...]:
    """审计六 case 的轴是否真正改变 typed 输入；不把 preflight 当能力证明。

    synonym/order/event-reference 只有在实际 typed 输入或事件槽可重建后才能升级；
    omission/explicit-repeat 的静态审计保持保守，unseen relation 仍保持未闭合。
    单独的 axis key、trace、case id 或 response mode 不能冒充这些输入存在。
    """
    if not isinstance(catalog, ConversationHeldOutInputCatalog):
        raise TypeError("axis audit catalog 类型错误")
    if (compiler is not None
            and not isinstance(compiler, ConversationQuestionInputCompiler)):
        raise TypeError("axis audit compiler 类型错误")
    if (relation_compiler is not None
            and not isinstance(
                relation_compiler, ConversationUnseenRelationCompiler)):
        raise TypeError("axis audit relation compiler 类型错误")
    if (source_compiler is not None
            and not isinstance(source_compiler, ConversationUnseenSourceCompiler)):
        raise TypeError("axis audit source compiler 类型错误")
    observed = []
    for case in catalog.cases:
        for axis in case.axis_keys:
            if axis not in observed:
                observed.append(axis)
    result = []
    for axis in observed:
        cases = tuple(case for case in catalog.cases if axis in case.axis_keys)
        case_keys = tuple(case.case_key for case in cases)
        bound = 0
        note = "当前只存在 manifest/fixture 轴，需补 typed 输入"
        if axis == AXIS_PROPOSITION_REFERENCE:
            bound = int(any(
                len(case.turns) >= 2
                and case.turns[1].reference_mode == REFERENCE_PROPOSITION
                and case.turns[1].request.target == case.turns[0].request.target
                for case in cases))
            note = "第二回合复用前回合 BoundProposition"
        elif axis == AXIS_SCOPE_DRIFT:
            bound = int(any(
                len(case.turns) >= 2
                and case.turns[1].request.response_scope
                != case.turns[0].request.response_scope
                for case in cases))
            note = "第二回合真实更换 response scope"
        elif axis == AXIS_ROLLBACK:
            bound = int(any(
                len(case.turns) >= 2
                and case.turns[1].rollback_mode == ROLLBACK_READ_ONLY
                for case in cases))
            note = "第二回合显式声明 read-only rollback"
        elif axis == AXIS_UNSEEN_SOURCE:
            if source_compiler is not None:
                sources = tuple(
                    source_compiler.compile(turn.request)
                    for case in cases for turn in case.turns)
                bound = int(bool(sources) and len(set(sources)) == 1)
            note = (
                "精确 SourceRef 未见，但 source kind/owner/version 域已见"
                if bound else "没有冻结 TRAIN SourceRef 分母")
        elif axis == AXIS_SYNONYM:
            inputs = tuple(
                turn.language_input.stable_key()
                for case in cases
                for turn in case.turns
                if turn.language_input is not None
            )
            bound = int(len(inputs) >= 2 and len(set(inputs)) == len(inputs))
            note = (
                "两个不同 Representation 经 lexical route 进入同一 typed case"
                if bound else "QuestionRequest 不携带 synonym/alias 表面")
        elif axis == AXIS_EVENT_REFERENCE:
            event_keys = tuple(
                _bound_event_keys(turn.request.target)
                for case in cases for turn in case.turns)
            bound = int(any(event_keys) and any(
                len(case.turns) >= 2
                and case.turns[1].reference_mode == REFERENCE_EVENT
                and _bound_event_keys(case.turns[1].request.target)
                == _bound_event_keys(case.turns[0].request.target)
                and case.turns[1].request.target != case.turns[0].request.target
                for case in cases))
            note = (
                "不同 BoundProposition 通过真实 Event filler 复用同一 Event"
                if bound else "只有 reference mode key，没有独立 event typed target")
        elif axis == AXIS_ORDER:
            inputs = tuple(
                turn.language_input
                for case in cases for turn in case.turns
                if turn.language_input is not None)
            bound = int(
                len(inputs) >= 2
                and tuple(
                    item.utterance.visible_forms for item in inputs)
                != tuple(reversed(tuple(
                    item.utterance.visible_forms for item in inputs))))
            note = (
                "两个 typed utterance 使用不同 Representation 顺序"
                if bound else "没有同输入不同问序的 typed utterance 序列")
        elif axis == AXIS_OMISSION:
            inputs = tuple(
                turn.language_input
                for case in cases for turn in case.turns
                if turn.language_input is not None)
            bound = int(any(
                item.provided_positions
                and len(item.provided_positions)
                < len(next(
                    turn.language_input
                    for case in cases
                    for turn in case.turns
                    if turn.language_input is not None
                ).utterance.visible_forms)
                for item in inputs))
            note = (
                "第二回合实际省略 Representation 位置并声明 context anchor"
                if bound else "context mode 不是实际省略的用户输入")
        elif axis == AXIS_EXPLICIT_REPEAT:
            bound = int(any(
                len(case.turns) >= 2
                and case.turns[1].request == case.turns[0].request
                for case in cases))
            note = (
                "第二回合复用完整 typed QuestionRequest"
                if bound else "重复轴仍只改变 trace/case 身份")
        elif axis == AXIS_UNSEEN_RELATION:
            if relation_compiler is not None:
                structures = tuple(
                    relation_compiler.compile(turn.request)
                    for case in cases for turn in case.turns)
                bound = int(bool(structures) and len(set(structures)) == 1)
            note = (
                "完整 relation 组合未见且 predicate/construction/Role-type 原语已见"
                if bound else "独立 identity 不等于未见 relation 结构")
        elif axis in {AXIS_CONFLICT, AXIS_MEMORY_MISS}:
            note = "候选/Memory fixture 可触发运行态，但不是 catalog 输入本身"
        semantic = 0
        if compiler is not None and axis in {AXIS_SYNONYM, AXIS_ORDER}:
            inputs = tuple(
                turn.language_input
                for case in cases for turn in case.turns
                if turn.language_input is not None)
            if inputs:
                normalized = tuple(
                    compiler.normalizer.normalize(item.utterance)
                    for item in inputs)
                compiled = tuple(compiler.compile(item) for item in inputs)
                requests = tuple(
                    next(
                        turn.request
                        for case in cases
                        for turn in case.turns
                        if turn.language_input == inputs[index])
                    for index in range(len(inputs)))
                compiled_equal = compiled == requests
                if axis == AXIS_SYNONYM:
                    semantic = int(
                        compiled_equal
                        and len({tuple(sorted(
                            atom.stable_key() for atom in item.semantic_atoms))
                            for item in normalized}) == 1)
                elif axis == AXIS_ORDER:
                    semantic = int(
                        compiled_equal
                        and len(normalized) == 2
                        and normalized[0].semantic_atoms
                        == tuple(reversed(normalized[1].semantic_atoms))
                        and requests[0].target != requests[1].target)
                else:
                    anchor_key = next(
                        turn.request.target.stable_key()
                        for case in cases
                        for turn in case.turns
                        if turn.language_input == inputs[0])
                    semantic = int(
                        compiled_equal
                        and any(item.provided_positions for item in inputs)
                        and len(inputs) == 2
                        and inputs[1].context_target_key
                        == anchor_key)
                if semantic:
                    note = (
                        "不同 Representation 已由 lexical evidence/compiler 归一并进入 QuestionRequest"
                        if axis == AXIS_SYNONYM else
                        "省略槽已由 ConversationContextRead target anchor 补全并进入 QuestionRequest"
                        if axis == AXIS_OMISSION else
                        "倒序 LanguageAtom 序列已由独立结构 frame 绑定不同 Proposition")
        result.append(ConversationHeldOutAxisInputAudit(
            axis, case_keys, bound, semantic, int(not bound), note))
    return tuple(result)


def _identity(kind: str, ordinal: int) -> CanonicalIdentity:
    """建立 preflight 专用的完整 cluster identity。"""
    return CanonicalIdentity.from_value(("dlg05-preflight-v1", kind, ordinal))


def _request(
        case: int,
        turn: int,
        *,
        candidate_count: int = 1,
        event_target: bool = False,
        relation_target: bool = False,
        ) -> QuestionRequest:
    """构造新的 typed Proposition/Source/Scope，不读取课程 episode。"""
    if type(candidate_count) is not int or candidate_count <= 0:
        raise ValueError("preflight candidate_count 必须为正整数")
    source = SourceRef(
        _NAMESPACE,
        100 + case,
        case,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(),
    )
    evidence_scope = document_scope(source)
    # DLG-04 需要真实 WorkMemory 生命周期；query 必须通过显式 episode
    # 挂在 document 下，不能用 detached document->query scope 冒充活动 query。
    response_episode = episode_scope(
        2000 + case * 10 + turn, parent=evidence_scope)
    response_scope = query_scope(
        1000 + case * 10 + turn, parent=response_episode)
    event_role = role_identity((_NAMESPACE, 17, case, 1))
    event = event_identity(source, (_NAMESPACE, 18, case, 1))
    relation_roles = tuple(
        role_identity((_NAMESPACE, 19, case, ordinal))
        for ordinal in (1, 2)
    )
    relation_entities = tuple(
        entity_identity(source, (_NAMESPACE, 20, case, ordinal))
        for ordinal in (1, 2)
    )
    if event_target and relation_target:
        raise ValueError("preflight target 不得同时声明 event/relation fixture")
    if event_target:
        bindings = (AtomicRoleBinding(event_role, event),)
    elif relation_target:
        bindings = tuple(
            AtomicRoleBinding(role, entity)
            for role, entity in zip(relation_roles, relation_entities)
        )
    else:
        bindings = ()
    definitions = tuple(
        AtomicPropositionDefinition(
            proposition_identity(
                source, (_NAMESPACE, 10, case, turn, index)),
            concept_identity((_NAMESPACE, 11, case, turn, index)),
            occurrence_identity(
                source,
                start=case + turn + index,
                end=case + turn + index + 1,
                ordinal=0),
            context_scope_identity(
                source, (_NAMESPACE, 12, case, turn, index)),
            bindings,
        )
        for index in range(1, candidate_count + 1)
    )
    graph = PropositionTemplateGraph(tuple(
        ScopedPropositionTemplate(
            definition,
            structure_concept_identity(
                (_NAMESPACE, 13, case, turn, index)),
        )
        for index, definition in enumerate(definitions, start=1)
    ))
    failures = BindingFailureProtocol(*tuple(
        minimal_instruction_identity((_NAMESPACE, 14, case, index))
        for index in range(1, 10)
    ))
    substituter = PropositionSubstituter(SubstitutionProtocol(
        minimal_instruction_identity((_NAMESPACE, 15, case)),
        failures,
    ))
    targets = tuple(
        substituter.substitute(definition.proposition, graph, BindingEnvironment())
        for definition in definitions
    )
    branch = language_branch_identity((_NAMESPACE, 16, case))
    return QuestionRequest(
        minimal_instruction_identity((_NAMESPACE, 20, case, 1)),
        minimal_instruction_identity((_NAMESPACE, 20, case, 2)),
        minimal_instruction_identity((_NAMESPACE, 20, case, 3)),
        targets[0],
        LogicEvidenceState(True, False),
        evidence_scope,
        response_scope,
        (_NAMESPACE, 21, case, turn),
        branch,
        targets if candidate_count > 1 else (),
    )


def _case(
        number: int,
        axes: tuple[ProtocolKey, ...],
        modes: tuple[ProtocolKey, ProtocolKey],
        context_second: ProtocolKey,
        reference_second: ProtocolKey,
        rollback_second: ProtocolKey,
        ) -> ConversationHeldOutCatalogCase:
    """把两个无标签 QuestionRequest 组成一个 typed 两回合 case。"""
    candidate_count = 2 if number == 2 else 1
    event_target = AXIS_EVENT_REFERENCE in axes
    relation_target = AXIS_UNSEEN_RELATION in axes
    first = _request(
        number, 1, candidate_count=candidate_count,
        event_target=event_target,
        relation_target=relation_target)
    second_base = _request(
        number, 2, candidate_count=candidate_count,
        event_target=event_target,
        relation_target=relation_target)
    second = replace(
        second_base,
        target=(
            second_base.target
            if AXIS_ORDER in axes or AXIS_EVENT_REFERENCE in axes
            else first.target
        ),
        evidence_scope=first.evidence_scope,
        response_scope=(
            second_base.response_scope
            if context_second == CONTEXT_SCOPE_CHANGE
            else first.response_scope
        ),
        target_branch=first.target_branch,
        authorized_candidate_targets=first.authorized_candidate_targets,
    )
    if context_second == CONTEXT_EXPLICIT_REPEAT:
        # 显式重复必须复用完整 typed request；只有本轮 context read/revision
        # 会在 runner 内追加新的 read digest，不能用 trace 差异冒充重复输入。
        second = first
    turns = (
        ConversationHeldOutCatalogTurn(
            ProtocolKey((_NAMESPACE, 100, number, 1)),
            1,
            first,
            CONTEXT_FRESH,
            modes[0],
            REFERENCE_NONE,
            ROLLBACK_NONE,
        ),
        ConversationHeldOutCatalogTurn(
            ProtocolKey((_NAMESPACE, 100, number, 2)),
            2,
            second,
            context_second,
            modes[1],
            reference_second,
            rollback_second,
        ),
    )
    return ConversationHeldOutCatalogCase(
        ProtocolKey((_NAMESPACE, 200, number)),
        _FAMILY,
        axes,
        _identity("dedup", number),
        _identity("provenance", number),
        turns,
    )


def build_dlg05_typed_preflight_catalog() -> ConversationHeldOutInputCatalog:
    """构造六 case、双回合、Memory ON/OFF 完整覆盖的无标签 catalog。"""
    cases = [
        _case(1, (AXIS_SYNONYM, AXIS_ORDER),
              (MEMORY_OFF, MEMORY_ON), CONTEXT_CARRY,
              REFERENCE_NONE, ROLLBACK_NONE),
        _case(2, (AXIS_OMISSION, AXIS_PROPOSITION_REFERENCE),
              (MEMORY_OFF, MEMORY_OFF), CONTEXT_CARRY,
              REFERENCE_PROPOSITION, ROLLBACK_NONE),
        _case(3, (AXIS_EVENT_REFERENCE, AXIS_UNSEEN_SOURCE),
              (MEMORY_OFF, MEMORY_ON), CONTEXT_CARRY,
              REFERENCE_EVENT, ROLLBACK_NONE),
        _case(4, (AXIS_UNSEEN_RELATION, AXIS_CONFLICT),
              (MEMORY_ON, MEMORY_OFF), CONTEXT_CARRY,
              REFERENCE_NONE, ROLLBACK_NONE),
        _case(5, (AXIS_MEMORY_MISS, AXIS_SCOPE_DRIFT),
              (MEMORY_OFF, MEMORY_ON), CONTEXT_SCOPE_CHANGE,
              REFERENCE_NONE, ROLLBACK_NONE),
        _case(6, (AXIS_EXPLICIT_REPEAT, AXIS_ROLLBACK),
              (MEMORY_OFF, MEMORY_ON), CONTEXT_EXPLICIT_REPEAT,
              REFERENCE_NONE, ROLLBACK_READ_ONLY),
    ]
    # case 1 的两个回合输入使用不同 Representation，且语义原子实际倒序。
    # 每个词形经 evidence-backed lexical route 归一；这是 synonym+order 的
    # typed 输入，不是修改 case id、turn ordinal 或 trace 来伪装语言变化。
    case = cases[0]
    forms = (
        representation_identity((_NAMESPACE, 300, 1), (1,)),
        representation_identity((_NAMESPACE, 300, 1), (2,)),
        representation_identity((_NAMESPACE, 300, 1), (3,)),
        representation_identity((_NAMESPACE, 300, 1), (4,)),
    )
    utterance_forms = ((forms[0], forms[1]), (forms[3], forms[2]))
    language_inputs = tuple(
        ConversationQuestionLanguageInput(
            ConversationTypedUtterance(
                case.turns[0].request.target_branch,
                utterance_forms[ordinal - 1]),
            structure_concept_identity((_NAMESPACE, 301, 1)),
            (ordinal,),
        )
        for ordinal, _visible_forms in enumerate(utterance_forms, start=1)
    )
    cases[0] = replace(
        case,
        turns=tuple(
            replace(turn, language_input=language_inputs[index])
            for index, turn in enumerate(case.turns)
        ),
    )
    # case 2 的第二回合真实省略位置 0；它只提供位置 1 的词形，
    # context_target_key 必须从第一回合的 typed target anchor 取得。
    case = cases[1]
    omission_forms = (
        representation_identity((_NAMESPACE, 310, 2), (1,)),
        representation_identity((_NAMESPACE, 310, 2), (2,)),
    )
    omission_inputs = (
        ConversationQuestionLanguageInput(
            ConversationTypedUtterance(
                case.turns[0].request.target_branch,
                omission_forms),
            structure_concept_identity((_NAMESPACE, 311, 2)),
            (1,),
        ),
        ConversationQuestionLanguageInput(
            ConversationTypedUtterance(
                case.turns[0].request.target_branch,
                (omission_forms[1],)),
            structure_concept_identity((_NAMESPACE, 311, 2)),
            (2,),
            (1,),
            case.turns[0].request.target.stable_key(),
        ),
    )
    cases[1] = replace(
        case,
        turns=tuple(
            replace(turn, language_input=omission_inputs[index])
            for index, turn in enumerate(case.turns)
        ),
    )
    return ConversationHeldOutInputCatalog(
        1,
        _FAMILY,
        (),
        (),
        (),
        tuple(cases),
    )


def build_dlg05_unseen_relation_compiler(
        catalog: ConversationHeldOutInputCatalog,
        ) -> ConversationUnseenRelationCompiler:
    """冻结 case 4 的 TRAIN 结构分母并构造 held-out 组合 compiler。

    两条 TRAIN 结构分别共享 held-out predicate 或 construction；Role/filler
    type 槽均已出现，但完整 predicate+construction+slot 组合没有出现。
    这是公开 preflight 分母，不是训练成功或关系事实生成证明。
    """
    if not isinstance(catalog, ConversationHeldOutInputCatalog):
        raise TypeError("unseen relation compiler catalog 类型错误")
    cases = tuple(
        case for case in catalog.cases
        if AXIS_UNSEEN_RELATION in case.axis_keys)
    if len(cases) != 1 or not cases[0].turns:
        raise ValueError("unseen relation case 必须唯一且非空")
    held_out = ConversationRelationStructure.from_target(
        cases[0].turns[0].request.target)
    train = (
        replace(
            held_out,
            construction=structure_concept_identity(
                (_NAMESPACE, 901, 1))),
        replace(
            held_out,
            predicate=concept_identity((_NAMESPACE, 902, 1))),
    )
    inventory = ConversationRelationTrainingInventory(tuple(sorted(
        train, key=lambda item: item.stable_key())))
    return ConversationUnseenRelationCompiler(inventory)


def build_dlg05_unseen_source_compiler(
        catalog: ConversationHeldOutInputCatalog,
        ) -> ConversationUnseenSourceCompiler:
    """为 case 3 冻结同域但不同 exact SourceRef 的 TRAIN 分母。"""
    if not isinstance(catalog, ConversationHeldOutInputCatalog):
        raise TypeError("unseen source compiler catalog 类型错误")
    cases = tuple(
        case for case in catalog.cases
        if AXIS_UNSEEN_SOURCE in case.axis_keys)
    if len(cases) != 1 or not cases[0].turns:
        raise ValueError("unseen source case 必须唯一且非空")
    held_out = cases[0].turns[0].request.source
    train_source = SourceRef(
        held_out.source_kind,
        held_out.source_id + 10_000,
        held_out.document_id + 10_000,
        held_out.owner,
        held_out.versions,
    )
    return ConversationUnseenSourceCompiler(
        ConversationSourceTrainingInventory((train_source,)))


def build_dlg05_preflight_language_compiler(
        catalog: ConversationHeldOutInputCatalog,
        ) -> ConversationQuestionInputCompiler:
    """为 case 1 的两个真实 Representation 输入建立 lexical/结构编译器。"""
    if not isinstance(catalog, ConversationHeldOutInputCatalog):
        raise TypeError("preflight language compiler catalog 类型错误")
    routes = []
    frames = []
    for case_number in (1, 2):
        case = catalog.cases[case_number - 1]
        inputs = tuple(
            turn.language_input for turn in case.turns
            if turn.language_input is not None)
        if len(inputs) != 2:
            raise ValueError("preflight language case 必须有两个 typed input")
        branch = case.turns[0].request.target_branch
        if branch is None:
            raise ValueError("preflight language case 缺少 target branch")
        atoms = tuple(
            language_atom_identity(
                branch, (_NAMESPACE, 303, case_number, index))
            for index in (1, 2)
        )
        full_atoms = atoms
        visible_atoms = []
        for item in inputs:
            if item.provided_positions:
                visible_atoms.append(tuple(
                    full_atoms[position]
                    for position in item.provided_positions))
            elif case_number == 1 and len(visible_atoms) == 1:
                visible_atoms.append((atoms[1], atoms[0]))
            else:
                visible_atoms.append(full_atoms)
        for item, semantic_atoms in zip(inputs, visible_atoms):
            for visible, atom in zip(
                    item.utterance.visible_forms, semantic_atoms):
                route_index = len(routes) + 1
                source_a = SourceRef(
                    _NAMESPACE, 304, route_index,
                    GLOBAL_OWNER_SCOPE, VersionBundle())
                source_b = SourceRef(
                    _NAMESPACE, 305, route_index,
                    GLOBAL_OWNER_SCOPE, VersionBundle())
                routes.append(ConversationLexicalRoute(
                    branch,
                    visible,
                    atom,
                    tuple(sorted((
                        ConversationLexicalEvidence(
                            source_a, (_NAMESPACE, 306, route_index, 1)),
                        ConversationLexicalEvidence(
                            source_b, (_NAMESPACE, 307, route_index, 1)),
                    ), key=lambda value: value.stable_key())),
                ))
        frame_turns = case.turns if case_number == 1 else case.turns[:1]
        for turn in frame_turns:
            request = turn.request
            semantic_atoms = full_atoms
            if case_number == 1 and turn.ordinal == 2:
                semantic_atoms = (atoms[1], atoms[0])
            frames.append(ConversationQuestionRequestFrame(
                turn.language_input.construction,
                branch,
                semantic_atoms,
                request.query_kind,
                request.intent,
                request.goal_kind,
                request.target,
                request.required,
                request.evidence_scope,
                request.response_scope,
                request.trace[:-1],
                request.target_branch,
                request.authorized_candidate_targets,
            ))
    return ConversationQuestionInputCompiler(
        ConversationLexicalNormalizer(tuple(routes)), tuple(frames))


def build_dlg05_typed_preflight_manifest() -> ConversationHeldOutManifest:
    """从六 case typed catalog 构造无标签 manifest。"""
    catalog = build_dlg05_typed_preflight_catalog()
    axes = []
    for case in catalog.cases:
        for axis in case.axis_keys:
            if axis not in axes:
                axes.append(axis)
    return build_conversation_heldout_manifest(catalog, tuple(axes))


def build_dlg05_preflight_evidence_plans(
        catalog: ConversationHeldOutInputCatalog,
        ) -> tuple[ConversationHeldOutPreflightEvidencePlan, ...]:
    """为六 case 建立 label-free typed Evidence 场景包。

    该包只描述 query 可见的 support/refute 候选，不保存 response-act、
    expected surface 或 evaluator label；最终 stance 仍必须由 G-01 policy
    从实际 ``GenerationCandidate`` 计算。
    """
    if not isinstance(catalog, ConversationHeldOutInputCatalog):
        raise TypeError("preflight catalog 类型错误")
    plans = []
    for case in catalog.cases:
        for turn in case.turns:
            if AXIS_MEMORY_MISS in case.axis_keys:
                states = ()
            elif AXIS_CONFLICT in case.axis_keys:
                states = (LogicEvidenceState(True, True),)
            elif AXIS_OMISSION in case.axis_keys:
                states = tuple(
                    LogicEvidenceState(True, False)
                    for _ in turn.request.authorized_candidate_targets
                )
            elif AXIS_EVENT_REFERENCE in case.axis_keys:
                states = (LogicEvidenceState(False, True),)
            else:
                states = (LogicEvidenceState(True, False),)
            plans.append(ConversationHeldOutPreflightEvidencePlan(
                turn.request.stable_key(), states))
    return tuple(plans)


def _evidence_id(
        request_key: tuple[int, ...],
        candidate_key: tuple[int, ...],
        stance: int,
        ordinal: int,
        ) -> int:
    """从本次请求、候选和方向产生正的稳定 Evidence id。"""
    fingerprint = integer_tuple_fingerprint(
        (*request_key, *candidate_key, stance, ordinal),
        domain="dlg05.preflight.evidence.v1",
    )
    result = int.from_bytes(bytes(fingerprint[2:10]), "big")
    result &= (1 << 63) - 1
    return result if result > 0 else 1


@dataclass(frozen=True, slots=True)
class ConversationHeldOutPreflightEvidencePlan:
    """无标签 query 输入的候选四态，仅描述可见 Evidence，不描述答案。"""

    request_key: tuple[int, ...]
    states: tuple[LogicEvidenceState, ...]

    def __post_init__(self) -> None:
        if (not isinstance(self.request_key, tuple) or not self.request_key
                or any(type(value) is not int for value in self.request_key)):
            raise TypeError("preflight evidence plan request_key 非法")
        if (not isinstance(self.states, tuple)
                or any(not isinstance(item, LogicEvidenceState)
                       for item in self.states)):
            raise TypeError("preflight evidence plan states 非法")
        if any(not (item.support or item.refute) for item in self.states):
            raise ValueError("preflight candidate state 必须有 support/refute Evidence")

    def stable_key(self) -> tuple[int, ...]:
        """返回不含 label 的完整 Evidence plan 整数身份。"""
        result = [1, len(self.request_key), *self.request_key, len(self.states)]
        for state in self.states:
            result.extend(state.stable_key())
        return tuple(result)


class ConversationHeldOutPreflightQuestionExecutor:
    """把 label-free typed Evidence plan 投影为真实 QuestionExecutionResult。"""

    def __init__(
            self,
            route: ObjectIdentity,
            executed_reason: ObjectIdentity,
            plans: tuple[ConversationHeldOutPreflightEvidencePlan, ...],
            ) -> None:
        """绑定唯一 query route 和只读候选计划，不保存 evaluator label。"""
        if (not isinstance(route, ObjectIdentity)
                or not isinstance(executed_reason, ObjectIdentity)):
            raise TypeError("preflight route/reason identity 类型错误")
        if (route.object_kind != OBJECT_MINIMAL_INSTRUCTION
                or executed_reason.object_kind != OBJECT_MINIMAL_INSTRUCTION):
            raise ValueError("preflight route/reason 必须是 MinimalInstruction")
        if (not isinstance(plans, tuple)
                or any(not isinstance(item, ConversationHeldOutPreflightEvidencePlan)
                       for item in plans)):
            raise TypeError("preflight plans 类型错误")
        by_key = {}
        for item in plans:
            previous = by_key.get(item.request_key)
            if previous is not None and previous != item:
                raise ValueError("preflight 同 request_key 的 Evidence plan 漂移")
            by_key[item.request_key] = item
        self.route = route
        self.executed_reason = executed_reason
        self._plans = by_key

    def execute(self, query: QuestionQuery) -> QuestionExecutionResult:
        """按 request authorized targets 形成候选，stance 留给 G-01 policy。"""
        if not isinstance(query, QuestionQuery):
            raise TypeError("preflight query 类型错误")
        if query.route != self.route:
            raise ValueError("preflight query route 漂移")
        request = query.request
        plan = self._plans.get(request.stable_key())
        if plan is None:
            raise ValueError("preflight 没有当前 request 的 Evidence plan")
        targets = request.authorized_candidate_targets or (request.target,)
        if len(plan.states) != len(targets):
            if plan.states:
                raise ValueError("preflight Evidence 数量与 authorized targets 不一致")
            targets = ()
        candidates = []
        for ordinal, (target, state) in enumerate(
                zip(targets, plan.states), start=1):
            hypothesis = HypothesisKey(
                (_NAMESPACE, 40),
                target.template.stable_key(),
                (_NAMESPACE, 41, *integer_tuple_fingerprint(
                    request.stable_key(),
                    domain="dlg05.preflight.competition.v1")),
                request.evidence_scope,
                request.source,
            )
            evidence = []
            for stance in (EVIDENCE_SUPPORT, EVIDENCE_REFUTE):
                if (stance == EVIDENCE_SUPPORT and not state.support) or (
                        stance == EVIDENCE_REFUTE and not state.refute):
                    continue
                evidence.append(EvidenceRecord(
                    _evidence_id(request.stable_key(), target.stable_key(),
                                 stance, ordinal),
                    hypothesis,
                    stance,
                    (_NAMESPACE, 42, stance),
                    request.source,
                    ordinal,
                    (ordinal, stance),
                ))
            candidates.append(GenerationCandidate(
                target,
                state,
                request.source,
                request.response_scope,
                tuple(evidence),
            ))
        return QuestionExecutionResult(
            query,
            self.executed_reason,
            tuple(candidates),
            (1, *integer_tuple_fingerprint(
                query.stable_key(), domain="dlg05.preflight.query.v1")),
        )


__all__ = [
    "ConversationHeldOutAxisInputAudit",
    "ConversationHeldOutPreflightEvidencePlan",
    "ConversationHeldOutPreflightQuestionExecutor",
    "audit_dlg05_preflight_axis_inputs",
    "build_dlg05_preflight_evidence_plans",
    "build_dlg05_preflight_language_compiler",
    "build_dlg05_typed_preflight_catalog",
    "build_dlg05_typed_preflight_manifest",
    "build_dlg05_unseen_relation_compiler",
    "build_dlg05_unseen_source_compiler",
]
