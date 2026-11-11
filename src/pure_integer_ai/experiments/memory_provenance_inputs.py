"""交付原输入是处理来源；只显式依赖的假设才重新成为语义成员。"""
from pure_integer_ai.cognition.shared.memory_event import (
    MEMORY_EVENT_OBSERVATION,MEMORY_OBJECT_HYPOTHESIS,MEMORY_OBJECT_OBSERVATION,
    MemoryObjectRef,ObservationPayload,
)
from pure_integer_ai.cognition.shared.query_state import BindingEntry,ROOT_EXPANDED,SPACE_MEMORY
from pure_integer_ai.experiments.trained_response_context import response_context_input_observations


def consumed_response_contexts(contexts,trace,role_evidence_keys):
    """由已消费的物理O或角色确定需审来源的context，候选存在本身不授读取权。"""
    observations={hop.proposition_key for hop in trace.hops()
        if hop.space==SPACE_MEMORY and hop.predicate_key==(MEMORY_EVENT_OBSERVATION,)}
    return tuple(context for context in contexts if (
        context.observation.stable_key() in observations
        or any(ref.stable_key() in observations for ref in response_context_input_observations((context,)))
        or any(role.stable_key() in role_evidence_keys for role in context.role_occurrences)))


def retain_context_input_dependencies(inputs,contexts):
    """保留完整物理Observation邻接，仅明确被context引用的H及其全部E进入查询。

    此处发生在QueryState构造前；既有内容/当前语境的H/E由调用方完整合并。
    Observation的声明邻接不改写，桥接只沿实际准入的Hypothesis对象展开。
    """
    from pure_integer_ai.experiments.trained_dialogue_memory_graph import MemoryActiveStructures
    required={ref.stable_key() for context in contexts for ref in context.memory_refs()
              if ref.object_kind==MEMORY_OBJECT_HYPOTHESIS}
    hypotheses=tuple(item for item in inputs.hypotheses if item.hypothesis_key in required)
    keys={item.hypothesis_key for item in hypotheses}
    evidence=tuple(item for item in inputs.evidence if item.hypothesis_key in keys)
    if any(set(item.evidence_keys)!={entry.evidence_key for entry in evidence
            if entry.hypothesis_key==item.hypothesis_key} for item in hypotheses):
        raise ValueError('delivery input dependency lacks its complete Evidence')
    return MemoryActiveStructures(inputs.observations,hypotheses,evidence,inputs.match_routes)


def verify_context_input_observations(contexts,*,roots,bindings,declaration,observation,source_hash,error):
    """独立context授权来源O，核对物理声明、完整展开根/绑定并返回实际O hop全集。"""
    expected=set()
    for ref in response_context_input_observations(contexts):
        payload_row=declaration(ref)
        payload=payload_row.event.payload
        if not isinstance(payload,ObservationPayload) or observation(payload.source)!=ref:
            raise error('delivery context input Observation differs from its original manifest')
        source=payload.source
        scope=payload_row.event.scope.stable_key()
        if (ref.owner!=source.owner or ref.versions!=source.versions
                or not any(root.owner_space==SPACE_MEMORY and root.status==ROOT_EXPANDED
                    and root.root_key==ref.stable_key()
                    and root.source_ref==(source_hash(source),*source.stable_key())
                    and root.scope_key==scope for root in roots)
                or BindingEntry((MEMORY_OBJECT_OBSERVATION,),ref.stable_key(),SPACE_MEMORY,scope).stable_key()
                    not in bindings):
            raise error('delivery input provenance lacks its exact current Observation root/binding')
        expected.add((ref.stable_key(),(MEMORY_EVENT_OBSERVATION,),source_hash(source)))
    return expected
