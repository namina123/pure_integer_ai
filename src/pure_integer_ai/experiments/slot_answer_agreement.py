"""不同合法关系解释的本次WH答案一致性，不把关系判为等价。"""
from pure_integer_ai.cognition.understanding.query_input_structure import STRUCTURE_OPEN
from pure_integer_ai.experiments.trained_question_open_frames import question_observed_role_contract


def select_convergent_slot_answer(validated):
    """只从已验证完整Core槽位中证明对象答案唯一，并选择规范物理代表。"""
    records = tuple(validated)
    if len(records) < 2 or any(parts is None for _, parts in records):
        return ()
    if len({parts[0].predicate_key for _, parts in records}) < 2:
        return ()
    contracts = set()
    for binding, parts in records:
        frame, relation, fact, answer, dynamic = parts
        fillers = {item.role.stable_key(): item.filler.stable_key() for item in fact.bindings}
        if (dynamic or frame.is_reference_slot or not frame.is_slot_question
                or len(fillers) != len(fact.bindings)
                or binding.frame_key != frame.stable_key()
                or binding.operation_key != frame.operation_key
                or binding.relation_key != relation.stable_key()
                or binding.answer_role_key != frame.answer_role_key
                or set(fillers) != {*frame.roles, frame.answer_role_key}
                or len(binding.slot_span_refs) != len(frame.roles)
                or not question_observed_role_contract(frame, tuple(fillers))
                or fillers.get(frame.answer_role_key) != answer
                or fact.predicate.stable_key() != frame.predicate_key
                or relation.predicate_key != frame.predicate_key
                or relation.proposition_key != fact.proposition.stable_key()
                or relation.source_ref != (fact.source_hash,)
                or relation.state != STRUCTURE_OPEN or relation.conflict_count
                or relation.role_coverage != len(frame.roles)
                or relation.required_count != len(frame.roles) + 2
                or relation.support_count != len(frame.roles) + 1
                or relation.predicate_coverage != 1 or relation.required_open != 1):
            return ()
        # 直接比较完整对象身份，不能以同名、来源摘要或跨谓词Role名称替代。
        contracts.add((binding.operation_key, frame.gaps, frame.wh_values, frame.slot_mode,
            tuple(zip(binding.slot_span_refs, (fillers[role] for role in frame.roles))), answer))
    if len(contracts) != 1:
        return ()
    def proof_key(record):
        """先消费完整问式自身的谓词证明，同层再按完整物理身份规范排序。"""
        binding, parts = record
        predicate_spans = set(parts[1].span_refs) - set(binding.slot_span_refs)
        return (int(bool(predicate_spans)), binding.stable_key())

    return (min(records, key=proof_key)[0],)
