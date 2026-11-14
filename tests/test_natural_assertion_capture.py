"""普通陈述捕获仍须原输入、图事实、投影、生成和同次交付完整。"""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from pure_integer_ai.cognition.shared.query_state import EvidenceEntry
from pure_integer_ai.experiments.batch_case_provenance import capture_factual_case_provenance
from pure_integer_ai.experiments.natural_assertion_capture import delivered_assertion_capture_witness
from pure_integer_ai.experiments.symmetric_input_order import symmetric_input_order_witnesses
from tests.test_batch_case_provenance import _call
from tests.test_symmetric_input_order import _fixture


def _assertion():
    """真实投影配合完整规范QueryState记录，不生成语言或虚构operation。"""
    structure, facts = _fixture()
    witness, = symmetric_input_order_witnesses(structure, facts)
    relation = witness.support
    query, delivery = _call()
    table = query['integer_key_table']
    def intern(key):
        """规范整数引用仅供构造测试记录。"""
        table.append(list(key))
        return len(table)
    query['query_ref'] = intern(structure.source_ref)
    query['input_structure'] = {'source_ref': {'integer_key_ref': query['query_ref']},
        'token_values': {'integer_key_ref': intern(structure.token_values)},
        'stable_key': {'integer_key_ref': intern(structure.stable_key())}}
    query['state_key']['query_ref'] = query['query_ref']
    query['state_key']['scalars'][11] = intern(relation.proposition_key)
    query['state_key']['bindings'] = [intern(item.stable_key()) for item in witness.original_bindings()]
    evidence = (*witness.original_evidence(), witness.query_evidence(),
        EvidenceEntry(relation.source_ref, relation.proposition_key, evidence_key=(800,)))
    query['state_key']['evidence'] = [intern(item.stable_key()) for item in evidence]
    query['operation_keys'], query['operation_contracts'] = [], []
    query['generation']['kind'] = 1
    query['response_plan']['claim_refs'] = [list(relation.proposition_key)]
    query['response_plan']['evidence_refs'] = [list(item.stable_key()) for item in evidence]
    query['response_plan']['claim_evidence_refs'] = [{'claim': list(relation.proposition_key),
        'evidence': [list(evidence[-1].stable_key())]}]
    delivery['evidence_ref_count'] = len(evidence)
    return structure, facts, query, delivery, relation.proposition_key


def test_plain_assertion_gets_same_call_capture_with_explicit_empty_operation():
    structure, facts, query, delivery, claim = _assertion()
    before = deepcopy(query)
    actual, witness = delivered_assertion_capture_witness(structure, facts, query)
    assert actual == claim
    captured = capture_factual_case_provenance(query_result=query, query_values=structure.token_values,
        expected_claim=claim, expected_operation=None, assertion_input=structure, core_fact_index=facts,
        input_observation=(103,), input_source=(104,), delivery_summary=delivery)
    assert captured['assertion_input_witness'] == list(witness)
    assert captured['expected_operation'] == []
    assert captured['status'] == 'CAPTURED_QUERY_PLAN_AND_DELIVERY_REFS'
    assert query == before


@pytest.mark.parametrize('drift', ('operation', 'contract', 'input', 'claim', 'primary',
    'generation_kind', 'positive_missing', 'binding_missing', 'projection_missing', 'core_source', 'core_missing'))
def test_assertion_capture_rejects_unproven_operation_claim_input_or_original_evidence(drift):
    structure, facts, query, delivery, claim = _assertion()
    if drift == 'operation': query['operation_keys'] = [[999]]
    elif drift == 'contract': query['operation_contracts'] = [[[999], [91672, 3]]]
    elif drift == 'input': query['input_structure']['stable_key']['integer_key_ref'] = 1
    elif drift == 'claim': query['response_plan']['claim_refs'] = [list(claim), list(claim)]
    elif drift == 'primary': query['state_key']['scalars'][11] = 1
    elif drift == 'generation_kind': query['generation']['kind'] = 2
    elif drift == 'positive_missing': query['state_key']['evidence'] = query['state_key']['evidence'][:-1]
    elif drift == 'binding_missing': query['state_key']['bindings'] = query['state_key']['bindings'][1:]
    elif drift == 'projection_missing': query['state_key']['evidence'] = query['state_key']['evidence'][1:]
    elif drift == 'core_source': facts[claim].source_hash += 1
    else: facts.pop(claim)
    with pytest.raises(ValueError): delivered_assertion_capture_witness(structure, facts, query)


def test_operation_capture_cannot_request_unverified_empty_operation():
    query, delivery = _call()
    with pytest.raises(ValueError):
        capture_factual_case_provenance(query_result=query, query_values=(65,), expected_claim=(7,),
            expected_operation=None, input_observation=(103,), input_source=(104,), delivery_summary=delivery)


def test_plain_capture_cannot_drop_another_scope_contract():
    structure, facts, query, _, _ = _assertion()
    fields = {name: getattr(structure, name) for name in structure.__dataclass_fields__}
    fields['operation_keys'] = ((999,),)
    with pytest.raises(ValueError): delivered_assertion_capture_witness(SimpleNamespace(**fields), facts, query)
