"""Reader provenance must not be added to a sparse operation execution."""
from types import SimpleNamespace

import pytest

from pure_integer_ai.cognition.shared.integer_operation_execution import UNKNOWN, execute_integer_operation
from pure_integer_ai.cognition.shared.query_state import BindingEntry, EvidenceEntry
from pure_integer_ai.experiments import operation_semantic_trace_audit as audit
from pure_integer_ai.experiments.operation_program_query import OPERATION_EXECUTION_ROLE
from test_integer_operation_program import context, _witness


def _records(context,monkeypatch):
    a,b,_c,_d,_first,_second,facts,program=context
    witness=_witness(program,facts)
    query=(92812,430)
    execution=execute_integer_operation(witness,query,((1,a),(2,b)),())
    assert execution.status==UNKNOWN and not execution.matches and not execution.outputs
    binding=BindingEntry(OPERATION_EXECUTION_ROLE,execution.stable_key(),space=1,scope_key=query)
    permission=EvidenceEntry(witness.source.stable_key(),program.operation_key,space=1,polarity=3,
        evidence_key=(92230,*query),scope_key=query,payload_key=execution.stable_key())
    prepared=SimpleNamespace(source_ref=query,integer_trace=lambda:dict(reader_identity=1))
    value=dict(query_ref=query,input_structure=prepared.integer_trace(),operation_keys=[program.operation_key],
        state_key=dict(query_ref=query,bindings=[binding.stable_key()],evidence=[permission.stable_key()],scalars=[0]*16),
        termination=8,response_plan=None,generation=None)
    trace=SimpleNamespace(resolve_ref=lambda ref:tuple(ref),operation_contracts=lambda:((program.operation_key,(1,)),))
    monkeypatch.setattr(audit,'GraphQueryTraceEvidence',lambda _value:trace)
    monkeypatch.setattr(audit,'_decode_input',lambda encoded,_trace:encoded)
    # Isolate the two validators: the real integer execution verifier checks
    # the empty sparse context; the input boundary requires its reader facts.
    monkeypatch.setattr(audit,'operation_execution_input_matches',
        lambda _execution,_prepared,inputs,**_kwargs:inputs==facts)
    return value,witness,prepared,facts,execution


def test_unknown_execution_preserves_empty_context_while_checking_full_reader_provenance(context,monkeypatch):
    value,witness,prepared,readers,execution=_records(context,monkeypatch)
    with pytest.raises(ValueError,match='complete current input variables'):
        audit.read_consumed_operation_selection(value,(),witness,prepared)
    restored=audit.read_consumed_operation_execution(value,(),witness,prepared,input_facts=readers)
    assert restored==execution and not restored.fact_proofs and not restored.outputs


@pytest.mark.parametrize('drift',('missing_reader','extra_execution_facts','permission'))
def test_reader_context_cannot_supply_execution_facts_or_missing_permission(context,monkeypatch,drift):
    value,witness,prepared,readers,_execution=_records(context,monkeypatch)
    facts=()
    if drift=='missing_reader':readers=readers[:-1]
    elif drift=='extra_execution_facts':facts=readers
    else:value['state_key']['evidence']=[]
    with pytest.raises(ValueError):
        audit.read_consumed_operation_execution(value,facts,witness,prepared,input_facts=readers)
