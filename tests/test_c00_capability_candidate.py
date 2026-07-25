"""C-00 provisional Capability candidate 的形成与反召回测试。"""
from __future__ import annotations

from dataclasses import replace

import pytest

from pure_integer_ai.cognition.shared.capability_candidate import (
    CapabilityCandidateProposal,
    CapabilityExample,
    CapabilityFormationInput,
    CapabilityFormationRequest,
    CapabilityFormationRuntime,
    CapabilityStatusProtocol,
)
from pure_integer_ai.cognition.shared.identity import (
    concept_identity,
    minimal_instruction_identity,
)
from pure_integer_ai.cognition.shared.memory_event import MemoryLinkedRef
from pure_integer_ai.cognition.shared.semantic_object import (
    binder_identity,
    proposition_identity,
    variable_identity,
)
from pure_integer_ai.numeric.symbol_domain import (
    OPCODE_ADD,
    OPCODE_HALT,
    OPCODE_LOAD,
)
from pure_integer_ai.training.formal_artifact_vm import encode_vm_program
from pure_integer_ai.vm.graph_compile import Instruction

from test_a06_artifact_binding_runtime import _runtime_case, _stop_work_memory
from test_s06_formal_artifact import _artifact


class _Former:
    """测试用 former，只提出调用方构造的 proposal，不参与 runtime 核验。"""

    def __init__(self, proposal):
        self.proposal = proposal
        self.calls = []

    def propose(self, formation_input):
        """记录脱敏形成输入并返回预设 proposal。"""
        self.calls.append(formation_input)
        return self.proposal


def _successful_example(left, right, total):
    """执行真实 A-06 双参数调用并返回脱离 WorkMemory 生命周期的不可变 example。"""
    backend, context, _, runtime, request = _runtime_case(
        left_payload=(left, 1),
        right_payload=(right, 1),
        expected_payload=(total, 1),
    )
    try:
        run = runtime.run(request)
        assert run.succeeded is True
        return CapabilityExample(run)
    finally:
        _stop_work_memory(context)
        backend.close()


def _status_protocol():
    """注入互异 provisional/verified/rejected 最小指令。"""
    return CapabilityStatusProtocol(
        minimal_instruction_identity((20700, 1)),
        minimal_instruction_identity((20700, 2)),
        minimal_instruction_identity((20700, 3)),
    )


def _proposal(examples):
    """构造类型不漂移、变量和执行绑定均不同的新 program proposal。"""
    first = examples[0].run.invocation.definition
    source = first.program.source
    bindings = ((201,), (202,))
    program = _artifact(
        source,
        None,
        first.program.artifact_kind,
        first.program.schema,
        20701,
        encode_vm_program((
            Instruction(OPCODE_LOAD, bindings[0]),
            Instruction(OPCODE_LOAD, bindings[1]),
            Instruction(OPCODE_ADD),
            Instruction(OPCODE_HALT),
        )),
    )
    binder = binder_identity(source, (20706, 1))
    parameters = tuple(
        replace(
            parameter,
            variable=variable_identity(
                binder,
                (20706, index + 2),
                parameter.schema.value_type,
            ),
            executor_binding=bindings[index],
        )
        for index, parameter in enumerate(first.parameters)
    )
    definition = replace(first, program=program, parameters=parameters)
    return CapabilityCandidateProposal(
        MemoryLinkedRef.object(concept_identity((20702, 1))),
        definition,
        source,
        (proposition_identity(source, (20703, 1)),),
        tuple(item.content_ref() for item in examples),
        minimal_instruction_identity((20704, 1)),
        (20705, 1, len(examples)),
    )


def test_c00_forms_provisional_candidate_from_distinct_successful_examples():
    """两个不同参数的真实 A-06 正例可形成新 program 的 provisional candidate。"""
    examples = (
        _successful_example(2, 3, 5),
        _successful_example(4, 6, 10),
    )
    proposal = _proposal(examples)
    former = _Former(proposal)
    request = CapabilityFormationRequest(
        examples,
        (),
        _status_protocol(),
    )

    candidate = CapabilityFormationRuntime(former).form(request)

    assert former.calls == [request.formation_input()]
    assert isinstance(former.calls[0], CapabilityFormationInput)
    assert not hasattr(former.calls[0], "examples")
    assert all(not hasattr(item, "run")
               and not hasattr(item, "definition")
               and not hasattr(item, "program")
               and not hasattr(item, "expected")
               for item in former.calls[0].demonstrations)
    assert candidate.state == request.status_protocol.provisional
    assert candidate.proposal.definition.program.identity not in {
        item.run.invocation.definition.program.identity for item in examples}
    assert candidate.proposal.example_refs == request.example_refs()
    assert candidate.proposal.contract_key()
    assert candidate.stable_key()


def test_c00_rejects_example_or_recalled_program_as_induction():
    """example 已有 program 和召回 program 都不得换标签冒充 induction。"""
    examples = (
        _successful_example(1, 2, 3),
        _successful_example(5, 7, 12),
    )
    first_program = examples[0].run.invocation.definition.program
    base = _proposal(examples)
    copied = replace(
        base,
        definition=replace(base.definition, program=first_program),
    )
    with pytest.raises(ValueError, match="example 中已有 program"):
        CapabilityFormationRuntime(_Former(copied)).form(
            CapabilityFormationRequest(examples, (), _status_protocol()))

    recalled_program = _artifact(
        base.definition.program.source,
        None,
        base.definition.program.artifact_kind,
        base.definition.program.schema,
        20707,
        base.definition.program.payload,
    )
    recalled = replace(base.definition, program=recalled_program)
    with pytest.raises(ValueError, match="已召回 program payload"):
        CapabilityFormationRuntime(_Former(base)).form(
            CapabilityFormationRequest(
                examples, (recalled,), _status_protocol()))


def test_c00_rejects_relabelled_payload_and_execution_contract():
    """只换 program 身份、载荷或执行绑定契约都不能伪造归纳。"""
    examples = (
        _successful_example(2, 4, 6),
        _successful_example(7, 8, 15),
    )
    first = examples[0].run.invocation.definition
    base = _proposal(examples)
    copied_program = _artifact(
        first.program.source,
        None,
        first.program.artifact_kind,
        first.program.schema,
        20708,
        first.program.payload,
    )
    copied_payload = replace(
        base,
        definition=replace(first, program=copied_program),
    )
    with pytest.raises(ValueError, match="example program payload"):
        CapabilityFormationRuntime(_Former(copied_payload)).form(
            CapabilityFormationRequest(examples, (), _status_protocol()))

    changed_program = _artifact(
        first.program.source,
        None,
        first.program.artifact_kind,
        first.program.schema,
        20709,
        encode_vm_program((
            Instruction(OPCODE_LOAD, first.parameters[0].executor_binding),
            Instruction(OPCODE_LOAD, first.parameters[1].executor_binding),
            Instruction(OPCODE_ADD),
            Instruction(OPCODE_HALT),
            Instruction(OPCODE_HALT),
        )),
    )
    copied_contract = replace(
        base,
        definition=replace(first, program=changed_program),
    )
    with pytest.raises(ValueError, match="example program 执行契约"):
        CapabilityFormationRuntime(_Former(copied_contract)).form(
            CapabilityFormationRequest(examples, (), _status_protocol()))


def test_c00_rejects_duplicate_arguments_and_missing_example_reference():
    """重复参数不能伪造多例，former 也不能漏掉不利 example。"""
    first = _successful_example(2, 8, 10)
    with pytest.raises(ValueError, match="不同参数"):
        CapabilityFormationRequest(
            (first, first), (), _status_protocol())

    examples = (first, _successful_example(3, 9, 12))
    proposal = replace(
        _proposal(examples),
        example_refs=(examples[0].content_ref(), examples[0].content_ref()),
    )
    with pytest.raises(ValueError, match="逐项引用"):
        CapabilityFormationRuntime(_Former(proposal)).form(
            CapabilityFormationRequest(examples, (), _status_protocol()))


def test_c00_rejects_contract_drift_and_failed_example():
    """参数 schema 漂移或形式失败都不能进入 provisional candidate。"""
    examples = (
        _successful_example(2, 3, 5),
        _successful_example(8, 9, 17),
    )
    proposal = _proposal(examples)
    first_parameter = proposal.definition.parameters[0]
    drift_type = concept_identity((20710, 1))
    drift_schema = replace(first_parameter.schema, value_type=drift_type)
    drift_parameter = replace(
        first_parameter,
        variable=variable_identity(
            binder_identity(proposal.source, (20710, 2)),
            (20710, 3),
            drift_type,
        ),
        schema=drift_schema,
    )
    drift = replace(
        proposal,
        definition=replace(
            proposal.definition,
            parameters=(
                drift_parameter,
                proposal.definition.parameters[1],
            ),
        ),
    )
    with pytest.raises(ValueError, match="类型契约漂移"):
        CapabilityFormationRuntime(_Former(drift)).form(
            CapabilityFormationRequest(examples, (), _status_protocol()))

    backend, context, _, runtime, request = _runtime_case(
        expected_payload=(99, 1))
    try:
        failed = runtime.run(request)
        assert failed.succeeded is False
        with pytest.raises(ValueError, match="成功 A-06"):
            CapabilityExample(failed)
    finally:
        _stop_work_memory(context)
        backend.close()
