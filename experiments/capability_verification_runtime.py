"""C-01 以 candidate definition 执行独立 held-out 的运行期接线。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.capability_candidate import (
    CapabilityCandidate,
)
from pure_integer_ai.cognition.shared.capability_verification import (
    CapabilityHeldOutCase,
    CapabilityVerificationReport,
    capability_argument_content_key,
    capability_verification_state,
)
from pure_integer_ai.cognition.shared.formal_artifact import (
    ArtifactArgument,
    ArtifactInvocation,
)
from pure_integer_ai.cognition.shared.formal_artifact_bridge import (
    FormalArtifactBridge,
    is_execution_result_artifact_of,
)


class CapabilityVerificationRuntime:
    """核验未见参数和独立规格后调用 S-06 bridge 形成 C-01 report。"""

    def __init__(self, bridge: FormalArtifactBridge) -> None:
        """绑定实际 executor/verifier 所在的现有形式域桥。"""
        if not isinstance(bridge, FormalArtifactBridge):
            raise TypeError("bridge 必须是 FormalArtifactBridge")
        self.bridge = bridge

    def verify(
            self,
            candidate: CapabilityCandidate,
            held_out: CapabilityHeldOutCase,
            ) -> CapabilityVerificationReport:
        """执行一个独立 held-out case，不写 Memory、Use 或 candidate 本体。"""
        if not isinstance(candidate, CapabilityCandidate):
            raise TypeError("candidate 必须是 CapabilityCandidate")
        if not isinstance(held_out, CapabilityHeldOutCase):
            raise TypeError("held_out 必须是 CapabilityHeldOutCase")
        if candidate.state != candidate.status_protocol.provisional:
            raise ValueError("C-01 只接受 provisional candidate")

        definition = candidate.proposal.definition
        example_sources = {
            item.run.invocation.source for item in candidate.examples}
        if (held_out.source == candidate.proposal.source
                or held_out.source in example_sources):
            raise ValueError("held-out source 必须与 candidate examples 分离")
        if held_out.specification_authority in (
                definition.executor, definition.verifier):
            raise ValueError("held-out specification authority 必须独立")
        if len(held_out.arguments) != len(definition.parameters):
            raise ValueError("held-out 参数数量未覆盖 candidate definition")
        for parameter, value in zip(definition.parameters, held_out.arguments):
            if value.schema != parameter.schema:
                raise ValueError("held-out 参数 schema 必须与 candidate 精确一致")
        if (held_out.expected.artifact_kind != definition.result_kind
                or held_out.expected.schema != definition.result_schema):
            raise ValueError("held-out expected 与 candidate 结果契约不一致")
        if is_execution_result_artifact_of(held_out.expected, definition):
            raise ValueError("candidate result Artifact 不能回流为 held-out expected")

        held_out_key = held_out.argument_content_key(definition)
        for example in candidate.examples:
            example_values = tuple(
                argument.value
                for argument in example.run.invocation.arguments)
            example_key = capability_argument_content_key(
                example.run.invocation.definition, example_values)
            if held_out_key == example_key:
                raise ValueError("held-out 参数重放了 C-00 example")

        candidate_before = candidate.stable_key()
        held_out_before = held_out.stable_key()
        invocation = ArtifactInvocation(
            held_out.proposition,
            definition,
            tuple(
                ArtifactArgument(parameter.variable, value)
                for parameter, value in zip(
                    definition.parameters, held_out.arguments)
            ),
            held_out.source,
            held_out.scope,
            held_out.case_key,
            held_out.expected,
        )
        result = self.bridge.invoke(invocation)
        if result.invocation != invocation:
            raise ValueError("held-out bridge 返回了其他 invocation 的结果")
        if (result.value is not None
                and result.value.identity == held_out.expected.identity):
            raise ValueError("candidate result Artifact 不能回流为 held-out expected")
        if candidate.stable_key() != candidate_before:
            raise ValueError("held-out 执行改写了 candidate")
        if held_out.stable_key() != held_out_before:
            raise ValueError("held-out 执行改写了预先冻结的 expected/spec")
        state = capability_verification_state(candidate, result)
        return CapabilityVerificationReport(
            candidate,
            held_out,
            invocation,
            result,
            candidate.state,
            state,
        )


__all__ = ["CapabilityVerificationRuntime"]
