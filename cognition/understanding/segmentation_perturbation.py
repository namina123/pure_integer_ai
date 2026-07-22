"""把已物化分词竞争候选适配到 H-02A 通用扰动 trace。

本模块只证明两个一等 Hypothesis 属于同一竞争边界并已在 L-04 图中物化，随后记录
候选边界替换。它不判断哪个边界正确，也不把 surface 差异当成语义负例；三态裁决
仍由调用方 verifier 完成。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.hypothesis import HypothesisKey
from pure_integer_ai.cognition.shared.perturbation import (
    PerturbationTrace,
    build_replacement_trace,
)
from pure_integer_ai.cognition.understanding.segmentation_span import (
    SegmentationSpanCandidate,
    SegmentationSpanMaterializer,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


def _protocol_key(value, *, where: str) -> tuple[int, ...]:
    """校验由课程或上层结构任务注入的开放整数变换键。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{where} 必须是非空整数 tuple")
    assert_int(*value, _where=where)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{where} 必须使用严格整数")
    return value


@dataclass(frozen=True)
class SegmentationPerturbationProtocol:
    """注入“已有分词候选边界替换”的变换身份。"""

    boundary_replacement_transform_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验边界替换键可作为稳定开放整数协议身份。"""
        _protocol_key(
            self.boundary_replacement_transform_key,
            where=(
                "SegmentationPerturbationProtocol."
                "boundary_replacement_transform_key"
            ),
        )


def _boundary_key(
        materializer: SegmentationSpanMaterializer,
        candidate: SegmentationSpanCandidate,
        ) -> tuple[int, ...]:
    """从权威 Span 成员恢复候选完整边界，不读取或编码 surface。"""
    values: list[int] = [len(candidate.parts)]
    for part in candidate.parts:
        members = materializer.spans.members_of(part)
        values.append(len(members))
        for start, end in members:
            values.extend((start, end))
    return tuple(values)


class SegmentationPerturbationAdapter:
    """只在同一已物化分词竞争组内构造候选替换 trace。"""

    def __init__(
            self, materializer: SegmentationSpanMaterializer,
            protocol: SegmentationPerturbationProtocol,
            ) -> None:
        """绑定 L-04 物化器与调用方变换协议，不拥有或另造 H-00 ledger。"""
        if not isinstance(materializer, SegmentationSpanMaterializer):
            raise TypeError("materializer 必须是 SegmentationSpanMaterializer")
        if not isinstance(protocol, SegmentationPerturbationProtocol):
            raise TypeError("protocol 必须是 SegmentationPerturbationProtocol")
        self.materializer = materializer
        self.protocol = protocol

    def build_boundary_replacement(
            self, original: HypothesisKey,
            transformed: HypothesisKey,
            ) -> PerturbationTrace:
        """用两个已物化同组候选的完整身份和 Span 边界构造替换 trace。"""
        if not isinstance(original, HypothesisKey):
            raise TypeError("original 必须是 HypothesisKey")
        if not isinstance(transformed, HypothesisKey):
            raise TypeError("transformed 必须是 HypothesisKey")
        if original == transformed:
            raise ValueError("边界替换必须使用两个不同 Hypothesis")
        if (
                original.hypothesis_kind != transformed.hypothesis_kind
                or original.competition_key != transformed.competition_key
                or original.scope != transformed.scope
                or original.observation != transformed.observation):
            raise ValueError("边界替换的两个 Hypothesis 必须属于同一竞争组")

        original_candidate = self.materializer.candidate(original)
        transformed_candidate = self.materializer.candidate(transformed)
        if original_candidate is None or transformed_candidate is None:
            raise LookupError("边界替换前必须先物化两个分词 Span 候选")
        original_boundary = _boundary_key(
            self.materializer, original_candidate)
        transformed_boundary = _boundary_key(
            self.materializer, transformed_candidate)
        if original_boundary == transformed_boundary:
            raise ValueError("分词边界替换必须真实改变 Span 成员边界")
        return build_replacement_trace(
            (original.object_identity(),),
            (transformed.object_identity(),),
            output_to_input=(-1,),
            transform_key=(
                self.protocol.boundary_replacement_transform_key),
            source=original.observation,
            scope=original.scope,
            metadata_keys=(
                original_boundary,
                transformed_boundary,
            ),
        )


__all__ = [
    "SegmentationPerturbationAdapter",
    "SegmentationPerturbationProtocol",
]
