"""将同次查询的原始 Memory O/H/E 分账交给 G-00，不伪装成已解析 M-07。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.generation_observed_surface import ObservedGenerationSpan
from pure_integer_ai.cognition.shared.identity import SourceRef
from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
from pure_integer_ai.cognition.shared.memory_event import (
    MEMORY_OBJECT_HYPOTHESIS, MEMORY_OBJECT_OBSERVATION, MemoryObjectRef,
)
from pure_integer_ai.cognition.shared.query_state import (
    SPACE_CORE, SPACE_DIALOGUE, SPACE_MEMORY, EvidenceEntry,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.typed_binding import BoundProposition


def _pack(key: tuple[int, ...]) -> tuple[int, ...]:
    """保留开放整数记录的完整长度边界。"""
    if type(key) is not tuple or not key or any(type(value) is not int for value in key):
        raise ValueError("观察生成证据必须为非空严格整数记录")
    return len(key), *key


@dataclass(frozen=True, slots=True)
class ObservedGenerationEvidence:
    """真实原始观察对运行期角色替换视图的证据；不创建 Core 命题或 M-07 决议。"""

    target: BoundProposition
    source: SourceRef
    scope: ScopeIdentity
    observation: MemoryObjectRef
    hypothesis: MemoryObjectRef
    hypothesis_kind: tuple[int, ...]
    competition_key: tuple[int, ...]
    spans: tuple[ObservedGenerationSpan, ...]
    evidence: tuple[EvidenceEntry, ...]
    binding_trace: tuple[int, ...]

    def __post_init__(self) -> None:
        """要求三图证据与原 O/H、source/scope、完整角色替换逐项对应。"""
        if not isinstance(self.target, BoundProposition) or not isinstance(self.source, SourceRef):
            raise TypeError("观察生成目标或来源类型错误")
        if (not isinstance(self.scope, ScopeIdentity) or self.scope.owner != self.source.owner
                or self.scope.versions != self.source.versions):
            raise ValueError("观察生成来源与作用域不一致")
        for ref, kind in ((self.observation, MEMORY_OBJECT_OBSERVATION),
                          (self.hypothesis, MEMORY_OBJECT_HYPOTHESIS)):
            if (not isinstance(ref, MemoryObjectRef) or ref.object_kind != kind
                    or ref.owner != self.source.owner or ref.versions != self.source.versions):
                raise ValueError("观察生成必须持有实际来源的 O/H 引用")
        if (type(self.spans) is not tuple or not self.spans
                or any(not isinstance(item, ObservedGenerationSpan) or item.source != self.source
                       or item.scope != self.scope or item.observation != self.observation
                       or item.hypothesis != self.hypothesis for item in self.spans)):
            raise ValueError("观察生成角色 Span 的 O/H 或来源漂移")
        if (len(self.target.bindings) != len(self.spans)
                or {(item.role, item.filler) for item in self.target.bindings}
                != {(item.role, item.origin) for item in self.spans}):
            raise ValueError("观察生成目标必须完整保留角色 Span，不得冒充旧例值")
        if (type(self.evidence) is not tuple or any(not isinstance(item, EvidenceEntry) for item in self.evidence)
                or {item.space for item in self.evidence} != {SPACE_CORE, SPACE_MEMORY, SPACE_DIALOGUE}
                or tuple(item.stable_key() for item in self.evidence)
                != tuple(sorted({item.stable_key() for item in self.evidence}))):
            raise ValueError("观察生成必须保留规范完整三图证据")
        for item in self.memory_evidence:
            if (item.hypothesis_key != self.hypothesis.stable_key()
                    or item.source_ref[1:] != self.source.stable_key()
                    or item.scope_key != self.scope.stable_key() or not item.evidence_key or not item.payload_key):
                raise ValueError("观察生成 Memory Evidence 缺少实际来源或 O/H 关联")
        if not self.memory_evidence:
            raise ValueError("观察生成不能使用空 Memory Evidence")
        self.stable_key()

    @property
    def memory_evidence(self) -> tuple[EvidenceEntry, ...]:
        """只用 Memory 对关系的原始立场判定真值，框架或语言支持不算事实支持。"""
        return tuple(item for item in self.evidence if item.space == SPACE_MEMORY)

    @property
    def state(self) -> LogicEvidenceState:
        """从实际 Memory 立场导出四态，不因能够发音而升级未知关系。"""
        return LogicEvidenceState(any(item.polarity == 1 for item in self.memory_evidence),
                                  any(item.polarity == 2 for item in self.memory_evidence))

    def stable_key(self) -> tuple[int, ...]:
        """保留绑定视图、O/H/E、各角色来源及完整三图证明。"""
        return (91526, 1, *_pack(self.target.stable_key()), *_pack(self.source.stable_key()),
                *_pack(self.scope.stable_key()), *_pack(self.observation.stable_key()),
                *_pack(self.hypothesis.stable_key()), *_pack(self.hypothesis_kind),
                *_pack(self.competition_key), len(self.spans),
                *(v for item in self.spans for v in _pack(item.stable_key())), len(self.evidence),
                *(v for item in self.evidence for v in _pack(item.stable_key())), *_pack(self.binding_trace))
