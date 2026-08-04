"""W08-04 到现役 A-03、A-08 与 free-text revision owner 的薄适配。"""
from __future__ import annotations

from collections.abc import Callable

from pure_integer_ai.cognition.shared.parser_revision import ParserRevisionRequest
from pure_integer_ai.cognition.understanding.memory_intake import (
    ObservationIntakeDraft,
)
from pure_integer_ai.experiments.free_text_revision_runtime import (
    FreeTextRevisionInvalidationReceipt,
    FreeTextRevisionInvalidator,
)
from pure_integer_ai.experiments.memory_reparse_runtime import (
    MemoryParserRevisionResult,
    MemoryParserRevisionRuntime,
)
from pure_integer_ai.experiments.parser_revision_runtime import (
    ParserRevisionResult,
    ParserRevisionRuntime,
)
from pure_integer_ai.experiments.ph2_w08_recompute_contract import (
    W08_RECOMPUTE_OWNER_KEYS,
)


class W08RecomputeAdapterError(ValueError):
    """现役 revision owner 返回了错误 receipt。"""


class W08A03ParserRevisionOwner:
    """绑定 A-03 的同原文 ParserVersion 原子 revision。"""

    owner_key = W08_RECOMPUTE_OWNER_KEYS[1]

    def __init__(self, runtime: ParserRevisionRuntime) -> None:
        if not isinstance(runtime, ParserRevisionRuntime):
            raise TypeError("W08 A-03 owner requires ParserRevisionRuntime")
        self.runtime = runtime

    def apply(self, request: ParserRevisionRequest) -> ParserRevisionResult:
        result = self.runtime.apply(request)
        if not isinstance(result, ParserRevisionResult):
            raise W08RecomputeAdapterError("A-03 returned an invalid revision result")
        return result


class W08A08MemoryRevisionOwner:
    """绑定 A-08 的长期 Memory 派生更新与旧 Use 保留审计。"""

    owner_key = W08_RECOMPUTE_OWNER_KEYS[1]

    def __init__(self, runtime: MemoryParserRevisionRuntime) -> None:
        if not isinstance(runtime, MemoryParserRevisionRuntime):
            raise TypeError("W08 A-08 owner requires MemoryParserRevisionRuntime")
        self.runtime = runtime

    def apply(
        self,
        request: ParserRevisionRequest,
        *,
        raw_text: str,
        license_id: str,
        batch_id: int,
        parser: object,
        materialize: Callable[
            [ObservationIntakeDraft], ObservationIntakeDraft
        ] | None = None,
        batch_fault_injector=None,
    ) -> MemoryParserRevisionResult:
        result = self.runtime.apply(
            request,
            raw_text=raw_text,
            license_id=license_id,
            batch_id=batch_id,
            parser=parser,
            materialize=materialize,
            batch_fault_injector=batch_fault_injector,
        )
        if not isinstance(result, MemoryParserRevisionResult):
            raise W08RecomputeAdapterError("A-08 returned an invalid revision result")
        return result


class W08FreeTextRevisionOwner:
    """绑定 R-03/free-text hierarchy、center、claim 的 typed 求交。"""

    owner_key = W08_RECOMPUTE_OWNER_KEYS[3]

    def __init__(self, invalidator: FreeTextRevisionInvalidator) -> None:
        if not isinstance(invalidator, FreeTextRevisionInvalidator):
            raise TypeError("W08 free-text owner requires FreeTextRevisionInvalidator")
        self.invalidator = invalidator

    def invalidate(self, changed_dependencies) -> FreeTextRevisionInvalidationReceipt:
        receipt = self.invalidator.invalidate(changed_dependencies)
        if not isinstance(receipt, FreeTextRevisionInvalidationReceipt):
            raise W08RecomputeAdapterError(
                "free-text invalidator returned an invalid receipt"
            )
        return receipt


__all__ = [
    "W08A03ParserRevisionOwner",
    "W08A08MemoryRevisionOwner",
    "W08FreeTextRevisionOwner",
    "W08RecomputeAdapterError",
]
