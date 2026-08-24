"""公开 W03-W05 受限课程的 raw-byte 完整命题句 demo。

这不是 F-00 问答、会话状态或开放自然语言入口。核心入口只接收
``tuple[int, ...]`` byte vector，先经过 DLG-RAW-00，再以已学习的
Unicode scalar 序列精确查找公开课程路由。命中时仅投影同次 W03-W05
proof 的 ``generated_proposition_surface``；未学习、歧义或运行时拒绝
均不产生自然语言补全。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.conversation_raw_intake import (
    ConversationRawIntake,
    encode_utf8_v1,
    intake_raw_conversation_vector,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_contract import (
    RawQuestionRequest,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_sparse_qa_runtime import (
    run_sparse_qa_query_with_typed_proof,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_sparse_qa_runtime_contract import (
    SparseQARuntime,
    SparseQASameDispatchProofProjection,
)


PUBLIC_SENTENCE_DEMO_RECORD_V1 = 1
PUBLIC_SENTENCE_DEMO_CATALOG_RECORD_V1 = 1
PUBLIC_SENTENCE_DEMO_ROUTE_RECORD_V1 = 1
PUBLIC_SENTENCE_DEMO_MAX_OUTPUT_BYTES = 4096

PUBLIC_SENTENCE_DEMO_ANSWER = 0
PUBLIC_SENTENCE_DEMO_REJECT_RAW = 1
PUBLIC_SENTENCE_DEMO_REJECT_LEXICAL_MISS = 2
PUBLIC_SENTENCE_DEMO_REJECT_LEXICAL_AMBIGUOUS = 3
PUBLIC_SENTENCE_DEMO_REJECT_RUNTIME_UNKNOWN = 4
PUBLIC_SENTENCE_DEMO_REJECT_RUNTIME_AMBIGUOUS = 5
PUBLIC_SENTENCE_DEMO_REJECT_OUTPUT_BUDGET = 6
PUBLIC_SENTENCE_DEMO_REJECT_RUNTIME_INCONSISTENT = 7

PUBLIC_SENTENCE_DEMO_ROUTE_NONE = 0
PUBLIC_SENTENCE_DEMO_ROUTE_EXACT = 1
PUBLIC_SENTENCE_DEMO_ROUTE_ALIAS = 2
PUBLIC_SENTENCE_DEMO_ROUTE_IMPLICIT = 3

PUBLIC_SENTENCE_DEMO_DISPATCH_NONE = 0
PUBLIC_SENTENCE_DEMO_DISPATCH_ANSWER = 1
PUBLIC_SENTENCE_DEMO_DISPATCH_UNKNOWN = 2
PUBLIC_SENTENCE_DEMO_DISPATCH_CLARIFY = 3

_RESULT_CODES = frozenset({
    PUBLIC_SENTENCE_DEMO_ANSWER,
    PUBLIC_SENTENCE_DEMO_REJECT_RAW,
    PUBLIC_SENTENCE_DEMO_REJECT_LEXICAL_MISS,
    PUBLIC_SENTENCE_DEMO_REJECT_LEXICAL_AMBIGUOUS,
    PUBLIC_SENTENCE_DEMO_REJECT_RUNTIME_UNKNOWN,
    PUBLIC_SENTENCE_DEMO_REJECT_RUNTIME_AMBIGUOUS,
    PUBLIC_SENTENCE_DEMO_REJECT_OUTPUT_BUDGET,
    PUBLIC_SENTENCE_DEMO_REJECT_RUNTIME_INCONSISTENT,
})
_ROUTE_KINDS = frozenset({
    PUBLIC_SENTENCE_DEMO_ROUTE_EXACT,
    PUBLIC_SENTENCE_DEMO_ROUTE_ALIAS,
    PUBLIC_SENTENCE_DEMO_ROUTE_IMPLICIT,
})
_DISPATCH_CODES = frozenset({
    PUBLIC_SENTENCE_DEMO_DISPATCH_NONE,
    PUBLIC_SENTENCE_DEMO_DISPATCH_ANSWER,
    PUBLIC_SENTENCE_DEMO_DISPATCH_UNKNOWN,
    PUBLIC_SENTENCE_DEMO_DISPATCH_CLARIFY,
})


# object-model: exception; interop=host-api-precondition
class PublicSentenceDemoError(ValueError):
    """公开课程 demo 的可迁移 record 或既有运行时边界发生漂移。"""


def _pack(result: list[int], value: tuple[int, ...]) -> None:
    """按长度前缀把一个整数段加入规范 record。"""
    result.extend((len(value), *value))


def _integer_vector(
        value: tuple[int, ...],
        *,
        label: str,
        allow_empty: bool,
        ) -> tuple[int, ...]:
    """核验不依赖宿主数值子类的有限严格整数序列。"""
    if not isinstance(value, tuple):
        raise TypeError(f"{label} 必须是整数 tuple")
    if (not allow_empty and not value) or any(type(item) is not int for item in value):
        raise PublicSentenceDemoError(f"{label} 不是规范严格整数序列")
    return value


def _unicode_scalars(
        value: tuple[int, ...],
        *,
        label: str,
        allow_empty: bool = False,
        ) -> tuple[int, ...]:
    """核验可由 UTF-8 v1 编码的 Unicode scalar 序列。"""
    scalars = _integer_vector(value, label=label, allow_empty=allow_empty)
    if any(
            item < 0 or item > 0x10FFFF or 0xD800 <= item <= 0xDFFF
            for item in scalars):
        raise PublicSentenceDemoError(f"{label} 含非法 Unicode scalar")
    return scalars


def _source_key(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验课程路由保留的非空来源整数 key。"""
    return _integer_vector(value, label=label, allow_empty=False)


def _registered_code(
        value: int,
        *,
        registered: frozenset[int],
        label: str,
        ) -> int:
    """拒绝 bool、整数子类与未注册的协议 code/tag。"""
    if type(value) is not int or value not in registered:
        raise PublicSentenceDemoError(f"{label} 未注册或不是严格整数")
    return value


def _surface_scalars(surface: str, *, label: str) -> tuple[int, ...]:
    """仅把既有学习产物表层复制为 scalar，不解析外部用户文本。"""
    if not isinstance(surface, str) or not surface:
        raise PublicSentenceDemoError(f"{label} 不是非空既有表层")
    return _unicode_scalars(
        tuple(ord(character) for character in surface),
        label=label,
    )


def _sha256_nibbles(value: str, *, label: str) -> tuple[int, ...]:
    """将既有 ASCII SHA-256 identity 显式降为 64 个整数 nibble。"""
    if not isinstance(value, str) or len(value) != 64:
        raise PublicSentenceDemoError(f"{label} 不是 SHA-256")
    result: list[int] = []
    for character in value:
        ordinal = ord(character)
        if 0x30 <= ordinal <= 0x39:
            result.append(ordinal - 0x30)
        elif 0x61 <= ordinal <= 0x66:
            result.append(ordinal - 0x61 + 10)
        else:
            raise PublicSentenceDemoError(f"{label} 不是小写 SHA-256")
    return tuple(result)


def _alias_question_surface(construction, alias_surface: str) -> str:
    """把一个已学习 alias 精确替换进已有 question construction。"""
    predicate_ordinals = tuple(
        ordinal for ordinal, segment in enumerate(construction.segments)
        if segment.kind == "PREDICATE"
    )
    if len(predicate_ordinals) != 1:
        raise PublicSentenceDemoError("公开课程 construction 缺少唯一 predicate")
    predicate_ordinal = predicate_ordinals[0]
    return "".join(
        alias_surface if ordinal == predicate_ordinal else segment.surface
        for ordinal, segment in enumerate(construction.segments)
    )


# object-model: value; representation=struct; interop=DLG-RAW-00-to-FT22
@dataclass(frozen=True, slots=True)
class PublicSentenceDemoRoute:
    """一个公开学习到的表层 scalar 到来源绑定旧运行时请求的有限路由。"""

    route_kind: int
    input_scalars: tuple[int, ...]
    request: RawQuestionRequest
    source_record_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """保证 Python legacy request 仅是与规范 scalar 同值的执行适配。"""
        route_kind = _registered_code(
            self.route_kind,
            registered=_ROUTE_KINDS,
            label="公开课程 route kind",
        )
        scalars = _unicode_scalars(
            self.input_scalars,
            label="公开课程 route input scalar",
        )
        if not isinstance(self.request, RawQuestionRequest):
            raise TypeError("公开课程 route request 非法")
        source = _source_key(
            self.source_record_key,
            label="公开课程 route SourceRef",
        )
        if self.request.source_record_key != source:
            raise PublicSentenceDemoError("公开课程 route 丢失 SourceRef 绑定")
        if _surface_scalars(
                self.request.question_surface,
                label="公开课程 route legacy surface",
        ) != scalars:
            raise PublicSentenceDemoError("公开课程 route scalar 与 legacy request 不一致")
        object.__setattr__(self, "route_kind", route_kind)
        object.__setattr__(self, "input_scalars", scalars)
        object.__setattr__(self, "source_record_key", source)

    def canonical_record(self) -> tuple[int, ...]:
        """返回不依赖 Python 字符串或对象身份的路由 record。"""
        result = [PUBLIC_SENTENCE_DEMO_ROUTE_RECORD_V1, self.route_kind]
        _pack(result, self.input_scalars)
        _pack(result, self.source_record_key)
        return tuple(result)


# object-model: value; representation=struct; interop=public-course-catalog
@dataclass(frozen=True, slots=True)
class PublicSentenceDemoCatalog:
    """绑定一个 FT22 runtime 的只读公开课程 scalar 路由目录。"""

    runtime_identity_sha256: str
    routes: tuple[PublicSentenceDemoRoute, ...]

    def __post_init__(self) -> None:
        """冻结运行时 identity 与规范排序；重复 route 保留为显式歧义。"""
        _sha256_nibbles(
            self.runtime_identity_sha256,
            label="公开课程 catalog runtime identity",
        )
        if (not isinstance(self.routes, tuple) or not self.routes
                or any(not isinstance(item, PublicSentenceDemoRoute)
                       for item in self.routes)):
            raise PublicSentenceDemoError("公开课程 catalog routes 非法")
        if self.routes != tuple(sorted(
                self.routes,
                key=PublicSentenceDemoRoute.canonical_record,
        )):
            raise PublicSentenceDemoError("公开课程 catalog routes 未规范排序")

    def matching_routes(
            self,
            input_scalars: tuple[int, ...],
            ) -> tuple[PublicSentenceDemoRoute, ...]:
        """线性精确比较 scalar，避免宿主字符串匹配和隐式 fallback。"""
        scalars = _unicode_scalars(
            input_scalars,
            label="公开课程 lookup scalar",
        )
        return tuple(
            item for item in self.routes if item.input_scalars == scalars)

    def canonical_record(self) -> tuple[int, ...]:
        """导出不含 legacy runtime identity 的 catalog 整数本体。"""
        result = [PUBLIC_SENTENCE_DEMO_CATALOG_RECORD_V1, len(self.routes)]
        for route in self.routes:
            _pack(result, route.canonical_record())
        return tuple(result)


# object-model: value; representation=struct; interop=public-sentence-demo
@dataclass(frozen=True, slots=True)
class PublicSentenceDemoResult:
    """一条 raw vector 的有限课程 demo 结果及其完整整数输出 record。"""

    result_code: int
    intake: ConversationRawIntake
    matched_route_count: int = 0
    selected_route_kind: int = PUBLIC_SENTENCE_DEMO_ROUTE_NONE
    selected_source_record_key: tuple[int, ...] = ()
    dispatch_status_code: int = PUBLIC_SENTENCE_DEMO_DISPATCH_NONE
    generated_proposition_scalars: tuple[int, ...] = ()
    output_bytes: tuple[int, ...] = ()
    generated_proposition_surface: str | None = None

    def __post_init__(self) -> None:
        """以 result code 冻结拒绝、来源与完整句投影的所有可观察状态。"""
        result_code = _registered_code(
            self.result_code,
            registered=_RESULT_CODES,
            label="公开课程 demo result code",
        )
        if not isinstance(self.intake, ConversationRawIntake):
            raise TypeError("公开课程 demo 缺少 DLG-RAW-00 intake")
        if type(self.matched_route_count) is not int or self.matched_route_count < 0:
            raise PublicSentenceDemoError("公开课程 demo route count 非法")
        selected_route_kind = _registered_code(
            self.selected_route_kind,
            registered=frozenset({
                PUBLIC_SENTENCE_DEMO_ROUTE_NONE,
                *_ROUTE_KINDS,
            }),
            label="公开课程 demo selected route kind",
        )
        dispatch_status_code = _registered_code(
            self.dispatch_status_code,
            registered=_DISPATCH_CODES,
            label="公开课程 demo dispatch status",
        )
        source = _integer_vector(
            self.selected_source_record_key,
            label="公开课程 demo selected SourceRef",
            allow_empty=True,
        )
        scalars = _unicode_scalars(
            self.generated_proposition_scalars,
            label="公开课程 demo output scalar",
            allow_empty=True,
        )
        output = _integer_vector(
            self.output_bytes,
            label="公开课程 demo output bytes",
            allow_empty=True,
        )
        if any(item < 0 or item > 255 for item in output):
            raise PublicSentenceDemoError("公开课程 demo output bytes 非法")
        has_output = bool(scalars or output or self.generated_proposition_surface)
        has_selected_route = selected_route_kind != PUBLIC_SENTENCE_DEMO_ROUTE_NONE
        if result_code != PUBLIC_SENTENCE_DEMO_ANSWER and self.generated_proposition_surface is not None:
            raise PublicSentenceDemoError("公开课程 demo 拒绝 record 不得携带表层文本")
        if result_code == PUBLIC_SENTENCE_DEMO_ANSWER:
            if (not self.intake.accepted or self.matched_route_count != 1
                    or not has_selected_route or not source
                    or dispatch_status_code
                    != PUBLIC_SENTENCE_DEMO_DISPATCH_ANSWER
                    or not scalars or not output
                    or not isinstance(self.generated_proposition_surface, str)
                    or _surface_scalars(
                        self.generated_proposition_surface,
                        label="公开课程 demo output surface",
                    ) != scalars
                    or encode_utf8_v1(scalars) != output
                    or len(output) > PUBLIC_SENTENCE_DEMO_MAX_OUTPUT_BYTES):
                raise PublicSentenceDemoError("公开课程 demo ANSWER record 非法")
        elif result_code == PUBLIC_SENTENCE_DEMO_REJECT_RAW:
            if (self.intake.accepted or self.matched_route_count != 0
                    or has_selected_route or source or has_output
                    or dispatch_status_code
                    != PUBLIC_SENTENCE_DEMO_DISPATCH_NONE):
                raise PublicSentenceDemoError("公开课程 demo raw 拒绝 record 非法")
        elif result_code == PUBLIC_SENTENCE_DEMO_REJECT_LEXICAL_MISS:
            if (not self.intake.accepted or self.matched_route_count != 0
                    or has_selected_route or source or has_output
                    or dispatch_status_code
                    != PUBLIC_SENTENCE_DEMO_DISPATCH_NONE):
                raise PublicSentenceDemoError("公开课程 demo lexical miss record 非法")
        elif result_code == PUBLIC_SENTENCE_DEMO_REJECT_LEXICAL_AMBIGUOUS:
            if (not self.intake.accepted or self.matched_route_count < 2
                    or has_selected_route or source or has_output
                    or dispatch_status_code
                    != PUBLIC_SENTENCE_DEMO_DISPATCH_NONE):
                raise PublicSentenceDemoError("公开课程 demo lexical ambiguity record 非法")
        elif result_code == PUBLIC_SENTENCE_DEMO_REJECT_RUNTIME_UNKNOWN:
            if (not self.intake.accepted or self.matched_route_count != 1
                    or not has_selected_route or not source or has_output
                    or dispatch_status_code
                    != PUBLIC_SENTENCE_DEMO_DISPATCH_UNKNOWN):
                raise PublicSentenceDemoError("公开课程 demo runtime unknown record 非法")
        elif result_code == PUBLIC_SENTENCE_DEMO_REJECT_RUNTIME_AMBIGUOUS:
            if (not self.intake.accepted or self.matched_route_count != 1
                    or not has_selected_route or not source or has_output
                    or dispatch_status_code
                    != PUBLIC_SENTENCE_DEMO_DISPATCH_CLARIFY):
                raise PublicSentenceDemoError("公开课程 demo runtime ambiguity record 非法")
        elif result_code == PUBLIC_SENTENCE_DEMO_REJECT_OUTPUT_BUDGET:
            if (not self.intake.accepted or self.matched_route_count != 1
                    or not has_selected_route or not source or has_output
                    or dispatch_status_code
                    != PUBLIC_SENTENCE_DEMO_DISPATCH_ANSWER):
                raise PublicSentenceDemoError("公开课程 demo output budget record 非法")
        elif (not self.intake.accepted or self.matched_route_count != 1
                or not has_selected_route or not source or has_output
                or dispatch_status_code
                != PUBLIC_SENTENCE_DEMO_DISPATCH_NONE):
            raise PublicSentenceDemoError("公开课程 demo runtime inconsistency record 非法")
        object.__setattr__(self, "result_code", result_code)
        object.__setattr__(self, "selected_route_kind", selected_route_kind)
        object.__setattr__(self, "dispatch_status_code", dispatch_status_code)
        object.__setattr__(self, "selected_source_record_key", source)
        object.__setattr__(self, "generated_proposition_scalars", scalars)
        object.__setattr__(self, "output_bytes", output)

    @property
    def accepted(self) -> bool:
        """仅在公开课程 proof 已实际投影为完整句时为真。"""
        return self.result_code == PUBLIC_SENTENCE_DEMO_ANSWER

    def canonical_record(self) -> tuple[int, ...]:
        """导出无 Python 文本/类身份依赖的 demo result 整数 record。"""
        result = [
            PUBLIC_SENTENCE_DEMO_RECORD_V1,
            self.result_code,
            self.dispatch_status_code,
            self.matched_route_count,
            self.selected_route_kind,
        ]
        _pack(result, self.intake.canonical_record())
        _pack(result, self.selected_source_record_key)
        _pack(result, self.generated_proposition_scalars)
        _pack(result, self.output_bytes)
        return tuple(result)


# object-model: host-adapter; representation=legacy-carrier; interop=not-canonical
@dataclass(frozen=True, slots=True)
class PublicSentenceDemoSameDispatchProofProjection:
    """仅供 Python host adapter 并排保留 demo 结果与同次 typed proof。

    该载体不提供 ``canonical_record``、``to_dict`` 或 ``sha256``。嵌套的
    legacy proof 只能作为下一层显式整数投影的读取源，不能进入可迁移状态或
    改变既有 ``PublicSentenceDemoResult`` 的公开协议。
    """

    demo_result: PublicSentenceDemoResult
    sparse_proof_projection: SparseQASameDispatchProofProjection | None
    host_adapter_only: int = 1

    def __post_init__(self) -> None:
        """只允许 ANSWER 公开同次 proof；所有 demo 拒绝保持零 proof。"""
        if not isinstance(self.demo_result, PublicSentenceDemoResult):
            raise TypeError("公开课程 host proof carrier 缺少 demo result")
        if type(self.host_adapter_only) is not int or self.host_adapter_only != 1:
            raise PublicSentenceDemoError(
                "公开课程 host proof carrier 缺少 host-only 标记")
        projection = self.sparse_proof_projection
        if self.demo_result.accepted:
            if (not isinstance(projection, SparseQASameDispatchProofProjection)
                    or projection.query_result.status != "ANSWER"
                    or projection.typed_proof is None
                    or projection.generated_proposition_surface is None
                    or projection.query_result.request.source_record_key
                    != self.demo_result.selected_source_record_key
                    or projection.generated_proposition_surface
                    != self.demo_result.generated_proposition_surface):
                raise PublicSentenceDemoError(
                    "公开课程 ANSWER 未绑定同次 typed proof")
            return
        if projection is None:
            return
        if (not isinstance(projection, SparseQASameDispatchProofProjection)
                or projection.query_result.status not in {"UNKNOWN", "CLARIFY"}
                or projection.typed_proof is not None
                or projection.generated_proposition_surface is not None
                or projection.raw_result is not None
                or projection.query_result.request.source_record_key
                != self.demo_result.selected_source_record_key):
            raise PublicSentenceDemoError(
                "公开课程 non-answer 泄漏或错配 typed proof")
        expected = (
            ("UNKNOWN", PUBLIC_SENTENCE_DEMO_REJECT_RUNTIME_UNKNOWN,
             PUBLIC_SENTENCE_DEMO_DISPATCH_UNKNOWN),
            ("CLARIFY", PUBLIC_SENTENCE_DEMO_REJECT_RUNTIME_AMBIGUOUS,
             PUBLIC_SENTENCE_DEMO_DISPATCH_CLARIFY),
        )
        if (projection.query_result.status,
                self.demo_result.result_code,
                self.demo_result.dispatch_status_code) not in expected:
            raise PublicSentenceDemoError(
                "公开课程 non-answer 未按 sparse dispatch 状态映射")


def _route_from_construction(
        route_kind: int,
        construction,
        question_surface: str | None = None,
        ) -> PublicSentenceDemoRoute:
    """从一个来源绑定学习 construction 建立精确 raw-scalar 路由。"""
    source = construction.source_record_key
    surface = construction.question_surface if question_surface is None else question_surface
    return PublicSentenceDemoRoute(
        route_kind,
        _surface_scalars(surface, label="公开课程 construction surface"),
        RawQuestionRequest(surface, source),
        source,
    )


def build_public_sentence_demo_catalog(
        runtime: SparseQARuntime,
        ) -> PublicSentenceDemoCatalog:
    """从实际 FT22 public runtime 的 exact/alias/implicit 学习对象冻结目录。"""
    if not isinstance(runtime, SparseQARuntime):
        raise TypeError("公开课程 demo catalog 只接受 SparseQARuntime")
    routes: list[PublicSentenceDemoRoute] = []
    for dispatch_entry in runtime.dispatch_index.entries:
        entry = dispatch_entry.entry
        explicit = entry.feature_catalog.catalog
        implicit = entry.implicit_bundle.catalog
        for construction in explicit:
            routes.append(_route_from_construction(
                PUBLIC_SENTENCE_DEMO_ROUTE_EXACT,
                construction,
            ))
        for construction in implicit:
            routes.append(_route_from_construction(
                PUBLIC_SENTENCE_DEMO_ROUTE_IMPLICIT,
                construction,
            ))
        for construction in explicit:
            link = construction.vertical_result.link
            if link is None:
                raise PublicSentenceDemoError(
                    "公开课程 alias construction 缺少 Proposition link")
            alias_routes = tuple(
                item for item in entry.alias_bridge.routes
                if item.proposition_key == link.proposition_key
            )
            if not alias_routes:
                raise PublicSentenceDemoError(
                    "公开课程 alias construction 缺少已学习 lexical route")
            for alias_route in alias_routes:
                routes.append(_route_from_construction(
                    PUBLIC_SENTENCE_DEMO_ROUTE_ALIAS,
                    construction,
                    _alias_question_surface(
                        construction,
                        alias_route.alias_surface,
                    ),
                ))
    ordered = tuple(sorted(
        routes,
        key=PublicSentenceDemoRoute.canonical_record,
    ))
    return PublicSentenceDemoCatalog(runtime.identity_sha256, ordered)


def _rejection(
        result_code: int,
        intake: ConversationRawIntake,
        *,
        matched_route_count: int = 0,
        route: PublicSentenceDemoRoute | None = None,
        dispatch_status_code: int = PUBLIC_SENTENCE_DEMO_DISPATCH_NONE,
        ) -> PublicSentenceDemoResult:
    """构造无自然语言输出的 fail-closed demo record。"""
    return PublicSentenceDemoResult(
        result_code,
        intake,
        matched_route_count,
        (PUBLIC_SENTENCE_DEMO_ROUTE_NONE
         if route is None else route.route_kind),
        () if route is None else route.source_record_key,
        dispatch_status_code,
    )


def run_public_sentence_demo_vector_with_typed_proof(
        runtime: SparseQARuntime,
        catalog: PublicSentenceDemoCatalog,
        raw_input_bytes: tuple[int, ...],
        ) -> PublicSentenceDemoSameDispatchProofProjection:
    """host-only：一次 dispatch 同时返回原 demo result 与其 typed proof。"""
    if not isinstance(runtime, SparseQARuntime):
        raise TypeError("公开课程 demo runtime 非法")
    if not isinstance(catalog, PublicSentenceDemoCatalog):
        raise TypeError("公开课程 demo catalog 非法")
    if catalog.runtime_identity_sha256 != runtime.identity_sha256:
        raise PublicSentenceDemoError("公开课程 catalog 不能跨 runtime 使用")
    intake = intake_raw_conversation_vector(raw_input_bytes)
    if not intake.accepted:
        return PublicSentenceDemoSameDispatchProofProjection(
            _rejection(PUBLIC_SENTENCE_DEMO_REJECT_RAW, intake),
            None,
        )
    matches = catalog.matching_routes(intake.unicode_scalars)
    if not matches:
        return PublicSentenceDemoSameDispatchProofProjection(
            _rejection(PUBLIC_SENTENCE_DEMO_REJECT_LEXICAL_MISS, intake),
            None,
        )
    if len(matches) != 1:
        return PublicSentenceDemoSameDispatchProofProjection(
            _rejection(
                PUBLIC_SENTENCE_DEMO_REJECT_LEXICAL_AMBIGUOUS,
                intake,
                matched_route_count=len(matches),
            ),
            None,
        )
    route = matches[0]
    projection = run_sparse_qa_query_with_typed_proof(runtime, route.request)
    query_result = projection.query_result
    if query_result.request != route.request:
        return PublicSentenceDemoSameDispatchProofProjection(
            _rejection(
                PUBLIC_SENTENCE_DEMO_REJECT_RUNTIME_INCONSISTENT,
                intake,
                matched_route_count=1,
                route=route,
            ),
            None,
        )
    if query_result.status == "UNKNOWN":
        return PublicSentenceDemoSameDispatchProofProjection(
            _rejection(
                PUBLIC_SENTENCE_DEMO_REJECT_RUNTIME_UNKNOWN,
                intake,
                matched_route_count=1,
                route=route,
                dispatch_status_code=PUBLIC_SENTENCE_DEMO_DISPATCH_UNKNOWN,
            ),
            projection,
        )
    if query_result.status == "CLARIFY":
        return PublicSentenceDemoSameDispatchProofProjection(
            _rejection(
                PUBLIC_SENTENCE_DEMO_REJECT_RUNTIME_AMBIGUOUS,
                intake,
                matched_route_count=1,
                route=route,
                dispatch_status_code=PUBLIC_SENTENCE_DEMO_DISPATCH_CLARIFY,
            ),
            projection,
        )
    if (query_result.status != "ANSWER"
            or query_result.selected_source_record_key
            != route.source_record_key
            or projection.generated_proposition_surface is None):
        return PublicSentenceDemoSameDispatchProofProjection(
            _rejection(
                PUBLIC_SENTENCE_DEMO_REJECT_RUNTIME_INCONSISTENT,
                intake,
                matched_route_count=1,
                route=route,
            ),
            None,
        )
    surface = projection.generated_proposition_surface
    scalars = _surface_scalars(surface, label="公开课程 proof output")
    output = encode_utf8_v1(scalars)
    if len(output) > PUBLIC_SENTENCE_DEMO_MAX_OUTPUT_BYTES:
        return PublicSentenceDemoSameDispatchProofProjection(
            _rejection(
                PUBLIC_SENTENCE_DEMO_REJECT_OUTPUT_BUDGET,
                intake,
                matched_route_count=1,
                route=route,
                dispatch_status_code=PUBLIC_SENTENCE_DEMO_DISPATCH_ANSWER,
            ),
            None,
        )
    return PublicSentenceDemoSameDispatchProofProjection(
        PublicSentenceDemoResult(
            PUBLIC_SENTENCE_DEMO_ANSWER,
            intake,
            1,
            route.route_kind,
            route.source_record_key,
            PUBLIC_SENTENCE_DEMO_DISPATCH_ANSWER,
            scalars,
            output,
            surface,
        ),
        projection,
    )


def run_public_sentence_demo_vector(
        runtime: SparseQARuntime,
        catalog: PublicSentenceDemoCatalog,
        raw_input_bytes: tuple[int, ...],
        ) -> PublicSentenceDemoResult:
    """运行原公开 demo 协议；内部委托 host-only proof carrier 后只返回旧 result。"""
    return run_public_sentence_demo_vector_with_typed_proof(
        runtime,
        catalog,
        raw_input_bytes,
    ).demo_result


def run_public_sentence_demo_bytes(
        runtime: SparseQARuntime,
        catalog: PublicSentenceDemoCatalog,
        raw_input_bytes: bytes,
        ) -> PublicSentenceDemoResult:
    """Python I/O 边界：复制 bytes 后交给只接受整数 tuple 的核心入口。"""
    if type(raw_input_bytes) is not bytes:
        raise TypeError("公开课程 demo bytes adapter 只接受 bytes")
    return run_public_sentence_demo_vector(
        runtime,
        catalog,
        tuple(raw_input_bytes),
    )


__all__ = [
    "PUBLIC_SENTENCE_DEMO_ANSWER",
    "PUBLIC_SENTENCE_DEMO_CATALOG_RECORD_V1",
    "PUBLIC_SENTENCE_DEMO_DISPATCH_ANSWER",
    "PUBLIC_SENTENCE_DEMO_DISPATCH_CLARIFY",
    "PUBLIC_SENTENCE_DEMO_DISPATCH_NONE",
    "PUBLIC_SENTENCE_DEMO_DISPATCH_UNKNOWN",
    "PUBLIC_SENTENCE_DEMO_MAX_OUTPUT_BYTES",
    "PUBLIC_SENTENCE_DEMO_RECORD_V1",
    "PUBLIC_SENTENCE_DEMO_REJECT_LEXICAL_AMBIGUOUS",
    "PUBLIC_SENTENCE_DEMO_REJECT_LEXICAL_MISS",
    "PUBLIC_SENTENCE_DEMO_REJECT_OUTPUT_BUDGET",
    "PUBLIC_SENTENCE_DEMO_REJECT_RAW",
    "PUBLIC_SENTENCE_DEMO_REJECT_RUNTIME_AMBIGUOUS",
    "PUBLIC_SENTENCE_DEMO_REJECT_RUNTIME_INCONSISTENT",
    "PUBLIC_SENTENCE_DEMO_REJECT_RUNTIME_UNKNOWN",
    "PUBLIC_SENTENCE_DEMO_ROUTE_ALIAS",
    "PUBLIC_SENTENCE_DEMO_ROUTE_EXACT",
    "PUBLIC_SENTENCE_DEMO_ROUTE_IMPLICIT",
    "PUBLIC_SENTENCE_DEMO_ROUTE_NONE",
    "PUBLIC_SENTENCE_DEMO_ROUTE_RECORD_V1",
    "PublicSentenceDemoCatalog",
    "PublicSentenceDemoError",
    "PublicSentenceDemoResult",
    "PublicSentenceDemoRoute",
    "PublicSentenceDemoSameDispatchProofProjection",
    "build_public_sentence_demo_catalog",
    "run_public_sentence_demo_bytes",
    "run_public_sentence_demo_vector",
    "run_public_sentence_demo_vector_with_typed_proof",
]
