"""公开对话的二进制终端适配器。

终端只是物理 transport。每个输入行被复制为有限 ``u8[]``，交给
DLG-RAW-00/01/02；输出只复制同次 G-03/G-04 已完成的 ``output_bytes``。
默认入口消费 DLG-RAW-14 outer response；兼容 runner 仍保留旧层 API。进程内会话仅保存
显式 typed context，不保留原始终端文本，也不写入
SQLite、长期 Memory 或其他持久状态。
"""
from __future__ import annotations

from pathlib import Path, PurePosixPath
import sys
import sysconfig
from typing import BinaryIO, Callable

from pure_integer_ai.experiments.conversation_public_dialogue_runtime import (
    PublicDialogueRuntimeV1,
    build_public_dialogue_runtime_v1,
)
from pure_integer_ai.experiments.conversation_public_source_payload_host import (
    load_public_source_payload_closure_from_root,
)
from pure_integer_ai.experiments.conversation_public_source_payload_provider import (
    PUBLIC_SOURCE_PAYLOAD_LOGICAL_KEYS_V1,
)
from pure_integer_ai.experiments.conversation_raw_answer_runtime import (
    ConversationRawAnswerResult,
)
from pure_integer_ai.experiments.conversation_public_proof_sentence_provider import (
    PUBLIC_PROOF_SENTENCE_PROVIDER_SNAPSHOT_RELATIVE_PATH,
    PublicProofSentenceProviderResultV1,
    load_public_proof_sentence_provider_from_root,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_sparse_qa_snapshot import (
    PUBLIC_SPARSE_QA_SOURCE_ARTIFACTS,
)
from pure_integer_ai.experiments.conversation_provider_origin_followup import (
    ProviderOriginFollowupResultV1,
)
from pure_integer_ai.experiments.conversation_raw_dialogue_session import (
    run_public_frame_dialogue_turn,
    start_public_frame_dialogue,
)
from pure_integer_ai.experiments.conversation_raw_mixed_focus_dialogue_session import (
    run_public_mixed_focus_dialogue_turn_v1,
    start_public_mixed_focus_dialogue,
)
from pure_integer_ai.experiments.conversation_raw_terminal_dialogue_act import (
    TerminalDialogueActRuntimeV1,
    TerminalDialogueResponseV1,
    build_terminal_dialogue_act_runtime_v1,
    run_public_terminal_dialogue_act_turn_v1,
    start_public_terminal_dialogue_act,
)
from pure_integer_ai.experiments.conversation_raw_route_clarification_dialogue import (
    ROUTE_CLARIFICATION_RESPONSE_KIND_PASSTHROUGH_V1,
    RouteClarificationDialogueResponseV1,
    RouteClarificationDialogueRuntimeV1,
    build_public_route_clarification_dialogue_runtime_v1,
    run_public_route_clarification_dialogue_turn_v1,
    start_public_route_clarification_dialogue,
)
from pure_integer_ai.experiments.conversation_raw_course_prepare import (
    PublicCoursePreparationCache,
)
from pure_integer_ai.experiments.alias_relation_course import (
    AliasRelationPreflightCache,
)
from pure_integer_ai.experiments.conversation_raw_intake import DLG_RAW_MAX_INPUT_BYTES
from pure_integer_ai.experiments.conversation_raw_intake import (
    DLG_RAW_REJECT_LEXICAL_MISS,
)


_TERMINAL_CONVERSATION_KEY = (65001, 34, 1)
_TERMINAL_READ_LIMIT = DLG_RAW_MAX_INPUT_BYTES + 1
_USER_PROMPT = b"\xe4\xbd\xa0> "
_SYSTEM_PROMPT = b"\xe7\xb3\xbb\xe7\xbb\x9f> "
_QUIT_LINES = frozenset({
    b":quit",
    b":quit\n",
    b":quit\r\n",
    b":exit",
    b":exit\n",
    b":exit\r\n",
})
_SOURCE_REPOSITORY = Path(__file__).resolve().parents[3]
_PUBLIC_DIALOGUE_DISTRIBUTION_SUBDIRECTORY = Path("share/pure_integer_ai")
_PUBLIC_DIALOGUE_REQUIRED_RESOURCES = tuple(sorted((
    *(item.decode("ascii") for item in PUBLIC_SOURCE_PAYLOAD_LOGICAL_KEYS_V1),
    PUBLIC_PROOF_SENTENCE_PROVIDER_SNAPSHOT_RELATIVE_PATH,
    *PUBLIC_SPARSE_QA_SOURCE_ARTIFACTS,
)))


def _public_dialogue_resource_root_is_complete(root: Path) -> bool:
    """仅在 terminal host 边界确认源码或安装目录具备完整公开资源闭包。"""
    try:
        base = root.resolve()
        for relative_path in _PUBLIC_DIALOGUE_REQUIRED_RESOURCES:
            logical = PurePosixPath(relative_path)
            if (logical.is_absolute() or logical.parts[:2] != ("data", "ph2")
                    or any(part in {"", ".", ".."} for part in logical.parts)):
                return False
            candidate = (base / Path(*logical.parts)).resolve()
            candidate.relative_to(base)
            if not candidate.is_file():
                return False
    except (OSError, ValueError):
        return False
    return True


def _installed_public_dialogue_resource_roots() -> tuple[Path, ...]:
    """枚举安装布局的 data 根；路径选择不进入对话核心语义。"""
    data_roots: list[Path] = []
    current = sysconfig.get_path("data")
    if current:
        data_roots.append(Path(current))
    for scheme in sysconfig.get_scheme_names():
        if not scheme.endswith("_user"):
            continue
        try:
            value = sysconfig.get_path("data", scheme=scheme)
        except (KeyError, TypeError, ValueError):
            continue
        if value:
            data_roots.append(Path(value))
    roots: list[Path] = []
    seen: set[Path] = set()
    for data_root in data_roots:
        root = (data_root / _PUBLIC_DIALOGUE_DISTRIBUTION_SUBDIRECTORY).resolve()
        if root not in seen:
            seen.add(root)
            roots.append(root)
    return tuple(roots)


def _default_public_dialogue_resource_root() -> Path:
    """优先完整源码闭包，否则选择完整安装闭包；缺失仍由 loader fail closed。"""
    for root in (
            _SOURCE_REPOSITORY,
            *_installed_public_dialogue_resource_roots(),
    ):
        if _public_dialogue_resource_root_is_complete(root):
            return root
    return _SOURCE_REPOSITORY


def _ascii_nonnegative_integer(value: int) -> bytes:
    """以显式十进制规则渲染非语义 terminal result code。"""
    if type(value) is not int or value < 0:
        raise ValueError("terminal result code 必须是非负严格整数")
    if value == 0:
        return b"0"
    digits: list[int] = []
    current = value
    while current:
        digits.append(0x30 + current % 10)
        current //= 10
    return bytes(reversed(digits))


def _host_bytes(value: tuple[int, ...], *, label: str) -> bytes:
    """在平台边缘复制规范 u8 vector，拒绝宿主字符串或隐式编码。"""
    if (not isinstance(value, tuple)
            or any(type(item) is not int or item < 0 or item > 255
                   for item in value)):
        raise ValueError(f"{label} 不是规范 u8 vector")
    return bytes(value)


def _read_bounded_line(input_stream: BinaryIO) -> tuple[bytes, bool]:
    """读取一条有界 input record；发现预算外字节后由调用方终止物理会话。"""
    raw = input_stream.readline(_TERMINAL_READ_LIMIT)
    if type(raw) is not bytes:
        raise TypeError("terminal input 必须返回 bytes")
    # 超过 4096 bytes 时 RAW-00 只需看到第 4097 个 byte 即可预算拒绝。终止
    # 本次 shell 可避免读取、截断或重解释同一物理行的余下 payload。
    return raw, len(raw) >= _TERMINAL_READ_LIMIT


def render_public_frame_dialogue_terminal_result(
        result: (ConversationRawAnswerResult
                 | PublicProofSentenceProviderResultV1
                 | ProviderOriginFollowupResultV1),
        ) -> bytes:
    """复制 Frame/provider/follow-up 的 canonical bytes，不组织替代回答。"""
    if not isinstance(result, (
            ConversationRawAnswerResult,
            PublicProofSentenceProviderResultV1,
            ProviderOriginFollowupResultV1,
    )):
        raise TypeError("terminal result 必须来自已登记 dialogue carrier")
    if result.accepted:
        output = (result.output_u8
                  if type(result) is ProviderOriginFollowupResultV1
                  else result.output_bytes)
        return _host_bytes(
            output,
            label="DLG-RAW terminal output",
        )
    result_code = (
        result.mapped_dlg_result_code
        if type(result) in {
            PublicProofSentenceProviderResultV1,
            ProviderOriginFollowupResultV1,
        }
        else result.result_code
    )
    return b"[REJECT:" + _ascii_nonnegative_integer(result_code) + b"]"


def render_terminal_dialogue_response_v1(
        response: TerminalDialogueResponseV1,
        ) -> bytes:
    """复制 DLG-RAW-13 统一 response 的已验证 raw-u8，不推断旧 carrier 类。"""
    if type(response) is not TerminalDialogueResponseV1:
        raise TypeError("terminal dialogue response 类型错误")
    return _host_bytes(
        response.output_u8,
        label="DLG-RAW terminal dialogue response",
    )


def render_route_clarification_dialogue_response_v1(
        response: RouteClarificationDialogueResponseV1,
        ) -> bytes:
    """复制 DLG-RAW-14 唯一 outer response 的已验证 raw-u8，保留课程内换行。"""
    if type(response) is not RouteClarificationDialogueResponseV1:
        raise TypeError("route clarification dialogue response 类型错误")
    return _host_bytes(
        response.output_u8,
        label="DLG-RAW route clarification dialogue response",
    )


def run_public_frame_dialogue_terminal(
        runtime: PublicDialogueRuntimeV1,
        input_stream: BinaryIO,
        output_stream: BinaryIO,
        *,
        prompts: bool = True,
        ) -> None:
    """运行保留的 V1 context shell；默认命令行入口不再调用本函数。"""
    if type(runtime) is not PublicDialogueRuntimeV1:
        raise TypeError("terminal public dialogue runtime 非法")
    if type(prompts) is not bool:
        raise TypeError("terminal prompts 必须是严格 bool")
    if not hasattr(input_stream, "readline") or not hasattr(output_stream, "write"):
        raise TypeError("terminal streams 必须提供 binary readline/write")
    state = start_public_frame_dialogue(_TERMINAL_CONVERSATION_KEY)
    preparation_cache = PublicCoursePreparationCache()
    preflight_cache = AliasRelationPreflightCache()
    while True:
        if prompts:
            output_stream.write(_USER_PROMPT)
            output_stream.flush()
        raw, budget_exceeded = _read_bounded_line(input_stream)
        if raw == b"" or raw in _QUIT_LINES:
            return
        turn = run_public_frame_dialogue_turn(
            state,
            tuple(raw),
            runtime,
            preparation_cache=preparation_cache,
            preflight_cache=preflight_cache,
        )
        state = turn.after
        result = (turn.answer if turn.answer is not None
                  else turn.provider_answer)
        if result is None:
            raise RuntimeError("RAW-04 turn 缺少回答 carrier")
        output_stream.write(_SYSTEM_PROMPT)
        output_stream.write(render_public_frame_dialogue_terminal_result(result))
        output_stream.write(b"\n")
        output_stream.flush()
        if budget_exceeded:
            return


def run_public_mixed_frame_dialogue_terminal(
        runtime: PublicDialogueRuntimeV1,
        input_stream: BinaryIO,
        output_stream: BinaryIO,
        *,
        prompts: bool = True,
        ) -> None:
    """运行默认 V3/V4 shell；连续焦点只进入 append-only V3 ledger。"""
    if type(runtime) is not PublicDialogueRuntimeV1:
        raise TypeError("terminal public dialogue runtime 非法")
    if type(prompts) is not bool:
        raise TypeError("terminal prompts 必须是严格 bool")
    if not hasattr(input_stream, "readline") or not hasattr(output_stream, "write"):
        raise TypeError("terminal streams 必须提供 binary readline/write")
    state = start_public_mixed_focus_dialogue(_TERMINAL_CONVERSATION_KEY)
    preparation_cache = PublicCoursePreparationCache()
    preflight_cache = AliasRelationPreflightCache()
    while True:
        if prompts:
            output_stream.write(_USER_PROMPT)
            output_stream.flush()
        raw, budget_exceeded = _read_bounded_line(input_stream)
        if raw == b"" or raw in _QUIT_LINES:
            return
        turn = run_public_mixed_focus_dialogue_turn_v1(
            state,
            tuple(raw),
            runtime,
            preparation_cache=preparation_cache,
            preflight_cache=preflight_cache,
        )
        state = turn.after
        result = (turn.answer if turn.answer is not None
                  else (turn.provider_answer if turn.provider_answer is not None
                        else turn.provider_followup_answer))
        if result is None:
            raise RuntimeError("DLG-RAW-12 turn 缺少回答 carrier")
        output_stream.write(_SYSTEM_PROMPT)
        output_stream.write(render_public_frame_dialogue_terminal_result(result))
        output_stream.write(b"\n")
        output_stream.flush()
        if budget_exceeded:
            return


def run_public_terminal_dialogue_act_terminal(
        runtime: TerminalDialogueActRuntimeV1,
        input_stream: BinaryIO,
        output_stream: BinaryIO,
        *,
        prompts: bool = True,
        ) -> None:
    """运行默认 DLG-RAW-13 shell；所有语言输出均来自统一 response record。"""
    if type(runtime) is not TerminalDialogueActRuntimeV1:
        raise TypeError("terminal dialogue act runtime 类型错误")
    if type(prompts) is not bool:
        raise TypeError("terminal prompts 必须是严格 bool")
    if not hasattr(input_stream, "readline") or not hasattr(output_stream, "write"):
        raise TypeError("terminal streams 必须提供 binary readline/write")
    state = start_public_terminal_dialogue_act(_TERMINAL_CONVERSATION_KEY)
    preparation_cache = PublicCoursePreparationCache()
    preflight_cache = AliasRelationPreflightCache()
    while True:
        if prompts:
            output_stream.write(_USER_PROMPT)
            output_stream.flush()
        raw, budget_exceeded = _read_bounded_line(input_stream)
        if raw == b"" or raw in _QUIT_LINES:
            return
        turn = run_public_terminal_dialogue_act_turn_v1(
            state,
            tuple(raw),
            runtime,
            preparation_cache=preparation_cache,
            preflight_cache=preflight_cache,
        )
        state = turn.after
        output_stream.write(_SYSTEM_PROMPT)
        output_stream.write(render_terminal_dialogue_response_v1(turn.response))
        output_stream.write(b"\n")
        output_stream.flush()
        if budget_exceeded:
            return


def run_public_route_clarification_dialogue_terminal(
        runtime: RouteClarificationDialogueRuntimeV1,
        input_stream: BinaryIO,
        output_stream: BinaryIO,
        *,
        prompts: bool = True,
        skip_initial_prompt: bool = False,
        fallback_runtime_factory: Callable[[], RouteClarificationDialogueRuntimeV1]
        | None = None,
        ) -> None:
    """运行默认 DLG-RAW-14 shell；每行仍先由 DLG-RAW-13 消费一次。"""
    if type(runtime) is not RouteClarificationDialogueRuntimeV1:
        raise TypeError("route clarification dialogue runtime 类型错误")
    if type(prompts) is not bool:
        raise TypeError("terminal prompts 必须是严格 bool")
    if type(skip_initial_prompt) is not bool:
        raise TypeError("terminal skip_initial_prompt 必须是严格 bool")
    if (fallback_runtime_factory is not None
            and not callable(fallback_runtime_factory)):
        raise TypeError("terminal fallback_runtime_factory 必须可调用")
    if not hasattr(input_stream, "readline") or not hasattr(output_stream, "write"):
        raise TypeError("terminal streams 必须提供 binary readline/write")
    state = start_public_route_clarification_dialogue(_TERMINAL_CONVERSATION_KEY)
    preparation_cache = PublicCoursePreparationCache()
    preflight_cache = AliasRelationPreflightCache()
    active_runtime = runtime
    provider_runtime_loaded = (
        active_runtime.terminal_runtime.inner_runtime.proof_sentence_provider
        is not None)
    first_iteration = True
    while True:
        if prompts and not (first_iteration and skip_initial_prompt):
            output_stream.write(_USER_PROMPT)
            output_stream.flush()
        raw, budget_exceeded = _read_bounded_line(input_stream)
        if raw == b"" or raw in _QUIT_LINES:
            return
        first_iteration = False
        turn = run_public_route_clarification_dialogue_turn_v1(
            state,
            tuple(raw),
            active_runtime,
            preparation_cache=preparation_cache,
            preflight_cache=preflight_cache,
        )
        # 冷启动不预建 provider-origin course。只有 fast runtime 的真实 RAW-01
        # lexical miss 才按需构建完整 provider runtime，并从同一 before state
        # 重放这一轮；任何已接受回答、歧义或 pending selection 都不走此分支。
        if (
                fallback_runtime_factory is not None
                and not provider_runtime_loaded
                and turn.response.response_kind
                == ROUTE_CLARIFICATION_RESPONSE_KIND_PASSTHROUGH_V1
                and turn.response.base_result_code
                == DLG_RAW_REJECT_LEXICAL_MISS
        ):
            candidate_runtime = fallback_runtime_factory()
            if type(candidate_runtime) is not RouteClarificationDialogueRuntimeV1:
                raise TypeError("terminal fallback runtime 类型错误")
            active_runtime = candidate_runtime
            provider_runtime_loaded = True
            turn = run_public_route_clarification_dialogue_turn_v1(
                state,
                tuple(raw),
                active_runtime,
                preparation_cache=preparation_cache,
                preflight_cache=preflight_cache,
            )
        state = turn.after
        output_stream.write(_SYSTEM_PROMPT)
        output_stream.write(
            render_route_clarification_dialogue_response_v1(turn.response))
        output_stream.write(b"\n")
        output_stream.flush()
        if budget_exceeded:
            return


def main(
        argv: list[str] | None = None,
        *,
        stdin: BinaryIO | None = None,
        stdout: BinaryIO | None = None,
        ) -> int:
    """加载冻结公开 closure 后启动 DLG-RAW-14；参数仅保留显式帮助入口。"""
    values = sys.argv[1:] if argv is None else argv
    if values:
        if values in (["-h"], ["--help"]):
            target = sys.stdout.buffer if stdout is None else stdout
            target.write(b"pure-integer-dialogue: public RAW-03 byte shell\n")
            target.flush()
            return 0
        raise SystemExit("pure-integer-dialogue does not accept arguments")
    input_stream = sys.stdin.buffer if stdin is None else stdin
    output_stream = sys.stdout.buffer if stdout is None else stdout
    # 先给 terminal 一个真实可见的 prompt，再做只读 closure/runtime 编译。
    # 这是 host 感知延迟优化，不改变任何 core transition 或验证顺序。
    output_stream.write(_USER_PROMPT)
    output_stream.flush()
    root = _default_public_dialogue_resource_root()
    source_payload_closure = load_public_source_payload_closure_from_root(root)
    # 首轮先使用不含 provider-origin course 的轻量 runtime；provider 仍由
    # terminal host 在真实 lexical miss 时按需构建并重放，不改变 core 语义。
    inner_runtime = build_public_dialogue_runtime_v1(source_payload_closure)
    runtime = build_public_route_clarification_dialogue_runtime_v1(inner_runtime)

    def build_provider_runtime() -> RouteClarificationDialogueRuntimeV1:
        provider = load_public_proof_sentence_provider_from_root(root)
        full_inner = build_public_dialogue_runtime_v1(
            source_payload_closure,
            proof_sentence_provider=provider,
        )
        return build_public_route_clarification_dialogue_runtime_v1(full_inner)

    run_public_route_clarification_dialogue_terminal(
        runtime,
        input_stream,
        output_stream,
        skip_initial_prompt=True,
        fallback_runtime_factory=build_provider_runtime,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "render_public_frame_dialogue_terminal_result",
    "render_route_clarification_dialogue_response_v1",
    "render_terminal_dialogue_response_v1",
    "run_public_frame_dialogue_terminal",
    "run_public_mixed_frame_dialogue_terminal",
    "run_public_route_clarification_dialogue_terminal",
    "run_public_terminal_dialogue_act_terminal",
    "main",
]
