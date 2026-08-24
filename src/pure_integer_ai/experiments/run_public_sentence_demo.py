"""公开受限课程 raw-byte 完整命题句 demo 的二进制终端适配器。

本模块只负责终端 transport：stdin 的原始 bytes 交给 demo core，core 返回的
UTF-8 bytes 原样写到 stdout。它不保存历史、不参与语言理解、也不构造回答。
"""
from __future__ import annotations

import sys
from typing import BinaryIO

from pure_integer_ai.experiments.conversation_public_sentence_demo import (
    PUBLIC_SENTENCE_DEMO_ANSWER,
    PublicSentenceDemoCatalog,
    PublicSentenceDemoResult,
    build_public_sentence_demo_catalog,
    run_public_sentence_demo_bytes,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_sparse_qa_runtime_contract import (
    SparseQARuntime,
)
from pure_integer_ai.experiments.conversation_raw_intake import (
    DLG_RAW_MAX_INPUT_BYTES,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_sparse_qa_snapshot import (
    load_or_rebuild_public_sparse_qa_runtime,
)


_USER_PROMPT = b"\xe4\xbd\xa0> "
_SYSTEM_PROMPT = b"\xe7\xb3\xbb\xe7\xbb\x9f> "
_TERMINAL_READ_LIMIT = DLG_RAW_MAX_INPUT_BYTES + 1
_QUIT_LINES = frozenset({
    b":quit",
    b":quit\n",
    b":quit\r\n",
    b":exit",
    b":exit\n",
    b":exit\r\n",
})


def _ascii_nonnegative_integer(value: int) -> bytes:
    """以显式十进制整数规则生成 transport 状态文本，拒绝宿主格式化语义。"""
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
    """在 terminal 边缘复制规范 u8 vector；core 不依赖 Python bytes。"""
    if (not isinstance(value, tuple)
            or any(type(item) is not int or item < 0 or item > 255
                   for item in value)):
        raise ValueError(f"{label} 不是规范 u8 vector")
    return bytes(value)


def _read_bounded_line(input_stream: BinaryIO) -> tuple[bytes, bool]:
    """读取一条有界 raw line，超过预算时交给 core 拒绝并结束 shell。

    ``readline(limit)`` 至少读取第 4097 个 byte，从而让 DLG-RAW-00
    产生明确的 input-budget rejection；调用方随后终止物理会话，不截断或
    重解释同一行的剩余 payload。
    """
    raw = input_stream.readline(_TERMINAL_READ_LIMIT)
    if type(raw) is not bytes:
        raise TypeError("terminal input 必须返回 bytes")
    return raw, len(raw) >= _TERMINAL_READ_LIMIT


def render_public_sentence_demo_terminal_result(
        result: PublicSentenceDemoResult,
        ) -> bytes:
    """将已完成 core record 投影为终端单行，不添加任何自然语言 fallback。"""
    if not isinstance(result, PublicSentenceDemoResult):
        raise TypeError("terminal result 必须来自公开课程 demo core")
    if result.result_code == PUBLIC_SENTENCE_DEMO_ANSWER:
        if not result.output_bytes:
            raise ValueError("公开课程 demo ANSWER 缺少 output bytes")
        return _host_bytes(
            result.output_bytes,
            label="公开课程 demo terminal output",
        )
    return b"[REJECT:" + _ascii_nonnegative_integer(result.result_code) + b"]"


def render_public_sentence_demo_question(
        runtime: SparseQARuntime,
        catalog: PublicSentenceDemoCatalog,
        question_surface: str,
        ) -> bytes:
    """将 argv 问句显式编码后交给 raw-byte core，返回同次结果 bytes。

    这是 host-only 的一次性入口；不做 trim、归一化、猜测或自然语言 fallback，
    因此不会把 Python 字符串语义带入可迁移核心。
    """
    if not isinstance(runtime, SparseQARuntime):
        raise TypeError("公开课程 demo runtime 非法")
    if not isinstance(catalog, PublicSentenceDemoCatalog):
        raise TypeError("公开课程 demo catalog 非法")
    if type(question_surface) is not str:
        raise TypeError("一次性问句必须是宿主 str")
    try:
        raw_question = question_surface.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("一次性问句含非法 Unicode scalar") from error
    result = run_public_sentence_demo_bytes(
        runtime,
        catalog,
        raw_question,
    )
    return render_public_sentence_demo_terminal_result(result)


def _question_argument(values: list[str]) -> str | None:
    """解析唯一的一次性 argv 形式，不引入 argparse 或宿主语义。"""
    if len(values) == 2 and values[0] == "--question":
        return values[1]
    if len(values) == 1 and values[0].startswith("--question="):
        return values[0][len("--question="):]
    return None


def run_public_sentence_demo_terminal(
        runtime: SparseQARuntime,
        catalog: PublicSentenceDemoCatalog,
        input_stream: BinaryIO,
        output_stream: BinaryIO,
        *,
        prompts: bool = True,
        skip_initial_prompt: bool = False,
        ) -> None:
    """运行无历史的 byte-line shell；每一行均是独立 raw transport record。"""
    if not isinstance(runtime, SparseQARuntime):
        raise TypeError("terminal runtime 非法")
    if not isinstance(catalog, PublicSentenceDemoCatalog):
        raise TypeError("terminal catalog 非法")
    if type(prompts) is not bool:
        raise TypeError("terminal prompts 必须是严格 bool")
    if type(skip_initial_prompt) is not bool:
        raise TypeError("terminal skip_initial_prompt 必须是严格 bool")
    if not hasattr(input_stream, "readline") or not hasattr(output_stream, "write"):
        raise TypeError("terminal streams 必须提供 binary readline/write")
    first_iteration = True
    while True:
        if prompts and not (first_iteration and skip_initial_prompt):
            output_stream.write(_USER_PROMPT)
            output_stream.flush()
        raw, budget_exceeded = _read_bounded_line(input_stream)
        if raw == b"":
            return
        if raw in _QUIT_LINES:
            return
        first_iteration = False
        result = run_public_sentence_demo_bytes(runtime, catalog, raw)
        output_stream.write(_SYSTEM_PROMPT)
        output_stream.write(render_public_sentence_demo_terminal_result(result))
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
    """加载一次公开 snapshot 后启动二进制交互；参数仅保留显式帮助入口。"""
    values = sys.argv[1:] if argv is None else argv
    if values:
        if values in (['-h'], ['--help']):
            target = sys.stdout.buffer if stdout is None else stdout
            target.write(
                b"pure-integer-dialogue-demo: public read-only course shell\n")
            target.flush()
            return 0
        question_surface = _question_argument(values)
        if question_surface is None:
            raise SystemExit(
                "pure-integer-dialogue-demo accepts --question <UTF-8 text>")
        output_stream = sys.stdout.buffer if stdout is None else stdout
        runtime = load_or_rebuild_public_sparse_qa_runtime()
        catalog = build_public_sentence_demo_catalog(runtime)
        output_stream.write(_SYSTEM_PROMPT)
        output_stream.write(render_public_sentence_demo_question(
            runtime,
            catalog,
            question_surface,
        ))
        output_stream.write(b"\n")
        output_stream.flush()
        return 0
    input_stream = sys.stdin.buffer if stdin is None else stdin
    output_stream = sys.stdout.buffer if stdout is None else stdout
    # 先给 terminal 一个真实可见的 prompt，再构建只读 runtime；这是 host
    # 感知延迟优化，不改变 core transition、验证顺序或回答语义。
    output_stream.write(_USER_PROMPT)
    output_stream.flush()
    runtime = load_or_rebuild_public_sparse_qa_runtime()
    catalog = build_public_sentence_demo_catalog(runtime)
    run_public_sentence_demo_terminal(
        runtime,
        catalog,
        input_stream,
        output_stream,
        skip_initial_prompt=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
