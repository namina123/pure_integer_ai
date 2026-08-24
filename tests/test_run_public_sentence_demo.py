"""公开 raw-byte 完整句终端适配器的定向验证。"""
from __future__ import annotations

from io import BytesIO

import pytest

from pure_integer_ai.experiments.conversation_public_sentence_demo import (
    PUBLIC_SENTENCE_DEMO_ANSWER,
    build_public_sentence_demo_catalog,
    run_public_sentence_demo_vector,
)
from pure_integer_ai.experiments.conversation_raw_intake import (
    DLG_RAW_MAX_INPUT_BYTES,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_sparse_qa_runtime import (
    build_public_sparse_qa_runtime,
)
from pure_integer_ai.experiments.run_public_sentence_demo import (
    main,
    render_public_sentence_demo_terminal_result,
    render_public_sentence_demo_question,
    run_public_sentence_demo_terminal,
)


@pytest.fixture(scope="module")
def runtime_and_catalog(tmp_path_factory):
    """构建一次真实公开课程 runtime 和只读 raw-byte catalog。"""
    runtime = build_public_sparse_qa_runtime(
        tmp_path_factory.mktemp("public_sentence_terminal"))
    return runtime, build_public_sentence_demo_catalog(runtime)


def test_terminal_writes_the_core_output_bytes_without_text_reencoding(
        runtime_and_catalog) -> None:
    """终端 ANSWER 必须逐字节复用 demo core 输出。"""
    runtime, catalog = runtime_and_catalog
    route = catalog.routes[0]
    result = run_public_sentence_demo_vector(
        runtime,
        catalog,
        tuple(route.request.question_surface.encode("utf-8")),
    )
    assert result.result_code == PUBLIC_SENTENCE_DEMO_ANSWER
    assert render_public_sentence_demo_terminal_result(result) == (
        bytes(result.output_bytes))


def test_terminal_does_not_retain_prior_line_as_conversation_memory(
        runtime_and_catalog) -> None:
    """每一行独立进入 core，未学习第二行只能得到 rejection。"""
    runtime, catalog = runtime_and_catalog
    route = catalog.routes[0]
    incoming = BytesIO(
        route.request.question_surface.encode("utf-8")
        + b"\n"
        + "它呢？".encode("utf-8")
        + b"\n:quit\n")
    outgoing = BytesIO()

    run_public_sentence_demo_terminal(
        runtime,
        catalog,
        incoming,
        outgoing,
        prompts=False,
    )

    lines = outgoing.getvalue().splitlines()
    first = run_public_sentence_demo_vector(
        runtime,
        catalog,
        tuple(route.request.question_surface.encode("utf-8") + b"\n"),
    )
    assert lines[0] == b"\xe7\xb3\xbb\xe7\xbb\x9f> " + bytes(
        first.output_bytes)
    assert lines[1] == b"\xe7\xb3\xbb\xe7\xbb\x9f> [REJECT:2]"


def test_terminal_rejects_non_binary_stream_output(runtime_and_catalog) -> None:
    """宿主边缘不允许 TextIO 悄然参与 raw-byte transport。"""
    runtime, catalog = runtime_and_catalog
    with pytest.raises(TypeError):
        run_public_sentence_demo_terminal(
            runtime,
            catalog,
            BytesIO(b":quit\n"),
            object(),
        )


def test_terminal_bounds_oversized_line_and_closes_session(
        runtime_and_catalog) -> None:
    """宿主读取不能在 RAW-00 预算前无界吸收一整行。"""
    runtime, catalog = runtime_and_catalog
    outgoing = BytesIO()
    run_public_sentence_demo_terminal(
        runtime,
        catalog,
        BytesIO(b"x" * (DLG_RAW_MAX_INPUT_BYTES + 1) + b"\n:quit\n"),
        outgoing,
        prompts=False,
    )
    assert outgoing.getvalue() == b"\xe7\xb3\xbb\xe7\xbb\x9f> [REJECT:1]\n"


def test_terminal_can_resume_after_host_printed_initial_prompt(
        runtime_and_catalog) -> None:
    """main 的预先 prompt 不应导致 terminal loop 重复打印首个 prompt。"""
    runtime, catalog = runtime_and_catalog
    route = catalog.routes[0]
    outgoing = BytesIO()
    run_public_sentence_demo_terminal(
        runtime,
        catalog,
        BytesIO(route.request.question_surface.encode("utf-8") + b"\n:quit\n"),
        outgoing,
        prompts=True,
        skip_initial_prompt=True,
    )
    output = outgoing.getvalue()
    assert output.startswith(b"\xe7\xb3\xbb\xe7\xbb\x9f> ")
    assert output.count(b"\xe4\xbd\xa0> ") == 1


def test_argv_question_adapter_preserves_core_answer_bytes(
        runtime_and_catalog) -> None:
    """一次性 argv 入口显式 UTF-8 编码，结果逐字节等于 core。"""
    runtime, catalog = runtime_and_catalog
    route = catalog.routes[0]
    expected = run_public_sentence_demo_vector(
        runtime,
        catalog,
        tuple(route.request.question_surface.encode("utf-8")),
    )
    assert render_public_sentence_demo_question(
        runtime,
        catalog,
        route.request.question_surface,
    ) == bytes(expected.output_bytes)


def test_main_accepts_one_shot_question_without_text_pipe(
        runtime_and_catalog, monkeypatch) -> None:
    """--question 只在 host 边界接收 str，不改变 raw-byte core。"""
    runtime, catalog = runtime_and_catalog
    route = catalog.routes[0]
    import pure_integer_ai.experiments.run_public_sentence_demo as module

    monkeypatch.setattr(
        module,
        "load_or_rebuild_public_sparse_qa_runtime",
        lambda: runtime,
    )
    monkeypatch.setattr(
        module,
        "build_public_sentence_demo_catalog",
        lambda value: catalog,
    )
    outgoing = BytesIO()
    assert main(
        ["--question", route.request.question_surface],
        stdout=outgoing,
    ) == 0
    assert outgoing.getvalue() == (
        b"\xe7\xb3\xbb\xe7\xbb\x9f> "
        + bytes(run_public_sentence_demo_vector(
            runtime,
            catalog,
            tuple(route.request.question_surface.encode("utf-8")),
        ).output_bytes)
        + b"\n"
    )


def test_main_rejects_unknown_argument_shape() -> None:
    """一次性入口不接受未注册的参数变体。"""
    with pytest.raises(SystemExit):
        main(["--question", "a", "extra"], stdout=BytesIO())
