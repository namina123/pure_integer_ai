"""公开 W03-W05 受限课程 raw-byte 完整句 demo 的定向验证。"""
from __future__ import annotations

from dataclasses import replace

import pytest

from pure_integer_ai.experiments.conversation_public_sentence_demo import (
    PUBLIC_SENTENCE_DEMO_ANSWER,
    PUBLIC_SENTENCE_DEMO_DISPATCH_NONE,
    PUBLIC_SENTENCE_DEMO_DISPATCH_UNKNOWN,
    PUBLIC_SENTENCE_DEMO_REJECT_LEXICAL_AMBIGUOUS,
    PUBLIC_SENTENCE_DEMO_REJECT_LEXICAL_MISS,
    PUBLIC_SENTENCE_DEMO_REJECT_RAW,
    PUBLIC_SENTENCE_DEMO_REJECT_RUNTIME_UNKNOWN,
    PublicSentenceDemoCatalog,
    PublicSentenceDemoError,
    build_public_sentence_demo_catalog,
    run_public_sentence_demo_bytes,
    run_public_sentence_demo_vector,
    run_public_sentence_demo_vector_with_typed_proof,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_sparse_qa_runtime import (
    build_public_sparse_qa_runtime,
    run_sparse_qa_sentence,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_sparse_qa_runtime_contract import (
    SparseQASameDispatchProofProjection,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_sparse_qa_snapshot import (
    load_public_sparse_qa_runtime_snapshot,
)


def _utf8_vector(surface: str, *, line_end: bool = False) -> tuple[int, ...]:
    """测试宿主边缘把文字交付为核心唯一接受的整数 byte vector。"""
    value = surface.encode("utf-8")
    if line_end:
        value += b"\r\n"
    return tuple(value)


@pytest.fixture(scope="module")
def runtime_and_catalog(tmp_path_factory):
    """一次性构建真实公开 FT22 runtime 与其派生受限课程目录。"""
    runtime = build_public_sparse_qa_runtime(
        tmp_path_factory.mktemp("public_sentence_demo"))
    return runtime, build_public_sentence_demo_catalog(runtime)


def test_catalog_is_real_public_course_projection(runtime_and_catalog) -> None:
    """目录只来自实际 exact/alias/implicit 学习对象，且保存整数本体。"""
    runtime, catalog = runtime_and_catalog
    assert catalog.runtime_identity_sha256 == runtime.identity_sha256
    assert len(catalog.routes) == 24
    assert {item.route_kind for item in catalog.routes} == {1, 2, 3}
    assert catalog.canonical_record()[0] == 1
    assert all(type(item) is int for item in catalog.canonical_record())
    assert all(item.request.source_record_key == item.source_record_key
               for item in catalog.routes)


def test_vector_answer_is_the_actual_selected_proof_sentence(
        runtime_and_catalog) -> None:
    """ANSWER 不套短答案模板，只返回同次 W03-W05 proof 的完整句。"""
    runtime, catalog = runtime_and_catalog
    route = next(item for item in catalog.routes if item.route_kind == 1)
    result = run_public_sentence_demo_vector(
        runtime,
        catalog,
        _utf8_vector(route.request.question_surface, line_end=True),
    )
    expected = run_sparse_qa_sentence(runtime, route.request)

    assert result.result_code == PUBLIC_SENTENCE_DEMO_ANSWER
    assert result.accepted is True
    assert result.intake.accepted is True
    assert result.intake.state_delta == ()
    assert result.selected_source_record_key == route.source_record_key
    assert result.generated_proposition_surface == (
        expected.generated_proposition_surface)
    assert result.generated_proposition_surface is not None
    assert result.output_bytes == tuple(
        result.generated_proposition_surface.encode("utf-8"))
    assert result.canonical_record()[0] == 1
    assert all(type(item) is int for item in result.canonical_record())


def test_host_only_demo_proof_api_uses_one_dispatch_and_preserves_result(
        monkeypatch) -> None:
    """新 host API 只保留一次 dispatch 的 proof，旧 demo 结果逐字节不变。"""
    from pure_integer_ai.experiments import (
        conversation_public_sentence_demo as demo_module,
    )

    runtime = load_public_sparse_qa_runtime_snapshot()
    catalog = build_public_sentence_demo_catalog(runtime)
    route = next(item for item in catalog.routes if item.route_kind == 1)
    payload = _utf8_vector(route.request.question_surface)
    calls = {"typed_proof_query": 0}
    original = demo_module.run_sparse_qa_query_with_typed_proof

    def counted_query(*args, **kwargs):
        calls["typed_proof_query"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        demo_module,
        "run_sparse_qa_query_with_typed_proof",
        counted_query,
    )
    paired = run_public_sentence_demo_vector_with_typed_proof(
        runtime,
        catalog,
        payload,
    )
    assert calls == {"typed_proof_query": 1}
    assert paired.host_adapter_only == 1
    assert paired.demo_result.result_code == PUBLIC_SENTENCE_DEMO_ANSWER
    assert paired.sparse_proof_projection is not None
    assert paired.sparse_proof_projection.typed_proof is not None
    assert paired.demo_result.generated_proposition_surface == (
        paired.sparse_proof_projection.generated_proposition_surface)
    assert paired.demo_result.output_bytes == tuple(
        paired.sparse_proof_projection.generated_proposition_surface.encode(
            "utf-8"))

    legacy = run_public_sentence_demo_vector(runtime, catalog, payload)
    assert calls == {"typed_proof_query": 2}
    assert legacy.canonical_record() == paired.demo_result.canonical_record()


def test_host_only_demo_proof_api_keeps_unknown_and_miss_zero_proof(
        monkeypatch) -> None:
    """UNKNOWN 可保留零-proof dispatch carrier，lexical miss 不进入 sparse runtime。"""
    from pure_integer_ai.experiments import (
        conversation_public_sentence_demo as demo_module,
    )

    runtime = load_public_sparse_qa_runtime_snapshot()
    catalog = build_public_sentence_demo_catalog(runtime)
    route = next(item for item in catalog.routes if item.route_kind == 1)
    answer = demo_module.run_sparse_qa_query_with_typed_proof(
        runtime,
        route.request,
    )
    unknown_result = replace(
        answer.query_result,
        status="UNKNOWN",
        answer_surface=None,
        decisive_phase=None,
        selected_entry_sha256=None,
        selected_source_record_key=None,
    )
    unknown_projection = SparseQASameDispatchProofProjection(
        unknown_result,
        None,
        None,
        None,
    )
    calls = {"typed_proof_query": 0}

    def unknown_query(*args, **kwargs):
        calls["typed_proof_query"] += 1
        return unknown_projection

    monkeypatch.setattr(
        demo_module,
        "run_sparse_qa_query_with_typed_proof",
        unknown_query,
    )
    unknown = run_public_sentence_demo_vector_with_typed_proof(
        runtime,
        catalog,
        _utf8_vector(route.request.question_surface),
    )
    assert calls == {"typed_proof_query": 1}
    assert unknown.demo_result.result_code == (
        PUBLIC_SENTENCE_DEMO_REJECT_RUNTIME_UNKNOWN)
    assert unknown.demo_result.dispatch_status_code == (
        PUBLIC_SENTENCE_DEMO_DISPATCH_UNKNOWN)
    assert unknown.demo_result.generated_proposition_surface is None
    assert unknown.sparse_proof_projection is unknown_projection
    assert unknown.sparse_proof_projection.typed_proof is None
    assert unknown.sparse_proof_projection.generated_proposition_surface is None

    def forbidden_query(*args, **kwargs):
        raise AssertionError("lexical miss entered sparse typed-proof dispatch")

    monkeypatch.setattr(
        demo_module,
        "run_sparse_qa_query_with_typed_proof",
        forbidden_query,
    )
    miss = run_public_sentence_demo_vector_with_typed_proof(
        runtime,
        catalog,
        _utf8_vector("未学习的公开问题？"),
    )
    assert miss.demo_result.result_code == PUBLIC_SENTENCE_DEMO_REJECT_LEXICAL_MISS
    assert miss.demo_result.generated_proposition_surface is None
    assert miss.sparse_proof_projection is None


def test_every_published_catalog_route_projects_one_actual_proof_sentence(
        runtime_and_catalog) -> None:
    """exact、alias、implicit 的全部公开课程路由均须回到真实 proof。"""
    runtime, catalog = runtime_and_catalog
    results = tuple(
        run_public_sentence_demo_vector(
            runtime,
            catalog,
            _utf8_vector(route.request.question_surface),
        )
        for route in catalog.routes
    )

    assert len(results) == len(catalog.routes)
    assert all(item.result_code == PUBLIC_SENTENCE_DEMO_ANSWER
               for item in results)
    assert all(item.generated_proposition_surface is not None
               for item in results)
    assert tuple(item.selected_source_record_key for item in results) == tuple(
        item.source_record_key for item in catalog.routes)


def test_unlearned_input_is_a_lexical_rejection_without_generated_text(
        runtime_and_catalog) -> None:
    """公开课程以外的可解码文本不得走自然语言 fallback。"""
    runtime, catalog = runtime_and_catalog
    result = run_public_sentence_demo_vector(
        runtime,
        catalog,
        _utf8_vector("未学习的公开问题？"),
    )

    assert result.result_code == PUBLIC_SENTENCE_DEMO_REJECT_LEXICAL_MISS
    assert result.generated_proposition_surface is None
    assert result.generated_proposition_scalars == result.output_bytes == ()
    assert result.dispatch_status_code == PUBLIC_SENTENCE_DEMO_DISPATCH_NONE


def test_ambiguous_catalog_input_is_rejected_without_selecting_a_route(
        runtime_and_catalog) -> None:
    """即使重复路由看似相同，也不得暗选其中一个来源或回答。"""
    runtime, catalog = runtime_and_catalog
    route = catalog.routes[0]
    ambiguous = PublicSentenceDemoCatalog(
        catalog.runtime_identity_sha256,
        tuple(sorted(
            (*catalog.routes, route),
            key=lambda item: item.canonical_record(),
        )),
    )
    result = run_public_sentence_demo_vector(
        runtime,
        ambiguous,
        _utf8_vector(route.request.question_surface),
    )

    assert result.result_code == PUBLIC_SENTENCE_DEMO_REJECT_LEXICAL_AMBIGUOUS
    assert result.matched_route_count == 2
    assert result.selected_source_record_key == ()
    assert result.generated_proposition_surface is None


def test_raw_rejection_precedes_catalog_and_never_produces_text(
        runtime_and_catalog) -> None:
    """DLG-RAW-00 失败不进入 lexical 或 W03-W05 运行时。"""
    runtime, catalog = runtime_and_catalog
    result = run_public_sentence_demo_vector(
        runtime,
        catalog,
        (0xEF, 0xBB, 0xBF, 0x61),
    )

    assert result.result_code == PUBLIC_SENTENCE_DEMO_REJECT_RAW
    assert result.intake.accepted is False
    assert result.generated_proposition_surface is None
    assert result.output_bytes == ()


def test_bytes_adapter_only_copies_to_the_integer_vector_core(
        runtime_and_catalog) -> None:
    """宿主 bytes adapter 与核心 vector 结果必须逐字段一致。"""
    runtime, catalog = runtime_and_catalog
    route = catalog.routes[0]
    payload = route.request.question_surface.encode("utf-8")
    assert run_public_sentence_demo_bytes(runtime, catalog, payload) == (
        run_public_sentence_demo_vector(runtime, catalog, tuple(payload)))
    with pytest.raises(TypeError):
        run_public_sentence_demo_vector(runtime, catalog, payload)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        run_public_sentence_demo_bytes(runtime, catalog, bytearray(payload))  # type: ignore[arg-type]


def test_catalog_runtime_identity_mismatch_fails_closed(runtime_and_catalog) -> None:
    """目录不能在另一 runtime 身份下静默复用。"""
    runtime, catalog = runtime_and_catalog
    other = replace(catalog, runtime_identity_sha256="0" * 64)
    assert other.canonical_record() == catalog.canonical_record()
    with pytest.raises(ValueError):
        run_public_sentence_demo_vector(
            runtime,
            other,
            _utf8_vector(catalog.routes[0].request.question_surface),
        )


def test_protocol_tags_reject_bool_and_rejections_carry_no_surface(
        runtime_and_catalog) -> None:
    """协议 tag 不能借 bool 伪装整数；拒绝 record 不能藏入空字符串。"""
    runtime, catalog = runtime_and_catalog
    route = next(item for item in catalog.routes if item.route_kind == 1)
    answer = run_public_sentence_demo_vector(
        runtime,
        catalog,
        _utf8_vector(route.request.question_surface),
    )
    miss = run_public_sentence_demo_vector(
        runtime,
        catalog,
        _utf8_vector("未学习的公开问题？"),
    )

    with pytest.raises(PublicSentenceDemoError):
        replace(route, route_kind=True)
    with pytest.raises(PublicSentenceDemoError):
        replace(answer, result_code=False)
    with pytest.raises(PublicSentenceDemoError):
        replace(answer, selected_route_kind=True)
    with pytest.raises(PublicSentenceDemoError):
        replace(answer, dispatch_status_code=True)
    with pytest.raises(PublicSentenceDemoError):
        replace(miss, generated_proposition_surface="")
