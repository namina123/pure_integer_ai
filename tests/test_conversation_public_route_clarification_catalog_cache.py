"""DLG-RAW-14 selector catalog derived-cache contract tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments import conversation_public_route_clarification_catalog as catalog_module
from pure_integer_ai.experiments.conversation_public_dialogue_runtime import (
    build_public_dialogue_runtime_v1,
)
from pure_integer_ai.experiments.conversation_public_proof_sentence_provider import (
    load_public_proof_sentence_provider_from_root,
)
from pure_integer_ai.experiments.conversation_public_source_payload_host import (
    load_public_source_payload_closure_from_root,
)
from pure_integer_ai.experiments.conversation_public_route_clarification_catalog import (
    PublicRouteClarificationCatalogError,
    PublicRouteClarificationCatalogValidationCacheV1,
    validate_public_route_clarification_catalog_cached_v1,
)
from pure_integer_ai.experiments.conversation_raw_route_clarification_dialogue import (
    build_public_route_clarification_dialogue_runtime_v1,
)


_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def runtime():
    closure = load_public_source_payload_closure_from_root(_ROOT)
    inner = build_public_dialogue_runtime_v1(
        closure,
        proof_sentence_provider=load_public_proof_sentence_provider_from_root(
            _ROOT),
    )
    return build_public_route_clarification_dialogue_runtime_v1(inner)


def test_catalog_cache_reuses_validated_derivation_and_clear(
        runtime,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """同一显式 key 第二次不重跑 JSONL 编译，clear 后重新编译。"""
    cache = PublicRouteClarificationCatalogValidationCacheV1()
    calls = 0
    original = catalog_module.load_public_route_clarification_catalog_from_closure

    def counted(closure):
        nonlocal calls
        calls += 1
        return original(closure)

    monkeypatch.setattr(
        catalog_module,
        "load_public_route_clarification_catalog_from_closure",
        counted,
    )
    first = validate_public_route_clarification_catalog_cached_v1(
        runtime.selector_catalog,
        runtime.terminal_runtime.inner_runtime.source_payload_closure,
        cache,
    )
    second = validate_public_route_clarification_catalog_cached_v1(
        runtime.selector_catalog,
        runtime.terminal_runtime.inner_runtime.source_payload_closure,
        cache,
    )
    assert calls == 1
    assert first.canonical_record() == second.canonical_record()
    assert len(cache.entries) == 1
    cache.clear()
    assert cache.entries == ()
    validate_public_route_clarification_catalog_cached_v1(
        runtime.selector_catalog,
        runtime.terminal_runtime.inner_runtime.source_payload_closure,
        cache,
    )
    assert calls == 2


def test_catalog_cache_does_not_hide_catalog_drift(runtime) -> None:
    """selector catalog 明文篡改必须 miss 并沿原 validator fail closed。"""
    catalog = runtime.selector_catalog
    form = catalog.forms[0]
    original_output = form.output_u8
    tampered = tuple(original_output[:-1]) + ((original_output[-1] ^ 1),)
    object.__setattr__(form, "output_u8", tampered)
    try:
        with pytest.raises((PublicRouteClarificationCatalogError, TypeError, ValueError)):
            validate_public_route_clarification_catalog_cached_v1(
                catalog,
                runtime.terminal_runtime.inner_runtime.source_payload_closure,
                runtime.validation_cache,
            )
    finally:
        object.__setattr__(form, "output_u8", original_output)


def test_catalog_cache_does_not_hide_closure_payload_drift(runtime) -> None:
    """closure raw payload 篡改不能借旧 entry 继续提供 selector capability。"""
    closure = runtime.terminal_runtime.inner_runtime.source_payload_closure
    route_index = next(
        index for index, record in enumerate(closure.records)
        if record.logical_key.endswith(
            b"dlg_raw_public_route_clarification_course_v1.jsonl.sample"))
    record = closure.records[route_index]
    original_payload = record.raw_payload
    object.__setattr__(record, "raw_payload", original_payload + b" ")
    try:
        with pytest.raises((PublicRouteClarificationCatalogError, TypeError, ValueError)):
            validate_public_route_clarification_catalog_cached_v1(
                runtime.selector_catalog,
                closure,
                runtime.validation_cache,
            )
    finally:
        object.__setattr__(record, "raw_payload", original_payload)
