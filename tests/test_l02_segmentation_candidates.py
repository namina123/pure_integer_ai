from __future__ import annotations

import pytest

from pure_integer_ai.cognition.shared.hypothesis import (
    EPISTEMIC_UNKNOWN,
    EVIDENCE_REFUTE,
    LIFECYCLE_ACTIVE,
    LIFECYCLE_SUPERSEDED,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    SourceRef,
    VersionBundle,
    minimal_instruction_identity,
)
from pure_integer_ai.cognition.shared.language_object_index import (
    LanguageObjectIndex,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.shared.types import LANG_ZH
from pure_integer_ai.cognition.understanding.segmentation_hypothesis import (
    SegmentationProtocol,
)
from pure_integer_ai.cognition.understanding.word_form_provider import (
    VisibleWordForm,
    WordFormProvider,
    WordFormProviderRegistry,
)
from pure_integer_ai.experiments.collection import CollectedItem
from pure_integer_ai.experiments.evaluation_isolation import isolated_evaluation
from pure_integer_ai.experiments.formal_train import (
    _apply_word_form_providers,
    make_train_context,
)
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import SOURCE_BARE_TEXT

_HYPOTHESIS_KEY = (92001,)
_LEXICAL_REASON_KEY = (92002,)
_OOV_REASON_KEY = (92003,)
_REFUTE_REASON_KEY = (92004,)


def _source(document_id: int, *, source_id: int = 501) -> SourceRef:
    """构造分词观察和反馈使用的稳定来源。"""
    return SourceRef(
        SOURCE_BARE_TEXT,
        source_id,
        document_id,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(),
    )


def _provider(words: tuple[str, ...], *, candidate_limit: int = 16):
    """构造带图内协议符号和课程词形的多候选 provider。"""
    backend = DictBackend()
    ctx = make_train_context(backend)
    branch = LanguageObjectIndex(ctx.graph_ontology).ensure_branch((91001,))
    catalog = {
        word: VisibleWordForm(
            _source(index + 1, source_id=601),
            1,
            SOURCE_BARE_TEXT,
            1,
            1,
        )
        for index, word in enumerate(words)
    }
    protocol = SegmentationProtocol(
        _HYPOTHESIS_KEY,
        _LEXICAL_REASON_KEY,
        _OOV_REASON_KEY,
        candidate_limit,
    )
    provider = WordFormProvider(
        backend=backend,
        concept_index=ctx.concept_index,
        ontology=ctx.graph_ontology,
        branch=branch,
        runtime_language=LANG_ZH,
        unicode_family_key=(91002,),
        inventory_relation_key=(91003,),
        catalog_identity=(1, 1, 1),
        catalog=catalog,
        segmentation_protocol=protocol,
    )
    registry = WordFormProviderRegistry()
    registry.register(provider)
    return ctx, provider, registry


def _parse(provider: WordFormProvider, text: str, *, document_id: int = 1):
    """按来源化 document scope 解析一段测试文本。"""
    observation = _source(document_id)
    return provider.parse_text(
        text,
        observation=observation,
        scope=document_scope(observation),
    )


def test_unknown_run_is_preserved_with_character_fallback_candidate():
    """未登录多字串当前 winner 保持连续，同时保留字符回退而不永久逐字化。"""
    _ctx, provider, _registry = _provider(("已知",))
    result = _parse(provider, "陌生连续串")

    assert result is not None
    assert result.tokens == ("陌生连续串",)
    assert len(result.candidates) >= 2
    assert tuple("陌生连续串") in {
        candidate.segmentation.tokens for candidate in result.candidates}
    assert result.selected is not None
    assert result.selected.snapshot.epistemic_status == EPISTEMIC_UNKNOWN


def test_required_baselines_survive_small_budget_and_dense_lattice():
    """候选预算最小时，密集词形命中也不能裁掉 FMM 和全字符双基线。"""
    _ctx, provider, _registry = _provider((
        "甲乙丙丁",
        "甲乙丙",
        "甲乙",
        "甲",
        "乙丙丁",
        "乙丙",
        "乙",
        "丙丁",
        "丙",
        "丁",
    ), candidate_limit=3)
    result = _parse(provider, "甲乙丙丁")

    assert result is not None
    tokenizations = {
        candidate.segmentation.tokens for candidate in result.candidates
    }
    assert ("甲乙丙丁",) in tokenizations
    assert ("甲", "乙", "丙", "丁") in tokenizations
    assert len(result.candidates) == 3


def test_whitespace_only_input_has_empty_projection_without_fake_candidate():
    """空白输入保持零 token，不制造边界候选，也不因启用协议而异常。"""
    _ctx, provider, _registry = _provider(("已知",))
    result = _parse(provider, " \t\n")

    assert result is not None
    assert result.candidates == ()
    assert result.tokens == ()


def test_long_word_and_short_word_paths_are_both_retained_without_frequency():
    """长词不能吞掉短词组合；候选排序不读取词频。"""
    _ctx, provider, _registry = _provider((
        "研究生命",
        "研究",
        "生命",
    ))
    result = _parse(provider, "研究生命")

    assert result is not None
    tokenizations = {
        candidate.segmentation.tokens for candidate in result.candidates}
    assert ("研究生命",) in tokenizations
    assert ("研究", "生命") in tokenizations
    assert result.tokens == ("研究生命",)
    assert len(tokenizations) == len(result.candidates)


def test_refuted_boundary_exits_consumer_but_keeps_append_only_history():
    """针对性反例 supersede 错误边界，旧候选和 Evidence 仍可审计。"""
    ctx, provider, _registry = _provider((
        "研究生命",
        "研究",
        "生命",
    ))
    first = _parse(provider, "研究生命")
    assert first is not None and first.selected is not None
    old = first.selected.hypothesis
    replacement = next(
        candidate.hypothesis for candidate in first.candidates
        if candidate.segmentation.tokens == ("研究", "生命"))

    feedback_source = _source(99, source_id=701)
    snapshot = provider.record_segmentation_feedback(
        old,
        stance=EVIDENCE_REFUTE,
        source=feedback_source,
        reason_key=_REFUTE_REASON_KEY,
        timestamp_seq=100,
        replacement=replacement,
    )
    assert snapshot.lifecycle == LIFECYCLE_SUPERSEDED
    assert ctx.graph_ontology.resolve(
        minimal_instruction_identity(_REFUTE_REASON_KEY)) is not None

    second = _parse(provider, "研究生命")
    assert second is not None
    assert second.tokens == ("研究", "生命")
    assert old not in {
        candidate.hypothesis for candidate in second.consumer_candidates}
    assert len(provider.segmentation_evidence_history(old)) == 2
    history = provider.segmentation_resolution_history(old)
    assert len(history) >= 2
    assert history[-1].candidate(old).transition_event_id > 0
    assert history[-1].previous_decision_id == history[-2].decision_id


def test_protocol_keys_are_materialized_as_injected_minimal_instructions():
    """分词 kind 和初始 Evidence 理由只由注入键定义，并在图内成为协议符号。"""
    ctx, _provider_instance, _registry = _provider(("已知",))

    for instruction_key in (
            _HYPOTHESIS_KEY, _LEXICAL_REASON_KEY, _OOV_REASON_KEY):
        assert ctx.graph_ontology.resolve(
            minimal_instruction_identity(instruction_key)) is not None


def test_formal_adapter_keeps_full_parse_beside_compatibility_tokens():
    """正式语料适配保存完整候选，tokens 只投影当前 winner。"""
    _ctx, _provider_instance, registry = _provider((
        "南京市",
        "南京",
        "市长",
    ))
    observation = _source(1)
    item = CollectedItem(
        tokens=["南京市长"],
        raw_text="南京市长",
        source=SOURCE_BARE_TEXT,
        source_ref=observation,
    )

    assert _apply_word_form_providers([item], registry) == 1
    assert item.word_form_parse is not None
    assert tuple(item.tokens) == item.word_form_parse.tokens
    assert len(item.word_form_parse.candidates) >= 2
    assert {
        candidate.segmentation.tokens
        for candidate in item.word_form_parse.candidates
    } >= {("南京市", "长"), ("南京", "市长")}


def test_probe_preview_does_not_write_formal_hypothesis_ledger():
    """split 前候选预览可生成 token/signature，但不把 probe Evidence 写入正式状态。"""
    _ctx, provider, registry = _provider(("研究生命", "研究", "生命"))
    observation = _source(8)
    item = CollectedItem(
        tokens=["研究生命"],
        raw_text="研究生命",
        source=SOURCE_BARE_TEXT,
        source_ref=observation,
    )

    _apply_word_form_providers(
        [item], registry, commit_evidence=False)
    assert item.word_form_parse is not None
    preview_hypothesis = item.word_form_parse.candidates[0].hypothesis
    with pytest.raises(KeyError, match="尚未登记"):
        provider.segmentation_snapshot(preview_hypothesis)

    _apply_word_form_providers(
        [item], registry, commit_evidence=True)
    assert provider.segmentation_snapshot(
        item.word_form_parse.candidates[0].hypothesis)


def test_same_protocol_and_source_produce_identical_candidate_keys():
    """候选身份和排序不依赖 Python 哈希随机化或登记顺序。"""
    _ctx1, provider1, _registry1 = _provider((
        "南京市",
        "南京",
        "市长",
    ))
    _ctx2, provider2, _registry2 = _provider((
        "市长",
        "南京",
        "南京市",
    ))
    first = _parse(provider1, "南京市长")
    second = _parse(provider2, "南京市长")

    assert first is not None and second is not None
    assert tuple(
        candidate.hypothesis.stable_key() for candidate in first.candidates
    ) == tuple(
        candidate.hypothesis.stable_key() for candidate in second.candidates)
    assert first.tokens == second.tokens


def test_evaluation_clone_feedback_does_not_mutate_host_ledger():
    """V-06 沙箱可修正分词候选，但宿主 ledger 和生命周期保持不变。"""
    ctx, provider, registry = _provider((
        "研究生命",
        "研究",
        "生命",
    ))
    ctx.word_form_providers = registry
    first = _parse(provider, "研究生命")
    assert first is not None and first.selected is not None
    old = first.selected.hypothesis
    replacement = next(
        candidate.hypothesis for candidate in first.candidates
        if candidate.segmentation.tokens == ("研究", "生命"))
    host_state = provider.segmentation_state()

    with isolated_evaluation(ctx, label="l02-feedback") as cloned:
        cloned_provider = cloned.word_form_providers.provider(LANG_ZH)
        assert cloned_provider is not None
        cloned_snapshot = cloned_provider.record_segmentation_feedback(
            old,
            stance=EVIDENCE_REFUTE,
            source=_source(99, source_id=702),
            reason_key=_REFUTE_REASON_KEY,
            timestamp_seq=100,
            replacement=replacement,
        )
        assert cloned_snapshot.lifecycle == LIFECYCLE_SUPERSEDED

    assert provider.segmentation_snapshot(old).lifecycle == LIFECYCLE_ACTIVE
    assert provider.segmentation_state() == host_state
