"""T1-G4 source-qualified obligation 到 DLG-RAW-16 shadow 的窄验证。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments.conversation_dlg_raw16_surface_organization_compile import (
    load_surface_organization_jsonl,
)
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_selector import (
    SurfaceSemantic,
)
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_shadow import (
    SHADOW_SELECTED,
    SurfaceShadowPlan,
)
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_structure_learner import (
    load_surface_evidence_jsonl,
    learn_surface_structure_model,
)
from pure_integer_ai.experiments.conversation_raw_lexical_evidence import (
    load_raw_lexical_evidence_jsonl,
)
from pure_integer_ai.experiments.conversation_raw_proposition_consumer import (
    RawPropositionQualification,
    consume_raw_proposition_relation,
)
from pure_integer_ai.experiments.conversation_raw_proposition_evidence import (
    bind_raw_proposition_relation,
    load_raw_proposition_jsonl,
)
from pure_integer_ai.experiments.conversation_raw_t1_shadow_adapter import (
    RawT1ShadowAdapterError,
    run_raw_t1_shadow_adapter,
)
from pure_integer_ai.experiments.conversation_raw_text_observation import (
    load_raw_text_observation_jsonl,
)
from pure_integer_ai.experiments.ph2_dataset_core import parse_canonical_json_bytes
from pure_integer_ai.experiments.conversation_raw_proposition_consumer import (
    load_raw_qualification_jsonl,
)


_ROOT = Path(__file__).resolve().parents[1]
_COURSE = _ROOT / "data/ph2/dlg_raw16_surface_organization_v1.jsonl.sample"
_SURFACE_EVIDENCE = _ROOT / "data/ph2/dlg_raw16_surface_slot_evidence_v1.jsonl.sample"
_OBS = _ROOT / "data/ph2/dlg_raw_text_observation_v1.jsonl.sample"
_LEX = _ROOT / "data/ph2/dlg_raw_lexical_evidence_v1.jsonl.sample"
_PROP = _ROOT / "data/ph2/dlg_raw_proposition_relation_evidence_v1.jsonl.sample"
_QUAL = _ROOT / "data/ph2/dlg_raw_proposition_qualification_v1.jsonl.sample"


@pytest.fixture(scope="module")
def models():
    records = tuple(item.record for item in
                    load_surface_organization_jsonl(_COURSE.read_bytes()))
    evidence = load_surface_evidence_jsonl(_SURFACE_EVIDENCE.read_bytes())
    return (
        learn_surface_structure_model(
            tuple(item for item in records if item.sample_id in {"s03", "s04"}),
            evidence),
        learn_surface_structure_model(
            tuple(item for item in records if item.sample_id in {"s09", "s10"}),
            evidence),
        learn_surface_structure_model(
            tuple(item for item in records if item.sample_id in {"s13", "s14"}),
            evidence),
    )


def _consumer(index: int):
    observations = load_raw_text_observation_jsonl(_OBS.read_bytes())
    lexical = load_raw_lexical_evidence_jsonl(_LEX.read_bytes())
    propositions = load_raw_proposition_jsonl(_PROP.read_bytes())
    qualifications = load_raw_qualification_jsonl(_QUAL.read_bytes())
    observation = observations[index]
    bound = bind_raw_proposition_relation(
        observation,
        tuple(item for item in lexical if item.observation_id == observation.observation_id),
        propositions[index],
    )
    return consume_raw_proposition_relation(bound, qualifications[index])


def test_supported_qualification_reaches_answer_shadow_without_replacement(models) -> None:
    consumer = _consumer(0)
    plan = SurfaceShadowPlan(
        SurfaceSemantic(consumer.proposition_id, "causal", "台风", "导致", "港口封闭"),
        "ANSWER", "neutral", ("cause", "relation", "effect"),
        (consumer.proposition_id,), (), (consumer.source_id,),
        consumer.context_id, consumer.family_id, "旧答案", 2, 80,
    )
    result = run_raw_t1_shadow_adapter(models[0], consumer, plan)
    assert result.shadow.status_code == SHADOW_SELECTED
    assert result.shadow.shadow_surface == "台风导致港口封闭。"
    assert result.replaced == 0
    assert result.canonical_record()


def test_unknown_qualification_reaches_zero_claim_shadow(models) -> None:
    consumer = _consumer(1)
    plan = SurfaceShadowPlan(
        SurfaceSemantic(consumer.proposition_id, "unknown", "青石台",
                        "运行预算", "未提供"),
        "UNKNOWN", "neutral", ("source", "scope"),
        (), (consumer.proposition_id,), (consumer.source_id,),
        consumer.context_id, consumer.family_id, "旧答案", 2, 80,
        ("当前", "青石台的运行预算"),
    )
    result = run_raw_t1_shadow_adapter(models[2], consumer, plan)
    assert result.shadow.status_code == SHADOW_SELECTED
    assert result.shadow.shadow_surface == "当前资料没有提供青石台的运行预算。"
    assert result.replaced == 0


def test_conflict_qualification_reaches_clarify_shadow(models) -> None:
    consumer = _consumer(0)
    conflict = RawPropositionQualification(
        "q-g4-conflict", consumer.proposition_id, consumer.observation_id,
        consumer.source_id, consumer.context_id, consumer.family_id,
        "t1-g0-public-v1", "train", "CONFLICT", "conflict-v1",
        consumer.evidence_ids, "public-authored-qualification-v1",
    )
    # Rebuild a conflict consumer over the same bound relation.
    observations = load_raw_text_observation_jsonl(_OBS.read_bytes())
    lexical = load_raw_lexical_evidence_jsonl(_LEX.read_bytes())
    propositions = load_raw_proposition_jsonl(_PROP.read_bytes())
    bound = bind_raw_proposition_relation(
        observations[0],
        tuple(item for item in lexical if item.observation_id == observations[0].observation_id),
        propositions[0],
    )
    conflict_consumer = consume_raw_proposition_relation(bound, conflict)
    plan = SurfaceShadowPlan(
        SurfaceSemantic(conflict_consumer.proposition_id, "scope", "资料A与资料B",
                        "需要", "补充范围"),
        "CLARIFY", "polite", ("choice", "target"),
        (), (), (conflict_consumer.source_id,),
        conflict_consumer.context_id, conflict_consumer.family_id, "旧答案", 2, 80,
        ("资料A与资料B", "补充范围"),
    )
    result = run_raw_t1_shadow_adapter(models[1], conflict_consumer, plan)
    assert result.shadow.status_code == SHADOW_SELECTED
    assert result.shadow.shadow_surface
    assert result.replaced == 0


def test_act_source_and_context_mismatch_fail_before_surface(models) -> None:
    consumer = _consumer(0)
    base = dict(
        semantic=SurfaceSemantic(consumer.proposition_id, "causal", "台风", "导致", "港口封闭"),
        dialogue_act="UNKNOWN", register="neutral",
        ordered_roles=("source", "scope"), required_proposition_ids=(),
        forbidden_proposition_ids=(), authorized_source_ids=(consumer.source_id,),
        context_id=consumer.context_id, family_id=consumer.family_id,
        legacy_surface="旧答案", min_chars=2, max_chars=80,
        slot_values=("当前", "资料"),
    )
    with pytest.raises(RawT1ShadowAdapterError, match="response-act"):
        run_raw_t1_shadow_adapter(models[2], consumer, SurfaceShadowPlan(**base))
    with pytest.raises(RawT1ShadowAdapterError, match="source"):
        run_raw_t1_shadow_adapter(
            models[0], consumer,
            SurfaceShadowPlan(**{**base, "dialogue_act": "ANSWER",
                                 "ordered_roles": ("cause", "relation", "effect"),
                                 "required_proposition_ids": (consumer.proposition_id,),
                                 "slot_values": (),
                                 "authorized_source_ids": ("other-source",)}),
        )
