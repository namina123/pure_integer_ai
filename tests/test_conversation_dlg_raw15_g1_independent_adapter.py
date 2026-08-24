"""DLG-RAW-15 G1 feasibility probe（明确 NOT_READY，不是 formal gate）。"""
from __future__ import annotations

from pathlib import Path

from pure_integer_ai.experiments.conversation_dlg_raw15_g1_independent_adapter import (
    DLG_RAW15_G1_FORMAL_STATUS,
    load_g1_physical_pack,
    load_g1_slot_catalog,
    make_closure,
)
from pure_integer_ai.experiments.conversation_public_source_payload_host import (
    load_public_source_payload_closure_from_root,
)
from pure_integer_ai.experiments.conversation_public_source_payload_provider import (
    PUBLIC_SOURCE_PAYLOAD_LOGICAL_KEYS_V1,
)
from pure_integer_ai.experiments.conversation_raw_answer_runtime import (
    run_public_frame_answer,
)
from pure_integer_ai.experiments.conversation_raw_intake import (
    DLG_RAW_ACCEPT,
    DLG_RAW_REJECT_LEXICAL_AMBIGUOUS,
    encode_utf8_v1,
    intake_raw_conversation_vector,
)
from pure_integer_ai.experiments.conversation_raw_lexical_ingress import (
    ingress_raw_lexical_frame,
)
from pure_integer_ai.experiments.conversation_source_bound_slot_catalog import (
    resolve_source_bound_slot_composition,
)


_ROOT = Path(__file__).resolve().parents[1]
_COURSE_KEY = b"data/ph2/grounded_answer_train_v1.jsonl.sample"
_RESPONSE_KEY = b"data/ph2/dlg_raw_public_response_act_frame_v2.jsonl.sample"
_LEXICAL_A_KEY = (
    b"data/ph2/dlg_raw_public_response_act_lexical_v2_a.txt.sample")
_LEXICAL_B_KEY = (
    b"data/ph2/dlg_raw_public_response_act_lexical_v2_b.txt.sample")
_AMBIGUOUS = tuple(ord(item) for item in "西岸入口预算是多少？")


def _production_closure():
    """只读 production，作为 feasibility probe 的 before/after 对照。"""
    return load_public_source_payload_closure_from_root(_ROOT)


def test_g1_probe_is_explicitly_not_ready_and_preserves_production() -> None:
    """探针只证明候选闭合，明确保留 formal G1 未就绪边界。"""
    assert DLG_RAW15_G1_FORMAL_STATUS == "FEASIBILITY_ONLY_NOT_READY"
    production_before = _production_closure()
    closure, episodes, base_catalog = make_closure()
    slot = load_g1_slot_catalog(closure, base_catalog)
    production_after = _production_closure()

    assert production_before.canonical_record() == production_after.canonical_record()
    assert production_before.closure_identity == production_after.closure_identity
    assert len(closure.records) == len(PUBLIC_SOURCE_PAYLOAD_LOGICAL_KEYS_V1)
    assert closure.closure_identity != production_before.closure_identity
    assert tuple(item["episode_id"] for item in episodes) == (
        "g1-seed-answer-v1",
        "g1-heldout-budget-1-v1",
        "g1-heldout-budget-2-v1",
    )
    assert tuple(item.frame_key for item in base_catalog.frames) == (
        "g1-frame-budget-1-v1",
        "g1-frame-budget-2-v1",
    )
    assert len({item.recipe.episode_id for item in base_catalog.frames}) == 2
    assert len({item.recipe.canonical_record() for item in base_catalog.frames}) == 2
    assert closure.payload_for(_COURSE_KEY) != production_before.payload_for(_COURSE_KEY)
    assert closure.payload_for(_RESPONSE_KEY) != production_before.payload_for(_RESPONSE_KEY)
    assert closure.payload_for(_LEXICAL_A_KEY) != production_before.payload_for(_LEXICAL_A_KEY)
    assert closure.payload_for(_LEXICAL_B_KEY) != production_before.payload_for(_LEXICAL_B_KEY)

    assert len(slot.families) == 1
    assert len(slot.bindings) == 2
    assert all(source.license_id == "CC0-1.0" for source in slot.source_records)
    assert len({source.source.stable_key() for source in slot.source_records}) == 6
    # 当前 source keys 仍属于 fixed registry namespace；这正是正式独立 pack
    # 尚未完成的审计边界，不能被本探针误报为独立 held-out evidence。
    assert tuple(source.source.stable_key()[0] for source in slot.source_records) == (
        65164, 65165, 65171, 65172, 65173, 65174,
    )


def test_g1_probe_resolves_ambiguity_but_only_replays_unknown() -> None:
    """双 binding 拒绝歧义；单 binding 可跑 RAW-01/02，但输出仍是 UNKNOWN。"""
    production_before = _production_closure()
    closure, _episodes, base_catalog = make_closure()
    slot = load_g1_slot_catalog(closure, base_catalog)
    ambiguous = resolve_source_bound_slot_composition(
        slot, base_catalog, base_catalog, _AMBIGUOUS, closure)
    assert ambiguous.result_code == DLG_RAW_REJECT_LEXICAL_AMBIGUOUS
    assert ambiguous.matched_frame_count == 2
    assert ambiguous.frame is None
    assert ambiguous.public_frame_catalog is None
    assert len(ambiguous.target_candidates) == 2
    assert len({item.target_key for item in ambiguous.target_candidates}) == 2
    assert len({item.recipe_record for item in ambiguous.target_candidates}) == 2

    outputs = []
    single_frames = []
    for index in (0, 1):
        single_closure, _single_episodes, single_base = make_closure(
            binding_indices=(index,))
        single_slot = load_g1_slot_catalog(single_closure, single_base)
        unique = resolve_source_bound_slot_composition(
            single_slot, single_base, single_base, _AMBIGUOUS, single_closure)
        assert unique.result_code == DLG_RAW_ACCEPT
        assert unique.accepted
        assert unique.frame is not None
        assert unique.public_frame_catalog is not None
        assert unique.matched_frame_count == 1
        single_frames.append(unique.frame)
        intake = intake_raw_conversation_vector(tuple(encode_utf8_v1(_AMBIGUOUS)))
        ingress = ingress_raw_lexical_frame(
            intake, unique.public_frame_catalog, (9001 + index,))
        assert ingress.result_code == DLG_RAW_ACCEPT
        answer = run_public_frame_answer(
            ingress, source_payload_closure=single_closure)
        assert answer.result_code == DLG_RAW_ACCEPT
        assert answer.accepted
        output = "".join(chr(item) for item in answer.output_scalars)
        assert "无法确定" in output
        outputs.append(output)
        assert answer.output_readback is not None
        assert answer.output_readback.unicode_scalars == answer.output_scalars

    assert len({item.frame_key for item in single_frames}) == 2
    assert len({item.raw_line_sha256 for item in single_frames}) == 2
    assert len({item.recipe.canonical_record() for item in single_frames}) == 2
    # 两支当前仍落到同一 UNKNOWN surface；这是未具备正向 G1 的证据。
    assert len(set(outputs)) == 1
    production_after = _production_closure()
    assert production_before.canonical_record() == production_after.canonical_record()
    assert production_before.closure_identity == production_after.closure_identity


def test_g1_physical_pack_loads_two_real_answer_frames_and_ambiguity() -> None:
    """独立 physical fixture 能真实装配两个 ANSWER frame；仍不升级 G1 状态。"""
    pack = load_g1_physical_pack()
    assert pack.status == "FEASIBILITY_ONLY_NOT_READY"
    assert pack.pack_id == "dlg-raw15-g1-independent-v1"
    assert pack.source_namespace == "PIAI-DLG-RAW-15-G1-V1"
    assert len(pack.base_catalog.frames) == 2
    assert len(pack.slot_catalog.bindings) == 2
    assert len({item.frame_key for item in pack.base_catalog.frames}) == 2
    assert len({item.recipe.episode_id for item in pack.base_catalog.frames}) == 2

    outputs = []
    for ordinal, frame in enumerate(pack.base_catalog.frames, start=1):
        intake = intake_raw_conversation_vector(
            tuple(encode_utf8_v1(frame.surface_scalars)))
        ingress = ingress_raw_lexical_frame(
            intake, pack.base_catalog, (8100, ordinal))
        answer = run_public_frame_answer(
            ingress, source_payload_closure=pack.closure)
        assert answer.result_code == DLG_RAW_ACCEPT
        outputs.append("".join(chr(item) for item in answer.output_scalars))
    assert outputs == [
        "玄衡台北区的建设预算为120万元。",
        "玄衡台南区的建设预算为230万元。",
    ]

    ambiguous = resolve_source_bound_slot_composition(
        pack.slot_catalog,
        pack.base_catalog,
        pack.base_catalog,
        tuple(ord(item) for item in "西岸入口预算是多少？"),
        pack.closure,
    )
    assert ambiguous.result_code == DLG_RAW_REJECT_LEXICAL_AMBIGUOUS
    assert ambiguous.matched_frame_count == 2
    assert len(ambiguous.target_candidates) == 2
    assert len({item.base_frame_key for item in ambiguous.target_candidates}) == 2
    assert len({item.recipe_record for item in ambiguous.target_candidates}) == 2
