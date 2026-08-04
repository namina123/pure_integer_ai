"""W08-02 中文变体、raw receipt、歧义与正交消融专项。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.identity import (
    CorpusVersion,
    CurriculumVersion,
    GLOBAL_OWNER_SCOPE,
    ParserVersion,
    PrimitiveVersion,
    SourceRef,
    VersionBundle,
)
from pure_integer_ai.experiments.ph2_w08_authority import W08_DIMENSION_KEYS
from pure_integer_ai.experiments.ph2_w08_contract import (
    W08_CONSUMER_KEYS,
    make_w08_request,
    open_w08_frozen_contract,
)
from pure_integer_ai.experiments.ph2_w08_firewall import W08PayloadFirewall
from pure_integer_ai.experiments.ph2_w08_payload import W08TrainingPayload
from pure_integer_ai.experiments.ph2_w08_variation import (
    W08_REFERENCE_MODES,
    W08_VARIATION_FAMILIES,
    W08VariationError,
    assess_w08_variation_ablation,
    learn_w08_variation,
    make_w08_surface_receipt,
    make_w08_variation_intake,
    make_w08_variation_keys,
    require_w08_variation_family_coverage,
    resolve_w08_variation,
    surface_from_w08_observation,
)


ROOT = Path(__file__).resolve().parents[1]
VERSIONS = VersionBundle(
    CorpusVersion(8),
    ParserVersion(2),
    PrimitiveVersion(1),
    CurriculumVersion(8),
)
SOURCE = SourceRef(80802, 11, 101, GLOBAL_OWNER_SCOPE, VERSIONS)


@pytest.fixture(scope="module")
def payload():
    context = open_w08_frozen_contract(ROOT)
    return W08PayloadFirewall.open(
        ROOT, context, make_w08_request(context)
    ).read_training_payload()


def _intake(
    surface: str,
    *,
    family: int = 1,
    content: int = 2,
    structures: tuple[tuple[int, ...], ...] = ((3,),),
    senses: tuple[tuple[int, ...], ...] = ((4,),),
    propositions: tuple[tuple[int, ...], ...] = ((5,),),
    mode: str = "NOUN_PHRASE",
    source: SourceRef = SOURCE,
):
    return make_w08_variation_intake(
        surface=surface,
        source=source,
        surface_family_key=(family,),
        content_key=(content,),
        structure_keys=structures,
        sense_keys=senses,
        proposition_keys=propositions,
        source_key=(source.source_kind, source.source_id),
        document_key=(source.document_id,),
        reference_mode=mode,
    )


def test_w08_variation_learns_six_schemas_and_evidence_identities_only(payload):
    learning = learn_w08_variation(payload)
    assert len(learning.payload_kinds) == 5
    assert len(learning.schema_fingerprints) == 5
    assert len(learning.evidence_bindings) == 63
    assert learning.source_parser_receipt_count == 4
    assert {
        "DISCOURSE_REVISION_LABEL",
        "DISCOURSE_INFORMATION_LABEL",
        "OPEN_SET_CLARIFICATION_LABEL",
        "ATTRIBUTION_QUOTATION_LABEL",
        "SOURCE_PARSER_RECEIPT_V1",
    } == {item[2] for item in learning.evidence_bindings}
    assert any(item.endswith("=REFERENCE") for item in learning.allowed_typed_operations)
    encoded = repr(learning).lower()
    assert "expected_payload" not in encoded
    assert "expected_state" not in encoded
    assert "甲船靠岸" not in encoded


def test_w08_variation_does_not_read_authored_teacher_answers(payload):
    class EvidenceWithoutAnswerAccess:
        def __init__(self, evidence):
            self._evidence = evidence

        def __getattr__(self, name):
            if name == "typed_evidence":
                raise AssertionError("authored answer was read")
            return getattr(self._evidence, name)

    guarded = tuple(
        evidence
        if evidence.evidence_kind == "SOURCE_PARSER_RECEIPT_V1"
        else EvidenceWithoutAnswerAccess(evidence)
        for evidence in payload.teacher_evidence
    )
    learning = learn_w08_variation(W08TrainingPayload(
        payload.source_refs, payload.observations, guarded
    ))
    assert len(learning.evidence_bindings) == 63


def test_w08_variation_rejects_evidence_kind_spoof(payload):
    first = payload.teacher_evidence[0]
    spoofed = replace(first, evidence_kind="SOURCE_PARSER_RECEIPT_V1")
    invalid = W08TrainingPayload(
        payload.source_refs,
        payload.observations,
        (spoofed, *payload.teacher_evidence[1:]),
    )
    with pytest.raises(W08VariationError, match="kind mismatch"):
        learn_w08_variation(invalid)


def test_w08_surface_receipt_preserves_raw_and_appends_unicode_receipts():
    surface = "數學\u3000Ａ？\n"
    receipt = make_w08_surface_receipt(surface)
    assert receipt.recover_raw() == surface
    assert tuple(chr(item) for item in receipt.normalized_codepoints) == tuple("數學 A? ")
    assert receipt.raw_codepoints != receipt.normalized_codepoints
    assert len(receipt.parser_candidate_keys) >= 2
    assert any(category.startswith("P") for category in receipt.unicode_categories)


def test_simplified_traditional_and_new_surfaces_share_typed_content_not_surface():
    simplified = _intake("数学", family=10)
    traditional = _intake("數學", family=11)
    assert simplified.receipt.recover_raw() == "数学"
    assert traditional.receipt.recover_raw() == "數學"
    assert simplified.bundle.representation != traditional.bundle.representation
    assert simplified.bundle.language_atom == traditional.bundle.language_atom
    assert simplified.bundle.senses == traditional.bundle.senses
    assert simplified.bundle.structures == traditional.bundle.structures
    assert simplified.bundle.propositions == traditional.bundle.propositions
    assert simplified.keys.combination_key != traditional.keys.combination_key


def test_width_whitespace_punctuation_and_word_order_do_not_define_semantics():
    variants = (
        _intake("甲问乙。", family=20),
        _intake("甲　问　乙．", family=21),
        _intake("乙被甲询问？", family=22),
        _intake("谁是甲询问的对象？", family=23),
        _intake("甲向那个人提出问题", family=24),
    )
    assert len({item.bundle.representation for item in variants}) == len(variants)
    assert len({item.bundle.language_atom for item in variants}) == 1
    assert len({item.bundle.propositions for item in variants}) == 1


@pytest.mark.parametrize("mode", W08_REFERENCE_MODES)
def test_ellipsis_zero_pronoun_and_np_are_explicit_typed_modes(mode):
    intake = _intake("后续片段", mode=mode)
    assert intake.reference_mode == mode
    assert resolve_w08_variation(intake, evidence_keys=((8, 2),)).uses[0].outcome_state == "RESOLVED"


def test_reference_mode_is_not_selected_from_chinese_cues():
    with pytest.raises(W08VariationError, match="typed"):
        _intake("它随后离开", mode="INFER_FROM_SURFACE")


def test_polysemy_and_multiple_legal_parses_remain_candidates_until_clarified():
    intake = _intake(
        "这个词在这里有两个读法",
        senses=((41,), (42,)),
        structures=((51,), (52,)),
        propositions=((61,),),
    )
    result = resolve_w08_variation(intake, evidence_keys=((71,),))
    assert len(result.candidate_keys) == 4
    assert result.selected_candidate_key is None
    assert {item.outcome_state for item in result.uses} == {"CLARIFY"}
    clarified = resolve_w08_variation(
        intake,
        evidence_keys=((71,),),
        clarification_candidate_key=result.candidate_keys[2],
    )
    assert clarified.selected_candidate_key == result.candidate_keys[2]
    assert {item.outcome_state for item in clarified.uses} == {"RESOLVED"}


def test_urg_share_bundle_but_keep_independent_choice_use_and_outcome_identity():
    intake = _intake("来源化的新表层", family=30)
    result = resolve_w08_variation(intake, evidence_keys=((91,), (92,)))
    assert tuple(item.consumer_key for item in result.uses) == W08_CONSUMER_KEYS
    assert len({item.request_key for item in result.uses}) == 3
    assert len({item.directional_choice_key for item in result.uses}) == 3
    assert len({item.outcome_key for item in result.uses}) == 3
    assert all(item.selected_candidate_key == result.candidate_keys[0]
               for item in result.uses)
    assert all(result.intake.bundle is intake.bundle for _ in result.uses)


def test_source_document_surface_content_structure_and_combination_keys_are_separate():
    keys = make_w08_variation_keys(
        surface_family_key=(1,),
        content_key=(1,),
        structure_key=(1,),
        source_key=(1,),
        document_key=(1,),
    )
    axes = (
        keys.surface_family_key,
        keys.content_key,
        keys.structure_key,
        keys.source_key,
        keys.document_key,
        keys.combination_key,
    )
    assert len(set(axes)) == 6
    other_page = make_w08_variation_keys(
        surface_family_key=(1,),
        content_key=(1,),
        structure_key=(1,),
        source_key=(1,),
        document_key=(2,),
    )
    assert keys.source_key == other_page.source_key
    assert keys.document_key != other_page.document_key
    assert keys.combination_key != other_page.combination_key


def test_all_authorized_source_surfaces_are_exactly_recoverable(payload):
    for observation in payload.observations:
        surface = surface_from_w08_observation(observation)
        assert make_w08_surface_receipt(surface).recover_raw() == surface


def test_variation_family_coverage_and_orthogonal_ablation():
    assert require_w08_variation_family_coverage(
        reversed(W08_VARIATION_FAMILIES)
    ) == tuple(reversed(W08_VARIATION_FAMILIES))
    with pytest.raises(W08VariationError, match="coverage"):
        require_w08_variation_family_coverage(W08_VARIATION_FAMILIES[:-1])
    full = {key: "PASS" for key in W08_DIMENSION_KEYS}
    ablated = dict(full)
    ablated["W-08-CHINESE_VARIATION"] = "FAIL"
    report = assess_w08_variation_ablation(
        full_dimension_outcomes=full,
        ablated_dimension_outcomes=ablated,
    )
    assert report.affected_dimensions == ("W-08-CHINESE_VARIATION",)
    assert set(report.unaffected_dimensions) == set(W08_DIMENSION_KEYS[1:])
    drifted = dict(ablated)
    drifted["W-08-DISCOURSE"] = "FAIL"
    with pytest.raises(W08VariationError, match="orthogonal"):
        assess_w08_variation_ablation(
            full_dimension_outcomes=full,
            ablated_dimension_outcomes=drifted,
        )
