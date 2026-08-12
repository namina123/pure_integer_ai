"""FT34 six-template review and narrow semantic projection tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line
from pure_integer_ai.experiments.ph2_mediawiki_inline_ast import (
    MediaWikiInlineParseError,
    project_mediawiki_inline,
)
from pure_integer_ai.experiments.ph2_mediawiki_semantic_template_ast import (
    MEDIAWIKI_SEMANTIC_TEMPLATE_PARSER_VERSION,
    MediaWikiSemanticReference,
    MediaWikiSemanticTemplateParseError,
    project_mediawiki_semantic_templates,
)
from pure_integer_ai.experiments.ph2_w03_public_template_review_ft34 import (
    FT34PublicTemplateReviewError,
    read_ft34_public_template_review,
    validate_ft34_public_template_review_sources,
)


REPOSITORY = Path(__file__).resolve().parents[1]
REVIEW = REPOSITORY / (
    "data/ph2/manifests/ft34_public_template_specification_review_v1.json")


def test_ft34_review_is_canonical_attributable_and_cross_checked() -> None:
    """All six decisions are bound to FT33 and frozen public revisions."""
    raw = REVIEW.read_bytes()
    manifest = read_ft34_public_template_review(REVIEW)
    validate_ft34_public_template_review_sources(
        manifest, repository_root=REPOSITORY)
    assert manifest.canonical_bytes() == raw
    assert tuple((item["template_name"], item["status"])
                 for item in manifest.reviews) == (
        ("alt form", "REVIEWED_AUTHORIZED"),
        ("rfdef", "REVIEWED_NOT_AUTHORIZED"),
        ("surname", "REVIEWED_AUTHORIZED"),
        ("syn of", "REVIEWED_AUTHORIZED"),
        ("zh-alt-form", "REVIEWED_AUTHORIZED"),
        ("†", "BLOCKED"),
    )
    assert sum(item["occurrence_count"] for item in manifest.reviews) == 30
    assert sum(item["renderer_authorized"] for item in manifest.reviews) == 4
    value = manifest.to_dict()
    assert len(value["evidence_pages"]) == 18
    assert value["snapshot_absence_evidence"]["missing_titles"] == [
        "Template:†"]
    assert all(item["license_id"] == "CC-BY-SA-4.0"
               and "Wiktionary contributors" in item["attribution"]
               for item in value["evidence_pages"])
    assert hashlib.sha256(raw).hexdigest() == (
        "622306008fa4247f2135d085a292da20aad5f10143ced56ded68f75fd3602da0")


def test_ft34_all_thirty_observed_occurrences_follow_their_decisions() -> None:
    """No FT33 occurrence is omitted behind a deduplicated profile test."""
    manifest = read_ft34_public_template_review(REVIEW)
    seen = 0
    for review in manifest.reviews:
        for observed in review["observed_definitions"]:
            seen += 1
            source = observed["raw_definition_text"]
            if review["renderer_authorized"] == 1:
                projection = project_mediawiki_semantic_templates(source)
                assert projection.document.source_text == source
                assert projection.display_text
                continue
            expected = (
                "MAINTENANCE_TEMPLATE"
                if review["template_name"] == "rfdef"
                else "BLOCKED_TEMPLATE")
            with pytest.raises(MediaWikiSemanticTemplateParseError) as caught:
                project_mediawiki_semantic_templates(source)
            assert caught.value.code == expected
    assert seen == 30


@pytest.mark.parametrize(("source", "expected"), (
    (
        "{{lb|zh|Teochew}} {{alt form|zh|焦|tr=-|t=[[乾]]}}",
        "（Teochew） 焦的另一種寫法（義：乾）",
    ),
    (
        "{{lb|zh|閩南語}} {{alt form|zh|毋|tr=m̄|t=不}}",
        "（閩南語） 毋的另一種寫法（轉寫：m̄；義：不）",
    ),
    ("{{alt form|zh|謀|t=思慮}}", "謀的另一種寫法（義：思慮）"),
    ("{{alt form|zh|掹}}", "掹的另一種寫法"),
    ("{{alt form|zh|銛|t=鋒利}}", "銛的另一種寫法（義：鋒利）"),
    ("{{syn of|zh|鮠}}", "鮠之同義詞"),
    ("{{syn of|zh|注音符號}}", "注音符號之同義詞"),
    ("{{syn of|zh|無依無靠}}", "無依無靠之同義詞"),
    ("{{syn of|zh|耶穌基督後期聖徒教會}}", "耶穌基督後期聖徒教會之同義詞"),
    ("{{syn of|zh|婊子立牌坊}}", "婊子立牌坊之同義詞"),
    ("{{syn of|zh|聖克里斯多福及尼維斯}}", "聖克里斯多福及尼維斯之同義詞"),
    ("{{zh-alt-form|訇}}", "訇的另一種寫法"),
    ("{{zh-alt-form|顎}}", "顎的另一種寫法"),
    ("{{zh-alt-form|輲}}", "輲的另一種寫法"),
    ("{{surname|zh}}", "姓氏"),
))
def test_ft34_authorized_observed_profiles_project_deterministically(
        source: str, expected: str) -> None:
    """Every distinct authorized FT33 source profile has one local display."""
    first = project_mediawiki_semantic_templates(source)
    second = project_mediawiki_semantic_templates(source)
    assert first.to_dict() == second.to_dict()
    assert first.display_text == expected
    assert first.document.source_text == source
    assert first.document.parser_version == (
        MEDIAWIKI_SEMANTIC_TEMPLATE_PARSER_VERSION)
    assert "".join(source[node.start:node.end]
                   for node in first.document.nodes) == source
    assert any(isinstance(node, MediaWikiSemanticReference)
               for node in first.document.nodes)


@pytest.mark.parametrize(("source", "code"), (
    ("{{rfdef|zh}}", "MAINTENANCE_TEMPLATE"),
    ("{{rfdef|zh|sort=土05}}", "MAINTENANCE_TEMPLATE"),
    ("{{†}} [[刻]]", "BLOCKED_TEMPLATE"),
    ("{{alt form|ja|風}}", "BAD_TEMPLATE_PROFILE"),
    ("{{alt form|zh|焦|from=潮州}}", "BAD_TEMPLATE_PROFILE"),
    ("{{alt form|zh|焦|t={{w|乾}}}}", "NESTED_MARKUP"),
    ("{{syn of|en|answer}}", "BAD_TEMPLATE_PROFILE"),
    ("{{syn of|zh|甲|乙}}", "BAD_TEMPLATE_PROFILE"),
    ("{{zh-alt-form|甲|乙}}", "BAD_TEMPLATE_PROFILE"),
    ("{{surname|en}}", "BAD_TEMPLATE_PROFILE"),
    ("{{surname|zh|from=父名}}", "BAD_TEMPLATE_PROFILE"),
    ("{{lb|zh|[[方言]]}} {{surname|zh}}", "NESTED_MARKUP"),
    ("{{place|zh|城市|c/烏克蘭}}", "UNKNOWN_TEMPLATE"),
))
def test_ft34_unauthorized_and_unobserved_profiles_fail_closed(
        source: str, code: str) -> None:
    """FT34 does not turn broad upstream templates into broad local code."""
    with pytest.raises(MediaWikiSemanticTemplateParseError) as caught:
        project_mediawiki_semantic_templates(source)
    assert caught.value.code == code


def test_ft34_does_not_mutate_the_frozen_v1_inline_parser() -> None:
    """The opt-in FT34 projection leaves all historical parser bytes stable."""
    for source in (
        "{{alt form|zh|掹}}",
        "{{syn of|zh|鮠}}",
        "{{zh-alt-form|顎}}",
        "{{surname|zh}}",
    ):
        with pytest.raises(MediaWikiInlineParseError) as caught:
            project_mediawiki_inline(source)
        assert caught.value.code == "UNKNOWN_TEMPLATE"


def test_ft34_rejects_forged_status_evidence_and_predecessor(tmp_path) -> None:
    """Authorization, occurrence evidence, and predecessor bytes are immutable."""
    value = json.loads(REVIEW.read_text(encoding="utf-8"))
    value["reviews"][1]["renderer_authorized"] = 1
    forged = tmp_path / "forged-authorization.json"
    forged.write_bytes(canonical_json_line(value))
    with pytest.raises(FT34PublicTemplateReviewError, match="decision drifted"):
        read_ft34_public_template_review(forged)

    value = json.loads(REVIEW.read_text(encoding="utf-8"))
    value["reviews"][0]["observed_definitions"].pop()
    forged = tmp_path / "forged-occurrences.json"
    forged.write_bytes(canonical_json_line(value))
    with pytest.raises(FT34PublicTemplateReviewError, match="inventory drifted"):
        read_ft34_public_template_review(forged)

    value = json.loads(REVIEW.read_text(encoding="utf-8"))
    value["unexpected"] = 1
    forged = tmp_path / "forged-extra-field.json"
    forged.write_bytes(canonical_json_line(value))
    with pytest.raises(FT34PublicTemplateReviewError, match="fields drifted"):
        read_ft34_public_template_review(forged)

    root = tmp_path / "repo"
    target = root / "data/ph2/manifests"
    target.mkdir(parents=True)
    for name in (
        "ft33_w03_public_definition_census_v4.json",
        "zhwiktionary_20260701.multistream_snapshot.json",
    ):
        source = REPOSITORY / "data/ph2/manifests" / name
        (target / name).write_bytes(source.read_bytes())
    census = target / "ft33_w03_public_definition_census_v4.json"
    census.write_bytes(census.read_bytes() + b" ")
    manifest = read_ft34_public_template_review(REVIEW)
    with pytest.raises(FT34PublicTemplateReviewError, match="predecessor SHA"):
        validate_ft34_public_template_review_sources(
            manifest, repository_root=root)
