"""FT32 public specification review and no-renderer authorization gate."""
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
from pure_integer_ai.experiments.ph2_w03_public_template_review_ft32 import (
    FT32PublicTemplateReviewError,
    read_ft32_public_template_review,
    validate_ft32_public_template_review_sources,
)


REPOSITORY = Path(__file__).resolve().parents[1]
REVIEW = REPOSITORY / (
    "data/ph2/manifests/ft32_public_template_specification_review_v1.json")
REVIEW_SHA256 = (
    "4470e7af54247720bf7137f3088a25deada266c6ccbcf61b8f0b7079e8351455")


def test_ft32_review_is_canonical_attributable_and_cross_checked() -> None:
    """The two outcomes close over FT31 frequency and the public snapshot."""
    raw = REVIEW.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == REVIEW_SHA256
    manifest = read_ft32_public_template_review(REVIEW)
    validate_ft32_public_template_review_sources(
        manifest, repository_root=REPOSITORY)
    reviews = manifest.reviews
    assert tuple(item["template_name"] for item in reviews) == (
        "place", "zh-div")
    assert tuple(item["status"] for item in reviews) == (
        "BLOCKED", "REVIEWED_NOT_AUTHORIZED")
    assert all(item["renderer_authorized"] == 0 for item in reviews)
    evidence = manifest.to_dict()["evidence_pages"]
    assert len(evidence) == 8
    assert all(item["license_id"] == "CC-BY-SA-4.0"
               and "Wiktionary contributors" in item["attribution"]
               for item in evidence)


def test_ft32_keeps_place_and_zh_div_fail_closed() -> None:
    """Review completion does not silently add either unsupported renderer."""
    for definition, code in (
        ("{{place|zh|城市|c/烏克蘭}}", "UNKNOWN_TEMPLATE"),
        ("{{zh-div|州}} {{place|zh|州|c/烏克蘭}}", "UNKNOWN_TEMPLATE"),
        ("{{place|zh|<<c/美國>><<首都>>}}", "UNSUPPORTED_INLINE_MARKUP"),
    ):
        with pytest.raises(MediaWikiInlineParseError) as caught:
            project_mediawiki_inline(definition)
        assert caught.value.code == code


def test_ft32_rejects_forged_authorization_and_evidence_identity(tmp_path) -> None:
    """Authorization and revision content cannot be edited in place."""
    value = json.loads(REVIEW.read_text(encoding="utf-8"))
    value["reviews"][0]["renderer_authorized"] = 1
    forged = tmp_path / "forged-authorization.json"
    forged.write_bytes(canonical_json_line(value))
    with pytest.raises(FT32PublicTemplateReviewError):
        read_ft32_public_template_review(forged)

    value = json.loads(REVIEW.read_text(encoding="utf-8"))
    value["reviews"][1]["unresolved_dependency_titles"].pop()
    forged = tmp_path / "forged-dependency-closure.json"
    forged.write_bytes(canonical_json_line(value))
    with pytest.raises(FT32PublicTemplateReviewError):
        read_ft32_public_template_review(forged)

    value = json.loads(REVIEW.read_text(encoding="utf-8"))
    value["reviews"][0]["observed_definitions"].pop()
    value["reviews"][0]["occurrence_count"] -= 1
    forged = tmp_path / "forged-occurrence-inventory.json"
    forged.write_bytes(canonical_json_line(value))
    with pytest.raises(FT32PublicTemplateReviewError):
        read_ft32_public_template_review(forged)

    value = json.loads(REVIEW.read_text(encoding="utf-8"))
    value["evidence_pages"][0]["content_sha256"] = "f" * 64
    forged = tmp_path / "forged-evidence.json"
    forged.write_bytes(canonical_json_line(value))
    with pytest.raises(FT32PublicTemplateReviewError):
        read_ft32_public_template_review(forged)


def test_ft32_rejects_predecessor_drift(tmp_path) -> None:
    """The review remains causally bound to the public census and snapshot."""
    manifest = read_ft32_public_template_review(REVIEW)
    root = tmp_path / "repo"
    target = root / "data/ph2/manifests"
    target.mkdir(parents=True)
    for name in (
        "ft31_w03_public_definition_census_v3.json",
        "zhwiktionary_20260701.multistream_snapshot.json",
    ):
        source = REPOSITORY / "data/ph2/manifests" / name
        (target / name).write_bytes(source.read_bytes())
    census = target / "ft31_w03_public_definition_census_v3.json"
    census.write_bytes(census.read_bytes() + b" ")
    with pytest.raises(FT32PublicTemplateReviewError, match="predecessor SHA"):
        validate_ft32_public_template_review_sources(
            manifest, repository_root=root)
