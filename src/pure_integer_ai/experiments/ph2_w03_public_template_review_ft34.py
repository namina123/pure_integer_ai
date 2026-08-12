"""FT34 public specification review for the six FT33-qualified templates."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_mediawiki_snapshot import (
    read_mediawiki_dump_snapshot,
)


FT34_REVIEW_FORMAT = "PH2_FT34_PUBLIC_TEMPLATE_SPECIFICATION_REVIEW"
FT34_REVIEW_VERSION = 1
FT34_STAGE = "FT34-PUBLIC-TEMPLATE-SPECIFICATION-REVIEW-V4-QUALIFIED-SIX"
FT34_REVIEW_DATE = "2026-08-12"
FT34_CENSUS_RELATIVE_PATH = (
    "data/ph2/manifests/ft33_w03_public_definition_census_v4.json")
FT34_CENSUS_SHA256 = (
    "3628324f0b334f7e08116a89202b4447afc947808ac0354dc4681da0ff0b2936")
FT34_SNAPSHOT_RELATIVE_PATH = (
    "data/ph2/manifests/zhwiktionary_20260701.multistream_snapshot.json")
FT34_SNAPSHOT_SHA256 = (
    "9d0c82e39719a8084eb5bd672ba984589952874ee248c04816a00c7be20f2fdc")
FT34_SOURCE_KEY = "ZHWIKTIONARY_20260701"
FT34_SNAPSHOT_ID = "zhwiktionary-20260701-adapter-v1-double-pass-v1"

# title -> namespace, page, revision, parent, timestamp, MediaWiki SHA-1,
# content SHA-256, UTF-8 bytes, role, direct frozen runtime dependencies.
FT34_EVIDENCE_IDENTITIES = {
    "Module:Form of": (
        828, 1365350, 9739602, 9739582, "2026-04-21T06:19:02Z",
        "q3ii7ilmph1d5gp4g8n2yerz780bvtb",
        "2e993972e95dc1addb1b1de12309f59af39792626dcaebdb2bf511f78e3dd3b2",
        66376, "TRANSITIVE_IMPLEMENTATION_MODULE", (
            "Module:JSON", "Module:debug/track", "Module:etymology",
            "Module:form of/cats", "Module:form of/data",
            "Module:form of/data/1", "Module:form of/data/2",
            "Module:form of/functions", "Module:form of/templates",
            "Module:fun", "Module:headword/data", "Module:labels",
            "Module:languages", "Module:links", "Module:load",
            "Module:parse utilities", "Module:string utilities",
            "Module:table", "Module:table/deepEquals", "Module:utilities")),
    "Module:Form of/templates": (
        828, 1382238, 9819295, 9739591, "2026-06-23T07:46:54Z",
        "38a1c3nf6rz0qufdeix2anxdtibqby6",
        "3326a1fb9470facab5e999d150747e49409f5805bee0138499da06f7045ed796",
        39500, "IMPLEMENTATION_MODULE", (
            "Module:debug/track", "Module:form of", "Module:fun",
            "Module:headword/data", "Module:languages", "Module:links",
            "Module:load", "Module:parameter utilities",
            "Module:parameters", "Module:parse interface",
            "Module:string utilities", "Module:table", "Module:utilities")),
    "Module:Labels/templates": (
        828, 1365117, 8342980, 8339507, "2024-04-20T20:30:21Z",
        "k5fqrzqnaqdaproyvdz6qokktnek03s",
        "a3a9af3d8796b6a85db5e87d424832ade0fdba55ba43b175f552e54c79fbf688",
        1332, "LABEL_IMPLEMENTATION_MODULE", (
            "Module:debug/track", "Module:labels", "Module:languages",
            "Module:parameters", "Module:template_link", "Module:utilities")),
    "Module:Names": (
        828, 1413518, 9651553, 9651552, "2026-02-13T01:10:50Z",
        "8wdrj46zt96u0aprarmzclmu9hqsfcc",
        "d3fdf0ddf27ddee83ce055cfbb6a206fc0a877af4c1d4d38fb953cd5299859f0",
        36240, "IMPLEMENTATION_MODULE", (
            "Module:debug", "Module:en-utilities", "Module:families",
            "Module:inflection utilities", "Module:languages",
            "Module:links", "Module:parameters", "Module:parse utilities",
            "Module:qualifier", "Module:scripts", "Module:table",
            "Module:utilities")),
    "Module:Zh/link": (
        828, 1889665, 8586237, 7637657, "2024-07-26T16:02:52Z",
        "e8o4pap9de6kjtv6up1qv5qr2okyt9f",
        "f5c52d31577dc75f10546745961a4463b44e66b22e312fced037d6037a2deb0e",
        4953, "IMPLEMENTATION_MODULE", (
            "Module:languages", "Module:links", "Module:parameters",
            "Module:zh", "Module:zh/extract")),
    "Template:Alt form": (
        10, 1405627, 7272485, 5739250, "2022-07-30T16:50:45Z",
        "0vctg9bbyhiu9oiawswwti1lnavxxfs",
        "9a9576d50669cc90ecb2ebd34e897569c010401892c6ef26df1445078da3476a",
        43, "REDIRECT_TEMPLATE_ENTRY", ("Template:Alternative form of",)),
    "Template:Alternative form of": (
        10, 1035888, 7829083, 7272487, "2023-08-27T20:34:05Z",
        "3y9h5vnm2rgu9wyvsbmx8x1h4b1ikq0",
        "78b9c1f660d220fe8e42d66ff243279263bd15f6ec28360ad84d91fef02dd20e",
        344, "TEMPLATE_ENTRY", (
            "Module:Form of/templates", "Module:Labels/templates")),
    "Template:Alternative form of/doc": (
        10, 2899377, 8955039, 0, "2024-11-21T02:29:46Z",
        "08zn173vlww9f8oz4fw3uag4r17kg9f",
        "a0a70b0c35ab8af628a462d3de639733b7437269ce27f8a3c0f079c308af9ef1",
        2333, "PUBLIC_DOCUMENTATION_AND_TEMPLATEDATA", ()),
    "Template:Rfdef": (
        10, 1365145, 8929978, 7811501, "2024-11-14T17:47:42Z",
        "niigtab855jmt2ojt7p9r386975n4lh",
        "1590ecbb86f87a4650d7953c3cae84049fcacd25eff5ad743d5e5c60b87c7fb9",
        947, "TEMPLATE_ENTRY", ("Module:Checkparams",)),
    "Template:Rfdef/doc": (
        10, 2886273, 8937175, 8937174, "2024-11-15T11:49:59Z",
        "np9wfdp0v5zo0iqunp3kpm90lg8d1pv",
        "e8df94db659e9b0014969793390da25f31a383cf0570212def6a2d2c0feffd9b",
        2704, "PUBLIC_DOCUMENTATION", ()),
    "Template:Surname": (
        10, 45121, 9300970, 7555962, "2025-06-24T04:44:49Z",
        "3mtqda6my2kbrm1x2w5xe3865ljxjq3",
        "dff9423e4e0fed7ee4b4b13e6d92385c7d8c893f4d40067b5aa586eddde6215d",
        87, "TEMPLATE_ENTRY", ("Module:Names",)),
    "Template:Surname/doc": (
        10, 2901272, 9301542, 9301541, "2025-06-24T07:53:05Z",
        "2s1vgneqzlqzgrt6t1bzzxrwbedhj74",
        "5a7ce25d0ccbe89536d27efa007380f8c971d3a9f31d1d15cd5d19a0822d9667",
        1799, "PUBLIC_DOCUMENTATION", ()),
    "Template:Syn of": (
        10, 1405407, 8059062, 6195383, "2024-02-12T20:11:45Z",
        "bz4j3pm5dkdm5rtk1e09g8ev4ap27b1",
        "70761ccbae52d73716a0a16e8015f931ce06cd3ca24fd02224839d356bff968f",
        35, "REDIRECT_TEMPLATE_ENTRY", ("Template:之同義詞",)),
    "Template:Zh-alt form": (
        10, 1365124, 8904466, 7516172, "2024-11-06T21:09:55Z",
        "2n8qzy99vn8w8zxe56aj2u3gkwr74ce",
        "409f61a84caf3f1d133b8cf66f3b4fad55f235fde6cbbc113de6c2d8bfbdd6a5",
        165, "TEMPLATE_ENTRY", ("Template:Zh-l",)),
    "Template:Zh-alt-form": (
        10, 1717558, 8339823, 6669971, "2024-04-19T20:11:42Z",
        "hunywzmuyr017ceak5iawzbu30x8nre",
        "76a342560ff83abaf209c3d07211562748d359425cf703b7820b8ce4907c6259",
        35, "REDIRECT_TEMPLATE_ENTRY", ("Template:Zh-alt form",)),
    "Template:Zh-l": (
        10, 1364576, 7637654, 7238439, "2023-06-11T20:11:36Z",
        "gt3gtcq2rddx4ssf6k09amhf8qiwd6f",
        "bfe0b0cbf86fd0d5b9be07abe63083b715d97b28c87446d5c132006bcc5a71f5",
        91, "TRANSITIVE_TEMPLATE_ENTRY", ("Module:Zh/link",)),
    "Template:Zh-l/doc": (
        10, 2767006, 9057970, 9020135, "2025-01-12T13:34:20Z",
        "2hliz1pnd71pbtykv7eawyccxb7tuai",
        "47ddc87570e7fa084803ddbcc4d73062b2f72c60ea7438b1e1ee3fd00613be6a",
        3079, "PUBLIC_DOCUMENTATION", ("Module:Zh/link",)),
    "Template:之同義詞": (
        10, 1405405, 7272898, 6206429, "2022-07-31T03:20:29Z",
        "iy5e3qv8otblhbq6hs9npz5yj7h0gqh",
        "076bb5ae0d6c56904e8986585867f6fb0786e76a7fa63e998b5abfff7fedca50",
        184, "TEMPLATE_ENTRY", ("Module:Form of/templates",)),
}

FT34_EVIDENCE_CONTRIBUTORS = {
    "Module:Form of": (53191, "TongcyDai"),
    "Module:Form of/templates": (53191, "TongcyDai"),
    "Module:Labels/templates": (53191, "TongcyDai"),
    "Module:Names": (79702, "列维劳德"),
    "Module:Zh/link": (53191, "TongcyDai"),
    "Template:Alt form": (25402, "Xiplus"),
    "Template:Alternative form of": (53191, "TongcyDai"),
    "Template:Alternative form of/doc": (92536, "Kethyga"),
    "Template:Rfdef": (53191, "TongcyDai"),
    "Template:Rfdef/doc": (92536, "Kethyga"),
    "Template:Surname": (55252, "Fglffer"),
    "Template:Surname/doc": (53191, "TongcyDai"),
    "Template:Syn of": (25402, "Xiplus"),
    "Template:Zh-alt form": (55252, "Fglffer"),
    "Template:Zh-alt-form": (25402, "Xiplus"),
    "Template:Zh-l": (25402, "Xiplus"),
    "Template:Zh-l/doc": (92536, "Kethyga"),
    "Template:之同義詞": (25402, "Xiplus"),
}

FT34_REVIEW_SPECS = {
    "alt form": {
        "status": "REVIEWED_AUTHORIZED",
        "renderer_authorized": 1,
        "parameter_profile": "ZH_TERM_OPTIONAL_EXPLICIT_TR_AND_GLOSS_V1",
        "evidence_titles": (
            "Module:Form of", "Module:Form of/templates",
            "Module:Labels/templates", "Template:Alt form",
            "Template:Alternative form of",
            "Template:Alternative form of/doc"),
        "findings": (
            "CALL_NAME_REDIRECTS_TO_ALTERNATIVE_FORM_OF",
            "EXPLICIT_GLOSS_ACCEPTS_SOURCE_PRESERVING_INLINE_LINKS",
            "OBSERVED_PROFILE_HAS_LANGUAGE_TARGET_OPTIONAL_TR_AND_T",
            "SEMANTIC_PROJECTION_EXCLUDES_CATEGORIES_LIVE_LINK_STATE_AND_AUTOMATIC_TRANSLITERATION",
            "TEMPLATEDATA_IDENTIFIES_LANGUAGE_AND_TARGET_AS_REQUIRED"),
        "blockers": (),
        "unresolved": (),
    },
    "rfdef": {
        "status": "REVIEWED_NOT_AUTHORIZED",
        "renderer_authorized": 0,
        "parameter_profile": "NO_LEXICAL_RENDERER",
        "evidence_titles": ("Template:Rfdef", "Template:Rfdef/doc"),
        "findings": (
            "DOCUMENTATION_REQUIRES_REMOVAL_AFTER_A_DEFINITION_IS_ADDED",
            "OUTPUT_IS_AN_EDITOR_REQUEST_NOT_A_LEXICAL_DEFINITION",
            "TEMPLATE_ADDS_LANGUAGE_DEFINITION_REQUEST_CATEGORIES"),
        "blockers": (
            "CATEGORY_SIDE_EFFECTS",
            "MAINTENANCE_REQUEST_NOT_LEXICAL_DEFINITION",
            "NO_TARGET_SEMANTIC_CONTENT"),
        "unresolved": ("Module:Checkparams",),
    },
    "surname": {
        "status": "REVIEWED_AUTHORIZED",
        "renderer_authorized": 1,
        "parameter_profile": "OBSERVED_ZH_LANGUAGE_ONLY_SURNAME_CLASS_V1",
        "evidence_titles": (
            "Module:Names", "Template:Surname", "Template:Surname/doc"),
        "findings": (
            "ALL_FT33_OCCURRENCES_USE_ONLY_POSITIONAL_LANGUAGE_ZH",
            "OBSERVED_PROFILE_DETERMINISTICALLY_DENOTES_THE_SURNAME_LEXICAL_CLASS",
            "SEMANTIC_PROJECTION_EXCLUDES_CATEGORIES_AND_UNOBSERVED_OPTIONAL_PARAMETERS"),
        "blockers": (),
        "unresolved": (),
    },
    "syn of": {
        "status": "REVIEWED_AUTHORIZED",
        "renderer_authorized": 1,
        "parameter_profile": "ZH_LANGUAGE_AND_TARGET_ONLY_SYNONYM_V1",
        "evidence_titles": (
            "Module:Form of", "Module:Form of/templates",
            "Template:Syn of", "Template:之同義詞"),
        "findings": (
            "CALL_NAME_REDIRECTS_TO_ZH_SYNONYM_TEMPLATE",
            "ALL_FT33_OCCURRENCES_HAVE_LANGUAGE_ZH_AND_ONE_TARGET",
            "SEMANTIC_PROJECTION_EXCLUDES_CATEGORIES_LIVE_LINK_STATE_AND_AUTOMATIC_TRANSLITERATION"),
        "blockers": (),
        "unresolved": (),
    },
    "zh-alt-form": {
        "status": "REVIEWED_AUTHORIZED",
        "renderer_authorized": 1,
        "parameter_profile": "ZH_TARGET_ONLY_ALTERNATIVE_FORM_V1",
        "evidence_titles": (
            "Module:Zh/link", "Template:Zh-alt form",
            "Template:Zh-alt-form", "Template:Zh-l", "Template:Zh-l/doc"),
        "findings": (
            "CALL_NAME_REDIRECTS_TO_ZH_ALT_FORM",
            "ALL_FT33_OCCURRENCES_HAVE_EXACTLY_ONE_TARGET",
            "TARGET_IS_PASSED_TO_ZH_LINK_AND_FOLLOWED_BY_ALTERNATIVE_FORM_TEXT",
            "SEMANTIC_PROJECTION_EXCLUDES_CATEGORIES_AND_LIVE_LINK_STATE"),
        "blockers": (),
        "unresolved": (),
    },
    "†": {
        "status": "BLOCKED",
        "renderer_authorized": 0,
        "parameter_profile": "NO_RENDERER_WITHOUT_FROZEN_TEMPLATE_IDENTITY",
        "evidence_titles": (),
        "findings": (
            "CALL_IS_SYNTACTICALLY_A_TEMPLATE_TRANSCLUSION",
            "NO_TEMPLATE_DAGGER_PAGE_EXISTS_IN_THE_FROZEN_FULL_INDEX",
            "SURROUNDING_DEFINITIONS_DO_NOT_PROVE_THE_SYMBOLS_INTENDED_SEMANTICS"),
        "blockers": (
            "INTENDED_SEMANTICS_UNSPECIFIED",
            "MISSING_TRANSCLUSION_TARGET",
            "TEMPLATE_PAGE_ABSENT_FROM_FROZEN_SNAPSHOT"),
        "unresolved": ("Template:†",),
    },
}


# object-model: exception
class FT34PublicTemplateReviewError(RuntimeError):
    """The FT34 evidence, review decision, or predecessor identity drifted."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _repository_file(root: Path, relative_path: str) -> Path:
    path = (root / Path(*relative_path.split("/"))).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FT34PublicTemplateReviewError("FT34 predecessor path escaped")
    return path


def _evidence_page(title: str) -> dict[str, object]:
    identity = FT34_EVIDENCE_IDENTITIES[title]
    contributor_id, contributor_name = FT34_EVIDENCE_CONTRIBUTORS[title]
    contributor = {
        "kind": "registered",
        "user_id": contributor_id,
        "username": contributor_name,
    }
    revision_id = identity[2]
    timestamp = identity[4]
    page_id = identity[1]
    return {
        "attribution": (
            f'Wiktionary contributors; page_title="{title}"; '
            f"page_id={page_id}; revision_id={revision_id}; "
            f"revision_timestamp={timestamp}; contributor="
            + json.dumps(
                contributor, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"))),
        "content_bytes": identity[7],
        "content_sha256": identity[6],
        "contributor": contributor,
        "direct_dependencies": sorted(identity[9]),
        "license_id": "CC-BY-SA-4.0",
        "mediawiki_revision_sha1": identity[5],
        "namespace_id": identity[0],
        "official_url": "https://dumps.wikimedia.org/zhwiktionary/20260701/",
        "page_id": page_id,
        "parent_revision_id": identity[3],
        "revision_id": revision_id,
        "revision_timestamp": timestamp,
        "revision_url": (
            "https://zh.wiktionary.org/w/index.php?title="
            + quote(title, safe="") + "&oldid=" + str(revision_id)),
        "role": identity[8],
        "title": title,
    }


def _observed_definition(item: dict[str, object]) -> dict[str, object]:
    return {
        "definition_ordinal": item["definition_ordinal"],
        "page_id": item["page_id"],
        "raw_definition_sha256": item["raw_definition_sha256"],
        "raw_definition_text": item["raw_definition_text"],
        "template_names": item["template_names"],
        "title": item["title"],
    }


def build_ft34_public_template_review_value(
        *, repository_root: str | Path) -> dict[str, object]:
    """Build the immutable review from the frozen public census and snapshot."""
    root = Path(repository_root).resolve()
    census_path = _repository_file(root, FT34_CENSUS_RELATIVE_PATH)
    snapshot_path = _repository_file(root, FT34_SNAPSHOT_RELATIVE_PATH)
    if (_file_sha256(census_path) != FT34_CENSUS_SHA256
            or _file_sha256(snapshot_path) != FT34_SNAPSHOT_SHA256):
        raise FT34PublicTemplateReviewError("FT34 predecessor SHA drifted")
    try:
        census = json.loads(census_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FT34PublicTemplateReviewError(
            "FT34 census is unreadable") from error
    snapshot = read_mediawiki_dump_snapshot(snapshot_path)
    if (snapshot.source_key != FT34_SOURCE_KEY
            or snapshot.snapshot_id != FT34_SNAPSHOT_ID
            or snapshot.license_id != "CC-BY-SA-4.0"
            or snapshot.final_parser_report.full_eof_verified != 1):
        raise FT34PublicTemplateReviewError("FT34 snapshot identity drifted")
    gate = census.get("template_evidence_gate")
    definitions = census.get("definitions")
    if not isinstance(gate, dict) or not isinstance(definitions, list):
        raise FT34PublicTemplateReviewError("FT34 census structure drifted")
    by_name = {
        item.get("template_name"): item for item in gate.get("templates", [])
        if isinstance(item, dict)
    }
    reviews = []
    for name in sorted(FT34_REVIEW_SPECS):
        spec = FT34_REVIEW_SPECS[name]
        gate_item = by_name.get(name)
        if not isinstance(gate_item, dict) or gate_item.get(
                "frequency_gate_met") != 1:
            raise FT34PublicTemplateReviewError(
                "FT34 qualified-template inventory drifted")
        observed = [
            _observed_definition(item) for item in definitions
            if isinstance(item, dict)
            and item.get("eligible_for_v4_artifact") == 1
            and name in item.get("template_names", [])
        ]
        observed.sort(key=lambda item: (
            item["page_id"], item["definition_ordinal"]))
        if len(observed) != gate_item.get("occurrence_count"):
            raise FT34PublicTemplateReviewError(
                "FT34 occurrence inventory drifted")
        reviews.append({
            "authorized_parameter_profile": spec["parameter_profile"],
            "determinism_blockers": sorted(spec["blockers"]),
            "distinct_page_count": gate_item["distinct_page_count"],
            "distinct_revision_count": gate_item["distinct_revision_count"],
            "evidence_titles": sorted(spec["evidence_titles"]),
            "observed_definitions": observed,
            "occurrence_count": gate_item["occurrence_count"],
            "public_specification_findings": sorted(spec["findings"]),
            "renderer_authorized": spec["renderer_authorized"],
            "status": spec["status"],
            "template_name": name,
            "unresolved_dependency_titles": sorted(spec["unresolved"]),
        })
    return {
        "artifact_kind": FT34_REVIEW_FORMAT,
        "boundary": {
            "dataset_expansions": 0,
            "formal_receipt_publications": 0,
            "mastery_changes": 0,
            "paper_modifications": 0,
            "readiness_changes": 0,
            "teacher_llm_calls": 0,
            "training_runs": 0,
        },
        "census_relative_path": FT34_CENSUS_RELATIVE_PATH,
        "census_sha256": FT34_CENSUS_SHA256,
        "evidence_pages": [
            _evidence_page(title) for title in sorted(FT34_EVIDENCE_IDENTITIES)],
        "format_version": FT34_REVIEW_VERSION,
        "license_evidence_url": (
            "https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use"),
        "license_id": "CC-BY-SA-4.0",
        "renderer_authorized_count": 4,
        "review_date": FT34_REVIEW_DATE,
        "reviewed_template_count": 6,
        "reviews": reviews,
        "snapshot_absence_evidence": {
            "full_index_entry_count": 3191659,
            "index_content_sha256": (
                "52c11a50e77d5427f0c197909804608b227e58b61486c0e9a88eafe32b8480e1"),
            "missing_titles": ["Template:†"],
            "scan_scope": "FULL_FROZEN_MULTISTREAM_INDEX",
        },
        "snapshot_id": FT34_SNAPSHOT_ID,
        "snapshot_manifest_relative_path": FT34_SNAPSHOT_RELATIVE_PATH,
        "snapshot_manifest_sha256": FT34_SNAPSHOT_SHA256,
        "source_key": FT34_SOURCE_KEY,
        "stage": FT34_STAGE,
    }


def _validate_basic(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FT34PublicTemplateReviewError("FT34 manifest is not an object")
    if set(value) != {
            "artifact_kind", "boundary", "census_relative_path",
            "census_sha256", "evidence_pages", "format_version",
            "license_evidence_url", "license_id",
            "renderer_authorized_count", "review_date",
            "reviewed_template_count", "reviews", "snapshot_absence_evidence",
            "snapshot_id", "snapshot_manifest_relative_path",
            "snapshot_manifest_sha256", "source_key", "stage"}:
        raise FT34PublicTemplateReviewError("FT34 manifest fields drifted")
    if (value.get("artifact_kind") != FT34_REVIEW_FORMAT
            or value.get("format_version") != FT34_REVIEW_VERSION
            or value.get("stage") != FT34_STAGE
            or value.get("review_date") != FT34_REVIEW_DATE
            or value.get("census_relative_path") != FT34_CENSUS_RELATIVE_PATH
            or value.get("census_sha256") != FT34_CENSUS_SHA256
            or value.get("snapshot_manifest_relative_path")
            != FT34_SNAPSHOT_RELATIVE_PATH
            or value.get("snapshot_manifest_sha256") != FT34_SNAPSHOT_SHA256
            or value.get("source_key") != FT34_SOURCE_KEY
            or value.get("snapshot_id") != FT34_SNAPSHOT_ID
            or value.get("license_id") != "CC-BY-SA-4.0"
            or value.get("renderer_authorized_count") != 4
            or value.get("reviewed_template_count") != 6):
        raise FT34PublicTemplateReviewError("FT34 manifest identity drifted")
    boundary = value.get("boundary")
    if (not isinstance(boundary, dict) or set(boundary) != {
            "dataset_expansions", "formal_receipt_publications",
            "mastery_changes", "paper_modifications", "readiness_changes",
            "teacher_llm_calls", "training_runs"}
            or any(item != 0 for item in boundary.values())):
        raise FT34PublicTemplateReviewError("FT34 boundary drifted")
    evidence = value.get("evidence_pages")
    expected_evidence = [
        _evidence_page(title) for title in sorted(FT34_EVIDENCE_IDENTITIES)]
    if evidence != expected_evidence:
        raise FT34PublicTemplateReviewError("FT34 evidence identity drifted")
    if value.get("snapshot_absence_evidence") != {
            "full_index_entry_count": 3191659,
            "index_content_sha256": (
                "52c11a50e77d5427f0c197909804608b227e58b61486c0e9a88eafe32b8480e1"),
            "missing_titles": ["Template:†"],
            "scan_scope": "FULL_FROZEN_MULTISTREAM_INDEX",
    }:
        raise FT34PublicTemplateReviewError("FT34 absence evidence drifted")
    reviews = value.get("reviews")
    if (not isinstance(reviews, list) or len(reviews) != 6
            or [item.get("template_name") for item in reviews]
            != sorted(FT34_REVIEW_SPECS)):
        raise FT34PublicTemplateReviewError("FT34 review inventory drifted")
    statuses = {item.get("status") for item in reviews}
    if not statuses.issubset({
            "REVIEWED_AUTHORIZED", "REVIEWED_NOT_AUTHORIZED", "BLOCKED"}):
        raise FT34PublicTemplateReviewError("FT34 review status drifted")
    for item in reviews:
        if set(item) != {
                "authorized_parameter_profile", "determinism_blockers",
                "distinct_page_count", "distinct_revision_count",
                "evidence_titles", "observed_definitions", "occurrence_count",
                "public_specification_findings", "renderer_authorized",
                "status", "template_name", "unresolved_dependency_titles"}:
            raise FT34PublicTemplateReviewError("FT34 review fields drifted")
        name = item["template_name"]
        spec = FT34_REVIEW_SPECS[name]
        expected_static = {
            "authorized_parameter_profile": spec["parameter_profile"],
            "determinism_blockers": sorted(spec["blockers"]),
            "evidence_titles": sorted(spec["evidence_titles"]),
            "public_specification_findings": sorted(spec["findings"]),
            "renderer_authorized": spec["renderer_authorized"],
            "status": spec["status"],
            "unresolved_dependency_titles": sorted(spec["unresolved"]),
        }
        if any(item.get(key) != expected
               for key, expected in expected_static.items()):
            raise FT34PublicTemplateReviewError("FT34 review decision drifted")
        observed = item.get("observed_definitions")
        if (not isinstance(observed, list)
                or len(observed) != item.get("occurrence_count")):
            raise FT34PublicTemplateReviewError(
                "FT34 observed inventory drifted")
        page_ids = set()
        for occurrence in observed:
            if (not isinstance(occurrence, dict) or set(occurrence) != {
                    "definition_ordinal", "page_id", "raw_definition_sha256",
                    "raw_definition_text", "template_names", "title"}):
                raise FT34PublicTemplateReviewError(
                    "FT34 observed fields drifted")
            raw_text = occurrence["raw_definition_text"]
            templates = occurrence["template_names"]
            if (not isinstance(raw_text, str) or not raw_text
                    or hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
                    != occurrence["raw_definition_sha256"]
                    or not isinstance(templates, list) or name not in templates
                    or templates != sorted(set(templates))
                    or type(occurrence["page_id"]) is not int
                    or type(occurrence["definition_ordinal"]) is not int):
                raise FT34PublicTemplateReviewError(
                    "FT34 observed identity drifted")
            page_ids.add(occurrence["page_id"])
        if len(page_ids) != item.get("distinct_page_count"):
            raise FT34PublicTemplateReviewError("FT34 page count drifted")
    return value


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class FT34PublicTemplateReviewManifest:
    """Canonical immutable six-template public review."""

    payload: CanonicalJsonObject

    def __post_init__(self) -> None:
        if not isinstance(self.payload, CanonicalJsonObject):
            raise TypeError("FT34 manifest payload type is invalid")
        _validate_basic(self.payload.to_value())

    def to_dict(self) -> dict[str, Any]:
        return self.payload.to_value()

    def canonical_bytes(self) -> bytes:
        return self.payload.payload + b"\n"

    @property
    def reviews(self) -> tuple[dict[str, Any], ...]:
        return tuple(self.to_dict()["reviews"])


def read_ft34_public_template_review(
        path: str | Path) -> FT34PublicTemplateReviewManifest:
    """Read canonical FT34 bytes and reject malformed review state."""
    try:
        payload = Path(path).resolve().read_bytes()
        if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
            raise FT34PublicTemplateReviewError("FT34 newline drifted")
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        manifest = FT34PublicTemplateReviewManifest(
            CanonicalJsonObject.from_value(value))
    except FT34PublicTemplateReviewError:
        raise
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise FT34PublicTemplateReviewError(
            "FT34 manifest is unreadable") from error
    if manifest.canonical_bytes() != payload:
        raise FT34PublicTemplateReviewError("FT34 manifest is not canonical")
    return manifest


def validate_ft34_public_template_review_sources(
        manifest: FT34PublicTemplateReviewManifest,
        *, repository_root: str | Path) -> None:
    """Recompute the review from frozen public predecessors."""
    if not isinstance(manifest, FT34PublicTemplateReviewManifest):
        raise TypeError("FT34 source validation requires a review manifest")
    expected = build_ft34_public_template_review_value(
        repository_root=repository_root)
    if manifest.to_dict() != expected:
        raise FT34PublicTemplateReviewError("FT34 review evidence drifted")


__all__ = [
    "FT34PublicTemplateReviewError",
    "FT34PublicTemplateReviewManifest",
    "FT34_REVIEW_FORMAT",
    "FT34_REVIEW_VERSION",
    "FT34_STAGE",
    "build_ft34_public_template_review_value",
    "read_ft34_public_template_review",
    "validate_ft34_public_template_review_sources",
]
