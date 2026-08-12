"""FT32 public specification review for frequent Wiktionary templates."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import quote

from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_mediawiki_snapshot import (
    read_mediawiki_dump_snapshot,
)


FT32_REVIEW_FORMAT = "PH2_FT32_PUBLIC_TEMPLATE_SPECIFICATION_REVIEW"
FT32_REVIEW_VERSION = 1
FT32_STAGE = "FT32-PUBLIC-TEMPLATE-SPECIFICATION-REVIEW-PLACE-ZH-DIV"
FT32_CENSUS_RELATIVE_PATH = (
    "data/ph2/manifests/ft31_w03_public_definition_census_v3.json")
FT32_CENSUS_SHA256 = (
    "4f637b372bc69c6c10e63ddba087eee1bfab108ed6381802323e39319242bdc4")
FT32_SNAPSHOT_RELATIVE_PATH = (
    "data/ph2/manifests/zhwiktionary_20260701.multistream_snapshot.json")
FT32_SNAPSHOT_SHA256 = (
    "9d0c82e39719a8084eb5bd672ba984589952874ee248c04816a00c7be20f2fdc")
FT32_SOURCE_KEY = "ZHWIKTIONARY_20260701"
FT32_SNAPSHOT_ID = "zhwiktionary-20260701-adapter-v1-double-pass-v1"

# title -> namespace, page, revision, parent, timestamp, MediaWiki SHA-1,
# content SHA-256, UTF-8 byte count, evidence role, direct dependencies.
FT32_EVIDENCE_IDENTITIES = {
    "Module:Place": (
        828, 1629826, 9800215, 9766382, "2026-05-29T18:30:39Z",
        "cyxf7ogd6qgww6jugsjbffw7n0hwqbf",
        "6446e354fbd38bc214e094e447381b5ef8dbb2466de6b6f25744c919fd05ae28",
        72816, "IMPLEMENTATION_MODULE", (
            "Module:links", "Module:memoize", "Module:parameters",
            "Module:place/data", "Module:place/placetypes",
            "Module:string utilities", "Module:table")),
    "Module:Place/data": (
        828, 1629828, 9800213, 9790150, "2026-05-29T18:30:02Z",
        "jc9e8gszwzawqlbh7k7umbl2tthz5ag",
        "d83acfde372882d6663b43f587bac883f871018bc6e534e17d182decda0a5840",
        30171, "LOCALIZATION_DATA_MODULE", ()),
    "Module:Place/placetypes": (
        828, 3437750, 9800214, 9790151, "2026-05-29T18:30:28Z",
        "ir4qrqnwy6heniff2wjaf03wiwrremk",
        "66763fa9f664c5577337e665eea9abbd7c4cf5cdf24c9237db8684d22057485e",
        239144, "PLACETYPE_MODULE", (
            "Module:headword/data", "Module:links", "Module:place/data",
            "Module:place/locations", "Module:string utilities",
            "Module:table")),
    "Module:Zh/templates": (
        828, 1889664, 9109462, 9109457, "2025-02-04T10:29:36Z",
        "r3lllsy17u7ogwxqo66r60gb6p6n73z",
        "151351230eca290cc95f64262647e5ff9a77a68d84a5e163220342eab79e7d14",
        10932, "IMPLEMENTATION_MODULE", (
            "Module:columns/old", "Module:debug/track", "Module:languages",
            "Module:links", "Module:parameters", "Module:zh",
            "Module:zh-cat", "Module:zh/data/ts", "Module:zh/extract",
            "Module:zh/link")),
    "Template:Place": (
        10, 1428131, 7272726, 5815423, "2022-07-31T02:55:57Z",
        "e5udahvzm41nr8p2qjycaj5kcvpvalr",
        "11433b755f77a4d904816ec4a393285d8727521d4a003903dd08c3be38b6ff0e",
        89, "TEMPLATE_ENTRY", ("Module:Place",)),
    "Template:Place/doc": (
        10, 1996020, 7365906, 7365892, "2022-09-30T08:05:40Z",
        "0delymgna2brm72r3tmi45g5o9xsaf0",
        "a52d34c527e3d4f7c5207c3de452b9d823f421922bc7bd99bd4ed0e362a02caf",
        22329, "PUBLIC_DOCUMENTATION", ("Module:Place/data",)),
    "Template:Zh-div": (
        10, 1428132, 7403525, 7238505, "2022-10-24T20:11:17Z",
        "44k1cyyeh7k13jvvv88hpif68uk863n",
        "bba4031d7651d6e9a23de96cf3d42c5c4257bbfda64a9617294dc2148817be9b",
        95, "TEMPLATE_ENTRY", ("Module:Zh/templates",)),
    "Template:Zh-div/doc": (
        10, 1632706, 6238445, 0, "2021-08-08T06:10:20Z",
        "jg5w3m913dlcreym4k4pqllaelhw35y",
        "66da9abdeb27edb21fd29002a8b64d6a670068e2421754b9851a07cfd6f5cc2b",
        925, "PUBLIC_DOCUMENTATION", ()),
}

FT32_EVIDENCE_CONTRIBUTORS = {
    "Module:Place": {
        "kind": "registered", "user_id": 53191, "username": "TongcyDai"},
    "Module:Place/data": {
        "kind": "registered", "user_id": 53191, "username": "TongcyDai"},
    "Module:Place/placetypes": {
        "kind": "registered", "user_id": 53191, "username": "TongcyDai"},
    "Module:Zh/templates": {
        "kind": "registered", "user_id": 53191, "username": "TongcyDai"},
    "Template:Place": {
        "kind": "registered", "user_id": 25402, "username": "Xiplus"},
    "Template:Place/doc": {
        "kind": "registered", "user_id": 63671, "username": "GnolizX"},
    "Template:Zh-div": {
        "kind": "registered", "user_id": 25402, "username": "Xiplus"},
    "Template:Zh-div/doc": {
        "ip": "180.217.38.176", "kind": "ip"},
}

# status, pages, revisions, occurrences, determinism blockers.
FT32_REVIEW_OUTCOMES = {
    "place": (
        "BLOCKED", 5, 5, 6, (
            "OUTPUT_SEMANTICS_NOT_CLOSED_FOR_ALL_PARAMETERS",
            "PUBLIC_SPECIFICATION_EXPERIMENTAL",
            "TRANSITIVE_DEPENDENCY_GRAPH_UNCLOSED")),
    "zh-div": (
        "REVIEWED_NOT_AUTHORIZED", 3, 3, 3, (
            "OUTPUT_DEPENDS_ON_CURRENT_PAGE_TITLE",
            "OUTPUT_DEPENDS_ON_LIVE_REDIRECT_STATE",
            "OUTPUT_DEPENDS_ON_LIVE_TARGET_EXISTENCE",
            "UNFROZEN_MODULE_DEPENDENCIES")),
}

# template -> evidence titles, specification findings, unresolved dependencies.
FT32_REVIEW_LIST_IDENTITIES = {
    "place": (
        (
            "Module:Place", "Module:Place/data", "Module:Place/placetypes",
            "Template:Place", "Template:Place/doc",
        ),
        (
            "ALTERNATIVE_FORMAT_SUPPORTS_EMBEDDED_PLACETYPES_AND_HOLONYMS",
            "DEFINITION_OVERRIDE_PARAMETER_EXISTS",
            "DOCUMENTATION_MARKS_TEMPLATE_EXPERIMENTAL",
            "ENTRY_INVOKES_MODULE_PLACE_SHOW",
            "POSITIONAL_AND_NAMED_PARAMETER_SURFACE_IS_OPEN_ENDED",
        ),
        (
            "Module:headword/data", "Module:links", "Module:memoize",
            "Module:parameters", "Module:place/locations",
            "Module:string utilities", "Module:table",
        ),
    ),
    "zh-div": (
        (
            "Module:Zh/templates", "Template:Zh-div", "Template:Zh-div/doc",
        ),
        (
            "DOCUMENTATION_DEFINES_ADMINISTRATIVE_OR_TAXONOMIC_LABEL",
            "ENTRY_INVOKES_MODULE_ZH_TEMPLATES_DIV",
            "FORMER_LABEL_PARAMETERS_ARE_OPTIONAL_AND_REPEATABLE",
            "IMPLEMENTATION_BRANCHES_ON_TARGET_EXISTENCE_AND_REDIRECT",
            "IMPLEMENTATION_READS_CURRENT_PAGE_TITLE",
        ),
        (
            "Module:columns/old", "Module:debug/track", "Module:languages",
            "Module:links", "Module:parameters", "Module:zh", "Module:zh-cat",
            "Module:zh/data/ts", "Module:zh/extract", "Module:zh/link",
        ),
    ),
}

FT32_OBSERVED_IDENTITIES = {
    "place": (
        (313346, "蜀", 7,
         "39c181810b489e0a4ddda9e0d243f43189fc8c8970d1412ec37854af14e57a94",
         ("place",)),
        (1375622, "江蘇", 1,
         "d0acdc7a490489a2767c48783345a7555c7225d150435790709a1ff66c575b61",
         ("place", "zh-div")),
        (2636880, "聖克里斯多福及尼維斯", 1,
         "96702dda807c934906dae6f5ab7e07db588d803f95e195e74f06d5c3a0a99358",
         ("lb", "place")),
        (2908808, "華盛頓哥倫比亞特區", 1,
         "70c294a0ffdd3d9bc395c69c0c19ab8721d19ed4738ec3ffe7cd829cb79f491a",
         ("place",)),
        (3198447, "第聶伯羅彼得羅夫斯克", 1,
         "7510f36cd2566cd4eacf02ab11d65d61598f6236ec79480af7360949338b6da0",
         ("place",)),
        (3198447, "第聶伯羅彼得羅夫斯克", 2,
         "8e238d7a60557f9a2176a2d029beba4a0d7a70879a91613840a86dbd3128bcd7",
         ("place", "zh-div")),
    ),
    "zh-div": (
        (313346, "蜀", 3,
         "fb54a1b60674db45c48dd158cd562a9aae2f2cf41ab141309528c0aefed3e1e7",
         ("w", "zh-div")),
        (1375622, "江蘇", 1,
         "d0acdc7a490489a2767c48783345a7555c7225d150435790709a1ff66c575b61",
         ("place", "zh-div")),
        (3198447, "第聶伯羅彼得羅夫斯克", 2,
         "8e238d7a60557f9a2176a2d029beba4a0d7a70879a91613840a86dbd3128bcd7",
         ("place", "zh-div")),
    ),
}


# object-model: exception
class FT32PublicTemplateReviewError(RuntimeError):
    """The public review or one of its predecessor identities drifted."""


def _exact(value: object, keys: tuple[str, ...], *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(keys):
        raise FT32PublicTemplateReviewError(f"{where} field set drifted")
    return value


def _sha256(value: object, *, where: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise FT32PublicTemplateReviewError(f"{where} is not SHA-256")
    return value


def _sorted_texts(value: object, *, where: str, allow_empty: bool = False) -> tuple[str, ...]:
    if (not isinstance(value, list)
            or any(not isinstance(item, str) or not item for item in value)):
        raise FT32PublicTemplateReviewError(f"{where} is invalid")
    result = tuple(value)
    if ((not allow_empty and not result)
            or tuple(sorted(set(result))) != result):
        raise FT32PublicTemplateReviewError(f"{where} is not canonical")
    return result


def _validate_contributor(value: object) -> None:
    if not isinstance(value, dict):
        raise FT32PublicTemplateReviewError("FT32 contributor is invalid")
    expected = {
        "registered": {"kind", "user_id", "username"},
        "ip": {"ip", "kind"},
    }.get(value.get("kind"))
    if expected is None or set(value) != expected:
        raise FT32PublicTemplateReviewError("FT32 contributor fields drifted")


def _observed_key(value: object, *, template_name: str) -> tuple[object, ...]:
    item = _exact(value, (
        "definition_ordinal", "page_id", "raw_definition_sha256",
        "template_names", "title",
    ), where="FT32 observed definition")
    templates = _sorted_texts(
        item["template_names"], where="FT32 observed templates")
    if (template_name not in templates
            or type(item["page_id"]) is not int or item["page_id"] <= 0
            or type(item["definition_ordinal"]) is not int
            or item["definition_ordinal"] <= 0
            or not isinstance(item["title"], str) or not item["title"]):
        raise FT32PublicTemplateReviewError("FT32 observed definition drifted")
    return (
        item["page_id"], item["title"], item["definition_ordinal"],
        _sha256(item["raw_definition_sha256"], where="FT32 definition SHA"),
        templates,
    )


def _validate_value(value: object) -> dict[str, Any]:
    root = _exact(value, (
        "artifact_kind", "boundary", "census_relative_path",
        "census_sha256", "evidence_pages", "format_version",
        "license_evidence_url", "license_id", "renderer_authorized_count",
        "review_date", "reviewed_template_count", "reviews", "snapshot_id",
        "snapshot_manifest_relative_path", "snapshot_manifest_sha256",
        "source_key", "stage",
    ), where="FT32 manifest")
    boundary = _exact(root["boundary"], (
        "formal_receipt_publications", "mastery_changes", "readiness_changes",
        "renderer_implementations", "teacher_llm_calls", "training_runs",
    ), where="FT32 boundary")
    if (
        root["artifact_kind"] != FT32_REVIEW_FORMAT
        or root["format_version"] != FT32_REVIEW_VERSION
        or root["stage"] != FT32_STAGE
        or root["census_relative_path"] != FT32_CENSUS_RELATIVE_PATH
        or root["census_sha256"] != FT32_CENSUS_SHA256
        or root["snapshot_manifest_relative_path"]
        != FT32_SNAPSHOT_RELATIVE_PATH
        or root["snapshot_manifest_sha256"] != FT32_SNAPSHOT_SHA256
        or root["source_key"] != FT32_SOURCE_KEY
        or root["snapshot_id"] != FT32_SNAPSHOT_ID
        or root["license_id"] != "CC-BY-SA-4.0"
        or root["license_evidence_url"]
        != "https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use"
        or root["review_date"] != "2026-08-12"
        or root["reviewed_template_count"] != 2
        or root["renderer_authorized_count"] != 0
        or any(item != 0 for item in boundary.values())
    ):
        raise FT32PublicTemplateReviewError("FT32 manifest boundary drifted")

    evidence = root["evidence_pages"]
    if not isinstance(evidence, list) or len(evidence) != 8:
        raise FT32PublicTemplateReviewError("FT32 evidence inventory drifted")
    evidence_titles = []
    for raw in evidence:
        item = _exact(raw, (
            "attribution", "content_bytes", "content_sha256", "contributor",
            "direct_dependencies", "license_id", "mediawiki_revision_sha1",
            "namespace_id", "official_url", "page_id", "parent_revision_id",
            "revision_id", "revision_timestamp", "revision_url", "role",
            "title",
        ), where="FT32 evidence")
        title = item["title"]
        expected = FT32_EVIDENCE_IDENTITIES.get(title)
        actual = (
            item["namespace_id"], item["page_id"], item["revision_id"],
            item["parent_revision_id"], item["revision_timestamp"],
            item["mediawiki_revision_sha1"], item["content_sha256"],
            item["content_bytes"], item["role"],
            _sorted_texts(item["direct_dependencies"],
                          where="FT32 dependencies", allow_empty=True),
        )
        if expected is None or actual != expected:
            raise FT32PublicTemplateReviewError("FT32 evidence identity drifted")
        _sha256(item["content_sha256"], where="FT32 evidence content")
        _validate_contributor(item["contributor"])
        contributor = FT32_EVIDENCE_CONTRIBUTORS[title]
        expected_revision_url = (
            "https://zh.wiktionary.org/w/index.php?title="
            + quote(title, safe="") + "&oldid=" + str(item["revision_id"]))
        expected_attribution = (
            f'Wiktionary contributors; page_title="{title}"; '
            f'page_id={item["page_id"]}; revision_id={item["revision_id"]}; '
            f'revision_timestamp={item["revision_timestamp"]}; contributor='
            + json.dumps(
                contributor, ensure_ascii=False, sort_keys=True,
                separators=(",", ":")))
        if (not re.fullmatch(r"[0-9a-z]{31}", item["mediawiki_revision_sha1"])
                or item["contributor"] != contributor
                or item["license_id"] != "CC-BY-SA-4.0"
                or item["official_url"]
                != "https://dumps.wikimedia.org/zhwiktionary/20260701/"
                or item["revision_url"] != expected_revision_url
                or item["attribution"] != expected_attribution):
            raise FT32PublicTemplateReviewError("FT32 evidence source drifted")
        evidence_titles.append(title)
    if tuple(evidence_titles) != tuple(sorted(FT32_EVIDENCE_IDENTITIES)):
        raise FT32PublicTemplateReviewError("FT32 evidence order drifted")

    reviews = root["reviews"]
    if not isinstance(reviews, list) or len(reviews) != 2:
        raise FT32PublicTemplateReviewError("FT32 review inventory drifted")
    review_names = []
    for raw in reviews:
        item = _exact(raw, (
            "determinism_blockers", "distinct_page_count",
            "distinct_revision_count", "evidence_titles",
            "observed_definitions", "occurrence_count",
            "public_specification_findings", "renderer_authorized", "status",
            "template_name", "unresolved_dependency_titles",
        ), where="FT32 review")
        name = item["template_name"]
        expected = FT32_REVIEW_OUTCOMES.get(name)
        blockers = _sorted_texts(
            item["determinism_blockers"], where="FT32 blockers")
        actual = (
            item["status"], item["distinct_page_count"],
            item["distinct_revision_count"], item["occurrence_count"],
            blockers,
        )
        observed = item["observed_definitions"]
        if (expected is None or actual != expected
                or item["renderer_authorized"] != 0
                or not isinstance(observed, list)
                or len(observed) != item["occurrence_count"]):
            raise FT32PublicTemplateReviewError("FT32 review outcome drifted")
        titles = _sorted_texts(
            item["evidence_titles"], where="FT32 review evidence")
        if not set(titles).issubset(evidence_titles):
            raise FT32PublicTemplateReviewError("FT32 review evidence is missing")
        findings = _sorted_texts(
            item["public_specification_findings"], where="FT32 findings")
        unresolved = _sorted_texts(
            item["unresolved_dependency_titles"],
            where="FT32 unresolved dependencies")
        if (titles, findings, unresolved) != FT32_REVIEW_LIST_IDENTITIES[name]:
            raise FT32PublicTemplateReviewError("FT32 review lists drifted")
        keys = tuple(_observed_key(entry, template_name=name)
                     for entry in observed)
        if (keys != FT32_OBSERVED_IDENTITIES[name]
                or len({entry[0] for entry in keys})
                != item["distinct_page_count"]):
            raise FT32PublicTemplateReviewError("FT32 occurrences drifted")
        review_names.append(name)
    if tuple(review_names) != tuple(sorted(FT32_REVIEW_OUTCOMES)):
        raise FT32PublicTemplateReviewError("FT32 review order drifted")
    return root


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class FT32PublicTemplateReviewManifest:
    """Canonical immutable review bytes with no renderer authorization."""

    payload: CanonicalJsonObject

    def __post_init__(self) -> None:
        if not isinstance(self.payload, CanonicalJsonObject):
            raise TypeError("FT32 manifest payload type is invalid")
        _validate_value(self.payload.to_value())

    def to_dict(self) -> dict[str, Any]:
        return self.payload.to_value()

    def canonical_bytes(self) -> bytes:
        return self.payload.payload + b"\n"

    @property
    def reviews(self) -> tuple[dict[str, Any], ...]:
        return tuple(self.to_dict()["reviews"])


def read_ft32_public_template_review(
        path: str | Path,
        ) -> FT32PublicTemplateReviewManifest:
    """Read canonical review bytes and reject status or evidence drift."""
    try:
        payload = Path(path).resolve().read_bytes()
        if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
            raise FT32PublicTemplateReviewError("FT32 manifest newline drifted")
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        manifest = FT32PublicTemplateReviewManifest(
            CanonicalJsonObject.from_value(value))
    except FT32PublicTemplateReviewError:
        raise
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise FT32PublicTemplateReviewError("FT32 manifest is unreadable") from error
    if manifest.canonical_bytes() != payload:
        raise FT32PublicTemplateReviewError("FT32 manifest is not canonical")
    return manifest


def _repository_file(root: Path, relative_path: str) -> Path:
    path = (root / Path(*relative_path.split("/"))).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FT32PublicTemplateReviewError("FT32 predecessor path escaped")
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def validate_ft32_public_template_review_sources(
        manifest: FT32PublicTemplateReviewManifest,
        *,
        repository_root: str | Path,
        ) -> None:
    """Cross-check the FT31 census and public snapshot without private data."""
    if not isinstance(manifest, FT32PublicTemplateReviewManifest):
        raise TypeError("FT32 source validation requires a review manifest")
    value = manifest.to_dict()
    root = Path(repository_root).resolve()
    census_path = _repository_file(root, value["census_relative_path"])
    snapshot_path = _repository_file(
        root, value["snapshot_manifest_relative_path"])
    if (_file_sha256(census_path) != value["census_sha256"]
            or _file_sha256(snapshot_path) != value["snapshot_manifest_sha256"]):
        raise FT32PublicTemplateReviewError("FT32 predecessor SHA drifted")
    snapshot = read_mediawiki_dump_snapshot(snapshot_path)
    if (snapshot.source_key != value["source_key"]
            or snapshot.snapshot_id != value["snapshot_id"]
            or snapshot.license_id != "CC-BY-SA-4.0"):
        raise FT32PublicTemplateReviewError("FT32 snapshot contract drifted")
    census_payload = census_path.read_bytes()
    census = parse_canonical_json_bytes(
        census_payload[:-1], require_object=True)
    gate = census.get("template_evidence_gate")
    definitions = census.get("definitions")
    if not isinstance(gate, dict) or not isinstance(definitions, list):
        raise FT32PublicTemplateReviewError("FT32 census structure drifted")
    templates = gate.get("templates")
    if not isinstance(templates, list):
        raise FT32PublicTemplateReviewError("FT32 template gate drifted")
    by_name = {item.get("template_name"): item for item in templates
               if isinstance(item, dict)}
    eligible = [item for item in definitions
                if isinstance(item, dict)
                and item.get("eligible_for_v3_artifact") == 1]
    for review in value["reviews"]:
        name = review["template_name"]
        gate_item = by_name.get(name)
        if not isinstance(gate_item, dict) or (
                gate_item.get("frequency_gate_met") != 1
                or gate_item.get("distinct_page_count")
                != review["distinct_page_count"]
                or gate_item.get("distinct_revision_count")
                != review["distinct_revision_count"]
                or gate_item.get("occurrence_count")
                != review["occurrence_count"]):
            raise FT32PublicTemplateReviewError("FT32 frequency evidence drifted")
        observed = {
            _observed_key(item, template_name=name)
            for item in review["observed_definitions"]
        }
        expected = {
            (
                item.get("page_id"), item.get("title"),
                item.get("definition_ordinal"),
                item.get("raw_definition_sha256"),
                tuple(sorted(item.get("template_names", []))),
            )
            for item in eligible
            if name in item.get("template_names", [])
        }
        if observed != expected:
            raise FT32PublicTemplateReviewError("FT32 definition evidence drifted")


__all__ = [
    "FT32PublicTemplateReviewError",
    "FT32PublicTemplateReviewManifest",
    "FT32_REVIEW_FORMAT",
    "FT32_REVIEW_VERSION",
    "FT32_STAGE",
    "read_ft32_public_template_review",
    "validate_ft32_public_template_review_sources",
]
