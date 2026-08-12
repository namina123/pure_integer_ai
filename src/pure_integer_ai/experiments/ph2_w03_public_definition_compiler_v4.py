"""FT33 public-definition v4 compact artifact and full census compiler."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_mediawiki_inline_ast import (
    MEDIAWIKI_INLINE_PARSER_VERSION,
)
from pure_integer_ai.experiments.ph2_mediawiki_multistream_adapter import (
    MediaWikiPageError,
    extract_balanced_templates,
)
from pure_integer_ai.experiments.ph2_source_pack_compiler import read_source_pack
from pure_integer_ai.experiments.ph2_source_pack_contract import (
    stable_source_pack_key,
)
from pure_integer_ai.experiments.ph2_w03_public_definition_compiler_v2 import (
    FT30_PAGE_STATUSES,
    FT30_ZH_SECTION_NAMES,
    _base_artifact,
    _definition_census_value,
    _sha256_path,
    _source_pack_identity,
    _source_ref,
    extract_ft30_definition_candidates,
)
from pure_integer_ai.experiments.ph2_w03_public_template_review_ft32 import (
    read_ft32_public_template_review,
    validate_ft32_public_template_review_sources,
)
from pure_integer_ai.experiments.ph2_w03_public_sense_contract import (
    W03_PUBLIC_SENSE_FORMAT,
    W03_PUBLIC_SENSE_MAX_ARTIFACT_BYTES,
    W03_PUBLIC_SENSE_SCHEMA_VERSION,
    W03PublicSenseAlias,
    W03PublicSenseArtifact,
    W03PublicSenseEntry,
)


FT33_PUBLIC_SENSE_ARTIFACT_VERSION = 4
FT33_MAX_SOURCE_COUNT = 512
FT33_MAX_DEFINITION_COUNT = 8192
FT33_MAX_ENTRY_COUNT = 16384
FT33_CENSUS_FORMAT = "PH2_FT33_PUBLIC_DEFINITION_CENSUS"
FT33_CENSUS_VERSION = 4
FT33_TEMPLATE_MIN_DISTINCT_PAGES = 3
FT33_TEMPLATE_MIN_DISTINCT_REVISIONS = 3
FT33_TEMPLATE_MIN_OCCURRENCES = 3


# object-model: exception
class W03PublicDefinitionCompilerV4Error(RuntimeError):
    """FT33 source, v3 base, FT32 review, census, or artifact drifted."""


def _read_v3_census(path: Path, expected_sha256: str) -> dict[str, object]:
    """Strictly recover the complete FT31 256-page census identity."""
    if _sha256_path(path) != expected_sha256:
        raise W03PublicDefinitionCompilerV4Error("FT33 v3 census SHA drifted")
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise W03PublicDefinitionCompilerV4Error(
            "FT33 v3 census is unreadable") from error
    if (
        canonical_json_bytes(value) + b"\n" != raw
        or not isinstance(value, dict)
        or value.get("artifact_kind")
        != "PH2_FT31_PUBLIC_DEFINITION_CENSUS"
        or value.get("artifact_version") != 3
        or value.get("page_count") != 256
        or value.get("definition_count") != 125
        or value.get("eligible_definition_count") != 65
        or value.get("page_status_counts") != {
            "ACCEPTED_DEFINITION": 41,
            "NON_CHINESE_DEFINITION": 12,
            "NO_DEFINITION": 197,
            "REDIRECT": 6,
        }
        or value.get("render_status_counts") != {
            "DISPLAY": 67,
            "MALFORMED_MARKUP": 1,
            "UNSUPPORTED_MARKUP": 57,
        }
    ):
        raise W03PublicDefinitionCompilerV4Error(
            "FT33 v3 census bearing identity drifted")
    gate = value.get("template_evidence_gate")
    if not isinstance(gate, dict) or gate.get("renderer_authorized_count") != 0:
        raise W03PublicDefinitionCompilerV4Error(
            "FT33 v3 renderer boundary drifted")
    qualified = {
        item.get("template_name")
        for item in gate.get("templates", [])
        if isinstance(item, dict) and item.get("frequency_gate_met") == 1
    }
    if qualified != {"place", "zh-div"}:
        raise W03PublicDefinitionCompilerV4Error(
            "FT33 v3 frequency evidence drifted")
    return value


def _review_identity(
        *,
        repository_root: Path,
        review_path: Path,
        review_relative_path: str,
        expected_sha256: str,
        ) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    """Validate FT32 and return compact inherited zero-authorization decisions."""
    if _sha256_path(review_path) != expected_sha256:
        raise W03PublicDefinitionCompilerV4Error("FT33 FT32 review SHA drifted")
    review = read_ft32_public_template_review(review_path)
    validate_ft32_public_template_review_sources(
        review, repository_root=repository_root)
    decisions = []
    by_name = {}
    for item in review.reviews:
        decision = {
            "determinism_blockers": item["determinism_blockers"],
            "renderer_authorized": item["renderer_authorized"],
            "status": item["status"],
            "template_name": item["template_name"],
            "unresolved_dependency_titles": item[
                "unresolved_dependency_titles"],
        }
        if decision["renderer_authorized"] != 0:
            raise W03PublicDefinitionCompilerV4Error(
                "FT33 inherited renderer authorization is not zero")
        decisions.append(decision)
        by_name[item["template_name"]] = decision
    if tuple(sorted(by_name)) != ("place", "zh-div"):
        raise W03PublicDefinitionCompilerV4Error(
            "FT33 inherited review inventory drifted")
    return ({
        "decisions": decisions,
        "renderer_authorized_count": 0,
        "review_manifest_relative_path": review_relative_path,
        "review_manifest_sha256": expected_sha256,
    }, by_name)


def _template_evidence(
        definitions: list[dict[str, object]],
        pages: list[dict[str, object]],
        *,
        inherited_reviews: dict[str, dict[str, object]],
        review_sha256: str,
        ) -> list[dict[str, object]]:
    """Count unknown templates while preserving FT32 authorization decisions."""
    revision_by_page = {
        item["page_id"]: item["revision_id"] for item in pages}
    names = sorted({
        name
        for item in definitions
        if item["eligible_for_v4_artifact"] == 1
        for name in item["unknown_template_names"]
    })
    values = []
    for name in names:
        matches = [
            item for item in definitions
            if item["eligible_for_v4_artifact"] == 1
            and name in item["unknown_template_names"]
        ]
        page_ids = sorted({item["page_id"] for item in matches})
        revisions = sorted({revision_by_page[item] for item in page_ids})
        occurrence_count = 0
        for item in matches:
            try:
                occurrence_count += sum(
                    template.name == name
                    for template in extract_balanced_templates(
                        item["raw_definition_text"],
                        max_templates=4096,
                        max_depth=64,
                    ))
            except MediaWikiPageError as error:
                raise W03PublicDefinitionCompilerV4Error(
                    "FT33 template evidence cannot be recomputed") from error
        review = inherited_reviews.get(name)
        values.append({
            "distinct_page_count": len(page_ids),
            "distinct_revision_count": len(revisions),
            "frequency_gate_met": int(
                len(page_ids) >= FT33_TEMPLATE_MIN_DISTINCT_PAGES
                and len(revisions) >= FT33_TEMPLATE_MIN_DISTINCT_REVISIONS
                and occurrence_count >= FT33_TEMPLATE_MIN_OCCURRENCES),
            "inherited_review_manifest_sha256": (
                review_sha256 if review is not None else None),
            "inherited_review_status": (
                review["status"] if review is not None else "UNREVIEWED"),
            "occurrence_count": occurrence_count,
            "page_ids": page_ids,
            "public_specification_reviewed": int(review is not None),
            "renderer_authorized": 0,
            "template_name": name,
        })
    return values


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class FT33PublicDefinitionBuildV4:
    """FT33 v4 compact artifact and full-census build result."""

    artifact: W03PublicSenseArtifact
    census_value: dict[str, object]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.artifact, W03PublicSenseArtifact)
            or not isinstance(self.census_value, dict)
        ):
            raise TypeError("FT33 build result type mismatch")


def build_ft33_public_definition_artifact_v4(
        *,
        repository_root: str | Path,
        base_artifact_path: str | Path,
        base_artifact_sha256: str,
        base_census_path: str | Path,
        base_census_sha256: str,
        review_manifest_path: str | Path,
        review_manifest_relative_path: str,
        review_manifest_sha256: str,
        expansion_pack_relative_path: str,
        expansion_pack_root: str | Path,
        selection_manifest_sha256: str,
        ) -> FT33PublicDefinitionBuildV4:
    """Append the 512-title slice to frozen v3 and audit every selected page."""
    base = _base_artifact(
        Path(base_artifact_path).resolve(),
        base_artifact_sha256,
        expected_artifact_version=3,
    )
    if (
        len(base.source_packs) != 4
        or len(base.source_revisions) != 1
        or len(base.entries) != 117
        or len(base.aliases) != 10
    ):
        raise W03PublicDefinitionCompilerV4Error(
            "FT33 v3 compact base inventory drifted")
    base_census = _read_v3_census(
        Path(base_census_path).resolve(), base_census_sha256)
    inherited_review, inherited_by_name = _review_identity(
        repository_root=Path(repository_root).resolve(),
        review_path=Path(review_manifest_path).resolve(),
        review_relative_path=review_manifest_relative_path,
        expected_sha256=review_manifest_sha256,
    )
    pack_root = Path(expansion_pack_root).resolve()
    bundle = read_source_pack(pack_root)
    if (
        bundle.manifest.redistribution_policy != "PUBLIC"
        or bundle.manifest.w_stages != ("W-03",)
        or len(bundle.sources) != FT33_MAX_SOURCE_COUNT
        or len(bundle.observations) != FT33_MAX_SOURCE_COUNT
    ):
        raise W03PublicDefinitionCompilerV4Error(
            "FT33 expansion source-pack contract drifted")
    source_by_key = {item.stable_key: item for item in bundle.sources}
    entries = []
    aliases = []
    page_values = []
    definition_values = []
    accepted_definition_count = 0
    for observation in sorted(
            bundle.observations,
            key=lambda item: item.stable_key.stable_key()):
        source = source_by_key.get(observation.source_ref_key)
        if source is None:
            raise W03PublicDefinitionCompilerV4Error(
                "FT33 Observation has no SourceRef")
        payload = observation.typed_payload.to_value()
        raw = payload.get("raw_observation")
        if (
            not isinstance(raw, dict)
            or set(raw) != {
                "contributor", "page_id", "redirect_title",
                "revision_id", "text", "timestamp", "title",
            }
        ):
            raise W03PublicDefinitionCompilerV4Error(
                "FT33 raw Observation fields drifted")
        span = source.source_span.to_value()
        if span.get("selection_manifest_sha256") != selection_manifest_sha256:
            raise W03PublicDefinitionCompilerV4Error(
                "FT33 source selection SHA drifted")
        title = raw["title"]
        redirect = raw["redirect_title"]
        page_id = raw["page_id"]
        source_ref = _source_ref(source, raw)
        definitions = extract_ft30_definition_candidates(raw["text"])
        eligible = tuple(
            item for item in definitions
            if item.language_section in FT30_ZH_SECTION_NAMES)
        if redirect:
            page_status = "REDIRECT"
            aliases.append(W03PublicSenseAlias(
                title,
                redirect,
                observation.language,
                source_ref,
                observation.stable_key.stable_key(),
            ))
        elif eligible:
            page_status = "ACCEPTED_DEFINITION"
        elif definitions:
            page_status = "NON_CHINESE_DEFINITION"
        else:
            page_status = "NO_DEFINITION"
        for ordinal, definition in enumerate(definitions, start=1):
            census_item = _definition_census_value(
                definition,
                observation_key=observation.stable_key.stable_key(),
                page_id=page_id,
                source_ref=source_ref,
                title=title,
                title_sha256=span["title_sha256"],
                stratum=span["title_length_stratum"],
                ordinal=ordinal,
            )
            census_item["eligible_for_v4_artifact"] = census_item.pop(
                "eligible_for_v2_artifact")
            definition_values.append(census_item)
            if definition not in eligible:
                continue
            accepted_definition_count += 1
            entry_key = stable_source_pack_key(
                "ft33_public_definition_entry",
                observation.stable_key.to_list(),
                ordinal,
                definition.text,
            ).stable_key()
            sense_key = stable_source_pack_key(
                "ft33_public_definition_sense",
                page_id,
                raw["revision_id"],
                ordinal,
                definition.text,
            ).stable_key()
            concept_key = stable_source_pack_key(
                "ft33_public_definition_concept",
                page_id,
                ordinal,
                definition.text,
            ).stable_key()
            entries.append(W03PublicSenseEntry(
                entry_key,
                title,
                title,
                observation.language,
                "DEFINITION",
                definition.text,
                sense_key,
                concept_key,
                observation.stable_key.stable_key(),
                source_ref,
                ("definition",),
                1,
            ))
        page_values.append({
            "accepted_definition_count": len(eligible),
            "detected_definition_count": len(definitions),
            "page_id": page_id,
            "page_status": page_status,
            "redirect": int(bool(redirect)),
            "revision_id": raw["revision_id"],
            "source_ref": source_ref.to_dict(),
            "text_sha256": span["text_sha256"],
            "title": title,
            "title_length_stratum": span["title_length_stratum"],
            "title_sha256": span["title_sha256"],
        })
    if (
        len(definition_values) > FT33_MAX_DEFINITION_COUNT
        or accepted_definition_count != len(entries)
        or len(base.entries) + len(entries) > FT33_MAX_ENTRY_COUNT
    ):
        raise W03PublicDefinitionCompilerV4Error(
            "FT33 definition or entry budget exceeded")
    expansion_identity = _source_pack_identity(
        expansion_pack_relative_path, pack_root, bundle)
    artifact = W03PublicSenseArtifact(
        tuple(sorted(
            (*base.source_packs, expansion_identity),
            key=lambda item: (item.source_key, item.relative_path),
        )),
        base.source_revisions,
        tuple(sorted((*base.entries, *entries), key=lambda item: item.entry_key)),
        tuple(sorted(
            (*base.aliases, *aliases),
            key=lambda item: (
                item.language, item.alias_surface, item.target_surface,
                item.observation_key,
            ),
        )),
    )
    status_counts = {
        status: sum(item["page_status"] == status for item in page_values)
        for status in sorted(FT30_PAGE_STATUSES)
    }
    render_counts: dict[str, int] = {}
    for item in definition_values:
        status = item["render_status"]
        render_counts[status] = render_counts.get(status, 0) + 1
    template_evidence = _template_evidence(
        definition_values,
        page_values,
        inherited_reviews=inherited_by_name,
        review_sha256=review_manifest_sha256,
    )
    if any(item["renderer_authorized"] != 0 for item in template_evidence):
        raise W03PublicDefinitionCompilerV4Error(
            "FT33 renderer authorization boundary drifted")
    census = {
        "artifact_kind": FT33_CENSUS_FORMAT,
        "artifact_version": FT33_CENSUS_VERSION,
        "base_artifact_identity": {
            "alias_count": len(base.aliases),
            "artifact_sha256": base_artifact_sha256,
            "entry_count": len(base.entries),
            "source_pack_count": len(base.source_packs),
            "source_revision_count": len(base.source_revisions),
        },
        "base_census_sha256": base_census_sha256,
        "base_v3_census_identity": {
            "definition_count": base_census["definition_count"],
            "eligible_definition_count": base_census[
                "eligible_definition_count"],
            "page_count": base_census["page_count"],
            "page_status_counts": base_census["page_status_counts"],
            "render_status_counts": base_census["render_status_counts"],
        },
        "definition_count": len(definition_values),
        "definitions": sorted(
            definition_values,
            key=lambda item: (item["page_id"], item["definition_ordinal"]),
        ),
        "eligible_definition_count": accepted_definition_count,
        "expansion_pack_manifest_sha256": expansion_identity.manifest_sha256,
        "format_version": 1,
        "inherited_template_review": inherited_review,
        "page_count": len(page_values),
        "page_status_counts": status_counts,
        "pages": sorted(page_values, key=lambda item: item["page_id"]),
        "parser_version": MEDIAWIKI_INLINE_PARSER_VERSION,
        "render_status_counts": {
            key: render_counts[key] for key in sorted(render_counts)},
        "selection_manifest_sha256": selection_manifest_sha256,
        "template_evidence_gate": {
            "minimum_distinct_pages": FT33_TEMPLATE_MIN_DISTINCT_PAGES,
            "minimum_distinct_revisions": (
                FT33_TEMPLATE_MIN_DISTINCT_REVISIONS),
            "minimum_occurrences": FT33_TEMPLATE_MIN_OCCURRENCES,
            "renderer_authorized_count": 0,
            "templates": template_evidence,
        },
    }
    return FT33PublicDefinitionBuildV4(artifact, census)


def ft33_public_sense_artifact_envelope_v4(
        artifact: W03PublicSenseArtifact,
        ) -> dict[str, object]:
    """Build the experimental v4 envelope with all formal flags at zero."""
    return {
        "artifact_version": FT33_PUBLIC_SENSE_ARTIFACT_VERSION,
        "experimental": 1,
        "formal_mastery_claim": 0,
        "format": W03_PUBLIC_SENSE_FORMAT,
        "mastery": 0,
        "payload": artifact.payload_value(),
        "payload_sha256": artifact.payload_sha256(),
        "readiness": 0,
        "schema_version": W03_PUBLIC_SENSE_SCHEMA_VERSION,
        "w02_runtime_evidenced": 0,
        "w03_started": 0,
    }


def _write_immutable(path: Path, payload: bytes, *, where: str) -> Path:
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise W03PublicDefinitionCompilerV4Error(
                f"FT33 {where} exists with different bytes")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except OSError as error:
        raise W03PublicDefinitionCompilerV4Error(
            f"FT33 {where} cannot be published") from error
    return path


def write_ft33_public_sense_artifact_v4(
        artifact: W03PublicSenseArtifact,
        path: str | Path,
        ) -> Path:
    """Publish only the v4 compact projection to the runtime data surface."""
    payload = canonical_json_bytes(
        ft33_public_sense_artifact_envelope_v4(artifact)) + b"\n"
    if len(payload) > W03_PUBLIC_SENSE_MAX_ARTIFACT_BYTES:
        raise W03PublicDefinitionCompilerV4Error(
            "FT33 compact artifact exceeds the runtime budget")
    return _write_immutable(
        Path(path).resolve(), payload, where="compact artifact")


def write_ft33_public_definition_census_v4(
        census_value: dict[str, object],
        path: str | Path,
        ) -> Path:
    """Publish the canonical full census outside the installed runtime."""
    return _write_immutable(
        Path(path).resolve(),
        canonical_json_bytes(census_value) + b"\n",
        where="definition census",
    )


__all__ = [
    "FT33_CENSUS_FORMAT",
    "FT33_CENSUS_VERSION",
    "FT33_PUBLIC_SENSE_ARTIFACT_VERSION",
    "FT33PublicDefinitionBuildV4",
    "W03PublicDefinitionCompilerV4Error",
    "build_ft33_public_definition_artifact_v4",
    "ft33_public_sense_artifact_envelope_v4",
    "write_ft33_public_definition_census_v4",
    "write_ft33_public_sense_artifact_v4",
]
