"""FT31 公开定义 v3 compact artifact 与规模 census 编译器。"""
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
    W03PublicDefinitionCompilerV2Error,
    _base_artifact,
    _base_definition_rendering_baseline,
    _definition_census_value,
    _sha256_path,
    _source_pack_identity,
    _source_ref,
    extract_ft30_definition_candidates,
)
from pure_integer_ai.experiments.ph2_w03_public_sense_contract import (
    W03_PUBLIC_SENSE_FORMAT,
    W03_PUBLIC_SENSE_MAX_ARTIFACT_BYTES,
    W03_PUBLIC_SENSE_SCHEMA_VERSION,
    W03PublicSenseAlias,
    W03PublicSenseArtifact,
    W03PublicSenseEntry,
)


FT31_PUBLIC_SENSE_ARTIFACT_VERSION = 3
FT31_MAX_SOURCE_COUNT = 256
FT31_MAX_DEFINITION_COUNT = 4096
FT31_MAX_ENTRY_COUNT = 8192
FT31_CENSUS_FORMAT = "PH2_FT31_PUBLIC_DEFINITION_CENSUS"
FT31_CENSUS_VERSION = 3
FT31_TEMPLATE_MIN_DISTINCT_PAGES = 3
FT31_TEMPLATE_MIN_DISTINCT_REVISIONS = 3
FT31_TEMPLATE_MIN_OCCURRENCES = 3


# object-model: exception
class W03PublicDefinitionCompilerV3Error(RuntimeError):
    """FT31 来源、v2 基线、census 或 v3 artifact 身份发生漂移。"""


def _read_v2_census(path: Path, expected_sha256: str) -> dict[str, object]:
    """严格回读 FT30 全选择 census，并冻结其承重计数。"""
    if _sha256_path(path) != expected_sha256:
        raise W03PublicDefinitionCompilerV3Error("FT31 v2 census SHA 漂移")
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise W03PublicDefinitionCompilerV3Error(
            "FT31 v2 census 不可读") from error
    if (canonical_json_bytes(value) + b"\n" != raw
            or not isinstance(value, dict)
            or value.get("artifact_kind") != (
                "PH2_FT30_PUBLIC_DEFINITION_CENSUS")
            or value.get("artifact_version") != 2
            or value.get("page_count") != 32
            or value.get("definition_count") != 18
            or value.get("eligible_definition_count") != 9
            or value.get("page_status_counts") != {
                "ACCEPTED_DEFINITION": 4,
                "NON_CHINESE_DEFINITION": 2,
                "NO_DEFINITION": 23,
                "REDIRECT": 3,
            }
            or value.get("render_status_counts") != {
                "DISPLAY": 14,
                "UNSUPPORTED_MARKUP": 4,
            }):
        raise W03PublicDefinitionCompilerV3Error(
            "FT31 v2 census 承重身份漂移")
    return value


def _template_evidence(
        definitions: list[dict[str, object]],
        pages: list[dict[str, object]],
        ) -> list[dict[str, object]]:
    """统计中文真实定义中的未知模板支持，不据此自动授权 renderer。"""
    revision_by_page = {
        item["page_id"]: item["revision_id"] for item in pages}
    names = sorted({
        name
        for item in definitions
        if item["eligible_for_v3_artifact"] == 1
        for name in item["unknown_template_names"]
    })
    values = []
    for name in names:
        matches = [
            item for item in definitions
            if item["eligible_for_v3_artifact"] == 1
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
                raise W03PublicDefinitionCompilerV3Error(
                    "FT31 template evidence 无法复算") from error
        frequency_gate = int(
            len(page_ids) >= FT31_TEMPLATE_MIN_DISTINCT_PAGES
            and len(revisions) >= FT31_TEMPLATE_MIN_DISTINCT_REVISIONS
            and occurrence_count >= FT31_TEMPLATE_MIN_OCCURRENCES)
        values.append({
            "distinct_page_count": len(page_ids),
            "distinct_revision_count": len(revisions),
            "frequency_gate_met": frequency_gate,
            "occurrence_count": occurrence_count,
            "page_ids": page_ids,
            "public_specification_reviewed": 0,
            "renderer_authorized": 0,
            "template_name": name,
        })
    return values


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class FT31PublicDefinitionBuildV3:
    """v3 artifact 与全选择 census 构建结果。"""

    artifact: W03PublicSenseArtifact
    census_value: dict[str, object]

    def __post_init__(self) -> None:
        if (not isinstance(self.artifact, W03PublicSenseArtifact)
                or not isinstance(self.census_value, dict)):
            raise TypeError("FT31 build result 类型非法")


def build_ft31_public_definition_artifact_v3(
        *,
        base_artifact_path: str | Path,
        base_artifact_sha256: str,
        base_census_path: str | Path,
        base_census_sha256: str,
        expansion_pack_relative_path: str,
        expansion_pack_root: str | Path,
        selection_manifest_sha256: str,
        ) -> FT31PublicDefinitionBuildV3:
    """在冻结 v2 上追加中文定义/redirect，并审计全部 256 页。"""
    base = _base_artifact(
        Path(base_artifact_path).resolve(),
        base_artifact_sha256,
        expected_artifact_version=2,
    )
    base_definition_baseline = _base_definition_rendering_baseline(
        base,
        base_artifact_sha256,
        expected_counts={
            "DISPLAY": 16,
            "NO_SOURCE_ANSWER": 2,
            "UNKNOWN_TEMPLATE": 3,
        },
    )
    base_census = _read_v2_census(
        Path(base_census_path).resolve(), base_census_sha256)
    pack_root = Path(expansion_pack_root).resolve()
    bundle = read_source_pack(pack_root)
    if (bundle.manifest.redistribution_policy != "PUBLIC"
            or bundle.manifest.w_stages != ("W-03",)
            or len(bundle.sources) != FT31_MAX_SOURCE_COUNT
            or len(bundle.observations) != FT31_MAX_SOURCE_COUNT):
        raise W03PublicDefinitionCompilerV3Error(
            "FT31 expansion source pack 合同漂移")
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
            raise W03PublicDefinitionCompilerV3Error(
                "FT31 Observation 缺 SourceRef")
        payload = observation.typed_payload.to_value()
        raw = payload.get("raw_observation")
        if (not isinstance(raw, dict)
                or set(raw) != {
                    "contributor", "page_id", "redirect_title",
                    "revision_id", "text", "timestamp", "title"}):
            raise W03PublicDefinitionCompilerV3Error(
                "FT31 raw Observation 字段漂移")
        span = source.source_span.to_value()
        if span.get("selection_manifest_sha256") != selection_manifest_sha256:
            raise W03PublicDefinitionCompilerV3Error(
                "FT31 source selection SHA 漂移")
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
            census_item["eligible_for_v3_artifact"] = census_item.pop(
                "eligible_for_v2_artifact")
            definition_values.append(census_item)
            if definition not in eligible:
                continue
            accepted_definition_count += 1
            entry_key = stable_source_pack_key(
                "ft31_public_definition_entry",
                observation.stable_key.to_list(),
                ordinal,
                definition.text,
            ).stable_key()
            sense_key = stable_source_pack_key(
                "ft31_public_definition_sense",
                page_id,
                raw["revision_id"],
                ordinal,
                definition.text,
            ).stable_key()
            concept_key = stable_source_pack_key(
                "ft31_public_definition_concept",
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
    if (len(definition_values) > FT31_MAX_DEFINITION_COUNT
            or accepted_definition_count != len(entries)
            or len(base.entries) + len(entries) > FT31_MAX_ENTRY_COUNT):
        raise W03PublicDefinitionCompilerV3Error(
            "FT31 definition/entry 预算超限")
    expansion_identity = _source_pack_identity(
        expansion_pack_relative_path, pack_root, bundle)
    artifact = W03PublicSenseArtifact(
        tuple(sorted(
            (*base.source_packs, expansion_identity),
            key=lambda item: (item.source_key, item.relative_path))),
        base.source_revisions,
        tuple(sorted((*base.entries, *entries), key=lambda item: item.entry_key)),
        tuple(sorted(
            (*base.aliases, *aliases),
            key=lambda item: (
                item.language, item.alias_surface, item.target_surface,
                item.observation_key))),
    )
    status_counts = {
        status: sum(item["page_status"] == status for item in page_values)
        for status in sorted(FT30_PAGE_STATUSES)
    }
    render_counts: dict[str, int] = {}
    for item in definition_values:
        status = item["render_status"]
        render_counts[status] = render_counts.get(status, 0) + 1
    template_evidence = _template_evidence(definition_values, page_values)
    census = {
        "artifact_kind": FT31_CENSUS_FORMAT,
        "artifact_version": FT31_CENSUS_VERSION,
        "base_artifact_sha256": base_artifact_sha256,
        "base_census_sha256": base_census_sha256,
        "base_definition_count": len(base_definition_baseline),
        "base_definition_rendering_baseline": list(base_definition_baseline),
        "base_definition_rendering_counts": {
            outcome: sum(
                item["display_outcome"] == outcome
                for item in base_definition_baseline)
            for outcome in (
                "DISPLAY", "NO_SOURCE_ANSWER", "UNKNOWN_TEMPLATE")
        },
        "base_v2_census_identity": {
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
            key=lambda item: (item["page_id"], item["definition_ordinal"])),
        "eligible_definition_count": accepted_definition_count,
        "expansion_pack_manifest_sha256": expansion_identity.manifest_sha256,
        "format_version": 1,
        "page_count": len(page_values),
        "page_status_counts": status_counts,
        "pages": sorted(page_values, key=lambda item: item["page_id"]),
        "parser_version": MEDIAWIKI_INLINE_PARSER_VERSION,
        "render_status_counts": {
            key: render_counts[key] for key in sorted(render_counts)},
        "selection_manifest_sha256": selection_manifest_sha256,
        "template_evidence_gate": {
            "minimum_distinct_pages": FT31_TEMPLATE_MIN_DISTINCT_PAGES,
            "minimum_distinct_revisions": (
                FT31_TEMPLATE_MIN_DISTINCT_REVISIONS),
            "minimum_occurrences": FT31_TEMPLATE_MIN_OCCURRENCES,
            "renderer_authorized_count": 0,
            "templates": template_evidence,
        },
    }
    return FT31PublicDefinitionBuildV3(artifact, census)


def ft31_public_sense_artifact_envelope_v3(
        artifact: W03PublicSenseArtifact,
        ) -> dict[str, object]:
    """构造 v3 experimental/formal 零边界 envelope。"""
    return {
        "artifact_version": FT31_PUBLIC_SENSE_ARTIFACT_VERSION,
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
    """独占或幂等发布规范字节。"""
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise W03PublicDefinitionCompilerV3Error(
                f"FT31 {where} 已存在且字节不同")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except OSError as error:
        raise W03PublicDefinitionCompilerV3Error(
            f"FT31 {where} 无法发布") from error
    return path


def write_ft31_public_sense_artifact_v3(
        artifact: W03PublicSenseArtifact,
        path: str | Path,
        ) -> Path:
    """发布仅含 compact projection 的 v3 runtime artifact。"""
    payload = canonical_json_bytes(
        ft31_public_sense_artifact_envelope_v3(artifact)) + b"\n"
    if len(payload) > W03_PUBLIC_SENSE_MAX_ARTIFACT_BYTES:
        raise W03PublicDefinitionCompilerV3Error(
            "FT31 compact artifact 超预算")
    return _write_immutable(
        Path(path).resolve(), payload, where="compact artifact")


def write_ft31_public_definition_census_v3(
        census_value: dict[str, object],
        path: str | Path,
        ) -> Path:
    """发布不进入 wheel 的 canonical 全选择 census。"""
    return _write_immutable(
        Path(path).resolve(),
        canonical_json_bytes(census_value) + b"\n",
        where="definition census",
    )


__all__ = [
    "FT31_CENSUS_FORMAT",
    "FT31_CENSUS_VERSION",
    "FT31_PUBLIC_SENSE_ARTIFACT_VERSION",
    "FT31PublicDefinitionBuildV3",
    "W03PublicDefinitionCompilerV3Error",
    "build_ft31_public_definition_artifact_v3",
    "ft31_public_sense_artifact_envelope_v3",
    "write_ft31_public_definition_census_v3",
    "write_ft31_public_sense_artifact_v3",
]
