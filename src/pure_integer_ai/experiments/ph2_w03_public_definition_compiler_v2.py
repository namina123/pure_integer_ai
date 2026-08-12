"""FT30 公开定义 v2 compact artifact 与全选择 census 编译器。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_mediawiki_inline_ast import (
    MEDIAWIKI_INLINE_PARSER_VERSION,
    MediaWikiInlineParseError,
    project_mediawiki_inline,
)
from pure_integer_ai.experiments.ph2_mediawiki_multistream_adapter import (
    MediaWikiPageError,
    extract_balanced_templates,
)
from pure_integer_ai.experiments.ph2_source_pack_compiler import (
    read_source_pack,
)
from pure_integer_ai.experiments.ph2_source_pack_contract import (
    stable_source_pack_key,
)
from pure_integer_ai.experiments.ph2_w03_public_sense_contract import (
    W03_PUBLIC_SENSE_FORMAT,
    W03_PUBLIC_SENSE_MAX_ARTIFACT_BYTES,
    W03_PUBLIC_SENSE_SCHEMA_VERSION,
    W03PublicSenseAlias,
    W03PublicSenseArtifact,
    W03PublicSenseEntry,
    W03PublicSenseSourcePackIdentity,
    W03PublicSenseSourceRef,
)
from pure_integer_ai.experiments.ph2_w03_public_sense_runtime import (
    W03PublicSenseRuntime,
)
from pure_integer_ai.experiments.ph2_w03_w04_source_bound_primitive import (
    project_w03_public_sense_to_w04_primitives,
)
from pure_integer_ai.experiments.ph2_w04_w05_source_bound_proposition import (
    project_w04_primitives_to_w05_source_bound_propositions,
)
from pure_integer_ai.experiments.ph2_w05_definition_rendering import (
    render_w05_definition_answer,
)
from pure_integer_ai.experiments.ph2_w05_raw_definition_qa import (
    answer_w05_raw_definition_question,
)
from pure_integer_ai.experiments.ph2_w05_raw_definition_qa_contract import (
    W05RawDefinitionRequest,
)


FT30_PUBLIC_SENSE_ARTIFACT_VERSION = 2
FT30_MAX_SOURCE_COUNT = 32
FT30_MAX_DEFINITION_COUNT = 256
FT30_MAX_ENTRY_COUNT = 512
FT30_CENSUS_FORMAT = "PH2_FT30_PUBLIC_DEFINITION_CENSUS"
FT30_CENSUS_VERSION = 2
FT30_ZH_SECTION_NAMES = frozenset({"中文", "汉语", "漢語"})
FT30_PAGE_STATUSES = frozenset({
    "ACCEPTED_DEFINITION",
    "NO_DEFINITION",
    "NON_CHINESE_DEFINITION",
    "REDIRECT",
})

_HEADING_RE = re.compile(r"^(=+)([^=\r\n]+)\1\s*$")
_DEFINITION_RE = re.compile(r"^#(?![#*:])\s*(\S.*?)(?:\r?\n)?$")


# object-model: exception
class W03PublicDefinitionCompilerV2Error(RuntimeError):
    """FT30 来源、定义投影、census 或 artifact 身份发生漂移。"""


def _sha256_path(path: Path) -> str:
    """以固定块大小计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    """返回 UTF-8 文本 SHA-256。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class FT30DefinitionCandidate:
    """一个保持正文字符跨度与完整章节路径的 definition。"""

    section_path: tuple[str, ...]
    start: int
    end: int
    text: str

    def __post_init__(self) -> None:
        if (not isinstance(self.section_path, tuple)
                or any(not isinstance(item, str) or not item
                       for item in self.section_path)
                or type(self.start) is not int or type(self.end) is not int
                or self.start < 0 or self.end <= self.start
                or not isinstance(self.text, str) or not self.text
                or self.text.strip() != self.text):
            raise W03PublicDefinitionCompilerV2Error(
                "FT30 definition candidate 非规范")

    @property
    def language_section(self) -> str | None:
        """返回顶层语言章节；根级 definition 返回空。"""
        return self.section_path[0] if self.section_path else None


def extract_ft30_definition_candidates(
        text: str,
        ) -> tuple[FT30DefinitionCandidate, ...]:
    """仅按冻结 heading/definition 语法提取，不展开模板。"""
    if not isinstance(text, str) or not text:
        raise W03PublicDefinitionCompilerV2Error("FT30 page text 非法")
    headings: dict[int, str] = {}
    definitions = []
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\r\n")
        heading = _HEADING_RE.fullmatch(stripped)
        if heading is not None:
            level = len(heading.group(1))
            name = heading.group(2).strip()
            if not name:
                raise W03PublicDefinitionCompilerV2Error(
                    "FT30 Wiktionary heading 为空")
            headings = {
                key: value for key, value in headings.items() if key < level}
            headings[level] = name
        else:
            definition = _DEFINITION_RE.fullmatch(line)
            if definition is not None:
                value = definition.group(1)
                start = offset + line.index(value)
                definitions.append(FT30DefinitionCandidate(
                    tuple(headings[key] for key in sorted(headings)),
                    start,
                    start + len(value),
                    value,
                ))
        offset += len(line)
    return tuple(definitions)


def _render_status(code: str) -> str:
    """沿用 FT29 稳定 failure code 到展示状态的映射。"""
    if code == "AMBIGUOUS_LINK":
        return "AMBIGUOUS_RENDERING"
    if code in {
            "NESTED_MARKUP", "UNKNOWN_TEMPLATE",
            "UNSUPPORTED_INLINE_MARKUP", "UNSUPPORTED_LINK_TARGET",
            "UNSUPPORTED_VARIABLE"}:
        return "UNSUPPORTED_MARKUP"
    return "MALFORMED_MARKUP"


def _definition_census_value(
        definition: FT30DefinitionCandidate,
        *,
        observation_key: tuple[int, ...],
        page_id: int,
        source_ref: W03PublicSenseSourceRef,
        title: str,
        title_sha256: str,
        stratum: str,
        ordinal: int,
        ) -> dict[str, object]:
    """对一个定义运行 FT29 parser，并保留模板/link/AST 审计。"""
    template_names: tuple[str, ...]
    try:
        templates = extract_balanced_templates(
            definition.text,
            max_templates=4096,
            max_depth=64,
        )
        template_names = tuple(sorted({item.name for item in templates}))
    except MediaWikiPageError:
        template_names = ()
    failure_code = None
    ast_sha256 = None
    projection_sha256 = None
    display_text_sha256 = None
    node_kinds: tuple[str, ...] = ()
    try:
        projection = project_mediawiki_inline(definition.text)
        status = "DISPLAY"
        ast_sha256 = projection.document.ast_sha256
        projection_sha256 = projection.projection_sha256
        display_text_sha256 = _sha256_text(projection.display_text)
        node_kinds = tuple(
            item.to_dict()["kind"] for item in projection.document.nodes)
    except MediaWikiInlineParseError as error:
        failure_code = error.code
        status = _render_status(error.code)
    eligible = int(definition.language_section in FT30_ZH_SECTION_NAMES)
    return {
        "ast_sha256": ast_sha256,
        "definition_ordinal": ordinal,
        "display_text_sha256": display_text_sha256,
        "eligible_for_v2_artifact": eligible,
        "end": definition.end,
        "failure_code": failure_code,
        "label_template_count": sum(
            item in {"label", "lb"} for item in template_names),
        "language_section": definition.language_section,
        "link_open_count": definition.text.count("[["),
        "node_kinds": list(node_kinds),
        "observation_key": list(observation_key),
        "page_id": page_id,
        "parser_version": MEDIAWIKI_INLINE_PARSER_VERSION,
        "projection_sha256": projection_sha256,
        "raw_definition_sha256": _sha256_text(definition.text),
        "raw_definition_text": definition.text,
        "render_status": status,
        "section_path": list(definition.section_path),
        "source_ref": source_ref.to_dict(),
        "start": definition.start,
        "template_names": list(template_names),
        "title": title,
        "title_length_stratum": stratum,
        "title_sha256": title_sha256,
        "unknown_template_names": [
            item for item in template_names if item not in {"label", "lb"}],
    }


def _source_ref(record, raw: dict[str, object]) -> W03PublicSenseSourceRef:
    """把完整 SourceRefRecord 压缩为可独立归属的公开来源投影。"""
    contributor = raw.get("contributor")
    title = raw.get("title")
    page_id = raw.get("page_id")
    timestamp = raw.get("timestamp")
    if (not isinstance(contributor, dict)
            or not isinstance(title, str) or not title
            or type(page_id) is not int or page_id <= 0
            or not isinstance(timestamp, str) or not timestamp):
        raise W03PublicDefinitionCompilerV2Error(
            "FT30 compact attribution metadata 非法")
    attribution = (
        f"{record.attribution}; page_title="
        f"{json.dumps(title, ensure_ascii=False)}; page_id={page_id}; "
        f"revision_timestamp={timestamp}; contributor="
        f"{canonical_json_bytes(contributor).decode('utf-8')}")
    return W03PublicSenseSourceRef(
        record.stable_key.stable_key(),
        record.source_key,
        record.snapshot_id,
        record.revision_id,
        record.source_identity,
        record.official_url,
        record.license_id,
        attribution,
        hashlib.sha256(canonical_json_bytes(record.to_dict())).hexdigest(),
    )


def _source_pack_identity(
        relative_path: str,
        root: Path,
        bundle,
        ) -> W03PublicSenseSourcePackIdentity:
    """冻结 FT30 source pack manifest 身份与计数。"""
    snapshots = {item.snapshot_id for item in bundle.sources}
    if len(snapshots) != 1:
        raise W03PublicDefinitionCompilerV2Error(
            "FT30 source pack snapshot_id 非唯一")
    return W03PublicSenseSourcePackIdentity(
        relative_path,
        _sha256_path(root / "manifest.json"),
        bundle.manifest.stable_key.stable_key(),
        bundle.manifest.source_key,
        bundle.manifest.license_partition,
        next(iter(snapshots)),
        len(bundle.sources),
        len(bundle.observations),
    )


def _base_artifact(path: Path, expected_sha256: str) -> W03PublicSenseArtifact:
    """严格回读冻结 v1 artifact payload，不经 runtime 默认路径。"""
    if _sha256_path(path) != expected_sha256:
        raise W03PublicDefinitionCompilerV2Error("FT30 base artifact SHA 漂移")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise W03PublicDefinitionCompilerV2Error(
            "FT30 base artifact 不可读") from error
    if (not isinstance(value, dict)
            or value.get("artifact_version") != 1
            or value.get("format") != W03_PUBLIC_SENSE_FORMAT
            or hashlib.sha256(canonical_json_bytes(value.get("payload"))).hexdigest()
            != value.get("payload_sha256")):
        raise W03PublicDefinitionCompilerV2Error(
            "FT30 base artifact envelope 漂移")
    return W03PublicSenseArtifact.from_payload_value(value["payload"])


def _base_definition_rendering_baseline(
        artifact: W03PublicSenseArtifact,
        artifact_sha256: str,
        ) -> tuple[dict[str, object], ...]:
    """冻结 v1 的逐定义 FT28/FT29 查询与渲染身份。"""
    sense_runtime = W03PublicSenseRuntime(artifact, artifact_sha256)
    primitive_runtime = project_w03_public_sense_to_w04_primitives(
        sense_runtime)
    proposition_runtime = (
        project_w04_primitives_to_w05_source_bound_propositions(
            primitive_runtime))
    values = []
    for entry in artifact.entries:
        if entry.relation_kind != "DEFINITION":
            continue
        if entry.definition_text is None:
            raise W03PublicDefinitionCompilerV2Error(
                "FT30 base DEFINITION 缺文本")
        source_answer = answer_w05_raw_definition_question(
            proposition_runtime,
            W05RawDefinitionRequest(
                "什么是" + entry.surface,
                entry.definition_text,
                entry.language,
            ),
        )
        display = render_w05_definition_answer(source_answer)
        outcome = display.failure_code or display.status
        values.append({
            "ast_sha256": display.ast_sha256,
            "definition_sha256": _sha256_text(entry.definition_text),
            "display_outcome": outcome,
            "display_projection_sha256": (
                display.display_projection_sha256),
            "display_status": display.status,
            "entry_key": list(entry.entry_key),
            "failure_code": display.failure_code,
            "inline_projection_sha256": (
                display.inline_projection_sha256),
            "language": entry.language,
            "observation_key": list(entry.observation_key),
            "source_answer_sha256": source_answer.sha256(),
            "source_answer_status": source_answer.status,
            "source_answer_trace_commitment_sha256": (
                source_answer.trace_commitment_sha256),
            "source_ref_key": list(entry.source_ref.stable_key),
            "surface_sha256": _sha256_text(entry.surface),
        })
    counts = {
        outcome: sum(item["display_outcome"] == outcome for item in values)
        for outcome in {item["display_outcome"] for item in values}
    }
    if counts != {
            "DISPLAY": 7,
            "NO_SOURCE_ANSWER": 2,
            "UNKNOWN_TEMPLATE": 3,
            }:
        raise W03PublicDefinitionCompilerV2Error(
            "FT30 base 7/3/2 definition rendering 基线漂移")
    return tuple(values)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class FT30PublicDefinitionBuildV2:
    """v2 artifact 与 census 构建结果。"""

    artifact: W03PublicSenseArtifact
    census_value: dict[str, object]

    def __post_init__(self) -> None:
        if (not isinstance(self.artifact, W03PublicSenseArtifact)
                or not isinstance(self.census_value, dict)):
            raise TypeError("FT30 build result 类型非法")


def build_ft30_public_definition_artifact_v2(
        *,
        base_artifact_path: str | Path,
        base_artifact_sha256: str,
        expansion_pack_relative_path: str,
        expansion_pack_root: str | Path,
        selection_manifest_sha256: str,
        ) -> FT30PublicDefinitionBuildV2:
    """在冻结 v1 上追加中文定义/redirect，并审计全部 32 页。"""
    base = _base_artifact(
        Path(base_artifact_path).resolve(), base_artifact_sha256)
    base_definition_baseline = _base_definition_rendering_baseline(
        base, base_artifact_sha256)
    pack_root = Path(expansion_pack_root).resolve()
    bundle = read_source_pack(pack_root)
    if (bundle.manifest.redistribution_policy != "PUBLIC"
            or bundle.manifest.w_stages != ("W-03",)
            or len(bundle.sources) != FT30_MAX_SOURCE_COUNT
            or len(bundle.observations) != FT30_MAX_SOURCE_COUNT):
        raise W03PublicDefinitionCompilerV2Error(
            "FT30 expansion source pack 合同漂移")
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
            raise W03PublicDefinitionCompilerV2Error(
                "FT30 Observation 缺 SourceRef")
        payload = observation.typed_payload.to_value()
        raw = payload.get("raw_observation")
        if (not isinstance(raw, dict)
                or set(raw) != {
                    "contributor", "page_id", "redirect_title",
                    "revision_id", "text", "timestamp", "title"}):
            raise W03PublicDefinitionCompilerV2Error(
                "FT30 raw Observation 字段漂移")
        span = source.source_span.to_value()
        if span.get("selection_manifest_sha256") != selection_manifest_sha256:
            raise W03PublicDefinitionCompilerV2Error(
                "FT30 source selection SHA 漂移")
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
            definition_values.append(_definition_census_value(
                definition,
                observation_key=observation.stable_key.stable_key(),
                page_id=page_id,
                source_ref=source_ref,
                title=title,
                title_sha256=span["title_sha256"],
                stratum=span["title_length_stratum"],
                ordinal=ordinal,
            ))
            if definition not in eligible:
                continue
            accepted_definition_count += 1
            entry_key = stable_source_pack_key(
                "ft30_public_definition_entry",
                observation.stable_key.to_list(),
                ordinal,
                definition.text,
            ).stable_key()
            sense_key = stable_source_pack_key(
                "ft30_public_definition_sense",
                page_id,
                raw["revision_id"],
                ordinal,
                definition.text,
            ).stable_key()
            concept_key = stable_source_pack_key(
                "ft30_public_definition_concept",
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
    if (len(definition_values) > FT30_MAX_DEFINITION_COUNT
            or accepted_definition_count != len(entries)
            or len(base.entries) + len(entries) > FT30_MAX_ENTRY_COUNT):
        raise W03PublicDefinitionCompilerV2Error(
            "FT30 definition/entry 预算超限")
    expansion_identity = _source_pack_identity(
        expansion_pack_relative_path, pack_root, bundle)
    artifact = W03PublicSenseArtifact(
        tuple(sorted(
            (*base.source_packs, expansion_identity),
            key=lambda item: (item.source_key, item.relative_path))),
        base.source_revisions,
        tuple(sorted(
            (*base.entries, *entries), key=lambda item: item.entry_key)),
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
    census = {
        "artifact_kind": FT30_CENSUS_FORMAT,
        "artifact_version": FT30_CENSUS_VERSION,
        "base_artifact_sha256": base_artifact_sha256,
        "base_definition_count": len(base_definition_baseline),
        "base_definition_rendering_baseline": list(
            base_definition_baseline),
        "base_definition_rendering_counts": {
            outcome: sum(
                item["display_outcome"] == outcome
                for item in base_definition_baseline)
            for outcome in (
                "DISPLAY", "NO_SOURCE_ANSWER", "UNKNOWN_TEMPLATE")
        },
        "definition_count": len(definition_values),
        "definitions": sorted(
            definition_values,
            key=lambda item: (
                item["page_id"], item["definition_ordinal"])),
        "eligible_definition_count": accepted_definition_count,
        "expansion_pack_manifest_sha256": (
            expansion_identity.manifest_sha256),
        "format_version": 1,
        "page_count": len(page_values),
        "page_status_counts": status_counts,
        "pages": sorted(page_values, key=lambda item: item["page_id"]),
        "parser_version": MEDIAWIKI_INLINE_PARSER_VERSION,
        "render_status_counts": {
            key: render_counts[key] for key in sorted(render_counts)},
        "selection_manifest_sha256": selection_manifest_sha256,
    }
    return FT30PublicDefinitionBuildV2(artifact, census)


def ft30_public_sense_artifact_envelope_v2(
        artifact: W03PublicSenseArtifact,
        ) -> dict[str, object]:
    """构造 v2 experimental/formal 零边界 envelope。"""
    return {
        "artifact_version": FT30_PUBLIC_SENSE_ARTIFACT_VERSION,
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
            raise W03PublicDefinitionCompilerV2Error(
                f"FT30 {where} 已存在且字节不同")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except OSError as error:
        raise W03PublicDefinitionCompilerV2Error(
            f"FT30 {where} 无法发布") from error
    return path


def write_ft30_public_sense_artifact_v2(
        artifact: W03PublicSenseArtifact,
        path: str | Path,
        ) -> Path:
    """发布仅含 compact projection 的 v2 runtime artifact。"""
    payload = canonical_json_bytes(
        ft30_public_sense_artifact_envelope_v2(artifact)) + b"\n"
    if len(payload) > W03_PUBLIC_SENSE_MAX_ARTIFACT_BYTES:
        raise W03PublicDefinitionCompilerV2Error(
            "FT30 compact artifact 超预算")
    return _write_immutable(
        Path(path).resolve(), payload, where="compact artifact")


def write_ft30_public_definition_census_v2(
        census_value: dict[str, object],
        path: str | Path,
        ) -> Path:
    """发布不进入 wheel 的 canonical 全选择 census。"""
    payload = canonical_json_bytes(census_value) + b"\n"
    return _write_immutable(
        Path(path).resolve(), payload, where="definition census")


__all__ = [
    "FT30_CENSUS_FORMAT",
    "FT30_CENSUS_VERSION",
    "FT30_MAX_DEFINITION_COUNT",
    "FT30_MAX_ENTRY_COUNT",
    "FT30_MAX_SOURCE_COUNT",
    "FT30_PUBLIC_SENSE_ARTIFACT_VERSION",
    "FT30_ZH_SECTION_NAMES",
    "FT30DefinitionCandidate",
    "FT30PublicDefinitionBuildV2",
    "W03PublicDefinitionCompilerV2Error",
    "build_ft30_public_definition_artifact_v2",
    "extract_ft30_definition_candidates",
    "ft30_public_sense_artifact_envelope_v2",
    "write_ft30_public_definition_census_v2",
    "write_ft30_public_sense_artifact_v2",
]
