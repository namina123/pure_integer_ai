"""从 FT26 public SourceRef/Observation pack 编译 compact sense artifact。"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_source_pack_compiler import (
    read_source_pack,
)
from pure_integer_ai.experiments.ph2_source_pack_contract import (
    stable_source_pack_key,
)
from pure_integer_ai.experiments.ph2_w03_adapter_extractors import (
    W03ExtractedCandidate,
    W03ExtractedObservation,
    extract_w03_observations,
)
from pure_integer_ai.experiments.ph2_w03_public_sense_contract import (
    W03_PUBLIC_SENSE_ARTIFACT_VERSION,
    W03_PUBLIC_SENSE_FORMAT,
    W03_PUBLIC_SENSE_MAX_ARTIFACT_BYTES,
    W03_PUBLIC_SENSE_SCHEMA_VERSION,
    W03PublicSenseAlias,
    W03PublicSenseArtifact,
    W03PublicSenseEntry,
    W03PublicSenseSourcePackIdentity,
    W03PublicSenseSourceRef,
    W03PublicSenseSourceRevision,
)
from pure_integer_ai.experiments.ph2_wikidata_adapter import (
    parse_wikidata_entity_terms_bytes,
)
from pure_integer_ai.experiments.ph2_wikidata_snapshot import (
    read_wikidata_revision_snapshot,
)


# object-model: exception
class W03PublicSenseCompilerError(RuntimeError):
    """source pack、候选抽取或 compact artifact 身份发生漂移。"""


def _sha256_path(path: Path) -> str:
    """以固定块大小计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _source_ref(record) -> W03PublicSenseSourceRef:
    """把完整 SourceRefRecord 压缩为可公开查询的来源投影。"""
    return W03PublicSenseSourceRef(
        record.stable_key.stable_key(),
        record.source_key,
        record.snapshot_id,
        record.revision_id,
        record.source_identity,
        record.official_url,
        record.license_id,
        record.attribution,
        hashlib.sha256(canonical_json_bytes(record.to_dict())).hexdigest(),
    )


def _wikidata_terms(observation) -> tuple[dict[str, Any], dict[str, Any]]:
    """按现有结构化 adapter 恢复 Wikidata terms 与 raw 元数据。"""
    payload = observation.typed_payload.to_value()
    raw = payload.get("raw_observation")
    if (not isinstance(raw, dict)
            or set(raw) != {"entity_json_utf8", "qid", "revision"}):
        return {}, {}
    terms = parse_wikidata_entity_terms_bytes(
        raw["entity_json_utf8"].encode("utf-8"),
        expected_qid=raw["qid"],
        expected_revision=raw["revision"],
    )
    return {
        "aliases": terms.aliases.to_value(),
        "descriptions": terms.descriptions.to_value(),
        "labels": terms.labels.to_value(),
    }, raw


def _wikidata_definition(
        terms: dict[str, Any],
        language: str,
        fallback_language: str,
        ) -> str | None:
    """按 exact language 后 base language 读取结构化 description。"""
    descriptions = terms.get("descriptions", {})
    value = descriptions.get(language)
    if value is None:
        value = descriptions.get(fallback_language)
    if not isinstance(value, dict):
        return None
    text = value.get("value")
    return text if isinstance(text, str) and text else None


def _canonical_wikidata_surfaces(
        extracted: W03ExtractedObservation,
        ) -> dict[tuple[int, ...], str]:
    """为同一 QID concept 选择稳定 label，alias 不成为 canonical。"""
    labels: dict[tuple[int, ...], list[tuple[int, str]]] = {}
    for candidate in extracted.candidates:
        roles = candidate.provenance.to_value().get("field_roles", [])
        if "label" not in roles:
            continue
        anchor = extracted.anchors[candidate.anchor_ordinal]
        priority = int(anchor.branch_language != extracted.observation.language)
        labels.setdefault(candidate.concept_key, []).append(
            (priority, anchor.surface))
    return {
        concept: sorted(values)[0][1]
        for concept, values in labels.items()
    }


def _entry_relation(candidate: W03ExtractedCandidate) -> tuple[str, tuple[str, ...]]:
    """从 extractor provenance 得到通用 relation 与字段角色。"""
    provenance = candidate.provenance.to_value()
    roles = tuple(sorted(set(provenance.get("field_roles", []))))
    if candidate.candidate_kind == "WIKTIONARY_SENSE":
        return "DEFINITION", ("definition",)
    if candidate.candidate_kind == "WIKIDATA_TERM":
        return ("LABEL" if "label" in roles else "ALIAS"), roles
    raise W03PublicSenseCompilerError(
        f"FT26 未注册 candidate kind: {candidate.candidate_kind}")


def _entry_definition(
        extracted: W03ExtractedObservation,
        candidate: W03ExtractedCandidate,
        terms: dict[str, Any],
        ) -> str | None:
    """从 Wiktionary definition 或 Wikidata description 恢复文本。"""
    anchor = extracted.anchors[candidate.anchor_ordinal]
    provenance = anchor.provenance.to_value()
    definition = provenance.get("definition_text")
    if isinstance(definition, str) and definition:
        return definition
    return _wikidata_definition(
        terms,
        anchor.branch_language,
        extracted.observation.language,
    )


def _entry_key(
        extracted: W03ExtractedObservation,
        candidate: W03ExtractedCandidate,
        relation_kind: str,
        ) -> tuple[int, ...]:
    """从 Observation、Sense 与 relation 生成版本化稳定 entry key。"""
    anchor = extracted.anchors[candidate.anchor_ordinal]
    return stable_source_pack_key(
        "ft26_public_sense_entry",
        extracted.observation.stable_key.to_list(),
        list(candidate.sense_key),
        anchor.branch_language,
        anchor.surface,
        relation_kind,
    ).stable_key()


def _entries_for_observation(
        extracted: W03ExtractedObservation,
        source_record,
        ) -> tuple[W03PublicSenseEntry, ...]:
    """把一个非 redirect Observation 的候选投影为 compact entries。"""
    terms, _ = _wikidata_terms(extracted.observation)
    canonical_by_concept = _canonical_wikidata_surfaces(extracted)
    candidates = extracted.candidates
    if candidates and candidates[0].candidate_kind == "WIKTIONARY_SENSE":
        first_provenance = candidates[0].provenance.to_value()
        first_path = first_provenance.get("section_path")
        if not isinstance(first_path, list) or not first_path:
            raise W03PublicSenseCompilerError(
                "FT26 Wiktionary 首候选缺 section_path")
        primary_section = first_path[0]
        candidates = tuple(
            item for item in candidates
            if (
                isinstance(
                    item.provenance.to_value().get("section_path"), list)
                and item.provenance.to_value()["section_path"]
                and item.provenance.to_value()["section_path"][0]
                == primary_section
            )
        )
        if not candidates:
            raise W03PublicSenseCompilerError(
                "FT26 Wiktionary 首语言章节没有 definition")
    values = []
    for candidate in candidates:
        anchor = extracted.anchors[candidate.anchor_ordinal]
        relation, roles = _entry_relation(candidate)
        canonical = canonical_by_concept.get(
            candidate.concept_key, anchor.surface)
        values.append(W03PublicSenseEntry(
            _entry_key(extracted, candidate, relation),
            anchor.surface,
            canonical,
            anchor.branch_language,
            relation,
            _entry_definition(extracted, candidate, terms),
            candidate.sense_key,
            candidate.concept_key,
            extracted.observation.stable_key.stable_key(),
            _source_ref(source_record),
            roles,
            1,
        ))
    return tuple(values)


def _redirect_alias(
        extracted: W03ExtractedObservation,
        source_record,
        ) -> W03PublicSenseAlias | None:
    """把无候选 Wiktionary redirect 转成显式 surface REFERS 边。"""
    if extracted.candidates or len(extracted.anchors) != 1:
        return None
    anchor = extracted.anchors[0]
    target = anchor.provenance.to_value().get("redirect_title")
    if not isinstance(target, str) or not target:
        return None
    return W03PublicSenseAlias(
        anchor.surface,
        target,
        anchor.branch_language,
        _source_ref(source_record),
        extracted.observation.stable_key.stable_key(),
    )


def _apply_supersedes(
        entries: tuple[W03PublicSenseEntry, ...],
        observations: tuple[object, ...],
        ) -> tuple[W03PublicSenseEntry, ...]:
    """按 Observation supersedes_key 关闭旧候选并绑定 active successor。"""
    observation_keys = {item.stable_key for item in observations}
    by_observation: dict[tuple[int, ...], list[W03PublicSenseEntry]] = {}
    for entry in entries:
        by_observation.setdefault(entry.observation_key, []).append(entry)
    supersedes: dict[tuple[int, ...], tuple[int, ...]] = {}
    for observation in observations:
        target = observation.supersedes_key
        if target is None:
            continue
        if target not in observation_keys:
            raise W03PublicSenseCompilerError(
                "FT26 supersedes_key 不在 source pack 输入")
        supersedes[observation.stable_key.stable_key()] = target.stable_key()
    visiting: set[tuple[int, ...]] = set()
    resolved: set[tuple[int, ...]] = set()

    def visit(key: tuple[int, ...]) -> None:
        if key in resolved or key not in supersedes:
            return
        if key in visiting:
            raise W03PublicSenseCompilerError("FT26 supersede chain 成环")
        visiting.add(key)
        visit(supersedes[key])
        visiting.remove(key)
        resolved.add(key)

    for key in supersedes:
        visit(key)
    superseded_observations = set(supersedes.values())
    updated = []
    for entry in entries:
        targets: tuple[tuple[int, ...], ...] = ()
        active = entry.observation_key not in superseded_observations
        target_observation = supersedes.get(entry.observation_key)
        if active and target_observation is not None:
            inherited = []
            while target_observation is not None:
                old = by_observation.get(target_observation, [])
                same_surface = tuple(
                    item.entry_key for item in old
                    if item.surface == entry.surface)
                inherited.extend(
                    same_surface or tuple(item.entry_key for item in old))
                target_observation = supersedes.get(target_observation)
            targets = tuple(sorted(set(inherited)))
        updated.append(replace(
            entry,
            active=int(active),
            supersedes_entry_keys=targets,
        ))
    return tuple(sorted(updated, key=lambda item: item.entry_key))


def _source_pack_identity(
        label: str,
        root: Path,
        bundle,
        ) -> W03PublicSenseSourcePackIdentity:
    """冻结一个完整 public source-pack manifest 的身份与计数。"""
    snapshots = {item.snapshot_id for item in bundle.sources}
    if len(snapshots) != 1:
        raise W03PublicSenseCompilerError(
            "FT26 source pack snapshot_id 非唯一")
    manifest_path = root / "manifest.json"
    return W03PublicSenseSourcePackIdentity(
        label,
        _sha256_path(manifest_path),
        bundle.manifest.stable_key.stable_key(),
        bundle.manifest.source_key,
        bundle.manifest.license_partition,
        next(iter(snapshots)),
        len(bundle.sources),
        len(bundle.observations),
    )


def _source_revisions(repository: Path) -> tuple[W03PublicSenseSourceRevision, ...]:
    """回读 Wikidata allowlist v2 对 v1 的真实 supersede 链。"""
    snapshot_path = (
        repository / "data/ph2/manifests/wikidata_revision_v1.pinned_snapshot.json")
    allowlist_path = repository / "data/ph2/wikidata_revision_v1_allowlist_v2.json"
    if not snapshot_path.is_file() or not allowlist_path.is_file():
        raise W03PublicSenseCompilerError(
            "FT26 Wikidata revision identity 文件缺失")
    snapshot = read_wikidata_revision_snapshot(snapshot_path)
    active_sha256 = _sha256_path(allowlist_path)
    if active_sha256 != snapshot.allowlist_sha256:
        raise W03PublicSenseCompilerError(
            "FT26 Wikidata active allowlist SHA 漂移")
    try:
        raw = json.loads(allowlist_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise W03PublicSenseCompilerError(
            "FT26 Wikidata allowlist JSON 非法") from error
    supersedes = raw.get("supersedes_sha256") if isinstance(raw, dict) else None
    if not isinstance(supersedes, str):
        raise W03PublicSenseCompilerError(
            "FT26 Wikidata allowlist 缺 supersedes_sha256")
    return (W03PublicSenseSourceRevision(
        "WIKIDATA_ALLOWLIST",
        active_sha256,
        supersedes,
    ),)


def build_w03_public_sense_artifact(
        repository_root: str | Path,
        pack_inputs: tuple[tuple[str, str | Path], ...],
        ) -> W03PublicSenseArtifact:
    """从完整 pack manifest 构造带 revision 状态的 compact artifact。"""
    repository = Path(repository_root).resolve()
    if (not isinstance(pack_inputs, tuple) or not pack_inputs
            or any(not isinstance(item, tuple) or len(item) != 2
                   for item in pack_inputs)):
        raise W03PublicSenseCompilerError("FT26 pack_inputs 非法")
    bundles = []
    identities = []
    all_sources = []
    all_observations = []
    for label, raw_root in pack_inputs:
        if (not isinstance(label, str) or not label
                or label.startswith(("/", "../")) or "\\" in label):
            raise W03PublicSenseCompilerError("FT26 pack label 非安全 POSIX 路径")
        root = Path(raw_root).resolve()
        bundle = read_source_pack(root)
        if (bundle.manifest.redistribution_policy != "PUBLIC"
                or bundle.manifest.w_stages != ("W-03",)
                or any(item.split != "train" for item in bundle.observations)):
            raise W03PublicSenseCompilerError(
                "FT26 只接受 public W-03 train source pack")
        bundles.append(bundle)
        identities.append(_source_pack_identity(label, root, bundle))
        all_sources.extend(bundle.sources)
        all_observations.extend(bundle.observations)
    if len({item.stable_key for item in all_sources}) != len(all_sources):
        raise W03PublicSenseCompilerError("FT26 SourceRef stable_key 重复")
    if len({item.stable_key for item in all_observations}) != len(all_observations):
        raise W03PublicSenseCompilerError("FT26 Observation stable_key 重复")
    source_by_key = {item.stable_key: item for item in all_sources}
    observations = tuple(sorted(
        all_observations, key=lambda item: item.stable_key.stable_key()))
    extracted = extract_w03_observations(observations)
    entries = []
    aliases = []
    for item in extracted:
        source = source_by_key.get(item.observation.source_ref_key)
        if source is None:
            raise W03PublicSenseCompilerError(
                "FT26 Observation 缺 SourceRef")
        entries.extend(_entries_for_observation(item, source))
        alias = _redirect_alias(item, source)
        if alias is not None:
            aliases.append(alias)
    if not entries or not aliases:
        raise W03PublicSenseCompilerError(
            "FT26 artifact 未同时形成候选与 redirect alias")
    active_entries = _apply_supersedes(tuple(entries), observations)
    artifact = W03PublicSenseArtifact(
        tuple(sorted(
            identities,
            key=lambda item: (item.source_key, item.relative_path))),
        (
            _source_revisions(repository)
            if any(item.source_key == "WIKIDATA_REVISION_V1"
                   for item in identities)
            else ()
        ),
        active_entries,
        tuple(sorted(
            aliases,
            key=lambda item: (
                item.language, item.alias_surface, item.target_surface,
                item.observation_key))),
    )
    return artifact


def w03_public_sense_artifact_envelope(
        artifact: W03PublicSenseArtifact,
        ) -> dict[str, object]:
    """构造固定 experimental/formal 状态的规范 artifact envelope。"""
    if not isinstance(artifact, W03PublicSenseArtifact):
        raise TypeError("FT26 artifact 类型非法")
    return {
        "artifact_version": W03_PUBLIC_SENSE_ARTIFACT_VERSION,
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


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W03PublicSenseArtifactBuild:
    """一次幂等 artifact 发布结果与完整字节身份。"""

    path: Path
    artifact: W03PublicSenseArtifact
    artifact_sha256: str
    size_bytes: int


def write_w03_public_sense_artifact(
        artifact: W03PublicSenseArtifact,
        path: str | Path,
        ) -> W03PublicSenseArtifactBuild:
    """独占或幂等写入 canonical artifact，禁止覆盖不同字节。"""
    target = Path(path).resolve()
    payload = canonical_json_bytes(
        w03_public_sense_artifact_envelope(artifact)) + b"\n"
    if len(payload) > W03_PUBLIC_SENSE_MAX_ARTIFACT_BYTES:
        raise W03PublicSenseCompilerError("FT26 compact artifact 超预算")
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise W03PublicSenseCompilerError(
                "FT26 artifact 目标已存在且字节不同")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with target.open("xb") as handle:
                handle.write(payload)
        except FileExistsError as error:
            raise W03PublicSenseCompilerError(
                "FT26 artifact 并发发布冲突") from error
    return W03PublicSenseArtifactBuild(
        target,
        artifact,
        hashlib.sha256(payload).hexdigest(),
        len(payload),
    )


__all__ = [
    "W03_PUBLIC_SENSE_MAX_ARTIFACT_BYTES",
    "W03PublicSenseArtifactBuild",
    "W03PublicSenseCompilerError",
    "build_w03_public_sense_artifact",
    "w03_public_sense_artifact_envelope",
    "write_w03_public_sense_artifact",
]
