"""派生 recovery-v7 neutral semantic external-source feasibility。

本模块只建立来源明确的离散词项、lexfile、roleset、role inventory 与
modality/negation cue 索引。词项命中不是 sense assignment，roleset 的可选角色也
不是占位符角色指派；两个边界在记录和摘要中始终分账。
"""
from __future__ import annotations

from collections import Counter, defaultdict
import gzip
import hashlib
from pathlib import Path, PurePosixPath
import stat
import xml.etree.ElementTree as ElementTree
from zipfile import BadZipFile, ZipFile

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_training_records import (
    V5_SOURCE_FAMILIES,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_conceptnet_alias_records import (
    neutral_source_phrases,
    neutral_source_units,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_neutral_source_projection_records import (
    GODOT_SOURCE_FAMILY,
    LIBREOFFICE_SOURCE_FAMILY,
    THUNDERBIRD_SOURCE_FAMILY,
    VSCODE_SOURCE_FAMILY,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
)


NEUTRAL_SEMANTIC_SOURCE_CANDIDATE_KIND = (
    "NORMALIZATION_RECOVERY_V7_NEUTRAL_SEMANTIC_SOURCE_CANDIDATE_V1")
NEUTRAL_SEMANTIC_SOURCE_CENSUS_KIND = (
    "NORMALIZATION_RECOVERY_V7_NEUTRAL_SEMANTIC_SOURCE_CENSUS_V1")
NEUTRAL_SEMANTIC_FAMILY_COVERAGE_KIND = (
    "NORMALIZATION_RECOVERY_V7_NEUTRAL_SEMANTIC_FAMILY_COVERAGE_V1")
NEUTRAL_SEMANTIC_PROPOSAL_COVERAGE_KIND = (
    "NORMALIZATION_RECOVERY_V7_NEUTRAL_SEMANTIC_PROPOSAL_COVERAGE_V1")
NEUTRAL_SEMANTIC_FACT_FAMILY_KIND = (
    "NORMALIZATION_RECOVERY_V7_NEUTRAL_SEMANTIC_FACT_FAMILY_V1")
NEUTRAL_SEMANTIC_TARGET_SCOPE = (
    "CROSS_PRODUCT_NEUTRAL_SEMANTIC_SOURCE_FEASIBILITY_V1")

OEWN_SOURCE_ID = "OPEN_ENGLISH_WORDNET_2025"
PROPBANK_SOURCE_ID = "PROPBANK_FRAMES_3_4_C66E0CCF"

SUPPORT_OEWN_ANY = "OEWN_ANY_LEMMA"
SUPPORT_OEWN_ACTION_STATE = "OEWN_DIRECT_ACTION_STATE_LEXFILE"
SUPPORT_PROPBANK_PREDICATE = "PROPBANK_PREDICATE_OR_ALIAS"
SUPPORT_PROPBANK_ROLE_INVENTORY = "PROPBANK_ROLE_INVENTORY"
SUPPORT_PROPBANK_MODAL_CUE = "PROPBANK_ARGM_MOD_CUE"
SUPPORT_PROPBANK_NEGATION_CUE = "PROPBANK_ARGM_NEG_CUE"
SUPPORT_TWO_SOURCE_LEXICAL = "OEWN_PROPBANK_EXACT_LEXICAL"
SUPPORT_TWO_SOURCE_ACTION_STATE = (
    "OEWN_ACTION_STATE_PROPBANK_PREDICATE_EXACT")

_SOURCE_ORDER = (
    GODOT_SOURCE_FAMILY,
    LIBREOFFICE_SOURCE_FAMILY,
    VSCODE_SOURCE_FAMILY,
    THUNDERBIRD_SOURCE_FAMILY,
)
_SUPPORT_ORDER = (
    SUPPORT_OEWN_ANY,
    SUPPORT_OEWN_ACTION_STATE,
    SUPPORT_PROPBANK_PREDICATE,
    SUPPORT_PROPBANK_ROLE_INVENTORY,
    SUPPORT_PROPBANK_MODAL_CUE,
    SUPPORT_PROPBANK_NEGATION_CUE,
    SUPPORT_TWO_SOURCE_LEXICAL,
    SUPPORT_TWO_SOURCE_ACTION_STATE,
)
_PROPBANK_MEMBER_MAX = 8_000
_PROPBANK_MEMBER_BYTES_MAX = 4 * 1024 * 1024
_PROPBANK_TOTAL_BYTES_MAX = 64 * 1024 * 1024


def _sha256(payload: bytes) -> str:
    """返回来源、集合或记录 identity 的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _record_id(identity: dict[str, object]) -> str:
    """从完整 identity 形成稳定记录 id。"""
    return _sha256(canonical_json_bytes(identity))


def _git_blob_sha1(payload: bytes) -> str:
    """重算 Git blob identity，不依赖工作树或 Git 命令。"""
    prefix = b"blob " + str(len(payload)).encode("ascii") + b"\x00"
    return hashlib.sha1(prefix + payload).hexdigest()


def _local_name(value: str) -> str:
    """移除可选 XML namespace，返回稳定 local tag。"""
    return value.rsplit("}", 1)[-1]


def normalize_optional_semantic_source_text(value: object) -> str:
    """规范非空来源文本；允许 schema 中明确可空的 alias/arg 字段。"""
    if not isinstance(value, str) or not value.strip():
        return ""
    return " ".join(neutral_source_units(value))


def _optional_phrases(value: object) -> frozenset[str]:
    """形成非空来源文本的一至四单元 phrase 集。"""
    if not isinstance(value, str) or not value.strip():
        return frozenset()
    return frozenset(neutral_source_phrases(value))


def parse_open_english_wordnet(
        source_path: str | Path,
        *,
        expected_lexicon_id: str = "oewn",
        expected_version: str = "2025",
        expected_license_url: str = (
            "https://creativecommons.org/licenses/by/4.0"),
        expected_repository_url: str = (
            "https://github.com/globalwordnet/english-wordnet"),
        ) -> tuple[dict[str, frozenset[str]], dict[str, object]]:
    """流式解析固定 OEWN LMF，并形成 lemma/action/state 离散索引。"""
    path = Path(source_path)
    entry_ids = set()
    synset_ids = set()
    lemma_synsets: dict[str, set[str]] = defaultdict(set)
    synset_lexfile = {}
    lexfile_counts = Counter()
    counters = Counter()
    lexicon_identity = None
    try:
        with gzip.open(path, "rb") as handle:
            for event, element in ElementTree.iterparse(
                    handle, events=("start", "end")):
                tag = _local_name(element.tag)
                if event == "start" and tag == "Lexicon":
                    if lexicon_identity is not None:
                        raise BroadQaExternalDataError(
                            "v7 neutral semantic OEWN Lexicon 重复")
                    lexicon_identity = {
                        "id": element.get("id", ""),
                        "language": element.get("language", ""),
                        "license": element.get("license", ""),
                        "url": element.get("url", ""),
                        "version": element.get("version", ""),
                    }
                    continue
                if event != "end":
                    continue
                if tag == "LexicalEntry":
                    counters["lexical_entry_count"] += 1
                    entry_id = element.get("id", "")
                    if not entry_id or entry_id in entry_ids:
                        raise BroadQaExternalDataError(
                            "v7 neutral semantic OEWN LexicalEntry id 非法")
                    entry_ids.add(entry_id)
                    lemma = next(
                        (child for child in element
                         if _local_name(child.tag) == "Lemma"), None)
                    phrase = normalize_optional_semantic_source_text(
                        lemma.get("writtenForm", "")
                        if lemma is not None else "")
                    if not phrase:
                        counters["empty_normalized_lemma_count"] += 1
                    for child in element:
                        if _local_name(child.tag) != "Sense":
                            continue
                        counters["sense_count"] += 1
                        synset_id = child.get("synset", "")
                        if not synset_id:
                            raise BroadQaExternalDataError(
                                "v7 neutral semantic OEWN Sense synset 为空")
                        if phrase:
                            lemma_synsets[phrase].add(synset_id)
                    element.clear()
                elif tag == "Synset":
                    counters["synset_count"] += 1
                    synset_id = element.get("id", "")
                    if not synset_id or synset_id in synset_ids:
                        raise BroadQaExternalDataError(
                            "v7 neutral semantic OEWN Synset id 非法")
                    synset_ids.add(synset_id)
                    lexfile = element.get("lexfile", "")
                    if not lexfile:
                        raise BroadQaExternalDataError(
                            "v7 neutral semantic OEWN lexfile 为空")
                    synset_lexfile[synset_id] = lexfile
                    lexfile_counts[lexfile] += 1
                    element.clear()
    except BroadQaExternalDataError:
        raise
    except (OSError, EOFError, ElementTree.ParseError) as error:
        raise BroadQaExternalDataError(
            "v7 neutral semantic OEWN gzip/XML 非法") from error
    expected_identity = {
        "id": expected_lexicon_id,
        "language": "en",
        "license": expected_license_url,
        "url": expected_repository_url,
        "version": expected_version,
    }
    if lexicon_identity != expected_identity:
        raise BroadQaExternalDataError(
            "v7 neutral semantic OEWN source identity 漂移")
    unresolved = {
        synset_id for values in lemma_synsets.values() for synset_id in values
        if synset_id not in synset_ids
    }
    if (unresolved or not entry_ids or not synset_ids
            or counters["empty_normalized_lemma_count"]):
        raise BroadQaExternalDataError(
            "v7 neutral semantic OEWN 引用或 lemma 非法")
    action = set()
    state = set()
    for phrase, values in lemma_synsets.items():
        lexfiles = {synset_lexfile[value] for value in values}
        if "noun.act" in lexfiles:
            action.add(phrase)
        if "noun.state" in lexfiles or "verb.stative" in lexfiles:
            state.add(phrase)
    all_phrases = set(lemma_synsets)
    action_state = action | state
    indexes = {
        "all": frozenset(all_phrases),
        "action": frozenset(action),
        "state": frozenset(state),
        "action_state": frozenset(action_state),
    }
    census = {
        "action_phrase_count": len(action),
        "action_state_overlap_phrase_count": len(action & state),
        "action_state_phrase_count": len(action_state),
        "empty_normalized_lemma_count": 0,
        "lexical_entry_count": counters["lexical_entry_count"],
        "normalized_lemma_phrase_count": len(all_phrases),
        "noun_act_synset_count": lexfile_counts["noun.act"],
        "noun_state_synset_count": lexfile_counts["noun.state"],
        "parse_anomaly_count": 0,
        "sense_count": counters["sense_count"],
        "state_phrase_count": len(state),
        "synset_count": counters["synset_count"],
        "unresolved_sense_synset_count": 0,
        "verb_stative_synset_count": lexfile_counts["verb.stative"],
    }
    return indexes, census


def _safe_propbank_members(archive: ZipFile) -> tuple[object, ...]:
    """核验 PropBank selection ZIP 的路径、类型和解压预算。"""
    infos = archive.infolist()
    if not infos or len(infos) > _PROPBANK_MEMBER_MAX:
        raise BroadQaExternalDataError(
            "v7 neutral semantic PropBank ZIP member 数非法")
    names = set()
    total_bytes = 0
    for info in infos:
        name = info.filename
        pure = PurePosixPath(name)
        mode = (info.external_attr >> 16) & 0o170000
        if (not name or "\\" in name or pure.is_absolute()
                or ".." in pure.parts or ":" in name
                or name in names or mode == stat.S_IFLNK
                or info.file_size > _PROPBANK_MEMBER_BYTES_MAX):
            raise BroadQaExternalDataError(
                "v7 neutral semantic PropBank ZIP member 非法")
        names.add(name)
        total_bytes += info.file_size
        allowed = (
            name in {"LICENSE", "README.md", "frames/",
                     "frames/.gitignore", "frames/README.txt",
                     "frames/frameset.dtd"}
            or (name.startswith("frames/") and name.endswith(".xml")
                and len(pure.parts) == 2))
        if not allowed:
            raise BroadQaExternalDataError(
                "v7 neutral semantic PropBank ZIP selection 漂移")
    if (total_bytes > _PROPBANK_TOTAL_BYTES_MAX
            or "LICENSE" not in names or "README.md" not in names):
        raise BroadQaExternalDataError(
            "v7 neutral semantic PropBank ZIP 预算或必需文件非法")
    return tuple(infos)


def parse_propbank_frames(
        source_path: str | Path,
        *,
        expected_license_sha256: str,
        expected_license_git_blob_sha1: str | None = None,
        ) -> tuple[dict[str, frozenset[str]], dict[str, object]]:
    """解析 PropBank frames selection，并显式记账坏 XML 与重复 roleset id。"""
    path = Path(source_path)
    predicates = set()
    role_inventory = set()
    modal_cues = set()
    negation_cues = set()
    roleset_paths: dict[str, list[str]] = defaultdict(list)
    resource_rolesets: dict[str, set[tuple[str, str]]] = defaultdict(set)
    resource_counts = Counter()
    counters = Counter()
    malformed = []
    try:
        with ZipFile(path) as archive:
            infos = _safe_propbank_members(archive)
            license_payload = archive.read("LICENSE")
            if (_sha256(license_payload) != expected_license_sha256
                    or (expected_license_git_blob_sha1 is not None
                        and _git_blob_sha1(license_payload)
                        != expected_license_git_blob_sha1)):
                raise BroadQaExternalDataError(
                    "v7 neutral semantic PropBank license identity 漂移")
            xml_names = sorted(
                info.filename for info in infos
                if info.filename.startswith("frames/")
                and info.filename.endswith(".xml"))
            if not xml_names:
                raise BroadQaExternalDataError(
                    "v7 neutral semantic PropBank XML selection 为空")
            for name in xml_names:
                payload = archive.read(name)
                try:
                    root = ElementTree.fromstring(payload)
                except ElementTree.ParseError:
                    malformed.append({
                        "payload_sha256": _sha256(payload),
                        "relative_path_sha256": _sha256(
                            name.encode("utf-8")),
                        "reason_code": "XML_PARSE_ERROR",
                    })
                    continue
                if _local_name(root.tag) != "frameset":
                    raise BroadQaExternalDataError(
                        "v7 neutral semantic PropBank XML root 非法")
                counters["valid_xml_file_count"] += 1
                for predicate in root.iter("predicate"):
                    predicate_phrase = normalize_optional_semantic_source_text(
                        predicate.get("lemma", ""))
                    if not predicate_phrase:
                        counters["empty_predicate_lemma_count"] += 1
                    for roleset in predicate.findall("roleset"):
                        counters["roleset_count"] += 1
                        roleset_id = roleset.get("id", "")
                        if not roleset_id:
                            raise BroadQaExternalDataError(
                                "v7 neutral semantic PropBank roleset id 为空")
                        roleset_paths[roleset_id].append(name)
                        aliases = set()
                        if predicate_phrase:
                            aliases.add(predicate_phrase)
                        for alias in roleset.findall("./aliases/alias"):
                            counters["alias_count"] += 1
                            phrase = normalize_optional_semantic_source_text(
                                alias.text or "")
                            if not phrase:
                                counters["empty_alias_count"] += 1
                                continue
                            aliases.add(phrase)
                        predicates.update(aliases)
                        roles = roleset.findall("./roles/role")
                        counters["role_count"] += len(roles)
                        if roles:
                            role_inventory.update(aliases)
                        for role in roles:
                            if (not (role.get("n") or "").strip()
                                    or not (role.get("descr") or "").strip()):
                                counters["role_missing_core_field_count"] += 1
                        resources = set()
                        for link in roleset.iter():
                            if _local_name(link.tag) not in {
                                    "rolelink", "lexlink"}:
                                continue
                            resource = (link.get("resource") or "").strip()
                            if not resource:
                                counters["empty_cross_link_resource_count"] += 1
                                continue
                            normalized_resource = resource.upper()
                            resource_counts[normalized_resource] += 1
                            resources.add(normalized_resource)
                        for resource in resources:
                            resource_rolesets[resource].add(
                                (name, roleset_id))
                        for argument in roleset.findall(
                                "./example/propbank/arg"):
                            counters["example_argument_count"] += 1
                            argument_type = (
                                argument.get("type") or "").upper()
                            phrase = normalize_optional_semantic_source_text(
                                "".join(argument.itertext()))
                            if not phrase:
                                counters["empty_argument_text_count"] += 1
                                continue
                            if argument_type == "ARGM-MOD":
                                counters["modal_cue_occurrence_count"] += 1
                                modal_cues.add(phrase)
                            elif argument_type == "ARGM-NEG":
                                counters["negation_cue_occurrence_count"] += 1
                                negation_cues.add(phrase)
            member_count = len(infos)
            uncompressed_bytes = sum(info.file_size for info in infos)
    except BroadQaExternalDataError:
        raise
    except (OSError, BadZipFile, KeyError, RuntimeError) as error:
        raise BroadQaExternalDataError(
            "v7 neutral semantic PropBank ZIP 非法") from error
    if (counters["empty_predicate_lemma_count"]
            or counters["role_missing_core_field_count"]
            or counters["empty_cross_link_resource_count"]):
        raise BroadQaExternalDataError(
            "v7 neutral semantic PropBank core field 非法")
    duplicate_rows = []
    for roleset_id, paths in roleset_paths.items():
        if len(paths) <= 1:
            continue
        duplicate_rows.append({
            "roleset_id_sha256": _sha256(roleset_id.encode("utf-8")),
            "source_path_set_sha256": _sha256(canonical_json_bytes(
                sorted(_sha256(path.encode("utf-8")) for path in paths))),
            "source_record_count": len(paths),
        })
    indexes = {
        "modal_cue": frozenset(modal_cues),
        "negation_cue": frozenset(negation_cues),
        "predicate": frozenset(predicates),
        "role_inventory": frozenset(role_inventory),
    }
    census = {
        "alias_count": counters["alias_count"],
        "archive_member_count": member_count,
        "archive_uncompressed_bytes": uncompressed_bytes,
        "cross_link_counts": {
            key: resource_counts[key] for key in sorted(resource_counts)},
        "cross_link_roleset_counts": {
            key: len(resource_rolesets[key])
            for key in sorted(resource_rolesets)},
        "duplicate_roleset_id_count": len(duplicate_rows),
        "duplicate_roleset_id_set_sha256": _sha256(
            canonical_json_bytes(sorted(
                duplicate_rows,
                key=lambda item: str(item["roleset_id_sha256"])))),
        "empty_alias_count": counters["empty_alias_count"],
        "empty_argument_text_count": counters["empty_argument_text_count"],
        "example_argument_count": counters["example_argument_count"],
        "malformed_xml_file_count": len(malformed),
        "malformed_xml_set_sha256": _sha256(canonical_json_bytes(sorted(
            malformed,
            key=lambda item: str(item["relative_path_sha256"])))),
        "modal_cue_occurrence_count": counters[
            "modal_cue_occurrence_count"],
        "modal_cue_phrase_count": len(modal_cues),
        "negation_cue_occurrence_count": counters[
            "negation_cue_occurrence_count"],
        "negation_cue_phrase_count": len(negation_cues),
        "parse_anomaly_count": len(malformed) + len(duplicate_rows),
        "predicate_alias_phrase_count": len(predicates),
        "role_count": counters["role_count"],
        "role_inventory_phrase_count": len(role_inventory),
        "roleset_count": counters["roleset_count"],
        "valid_xml_file_count": counters["valid_xml_file_count"],
        "xml_file_count": counters["valid_xml_file_count"] + len(malformed),
    }
    return indexes, census


def _candidate_records() -> tuple[dict[str, object], ...]:
    """冻结本轮已核实的 selected/blocked/deferred source roster。"""
    candidates = (
        {
            "candidate_id": OEWN_SOURCE_ID,
            "field_scope": [
                "LEMMA_PART_OF_SPEECH", "SENSE_SYNSET_REFERENCE",
                "SYNSET_LEXFILE", "SENSE_RELATION"],
            "license_id": "WORDNET_LICENSE_PLUS_CC-BY-4.0",
            "official_url": (
                "https://en-word.net/static/english-wordnet-2025.xml.gz"),
            "repository_commit": (
                "02ff9f3f5bc0a25592e7263ffdbc9bcb6564936b"),
            "repository_tree": (
                "96ecd968b9f047b6e50e5d79b7e004e95864f715"),
            "repository_url": (
                "https://github.com/globalwordnet/english-wordnet"),
            "selection_status": "SELECTED_LICENSE_BOUND",
            "version": "2025",
        },
        {
            "candidate_id": PROPBANK_SOURCE_ID,
            "field_scope": [
                "PREDICATE_LEMMA", "ROLESET_ID_NAME_ALIAS",
                "ROLE_INVENTORY", "EXAMPLE_ARGM_MOD_NEG",
                "FRAMENET_VERBNET_CROSS_LINK"],
            "license_id": "CC-BY-SA-4.0",
            "official_url": (
                "https://github.com/propbank/propbank-frames"),
            "repository_commit": (
                "c66e0ccf28b53f00051b187db83e937b5bee2e32"),
            "repository_tree": (
                "d1e1ef0c13c5ec6e06096b1448cb5f65d4e1b8c7"),
            "repository_url": (
                "https://github.com/propbank/propbank-frames"),
            "selection_status": "SELECTED_LICENSE_BOUND",
            "version": "3.4",
        },
        {
            "candidate_id": "VERBNET_REPOSITORY",
            "field_scope": ["THEMATIC_ROLE", "SYNTAX", "SEMANTICS"],
            "license_id": "UNBOUND",
            "official_url": "https://github.com/cu-clear/verbnet",
            "repository_commit": (
                "ae8e9cfdc2c0d3414b748763612f1a0a34194cc1"),
            "repository_tree": (
                "45591aa8e0b8365f3a927c94370f88e5a1073e3d"),
            "repository_url": "https://github.com/cu-clear/verbnet",
            "selection_status": "BLOCKED_LICENSE_NOT_BOUND",
            "version": "UNBOUND_AT_AUDITED_COMMIT",
        },
        {
            "candidate_id": "UD_ENGLISH_EWT_V2_18",
            "field_scope": ["DEPENDENCY_ANNOTATION"],
            "license_id": "CC-BY-SA-4.0_ANNOTATION_MIXED_TEXT_RIGHTS",
            "official_url": (
                "https://github.com/UniversalDependencies/UD_English-EWT"),
            "repository_commit": (
                "4a4d77f599ea53cc405f85d0cec4b2f14f81d42b"),
            "repository_tree": (
                "ca5091e85212d4715cd21288ae67b7753e98a557"),
            "repository_url": (
                "https://github.com/UniversalDependencies/UD_English-EWT"),
            "selection_status": "DEFERRED_SOURCE_LEVEL_RIGHTS_PARTITION_REQUIRED",
            "version": "2.18",
        },
        {
            "candidate_id": "UD_ENGLISH_GUM_V2_18",
            "field_scope": ["DEPENDENCY_ANNOTATION"],
            "license_id": "CC-BY-NC-SA-4.0",
            "official_url": (
                "https://github.com/UniversalDependencies/UD_English-GUM"),
            "repository_commit": (
                "1fe635509c649e376dfb449d528424ab78f4eaee"),
            "repository_tree": "",
            "repository_url": (
                "https://github.com/UniversalDependencies/UD_English-GUM"),
            "selection_status": "REJECTED_NONCOMMERCIAL_OR_MIXED_NC_CONTENT",
            "version": "2.18",
        },
        {
            "candidate_id": "FRAMENET_RAW_DATA",
            "field_scope": ["FRAME", "FRAME_ELEMENT"],
            "license_id": "NO_STABLE_BOUND_DATA_CONTRACT_FOUND",
            "official_url": "https://framenet.icsi.berkeley.edu/",
            "repository_commit": "",
            "repository_tree": "",
            "repository_url": "",
            "selection_status": "DEFERRED_LICENSE_AND_VERSION_CONTRACT_REQUIRED",
            "version": "UNBOUND",
        },
    )
    records = []
    for value in candidates:
        identity = {
            "candidate_id": value["candidate_id"],
            "target_scope": NEUTRAL_SEMANTIC_TARGET_SCOPE,
        }
        records.append({
            **identity,
            **value,
            "format_version": 1,
            "record_id": _record_id(identity),
            "record_kind": NEUTRAL_SEMANTIC_SOURCE_CANDIDATE_KIND,
        })
    return tuple(sorted(records, key=lambda item: str(item["candidate_id"])))


def _source_census_records(
        *,
        oewn_census: dict[str, object],
        propbank_census: dict[str, object],
        source_identities: dict[str, dict[str, object]],
        ) -> tuple[dict[str, object], ...]:
    """形成两份只含 aggregate/commitment 的 source census 记录。"""
    records = []
    for source_id, census in (
            (OEWN_SOURCE_ID, oewn_census),
            (PROPBANK_SOURCE_ID, propbank_census)):
        identity = {
            "source_id": source_id,
            "target_scope": NEUTRAL_SEMANTIC_TARGET_SCOPE,
        }
        records.append({
            **identity,
            "census": census,
            "format_version": 1,
            "raw_or_lexical_surface_published": 0,
            "record_id": _record_id(identity),
            "record_kind": NEUTRAL_SEMANTIC_SOURCE_CENSUS_KIND,
            "source_identity": source_identities[source_id],
        })
    return tuple(records)


def _support_sets(
        oewn: dict[str, frozenset[str]],
        propbank: dict[str, frozenset[str]],
        ) -> dict[str, frozenset[str]]:
    """形成词项、action/state、role inventory 与 cue 的分账支持集。"""
    return {
        SUPPORT_OEWN_ANY: oewn["all"],
        SUPPORT_OEWN_ACTION_STATE: oewn["action_state"],
        SUPPORT_PROPBANK_PREDICATE: propbank["predicate"],
        SUPPORT_PROPBANK_ROLE_INVENTORY: propbank["role_inventory"],
        SUPPORT_PROPBANK_MODAL_CUE: propbank["modal_cue"],
        SUPPORT_PROPBANK_NEGATION_CUE: propbank["negation_cue"],
        SUPPORT_TWO_SOURCE_LEXICAL: frozenset(
            oewn["all"] & propbank["predicate"]),
        SUPPORT_TWO_SOURCE_ACTION_STATE: frozenset(
            oewn["action_state"] & propbank["predicate"]),
    }


def _family_coverage_records(
        *,
        rows_by_family: dict[str, tuple[dict[str, object], ...]],
        support_sets: dict[str, frozenset[str]],
        ) -> tuple[
            tuple[dict[str, object], ...],
            dict[str, dict[str, frozenset[str]]],
        ]:
    """计算四 product family 的 exact lexical phrase/pair coverage。"""
    if set(rows_by_family) != set(_SOURCE_ORDER):
        raise BroadQaExternalDataError(
            "v7 neutral semantic neutral family roster 漂移")
    matched_by_family: dict[str, dict[str, set[str]]] = {
        family: {support: set() for support in _SUPPORT_ORDER}
        for family in _SOURCE_ORDER}
    records = []
    seen_pairs = set()
    for family in _SOURCE_ORDER:
        pair_counts = Counter()
        rows = rows_by_family[family]
        for row in rows:
            pair_id = row.get("pair_id") if isinstance(row, dict) else None
            surface = row.get("_neutral_surface") \
                if isinstance(row, dict) else None
            if (not isinstance(pair_id, str) or len(pair_id) != 64
                    or pair_id in seen_pairs
                    or not isinstance(surface, str) or not surface):
                raise BroadQaExternalDataError(
                    "v7 neutral semantic neutral row 非法")
            seen_pairs.add(pair_id)
            row_phrases = _optional_phrases(surface)
            for support in _SUPPORT_ORDER:
                hits = row_phrases & support_sets[support]
                if hits:
                    pair_counts[support] += 1
                    matched_by_family[family][support].update(hits)
        identity = {
            "source_family": family,
            "target_scope": NEUTRAL_SEMANTIC_TARGET_SCOPE,
        }
        records.append({
            **identity,
            "coverage": {
                support: {
                    "matched_pair_count": pair_counts[support],
                    "matched_phrase_count": len(
                        matched_by_family[family][support]),
                } for support in _SUPPORT_ORDER
            },
            "format_version": 1,
            "lexical_match_assigns_semantic_sense": 0,
            "projected_pair_count": len(rows),
            "record_id": _record_id(identity),
            "record_kind": NEUTRAL_SEMANTIC_FAMILY_COVERAGE_KIND,
            "role_inventory_assigns_placeholder_role": 0,
        })
    frozen = {
        family: {
            support: frozenset(values)
            for support, values in supports.items()
        } for family, supports in matched_by_family.items()
    }
    return tuple(records), frozen


def _cross_family_phrase_counts(
        matched_by_family: dict[str, dict[str, frozenset[str]]],
        ) -> dict[str, int]:
    """统计至少两家 product family 命中的同一来源 phrase 数。"""
    values = {}
    for support in _SUPPORT_ORDER:
        counts = Counter()
        for family in _SOURCE_ORDER:
            counts.update(matched_by_family[family][support])
        values[support] = sum(count >= 2 for count in counts.values())
    return values


def _proposal_coverage_records(
        *,
        proposals: tuple[dict[str, object], ...],
        rows_by_family: dict[str, tuple[dict[str, object], ...]],
        support_sets: dict[str, frozenset[str]],
        ) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    """对冻结 proposal 只读 neutral source，统计各 fact family 可用性。"""
    row_by_pair = {}
    for family in _SOURCE_ORDER:
        for row in rows_by_family[family]:
            pair_id = str(row["pair_id"])
            if pair_id in row_by_pair:
                raise BroadQaExternalDataError(
                    "v7 neutral semantic proposal pair identity 冲突")
            row_by_pair[pair_id] = row
    by_family = {
        family: {"coverage": Counter(), "outcomes": Counter(), "count": 0}
        for family in _SOURCE_ORDER}
    aggregate_coverage = Counter()
    aggregate_outcomes = Counter()
    for proposal in proposals:
        held_family = proposal.get("held_out_source_family") \
            if isinstance(proposal, dict) else None
        pair_id = proposal.get("source_pair_id") \
            if isinstance(proposal, dict) else None
        outcome = proposal.get("pre_authorization_outcome") \
            if isinstance(proposal, dict) else None
        if (held_family not in _SOURCE_ORDER
                or not isinstance(pair_id, str) or pair_id not in row_by_pair
                or outcome not in {"EXACT", "UNKNOWN", "WRONG"}
                or row_by_pair[pair_id].get("source_family") != held_family):
            raise BroadQaExternalDataError(
                "v7 neutral semantic proposal identity 非法")
        row_phrases = _optional_phrases(
            row_by_pair[pair_id]["_neutral_surface"])
        by_family[str(held_family)]["count"] += 1
        by_family[str(held_family)]["outcomes"][str(outcome)] += 1
        aggregate_outcomes[str(outcome)] += 1
        for support in _SUPPORT_ORDER:
            available = int(bool(row_phrases & support_sets[support]))
            by_family[str(held_family)]["coverage"][support] += available
            aggregate_coverage[support] += available
    records = []
    for family in _SOURCE_ORDER:
        identity = {
            "held_out_source_family": family,
            "target_scope": NEUTRAL_SEMANTIC_TARGET_SCOPE,
        }
        records.append({
            **identity,
            "coverage_available_counts": {
                support: by_family[family]["coverage"][support]
                for support in _SUPPORT_ORDER},
            "format_version": 1,
            "pre_authorization_outcome_counts": {
                key: by_family[family]["outcomes"][key]
                for key in ("EXACT", "UNKNOWN", "WRONG")},
            "proposal_count": by_family[family]["count"],
            "record_id": _record_id(identity),
            "record_kind": NEUTRAL_SEMANTIC_PROPOSAL_COVERAGE_KIND,
            "source_coverage_authorizes_transformation": 0,
        })
    summary = {
        "coverage_available_counts": {
            support: aggregate_coverage[support]
            for support in _SUPPORT_ORDER},
        "pre_authorization_outcome_counts": {
            key: aggregate_outcomes[key]
            for key in ("EXACT", "UNKNOWN", "WRONG")},
        "proposal_count": len(proposals),
    }
    return tuple(records), summary


def _fact_family_records(
        *,
        cross_family_counts: dict[str, int],
        ) -> tuple[dict[str, object], ...]:
    """冻结可用 fact family，并保留 sense/role assignment 缺口。"""
    values = (
        (
            "PREDICATE_ACTION_STATE_EVIDENCE",
            "AVAILABLE_NONZERO_TWO_SOURCE_LEXICAL_SUPPORT",
            2, cross_family_counts[SUPPORT_TWO_SOURCE_ACTION_STATE],
            "LEXFILE_AND_PREDICATE_MATCH_DO_NOT_ASSIGN_SENSE",
        ),
        (
            "ARGUMENT_SEMANTIC_ROLE_INVENTORY",
            "AVAILABLE_INVENTORY_ONLY_NOT_PLACEHOLDER_ASSIGNMENT",
            1, cross_family_counts[SUPPORT_PROPBANK_ROLE_INVENTORY],
            "ROLESET_INVENTORY_DOES_NOT_ASSIGN_SOURCE_PLACEHOLDER_ROLE",
        ),
        (
            "MODALITY_CUE_EVIDENCE",
            "AVAILABLE_EXAMPLE_CUE_NOT_SCOPE_ASSIGNMENT",
            1, cross_family_counts[SUPPORT_PROPBANK_MODAL_CUE],
            "ARGM_MOD_SURFACE_CUE_DOES_NOT_ASSIGN_MODAL_SCOPE",
        ),
        (
            "NEGATION_CUE_EVIDENCE",
            "AVAILABLE_EXAMPLE_CUE_NOT_SCOPE_ASSIGNMENT",
            1, cross_family_counts[SUPPORT_PROPBANK_NEGATION_CUE],
            "ARGM_NEG_SURFACE_CUE_DOES_NOT_ASSIGN_NEGATION_SCOPE",
        ),
        (
            "LEXICAL_SENSE_ASSIGNMENT",
            "NE_NOT_PRESENT",
            0, 0, "LEXICAL_MATCH_IS_NOT_WORD_SENSE_DISAMBIGUATION",
        ),
        (
            "PLACEHOLDER_ROLE_ASSIGNMENT",
            "NE_NOT_PRESENT",
            0, 0, "ROLE_INVENTORY_IS_NOT_ARGUMENT_ROLE_ASSIGNMENT",
        ),
        (
            "TARGET_SPACING_PUNCTUATION_POLICY",
            "NE_NOT_PRESENT",
            0, 0, "SOURCE_SEMANTICS_DO_NOT_DEFINE_UNSEEN_PRODUCT_STYLE",
        ),
    )
    records = []
    for family, outcome, source_count, evidence_count, reason in values:
        identity = {
            "fact_family": family,
            "target_scope": NEUTRAL_SEMANTIC_TARGET_SCOPE,
        }
        records.append({
            **identity,
            "cross_family_evidence_phrase_count": evidence_count,
            "format_version": 1,
            "outcome": outcome,
            "reason_code": reason,
            "record_id": _record_id(identity),
            "record_kind": NEUTRAL_SEMANTIC_FACT_FAMILY_KIND,
            "selected_source_count": source_count,
        })
    return tuple(records)


def derive_neutral_semantic_source_feasibility(
        *,
        oewn_source_path: str | Path,
        propbank_source_path: str | Path,
        propbank_license_sha256: str,
        propbank_license_git_blob_sha1: str,
        rows_by_family: dict[str, tuple[dict[str, object], ...]],
        proposals: tuple[dict[str, object], ...],
        source_identities: dict[str, dict[str, object]],
        ) -> tuple[
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            dict[str, object],
        ]:
    """重解析两源并派生 candidate/source/family/proposal/fact 记录。"""
    if set(source_identities) != {OEWN_SOURCE_ID, PROPBANK_SOURCE_ID}:
        raise BroadQaExternalDataError(
            "v7 neutral semantic selected source identity roster 漂移")
    oewn, oewn_census = parse_open_english_wordnet(oewn_source_path)
    propbank, propbank_census = parse_propbank_frames(
        propbank_source_path,
        expected_license_sha256=propbank_license_sha256,
        expected_license_git_blob_sha1=propbank_license_git_blob_sha1,
    )
    support_sets = _support_sets(oewn, propbank)
    family_records, matched_by_family = _family_coverage_records(
        rows_by_family=rows_by_family, support_sets=support_sets)
    cross_family_counts = _cross_family_phrase_counts(matched_by_family)
    proposal_records, proposal_summary = _proposal_coverage_records(
        proposals=proposals,
        rows_by_family=rows_by_family,
        support_sets=support_sets,
    )
    candidates = _candidate_records()
    source_census = _source_census_records(
        oewn_census=oewn_census,
        propbank_census=propbank_census,
        source_identities=source_identities,
    )
    fact_records = _fact_family_records(
        cross_family_counts=cross_family_counts)
    selected_count = sum(
        item["selection_status"] == "SELECTED_LICENSE_BOUND"
        for item in candidates)
    feasibility = (
        "PASS_NONZERO_CROSS_FAMILY_DISCRETE_SUPPORT"
        if selected_count >= 2
        and cross_family_counts[SUPPORT_TWO_SOURCE_ACTION_STATE] > 0
        and cross_family_counts[SUPPORT_PROPBANK_ROLE_INVENTORY] > 0
        else "NE_ZERO_REQUIRED_SOURCE_SUPPORT")
    return (
        candidates,
        source_census,
        family_records,
        proposal_records,
        fact_records,
        {
            "candidate_source_count": len(candidates),
            "capability_outcome": "NE_SOURCE_FEASIBILITY_NOT_AUTHORIZATION",
            "cross_family_matched_phrase_counts": {
                support: cross_family_counts[support]
                for support in _SUPPORT_ORDER},
            "feasibility_outcome": feasibility,
            "lexical_match_assigns_semantic_sense": 0,
            "oewn": oewn_census,
            "placeholder_role_assignment_count": 0,
            "propbank": propbank_census,
            "proposals": proposal_summary,
            "raw_or_lexical_surface_published": 0,
            "selected_source_count": selected_count,
            "source_family_count": len(V5_SOURCE_FAMILIES),
        },
    )


__all__ = [
    "NEUTRAL_SEMANTIC_FACT_FAMILY_KIND",
    "NEUTRAL_SEMANTIC_FAMILY_COVERAGE_KIND",
    "NEUTRAL_SEMANTIC_PROPOSAL_COVERAGE_KIND",
    "NEUTRAL_SEMANTIC_SOURCE_CANDIDATE_KIND",
    "NEUTRAL_SEMANTIC_SOURCE_CENSUS_KIND",
    "NEUTRAL_SEMANTIC_TARGET_SCOPE",
    "OEWN_SOURCE_ID",
    "PROPBANK_SOURCE_ID",
    "SUPPORT_OEWN_ACTION_STATE",
    "SUPPORT_OEWN_ANY",
    "SUPPORT_PROPBANK_MODAL_CUE",
    "SUPPORT_PROPBANK_NEGATION_CUE",
    "SUPPORT_PROPBANK_PREDICATE",
    "SUPPORT_PROPBANK_ROLE_INVENTORY",
    "SUPPORT_TWO_SOURCE_ACTION_STATE",
    "SUPPORT_TWO_SOURCE_LEXICAL",
    "derive_neutral_semantic_source_feasibility",
    "normalize_optional_semantic_source_text",
    "parse_open_english_wordnet",
    "parse_propbank_frames",
]
