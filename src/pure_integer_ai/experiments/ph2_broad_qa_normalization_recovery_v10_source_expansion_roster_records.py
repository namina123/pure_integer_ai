"""冻结 recovery-v10 独立 TRAIN 来源扩充候选名册。

本模块只保存已经核对的上游 Git revision、许可、locale 路径和选择理由。
任何翻译正文、pair、词频或评测标签都不在本边界内读取。OBS Studio 只
保留为后继 fresh formal 候选，不能被本轮 TRAIN 内容审计消费。
"""
from __future__ import annotations

from collections import Counter
import hashlib
import re

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
)


V10_SOURCE_EXPANSION_CANDIDATE_RECORD_KIND = (
    "NORMALIZATION_RECOVERY_V10_SOURCE_EXPANSION_CANDIDATE_V1")
V10_SOURCE_EXPANSION_EXCLUSION_RECORD_KIND = (
    "NORMALIZATION_RECOVERY_V10_SOURCE_EXPANSION_EXCLUSION_V1")
V10_SOURCE_EXPANSION_CENSUS_RECORD_KIND = (
    "NORMALIZATION_RECOVERY_V10_SOURCE_EXPANSION_CENSUS_V1")
V10_SOURCE_EXPANSION_SCOPE = (
    "INDEPENDENT_ZH_TW_TO_ZH_CN_SOURCE_CONTEXT_EXPANSION_V1")

_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def _blob(
        path: str,
        *,
        size: int,
        sha1: str,
        sha256: str = "",
        locale: str = "",
        role: str = "",
        ) -> dict[str, object]:
    """构造一个不含正文的固定 Git blob identity。"""
    value: dict[str, object] = {
        "bytes": size,
        "git_blob_sha1": sha1,
        "relative_path": path,
    }
    if sha256:
        value["sha256"] = sha256
    if locale:
        value["locale"] = locale
    if role:
        value["role"] = role
    return value


_MIXXX_LICENSE_FILES = (
    _blob(
        "COPYING",
        size=210,
        sha1="5914766cf2e120f3c95385c34fec80cbb9200641",
        sha256=(
            "98adb836b75addd92719b0b90ae3200c4b2e2074508135d9057ee0ca4ca9faa5"),
        role="PROJECT_COPYRIGHT_AND_LICENSE_POINTER",
    ),
    _blob(
        "LICENSE",
        size=177_985,
        sha1="8ea4ac28edd6d60cf9f0a23deddf92f9ec924ebe",
        sha256=(
            "c10911954286426c2335e8ccddb5496b75a89e797cfe4ada9f3e64d744759a0c"),
        role="PRIMARY_LICENSE_AND_BUNDLED_NOTICES",
    ),
)
_MIXXX_LOCALE_FILES = (
    _blob(
        "res/translations/mixxx_zh_CN.ts",
        size=872_354,
        sha1="16d427eb0e86b095304828f0f853e61437271e35",
        locale="zh_CN",
        role="TRAIN_TRANSLATION_WITH_OFFICIAL_SOURCE",
    ),
    _blob(
        "res/translations/mixxx_zh_TW.ts",
        size=872_519,
        sha1="041596c86e0525c55f02727227ba41d3a56c7052",
        locale="zh_TW",
        role="TRAIN_TRANSLATION_WITH_OFFICIAL_SOURCE",
    ),
)
_MIXXX_EXCLUDED_DERIVATIVES = (
    _blob(
        "res/translations/mixxx_zh_CN.qm",
        size=340_228,
        sha1="fc06bc4fe9bd92418b93eae8d5d3ceeab1c02845",
        locale="zh_CN",
        role="EXCLUDED_COMPILED_QM_DERIVATIVE",
    ),
    _blob(
        "res/translations/mixxx_zh_TW.qm",
        size=340_481,
        sha1="07d3a556230e93d8dcaba9bea1df7291457690a8",
        locale="zh_TW",
        role="EXCLUDED_COMPILED_QM_DERIVATIVE",
    ),
)

_MUMBLE_LICENSE_FILES = (
    _blob(
        "LICENSE",
        size=1_578,
        sha1="8c0a1ccea030f93f3a0e356b5f240b0232a5e02a",
        sha256=(
            "b4648cb20ff93945b85c82f470050e2a67d2b2f6b2f8103cf72f0fa57a9bcad5"),
        role="PRIMARY_PROJECT_LICENSE",
    ),
)
_MUMBLE_LOCALE_FILES = (
    _blob(
        "src/mumble/mumble_zh_CN.ts",
        size=451_928,
        sha1="f6f9f752db3057568d2dea44a2cc6dd3b480f9fc",
        locale="zh_CN",
        role="TRAIN_TRANSLATION_WITH_OFFICIAL_SOURCE",
    ),
    _blob(
        "src/mumble/mumble_zh_TW.ts",
        size=418_179,
        sha1="3d6f8ad24ccdc5b1876a349f8214ce65feaf0b85",
        locale="zh_TW",
        role="TRAIN_TRANSLATION_WITH_OFFICIAL_SOURCE",
    ),
)
_MUMBLE_UNPAIRED_LOCALE_FILES = (
    _blob(
        "src/mumble/qttranslations/qt_zh_CN.ts",
        size=307_789,
        sha1="f7aef4c9cff451cf0443214b152b9bac35749834",
        locale="zh_CN",
        role="EXCLUDED_NO_ZH_TW_COUNTERPART",
    ),
)

_OBS_LICENSE_FILES = (
    _blob(
        "COPYING",
        size=18_092,
        sha1="d159169d1050894d3ea3b98e1c965c4058208fe1",
        sha256=(
            "8177f97513213526df2cf6184d8ff986c675afb514d4e68a404010521b880643"),
        role="PRIMARY_PROJECT_LICENSE",
    ),
)


_HISTORICAL_EXCLUSIONS = (
    ("AUDACITY_PROJECT", "https://github.com/audacity/audacity.git",
     "FORMAL_FAMILY_CONSUMED"),
    ("FIREFOX_PROJECT", "https://github.com/mozilla-l10n/firefox-l10n",
     "FORMAL_FAMILY_CONSUMED"),
    ("GIMP_PROJECT", "https://gitlab.gnome.org/GNOME/gimp.git",
     "FORMAL_FAMILY_CONSUMED"),
    ("QT_PROJECT", "https://github.com/qt/qttranslations.git",
     "FORMAL_FAMILY_CONSUMED"),
    ("VLC_PROJECT", "https://github.com/videolan/vlc.git",
     "FORMAL_FAMILY_CONSUMED"),
    ("GODOT_ENGINE_PROJECT", "https://github.com/godotengine/godot",
     "HISTORICAL_TRAIN_OR_FEASIBILITY_FAMILY"),
    ("LIBREOFFICE_PROJECT", "https://github.com/LibreOffice/translations.git",
     "HISTORICAL_TRAIN_OR_FEASIBILITY_FAMILY"),
    ("THUNDERBIRD_PROJECT", "https://github.com/thunderbird/thunderbird-l10n",
     "HISTORICAL_TRAIN_OR_FEASIBILITY_FAMILY"),
    ("VSCODE_PROJECT", "https://github.com/microsoft/vscode-loc",
     "HISTORICAL_TRAIN_OR_FEASIBILITY_FAMILY"),
    ("KEEPASSXC_PROJECT", "https://github.com/keepassxreboot/keepassxc.git",
     "ACTIVE_PREDECESSOR_TRAIN_FAMILY"),
    ("QBITTORRENT_PROJECT", "https://github.com/qbittorrent/qBittorrent.git",
     "ACTIVE_PREDECESSOR_TRAIN_FAMILY"),
    ("STELLARIUM_PROJECT", "https://github.com/Stellarium/stellarium.git",
     "ACTIVE_PREDECESSOR_TRAIN_FAMILY"),
    ("BITCOIN_CORE_PROJECT", "https://github.com/bitcoin/bitcoin.git",
     "PREDECESSOR_REJECTED_ZERO_ACTIVE_COMMON_PAIR"),
    ("CC_CEDICT", "https://www.mdbg.net/chinese/dictionary?page=cc-cedict",
     "LICENSE_RECONCILIATION_BLOCKED"),
)


def _sha256(payload: bytes) -> str:
    """返回规范记录 identity。"""
    return hashlib.sha256(payload).hexdigest()


def _record_id(value: dict[str, object]) -> str:
    """从固定上游与选择身份形成记录 id。"""
    return _sha256(canonical_json_bytes(value))


def _validate_blob(item: object, *, license_file: bool) -> None:
    """核验 blob 只含合法路径、大小与摘要元数据。"""
    if not isinstance(item, dict):
        raise BroadQaExternalDataError("v10 source roster blob 非对象")
    path = item.get("relative_path")
    size = item.get("bytes")
    sha1 = item.get("git_blob_sha1")
    if (not isinstance(path, str) or not path or path.startswith("/")
            or "\\" in path or ".." in path.split("/")
            or type(size) is not int or size <= 0
            or not isinstance(sha1, str) or not _SHA1.fullmatch(sha1)):
        raise BroadQaExternalDataError("v10 source roster blob identity 漂移")
    if license_file:
        sha256 = item.get("sha256")
        if not isinstance(sha256, str) or not _SHA256.fullmatch(sha256):
            raise BroadQaExternalDataError("v10 source roster license SHA 漂移")


def _candidate(
        *,
        source_family: str,
        repository: str,
        commit: str,
        commit_date: str,
        root_tree: str,
        selection_status: str,
        format_name: str,
        license_expression: str,
        license_files: tuple[dict[str, object], ...] = (),
        locale_files: tuple[dict[str, object], ...] = (),
        excluded_locale_files: tuple[dict[str, object], ...] = (),
        complete_locale_pair_count: int = 0,
        complete_locale_total_bytes: int = 0,
        official_source_binding: str = "",
        license_detection_source: str = "OFFICIAL_LICENSE_TEXT",
        ) -> dict[str, object]:
    """构造一个标签盲来源候选决策记录。"""
    if (not source_family or not repository.startswith("https://")
            or not _SHA1.fullmatch(commit) or not _SHA1.fullmatch(root_tree)
            or not _DATE.fullmatch(commit_date)
            or type(complete_locale_pair_count) is not int
            or complete_locale_pair_count < 0
            or type(complete_locale_total_bytes) is not int
            or complete_locale_total_bytes < 0):
        raise BroadQaExternalDataError("v10 source roster candidate identity 漂移")
    for item in license_files:
        _validate_blob(item, license_file=True)
    for item in locale_files + excluded_locale_files:
        _validate_blob(item, license_file=False)
    identity = {
        "commit": commit,
        "repository": repository,
        "selection_status": selection_status,
        "source_family": source_family,
        "target_scope": V10_SOURCE_EXPANSION_SCOPE,
    }
    return {
        **identity,
        "commit_date": commit_date,
        "complete_locale_pair_count": complete_locale_pair_count,
        "complete_locale_total_bytes": complete_locale_total_bytes,
        "excluded_locale_files": list(excluded_locale_files),
        "format": format_name,
        "format_version": 1,
        "label_or_translation_read_count": 0,
        "license": {
            "detection_source": license_detection_source,
            "expression": license_expression,
            "files": list(license_files),
        },
        "license_file_content_read_count": len(license_files),
        "locale_blob_content_read_count": 0,
        "locale_files": list(locale_files),
        "official_source_binding": official_source_binding,
        "record_id": _record_id(identity),
        "record_kind": V10_SOURCE_EXPANSION_CANDIDATE_RECORD_KIND,
        "root_tree": root_tree,
        "surface_published": 0,
    }


def _candidates() -> tuple[dict[str, object], ...]:
    """形成两家 TRAIN、一个 formal 保留和一个拒绝候选。"""
    return (
        _candidate(
            source_family="MIXXX_PROJECT",
            repository="https://github.com/mixxxdj/mixxx.git",
            commit="9e670c1120cc82304c4d5dcaa11a36367c5d50c3",
            commit_date="2026-08-12T19:59:55Z",
            root_tree="cde0e59a34502e83c66453a0c68073689ef0d74c",
            selection_status="SELECTED_TRAIN_CONTENT_FEASIBILITY_PENDING",
            format_name="QT_TS_XML_V1",
            license_expression="GPL-2.0-or-later",
            license_files=_MIXXX_LICENSE_FILES,
            locale_files=_MIXXX_LOCALE_FILES,
            excluded_locale_files=_MIXXX_EXCLUDED_DERIVATIVES,
            complete_locale_pair_count=1,
            complete_locale_total_bytes=1_744_873,
            official_source_binding="QT_TS_SOURCE_ELEMENT",
        ),
        _candidate(
            source_family="MUMBLE_PROJECT",
            repository="https://github.com/mumble-voip/mumble.git",
            commit="66b7c072b20dee26a63697c9e92b199b740fad99",
            commit_date="2026-08-15T14:40:47Z",
            root_tree="802f5a6a40513b47a1e704bad281f322e03613bb",
            selection_status="SELECTED_TRAIN_CONTENT_FEASIBILITY_PENDING",
            format_name="QT_TS_XML_V1",
            license_expression="BSD-3-Clause",
            license_files=_MUMBLE_LICENSE_FILES,
            locale_files=_MUMBLE_LOCALE_FILES,
            excluded_locale_files=_MUMBLE_UNPAIRED_LOCALE_FILES,
            complete_locale_pair_count=1,
            complete_locale_total_bytes=870_107,
            official_source_binding="QT_TS_SOURCE_ELEMENT",
        ),
        _candidate(
            source_family="OBS_STUDIO_PROJECT",
            repository="https://github.com/obsproject/obs-studio.git",
            commit="1bf1379faa8b87dbb1cb75635e7880e7d9625b8c",
            commit_date="2026-08-14T20:57:12Z",
            root_tree="abc9ae49a59afa26bc487063d8c2a22f4567cd4a",
            selection_status="RESERVED_UNREAD_FRESH_FORMAL_CANDIDATE",
            format_name="OBS_LOCALE_INI_V1",
            license_expression="GPL-2.0-only",
            license_files=_OBS_LICENSE_FILES,
            complete_locale_pair_count=38,
            complete_locale_total_bytes=261_085,
            official_source_binding="OBS_LOCALE_KEY_WITH_EN_US_SOURCE",
        ),
        _candidate(
            source_family="WIRESHARK_PROJECT",
            repository="https://github.com/wireshark/wireshark.git",
            commit="716a200295d2f91629211fd8660e8cce37c13884",
            commit_date="2026-08-16T02:15:23Z",
            root_tree="b3b990adeb4d2f52e3e906b4d92517d2101ce9a4",
            selection_status="REJECTED_NO_ZH_TW_COUNTERPART",
            format_name="QT_TS_XML_V1",
            license_expression="GPL-2.0-or-later",
            license_detection_source=(
                "OFFICIAL_REPOSITORY_LICENSE_METADATA"),
            excluded_locale_files=(_blob(
                "ui/qt/wireshark_zh_CN.ts",
                size=642_165,
                sha1="dd599595176751f3f4568ff6faf0d72a9e508368",
                locale="zh_CN",
                role="REJECTED_NO_ZH_TW_COUNTERPART",
            ),),
            official_source_binding="QT_TS_SOURCE_ELEMENT",
        ),
    )


def _exclusions() -> tuple[dict[str, object], ...]:
    """冻结全部历史来源与许可 blocker 的排除账。"""
    records = []
    for source_family, repository, reason in _HISTORICAL_EXCLUSIONS:
        identity = {
            "repository": repository,
            "source_family": source_family,
            "target_scope": V10_SOURCE_EXPANSION_SCOPE,
        }
        records.append({
            **identity,
            "exclusion_reason": reason,
            "format_version": 1,
            "record_id": _record_id(identity),
            "record_kind": V10_SOURCE_EXPANSION_EXCLUSION_RECORD_KIND,
        })
    records.sort(key=lambda item: str(item["source_family"]))
    return tuple(records)


def derive_normalization_recovery_v10_source_expansion_roster() -> tuple[
        tuple[dict[str, object], ...],
        tuple[dict[str, object], ...],
        dict[str, object],
        ]:
    """派生标签盲候选、历史排除账和完整 census。"""
    candidates = _candidates()
    exclusions = _exclusions()
    families = [str(item["source_family"]) for item in candidates]
    repositories = [str(item["repository"]) for item in candidates]
    exclusion_repositories = {str(item["repository"]) for item in exclusions}
    if (len(families) != len(set(families))
            or len(repositories) != len(set(repositories))
            or set(repositories).intersection(exclusion_repositories)):
        raise BroadQaExternalDataError("v10 source roster 独立性账漂移")
    status_counts = Counter(str(item["selection_status"])
                            for item in candidates)
    selected = tuple(item for item in candidates if item["selection_status"]
                     == "SELECTED_TRAIN_CONTENT_FEASIBILITY_PENDING")
    if ({str(item["source_family"]) for item in selected}
            != {"MIXXX_PROJECT", "MUMBLE_PROJECT"}
            or status_counts["RESERVED_UNREAD_FRESH_FORMAL_CANDIDATE"] != 1
            or status_counts["REJECTED_NO_ZH_TW_COUNTERPART"] != 1):
        raise BroadQaExternalDataError("v10 source roster 决策账漂移")
    census = {
        "candidate_count": len(candidates),
        "candidate_locale_blob_content_read_count": 0,
        "candidate_surface_published": 0,
        "historical_exclusion_count": len(exclusions),
        "rejected_candidate_count": 1,
        "reserved_formal_candidate_count": 1,
        "selected_train_candidate_count": len(selected),
        "selected_train_complete_locale_pair_count": sum(
            int(item["complete_locale_pair_count"]) for item in selected),
        "selected_train_locale_file_count": sum(
            len(item["locale_files"]) for item in selected),
        "selected_train_locale_total_bytes": sum(
            int(item["complete_locale_total_bytes"]) for item in selected),
    }
    return candidates, exclusions, census


__all__ = [
    "V10_SOURCE_EXPANSION_CANDIDATE_RECORD_KIND",
    "V10_SOURCE_EXPANSION_CENSUS_RECORD_KIND",
    "V10_SOURCE_EXPANSION_EXCLUSION_RECORD_KIND",
    "V10_SOURCE_EXPANSION_SCOPE",
    "derive_normalization_recovery_v10_source_expansion_roster",
]
