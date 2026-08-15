"""冻结 recovery-v9 GIMP held-out 来源的标签盲元数据。

本模块只承诺官方仓库、固定 revision、许可与完整简繁 locale blob 清单。
任何 locale 正文、translation、pair、词频或评测标签都不在该边界内读取。
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


NORMALIZATION_RECOVERY_V9_SOURCE_ROSTER_RECORD_KIND = (
    "NORMALIZATION_RECOVERY_V9_GIMP_EVALUATION_SOURCE_ROSTER_RECORD_V1")
NORMALIZATION_RECOVERY_V9_SOURCE_ROSTER_CENSUS_KIND = (
    "NORMALIZATION_RECOVERY_V9_GIMP_EVALUATION_SOURCE_ROSTER_CENSUS_V1")
NORMALIZATION_RECOVERY_V9_SOURCE_ROSTER_SCOPE = (
    "GIMP_ALL_EIGHT_DOMAIN_ZH_TW_TO_ZH_CN_HELD_OUT_V1")

_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _blob(
        path: str,
        *,
        size: int,
        sha1: str,
        sha256: str = "",
        domain: str = "",
        locale: str = "",
        ) -> dict[str, object]:
    """构造一个不含正文的固定 Git blob identity。"""
    value: dict[str, object] = {
        "bytes": size,
        "git_blob_sha1": sha1,
        "relative_path": path,
    }
    if sha256:
        value["sha256"] = sha256
    if domain:
        value["domain"] = domain
    if locale:
        value["locale"] = locale
    return value


_LICENSE_FILES = (
    _blob(
        "COPYING",
        size=35_151,
        sha1="e60008693e017bec1b4eb49c84be3898e26fcf2a",
        sha256=(
            "e79e9c8a0c85d735ff98185918ec94ed7d175efc377012787aebcf3b80f0d90b"),
    ),
    _blob(
        "LICENSE",
        size=2_823,
        sha1="eb43828b995bd169913e2032aed4201194562d70",
        sha256=(
            "0986a9c943105de194155dd1897a58b81f941c9045e361b5477c958cfbaf7b0a"),
    ),
)

_LOCALE_FILES = (
    _blob(
        "po/zh_CN.po", size=891_194,
        sha1="dd58b4caab8eacff1ac496fa436d3acd478cc317",
        domain="po", locale="zh_CN"),
    _blob(
        "po/zh_TW.po", size=810_090,
        sha1="ed2b87e76ab669e6c8fdf6469147b3fcfd7814e5",
        domain="po", locale="zh_TW"),
    _blob(
        "po-libgimp/zh_CN.po", size=88_553,
        sha1="022cf1dd932736e0192f90bc94399d4302dbcfc4",
        domain="po-libgimp", locale="zh_CN"),
    _blob(
        "po-libgimp/zh_TW.po", size=83_082,
        sha1="3e2dfb076a84a8c2005a7cb08118385a8316abf8",
        domain="po-libgimp", locale="zh_TW"),
    _blob(
        "po-plug-ins/zh_CN.po", size=586_321,
        sha1="ccda1b913cc95c9e5404f7868fc41d2d7ff3b792",
        domain="po-plug-ins", locale="zh_CN"),
    _blob(
        "po-plug-ins/zh_TW.po", size=531_159,
        sha1="281fa1e37b02a9776311b1ff6fc44387f3a5ab2f",
        domain="po-plug-ins", locale="zh_TW"),
    _blob(
        "po-python/zh_CN.po", size=36_648,
        sha1="d16ec5c42fb48512eff8f46a90a9332ac6edc03d",
        domain="po-python", locale="zh_CN"),
    _blob(
        "po-python/zh_TW.po", size=31_211,
        sha1="58901034aca49ca07d44a6f18107c8c38dea77f8",
        domain="po-python", locale="zh_TW"),
    _blob(
        "po-script-fu/zh_CN.po", size=67_847,
        sha1="ed861450205529ee870508572934813ab52ca130",
        domain="po-script-fu", locale="zh_CN"),
    _blob(
        "po-script-fu/zh_TW.po", size=43_450,
        sha1="2991ebe9aafbb53ee650301ef2d3df49de2e3c2e",
        domain="po-script-fu", locale="zh_TW"),
    _blob(
        "po-tags/zh_CN.po", size=927,
        sha1="b141afe4077f77656af9d71853773b94f5f3dc17",
        domain="po-tags", locale="zh_CN"),
    _blob(
        "po-tags/zh_TW.po", size=870,
        sha1="7a1d043df30a9ab0e78b84c9321310580cb4b83c",
        domain="po-tags", locale="zh_TW"),
    _blob(
        "po-tips/zh_CN.po", size=13_774,
        sha1="82adc71e5677ee7bd3e7b8f5a2537b1ecf21e418",
        domain="po-tips", locale="zh_CN"),
    _blob(
        "po-tips/zh_TW.po", size=14_911,
        sha1="6652b7e8b4367c3688e1a3ae6eb58b0233124278",
        domain="po-tips", locale="zh_TW"),
    _blob(
        "po-windows-installer/zh_CN.po", size=13_623,
        sha1="9ffe5075f1d38591c53967d99c155928fdb9e181",
        domain="po-windows-installer", locale="zh_CN"),
    _blob(
        "po-windows-installer/zh_TW.po", size=14_425,
        sha1="711d090be2e874c1cc0d60eb1200c7d95f1144cd",
        domain="po-windows-installer", locale="zh_TW"),
)

_PRIOR_SOURCE_REPOSITORIES = frozenset({
    "https://github.com/LibreOffice/translations.git",
    "https://github.com/Stellarium/stellarium.git",
    "https://github.com/audacity/audacity.git",
    "https://github.com/bitcoin/bitcoin.git",
    "https://github.com/godotengine/godot",
    "https://github.com/keepassxreboot/keepassxc.git",
    "https://github.com/microsoft/vscode-loc",
    "https://github.com/mozilla-l10n/firefox-l10n",
    "https://github.com/qbittorrent/qBittorrent.git",
    "https://github.com/qt/qttranslations.git",
    "https://github.com/thunderbird/thunderbird-l10n",
    "https://github.com/videolan/vlc.git",
})


def _sha256(payload: bytes) -> str:
    """返回规范 roster record identity。"""
    return hashlib.sha256(payload).hexdigest()


def _record_id(value: dict[str, object]) -> str:
    """从完整 project/revision/scope 形成不可变记录 identity。"""
    return _sha256(canonical_json_bytes(value))


def _validate_blob(item: dict[str, object], *, license_file: bool) -> None:
    """核验 blob commitment 只含路径、大小与摘要元数据。"""
    path = item.get("relative_path")
    size = item.get("bytes")
    sha1 = item.get("git_blob_sha1")
    if (not isinstance(path, str) or not path or path.startswith("/")
            or "\\" in path or ".." in path.split("/")
            or type(size) is not int or size <= 0
            or not isinstance(sha1, str) or not _SHA1.fullmatch(sha1)):
        raise BroadQaExternalDataError("v9 source roster blob identity 漂移")
    if license_file:
        sha256 = item.get("sha256")
        if not isinstance(sha256, str) or not _SHA256.fullmatch(sha256):
            raise BroadQaExternalDataError("v9 source roster license SHA 漂移")
    elif (item.get("locale") not in {"zh_CN", "zh_TW"}
          or not isinstance(item.get("domain"), str)
          or not item["domain"]):
        raise BroadQaExternalDataError("v9 source roster locale identity 漂移")


def _locale_pair_count(files: tuple[dict[str, object], ...]) -> int:
    """要求每个已发现 GIMP gettext domain 完整保留简繁双侧。"""
    counts = Counter(
        (str(item["domain"]), str(item["locale"])) for item in files)
    domains = {domain for domain, _locale in counts}
    if (len(domains) != 8 or any(
            counts[(domain, locale)] != 1
            for domain in domains for locale in ("zh_CN", "zh_TW"))):
        raise BroadQaExternalDataError("v9 source roster domain pair 漂移")
    return len(domains)


def derive_normalization_recovery_v9_source_roster() -> tuple[
        tuple[dict[str, object], ...], dict[str, object]]:
    """派生 GIMP 八 domain held-out 来源的标签盲冻结记录。"""
    repository = "https://gitlab.gnome.org/GNOME/gimp.git"
    if repository in _PRIOR_SOURCE_REPOSITORIES:
        raise BroadQaExternalDataError("v9 source roster repository 非独立")
    for item in _LICENSE_FILES:
        _validate_blob(item, license_file=True)
    for item in _LOCALE_FILES:
        _validate_blob(item, license_file=False)
    paths = [str(item["relative_path"])
             for item in _LICENSE_FILES + _LOCALE_FILES]
    hashes = [str(item["git_blob_sha1"])
              for item in _LICENSE_FILES + _LOCALE_FILES]
    if len(paths) != len(set(paths)) or len(hashes) != len(set(hashes)):
        raise BroadQaExternalDataError("v9 source roster file inventory 漂移")
    pair_count = _locale_pair_count(_LOCALE_FILES)
    identity = {
        "commit": "78fc57122afa94d34adeca670b7d89d663f77789",
        "repository": repository,
        "root_tree": "efc8a0d0df6606bc8b61b86b936e151f496013c8",
        "source_family": "GIMP_PROJECT",
        "target_scope": NORMALIZATION_RECOVERY_V9_SOURCE_ROSTER_SCOPE,
    }
    record = {
        **identity,
        "commit_date": "2026-08-15T21:03:16Z",
        "family_independence_group": "GIMP_UPSTREAM",
        "format_version": 1,
        "label_or_translation_read_count": 0,
        "license": {
            "detection_key": "gpl-3.0+",
            "detection_name": "GNU General Public License v3.0 or later",
            "detection_source": "OFFICIAL_GITLAB_PROJECT_LICENSE_METADATA",
            "expression": "GPL-3.0-or-later",
            "files": list(_LICENSE_FILES),
        },
        "license_file_content_read_count": 2,
        "locale_blob_content_read_count": 0,
        "locale_file_count": len(_LOCALE_FILES),
        "locale_files": list(_LOCALE_FILES),
        "locale_pair_count": pair_count,
        "official_source_binding": (
            "GETTEXT_DOMAIN_MSGCTXT_MSGID_MSGID_PLURAL"),
        "parser_identity": "POLIB_1_2_0_GETTEXT_PO_V1",
        "prior_source_repository_count": len(_PRIOR_SOURCE_REPOSITORIES),
        "prior_source_repository_overlap_count": 0,
        "record_id": _record_id(identity),
        "record_kind": NORMALIZATION_RECOVERY_V9_SOURCE_ROSTER_RECORD_KIND,
        "selection_status": (
            "SELECTED_ALL_EIGHT_DOMAIN_TREE_LICENSE_PATH_FROZEN"),
        "source_identity_fields": [
            "domain", "msgctxt", "msgid", "msgid_plural"],
        "source_policy_scope": (
            "GIMP_ALL_DOMAIN_ZH_TW_TO_ZH_CN_FIXED_COMMIT_V1"),
        "surface_published": 0,
    }
    census = {
        "all_discovered_complete_domain_pairs_selected": 1,
        "family_independence_group_count": 1,
        "label_or_translation_read_count": 0,
        "license_file_content_read_count": 2,
        "license_file_count": len(_LICENSE_FILES),
        "locale_blob_content_read_count": 0,
        "locale_file_count": len(_LOCALE_FILES),
        "locale_pair_count": pair_count,
        "locale_total_bytes": sum(int(item["bytes"])
                                  for item in _LOCALE_FILES),
        "official_source_bound_family_count": 1,
        "prior_source_repository_count": len(_PRIOR_SOURCE_REPOSITORIES),
        "prior_source_repository_overlap_count": 0,
        "selected_source_family_count": 1,
        "surface_published": 0,
    }
    return (record,), census


__all__ = [
    "NORMALIZATION_RECOVERY_V9_SOURCE_ROSTER_CENSUS_KIND",
    "NORMALIZATION_RECOVERY_V9_SOURCE_ROSTER_RECORD_KIND",
    "NORMALIZATION_RECOVERY_V9_SOURCE_ROSTER_SCOPE",
    "derive_normalization_recovery_v9_source_roster",
]
