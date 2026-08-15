"""冻结 recovery-v8 新 TRAIN source family 的无表面 roster。

本模块只保存远端 Git tree 已核对的 repository、revision、license 与 locale
blob identity。locale 内容必须等 roster commitment 发布后才可读取，且后续
source pack 仍需逐 blob、parser、许可与完整 selection 重新核验。
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


NORMALIZATION_RECOVERY_V8_SOURCE_ROSTER_RECORD_KIND = (
    "NORMALIZATION_RECOVERY_V8_TRAIN_SOURCE_ROSTER_RECORD_V1")
NORMALIZATION_RECOVERY_V8_SOURCE_ROSTER_CENSUS_KIND = (
    "NORMALIZATION_RECOVERY_V8_TRAIN_SOURCE_ROSTER_CENSUS_V1")
NORMALIZATION_RECOVERY_V8_SOURCE_ROSTER_SCOPE = (
    "ZH_CN_CROSS_PRODUCT_SOURCE_CONDITIONED_TRANSFER_V8")

_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def _blob(path: str, *, size: int, sha1: str) -> dict[str, object]:
    """构造一个未读内容的 Git blob identity。"""
    return {"bytes": size, "git_blob_sha1": sha1, "relative_path": path}


_CANDIDATES = (
    {
        "commit": "c90c23d388f66b7eef67f4c6f69184c088727d6a",
        "commit_date": "2026-08-15T14:20:47Z",
        "family_independence_group": "BITCOIN_CORE_UPSTREAM",
        "format": "QT_TS_XML_V1",
        "license": {
            "expression": "MIT",
            "files": (_blob(
                "COPYING", size=1_142,
                sha1="89960cbf2f221a29852ed162b25bda2afc0b2dd6"),),
            "primary_bytes": 1_142,
            "primary_sha256": (
                "b028769f3852a9368ab10bd754ff01ebb741f84a2fa658c9aff82a631bc6ecfc"),
        },
        "locale_files": (
            _blob(
                "src/qt/locale/bitcoin_zh_CN.ts", size=233_165,
                sha1="d5255cff78dd91dca41516c141fb8a1693c44c8d"),
            _blob(
                "src/qt/locale/bitcoin_zh_TW.ts", size=233_129,
                sha1="b2c3e1fe56c9cfc5098b3c4e7d37507b5ee05e2f"),
        ),
        "official_source_binding": "QT_TS_SOURCE_ELEMENT",
        "repository": "https://github.com/bitcoin/bitcoin.git",
        "root_tree": "a81f8cb056b0b1565bccac4058c3160000908596",
        "source_family": "BITCOIN_CORE_PROJECT",
        "source_policy_scope": "BITCOIN_ZH_TW_TO_ZH_CN_FIXED_COMMIT_V1",
    },
    {
        "commit": "27a1fab8de2f6f8ce1473daca07c1dddda15ae20",
        "commit_date": "2026-08-15T10:32:15Z",
        "family_independence_group": "QBITTORRENT_UPSTREAM",
        "format": "QT_TS_XML_V1",
        "license": {
            "expression": (
                "LicenseRef-qBittorrent-GPL-2.0-or-later-with-exception"),
            "files": (
                _blob(
                    "COPYING", size=1_056,
                    sha1="9570e1fa841cf23bec987f92f241718ff6e5f46f"),
                _blob(
                    "COPYING.GPLv2", size=17_984,
                    sha1="9efa6fbc962836e243e20f7f23db062e2c077d28"),
                _blob(
                    "COPYING.GPLv3", size=35_149,
                    sha1="f288702d2fa16d3cdf0035b15a9fcbc552cd88e7"),
            ),
            "primary_bytes": 1_056,
            "primary_sha256": (
                "e675cd856f9817474455200ba7e6f5b7cc42d6598a5eecbbbdaa0e6fd304d6b7"),
        },
        "locale_files": (
            _blob(
                "src/lang/qbittorrent_zh_CN.ts", size=599_219,
                sha1="db876379673fa512b6804403890c811eb1d7e808"),
            _blob(
                "src/lang/qbittorrent_zh_TW.ts", size=599_674,
                sha1="84b1b90ea074f03c618a6c3e58151a25d6f8841a"),
        ),
        "official_source_binding": "QT_TS_SOURCE_ELEMENT",
        "repository": "https://github.com/qbittorrent/qBittorrent.git",
        "root_tree": "37703683fa1be10ad0a840b98a2b2337b35e027b",
        "source_family": "QBITTORRENT_PROJECT",
        "source_policy_scope": "QBITTORRENT_ZH_TW_TO_ZH_CN_FIXED_COMMIT_V1",
    },
    {
        "commit": "e9d341bad27ca657128a94d49aff7f2e5d8ff530",
        "commit_date": "2026-08-15T19:09:45Z",
        "family_independence_group": "STELLARIUM_UPSTREAM",
        "format": "GETTEXT_PO_V1",
        "license": {
            "expression": "GPL-2.0-only",
            "files": (_blob(
                "COPYING", size=17_992,
                sha1="b35f35c99338ef974fc40dd4c11320d54be65289"),),
            "primary_bytes": 17_992,
            "primary_sha256": (
                "3aeeb5bb98bf7041ab82cffe15efa28ac58ee2bdf162b71301f5c192be631259"),
        },
        "locale_files": (
            _blob(
                "po/stellarium/zh_CN.po", size=1_245_642,
                sha1="3902dd6d975bb75ed041b30ea87ddcced05cfc96"),
            _blob(
                "po/stellarium/zh_TW.po", size=1_220_706,
                sha1="f27692748194d3391b70b83a828cb370d9a19380"),
            _blob(
                "po/stellarium-desktop/zh_CN.po", size=1_607,
                sha1="77a591179b45db472528ef4bea1d923e7f572438"),
            _blob(
                "po/stellarium-desktop/zh_TW.po", size=1_637,
                sha1="d648ad2de995b50e39b06e8bf9b534adf38ce881"),
            _blob(
                "po/stellarium-landscapes-descriptions/zh_CN.po", size=8_060,
                sha1="317701abc5357acd95b9d55f4358189c02d17c1f"),
            _blob(
                "po/stellarium-landscapes-descriptions/zh_TW.po", size=8_153,
                sha1="940c15f324ac137749433198d17a54f881237095"),
            _blob(
                "po/stellarium-metainfo/zh_CN.po", size=3_362,
                sha1="8928dee859188e28b77884075a34e3e696e84a78"),
            _blob(
                "po/stellarium-metainfo/zh_TW.po", size=3_084,
                sha1="9fdb0a3392c463a5e779b7cbca5a37e456bd0f2e"),
            _blob(
                "po/stellarium-planetary-features/zh_CN.po", size=3_933_672,
                sha1="42bd4cc9ff0d24a81149d40bd6e541884a2c7214"),
            _blob(
                "po/stellarium-planetary-features/zh_TW.po", size=3_791_574,
                sha1="68070394bb4f0d7f7fb99e14c0a85ade3be0df04"),
            _blob(
                "po/stellarium-remotecontrol/zh_CN.po", size=29_287,
                sha1="b33fd59eecd5f8be5454ff80a8fb95ff80fc9c9c"),
            _blob(
                "po/stellarium-remotecontrol/zh_TW.po", size=26_548,
                sha1="79945b2d41d5fcbf5984b36c061148d440239798"),
            _blob(
                "po/stellarium-scenery3d-descriptions/zh_CN.po", size=12_382,
                sha1="0e87f19970b33e2358ea5c63ee58cfa434ec40f8"),
            _blob(
                "po/stellarium-scenery3d-descriptions/zh_TW.po", size=12_664,
                sha1="bd6842f3418e0089ebda3f8e5a8c028f5dbf8b12"),
            _blob(
                "po/stellarium-scripts/zh_CN.po", size=49_189,
                sha1="aa66500cc7efc59dcf722d309ea08a6601009e10"),
            _blob(
                "po/stellarium-scripts/zh_TW.po", size=47_624,
                sha1="79eb8951f2fd58b8f4e4ad90a927f29b6bed7793"),
            _blob(
                "po/stellarium-sky/zh_CN.po", size=285_842,
                sha1="aa7f96d202fc02d278bd3210454c6c28b2e52a0d"),
            _blob(
                "po/stellarium-sky/zh_TW.po", size=287_090,
                sha1="9d20d9c0db9b9e83644f30cc0ac2e191e5ee03c1"),
            _blob(
                "po/stellarium-skycultures/zh_CN.po", size=2_908_339,
                sha1="02ccf663c0e11eb3b98941516ecd1794201fc7b1"),
            _blob(
                "po/stellarium-skycultures/zh_TW.po", size=2_849_024,
                sha1="3a1c75d329b49b1032bf11084328acc169bad8c3"),
            _blob(
                "po/stellarium-skycultures-descriptions/zh_CN.po",
                size=1_494_356,
                sha1="59040c284539c8d85cd831f958f37c0fb8504aab"),
            _blob(
                "po/stellarium-skycultures-descriptions/zh_TW.po", size=880_631,
                sha1="ba27998faadd49502074e35f092d079bf6b453d8"),
        ),
        "official_source_binding": "GETTEXT_MSGID",
        "repository": "https://github.com/Stellarium/stellarium.git",
        "root_tree": "c89ce90590c9b0f526a32192d4b2097bd5888c67",
        "source_family": "STELLARIUM_PROJECT",
        "source_policy_scope": "STELLARIUM_ZH_TW_TO_ZH_CN_FIXED_COMMIT_V1",
    },
)


def _sha256(payload: bytes) -> str:
    """返回规范 roster identity。"""
    return hashlib.sha256(payload).hexdigest()


def _record_id(value: dict[str, object]) -> str:
    """由完整 source family identity 形成记录 id。"""
    return _sha256(canonical_json_bytes(value))


def _locale_pair_count(files: tuple[dict[str, object], ...]) -> int:
    """核验每个固定 domain 都同时存在 zh-CN 与 zh-TW blob。"""
    domains = Counter()
    for item in files:
        path = str(item["relative_path"])
        if "zh_CN" in path:
            domain = path.replace("zh_CN", "{locale}")
            domains[(domain, "zh_CN")] += 1
        elif "zh_TW" in path:
            domain = path.replace("zh_TW", "{locale}")
            domains[(domain, "zh_TW")] += 1
        else:
            raise BroadQaExternalDataError("v8 roster locale path 非简繁对")
    names = {domain for domain, _locale in domains}
    if (not names or any(
            domains[(domain, locale)] != 1
            for domain in names for locale in ("zh_CN", "zh_TW"))):
        raise BroadQaExternalDataError("v8 roster locale domain 不成对")
    return len(names)


def _validate_blob(item: object, *, label: str) -> dict[str, object]:
    """核验一个 tree blob commitment 不含 surface。"""
    if not isinstance(item, dict):
        raise BroadQaExternalDataError(f"v8 roster {label} blob 非对象")
    path = item.get("relative_path")
    size = item.get("bytes")
    sha1 = item.get("git_blob_sha1")
    if (not isinstance(path, str) or not path or path.startswith("/")
            or "\\" in path or ".." in path.split("/")
            or type(size) is not int or size <= 0
            or not isinstance(sha1, str) or not _SHA1.fullmatch(sha1)):
        raise BroadQaExternalDataError(f"v8 roster {label} blob identity 漂移")
    return item


def derive_normalization_recovery_v8_source_roster() -> tuple[
        tuple[dict[str, object], ...], dict[str, object]]:
    """派生三个新 TRAIN family 的无表面冻结 roster。"""
    records = []
    families = set()
    groups = set()
    repositories = set()
    total_locale_bytes = 0
    total_locale_files = 0
    total_license_files = 0
    total_pairs = 0
    for candidate in _CANDIDATES:
        family = candidate["source_family"]
        group = candidate["family_independence_group"]
        repository = candidate["repository"]
        commit = candidate["commit"]
        tree = candidate["root_tree"]
        date = candidate["commit_date"]
        if (not isinstance(family, str) or not family
                or family in families or group in groups
                or repository in repositories
                or not isinstance(repository, str)
                or not repository.startswith("https://github.com/")
                or not repository.endswith(".git")
                or not isinstance(commit, str) or not _SHA1.fullmatch(commit)
                or not isinstance(tree, str) or not _SHA1.fullmatch(tree)
                or not isinstance(date, str) or not _DATE.fullmatch(date)
                or candidate["format"] not in {
                    "GETTEXT_PO_V1", "QT_TS_XML_V1"}
                or candidate["official_source_binding"] not in {
                    "GETTEXT_MSGID", "QT_TS_SOURCE_ELEMENT"}):
            raise BroadQaExternalDataError("v8 roster candidate identity 漂移")
        license_value = candidate["license"]
        if (not isinstance(license_value, dict)
                or not isinstance(license_value.get("expression"), str)
                or not license_value["expression"]
                or type(license_value.get("primary_bytes")) is not int
                or license_value["primary_bytes"] <= 0
                or not isinstance(license_value.get("primary_sha256"), str)
                or not _SHA256.fullmatch(license_value["primary_sha256"])):
            raise BroadQaExternalDataError("v8 roster license identity 漂移")
        license_files = tuple(
            _validate_blob(item, label="license")
            for item in license_value.get("files", ()))
        locale_files = tuple(
            _validate_blob(item, label="locale")
            for item in candidate.get("locale_files", ()))
        all_paths = [str(item["relative_path"])
                     for item in license_files + locale_files]
        if (not license_files or len(all_paths) != len(set(all_paths))
                or license_value["primary_bytes"]
                != license_files[0]["bytes"]):
            raise BroadQaExternalDataError("v8 roster file inventory 漂移")
        pair_count = _locale_pair_count(locale_files)
        identity = {
            "commit": commit,
            "repository": repository,
            "root_tree": tree,
            "source_family": family,
            "target_scope": NORMALIZATION_RECOVERY_V8_SOURCE_ROSTER_SCOPE,
        }
        records.append({
            **identity,
            "commit_date": date,
            "family_independence_group": group,
            "format_version": 1,
            "license": {
                "expression": license_value["expression"],
                "files": list(license_files),
                "primary_bytes": license_value["primary_bytes"],
                "primary_sha256": license_value["primary_sha256"],
            },
            "locale_blob_content_read_count": 0,
            "locale_file_count": len(locale_files),
            "locale_files": list(locale_files),
            "locale_pair_count": pair_count,
            "official_source_binding": candidate[
                "official_source_binding"],
            "parser_identity": candidate["format"],
            "record_id": _record_id(identity),
            "record_kind": (
                NORMALIZATION_RECOVERY_V8_SOURCE_ROSTER_RECORD_KIND),
            "selection_status": "SELECTED_TREE_LICENSE_PATH_FROZEN",
            "source_policy_scope": candidate["source_policy_scope"],
            "surface_published": 0,
        })
        families.add(str(family))
        groups.add(str(group))
        repositories.add(str(repository))
        total_locale_bytes += sum(int(item["bytes"]) for item in locale_files)
        total_locale_files += len(locale_files)
        total_license_files += len(license_files)
        total_pairs += pair_count
    records.sort(key=lambda item: str(item["source_family"]))
    if len(records) < 2:
        raise BroadQaExternalDataError("v8 roster 独立 TRAIN family 不足")
    return tuple(records), {
        "family_independence_group_count": len(groups),
        "locale_blob_content_read_count": 0,
        "locale_file_count": total_locale_files,
        "locale_pair_count": total_pairs,
        "locale_total_bytes": total_locale_bytes,
        "license_file_count": total_license_files,
        "official_source_bound_family_count": sum(
            item["official_source_binding"] in {
                "GETTEXT_MSGID", "QT_TS_SOURCE_ELEMENT"}
            for item in records),
        "selected_source_family_count": len(records),
        "surface_published": 0,
    }


__all__ = [
    "NORMALIZATION_RECOVERY_V8_SOURCE_ROSTER_CENSUS_KIND",
    "NORMALIZATION_RECOVERY_V8_SOURCE_ROSTER_RECORD_KIND",
    "NORMALIZATION_RECOVERY_V8_SOURCE_ROSTER_SCOPE",
    "derive_normalization_recovery_v8_source_roster",
]
