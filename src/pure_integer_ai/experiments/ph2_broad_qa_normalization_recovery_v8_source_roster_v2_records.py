"""派生 recovery-v8 source roster v2 replacement records。

v2只消费v1无表面roster与aggregate content结果：保留content PASS的
qBittorrent/Stellarium，永久淘汰零active的Bitcoin，并在未读locale内容时
加入固定KeePassXC tree/blob。它不读取任何source raw或translation surface。
"""
from __future__ import annotations

import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
)


NORMALIZATION_RECOVERY_V8_SOURCE_ROSTER_V2_RECORD_KIND = (
    "NORMALIZATION_RECOVERY_V8_TRAIN_SOURCE_ROSTER_RECORD_V2")
NORMALIZATION_RECOVERY_V8_SOURCE_ROSTER_V2_CENSUS_KIND = (
    "NORMALIZATION_RECOVERY_V8_TRAIN_SOURCE_ROSTER_CENSUS_V2")

_CONTENT_PASS = "PASS_NONZERO_ACTIVE_COMMON_PAIR"
_BITCOIN_REJECT = "REJECTED_ZERO_ACTIVE_COMMON_PAIR"
_INHERITED = ("QBITTORRENT_PROJECT", "STELLARIUM_PROJECT")


def _sha256(payload: bytes) -> str:
    """返回规范v2 roster identity。"""
    return hashlib.sha256(payload).hexdigest()


def _record_id(value: dict[str, object]) -> str:
    """从完整family/revision identity形成记录id。"""
    return _sha256(canonical_json_bytes(value))


def _blob(path: str, *, size: int, sha1: str) -> dict[str, object]:
    """构造一个未读内容的Git blob identity。"""
    return {"bytes": size, "git_blob_sha1": sha1, "relative_path": path}


def _keepassxc_record() -> dict[str, object]:
    """构造未读locale的KeePassXC replacement commitment。"""
    identity = {
        "commit": "0e1510d71ab63ce1edddb71257bce34a7cee2f0d",
        "repository": "https://github.com/keepassxreboot/keepassxc.git",
        "root_tree": "0c01585ce330ea3f15d2b365e2844b1d162e56d4",
        "source_family": "KEEPASSXC_PROJECT",
        "source_roster_revision": 2,
        "target_scope": "ZH_CN_CROSS_PRODUCT_SOURCE_CONDITIONED_TRANSFER_V8",
    }
    return {
        **identity,
        "commit_date": "2026-08-12T19:03:33Z",
        "content_feasibility_outcome": "NOT_READ_ROSTER_V2_REPLACEMENT",
        "family_independence_group": "KEEPASSXC_UPSTREAM",
        "format_version": 1,
        "license": {
            "expression": "GPL-2.0-only OR GPL-3.0-only",
            "files": [
                _blob(
                    "COPYING", size=14_251,
                    sha1="a00aaf28c357bc0b9319103c02d3fee891d2c673"),
                _blob(
                    "LICENSE.GPL-2", size=18_092,
                    sha1="d159169d1050894d3ea3b98e1c965c4058208fe1"),
                _blob(
                    "LICENSE.GPL-3", size=35_149,
                    sha1="f288702d2fa16d3cdf0035b15a9fcbc552cd88e7"),
            ],
            "primary_bytes": 14_251,
            "primary_sha256": (
                "f4cf558763d725e47a55da6f32735b3f0b3d69184870f921015b3c54b58bfb36"),
        },
        "locale_blob_content_read_count": 0,
        "locale_file_count": 2,
        "locale_files": [
            _blob(
                "share/translations/keepassxc_zh_CN.ts", size=385_277,
                sha1="449aa42335a7b9ddc992e16d8e4c9c9cb73f4844"),
            _blob(
                "share/translations/keepassxc_zh_TW.ts", size=385_886,
                sha1="8c13b313b2da39cf4384ce109f3bcc850bd6e1e8"),
        ],
        "locale_pair_count": 1,
        "official_source_binding": "QT_TS_SOURCE_ELEMENT",
        "parser_identity": "QT_TS_XML_V1",
        "predecessor_record_id": "",
        "record_id": _record_id(identity),
        "record_kind": NORMALIZATION_RECOVERY_V8_SOURCE_ROSTER_V2_RECORD_KIND,
        "selection_status": "SELECTED_V2_REPLACEMENT_TREE_LICENSE_PATH_FROZEN",
        "source_policy_scope": "KEEPASSXC_ZH_TW_TO_ZH_CN_FIXED_COMMIT_V1",
        "surface_published": 0,
    }


def derive_normalization_recovery_v8_source_roster_v2(
        *,
        v1_roster: tuple[dict[str, object], ...],
        content_records: tuple[dict[str, object], ...],
        ) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    """由v1/content aggregate派生两家继承加一家replacement的v2 roster。"""
    v1 = {str(item.get("source_family")): item for item in v1_roster
          if isinstance(item, dict)}
    content = {str(item.get("source_family")): item
               for item in content_records if isinstance(item, dict)}
    expected = {
        "BITCOIN_CORE_PROJECT", "QBITTORRENT_PROJECT", "STELLARIUM_PROJECT"}
    if (set(v1) != expected or set(content) != expected
            or content["BITCOIN_CORE_PROJECT"].get("content_outcome")
            != _BITCOIN_REJECT
            or any(content[family].get("content_outcome") != _CONTENT_PASS
                   for family in _INHERITED)):
        raise BroadQaExternalDataError(
            "v8 roster v2 predecessor/content outcome 漂移")
    records = []
    for family in _INHERITED:
        predecessor = v1[family]
        identity = {
            "commit": predecessor["commit"],
            "repository": predecessor["repository"],
            "root_tree": predecessor["root_tree"],
            "source_family": family,
            "source_roster_revision": 2,
            "target_scope": predecessor["target_scope"],
        }
        records.append({
            **predecessor,
            **identity,
            "content_feasibility_outcome": _CONTENT_PASS,
            "locale_blob_content_read_count": 1,
            "predecessor_record_id": predecessor["record_id"],
            "record_id": _record_id(identity),
            "record_kind": (
                NORMALIZATION_RECOVERY_V8_SOURCE_ROSTER_V2_RECORD_KIND),
            "selection_status": "INHERITED_V1_CONTENT_PASS",
        })
    records.append(_keepassxc_record())
    records.sort(key=lambda item: str(item["source_family"]))
    if (len(records) != 3
            or len({item["family_independence_group"] for item in records})
            != 3
            or len({item["repository"] for item in records}) != 3):
        raise BroadQaExternalDataError("v8 roster v2 independence 漂移")
    summary = {
        "content_pass_inherited_family_count": 2,
        "family_independence_group_count": 3,
        "locale_blob_content_read_family_count": 2,
        "locale_blob_content_unread_family_count": 1,
        "locale_file_count": sum(int(item["locale_file_count"])
                                 for item in records),
        "locale_pair_count": sum(int(item["locale_pair_count"])
                                 for item in records),
        "locale_total_bytes": sum(
            int(file["bytes"])
            for item in records for file in item["locale_files"]),
        "license_file_count": sum(len(item["license"]["files"])
                                  for item in records),
        "official_source_bound_family_count": 3,
        "rejected_predecessor_family_count": 1,
        "replacement_locale_blob_content_read_count": 0,
        "selected_source_family_count": 3,
        "surface_published": 0,
    }
    return tuple(records), summary


__all__ = [
    "NORMALIZATION_RECOVERY_V8_SOURCE_ROSTER_V2_CENSUS_KIND",
    "NORMALIZATION_RECOVERY_V8_SOURCE_ROSTER_V2_RECORD_KIND",
    "derive_normalization_recovery_v8_source_roster_v2",
]
