"""从冻结roster与已核验blob派生单个v8 source family records。"""
from __future__ import annotations

from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_gettext_source_records import (
    derive_normalization_recovery_v8_gettext_source_records,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_structured_source_records import (
    derive_normalization_recovery_v8_qt_ts_source_records,
)


def _qt_pair_spec(
        *,
        domain: str,
        hans_language: str,
        hans_path: str,
        hant_path: str,
        ) -> tuple[dict[str, object], ...]:
    """构造一个固定Qt TS domain pair spec。"""
    return ({
        "domain": domain,
        "zh_Hans": {
            "expected_language": hans_language,
            "relative_path": hans_path,
        },
        "zh_Hant": {
            "expected_language": "zh_TW",
            "relative_path": hant_path,
        },
    },)


def _stellarium_specs(
        locale_paths: tuple[str, ...],
        ) -> tuple[dict[str, object], ...]:
    """从冻结路径重建11个gettext domain pair spec。"""
    domains = {}
    for path in locale_paths:
        value = Path(path)
        if (len(value.parts) != 3 or value.parts[0] != "po"
                or value.name not in {"zh_CN.po", "zh_TW.po"}):
            raise BroadQaExternalDataError(
                "v8 Stellarium locale path 漂移")
        domains.setdefault(value.parent.name, {})[value.name] = path
    if (len(domains) != 11 or any(
            set(paths) != {"zh_CN.po", "zh_TW.po"}
            for paths in domains.values())):
        raise BroadQaExternalDataError(
            "v8 Stellarium domain roster 漂移")
    return tuple({
        "domain": domain,
        "zh_Hans": {
            "expected_language": "zh_CN",
            "relative_path": paths["zh_CN.po"],
        },
        "zh_Hant": {
            "expected_language": "zh_TW",
            "relative_path": paths["zh_TW.po"],
        },
    } for domain, paths in sorted(domains.items()))


def derive_normalization_recovery_v8_source_family_records(
        roster_record: dict[str, object],
        payloads: dict[str, bytes],
        ) -> tuple[
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            dict[str, object],
        ]:
    """按family固定合同调用共享Qt/gettext parser并返回完整records。"""
    family = str(roster_record.get("source_family"))
    license_value = roster_record.get("license")
    locale_files = roster_record.get("locale_files")
    policy = roster_record.get("source_policy_scope")
    if (not isinstance(license_value, dict)
            or not isinstance(license_value.get("expression"), str)
            or not isinstance(locale_files, list)
            or not isinstance(policy, str) or not policy):
        raise BroadQaExternalDataError(
            "v8 source family roster contract 漂移")
    locale_paths = tuple(str(item.get("relative_path"))
                         for item in locale_files if isinstance(item, dict))
    if (len(locale_paths) != len(locale_files)
            or any(path not in payloads for path in locale_paths)):
        raise BroadQaExternalDataError(
            "v8 source family locale payload inventory 漂移")
    files = {path: payloads[path] for path in locale_paths}
    if family == "BITCOIN_CORE_PROJECT":
        specs = _qt_pair_spec(
            domain="bitcoin",
            hans_language="zh_CN",
            hans_path="src/qt/locale/bitcoin_zh_CN.ts",
            hant_path="src/qt/locale/bitcoin_zh_TW.ts",
        )
        derive = derive_normalization_recovery_v8_qt_ts_source_records
    elif family == "QBITTORRENT_PROJECT":
        specs = _qt_pair_spec(
            domain="qbittorrent",
            hans_language="zh",
            hans_path="src/lang/qbittorrent_zh_CN.ts",
            hant_path="src/lang/qbittorrent_zh_TW.ts",
        )
        derive = derive_normalization_recovery_v8_qt_ts_source_records
    elif family == "KEEPASSXC_PROJECT":
        specs = _qt_pair_spec(
            domain="keepassxc",
            hans_language="zh_CN",
            hans_path="share/translations/keepassxc_zh_CN.ts",
            hant_path="share/translations/keepassxc_zh_TW.ts",
        )
        derive = derive_normalization_recovery_v8_qt_ts_source_records
    elif family == "STELLARIUM_PROJECT":
        specs = _stellarium_specs(locale_paths)
        derive = derive_normalization_recovery_v8_gettext_source_records
    else:
        raise BroadQaExternalDataError(
            "v8 source family roster 未支持")
    return derive(
        source_family=family,
        source_policy_scope=policy,
        license_expression=license_value["expression"],
        pair_specs=specs,
        files=files,
    )


__all__ = [
    "derive_normalization_recovery_v8_source_family_records",
]
