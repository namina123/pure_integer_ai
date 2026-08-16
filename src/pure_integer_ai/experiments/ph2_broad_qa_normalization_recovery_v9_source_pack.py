"""发布并严格回读 recovery-v9 GIMP 自包含 held-out source pack。"""
from __future__ import annotations

from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import stat
import zipfile

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_localization_structure import (
    git_blob_sha1,
    localization_record_id,
    localization_structure_token_category,
    localization_structure_tokens,
    read_exact_localization_zip,
    sha256_hex,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v9_gettext_source_records import (
    derive_normalization_recovery_v9_gettext_source_records,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v9_source_content_audit import (
    read_normalization_recovery_v9_source_content_aggregate,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v9_source_roster import (
    read_normalization_recovery_v9_source_roster,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    canonical_json_line,
)


NORMALIZATION_RECOVERY_V9_SOURCE_PACK_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V9_GIMP_SOURCE_PACK_V1")
NORMALIZATION_RECOVERY_V9_SOURCE_PACK_STATUS = (
    "GIMP_RAW_AND_LABEL_FREE_IDENTITY_FROZEN_NOT_FORMAL")
V9_GIMP_SOURCE_FILE_RECORD_KIND = "V9_GIMP_SOURCE_FILE_V1"
V9_GIMP_LABEL_FREE_IDENTITY_RECORD_KIND = (
    "V9_GIMP_LABEL_FREE_PAIR_IDENTITY_V1")
V9_GIMP_RUNTIME_SHAPE_RECORD_KIND = "V9_GIMP_LABEL_BLIND_RUNTIME_SHAPE_V1"
V9_GIMP_SOURCE_CENSUS_RECORD_KIND = "V9_GIMP_SOURCE_PACK_CENSUS_V1"

V9_SOURCE_ROSTER_MANIFEST_SHA256 = (
    "fd036c301ed901a861c7e58b62359b30dc2ed98a9f836afc76453730112f92d8")
V9_SOURCE_CONTENT_MANIFEST_SHA256 = (
    "7e8066bcae5852965c76b50d0cbc0851ef60fe7b90824ac95612913f22029331")

_ARCHIVE_NAME = "gimp-78fc57122afa94d3-zh-raw-v1.zip"
_OUTPUT_FILES = (
    (_ARCHIVE_NAME, "GIMP_FIXED_RAW_ARCHIVE", 0),
    ("source-files.jsonl", "GIMP_SOURCE_FILES", None),
    ("pair-identities.jsonl", "GIMP_LABEL_FREE_PAIR_IDENTITIES", None),
    ("runtime-shapes.jsonl", "GIMP_LABEL_BLIND_RUNTIME_SHAPES", None),
    ("source-census.jsonl", "GIMP_SOURCE_PACK_CENSUS", 1),
)

_TOKEN_REPRESENTATIVES = {
    "BBCODE_CLOSE": "[/b]",
    "BBCODE_OPEN": "[b]",
    "BRACE_PLACEHOLDER": "{x}",
    "CODE_FENCE": "`",
    "DOLLAR_BRACKET": "$[x]",
    "DOLLAR_PAREN": "$(x)",
    "ENTITY": "&x;",
    "ESCAPE": "\\n",
    "HTML_CLOSE": "</b>",
    "HTML_OPEN": "<b>",
    "HTML_SELF": "<b/>",
    "OTHER_STRUCTURE": "%s",
    "PERCENT_PLACEHOLDER": "%s",
}


def _sha256(payload: bytes) -> str:
    """返回文件、artifact 或规范值 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _require_k_root(value: str | Path) -> Path:
    """要求显式 run root 位于已存在 K 盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError("v9 source pack run root 必须在K盘")
    return root


def _within(root: Path, value: str | Path, *, label: str) -> Path:
    """解析并限制输入输出仍位于本次 K 盘 run root。"""
    path = Path(value).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise BroadQaExternalDataError(
            f"v9 source pack {label} 越出run root") from error
    return path


def _overlap(left: Path, right: Path) -> bool:
    """判断两个 artifact/input 根是否相同或互为祖先。"""
    try:
        left.relative_to(right)
        return True
    except ValueError:
        pass
    try:
        right.relative_to(left)
        return True
    except ValueError:
        return False


def _read_exact_root(
        root: Path,
        items: list[dict[str, object]],
        *,
        label: str,
        require_sha256: bool,
        ) -> dict[str, bytes]:
    """按 roster 精确读取一个物理 root 并拒绝额外文件。"""
    expected = {str(item.get("relative_path")): item
                for item in items if isinstance(item, dict)}
    physical = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*") if path.is_file()
    }
    if len(expected) != len(items) or physical != set(expected):
        raise BroadQaExternalDataError(
            f"v9 source pack {label} physical inventory 漂移")
    payloads = {}
    for relative, item in expected.items():
        path = (root / Path(relative)).resolve()
        try:
            path.relative_to(root)
            payload = path.read_bytes()
        except (ValueError, OSError) as error:
            raise BroadQaExternalDataError(
                f"v9 source pack {label} blob 不可读") from error
        if (len(payload) != item.get("bytes")
                or git_blob_sha1(payload) != item.get("git_blob_sha1")
                or (require_sha256
                    and sha256_hex(payload) != item.get("sha256"))):
            raise BroadQaExternalDataError(
                f"v9 source pack {label} blob identity 漂移")
        payloads[relative] = payload
    return payloads


def _archive_payload(files: dict[str, bytes]) -> bytes:
    """以固定1980 UTC、stored compression构造确定性 raw ZIP。"""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, payload)
    return buffer.getvalue()


def _archive_files(
        payload: bytes,
        expected_paths: tuple[str, ...],
        ) -> dict[str, bytes]:
    """严格读取固定18文件raw ZIP。"""
    return read_exact_localization_zip(
        payload,
        expected_files=expected_paths,
        label="v9 GIMP source pack",
        member_count_max=len(expected_paths),
        uncompressed_bytes_max=4_000_000,
    )


def _pair_specs(record: dict[str, object]) -> tuple[dict[str, object], ...]:
    """从 roster 派生八 domain parser specs。"""
    by_domain: dict[str, dict[str, str]] = {}
    for item in record["locale_files"]:
        domain = str(item["domain"])
        locale = str(item["locale"])
        path = str(item["relative_path"])
        if locale in by_domain.setdefault(domain, {}):
            raise BroadQaExternalDataError("v9 source pack locale 重复")
        by_domain[domain][locale] = path
    if len(by_domain) != 8 or any(
            set(values) != {"zh_CN", "zh_TW"}
            for values in by_domain.values()):
        raise BroadQaExternalDataError("v9 source pack domain pair 漂移")
    return tuple({
        "domain": domain,
        "zh_Hans": {
            "expected_language": "zh_CN",
            "relative_path": values["zh_CN"],
        },
        "zh_Hant": {
            "expected_language": "zh_TW",
            "relative_path": values["zh_TW"],
        },
    } for domain, values in sorted(by_domain.items()))


def _license_source_records(
        record: dict[str, object],
        payloads: dict[str, bytes],
        ) -> tuple[dict[str, object], ...]:
    """为两份根许可形成来源记录。"""
    values = []
    for item in record["license"]["files"]:
        relative = str(item["relative_path"])
        payload = payloads[relative]
        identity = {
            "git_blob_sha1": git_blob_sha1(payload),
            "relative_path": relative,
            "sha256": sha256_hex(payload),
        }
        values.append({
            **identity,
            "bytes": len(payload),
            "file_id": localization_record_id(identity),
            "format_version": 1,
            "license_expression": record["license"]["expression"],
            "record_kind": V9_GIMP_SOURCE_FILE_RECORD_KIND,
            "role": "LICENSE_TEXT",
        })
    return tuple(values)


def _identity_records(
        pairs: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """去除双侧translation，只保留正式分母source identity。"""
    values = []
    for item in pairs:
        values.append({
            "format_version": 1,
            "license_expression": item["license_expression"],
            "pair_id": item["pair_id"],
            "record_kind": V9_GIMP_LABEL_FREE_IDENTITY_RECORD_KIND,
            "source_family": item["source_family"],
            "source_identity": item["source_identity"],
            "source_identity_sha256": item["source_identity_sha256"],
            "source_policy_scope": item["source_policy_scope"],
        })
    return tuple(values)


def _synthetic_input(
        scalar_count: int,
        categories: tuple[str, ...],
        ) -> str:
    """按长度与结构类别构造不含真实翻译 surface 的输入。"""
    tokens = []
    previous = ""
    for category in categories:
        token = _TOKEN_REPRESENTATIVES.get(category)
        if token is None:
            raise BroadQaExternalDataError(
                "v9 source pack runtime shape category 未冻结")
        if previous == "CODE_FENCE" and category == "CODE_FENCE":
            tokens.append("測")
        tokens.append(token)
        previous = category
    base = "".join(tokens)
    if len(base) > scalar_count:
        raise BroadQaExternalDataError(
            "v9 source pack runtime shape 长度无法保持")
    value = base + ("測" * (scalar_count - len(base)))
    derived = tuple(localization_structure_token_category(token)
                    for token in localization_structure_tokens(value))
    if derived != categories or len(value) != scalar_count:
        raise BroadQaExternalDataError(
            "v9 source pack runtime shape 结构无法重建")
    return value


def _runtime_shapes(
        pairs: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """生成与正式分母同长度/结构分布的无真实surface query。"""
    values = []
    for ordinal, item in enumerate(pairs):
        input_text = str(item["zh_hant"]["msgstr"])
        source_text = str(item["official_source_text"])
        categories = tuple(
            localization_structure_token_category(str(token))
            for token in item["zh_hant_structure_tokens"])
        synthetic = _synthetic_input(len(input_text), categories)
        query = {
            "input_text": synthetic,
            "official_source_text": "S" * len(source_text),
            "structure_tokens": list(localization_structure_tokens(synthetic)),
        }
        identity = {
            "ordinal": ordinal,
            "query": query,
            "record_kind": V9_GIMP_RUNTIME_SHAPE_RECORD_KIND,
        }
        values.append({
            **identity,
            "format_version": 1,
            "input_scalar_count": len(input_text),
            "official_source_scalar_count": len(source_text),
            "shape_id": localization_record_id(identity),
            "structure_category_sequence": list(categories),
            "structure_token_count": len(categories),
            "synthetic_surface_only": 1,
        })
    return tuple(values)


def _histogram(values: list[int]) -> dict[str, int]:
    """形成键排序稳定的整数 histogram。"""
    return {str(key): count for key, count in sorted(Counter(values).items())}


def _census_record(
        *,
        parser_summary: dict[str, object],
        source_files: tuple[dict[str, object], ...],
        identities: tuple[dict[str, object], ...],
        shapes: tuple[dict[str, object], ...],
        ) -> dict[str, object]:
    """形成source pack分母与label-blind runtime分布。"""
    category_counts = Counter(
        category for item in shapes
        for category in item["structure_category_sequence"])
    return {
        "format_version": 1,
        "input_scalar_length_histogram": _histogram([
            int(item["input_scalar_count"]) for item in shapes]),
        "label_free_identity_count": len(identities),
        "label_or_translation_surface_published": 0,
        "official_source_scalar_length_histogram": _histogram([
            int(item["official_source_scalar_count"]) for item in shapes]),
        "parser_summary": parser_summary,
        "record_kind": V9_GIMP_SOURCE_CENSUS_RECORD_KIND,
        "runtime_shape_count": len(shapes),
        "source_file_count": len(source_files),
        "structure_category_histogram": dict(sorted(category_counts.items())),
        "structure_token_count_histogram": _histogram([
            int(item["structure_token_count"]) for item in shapes]),
        "synthetic_runtime_surface_only": 1,
    }


def _derive_material(
        *,
        roster: tuple[dict[str, object], ...],
        content_records: tuple[dict[str, object], ...],
        payloads: dict[str, bytes],
        ) -> tuple[dict[str, object], dict[str, bytes]]:
    """由冻结raw、roster和content aggregate重派生完整pack。"""
    if len(roster) != 1 or len(content_records) != 1:
        raise BroadQaExternalDataError("v9 source pack predecessor 分母漂移")
    record = roster[0]
    content = content_records[0]
    locale_paths = {str(item["relative_path"])
                    for item in record["locale_files"]}
    locale_payloads = {path: payloads[path] for path in locale_paths}
    locale_files, pairs, parser_summary = (
        derive_normalization_recovery_v9_gettext_source_records(
            source_family="GIMP_PROJECT",
            source_policy_scope=str(record["source_policy_scope"]),
            license_expression="GPL-3.0-or-later",
            pair_specs=_pair_specs(record),
            files=locale_payloads,
        ))
    parser_file_sha = _sha256(canonical_json_bytes(locale_files))
    identities = _identity_records(pairs)
    identity_sha = _sha256(canonical_json_bytes([{
        "pair_id": item["pair_id"],
        "source_identity": item["source_identity"],
    } for item in identities]))
    if (content.get("content_outcome")
            != "PASS_NONZERO_ACTIVE_COMMON_PAIR"
            or content.get("parser_summary") != parser_summary
            or content.get("transient_pair_count") != len(pairs)
            or content.get("locale_file_commitment_sha256")
            != parser_file_sha
            or content.get("pair_identity_roster_sha256") != identity_sha):
        raise BroadQaExternalDataError(
            "v9 source pack content aggregate 重派生漂移")
    license_paths = {str(item["relative_path"])
                     for item in record["license"]["files"]}
    license_payloads = {path: payloads[path] for path in license_paths}
    source_files = tuple(sorted(
        _license_source_records(record, license_payloads) + locale_files,
        key=lambda item: str(item["relative_path"])))
    shapes = _runtime_shapes(pairs)
    census = _census_record(
        parser_summary=parser_summary,
        source_files=source_files,
        identities=identities,
        shapes=shapes,
    )
    archive = _archive_payload(payloads)
    output_payloads = {
        _ARCHIVE_NAME: archive,
        "source-files.jsonl": b"".join(
            canonical_json_line(item) for item in source_files),
        "pair-identities.jsonl": b"".join(
            canonical_json_line(item) for item in identities),
        "runtime-shapes.jsonl": b"".join(
            canonical_json_line(item) for item in shapes),
        "source-census.jsonl": canonical_json_line(census),
    }
    summary = {
        "label_free_identity_count": len(identities),
        "label_or_translation_surface_published": 0,
        "plain_pair_count": parser_summary["plain_pair_count"],
        "raw_archive_bytes": len(archive),
        "runtime_shape_count": len(shapes),
        "source_file_count": len(source_files),
        "synthetic_runtime_surface_only": 1,
    }
    manifest = {
        "artifact_kind": NORMALIZATION_RECOVERY_V9_SOURCE_PACK_KIND,
        "files": [
            _artifact_payload(
                name,
                role=role,
                payload=output_payloads[name],
                count=(fixed_count if fixed_count is not None else (
                    len(source_files) if name == "source-files.jsonl" else
                    len(identities) if name == "pair-identities.jsonl" else
                    len(shapes))),
            )
            for name, role, fixed_count in _OUTPUT_FILES
        ],
        "format_version": 1,
        "individual_label_print_count": 0,
        "inputs": {
            "source_content_manifest_sha256": (
                V9_SOURCE_CONTENT_MANIFEST_SHA256),
            "source_roster_manifest_sha256": V9_SOURCE_ROSTER_MANIFEST_SHA256,
        },
        "label_or_translation_surface_published": 0,
        "mastery_claimed": 0,
        "production_enabled": 0,
        "runtime_program_published": 0,
        "status": NORMALIZATION_RECOVERY_V9_SOURCE_PACK_STATUS,
        "summary": summary,
        "teacher_api_llm_call_count": 0,
    }
    return manifest, output_payloads


def _artifact_payload(
        name: str,
        *,
        role: str,
        payload: bytes,
        count: int,
        ) -> dict[str, object]:
    """构造一个source-pack文件 commitment。"""
    return {
        "bytes": len(payload),
        "record_count": count,
        "relative_path": name,
        "role": role,
        "sha256": _sha256(payload),
    }


def _predecessors(
        roster_dir: Path,
        content_dir: Path,
        ) -> tuple[
            tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    """严格回读roster与aggregate content，不重开旧raw。"""
    _roster_manifest, roster_outputs = (
        read_normalization_recovery_v9_source_roster(
            roster_dir,
            expected_manifest_sha256=V9_SOURCE_ROSTER_MANIFEST_SHA256,
        ))
    _content_manifest, content_outputs = (
        read_normalization_recovery_v9_source_content_aggregate(
            content_dir,
            expected_manifest_sha256=V9_SOURCE_CONTENT_MANIFEST_SHA256,
        ))
    return (
        roster_outputs["source-roster.jsonl"],
        content_outputs["source-content.jsonl"],
    )


def _root_payloads(
        record: dict[str, object],
        *,
        license_root: Path,
        locale_root: Path,
        ) -> dict[str, bytes]:
    """从两个固定root读取18份blob并组合为单一archive inventory。"""
    license_payloads = _read_exact_root(
        license_root,
        record["license"]["files"],
        label="license",
        require_sha256=True,
    )
    locale_payloads = _read_exact_root(
        locale_root,
        record["locale_files"],
        label="locale",
        require_sha256=False,
    )
    overlap = set(license_payloads).intersection(locale_payloads)
    if overlap:
        raise BroadQaExternalDataError("v9 source pack raw path 冲突")
    return {**license_payloads, **locale_payloads}


def publish_normalization_recovery_v9_source_pack(
        *,
        run_root: str | Path,
        roster_dir: str | Path,
        content_audit_dir: str | Path,
        license_root: str | Path,
        locale_root: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """不可覆盖发布自包含 GIMP raw/identity/runtime-shape pack。"""
    root = _require_k_root(run_root)
    paths = tuple(_within(root, value, label=str(index)) for index, value in
                  enumerate((roster_dir, content_audit_dir, license_root,
                             locale_root, target_dir)))
    roster, content, license_path, locale_path, target = paths
    if (target.exists() or any(not path.is_dir() for path in paths[:-1])
            or any(_overlap(left, right)
                   for index, left in enumerate(paths)
                   for right in paths[index + 1:])):
        raise BroadQaExternalDataError("v9 source pack input/target path 非法")
    roster_records, content_records = _predecessors(roster, content)
    payloads = _root_payloads(
        roster_records[0],
        license_root=license_path,
        locale_root=locale_path,
    )
    manifest, outputs = _derive_material(
        roster=roster_records,
        content_records=content_records,
        payloads=payloads,
    )
    target.mkdir()
    for name, _role, _count in _OUTPUT_FILES:
        with (target / name).open("xb") as handle:
            handle.write(outputs[name])
    path = target / "manifest.json"
    with path.open("xb") as handle:
        handle.write(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": _sha256(path.read_bytes())}


def read_normalization_recovery_v9_source_pack(
        source_dir: str | Path,
        *,
        roster_dir: str | Path,
        content_audit_dir: str | Path,
        expected_manifest_sha256: str,
        ) -> tuple[
            dict[str, object], dict[str, tuple[dict[str, object], ...]]]:
    """从pack内raw严格重派生全部label-free输出。"""
    root = Path(source_dir).resolve()
    expected_names = {
        "manifest.json", *[name for name, _role, _count in _OUTPUT_FILES]}
    try:
        physical_names = {item.name for item in root.iterdir()}
        encoded = (root / "manifest.json").read_bytes()
        stored = json.loads(encoded)
        archive = (root / _ARCHIVE_NAME).read_bytes()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError("v9 source pack 不可读") from error
    if (physical_names != expected_names
            or _sha256(encoded) != expected_manifest_sha256
            or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded):
        raise BroadQaExternalDataError("v9 source pack manifest identity 漂移")
    roster_records, content_records = _predecessors(
        Path(roster_dir).resolve(), Path(content_audit_dir).resolve())
    record = roster_records[0]
    expected_paths = tuple(sorted(
        str(item["relative_path"])
        for item in record["license"]["files"] + record["locale_files"]))
    payloads = _archive_files(archive, expected_paths)
    for item in record["license"]["files"] + record["locale_files"]:
        payload = payloads[str(item["relative_path"])]
        if (len(payload) != item["bytes"]
                or git_blob_sha1(payload) != item["git_blob_sha1"]
                or ("sha256" in item
                    and sha256_hex(payload) != item["sha256"])):
            raise BroadQaExternalDataError(
                "v9 source pack archive blob identity 漂移")
    expected, outputs = _derive_material(
        roster=roster_records,
        content_records=content_records,
        payloads=payloads,
    )
    if stored != expected:
        raise BroadQaExternalDataError("v9 source pack fields 漂移")
    parsed_outputs = {}
    by_name = {str(item["relative_path"]): item for item in stored["files"]}
    for name, role, fixed_count in _OUTPUT_FILES:
        try:
            payload = (root / name).read_bytes()
        except OSError as error:
            raise BroadQaExternalDataError(
                f"v9 source pack {name} 不可读") from error
        if payload != outputs[name]:
            raise BroadQaExternalDataError(
                f"v9 source pack {name} 重派生漂移")
        if name.endswith(".jsonl"):
            values = tuple(json.loads(line) for line in payload.splitlines())
            parsed_outputs[name] = values
            count = len(values)
        else:
            count = 0
        if by_name.get(name) != _artifact_payload(
                name, role=role, payload=payload,
                count=fixed_count if fixed_count is not None else count):
            raise BroadQaExternalDataError(
                f"v9 source pack {name} commitment 漂移")
    return {**stored, "manifest_sha256": _sha256(encoded)}, parsed_outputs


def materialize_normalization_recovery_v9_source_pairs_after_guard(
        source_dir: str | Path, *,
        expected_manifest_sha256: str,
        guard_consumed: int,
        ) -> tuple[dict[str, object], tuple[dict[str, object], ...],
                   dict[str, object]]:
    """只在formal guard后从自包含raw重建并核对完整GIMP pairs。"""
    if type(guard_consumed) is not int or guard_consumed != 1:
        raise BroadQaExternalDataError(
            "v9 GIMP source pairs只能在formal guard后物化")
    root = Path(source_dir).resolve()
    expected_names = {
        "manifest.json", *[name for name, _role, _count in _OUTPUT_FILES]}
    try:
        physical = {item.name for item in root.iterdir()}
        encoded = (root / "manifest.json").read_bytes()
        stored = json.loads(encoded)
        payloads = {name: (root / name).read_bytes()
                    for name, _role, _count in _OUTPUT_FILES}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "v9 GIMP guard后source pack不可读") from error
    if (physical != expected_names
            or _sha256(encoded) != expected_manifest_sha256
            or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded):
        raise BroadQaExternalDataError(
            "v9 GIMP guard后source manifest漂移")
    by_name = {str(item.get("relative_path")): item
               for item in stored.get("files", [])
               if isinstance(item, dict)}
    parsed = {}
    for name, role, fixed_count in _OUTPUT_FILES:
        payload = payloads[name]
        if name.endswith(".jsonl"):
            try:
                values = tuple(json.loads(line) for line in payload.splitlines())
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise BroadQaExternalDataError(
                    f"v9 GIMP guard后{name}不可读") from error
            if b"".join(canonical_json_line(item) for item in values) != payload:
                raise BroadQaExternalDataError(
                    f"v9 GIMP guard后{name}非规范")
            parsed[name] = values
            count = len(values)
        else:
            count = 0
        expected_count = fixed_count if fixed_count is not None else count
        if by_name.get(name) != _artifact_payload(
                name, role=role, payload=payload, count=expected_count):
            raise BroadQaExternalDataError(
                f"v9 GIMP guard后{name} commitment漂移")
    source_files = parsed["source-files.jsonl"]
    identities = parsed["pair-identities.jsonl"]
    census_values = parsed["source-census.jsonl"]
    stored_summary = stored.get("summary")
    expected_pair_count = stored_summary.get(
        "label_free_identity_count") if isinstance(stored_summary, dict) else None
    if (len(source_files) != 18
            or type(expected_pair_count) is not int or expected_pair_count <= 0
            or len(identities) != expected_pair_count
            or len(census_values) != 1):
        raise BroadQaExternalDataError("v9 GIMP guard后分母漂移")
    expected_paths = tuple(sorted(
        str(item["relative_path"]) for item in source_files))
    archive_files = _archive_files(payloads[_ARCHIVE_NAME], expected_paths)
    for item in source_files:
        payload = archive_files[str(item["relative_path"])]
        if (len(payload) != item.get("bytes")
                or git_blob_sha1(payload) != item.get("git_blob_sha1")
                or sha256_hex(payload) != item.get("sha256")):
            raise BroadQaExternalDataError(
                "v9 GIMP guard后archive file identity漂移")
    locale_records = tuple(item for item in source_files
                           if item.get("role")
                           == "TRANSLATION_WITH_OFFICIAL_SOURCE_GETTEXT_PO")
    by_domain: dict[str, dict[str, str]] = {}
    for item in locale_records:
        domain = str(item.get("domain"))
        role = str(item.get("locale_role"))
        relative = str(item.get("relative_path"))
        if role in by_domain.setdefault(domain, {}):
            raise BroadQaExternalDataError(
                "v9 GIMP guard后locale domain重复")
        by_domain[domain][role] = relative
    if len(by_domain) != 8 or any(
            set(values) != {"zh_Hans", "zh_Hant"}
            for values in by_domain.values()):
        raise BroadQaExternalDataError("v9 GIMP guard后locale roster漂移")
    specs = tuple({
        "domain": domain,
        "zh_Hans": {
            "expected_language": "zh_CN",
            "relative_path": values["zh_Hans"],
        },
        "zh_Hant": {
            "expected_language": "zh_TW",
            "relative_path": values["zh_Hant"],
        },
    } for domain, values in sorted(by_domain.items()))
    locale_payloads = {str(item["relative_path"]): archive_files[
        str(item["relative_path"])] for item in locale_records}
    derived_files, pairs, summary = (
        derive_normalization_recovery_v9_gettext_source_records(
            source_family="GIMP_PROJECT",
            source_policy_scope=str(identities[0]["source_policy_scope"]),
            license_expression="GPL-3.0-or-later",
            pair_specs=specs,
            files=locale_payloads,
        ))
    census = census_values[0]
    if (derived_files != locale_records
            or _identity_records(pairs) != identities
            or census.get("parser_summary") != summary
            or census.get("label_free_identity_count") != len(pairs)):
        raise BroadQaExternalDataError(
            "v9 GIMP guard后pair/identity/census重派生漂移")
    return ({**stored, "manifest_sha256": expected_manifest_sha256},
            pairs, summary)


__all__ = [
    "NORMALIZATION_RECOVERY_V9_SOURCE_PACK_KIND",
    "NORMALIZATION_RECOVERY_V9_SOURCE_PACK_STATUS",
    "V9_SOURCE_CONTENT_MANIFEST_SHA256",
    "V9_SOURCE_ROSTER_MANIFEST_SHA256",
    "publish_normalization_recovery_v9_source_pack",
    "read_normalization_recovery_v9_source_pack",
    "materialize_normalization_recovery_v9_source_pairs_after_guard",
]
