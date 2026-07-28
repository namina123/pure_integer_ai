"""ConceptNet 5.7.0 assertion 的许可分区、URI 和纯整数 weight adapter。"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    canonical_json_line,
)
from pure_integer_ai.experiments.ph2_raw_snapshot import sha256_path


SOURCE_KEY = "CONCEPTNET_5_7_0"
SNAPSHOT_ID = "5.7.0"
OFFICIAL_URL = (
    "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/"
    "conceptnet-assertions-5.7.0.csv.gz"
)
ADAPTER_VERSION = 1
PARSER_VERSION = 1

LICENSE_PARTITIONS = {
    "cc:by/4.0": "CC-BY-4.0",
    "cc:by-sa/4.0": "CC-BY-SA-4.0",
}
TARGET_RELATIONS = (
    "/r/Antonym",
    "/r/Causes",
    "/r/HasProperty",
    "/r/IsA",
    "/r/PartOf",
    "/r/Synonym",
)

ENDPOINT_CONCEPT = 1
ENDPOINT_EXTERNAL = 2

_DECIMAL_PATTERN = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?")


class _JsonDecimal(str):
    """区分 JSON 数字小数与 JSON string，同时保留原始十进制文本。"""


class ConceptNetAdapterError(RuntimeError):
    """ConceptNet 行、URI、metadata、许可、weight 或 gzip 非法。"""

    def __init__(self, code: str, message: str) -> None:
        """保存稳定 anomaly code，并保留面向审计的中文错误。"""
        super().__init__(message)
        self.code = code


def decimal_text_to_ratio(value: str) -> tuple[int, int]:
    """把正十进制原文约成最简整数分子/分母，绝不经过 binary float。"""
    if not isinstance(value, str) or not _DECIMAL_PATTERN.fullmatch(value):
        raise ConceptNetAdapterError("BAD_WEIGHT", "ConceptNet weight 非十进制")
    whole, dot, fraction = value.partition(".")
    denominator = 10 ** len(fraction) if dot else 1
    numerator = int(whole + fraction) if dot else int(whole)
    if numerator <= 0:
        raise ConceptNetAdapterError("BAD_WEIGHT", "ConceptNet weight 必须为正")
    divisor = math.gcd(numerator, denominator)
    return numerator // divisor, denominator // divisor


def _object_without_duplicates(
        pairs: list[tuple[str, Any]],
        ) -> dict[str, Any]:
    """构造 JSON object 时拒绝重复 key，防止许可或 weight 被覆盖。"""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ConceptNetAdapterError(
                "BAD_METADATA", "ConceptNet metadata key 重复")
        result[key] = value
    return result


def _metadata(value: str) -> dict[str, Any]:
    """严格解析 metadata，并把所有 JSON 小数保留为原始十进制文本。"""
    try:
        result = json.loads(
            value,
            parse_float=_JsonDecimal,
            object_pairs_hook=_object_without_duplicates,
        )
    except (ConceptNetAdapterError, json.JSONDecodeError) as error:
        raise ConceptNetAdapterError(
            "BAD_METADATA", "ConceptNet metadata JSON 非法") from error
    if not isinstance(result, dict):
        raise ConceptNetAdapterError(
            "BAD_METADATA", "ConceptNet metadata 必须是 object")
    return result


def _freeze_json(value: Any) -> Any:
    """把已解析 JSON 转成可哈希整数/文本 tuple，用于 source 去重。"""
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze_json(item))
                            for key, item in value.items()))
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    if value is None or type(value) in {bool, int, str, _JsonDecimal}:
        return value
    raise ConceptNetAdapterError(
        "BAD_METADATA", "ConceptNet metadata 含非规范 JSON 值")


def _validated_sources(value: Any) -> list[dict[str, Any]]:
    """校验 sources 为非空、无重复的 JSON object 列表。"""
    if (not isinstance(value, list) or not value
            or any(not isinstance(item, dict) or not item for item in value)):
        raise ConceptNetAdapterError("BAD_SOURCES", "ConceptNet sources 缺失或非法")
    identities = [_freeze_json(item) for item in value]
    if len(set(identities)) != len(identities):
        raise ConceptNetAdapterError("BAD_SOURCES", "ConceptNet source object 重复")
    return value


def _endpoint_language(value: str) -> str:
    """轻量校验 endpoint，并只返回 concept language 或空文本。"""
    if (not value or value.strip() != value
            or any(character in value for character in "\t\r\n")):
        raise ConceptNetAdapterError("BAD_URI", "ConceptNet endpoint URI 非法")
    if value.startswith("/c/"):
        parts = value.split("/")
        suffix = parts[4:]
        if suffix and suffix[-1] == "":
            suffix = suffix[:-1]
        if (len(parts) < 4 or parts[0] or parts[1] != "c"
                or not parts[2] or not parts[3]
                or any(not item for item in suffix)):
            raise ConceptNetAdapterError("BAD_URI", "ConceptNet concept URI 非法")
        return parts[2]
    if not (value.startswith("/") or value.startswith("https://")
            or value.startswith("http://")):
        raise ConceptNetAdapterError("BAD_URI", "ConceptNet external URI 非法")
    return ""


def _assertion_component(value: str) -> str:
    """按 ConceptNet assertion URI 语法补前导/终止 slash。"""
    component = value if value.startswith("/") else "/" + value
    return component if component.endswith("/") else component + "/"


def _assertion_fields(
        line: str,
        ) -> tuple[
            str, str, str, str, str, str, str,
            list[dict[str, Any]], str, int, int,
        ]:
    """共享解析热路径，返回 rich parser 与全量 scanner 所需字段。"""
    if (not isinstance(line, str) or not line or line.strip() != line
            or "\r" in line or "\n" in line):
        raise ConceptNetAdapterError("BAD_ROW", "ConceptNet row 空或有首尾空白")
    columns = line.split("\t")
    if len(columns) != 5:
        raise ConceptNetAdapterError("BAD_ROW", "ConceptNet row 必须恰有五列")
    assertion_uri, relation, start_text, end_text, metadata_text = columns
    if (not relation.startswith("/r/") or len(relation) <= 3
            or relation.strip() != relation):
        raise ConceptNetAdapterError("BAD_RELATION", "ConceptNet relation 非法")
    start_language = _endpoint_language(start_text)
    end_language = _endpoint_language(end_text)
    expected_uri = "/a/[" + ",".join((
        _assertion_component(relation),
        _assertion_component(start_text),
        _assertion_component(end_text),
    )) + "]"
    if assertion_uri != expected_uri:
        raise ConceptNetAdapterError(
            "BAD_ASSERTION_URI", "ConceptNet assertion URI 与三元组不一致")
    metadata = _metadata(metadata_text)
    dataset = metadata.get("dataset")
    license_text = metadata.get("license")
    if not isinstance(dataset, str) or not dataset.startswith("/d/"):
        raise ConceptNetAdapterError("BAD_DATASET", "ConceptNet dataset 缺失或非法")
    if not isinstance(license_text, str) or license_text not in LICENSE_PARTITIONS:
        raise ConceptNetAdapterError(
            "BAD_LICENSE", "ConceptNet assertion 许可不在 allowlist")
    sources = _validated_sources(metadata.get("sources"))
    parsed_weight = metadata.get("weight")
    if type(parsed_weight) is int:
        weight_text = str(parsed_weight)
    elif type(parsed_weight) is _JsonDecimal:
        weight_text = str(parsed_weight)
    else:
        raise ConceptNetAdapterError(
            "BAD_WEIGHT", "ConceptNet weight 必须是 JSON number")
    numerator, denominator = decimal_text_to_ratio(weight_text)
    return (
        assertion_uri,
        relation,
        start_text,
        end_text,
        start_language,
        end_language,
        dataset,
        sources,
        license_text,
        numerator,
        denominator,
    )


@dataclass(frozen=True)
class ConceptNetEndpoint:
    """ConceptNet concept URI 或外部 URI 的无损 typed 边界。"""

    kind: int
    uri: str
    language: str
    term: str
    suffix: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.kind not in {ENDPOINT_CONCEPT, ENDPOINT_EXTERNAL}:
            raise ConceptNetAdapterError(
                "BAD_URI", "ConceptNet endpoint kind 非法")
        if (not isinstance(self.uri, str) or not self.uri
                or self.uri.strip() != self.uri
                or any(character in self.uri for character in "\t\r\n")):
            raise ConceptNetAdapterError("BAD_URI", "ConceptNet endpoint URI 非法")
        if self.kind == ENDPOINT_CONCEPT:
            if not self.language or not self.term:
                raise ConceptNetAdapterError(
                    "BAD_URI", "ConceptNet concept 缺 language/term")
        elif self.language or self.term or self.suffix:
            raise ConceptNetAdapterError(
                "BAD_URI", "ConceptNet external endpoint 不得伪造 concept 字段")

    def to_dict(self) -> dict[str, Any]:
        """导出 endpoint，并显式保留 concept 的语言和 suffix。"""
        return {
            "kind": self.kind,
            "language": self.language,
            "suffix": list(self.suffix),
            "term": self.term,
            "uri": self.uri,
        }


def parse_conceptnet_endpoint(value: str) -> ConceptNetEndpoint:
    """解析 `/c/<lang>/<term>/...`；其他合法 URI 作为 opaque external。"""
    if not isinstance(value, str) or not value:
        raise ConceptNetAdapterError("BAD_URI", "ConceptNet endpoint 为空")
    _endpoint_language(value)
    if value.startswith("/c/"):
        parts = value.split("/")
        if len(parts) < 4 or parts[0] or parts[1] != "c":
            raise ConceptNetAdapterError("BAD_URI", "ConceptNet concept URI 非法")
        language, term = parts[2], parts[3]
        suffix_items = parts[4:]
        if suffix_items and suffix_items[-1] == "":
            suffix_items = suffix_items[:-1]
        suffix = tuple(suffix_items)
        if (not language or not term or any(not item for item in suffix)
                or language.strip() != language or term.strip() != term):
            raise ConceptNetAdapterError("BAD_URI", "ConceptNet concept URI 非法")
        return ConceptNetEndpoint(
            ENDPOINT_CONCEPT, value, language, term, suffix)
    if not (value.startswith("/") or value.startswith("https://")
            or value.startswith("http://")):
        raise ConceptNetAdapterError("BAD_URI", "ConceptNet external URI 非法")
    return ConceptNetEndpoint(ENDPOINT_EXTERNAL, value, "", "", ())


@dataclass(frozen=True)
class ConceptNetAssertion:
    """一条保留许可、sources、relation 和纯整数 weight 的外部 assertion。"""

    line_number: int
    assertion_uri: str
    relation: str
    start: ConceptNetEndpoint
    end: ConceptNetEndpoint
    dataset: str
    license_text: str
    license_partition: str
    sources: tuple[CanonicalJsonObject, ...]
    source_cluster_sha256: str
    weight_text: str
    weight_numerator: int
    weight_denominator: int
    metadata_sha256: str

    def __post_init__(self) -> None:
        if type(self.line_number) is not int or self.line_number <= 0:
            raise ConceptNetAdapterError(
                "BAD_LINE_NUMBER", "ConceptNet line number 非法")
        if not self.assertion_uri.startswith("/a/["):
            raise ConceptNetAdapterError("BAD_URI", "ConceptNet assertion URI 非法")
        if not self.relation.startswith("/r/") or len(self.relation) <= 3:
            raise ConceptNetAdapterError("BAD_RELATION", "ConceptNet relation 非法")
        if self.license_partition != LICENSE_PARTITIONS.get(self.license_text):
            raise ConceptNetAdapterError(
                "BAD_LICENSE", "ConceptNet license partition 非法")
        if not self.dataset.startswith("/d/"):
            raise ConceptNetAdapterError("BAD_DATASET", "ConceptNet dataset 非法")
        if not self.sources or any(
                not isinstance(item, CanonicalJsonObject) for item in self.sources):
            raise ConceptNetAdapterError("BAD_SOURCES", "ConceptNet sources 非法")
        if (type(self.weight_numerator) is not int
                or type(self.weight_denominator) is not int
                or self.weight_numerator <= 0 or self.weight_denominator <= 0):
            raise ConceptNetAdapterError("BAD_WEIGHT", "ConceptNet weight ratio 非法")
        for digest in (self.source_cluster_sha256, self.metadata_sha256):
            if (len(digest) != 64
                    or any(item not in "0123456789abcdef" for item in digest)):
                raise ConceptNetAdapterError("BAD_DIGEST", "ConceptNet digest 非法")

    def to_dict(self) -> dict[str, Any]:
        """导出外部 assertion，钉死 relation 与项目关系/真值不等价。"""
        return {
            "assertion_uri": self.assertion_uri,
            "dataset": self.dataset,
            "definitive_truth_authoritative": 0,
            "end": self.end.to_dict(),
            "license_partition": self.license_partition,
            "license_text": self.license_text,
            "line_number": self.line_number,
            "metadata_sha256": self.metadata_sha256,
            "project_relation_authoritative": 0,
            "relation": self.relation,
            "source_cluster_sha256": self.source_cluster_sha256,
            "sources": [item.to_value() for item in self.sources],
            "start": self.start.to_dict(),
            "weight_denominator": self.weight_denominator,
            "weight_numerator": self.weight_numerator,
            "weight_text": self.weight_text,
        }


def parse_conceptnet_assertion(
        line: str,
        *,
        line_number: int,
        ) -> ConceptNetAssertion:
    """严格解析五列 assertion，并验证 URI、许可、sources 和 weight 原文。"""
    (
        assertion_uri,
        relation,
        start_text,
        end_text,
        _,
        _,
        dataset,
        raw_sources,
        license_text,
        numerator,
        denominator,
    ) = _assertion_fields(line)
    metadata_text = line.rsplit("\t", 1)[1]
    start = parse_conceptnet_endpoint(start_text)
    end = parse_conceptnet_endpoint(end_text)
    try:
        sources = tuple(CanonicalJsonObject.from_value(item)
                        for item in raw_sources)
    except Exception as error:
        raise ConceptNetAdapterError(
            "BAD_SOURCES", "ConceptNet source object 非规范") from error
    metadata = _metadata(metadata_text)
    weight_text = str(metadata["weight"])
    source_values = [item.to_value() for item in sources]
    source_cluster_sha256 = hashlib.sha256(canonical_json_line({
        "dataset": dataset,
        "sources": sorted(
            source_values,
            key=lambda item: canonical_json_line(item),
        ),
    })).hexdigest()
    return ConceptNetAssertion(
        line_number,
        assertion_uri,
        relation,
        start,
        end,
        dataset,
        license_text,
        LICENSE_PARTITIONS[license_text],
        sources,
        source_cluster_sha256,
        weight_text,
        numerator,
        denominator,
        hashlib.sha256(metadata_text.encode("utf-8")).hexdigest(),
    )


@dataclass(frozen=True)
class ConceptNetScanReport:
    """ConceptNet 完整 gzip 的许可分区、目标关系和事件双遍摘要。"""

    compressed_sha256: str
    compressed_size_bytes: int
    line_count: int
    assertion_count: int
    anomaly_count: int
    terminal_newline_present: int
    zh_endpoint_count: int
    zh_zh_count: int
    license_counts: CanonicalJsonObject
    license_event_sha256: CanonicalJsonObject
    target_relation_counts: CanonicalJsonObject
    anomaly_codes: CanonicalJsonObject
    event_encoding: str
    event_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """导出规范扫描报告。"""
        return {
            "anomaly_codes": self.anomaly_codes.to_value(),
            "anomaly_count": self.anomaly_count,
            "assertion_count": self.assertion_count,
            "compressed_sha256": self.compressed_sha256,
            "compressed_size_bytes": self.compressed_size_bytes,
            "event_sha256": self.event_sha256,
            "event_encoding": self.event_encoding,
            "license_counts": self.license_counts.to_value(),
            "license_event_sha256": self.license_event_sha256.to_value(),
            "line_count": self.line_count,
            "target_relation_counts": self.target_relation_counts.to_value(),
            "terminal_newline_present": self.terminal_newline_present,
            "zh_endpoint_count": self.zh_endpoint_count,
            "zh_zh_count": self.zh_zh_count,
        }


def scan_conceptnet_gzip(
        path: str | Path,
        *,
        expected_compressed_sha256: str,
        complete_snapshot: bool,
        ) -> ConceptNetScanReport:
    """流式扫描 gzip，分别形成两许可事件 hash，并对坏行保留 anomaly。"""
    source = Path(path)
    if (not source.is_file()
            or sha256_path(source) != expected_compressed_sha256):
        raise ConceptNetAdapterError(
            "BAD_FILE_HASH", "ConceptNet compressed SHA-256 不一致")
    event_digest = hashlib.sha256()
    license_digests = {
        partition: hashlib.sha256()
        for partition in sorted(LICENSE_PARTITIONS.values())
    }
    license_counts = {partition: 0 for partition in license_digests}
    target_counts = {
        relation: {"all": 0, "zh_endpoint": 0, "zh_zh": 0}
        for relation in TARGET_RELATIONS
    }
    anomaly_codes: dict[str, int] = {}
    line_count = 0
    assertion_count = 0
    anomaly_count = 0
    terminal_newline = 0
    zh_endpoint_count = 0
    zh_zh_count = 0
    previous_assertion_uri = ""
    try:
        with gzip.open(
                source,
                "rt",
                encoding="utf-8",
                errors="strict",
                newline="",
                ) as handle:
            for line_number, raw in enumerate(handle, start=1):
                line_count = line_number
                terminal_newline = 1 if raw.endswith(("\n", "\r")) else 0
                line = raw.rstrip("\r\n")
                try:
                    (
                        assertion_uri,
                        relation,
                        _,
                        _,
                        start_language,
                        end_language,
                        _,
                        _,
                        license_text,
                        _,
                        _,
                    ) = _assertion_fields(line)
                    if (previous_assertion_uri
                            and assertion_uri <= previous_assertion_uri):
                        raise ConceptNetAdapterError(
                            "BAD_ORDER", "ConceptNet assertion URI 非严格递增")
                except ConceptNetAdapterError as error:
                    anomaly_count += 1
                    anomaly_codes[error.code] = anomaly_codes.get(error.code, 0) + 1
                    event = {
                        "code": error.code,
                        "kind": "anomaly",
                        "line_number": line_number,
                        "raw_line_sha256": hashlib.sha256(
                            line.encode("utf-8")).hexdigest(),
                    }
                else:
                    previous_assertion_uri = assertion_uri
                    assertion_count += 1
                    event_bytes = b"A\0" + line.encode("utf-8") + b"\n"
                    license_partition = LICENSE_PARTITIONS[license_text]
                    license_counts[license_partition] += 1
                    license_digests[license_partition].update(event_bytes)
                    start_zh = start_language == "zh"
                    end_zh = end_language == "zh"
                    if start_zh or end_zh:
                        zh_endpoint_count += 1
                    if start_zh and end_zh:
                        zh_zh_count += 1
                    if relation in target_counts:
                        target = target_counts[relation]
                        target["all"] += 1
                        if start_zh or end_zh:
                            target["zh_endpoint"] += 1
                        if start_zh and end_zh:
                            target["zh_zh"] += 1
                    event_digest.update(event_bytes)
                    continue
                event_digest.update(b"E\0" + canonical_json_line(event))
    except (OSError, EOFError, UnicodeError) as error:
        raise ConceptNetAdapterError(
            "BAD_GZIP", "ConceptNet gzip/UTF-8 损坏") from error
    if sha256_path(source) != expected_compressed_sha256:
        raise ConceptNetAdapterError(
            "MIDREAD_CHANGE", "ConceptNet 文件读取期间 SHA-256 漂移")
    if complete_snapshot and (line_count == 0 or anomaly_count
                              or assertion_count != line_count):
        raise ConceptNetAdapterError(
            "INCOMPLETE_SNAPSHOT", "ConceptNet 完整快照含 anomaly 或计数不一致")
    return ConceptNetScanReport(
        expected_compressed_sha256,
        source.stat().st_size,
        line_count,
        assertion_count,
        anomaly_count,
        terminal_newline,
        zh_endpoint_count,
        zh_zh_count,
        CanonicalJsonObject.from_value(license_counts),
        CanonicalJsonObject.from_value({
            key: digest.hexdigest() for key, digest in license_digests.items()
        }),
        CanonicalJsonObject.from_value(target_counts),
        CanonicalJsonObject.from_value(anomaly_codes),
        "VALIDATED_RAW_LINE_V1",
        event_digest.hexdigest(),
    )


def read_conceptnet_sample(path: str | Path) -> tuple[ConceptNetAssertion, ...]:
    """读取无空行的公开小样，并逐行恢复 typed assertion。"""
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="strict").splitlines()
    except (OSError, UnicodeError) as error:
        raise ConceptNetAdapterError(
            "BAD_SAMPLE", "ConceptNet sample 无法读取") from error
    if not lines or any(not line for line in lines):
        raise ConceptNetAdapterError(
            "BAD_SAMPLE", "ConceptNet sample 为空或含空行")
    return tuple(parse_conceptnet_assertion(line, line_number=index)
                 for index, line in enumerate(lines, start=1))


__all__ = [
    "ADAPTER_VERSION",
    "ConceptNetAdapterError",
    "ConceptNetAssertion",
    "ConceptNetEndpoint",
    "ConceptNetScanReport",
    "ENDPOINT_CONCEPT",
    "ENDPOINT_EXTERNAL",
    "LICENSE_PARTITIONS",
    "OFFICIAL_URL",
    "PARSER_VERSION",
    "SNAPSHOT_ID",
    "SOURCE_KEY",
    "TARGET_RELATIONS",
    "decimal_text_to_ratio",
    "parse_conceptnet_assertion",
    "parse_conceptnet_endpoint",
    "read_conceptnet_sample",
    "scan_conceptnet_gzip",
]
