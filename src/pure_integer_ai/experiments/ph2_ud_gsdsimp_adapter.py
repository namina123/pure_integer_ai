"""UD Chinese GSDSimp r2.18 CoNLL-U 的整数 tuple ID 和双遍扫描 adapter。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line
from pure_integer_ai.experiments.ph2_raw_snapshot import sha256_path


SOURCE_KEY = "UD_ZH_GSDSIMP_R2_18"
REPOSITORY_URL = "https://github.com/UniversalDependencies/UD_Chinese-GSDSimp"
TAG = "r2.18"
COMMIT_SHA1 = "7b61ed473f963e911788efdf1f478154bc1053e4"
LICENSE_ID = "CC-BY-SA-4.0"
ADAPTER_VERSION = 1
PARSER_VERSION = 1

NODE_WORD = 1
NODE_RANGE = 2
NODE_EMPTY = 3


class UdGsdsimpAdapterError(RuntimeError):
    """CoNLL-U 行、ID、sentence graph、UTF-8 或文件 hash 不合法。"""


def _decimal(value: str, *, where: str, allow_zero: bool = False) -> int:
    """解析无符号十进制，拒绝前导零和任何 float 风格。"""
    if (not value or not value.isascii() or not value.isdigit()
            or (len(value) > 1 and value.startswith("0"))):
        raise UdGsdsimpAdapterError(f"CoNLL-U {where} 非规范十进制")
    result = int(value)
    if result < (0 if allow_zero else 1):
        raise UdGsdsimpAdapterError(f"CoNLL-U {where} 超出范围")
    return result


@dataclass(frozen=True, order=True)
class ConlluNodeId:
    """以三个严格整数表达 word、range 或 empty node，绝不经过 float。"""

    kind: int
    major: int
    tail: int

    def __post_init__(self) -> None:
        if self.kind not in {NODE_WORD, NODE_RANGE, NODE_EMPTY}:
            raise UdGsdsimpAdapterError("CoNLL-U node kind 非法")
        if type(self.major) is not int or self.major <= 0:
            raise UdGsdsimpAdapterError("CoNLL-U node major 必须为正整数")
        if type(self.tail) is not int or self.tail < 0:
            raise UdGsdsimpAdapterError("CoNLL-U node tail 必须为非负整数")
        if self.kind == NODE_WORD and self.tail != 0:
            raise UdGsdsimpAdapterError("word ID tail 必须为 0")
        if self.kind in {NODE_RANGE, NODE_EMPTY} and self.tail <= 0:
            raise UdGsdsimpAdapterError("range/empty ID tail 必须为正整数")
        if self.kind == NODE_RANGE and self.tail <= self.major:
            raise UdGsdsimpAdapterError("range ID end 必须大于 start")

    def stable_key(self) -> tuple[int, int, int]:
        """返回完整 integer tuple 身份。"""
        return self.kind, self.major, self.tail

    def to_list(self) -> list[int]:
        """导出 JSON 整数列表。"""
        return [self.kind, self.major, self.tail]


def parse_conllu_node_id(value: str) -> ConlluNodeId:
    """按 `N`、`N-M`、`N.M` 解析，不调用 float。"""
    if not isinstance(value, str) or not value:
        raise UdGsdsimpAdapterError("CoNLL-U ID 不能为空")
    if value.isdigit():
        return ConlluNodeId(NODE_WORD, _decimal(value, where="word ID"), 0)
    if "-" in value and "." not in value:
        parts = value.split("-")
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            raise UdGsdsimpAdapterError("CoNLL-U range ID 非法")
        return ConlluNodeId(
            NODE_RANGE,
            _decimal(parts[0], where="range start"),
            _decimal(parts[1], where="range end"),
        )
    if "." in value and "-" not in value:
        parts = value.split(".")
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            raise UdGsdsimpAdapterError("CoNLL-U empty ID 非法")
        return ConlluNodeId(
            NODE_EMPTY,
            _decimal(parts[0], where="empty major"),
            _decimal(parts[1], where="empty minor"),
        )
    raise UdGsdsimpAdapterError("CoNLL-U ID 非 word/range/empty")


def _pairs(value: str, *, field: str) -> tuple[tuple[str, str], ...]:
    """解析 FEATS/MISC 的 `key=value|...`，`_` 为空集合。"""
    if value == "_":
        return ()
    pairs: list[tuple[str, str]] = []
    for item in value.split("|"):
        if "=" not in item:
            if field == "MISC" and item:
                pairs.append((item, ""))
                continue
            raise UdGsdsimpAdapterError(f"CoNLL-U {field} 非 key=value")
        name, content = item.split("=", 1)
        if not name or not content:
            raise UdGsdsimpAdapterError(f"CoNLL-U {field} key/value 为空")
        pairs.append((name, content))
    if len({name for name, _ in pairs}) != len(pairs):
        raise UdGsdsimpAdapterError(f"CoNLL-U {field} key 重复")
    return tuple(pairs)


def _dependency_head(value: str) -> tuple[int, int]:
    """解析 enhanced dependency head 为 `(major, minor)`，0 为 root。"""
    if value == "0":
        return 0, 0
    node = parse_conllu_node_id(value)
    if node.kind == NODE_RANGE:
        raise UdGsdsimpAdapterError("CoNLL-U DEPS head 不得为 range")
    return node.major, node.tail


def _dependencies(value: str) -> tuple[tuple[int, int, str], ...]:
    """解析 DEPS，保留 relation 文本但不映射项目 Role。"""
    if value == "_":
        return ()
    result: list[tuple[int, int, str]] = []
    for item in value.split("|"):
        if ":" not in item:
            raise UdGsdsimpAdapterError("CoNLL-U DEPS 缺少 relation")
        head_text, relation = item.split(":", 1)
        if not relation:
            raise UdGsdsimpAdapterError("CoNLL-U DEPS relation 为空")
        major, minor = _dependency_head(head_text)
        result.append((major, minor, relation))
    if len(set(result)) != len(result):
        raise UdGsdsimpAdapterError("CoNLL-U DEPS 重复")
    return tuple(result)


@dataclass(frozen=True)
class ConlluRow:
    """一行 CoNLL-U token/range/empty node 的 typed 外部注释。"""

    node_id: ConlluNodeId
    form: str
    lemma: str
    upos: str
    xpos: str
    feats: tuple[tuple[str, str], ...]
    head: int | None
    deprel: str
    deps: tuple[tuple[int, int, str], ...]
    misc: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, ConlluNodeId):
            raise UdGsdsimpAdapterError("ConlluRow.node_id 类型错误")
        if not self.form:
            raise UdGsdsimpAdapterError("CoNLL-U FORM 不能为空")
        if self.node_id.kind == NODE_WORD:
            if type(self.head) is not int or self.head < 0:
                raise UdGsdsimpAdapterError("word HEAD 必须为非负整数")
            if not self.deprel or self.deprel == "_":
                raise UdGsdsimpAdapterError("word DEPREL 不能为空")
        elif self.node_id.kind == NODE_RANGE:
            if (self.lemma != "_" or self.upos != "_" or self.xpos != "_"
                    or self.feats or self.head is not None
                    or self.deprel != "_" or self.deps):
                raise UdGsdsimpAdapterError(
                    "range LEMMA/UPOS/XPOS/FEATS/HEAD/DEPREL/DEPS 必须为空")
        else:
            if self.head is not None or self.deprel != "_" or not self.deps:
                raise UdGsdsimpAdapterError("empty node 必须只用 enhanced DEPS")

    def to_dict(self) -> dict[str, Any]:
        """导出外部注释，显式钉死 dependency/Role 非等价。"""
        return {
            "dependency_label_authoritative": 0,
            "deprel": self.deprel,
            "deps": [list(item) for item in self.deps],
            "feats": [[name, value] for name, value in self.feats],
            "form": self.form,
            "head": self.head,
            "lemma": self.lemma,
            "misc": [[name, value] for name, value in self.misc],
            "node_id": self.node_id.to_list(),
            "project_role_authoritative": 0,
            "upos": self.upos,
            "xpos": self.xpos,
        }


def parse_conllu_row(line: str) -> ConlluRow:
    """严格解析十列 CoNLL-U 行。"""
    if not isinstance(line, str) or not line or line.strip() != line:
        raise UdGsdsimpAdapterError("CoNLL-U row 空或有首尾空白")
    columns = line.split("\t")
    if len(columns) != 10:
        raise UdGsdsimpAdapterError("CoNLL-U row 必须恰有十列")
    node_id = parse_conllu_node_id(columns[0])
    head: int | None
    if columns[6] == "_":
        head = None
    elif columns[6].isdigit():
        head = _decimal(columns[6], where="HEAD", allow_zero=True)
    else:
        raise UdGsdsimpAdapterError("CoNLL-U HEAD 必须为整数或 _")
    return ConlluRow(
        node_id,
        columns[1],
        columns[2],
        columns[3],
        columns[4],
        _pairs(columns[5], field="FEATS"),
        head,
        columns[7],
        _dependencies(columns[8]),
        _pairs(columns[9], field="MISC"),
    )


@dataclass(frozen=True)
class ConlluSentence:
    """一个带 sent_id/text provenance 的完整 CoNLL-U sentence block。"""

    sent_id: str
    text: str
    comments: tuple[tuple[str, str], ...]
    rows: tuple[ConlluRow, ...]
    first_line: int
    last_line: int

    def __post_init__(self) -> None:
        if not self.sent_id or not self.text or not self.rows:
            raise UdGsdsimpAdapterError("CoNLL-U sentence 缺 sent_id/text/rows")
        if self.first_line <= 0 or self.last_line < self.first_line:
            raise UdGsdsimpAdapterError("CoNLL-U sentence line span 非法")
        word_rows = [row for row in self.rows if row.node_id.kind == NODE_WORD]
        word_ids = [row.node_id.major for row in word_rows]
        node_keys = [row.node_id.stable_key() for row in self.rows]
        if len(node_keys) != len(set(node_keys)):
            raise UdGsdsimpAdapterError("CoNLL-U node ID 重复")
        if word_ids != list(range(1, len(word_rows) + 1)):
            raise UdGsdsimpAdapterError("CoNLL-U word ID 必须从 1 连续递增")
        word_set = set(word_ids)
        if sum(row.head == 0 for row in word_rows) != 1:
            raise UdGsdsimpAdapterError("CoNLL-U sentence 必须恰有一个 basic root")
        if any(row.head not in word_set | {0} for row in word_rows):
            raise UdGsdsimpAdapterError("CoNLL-U basic HEAD 引用缺失 word")
        ranges: list[set[int]] = []
        empty_ids: set[tuple[int, int]] = set()
        for row in self.rows:
            if row.node_id.kind == NODE_RANGE:
                covered = set(range(row.node_id.major, row.node_id.tail + 1))
                if not covered <= word_set:
                    raise UdGsdsimpAdapterError("CoNLL-U range 覆盖缺失 word")
                if any(covered & prior for prior in ranges):
                    raise UdGsdsimpAdapterError("CoNLL-U range 不得重叠")
                ranges.append(covered)
            if row.node_id.kind == NODE_EMPTY:
                if row.node_id.major not in word_set:
                    raise UdGsdsimpAdapterError(
                        "CoNLL-U empty node major 缺失 word")
                empty_ids.add((row.node_id.major, row.node_id.tail))
        by_major: dict[int, list[int]] = {}
        for major, minor in empty_ids:
            by_major.setdefault(major, []).append(minor)
        if any(sorted(minors) != list(range(1, len(minors) + 1))
               for minors in by_major.values()):
            raise UdGsdsimpAdapterError("CoNLL-U empty node minor 必须连续")
        dependency_heads = {(0, 0)} | {(item, 0) for item in word_set} | empty_ids
        if any((major, minor) not in dependency_heads
               for row in self.rows
               for major, minor, _ in row.deps):
            raise UdGsdsimpAdapterError("CoNLL-U enhanced DEPS 引用缺失 node")

    def to_dict(self) -> dict[str, Any]:
        """导出 sentence parser event。"""
        return {
            "comments": [[name, value] for name, value in self.comments],
            "first_line": self.first_line,
            "last_line": self.last_line,
            "rows": [row.to_dict() for row in self.rows],
            "sent_id": self.sent_id,
            "text": self.text,
        }


@dataclass(frozen=True)
class UdConlluScanReport:
    """单个 upstream split 文件的 sentence/node/anomaly 双遍摘要。"""

    relative_path: str
    split: str
    file_sha256: str
    size_bytes: int
    sentence_count: int
    word_count: int
    range_count: int
    empty_count: int
    anomaly_count: int
    terminal_newline_present: int
    event_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """导出规范扫描报告。"""
        return {
            "anomaly_count": self.anomaly_count,
            "empty_count": self.empty_count,
            "event_sha256": self.event_sha256,
            "file_sha256": self.file_sha256,
            "range_count": self.range_count,
            "relative_path": self.relative_path,
            "sentence_count": self.sentence_count,
            "size_bytes": self.size_bytes,
            "split": self.split,
            "terminal_newline_present": self.terminal_newline_present,
            "word_count": self.word_count,
        }


def _parse_sentence_block(
        block: list[tuple[int, str]],
        ) -> ConlluSentence:
    """解析并校验一个非空 sentence block。"""
    comments: list[tuple[str, str]] = []
    rows: list[ConlluRow] = []
    for _, line in block:
        if line.startswith("#"):
            content = line[1:].strip()
            if " = " in content:
                name, value = content.split(" = ", 1)
            else:
                name, value = "comment", content
            comments.append((name, value))
        else:
            rows.append(parse_conllu_row(line))
    values: dict[str, list[str]] = {}
    for name, value in comments:
        values.setdefault(name, []).append(value)
    if len(values.get("sent_id", ())) != 1 or len(values.get("text", ())) != 1:
        raise UdGsdsimpAdapterError("CoNLL-U sent_id/text 不唯一或缺失")
    return ConlluSentence(
        values["sent_id"][0],
        values["text"][0],
        tuple(comments),
        tuple(rows),
        block[0][0],
        block[-1][0],
    )


def _blocks(path: Path) -> Iterator[tuple[list[tuple[int, str]], int]]:
    """strict UTF-8 流式分 sentence block，并返回最后一行换行事实。"""
    try:
        with path.open("rt", encoding="utf-8", errors="strict", newline="") as handle:
            block: list[tuple[int, str]] = []
            last_has_newline = 0
            for line_number, raw in enumerate(handle, start=1):
                last_has_newline = 1 if raw.endswith(("\n", "\r")) else 0
                line = raw.rstrip("\r\n")
                if not line:
                    if block:
                        yield block, last_has_newline
                        block = []
                    continue
                block.append((line_number, line))
            if block:
                yield block, last_has_newline
    except (OSError, UnicodeError) as error:
        raise UdGsdsimpAdapterError("CoNLL-U 文件/UTF-8 损坏") from error


def iter_ud_conllu_sentences(path: str | Path) -> Iterator[ConlluSentence]:
    """流式返回严格校验的 sentence，供后继课程复用同一解析边界。"""
    source = Path(path)
    if not source.is_file():
        raise UdGsdsimpAdapterError("CoNLL-U 文件不存在")
    for block, _ in _blocks(source):
        yield _parse_sentence_block(block)


def scan_ud_conllu(
        path: str | Path,
        *,
        relative_path: str,
        split: str,
        expected_sha256: str,
        ) -> UdConlluScanReport:
    """前后核验文件 hash，并保留 sentence anomaly 而不猜修复。"""
    if split not in {"train", "dev", "held_out"}:
        raise UdGsdsimpAdapterError("UD split 非法")
    source = Path(path)
    if not source.is_file() or sha256_path(source) != expected_sha256:
        raise UdGsdsimpAdapterError("UD file SHA-256 不一致")
    digest = hashlib.sha256()
    sentence_count = 0
    word_count = 0
    range_count = 0
    empty_count = 0
    anomalies = 0
    sent_ids: set[str] = set()
    terminal_newline = 0
    for block, last_has_newline in _blocks(source):
        terminal_newline = last_has_newline
        try:
            sentence = _parse_sentence_block(block)
            if sentence.sent_id in sent_ids:
                raise UdGsdsimpAdapterError("UD sent_id 重复")
        except UdGsdsimpAdapterError:
            anomalies += 1
            event = {
                "block_sha256": hashlib.sha256(
                    "\n".join(line for _, line in block).encode(
                        "utf-8")).hexdigest(),
                "first_line": block[0][0],
                "kind": "anomaly",
                "last_line": block[-1][0],
            }
        else:
            sent_ids.add(sentence.sent_id)
            sentence_count += 1
            word_count += sum(
                row.node_id.kind == NODE_WORD for row in sentence.rows)
            range_count += sum(
                row.node_id.kind == NODE_RANGE for row in sentence.rows)
            empty_count += sum(
                row.node_id.kind == NODE_EMPTY for row in sentence.rows)
            event = sentence.to_dict()
            event["kind"] = "sentence"
        digest.update(canonical_json_line(event))
    if sha256_path(source) != expected_sha256:
        raise UdGsdsimpAdapterError("UD file 读取期间 SHA-256 漂移")
    return UdConlluScanReport(
        relative_path,
        split,
        expected_sha256,
        source.stat().st_size,
        sentence_count,
        word_count,
        range_count,
        empty_count,
        anomalies,
        terminal_newline,
        digest.hexdigest(),
    )


__all__ = [
    "ADAPTER_VERSION",
    "COMMIT_SHA1",
    "ConlluNodeId",
    "ConlluRow",
    "ConlluSentence",
    "LICENSE_ID",
    "NODE_EMPTY",
    "NODE_RANGE",
    "NODE_WORD",
    "PARSER_VERSION",
    "REPOSITORY_URL",
    "SOURCE_KEY",
    "TAG",
    "UdConlluScanReport",
    "UdGsdsimpAdapterError",
    "iter_ud_conllu_sentences",
    "parse_conllu_node_id",
    "parse_conllu_row",
    "scan_ud_conllu",
]
