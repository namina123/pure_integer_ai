"""冻结外部中文阅读理解来源，并生成标签隔离的评测包。"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

from opencc import OpenCC

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


EXTERNAL_EVAL_KIND = "PH2_BROAD_QA_EXTERNAL_EVAL_PACK_V1"
EXTERNAL_SELECTION_RULE = "TITLE_BUCKET_THEN_ITEM_SHA256_V1"
_TO_SIMPLIFIED = OpenCC("t2s")
_TO_TRADITIONAL = OpenCC("s2t")
_CMRC_REVISION = "c0eb1b6ba219847457e6af3180da722bbeb656af"
_DRCD_REVISION = "b944790de5af02c5fbb7cd9cb1473d27d169eebf"
_CMRC_FILES = (
    ("train", "data/cmrc2018_train.json",
     "d935a2d2c3ea8fef4b6b3d7f1c876f5870b720fc0c9caf4d14989f394a3d4745"),
    ("dev", "data/cmrc2018_dev.json",
     "5cfe4414c28a8ecbb51670f78c0dc7d1049f286c2d5769b52f1f94bcc0752cf1"),
    ("trial", "data/cmrc2018_trial.json",
     "a976d1fd5efc173bd58ff1c57e958de5f49fed633a7bfb8e0e402e5490d75f5e"),
)
_DRCD_FILES = (
    ("training", "DRCD_training.json",
     "5e6268091ab98f0bb858f03c77fba85e23b31e49a91638e4f22d0d8ec703f79a"),
    ("dev", "DRCD_dev.json",
     "e236df03861ba241bd7b29628cd48b3b8e339c60137e041a4358b5320ed61928"),
    ("test", "DRCD_test.json",
     "d9de0b4d247a8391ac8daee6c02c2c8272b75562a1e2377b50ae04ecc09a7438"),
)


# object-model: exception
class BroadQaExternalDataError(RuntimeError):
    """外部来源、schema、划分或冻结 artifact 发生漂移。"""


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ExternalQaSourceFile:
    """声明一份官方文件的身份、格式、许可和来源分区。"""

    source_key: str
    source_partition: str
    format_kind: str
    path: Path
    expected_sha256: str
    revision: str
    license_id: str
    upstream_url: str

    def __post_init__(self) -> None:
        """核验调用方提供的来源身份不变量。"""
        if (not self.source_key or not self.source_partition
                or self.format_kind not in {"CMRC2018", "DRCD"}
                or not isinstance(self.path, Path) or not self.path.is_file()
                or len(self.expected_sha256) != 64
                or any(item not in "0123456789abcdef"
                       for item in self.expected_sha256)
                or not self.revision or not self.license_id
                or not self.upstream_url.startswith("https://")):
            raise BroadQaExternalDataError("external source identity 非法")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ExternalQaItem:
    """表达来源绑定、问题与金答案，供冻结器分离 questions/labels。"""

    item_id: str
    source_key: str
    source_partition: str
    source_revision: str
    source_question_id: str
    title: str
    context: str
    question: str
    gold_answers: tuple[str, ...]
    license_id: str
    upstream_url: str

    def __post_init__(self) -> None:
        """核验问题、上下文、答案和稳定身份均完整。"""
        strings = (
            self.item_id, self.source_key, self.source_partition,
            self.source_revision, self.source_question_id, self.title,
            self.context, self.question, self.license_id, self.upstream_url,
        )
        if (any(not isinstance(value, str) or not value for value in strings)
                or len(self.item_id) != 64
                or any(item not in "0123456789abcdef" for item in self.item_id)
                or not isinstance(self.gold_answers, tuple)
                or not self.gold_answers
                or any(not isinstance(value, str) or not value
                       for value in self.gold_answers)):
            raise BroadQaExternalDataError("external QA item 非法")

    @property
    def title_key(self) -> str:
        """返回跨简繁、跨来源稳定的标题分组键。"""
        return normalize_external_text(self.title)

    def question_record(self, split: str) -> dict[str, object]:
        """导出不携带金答案的规范预测输入。"""
        return {
            "context": self.context,
            "context_sha256": hashlib.sha256(
                self.context.encode("utf-8")).hexdigest(),
            "format_version": 1,
            "item_id": self.item_id,
            "license_id": self.license_id,
            "question": self.question,
            "record_kind": "PH2_BROAD_QA_EXTERNAL_QUESTION_V1",
            "source_key": self.source_key,
            "source_partition": self.source_partition,
            "source_question_id": self.source_question_id,
            "source_revision": self.source_revision,
            "split": split,
            "title": self.title,
            "upstream_url": self.upstream_url,
        }

    def label_record(self, split: str) -> dict[str, object]:
        """导出独立于预测输入的规范评分标签。"""
        return {
            "format_version": 1,
            "gold_answers": list(self.gold_answers),
            "item_id": self.item_id,
            "record_kind": "PH2_BROAD_QA_EXTERNAL_LABEL_V1",
            "split": split,
        }


def official_external_qa_sources(
        cmrc_root: str | Path,
        drcd_root: str | Path,
        ) -> tuple[ExternalQaSourceFile, ...]:
    """把官方 checkout 的 commit、文件 hash、许可和 URL 绑定为来源。"""
    cmrc = Path(cmrc_root).resolve()
    drcd = Path(drcd_root).resolve()
    values = []
    for partition, relative, sha256 in _CMRC_FILES:
        values.append(ExternalQaSourceFile(
            "CMRC2018", partition, "CMRC2018", cmrc / relative,
            sha256, _CMRC_REVISION, "CC-BY-SA-4.0",
            "https://github.com/ymcui/cmrc2018"))
    for partition, relative, sha256 in _DRCD_FILES:
        values.append(ExternalQaSourceFile(
            "DRCD", partition, "DRCD", drcd / relative,
            sha256, _DRCD_REVISION, "CC-BY-SA-3.0",
            "https://github.com/DRCKnowledgeTeam/DRCD"))
    return tuple(values)


def normalize_external_text(value: str) -> str:
    """以简体、无空白、casefold 形式做标题和答案严格比对。"""
    if not isinstance(value, str):
        raise TypeError("external normalized value 必须是字符串")
    return "".join(_TO_SIMPLIFIED.convert(value).split()).casefold()


def external_title_surfaces(value: str) -> tuple[str, ...]:
    """返回来源标题的原文、简体和繁体精确表面形式。"""
    if not isinstance(value, str) or not value:
        raise BroadQaExternalDataError("external title surface 非法")
    return tuple(sorted({
        value, _TO_SIMPLIFIED.convert(value), _TO_TRADITIONAL.convert(value),
    }))


def _sha256_file(path: Path) -> str:
    """流式计算来源或已冻结文件的 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_source(source: ExternalQaSourceFile) -> object:
    """在解析前核验完整文件身份并读取 JSON。"""
    actual = _sha256_file(source.path)
    if actual != source.expected_sha256:
        raise BroadQaExternalDataError(
            f"external source SHA 漂移: {source.source_partition}")
    try:
        return json.loads(source.path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            f"external source JSON 非法: {source.source_partition}") from error


def _item_id(
        source: ExternalQaSourceFile,
        *,
        question_id: str,
        title: str,
        context: str,
        question: str,
        ) -> str:
    """从来源身份与原始内容生成问题稳定身份。"""
    payload = "\0".join((
        source.source_key, source.revision, source.source_partition,
        question_id, title, context, question,
    )).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _unique_answers(values: Iterable[str]) -> tuple[str, ...]:
    """按首次出现顺序去重答案，保留官方表面形式。"""
    result = []
    seen = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


def _load_cmrc(
        source: ExternalQaSourceFile,
        value: object,
        anomalies: Counter[str],
        ) -> list[ExternalQaItem]:
    """严格解析 CMRC 原生扁平 context 格式并排除非字符串答案。"""
    if not isinstance(value, list):
        raise BroadQaExternalDataError("CMRC 根必须是 array")
    result = []
    for article in value:
        if (not isinstance(article, dict)
                or set(article) != {"context_id", "context_text", "qas", "title"}
                or not isinstance(article["context_text"], str)
                or not isinstance(article["title"], str)
                or not isinstance(article["qas"], list)):
            raise BroadQaExternalDataError("CMRC context schema 漂移")
        context = article["context_text"]
        title = article["title"]
        if not title:
            anomalies["EMPTY_TITLE_QUESTION"] += len(article["qas"])
            continue
        for qa in article["qas"]:
            if (not isinstance(qa, dict)
                    or set(qa) != {"query_id", "query_text", "answers"}
                    or not isinstance(qa["query_id"], str)
                    or not isinstance(qa["query_text"], str)
                    or not isinstance(qa["answers"], list)
                    or not qa["answers"]):
                raise BroadQaExternalDataError("CMRC question schema 漂移")
            if not qa["query_text"]:
                anomalies["EMPTY_QUESTION"] += 1
                continue
            if any(type(answer) is not str for answer in qa["answers"]):
                anomalies["NON_STRING_ANSWER_QUESTION"] += 1
                continue
            answers = _unique_answers(qa["answers"])
            if (not answers or any(not answer or answer not in context
                                   for answer in answers)):
                anomalies["ANSWER_NOT_IN_CONTEXT_QUESTION"] += 1
                continue
            result.append(ExternalQaItem(
                _item_id(
                    source, question_id=qa["query_id"], title=title,
                    context=context, question=qa["query_text"]),
                source.source_key, source.source_partition, source.revision,
                qa["query_id"], title, context, qa["query_text"], answers,
                source.license_id, source.upstream_url,
            ))
    return result


def _load_drcd(
        source: ExternalQaSourceFile,
        value: object,
        anomalies: Counter[str],
        ) -> list[ExternalQaItem]:
    """严格解析 DRCD SQuAD 结构，并逐答案核验官方字符 span。"""
    if (not isinstance(value, dict) or set(value) != {"version", "data"}
            or not isinstance(value["version"], str)
            or not isinstance(value["data"], list)):
        raise BroadQaExternalDataError("DRCD 根 schema 漂移")
    result = []
    for article in value["data"]:
        if (not isinstance(article, dict)
                or set(article) != {"id", "paragraphs", "title"}
                or not isinstance(article["title"], str)
                or not isinstance(article["paragraphs"], list)):
            raise BroadQaExternalDataError("DRCD article schema 漂移")
        title = article["title"]
        if not title:
            anomalies["EMPTY_TITLE_QUESTION"] += sum(
                len(paragraph.get("qas", ()))
                for paragraph in article["paragraphs"]
                if isinstance(paragraph, dict))
            continue
        for paragraph in article["paragraphs"]:
            if (not isinstance(paragraph, dict)
                    or set(paragraph) != {"context", "id", "qas"}
                    or not isinstance(paragraph["context"], str)
                    or not isinstance(paragraph["qas"], list)):
                raise BroadQaExternalDataError("DRCD paragraph schema 漂移")
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                if (not isinstance(qa, dict)
                        or set(qa) != {"answers", "id", "question"}
                        or not isinstance(qa["id"], str)
                        or not isinstance(qa["question"], str)
                        or not isinstance(qa["answers"], list)
                        or not qa["answers"]):
                    raise BroadQaExternalDataError("DRCD question schema 漂移")
                if not qa["question"]:
                    anomalies["EMPTY_QUESTION"] += 1
                    continue
                answers = []
                invalid = False
                for answer in qa["answers"]:
                    if (not isinstance(answer, dict)
                            or set(answer) != {"answer_start", "id", "text"}
                            or type(answer["answer_start"]) is not int
                            or answer["answer_start"] < 0
                            or type(answer["text"]) is not str
                            or not answer["text"]
                            or answer["answer_start"] + len(answer["text"])
                            > len(context)
                            or context[
                                answer["answer_start"]:
                                answer["answer_start"] + len(answer["text"])
                            ] != answer["text"]):
                        invalid = True
                        break
                    answers.append(answer["text"])
                if invalid:
                    anomalies["INVALID_SPAN_QUESTION"] += 1
                    continue
                result.append(ExternalQaItem(
                    _item_id(
                        source, question_id=qa["id"], title=title,
                        context=context, question=qa["question"]),
                    source.source_key, source.source_partition,
                    source.revision, qa["id"], title, context,
                    qa["question"], _unique_answers(answers),
                    source.license_id, source.upstream_url,
                ))
    return result


def load_external_qa_sources(
        sources: Iterable[ExternalQaSourceFile],
        ) -> tuple[tuple[ExternalQaItem, ...], dict[str, object]]:
    """加载多份官方来源，返回稳定 item inventory 与异常分账。"""
    source_list = tuple(sources)
    if not source_list:
        raise BroadQaExternalDataError("external source inventory 为空")
    anomalies: Counter[str] = Counter()
    items = []
    source_reports = []
    for source in source_list:
        value = _read_source(source)
        before = len(items)
        source_anomalies: Counter[str] = Counter()
        if source.format_kind == "CMRC2018":
            items.extend(_load_cmrc(source, value, source_anomalies))
        else:
            items.extend(_load_drcd(source, value, source_anomalies))
        anomalies.update(source_anomalies)
        source_reports.append({
            "accepted_question_count": len(items) - before,
            "anomalies": dict(sorted(source_anomalies.items())),
            "format_kind": source.format_kind,
            "license_id": source.license_id,
            "revision": source.revision,
            "sha256": source.expected_sha256,
            "source_key": source.source_key,
            "source_partition": source.source_partition,
            "upstream_url": source.upstream_url,
        })
    identities = [item.item_id for item in items]
    if len(set(identities)) != len(identities):
        raise BroadQaExternalDataError("external item identity 重复")
    ordered = tuple(sorted(items, key=lambda item: item.item_id))
    return ordered, {
        "accepted_question_count": len(ordered),
        "anomalies": dict(sorted(anomalies.items())),
        "source_files": source_reports,
    }


def _split_for_title(title_key: str) -> str:
    """仅由规范标题决定 dev/held-out 域，防止同标题跨分区。"""
    digest = hashlib.sha256(title_key.encode("utf-8")).digest()
    return "dev" if int.from_bytes(digest[:4], "big") % 5 < 2 else "held_out"


def select_external_source_pack(
        items: Iterable[ExternalQaItem],
        *,
        dev_per_source: int = 100,
        held_out_per_source: int = 150,
        ) -> dict[str, tuple[ExternalQaItem, ...]]:
    """按全局标题桶隔离并为每个来源稳定抽取 100/150 问。"""
    if (type(dev_per_source) is not int or dev_per_source <= 0
            or type(held_out_per_source) is not int
            or held_out_per_source <= 0):
        raise BroadQaExternalDataError("external split quota 非法")
    grouped: dict[tuple[str, str], list[ExternalQaItem]] = defaultdict(list)
    source_keys = set()
    for item in items:
        split = _split_for_title(item.title_key)
        grouped[(item.source_key, split)].append(item)
        source_keys.add(item.source_key)
    selected: dict[str, list[ExternalQaItem]] = {"dev": [], "held_out": []}
    for source_key in sorted(source_keys):
        for split, quota in (
                ("dev", dev_per_source),
                ("held_out", held_out_per_source)):
            candidates = sorted(
                grouped[(source_key, split)], key=lambda item: item.item_id)
            if len(candidates) < quota:
                raise BroadQaExternalDataError(
                    f"external split 候选不足: {source_key}/{split}")
            selected[split].extend(candidates[:quota])
    dev_titles = {item.title_key for item in selected["dev"]}
    held_titles = {item.title_key for item in selected["held_out"]}
    if dev_titles.intersection(held_titles):
        raise BroadQaExternalDataError("external title 泄漏到两个 split")
    return {
        split: tuple(sorted(values, key=lambda item: item.item_id))
        for split, values in selected.items()
    }


def select_external_successor_pack(
        items: Iterable[ExternalQaItem],
        *,
        excluded_title_keys: Iterable[str],
        dev_per_source: int = 100,
        held_out_per_source: int = 150,
        ) -> dict[str, tuple[ExternalQaItem, ...]]:
    """排除已消费标题域，再以同一预定义桶规则冻结 successor family。"""
    excluded = frozenset(excluded_title_keys)
    if (not excluded
            or any(not isinstance(item, str) or not item for item in excluded)):
        raise BroadQaExternalDataError("successor excluded title inventory 非法")
    filtered = tuple(item for item in items if item.title_key not in excluded)
    selected = select_external_source_pack(
        filtered, dev_per_source=dev_per_source,
        held_out_per_source=held_out_per_source)
    if any(item.title_key in excluded
           for values in selected.values() for item in values):
        raise BroadQaExternalDataError("successor family 复用了已消费标题域")
    return selected


def _write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> int:
    """以规范单行 JSON 写入不可覆盖文件并返回记录数。"""
    if path.exists():
        raise BroadQaExternalDataError(f"external artifact 禁止覆盖: {path.name}")
    count = 0
    with path.open("xb") as handle:
        for record in records:
            handle.write(canonical_json_line(record))
            count += 1
    return count


def freeze_external_source_pack(
        selected: dict[str, tuple[ExternalQaItem, ...]],
        *,
        target_dir: str | Path,
        source_report: dict[str, object],
        ) -> dict[str, object]:
    """不可覆盖地发布 questions/labels，并最后写入冻结 manifest。"""
    if set(selected) != {"dev", "held_out"}:
        raise BroadQaExternalDataError("external selected split 漂移")
    target = Path(target_dir).resolve()
    if target.exists():
        raise BroadQaExternalDataError("external freeze target 已存在")
    target.mkdir(parents=True)
    artifacts = []
    for split in ("dev", "held_out"):
        values = selected[split]
        for role, records in (
                ("questions", (item.question_record(split) for item in values)),
                ("labels", (item.label_record(split) for item in values))):
            path = target / f"{split}.{role}.jsonl"
            count = _write_jsonl(path, records)
            artifacts.append({
                "bytes": path.stat().st_size,
                "record_count": count,
                "role": f"{split}_{role}",
                "sha256": _sha256_file(path),
            })
    manifest = {
        "artifact_kind": EXTERNAL_EVAL_KIND,
        "artifacts": artifacts,
        "format_version": 1,
        "selection_rule": EXTERNAL_SELECTION_RULE,
        "source_report": source_report,
        "splits": {
            split: {
                "question_count": len(selected[split]),
                "source_counts": dict(sorted(Counter(
                    item.source_key for item in selected[split]).items())),
                "title_count": len({item.title_key for item in selected[split]}),
            }
            for split in ("dev", "held_out")
        },
        "status": "FROZEN_NOT_RUN",
        "title_domain_overlap_count": 0,
    }
    manifest_path = target / "manifest.json"
    manifest_path.write_bytes(canonical_json_line(manifest))
    return {
        **manifest,
        "manifest_sha256": _sha256_file(manifest_path),
    }


__all__ = [
    "BroadQaExternalDataError",
    "EXTERNAL_EVAL_KIND",
    "EXTERNAL_SELECTION_RULE",
    "ExternalQaItem",
    "ExternalQaSourceFile",
    "freeze_external_source_pack",
    "external_title_surfaces",
    "load_external_qa_sources",
    "normalize_external_text",
    "official_external_qa_sources",
    "select_external_source_pack",
    "select_external_successor_pack",
]
