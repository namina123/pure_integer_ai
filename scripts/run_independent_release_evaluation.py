"""Run a bounded, reproducible evaluation of an independent public release.

The evaluator deliberately uses only public release data plus an explicitly
generated Runtime conflict fixture.  It never reads private labels and writes
all evidence to a caller-selected K: directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sqlite3
import sys
from typing import Any

from pure_integer_ai.experiments.conversation_runtime_material_cli import (
    build_runtime_material_run,
)
from pure_integer_ai.experiments.public_model_release import (
    load_public_model_release,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME_LEDGER = Path(
    r"K:\pure_integer_ai_work\dialogue_sessions\multi-source-supported-20260825-a")
EVALUATION_FORMAT = "PURE_INTEGER_AI_INDEPENDENT_RELEASE_EVALUATION_V1"


def _canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _require_k_directory(value: str | Path, *, label: str) -> Path:
    path = Path(value).resolve()
    if path.drive.upper() != "K:" or not path.is_dir():
        raise ValueError(f"{label} 必须是 K 盘已存在目录")
    return path


def _course_questions(release_root: Path, *, limit: int = 5) -> tuple[str, ...]:
    courses = sorted(
        (release_root / "data" / "ph2").glob(
            "broad_wikipedia_passage_v1_compact_20k_v2.course*.jsonl"))
    titles: list[str] = []
    for path in courses:
        for line in path.read_bytes().splitlines():
            if not line.strip():
                continue
            row = json.loads(line.decode("utf-8"))
            if (row.get("split") == "heldout"
                    and row.get("sample_kind") == "POSITIVE"
                    and isinstance(row.get("title"), str)
                    and row["title"] not in titles):
                titles.append(row["title"])
                if len(titles) == limit:
                    return tuple(titles)
    # Current public releases may intentionally omit the large Wikipedia
    # course after its source has been reduced to the frozen QA index.  The
    # index itself is the authoritative public held-out surface in that case;
    # choose deterministic document titles rather than requiring a host file
    # that is not part of the release manifest.
    database = release_root / "knowledge" / "broad_qa.sqlite3"
    if database.is_file():
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        try:
            for (title,) in connection.execute(
                    "SELECT title FROM document "
                    "WHERE title IS NOT NULL AND title <> '' "
                    "ORDER BY doc_id LIMIT ?", (limit,)):
                if isinstance(title, str) and title not in titles:
                    titles.append(title)
        finally:
            connection.close()
        if len(titles) == limit:
            return tuple(titles)
    raise ValueError("release held-out 课程不足")


def _run_protocol(
        release_root: Path,
        requests: tuple[dict[str, object], ...],
        *,
        session_root: Path | None = None,
        runtime_root: Path | None = None,
        runtime_database: Path | None = None,
        metrics_output: Path | None = None,
        performance_tier: str = "deferred-narrow-fast",
        ) -> tuple[dict[str, object], ...]:
    if performance_tier not in {
            "strict", "deferred-narrow", "deferred-narrow-fast"}:
        raise ValueError("performance_tier 非法")
    command = [
        sys.executable, "-m",
        "pure_integer_ai.experiments.run_dialogue_protocol",
        "--release-root", str(release_root),
        "--performance-tier", performance_tier,
    ]
    if session_root is not None:
        command.extend(("--session-root", str(session_root)))
    if runtime_root is not None or runtime_database is not None:
        if runtime_root is None or runtime_database is None:
            raise ValueError("runtime root/database 必须同时提供")
        command.extend(("--runtime-material-ledger-root", str(runtime_root),
                        "--runtime-material-sqlite", str(runtime_database)))
    if metrics_output is not None:
        command.extend(("--metrics-output", str(metrics_output)))
    payload = b"".join(_canonical(item) for item in requests)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT / "src") + os.pathsep + environment.get(
        "PYTHONPATH", "")
    completed = subprocess.run(
        command, cwd=REPOSITORY_ROOT, input=payload,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=environment, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"独立协议进程失败 ({completed.returncode}): "
            f"{completed.stderr.decode('utf-8', errors='replace')[-2000:]}")
    responses: list[dict[str, object]] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        value = json.loads(line.decode("utf-8"))
        if not isinstance(value, dict):
            raise RuntimeError("独立协议响应必须是 JSON object")
        responses.append(value)
    return tuple(responses)


def _turns(responses: tuple[dict[str, object], ...]) -> tuple[dict[str, object], ...]:
    return tuple(item for item in responses if item.get("type") == "response")


def _wire_source_title(item: dict[str, object]) -> str | None:
    """读取正式 JSONL 的可见来源标题，不依赖内部 DialogueTurn 字段。"""
    source = item.get("source")
    if not isinstance(source, dict):
        return None
    title = source.get("title")
    return title if isinstance(title, str) and title else None


def _wire_text(item: dict[str, object]) -> str:
    value = item.get("text")
    return value if isinstance(value, str) else ""


def _wire_status(item: dict[str, object]) -> str:
    """把自然语言协议投影成 evaluator 自用标签，绝不写入用户接口。"""
    if _wire_source_title(item) is not None:
        return "ANSWER"
    text = _wire_text(item)
    if "请补充问题的范围或限定条件" in text:
        return "CLARIFY"
    if "当前公开资料无法确认这个问题" in text:
        return "UNKNOWN"
    return "OTHER"


def _wire_answer(item: dict[str, object], expected_title: str) -> bool:
    return (
        item.get("type") == "response"
        and _wire_source_title(item) == expected_title
        and bool(_wire_text(item).strip())
    )


def _wire_unknown(item: dict[str, object]) -> bool:
    return item.get("type") == "response" and _wire_status(item) == "UNKNOWN"


def _wire_clarify(item: dict[str, object]) -> bool:
    return item.get("type") == "response" and _wire_status(item) == "CLARIFY"


def _build_conflict_runtime(root: Path) -> tuple[Path, Path]:
    prefix = root.name
    material = root.parent / f"{prefix}-conflict-material.txt"
    runtime_root = root.parent / f"{prefix}-conflict-runtime"
    if material.exists() or runtime_root.exists():
        raise ValueError("冲突评测 fixture 已存在，拒绝覆盖")
    material.write_text(
        "甲来源说夜间模式降低亮度。乙来源说夜间模式提高亮度。",
        encoding="utf-8", newline="\n")
    _root, database = build_runtime_material_run(
        material_file=material,
        output_root=runtime_root,
        source_kind=94,
        source_id=2026082601,
        document_id=0,
        scope_id=2026082601,
        license_id="CC0-1.0",
        batch_id=1,
        authority_key=(7, 2026082601),
        version_key=(1, 2026082601),
        question="夜间模式如何影响亮度？",
        qualification_state="CONFLICT",
        reason_id="independent-source-conflict",
        source_title="冲突评测资料",
        source_url="https://example.invalid/conflict-evaluation",
    )
    return runtime_root, database


def _metric_summary(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "turn_count", "p50_us", "p95_us", "sqlite_statement_count_total",
        "sqlite_statement_count_per_turn", "peak_working_set_bytes",
        "performance_tier",
    }
    if not isinstance(value, dict) or not required.issubset(value):
        raise ValueError(f"性能摘要字段不完整: {path}")
    return {key: value[key] for key in sorted(required)}


def evaluate(
        release_root: str | Path,
        *,
        output_root: str | Path,
        runtime_ledger: str | Path = DEFAULT_RUNTIME_LEDGER,
        ) -> dict[str, object]:
    release = _require_k_directory(release_root, label="release_root")
    # 发布闭合、逐文件 SHA、嵌套 manifest 和 private 边界只严格验证一次；
    # 后续独立子进程仍检查闭合集合与文件大小，但不重复扫描全部大文件。
    load_public_model_release(release)
    output = Path(output_root).resolve()
    if output.drive.upper() != "K:":
        raise ValueError("output_root 必须位于 K 盘")
    if output.exists():
        raise ValueError("output_root 必须是尚不存在的新目录")
    output.mkdir(parents=True)
    runtime = _require_k_directory(runtime_ledger, label="runtime ledger")
    questions = _course_questions(release)
    heldout_responses = _run_protocol(
        release, tuple(
            {"id": f"heldout-{index}", "op": "turn",
             "text": f"{question}是什么？"}
            for index, question in enumerate(questions)
        ) + ({"id": "heldout-quit", "op": "quit"},))
    heldout_turns = _turns(heldout_responses)

    unknown_questions = (
        "不存在的独立评测实体_20260826是什么？",
        "虚构的随机未知对象_20260826位于哪里？",
    )
    unknown_turns = _turns(_run_protocol(
        release, tuple({"id": f"unknown-{i}", "op": "turn", "text": q}
                       for i, q in enumerate(unknown_questions))
        + ({"id": "unknown-quit", "op": "quit"},)))
    negative_questions = (
        "虚构的开放集负例_20260826是什么？",
        "不存在的负例对象_20260826位于哪里？",
    )
    negative_turns = _turns(_run_protocol(
        release, tuple({"id": f"negative-{i}", "op": "turn", "text": q}
                       for i, q in enumerate(negative_questions))
        + ({"id": "negative-quit", "op": "quit"},)))

    session_root = output / "checkpoint-session"
    session_root.mkdir()
    first = _turns(_run_protocol(
        release, ({"id": "checkpoint-first", "op": "turn",
                   "text": "浙江卫视是什么？"},
                  {"id": "checkpoint-quit", "op": "quit"}),
        session_root=session_root))
    second = _turns(_run_protocol(
        release, ({"id": "checkpoint-followup", "op": "turn",
                   "text": "它在哪里？"},
                  {"id": "checkpoint-quit-2", "op": "quit"}),
        session_root=session_root))

    conflict_root, conflict_database = _build_conflict_runtime(output)
    conflict_turns = _turns(_run_protocol(
        release, ({"id": "conflict", "op": "turn",
                   "text": "夜间模式如何影响亮度？"},
                  {"id": "conflict-quit", "op": "quit"}),
        runtime_root=conflict_root, runtime_database=conflict_database))
    cross_turns = _turns(_run_protocol(
        release, ({"id": "cross-source", "op": "turn",
                   "text": "夜间模式的操作顺序是什么？"},
                  {"id": "cross-source-quit", "op": "quit"}),
        runtime_root=runtime, runtime_database=runtime / "runtime.sqlite3"))

    cold_metrics = output / "performance-cold.json"
    _run_protocol(
        release, ({"id": "cold", "op": "turn",
                   "text": f"{questions[0]}是什么？"},
                  {"id": "cold-quit", "op": "quit"}),
        metrics_output=cold_metrics, performance_tier="strict")
    warm_metrics = output / "performance-warm.json"
    _run_protocol(
        release,
        tuple({"id": f"warm-{i}", "op": "turn",
               "text": f"{questions[i % len(questions)]}是什么？"}
              for i in range(10)) + ({"id": "warm-quit", "op": "quit"},),
        metrics_output=warm_metrics,
        performance_tier="deferred-narrow-fast")

    checks = {
        "heldout": len(heldout_turns) == len(questions)
        and all(
            _wire_answer(item, expected_title)
            for item, expected_title in zip(heldout_turns, questions)
        ),
        "unknown": len(unknown_turns) == len(unknown_questions)
        and all(_wire_unknown(item) for item in unknown_turns),
        "negative": len(negative_turns) == len(negative_questions)
        and all(_wire_unknown(item) for item in negative_turns),
        "conflict_clarify": len(conflict_turns) == 1
        and _wire_clarify(conflict_turns[0]) if conflict_turns else False,
        "cross_source_citations": len(cross_turns) == 1
        and _wire_status(cross_turns[0]) == "ANSWER"
        and len(cross_turns[0].get("citations", ())) >= 2,
        "checkpoint": len(first) == 1 and len(second) == 1
        and _wire_status(first[0]) == "ANSWER"
        and _wire_status(second[0]) == "ANSWER"
        and second[0].get("ordinal") == first[0].get("ordinal", -1) + 1,
    }
    metrics = {"cold": _metric_summary(cold_metrics),
               "warm": _metric_summary(warm_metrics)}
    checks["performance"] = all(
        type(metrics[tier]["p50_us"]) is int
        and type(metrics[tier]["p95_us"]) is int
        and metrics[tier]["p50_us"] <= metrics[tier]["p95_us"]
        and metrics[tier]["peak_working_set_bytes"] > 0
        for tier in ("cold", "warm"))
    evidence = {
        "format": EVALUATION_FORMAT,
        "schema_version": 1,
        "release_root": release.name,
        "heldout": [{"expected_source_title": expected_title,
                      "wire_status": _wire_status(item),
                      "text": _wire_text(item),
                      "source_title": _wire_source_title(item)}
                     for item, expected_title in zip(
                         heldout_turns, questions)],
        "unknown": [{"wire_status": _wire_status(item),
                     "text": _wire_text(item)} for item in unknown_turns],
        "negative": [{"wire_status": _wire_status(item),
                      "text": _wire_text(item)} for item in negative_turns],
        "conflict": [{"wire_status": _wire_status(item),
                      "text": _wire_text(item)} for item in conflict_turns],
        "cross_source": [{"wire_status": _wire_status(item),
                          "citation_count": len(item.get("citations", ())),
                          "source_title": _wire_source_title(item),
                          "text": _wire_text(item)}
                         for item in cross_turns],
        "checkpoint": {"first_ordinal": first[0].get("ordinal") if first else None,
                       "second_ordinal": second[0].get("ordinal") if second else None,
                       "first_wire_status": _wire_status(first[0]) if first else None,
                       "second_wire_status": _wire_status(second[0]) if second else None},
        "performance": metrics,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "NE",
    }
    evidence_path = output / "independent_release_evaluation.json"
    evidence_path.write_bytes(_canonical(evidence))
    (output / "independent_release_evaluation.sha256").write_text(
        hashlib.sha256(evidence_path.read_bytes()).hexdigest() + "\n",
        encoding="ascii", newline="\n")
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="evaluate an independent public release")
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--runtime-ledger", default=str(DEFAULT_RUNTIME_LEDGER))
    args = parser.parse_args(argv)
    result = evaluate(args.release_root, output_root=args.output_root,
                      runtime_ledger=args.runtime_ledger)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
