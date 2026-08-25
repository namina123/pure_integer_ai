"""将 compact JSONL 课程切成可恢复 shard，不复制整数 sidecar 内容。

每个 shard 保存原记录的顺序子集，并复制同一个 sidecar；记录中的相对
``token_index_file``/``aggregate_index_file`` 不变。shard 仅是训练编排边界，
不改变 course identity、split 或 sidecar hash。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


class IndexedCourseSplitError(ValueError):
    """compact course 无法安全切分。"""


def split_indexed_dialogue_course(
        source: str | Path, output_dir: str | Path, *, shard_size: int,
        require_k_drive: bool = True,
        ) -> tuple[dict[str, object], ...]:
    source_path = Path(source).resolve()
    target = Path(output_dir).resolve()
    if require_k_drive and source_path.drive.upper() != "K:":
        raise IndexedCourseSplitError("source 必须位于 K 盘")
    if target.exists():
        raise IndexedCourseSplitError("output_dir 已存在，拒绝覆盖")
    if type(shard_size) is not int or shard_size < 1:
        raise IndexedCourseSplitError("shard_size 必须是正整数")
    sidecar = source_path.with_name(source_path.name + ".tokens.int.json")
    if not source_path.is_file() or not sidecar.is_file():
        raise IndexedCourseSplitError("课程或整数 sidecar 缺失")
    records: list[bytes] = []
    for line_no, raw in enumerate(source_path.read_bytes().splitlines(), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise IndexedCourseSplitError(f"课程 JSONL 非法: {line_no}") from error
        if not isinstance(value, dict):
            raise IndexedCourseSplitError(f"课程记录必须是对象: {line_no}")
        if value.get("token_index_file") != sidecar.name:
            raise IndexedCourseSplitError("课程 token_index_file 与 sidecar 不一致")
        records.append(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                  separators=(",", ":")).encode("utf-8") + b"\n")
    if not records:
        raise IndexedCourseSplitError("课程为空")
    aggregate_names = {
        json.loads(item.decode("utf-8")).get("aggregate_index_file")
        for item in records
    }
    aggregate_names.discard(None)
    if len(aggregate_names) > 1:
        raise IndexedCourseSplitError("课程 aggregate_index_file 不一致")
    aggregate_sidecar = None
    aggregate_sha = None
    if aggregate_names:
        aggregate_name = next(iter(aggregate_names))
        if not isinstance(aggregate_name, str) or not aggregate_name:
            raise IndexedCourseSplitError("课程 aggregate_index_file 非法")
        aggregate_sidecar = (source_path.parent / Path(aggregate_name)).resolve()
        try:
            aggregate_sidecar.relative_to(source_path.parent.resolve())
        except ValueError as error:
            raise IndexedCourseSplitError("aggregate sidecar 路径越界") from error
        if not aggregate_sidecar.is_file():
            raise IndexedCourseSplitError("aggregate sidecar 缺失")
        aggregate_sha = hashlib.sha256(aggregate_sidecar.read_bytes()).hexdigest()
    target.mkdir(parents=True)
    sidecar_target = target / sidecar.name
    shutil.copyfile(sidecar, sidecar_target)
    sidecar_sha = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    if aggregate_sidecar is not None:
        shutil.copyfile(aggregate_sidecar, target / aggregate_sidecar.name)
    reports: list[dict[str, object]] = []
    for shard_no, start in enumerate(range(0, len(records), shard_size), 1):
        rows = records[start:start + shard_size]
        path = target / f"{source_path.stem}.shard-{shard_no:04d}.course.jsonl"
        payload = b"".join(rows)
        path.write_bytes(payload)
        reports.append({
            "shard": shard_no,
            "path": path.as_posix(),
            "record_start": start,
            "record_count": len(rows),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "sidecar": sidecar_target.as_posix(),
            "sidecar_sha256": sidecar_sha,
        })
    manifest_value = {"format": "PURE_INTEGER_AI_INDEXED_COURSE_SHARDS_V1",
                      "source": source_path.name, "record_count": len(records),
                      "shard_size": shard_size, "sidecar": sidecar.name,
                      "sidecar_sha256": sidecar_sha, "shards": reports}
    if aggregate_sidecar is not None:
        manifest_value["aggregate_sidecar"] = aggregate_sidecar.name
        manifest_value["aggregate_sidecar_sha256"] = aggregate_sha
    (target / "shards.manifest.json").write_text(
        json.dumps(manifest_value,
                   ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8", newline="\n")
    return tuple(reports)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="split compact indexed dialogue course")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-size", type=int, default=5000)
    args = parser.parse_args(argv)
    print(json.dumps(split_indexed_dialogue_course(
        args.source, args.output_dir, shard_size=args.shard_size),
        ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["IndexedCourseSplitError", "split_indexed_dialogue_course", "main"]
