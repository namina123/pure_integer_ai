"""目录级只读体量与重复内容审计。

该审计器面向 K/F/D 三类运行目录，不读取 SQLite 表语义，也不修改任何
输入文件。它按文件内容 SHA-256、device/inode、大小、后缀和路径角色归因
物理重复副本，并把不增加占用的硬链接别名单独报告，
用来区分一次训练的真实图体量与 model release、checkpoint、dump、query
trace 或历史 run 的物理重复。核心算法只使用 Python 标准库。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


_CHUNK = 8 * 1024 * 1024
_FORMAT = "PURE_INTEGER_RUN_STORAGE_AUDIT_V1"


def _sha256(path: Path) -> str:
    """流式计算文件摘要，不把大文件载入内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(_CHUNK)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def _location_class(path: Path) -> str:
    """按物理路径给出可解释的保存位置分类，不参与语义判断。"""
    parts = {item.casefold() for item in path.parts}
    if "current_runs" in parts:
        return "current_run"
    if "model_releases" in parts:
        return "model_release"
    if "stage_ec_training_20260903" in parts:
        return "stage_training"
    if "portable_packages" in parts:
        return "portable_package"
    if "k_cold_storage_20260903" in parts:
        return "cold_archive"
    if "archive" in parts or "pure_integer_ai_archive" in parts:
        return "archive"
    if path.drive.upper() == "K:":
        return "k_other"
    if path.drive.upper() == "F:":
        return "f_other"
    if path.drive.upper() == "D:":
        return "workspace"
    return "other"


def _file_role(path: Path) -> str:
    """按文件名/目录角色分类，保留未知项而不是强行归类。"""
    name = path.name.casefold()
    if name == "training.sqlite3":
        return "training_database"
    if name == "session.sqlite3":
        return "session_database"
    if name == "global_identity.dump":
        return "global_identity_dump"
    if name.startswith("space_") and name.endswith(".dump"):
        return "space_dump"
    if name.endswith(".sqlite3"):
        return "sqlite_database"
    if name.endswith(".json.gz") or name.endswith(".jsonl.gz"):
        return "compressed_trace_or_course"
    if name.endswith(".jsonl"):
        return "course_or_trace_jsonl"
    if name.endswith(".zip"):
        return "portable_archive"
    if name.endswith(".json"):
        return "manifest_or_report"
    if name.endswith(".int"):
        return "cursor"
    return "other"


def _iter_files(roots: tuple[Path, ...]) -> tuple[Path, ...]:
    """返回去重后的普通文件路径；符号链接不跟随。"""
    result: dict[str, Path] = {}
    for root in roots:
        root = root.resolve(strict=True)
        if not root.is_dir():
            raise ValueError(f"审计根不是目录: {root}")
        for path in root.rglob("*"):
            if path.is_file() and not path.is_symlink():
                result[str(path.resolve())] = path.resolve()
    return tuple(sorted(result.values(), key=lambda item: str(item)))


def audit_roots(roots: tuple[str | Path, ...]) -> dict[str, object]:
    """对多个运行根做内容级重复审计，返回可序列化整数/字符串报告。

    同一 device/inode 的多个路径是硬链接别名，不重复占用物理字节。
    内容相同但 inode 不同才计入 ``duplicate_excess_bytes``。
    """
    paths = tuple(Path(item) for item in roots)
    if not paths:
        raise ValueError("至少需要一个审计根")
    files = _iter_files(paths)
    entries: list[dict[str, object]] = []
    by_content: dict[tuple[int, str], list[dict[str, object]]] = {}
    physical_sizes: dict[tuple[int, int] | tuple[str, str], int] = {}
    for path in files:
        stat = path.stat()
        size = stat.st_size
        digest = _sha256(path)
        # Some filesystems report inode 0.  In that case fail closed to the
        # resolved path instead of accidentally collapsing unrelated files.
        physical_identity: tuple[int, int] | tuple[str, str] = (
            (int(stat.st_dev), int(stat.st_ino))
            if int(stat.st_ino) != 0 else ("path", str(path.resolve()))
        )
        entry = {
            "path": str(path),
            "size": size,
            "sha256": digest,
            "device": int(stat.st_dev),
            "inode": int(stat.st_ino),
            "link_count": int(stat.st_nlink),
            "location_class": _location_class(path),
            "file_role": _file_role(path),
        }
        entries.append(entry)
        by_content.setdefault((size, digest), []).append(entry)
        physical_sizes.setdefault(physical_identity, size)

    duplicate_groups = []
    hardlink_groups = []
    duplicate_bytes = 0
    duplicate_file_count = 0
    hardlink_alias_count = 0
    unique_content_bytes = 0
    for (size, digest), group in sorted(
            by_content.items(), key=lambda item: (item[0][0], item[0][1])):
        unique_content_bytes += size
        if len(group) <= 1:
            continue
        physical_copies: dict[tuple[int, int] | tuple[str, str], list[dict[str, object]]] = {}
        for item in group:
            identity = (
                (int(item["device"]), int(item["inode"]))
                if int(item["inode"]) != 0 else ("path", str(item["path"]))
            )
            physical_copies.setdefault(identity, []).append(item)
        physical_copy_count = len(physical_copies)
        group_hardlink_alias_count = len(group) - physical_copy_count
        hardlink_alias_count += group_hardlink_alias_count
        excess = size * (physical_copy_count - 1)
        common = {
            "size": size,
            "sha256": digest,
            "count": len(group),
            "physical_copy_count": physical_copy_count,
            "hardlink_alias_count": group_hardlink_alias_count,
            "locations": [item["location_class"] for item in group],
            "roles": sorted({item["file_role"] for item in group}),
            "paths": [item["path"] for item in group],
        }
        if physical_copy_count == 1:
            hardlink_groups.append(common)
            continue
        duplicate_bytes += excess
        duplicate_file_count += physical_copy_count - 1
        duplicate_groups.append({
            **common,
            "duplicate_excess_bytes": excess,
        })

    by_role: dict[str, dict[str, int]] = {}
    by_location: dict[str, dict[str, int]] = {}
    for item in entries:
        for key, name in ((by_role, str(item["file_role"])),
                          (by_location, str(item["location_class"]))):
            summary = key.setdefault(name, {"file_count": 0, "bytes": 0})
            summary["file_count"] += 1
            summary["bytes"] += int(item["size"])
    return {
        "format": _FORMAT,
        "schema_version": 1,
        "read_only": 1,
        "content_hash": "sha256",
        "roots": [str(Path(item).resolve()) for item in paths],
        "file_count": len(entries),
        "total_bytes": sum(int(item["size"]) for item in entries),
        "physical_bytes": sum(physical_sizes.values()),
        "unique_content_bytes": unique_content_bytes,
        "duplicate_excess_bytes": duplicate_bytes,
        "duplicate_excess_file_count": duplicate_file_count,
        "duplicate_group_count": len(duplicate_groups),
        "hardlink_alias_count": hardlink_alias_count,
        "hardlink_group_count": len(hardlink_groups),
        "by_role": {key: by_role[key] for key in sorted(by_role)},
        "by_location": {key: by_location[key] for key in sorted(by_location)},
        "duplicate_groups": duplicate_groups,
        "hardlink_groups": hardlink_groups,
        "files": entries,
    }


def main(argv: list[str] | None = None) -> int:
    """执行目录审计；输出文件必须位于审计根之外。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    roots = tuple(path.resolve(strict=True) for path in args.roots)
    target = args.output.resolve()
    if any(target == root or root in target.parents for root in roots):
        raise ValueError("审计报告不得写入被审计运行根")
    report = audit_roots(roots)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=True, sort_keys=True,
                                 separators=(",", ":")) + "\n", encoding="ascii")
    print(json.dumps({
        "file_count": report["file_count"],
        "total_bytes": report["total_bytes"],
        "physical_bytes": report["physical_bytes"],
        "unique_content_bytes": report["unique_content_bytes"],
        "duplicate_excess_bytes": report["duplicate_excess_bytes"],
        "duplicate_group_count": report["duplicate_group_count"],
        "hardlink_alias_count": report["hardlink_alias_count"],
        "hardlink_group_count": report["hardlink_group_count"],
    }, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["audit_roots", "main"]
