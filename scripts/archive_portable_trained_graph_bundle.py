"""将训练图便携目录制作为 ZIP64 传输包。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import os
import zipfile


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def archive(bundle_root: str | Path, archive_path: str | Path) -> dict[str, object]:
    root = Path(bundle_root).resolve()
    output = Path(archive_path).resolve()
    manifest = root / "portable_bundle_manifest.json"
    digest = root / "portable_bundle_manifest.json.sha256"
    if not root.is_dir() or not manifest.is_file() or not digest.is_file():
        raise ValueError("bundle_root 不是完整便携包")
    if digest.read_text(encoding="ascii").strip() != _sha256(manifest):
        raise ValueError("便携包 manifest SHA-256 不匹配")
    if output.exists():
        raise FileExistsError("archive_path 已存在，拒绝覆盖")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    if temporary.exists():
        raise FileExistsError("archive partial 已存在")
    prefix = root.name
    count = 0
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED,
                             allowZip64=True) as target:
            for path in sorted(root.rglob("*")):
                if not path.is_file() or "__pycache__" in path.parts:
                    continue
                relative = path.relative_to(root)
                if path.suffix in {".pyc", ".pyo"}:
                    continue
                target.write(path, (Path(prefix) / relative).as_posix())
                count += 1
        os.replace(temporary, output)
    except Exception:
        print(f"partial_archive={temporary}")
        raise
    checksum = _sha256(output)
    output.with_name(output.name + ".sha256").write_text(
        checksum + "  " + output.name + "\n", encoding="ascii", newline="\n")
    return {"status": "BUILT", "archive": str(output), "file_count": count,
            "bytes": output.stat().st_size, "sha256": checksum}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="归档训练图便携包")
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--archive-path", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(archive(args.bundle_root, args.archive_path),
                     ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
