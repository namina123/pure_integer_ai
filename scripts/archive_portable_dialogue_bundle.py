"""Create a clean ZIP64 transfer archive from a verified portable bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _excluded(relative: Path) -> bool:
    return (
        "__pycache__" in relative.parts
        or relative.suffix in {".pyc", ".pyo"}
        or relative.parts and relative.parts[0] == "runtime"
    )


def archive(bundle_root: str | Path, archive_path: str | Path) -> dict[str, object]:
    root = Path(bundle_root).resolve()
    output = Path(archive_path).resolve()
    manifest_path = root / "portable_bundle_manifest.json"
    digest_path = root / "portable_bundle_manifest.json.sha256"
    if not root.is_dir() or not manifest_path.is_file() or not digest_path.is_file():
        raise ValueError("bundle_root 不是完整便携包")
    if digest_path.read_text(encoding="ascii").strip() != _sha256(manifest_path):
        raise ValueError("便携包 manifest SHA-256 漂移")
    if output.exists():
        raise FileExistsError("archive_path 已存在，禁止覆盖")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    if temporary.exists():
        raise FileExistsError("archive partial 已存在，请先人工处理")
    prefix = root.name
    file_count = 0
    try:
        with zipfile.ZipFile(
                temporary, "w", compression=zipfile.ZIP_STORED,
                allowZip64=True) as target:
            for suffix in ("runtime/", "runtime/session/"):
                target.writestr(prefix + "/" + suffix, b"")
            for path in sorted(root.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(root)
                if _excluded(relative):
                    continue
                target.write(path, (Path(prefix) / relative).as_posix())
                file_count += 1
        os.replace(temporary, output)
    except Exception:
        print(f"partial_archive={temporary}")
        raise
    digest = _sha256(output)
    checksum = output.with_name(output.name + ".sha256")
    checksum.write_text(digest + "  " + output.name + "\n",
                        encoding="ascii", newline="\n")
    return {
        "archive": str(output),
        "bytes": output.stat().st_size,
        "file_count": file_count,
        "sha256": digest,
        "status": "BUILT",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="archive portable dialogue bundle")
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--archive-path", required=True)
    args = parser.parse_args(argv)
    result = archive(args.bundle_root, args.archive_path)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
