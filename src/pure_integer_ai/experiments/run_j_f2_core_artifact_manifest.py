"""J-F2 Core/artifact manifest 的一次性公开发布入口。"""
from __future__ import annotations

import argparse
from pathlib import Path

from pure_integer_ai.experiments.j_f2_core_artifact_manifest import (
    MANIFEST_PATH,
    publish_core_artifact_manifest,
)


def main() -> int:
    """构建、排他发布并回读 J-F2 Core manifest。"""
    parser = argparse.ArgumentParser(description="publish J-F2 Core artifact manifest")
    parser.add_argument("repository", nargs="?", default=".")
    args = parser.parse_args()
    manifest = publish_core_artifact_manifest(Path(args.repository), target=MANIFEST_PATH)
    print(f"published={MANIFEST_PATH}")
    print(f"files={len(manifest.file_bindings)}")
    print(f"sha256={manifest.sha256()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
