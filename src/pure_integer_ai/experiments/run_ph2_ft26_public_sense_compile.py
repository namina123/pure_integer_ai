"""FT26 两个 source pack 与 compact public sense artifact 的构建入口。"""
from __future__ import annotations

from pathlib import Path

from pure_integer_ai.experiments.ph2_w03_public_sense_compiler import (
    build_w03_public_sense_artifact,
    write_w03_public_sense_artifact,
)
from pure_integer_ai.experiments.ph2_w03_public_sense_source_catalog import (
    FT26_PUBLIC_SENSE_SOURCE_ARTIFACT_ROOT,
    compile_ft26_public_sense_source_packs,
)


REPOSITORY = Path(__file__).resolve().parents[3]


def main() -> int:
    """从 frozen raw 构建并幂等发布 FT26 的全部公开数据产物。"""
    raw_root = REPOSITORY.parent / "ph2_dataset_raw"
    source_root = REPOSITORY / FT26_PUBLIC_SENSE_SOURCE_ARTIFACT_ROOT
    builds = compile_ft26_public_sense_source_packs(
        REPOSITORY, raw_root, source_root)
    inputs = tuple(
        (
            build.pack_root.relative_to(REPOSITORY).as_posix(),
            build.pack_root,
        )
        for _, build in builds
    )
    artifact = build_w03_public_sense_artifact(REPOSITORY, inputs)
    output = write_w03_public_sense_artifact(
        artifact,
        REPOSITORY / "data/ph2/w03_public_sense_runtime_v1.json",
    )
    print(f"SOURCE_PACK_COUNT={len(builds)}")
    print(f"ENTRY_COUNT={len(output.artifact.entries)}")
    print(f"ALIAS_COUNT={len(output.artifact.aliases)}")
    print(f"ARTIFACT_SIZE_BYTES={output.size_bytes}")
    print(f"ARTIFACT_SHA256={output.artifact_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
