"""J-F2 final joint seal 的一次性公开发布入口。"""
from __future__ import annotations

import argparse
from pathlib import Path

from pure_integer_ai.experiments.j_f2_final_joint_seal import (
    SEAL_PATH,
    publish_final_joint_seal,
)


def main() -> int:
    """构建、排他发布并回读唯一 final joint seal。"""
    parser = argparse.ArgumentParser(description="publish J-F2 final joint seal")
    parser.add_argument("repository", nargs="?", default=".")
    args = parser.parse_args()
    seal = publish_final_joint_seal(Path(args.repository), target=SEAL_PATH)
    print(f"published={SEAL_PATH}")
    print(f"dependencies={len(seal.dependency_bindings)}")
    print(f"hard_conjuncts={len(seal.hard_conjuncts)}")
    print(f"sha256={seal.sha256()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
