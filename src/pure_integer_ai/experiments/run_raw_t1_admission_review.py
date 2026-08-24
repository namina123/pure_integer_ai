"""T1-G32：公开样例的 developer-only admission review 只读命令。

运行：
``py -3.11 -m pure_integer_ai.experiments.run_raw_t1_admission_review``

该入口只读取仓库中的公开 sample，打印中文观察报告，不接默认 terminal、不调用 LLM、不写盘。
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import TextIO

from pure_integer_ai.experiments.conversation_raw_t1_admission_review import (
    build_raw_t1_admission_review,
    render_raw_t1_admission_review_zh,
)
from pure_integer_ai.experiments.conversation_raw_t1_training_admission import (
    admit_raw_t1_training_pack,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_PUBLIC_SAMPLE_NAMES = (
    "dlg_raw_text_observation_v1.jsonl.sample",
    "dlg_raw_lexical_evidence_v1.jsonl.sample",
    "dlg_raw_proposition_relation_evidence_v1.jsonl.sample",
    "dlg_raw_proposition_qualification_v1.jsonl.sample",
)


class RawT1AdmissionReviewCommandError(ValueError):
    """developer-only review command 无法读取完整公开样例。"""


def _sample_payloads(repository_root: Path) -> tuple[bytes, ...]:
    if not isinstance(repository_root, Path):
        raise TypeError("repository_root 必须是 Path")
    root = repository_root.resolve()
    if not root.is_dir():
        raise RawT1AdmissionReviewCommandError("repository root 不存在")
    data_root = (root / "data" / "ph2").resolve()
    try:
        data_root.relative_to(root)
    except ValueError as error:
        raise RawT1AdmissionReviewCommandError("公开样例路径逃逸") from error
    payloads = []
    for name in _PUBLIC_SAMPLE_NAMES:
        path = (data_root / name).resolve()
        try:
            path.relative_to(data_root)
        except ValueError as error:
            raise RawT1AdmissionReviewCommandError("公开样例路径非法") from error
        if not path.is_file():
            raise RawT1AdmissionReviewCommandError(f"公开样例缺失：{name}")
        payloads.append(path.read_bytes())
    return tuple(payloads)


def run_raw_t1_admission_review(
        *,
        repository_root: Path = _REPOSITORY_ROOT,
        ) -> str:
    """读取公开样例并返回中文只读报告，不写入任何介质。"""
    payloads = _sample_payloads(repository_root)
    admission = admit_raw_t1_training_pack(*payloads)
    review = build_raw_t1_admission_review(admission)
    return render_raw_t1_admission_review_zh(review)


def main(
        argv: list[str] | None = None,
        *,
        stdout: TextIO | None = None,
        ) -> int:
    """developer-only 命令入口；不接受路径参数，避免绕过公开样例边界。"""
    values = sys.argv[1:] if argv is None else argv
    target = sys.stdout if stdout is None else stdout
    if values:
        if values in (["-h"], ["--help"]):
            target.write("pure-integer-t1-review: 公开样例只读 admission 观察\n")
            return 0
        raise RawT1AdmissionReviewCommandError("该 developer-only 命令不接受参数")
    target.write(run_raw_t1_admission_review())
    target.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RawT1AdmissionReviewCommandError",
    "main",
    "run_raw_t1_admission_review",
]
