"""T1-G32：developer-only review 命令的只读端到端测试。"""
from __future__ import annotations

from io import StringIO

import pytest

from pure_integer_ai.experiments.run_raw_t1_admission_review import (
    RawT1AdmissionReviewCommandError,
    main,
    run_raw_t1_admission_review,
)


def test_public_review_command_renders_human_readable_report_without_arguments() -> None:
    output = StringIO()
    assert main([], stdout=output) == 0
    text = output.getvalue()
    assert "T1 原始文本 admission 观察报告" in text
    assert "可送独立标注审核" in text
    assert "仅保留为负例见证" in text
    assert "不代表词义、命题或现实真值" in text
    assert "设备" not in text  # 当前公开 G0 sample 不应被命令改写成训练实体


def test_review_command_is_read_only_and_rejects_path_or_unknown_arguments() -> None:
    first = run_raw_t1_admission_review()
    second = run_raw_t1_admission_review()
    assert first == second
    with pytest.raises(RawT1AdmissionReviewCommandError):
        main(["--root", "K:\\training"])
