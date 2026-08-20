"""DLG-05 v4 确定性中文阅读投影专项。"""
from __future__ import annotations

from pathlib import Path
import pytest

from tests.test_ph2_conversation_heldout_v4_bundle import _bundle_fixture
from pure_integer_ai.experiments.conversation_heldout_v4_projection import (
    ConversationHeldOutV4ProjectionError,
    publish_v4_projection,
    render_v4_html,
    render_v4_markdown,
)


def test_v4_reading_projection_is_deterministic_and_label_free():
    """Markdown/HTML 只从 bundle 生成，重复渲染逐字节一致。"""
    fixture, bundle, _turn, _source = _bundle_fixture()
    try:
        markdown_a = render_v4_markdown(bundle)
        markdown_b = render_v4_markdown(bundle)
        html_a = render_v4_html(bundle)
        html_b = render_v4_html(bundle)
        assert markdown_a == markdown_b
        assert html_a == html_b
        assert "公开来源正文" in markdown_a
        assert "档案显示" not in markdown_a
        assert "selected_candidate" not in markdown_a
        assert "response_act" not in markdown_a
        assert "PASS" not in markdown_a
        assert "labels" not in html_a
        assert "<script" not in html_a
    finally:
        fixture.close()


def test_v4_projection_publish_is_idempotent_and_non_overwriting(tmp_path):
    """投影首次写入后只能精确复写，漂移内容必须拒绝。"""
    fixture, bundle, _turn, _source = _bundle_fixture()
    try:
        paths = publish_v4_projection(bundle, tmp_path)
        assert tuple(path.exists() for path in paths) == (True, True)
        assert publish_v4_projection(bundle, tmp_path) == paths
        paths[0].write_text("漂移", encoding="utf-8")
        with pytest.raises(ConversationHeldOutV4ProjectionError, match="覆盖"):
            publish_v4_projection(bundle, tmp_path)
    finally:
        fixture.close()


def test_v4_projection_has_no_runtime_readback_edge():
    """生产 candidate runtime 不得导入或解析 Markdown/HTML 投影。"""
    runtime = Path(__file__).parents[1] / "src" / "pure_integer_ai" / "experiments" / "conversation_heldout_candidate_runtime.py"
    text = runtime.read_text(encoding="utf-8")
    assert "conversation_heldout_v4_projection" not in text
    assert "render_v4_markdown" not in text
    assert "render_v4_html" not in text
