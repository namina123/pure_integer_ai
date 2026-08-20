"""DLG-05 v4 的确定性中文 Markdown/HTML 阅读投影。

投影只读 ``ConversationHeldOutV4SourceBundle``，用于独立 owner 阅读完整问题、
所有候选、Evidence/SourceRef 链和来源原文。它不含标签字段、不作选择，不是
SQLite/Core/Memory 的输入；生产 runtime 没有反向解析本模块输出的接口。
"""
from __future__ import annotations

from html import escape
from pathlib import Path

from pure_integer_ai.experiments.conversation_heldout_v4_bundle import (
    ConversationHeldOutV4SourceBundle,
    ConversationHeldOutV4SourceRecord,
    digest_hex,
    scalars_text,
)


class ConversationHeldOutV4ProjectionError(ValueError):
    """阅读投影输入不完整或无法保持确定性。"""


def _write_once(path: Path, payload: bytes) -> Path:
    """只写新文件或精确复写同一字节，不允许覆盖漂移投影。"""
    if path.exists():
        if path.read_bytes() != payload:
            raise ConversationHeldOutV4ProjectionError(
                "阅读投影已存在且内容不同，不允许覆盖")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _key_text(value: tuple[int, ...]) -> str:
    """把完整整数键按固定逗号格式变成可读文本。"""
    return "(" + ",".join(str(item) for item in value) + ")"


def _source_markdown(source: ConversationHeldOutV4SourceRecord) -> list[str]:
    """渲染单条来源的原文、许可、归属和完整 SourceRef。"""
    return [
        f"### 来源 {_key_text(source.source_key)}",
        f"- 原文：{source.raw_text}",
        f"- 内容 SHA-256：`{digest_hex(source.content_sha256)}`",
        f"- 许可：{scalars_text(source.license_scalars, allow_empty=False)}",
        f"- 归属：{scalars_text(source.attribution_scalars, allow_empty=False)}",
        f"- 官方来源：{scalars_text(source.source_uri_scalars)}",
    ]


def render_v4_markdown(bundle: ConversationHeldOutV4SourceBundle) -> str:
    """确定性生成中文 Markdown 阅读投影，不携带任何答案标签。"""
    if not isinstance(bundle, ConversationHeldOutV4SourceBundle):
        raise TypeError("bundle 必须是 ConversationHeldOutV4SourceBundle")
    lines = [
        "# DLG-05 v4 无标签阅读材料",
        "",
        "本文件由完整整数 source bundle 确定性生成，只供独立阅读；不包含评测标签、",
        "选中候选或通过/失败结论，也不是运行时存储。",
        "",
        f"- Family：`{_key_text(bundle.family_key.components)}`",
        f"- Bundle payload size：`{bundle.payload_size}`",
        f"- Bundle payload SHA-256：`{digest_hex(bundle.payload_sha256)}`",
        "",
        "## 回合",
    ]
    for turn in sorted(bundle.turns, key=lambda item: (
            item.case_key.components, item.turn_key.components)):
        lines.extend([
            "",
            f"### Case `{_key_text(turn.case_key.components)}` / Turn `"
            f"{_key_text(turn.turn_key.components)}`",
            f"- 输入 Representation：`{len(turn.representations)}` 个",
            f"- 输入表面：{''.join(item.text for item in turn.representations)}",
            f"- 候选数量：`{len(turn.candidates)}`（仅列出候选，不作选择）",
        ])
        for index, candidate in enumerate(turn.candidates, start=1):
            lines.extend([
                "",
                f"#### 候选 {index}",
                f"- 候选身份：`{_key_text(candidate.candidate_key)}`",
                f"- 候选表面：{scalars_text(candidate.surface_scalars, allow_empty=False)}",
                "- 候选表面 Representation："
                + "；".join(
                    f"`{_key_text(item.representation.stable_key())}`={item.text}"
                    for item in turn.surface_representations
                    if item.representation in set(candidate.surface_representations)
                ),
                f"- Evidence 数量：`{len(candidate.evidence)}`",
                f"- SourceRef 链：`{len(candidate.source_chain)}` 条",
            ])
            for evidence in candidate.evidence:
                lines.append(
                    f"- Evidence：`{_key_text(evidence.stable_key())}`；来源："
                    f"`{_key_text(evidence.source.stable_key())}`")
        lines.append(f"- 回合来源数：`{len(turn.source_keys)}`")
    lines.extend(["", "## 来源原文", ""])
    for source in sorted(bundle.sources, key=lambda item: item.source_key):
        lines.extend(_source_markdown(source))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_v4_html(bundle: ConversationHeldOutV4SourceBundle) -> str:
    """确定性生成中文 HTML 阅读投影，所有可读文本均经 HTML escaping。"""
    # 这里仅建立稳定、无脚本的阅读边界；运行时没有 HTML 解析/回读路径。
    sections = [
        '<!doctype html>',
        '<html lang="zh-CN"><head><meta charset="utf-8">',
        "<title>DLG-05 v4 无标签阅读材料</title></head><body>",
        "<h1>DLG-05 v4 无标签阅读材料</h1>",
        f"<p>Bundle payload SHA-256：<code>{escape(digest_hex(bundle.payload_sha256))}</code></p>",
    ]
    for turn in sorted(bundle.turns, key=lambda item: (
            item.case_key.components, item.turn_key.components)):
        sections.append(
            f"<section><h2>Case <code>{escape(_key_text(turn.case_key.components))}"
            f"</code> / Turn <code>{escape(_key_text(turn.turn_key.components))}</code></h2>"
            f"<p>输入表面：{escape(''.join(item.text for item in turn.representations))}</p>")
        sections.append("<h3>全部候选</h3><ol>")
        for candidate in turn.candidates:
            sections.append(
                f"<li><p>候选表面：{escape(scalars_text(candidate.surface_scalars, allow_empty=False))}</p>"
                f"<p>候选身份：<code>{escape(_key_text(candidate.candidate_key))}</code></p>"
                f"<p>候选表面 Representation：{escape('；'.join(
                    f'({_key_text(item.representation.stable_key())})={item.text}'
                    for item in turn.surface_representations
                    if item.representation in set(candidate.surface_representations)
                ))}</p>"
                f"<p>Evidence 数量：{len(candidate.evidence)}；SourceRef 链："
                f"{len(candidate.source_chain)} 条</p></li>")
        sections.append("</ol></section>")
    sections.append("<h2>来源原文</h2>")
    for source in sorted(bundle.sources, key=lambda item: item.source_key):
        sections.append(
            f"<article><h3>来源 <code>{escape(_key_text(source.source_key))}</code></h3>"
            f"<p>原文：{escape(source.raw_text)}</p>"
            f"<p>许可：{escape(scalars_text(source.license_scalars, allow_empty=False))}；"
            f"归属：{escape(scalars_text(source.attribution_scalars, allow_empty=False))}；"
            f"官方来源：{escape(scalars_text(source.source_uri_scalars))}</p></article>")
    sections.append("</body></html>\n")
    return "".join(sections)


def publish_v4_projection(
        bundle: ConversationHeldOutV4SourceBundle,
        target_directory: str | Path,
        ) -> tuple[Path, Path]:
    """在指定外部目录一次性发布 Markdown/HTML，只允许幂等同字节复写。

    该函数只写阅读边界文件，不把文件作为 bundle 输入，也不提供回读解析器。
    """
    if not isinstance(bundle, ConversationHeldOutV4SourceBundle):
        raise TypeError("bundle 必须是 ConversationHeldOutV4SourceBundle")
    root = Path(target_directory).resolve()
    markdown_path = root / "dlg05_v4_reading.md"
    html_path = root / "dlg05_v4_reading.html"
    markdown = render_v4_markdown(bundle).encode("utf-8")
    html = render_v4_html(bundle).encode("utf-8")
    return _write_once(markdown_path, markdown), _write_once(html_path, html)


__all__ = [
    "ConversationHeldOutV4ProjectionError",
    "render_v4_html",
    "render_v4_markdown",
    "publish_v4_projection",
]
