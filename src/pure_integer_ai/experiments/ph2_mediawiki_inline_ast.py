"""Deterministic, fail-closed MediaWiki inline display projection.

The parser intentionally supports only markup whose readable projection is
unambiguous without expanding templates or consulting a wiki installation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_mediawiki_multistream_adapter import (
    MediaWikiPageError,
    extract_balanced_templates,
)


MEDIAWIKI_INLINE_PARSER_VERSION = "MEDIAWIKI_INLINE_DISPLAY_V1"
MEDIAWIKI_INLINE_FAILURE_CODES = {
    "AMBIGUOUS_LINK",
    "BAD_LINK",
    "BAD_LABEL_TEMPLATE",
    "ILLEGAL_ESCAPE",
    "NESTED_MARKUP",
    "UNBALANCED_LINK",
    "UNBALANCED_TEMPLATE",
    "UNKNOWN_TEMPLATE",
    "UNSUPPORTED_LINK_TARGET",
    "UNSUPPORTED_INLINE_MARKUP",
    "UNSUPPORTED_VARIABLE",
}


# object-model: exception
class MediaWikiInlineParseError(ValueError):
    """A stable reason why source markup has no authorized projection."""

    def __init__(self, code: str, message: str) -> None:
        if code not in MEDIAWIKI_INLINE_FAILURE_CODES:
            raise ValueError("unknown MediaWiki inline failure code")
        super().__init__(message)
        self.code = code


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _span(start: int, end: int) -> None:
    if (type(start) is not int or type(end) is not int
            or start < 0 or end <= start):
        raise ValueError("MediaWiki inline node span is invalid")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class MediaWikiInlineText:
    start: int
    end: int
    text: str

    def __post_init__(self) -> None:
        _span(self.start, self.end)
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("MediaWiki text node is empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "end": self.end,
            "kind": "TEXT",
            "start": self.start,
            "text": self.text,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class MediaWikiInlineLink:
    start: int
    end: int
    target: str
    display_label: str | None

    def __post_init__(self) -> None:
        _span(self.start, self.end)
        if (not isinstance(self.target, str) or not self.target
                or self.target.strip() != self.target):
            raise ValueError("MediaWiki link target is not canonical")
        if self.display_label is not None and (
                not isinstance(self.display_label, str)
                or not self.display_label
                or self.display_label.strip() != self.display_label):
            raise ValueError("MediaWiki link display label is not canonical")

    def to_dict(self) -> dict[str, object]:
        return {
            "display_label": self.display_label,
            "end": self.end,
            "kind": "LINK",
            "start": self.start,
            "target": self.target,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class MediaWikiInlineLabel:
    start: int
    end: int
    template_name: str
    language: str
    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        _span(self.start, self.end)
        if self.template_name not in {"label", "lb"}:
            raise ValueError("MediaWiki label template name drifted")
        if (not isinstance(self.language, str) or not self.language
                or self.language.strip() != self.language):
            raise ValueError("MediaWiki label language is not canonical")
        if (not isinstance(self.labels, tuple) or not self.labels
                or any(not isinstance(item, str) or not item
                       or item.strip() != item for item in self.labels)):
            raise ValueError("MediaWiki label values are not canonical")

    def to_dict(self) -> dict[str, object]:
        return {
            "end": self.end,
            "kind": "LABEL",
            "labels": list(self.labels),
            "language": self.language,
            "start": self.start,
            "template_name": self.template_name,
        }


MediaWikiInlineNode = (
    MediaWikiInlineText | MediaWikiInlineLink | MediaWikiInlineLabel)


def _node_source(node: MediaWikiInlineNode) -> str:
    if isinstance(node, MediaWikiInlineText):
        return node.text
    if isinstance(node, MediaWikiInlineLink):
        label = "" if node.display_label is None else "|" + node.display_label
        return "[[" + node.target + label + "]]"
    if isinstance(node, MediaWikiInlineLabel):
        return "{{" + "|".join((
            node.template_name, node.language, *node.labels)) + "}}"
    raise TypeError("unknown MediaWiki inline node")


def _render_nodes(nodes: tuple[MediaWikiInlineNode, ...]) -> str:
    values = []
    for node in nodes:
        if isinstance(node, MediaWikiInlineText):
            values.append(node.text)
        elif isinstance(node, MediaWikiInlineLink):
            values.append(node.target if node.display_label is None
                          else node.display_label)
        elif isinstance(node, MediaWikiInlineLabel):
            values.append("（" + "、".join(node.labels) + "）")
        else:
            raise TypeError("unknown MediaWiki inline node")
    return "".join(values)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class MediaWikiInlineDocument:
    source_text: str
    nodes: tuple[MediaWikiInlineNode, ...]
    ast_sha256: str
    parser_version: str = MEDIAWIKI_INLINE_PARSER_VERSION

    def __post_init__(self) -> None:
        if (not isinstance(self.source_text, str) or not self.source_text
                or self.source_text.strip() != self.source_text):
            raise ValueError("MediaWiki inline source is not canonical text")
        if (not isinstance(self.nodes, tuple) or not self.nodes
                or any(not isinstance(item, (
                    MediaWikiInlineText,
                    MediaWikiInlineLink,
                    MediaWikiInlineLabel,
                )) for item in self.nodes)):
            raise ValueError("MediaWiki inline AST node inventory is invalid")
        position = 0
        for node in self.nodes:
            if node.start != position or node.end > len(self.source_text):
                raise ValueError("MediaWiki inline AST spans are not contiguous")
            if self.source_text[node.start:node.end] != _node_source(node):
                raise ValueError("MediaWiki inline AST lost source bytes")
            position = node.end
        if position != len(self.source_text):
            raise ValueError("MediaWiki inline AST does not cover source")
        if self.parser_version != MEDIAWIKI_INLINE_PARSER_VERSION:
            raise ValueError("MediaWiki inline parser version drifted")
        expected = _sha({
            "nodes": [item.to_dict() for item in self.nodes],
            "parser_version": self.parser_version,
            "source_text": self.source_text,
        })
        if self.ast_sha256 != expected:
            raise ValueError("MediaWiki inline AST commitment drifted")

    def to_dict(self) -> dict[str, object]:
        return {
            "ast_sha256": self.ast_sha256,
            "nodes": [item.to_dict() for item in self.nodes],
            "parser_version": self.parser_version,
            "source_text": self.source_text,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class MediaWikiInlineProjection:
    document: MediaWikiInlineDocument
    display_text: str
    projection_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.document, MediaWikiInlineDocument):
            raise TypeError("MediaWiki inline projection document is invalid")
        expected_display = _render_nodes(self.document.nodes)
        if (not isinstance(self.display_text, str) or not self.display_text
                or self.display_text.strip() != self.display_text
                or self.display_text != expected_display):
            raise ValueError("MediaWiki inline display projection drifted")
        expected = _sha({
            "ast_sha256": self.document.ast_sha256,
            "display_text": self.display_text,
            "parser_version": self.document.parser_version,
            "source_text": self.document.source_text,
        })
        if self.projection_sha256 != expected:
            raise ValueError("MediaWiki inline display commitment drifted")

    def to_dict(self) -> dict[str, object]:
        return {
            "ast_sha256": self.document.ast_sha256,
            "display_text": self.display_text,
            "parser_version": self.document.parser_version,
            "projection_sha256": self.projection_sha256,
            "source_text": self.document.source_text,
        }


def _reject_nested(inner: str) -> None:
    if any(marker in inner for marker in ("{{", "}}", "[[", "]]")):
        raise MediaWikiInlineParseError(
            "NESTED_MARKUP",
            "nested MediaWiki markup has no single local projection",
        )
    if any(_unsupported_inline_at(inner, index)
           for index in range(len(inner))):
        raise MediaWikiInlineParseError(
            "UNSUPPORTED_INLINE_MARKUP",
            "template or link field contains unsupported inline markup",
        )


def _unsupported_inline_at(source: str, index: int) -> bool:
    if source.startswith(("<!--", "''"), index):
        return True
    if source[index] in "[]" and not source.startswith(("[[", "]]"), index):
        return True
    if source[index] == "<" and index + 1 < len(source):
        following = source[index + 1]
        return following.isalpha() or following in "!/"
    return False


def _link_node(source: str, start: int) -> tuple[MediaWikiInlineLink, int]:
    close = source.find("]]", start + 2)
    if close < 0:
        raise MediaWikiInlineParseError(
            "UNBALANCED_LINK", "MediaWiki link is not closed")
    inner = source[start + 2:close]
    _reject_nested(inner)
    parts = inner.split("|")
    if len(parts) > 2:
        raise MediaWikiInlineParseError(
            "AMBIGUOUS_LINK", "MediaWiki link has multiple display fields")
    target = parts[0].strip()
    label = None if len(parts) == 1 else parts[1].strip()
    if (not target or target != parts[0]
            or (label is not None and (not label or label != parts[1]))):
        raise MediaWikiInlineParseError(
            "BAD_LINK", "MediaWiki link fields are not canonical")
    if ":" in target or "#" in target:
        raise MediaWikiInlineParseError(
            "UNSUPPORTED_LINK_TARGET",
            "namespaced or interwiki link requires source-specific semantics",
        )
    end = close + 2
    return MediaWikiInlineLink(start, end, target, label), end


def _template_node(
        source: str,
        start: int,
        ) -> tuple[MediaWikiInlineLabel, int]:
    if source.startswith("{{{", start):
        raise MediaWikiInlineParseError(
            "UNSUPPORTED_VARIABLE", "template variables are not expanded")
    close = source.find("}}", start + 2)
    if close < 0:
        raise MediaWikiInlineParseError(
            "UNBALANCED_TEMPLATE", "MediaWiki template is not closed")
    inner = source[start + 2:close]
    _reject_nested(inner)
    raw_parts = inner.split("|")
    parts = tuple(item.strip() for item in raw_parts)
    name = parts[0].casefold() if parts else ""
    if name not in {"label", "lb"}:
        raise MediaWikiInlineParseError(
            "UNKNOWN_TEMPLATE", "template has no authorized local renderer")
    if (len(parts) < 3 or parts[0] != name
            or any(not item for item in parts)
            or any("=" in item for item in parts[1:])
            or parts != tuple(raw_parts)):
        raise MediaWikiInlineParseError(
            "BAD_LABEL_TEMPLATE", "label template fields are ambiguous")
    end = close + 2
    return MediaWikiInlineLabel(
        start, end, name, parts[1], parts[2:]), end


def parse_mediawiki_inline(source_text: str) -> MediaWikiInlineDocument:
    """Parse the supported inline subset and retain exact source spans."""
    if (not isinstance(source_text, str) or not source_text
            or source_text.strip() != source_text):
        raise ValueError("MediaWiki inline source is not canonical text")
    try:
        extract_balanced_templates(
            source_text,
            max_templates=max(1, len(source_text) // 4 + 1),
            max_depth=64,
        )
    except MediaWikiPageError as error:
        raise MediaWikiInlineParseError(
            "UNBALANCED_TEMPLATE",
            "MediaWiki template balance check failed",
        ) from error

    nodes: list[MediaWikiInlineNode] = []
    index = 0
    text_start = 0
    while index < len(source_text):
        if (source_text[index] == "\\" and index + 1 < len(source_text)
                and source_text[index + 1] in "[]{}|\\"):
            raise MediaWikiInlineParseError(
                "ILLEGAL_ESCAPE",
                "backslash escaping of MediaWiki structure is unsupported",
            )
        if _unsupported_inline_at(source_text, index):
            raise MediaWikiInlineParseError(
                "UNSUPPORTED_INLINE_MARKUP",
                "inline markup has no authorized local renderer",
            )
        if source_text.startswith("{{", index):
            if text_start < index:
                nodes.append(MediaWikiInlineText(
                    text_start, index, source_text[text_start:index]))
            node, index = _template_node(source_text, index)
            nodes.append(node)
            text_start = index
            continue
        if source_text.startswith("[[", index):
            if text_start < index:
                nodes.append(MediaWikiInlineText(
                    text_start, index, source_text[text_start:index]))
            node, index = _link_node(source_text, index)
            nodes.append(node)
            text_start = index
            continue
        if source_text.startswith("}}", index):
            raise MediaWikiInlineParseError(
                "UNBALANCED_TEMPLATE", "unexpected template close")
        if source_text.startswith("]]", index):
            raise MediaWikiInlineParseError(
                "UNBALANCED_LINK", "unexpected link close")
        index += 1
    if text_start < len(source_text):
        nodes.append(MediaWikiInlineText(
            text_start, len(source_text), source_text[text_start:]))
    ast_sha256 = _sha({
        "nodes": [item.to_dict() for item in nodes],
        "parser_version": MEDIAWIKI_INLINE_PARSER_VERSION,
        "source_text": source_text,
    })
    return MediaWikiInlineDocument(
        source_text, tuple(nodes), ast_sha256)


def project_mediawiki_inline(source_text: str) -> MediaWikiInlineProjection:
    """Return one source-preserving readable projection or fail closed."""
    document = parse_mediawiki_inline(source_text)
    display = _render_nodes(document.nodes)
    commitment = _sha({
        "ast_sha256": document.ast_sha256,
        "display_text": display,
        "parser_version": document.parser_version,
        "source_text": document.source_text,
    })
    return MediaWikiInlineProjection(document, display, commitment)


__all__ = [
    "MEDIAWIKI_INLINE_FAILURE_CODES",
    "MEDIAWIKI_INLINE_PARSER_VERSION",
    "MediaWikiInlineDocument",
    "MediaWikiInlineLabel",
    "MediaWikiInlineLink",
    "MediaWikiInlineParseError",
    "MediaWikiInlineProjection",
    "MediaWikiInlineText",
    "parse_mediawiki_inline",
    "project_mediawiki_inline",
]
