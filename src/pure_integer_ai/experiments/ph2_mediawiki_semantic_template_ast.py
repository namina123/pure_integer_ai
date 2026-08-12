"""Source-preserving projection for the narrow templates authorized by FT34."""
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


MEDIAWIKI_SEMANTIC_TEMPLATE_PARSER_VERSION = (
    "MEDIAWIKI_SEMANTIC_TEMPLATE_DISPLAY_FT34_V1")
MEDIAWIKI_SEMANTIC_TEMPLATE_FAILURE_CODES = frozenset({
    "BAD_LINK",
    "BAD_TEMPLATE_PROFILE",
    "BLOCKED_TEMPLATE",
    "DUPLICATE_PARAMETER",
    "MAINTENANCE_TEMPLATE",
    "NESTED_MARKUP",
    "UNBALANCED_LINK",
    "UNBALANCED_TEMPLATE",
    "UNKNOWN_TEMPLATE",
    "UNSUPPORTED_INLINE_MARKUP",
    "UNSUPPORTED_LINK_TARGET",
    "UNSUPPORTED_VARIABLE",
})


# object-model: exception
class MediaWikiSemanticTemplateParseError(ValueError):
    """A stable reason why FT34 cannot project source markup."""

    def __init__(self, code: str, message: str) -> None:
        if code not in MEDIAWIKI_SEMANTIC_TEMPLATE_FAILURE_CODES:
            raise ValueError("unknown semantic-template failure code")
        super().__init__(message)
        self.code = code


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _span(start: int, end: int) -> None:
    if (type(start) is not int or type(end) is not int
            or start < 0 or end <= start):
        raise ValueError("semantic inline node span is invalid")


def _canonical_field(value: str, *, where: str) -> str:
    if not value or value.strip() != value:
        raise MediaWikiSemanticTemplateParseError(
            "BAD_TEMPLATE_PROFILE", f"{where} is not canonical")
    return value


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class MediaWikiSemanticText:
    start: int
    end: int
    text: str

    def __post_init__(self) -> None:
        _span(self.start, self.end)
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("semantic text node is empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "end": self.end,
            "kind": "TEXT",
            "start": self.start,
            "text": self.text,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class MediaWikiSemanticLink:
    start: int
    end: int
    target: str
    display_label: str | None

    def __post_init__(self) -> None:
        _span(self.start, self.end)
        if (not isinstance(self.target, str) or not self.target
                or self.target.strip() != self.target):
            raise ValueError("semantic link target is invalid")
        if self.display_label is not None and (
                not self.display_label
                or self.display_label.strip() != self.display_label):
            raise ValueError("semantic link label is invalid")

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
class MediaWikiSemanticLabel:
    start: int
    end: int
    raw_source: str
    template_name: str
    language: str
    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        _span(self.start, self.end)
        if (self.template_name not in {"label", "lb"}
                or not self.language or not self.labels
                or any(not item for item in self.labels)):
            raise ValueError("semantic label node is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "end": self.end,
            "kind": "LABEL",
            "labels": list(self.labels),
            "language": self.language,
            "raw_source": self.raw_source,
            "start": self.start,
            "template_name": self.template_name,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class MediaWikiSemanticReference:
    start: int
    end: int
    raw_source: str
    template_name: str
    relation_kind: str
    language: str
    target: str | None
    transliteration: str | None
    gloss_source: str | None
    gloss_display: str | None

    def __post_init__(self) -> None:
        _span(self.start, self.end)
        allowed = {
            "ALTERNATIVE_FORM", "LEXICAL_CLASS", "SYNONYM"}
        if (self.relation_kind not in allowed or not self.raw_source
                or not self.template_name or not self.language):
            raise ValueError("semantic reference identity is invalid")
        if self.relation_kind == "LEXICAL_CLASS":
            if (self.target is not None or self.transliteration is not None
                    or self.gloss_source is not None
                    or self.gloss_display is not None):
                raise ValueError("lexical-class node has target metadata")
        elif not self.target:
            raise ValueError("semantic reference target is missing")
        if (self.gloss_source is None) != (self.gloss_display is None):
            raise ValueError("semantic reference gloss pair drifted")

    def to_dict(self) -> dict[str, object]:
        return {
            "end": self.end,
            "gloss_display": self.gloss_display,
            "gloss_source": self.gloss_source,
            "kind": "SEMANTIC_REFERENCE",
            "language": self.language,
            "raw_source": self.raw_source,
            "relation_kind": self.relation_kind,
            "start": self.start,
            "target": self.target,
            "template_name": self.template_name,
            "transliteration": self.transliteration,
        }


MediaWikiSemanticNode = (
    MediaWikiSemanticText | MediaWikiSemanticLink
    | MediaWikiSemanticLabel | MediaWikiSemanticReference)


def _node_source(node: MediaWikiSemanticNode) -> str:
    if isinstance(node, MediaWikiSemanticText):
        return node.text
    if isinstance(node, MediaWikiSemanticLink):
        label = "" if node.display_label is None else "|" + node.display_label
        return "[[" + node.target + label + "]]"
    if isinstance(node, (MediaWikiSemanticLabel, MediaWikiSemanticReference)):
        return node.raw_source
    raise TypeError("unknown semantic inline node")


def _render_reference(node: MediaWikiSemanticReference) -> str:
    if node.relation_kind == "LEXICAL_CLASS":
        return "姓氏"
    assert node.target is not None
    if node.relation_kind == "SYNONYM":
        return node.target + "之同義詞"
    text = node.target + "的另一種寫法"
    details = []
    if node.transliteration not in {None, "-"}:
        details.append("轉寫：" + node.transliteration)
    if node.gloss_display is not None:
        details.append("義：" + node.gloss_display)
    if details:
        text += "（" + "；".join(details) + "）"
    return text


def _render_nodes(nodes: tuple[MediaWikiSemanticNode, ...]) -> str:
    values = []
    for node in nodes:
        if isinstance(node, MediaWikiSemanticText):
            values.append(node.text)
        elif isinstance(node, MediaWikiSemanticLink):
            values.append(node.target if node.display_label is None
                          else node.display_label)
        elif isinstance(node, MediaWikiSemanticLabel):
            values.append("（" + "、".join(node.labels) + "）")
        elif isinstance(node, MediaWikiSemanticReference):
            values.append(_render_reference(node))
        else:
            raise TypeError("unknown semantic inline node")
    return "".join(values)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class MediaWikiSemanticDocument:
    source_text: str
    nodes: tuple[MediaWikiSemanticNode, ...]
    ast_sha256: str
    parser_version: str = MEDIAWIKI_SEMANTIC_TEMPLATE_PARSER_VERSION

    def __post_init__(self) -> None:
        if (not isinstance(self.source_text, str) or not self.source_text
                or self.source_text.strip() != self.source_text
                or not self.nodes):
            raise ValueError("semantic document source is invalid")
        position = 0
        for node in self.nodes:
            if node.start != position or node.end > len(self.source_text):
                raise ValueError("semantic document spans are not contiguous")
            if self.source_text[node.start:node.end] != _node_source(node):
                raise ValueError("semantic document lost source text")
            position = node.end
        if position != len(self.source_text):
            raise ValueError("semantic document does not cover source")
        expected = _sha({
            "nodes": [item.to_dict() for item in self.nodes],
            "parser_version": self.parser_version,
            "source_text": self.source_text,
        })
        if self.parser_version != MEDIAWIKI_SEMANTIC_TEMPLATE_PARSER_VERSION:
            raise ValueError("semantic document parser version drifted")
        if self.ast_sha256 != expected:
            raise ValueError("semantic document commitment drifted")

    def to_dict(self) -> dict[str, object]:
        return {
            "ast_sha256": self.ast_sha256,
            "nodes": [item.to_dict() for item in self.nodes],
            "parser_version": self.parser_version,
            "source_text": self.source_text,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class MediaWikiSemanticProjection:
    document: MediaWikiSemanticDocument
    display_text: str
    projection_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.document, MediaWikiSemanticDocument):
            raise TypeError("semantic projection document is invalid")
        expected_display = _render_nodes(self.document.nodes)
        if (not self.display_text or self.display_text.strip() != self.display_text
                or self.display_text != expected_display):
            raise ValueError("semantic projection display drifted")
        expected = _sha({
            "ast_sha256": self.document.ast_sha256,
            "display_text": self.display_text,
            "parser_version": self.document.parser_version,
            "source_text": self.document.source_text,
        })
        if self.projection_sha256 != expected:
            raise ValueError("semantic projection commitment drifted")

    def to_dict(self) -> dict[str, object]:
        return {
            "ast_sha256": self.document.ast_sha256,
            "display_text": self.display_text,
            "parser_version": self.document.parser_version,
            "projection_sha256": self.projection_sha256,
            "source_text": self.document.source_text,
        }


def _top_level_parts(inner: str) -> tuple[str, ...]:
    parts = []
    start = 0
    template_depth = 0
    link_depth = 0
    index = 0
    while index < len(inner):
        if inner.startswith("{{{", index):
            raise MediaWikiSemanticTemplateParseError(
                "UNSUPPORTED_VARIABLE", "template variables are unsupported")
        if inner.startswith("{{", index):
            template_depth += 1
            index += 2
            continue
        if inner.startswith("}}", index):
            if template_depth <= 0:
                raise MediaWikiSemanticTemplateParseError(
                    "UNBALANCED_TEMPLATE", "nested template close is invalid")
            template_depth -= 1
            index += 2
            continue
        if inner.startswith("[[", index):
            link_depth += 1
            index += 2
            continue
        if inner.startswith("]]", index):
            if link_depth <= 0:
                raise MediaWikiSemanticTemplateParseError(
                    "UNBALANCED_LINK", "nested link close is invalid")
            link_depth -= 1
            index += 2
            continue
        if inner[index] == "|" and template_depth == link_depth == 0:
            parts.append(inner[start:index])
            start = index + 1
        index += 1
    if template_depth or link_depth:
        raise MediaWikiSemanticTemplateParseError(
            "NESTED_MARKUP", "template field nesting is unbalanced")
    parts.append(inner[start:])
    return tuple(parts)


def _named_parameters(
        values: tuple[str, ...]) -> tuple[tuple[str, ...], dict[str, str]]:
    positional = []
    named: dict[str, str] = {}
    named_started = False
    for raw in values:
        if "=" not in raw:
            if named_started:
                raise MediaWikiSemanticTemplateParseError(
                    "BAD_TEMPLATE_PROFILE",
                    "positional parameter follows a named parameter")
            positional.append(_canonical_field(raw, where="positional field"))
            continue
        named_started = True
        key, value = raw.split("=", 1)
        key = _canonical_field(key, where="parameter name")
        value = _canonical_field(value, where="parameter value")
        if key in named:
            raise MediaWikiSemanticTemplateParseError(
                "DUPLICATE_PARAMETER", "named parameter is duplicated")
        named[key] = value
    return tuple(positional), named


def _simple_term(value: str, *, where: str) -> str:
    value = _canonical_field(value, where=where)
    if any(marker in value for marker in (
            "{{", "}}", "[[", "]]", "<", ">", "''")):
        raise MediaWikiSemanticTemplateParseError(
            "NESTED_MARKUP", f"{where} contains unsupported markup")
    return value


def _project_gloss(value: str) -> str:
    nodes = _parse_fragment(value, offset=0, allow_templates=False)
    display = _render_nodes(nodes)
    if not display or display.strip() != display:
        raise MediaWikiSemanticTemplateParseError(
            "BAD_TEMPLATE_PROFILE", "gloss projection is not canonical")
    return display


def _template_node(
        raw: str, *, start: int, end: int) -> MediaWikiSemanticNode:
    parts = _top_level_parts(raw[2:-2])
    if not parts:
        raise MediaWikiSemanticTemplateParseError(
            "BAD_TEMPLATE_PROFILE", "template name is missing")
    name = _canonical_field(parts[0], where="template name")
    if name in {"label", "lb"}:
        positional, named = _named_parameters(parts[1:])
        if named or len(positional) < 2:
            raise MediaWikiSemanticTemplateParseError(
                "BAD_TEMPLATE_PROFILE", "label profile is unsupported")
        language = _simple_term(positional[0], where="label language")
        labels = tuple(
            _simple_term(item, where="label value")
            for item in positional[1:])
        return MediaWikiSemanticLabel(
            start, end, raw, name, language, labels)
    if name == "rfdef":
        raise MediaWikiSemanticTemplateParseError(
            "MAINTENANCE_TEMPLATE", "rfdef is not lexical content")
    if name == "†":
        raise MediaWikiSemanticTemplateParseError(
            "BLOCKED_TEMPLATE", "dagger template identity is absent")
    if name not in {"alt form", "surname", "syn of", "zh-alt-form"}:
        raise MediaWikiSemanticTemplateParseError(
            "UNKNOWN_TEMPLATE", "template has no FT34 authorization")
    positional, named = _named_parameters(parts[1:])
    if name == "surname":
        if positional != ("zh",) or named:
            raise MediaWikiSemanticTemplateParseError(
                "BAD_TEMPLATE_PROFILE", "surname profile is not authorized")
        return MediaWikiSemanticReference(
            start, end, raw, name, "LEXICAL_CLASS", "zh",
            None, None, None, None)
    if name == "syn of":
        if len(positional) != 2 or positional[0] != "zh" or named:
            raise MediaWikiSemanticTemplateParseError(
                "BAD_TEMPLATE_PROFILE", "synonym profile is not authorized")
        target = _simple_term(positional[1], where="synonym target")
        return MediaWikiSemanticReference(
            start, end, raw, name, "SYNONYM", "zh",
            target, None, None, None)
    if name == "zh-alt-form":
        if len(positional) != 1 or named:
            raise MediaWikiSemanticTemplateParseError(
                "BAD_TEMPLATE_PROFILE", "zh-alt-form profile is not authorized")
        target = _simple_term(positional[0], where="alternative-form target")
        return MediaWikiSemanticReference(
            start, end, raw, name, "ALTERNATIVE_FORM", "zh",
            target, None, None, None)
    if (len(positional) != 2 or positional[0] != "zh"
            or not set(named).issubset({"t", "tr"})):
        raise MediaWikiSemanticTemplateParseError(
            "BAD_TEMPLATE_PROFILE", "alt-form profile is not authorized")
    target = _simple_term(positional[1], where="alternative-form target")
    transliteration = None
    if "tr" in named:
        transliteration = _simple_term(
            named["tr"], where="explicit transliteration")
    gloss_source = named.get("t")
    gloss_display = (
        None if gloss_source is None else _project_gloss(gloss_source))
    return MediaWikiSemanticReference(
        start, end, raw, name, "ALTERNATIVE_FORM", "zh",
        target, transliteration, gloss_source, gloss_display)


def _link_node(source: str, start: int) -> tuple[MediaWikiSemanticLink, int]:
    close = source.find("]]", start + 2)
    if close < 0:
        raise MediaWikiSemanticTemplateParseError(
            "UNBALANCED_LINK", "MediaWiki link is not closed")
    inner = source[start + 2:close]
    if any(marker in inner for marker in ("{{", "}}", "[[", "]]")):
        raise MediaWikiSemanticTemplateParseError(
            "NESTED_MARKUP", "nested link markup is unsupported")
    parts = inner.split("|")
    if len(parts) > 2:
        raise MediaWikiSemanticTemplateParseError(
            "BAD_LINK", "MediaWiki link has too many fields")
    target = _canonical_field(parts[0], where="link target")
    label = None if len(parts) == 1 else _canonical_field(
        parts[1], where="link display")
    if ":" in target or "#" in target:
        raise MediaWikiSemanticTemplateParseError(
            "UNSUPPORTED_LINK_TARGET", "link target requires wiki state")
    end = close + 2
    return MediaWikiSemanticLink(start, end, target, label), end


def _parse_fragment(
        source: str, *, offset: int,
        allow_templates: bool) -> tuple[MediaWikiSemanticNode, ...]:
    try:
        spans = extract_balanced_templates(
            source,
            max_templates=max(1, len(source) // 4 + 1),
            max_depth=64,
        )
    except MediaWikiPageError as error:
        raise MediaWikiSemanticTemplateParseError(
            "UNBALANCED_TEMPLATE", "template balance check failed") from error
    templates = {item.start: item for item in spans}
    nodes: list[MediaWikiSemanticNode] = []
    index = 0
    text_start = 0
    while index < len(source):
        if source.startswith("{{{", index):
            raise MediaWikiSemanticTemplateParseError(
                "UNSUPPORTED_VARIABLE", "template variables are unsupported")
        if source.startswith("{{", index):
            span = templates.get(index)
            if span is None:
                raise MediaWikiSemanticTemplateParseError(
                    "NESTED_MARKUP", "nested template is unsupported")
            if not allow_templates:
                raise MediaWikiSemanticTemplateParseError(
                    "NESTED_MARKUP", "template is not permitted in this field")
            if text_start < index:
                nodes.append(MediaWikiSemanticText(
                    offset + text_start, offset + index,
                    source[text_start:index]))
            raw = source[index:span.end]
            nodes.append(_template_node(
                raw, start=offset + index, end=offset + span.end))
            index = span.end
            text_start = index
            continue
        if source.startswith("[[", index):
            if text_start < index:
                nodes.append(MediaWikiSemanticText(
                    offset + text_start, offset + index,
                    source[text_start:index]))
            node, end = _link_node(source, index)
            nodes.append(MediaWikiSemanticLink(
                offset + node.start, offset + node.end,
                node.target, node.display_label))
            index = end
            text_start = index
            continue
        if source.startswith(("}}", "]]"), index):
            raise MediaWikiSemanticTemplateParseError(
                "UNBALANCED_TEMPLATE" if source.startswith("}}", index)
                else "UNBALANCED_LINK", "unexpected closing markup")
        if source.startswith(("<!--", "''"), index) or (
                source[index] == "<" and index + 1 < len(source)):
            raise MediaWikiSemanticTemplateParseError(
                "UNSUPPORTED_INLINE_MARKUP", "inline markup is unsupported")
        index += 1
    if text_start < len(source):
        nodes.append(MediaWikiSemanticText(
            offset + text_start, offset + len(source), source[text_start:]))
    return tuple(nodes)


def parse_mediawiki_semantic_templates(
        source_text: str) -> MediaWikiSemanticDocument:
    """Parse only FT34-authorized profiles while preserving every source char."""
    if (not isinstance(source_text, str) or not source_text
            or source_text.strip() != source_text):
        raise ValueError("semantic template source is not canonical text")
    nodes = _parse_fragment(source_text, offset=0, allow_templates=True)
    ast_sha256 = _sha({
        "nodes": [item.to_dict() for item in nodes],
        "parser_version": MEDIAWIKI_SEMANTIC_TEMPLATE_PARSER_VERSION,
        "source_text": source_text,
    })
    return MediaWikiSemanticDocument(source_text, nodes, ast_sha256)


def project_mediawiki_semantic_templates(
        source_text: str) -> MediaWikiSemanticProjection:
    """Return a deterministic semantic display or fail closed."""
    document = parse_mediawiki_semantic_templates(source_text)
    display = _render_nodes(document.nodes)
    commitment = _sha({
        "ast_sha256": document.ast_sha256,
        "display_text": display,
        "parser_version": document.parser_version,
        "source_text": document.source_text,
    })
    return MediaWikiSemanticProjection(document, display, commitment)


__all__ = [
    "MEDIAWIKI_SEMANTIC_TEMPLATE_FAILURE_CODES",
    "MEDIAWIKI_SEMANTIC_TEMPLATE_PARSER_VERSION",
    "MediaWikiSemanticDocument",
    "MediaWikiSemanticLabel",
    "MediaWikiSemanticLink",
    "MediaWikiSemanticProjection",
    "MediaWikiSemanticReference",
    "MediaWikiSemanticTemplateParseError",
    "MediaWikiSemanticText",
    "parse_mediawiki_semantic_templates",
    "project_mediawiki_semantic_templates",
]
