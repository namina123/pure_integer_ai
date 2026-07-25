"""词形目录的权威 Representation 路径和 legacy 兼容路径。

新路径以 LanguageBranch 到 Unicode Representation 的图内 statement 作为权威事实；
旧 ``NODE_WORD`` 表仅供迁移和尚未接线的 caller 使用，不再定义语言原子或表示身份。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.graph_ontology import (
    GraphOntology,
    relation_concept_identity,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_LANGUAGE_BRANCH,
    OBJECT_REPRESENTATION,
    ObjectIdentity,
    TypedRef,
    representation_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.unicode_representation import (
    validate_unicode_scalars,
)
from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.crosscut.integer.unicode_codec import decode, encode
from pure_integer_ai.storage.abstract_mark import MARK_LANG, set_mark
from pure_integer_ai.storage.concept_correspondence import (
    CORR_ORDINAL,
    record_correspondence,
)
from pure_integer_ai.storage.node_store import (
    NODE_CONCEPT,
    NODE_WORD,
    TIER_PRIMARY,
)
from pure_integer_ai.storage.word_form_index import (
    load_word_forms,
    record_legacy_word_form_bridge,
    record_word_form,
)

_LEGACY_WORD_IDENTITY = Hasher("pure_integer_ai.word_form.v1")


class WordFormIdentityAmbiguity(ValueError):
    """同一目录键命中多个不同对象时拒绝静默选取。"""


@dataclass(frozen=True)
class WordFormMigration:
    """一次 legacy 迁移产生的权威表示和全部旧节点桥。"""

    representation: TypedRef
    legacy_refs: tuple[tuple[int, int], ...]


class WordFormIndex:
    """提供图内权威词形目录，并隔离旧按语言 surface 目录。"""

    def __init__(self, backend, concept_index=None, *,
                 ontology: GraphOntology | None = None,
                 unicode_family_key: tuple[int, ...] | None = None,
                 inventory_relation_key: tuple[int, ...] | None = None) -> None:
        self._backend = backend
        self._concept_index = concept_index
        self._ontology = ontology
        self._unicode_family_key = unicode_family_key
        self._inventory_relation_key = inventory_relation_key
        self._legacy_cache: dict[
            tuple[int, int],
            dict[tuple[int, ...], tuple[int, int]],
        ] = {}
        self._authoritative_cache: dict[
            TypedRef, dict[tuple[int, ...], TypedRef],
        ] = {}
        self._course_catalogs: dict[
            TypedRef, tuple[tuple[int, ...], frozenset[tuple[int, ...]]],
        ] = {}
        self._segment_cache: dict[
            tuple[str, object],
            tuple[tuple[int, ...], dict[int, set[tuple[int, ...]]]],
        ] = {}

    def ensure(self, surface: str, *, branch: TypedRef,
               scope: ScopeIdentity, provenance_kind: int,
               epistemic_origin: int = 0,
               content_version: int = 0,
               tier: int = TIER_PRIMARY) -> TypedRef:
        """物化 Unicode Representation，并登记其在语言分支中的可见性。"""
        self._validate_surface(surface)
        ontology = self._require_authoritative()
        self._validate_branch(branch)
        assert_int(
            provenance_kind,
            epistemic_origin,
            content_version,
            tier,
            _where="WordFormIndex.ensure",
        )
        representation = ontology.materialize(
            self._representation_identity(encode(surface)), tier=tier)
        predicate = ontology.materialize(self._inventory_predicate_identity())
        ontology.relate(
            predicate,
            branch,
            representation,
            scope=scope,
            provenance_kind=provenance_kind,
            epistemic_origin=epistemic_origin,
            content_version=content_version,
        )
        self._authoritative_cache.pop(branch, None)
        catalog = self._course_catalogs.get(branch)
        if catalog is None or encode(surface) not in catalog[1]:
            self._segment_cache.pop(("authoritative", branch), None)
        return representation

    def install_course_catalog(
            self, *, branch: TypedRef, catalog_identity: tuple[int, ...],
            surfaces) -> int:
        """安装可重建的课程可见词形目录，不提前把全量目录写入权威图。"""
        self._validate_branch(branch)
        if not isinstance(catalog_identity, tuple) or not catalog_identity:
            raise ValueError("课程目录身份必须是非空整数 tuple")
        assert_int(*catalog_identity, _where="WordFormIndex.course_catalog")
        if any(type(value) is not int for value in catalog_identity):
            raise ValueError("课程目录身份必须使用严格整数")
        codepoints: set[tuple[int, ...]] = set()
        for surface in surfaces:
            self._validate_surface(surface)
            codepoints.add(validate_unicode_scalars(encode(surface)))
        frozen = frozenset(codepoints)
        existing = self._course_catalogs.get(branch)
        if existing is not None and existing != (catalog_identity, frozen):
            raise WordFormIdentityAmbiguity(
                "同一 WordFormIndex 分支安装了不同课程目录")
        self._course_catalogs[branch] = (catalog_identity, frozen)
        self._segment_cache.pop(("authoritative", branch), None)
        return len(frozen)

    def course_catalog_size(self, branch: TypedRef) -> int:
        """返回当前实例中可由外部 manifest 重建的课程词形数。"""
        self._validate_branch(branch)
        catalog = self._course_catalogs.get(branch)
        return len(catalog[1]) if catalog is not None else 0

    def register(self, surface: str, *, language: int, space_id: int,
                 tier: int = TIER_PRIMARY) -> tuple[int, int]:
        """注册 legacy ``NODE_WORD``；新代码应改用 ``ensure``。"""
        self._validate_surface(surface)
        if self._concept_index is None:
            raise RuntimeError("legacy 注册需要 ConceptIndex")
        assert_int(language, space_id, tier, _where="WordFormIndex.register")
        cps = encode(surface)
        identity = _LEGACY_WORD_IDENTITY.h63((language, cps))
        ref = self._concept_index.ensure(
            identity,
            space_id=space_id,
            tier=tier,
            node_type=NODE_WORD,
        )
        record_correspondence(
            self._backend,
            space_id=ref[0],
            local_id=ref[1],
            corr_kind=CORR_ORDINAL,
            codepoints=cps,
        )
        set_mark(
            self._backend,
            ref=ref,
            mark_kind=MARK_LANG,
            mark_value=language,
        )
        record_word_form(
            self._backend,
            space_id=space_id,
            language=language,
            word_ref=ref,
            codepoints=cps,
        )
        self._legacy_cache.pop((space_id, language), None)
        self._segment_cache.pop(("legacy", (language, space_id)), None)
        return ref

    def forms(self, *, branch: TypedRef | None = None,
              language: int | None = None,
              space_id: int | None = None):
        """读取权威 branch 目录；显式 language 参数只读取 legacy 目录。"""
        if branch is not None:
            if language is not None or space_id is not None:
                raise ValueError("branch 与 legacy language/space_id 不得混用")
            return self._authoritative_forms(branch)
        if language is None or space_id is None:
            raise ValueError("读取 legacy 目录必须同时提供 language 和 space_id")
        return self.forms_legacy(language=language, space_id=space_id)

    def forms_legacy(self, *, language: int,
                     space_id: int) -> dict[tuple[int, ...], tuple[int, int]]:
        """读取旧 ``(language, codepoints)`` 目录，重复键异值时 fail closed。"""
        assert_int(language, space_id, _where="WordFormIndex.forms_legacy")
        key = (space_id, language)
        cached = self._legacy_cache.get(key)
        if cached is not None:
            return dict(cached)
        entries = load_word_forms(
            self._backend, space_id=space_id, language=language)
        out: dict[tuple[int, ...], tuple[int, int]] = {}
        for cps, ref in entries:
            existing = out.get(cps)
            if existing is not None and existing != ref:
                raise WordFormIdentityAmbiguity(
                    "同一 legacy 词形键命中多个节点")
            out[cps] = ref
        self._legacy_cache[key] = out
        return dict(out)

    def lookup(self, surface: str, *, branch: TypedRef | None = None,
               language: int | None = None,
               space_id: int | None = None):
        """查权威 Representation；显式 language 参数只查 legacy 节点。"""
        if not isinstance(surface, str) or not surface:
            return None
        if branch is not None:
            if language is not None or space_id is not None:
                raise ValueError("branch 与 legacy language/space_id 不得混用")
            return self._lookup_authoritative(surface, branch)
        if language is None or space_id is None:
            raise ValueError("查 legacy 词形必须同时提供 language 和 space_id")
        return self.lookup_legacy(
            surface, language=language, space_id=space_id)

    def lookup_legacy(self, surface: str, *, language: int,
                      space_id: int) -> tuple[int, int] | None:
        """只读查找已登记的 legacy ``NODE_WORD``。"""
        if not isinstance(surface, str) or not surface:
            return None
        cps = encode(surface)
        return self.forms_legacy(
            language=language, space_id=space_id).get(cps)

    def segment(self, text: str, *, branch: TypedRef | None = None,
                language: int | None = None,
                space_id: int | None = None) -> list[str]:
        """按权威图与课程可见目录的并集执行确定性正向最大匹配。"""
        if not isinstance(text, str):
            raise TypeError("待分词文本必须是字符串")
        compiled = self._compiled_catalog(
            branch=branch, language=language, space_id=space_id)
        return self._segment_compiled(text, compiled)

    def match_lattice(self, text: str, *, branch: TypedRef | None = None,
                      language: int | None = None,
                      space_id: int | None = None
                      ) -> tuple[tuple[str, ...], ...]:
        """返回每个码点起点的全部词形命中，按长度降序且不裁成单一 FMM。"""
        if not isinstance(text, str):
            raise TypeError("待匹配文本必须是字符串")
        compiled = self._compiled_catalog(
            branch=branch, language=language, space_id=space_id)
        lengths, by_length = compiled
        cps = encode(text)
        lattice: list[tuple[str, ...]] = []
        for pos in range(len(cps)):
            matches: list[tuple[int, ...]] = []
            for length in lengths:
                candidate = cps[pos:pos + length]
                if candidate in by_length[length]:
                    matches.append(candidate)
            lattice.append(tuple(decode(match) for match in matches))
        return tuple(lattice)

    def _compiled_catalog(
            self, *, branch: TypedRef | None,
            language: int | None,
            space_id: int | None,
            ) -> tuple[tuple[int, ...], dict[int, set[tuple[int, ...]]]]:
        """读取或编译一次目录，供 FMM 和多候选 lattice 共享。"""
        if branch is not None:
            cache_key: tuple[str, object] = ("authoritative", branch)
            course = self._course_catalogs.get(branch)
        else:
            cache_key = ("legacy", (language, space_id))
            course = None
        compiled = self._segment_cache.get(cache_key)
        if compiled is None:
            forms = self.forms(
                branch=branch, language=language, space_id=space_id)
            candidates = set(forms)
            if course is not None:
                candidates.update(course[1])
            compiled = self._compile_codepoints(tuple(candidates))
            self._segment_cache[cache_key] = compiled
        return compiled

    def migrate_legacy(self, surface: str, *, language: int, space_id: int,
                       branch: TypedRef, scope: ScopeIdentity,
                       provenance_kind: int, epistemic_origin: int = 0,
                       content_version: int = 0,
                       tier: int = TIER_PRIMARY) -> WordFormMigration:
        """把所有可见旧 surface/词形节点显式桥接到权威 Representation。"""
        if self._concept_index is None:
            raise RuntimeError("legacy 迁移需要 ConceptIndex")
        representation = self.ensure(
            surface,
            branch=branch,
            scope=scope,
            provenance_kind=provenance_kind,
            epistemic_origin=epistemic_origin,
            content_version=content_version,
            tier=tier,
        )
        refs: set[tuple[int, int]] = set()
        legacy_word = self.lookup_legacy(
            surface, language=language, space_id=space_id)
        if legacy_word is not None:
            refs.add(legacy_word)
        legacy_surface = self._concept_index.lookup(surface, space_id)
        if legacy_surface is not None:
            refs.add(legacy_surface)
        for ref in sorted(refs):
            row = self._backend.select("concept_node", where={
                "space_id": ref[0], "local_id": ref[1],
            })
            if len(row) != 1 or row[0]["type"] not in {
                    NODE_CONCEPT, NODE_WORD}:
                raise WordFormIdentityAmbiguity(
                    "legacy 迁移端点不存在或节点类型非法")
            record_legacy_word_form_bridge(
                self._backend,
                legacy_ref=ref,
                legacy_node_type=row[0]["type"],
                object_ref=(
                    representation.object_kind,
                    representation.space_id,
                    representation.local_id,
                ),
            )
        return WordFormMigration(representation, tuple(sorted(refs)))

    def _authoritative_forms(
            self, branch: TypedRef) -> dict[tuple[int, ...], TypedRef]:
        """从图 statement 恢复 branch 的 Unicode Representation 目录。"""
        ontology = self._require_authoritative()
        self._validate_branch(branch)
        cached = self._authoritative_cache.get(branch)
        if cached is not None:
            return dict(cached)
        predicate = ontology.resolve(self._inventory_predicate_identity())
        if predicate is None:
            return {}
        out: dict[tuple[int, ...], TypedRef] = {}
        for ref in ontology.follow(branch, (predicate,)):
            identity = ontology.identity_of(ref)
            parts = self._representation_parts(identity)
            if parts is None:
                continue
            cps = parts[1]
            existing = out.get(cps)
            if existing is not None and existing != ref:
                raise WordFormIdentityAmbiguity(
                    "同一 branch 和码点序列命中多个 Representation")
            out[cps] = ref
        ordered = {key: out[key] for key in sorted(out)}
        self._authoritative_cache[branch] = ordered
        return dict(ordered)

    def _lookup_authoritative(self, surface: str,
                              branch: TypedRef) -> TypedRef | None:
        """只在 branch 已有图内可见性 statement 时返回 Representation。"""
        ontology = self._require_authoritative()
        self._validate_branch(branch)
        representation = ontology.resolve(
            self._representation_identity(encode(surface)))
        if representation is None:
            return None
        predicate = ontology.resolve(self._inventory_predicate_identity())
        if predicate is None:
            return None
        statements = ontology.statements(
            predicate=predicate,
            subject=branch,
            object_ref=representation,
        )
        return representation if statements else None

    def _representation_identity(
            self, codepoints: tuple[int, ...]) -> ObjectIdentity:
        """按注入的 Unicode 表示族构造与语言分支无关的身份。"""
        if self._unicode_family_key is None:
            raise RuntimeError("权威词形入口缺 unicode_family_key")
        return representation_identity(
            self._unicode_family_key,
            validate_unicode_scalars(tuple(codepoints)),
        )

    def _inventory_predicate_identity(self) -> ObjectIdentity:
        """按注入键构造 branch 词形可见性 predicate。"""
        if self._inventory_relation_key is None:
            raise RuntimeError("权威词形入口缺 inventory_relation_key")
        return relation_concept_identity(self._inventory_relation_key)

    def _representation_parts(
            self, identity: ObjectIdentity
            ) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
        """解析 Representation 身份，并过滤非本 Unicode 表示族对象。"""
        if identity.object_kind != OBJECT_REPRESENTATION:
            return None
        values = identity.components
        if not values:
            raise WordFormIdentityAmbiguity("Representation 身份缺少组成键")
        family_size = values[0]
        family_end = 1 + family_size
        if family_size <= 0 or family_end >= len(values):
            raise WordFormIdentityAmbiguity("Representation 表示族键损坏")
        family = values[1:family_end]
        representation_size = values[family_end]
        representation = values[family_end + 1:]
        if representation_size <= 0 or len(representation) != representation_size:
            raise WordFormIdentityAmbiguity("Representation 内容键损坏")
        if family != self._unicode_family_key:
            return None
        return family, validate_unicode_scalars(representation)

    def _validate_branch(self, branch: TypedRef) -> None:
        """核验 branch 是当前图内已物化的 LanguageBranch。"""
        identity = self._require_authoritative().identity_of(branch)
        if identity.object_kind != OBJECT_LANGUAGE_BRANCH:
            raise ValueError("branch 必须是已物化 LanguageBranch")

    def _require_authoritative(self) -> GraphOntology:
        """返回权威图入口，配置缺失时拒绝退回 legacy。"""
        if self._ontology is None:
            raise RuntimeError("WordFormIndex 未配置权威 GraphOntology")
        return self._ontology

    @staticmethod
    def _validate_surface(surface: str) -> None:
        """校验 surface 仅作为 Unicode 表示输入，不参与语言原子身份。"""
        if not isinstance(surface, str) or not surface:
            raise ValueError("词形必须是非空字符串")

    @staticmethod
    def _segment_with_codepoints(
            text: str, forms: tuple[tuple[int, ...], ...]) -> list[str]:
        """对给定码点目录执行 FMM，未知码点保持单 scalar 回退。"""
        return WordFormIndex._segment_compiled(
            text, WordFormIndex._compile_codepoints(forms))

    @staticmethod
    def _compile_codepoints(
            forms: tuple[tuple[int, ...], ...]
            ) -> tuple[tuple[int, ...], dict[int, set[tuple[int, ...]]]]:
        """把词形目录一次编译为按长度分桶的 FMM 查询结构。"""
        by_length: dict[int, set[tuple[int, ...]]] = {}
        for cps in forms:
            by_length.setdefault(len(cps), set()).add(cps)
        return tuple(sorted(by_length, reverse=True)), by_length

    @staticmethod
    def _segment_compiled(
            text: str,
            compiled: tuple[
                tuple[int, ...], dict[int, set[tuple[int, ...]]],
            ]) -> list[str]:
        """消费已编译目录执行 FMM，动态物化不会反复重建全目录。"""
        lengths, by_length = compiled
        cps = encode(text)
        out: list[str] = []
        pos = 0
        while pos < len(cps):
            matched = None
            for length in lengths:
                candidate = cps[pos:pos + length]
                if candidate in by_length[length]:
                    matched = candidate
                    break
            if matched is None:
                matched = cps[pos:pos + 1]
            out.append(decode(matched))
            pos += len(matched)
        return out


__all__ = [
    "WordFormIdentityAmbiguity",
    "WordFormIndex",
    "WordFormMigration",
]
