"""来源化语言信号 seed 的图内物化和只读竞争查询。

本模块不解释任何语言词形、否定或动作含义。seed 只携带外部注入的整数身份，
运行时把 surface 作为 Representation 输入，把语义目标作为 MinimalInstruction
身份，并用图 statement 保存二者之间的关联。兼容旧词形的投影不属于本模块。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from pure_integer_ai.cognition.shared.graph_ontology import (
    GraphOntology,
    relation_concept_identity,
)
from pure_integer_ai.cognition.shared.identity import (
    CorpusVersion,
    CurriculumVersion,
    OBJECT_MINIMAL_INSTRUCTION,
    OBJECT_LANGUAGE_ATOM,
    GLOBAL_OWNER_SCOPE,
    ParserVersion,
    PrimitiveVersion,
    SourceRef,
    TypedRef,
    VersionBundle,
    minimal_instruction_identity,
)
from pure_integer_ai.cognition.shared.language_object_index import (
    LanguageObjectIndex,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.understanding.word_form_index import WordFormIndex
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage.node_store import TIER_PRIMARY


def _integer_tuple(value: Any, *, where: str, allow_empty: bool = False) -> tuple[int, ...]:
    """把外部 JSON 键校验为不可变的严格整数元组。"""
    if not isinstance(value, list):
        raise ValueError(f"{where} 必须是 JSON 数组")
    if not value and not allow_empty:
        raise ValueError(f"{where} 不得为空")
    if any(type(item) is not int for item in value):
        raise ValueError(f"{where} 只能包含严格整数")
    return tuple(value)


def _positive_int(value: Any, *, where: str) -> int:
    """把外部 JSON 标量校验为严格正整数。"""
    if type(value) is not int or value <= 0:
        raise ValueError(f"{where} 必须是严格正整数")
    return value


@dataclass(frozen=True)
class LanguageSignalSeed:
    """一条带文件来源的 surface、语言原子和最小指令候选。"""

    source: SourceRef
    language: int
    surface: str
    atom_key: tuple[int, ...]
    instruction_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """校验 seed 不把 surface 偷换成语言原子身份。"""
        if not isinstance(self.source, SourceRef):
            raise TypeError("LanguageSignalSeed.source 必须是 SourceRef")
        assert_int(self.language, _where="LanguageSignalSeed.language")
        if self.language <= 0:
            raise ValueError("LanguageSignalSeed.language 必须为正")
        if not isinstance(self.surface, str) or not self.surface:
            raise ValueError("LanguageSignalSeed.surface 必须是非空字符串")
        for value, where in (
                (self.atom_key, "LanguageSignalSeed.atom_key"),
                (self.instruction_key, "LanguageSignalSeed.instruction_key")):
            if not isinstance(value, tuple) or not value:
                raise ValueError(f"{where} 必须是非空 tuple")
            assert_int(*value, _where=where)

    def stable_key(self) -> tuple[int, ...]:
        """返回不依赖 surface 排序的完整 seed 键。"""
        return (
            *self.source.stable_key(),
            self.language,
            len(self.atom_key),
            *self.atom_key,
            len(self.instruction_key),
            *self.instruction_key,
        )


@dataclass(frozen=True)
class LanguageSignalCatalog:
    """由外部文件恢复的来源化语言信号目录。"""

    schema_version: int
    source_kind: int
    versions: VersionBundle
    unicode_family_key: tuple[int, ...]
    branch_keys: tuple[tuple[int, tuple[int, ...]], ...]
    branch_inventory_relation_key: tuple[int, ...]
    branch_atom_relation_key: tuple[int, ...]
    atom_representation_relation_key: tuple[int, ...]
    atom_instruction_relation_key: tuple[int, ...]
    entries: tuple[LanguageSignalSeed, ...]
    content_sha256: str

    def branch_key(self, language: int) -> tuple[int, ...]:
        """按外部语言键读取 branch 身份，不使用语言名称推断。"""
        key = self.find_branch_key(language)
        if key is not None:
            return key
        raise LookupError("seed 没有对应的语言分支")

    def find_branch_key(self, language: int) -> tuple[int, ...] | None:
        """只读查询语言分支键，未知语言保持无副作用的空结果。"""
        for candidate, key in self.branch_keys:
            if candidate == language:
                return key
        return None

    def identity_key(self) -> tuple[int, ...]:
        """返回目录内容的稳定整数身份。"""
        digest_key = int(self.content_sha256[:15], 16) or 1
        return (self.schema_version, self.source_kind, digest_key, len(self.entries))


@dataclass(frozen=True)
class LanguageSignalCandidate:
    """一次 surface 查询得到的完整图候选，保留来源和两端对象。"""

    source: SourceRef
    language: int
    surface: str
    branch: TypedRef
    atom: TypedRef
    representation: TypedRef
    instruction: TypedRef

    def stable_key(self) -> tuple[int, ...]:
        """返回候选的完整身份排序键。"""
        return (
            *self.source.stable_key(),
            self.language,
            *self.atom.stable_key(),
            *self.instruction.stable_key(),
        )


@dataclass(frozen=True)
class LanguageSignalInstallReport:
    """记录一次 seed 图安装的来源摘要和四类 statement 提交数。"""

    content_sha256: str
    entry_count: int
    statement_count: int


@dataclass(frozen=True)
class LanguageSignalInstructionResolution:
    """保留图证据存在性与一致指令身份，避免把冲突误作无候选。"""

    has_evidence: bool
    instruction_key: tuple[int, ...] | None

    def __post_init__(self) -> None:
        """校验无证据状态不能携带伪造的指令结果。"""
        if not self.has_evidence and self.instruction_key is not None:
            raise ValueError("无图证据的解析不得携带 instruction_key")


def read_language_signal_catalog(path: str | Path) -> LanguageSignalCatalog:
    """读取并严格校验文件化语言信号 seed，逐行派生 SourceRef。"""
    file_path = Path(path)
    raw = file_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("语言信号 seed 不是有效 UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("语言信号 seed 顶层必须是对象")

    schema_version = _positive_int(payload.get("schema_version"),
                                   where="schema_version")
    source_kind = _positive_int(payload.get("source_kind"),
                                where="source_kind")
    version_values = _integer_tuple(
        payload.get("versions"), where="versions")
    if len(version_values) != 4 or any(value < 0 for value in version_values):
        raise ValueError("versions 必须是四个非负整数")
    versions = VersionBundle(
        CorpusVersion(version_values[0]),
        ParserVersion(version_values[1]),
        PrimitiveVersion(version_values[2]),
        CurriculumVersion(version_values[3]),
    )
    unicode_family_key = _integer_tuple(
        payload.get("unicode_family_key"), where="unicode_family_key")

    branch_payload = payload.get("branch_keys")
    if not isinstance(branch_payload, dict) or not branch_payload:
        raise ValueError("branch_keys 必须是非空对象")
    branches_list: list[tuple[int, tuple[int, ...]]] = []
    seen_languages: set[int] = set()
    for language_key, key in branch_payload.items():
        try:
            language = int(language_key)
        except (TypeError, ValueError) as exc:
            raise ValueError("branch_keys.language 必须是正整数字符串") from exc
        language = _positive_int(language, where="branch_keys.language")
        if str(language) != language_key or language in seen_languages:
            raise ValueError("branch_keys.language 必须使用唯一规范十进制形式")
        seen_languages.add(language)
        branches_list.append((
            language,
            _integer_tuple(key, where=f"branch_keys.{language_key}"),
        ))
    branches = tuple(sorted(branches_list))

    relation_payload = payload.get("relation_keys")
    if not isinstance(relation_payload, dict):
        raise ValueError("relation_keys 必须是对象")

    def relation_key(name: str) -> tuple[int, ...]:
        """读取一个由 manifest 注入的图 predicate 键。"""
        return _integer_tuple(relation_payload.get(name),
                              where=f"relation_keys.{name}")

    inventory_relation = relation_key("branch_inventory")
    branch_relation = relation_key("branch_atom")
    representation_relation = relation_key("atom_representation")
    instruction_relation = relation_key("atom_instruction")
    relation_keys = (
        inventory_relation,
        branch_relation,
        representation_relation,
        instruction_relation,
    )
    if len(set(relation_keys)) != len(relation_keys):
        raise ValueError("语言信号 relation_keys 必须两两不同")
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("entries 必须是非空数组")

    source_id = int(digest[:15], 16) or 1
    entries: list[LanguageSignalSeed] = []
    seen: set[tuple[int, str, tuple[int, ...], tuple[int, ...]]] = set()
    for ordinal, item in enumerate(raw_entries):
        if not isinstance(item, dict):
            raise ValueError("seed entry 必须是对象")
        language = _positive_int(item.get("language"),
                                where=f"entries[{ordinal}].language")
        if language not in {candidate for candidate, _ in branches}:
            raise ValueError("seed entry 的语言没有 branch_keys")
        surface = item.get("surface")
        atom_key = _integer_tuple(
            item.get("atom_key"), where=f"entries[{ordinal}].atom_key")
        instruction_key = _integer_tuple(
            item.get("instruction_key"),
            where=f"entries[{ordinal}].instruction_key")
        if not isinstance(surface, str) or not surface:
            raise ValueError(f"entries[{ordinal}].surface 必须是非空字符串")
        identity = (language, surface, atom_key, instruction_key)
        if identity in seen:
            raise ValueError("seed entry 重复")
        seen.add(identity)
        entries.append(LanguageSignalSeed(
            SourceRef(source_kind, source_id, ordinal, GLOBAL_OWNER_SCOPE,
                      versions),
            language,
            surface,
            atom_key,
            instruction_key,
        ))

    return LanguageSignalCatalog(
        schema_version,
        source_kind,
        versions,
        unicode_family_key,
        branches,
        inventory_relation,
        branch_relation,
        representation_relation,
        instruction_relation,
        tuple(entries),
        digest,
    )


class LanguageSignalRuntime:
    """把来源化 seed 接入图，并提供不私选候选的只读查询。"""

    def __init__(self, *, backend, concept_index, ontology: GraphOntology,
                 catalog: LanguageSignalCatalog) -> None:
        self.backend = backend
        self.concept_index = concept_index
        self.ontology = ontology
        self.catalog = catalog
        self._objects = LanguageObjectIndex(ontology)
        self._word_forms = WordFormIndex(
            backend,
            concept_index,
            ontology=ontology,
            unicode_family_key=catalog.unicode_family_key,
            inventory_relation_key=catalog.branch_inventory_relation_key,
        )
        self._installed = False

    @property
    def word_forms(self) -> WordFormIndex:
        """返回使用独立分支目录 predicate 的权威 Representation 索引。"""
        return self._word_forms

    def install(self) -> LanguageSignalInstallReport:
        """按稳定 seed 序写入分支目录、原子、表示和指令四类关联。"""
        if self._installed:
            return LanguageSignalInstallReport(
                self.catalog.content_sha256, len(self.catalog.entries), 0)
        predicates = tuple(self.ontology.materialize(
            relation_concept_identity(key)) for key in (
                self.catalog.branch_atom_relation_key,
                self.catalog.atom_representation_relation_key,
                self.catalog.atom_instruction_relation_key,
            ))
        statement_count = 0
        for seed in sorted(self.catalog.entries, key=LanguageSignalSeed.stable_key):
            branch = self._objects.ensure_branch(
                self.catalog.branch_key(seed.language), tier=TIER_PRIMARY)
            branch_identity = self.ontology.identity_of(branch)
            atom = self._objects.ensure_atom(
                branch, seed.atom_key, tier=TIER_PRIMARY)
            representation = self._word_forms.ensure(
                seed.surface,
                branch=branch,
                scope=document_scope(seed.source),
                provenance_kind=self.catalog.source_kind,
                content_version=self.catalog.versions.curriculum.value,
                tier=TIER_PRIMARY,
            )
            instruction = self.ontology.materialize(
                minimal_instruction_identity(
                    seed.instruction_key,
                    versions=self.catalog.versions,
                ),
                tier=TIER_PRIMARY,
            )
            self.ontology.relate(
                predicates[0], branch, atom,
                scope=document_scope(seed.source),
                provenance_kind=self.catalog.source_kind,
                content_version=self.catalog.versions.curriculum.value,
            )
            self.ontology.relate(
                predicates[1], atom, representation,
                scope=document_scope(seed.source),
                provenance_kind=self.catalog.source_kind,
                content_version=self.catalog.versions.curriculum.value,
            )
            self.ontology.relate(
                predicates[2], atom, instruction,
                scope=document_scope(seed.source),
                provenance_kind=self.catalog.source_kind,
                content_version=self.catalog.versions.curriculum.value,
            )
            statement_count += 4
            if branch_identity.object_kind != branch.object_kind:
                raise ValueError("branch 身份恢复失败")
        self._installed = True
        return LanguageSignalInstallReport(
            self.catalog.content_sha256, len(self.catalog.entries), statement_count)

    def lookup(self, surface: str, *, language: int) -> tuple[LanguageSignalCandidate, ...]:
        """沿同一来源的四类图 statement 返回全部竞争候选。"""
        if not isinstance(surface, str) or not surface:
            return ()
        branch_key = self.catalog.find_branch_key(language)
        if branch_key is None:
            return ()
        branch = self._objects.lookup_branch(branch_key)
        if branch is None:
            return ()
        inventory_predicate = self.ontology.resolve(
            relation_concept_identity(
                self.catalog.branch_inventory_relation_key))
        branch_predicate = self.ontology.resolve(
            relation_concept_identity(self.catalog.branch_atom_relation_key))
        representation_predicate = self.ontology.resolve(
            relation_concept_identity(
                self.catalog.atom_representation_relation_key))
        instruction_predicate = self.ontology.resolve(
            relation_concept_identity(self.catalog.atom_instruction_relation_key))
        if any(item is None for item in (
                inventory_predicate, branch_predicate, representation_predicate,
                instruction_predicate)):
            return ()
        representation = self._word_forms.lookup(surface, branch=branch)
        if representation is None:
            return ()
        inventory_sources = {
            item.assertion.scope.source
            for item in self.ontology.statements(
                predicate=inventory_predicate,
                subject=branch,
                object_ref=representation,
            )
            if item.assertion.scope.source is not None
        }
        output: dict[tuple[int, ...], LanguageSignalCandidate] = {}
        for representation_statement in self.ontology.statements(
                predicate=representation_predicate,
                object_ref=representation):
            source = representation_statement.assertion.scope.source
            if source is None or source not in inventory_sources:
                continue
            atom = representation_statement.subject
            if self.ontology.identity_of(atom).object_kind != OBJECT_LANGUAGE_ATOM:
                raise ValueError("语言表示关联起点不是 LanguageAtom")
            branch_matches = self.ontology.statements(
                predicate=branch_predicate,
                subject=branch,
                object_ref=atom,
            )
            if not any(item.assertion.scope.source == source
                       for item in branch_matches):
                continue
            for signal in self.ontology.statements(
                    predicate=instruction_predicate, subject=atom):
                if signal.assertion.scope.source != source:
                    continue
                identity = self.ontology.identity_of(signal.object)
                if identity.object_kind != OBJECT_MINIMAL_INSTRUCTION:
                    raise ValueError("语言信号目标不是 MinimalInstruction")
                candidate = LanguageSignalCandidate(
                    source, language, surface, branch, atom,
                    representation, signal.object)
                output[candidate.stable_key()] = candidate
        return tuple(output[key] for key in sorted(output))

    def matches_instruction(
            self, surface: str, *, language: int,
            instruction_key: tuple[int, ...]) -> bool | None:
        """判断一个表示的全部图候选是否一致指向调用方给定的最小指令。

        无候选返回 ``None``，允许上层区分“图无证据”和明确的非目标候选；
        全部候选一致命中才返回 ``True``，任一非目标或混合候选都返回 ``False``。
        """
        if not isinstance(instruction_key, tuple) or not instruction_key:
            raise ValueError("instruction_key 必须是非空整数 tuple")
        if any(type(value) is not int for value in instruction_key):
            raise TypeError("instruction_key 只能包含严格整数")
        resolution = self.resolve_instruction(surface, language=language)
        if not resolution.has_evidence:
            return None
        return resolution.instruction_key == instruction_key

    def resolve_instruction(
            self, surface: str, *,
            language: int) -> LanguageSignalInstructionResolution:
        """解析全部图候选的一致指令身份，并显式保留冲突与无证据的差别。"""
        candidates = self.lookup(surface, language=language)
        if not candidates:
            return LanguageSignalInstructionResolution(False, None)
        instruction_keys = {
            self.ontology.identity_of(candidate.instruction).components
            for candidate in candidates
        }
        if len(instruction_keys) != 1:
            return LanguageSignalInstructionResolution(True, None)
        return LanguageSignalInstructionResolution(
            True, next(iter(instruction_keys)))

    def clone_for_context(self, *, backend, concept_index,
                          ontology: GraphOntology) -> "LanguageSignalRuntime":
        """在隔离图和后端上重建目录 runtime，不共享可变查询缓存。"""
        return LanguageSignalRuntime(
            backend=backend,
            concept_index=concept_index,
            ontology=ontology,
            catalog=self.catalog,
        )


__all__ = [
    "LanguageSignalCandidate",
    "LanguageSignalCatalog",
    "LanguageSignalInstallReport",
    "LanguageSignalInstructionResolution",
    "LanguageSignalRuntime",
    "LanguageSignalSeed",
    "read_language_signal_catalog",
]
