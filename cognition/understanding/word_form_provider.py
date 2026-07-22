"""正式语言入口使用的课程词形 provider 与多语言注册表。

课程目录保持为可由外部 manifest 重建的只读候选；只有真实输入命中的词形才惰性物化
Representation，并显式桥接 observe 仍需创建的 legacy 词节点。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.identity import (
    SourceRef,
    TypedRef,
    minimal_instruction_identity,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EvidenceRecord,
    HypothesisKey,
    HypothesisSnapshot,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    CLOCK_OBSERVATION,
    LogicalClockIdentity,
    LogicalTimestamp,
    ScopeIdentity,
    document_scope,
)
from pure_integer_ai.cognition.understanding.segmentation_hypothesis import (
    SegmentationHypothesisEngine,
    SegmentationProtocol,
    SegmentationResult,
)
from pure_integer_ai.cognition.understanding.word_form_index import WordFormIndex
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage.node_store import TIER_PRIMARY


@dataclass(frozen=True)
class VisibleWordForm:
    """一个课程可见词形的来源和图 statement 写入参数。"""

    source_ref: SourceRef
    course_split: int
    provenance_kind: int
    epistemic_origin: int
    content_version: int

    def __post_init__(self) -> None:
        assert_int(
            self.course_split,
            self.provenance_kind,
            self.epistemic_origin,
            self.content_version,
            _where="VisibleWordForm",
        )
        if (self.course_split <= 0 or self.provenance_kind <= 0
                or self.content_version < 0):
            raise ValueError(
                "词形课程 split、来源类型必须为正且内容版本不得为负")


class WordFormProvider:
    """为一个运行期语言键提供课程 FMM、惰性物化和 legacy 桥接。"""

    def __init__(
            self, *, backend, concept_index, ontology,
            branch: TypedRef, runtime_language: int,
            unicode_family_key: tuple[int, ...],
            inventory_relation_key: tuple[int, ...],
            catalog_identity: tuple[int, ...],
            catalog: dict[str, VisibleWordForm],
            segmentation_protocol: SegmentationProtocol | None = None,
            ) -> None:
        assert_int(runtime_language, _where="WordFormProvider.runtime_language")
        if runtime_language <= 0:
            raise ValueError("运行期语言键必须为正")
        self.backend = backend
        self.concept_index = concept_index
        self.ontology = ontology
        self.branch = branch
        self.runtime_language = runtime_language
        self.unicode_family_key = unicode_family_key
        self.inventory_relation_key = inventory_relation_key
        self.catalog_identity = catalog_identity
        self._catalog = dict(catalog)
        self._materialized: dict[tuple[int, str], TypedRef] = {}
        self.segmentation_protocol = segmentation_protocol
        self._segmentation_span_materializer = None
        self._segmentation = (
            None if segmentation_protocol is None
            else SegmentationHypothesisEngine(segmentation_protocol))
        if segmentation_protocol is not None:
            for instruction_key in segmentation_protocol.instruction_keys():
                ontology.materialize(
                    minimal_instruction_identity(instruction_key),
                    tier=TIER_PRIMARY,
                )
        self.index = WordFormIndex(
            backend,
            concept_index,
            ontology=ontology,
            unicode_family_key=unicode_family_key,
            inventory_relation_key=inventory_relation_key,
        )
        installed = self.index.install_course_catalog(
            branch=branch,
            catalog_identity=catalog_identity,
            surfaces=self._catalog,
        )
        if installed != len(self._catalog):
            raise ValueError("课程词形 surface 与码点目录去重结果不一致")

    @property
    def catalog_size(self) -> int:
        """返回当前语言分支可见的唯一课程词形数。"""
        return len(self._catalog)

    @property
    def materialized_count(self) -> int:
        """返回当前进程已因真实观察而物化或恢复的词形桥数量。"""
        return len(self._materialized)

    def visible_form(self, surface: str) -> VisibleWordForm | None:
        """只读返回词形的来源和课程 split；不可见词形返回 None。"""
        return self._catalog.get(surface)

    def segmentation_snapshot(
            self, hypothesis: HypothesisKey) -> HypothesisSnapshot:
        """读取一个分词候选的当前 H-00 快照。"""
        if self._segmentation is None:
            raise RuntimeError("当前 provider 未配置多候选分词协议")
        return self._segmentation.ledger.snapshot(hypothesis)

    def segmentation_evidence_history(
            self, hypothesis: HypothesisKey) -> tuple[EvidenceRecord, ...]:
        """读取一个分词候选的完整 append-only Evidence 历史。"""
        if self._segmentation is None:
            raise RuntimeError("当前 provider 未配置多候选分词协议")
        return self._segmentation.ledger.evidence_history(hypothesis)

    def segmentation_resolution_history(
            self, hypothesis: HypothesisKey) -> tuple:
        """读取候选竞争组的完整 H-04 决策、降级和替代历史。"""
        if self._segmentation is None:
            raise RuntimeError("当前 provider 未配置多候选分词协议")
        return self._segmentation.resolution_history(hypothesis)

    def segment_text(self, text: str) -> list[str]:
        """按上游空白边界分单元，再用 WordFormIndex.segment 执行课程 FMM。"""
        if not isinstance(text, str):
            raise TypeError("待分词文本必须是字符串")
        tokens: list[str] = []
        for unit in text.split():
            tokens.extend(self.index.segment(unit, branch=self.branch))
        return tokens

    def parse_text(
            self, text: str, *, observation: SourceRef,
            scope: ScopeIdentity,
            commit_evidence: bool = True,
            ) -> SegmentationResult | None:
        """生成带 H-00 Evidence 的多分词候选；未配置协议时返回 None。"""
        if self._segmentation is None:
            return None
        lattice = self.index.match_lattice(text, branch=self.branch)
        return self._segmentation.parse(
            text,
            lattice=lattice,
            branch=self.branch,
            observation=observation,
            scope=scope,
            visible_form=self.visible_form,
            commit=commit_evidence,
        )

    def record_segmentation_feedback(
            self, hypothesis: HypothesisKey, *, stance: int,
            source: SourceRef, reason_key: tuple[int, ...],
            timestamp_seq: int,
            replacement: HypothesisKey | None = None,
            ) -> HypothesisSnapshot:
        """追加分词边界反馈，并把调用方 reason 物化为图内最小协议符号。"""
        if self._segmentation is None:
            raise RuntimeError("当前 provider 未配置多候选分词协议")
        self._segmentation.validate_feedback(
            hypothesis,
            stance=stance,
            source=source,
            reason_key=reason_key,
            timestamp_seq=timestamp_seq,
            replacement=replacement,
        )
        timestamp = None
        if (self._segmentation_span_materializer is not None
                and stance == EVIDENCE_REFUTE
                and replacement is not None):
            timestamp = LogicalTimestamp(
                LogicalClockIdentity(
                    hypothesis.scope,
                    CLOCK_OBSERVATION,
                ),
                timestamp_seq,
            )
            self._segmentation_span_materializer.validate_candidate_supersede(
                hypothesis,
                replacement,
                timestamp,
            )
        self.ontology.materialize(
            minimal_instruction_identity(reason_key),
            tier=TIER_PRIMARY,
        )
        snapshot = self._segmentation.record_feedback(
            hypothesis,
            stance=stance,
            source=source,
            reason_key=reason_key,
            timestamp_seq=timestamp_seq,
            replacement=replacement,
        )
        if timestamp is not None:
            self._segmentation_span_materializer.supersede_candidate(
                hypothesis,
                replacement,
                timestamp,
            )
        return snapshot

    def install_segmentation_span_materializer(self, materializer) -> None:
        """安装 L-04 生命周期同步器，同一 provider 不得静默替换实例。"""
        if materializer is None:
            raise TypeError("materializer 不得为 None")
        existing = self._segmentation_span_materializer
        if existing is not None and existing is not materializer:
            raise ValueError("同一 provider 已安装不同 Span materializer")
        self._segmentation_span_materializer = materializer

    def segmentation_state(self) -> tuple:
        """返回分词 H-00 事件与 H-04 决策链，供评测隔离核验。"""
        if self._segmentation is None:
            return ()
        return self._segmentation.state_key()

    def observe_surface(self, surface: str, *, space_id: int) -> TypedRef | None:
        """命中课程词形时惰性物化 Representation，并桥接同 surface 的旧节点。"""
        assert_int(space_id, _where="WordFormProvider.observe_surface")
        visible = self._catalog.get(surface)
        if visible is None:
            return None
        key = (space_id, surface)
        cached = self._materialized.get(key)
        if cached is not None:
            return cached
        migration = self.index.migrate_legacy(
            surface,
            language=self.runtime_language,
            space_id=space_id,
            branch=self.branch,
            scope=document_scope(visible.source_ref),
            provenance_kind=visible.provenance_kind,
            epistemic_origin=visible.epistemic_origin,
            content_version=visible.content_version,
        )
        if migration.legacy_refs:
            self._materialized[key] = migration.representation
        return migration.representation

    def clone_for_context(self, *, backend, concept_index,
                          ontology) -> "WordFormProvider":
        """在评测克隆后端上重建 provider，不共享可变索引和物化缓存。"""
        cloned = WordFormProvider(
            backend=backend,
            concept_index=concept_index,
            ontology=ontology,
            branch=self.branch,
            runtime_language=self.runtime_language,
            unicode_family_key=self.unicode_family_key,
            inventory_relation_key=self.inventory_relation_key,
            catalog_identity=self.catalog_identity,
            catalog=self._catalog,
            segmentation_protocol=self.segmentation_protocol,
        )
        if self._segmentation is not None:
            cloned._segmentation = self._segmentation.clone()
        return cloned


class WordFormProviderRegistry:
    """按运行期语言键派发 provider，后续新增语言不修改 parser 逻辑。"""

    def __init__(self) -> None:
        self._providers: dict[int, WordFormProvider] = {}

    def register(self, provider: WordFormProvider) -> None:
        """登记一个语言 provider，同键异实例拒绝静默覆盖。"""
        existing = self._providers.get(provider.runtime_language)
        if existing is not None and existing is not provider:
            raise ValueError("同一运行期语言键重复登记不同词形 provider")
        self._providers[provider.runtime_language] = provider

    def supports(self, runtime_language: int) -> bool:
        """判断给定运行期语言键是否已有课程 provider。"""
        return runtime_language in self._providers

    def segment_text(self, text: str, *, runtime_language: int) -> list[str]:
        """把原始语言文本派发给对应分支；缺 provider 时 fail closed。"""
        provider = self._providers.get(runtime_language)
        if provider is None:
            raise LookupError("运行期语言键没有词形 provider")
        return provider.segment_text(text)

    def parse_text(self, text: str, *, runtime_language: int,
                   observation: SourceRef, scope,
                   commit_evidence: bool = True) -> SegmentationResult | None:
        """派发多候选分词；对应 provider 未配置 H-00 协议时返回 None。"""
        provider = self._providers.get(runtime_language)
        if provider is None:
            raise LookupError("运行期语言键没有词形 provider")
        return provider.parse_text(
            text,
            observation=observation,
            scope=scope,
            commit_evidence=commit_evidence,
        )

    def observe_surface(self, surface: str, *, runtime_language: int,
                        space_id: int) -> TypedRef | None:
        """把观察到的 surface 派发给对应分支执行惰性物化。"""
        provider = self._providers.get(runtime_language)
        if provider is None:
            return None
        return provider.observe_surface(surface, space_id=space_id)

    def provider(self, runtime_language: int) -> WordFormProvider | None:
        """返回指定语言 provider，供遥测和测试读取状态。"""
        return self._providers.get(runtime_language)

    def clone_for_context(self, *, backend, concept_index,
                          ontology) -> "WordFormProviderRegistry":
        """为评测上下文重建全部语言 provider。"""
        cloned = WordFormProviderRegistry()
        for runtime_language in sorted(self._providers):
            cloned.register(self._providers[runtime_language].clone_for_context(
                backend=backend,
                concept_index=concept_index,
                ontology=ontology,
            ))
        return cloned

    def segmentation_state(self) -> tuple:
        """按语言键汇总全部 provider 的分词状态，不复制课程词形目录。"""
        return tuple(
            (runtime_language,
             self._providers[runtime_language].segmentation_state())
            for runtime_language in sorted(self._providers)
        )

    def install_segmentation_span_materializer(self, materializer) -> None:
        """给全部已登记语言 provider 安装同一上下文的 L-04 同步器。"""
        for runtime_language in sorted(self._providers):
            self._providers[
                runtime_language].install_segmentation_span_materializer(
                    materializer)


__all__ = [
    "VisibleWordForm",
    "WordFormProvider",
    "WordFormProviderRegistry",
]
