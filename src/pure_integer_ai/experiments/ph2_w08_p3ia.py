"""编排现有层级、召回、问答与修订 owner 的 W08 P3-Ia facade。"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pure_integer_ai.cognition.shared.hypothesis import EvidenceRecord
from pure_integer_ai.cognition.shared.identity import ObjectIdentity
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.shared.memory_query import MemoryCurrentQuery
from pure_integer_ai.cognition.shared.question_answer import (
    QuestionExecutionResult,
    QuestionQuery,
)
from pure_integer_ai.experiments.free_text_hierarchy_runtime import (
    FormedTextHierarchy,
    MechanicalTextHierarchyFormer,
)
from pure_integer_ai.experiments.free_text_recall_runtime import (
    FreeTextRecallRun,
    FreeTextRecallRuntime,
    RecallIndexEntry,
    RecalledFactQuestionExecutor,
)
from pure_integer_ai.experiments.ph2_dataset_contract import StableRecordKey
from pure_integer_ai.experiments.ph2_free_text_hierarchy_recall_contract import (
    RecallBudget,
    SourceDocument,
)
from pure_integer_ai.experiments.ph2_w05_contract import digest_value
from pure_integer_ai.experiments.ph2_w08_contract import W08_CONSUMER_KEYS
from pure_integer_ai.experiments.ph2_w08_p3ia_contract import (
    W08P3IaAuditReceipt,
    W08P3IaError,
    W08P3IaRequest,
    W08P3IaResourceReceipt,
    W08P3IaTrace,
    W08P3IaUse,
    W08_P3IA_COMPONENT_KEYS,
    W08_P3IA_OWNER_KEYS,
)
from pure_integer_ai.experiments.ph2_w08_p3ia_training import (
    W08P3IaTrainingBundle,
    W08P3IaTrainingCase,
)
from pure_integer_ai.experiments.ph2_w08_recompute_contract import (
    W08LocalRecomputeAuditReceipt,
)
from pure_integer_ai.experiments.question_answer_runtime import QuestionAnswerRun


@dataclass(frozen=True)
class W08P3IaExecution:
    training: W08P3IaTrainingBundle
    case: W08P3IaTrainingCase
    document: SourceDocument
    raw_query: str
    history: tuple[EvidenceRecord, ...]
    current: MemoryCurrentQuery
    index: tuple[RecallIndexEntry, ...]
    access: MemoryAccessContext
    recall_budget: RecallBudget
    question_query: QuestionQuery
    question_reason: ObjectIdentity
    revision: W08LocalRecomputeAuditReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.training, W08P3IaTrainingBundle):
            raise TypeError("P3-Ia execution training type is invalid")
        if not isinstance(self.case, W08P3IaTrainingCase) or self.case not in self.training.cases:
            raise W08P3IaError("P3-Ia execution case is not preregistered")
        if not isinstance(self.document, SourceDocument):
            raise TypeError("P3-Ia execution document type is invalid")
        if self.document.raw_text != self.training.long_context.document_text:
            raise W08P3IaError("P3-Ia document did not come from W08-visible material")
        if self.raw_query != self.case.paraphrase_surface:
            raise W08P3IaError("P3-Ia query was not mechanically derived")
        if not isinstance(self.history, tuple) or any(
            not isinstance(item, EvidenceRecord) for item in self.history
        ):
            raise TypeError("P3-Ia Evidence history type is invalid")
        if not isinstance(self.current, MemoryCurrentQuery):
            raise TypeError("P3-Ia current query type is invalid")
        if not isinstance(self.index, tuple) or any(
            not isinstance(item, RecallIndexEntry) for item in self.index
        ):
            raise TypeError("P3-Ia recall index type is invalid")
        if not isinstance(self.access, MemoryAccessContext):
            raise TypeError("P3-Ia access context type is invalid")
        if not isinstance(self.recall_budget, RecallBudget):
            raise TypeError("P3-Ia recall budget type is invalid")
        if not isinstance(self.question_query, QuestionQuery):
            raise TypeError("P3-Ia question query type is invalid")
        if not isinstance(self.question_reason, ObjectIdentity):
            raise TypeError("P3-Ia question reason type is invalid")
        if not isinstance(self.revision, W08LocalRecomputeAuditReceipt):
            raise TypeError("P3-Ia revision receipt type is invalid")


class W08P3IaGenerationOwner:
    """以召回的问答候选核验真实 G-00 至 G-04 运行。"""

    owner_key = W08_P3IA_OWNER_KEYS[3]

    def __init__(
        self,
        producer: Callable[[FreeTextRecallRun, QuestionExecutionResult], QuestionAnswerRun],
    ) -> None:
        if not callable(producer):
            raise TypeError("P3-Ia generation producer must be callable")
        self.producer = producer

    def consume(
        self,
        recall: FreeTextRecallRun,
        qa_result: QuestionExecutionResult,
    ) -> QuestionAnswerRun:
        run = self.producer(recall, qa_result)
        if not isinstance(run, QuestionAnswerRun):
            raise TypeError("P3-Ia generation owner returned an invalid run")
        if (
            not run.complete
            or run.postcheck is None
            or not run.postcheck.complete
            or run.planning_request is None
            or run.selection is None
        ):
            raise W08P3IaError("P3-Ia generation did not pass same-run postcheck")
        expected = {item.stable_key() for item in qa_result.candidates}
        selected = set(run.selection.selected_candidate_keys)
        actual = {item.stable_key() for item in run.planning_request.candidates}
        if not expected or not selected or not selected <= expected or not expected <= actual:
            raise W08P3IaError("P3-Ia generation replaced the recalled QA candidate")
        return run


@dataclass(frozen=True)
class W08P3IaOwners:
    hierarchy: MechanicalTextHierarchyFormer
    recall: FreeTextRecallRuntime
    generation: W08P3IaGenerationOwner

    def __post_init__(self) -> None:
        if not isinstance(self.hierarchy, MechanicalTextHierarchyFormer):
            raise TypeError("P3-Ia hierarchy owner type is invalid")
        if not isinstance(self.recall, FreeTextRecallRuntime):
            raise TypeError("P3-Ia recall owner type is invalid")
        if not isinstance(self.generation, W08P3IaGenerationOwner):
            raise TypeError("P3-Ia generation owner type is invalid")


class W08P3IaFacade:
    """把 W08 可见资料绑定到生产 P3-Ia owner 链。"""

    def __init__(self, owners: W08P3IaOwners) -> None:
        if not isinstance(owners, W08P3IaOwners):
            raise TypeError("P3-Ia facade requires W08P3IaOwners")
        self.owners = owners

    @staticmethod
    def _resources(
        *,
        hierarchy: FormedTextHierarchy | None = None,
        recall: FreeTextRecallRun | None = None,
        consumers: int = 0,
        recompute_objects: int = 0,
    ) -> W08P3IaResourceReceipt:
        metrics = None
        if recall is not None and recall.exact_read is not None:
            metrics = recall.exact_read.metrics
        return W08P3IaResourceReceipt(
            0 if hierarchy is None else len(hierarchy.candidates),
            0 if recall is None else len(recall.matched_features),
            0 if recall is None else len(recall.centers),
            (
                0
                if recall is None or recall.exact_read is None
                else len(recall.exact_read.receipt.segment_keys)
            ),
            0 if metrics is None else metrics.page_faults,
            0 if metrics is None else metrics.cold_read_bytes,
            0 if metrics is None else metrics.page_in_records,
            consumers,
            recompute_objects,
            (
                (0 if hierarchy is None else len(hierarchy.candidates))
                + (0 if recall is None else len(recall.matched_features) + len(recall.centers))
                + consumers
                + recompute_objects
            ),
        )

    @staticmethod
    def _stopped(
        request: W08P3IaRequest,
        state: str,
        calls: list[str],
        *,
        blocked_component: str = "",
        hierarchy: FormedTextHierarchy | None = None,
        recall: FreeTextRecallRun | None = None,
    ) -> W08P3IaAuditReceipt:
        return W08P3IaAuditReceipt(
            request.request_key,
            state,
            None,
            W08P3IaFacade._resources(hierarchy=hierarchy, recall=recall),
            (),
            tuple(calls),
            blocked_component,
        )

    @staticmethod
    def _hierarchy_target(
        hierarchy: FormedTextHierarchy,
        case: W08P3IaTrainingCase,
    ):
        targets = tuple(
            item
            for item in hierarchy.candidates
            if item.candidate_kind == "PROPOSITION"
            and item.span.start <= case.citation_start
            and case.citation_end <= item.span.end
        )
        if len(targets) != 1:
            raise W08P3IaError("P3-Ia citation did not resolve to one proposition")
        by_key = {item.candidate_key: item for item in hierarchy.candidates}
        paragraph = by_key.get(targets[0].parent_key)
        section = None if paragraph is None else by_key.get(paragraph.parent_key)
        if (
            paragraph is None
            or paragraph.candidate_kind != "PARAGRAPH"
            or section is None
            or section.candidate_kind != "SECTION"
        ):
            raise W08P3IaError("P3-Ia section/paragraph/proposition chain is incomplete")
        return section, paragraph, targets[0]

    def execute(
        self,
        request: W08P3IaRequest,
        execution: W08P3IaExecution,
    ) -> W08P3IaAuditReceipt:
        if not isinstance(request, W08P3IaRequest) or not isinstance(
            execution, W08P3IaExecution
        ):
            raise TypeError("P3-Ia request/execution type is invalid")
        if request.case_key != execution.case.case_key:
            raise W08P3IaError("P3-Ia request/case identity drifted")
        calls: list[str] = []
        for component, state in (
            (W08_P3IA_COMPONENT_KEYS[0], "GROUNDING_BLOCKED"),
            (W08_P3IA_COMPONENT_KEYS[1], "GROUNDING_BLOCKED"),
            (W08_P3IA_COMPONENT_KEYS[2], "GROUNDING_BLOCKED"),
            (W08_P3IA_COMPONENT_KEYS[3], "ACCESS_BLOCKED"),
        ):
            if not request.component_enabled(component):
                return self._stopped(
                    request,
                    state,
                    calls,
                    blocked_component=component,
                )

        hierarchy = self.owners.hierarchy.form(execution.document)
        calls.append(W08_P3IA_OWNER_KEYS[0])
        section, paragraph, proposition = self._hierarchy_target(
            hierarchy, execution.case
        )
        recall = self.owners.recall.resolve(
            execution.raw_query,
            execution.history,
            execution.current,
            execution.index,
            execution.access,
            execution.recall_budget,
            reader_key=request.reader_key,
        )
        calls.append(W08_P3IA_OWNER_KEYS[1])
        if recall.stop_reason != "RESOLVED":
            state = {
                "CLARIFY": "CLARIFY",
                "UNKNOWN": "UNKNOWN",
                "UNAUTHORIZED": "ACCESS_BLOCKED",
                "BUDGET_EXHAUSTED": "BUDGET_EXHAUSTED",
            }.get(recall.stop_reason, "UNKNOWN")
            return self._stopped(
                request,
                state,
                calls,
                hierarchy=hierarchy,
                recall=recall,
            )
        if (
            recall.exact_read is None
            or recall.exact_read.payload is None
            or recall.selected_center_key is None
            or len(recall.centers) != 1
        ):
            raise W08P3IaError("P3-Ia resolved recall is incomplete")
        exact = recall.exact_read
        matches = tuple(
            item
            for item in recall.matched_features
            if item.feature_key == execution.case.feature_key
        )
        if not matches:
            raise W08P3IaError("P3-Ia paraphrase Evidence did not bear on recall")
        citations = exact.receipt.citations
        if len(citations) != 1:
            raise W08P3IaError("P3-Ia exact recall did not return one citation")
        citation = citations[0]
        if (
            citation.span.start != execution.case.citation_start
            or citation.span.end != execution.case.citation_end
            or citation.source_ref != execution.document.source_ref
            or exact.acl_checked_before_payload != 1
            or exact.metrics.page_faults <= 0
            or exact.metrics.page_in_records != 1
        ):
            raise W08P3IaError("P3-Ia ACL/cold page-in/citation evidence drifted")

        qa_result = RecalledFactQuestionExecutor(
            recall,
            route=execution.question_query.route,
            executed_reason=execution.question_reason,
            trace_prefix=(80806, 201, *request.request_key),
        ).execute(execution.question_query)
        calls.append(W08_P3IA_OWNER_KEYS[2])
        generation = self.owners.generation.consume(recall, qa_result)
        calls.append(W08_P3IA_OWNER_KEYS[3])

        revision = execution.revision
        invalidated = set(revision.free_text.invalidated_keys)
        if (
            revision.stop_state != "RESOLVED"
            or revision.full_document_reparse_count != 0
            or revision.additional_payload_get_count != 0
            or not invalidated.intersection(exact.payload.dependency_keys)
            or not revision.free_text.preserved_keys
            or any(item.bit_identical != 1 for item in revision.preservations)
        ):
            raise W08P3IaError("P3-Ia revision locality did not bear")
        calls.append(W08_P3IA_OWNER_KEYS[4])

        paraphrase_evidence = tuple(
            sorted(
                {
                    StableRecordKey((8080605, item.evidence_id))
                    for item in matches
                }
            )
        )
        evidence = tuple(
            sorted({*paraphrase_evidence, citation.record_key})
        )
        selected_generation = tuple(generation.selection.selected_candidate_keys)
        if len(selected_generation) != 1:
            raise W08P3IaError("P3-Ia generation selection is not exact")
        selected = (
            proposition.candidate_key.components,
            qa_result.candidates[0].stable_key(),
            selected_generation[0],
        )
        uses = tuple(
            W08P3IaUse(
                consumer,
                request.request_key,
                selected[index],
                evidence,
                digest_value(
                    {
                        "request": list(request.request_key),
                        "consumer": consumer,
                        "kind": "choice",
                    }
                ),
                digest_value(
                    {
                        "request": list(request.request_key),
                        "consumer": consumer,
                        "kind": "use",
                    }
                ),
                "RESOLVED",
                digest_value(
                    {
                        "request": list(request.request_key),
                        "consumer": consumer,
                        "kind": "outcome",
                    }
                ),
            )
            for index, consumer in enumerate(W08_CONSUMER_KEYS)
        )
        hierarchy_key = digest_value(
            {
                "document": list(hierarchy.document.document_key.components),
                "ranges": [list(item) for item in hierarchy.ranges()],
                "evidence": [
                    list(item.evidence_key.components) for item in hierarchy.evidence
                ],
            }
        )
        trace = W08P3IaTrace(
            hierarchy_key,
            tuple(sorted((section.candidate_key, paragraph.candidate_key, proposition.candidate_key))),
            paraphrase_evidence,
            tuple(sorted(item.center_key for item in recall.centers)),
            exact.receipt.obligation_key.components,
            citation.record_key,
            citation.span.start,
            citation.span.end,
            qa_result.stable_key(),
            generation.stable_key(),
            revision.result_key(),
        )
        return W08P3IaAuditReceipt(
            request.request_key,
            "RESOLVED",
            trace,
            self._resources(
                hierarchy=hierarchy,
                recall=recall,
                consumers=len(uses),
                recompute_objects=revision.recompute_object_count,
            ),
            uses,
            tuple(calls),
        )


__all__ = [
    "W08P3IaExecution",
    "W08P3IaFacade",
    "W08P3IaGenerationOwner",
    "W08P3IaOwners",
]
