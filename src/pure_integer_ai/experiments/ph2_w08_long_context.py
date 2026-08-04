"""协调现役长上下文 owner 的 W08-05 facade。"""
from __future__ import annotations

from pure_integer_ai.experiments.authorized_center_runtime import AuthorizedCenterAgendaRun
from pure_integer_ai.experiments.long_generation_checkpoint import (
    LONG_GENERATION_COMPLETE,
    LongGenerationCheckpoint,
)
from pure_integer_ai.experiments.ph2_w08_contract import W08_CONSUMER_KEYS
from pure_integer_ai.experiments.ph2_w08_long_context_adapters import (
    W08LongContextExecution,
    W08LongContextOwners,
)
from pure_integer_ai.experiments.ph2_w08_long_context_contract import (
    W08LongContextAuditReceipt,
    W08LongContextError,
    W08LongContextRequest,
    W08LongContextResourceReceipt,
    W08LongContextTrace,
    W08LongContextUse,
    W08_LONG_CONTEXT_COMPONENT_KEYS,
    W08_LONG_CONTEXT_OWNER_KEYS,
)


class W08LongContextFacade:
    """把 typed W08 请求绑定到 R-06/R-04 owner，不建立第二套引擎。"""

    def __init__(self, owners: W08LongContextOwners) -> None:
        if not isinstance(owners, W08LongContextOwners):
            raise TypeError("W08 long-context facade requires W08LongContextOwners")
        self.owners = owners

    @staticmethod
    def _resources(
        *,
        state: str,
        execution: W08LongContextExecution,
        agenda_entries: int,
        centers: AuthorizedCenterAgendaRun | None,
        checkpoint: LongGenerationCheckpoint | None,
        uses: tuple[W08LongContextUse, ...],
    ) -> W08LongContextResourceReceipt:
        metrics = () if centers is None else tuple(
            item.exact.metrics for item in centers.record_reads
        )
        opened_pages = sum(item.page_faults for item in metrics)
        page_in_records = sum(item.page_in_records for item in metrics)
        payload_bytes = sum(item.cold_read_bytes for item in metrics)
        payload_gets = opened_pages
        logic_operations = (
            len(execution.material.seeds)
            + (0 if centers is None else len(centers.states))
            + (0 if checkpoint is None else checkpoint.next_cursor)
            + len(uses)
        )
        opened_segments = 0
        if centers is not None:
            opened_segments = len({
                item.segment_key
                for item in execution.authorization.bindings
                if item.center_key in {
                    state.center.center_key
                    for state in centers.states
                    if state.receipt.state in {"READY", "STOPPED"}
                }
            })
        return W08LongContextResourceReceipt(
            opened_segments,
            opened_pages,
            page_in_records,
            payload_gets,
            payload_bytes,
            agenda_entries,
            len(uses),
            execution.recompute_objects,
            logic_operations,
            0 if checkpoint is None else checkpoint.revision + 1,
            state,
        )

    @staticmethod
    def _stopped(
        request,
        execution,
        state: str,
        calls: list[str],
        *,
        blocked_component: str = "",
        agenda_entries: int = 0,
    ) -> W08LongContextAuditReceipt:
        return W08LongContextAuditReceipt(
            request.request_key,
            state,
            None,
            W08LongContextFacade._resources(
                state=state,
                execution=execution,
                agenda_entries=agenda_entries,
                centers=None,
                checkpoint=None,
                uses=(),
            ),
            (),
            tuple(calls),
            blocked_component,
        )

    @staticmethod
    def _failure_state(run: AuthorizedCenterAgendaRun) -> str:
        states = tuple(item.receipt.state for item in run.states)
        if any(state in {"ACL_DENIED", "CENTER_UNBOUND", "SCOPE_MISMATCH", "SOURCE_MISMATCH", "VERSION_MISMATCH", "MANIFEST_STALE", "POLICY_STALE", "SEGMENT_MISMATCH", "SEGMENT_NOT_ISOLATED"} for state in states):
            return "ACCESS_BLOCKED"
        if "READ_FAILED" in states:
            return "BUDGET_EXHAUSTED"
        return "UNKNOWN"

    def execute(self, request, execution: W08LongContextExecution) -> W08LongContextAuditReceipt:
        if not isinstance(request, W08LongContextRequest) or not isinstance(
            execution, W08LongContextExecution
        ):
            raise TypeError("W08 long-context request/execution type is invalid")
        calls: list[str] = []
        if request.training_material_key != execution.material.training_material_key:
            raise W08LongContextError("long-context request/material identity drifted")
        if not request.required_center_keys:
            return self._stopped(request, execution, "UNKNOWN", calls)
        if len(request.clarification_candidate_keys) > 1 and request.clarification_resolution_key is None:
            return self._stopped(request, execution, "CLARIFY", calls)
        if set(request.conflict_center_keys) - set(request.resolved_conflict_keys):
            return self._stopped(request, execution, "CLARIFY", calls)
        for component, stop_state in (
            (W08_LONG_CONTEXT_COMPONENT_KEYS[0], "GROUNDING_BLOCKED"),
            (W08_LONG_CONTEXT_COMPONENT_KEYS[1], "GROUNDING_BLOCKED"),
            (W08_LONG_CONTEXT_COMPONENT_KEYS[2], "ACCESS_BLOCKED"),
            (W08_LONG_CONTEXT_COMPONENT_KEYS[3], "GROUNDING_BLOCKED"),
        ):
            if not request.component_enabled(component):
                return self._stopped(
                    request,
                    execution,
                    stop_state,
                    calls,
                    blocked_component=component,
                )

        hierarchy, _ = self.owners.hierarchy.build(execution.material)
        calls.append(self.owners.hierarchy.owner_key)
        available = {item.center_key: item for item in execution.centers}
        if not set(request.required_center_keys).issubset(available):
            return self._stopped(request, execution, "UNKNOWN", calls)
        selected = tuple(available[key] for key in request.required_center_keys)
        agenda, bound = self.owners.agenda.open(
            request,
            execution.agenda_key,
            selected,
        )
        calls.append(self.owners.agenda.owner_key)
        run = self.owners.centers.run(request, execution, bound)
        calls.append(self.owners.centers.owner_key)
        if any(item.receipt.state != "READY" for item in run.states):
            return self._stopped(
                request,
                execution,
                self._failure_state(run),
                calls,
                agenda_entries=len(agenda.centers),
            )
        checkpoint = self.owners.checkpoint.run(request, execution, run)
        calls.append(self.owners.checkpoint.owner_key)
        state = (
            "RESOLVED"
            if checkpoint.status == LONG_GENERATION_COMPLETE
            else "BUDGET_EXHAUSTED"
        )
        if state != "RESOLVED":
            self.owners.agenda.record(request, agenda, run, checkpoint)
            return W08LongContextAuditReceipt(
                request.request_key,
                state,
                None,
                self._resources(
                    state=state,
                    execution=execution,
                    agenda_entries=len(agenda.centers),
                    centers=run,
                    checkpoint=checkpoint,
                    uses=(),
                ),
                (),
                tuple(calls),
            )
        advanced_agenda = self.owners.agenda.record(
            request,
            agenda,
            run,
            checkpoint,
        )
        uses = tuple(
            self.owners.consumers.consume(request, consumer, run, checkpoint)
            for consumer in W08_CONSUMER_KEYS
        )
        calls.append(self.owners.consumers.owner_key)
        if tuple(item.consumer_key for item in uses) != W08_CONSUMER_KEYS:
            raise W08LongContextError("long-context consumers are not in canonical order")
        citation_keys = tuple(sorted({
            citation.record_key
            for read in run.record_reads
            for citation in read.exact.receipt.citations
        }))
        receipt_keys = tuple(sorted(item.receipt.receipt_key for item in run.states))
        prefix_content = tuple(
            (record.prefix_digest, record.content_digest)
            for record in hierarchy.records
        )
        trace = W08LongContextTrace(
            hierarchy.stable_key(),
            prefix_content,
            tuple(sorted(item.center.center_key for item in run.states)),
            receipt_keys,
            citation_keys,
            advanced_agenda.digest(),
            checkpoint.digest(),
            checkpoint.prefix_digest,
            checkpoint.next_cursor,
        )
        return W08LongContextAuditReceipt(
            request.request_key,
            state,
            trace,
            self._resources(
                state=state,
                execution=execution,
                agenda_entries=len(advanced_agenda.centers),
                centers=run,
                checkpoint=checkpoint,
                uses=uses,
            ),
            uses,
            tuple(calls),
        )


__all__ = ["W08LongContextFacade"]
