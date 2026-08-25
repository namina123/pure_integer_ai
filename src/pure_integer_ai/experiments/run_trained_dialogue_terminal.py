"""K 盘训练状态驱动的公开交互终端。

这是面向真实训练结果的轻量 host 入口：QA SQLite 以只读方式打开，训练 run
只读加载，回答仍由既有窄域/广域查询和表层组织 runtime 产生。Core、Runtime
和训练账本不会因交互写入；默认会话热区仅存在当前进程，也可通过独立 K 盘
``session_root`` 以纯整数 checkpoint 持久化最近 8 轮。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys
import time
from typing import BinaryIO, Callable
from typing import Any

from pure_integer_ai.experiments.conversation_broad_qa_runtime import (
    BroadDialogueState,
    answer_broad_dialogue_turn,
)
from pure_integer_ai.experiments.conversation_broad_dialogue_persistence import (
    BroadDialoguePersistenceError,
    recover_broad_dialogue_checkpoint,
    write_broad_dialogue_checkpoint,
)
from pure_integer_ai.experiments.conversation_broad_runtime_memory_bridge import (
    BroadDialogueRuntimeMemoryError,
    append_dialogue_turn_to_runtime_memory,
    empty_runtime_memory_for_conversation,
    replay_dialogue_state_to_runtime_memory,
)
from pure_integer_ai.experiments.conversation_dialogue_scale_showcase import (
    load_training_observation,
)
from pure_integer_ai.experiments.conversation_public_sentence_demo import (
    build_public_sentence_demo_catalog,
    run_public_sentence_demo_bytes,
)
from pure_integer_ai.experiments.conversation_training_pack import (
    load_dialogue_training_pack,
)
from pure_integer_ai.experiments.conversation_runtime_material_binding_persistence import (
    RuntimeMaterialBindingPersistenceError,
    load_runtime_material_response_provider,
)
from pure_integer_ai.experiments.conversation_runtime_material_persistence import (
    RuntimeMaterialPersistenceError,
    load_runtime_material_runtime,
    open_runtime_material_sqlite,
    rebuild_runtime_material_observations,
)
from pure_integer_ai.experiments.conversation_trained_surface_runtime import (
    TrainedSurfaceRuntime,
    load_trained_surface_runtime,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_sparse_qa_snapshot import (
    load_or_rebuild_public_sparse_qa_runtime,
)
from pure_integer_ai.experiments.run_conversation_training import (
    default_course_paths,
)
from pure_integer_ai.experiments.ph2_broad_qa_relation_evidence_learning import (
    learn_relation_evidence_model,
)
from pure_integer_ai.experiments.ph2_broad_qa_relation_role_evidence_learning import (
    learn_relation_role_evidence_model,
)
from pure_integer_ai.experiments.ph2_broad_qa_relation_marker_evidence_learning import (
    learn_relation_marker_evidence_model,
)
from pure_integer_ai.experiments.ph2_broad_qa_relation_answer_frame_learning import (
    learn_relation_answer_frame_model,
)
from pure_integer_ai.storage.k_run_boundary import (
    ensure_normal_relative_directory,
    open_existing_run_root,
)
from pure_integer_ai.storage.source_record import SourceRecordRepository


def _require_k_file(value: str | Path, *, label: str) -> Path:
    """核验只读外部索引位于 K 盘且确实存在。"""
    path = Path(value).resolve()
    if path.drive.upper() != "K:" or not path.is_file():
        raise ValueError(f"{label} 必须是 K 盘已存在文件")
    return path


def _course_paths_for_training_run(
        project_root: Path,
        training_run_root: Path,
        extra_course_paths: tuple[str | Path, ...],
        ) -> tuple[Path, ...]:
    """按训练 run 冻结的 source_files 重建完全相同的公开 pack。

    默认课程 glob 会遗漏训练时显式附加的公开课程；终端必须读取 run 自己
    的 manifest，不能靠调用方记忆补参数，也不能放宽 pack SHA 校验。
    """
    manifest_path = training_run_root / "dialogue_pack_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("training run 缺少 dialogue_pack_manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("training run pack manifest 不可回读") from error
    rows = manifest.get("source_files") if isinstance(manifest, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("training run pack manifest 缺少 source_files")
    resolved: list[Path] = []
    for row in rows:
        if not isinstance(row, list) or not row:
            raise ValueError("training run source_files 记录非法")
        candidate = Path(str(row[0])).resolve()
        try:
            candidate.relative_to(project_root)
        except ValueError as error:
            raise ValueError("training run source_files 越出 project_root") from error
        if not candidate.is_file():
            raise ValueError(f"training run source_files 缺失: {candidate}")
        resolved.append(candidate)
    for value in extra_course_paths:
        candidate = Path(value).resolve()
        try:
            candidate.relative_to(project_root)
        except ValueError as error:
            raise ValueError("extra course 越出 project_root") from error
        if candidate not in resolved:
            resolved.append(candidate)
    if len(resolved) != len(set(resolved)):
        raise ValueError("training run source_files 重复")
    return tuple(resolved)


def _narrow_answer(runtime, catalog, trained_surface: TrainedSurfaceRuntime | None):
    """建立只读窄域消费者；未命中交给广域来源查询。"""
    def answer(question: str) -> tuple[str, str] | None:
        result = run_public_sentence_demo_bytes(
            runtime, catalog, question.encode("utf-8"))
        if result.generated_proposition_surface is None:
            return None
        surface = result.generated_proposition_surface
        if trained_surface is not None:
            rendered = trained_surface.render(surface, response_act="ANSWER")
            if rendered.used:
                surface = rendered.surface
        return surface, "ANSWER"
    return answer


def _display(turn) -> str:
    """将既有 turn 的答案载荷投影为人可读的一行，不改动 turn 语义。"""
    if turn.citations:
        blocks = []
        for citation in turn.citations:
            block = citation.surface
            if citation.source_title:
                source = f"来源：{citation.source_title}"
                if citation.source_url:
                    source += f"（{citation.source_url}）"
                block += f"\n{source}"
            blocks.append(block)
        return "\n\n".join(blocks)
    if turn.display_answer:
        answer = turn.display_answer
    elif turn.status == "CLARIFY":
        answer = "请补充问题的范围或限定条件。"
    elif turn.status == "REPAIR":
        answer = "当前问题需要补充信息后才能回答。"
    else:
        answer = "当前公开资料无法确认这个问题。"
    if turn.source_title:
        source = f"来源：{turn.source_title}"
        if turn.source_url:
            source += f"（{turn.source_url}）"
        return f"{answer}\n{source}"
    return answer


def _protocol_turn_payload(turn) -> dict[str, object]:
    """把 DialogueTurn 投影为稳定的机器可读响应，不改变语义。"""
    return {
        "type": "response",
        "ordinal": turn.ordinal,
        "question": turn.question,
        "retrieval_question": turn.retrieval_question,
        "status": turn.status,
        "answer": turn.answer,
        "display_answer": turn.display_answer,
        "source_title": turn.source_title,
        "source_url": turn.source_url,
        "citations": [
            {
                "surface": citation.surface,
                "source_title": citation.source_title,
                "source_url": citation.source_url,
            }
            for citation in turn.citations
        ],
        "turn_key": list(turn.turn_key),
    }


def _write_protocol_payload(stream_out: BinaryIO,
                            payload: dict[str, object]) -> None:
    """以无 BOM UTF-8 JSONL 写出一条协议响应。"""
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    stream_out.write(encoded)
    stream_out.flush()


def run_trained_dialogue_terminal(
        *,
        project_root: str | Path,
        qa_database: str | Path,
        training_run_root: str | Path | None = None,
        session_root: str | Path | None = None,
        extra_course_paths: tuple[str | Path, ...] = (),
        extra_variant_course_paths: tuple[str | Path, ...] = (),
        extra_variant_evidence_paths: tuple[str | Path, ...] = (),
        runtime_material_answer: Callable[
            [str], tuple[str, str | None, str | None] | None] | None = None,
        runtime_material_response: Callable[
            [str], tuple[object, ...] | None]
        | None = None,
        runtime_material_binding_root: str | Path | None = None,
        runtime_material_source_records: SourceRecordRepository | None = None,
        runtime_material_observations: tuple[object, ...] = (),
        runtime_material_runtime_root: str | Path | None = None,
        runtime_material_runtime_database: str | Path | None = None,
        runtime_material_binding_relative_path: str | Path = (
            "runtime_material_response/bindings.int"),
        learned_relation_evidence_model: Any | None = None,
        learned_relation_role_evidence_model: Any | None = None,
        learned_relation_marker_evidence_model: Any | None = None,
        learned_relation_answer_frame_model: Any | None = None,
        input_stream: BinaryIO | None = None,
        output_stream: BinaryIO | None = None,
        protocol_stream: bool = False,
        metrics_output: str | Path | None = None,
        ) -> int:
    """运行可回放的只读交互会话。

    默认使用人类终端；``protocol_stream=True`` 时读取无 BOM UTF-8 JSONL，
    每行一个 ``{"id": ..., "op": "turn", "text": ...}`` 请求并输出一条
    稳定 JSON 响应。两种入口共享完全相同的查询、证据、拒答和 checkpoint 路径。
    """
    runtime_sqlite_runtime = None
    runtime_material_response_provider = None
    if runtime_material_runtime_database is not None:
        if runtime_material_runtime_root is None:
            raise ValueError("runtime material SQLite 必须同时指定 runtime ledger root")
        if (runtime_material_binding_root is not None
                and Path(runtime_material_binding_root).resolve()
                != Path(runtime_material_runtime_root).resolve()):
            raise ValueError("runtime material binding/runtime ledger root 不一致")
        runtime_sqlite_runtime = open_runtime_material_sqlite(
            runtime_material_runtime_database)
        runtime_material_source_records = runtime_sqlite_runtime.source_records
        runtime_material_runtime_context = runtime_sqlite_runtime.context
        runtime_material_binding_root = runtime_material_runtime_root
    else:
        runtime_material_runtime_context = None

    if runtime_material_runtime_root is not None:
        if runtime_material_binding_root is not None and (
                Path(runtime_material_binding_root).resolve()
                != Path(runtime_material_runtime_root).resolve()):
            raise ValueError("runtime material binding/runtime ledger root 不一致")
        if runtime_material_response is not None or runtime_material_answer is not None:
            raise ValueError(
                "runtime ledger 不得与手工 Runtime provider 同时指定")
        def _runtime_fail_closed(_question: str):
            return "UNKNOWN", None, None, None
        if (runtime_material_runtime_context is None
                or not isinstance(runtime_material_source_records,
                                   SourceRecordRepository)):
            runtime_material_response = _runtime_fail_closed
        else:
            try:
                recovery = load_runtime_material_runtime(
                    runtime_material_runtime_root,
                    source_records=runtime_material_source_records,
                )
                observations = rebuild_runtime_material_observations(
                    runtime_material_runtime_context,
                    recovery,
                    source_records=runtime_material_source_records,
                )
                restored_provider = load_runtime_material_response_provider(
                    runtime_material_runtime_root,
                    source_records=runtime_material_source_records,
                    observations=observations,
                    relative_path=runtime_material_binding_relative_path,
                )
                runtime_material_response_provider = restored_provider
                runtime_material_response = restored_provider.response
            except (RuntimeMaterialPersistenceError,
                    RuntimeMaterialBindingPersistenceError,
                    TypeError, ValueError):
                runtime_material_response = _runtime_fail_closed

    if runtime_material_binding_root is not None and runtime_material_runtime_root is None:
        if runtime_material_response is not None or runtime_material_answer is not None:
            raise ValueError(
                "binding ledger 不得与手工 Runtime provider 同时指定")
        def _binding_fail_closed(_question: str):
            return "UNKNOWN", None, None, None
        if (not isinstance(runtime_material_source_records, SourceRecordRepository)
                or not isinstance(runtime_material_observations, tuple)
                or not runtime_material_observations):
            # ledger 存在但 observation source 缺失时，禁止落入窄域/广域
            # fallback；调用方仍可结束会话，但所有问题都保持未知。
            runtime_material_response = _binding_fail_closed
        else:
            try:
                restored_provider = load_runtime_material_response_provider(
                    runtime_material_binding_root,
                    source_records=runtime_material_source_records,
                    observations=runtime_material_observations,
                    relative_path=runtime_material_binding_relative_path,
                )
                runtime_material_response_provider = restored_provider
                runtime_material_response = restored_provider.response
            except (RuntimeMaterialBindingPersistenceError, TypeError, ValueError):
                runtime_material_response = _binding_fail_closed
    database = _require_k_file(qa_database, label="qa_database")
    root = Path(project_root).resolve()
    trained_surface = None
    if training_run_root is not None:
        run_root = Path(training_run_root).resolve()
        if run_root.drive.upper() != "K:" or not run_root.is_dir():
            raise ValueError("training_run_root 必须是 K 盘已存在目录")
        course_paths = _course_paths_for_training_run(
            root, run_root, extra_course_paths)
        pack = load_dialogue_training_pack(course_paths)
        load_training_observation(
            run_root, expected_pack_sha256=pack.pack_sha256)
        trained_surface = load_trained_surface_runtime(
            project_root=root,
            training_run_root=run_root,
            expected_pack_sha256=pack.pack_sha256,
            extra_variant_course_paths=tuple(
                Path(item).resolve() for item in extra_variant_course_paths),
            extra_variant_evidence_paths=tuple(
                Path(item).resolve() for item in extra_variant_evidence_paths),
        )
    session_capability = None
    if session_root is not None:
        session_path = Path(session_root).resolve()
        if session_path.drive.upper() != "K:" or not session_path.is_dir():
            raise ValueError("session_root 必须是 K 盘已存在目录")
        if training_run_root is not None and session_path == Path(training_run_root).resolve():
            raise ValueError("session_root 不得与 training_run_root 相同")
        session_capability = open_existing_run_root(
            session_path, label="broad dialogue session root")
        ensure_normal_relative_directory(
            session_capability, "broad_dialogue_checkpoints",
            label="broad dialogue checkpoint directory")
    sparse_runtime = load_or_rebuild_public_sparse_qa_runtime()
    narrow = _narrow_answer(
        sparse_runtime,
        build_public_sentence_demo_catalog(sparse_runtime),
        trained_surface,
    )
    stream_in = sys.stdin.buffer if input_stream is None else input_stream
    stream_out = sys.stdout.buffer if output_stream is None else output_stream
    state = BroadDialogueState((1, 1, 8))
    runtime_memory_state = None
    if session_capability is not None:
        checkpoint_dir = session_capability.path / "broad_dialogue_checkpoints"
        if any(item.is_file() and item.suffix == ".int"
               and item.name.startswith("checkpoint-")
               for item in checkpoint_dir.iterdir()):
            try:
                recovery = recover_broad_dialogue_checkpoint(
                    session_capability.path,
                    require_k_drive=not session_capability.test_transport,
                )
                state = recovery.checkpoint.state
                runtime_memory_state = recovery.checkpoint.runtime_memory_state
                if runtime_memory_state is None:
                    runtime_memory_state = replay_dialogue_state_to_runtime_memory(state)
            except (BroadDialoguePersistenceError,
                    BroadDialogueRuntimeMemoryError) as error:
                raise ValueError(f"会话检查点无法恢复: {error}") from error
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    latency_us: list[int] = []
    try:
        while True:
            if not protocol_stream:
                stream_out.write("你> ".encode("utf-8"))
                stream_out.flush()
            raw = stream_in.readline()
            if raw == b"":
                break
            if protocol_stream:
                request: dict[str, object] = {}
                try:
                    request = json.loads(raw.decode("utf-8"))
                    if not isinstance(request, dict):
                        raise ValueError("请求必须是 JSON 对象")
                    request_id = request.get("id")
                    operation = request.get("op", "turn")
                    if operation in {"quit", "exit"}:
                        _write_protocol_payload(stream_out, {
                            "id": request_id, "type": "bye", "status": "OK",
                        })
                        break
                    if operation != "turn":
                        raise ValueError("op 必须是 turn、quit 或 exit")
                    question = request.get("text")
                    if type(question) is not str or not question.strip():
                        raise ValueError("turn.text 必须是非空字符串")
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                    _write_protocol_payload(stream_out, {
                        "id": request.get("id") if 'request' in locals()
                        and isinstance(request, dict) else None,
                        "type": "error",
                        "status": "INVALID_REQUEST",
                        "error": str(error),
                    })
                    continue
            else:
                payload = raw.rstrip(b"\r\n")
                if payload in {b":quit", b":exit"}:
                    break
                try:
                    question = payload.decode("utf-8")
                except UnicodeDecodeError:
                    stream_out.write("系统> 输入必须是 UTF-8 文本。\n".encode("utf-8"))
                    stream_out.flush()
                    continue
            if not question.strip():
                continue
            runtime_response = runtime_material_response
            if runtime_material_response_provider is not None:
                provider = runtime_material_response_provider
                prior_state = state

                def _runtime_response_with_focus(
                        value: str,
                        *,
                        _provider=provider,
                        _prior_state=prior_state,
                        ):
                    exact = _provider.response_with_citations(value)
                    if exact is not None:
                        return exact
                    if _prior_state.turns:
                        previous = _prior_state.turns[-1]
                        if (previous.status == "ANSWER"
                                and previous.source_title):
                            followup = _provider.response_followup_with_citations(
                                value, previous.source_title)
                            if followup is not None:
                                return followup
                            related = _provider.response_related_with_citations(
                                value, previous.source_title)
                            if related is not None:
                                return related
                    related = _provider.response_related_with_citations(value)
                    if related is not None:
                        return related
                    return None

                runtime_response = _runtime_response_with_focus
            started_ns = time.perf_counter_ns()
            state, turn = answer_broad_dialogue_turn(
                state, question, connection, narrow_answer=narrow,
                runtime_material_answer=runtime_material_answer,
                runtime_material_response=runtime_response,
                learned_relation_evidence_model=(
                    learned_relation_evidence_model),
                learned_relation_role_evidence_model=(
                    learned_relation_role_evidence_model),
                learned_relation_marker_evidence_model=(
                    learned_relation_marker_evidence_model),
                learned_relation_answer_frame_model=(
                    learned_relation_answer_frame_model))
            latency_us.append(max(0, (time.perf_counter_ns() - started_ns) // 1000))
            if session_capability is not None:
                try:
                    if runtime_memory_state is None:
                        runtime_memory_state = empty_runtime_memory_for_conversation(
                            state.conversation_key)
                    runtime_memory_state = append_dialogue_turn_to_runtime_memory(
                        runtime_memory_state, state.conversation_key, turn,
                    ).memory_after
                    write_broad_dialogue_checkpoint(
                        session_capability, state,
                        runtime_memory_state=runtime_memory_state,
                    )
                except (BroadDialoguePersistenceError,
                        BroadDialogueRuntimeMemoryError) as error:
                    raise ValueError(f"会话记忆写入失败: {error}") from error
            if protocol_stream:
                response = _protocol_turn_payload(turn)
                response["id"] = request_id
                _write_protocol_payload(stream_out, response)
            else:
                stream_out.write(("系统> " + _display(turn) + "\n").encode("utf-8"))
                stream_out.flush()
    finally:
        connection.close()
        if runtime_sqlite_runtime is not None:
            runtime_sqlite_runtime.close()
        if metrics_output is not None:
            metrics_path = Path(metrics_output).resolve()
            if metrics_path.drive.upper() != "K:":
                raise ValueError("metrics_output 必须是 K 盘路径")
            if not latency_us:
                raise ValueError("metrics_output 没有可记录的对话轮次")
            ordered = sorted(latency_us)
            p50 = ordered[(len(ordered) - 1) * 50 // 100]
            p95 = ordered[(len(ordered) - 1) * 95 // 100]
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(
                json.dumps({
                    "protocol": "jsonl" if protocol_stream else "terminal",
                    "turn_count": len(latency_us),
                    "latency_us": latency_us,
                    "p50_us": p50,
                    "p95_us": p95,
                }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n",
                encoding="utf-8",
                newline="\n",
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="run read-only trained dialogue terminal")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--qa-database", required=True)
    parser.add_argument("--training-run-root", default=None)
    parser.add_argument("--session-root", default=None,
                        help="可选 K 盘会话根；启用后跨进程恢复最近 8 轮")
    parser.add_argument("--protocol", choices=("terminal", "jsonl"),
                        default="terminal",
                        help="交互入口：人类终端或 UTF-8 JSONL 协议")
    parser.add_argument("--metrics-output", default=None,
                        help="可选 K 盘性能摘要 JSON 路径")
    parser.add_argument("--extra-course", action="append", default=[])
    parser.add_argument("--variant-course", action="append", default=[])
    parser.add_argument("--variant-evidence", action="append", default=[])
    parser.add_argument("--runtime-material-ledger-root", default=None)
    parser.add_argument("--runtime-material-sqlite", default=None)
    parser.add_argument("--relation-evidence-course", action="append", default=[],
                        help="可选公开关系/证据课程，可重复")
    parser.add_argument("--relation-role-evidence-course", action="append",
                        default=[], help="可选公开 value/qualifier 课程")
    parser.add_argument("--relation-marker-evidence-course", action="append",
                        default=[], help="可选公开 marker/value 课程")
    parser.add_argument("--relation-answer-frame-course", action="append",
                        default=[], help="可选公开回答句面课程")
    args = parser.parse_args(argv)
    relation_courses = tuple(Path(item).resolve()
                             for item in args.relation_evidence_course)
    relation_role_courses = tuple(Path(item).resolve()
                                  for item in args.relation_role_evidence_course)
    relation_marker_courses = tuple(Path(item).resolve()
                                    for item in args.relation_marker_evidence_course)
    relation_frame_courses = tuple(Path(item).resolve()
                                   for item in args.relation_answer_frame_course)
    return run_trained_dialogue_terminal(
        project_root=args.project_root,
        qa_database=args.qa_database,
        training_run_root=args.training_run_root,
        session_root=args.session_root,
        extra_course_paths=tuple(args.extra_course),
        extra_variant_course_paths=tuple(args.variant_course),
        extra_variant_evidence_paths=tuple(args.variant_evidence),
        runtime_material_runtime_root=args.runtime_material_ledger_root,
        runtime_material_runtime_database=args.runtime_material_sqlite,
        protocol_stream=(args.protocol == "jsonl"),
        metrics_output=args.metrics_output,
        learned_relation_evidence_model=(
            learn_relation_evidence_model(relation_courses)
            if relation_courses else None),
        learned_relation_role_evidence_model=(
            learn_relation_role_evidence_model(relation_role_courses)
            if relation_role_courses else None),
        learned_relation_marker_evidence_model=(
            learn_relation_marker_evidence_model(relation_marker_courses)
            if relation_marker_courses else None),
        learned_relation_answer_frame_model=(
            learn_relation_answer_frame_model(relation_frame_courses)
            if relation_frame_courses else None),
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "run_trained_dialogue_terminal"]
