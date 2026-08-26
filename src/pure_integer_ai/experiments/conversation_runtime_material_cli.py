"""Runtime 资料导入的用户侧装配入口。

该模块把一份 UTF-8 资料文件接到既有 Runtime/Companion/语言 observation
管线，再在一次新的 K 盘 run 中排他发布 SQLite、event/observation 账本和
显式 qualification binding。它不调用 LLM、不写 Core，也不从问题或原文猜
答案；``state``、``reason`` 和问题绑定必须由调用方明确提供。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    SourceRef,
    VersionBundle,
)
from pure_integer_ai.cognition.shared.learning_input_capsule import RuntimeMemoryState
from pure_integer_ai.cognition.shared.scope_identity import session_scope
from pure_integer_ai.experiments.conversation_raw_proposition_consumer import (
    QUALIFICATION_STATES,
    RawPropositionQualification,
)
from pure_integer_ai.experiments.conversation_runtime_material_binding_persistence import (
    persist_runtime_material_response_bindings,
)
from pure_integer_ai.experiments.conversation_runtime_material_ingest import (
    ingest_runtime_material,
)
from pure_integer_ai.experiments.conversation_runtime_material_language import (
    observe_runtime_material_language,
)
from pure_integer_ai.experiments.conversation_runtime_material_persistence import (
    persist_runtime_material_observation,
    write_runtime_material_manifest,
)
from pure_integer_ai.experiments.conversation_runtime_material_response import (
    RuntimeMaterialResponseSpec,
    build_runtime_material_response_provider,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import SQLiteBackend
from pure_integer_ai.storage.k_run_boundary import create_new_run_root


class RuntimeMaterialCliError(ValueError):
    """导入参数、来源链或发布边界不满足合同。"""


@dataclass(frozen=True, slots=True)
class RuntimeMaterialBindingRequest:
    """一条由调用方明确给出的资料问题、关系和资格。"""

    question: str
    qualification_state: str
    reason_id: str
    relation_index: int = 0
    source_title: str | None = None
    source_url: str | None = None

    def __post_init__(self) -> None:
        if type(self.question) is not str or not self.question.strip():
            raise RuntimeMaterialCliError("binding question 不得为空")
        if (type(self.qualification_state) is not str
                or self.qualification_state not in QUALIFICATION_STATES):
            raise RuntimeMaterialCliError("binding qualification_state 未注册")
        if type(self.reason_id) is not str or not self.reason_id.strip():
            raise RuntimeMaterialCliError("binding reason_id 不得为空")
        if type(self.relation_index) is not int or self.relation_index < 0:
            raise RuntimeMaterialCliError("binding relation_index 必须为非负整数")
        for value, label in ((self.source_title, "binding source_title"),
                             (self.source_url, "binding source_url")):
            if value is not None and (
                    type(value) is not str or not value.strip()):
                raise RuntimeMaterialCliError(f"{label} 非法")


@dataclass(frozen=True, slots=True)
class RuntimeMaterialDocumentRequest:
    """批量资料清单中的一份来源、scope 和问题绑定。"""

    material_file: str
    source_kind: int
    source_id: int
    document_id: int
    scope_id: int
    license_id: str
    batch_id: int
    authority_key: tuple[int, ...]
    version_key: tuple[int, ...]
    bindings: tuple[RuntimeMaterialBindingRequest, ...]

    def __post_init__(self) -> None:
        if type(self.material_file) is not str or not self.material_file.strip():
            raise RuntimeMaterialCliError("manifest material_file 不得为空")
        for value, label in ((self.source_kind, "source_kind"),
                             (self.source_id, "source_id"),
                             (self.scope_id, "scope_id")):
            if type(value) is not int or value <= 0:
                raise RuntimeMaterialCliError(
                    f"manifest {label} 必须为正整数")
        for value, label in ((self.document_id, "document_id"),
                             (self.batch_id, "batch_id")):
            if type(value) is not int or value < 0:
                raise RuntimeMaterialCliError(
                    f"manifest {label} 必须为非负整数")
        if type(self.license_id) is not str or not self.license_id.strip():
            raise RuntimeMaterialCliError("manifest license_id 不得为空")
        for value, label in ((self.authority_key, "authority_key"),
                             (self.version_key, "version_key")):
            if (not isinstance(value, tuple) or not value
                    or any(type(item) is not int or item < 0 for item in value)):
                raise RuntimeMaterialCliError(
                    f"manifest {label} 必须是非空整数数组")
        if (not isinstance(self.bindings, tuple) or not self.bindings
                or any(not isinstance(item, RuntimeMaterialBindingRequest)
                       for item in self.bindings)):
            raise RuntimeMaterialCliError("manifest bindings 不得为空")


def _positive(value: str, *, label: str) -> int:
    try:
        result = int(value, 10)
    except (TypeError, ValueError) as error:
        raise RuntimeMaterialCliError(f"{label} 必须是十进制整数") from error
    if result <= 0:
        raise RuntimeMaterialCliError(f"{label} 必须为正整数")
    return result


def _nonnegative(value: str, *, label: str) -> int:
    try:
        result = int(value, 10)
    except (TypeError, ValueError) as error:
        raise RuntimeMaterialCliError(f"{label} 必须是十进制整数") from error
    if result < 0:
        raise RuntimeMaterialCliError(f"{label} 必须为非负整数")
    return result


def _integer_key(value: str, *, label: str) -> tuple[int, ...]:
    parts = tuple(item.strip() for item in value.split(","))
    if not parts or any(not item for item in parts):
        raise RuntimeMaterialCliError(f"{label} 必须是逗号分隔的整数")
    result = tuple(_nonnegative(item, label=f"{label}[{index}]")
                   for index, item in enumerate(parts))
    if not result:
        raise RuntimeMaterialCliError(f"{label} 不得为空")
    return result


def _read_utf8(path: str | Path) -> str:
    candidate = Path(path).resolve()
    if not candidate.is_file():
        raise RuntimeMaterialCliError(f"资料文件不存在: {candidate}")
    try:
        payload = candidate.read_bytes()
        # DLG-RAW-00 明确拒绝 BOM；报告可操作原因，避免用户看到后续
        # "没有机械结构候选" 这一间接错误。
        if payload[:3] == b"\xef\xbb\xbf":
            raise RuntimeMaterialCliError(
                "资料必须是无 BOM 的 UTF-8；请以 UTF-8 (无签名) 重新保存")
        value = payload.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise RuntimeMaterialCliError("资料文件必须是可回读的 UTF-8") from error
    if not value or not value.strip():
        raise RuntimeMaterialCliError("资料文件不得为空")
    return value


def _require_new_k_root(path: str | Path, *, require_k_drive: bool) -> Path:
    candidate = Path(path).resolve()
    if require_k_drive and candidate.drive.upper() != "K:":
        raise RuntimeMaterialCliError("output_root 必须位于 K 盘")
    if candidate.exists():
        raise RuntimeMaterialCliError(
            "output_root 必须是尚不存在的新目录，不覆盖已有 Runtime 资料")
    return candidate


def _read_binding_requests(
        path: str | Path,
        *,
        require_k_drive: bool,
        default_source_title: str | None = None,
        default_source_url: str | None = None,
        ) -> tuple[RuntimeMaterialBindingRequest, ...]:
    """读取严格 JSONL binding 清单；正文和语义资格仍由调用方提供。"""
    candidate = Path(path).resolve()
    if require_k_drive and candidate.drive.upper() != "K:":
        raise RuntimeMaterialCliError("binding_file 必须位于 K 盘")
    if not candidate.is_file():
        raise RuntimeMaterialCliError(f"binding_file 不存在: {candidate}")
    try:
        payload = candidate.read_bytes()
        if payload[:3] == b"\xef\xbb\xbf":
            raise RuntimeMaterialCliError(
                "binding_file 必须是无 BOM 的 UTF-8")
        text = payload.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise RuntimeMaterialCliError(
            "binding_file 必须是可回读的无 BOM UTF-8") from error
    allowed = frozenset({
        "question", "qualification_state", "reason_id", "relation_index",
        "source_title", "source_url",
    })
    requests: list[RuntimeMaterialBindingRequest] = []
    seen_bindings: set[tuple[str, int]] = set()
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeMaterialCliError(
                f"binding_file JSONL 非法: line {line_number}") from error
        if not isinstance(record, dict) or not set(record).issubset(allowed):
            raise RuntimeMaterialCliError(
                f"binding_file 字段非法: line {line_number}")
        required = {"question", "qualification_state", "reason_id"}
        if not required.issubset(record):
            raise RuntimeMaterialCliError(
                f"binding_file 缺少必填字段: line {line_number}")
        relation_index = record.get("relation_index", 0)
        if type(relation_index) is not int or relation_index < 0:
            raise RuntimeMaterialCliError(
                f"binding_file relation_index 非法: line {line_number}")
        request = RuntimeMaterialBindingRequest(
            record["question"],
            record["qualification_state"],
            record["reason_id"],
            relation_index,
            record.get("source_title", default_source_title),
            record.get("source_url", default_source_url),
        )
        binding_key = (request.question, request.relation_index)
        if binding_key in seen_bindings:
            raise RuntimeMaterialCliError(
                f"binding_file question/relation 重复: line {line_number}")
        seen_bindings.add(binding_key)
        requests.append(request)
    if not requests:
        raise RuntimeMaterialCliError("binding_file 不得为空")
    return tuple(requests)


def _manifest_key(value: object, *, label: str) -> tuple[int, ...]:
    """核验 JSON manifest 中显式给出的非空非负整数数组。"""
    if (not isinstance(value, list) or not value
            or any(type(item) is not int or item < 0 for item in value)):
        raise RuntimeMaterialCliError(f"{label} 必须是非空非负整数数组")
    return tuple(value)


def _read_document_manifest(
        path: str | Path,
        *,
        require_k_drive: bool,
        ) -> tuple[RuntimeMaterialDocumentRequest, ...]:
    """读取多资料 JSONL manifest；每行保留独立来源和资格边界。"""
    candidate = Path(path).resolve()
    if require_k_drive and candidate.drive.upper() != "K:":
        raise RuntimeMaterialCliError("material_manifest 必须位于 K 盘")
    if not candidate.is_file():
        raise RuntimeMaterialCliError(
            f"material_manifest 不存在: {candidate}")
    payload = candidate.read_bytes()
    if payload[:3] == b"\xef\xbb\xbf":
        raise RuntimeMaterialCliError("material_manifest 必须是无 BOM UTF-8")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeMaterialCliError(
            "material_manifest 必须是无 BOM UTF-8") from error
    allowed = frozenset({
        "material_file", "source_kind", "source_id", "document_id",
        "scope_id", "license_id", "batch_id", "authority_key",
        "version_key", "bindings",
    })
    binding_allowed = frozenset({
        "question", "qualification_state", "reason_id", "relation_index",
        "source_title", "source_url",
    })
    documents: list[RuntimeMaterialDocumentRequest] = []
    source_keys: set[tuple[int, int, int]] = set()
    scopes: set[int] = set()
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeMaterialCliError(
                f"material_manifest JSONL 非法: line {line_number}") from error
        if not isinstance(record, dict) or set(record) != allowed:
            raise RuntimeMaterialCliError(
                f"material_manifest 字段集合漂移: line {line_number}")
        raw_bindings = record["bindings"]
        if not isinstance(raw_bindings, list) or not raw_bindings:
            raise RuntimeMaterialCliError(
                f"material_manifest bindings 为空: line {line_number}")
        bindings: list[RuntimeMaterialBindingRequest] = []
        binding_keys: set[tuple[str, int]] = set()
        for index, raw in enumerate(raw_bindings):
            if not isinstance(raw, dict) or not set(raw).issubset(binding_allowed):
                raise RuntimeMaterialCliError(
                    f"material_manifest binding 字段非法: line {line_number}")
            required = {"question", "qualification_state", "reason_id"}
            if not required.issubset(raw):
                raise RuntimeMaterialCliError(
                    f"material_manifest binding 缺字段: line {line_number}")
            binding = RuntimeMaterialBindingRequest(
                raw["question"], raw["qualification_state"], raw["reason_id"],
                raw.get("relation_index", 0), raw.get("source_title"),
                raw.get("source_url"))
            binding_key = (binding.question, binding.relation_index)
            if binding_key in binding_keys:
                raise RuntimeMaterialCliError(
                    f"material_manifest 文档内问题/relation 重复: line {line_number}")
            binding_keys.add(binding_key)
            bindings.append(binding)
        document = RuntimeMaterialDocumentRequest(
            record["material_file"], record["source_kind"], record["source_id"],
            record["document_id"], record["scope_id"], record["license_id"],
            record["batch_id"],
            _manifest_key(record["authority_key"], label="authority_key"),
            _manifest_key(record["version_key"], label="version_key"),
            tuple(bindings),
        )
        source_key = (document.source_kind, document.source_id,
                      document.document_id)
        if source_key in source_keys or document.scope_id in scopes:
            raise RuntimeMaterialCliError(
                f"material_manifest source/scope 重复: line {line_number}")
        source_keys.add(source_key)
        scopes.add(document.scope_id)
        documents.append(document)
    if not documents:
        raise RuntimeMaterialCliError("material_manifest 不得为空")
    return tuple(documents)


def build_runtime_material_run(
        *,
        material_file: str | Path,
        output_root: str | Path,
        source_kind: int,
        source_id: int,
        document_id: int,
        scope_id: int,
        license_id: str,
        batch_id: int,
        authority_key: tuple[int, ...],
        version_key: tuple[int, ...],
        question: str | None = None,
        qualification_state: str | None = None,
        reason_id: str | None = None,
        source_title: str | None = None,
        source_url: str | None = None,
        relation_index: int = 0,
        binding_requests: tuple[RuntimeMaterialBindingRequest, ...] | None = None,
        require_k_drive: bool = True,
        ) -> tuple[Path, Path]:
    """创建一份可直接被训练终端读取的 K 盘 Runtime 资料 run。

    旧的单问题参数仍可用；``binding_requests`` 用于一次发布同一资料的
    多条显式问题绑定。两种入口不能混用，也不会自动从正文生成问题或资格。
    """
    if type(source_kind) is not int or source_kind <= 0:
        raise RuntimeMaterialCliError("source_kind 必须为正整数")
    if type(source_id) is not int or source_id <= 0:
        raise RuntimeMaterialCliError("source_id 必须为正整数")
    if type(document_id) is not int or document_id < 0:
        raise RuntimeMaterialCliError("document_id 必须为非负整数")
    if type(scope_id) is not int or scope_id <= 0:
        raise RuntimeMaterialCliError("scope_id 必须为正整数")
    if type(batch_id) is not int or batch_id < 0:
        raise RuntimeMaterialCliError("batch_id 必须为非负整数")
    if not isinstance(license_id, str) or not license_id.strip():
        raise RuntimeMaterialCliError("license_id 不得为空")
    if binding_requests is not None:
        if (not isinstance(binding_requests, tuple)
                or not binding_requests
                or any(not isinstance(item, RuntimeMaterialBindingRequest)
                       for item in binding_requests)):
            raise RuntimeMaterialCliError(
                "binding_requests 必须是非空 RuntimeMaterialBindingRequest tuple")
        if any(value is not None for value in (question, qualification_state,
                                               reason_id)):
            raise RuntimeMaterialCliError(
                "binding_requests 不得与单问题参数同时提供")
        if len({(item.question, item.relation_index)
                for item in binding_requests}) != len(binding_requests):
            raise RuntimeMaterialCliError(
                "binding_requests question/relation 不得重复")
    else:
        if not isinstance(question, str) or not question.strip():
            raise RuntimeMaterialCliError("question 不得为空")
        if qualification_state not in QUALIFICATION_STATES:
            raise RuntimeMaterialCliError("qualification_state 未注册")
        if not isinstance(reason_id, str) or not reason_id.strip():
            raise RuntimeMaterialCliError("reason_id 不得为空")
        binding_requests = (RuntimeMaterialBindingRequest(
            question, qualification_state, reason_id, relation_index,
            source_title, source_url),)
    if type(relation_index) is not int or relation_index < 0:
        raise RuntimeMaterialCliError("relation_index 必须为非负整数")
    if type(require_k_drive) is not bool:
        raise TypeError("require_k_drive 必须是 bool")
    raw_text = _read_utf8(material_file)
    root_path = _require_new_k_root(
        output_root, require_k_drive=require_k_drive)
    root = create_new_run_root(
        root_path,
        require_k_drive=require_k_drive,
        label="runtime material output root",
    )
    database = root.path / "runtime.sqlite3"
    backend = SQLiteBackend(str(database))
    try:
        bootstrap(backend)
        context = make_train_context(backend, companion=True)
        repository = context.memory_read_intake.source_intake.repository
        companion = context.memory_read_intake.source_intake.companion
        source = SourceRef(
            source_kind, source_id, document_id,
            GLOBAL_OWNER_SCOPE, VersionBundle(),
        )
        scope = session_scope(scope_id, source=source)
        source_intake = context.memory_read_intake.source_intake
        # Companion association is an internal append-only identity.  Generate
        # it through SourceIntake, then reuse the returned metadata; callers
        # must not guess or provide the association ordinal.
        source_record = source_intake.ensure(
            source, raw_text, license_id=license_id, batch_id=batch_id)
        ingest = ingest_runtime_material(
            RuntimeMemoryState(scope.stable_key()),
            source=source,
            scope=scope,
            raw_text=raw_text,
            source_records=repository,
            metadata=source_record.metadata,
            source_intake=source_intake,
            version_key=version_key,
            authority_key=authority_key,
        )
        observation = observe_runtime_material_language(
            context,
            ingest,
            observation_id=f"runtime-cli-{source_id}",
            context_id=f"runtime-cli-context-{scope_id}",
            family_id="runtime-cli-material-v1",
            source_namespace="runtime-cli",
        )
        persist_runtime_material_observation(root, observation)
        specs: list[RuntimeMaterialResponseSpec] = []
        for ordinal, request in enumerate(binding_requests):
            if request.relation_index >= len(observation.relation_candidates):
                raise RuntimeMaterialCliError(
                    "relation_index 超出资料 observation 的真实 relation candidate")
            relation = observation.relation_candidates[request.relation_index]
            qualification = RawPropositionQualification(
                f"runtime-cli-qualification-{source_id}-{ordinal}",
                relation.proposition.proposition_id,
                observation.raw_observation.observation_id,
                observation.raw_observation.source_id,
                observation.raw_observation.context_id,
                observation.raw_observation.family_id,
                observation.raw_observation.source_namespace,
                observation.raw_observation.split,
                request.qualification_state,
                request.reason_id,
                tuple(item.evidence_id for item in relation.evidence),
                "runtime-cli-explicit-authority",
            )
            specs.append(RuntimeMaterialResponseSpec(
                observation,
                qualification,
                request.question,
                request.relation_index,
                request.source_title,
                request.source_url,
            ))
        provider = build_runtime_material_response_provider(
            tuple(specs), source_records=repository)
        persist_runtime_material_response_bindings(root, provider)
        backend.commit()
        write_runtime_material_manifest(
            root, source_records=repository, database_path=database)
    finally:
        backend.close()
    return root.path, database


def build_runtime_material_manifest_run(
        *,
        documents: tuple[RuntimeMaterialDocumentRequest, ...],
        output_root: str | Path,
        require_k_drive: bool = True,
        ) -> tuple[Path, Path]:
    """把多份独立资料发布为一个 SQLite、多 scope 账本和统一 binding ledger。"""
    if (not isinstance(documents, tuple) or not documents
            or any(not isinstance(item, RuntimeMaterialDocumentRequest)
                   for item in documents)):
        raise RuntimeMaterialCliError(
            "documents 必须是非空 RuntimeMaterialDocumentRequest tuple")
    if len({(item.source_kind, item.source_id, item.document_id)
            for item in documents}) != len(documents):
        raise RuntimeMaterialCliError("documents source identity 不得重复")
    if len({item.scope_id for item in documents}) != len(documents):
        raise RuntimeMaterialCliError("documents scope identity 不得重复")
    root_path = _require_new_k_root(output_root, require_k_drive=require_k_drive)
    root = create_new_run_root(
        root_path, require_k_drive=require_k_drive,
        label="runtime material manifest output root")
    database = root.path / "runtime.sqlite3"
    backend = SQLiteBackend(str(database))
    try:
        bootstrap(backend)
        context = make_train_context(backend, companion=True)
        repository = context.memory_read_intake.source_intake.repository
        source_intake = context.memory_read_intake.source_intake
        specs: list[RuntimeMaterialResponseSpec] = []
        for document_ordinal, document in enumerate(documents):
            raw_text = _read_utf8(document.material_file)
            source = SourceRef(
                document.source_kind, document.source_id, document.document_id,
                GLOBAL_OWNER_SCOPE, VersionBundle())
            scope = session_scope(document.scope_id, source=source)
            source_record = source_intake.ensure(
                source, raw_text, license_id=document.license_id,
                batch_id=document.batch_id)
            ingest = ingest_runtime_material(
                RuntimeMemoryState(scope.stable_key()), source=source, scope=scope,
                raw_text=raw_text, source_records=repository,
                metadata=source_record.metadata, source_intake=source_intake,
                version_key=document.version_key,
                authority_key=document.authority_key)
            observation = observe_runtime_material_language(
                context, ingest,
                observation_id=f"runtime-manifest-{document.source_id}",
                context_id=f"runtime-manifest-context-{document.scope_id}",
                family_id="runtime-manifest-material-v1",
                source_namespace="runtime-manifest")
            persist_runtime_material_observation(root, observation)
            for binding_ordinal, request in enumerate(document.bindings):
                if request.relation_index >= len(observation.relation_candidates):
                    raise RuntimeMaterialCliError(
                        "manifest relation_index 超出真实 relation candidate")
                relation = observation.relation_candidates[request.relation_index]
                qualification = RawPropositionQualification(
                    "runtime-manifest-qualification-"
                    f"{document.source_id}-{binding_ordinal}",
                    relation.proposition.proposition_id,
                    observation.raw_observation.observation_id,
                    observation.raw_observation.source_id,
                    observation.raw_observation.context_id,
                    observation.raw_observation.family_id,
                    observation.raw_observation.source_namespace,
                    observation.raw_observation.split,
                    request.qualification_state, request.reason_id,
                    tuple(item.evidence_id for item in relation.evidence),
                    "runtime-manifest-explicit-authority")
                specs.append(RuntimeMaterialResponseSpec(
                    observation, qualification, request.question,
                    request.relation_index, request.source_title,
                    request.source_url))
        provider = build_runtime_material_response_provider(
            tuple(specs), source_records=repository)
        persist_runtime_material_response_bindings(root, provider)
        backend.commit()
        write_runtime_material_manifest(
            root, source_records=repository, database_path=database)
    finally:
        backend.close()
    return root.path, database


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="将一份或多份 UTF-8 资料接入 K 盘 Runtime 对话账本")
    material_group = parser.add_mutually_exclusive_group(required=True)
    material_group.add_argument("--material-file",
                        help="单份 UTF-8 原文文件；不会复制进 D 盘或 Git")
    material_group.add_argument("--material-manifest",
                        help="K 盘无 BOM UTF-8 JSONL 多资料清单")
    parser.add_argument("--output-root", required=True,
                        help="K 盘新目录；必须不存在")
    parser.add_argument("--source-kind", default=None,
                        help="正整数来源类型")
    parser.add_argument("--source-id", default=None,
                        help="正整数来源 ID")
    parser.add_argument("--document-id", default="0",
                        help="非负文档 ID，默认 0")
    parser.add_argument("--scope-id", default=None,
                        help="正整数 Runtime session scope ID")
    parser.add_argument("--license-id", default=None,
                        help="原文许可标识，例如 CC0-1.0")
    parser.add_argument("--batch-id", default=None,
                        help="非负来源批次 ID")
    parser.add_argument("--authority-key", default=None,
                        help="逗号分隔的非负整数 authority key")
    parser.add_argument("--version-key", default="1",
                        help="逗号分隔的非负整数版本 key，默认 1")
    question_group = parser.add_mutually_exclusive_group()
    question_group.add_argument("--question",
                        help="显式绑定到该资料 relation 的单个问题")
    question_group.add_argument("--binding-file", default=None,
                        help="K 盘无 BOM UTF-8 JSONL 多问题绑定清单")
    parser.add_argument("--qualification-state", default=None,
                        choices=tuple(sorted(QUALIFICATION_STATES)),
                        help="显式资格：SUPPORTED/UNKNOWN/CONFLICT")
    parser.add_argument("--reason-id", default=None,
                        help="资格理由标识，不接受空值")
    parser.add_argument("--relation-index", default="0",
                        help="资料 observation 中的 relation candidate 序号")
    parser.add_argument("--source-title", default=None)
    parser.add_argument("--source-url", default=None)
    args = parser.parse_args(argv)
    if args.material_manifest is not None:
        forbidden = (
            args.source_kind, args.source_id, args.scope_id, args.license_id,
            args.batch_id, args.authority_key, args.question, args.binding_file,
            args.qualification_state, args.reason_id, args.source_title,
            args.source_url,
        )
        if any(value is not None for value in forbidden):
            parser.error("--material-manifest 不得与单资料参数混用")
        documents = _read_document_manifest(
            args.material_manifest, require_k_drive=True)
        root, database = build_runtime_material_manifest_run(
            documents=documents, output_root=args.output_root)
        print(f"已创建多资料 Runtime run：{root}")
        print(f"Runtime SQLite：{database}")
        print("多份资料保留独立来源/scope；统一 event/observation/binding 为整数账本。")
        return 0
    required_values = {
        "source_kind": args.source_kind, "source_id": args.source_id,
        "scope_id": args.scope_id, "license_id": args.license_id,
        "batch_id": args.batch_id, "authority_key": args.authority_key,
    }
    missing = tuple(name for name, value in required_values.items()
                    if value is None)
    if missing:
        parser.error("单资料入口缺少参数: " + ", ".join(missing))
    if args.question is None and args.binding_file is None:
        parser.error("单资料入口必须提供 --question 或 --binding-file")
    binding_requests = None
    if args.binding_file is not None:
        binding_requests = _read_binding_requests(
            args.binding_file,
            require_k_drive=True,
            default_source_title=args.source_title,
            default_source_url=args.source_url,
        )
    elif args.qualification_state is None or args.reason_id is None:
        parser.error("单问题入口必须同时提供 --qualification-state 和 --reason-id")
    root, database = build_runtime_material_run(
        material_file=args.material_file,
        output_root=args.output_root,
        source_kind=_positive(args.source_kind, label="source_kind"),
        source_id=_positive(args.source_id, label="source_id"),
        document_id=_nonnegative(args.document_id, label="document_id"),
        scope_id=_positive(args.scope_id, label="scope_id"),
        license_id=args.license_id,
        batch_id=_nonnegative(args.batch_id, label="batch_id"),
        authority_key=_integer_key(args.authority_key, label="authority_key"),
        version_key=_integer_key(args.version_key, label="version_key"),
        question=args.question,
        qualification_state=args.qualification_state,
        reason_id=args.reason_id,
        source_title=args.source_title,
        source_url=args.source_url,
        relation_index=_nonnegative(args.relation_index, label="relation_index"),
        binding_requests=binding_requests,
    )
    print(f"已创建 Runtime 资料 run：{root}")
    print(f"Runtime SQLite：{database}")
    print("资料正文留在 K 盘 SQLite Companion 层；event/observation/binding 为整数账本。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RuntimeMaterialCliError",
    "RuntimeMaterialBindingRequest",
    "RuntimeMaterialDocumentRequest",
    "build_runtime_material_run",
    "build_runtime_material_manifest_run",
    "main",
]
