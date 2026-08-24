"""P3-C3 的受预算 runtime artifact test-transport bridge.

这个模块刻意不复用 legacy runtime artifact writer。唯一公开入口接收同次 C2
typed input；它在内部重新执行 C2，随后才从 C1c descriptor 的固定 child 读取 capsule。
返回值不含本机路径、capsule、runtime result 或任何 runtime payload。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Callable

from pure_integer_ai.experiments.conversation_heldout_v4_bound_capsule_consumer_gate import (
    ConversationHeldOutV4BoundCapsuleConsumerGateError,
    ConversationHeldOutV4BoundCapsuleConsumerGateInput,
    ConversationHeldOutV4BoundCapsuleConsumerGateResult,
    V4_BOUND_CAPSULE_CONSUMER_GATE_STATUS_TEST_ONLY,
    run_v4_bound_capsule_consumer_zero_write_gate,
)
from pure_integer_ai.experiments.conversation_heldout_v4_candidate_runtime import (
    ConversationHeldOutV4CandidateRuntimeResult,
    ConversationHeldOutV4RuntimeSourceCapsule,
    ConversationHeldOutV4RuntimeStaticAssetReadBudget,
    ConversationHeldOutV4RuntimeStaticAssets,
    V4_RUNTIME_SOURCE_ORIGIN_EXTERNAL,
    read_v4_runtime_static_assets,
    run_v4_candidate_runtime,
)
from pure_integer_ai.experiments.conversation_heldout_v4_external_input_capsule import (
    ConversationHeldOutV4ExternalCapsuleError,
    read_budgeted_v4_external_input_capsule,
)
from pure_integer_ai.experiments.conversation_heldout_v4_freeze import verify_v4_freeze
from pure_integer_ai.experiments.conversation_heldout_v4_projection import (
    render_v4_html,
    render_v4_markdown,
)
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.experiments.ph2_dataset_core import canonical_json_bytes
from pure_integer_ai.storage.integer_codec import encode_integer_tuple, pack_key, strict_integer_tuple
from pure_integer_ai.storage.k_run_boundary import (
    KRunBoundaryError,
    KRunRoot,
    create_new_run_root,
    ensure_normal_relative_directory,
    open_existing_run_root,
    publish_manifest_last,
    require_disjoint_run_roots,
    require_exact_file_closure,
    require_fresh_empty_run_root,
    sha256_plain_file,
    write_exclusive_bytes,
)


V4_BOUND_RUNTIME_ARTIFACT_STATUS_TEST_TRANSPORT_UNQUALIFIED = (
    "P3_C3_BOUND_RUNTIME_ARTIFACT=TEST_TRANSPORT_UNQUALIFIED")
V4_BOUND_RUNTIME_ARTIFACT_ACTIVE_PROVENANCE_COVERAGE = "NE"
V4_BOUND_RUNTIME_ARTIFACT_MANIFEST_SCHEMA = (
    "dlg05-v4-p3c3-bound-runtime-artifact-v1")
V4_BOUND_RUNTIME_ARTIFACT_RESULT_SCHEMA = 1
V4_BOUND_RUNTIME_ARTIFACT_OUTPUT_BUDGET_SCHEMA = 1

_BUNDLE_FILE = Path("bundle.canonical.ints")
_RECEIPT_FILE = Path("runtime_receipt.canonical.ints")
_FREEZE_FILE = Path("freeze.json")
_PROJECTION_DIRECTORY = Path("projection")
_MARKDOWN_FILE = _PROJECTION_DIRECTORY / "dlg05_v4_reading.md"
_HTML_FILE = _PROJECTION_DIRECTORY / "dlg05_v4_reading.html"
_MANIFEST_FILE = Path("artifact_manifest.json")
_PRE_MANIFEST_FILES = frozenset({
    _BUNDLE_FILE, _RECEIPT_FILE, _FREEZE_FILE, _MARKDOWN_FILE, _HTML_FILE,
})
_EXPECTED_FILES = _PRE_MANIFEST_FILES | {_MANIFEST_FILE}
_C1C_RUNTIME_CAPSULE_DOMAIN = b"dlg05.v4.p3c1c.runtime-capsule.v1\x00"


class ConversationHeldOutV4BoundRuntimeArtifactWriterError(RuntimeError):
    """C3 的 C2 binding、受预算读取、runtime 或 artifact 闭合失败。"""


def _fail(message: str) -> None:
    raise ConversationHeldOutV4BoundRuntimeArtifactWriterError(message)


def _positive(value: int, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive strict integer")
    return value


def _stage(value: str) -> str:
    if (not isinstance(value, str) or not value or len(value) > 56
            or not value[0].isascii() or not value[0].isalnum()
            or any(not item.isascii() or not (item.isalnum() or item in {"-", "_", "."})
                   for item in value)):
        raise ValueError("P3-C3 logical_stage_name is invalid")
    return value


def _digest(value: bytes) -> tuple[int, ...]:
    return tuple(hashlib.sha256(value).digest())


def _key_digest(value: tuple[int, ...]) -> tuple[int, ...]:
    strict_integer_tuple(value, label="P3-C3 stable key")
    return _digest(encode_integer_tuple(value))


def _digest_hex(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _identity_document(value: tuple[int, ...]) -> str:
    return bytes(_key_digest(value)).hex()


@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4BoundRuntimeArtifactBudget:
    """C3 固定六文件闭包的单文件、总输出和限定读回上限。"""

    max_bundle_bytes: int
    max_runtime_receipt_bytes: int
    max_freeze_bytes: int
    max_projection_bytes: int
    max_manifest_bytes: int
    max_total_output_bytes: int
    max_total_read_bytes: int

    def __post_init__(self) -> None:
        values = self.integer_stream()[1:]
        for label, item in zip((
                "max_bundle_bytes", "max_runtime_receipt_bytes", "max_freeze_bytes",
                "max_projection_bytes", "max_manifest_bytes", "max_total_output_bytes",
                "max_total_read_bytes"), values):
            _positive(item, label=f"P3-C3 {label}")
        if self.max_total_read_bytes < self.max_manifest_bytes:
            raise ValueError("P3-C3 total read budget cannot hold manifest")

    def integer_stream(self) -> tuple[int, ...]:
        return (
            V4_BOUND_RUNTIME_ARTIFACT_OUTPUT_BUDGET_SCHEMA,
            self.max_bundle_bytes,
            self.max_runtime_receipt_bytes,
            self.max_freeze_bytes,
            self.max_projection_bytes,
            self.max_manifest_bytes,
            self.max_total_output_bytes,
            self.max_total_read_bytes,
        )


@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4BoundRuntimeArtifactWriterInput:
    """C3 唯一公开入口；不接受 root、capsule 或预构造 runtime result。"""

    bound_capsule_consumer_gate_input: ConversationHeldOutV4BoundCapsuleConsumerGateInput
    runtime_static_asset_budget: ConversationHeldOutV4RuntimeStaticAssetReadBudget
    artifact_budget: ConversationHeldOutV4BoundRuntimeArtifactBudget
    bridge_code_identity: ProtocolKey
    logical_stage_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.bound_capsule_consumer_gate_input,
                          ConversationHeldOutV4BoundCapsuleConsumerGateInput):
            raise TypeError("P3-C3 must receive a typed C2 input")
        if not isinstance(self.runtime_static_asset_budget,
                          ConversationHeldOutV4RuntimeStaticAssetReadBudget):
            raise TypeError("P3-C3 runtime static asset budget type is invalid")
        if not isinstance(self.artifact_budget,
                          ConversationHeldOutV4BoundRuntimeArtifactBudget):
            raise TypeError("P3-C3 artifact budget type is invalid")
        if not isinstance(self.bridge_code_identity, ProtocolKey):
            raise TypeError("P3-C3 bridge_code_identity must be ProtocolKey")
        _stage(self.logical_stage_name)


@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4BoundRuntimeArtifactCounts:
    """C3 唯一实际 C2、capsule、static/runtime 与 artifact I/O 账户。"""

    c2_runs: int
    capsule_reads: int
    static_asset_reads: int
    runtime_runs: int
    output_file_count: int
    output_byte_count: int
    output_read_byte_count: int

    def __post_init__(self) -> None:
        if (type(self.c2_runs) is not int or self.c2_runs != 1
                or type(self.capsule_reads) is not int or self.capsule_reads != 2
                or type(self.static_asset_reads) is not int or self.static_asset_reads != 2
                or type(self.runtime_runs) is not int or self.runtime_runs != 1
                or type(self.output_file_count) is not int or self.output_file_count != 6):
            raise ValueError("P3-C3 fixed operation counts are invalid")
        for label, item in (("output_byte_count", self.output_byte_count),
                            ("output_read_byte_count", self.output_read_byte_count)):
            if type(item) is not int or item <= 0:
                raise ValueError(f"P3-C3 {label} must be positive")

    def integer_stream(self) -> tuple[int, ...]:
        return (
            self.c2_runs, self.capsule_reads, self.static_asset_reads,
            self.runtime_runs, self.output_file_count, self.output_byte_count,
            self.output_read_byte_count,
        )


@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4BoundRuntimeArtifactWriterResult:
    """C3 payload-free result；路径、正文、问题、答案、label 与 runtime result 均不外泄。"""

    bound_gate_identity_sha256: tuple[int, ...]
    caller_identity_sha256: tuple[int, ...]
    capsule_budget_identity_sha256: tuple[int, ...]
    runtime_asset_budget_identity_sha256: tuple[int, ...]
    artifact_budget_identity_sha256: tuple[int, ...]
    artifact_manifest_sha256: tuple[int, ...]
    counts: ConversationHeldOutV4BoundRuntimeArtifactCounts
    status: str
    active_provenance_coverage: str

    def __post_init__(self) -> None:
        for label, value in (
                ("bound gate", self.bound_gate_identity_sha256),
                ("caller", self.caller_identity_sha256),
                ("capsule budget", self.capsule_budget_identity_sha256),
                ("runtime asset budget", self.runtime_asset_budget_identity_sha256),
                ("artifact budget", self.artifact_budget_identity_sha256),
                ("artifact manifest", self.artifact_manifest_sha256)):
            if (not isinstance(value, tuple) or len(value) != 32
                    or any(type(item) is not int or item < 0 or item > 255
                           for item in value)):
                raise ValueError(f"P3-C3 {label} digest is invalid")
        if not isinstance(self.counts, ConversationHeldOutV4BoundRuntimeArtifactCounts):
            raise TypeError("P3-C3 result counts type is invalid")
        if (self.status != V4_BOUND_RUNTIME_ARTIFACT_STATUS_TEST_TRANSPORT_UNQUALIFIED
                or self.active_provenance_coverage
                != V4_BOUND_RUNTIME_ARTIFACT_ACTIVE_PROVENANCE_COVERAGE):
            raise ValueError("P3-C3 result status is invalid")

    def stable_key(self) -> tuple[int, ...]:
        result = [V4_BOUND_RUNTIME_ARTIFACT_RESULT_SCHEMA]
        for value in (
                self.bound_gate_identity_sha256, self.caller_identity_sha256,
                self.capsule_budget_identity_sha256,
                self.runtime_asset_budget_identity_sha256,
                self.artifact_budget_identity_sha256, self.artifact_manifest_sha256,
                self.counts.integer_stream(),
                tuple(ord(item) for item in self.status),
                tuple(ord(item) for item in self.active_provenance_coverage)):
            pack_key(result, value)
        return tuple(result)


def _source_root_for(value: ConversationHeldOutV4BoundCapsuleConsumerGateInput) -> KRunRoot:
    """只由 C1c descriptor 的固定 child 重建 C3 的 source capability。"""
    adapter = value.adapter_result
    descriptor = adapter.capsule_descriptor
    caller = value.active_caller_gate_input
    expected = descriptor.parent_run_root.path / descriptor.relative_root
    if caller.source_capsule_root != expected:
        _fail("P3-C3 C2 source root is not the fixed C1c descriptor child")
    for label, path in (
            ("C1c capsule parent", descriptor.parent_run_root.path),
            ("C1c capsule child", expected),
            ("future artifact root", caller.future_artifact_root)):
        if path.drive.upper() != "D:":
            _fail(f"P3-C3 test transport {label} must be on explicit D drive")
    try:
        return open_existing_run_root(
            expected, require_k_drive=False, label="P3-C3 C1c capsule child")
    except KRunBoundaryError as exc:
        raise ConversationHeldOutV4BoundRuntimeArtifactWriterError(
            "P3-C3 C1c capsule child boundary failed") from exc


def _read_bound_capsule(
        source_root: KRunRoot,
        value: ConversationHeldOutV4BoundCapsuleConsumerGateInput,
        ) -> ConversationHeldOutV4RuntimeSourceCapsule:
    """仅通过 C1c 的 explicit external-capsule budget 读取 source。"""
    adapter = value.adapter_result
    descriptor = adapter.capsule_descriptor
    try:
        capsule = read_budgeted_v4_external_input_capsule(
            source_root.path,
            budget=adapter.receipt.budget.capsule_budget,
            require_k_drive=False,
        )
    except (ConversationHeldOutV4ExternalCapsuleError, KRunBoundaryError,
            TypeError, ValueError) as exc:
        raise ConversationHeldOutV4BoundRuntimeArtifactWriterError(
            "P3-C3 budgeted C1c capsule read failed") from exc
    expected_identity = _digest(
        _C1C_RUNTIME_CAPSULE_DOMAIN + encode_integer_tuple(capsule.stable_key()))
    if (capsule.origin != V4_RUNTIME_SOURCE_ORIGIN_EXTERNAL
            or capsule.manifest_sha256 != descriptor.manifest_sha256
            or expected_identity != descriptor.runtime_capsule_identity_sha256
            or len(capsule.inputs) != descriptor.turn_count):
        _fail("P3-C3 budgeted capsule does not match C1c descriptor identity")
    return capsule


def _freeze_document(result: ConversationHeldOutV4CandidateRuntimeResult) -> dict[str, object]:
    return {
        "artifact_kind": "DLG05_V4_P3C3_BOUND_RUNTIME_ARTIFACT_V1",
        "bundle_payload_sha256": bytes(result.bundle.payload_sha256).hex(),
        "bundle_payload_size": result.bundle.payload_size,
        "freeze_stable_key_sha256": _identity_document(result.freeze.stable_key()),
        "source_count": len(result.bundle.sources),
        "turn_count": len(result.bundle.turns),
    }


def _payloads(result: ConversationHeldOutV4CandidateRuntimeResult) -> dict[Path, bytes]:
    verify_v4_freeze(result.bundle, result.freeze)
    payloads = {
        _BUNDLE_FILE: encode_integer_tuple(result.bundle.canonical_payload),
        _RECEIPT_FILE: encode_integer_tuple(result.receipt.canonical_payload),
        _FREEZE_FILE: canonical_json_bytes(_freeze_document(result)),
        _MARKDOWN_FILE: render_v4_markdown(result.bundle).encode("utf-8"),
        _HTML_FILE: render_v4_html(result.bundle).encode("utf-8"),
    }
    if set(payloads) != _PRE_MANIFEST_FILES:
        _fail("P3-C3 fixed artifact payload closure drifted")
    return payloads


def _check_payload_budget(
        payloads: dict[Path, bytes], budget: ConversationHeldOutV4BoundRuntimeArtifactBudget,
        ) -> int:
    limits = {
        _BUNDLE_FILE: budget.max_bundle_bytes,
        _RECEIPT_FILE: budget.max_runtime_receipt_bytes,
        _FREEZE_FILE: budget.max_freeze_bytes,
        _MARKDOWN_FILE: budget.max_projection_bytes,
        _HTML_FILE: budget.max_projection_bytes,
    }
    total = 0
    for relative, payload in payloads.items():
        if not isinstance(payload, bytes) or not payload or len(payload) > limits[relative]:
            _fail("P3-C3 artifact payload exceeds explicit output budget")
        total += len(payload)
    if total > budget.max_total_output_bytes:
        _fail("P3-C3 aggregate artifact output exceeds budget")
    return total


def _preflight_artifact_closure_budget(
        *, payloads: dict[Path, bytes], manifest: bytes,
        budget: ConversationHeldOutV4BoundRuntimeArtifactBudget,
        ) -> int:
    """在 create-new 前核验包含 manifest 的全部 write/read 上限。"""
    payload_total = _check_payload_budget(payloads, budget)
    if not isinstance(manifest, bytes) or not manifest:
        _fail("P3-C3 manifest payload is invalid")
    if len(manifest) > budget.max_manifest_bytes:
        _fail("P3-C3 manifest exceeds output budget")
    closure_total = payload_total + len(manifest)
    if closure_total > budget.max_total_output_bytes:
        _fail("P3-C3 artifact closure exceeds output budget")
    if closure_total > budget.max_total_read_bytes:
        _fail("P3-C3 artifact closure exceeds read budget")
    return closure_total


def _manifest_payload(
        *, gate: ConversationHeldOutV4BoundCapsuleConsumerGateResult,
        value: ConversationHeldOutV4BoundRuntimeArtifactWriterInput,
        static_assets: ConversationHeldOutV4RuntimeStaticAssets,
        payloads: dict[Path, bytes],
        ) -> bytes:
    adapter = value.bound_capsule_consumer_gate_input.adapter_result
    caller = gate.active_caller_gate_result
    document = {
        "active_provenance_coverage": V4_BOUND_RUNTIME_ARTIFACT_ACTIVE_PROVENANCE_COVERAGE,
        "artifact_budget_identity_sha256": _identity_document(value.artifact_budget.integer_stream()),
        "bridge_code_identity": list(value.bridge_code_identity.components),
        "c1c_capsule_budget_identity_sha256": _identity_document(
            adapter.receipt.budget.capsule_budget.integer_tuple()),
        "c2_stable_key": list(gate.stable_key()),
        "caller_code_identity": list(caller.caller_code_identity.components),
        "caller_logical_stage_name": caller.logical_stage_name,
        "files": {
            relative.as_posix(): {"sha256": _digest_hex(payload), "size": len(payload)}
            for relative, payload in sorted(payloads.items(), key=lambda item: item[0].as_posix())
        },
        "files_scope": "PRE_MANIFEST_PAYLOAD_FILES_ONLY",
        "logical_stage_name": value.logical_stage_name,
        "runtime_asset_budget_identity_sha256": _identity_document(
            value.runtime_static_asset_budget.integer_stream()),
        "runtime_inventory_identity_sha256": _identity_document(
            static_assets.inventory.stable_key()),
        "runtime_static_assets": {
            "code_file_count": len(static_assets.inventory.execution_code),
            "code_total_size": static_assets.inventory.execution_code_total_size,
            "surface_sample_sha256": bytes(static_assets.inventory.surface_sample_sha256).hex(),
            "surface_sample_size": static_assets.inventory.surface_sample_size,
        },
        "schema": V4_BOUND_RUNTIME_ARTIFACT_MANIFEST_SCHEMA,
        "status": V4_BOUND_RUNTIME_ARTIFACT_STATUS_TEST_TRANSPORT_UNQUALIFIED,
        "test_transport": True,
    }
    return canonical_json_bytes(document)


def _write_artifact(
        root: KRunRoot, *, payloads: dict[Path, bytes], manifest: bytes,
        budget: ConversationHeldOutV4BoundRuntimeArtifactBudget,
        verify_source_before_manifest: Callable[[], None],
        ) -> int:
    if len(manifest) > budget.max_manifest_bytes:
        _fail("P3-C3 manifest exceeds output budget")
    if sum(len(value) for value in payloads.values()) + len(manifest) > budget.max_total_output_bytes:
        _fail("P3-C3 artifact closure exceeds output budget")
    try:
        require_fresh_empty_run_root(root, label="P3-C3 artifact root")
        # Only this fixed directory is needed; all files use exclusive writes.
        ensure_normal_relative_directory(root, _PROJECTION_DIRECTORY,
                                         label="P3-C3 projection directory")
        for relative in (_BUNDLE_FILE, _RECEIPT_FILE, _FREEZE_FILE,
                         _MARKDOWN_FILE, _HTML_FILE):
            write_exclusive_bytes(root, relative, payloads[relative],
                                  label="P3-C3 artifact payload")
        require_exact_file_closure(root, _PRE_MANIFEST_FILES,
                                   label="P3-C3 pre-manifest artifact")
        # The second source read is the source-side closure immediately before
        # the single publication marker.  A failure leaves this partial root.
        verify_source_before_manifest()
        publish_manifest_last(root, _MANIFEST_FILE, manifest, _PRE_MANIFEST_FILES,
                              label="P3-C3 artifact")
        require_exact_file_closure(root, _EXPECTED_FILES, label="P3-C3 artifact")
        read_total = 0
        limits = {
            _BUNDLE_FILE: budget.max_bundle_bytes,
            _RECEIPT_FILE: budget.max_runtime_receipt_bytes,
            _FREEZE_FILE: budget.max_freeze_bytes,
            _MARKDOWN_FILE: budget.max_projection_bytes,
            _HTML_FILE: budget.max_projection_bytes,
            _MANIFEST_FILE: budget.max_manifest_bytes,
        }
        expected = dict(payloads)
        expected[_MANIFEST_FILE] = manifest
        for relative in sorted(_EXPECTED_FILES, key=Path.as_posix):
            physical = sha256_plain_file(root, relative, max_bytes=limits[relative],
                                         label="P3-C3 artifact readback")
            if physical.byte_count != len(expected[relative]) or physical.sha256 != _digest(expected[relative]):
                _fail("P3-C3 artifact bounded readback drifted")
            read_total += physical.byte_count
            if read_total > budget.max_total_read_bytes:
                _fail("P3-C3 artifact readback exceeds budget")
        return read_total
    except KRunBoundaryError as exc:
        raise ConversationHeldOutV4BoundRuntimeArtifactWriterError(
            "P3-C3 artifact boundary failed") from exc


def write_v4_bound_runtime_artifact(
        value: ConversationHeldOutV4BoundRuntimeArtifactWriterInput,
        ) -> ConversationHeldOutV4BoundRuntimeArtifactWriterResult:
    """重跑 C2 后，以两次有界 capsule read 执行一次现役 runtime 并发布 C3 闭包。"""
    if not isinstance(value, ConversationHeldOutV4BoundRuntimeArtifactWriterInput):
        raise TypeError("P3-C3 writer requires BoundRuntimeArtifactWriterInput")
    try:
        gate = run_v4_bound_capsule_consumer_zero_write_gate(
            value.bound_capsule_consumer_gate_input)
    except (ConversationHeldOutV4BoundCapsuleConsumerGateError, TypeError, ValueError) as exc:
        raise ConversationHeldOutV4BoundRuntimeArtifactWriterError(
            "P3-C3 internal C2 revalidation failed") from exc
    if (gate.status != V4_BOUND_CAPSULE_CONSUMER_GATE_STATUS_TEST_ONLY
            or not gate.active_caller_gate_result.test_transport):
        _fail("P3-C3 only permits explicit test transport")
    source_root = _source_root_for(value.bound_capsule_consumer_gate_input)
    source_before = _read_bound_capsule(source_root, value.bound_capsule_consumer_gate_input)
    try:
        static_assets_before = read_v4_runtime_static_assets(
            value.runtime_static_asset_budget, test_transport=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ConversationHeldOutV4BoundRuntimeArtifactWriterError(
            "P3-C3 bounded runtime static asset read failed") from exc
    try:
        runtime_result = run_v4_candidate_runtime(
            source_before, static_assets=static_assets_before)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ConversationHeldOutV4BoundRuntimeArtifactWriterError(
            "P3-C3 current runtime execution failed") from exc
    if (not isinstance(runtime_result, ConversationHeldOutV4CandidateRuntimeResult)
            or runtime_result.capsule != source_before
            or runtime_result.identity.runtime_inventory
            != static_assets_before.inventory):
        _fail("P3-C3 runtime result did not bind verified source/static assets")
    payloads = _payloads(runtime_result)
    _check_payload_budget(payloads, value.artifact_budget)
    try:
        static_assets_after = read_v4_runtime_static_assets(
            value.runtime_static_asset_budget, test_transport=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ConversationHeldOutV4BoundRuntimeArtifactWriterError(
            "P3-C3 post-runtime static asset read failed") from exc
    if static_assets_after.stable_key() != static_assets_before.stable_key():
        _fail("P3-C3 runtime static assets drifted during execution")
    manifest = _manifest_payload(gate=gate, value=value, static_assets=static_assets_after,
                                 payloads=payloads)
    _preflight_artifact_closure_budget(
        payloads=payloads, manifest=manifest, budget=value.artifact_budget)
    caller = value.bound_capsule_consumer_gate_input.active_caller_gate_input
    try:
        artifact_root = create_new_run_root(
            caller.future_artifact_root, require_k_drive=False,
            label="P3-C3 future artifact root")
        require_disjoint_run_roots(source_root, artifact_root,
                                   label="P3-C3 source/artifact roots")
    except KRunBoundaryError as exc:
        raise ConversationHeldOutV4BoundRuntimeArtifactWriterError(
            "P3-C3 future artifact root boundary failed") from exc
    # A failure below intentionally leaves this fresh root without a manifest.
    def verify_source_before_manifest() -> None:
        source_after = _read_bound_capsule(
            source_root, value.bound_capsule_consumer_gate_input)
        if source_after != source_before:
            _fail("P3-C3 source capsule drifted during artifact publication")

    read_total = _write_artifact(
        artifact_root, payloads=payloads, manifest=manifest,
        budget=value.artifact_budget,
        verify_source_before_manifest=verify_source_before_manifest)
    counts = ConversationHeldOutV4BoundRuntimeArtifactCounts(
        1, 2, 2, 1, 6,
        sum(len(item) for item in payloads.values()) + len(manifest), read_total)
    adapter = value.bound_capsule_consumer_gate_input.adapter_result
    return ConversationHeldOutV4BoundRuntimeArtifactWriterResult(
        _key_digest(gate.stable_key()),
        _key_digest(gate.active_caller_gate_result.caller_code_identity.components),
        _key_digest(adapter.receipt.budget.capsule_budget.integer_tuple()),
        _key_digest(value.runtime_static_asset_budget.integer_stream()),
        _key_digest(value.artifact_budget.integer_stream()),
        _digest(manifest), counts,
        V4_BOUND_RUNTIME_ARTIFACT_STATUS_TEST_TRANSPORT_UNQUALIFIED,
        V4_BOUND_RUNTIME_ARTIFACT_ACTIVE_PROVENANCE_COVERAGE,
    )


__all__ = [
    "ConversationHeldOutV4BoundRuntimeArtifactBudget",
    "ConversationHeldOutV4BoundRuntimeArtifactCounts",
    "ConversationHeldOutV4BoundRuntimeArtifactWriterError",
    "ConversationHeldOutV4BoundRuntimeArtifactWriterInput",
    "ConversationHeldOutV4BoundRuntimeArtifactWriterResult",
    "ConversationHeldOutV4RuntimeStaticAssetReadBudget",
    "V4_BOUND_RUNTIME_ARTIFACT_ACTIVE_PROVENANCE_COVERAGE",
    "V4_BOUND_RUNTIME_ARTIFACT_MANIFEST_SCHEMA",
    "V4_BOUND_RUNTIME_ARTIFACT_STATUS_TEST_TRANSPORT_UNQUALIFIED",
    "write_v4_bound_runtime_artifact",
]
