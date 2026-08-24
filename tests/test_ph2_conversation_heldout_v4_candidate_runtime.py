"""DLG-05 v4 真实 candidate runtime 的有界回归。"""
from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

import pure_integer_ai.experiments.conversation_heldout_v4_candidate_runtime as runtime_module
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
)
from pure_integer_ai.experiments.conversation_heldout_v4_bundle import (
    unicode_scalars,
)
from pure_integer_ai.experiments.conversation_heldout_v4_external_input_capsule import (
    read_budgeted_v4_external_input_capsule,
)
from pure_integer_ai.experiments.ph2_grounded_answer_connector import (
    GroundedAnswerConnectorError,
    GroundedAnswerConnectorTarget,
)
from pure_integer_ai.experiments.ph2_grounded_answer_response_act_compile import (
    GroundedResponseActCompileError,
    GroundedResponseActCompileTarget,
)
from pure_integer_ai.cognition.shared.identity import language_branch_identity
from pure_integer_ai.experiments.conversation_heldout_v4_candidate_runtime import (
    ConversationHeldOutV4RuntimeError,
    ConversationHeldOutV4RuntimeReceipt,
    V4_RUNTIME_CODE_RELATIVE_PATH,
    V4_RUNTIME_DEFAULT_STATIC_ASSET_READ_BUDGET,
    V4_RUNTIME_EXECUTION_CODE_RELATIVE_PATHS,
    V4_RUNTIME_SURFACE_SAMPLE_RELATIVE_PATH,
    build_v4_synthetic_runtime_fixture,
    read_v4_runtime_inventory,
    read_v4_runtime_static_assets,
    run_v4_candidate_runtime,
    run_v4_synthetic_candidate_runtime,
)

from test_ph2_conversation_heldout_v4_bound_capsule_consumer_gate import (  # noqa: E402
    _bound_input,
)


@pytest.fixture(scope="module")
def runtime_result():
    """只运行一次默认公开 runtime slice，供同次性和漂移专项复用。"""
    return run_v4_synthetic_candidate_runtime()


def test_v4_candidate_runtime_invokes_real_executor_once_for_each_turn(monkeypatch):
    """每个导出 turn 都必须触发一次实际 FactQuestionExecutor.execute。"""
    calls = []
    original = runtime_module._CountingFactQuestionExecutor.execute

    def counted_execute(self, query):
        calls.append(query)
        return original(self, query)

    monkeypatch.setattr(
        runtime_module._CountingFactQuestionExecutor,
        "execute",
        counted_execute,
    )
    result = run_v4_synthetic_candidate_runtime()

    assert len(calls) == len(result.frames) == 12
    assert tuple(call.request for call in calls) == tuple(
        frame.input.request for frame in result.frames)
    for frame in result.frames:
        turn = result.bundle.turn_for(
            frame.input.case_key, frame.input.turn_key)
        assert frame.executor_calls == 1
        assert tuple(item.candidate for item in turn.candidates) == (
            frame.execution.candidates)


def test_v4_candidate_runtime_binds_actual_g01_to_generation_and_renderer(
        runtime_result):
    """G-01、G-00 和 G-03 必须消费同一次 execution，而非外部重选或查表。"""
    result = runtime_result

    assert len(result.frames) == 12
    assert len(result.bundle.turns) == 12
    assert sum(len(turn.candidates) for turn in result.bundle.turns) == 12
    for frame in result.frames:
        planning = frame.execution.planning_request()
        assert frame.selection.request == planning
        assert frame.generation.plan.request == planning
        assert frame.generation.rendered == frame.rendered
        stance, content = frame.generation.plan.layers[:2]
        assert stance.payload == content.payload == frame.selection.stable_key()
        execution_keys = {
            candidate.stable_key() for candidate in frame.execution.candidates
        }
        assert set(frame.selection.selected_candidate_keys) <= execution_keys
        assert {item.candidate.stable_key()
                for item in frame.candidate_realizations} == execution_keys
        for realization in frame.candidate_realizations:
            assert realization.planning.candidates == (realization.candidate,)
            assert realization.selection.request == realization.planning
            assert realization.generation.plan.request == realization.planning
            assert realization.generation.rendered == realization.rendered
        if frame.execution.candidates:
            assert tuple(item.scalars for item in frame.surface_representations)
        else:
            assert frame.surface_representations == ()
        assert frame.rendered.renderer == (
            frame.generation.rendered.renderer)


def test_v4_candidate_runtime_rejects_static_candidates_and_prewritten_surfaces(
        runtime_result):
    """变化的 ledger/source 输入必须改变实际 candidate 或 G-03 surface。"""
    capsule = build_v4_synthetic_runtime_fixture()
    inputs = list(capsule.inputs)
    changed_text = "runtime 输入已变化，不能复用静态答案。"
    first = inputs[0]
    changed_record = replace(
        first.source_records[0],
        raw_text_scalars=unicode_scalars(changed_text),
        content_sha256=tuple(hashlib.sha256(
            changed_text.encode("utf-8")).digest()),
    )
    inputs[0] = replace(first, source_records=(changed_record,))
    inputs[1] = replace(
        inputs[1],
        source_records=(changed_record,),
        evidence_plans=(),
    )

    changed = run_v4_candidate_runtime(replace(capsule, inputs=tuple(inputs)))
    first_turn = changed.bundle.turn_for(
        changed.frames[0].input.case_key,
        changed.frames[0].input.turn_key,
    )
    second_turn = changed.bundle.turn_for(
        changed.frames[1].input.case_key,
        changed.frames[1].input.turn_key,
    )
    rendered_text = "".join(chr(value) for value in changed.frames[0].rendered.units)

    assert changed.bundle.payload_sha256 != runtime_result.bundle.payload_sha256
    assert changed_text in rendered_text
    assert len(first_turn.candidates) == 1
    assert second_turn.candidates == ()
    assert changed.frames[1].selection.selected_candidate_keys == ()


def test_v4_candidate_runtime_fails_closed_on_frame_or_runtime_identity_drift(
        runtime_result):
    """伪造 execute 次数或替换 runtime identity 都必须在 receipt 前被拒绝。"""
    with pytest.raises(ConversationHeldOutV4RuntimeError, match="恰好调用一次"):
        replace(runtime_result.frames[0], executor_calls=2)

    drifted_identity = replace(
        runtime_result.identity,
        renderer_sha256=(0,) * 32,
    )
    with pytest.raises(ConversationHeldOutV4RuntimeError, match="renderer identity"):
        ConversationHeldOutV4RuntimeReceipt(
            runtime_result.bundle.family_key,
            drifted_identity,
            runtime_result.frames,
            runtime_result.bundle,
        )


def test_v4_runtime_requires_explicit_capsule_and_marks_fixture_synthetic():
    """代码内 fixture 必须显式传入，不能伪装成生产来源默认值。"""
    capsule = build_v4_synthetic_runtime_fixture()
    assert not capsule.is_external
    with pytest.raises(TypeError, match="required positional argument"):
        run_v4_candidate_runtime()  # type: ignore[call-arg]


def test_v4_runtime_uses_a_selector_binding_for_each_external_scope(tmp_path):
    """真实 C1c 输入的不同版本 turn 必须各自消费同边界 G-01 selector。"""
    adapter, caller = _bound_input(tmp_path)
    capsule = read_budgeted_v4_external_input_capsule(
        caller.source_capsule_root,
        budget=adapter.receipt.budget.capsule_budget,
        require_k_drive=False,
    )
    result = run_v4_candidate_runtime(
        capsule,
        static_assets=read_v4_runtime_static_assets(
            V4_RUNTIME_DEFAULT_STATIC_ASSET_READ_BUDGET,
            test_transport=True,
        ),
    )

    assert capsule.is_external
    assert len(result.frames) == len(result.identity.selection_protocols) == 2
    assert tuple(
        binding.scope for binding in result.identity.selection_protocols
    ) == tuple(sorted(
        (frame.input.request.response_scope for frame in result.frames),
        key=lambda scope: scope.stable_key(),
    ))
    assert len({binding.scope.versions
                for binding in result.identity.selection_protocols}) == 2
    for frame in result.frames:
        binding = frame.selection_protocol
        assert binding.scope == frame.input.request.response_scope
        assert frame.selection.protocol == binding.content
        assert frame.selection.stance in binding.content.stances()

    global_branch = language_branch_identity((20260822, 405, 999))
    with pytest.raises(GroundedAnswerConnectorError, match="owner/version"):
        GroundedAnswerConnectorTarget(
            result.frames[0].input.request.target,
            global_branch,
            (20260822, 405, 998),
        )
    with pytest.raises(GroundedResponseActCompileError, match="owner/version"):
        GroundedResponseActCompileTarget(
            "UNKNOWN",
            result.frames[0].selection_protocol.content.unknown,
            global_branch,
            (20260822, 405, 997),
        )


def test_v4_runtime_inventory_binds_explicit_execution_code_closure():
    """运行 identity 必须逐项绑定固定的本地执行代码闭包，而非只 hash 入口模块。"""
    inventory = read_v4_runtime_inventory()

    assert inventory.schema_version == 2
    assert tuple(item.relative_path for item in inventory.execution_code) == (
        V4_RUNTIME_EXECUTION_CODE_RELATIVE_PATHS)
    assert V4_RUNTIME_CODE_RELATIVE_PATH in {
        item.relative_path for item in inventory.execution_code}
    assert len(inventory.execution_code) > 1
    assert inventory.execution_code_total_size == sum(
        item.size for item in inventory.execution_code)
    assert len(inventory.execution_code_closure_sha256) == 32


def test_v4_runtime_static_assets_are_budgeted_payloads_without_path_reread(
        monkeypatch, runtime_result):
    """runtime 只能消费 static loader 已核验 payload，不能向 course/compiler 交出路径。"""
    read_bytes_calls = []

    def reject_read_bytes(path):
        """任何恢复到 Path.read_bytes 的静态资产读取都会直接暴露。"""
        read_bytes_calls.append(path)
        raise AssertionError("runtime static assets 不得调用 Path.read_bytes")

    monkeypatch.setattr(Path, "read_bytes", reject_read_bytes)
    assets = read_v4_runtime_static_assets(
        V4_RUNTIME_DEFAULT_STATIC_ASSET_READ_BUDGET,
        test_transport=True,
    )
    result = run_v4_candidate_runtime(
        build_v4_synthetic_runtime_fixture(),
        static_assets=assets,
    )

    assert read_bytes_calls == []
    assert assets.inventory == runtime_result.identity.runtime_inventory
    assert result == runtime_result
    assert assets.surface_sample_payload.endswith(b"\n")
    assert assets.inventory.surface_sample_path == (
        V4_RUNTIME_SURFACE_SAMPLE_RELATIVE_PATH)


def test_v4_runtime_static_assets_preflight_all_budgets_before_payload_read(
        monkeypatch):
    """文件数或 sample 上限不足时，loader 不得先读取部分代码 payload。"""
    payload_reads = []
    original = runtime_module._read_bounded_runtime_static_asset_payload

    def counted_read(*args, **kwargs):
        """记录任何 payload 打开，供预算预检的零读取断言使用。"""
        payload_reads.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        runtime_module,
        "_read_bounded_runtime_static_asset_payload",
        counted_read,
    )
    too_few_files = replace(
        V4_RUNTIME_DEFAULT_STATIC_ASSET_READ_BUDGET,
        max_execution_code_file_count=(
            len(V4_RUNTIME_EXECUTION_CODE_RELATIVE_PATHS) - 1),
    )
    with pytest.raises(ConversationHeldOutV4RuntimeError, match="file count"):
        read_v4_runtime_static_assets(too_few_files, test_transport=True)
    assert payload_reads == []

    too_small_sample = replace(
        V4_RUNTIME_DEFAULT_STATIC_ASSET_READ_BUDGET,
        max_surface_sample_bytes=1,
    )
    with pytest.raises(ConversationHeldOutV4RuntimeError, match="预检超过读取预算"):
        read_v4_runtime_static_assets(too_small_sample, test_transport=True)
    assert payload_reads == []


def test_v4_runtime_static_assets_reject_post_read_identity_drift(monkeypatch):
    """每个静态文件在流式读取后都必须重新绑定原普通文件 identity。"""
    original = runtime_module.require_plain_file_identity

    def reject_sample_drift(root, relative, identity, *, label):
        """模拟 sample 在打开后被同路径替换，不能让 runtime 继续消费。"""
        if str(relative).replace("\\", "/") == (
                V4_RUNTIME_SURFACE_SAMPLE_RELATIVE_PATH):
            raise runtime_module.KRunBoundaryError("injected static asset drift")
        return original(root, relative, identity, label=label)

    monkeypatch.setattr(
        runtime_module,
        "require_plain_file_identity",
        reject_sample_drift,
    )
    with pytest.raises(ConversationHeldOutV4RuntimeError, match="文件身份漂移"):
        read_v4_runtime_static_assets(
            V4_RUNTIME_DEFAULT_STATIC_ASSET_READ_BUDGET,
            test_transport=True,
        )


def test_v4_execution_code_closure_matches_static_local_import_guard():
    """新增本地 import、star import 或动态 import 时必须人工审计并更新固定闭包。"""
    repository_root = Path(runtime_module.__file__).resolve().parents[3]
    source_root = repository_root / "src"
    package_root = "pure_integer_ai"
    entry_module = (
        "pure_integer_ai.experiments.conversation_heldout_v4_candidate_runtime")

    def source_for(module_name):
        """定位一个本地模块或包的公开 Python 源文件。"""
        if (module_name != package_root
                and not module_name.startswith(package_root + ".")):
            return None
        parts = module_name.split(".")
        if any(not part.isidentifier() for part in parts):
            return None
        base = source_root.joinpath(*parts)
        module = base.with_suffix(".py")
        package = base / "__init__.py"
        if module.is_file():
            return module, False
        if package.is_file():
            return package, True
        return None

    def add_with_parents(queue, module_name):
        """将模块及 Python import 期间会加载的全部父包加入待解析集合。"""
        parts = module_name.split(".")
        for index in range(1, len(parts) + 1):
            parent = ".".join(parts[:index])
            if source_for(parent) is not None:
                queue.add(parent)

    def relative_base(module_name, is_package, node):
        """按当前文件的模块/包位置解析一个 `from ... import` 基础模块。"""
        if node.level == 0:
            return node.module
        parts = (module_name.split(".") if is_package
                 else module_name.split(".")[:-1])
        if node.level > len(parts):
            raise AssertionError("v4 execution code 含越界 relative import")
        base = parts[:len(parts) - (node.level - 1)]
        if node.module:
            base.extend(node.module.split("."))
        return ".".join(base)

    pending = set()
    add_with_parents(pending, entry_module)
    visited = set()
    paths = {}
    dynamic_imports = []
    star_imports = []
    while pending:
        module_name = min(pending)
        pending.remove(module_name)
        if module_name in visited:
            continue
        source = source_for(module_name)
        assert source is not None
        path, is_package = source
        visited.add(module_name)
        paths[path.resolve()] = path
        tree = ast.parse(path.read_bytes(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if (isinstance(node.func, ast.Name)
                        and node.func.id == "__import__"):
                    dynamic_imports.append((module_name, node.lineno))
                if (isinstance(node.func, ast.Attribute)
                        and node.func.attr == "import_module"):
                    dynamic_imports.append((module_name, node.lineno))
            if isinstance(node, ast.Import):
                for alias in node.names:
                    add_with_parents(pending, alias.name)
            elif isinstance(node, ast.ImportFrom):
                base = relative_base(module_name, is_package, node)
                if base is None:
                    continue
                add_with_parents(pending, base)
                for alias in node.names:
                    if alias.name == "*":
                        star_imports.append((module_name, node.lineno, base))
                        continue
                    child = base + "." + alias.name
                    if source_for(child) is not None:
                        add_with_parents(pending, child)

    actual = tuple(sorted(
        path.relative_to(repository_root).as_posix()
        for path in paths.values()))
    assert dynamic_imports == []
    assert star_imports == []
    assert actual == V4_RUNTIME_EXECUTION_CODE_RELATIVE_PATHS


def test_v4_runtime_keeps_unselected_candidate_with_own_actual_surface():
    """G-01 仅选一个时，完整 candidate 集合和逐 candidate 表面都不得丢失或复用 response。"""
    capsule = build_v4_synthetic_runtime_fixture()
    inputs = list(capsule.inputs)
    multi = inputs[2]
    first, second = multi.evidence_plans
    inputs[2] = replace(
        multi,
        evidence_plans=(
            replace(first, stances=(EVIDENCE_SUPPORT,)),
            replace(
                second,
                competition_key=(*second.competition_key, 2),
                stances=(EVIDENCE_REFUTE,),
            ),
        ),
    )

    result = run_v4_candidate_runtime(replace(capsule, inputs=tuple(inputs)))
    frame = result.frames[2]
    turn = result.bundle.turn_for(frame.input.case_key, frame.input.turn_key)

    assert len(frame.execution.candidates) == 2
    assert frame.selection.stance == frame.selection_protocol.content.answer
    assert frame.selection_protocol in result.identity.selection_protocols
    assert len(frame.selection.selected_candidate_keys) == 1
    assert set(frame.selection.selected_candidate_keys) < {
        candidate.stable_key() for candidate in frame.execution.candidates
    }
    assert len(turn.candidates) == 2
    for exported in turn.candidates:
        rendered = frame.render_candidate(exported.candidate)
        assert exported.surface_scalars == rendered.units
        assert exported.surface_representations == rendered.representations
        assert exported.surface_representations != frame.rendered.representations
