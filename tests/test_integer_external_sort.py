"""P2 最小整数外排的 bounded、封存、确定性与领域边界专项。"""
from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest

import pure_integer_ai.storage.integer_external_sort as sort_module
from pure_integer_ai.storage.integer_codec import (
    IntegerFramedStreamBudgetExceeded,
    IntegerFramedStreamError,
    IntegerFramedStreamReader,
    IntegerFramedStreamWriter,
)
from pure_integer_ai.storage.integer_external_sort import (
    INTEGER_EXTERNAL_SORT_BUDGET_SCHEMA,
    INTEGER_EXTERNAL_SORT_IDENTITY_SCHEMA,
    INTEGER_EXTERNAL_SORT_INPUT_IDENTITY_SCHEMA,
    INTEGER_EXTERNAL_SORT_RESULT_SCHEMA,
    IntegerExternalSortBudget,
    IntegerExternalSortBudgetExceeded,
    IntegerExternalSortError,
    external_sort_sealed_integer_records,
)
from pure_integer_ai.storage.k_run_boundary import (
    KRunBoundaryError,
    ensure_normal_relative_directory,
    open_exclusive_binary,
    open_existing_run_root,
    open_plain_binary,
)


def _root(tmp_path, name: str):
    """创建显式 test transport root，生产 K 盘策略不在该专项中被绕过。"""
    path = tmp_path / name
    path.mkdir()
    return open_existing_run_root(path, require_k_drive=False, label=name)


def _budget(**changes) -> IntegerExternalSortBudget:
    """提供足够小以强制多 run、又能覆盖 P0 footer 的固定测试预算。"""
    values = {
        "max_input_file_count": 8,
        "max_input_physical_bytes": 100_000,
        "max_input_record_count": 128,
        "max_input_payload_bytes": 10_000,
        "max_record_payload_bytes": 1_000,
        "max_batch_record_count": 2,
        "max_batch_payload_bytes": 1_000,
        "max_batch_sort_key_bytes": 1_000,
        "max_temporary_run_count": 64,
        "max_temporary_record_count": 1_000,
        "max_temporary_payload_bytes": 100_000,
        "max_temporary_physical_bytes": 100_000,
        "max_output_physical_bytes": 100_000,
        "merge_fan_in": 2,
        "max_open_files": 3,
        "max_merge_pass_count": 8,
    }
    values.update(changes)
    return IntegerExternalSortBudget(**values)


def _write_stream(root, relative: Path, records: tuple[tuple[int, ...], ...], *, seal: bool = True) -> None:
    """仅经 K capability 和 P0 from_open_binary 建立测试输入或既有输出。"""
    if relative.parent.parts:
        ensure_normal_relative_directory(root, relative.parent, label="test stream parent")
    stream = open_exclusive_binary(root, relative, label="test stream")
    writer = IntegerFramedStreamWriter.from_open_binary(stream, path=relative)
    for record in records:
        writer.append(record)
    if seal:
        writer.seal()
    else:
        writer.close()


def _read_records(root, relative: Path, budget: IntegerExternalSortBudget) -> tuple[tuple[int, ...], ...]:
    """以 P0 reader 读回输出，不将 raw bytes 当作领域格式解释。"""
    stream = open_plain_binary(root, relative, label="test read")
    reader = IntegerFramedStreamReader.from_open_binary(
        stream,
        path=relative,
        max_frame_bytes=budget.max_record_payload_bytes,
        max_record_count=budget.max_input_record_count,
        max_total_payload_bytes=budget.max_input_payload_bytes,
    )
    try:
        return tuple(reader)
    finally:
        reader.close()


def _raw_stream_bytes(root, relative: Path) -> bytes:
    """仅用于逐字节确定性断言，经 capability 打开已封存测试 artifact。"""
    with open_plain_binary(root, relative, label="test raw stream") as stream:
        return stream.read()


def _sort(root, work_root, budget: IntegerExternalSortBudget):
    """执行固定两输入排序；输入顺序也定义重复 sort key 的稳定次序。"""
    return external_sort_sealed_integer_records(
        root,
        (Path("first.pifrs"), Path("nested") / "second.pifrs"),
        work_root,
        output_relative_path=Path("published") / "sorted.pifrs",
        logical_stage_name="records-by-leading-integer.v1",
        sort_key=lambda record: (record[0],),
        budget=budget,
    )


def test_multiple_runs_two_merge_passes_preserve_records_stable_ties_and_bytes(tmp_path):
    """小 batch 加 fan-in=2 必须多轮归并，重复 key 仍按输入流与原序稳定。"""
    input_root = _root(tmp_path, "input")
    work_one = _root(tmp_path, "work-one")
    work_two = _root(tmp_path, "work-two")
    budget = _budget()
    _write_stream(input_root, Path("first.pifrs"), (
        (2, 100), (1, 101), (2, 102), (5, 103), (0, 104),
    ))
    _write_stream(input_root, Path("nested") / "second.pifrs", (
        (1, 200), (2, 201), (0, 202), (-1, 203), (2, 204),
    ))

    first = _sort(input_root, work_one, budget)
    second = _sort(input_root, work_two, budget)
    output_relative = Path("published") / "sorted.pifrs"
    expected = (
        (-1, 203), (0, 104), (0, 202), (1, 101), (1, 200),
        (2, 100), (2, 102), (2, 201), (2, 204), (5, 103),
    )

    assert _read_records(work_one, output_relative, budget) == expected
    assert _read_records(work_two, output_relative, budget) == expected
    assert _raw_stream_bytes(work_one, output_relative) == _raw_stream_bytes(
        work_two,
        output_relative,
    )
    assert first.identity.output_physical == second.identity.output_physical
    assert first.identity.input_identities == second.identity.input_identities
    assert first.initial_run_count == 5
    assert first.merge_pass_count == 2
    assert first.temporary_run_count == 8
    assert first.temporary_record_count == 26
    assert first.merge_fan_in == 2
    assert first.max_open_files == 3
    assert first.budget == budget


def test_empty_sealed_input_writes_a_sealed_empty_output_without_temporary_run(tmp_path):
    """非空输入路径 tuple 可以承载零 record P0 stream，输出仍须是封存 artifact。"""
    input_root = _root(tmp_path, "input")
    work_root = _root(tmp_path, "work")
    budget = _budget()
    _write_stream(input_root, Path("first.pifrs"), ())
    _write_stream(input_root, Path("nested") / "second.pifrs", ())

    result = _sort(input_root, work_root, budget)

    assert _read_records(work_root, Path("published") / "sorted.pifrs", budget) == ()
    assert result.initial_run_count == 0
    assert result.temporary_run_count == 0
    assert result.identity.output_footer.record_count == 0
    assert result.identity.output_footer.total_payload_bytes == 0


def test_canonical_integer_keys_bind_budget_paths_physical_and_output_without_roots(
        tmp_path, monkeypatch):
    """公开 value struct 的稳定键必须完整、纯整数且不泄露绝对 root。"""
    input_one = _root(tmp_path, "input-one")
    input_two = _root(tmp_path, "input-two")
    work_one = _root(tmp_path, "work-one")
    work_two = _root(tmp_path, "work-two")
    budget = _budget()
    records_one = ((2, 10), (1, 11))
    records_two = ((3, 20),)
    for root in (input_one, input_two):
        _write_stream(root, Path("first.pifrs"), records_one)
        _write_stream(root, Path("nested") / "second.pifrs", records_two)

    first = _sort(input_one, work_one, budget)
    second = _sort(input_two, work_two, budget)
    input_identity = first.identity.input_identities[0]
    changed_budget = replace(
        budget,
        max_input_physical_bytes=budget.max_input_physical_bytes + 1,
    )
    changed_input_path = replace(
        input_identity,
        relative_path=Path("renamed.pifrs"),
    )
    changed_input_physical = replace(
        input_identity,
        physical=replace(
            input_identity.physical,
            byte_count=input_identity.physical.byte_count + 1,
        ),
    )
    changed_output = replace(
        first.identity,
        output_relative_path=Path("published") / "other.pifrs",
    )
    changed_output_physical = replace(
        first.identity,
        output_physical=replace(
            first.identity.output_physical,
            byte_count=first.identity.output_physical.byte_count + 1,
        ),
    )
    changed_stage = replace(
        first.identity,
        logical_stage_name="records-by-leading-integer.v2",
    )

    assert budget.integer_stream()[0] == INTEGER_EXTERNAL_SORT_BUDGET_SCHEMA
    assert input_identity.integer_stream()[0] == (
        INTEGER_EXTERNAL_SORT_INPUT_IDENTITY_SCHEMA)
    assert first.identity.integer_stream()[0] == INTEGER_EXTERNAL_SORT_IDENTITY_SCHEMA
    assert first.integer_stream()[0] == INTEGER_EXTERNAL_SORT_RESULT_SCHEMA
    assert first.stable_key() == first.integer_stream()
    assert first.identity.stable_key() == first.identity.integer_stream()
    assert input_identity.stable_key() == input_identity.integer_stream()
    assert budget.stable_key() == budget.integer_stream()
    assert all(type(item) is int for item in first.stable_key())

    # 相同物理内容、相对路径和配置在不同输入/输出 root 下必须同一身份。
    assert first.identity.stable_key() == second.identity.stable_key()
    assert first.stable_key() == second.stable_key()
    assert budget.stable_key() != changed_budget.stable_key()
    assert input_identity.stable_key() != changed_input_path.stable_key()
    assert input_identity.stable_key() != changed_input_physical.stable_key()
    assert first.identity.stable_key() != changed_output.stable_key()
    assert first.identity.stable_key() != changed_output_physical.stable_key()
    assert first.identity.stable_key() != changed_stage.stable_key()
    assert first.stable_key() != replace(first, budget=changed_budget).stable_key()

    # 以 Windows normcase 模拟大小写折叠，同时仍固定输出 POSIX 分隔符。
    monkeypatch.setattr(
        sort_module.os.path,
        "normcase",
        lambda value: value.lower().replace("/", chr(92)),
    )
    case_variant = replace(
        first.identity,
        output_relative_path=Path("PUBLISHED") / "SORTED.PIFRS",
    )
    assert first.identity.stable_key() == case_variant.stable_key()


def test_unsealed_input_non_strict_key_and_budget_exhaustion_fail_without_output(tmp_path):
    """未封存 P0、bool key 与输入 record 超限均不能形成固定 output。"""
    input_root = _root(tmp_path, "input")
    unsealed_work = _root(tmp_path, "unsealed-work")
    key_work = _root(tmp_path, "key-work")
    budget_work = _root(tmp_path, "budget-work")
    _write_stream(input_root, Path("first.pifrs"), ((1,), (2,)), seal=False)
    _write_stream(input_root, Path("nested") / "second.pifrs", ())
    with pytest.raises(IntegerFramedStreamError):
        _sort(input_root, unsealed_work, _budget())
    assert not (unsealed_work.path / "published" / "sorted.pifrs").exists()

    valid_input = _root(tmp_path, "valid-input")
    _write_stream(valid_input, Path("first.pifrs"), ((1,), (2,)))
    _write_stream(valid_input, Path("nested") / "second.pifrs", ((3,),))
    with pytest.raises(IntegerExternalSortError, match="严格整数"):
        external_sort_sealed_integer_records(
            valid_input,
            (Path("first.pifrs"), Path("nested") / "second.pifrs"),
            key_work,
            output_relative_path=Path("published") / "sorted.pifrs",
            logical_stage_name="non-strict-key.v1",
            sort_key=lambda _record: (True,),
            budget=_budget(),
        )
    assert not (key_work.path / "published" / "sorted.pifrs").exists()

    with pytest.raises((
            IntegerExternalSortBudgetExceeded,
            IntegerFramedStreamBudgetExceeded,
    )):
        _sort(
            valid_input,
            budget_work,
            _budget(max_input_record_count=1, max_batch_record_count=1),
        )
    assert not (budget_work.path / "published" / "sorted.pifrs").exists()


def test_preexisting_output_nested_roots_and_post_read_identity_drift_fail_closed(tmp_path, monkeypatch):
    """既有 output、互相嵌套 root 和 capture 后身份漂移都必须拒绝，不覆盖残片。"""
    input_root = _root(tmp_path, "input")
    output_work = _root(tmp_path, "output-work")
    drift_work = _root(tmp_path, "drift-work")
    nested_work = ensure_normal_relative_directory(input_root, "nested-work", label="nested work")
    budget = _budget()
    _write_stream(input_root, Path("first.pifrs"), ((2,), (1,)))
    _write_stream(input_root, Path("nested") / "second.pifrs", ((0,),))
    _write_stream(output_work, Path("published") / "sorted.pifrs", ((99,),))

    with pytest.raises(IntegerExternalSortError, match="已存在"):
        _sort(input_root, output_work, budget)
    assert not (output_work.path / "integer-external-sort").exists()

    with pytest.raises(KRunBoundaryError, match="嵌套"):
        _sort(input_root, nested_work, budget)

    original_recheck = sort_module.require_plain_file_identity

    def injected_drift(root, relative, identity, *, label):
        """仅在 source 被完整消费后模拟边界检测到的路径身份漂移。"""
        if label == "external sort input[0] post-read":
            raise KRunBoundaryError("injected input identity drift")
        return original_recheck(root, relative, identity, label=label)

    monkeypatch.setattr(sort_module, "require_plain_file_identity", injected_drift)
    with pytest.raises(KRunBoundaryError, match="injected input identity drift"):
        _sort(input_root, drift_work, budget)
    assert not (drift_work.path / "published" / "sorted.pifrs").exists()


def test_merge_pass_budget_and_path_escape_fail_closed(tmp_path):
    """merge、词法越界及 Windows 大小写别名均在固定 output 之前失败。"""
    input_root = _root(tmp_path, "input")
    pass_work = _root(tmp_path, "pass-work")
    path_work = _root(tmp_path, "path-work")
    case_work = _root(tmp_path, "case-work")
    reserved_work = _root(tmp_path, "reserved-work")
    _write_stream(input_root, Path("first.pifrs"), tuple((value,) for value in range(5)))
    _write_stream(input_root, Path("nested") / "second.pifrs", tuple((value,) for value in range(5, 10)))
    with pytest.raises(IntegerExternalSortBudgetExceeded, match="merge pass"):
        _sort(input_root, pass_work, _budget(max_merge_pass_count=1))
    assert not (pass_work.path / "published" / "sorted.pifrs").exists()

    with pytest.raises(ValueError, match="相对 Path"):
        external_sort_sealed_integer_records(
            input_root,
            (Path("..") / "outside.pifrs",),
            path_work,
            output_relative_path=Path("published") / "sorted.pifrs",
            logical_stage_name="path-escape.v1",
            sort_key=lambda record: record,
            budget=_budget(),
        )

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(sort_module.os.path, "normcase", lambda value: value.lower())
    try:
        with pytest.raises(IntegerExternalSortError, match="路径不得重复"):
            external_sort_sealed_integer_records(
                input_root,
                (Path("first.pifrs"), Path("FIRST.PIFRS")),
                case_work,
                output_relative_path=Path("published") / "sorted.pifrs",
                logical_stage_name="case-alias.v1",
                sort_key=lambda record: record,
                budget=_budget(),
            )
        with pytest.raises(IntegerExternalSortError, match="保留临时"):
            external_sort_sealed_integer_records(
                input_root,
                (Path("first.pifrs"),),
                reserved_work,
                output_relative_path=(
                    Path("INTEGER-EXTERNAL-SORT")
                    / "reserved-stage.v1"
                    / "output.pifrs"
                ),
                logical_stage_name="reserved-stage.v1",
                sort_key=lambda record: record,
                budget=_budget(),
            )
    finally:
        monkeypatch.undo()


def test_module_is_p0_and_boundary_only_without_unsafe_file_or_domain_fallbacks():
    """外排实现不得吸收 P2 领域、临时目录、覆盖删除或裸 Path.open 回退。"""
    source_path = Path(sort_module.__file__)
    source_text = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source_text, filename=str(source_path))
    dynamic_imports: list[int] = []
    destructive_calls: list[int] = []
    path_open_calls: list[int] = []
    names: set[str] = set()
    allowed_modules = {
        "pure_integer_ai.storage.integer_codec",
        "pure_integer_ai.storage.k_run_boundary",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("pure_integer_ai"):
                assert node.module in allowed_modules
        elif isinstance(node, ast.Call):
            if ((isinstance(node.func, ast.Name) and node.func.id == "__import__")
                    or (isinstance(node.func, ast.Attribute)
                        and node.func.attr == "import_module")):
                dynamic_imports.append(node.lineno)
            if (isinstance(node.func, ast.Attribute)
                    and node.func.attr in {
                        "remove", "unlink", "rmdir", "rename", "replace",
                    }):
                destructive_calls.append(node.lineno)
            if (isinstance(node.func, ast.Attribute)
                    and node.func.attr == "open"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "Path"):
                path_open_calls.append(node.lineno)

    assert dynamic_imports == []
    assert destructive_calls == []
    assert path_open_calls == []
    assert names.isdisjoint({
        "tempfile", "json", "sqlite3", "shutil", "glob", "iterdir",
        "read_bytes", "write_bytes", "unlink", "rmtree", "environ", "getenv",
    })
    assert "repr(" not in source_text
    assert "json" not in source_text.lower()
    assert all(term not in source_text for term in (
        "SourceRef", "v4", "private", "formal", "sqlite", "json",
    ))
