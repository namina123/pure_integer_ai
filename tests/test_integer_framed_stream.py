"""R04b P0 规范整数分帧流的有界、封存与损坏拒绝专项。"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

import pure_integer_ai.storage.integer_codec as codec_module
from pure_integer_ai.storage.integer_codec import (
    INTEGER_FRAMED_STREAM_MAGIC,
    IntegerFramedStreamBudgetExceeded,
    IntegerFramedStreamError,
    IntegerFramedStreamReader,
    IntegerFramedStreamStateError,
    IntegerFramedStreamWriter,
    encode_integer_tuple,
)


def _write_sealed(
        path: Path, records: tuple[tuple[int, ...], ...],
        ):
    """通过唯一 writer 生成一个最小 sealed frame stream。"""
    with IntegerFramedStreamWriter(path) as writer:
        for record in records:
            writer.append(record)
        footer = writer.seal()
    assert writer.sealed
    assert writer.footer == footer
    return footer


def _read_all(
        path: Path, *, max_frame_bytes: int = 4096,
        max_record_count: int = 64, max_total_payload_bytes: int = 16384,
        ) -> tuple[tuple[tuple[int, ...], ...], object]:
    """逐条读完 stream，并返回 records 与 footer 以供完整性断言。"""
    with IntegerFramedStreamReader(
            path,
            max_frame_bytes=max_frame_bytes,
            max_record_count=max_record_count,
            max_total_payload_bytes=max_total_payload_bytes,
    ) as reader:
        records = tuple(reader)
        return records, reader.footer


def test_framed_stream_round_trip_is_incremental_and_footer_bound(tmp_path):
    """空 record、负值和大整数均可逐条恢复，footer 只在读完整流后可见。"""
    path = tmp_path / "records.ints"
    records = ((), (0, -1, 1, -64, 8192), (2 ** 255, -(2 ** 255)))
    expected_footer = _write_sealed(path, records)

    with IntegerFramedStreamReader(
            path,
            max_frame_bytes=4096,
            max_record_count=len(records),
            max_total_payload_bytes=8192,
    ) as reader:
        assert reader.read_record() == records[0]
        assert reader.footer is None
        assert reader.record_count == 1
        assert tuple(reader) == records[1:]
        assert reader.footer == expected_footer
        assert reader.total_payload_bytes == expected_footer.total_payload_bytes


def test_framed_stream_accepts_sealed_empty_stream_with_zero_budgets(tmp_path):
    """只有 footer 的空 stream 可在零 record/byte 预算下完成封存校验。"""
    path = tmp_path / "empty.ints"
    expected_footer = _write_sealed(path, ())

    records, footer = _read_all(
        path,
        max_frame_bytes=0,
        max_record_count=0,
        max_total_payload_bytes=0,
    )

    assert records == ()
    assert footer == expected_footer


def test_framed_stream_can_take_over_explicitly_open_binary_handles(tmp_path):
    """上层物理边界可先打开句柄，P0 不得为同一 stream 重新按路径打开。"""
    path = tmp_path / "preopened.ints"
    write_stream = path.open("xb", buffering=0)
    writer = IntegerFramedStreamWriter.from_open_binary(
        write_stream,
        path=path,
    )
    writer.append((1, -2, 3))
    footer = writer.seal()
    assert write_stream.closed

    read_stream = path.open("rb", buffering=0)
    reader = IntegerFramedStreamReader.from_open_binary(
        read_stream,
        path=path,
        max_frame_bytes=64,
        max_record_count=4,
        max_total_payload_bytes=64,
    )
    assert tuple(reader) == ((1, -2, 3),)
    assert reader.footer == footer
    assert read_stream.closed


def test_framed_reader_validates_budget_before_opening_or_taking_handle(tmp_path):
    """非法预算必须先失败，不能因不存在路径或已给句柄改变错误/资源顺序。"""
    missing = tmp_path / "must-not-open.ints"
    with pytest.raises(ValueError, match="max_frame_bytes"):
        IntegerFramedStreamReader(
            missing,
            max_frame_bytes=-1,
            max_record_count=4,
            max_total_payload_bytes=64,
        )

    existing = tmp_path / "external.ints"
    stream = existing.open("xb", buffering=0)
    try:
        with pytest.raises(ValueError, match="max_record_count"):
            IntegerFramedStreamReader.from_open_binary(
                stream,
                path=existing,
                max_frame_bytes=64,
                max_record_count=-1,
                max_total_payload_bytes=64,
            )
        assert stream.closed is False
    finally:
        stream.close()


def test_framed_stream_rejects_unsealed_and_duplicate_writer_lifecycle(tmp_path):
    """未写 footer 的残片不得读取为发布流，writer 也不得覆盖、追加或重复 seal。"""
    path = tmp_path / "partial.ints"
    writer = IntegerFramedStreamWriter(path)
    writer.append((1,))
    writer.close()

    with pytest.raises(IntegerFramedStreamError):
        with IntegerFramedStreamReader(
                path,
                max_frame_bytes=64,
                max_record_count=4,
                max_total_payload_bytes=64,
        ) as reader:
            reader.finish()
    with pytest.raises(FileExistsError):
        IntegerFramedStreamWriter(path)

    sealed_path = tmp_path / "sealed.ints"
    sealed_writer = IntegerFramedStreamWriter(sealed_path)
    sealed_writer.append((2,))
    sealed_writer.seal()
    with pytest.raises(IntegerFramedStreamStateError):
        sealed_writer.append((3,))
    with pytest.raises(IntegerFramedStreamStateError):
        sealed_writer.seal()


def test_framed_stream_rejects_frame_budget_before_payload_read(tmp_path, monkeypatch):
    """frame 长度一旦超预算，reader 不得进入对应 payload 的完整读取。"""
    path = tmp_path / "oversized.ints"
    _write_sealed(path, ((1,),))
    original_read_exact = codec_module._read_exact

    def guarded_read_exact(stream, size, *, label):
        """若预算门失效，专项立即暴露读取 oversized payload 的行为。"""
        if label == "整数分帧流 record payload":
            raise AssertionError("超预算 frame 不得读取 record payload")
        return original_read_exact(stream, size, label=label)

    monkeypatch.setattr(codec_module, "_read_exact", guarded_read_exact)
    with IntegerFramedStreamReader(
            path,
            max_frame_bytes=0,
            max_record_count=4,
            max_total_payload_bytes=64,
    ) as reader:
        with pytest.raises(IntegerFramedStreamBudgetExceeded):
            reader.read_record()


@pytest.mark.parametrize("kind", [
    "magic", "noncanonical_length", "payload", "truncated", "footer", "trailing",
])
def test_framed_stream_rejects_physical_or_footer_tampering(tmp_path, kind):
    """magic、varint、record、footer 和尾随字节任一漂移都不能恢复 sealed 结论。"""
    path = tmp_path / f"{kind}.ints"
    _write_sealed(path, ((5,),))
    payload = path.read_bytes()
    if kind == "magic":
        payload = b"BAD" + payload[len(b"BAD"):]
    elif kind == "noncanonical_length":
        payload = INTEGER_FRAMED_STREAM_MAGIC + b"\x82\x00" + payload[
            len(INTEGER_FRAMED_STREAM_MAGIC) + 1:]
    elif kind == "payload":
        offset = len(INTEGER_FRAMED_STREAM_MAGIC) + 1
        payload = payload[:offset] + bytes([payload[offset] ^ 1]) + payload[offset + 1:]
    elif kind == "truncated":
        payload = payload[:-1]
    elif kind == "footer":
        payload = payload[:-1] + bytes([payload[-1] ^ 1])
    elif kind == "trailing":
        payload += b"\x00"
    else:
        raise AssertionError("unknown tamper kind")
    path.write_bytes(payload)

    with pytest.raises(IntegerFramedStreamError):
        records, footer = _read_all(path)
        assert records or footer


def test_framed_stream_reader_close_does_not_turn_partial_read_into_verified_footer(tmp_path):
    """提前关闭只释放资源；没有消费 footer 就没有可用的完整性身份。"""
    path = tmp_path / "early-close.ints"
    _write_sealed(path, ((1,), (2,)))

    reader = IntegerFramedStreamReader(
        path,
        max_frame_bytes=64,
        max_record_count=4,
        max_total_payload_bytes=64,
    )
    assert reader.read_record() == (1,)
    assert reader.footer is None
    reader.close()
    assert reader.footer is None
    with pytest.raises(IntegerFramedStreamStateError):
        reader.finish()


def test_framed_stream_footer_payload_uses_existing_canonical_integer_codec(tmp_path):
    """封存 footer 的物理 payload 必须仍由现役 canonical integer codec 产生。"""
    path = tmp_path / "footer.ints"
    footer = _write_sealed(path, ((1,),))
    payload = path.read_bytes()
    footer_offset = payload.index(b"\x00", len(INTEGER_FRAMED_STREAM_MAGIC))

    assert payload[footer_offset + 1:] == encode_integer_tuple(footer.integer_tuple())


def test_framed_stream_writer_poisoned_after_partial_append_write(
        tmp_path, monkeypatch):
    """append 只要发生部分物理写入，残片 writer 就不得再追加或 seal。"""
    path = tmp_path / "partial-append.ints"
    writer = IntegerFramedStreamWriter(path)
    original_write_all = codec_module._write_all
    record_payload = encode_integer_tuple((7,))

    def partial_append_write(stream, data):
        """模拟 frame header 已写完后 payload 只写一个字节便发生 I/O 故障。"""
        if data == record_payload:
            assert stream.write(data[:1]) == 1
            raise OSError("injected partial append write failure")
        original_write_all(stream, data)

    monkeypatch.setattr(codec_module, "_write_all", partial_append_write)
    with pytest.raises(OSError, match="partial append"):
        writer.append((7,))

    assert writer.sealed is False
    assert writer.footer is None
    with pytest.raises(IntegerFramedStreamStateError, match="失效"):
        writer.append((8,))
    with pytest.raises(IntegerFramedStreamStateError, match="失效"):
        writer.seal()
    with pytest.raises(IntegerFramedStreamError):
        _read_all(path)


def test_framed_stream_writer_poisoned_after_partial_footer_write(
        tmp_path, monkeypatch):
    """footer 半写失败后不得第二次 seal，避免把不可恢复残片伪装为封存流。"""
    path = tmp_path / "partial-footer.ints"
    writer = IntegerFramedStreamWriter(path)
    writer.append((9,))
    original_write_all = codec_module._write_all

    def partial_footer_write(stream, data):
        """允许 sentinel 落盘，再让 footer payload 只写一个字节后失败。"""
        if data != b"\x00":
            assert stream.write(data[:1]) == 1
            raise OSError("injected partial footer write failure")
        original_write_all(stream, data)

    monkeypatch.setattr(codec_module, "_write_all", partial_footer_write)
    with pytest.raises(OSError, match="partial footer"):
        writer.seal()

    assert writer.sealed is False
    assert writer.footer is None
    with pytest.raises(IntegerFramedStreamStateError, match="失效"):
        writer.seal()
    with pytest.raises(IntegerFramedStreamStateError, match="失效"):
        writer.append((10,))
    with pytest.raises(IntegerFramedStreamError):
        _read_all(path)


def test_framed_stream_writer_poisoned_after_footer_flush_failure(tmp_path):
    """footer 已写入但 flush 失败时，writer 不得把结果视为已封存。"""
    path = tmp_path / "flush-failure.ints"
    writer = IntegerFramedStreamWriter(path)
    writer.append((11,))
    raw_stream = writer._stream
    assert raw_stream is not None

    def fail_flush():
        """模拟 footer 物理写入后，持久化边界报告失败。"""
        raise OSError("injected footer flush failure")

    writer._stream = SimpleNamespace(
        write=raw_stream.write,
        flush=fail_flush,
        close=raw_stream.close,
    )
    with pytest.raises(OSError, match="footer flush"):
        writer.seal()

    assert raw_stream.closed
    assert writer.sealed is False
    assert writer.footer is None
    with pytest.raises(IntegerFramedStreamStateError, match="失效"):
        writer.seal()
    with pytest.raises(IntegerFramedStreamStateError, match="失效"):
        writer.append((12,))


def test_framed_stream_writer_poisoned_after_footer_close_failure(
        tmp_path, monkeypatch):
    """footer 写完后的 close 失败也必须永久终止该 writer 生命周期。"""
    path = tmp_path / "close-failure.ints"
    writer = IntegerFramedStreamWriter(path)
    writer.append((13,))
    original_close = IntegerFramedStreamWriter.close

    def close_then_fail(self):
        """先释放真实句柄，再模拟 close 向调用方报告 I/O 失败。"""
        stream = self._stream
        original_close(self)
        if stream is not None:
            raise OSError("injected footer close failure")

    monkeypatch.setattr(IntegerFramedStreamWriter, "close", close_then_fail)
    with pytest.raises(OSError, match="footer close"):
        writer.seal()

    assert writer.sealed is False
    assert writer.footer is None
    with pytest.raises(IntegerFramedStreamStateError, match="失效"):
        writer.seal()
    with pytest.raises(IntegerFramedStreamStateError, match="失效"):
        writer.append((14,))


def test_framed_stream_module_has_no_dlg05_or_runtime_dependency():
    """P0 只承担通用 storage framing，不能耦合 R04、runtime、owner 或 formal。"""
    source_path = Path(codec_module.__file__)
    tree = ast.parse(source_path.read_bytes(), filename=str(source_path))
    local_imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(
                "pure_integer_ai"):
            local_imports.append(node.module)
    assert all("conversation_heldout" not in module for module in local_imports)
    assert all("owner" not in module and "formal" not in module
               for module in local_imports)
