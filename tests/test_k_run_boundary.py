"""P2 通用 K 盘 run boundary 的目标物理边界专项。"""
from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
import stat as stat_module
from types import SimpleNamespace

import pytest

import pure_integer_ai.storage.k_run_boundary as boundary_module
from pure_integer_ai.storage.k_run_boundary import (
    KRunBoundaryError,
    KRunRoot,
    capture_plain_file_identity,
    create_new_run_root,
    ensure_normal_relative_directory,
    open_existing_run_root,
    open_exclusive_binary,
    open_plain_binary,
    publish_manifest_last,
    relative_path,
    require_created_run_root,
    require_disjoint_run_roots,
    require_exact_file_closure,
    require_fresh_empty_run_root,
    require_plain_file,
    require_plain_file_identity,
    sha256_plain_file,
    write_exclusive_bytes,
)


def _root(tmp_path: Path, name: str = "run") -> KRunRoot:
    """仅以显式 test transport 开关创建 D 盘临时 run root。"""
    return create_new_run_root(
        tmp_path / name,
        require_k_drive=False,
        label="test transport root",
    )


def test_root_creation_defaults_to_k_and_test_transport_is_explicit(tmp_path):
    """生产默认拒绝 D 盘，测试 transport 不能隐式继承这一例外。"""
    rejected = tmp_path / "production-root"
    with pytest.raises(KRunBoundaryError, match="K 盘"):
        create_new_run_root(rejected)
    assert not rejected.exists()

    root = _root(tmp_path)
    assert root.path == (tmp_path / "run").resolve()
    assert open_existing_run_root(
        root.path,
        require_k_drive=False,
        label="test transport root",
    ) == root
    with pytest.raises(KRunBoundaryError, match="此前不存在|已存在"):
        create_new_run_root(
            root.path,
            require_k_drive=False,
            label="test transport root",
        )
    with pytest.raises(KRunBoundaryError, match="K 盘"):
        open_existing_run_root(root.path)
    volume_root = Path(tmp_path.anchor)
    assert boundary_module._normal_existing_directory(
        volume_root,
        label="test volume root",
        allow_volume_root=True,
    ) == volume_root.resolve()


def test_fresh_empty_root_requires_new_capability_identity_and_o1_empty_check(
        tmp_path, monkeypatch):
    """B1 work root 必须来自本次排他新建，目录替换或任意 child 都不得通过。"""
    fresh = _root(tmp_path, "fresh")
    require_fresh_empty_run_root(fresh, label="fresh root")
    require_created_run_root(fresh, label="created root")

    reopened = open_existing_run_root(
        fresh.path,
        require_k_drive=False,
        label="reopened root",
    )
    direct = KRunRoot(fresh.path, test_transport=True)
    with pytest.raises(KRunBoundaryError, match="create_new_run_root"):
        require_fresh_empty_run_root(reopened, label="reopened root")
    with pytest.raises(KRunBoundaryError, match="create_new_run_root"):
        require_fresh_empty_run_root(direct, label="direct root")
    with pytest.raises(KRunBoundaryError, match="create_new_run_root"):
        require_created_run_root(reopened, label="reopened created root")
    with pytest.raises(KRunBoundaryError, match="create_new_run_root"):
        require_created_run_root(direct, label="direct created root")

    dirty = _root(tmp_path, "dirty")
    ensure_normal_relative_directory(dirty, "empty-child")
    with pytest.raises(KRunBoundaryError, match="必须为空"):
        require_fresh_empty_run_root(dirty, label="dirty root")
    require_created_run_root(dirty, label="dirty created root")

    original_identity = boundary_module._directory_identity_from_path

    def drifted_identity(path, *, label):
        """模拟 root 路径被替换为另一个普通目录后 identity 已不可继续使用。"""
        actual = original_identity(path, label=label)
        return boundary_module._KRunDirectoryIdentity(
            actual.device,
            actual.inode + 1,
        )

    monkeypatch.setattr(
        boundary_module,
        "_directory_identity_from_path",
        drifted_identity,
    )
    with pytest.raises(KRunBoundaryError, match="目录身份漂移"):
        require_fresh_empty_run_root(fresh, label="fresh drift")


def test_relative_directories_and_exclusive_files_cannot_escape_or_overwrite(tmp_path):
    """父目录须显式建立，文件只能在 root 内排他新建一次。"""
    root = _root(tmp_path)
    directory = ensure_normal_relative_directory(root, Path("work") / "attempt")
    assert directory.path == root.path / "work" / "attempt"

    with open_exclusive_binary(root, Path("work") / "attempt" / "record.bin") as stream:
        assert stream.write(b"abc") == 3
    assert require_plain_file(root, Path("work") / "attempt" / "record.bin")
    with pytest.raises(KRunBoundaryError, match="已存在"):
        open_exclusive_binary(root, Path("work") / "attempt" / "record.bin")

    for invalid in (
            Path(),
            Path("..") / "escape",
            Path("/absolute"),
            Path("C:drive-relative"),
            ):
        with pytest.raises(KRunBoundaryError):
            relative_path(root, invalid)
    with pytest.raises(KRunBoundaryError, match="普通目录"):
        open_exclusive_binary(root, Path("missing") / "record.bin")
    with pytest.raises(TypeError, match="KRunRoot"):
        ensure_normal_relative_directory(root.path, "unauthorized-path")


def test_stream_digest_is_bounded_and_plain_file_rejects_hardlink(tmp_path):
    """SHA 只经受预算约束的流式读取，硬链接不得进入 run artifact。"""
    root = _root(tmp_path)
    ensure_normal_relative_directory(root, "published")
    payload = b"bounded-digest"
    path = write_exclusive_bytes(root, Path("published") / "data.bin", payload)

    digest = sha256_plain_file(
        root,
        Path("published") / "data.bin",
        max_bytes=len(payload),
        chunk_bytes=3,
    )
    assert digest.byte_count == len(payload)
    assert digest.sha256 == tuple(hashlib.sha256(payload).digest())
    with pytest.raises(KRunBoundaryError, match="预算"):
        sha256_plain_file(
            root,
            Path("published") / "data.bin",
            max_bytes=len(payload) - 1,
        )

    linked = root.path / "published" / "linked.bin"
    try:
        os.link(path, linked)
    except OSError:
        pytest.skip("当前文件系统不支持创建硬链接专项")
    with pytest.raises(KRunBoundaryError, match="硬链接"):
        require_plain_file(root, Path("published") / "data.bin")


def test_open_plain_binary_returns_verified_caller_owned_handle(tmp_path):
    """P2 可将由边界验证后的句柄交给 P0，而非重新对路径执行裸打开。"""
    root = _root(tmp_path)
    ensure_normal_relative_directory(root, "published")
    relative = Path("published") / "data.bin"
    write_exclusive_bytes(root, relative, b"verified")

    with open_plain_binary(root, relative) as stream:
        assert stream.read() == b"verified"
        assert stream.closed is False
    assert stream.closed is True


def test_captured_identity_pins_open_and_post_read_recheck_to_same_file(tmp_path):
    """大 stream 可用一次解码读取后做 O(1) identity 重验，而非再次完整扫 SHA。"""
    root = _root(tmp_path)
    ensure_normal_relative_directory(root, "published")
    relative = Path("published") / "data.bin"
    path = write_exclusive_bytes(root, relative, b"expected")
    identity = capture_plain_file_identity(root, relative)

    with open_plain_binary(root, relative, expected_identity=identity) as stream:
        assert stream.read() == b"expected"
    assert require_plain_file_identity(root, relative, identity) == path

    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"changed!")
    os.replace(replacement, path)
    with pytest.raises(KRunBoundaryError, match="身份漂移"):
        require_plain_file_identity(root, relative, identity)
    with pytest.raises(KRunBoundaryError, match="打开前文件身份漂移"):
        open_plain_binary(root, relative, expected_identity=identity)


def test_file_fingerprint_excludes_mutable_close_timestamps():
    """Windows flush/close 可改变 ctime；它不是同一文件 handle 的身份字段。"""
    common = {
        "st_mode": stat_module.S_IFREG | 0o600,
        "st_dev": 7,
        "st_ino": 19,
        "st_size": 23,
        "st_nlink": 1,
    }
    before = SimpleNamespace(
        **common,
        st_mtime_ns=101,
        st_ctime_ns=103,
    )
    after = SimpleNamespace(
        **common,
        st_mtime_ns=107,
        st_ctime_ns=109,
    )

    assert boundary_module._fingerprint_from_stat(
        before, label="timestamp regression") == boundary_module._fingerprint_from_stat(
            after, label="timestamp regression")


def test_synthetic_reparse_and_hardlinked_file_stats_fail_closed():
    """即使宿主不允许建 junction/hardlink，lstat 结果中的两类风险仍必须被拒绝。"""
    common = {
        "st_mode": stat_module.S_IFREG | 0o600,
        "st_dev": 7,
        "st_ino": 19,
        "st_size": 23,
        "st_nlink": 1,
    }
    reparse = SimpleNamespace(
        **common,
        st_file_attributes=boundary_module._REPARSE_POINT,
    )
    with pytest.raises(KRunBoundaryError, match="普通文件"):
        boundary_module._fingerprint_from_stat(reparse, label="synthetic reparse")

    hardlinked = SimpleNamespace(**{
        **common,
        "st_nlink": 2,
        "st_file_attributes": 0,
    })
    with pytest.raises(KRunBoundaryError, match="硬链接"):
        boundary_module._fingerprint_from_stat(
            hardlinked,
            label="synthetic hardlink",
        )


def test_stream_digest_rejects_substituted_open_handle_before_payload_read(
        tmp_path, monkeypatch):
    """无原子目录 no-follow 时，lstat/fstat 交叉核验仍须在读正文前拒绝替换。"""
    root = _root(tmp_path)
    ensure_normal_relative_directory(root, "published")
    relative = Path("published") / "data.bin"
    write_exclusive_bytes(root, relative, b"expected")
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"substituted")
    expected_path = root.path / relative
    real_open = boundary_module.os.open

    def substituted_open(path, flags, mode=0o777):
        """只替换目标读取的 handle；路径本身仍指向原始 artifact。"""
        if Path(path) == expected_path:
            return real_open(outside, flags, mode)
        return real_open(path, flags, mode)

    monkeypatch.setattr(boundary_module.os, "open", substituted_open)
    with pytest.raises(KRunBoundaryError, match="打开前后文件身份漂移"):
        sha256_plain_file(root, relative, max_bytes=64)


def test_exclusive_open_and_digest_recheck_parent_after_handle_open(
        tmp_path, monkeypatch):
    """父目录在 precheck 后被替换时，handle 接受或 payload 读取前必须再次拒绝。"""
    root = _root(tmp_path)
    ensure_normal_relative_directory(root, "published")
    open_relative = Path("published") / "open.bin"
    digest_relative = Path("published") / "data.bin"
    original_parent_check = boundary_module._require_normal_relative_directory
    parent_checks = 0

    def fail_second_parent_check(*args, **kwargs):
        """模拟 handle 已获得后发现父目录成为 reparse point。"""
        nonlocal parent_checks
        parent_checks += 1
        if parent_checks == 2:
            raise KRunBoundaryError("injected post-open parent reparse")
        return original_parent_check(*args, **kwargs)

    monkeypatch.setattr(
        boundary_module,
        "_require_normal_relative_directory",
        fail_second_parent_check,
    )
    with pytest.raises(KRunBoundaryError, match="post-open parent reparse"):
        open_exclusive_binary(root, open_relative)
    assert parent_checks == 2

    monkeypatch.setattr(
        boundary_module,
        "_require_normal_relative_directory",
        original_parent_check,
    )
    write_exclusive_bytes(root, digest_relative, b"expected")
    parent_checks = 0
    monkeypatch.setattr(
        boundary_module,
        "_require_normal_relative_directory",
        fail_second_parent_check,
    )

    class _NoPayloadDigest:
        """若 parent postcheck 失效并读到正文，专项立即失败。"""

        def update(self, _chunk):
            """禁止测试路径处理任何 payload。"""
            raise AssertionError("post-open parent recheck 前不得读取 payload")

    monkeypatch.setattr(boundary_module.hashlib, "sha256", _NoPayloadDigest)
    with pytest.raises(KRunBoundaryError, match="post-open parent reparse"):
        sha256_plain_file(root, digest_relative, max_bytes=64)
    assert parent_checks == 2


def test_link_escape_is_rejected_when_platform_allows_link_creation(tmp_path):
    """目录链接不能被 relative path、建目录或文件边界跟随。"""
    root = _root(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = root.path / "linked"
    try:
        os.symlink(outside, linked, target_is_directory=True)
    except OSError:
        pytest.skip("当前环境不允许创建目录链接专项")
    with pytest.raises(KRunBoundaryError, match="普通目录"):
        ensure_normal_relative_directory(root, Path("linked") / "child")
    with pytest.raises(KRunBoundaryError):
        open_exclusive_binary(root, Path("linked") / "record.bin")


def test_manifest_last_requires_exact_preclosure_and_rejects_late_extra_file(tmp_path):
    """manifest 只能在完整 payload 闭包后排他写入，发布后额外文件即失效。"""
    root = _root(tmp_path)
    publication = ensure_normal_relative_directory(root, "published")
    first = Path("first.pifrs")
    second = Path("nested") / "second.pifrs"
    manifest = Path("manifest.pifrs")
    ensure_normal_relative_directory(publication, "nested")
    write_exclusive_bytes(publication, first, b"first")
    expected = frozenset({first, second})
    with pytest.raises(KRunBoundaryError, match="闭包"):
        publish_manifest_last(publication, manifest, b"manifest", expected)

    write_exclusive_bytes(publication, second, b"second")
    manifest_path = publish_manifest_last(
        publication,
        manifest,
        b"manifest",
        expected,
    )
    assert manifest_path == publication.path / manifest
    require_exact_file_closure(
        publication,
        frozenset({first, second, manifest}),
    )
    with pytest.raises(KRunBoundaryError, match="已存在"):
        publish_manifest_last(publication, manifest, b"again", expected)

    write_exclusive_bytes(publication, "extra.pifrs", b"extra")
    with pytest.raises(KRunBoundaryError, match="闭包"):
        require_exact_file_closure(
            publication,
            frozenset({first, second, manifest}),
        )


def test_disjoint_run_roots_rejects_same_or_nested_paths(tmp_path):
    """输入、work 和输出根不得借同一目录或嵌套路径重叠。"""
    root = _root(tmp_path)
    child = ensure_normal_relative_directory(root, "child")
    other = _root(tmp_path, "other")
    with pytest.raises(KRunBoundaryError, match="嵌套"):
        require_disjoint_run_roots(root, child)
    with pytest.raises(KRunBoundaryError, match="相同"):
        require_disjoint_run_roots(root, root)
    require_disjoint_run_roots(root, other)


def test_boundary_module_is_domain_neutral_and_has_no_unsafe_fallbacks():
    """通用边界不能依赖 v4/训练域，也不得引入删除、临时或环境回退。"""
    source_path = Path(boundary_module.__file__)
    tree = ast.parse(source_path.read_bytes(), filename=str(source_path))
    dynamic_imports = []
    destructive_attribute_calls = []
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        if isinstance(node, ast.Import):
            assert all(not alias.name.startswith("pure_integer_ai")
                       for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert not (node.module and node.module.startswith("pure_integer_ai"))
        elif isinstance(node, ast.Call):
            if ((isinstance(node.func, ast.Name) and node.func.id == "__import__")
                    or (isinstance(node.func, ast.Attribute)
                        and node.func.attr == "import_module")):
                dynamic_imports.append(node.lineno)
            if (isinstance(node.func, ast.Attribute)
                    and node.func.attr in {
                        "remove", "unlink", "rmdir", "rename", "replace",
                    }):
                destructive_attribute_calls.append(node.lineno)

    assert dynamic_imports == []
    assert destructive_attribute_calls == []
    assert names.isdisjoint({
        "environ", "getenv", "tempfile", "json", "sqlite3", "shutil",
        "unlink", "rmtree", "read_bytes", "write_bytes",
    })
    source_text = source_path.read_text(encoding="utf-8")
    assert "require_fresh_empty_run_root" in source_text
    assert "os.scandir" in source_text
    assert "tuple(root_path.iterdir())" not in source_text
