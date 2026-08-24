"""K 盘大规模运行目录的通用物理边界。

本模块只管理已显式给出的 run root、普通文件和 manifest-last 发布顺序；不理解训练、
来源、DLG、JSON、SQLite 或任何语义对象。生产入口默认只接受 K 盘，测试必须显式
传入 ``require_k_drive=False``。本模块从不删除、覆盖或回退到其他目录。

路径边界在每次 I/O 前后以不跟随链接的目录/文件检查、已打开 handle 的 ``fstat`` 和
最终路径的 ``lstat`` 交叉复核。POSIX 提供 ``O_NOFOLLOW`` 时会额外用于叶文件；Windows
标准库没有目录句柄相对打开或全路径原子 no-follow 原语，因此 hostile concurrent filesystem
不能被本模块单独证明为原子安全。出现可观测替换即 fail closed；生产调用方仍须把 K run root
置于受控 ACL 下，且不得把本模块的机械检查宣传为不可变快照证明。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import stat as stat_module
from typing import BinaryIO


K_RUN_DRIVE = "K:"
"""训练和大规模运行唯一允许的数据盘符。"""

_REPARSE_POINT = 0x0400
_SHA256_SIZE = hashlib.sha256().digest_size
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_BINARY = getattr(os, "O_BINARY", 0)


# object-model: exception
class KRunBoundaryError(RuntimeError):
    """K 盘 run root、普通文件、预算或 manifest-last 物理边界不成立。"""


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class _KRunDirectoryIdentity:
    """目录 capability 捕获的稳定设备/索引身份，不使用会随子项改变的统计字段。"""

    device: int
    inode: int

    def __post_init__(self) -> None:
        """拒绝 bool、负值或宿主未提供的目录身份字段。"""
        if (type(self.device) is not int or type(self.inode) is not int
                or self.device < 0 or self.inode < 0):
            raise ValueError("K run 目录 identity 必须是非负严格 device/inode")


# object-model: capability_token; representation=private; interop=none
class _FreshRunRootCreationToken:
    """仅由 create_new_run_root 生成的模块私有创建见证，不能经公开构造器传入。"""

    __slots__ = ("directory_identity",)

    def __init__(self, directory_identity: _KRunDirectoryIdentity) -> None:
        """绑定新建时目录身份，后续 root 路径替换不能继承该见证。"""
        if not isinstance(directory_identity, _KRunDirectoryIdentity):
            raise TypeError("fresh run root token identity 类型错误")
        self.directory_identity = directory_identity


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class KRunRoot:
    """经过物理目录与 K 盘策略核验的 run root capability。"""

    path: Path
    test_transport: bool = False
    _directory_identity: _KRunDirectoryIdentity = field(
        init=False,
        repr=False,
        compare=False,
    )
    _fresh_creation_token: _FreshRunRootCreationToken | None = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """固定普通目录、目录 identity 和明确的生产/test transport 策略。"""
        if not isinstance(self.path, Path):
            raise TypeError("KRunRoot.path 必须是 Path")
        if type(self.test_transport) is not bool:
            raise TypeError("KRunRoot.test_transport 必须是 bool")
        target = _normal_existing_directory(self.path, label="K run root")
        _require_k_drive(
            target,
            required=not self.test_transport,
            label="K run root",
        )
        object.__setattr__(self, "path", target)
        object.__setattr__(
            self,
            "_directory_identity",
            _directory_identity_from_path(target, label="K run root"),
        )
        object.__setattr__(self, "_fresh_creation_token", None)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class KRunFileDigest:
    """一个已按预算流式读取的普通文件的字节数和完整 SHA-256。"""

    byte_count: int
    sha256: tuple[int, ...]

    def __post_init__(self) -> None:
        """固定文件长度和完整 SHA-256 的严格整数表示。"""
        if type(self.byte_count) is not int or self.byte_count < 0:
            raise ValueError("K run 文件 byte_count 必须是非负严格整数")
        if (not isinstance(self.sha256, tuple)
                or len(self.sha256) != _SHA256_SIZE
                or any(type(item) is not int or item < 0 or item > 255
                       for item in self.sha256)):
            raise ValueError("K run 文件 SHA-256 必须是完整字节整数 tuple")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class KRunFileIdentity:
    """一个时点的普通单硬链接文件对象身份，不对外宣称不可变快照。

    时间戳不属于文件身份：Windows 可以在 flush/close 后改变 ``ctime``，即使文件没有被
    替换。长度在写入完成后才参与比较，设备和文件索引负责识别通常的路径替换。
    """

    device: int
    inode: int
    byte_count: int
    link_count: int


def _fail(message: str) -> None:
    """统一产生 fail-closed 的 K run boundary 错误。"""
    raise KRunBoundaryError(message)


def _require_bool(value: bool, *, label: str) -> bool:
    """拒绝把整数或其他真值对象隐式当作 K 盘策略开关。"""
    if type(value) is not bool:
        raise TypeError(f"{label} 必须是 bool")
    return value


def _lstat(path: Path, *, label: str) -> os.stat_result:
    """读取最终目录项的 non-follow stat，拒绝无法审计的路径。"""
    try:
        return os.stat(path, follow_symlinks=False)
    except OSError as exc:
        _fail(f"{label} 不可 stat")
        raise AssertionError from exc


def _require_normal_directory_stat(
        stat_result: os.stat_result, *, label: str,
        ) -> None:
    """核验一个已由 lstat 取得的目录项确为非 reparse 普通目录。"""
    attributes = getattr(stat_result, "st_file_attributes", 0)
    if (not stat_module.S_ISDIR(stat_result.st_mode)
            or stat_module.S_ISLNK(stat_result.st_mode)
            or attributes & _REPARSE_POINT):
        _fail(f"{label} 不是普通目录")


def _directory_identity_from_path(
        path: Path, *, label: str,
        ) -> _KRunDirectoryIdentity:
    """从 non-follow 目录 stat 提取不随子目录增减变化的 capability identity。"""
    stat_result = _lstat(path, label=label)
    _require_normal_directory_stat(stat_result, label=label)
    return _KRunDirectoryIdentity(stat_result.st_dev, stat_result.st_ino)


def _lexists(path: Path) -> bool:
    """在 broken link 情况下也保留路径存在性，避免被 exists 静默绕过。"""
    return os.path.lexists(path)


def _absolute_path(
        value: str | Path, *, label: str, allow_volume_root: bool = False,
        ) -> Path:
    """拒绝相对和 ``..``；仅父目录验证可接受盘符根。"""
    if not isinstance(value, (str, Path)):
        raise TypeError(f"{label} 必须是 str 或 Path")
    path = Path(value)
    if (not path.is_absolute() or ".." in path.parts
            or (not allow_volume_root and not path.name)):
        _fail(f"{label} 必须是不含 .. 的非根绝对路径")
    return path


def _require_k_drive(path: Path, *, required: bool, label: str) -> None:
    """生产策略开启时仅允许 K 盘，不把 test transport 当作生产回退。"""
    if required and path.drive.upper() != K_RUN_DRIVE:
        _fail(f"{label} 必须位于 K 盘")


def _normal_existing_directory(
        path: Path, *, label: str, allow_volume_root: bool = False,
        ) -> Path:
    """逐级验证既有普通目录，防止 resolve 跟随 link/reparse 越界。"""
    path = _absolute_path(
        path,
        label=label,
        allow_volume_root=allow_volume_root,
    )
    for current in (path, *path.parents):
        if not _lexists(current):
            _fail(f"{label} 不存在")
        stat_result = _lstat(current, label=label)
        _require_normal_directory_stat(stat_result, label=label)
        if current == current.parent:
            break
    return path


def _root_path(root: KRunRoot) -> Path:
    """在每次 I/O 前重验普通根、盘符策略及捕获目录 identity。"""
    if not isinstance(root, KRunRoot):
        raise TypeError("root 必须是 KRunRoot")
    target = _normal_existing_directory(root.path, label="K run root")
    _require_k_drive(
        target,
        required=not root.test_transport,
        label="K run root",
    )
    if _directory_identity_from_path(target, label="K run root") != root._directory_identity:
        _fail("K run root 目录身份漂移")
    return target


def _fingerprint_from_stat(
        stat_result: os.stat_result, *, label: str,
        ) -> KRunFileIdentity:
    """从已打开或不跟随链接的 stat 结果固定普通文件身份。"""
    if (not stat_module.S_ISREG(stat_result.st_mode)
            or getattr(stat_result, "st_file_attributes", 0)
            & _REPARSE_POINT):
        _fail(f"{label} 不是普通文件")
    values = (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_nlink,
    )
    if (any(type(value) is not int for value in values)
            or stat_result.st_size < 0 or stat_result.st_nlink != 1):
        _fail(f"{label} 文件身份或硬链接数非法")
    return KRunFileIdentity(*values)


def _path_fingerprint(path: Path, *, label: str) -> KRunFileIdentity:
    """读取最终路径的 non-follow 指纹，供 handle 与发布后路径交叉核验。"""
    stat_result = _lstat(path, label=label)
    return _fingerprint_from_stat(stat_result, label=label)


def open_existing_run_root(
        root: str | Path, *, require_k_drive: bool = True,
        label: str = "K run root",
        ) -> KRunRoot:
    """打开既有普通 run root；生产默认拒绝 K 盘以外的路径。"""
    _require_bool(require_k_drive, label="require_k_drive")
    if not isinstance(label, str) or not label:
        raise TypeError("label 必须是非空 str")
    target = _normal_existing_directory(Path(root), label=label)
    _require_k_drive(target, required=require_k_drive, label=label)
    return KRunRoot(target, test_transport=not require_k_drive)


def create_new_run_root(
        root: str | Path, *, require_k_drive: bool = True,
        label: str = "K run root",
        ) -> KRunRoot:
    """排他创建此前不存在的 run root；失败后不尝试删除任何残留。"""
    _require_bool(require_k_drive, label="require_k_drive")
    if not isinstance(label, str) or not label:
        raise TypeError("label 必须是非空 str")
    raw = _absolute_path(root, label=label)
    if _lexists(raw):
        _fail(f"{label} 必须此前不存在且不是链接")
    parent = _normal_existing_directory(
        raw.parent,
        label=f"{label} parent",
        allow_volume_root=True,
    )
    target = parent / raw.name
    _require_k_drive(target, required=require_k_drive, label=label)
    if _lexists(target):
        _fail(f"{label} 已存在")
    try:
        target.mkdir()
    except FileExistsError as exc:
        _fail(f"{label} 已被并发创建")
        raise AssertionError from exc
    except OSError as exc:
        _fail(f"{label} 创建失败")
        raise AssertionError from exc
    result = open_existing_run_root(
        target,
        require_k_drive=require_k_drive,
        label=label,
    )
    object.__setattr__(
        result,
        "_fresh_creation_token",
        _FreshRunRootCreationToken(result._directory_identity),
    )
    return result


def require_disjoint_run_roots(
        left: KRunRoot, right: KRunRoot, *, label: str = "K run roots",
        ) -> None:
    """拒绝 source/run 或两个 run root 相同、嵌套或可相互覆盖。"""
    left_path = _root_path(left)
    right_path = _root_path(right)
    if (left_path == right_path or left_path.is_relative_to(right_path)
            or right_path.is_relative_to(left_path)):
        _fail(f"{label} 不得相同或嵌套")


def require_fresh_empty_run_root(
        root: KRunRoot, *, label: str = "K run root",
        ) -> None:
    """要求 create-new capability 未被替换且当前没有任何目录项。

    此检查不能由 ``open_existing_run_root`` 或公开 ``KRunRoot`` 构造伪造：只有
    ``create_new_run_root`` 会绑定模块私有创建 token。目录枚举只读取首个 child，故对
    误传的大目录仍为 O(1) 内存；任何普通文件、空目录、链接或 reparse child 都拒绝。
    """
    root_path = _root_path(root)
    token = root._fresh_creation_token
    if (not isinstance(token, _FreshRunRootCreationToken)
            or token.directory_identity != root._directory_identity):
        _fail(f"{label} 必须是本次 create_new_run_root 创建的 capability")
    try:
        with os.scandir(root_path) as children:
            first_child = next(children, None)
    except OSError as exc:
        _fail(f"{label} 不可枚举")
        raise AssertionError from exc
    if first_child is not None:
        _fail(f"{label} 必须为空")
    _root_path(root)


def require_created_run_root(
        root: KRunRoot, *, label: str = "K run root",
        ) -> None:
    """要求 capability 来自本次排他新建，但允许已登记的 stage 继续存在。

    多阶段 K run 在首个阶段后不再为空，仍须拒绝通过重新打开路径或直接构造
    ``KRunRoot`` 伪造的 capability。该检查只验证 root 身份与私有创建 token；调用方
    必须自行对将要新建的 stage/file 使用排他创建和独立的空目录核验。
    """
    _root_path(root)
    token = root._fresh_creation_token
    if (not isinstance(token, _FreshRunRootCreationToken)
            or token.directory_identity != root._directory_identity):
        _fail(f"{label} 必须是本次 create_new_run_root 创建的 capability")


def is_created_run_root(root: KRunRoot) -> bool:
    """在已重验物理 root 后报告它是否保有本进程的新建见证，不授予写入权限。"""
    _root_path(root)
    token = root._fresh_creation_token
    return (isinstance(token, _FreshRunRootCreationToken)
            and token.directory_identity == root._directory_identity)


def _relative_parts(value: str | Path, *, label: str) -> tuple[str, ...]:
    """拆解安全相对路径，拒绝根、空段、绝对路径和向上逃逸。"""
    if not isinstance(value, (str, Path)):
        raise TypeError(f"{label} 必须是 str 或 Path")
    relative = Path(value)
    if (relative.is_absolute() or relative.drive or relative.root
            or not relative.parts
            or ".." in relative.parts):
        _fail(f"{label} 必须是非空且不含 .. 的相对路径")
    if any(part in {"", ".", ".."} for part in relative.parts):
        _fail(f"{label} 含非法路径成分")
    return relative.parts


def relative_path(
        root: KRunRoot, relative: str | Path,
        *, label: str = "K run relative path",
        ) -> Path:
    """将一条已验证相对路径限制在既有 run root 的词法边界内。"""
    root_path = _root_path(root)
    parts = _relative_parts(relative, label=label)
    result = root_path.joinpath(*parts)
    if not result.is_relative_to(root_path):
        _fail(f"{label} 路径越界")
    return result


def _require_normal_relative_directory(
        root: Path, relative: str | Path, *, label: str,
        ) -> Path:
    """读取既有 root 内目录并拒绝每一级 symlink/reparse。"""
    parts = _relative_parts(relative, label=label)
    current = root
    for part in parts:
        current = current / part
        if not _lexists(current):
            _fail(f"{label} 不是普通目录")
        _require_normal_directory_stat(_lstat(current, label=label), label=label)
        if not current.is_relative_to(root):
            _fail(f"{label} 路径越界")
    return current


def ensure_normal_relative_directory(
        root: KRunRoot, relative: str | Path,
        *, label: str = "K run relative directory",
        ) -> KRunRoot:
    """按层创建或复核 root 内普通目录，不跟随 link/reparse 且不清理旧目录。"""
    root_path = _root_path(root)
    parts = _relative_parts(relative, label=label)
    current = root_path
    for index, part in enumerate(parts):
        current = current / part
        if _lexists(current):
            _require_normal_directory_stat(
                _lstat(current, label=label), label=label)
        else:
            try:
                current.mkdir()
            except FileExistsError as exc:
                _fail(f"{label} 已被并发替换")
                raise AssertionError from exc
            except OSError as exc:
                _fail(f"{label} 创建失败")
                raise AssertionError from exc
        _require_normal_directory_stat(_lstat(current, label=label), label=label)
        _require_normal_relative_directory(
            root_path,
            Path(*parts[:index + 1]),
            label=label,
        )
        if not current.is_relative_to(root_path):
            _fail(f"{label} 路径越界")
    return KRunRoot(current, test_transport=root.test_transport)


def require_plain_file(
        root: KRunRoot, relative: str | Path, *, label: str = "K run file",
        ) -> Path:
    """返回 root 内普通单硬链接文件，拒绝链接、reparse、目录和越界。"""
    root_path = _root_path(root)
    relative_value = Path(relative)
    path = relative_path(root, relative_value, label=label)
    if relative_value.parent.parts:
        _require_normal_relative_directory(
            root_path,
            relative_value.parent,
            label=f"{label} parent",
        )
    if not _lexists(path):
        _fail(f"{label} 不是普通文件")
    _path_fingerprint(path, label=label)
    if not path.is_relative_to(root_path):
        _fail(f"{label} 路径越界")
    return path


def capture_plain_file_identity(
        root: KRunRoot, relative: str | Path,
        *, label: str = "K run file",
        ) -> KRunFileIdentity:
    """捕获当前普通文件对象身份，供同一有界操作结束后复核。

    该值只用于检测可观测的路径替换、长度或硬链接漂移；它不是不可变 snapshot，
    更不能检测同 inode、同长度的并发原地改写。
    """
    path = require_plain_file(root, relative, label=label)
    return _path_fingerprint(path, label=label)


def require_plain_file_identity(
        root: KRunRoot, relative: str | Path,
        identity: KRunFileIdentity,
        *, label: str = "K run file",
        ) -> Path:
    """要求当前普通文件仍与此前捕获的对象身份完全一致。"""
    if not isinstance(identity, KRunFileIdentity):
        raise TypeError("identity 必须是 KRunFileIdentity")
    path = require_plain_file(root, relative, label=label)
    if _path_fingerprint(path, label=label) != identity:
        _fail(f"{label} 文件身份漂移")
    return path


def open_exclusive_binary(
        root: KRunRoot, relative: str | Path,
        *, label: str = "K run file",
        ) -> BinaryIO:
    """排他新建一个 root 内文件；父目录必须已由调用方显式建立。"""
    relative_value = Path(relative)
    path = relative_path(root, relative_value, label=label)
    if relative_value.parent.parts:
        _require_normal_relative_directory(
            _root_path(root),
            relative_value.parent,
            label=f"{label} parent",
        )
    if _lexists(path):
        _fail(f"{label} 已存在或不是可新建普通文件")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_BINARY | _O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags, 0o600)
        stream = os.fdopen(descriptor, "wb", buffering=0)
        descriptor = None
    except FileExistsError as exc:
        _fail(f"{label} 已被并发创建")
        raise AssertionError from exc
    except OSError as exc:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        _fail(f"{label} 排他创建失败")
        raise AssertionError from exc
    try:
        handle_fingerprint = _fingerprint_from_stat(
            os.fstat(stream.fileno()),
            label=label,
        )
        if relative_value.parent.parts:
            _require_normal_relative_directory(
                _root_path(root),
                relative_value.parent,
                label=f"{label} parent",
            )
        else:
            _root_path(root)
        if _path_fingerprint(path, label=label) != handle_fingerprint:
            _fail(f"{label} 排他打开前后文件身份漂移")
    except BaseException:
        try:
            stream.close()
        except OSError:
            pass
        raise
    return stream


def _write_all(stream: BinaryIO, payload: bytes) -> None:
    """将有限 bytes 完整写入一个已排他打开的二进制文件。"""
    view = memoryview(payload)
    while view:
        written = stream.write(view)
        if type(written) is not int or written <= 0 or written > len(view):
            raise OSError("K run 文件写入未完成")
        view = view[written:]


def write_exclusive_bytes(
        root: KRunRoot, relative: str | Path, payload: bytes,
        *, label: str = "K run file",
        ) -> Path:
    """排他写入有限 bytes 并 flush；失败不清理残片，也不发布 manifest。"""
    if not isinstance(payload, bytes):
        raise TypeError("payload 必须是 bytes")
    path = relative_path(root, relative, label=label)
    handle_fingerprint: KRunFileIdentity | None = None
    try:
        with open_exclusive_binary(root, relative, label=label) as stream:
            _write_all(stream, payload)
            stream.flush()
            handle_fingerprint = _fingerprint_from_stat(
                os.fstat(stream.fileno()),
                label=label,
            )
    except KRunBoundaryError:
        raise
    except OSError as exc:
        _fail(f"{label} 写入或关闭失败")
        raise AssertionError from exc
    if handle_fingerprint is None:
        raise AssertionError("K run 文件写入后缺 handle fingerprint")
    require_plain_file(root, relative, label=label)
    if _path_fingerprint(path, label=label) != handle_fingerprint:
        _fail(f"{label} 写入后文件身份漂移")
    return path


def open_plain_binary(
        root: KRunRoot, relative: str | Path,
        *,
        label: str = "K run file",
        expected_identity: KRunFileIdentity | None = None,
        ) -> BinaryIO:
    """打开一个已复核的普通文件句柄，调用方负责关闭。

    先核验 leaf 和全部父目录，再以可用的 ``O_NOFOLLOW`` 打开；打开后必须让 handle
    的身份、当前 leaf 和父链再次一致，才把句柄交给调用方。Windows 上该流程只能检测
    可观测的并发替换，不能替代受控 ACL 或声称全路径原子 no-follow。
    """
    if (expected_identity is not None
            and not isinstance(expected_identity, KRunFileIdentity)):
        raise TypeError("expected_identity 必须是 KRunFileIdentity 或 None")
    path = require_plain_file(root, relative, label=label)
    fingerprint_before = _path_fingerprint(path, label=label)
    if (expected_identity is not None
            and fingerprint_before != expected_identity):
        _fail(f"{label} 打开前文件身份漂移")
    flags = os.O_RDONLY | _O_BINARY | _O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        stream = os.fdopen(descriptor, "rb", buffering=0)
        descriptor = None
    except OSError as exc:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        _fail(f"{label} 读取打开失败")
        raise AssertionError from exc
    try:
        handle_fingerprint = _fingerprint_from_stat(
            os.fstat(stream.fileno()),
            label=label,
        )
        if handle_fingerprint != fingerprint_before:
            _fail(f"{label} 打开前后文件身份漂移")
        relative_value = Path(relative)
        if relative_value.parent.parts:
            _require_normal_relative_directory(
                _root_path(root),
                relative_value.parent,
                label=f"{label} parent",
            )
        else:
            _root_path(root)
        if _path_fingerprint(path, label=label) != handle_fingerprint:
            _fail(f"{label} 打开前后路径身份漂移")
    except BaseException:
        try:
            stream.close()
        except OSError:
            pass
        raise
    return stream


def sha256_plain_file(
        root: KRunRoot, relative: str | Path, *, max_bytes: int,
        chunk_bytes: int = 64 * 1024,
        label: str = "K run file",
        ) -> KRunFileDigest:
    """按显式预算流式读取普通文件，并在读取前后复核路径和文件长度。"""
    if type(max_bytes) is not int or max_bytes < 0:
        raise ValueError("max_bytes 必须是非负严格整数")
    if type(chunk_bytes) is not int or chunk_bytes <= 0:
        raise ValueError("chunk_bytes 必须是正严格整数")
    path = relative_path(root, relative, label=label)
    digest = hashlib.sha256()
    byte_count = 0
    fingerprint_before: KRunFileIdentity | None = None
    try:
        with open_plain_binary(root, relative, label=label) as stream:
            fingerprint_before = _fingerprint_from_stat(
                os.fstat(stream.fileno()),
                label=label,
            )
            if fingerprint_before.byte_count > max_bytes:
                _fail(f"{label} 超过读取预算")
            while True:
                remaining_with_sentinel = max_bytes - byte_count + 1
                chunk = stream.read(min(chunk_bytes, remaining_with_sentinel))
                if not isinstance(chunk, bytes):
                    _fail(f"{label} 返回非 bytes 读取结果")
                if not chunk:
                    break
                byte_count += len(chunk)
                if byte_count > max_bytes:
                    _fail(f"{label} 超过读取预算")
                digest.update(chunk)
            if (_fingerprint_from_stat(
                    os.fstat(stream.fileno()),
                    label=label,
            ) != fingerprint_before):
                _fail(f"{label} 读取期间 handle 身份漂移")
    except KRunBoundaryError:
        raise
    except OSError as exc:
        _fail(f"{label} 流式读取失败")
        raise AssertionError from exc
    if fingerprint_before is None:
        raise AssertionError("K run 文件读取后缺 handle fingerprint")
    require_plain_file(root, relative, label=label)
    fingerprint_after = _path_fingerprint(path, label=label)
    if (fingerprint_before != fingerprint_after
            or fingerprint_before.byte_count != byte_count):
        _fail(f"{label} 读取期间文件身份漂移")
    return KRunFileDigest(byte_count, tuple(digest.digest()))


def _collect_normal_files(root: KRunRoot, *, label: str) -> frozenset[Path]:
    """递归枚举 root 内叶文件，拒绝任意 link/reparse、硬链接或非普通对象。"""
    found: set[Path] = set()

    root_path = _root_path(root)

    def visit(directory: Path, relative: Path) -> None:
        if relative.parts:
            _require_normal_relative_directory(
                root_path,
                relative,
                label=f"{label} directory",
            )
        try:
            children = tuple(directory.iterdir())
        except OSError as exc:
            _fail(f"{label} 不可枚举")
            raise AssertionError from exc
        for child in children:
            child_relative = relative / child.name
            child_stat = _lstat(child, label=label)
            if stat_module.S_ISDIR(child_stat.st_mode):
                _require_normal_directory_stat(child_stat, label=label)
                _require_normal_relative_directory(
                    root_path,
                    child_relative,
                    label=f"{label} directory",
                )
                visit(child, child_relative)
                continue
            if not stat_module.S_ISREG(child_stat.st_mode):
                _fail(f"{label} 含非普通文件")
            require_plain_file(root, child_relative, label=f"{label} file")
            found.add(child_relative)

        if relative.parts:
            _require_normal_relative_directory(
                root_path,
                relative,
                label=f"{label} directory",
            )

    visit(root_path, Path())
    return frozenset(found)


def require_exact_file_closure(
        root: KRunRoot, expected_files: frozenset[Path],
        *, label: str = "K run publication",
        ) -> None:
    """要求专用 publication root 的递归文件闭包精确等于声明集合。"""
    if not isinstance(expected_files, frozenset) or not expected_files:
        raise ValueError("expected_files 必须是非空 frozenset[Path]")
    normalized: set[Path] = set()
    for relative in expected_files:
        if not isinstance(relative, Path):
            raise TypeError("expected_files 必须只含 Path")
        _relative_parts(relative, label=f"{label} expected file")
        normalized.add(relative)
    if len(normalized) != len(expected_files):
        raise ValueError("expected_files 不得有重复路径")
    actual = _collect_normal_files(root, label=label)
    if actual != frozenset(normalized):
        _fail(f"{label} 文件闭包与声明不一致")


def publish_manifest_last(
        root: KRunRoot, manifest_relative: Path, payload: bytes,
        pre_manifest_files: frozenset[Path],
        *, label: str = "K run publication",
        ) -> Path:
    """只在 pre-manifest 闭包完整且 manifest 不存在时排他写入最终发布标记。"""
    if not isinstance(manifest_relative, Path):
        raise TypeError("manifest_relative 必须是 Path")
    _relative_parts(manifest_relative, label=f"{label} manifest")
    if manifest_relative in pre_manifest_files:
        raise ValueError("manifest_relative 不得属于 pre_manifest_files")
    manifest_path = relative_path(
        root,
        manifest_relative,
        label=f"{label} manifest",
    )
    if _lexists(manifest_path):
        _fail(f"{label} manifest 已存在")
    require_exact_file_closure(
        root,
        pre_manifest_files,
        label=f"{label} pre-manifest",
    )
    path = write_exclusive_bytes(
        root,
        manifest_relative,
        payload,
        label=f"{label} manifest",
    )
    require_exact_file_closure(
        root,
        frozenset((*pre_manifest_files, manifest_relative)),
        label=label,
    )
    return path


__all__ = [
    "K_RUN_DRIVE",
    "KRunBoundaryError",
    "KRunFileDigest",
    "KRunFileIdentity",
    "KRunRoot",
    "capture_plain_file_identity",
    "create_new_run_root",
    "ensure_normal_relative_directory",
    "is_created_run_root",
    "open_existing_run_root",
    "open_exclusive_binary",
    "open_plain_binary",
    "publish_manifest_last",
    "relative_path",
    "require_created_run_root",
    "require_disjoint_run_roots",
    "require_fresh_empty_run_root",
    "require_exact_file_closure",
    "require_plain_file_identity",
    "require_plain_file",
    "sha256_plain_file",
    "write_exclusive_bytes",
]
