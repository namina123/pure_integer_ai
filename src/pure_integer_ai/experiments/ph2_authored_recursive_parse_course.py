"""LC-04 递归结构、联合 parse competition 与局部 reparse 的原创课程。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pure_integer_ai.experiments.ph2_authored_course_common import (
    AuthoredCompiledSeed,
    AuthoredCourseBuild,
    AuthoredCourseSpec,
    publish_authored_course,
)
from pure_integer_ai.experiments.ph2_capability_course_contract import (
    COURSE_EXECUTION_STATE,
    COURSE_INVARIANTS,
    COURSE_SPLIT_AXES,
    CapabilityCourseManifest,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    EXPECTED_STATES,
    SAMPLE_ROLES,
    CanonicalJsonObject,
    canonical_json_line,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_language_course_contract import (
    LANGUAGE_OBJECTIVE_KEYS,
)
from pure_integer_ai.experiments.ph2_language_coverage_contract import (
    SAMPLE_FAMILIES,
)


SOURCE_KEY = "AUTHORED_CC0_V1"
LICENSE_ID = "CC0-1.0"
COURSE_VERSION = 1
ARTIFACT_VERSION = 1
ADAPTER_VERSION = 1
GENERATOR_VERSION = 1
PARSER_VERSION = 1
PACK_NAME = "AUTHORED_CC0_V1--CC0-1.0--lc04-recursive-parse-v1"
STAGE = "W-05"
SUBSTAGE = "LC04_RECURSIVE_PARSE"
PAYLOAD_KIND = "RecursiveParseCandidateV1"
COURSE_MANIFEST_ARTIFACT_VERSION = "LC-04-recursive-parse-course-v1"
COURSE_MANIFEST_PATH = Path(
    "data/ph2/manifests/lc04_recursive_parse_course_v1.json")
FORMAL_ARTIFACT_RELATIVE_ROOT = (
    "ph2_dataset_artifacts/d02_language_courses_v1")

CANDIDATE_KINDS = (
    "AMBIGUOUS",
    "COORDINATED",
    "DISCONTINUOUS_DEPENDENCY",
    "GENERATION",
    "NESTED",
    "OPTIONAL",
    "PRESELECTED_TREE",
    "REPEATED",
    "RETENTION",
    "REVISION",
    "UNKNOWN",
)
NODE_KINDS = (
    "CLAUSE",
    "CONSTITUENT",
    "COORDINATE",
    "EMPTY",
    "ROOT",
    "TOKEN_GROUP",
)
EDGE_KINDS = (
    "CONTAINS",
    "COORDINATES",
    "DISCONTINUOUS",
    "ROLE_BINDS",
    "SCOPE_CONTAINS",
)
REGISTER_KEYS = ("COLLOQUIAL", "FORMAL", "NEUTRAL")
BASELINE_KINDS = (
    "JOINT_PARSE_CANDIDATES_PRESENT",
    "PRESELECTED_TREE_ONLY",
)
EVALUATOR_DIMENSIONS = (
    "AMBIGUOUS_PARSE_COMPETITION",
    "COORDINATION_STRUCTURE",
    "DISCONTINUOUS_DEPENDENCY",
    "HELD_OUT_FILLER_PARSE_DEPTH",
    "LOCAL_REPARSE_SUPERSEDE",
    "NESTED_DEPTH",
    "NULL_OPTIONAL_CONSTITUENT",
    "PRESELECTED_TREE_REJECT",
    "REPEATED_TOKEN_IDENTITY",
    "RETENTION_REVERIFY",
    "REVERSE_LINEARIZATION",
    "ROLE_SCOPE_PRESERVATION",
)
VERIFIER_NE_CONDITIONS = (
    "CAPABILITY_LEARNED_REQUESTED",
    "GENERATION_RESULT_NOT_EXECUTED",
    "NO_EVALUATOR_LABEL",
    "PARSE_CANDIDATE_MISSING",
    "RECURSION_BUDGET_EXCEEDED",
    "SEMANTIC_TRUTH_REQUESTED",
)
RETENTION_PROTOCOLS = (
    "A_TO_LC04_REVERIFY_A",
    "DUMP_RESUME_REQUIRED_AT_RUNTIME",
    "LOCAL_REPARSE_SUPERSEDE_ONLY",
)
COMBINATION_AXES = (
    "DEPTH_FAMILY",
    "FILLER_FAMILY",
    "FILLER_X_PARSE_DEPTH",
    "PARSE_FAMILY",
)
ABLATION_KEYS = (
    "DROP_INNER_SPAN",
    "DROP_PARSE_CANDIDATE",
    "PRESELECTED_TREE_ONLY",
    "REMOVE_ROLE_BINDING",
    "REMOVE_SCOPE_EDGE",
)
PAYLOAD_KEYS = (
    "baseline_kind",
    "candidate_kind",
    "generation_constraint",
    "objective_keys",
    "observed_surface",
    "parse_budget",
    "parse_candidates",
    "proposition_group",
    "retention_anchor_id",
    "sample_family",
    "selection_state",
    "split_identity",
    "surface_scope",
)

_SEED_FIELDS = {
    "baseline_kind", "candidate_kind", "depth_family", "evaluation_dimension",
    "expected_payload", "expected_state", "family", "filler_family",
    "generation_constraint", "label_owner", "license_id", "logical_order",
    "objective_keys", "observed_text", "parse_budget", "parse_candidates",
    "parse_family", "perturbation_kind", "proposition_group",
    "retention_anchor_id", "sample_family", "sample_role", "seed_id", "split",
    "supersedes_seed_id", "surface_scope", "template_family",
}
_SCOPE_FIELDS = {"genre", "language", "register", "script"}
_BUDGET_FIELDS = {
    "max_candidates", "max_depth", "max_linearization_units", "max_nodes",
}
_PARSE_FIELDS = {
    "candidate_id", "edges", "linearization_key", "nodes", "parse_key",
    "parser_version", "role_signature", "root_node_id", "scope_key",
}
_NODE_FIELDS = {
    "members", "node_id", "node_kind", "optional", "parent_node_id",
    "repeat_ordinal", "role_key", "scope_key",
}
_EDGE_FIELDS = {
    "edge_kind", "from_node_id", "order_index", "to_node_id",
}
_GENERATION_FIELDS = {
    "direction", "input_candidate_ids", "output_surface_hidden",
    "requires_joint_parse", "role_scope_preservation_required",
}
_EXPECTED_FIELDS = {
    "accepted", "accepted_surfaces", "analysis_key", "reason_code",
}
_ROLE_BY_FAMILY = {
    "AMBIGUOUS": "conflict",
    "NEGATIVE": "refute",
    "POSITIVE": "support",
    "RETENTION": "read_only_probe",
    "REVISION": "supersede",
    "UNKNOWN": "anomaly",
}
_STATE_BY_FAMILY = {
    "AMBIGUOUS": "CONFLICT",
    "NEGATIVE": "FALSE",
    "POSITIVE": "TRUE",
    "RETENTION": "TRUE",
    "REVISION": "TRUE",
    "UNKNOWN": "UNKNOWN",
}


class AuthoredRecursiveParseCourseError(RuntimeError):
    """LC-04 seed、递归 parse payload 或组合 split 不满足冻结合同。"""


def _exact_keys(value: Any, keys: set[str], *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise AuthoredRecursiveParseCourseError(f"{where} 字段集合非法")
    return value


def _text(value: Any, *, where: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or value.strip() != value:
        raise AuthoredRecursiveParseCourseError(f"{where} 必须是规范文本")
    if not allow_empty and not value:
        raise AuthoredRecursiveParseCourseError(f"{where} 不能为空")
    return value


def _integer(value: Any, *, where: str, positive: bool = False) -> int:
    if type(value) is not int or value < int(positive):
        qualifier = "正" if positive else "非负"
        raise AuthoredRecursiveParseCourseError(
            f"{where} 必须是{qualifier}严格整数")
    return value


def _flag(value: Any, *, where: str) -> int:
    if type(value) is not int or value not in {0, 1}:
        raise AuthoredRecursiveParseCourseError(f"{where} 必须是 0/1")
    return value


def _string_tuple(
        value: Any,
        *,
        where: str,
        sorted_unique: bool = False,
        allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise AuthoredRecursiveParseCourseError(f"{where} 必须是文本数组")
    result = tuple(_text(item, where=where) for item in value)
    if len(result) != len(set(result)):
        raise AuthoredRecursiveParseCourseError(f"{where} 不得重复")
    if sorted_unique and tuple(sorted(result)) != result:
        raise AuthoredRecursiveParseCourseError(f"{where} 必须排序")
    return result


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_expected(
        payload: CanonicalJsonObject,
        *,
        expected_state: str) -> None:
    raw = _exact_keys(payload.to_value(), _EXPECTED_FIELDS, where="expected")
    accepted = _flag(raw["accepted"], where="expected accepted")
    surfaces = _string_tuple(
        raw["accepted_surfaces"], where="accepted surfaces", allow_empty=True)
    _text(raw["analysis_key"], where="analysis_key")
    _text(raw["reason_code"], where="reason_code")
    if expected_state == "TRUE" and (accepted != 1 or not surfaces):
        raise AuthoredRecursiveParseCourseError("TRUE label 必须给出私有合法表层")
    if expected_state != "TRUE" and (accepted != 0 or surfaces):
        raise AuthoredRecursiveParseCourseError("非 TRUE label 不得给出采用表层")


@dataclass(frozen=True)
class RecursiveParsePayloadAudit:
    """返回 parse 数、最大深度、非连续数、组合键和单树基线。"""

    parse_candidate_count: int
    maximum_depth: int
    discontinuous_node_count: int
    combination_key: str
    single_tree_only: int


def _validate_members(
        value: Any,
        *,
        text_length: int,
        where: str) -> int:
    if not isinstance(value, list) or not value:
        raise AuthoredRecursiveParseCourseError(f"{where} members 不能为空")
    previous_end = -1
    zero_width = 0
    for item in value:
        if not isinstance(item, list) or len(item) != 2:
            raise AuthoredRecursiveParseCourseError(
                f"{where} member 必须是二元区间")
        start = _integer(item[0], where=f"{where} start")
        end = _integer(item[1], where=f"{where} end")
        if not (start <= end <= text_length):
            raise AuthoredRecursiveParseCourseError(f"{where} member 越界")
        if start == end:
            zero_width += 1
        if previous_end >= 0 and start <= previous_end:
            raise AuthoredRecursiveParseCourseError(
                f"{where} members 必须有序、不相邻、不重叠")
        previous_end = end
    if zero_width and len(value) != 1:
        raise AuthoredRecursiveParseCourseError("零宽 constituent 不得混合成员")
    return int(len(value) > 1)


def _parse_depth(nodes: dict[str, dict[str, Any]], node_id: str) -> int:
    seen: set[str] = set()
    depth = 0
    current = node_id
    while current:
        if current in seen:
            raise AuthoredRecursiveParseCourseError("parse parent 出现环")
        seen.add(current)
        depth += 1
        parent = nodes[current]["parent_node_id"]
        if parent and parent not in nodes:
            raise AuthoredRecursiveParseCourseError("parse parent 引用未知 node")
        current = parent
    return depth


def validate_recursive_parse_payload(
        payload: CanonicalJsonObject | dict[str, Any],
        ) -> RecursiveParsePayloadAudit:
    """校验递归 constituent、联合候选、预算、reparse 与线性化输入。"""
    value = payload.to_value() if isinstance(payload, CanonicalJsonObject) else payload
    raw = _exact_keys(value, set(PAYLOAD_KEYS), where="recursive parse payload")
    candidate_kind = _text(raw["candidate_kind"], where="candidate_kind")
    if candidate_kind not in CANDIDATE_KINDS:
        raise AuthoredRecursiveParseCourseError("candidate_kind 未登记")
    baseline = _text(raw["baseline_kind"], where="baseline_kind")
    if baseline not in BASELINE_KINDS:
        raise AuthoredRecursiveParseCourseError("baseline_kind 未登记")
    if raw["selection_state"] != "UNSELECTED":
        raise AuthoredRecursiveParseCourseError("Observation 不得预选 parse tree")
    if raw["sample_family"] not in SAMPLE_FAMILIES:
        raise AuthoredRecursiveParseCourseError("sample_family 未登记")
    _text(raw["proposition_group"], where="proposition_group")

    scope = _exact_keys(raw["surface_scope"], _SCOPE_FIELDS, where="scope")
    if scope["language"] != "zh" or scope["script"] != "HAN":
        raise AuthoredRecursiveParseCourseError("LC-04 首阶段只冻结 zh/HAN")
    if scope["register"] not in REGISTER_KEYS:
        raise AuthoredRecursiveParseCourseError("register 未登记")
    _text(scope["genre"], where="genre")

    observed = _exact_keys(raw["observed_surface"], {
        "append_only", "sha256", "target_hidden", "text",
    }, where="observed surface")
    text = _text(observed["text"], where="observed text")
    if _flag(observed["append_only"], where="append_only") != 1:
        raise AuthoredRecursiveParseCourseError("observed surface 必须 append-only")
    if observed["sha256"] != _sha256_text(text):
        raise AuthoredRecursiveParseCourseError("observed surface SHA-256 漂移")
    target_hidden = _flag(observed["target_hidden"], where="target_hidden")

    budget = _exact_keys(raw["parse_budget"], _BUDGET_FIELDS, where="parse budget")
    maximum_candidates = _integer(
        budget["max_candidates"], where="max_candidates", positive=True)
    maximum_depth_budget = _integer(
        budget["max_depth"], where="max_depth", positive=True)
    maximum_units = _integer(
        budget["max_linearization_units"],
        where="max_linearization_units", positive=True)
    maximum_nodes = _integer(
        budget["max_nodes"], where="max_nodes", positive=True)

    candidates_value = raw["parse_candidates"]
    if not isinstance(candidates_value, list) or not candidates_value:
        raise AuthoredRecursiveParseCourseError("parse_candidates 不能为空")
    if len(candidates_value) > maximum_candidates:
        raise AuthoredRecursiveParseCourseError("parse candidate 超预算")
    candidate_ids: set[str] = set()
    parse_keys: set[str] = set()
    maximum_depth = 0
    discontinuous = 0
    total_nodes = 0
    has_optional = False
    has_repeat = False
    has_coordination = False
    for candidate_value in candidates_value:
        candidate = _exact_keys(
            candidate_value, _PARSE_FIELDS, where="parse candidate")
        candidate_id = _text(candidate["candidate_id"], where="candidate_id")
        parse_key = _text(candidate["parse_key"], where="parse_key")
        if candidate_id in candidate_ids or parse_key in parse_keys:
            raise AuthoredRecursiveParseCourseError("parse candidate 身份重复")
        candidate_ids.add(candidate_id)
        parse_keys.add(parse_key)
        _integer(candidate["parser_version"], where="parser_version", positive=True)
        _text(candidate["linearization_key"], where="linearization_key")
        _text(candidate["scope_key"], where="parse scope_key")
        _text(candidate["role_signature"], where="role_signature")
        nodes_value = candidate["nodes"]
        if not isinstance(nodes_value, list) or not nodes_value:
            raise AuthoredRecursiveParseCourseError("parse nodes 不能为空")
        total_nodes += len(nodes_value)
        nodes: dict[str, dict[str, Any]] = {}
        role_repeat: dict[str, set[int]] = {}
        roots: list[str] = []
        for node_value in nodes_value:
            node = _exact_keys(node_value, _NODE_FIELDS, where="parse node")
            node_id = _text(node["node_id"], where="node_id")
            if node_id in nodes:
                raise AuthoredRecursiveParseCourseError("node_id 重复")
            if node["node_kind"] not in NODE_KINDS:
                raise AuthoredRecursiveParseCourseError("node_kind 未登记")
            parent = _text(
                node["parent_node_id"], where="parent_node_id", allow_empty=True)
            optional = _flag(node["optional"], where="node optional")
            repeat = _integer(node["repeat_ordinal"], where="repeat ordinal")
            role = _text(node["role_key"], where="role_key")
            _text(node["scope_key"], where="node scope_key")
            discontinuous += _validate_members(
                node["members"], text_length=len(text), where="node")
            if parent == "":
                roots.append(node_id)
            has_optional = has_optional or bool(optional)
            role_repeat.setdefault(role, set()).add(repeat)
            nodes[node_id] = node
        root = _text(candidate["root_node_id"], where="root_node_id")
        if roots != [root] or root not in nodes:
            raise AuthoredRecursiveParseCourseError("parse 必须有唯一声明 root")
        for node_id, node in nodes.items():
            if node["parent_node_id"] == node_id:
                raise AuthoredRecursiveParseCourseError("parse parent 不得自环")
            maximum_depth = max(maximum_depth, _parse_depth(nodes, node_id))
        has_repeat = has_repeat or any(
            len(values) >= 2 and max(values) > 0 for values in role_repeat.values())

        edges_value = candidate["edges"]
        if not isinstance(edges_value, list):
            raise AuthoredRecursiveParseCourseError("parse edges 必须是数组")
        edge_orders: list[int] = []
        for edge_value in edges_value:
            edge = _exact_keys(edge_value, _EDGE_FIELDS, where="parse edge")
            if edge["edge_kind"] not in EDGE_KINDS:
                raise AuthoredRecursiveParseCourseError("edge_kind 未登记")
            source = _text(edge["from_node_id"], where="from_node_id")
            target = _text(edge["to_node_id"], where="to_node_id")
            if source not in nodes or target not in nodes or source == target:
                raise AuthoredRecursiveParseCourseError("parse edge 端点非法")
            edge_orders.append(_integer(
                edge["order_index"], where="edge order", positive=True))
            has_coordination = has_coordination or (
                edge["edge_kind"] == "COORDINATES")
            if edge["edge_kind"] == "DISCONTINUOUS":
                discontinuous += 1
        if edge_orders != list(range(1, len(edge_orders) + 1)):
            raise AuthoredRecursiveParseCourseError("parse edge order 必须连续递增")

    if total_nodes > maximum_nodes:
        raise AuthoredRecursiveParseCourseError("parse node 超预算")
    if maximum_depth > maximum_depth_budget:
        raise AuthoredRecursiveParseCourseError("parse recursion depth 超预算")
    if len(text) > maximum_units:
        raise AuthoredRecursiveParseCourseError("linearization unit 超预算")
    if candidate_kind == "AMBIGUOUS" and len(candidates_value) < 2:
        raise AuthoredRecursiveParseCourseError("歧义样本不得先选一棵树")
    if candidate_kind != "AMBIGUOUS" and len(candidates_value) != 1:
        raise AuthoredRecursiveParseCourseError("非歧义样本 parse 数非法")
    if candidate_kind == "OPTIONAL" and not has_optional:
        raise AuthoredRecursiveParseCourseError("可空构式缺 optional constituent")
    if candidate_kind == "REPEATED" and not has_repeat:
        raise AuthoredRecursiveParseCourseError("重复 token 未保留 occurrence identity")
    if candidate_kind == "COORDINATED" and not has_coordination:
        raise AuthoredRecursiveParseCourseError("协调结构缺 COORDINATES edge")
    if candidate_kind == "NESTED" and maximum_depth < 3:
        raise AuthoredRecursiveParseCourseError("嵌套 parse 深度不足")
    if candidate_kind == "DISCONTINUOUS_DEPENDENCY" and discontinuous == 0:
        raise AuthoredRecursiveParseCourseError("非连续依赖未保留")

    generation = _exact_keys(
        raw["generation_constraint"], _GENERATION_FIELDS,
        where="generation constraint")
    direction = _text(generation["direction"], where="generation direction")
    if direction not in {"BIDIRECTIONAL", "GENERATION", "UNDERSTANDING"}:
        raise AuthoredRecursiveParseCourseError("generation direction 未登记")
    inputs = _string_tuple(
        generation["input_candidate_ids"], where="input_candidate_ids")
    if any(item not in candidate_ids for item in inputs):
        raise AuthoredRecursiveParseCourseError("generation input 引用未知 parse")
    hidden = _flag(
        generation["output_surface_hidden"], where="output_surface_hidden")
    if hidden != target_hidden:
        raise AuthoredRecursiveParseCourseError("generation hidden 标志不一致")
    joint = _flag(
        generation["requires_joint_parse"], where="requires_joint_parse")
    preserve = _flag(
        generation["role_scope_preservation_required"],
        where="role_scope_preservation_required")
    single_tree = int(baseline == "PRESELECTED_TREE_ONLY")
    if single_tree != int(candidate_kind == "PRESELECTED_TREE"):
        raise AuthoredRecursiveParseCourseError("single-tree baseline 身份不一致")
    if joint != int(not single_tree):
        raise AuthoredRecursiveParseCourseError("joint parse requirement 漂移")
    if preserve != 1:
        raise AuthoredRecursiveParseCourseError("线性化必须保持 Role/scope")
    if candidate_kind == "GENERATION":
        if direction != "GENERATION" or hidden != 1:
            raise AuthoredRecursiveParseCourseError("反向线性化必须隐藏目标")
    elif hidden:
        raise AuthoredRecursiveParseCourseError("非生成样本不得隐藏目标")

    split = _exact_keys(raw["split_identity"], {
        "combination_key", "depth_family", "filler_family", "isolation_axis",
        "parse_family",
    }, where="split identity")
    filler = _text(split["filler_family"], where="filler_family")
    depth_family = _text(split["depth_family"], where="depth_family")
    _text(split["parse_family"], where="parse_family")
    combination = _text(split["combination_key"], where="combination_key")
    if combination != f"{filler}::{depth_family}":
        raise AuthoredRecursiveParseCourseError("filler×parse-depth 组合键漂移")
    if split["isolation_axis"] != "FILLER_X_PARSE_DEPTH":
        raise AuthoredRecursiveParseCourseError("组合 held-out 轴未冻结")

    objectives = tuple(raw["objective_keys"])
    if objectives != tuple(sorted(set(objectives))):
        raise AuthoredRecursiveParseCourseError("objective_keys 必须排序去重")
    if not objectives or any(
            item not in LANGUAGE_OBJECTIVE_KEYS for item in objectives):
        raise AuthoredRecursiveParseCourseError("objective_keys 未登记")
    anchor = _text(
        raw["retention_anchor_id"], where="retention_anchor_id",
        allow_empty=True)
    if raw["sample_family"] == "RETENTION" and not anchor:
        raise AuthoredRecursiveParseCourseError("retention 必须绑定旧 anchor")
    if raw["sample_family"] != "RETENTION" and anchor:
        raise AuthoredRecursiveParseCourseError("非 retention 不得绑定 anchor")
    return RecursiveParsePayloadAudit(
        len(candidates_value), maximum_depth, discontinuous,
        combination, single_tree)


@dataclass(frozen=True)
class AuthoredRecursiveParseSeed:
    """一个保留联合 parse 候选、递归 span 和 reparse lineage 的 seed。"""

    seed_id: str
    family: str
    template_family: str
    label_owner: str
    split: str
    sample_family: str
    sample_role: str
    candidate_kind: str
    observed_text: str
    surface_scope: CanonicalJsonObject
    parse_budget: CanonicalJsonObject
    parse_candidates: tuple[CanonicalJsonObject, ...]
    generation_constraint: CanonicalJsonObject
    filler_family: str
    parse_family: str
    depth_family: str
    proposition_group: str
    baseline_kind: str
    objective_keys: tuple[str, ...]
    expected_state: str
    expected_payload: CanonicalJsonObject
    evaluation_dimension: str
    perturbation_kind: str
    supersedes_seed_id: str
    retention_anchor_id: str
    logical_order: int
    license_id: str

    def __post_init__(self) -> None:
        for name, value in (
                ("seed_id", self.seed_id), ("family", self.family),
                ("template_family", self.template_family),
                ("filler_family", self.filler_family),
                ("parse_family", self.parse_family),
                ("depth_family", self.depth_family),
                ("proposition_group", self.proposition_group),
                ("perturbation_kind", self.perturbation_kind)):
            _text(value, where=name)
        if self.label_owner not in {"teacher", "evaluator"}:
            raise AuthoredRecursiveParseCourseError("label_owner 非法")
        if self.split != ("train" if self.label_owner == "teacher" else "held_out"):
            raise AuthoredRecursiveParseCourseError("label_owner 与 split 不一致")
        if self.sample_family not in SAMPLE_FAMILIES:
            raise AuthoredRecursiveParseCourseError("sample_family 未登记")
        if self.sample_role not in SAMPLE_ROLES:
            raise AuthoredRecursiveParseCourseError("sample_role 未登记")
        if self.candidate_kind not in CANDIDATE_KINDS:
            raise AuthoredRecursiveParseCourseError("candidate_kind 未登记")
        _text(self.observed_text, where="observed_text")
        for name in ("surface_scope", "parse_budget", "generation_constraint"):
            if not isinstance(getattr(self, name), CanonicalJsonObject):
                raise AuthoredRecursiveParseCourseError(f"{name} 类型非法")
        if (not isinstance(self.parse_candidates, tuple)
                or not self.parse_candidates
                or any(not isinstance(item, CanonicalJsonObject)
                       for item in self.parse_candidates)):
            raise AuthoredRecursiveParseCourseError("parse_candidates 类型非法")
        if self.baseline_kind not in BASELINE_KINDS:
            raise AuthoredRecursiveParseCourseError("baseline_kind 未登记")
        if self.objective_keys != tuple(sorted(set(self.objective_keys))):
            raise AuthoredRecursiveParseCourseError("objective_keys 必须排序去重")
        if not self.objective_keys or any(
                item not in LANGUAGE_OBJECTIVE_KEYS for item in self.objective_keys):
            raise AuthoredRecursiveParseCourseError("objective_keys 未登记")
        if self.expected_state not in EXPECTED_STATES:
            raise AuthoredRecursiveParseCourseError("expected_state 非四态")
        if not isinstance(self.expected_payload, CanonicalJsonObject):
            raise AuthoredRecursiveParseCourseError("expected_payload 类型非法")
        _validate_expected(self.expected_payload, expected_state=self.expected_state)
        if self.evaluation_dimension not in EVALUATOR_DIMENSIONS:
            raise AuthoredRecursiveParseCourseError("evaluation_dimension 未登记")
        _text(self.supersedes_seed_id, where="supersedes_seed_id", allow_empty=True)
        _text(self.retention_anchor_id, where="retention_anchor_id", allow_empty=True)
        if type(self.logical_order) is not int or self.logical_order <= 0:
            raise AuthoredRecursiveParseCourseError("logical_order 必须为正严格整数")
        if self.license_id != LICENSE_ID:
            raise AuthoredRecursiveParseCourseError("原创递归 parse 课程必须为 CC0-1.0")
        validate_recursive_parse_payload(self.observation_payload())

    def observation_payload(self) -> CanonicalJsonObject:
        generation = self.generation_constraint.to_value()
        return CanonicalJsonObject.from_value({
            "baseline_kind": self.baseline_kind,
            "candidate_kind": self.candidate_kind,
            "generation_constraint": generation,
            "objective_keys": list(self.objective_keys),
            "observed_surface": {
                "append_only": 1,
                "sha256": _sha256_text(self.observed_text),
                "target_hidden": generation["output_surface_hidden"],
                "text": self.observed_text,
            },
            "parse_budget": self.parse_budget.to_value(),
            "parse_candidates": [item.to_value() for item in self.parse_candidates],
            "proposition_group": self.proposition_group,
            "retention_anchor_id": self.retention_anchor_id,
            "sample_family": self.sample_family,
            "selection_state": "UNSELECTED",
            "split_identity": {
                "combination_key": f"{self.filler_family}::{self.depth_family}",
                "depth_family": self.depth_family,
                "filler_family": self.filler_family,
                "isolation_axis": "FILLER_X_PARSE_DEPTH",
                "parse_family": self.parse_family,
            },
            "surface_scope": self.surface_scope.to_value(),
        })

    def compiled_seed(self) -> AuthoredCompiledSeed:
        payload = self.observation_payload()
        audit = validate_recursive_parse_payload(payload)
        parse_ids = tuple(
            item.to_value()["candidate_id"] for item in self.parse_candidates)
        return AuthoredCompiledSeed(
            self.seed_id,
            self.family,
            self.template_family,
            self.label_owner,
            self.split,
            self.sample_role,
            PAYLOAD_KIND,
            payload,
            self.expected_state,
            self.expected_payload,
            self.perturbation_kind,
            self.supersedes_seed_id,
            self.logical_order,
            (self.seed_id, parse_ids, audit.combination_key),
            (self.observed_text, self.proposition_group),
            (
                self.candidate_kind, self.parse_family, self.depth_family,
                audit.parse_candidate_count, audit.maximum_depth,
                audit.discontinuous_node_count,
            ),
            self.evaluation_dimension,
        )


def _seed_from_value(value: dict[str, Any]) -> AuthoredRecursiveParseSeed:
    raw = _exact_keys(value, _SEED_FIELDS, where="recursive parse seed")
    candidates = raw["parse_candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise AuthoredRecursiveParseCourseError("parse_candidates 不能为空")
    expected = raw["expected_payload"]
    if not isinstance(expected, dict):
        raise AuthoredRecursiveParseCourseError("expected_payload 必须是对象")
    return AuthoredRecursiveParseSeed(
        _text(raw["seed_id"], where="seed_id"),
        _text(raw["family"], where="family"),
        _text(raw["template_family"], where="template_family"),
        _text(raw["label_owner"], where="label_owner"),
        _text(raw["split"], where="split"),
        _text(raw["sample_family"], where="sample_family"),
        _text(raw["sample_role"], where="sample_role"),
        _text(raw["candidate_kind"], where="candidate_kind"),
        _text(raw["observed_text"], where="observed_text"),
        CanonicalJsonObject.from_value(_exact_keys(
            raw["surface_scope"], _SCOPE_FIELDS, where="surface_scope")),
        CanonicalJsonObject.from_value(_exact_keys(
            raw["parse_budget"], _BUDGET_FIELDS, where="parse_budget")),
        tuple(CanonicalJsonObject.from_value(_exact_keys(
            item, _PARSE_FIELDS, where="parse candidate")) for item in candidates),
        CanonicalJsonObject.from_value(_exact_keys(
            raw["generation_constraint"], _GENERATION_FIELDS,
            where="generation_constraint")),
        _text(raw["filler_family"], where="filler_family"),
        _text(raw["parse_family"], where="parse_family"),
        _text(raw["depth_family"], where="depth_family"),
        _text(raw["proposition_group"], where="proposition_group"),
        _text(raw["baseline_kind"], where="baseline_kind"),
        _string_tuple(
            raw["objective_keys"], where="objective_keys", sorted_unique=True),
        _text(raw["expected_state"], where="expected_state"),
        CanonicalJsonObject.from_value(_exact_keys(
            expected, _EXPECTED_FIELDS, where="expected_payload")),
        _text(raw["evaluation_dimension"], where="evaluation_dimension"),
        _text(raw["perturbation_kind"], where="perturbation_kind"),
        _text(raw["supersedes_seed_id"], where="supersedes_seed_id", allow_empty=True),
        _text(raw["retention_anchor_id"], where="retention_anchor_id", allow_empty=True),
        _integer(raw["logical_order"], where="logical_order", positive=True),
        _text(raw["license_id"], where="license_id"),
    )


def read_authored_recursive_parse_seeds(
        path: str | Path) -> tuple[AuthoredRecursiveParseSeed, ...]:
    """严格读取 JSONL，并核双 owner、联合候选、reparse 和组合 held-out。"""
    try:
        payload = Path(path).read_bytes()
    except OSError as error:
        raise AuthoredRecursiveParseCourseError("recursive parse sample 无法读取") from error
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise AuthoredRecursiveParseCourseError("recursive parse sample 换行非法")
    seeds: list[AuthoredRecursiveParseSeed] = []
    try:
        for line in payload.splitlines(keepends=True):
            value = parse_canonical_json_bytes(line[:-1], require_object=True)
            assert isinstance(value, dict)
            if canonical_json_line(value) != line:
                raise AuthoredRecursiveParseCourseError("recursive parse seed 非规范 JSON")
            seeds.append(_seed_from_value(value))
    except AuthoredRecursiveParseCourseError:
        raise
    except Exception as error:
        raise AuthoredRecursiveParseCourseError("recursive parse seed 损坏") from error
    identifiers = [item.seed_id for item in seeds]
    orders = [item.logical_order for item in seeds]
    if not seeds or len(identifiers) != len(set(identifiers)):
        raise AuthoredRecursiveParseCourseError("seed_id 空或重复")
    if orders != sorted(orders) or len(orders) != len(set(orders)):
        raise AuthoredRecursiveParseCourseError("logical_order 必须严格递增")
    index = {item.seed_id: item for item in seeds}
    for seed in seeds:
        expected_role = _ROLE_BY_FAMILY.get(seed.sample_family)
        if expected_role is not None and seed.sample_role != expected_role:
            raise AuthoredRecursiveParseCourseError("sample family/role 不一致")
        if seed.sample_family == "GENERATION" and seed.sample_role not in {
                "support", "read_only_probe", "refute"}:
            raise AuthoredRecursiveParseCourseError("generation sample_role 非法")
        expected_state = _STATE_BY_FAMILY.get(seed.sample_family)
        if expected_state is not None and seed.expected_state != expected_state:
            raise AuthoredRecursiveParseCourseError("sample family/state 不一致")
        if seed.sample_family == "GENERATION" and seed.expected_state not in {
                "TRUE", "FALSE"}:
            raise AuthoredRecursiveParseCourseError("generation state 非法")
        if seed.sample_family == "REVISION":
            target = index.get(seed.supersedes_seed_id)
            if (target is None or target.logical_order >= seed.logical_order
                    or target.family != seed.family or target.split != seed.split):
                raise AuthoredRecursiveParseCourseError(
                    "revision 必须 supersede 同族更早 seed")
            target_versions = {
                item.to_value()["parser_version"] for item in target.parse_candidates
            }
            new_versions = {
                item.to_value()["parser_version"] for item in seed.parse_candidates
            }
            if min(new_versions) <= max(target_versions):
                raise AuthoredRecursiveParseCourseError(
                    "local reparse parser version 必须严格前进")
        elif seed.supersedes_seed_id:
            raise AuthoredRecursiveParseCourseError("非 revision 不得 supersede")
        if seed.sample_family == "RETENTION":
            anchor = index.get(seed.retention_anchor_id)
            if (anchor is None or anchor.logical_order >= seed.logical_order
                    or anchor.split != seed.split):
                raise AuthoredRecursiveParseCourseError(
                    "retention anchor 必须是同 split 更早 seed")
        elif seed.retention_anchor_id:
            raise AuthoredRecursiveParseCourseError("非 retention 不得绑定 anchor")

    teacher = tuple(item for item in seeds if item.label_owner == "teacher")
    evaluator = tuple(item for item in seeds if item.label_owner == "evaluator")
    for owner, owner_seeds in (("teacher", teacher), ("evaluator", evaluator)):
        if {item.sample_family for item in owner_seeds} != set(SAMPLE_FAMILIES):
            raise AuthoredRecursiveParseCourseError(f"{owner} 未覆盖七类 sample family")
        if {item.candidate_kind for item in owner_seeds} != set(CANDIDATE_KINDS):
            raise AuthoredRecursiveParseCourseError(f"{owner} 未覆盖完整 parse 候选族")
        baseline = tuple(item for item in owner_seeds
                         if item.baseline_kind == "PRESELECTED_TREE_ONLY")
        if len(baseline) != 1 or baseline[0].expected_state != "FALSE":
            raise AuthoredRecursiveParseCourseError(
                f"{owner} preselected-tree 负基线未冻结")
        ambiguous = tuple(item for item in owner_seeds
                          if item.candidate_kind == "AMBIGUOUS")
        if len(ambiguous) != 1 or len(ambiguous[0].parse_candidates) < 2:
            raise AuthoredRecursiveParseCourseError(
                f"{owner} 联合 parse competition 缺失")
    if ({item.family for item in teacher} & {item.family for item in evaluator}
            or {item.template_family for item in teacher}
            & {item.template_family for item in evaluator}):
        raise AuthoredRecursiveParseCourseError("teacher/evaluator family/template 泄漏")

    teacher_fillers = {item.filler_family for item in teacher}
    teacher_depths = {item.depth_family for item in teacher}
    teacher_pairs = {(item.filler_family, item.depth_family) for item in teacher}
    held_out = {
        (item.filler_family, item.depth_family) for item in evaluator
        if item.filler_family in teacher_fillers
        and item.depth_family in teacher_depths
        and (item.filler_family, item.depth_family) not in teacher_pairs
    }
    if len(held_out) < 2:
        raise AuthoredRecursiveParseCourseError(
            "held-out filler×parse-depth 组合不足")
    if {item.evaluation_dimension for item in evaluator} != set(
            EVALUATOR_DIMENSIONS):
        raise AuthoredRecursiveParseCourseError("独立 evaluator 维度未列全")
    return tuple(seeds)


def _course_spec() -> AuthoredCourseSpec:
    return AuthoredCourseSpec(
        SOURCE_KEY,
        LICENSE_ID,
        COURSE_VERSION,
        ARTIFACT_VERSION,
        ADAPTER_VERSION,
        GENERATOR_VERSION,
        PARSER_VERSION,
        PACK_NAME,
        STAGE,
        SUBSTAGE,
        "authored-recursive-parse-seed-v1",
        "urn:pure-integer-ai:ph2:authored-recursive-parse-v1",
        "Pure Integer AI PH2 authored recursive parse seed",
        "RECURSIVE_PARSE_LABEL",
        "lc04-recursive-parse",
        320,
    )


def compile_authored_recursive_parse_course(
        sample_path: str | Path,
        release_root: str | Path) -> AuthoredCourseBuild:
    seeds = read_authored_recursive_parse_seeds(sample_path)
    return publish_authored_course(
        tuple(item.compiled_seed() for item in seeds),
        sample_path,
        release_root,
        _course_spec(),
    )


def build_recursive_parse_course_manifest(
        sample_path: str | Path,
        build: AuthoredCourseBuild,
        *,
        artifact_relative_root: str = FORMAL_ARTIFACT_RELATIVE_ROOT,
        ) -> CapabilityCourseManifest:
    sample = Path(sample_path)
    seeds = read_authored_recursive_parse_seeds(sample)
    teacher = tuple(item for item in seeds if item.label_owner == "teacher")
    evaluator = tuple(item for item in seeds if item.label_owner == "evaluator")
    return CapabilityCourseManifest(
        1,
        COURSE_MANIFEST_ARTIFACT_VERSION,
        "COURSE_FROZEN",
        "NOT_STARTED",
        ("LC-04",),
        ("RECURSIVE_PARSE",),
        STAGE,
        SUBSTAGE,
        SOURCE_KEY,
        LICENSE_ID,
        f"data/ph2/{sample.name}",
        hashlib.sha256(sample.read_bytes()).hexdigest(),
        len(seeds),
        SAMPLE_FAMILIES,
        COURSE_SPLIT_AXES,
        tuple(sorted({item.family for item in teacher})),
        tuple(sorted({item.family for item in evaluator})),
        tuple(sorted({item.template_family for item in teacher})),
        tuple(sorted({item.template_family for item in evaluator})),
        PAYLOAD_KIND,
        PAYLOAD_KEYS,
        tuple(sorted({key for item in seeds for key in item.objective_keys})),
        EVALUATOR_DIMENSIONS,
        VERIFIER_NE_CONDITIONS,
        RETENTION_PROTOCOLS,
        COMBINATION_AXES,
        BASELINE_KINDS,
        ABLATION_KEYS,
        f"{artifact_relative_root}/packs/{PACK_NAME}/manifest.json",
        build.manifest.sha256(),
        build.manifest.record_count,
        build.manifest.splits,
        CanonicalJsonObject.from_value(COURSE_INVARIANTS),
        CanonicalJsonObject.from_value(COURSE_EXECUTION_STATE),
    )


__all__ = [
    "COURSE_MANIFEST_PATH",
    "EVALUATOR_DIMENSIONS",
    "FORMAL_ARTIFACT_RELATIVE_ROOT",
    "PACK_NAME",
    "PAYLOAD_KIND",
    "AuthoredRecursiveParseCourseError",
    "AuthoredRecursiveParseSeed",
    "RecursiveParsePayloadAudit",
    "build_recursive_parse_course_manifest",
    "compile_authored_recursive_parse_course",
    "read_authored_recursive_parse_seeds",
    "validate_recursive_parse_payload",
]
