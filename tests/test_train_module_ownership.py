"""B-03 训练模块职责和反向依赖防回退测试。"""
from __future__ import annotations

import ast
import importlib
import inspect

from pure_integer_ai.experiments.train_module_ownership import (
    TRAIN_MODULE_OWNERSHIP,
    ownership_for,
)


def test_train_module_ownership_is_unique_and_importable():
    modules = [item.module for item in TRAIN_MODULE_OWNERSHIP]
    assert len(modules) == len(set(modules))
    for module in modules:
        imported = importlib.import_module(f"pure_integer_ai.{module}")
        assert imported is not None
        assert ownership_for(module).module == module


def test_independent_train_modules_do_not_import_formal_train():
    for record in TRAIN_MODULE_OWNERSHIP:
        if record.may_import_runner:
            continue
        module = importlib.import_module(f"pure_integer_ai.{record.module}")
        source = inspect.getsource(module)
        assert "experiments.formal_train" not in source


def test_formal_train_reexports_compatibility_interfaces():
    formal_train = importlib.import_module(
        "pure_integer_ai.experiments.formal_train")
    round_runtime = importlib.import_module(
        "pure_integer_ai.experiments.round_runtime")
    task_generation = importlib.import_module(
        "pure_integer_ai.experiments.task_generation_runtime")
    arithmetic_structure = importlib.import_module(
        "pure_integer_ai.experiments.arithmetic_structure_runtime")
    evaluation_runtime = importlib.import_module(
        "pure_integer_ai.experiments.evaluation_runtime")
    preflight_runtime = importlib.import_module(
        "pure_integer_ai.experiments.preflight_runtime")
    stage_learning = importlib.import_module(
        "pure_integer_ai.experiments.stage_learning_runtime")
    train_diagnostics = importlib.import_module(
        "pure_integer_ai.experiments.train_diagnostics")
    observation = importlib.import_module(
        "pure_integer_ai.experiments.language_observation")
    protocol_runtime = importlib.import_module(
        "pure_integer_ai.experiments.language_protocol_runtime")
    structure = importlib.import_module(
        "pure_integer_ai.experiments.language_structure_runtime")
    context = importlib.import_module(
        "pure_integer_ai.experiments.train_context")

    assert formal_train.TrainContext is context.TrainContext
    assert formal_train.make_train_context is context.make_train_context
    assert formal_train.RoundRunner is round_runtime.RoundRunner
    assert formal_train.RoundResult is round_runtime.RoundResult
    assert formal_train.DefaultRoundRunner is round_runtime.DefaultRoundRunner
    assert formal_train.GenerateSummary is task_generation.GenerateSummary
    assert (
        formal_train._run_task_driven_generate
        is task_generation._run_task_driven_generate
    )
    assert (
        formal_train._discover_and_recognize_arith_operators
        is arithmetic_structure._discover_and_recognize_arith_operators
    )
    assert (
        formal_train._verify_generalization
        is arithmetic_structure._verify_generalization
    )
    assert formal_train.CalibrationSample is evaluation_runtime.CalibrationSample
    assert formal_train._h2_calibrate is evaluation_runtime._h2_calibrate
    assert (
        formal_train._measure_floor_pass
        is evaluation_runtime._measure_floor_pass
    )
    assert (
        formal_train._run_simulated_offline_eval
        is evaluation_runtime._run_simulated_offline_eval
    )
    assert formal_train.pre_flight is preflight_runtime.pre_flight
    assert formal_train.PreFlightReport is preflight_runtime.PreFlightReport
    assert formal_train._inject_base_freq is stage_learning._inject_base_freq
    assert formal_train._promote_eligible is stage_learning._promote_eligible
    assert (
        formal_train._anti_collapse_summary
        is train_diagnostics._anti_collapse_summary
    )
    assert formal_train._rebuild_path is round_runtime._rebuild_path
    assert formal_train._build_space_ctx is round_runtime._build_space_ctx
    assert (
        formal_train._collect_action_seed_candidates
        is round_runtime._collect_action_seed_candidates
    )
    assert formal_train._split_item_to_segments is observation._split_item_to_segments
    assert (
        formal_train.install_language_graph_protocols
        is protocol_runtime.install_language_graph_protocols
    )
    assert (
        formal_train._discover_and_recognize_lang_structures
        is structure._discover_and_recognize_lang_structures
    )


def test_formal_train_does_not_redefine_round_runtime():
    formal_train = importlib.import_module(
        "pure_integer_ai.experiments.formal_train")
    tree = ast.parse(inspect.getsource(formal_train))
    forbidden = {
        "RoundRunner",
        "RoundResult",
        "DefaultRoundRunner",
        "_hotzone_dag_edges",
        "_reachable_dag_edges",
        "_rebuild_path",
        "_build_space_ctx",
        "_resolve_emergent_excluded_refs",
        "_run_emergence_hook",
        "_feed_action_experience",
        "_collect_action_seed_candidates",
        "GenerateSummary",
        "_run_task_driven_generate",
        "_discover_and_recognize_arith_operators",
        "_verify_generalization",
        "CalibrationSample",
        "_observe_eval_item",
        "_make_calib_judge_fn",
        "_run_calibration_phase",
        "_run_calibration_phase_impl",
        "_run_simulated_offline_eval",
        "_run_simulated_offline_eval_impl",
        "_held_out_discovery_tally_free",
        "_measure_floor_pass",
        "_measure_floor_pass_impl",
        "_h2_calibrate",
        "_h2_calibrate_impl",
        "PreFlightReport",
        "pre_flight",
        "_pre_flight_impl",
        "_graph_size",
        "_edge_count",
        "_anti_collapse_summary",
        "_weaning_blockers",
        "_causes_coverage",
        "_inject_base_freq",
        "_promote_eligible",
        "_run_round_batch",
    }
    definitions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }
    assert definitions.isdisjoint(forbidden)


def test_production_runners_do_not_assign_shared_runtime_module_attributes():
    module_names = (
        "pure_integer_ai.experiments.formal_train",
        "pure_integer_ai.experiments.capability_exam",
        "pure_integer_ai.experiments.run_weaning_train",
    )
    for module_name in module_names:
        module = importlib.import_module(module_name)
        tree = ast.parse(inspect.getsource(module))
        assignments = []
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            elif isinstance(node, ast.AugAssign):
                targets = [node.target]
            for target in targets:
                if (isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id in {"gates", "stages"}):
                    assignments.append((target.value.id, target.attr))
        assert assignments == [], (module_name, assignments)


def test_formal_train_uses_shared_production_gate_profile():
    """正式训练不得重新持有独立的生产 gate 配置真源。"""
    formal_train = importlib.import_module(
        "pure_integer_ai.experiments.formal_train")
    gate_profile = importlib.import_module(
        "pure_integer_ai.experiments.train_gate_profile")
    source = inspect.getsource(formal_train)

    assert "push_production_training_gates()" in source
    assert "reset_production_training_gates(training_gate_token)" in source
    assert '"SELECTION_PREF_MODE": True' not in source
    assert gate_profile.production_training_gate_overrides()[
        "SELECTION_PREF_MODE"
    ] is True
