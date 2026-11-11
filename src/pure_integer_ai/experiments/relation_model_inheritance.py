"""无正文知识增量的父模型清单与原生语法继承，不复制训练来源或回放。"""
from pathlib import Path
import shutil

from pure_integer_ai.experiments.authored_relation_sqlite_course import file_sha256
from pure_integer_ai.experiments.relation_knowledge_checkpoint import model_source_body_count
from pure_integer_ai.experiments.trained_relation_surface_artifact import (
    CARRIER_LANGUAGE_ARTIFACT_SCHEMA_VERSION, read_relation_surface_artifact, validate_language_members,
)


REQUIRED_COMPONENTS = (
    'training.sqlite3', 'relation_surfaces.sqlite3', 'assertion_scopes.sqlite3',
    'composition.sqlite3', 'programs.sqlite3', 'relation_questions.sqlite3',
    'requests.sqlite3', 'revision_cues.sqlite3', 'sequence.sqlite3',
)
OPTIONAL_COMPONENTS = ('input_dialogue_act.sqlite3', 'outputs.sqlite3')


def parent_model_identity(parent):
    """冻结所有许可组件及V3父语言；未知侧库、正文和缺失组件均拒绝。"""
    parent = Path(parent).resolve(strict=True)
    if parent.drive.upper() != 'K:':
        raise ValueError('parent model must remain on K')
    files = {path.name: path for path in parent.iterdir() if path.is_file()}
    if (set(REQUIRED_COMPONENTS) - files.keys()
            or files.keys() - set((*REQUIRED_COMPONENTS, *OPTIONAL_COMPONENTS))
            or any(path.resolve().parent != parent for path in files.values())):
        raise ValueError('parent model component manifest is incomplete or unregistered')
    identities = {name: file_sha256(path).hex() for name, path in sorted(files.items())}
    if model_source_body_count(files['training.sqlite3']):
        raise ValueError('knowledge extension requires a source-body-free parent Core')
    facts, frames, metadata = read_relation_surface_artifact(files['relation_surfaces.sqlite3'],
        expected_core_sha256=identities['training.sqlite3'])
    if metadata['schema_version'] != CARRIER_LANGUAGE_ARTIFACT_SCHEMA_VERSION:
        raise ValueError('knowledge extension requires a complete V3 parent language')
    validate_language_members(facts, frames, require_carrier=True)
    return identities


def inherit_parent_model_sidecars(parent, run_root, identities):
    """只复制冻结清单内的侧组件，目标必须在本次显式K根内且旧文件不能漂移。"""
    parent, root = Path(parent).resolve(strict=True), Path(run_root).resolve(strict=True)
    model = (root/'model').resolve(strict=True)
    if (root.drive.upper() != 'K:' or root == Path(root.anchor)
            or not model.is_relative_to(root) or parent.is_relative_to(root)
            or root.is_relative_to(parent) or parent_model_identity(parent) != identities):
        raise ValueError('frozen independent parent or explicit K output differs')
    copied = {}
    for name, digest in sorted(identities.items()):
        if name in ('training.sqlite3', 'relation_surfaces.sqlite3'):
            continue
        source, target = parent/name, (model/name).resolve()
        if not target.is_relative_to(root) or source.stat().st_size > 16*1024**2:
            raise ValueError('inherited sidecar exceeds its explicit location or size budget')
        if target.exists() and file_sha256(target).hex() != digest:
            raise ValueError('existing inherited model component differs')
        if not target.exists():
            shutil.copyfile(source, target)
        if file_sha256(target).hex() != digest:
            raise ValueError('copied inherited model component differs')
        copied[name] = digest
    return copied


def validate_inherited_native_graphs(model, core, generator):
    """由现役消费者核对新Core中的全部父原生模板、序列和可选输出合同。"""
    from pure_integer_ai.experiments.operation_output_course import read_operation_output_graph
    from pure_integer_ai.experiments.trained_operation_composition import TrainedOperationCompositionRuntime
    from pure_integer_ai.experiments.trained_operation_sequence import TrainedOperationSequenceRuntime
    model = Path(model).resolve(strict=True)
    native = TrainedOperationCompositionRuntime(model/'composition.sqlite3', generator, core,
        model/'programs.sqlite3')
    sequence = TrainedOperationSequenceRuntime(model/'sequence.sqlite3', native)
    outputs = (read_operation_output_graph(model/'outputs.sqlite3', native.operation_witnesses)
        if (model/'outputs.sqlite3').is_file() else ())
    return dict(native_template_count=len(native.witnesses), sequence_rule_count=len(sequence.rules),
        inherited_output_contract_count=len(outputs))
