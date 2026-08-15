"""Output contract for recovery-v8 family-LOSO learning."""

NORMALIZATION_RECOVERY_V8_OUTPUT_FILE_ROLES = (
    ("loso-training-views.jsonl", "FAMILY_LOSO_TRAINING_VIEWS", "view_id"),
    ("orthographic-rules.jsonl", "LOSO_ORTHOGRAPHIC_RULES", "rule_id"),
    ("source-conditioned-lexical-rules.jsonl",
     "LOSO_SOURCE_CONDITIONED_LEXICAL_RULES", "rule_id"),
    ("layout-morphology-obligations.jsonl",
     "LOSO_LAYOUT_MORPHOLOGY_OBLIGATIONS", "rule_id"),
    ("identity-veto-rules.jsonl", "LOSO_IDENTITY_VETO_RULES", "rule_id"),
)

__all__ = ["NORMALIZATION_RECOVERY_V8_OUTPUT_FILE_ROLES"]
