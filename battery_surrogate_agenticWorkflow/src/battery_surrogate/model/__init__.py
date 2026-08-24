"""Point-wise MLP model package for battery surrogate training.

Note: torch-dependent modules (mlp_pointwise, registry, evaluate) are imported lazily
to avoid slow startup times in WSL environments. Import directly when needed:

  from battery_surrogate.model.mlp_pointwise import PointwiseMLP
  from battery_surrogate.model.registry import build_model, build_datasets
  from battery_surrogate.model.evaluate import evaluate_on_ops, bc_v_spatial_variance

Fast-import modules (no torch):
  from battery_surrogate.model.normalizer import PointwiseNormalizer
  from battery_surrogate.model.split import resolve_split, validate_coverage
  from battery_surrogate.model.features_sequence import build_sequence_for_sensor
  from battery_surrogate.model.dataset_sequence import SequenceDataset
"""


def __getattr__(name):
    """Lazy load torch-dependent modules on first access."""

    imports = {
        "LearnableSwish": ("mlp_pointwise", "LearnableSwish"),
        "PointwiseMLP": ("mlp_pointwise", "PointwiseMLP"),
        "build_model": ("registry", "build_model"),
        "build_datasets": ("registry", "build_datasets"),
        "RecurrentPointwise": ("recurrent_pointwise", "RecurrentPointwise"),
        "train_sequence_model": ("trainer_sequence", "train_sequence_model"),
        "evaluate_on_ops": ("evaluate", "evaluate_on_ops"),
        "bc_v_spatial_variance": ("evaluate", "bc_v_spatial_variance"),
        "PointwiseNormalizer": ("normalizer", "PointwiseNormalizer"),
        "resolve_split": ("split", "resolve_split"),
        "validate_coverage": ("split", "validate_coverage"),
        "SequenceDataset": ("dataset_sequence", "SequenceDataset"),
        "build_sequence_for_sensor": ("features_sequence", "build_sequence_for_sensor"),
        "build_history_lags": ("features_sequence", "build_history_lags"),
        "evaluate_sequence_model": ("evaluate_sequence", "evaluate_sequence_model"),
        "history_length_benchmark": ("evaluate_sequence", "history_length_benchmark"),
        "benchmark_history_lengths": ("evaluate_sequence", "benchmark_history_lengths"),
        "plot_error_curves": ("evaluate_sequence", "plot_error_curves"),
    }

    if name in imports:
        module_name, attr_name = imports[name]
        module = __import__(f".{module_name}", fromlist=[attr_name], level=1)
        return getattr(module, attr_name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LearnableSwish",
    "PointwiseMLP",
    "PointwiseNormalizer",
    "resolve_split",
    "validate_coverage",
    "build_model",
    "build_datasets",
    "RecurrentPointwise",
    "train_sequence_model",
    "SequenceDataset",
    "build_sequence_for_sensor",
    "build_history_lags",
    "evaluate_on_ops",
    "bc_v_spatial_variance",
    "evaluate_sequence_model",
    "history_length_benchmark",
    "benchmark_history_lengths",
    "plot_error_curves",
]
