"""Model and dataset registries for config-driven instantiation.

New datasets or models are registered here. Training and evaluation code
queries the registry by name instead of importing every class directly,
so scripts stay agnostic to which dataset source or model variant is active.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch.nn as nn


def _lazy_model_registry() -> dict[str, type]:
    """Return the model registry (imports deferred to avoid circular deps)."""
    from models.flow_matching.model import LatentFlowMatcher
    from models.vqvae.model import TransformerVQVAE

    return {
        "vqvae": TransformerVQVAE,
        "flow_matching": LatentFlowMatcher,
    }


def _lazy_dataset_registry() -> dict[str, Any]:
    """Return the dataset-source registry.

    Each entry is the `prepare_split` callable for that source (see
    data/generate_dataset.py), not a Dataset class -- the on-disk
    format is unified (.npz) after preparation, so a single
    NS2DDataset (data/datamodule.py) reads all three sources.
    """
    from data.pdebench_loader import prepare_pdebench_split
    from data.synthetic_solver import prepare_synthetic_split
    from data.the_well_loader import prepare_the_well_split

    return {
        "pdebench": prepare_pdebench_split,
        "the_well": prepare_the_well_split,
        "synthetic": prepare_synthetic_split,
    }


def build_model(name: str, **kwargs: Any) -> "nn.Module":
    """Instantiate a model by registry name.

    Args:
        name: Model name ('vqvae' or 'flow_matching').
        **kwargs: Constructor arguments forwarded to the model class.

    Returns:
        Instantiated nn.Module.

    Raises:
        KeyError: If name is not registered.
    """
    registry = _lazy_model_registry()
    if name not in registry:
        raise KeyError(f"Unknown model '{name}'. Available: {list(registry)}")
    return registry[name](**kwargs)


def get_dataset_preparer(source: str):
    """Return the split-preparation callable for a dataset source.

    Args:
        source: One of 'pdebench', 'the_well', 'synthetic'.

    Returns:
        Callable with signature matching data/generate_dataset.py's
        expectations (see prepare_synthetic_split for the reference
        signature).

    Raises:
        KeyError: If source is not registered.
    """
    registry = _lazy_dataset_registry()
    if source not in registry:
        raise KeyError(f"Unknown dataset source '{source}'. Available: {list(registry)}")
    return registry[source]


def list_models() -> list[str]:
    """Return sorted list of registered model names."""
    return sorted(_lazy_model_registry().keys())


def list_dataset_sources() -> list[str]:
    """Return sorted list of registered dataset sources."""
    return sorted(_lazy_dataset_registry().keys())
