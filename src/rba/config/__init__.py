from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

PROJ_ROOT = Path(__file__).resolve().parents[3]

# This package directory — home of the YAML experiment configs (features.yaml,
# targets.yaml, models.yaml). Runtime constants live in this module; YAML holds
# experiment/feature configuration (CONTEXT.md "Configs" convention).
CONFIG_DIR = Path(__file__).resolve().parent
FEATURES_CONFIG_PATH = CONFIG_DIR / "features.yaml"
MODELS_CONFIG_PATH = CONFIG_DIR / "models.yaml"

# Top-level ``models.yaml`` key that holds run-level defaults rather than a model.
_MODELS_DEFAULTS_KEY = "defaults"

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MLRUNS_DIR = PROJ_ROOT / "mlruns"
MLFLOW_TRACKING_URI = MLRUNS_DIR.as_uri()

RANDOM_SEED = 12

try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load a YAML config file into a plain dict.

    Parameters
    ----------
    path
        Path to a YAML document (e.g. :data:`FEATURES_CONFIG_PATH`).

    Returns
    -------
    dict
        The parsed top-level mapping.
    """
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a mapping at the top of {path}, got {type(loaded).__name__}.")
    return loaded


def load_features_config(path: Path = FEATURES_CONFIG_PATH) -> dict[str, Any]:
    """Load ``features.yaml`` — the feature-pipeline configuration.

    Thin wrapper over :func:`load_yaml_config` pinned to the project's feature
    config, so callers (feature builders) need not know the path.
    """
    return load_yaml_config(path)


def load_models_config(path: Path = MODELS_CONFIG_PATH) -> dict[str, Any]:
    """Load the full ``models.yaml`` mapping (every model entry + ``defaults``).

    Thin wrapper over :func:`load_yaml_config` pinned to the project's model
    config. Callers usually want a single entry — see :func:`load_model_config`.
    """
    return load_yaml_config(path)


def load_model_config(name: str, path: Path = MODELS_CONFIG_PATH) -> dict[str, Any]:
    """Load a single named model entry from ``models.yaml``.

    Returns the entry as a shallow copy with its ``name`` injected (handy for
    logging / MLflow run naming), so it can be passed straight to
    :func:`rba.models.build_model`.

    Parameters
    ----------
    name
        A model key in ``models.yaml`` (e.g. ``"majority_class"``). The run-level
        ``defaults`` section is not a model and cannot be requested.
    path
        Override for the config location (injectable for tests).

    Returns
    -------
    dict
        The model entry — ``module`` / ``class`` / ``task`` / ``default`` /
        ``search_space`` (as present) plus an injected ``name``.

    Raises
    ------
    KeyError
        If ``name`` is absent from the config (or is the ``defaults`` section). The
        message lists the available model names.
    """
    config = load_models_config(path)
    if name == _MODELS_DEFAULTS_KEY or name not in config:
        available = sorted(k for k in config if k != _MODELS_DEFAULTS_KEY)
        raise KeyError(f"No model named {name!r} in {path}. Available: {available}.")
    entry = dict(config[name])
    entry.setdefault("name", name)
    return entry
