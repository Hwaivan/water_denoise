"""YAML configuration loading and validation helpers."""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML file into a plain dictionary."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError("Configuration file not found: {}".format(config_path))
    with config_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError("The top level of the configuration must be a mapping")
    return data


def deep_update(base: Mapping[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a recursive merge without mutating either input."""
    result = deepcopy(dict(base))
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def require_sections(config: Mapping[str, Any], *sections: str) -> None:
    """Raise a helpful error when a required configuration section is absent."""
    missing = [name for name in sections if name not in config]
    if missing:
        raise KeyError("Missing configuration sections: {}".format(", ".join(missing)))

