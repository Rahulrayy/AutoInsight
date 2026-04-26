import os
import yaml

_CONFIG = None
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")


def _load():
    global _CONFIG
    if _CONFIG is None:
        try:
            with open(_CONFIG_PATH, "r") as f:
                _CONFIG = yaml.safe_load(f) or {}
        except FileNotFoundError:
            _CONFIG = {}
    return _CONFIG


def get(section, key, default=None):
    return _load().get(section, {}).get(key, default)
