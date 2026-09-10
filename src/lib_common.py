"""Shared configuration and architecture helpers; accelerator imports are lazy."""
import os
import re
import yaml
from pathlib import Path
from contextlib import contextmanager


class UniqueKeyLoader(yaml.SafeLoader):
    """Reject YAML keys that would silently overwrite a class or option."""


def _unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"Duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def read_yaml(file_path):
    with open(file_path, 'r') as f:
        return yaml.load(f, Loader=UniqueKeyLoader)


def _typed_option(name, raw, default):
    if isinstance(default, bool):  # bool is also an int: this must come first.
        if str(raw).strip().lower() in ('true', '1', 'yes', 'on'):
            return True
        if str(raw).strip().lower() in ('false', '0', 'no', 'off'):
            return False
        raise ValueError(f"{name} must be a boolean, got {raw!r}")
    if isinstance(default, int):
        if not re.fullmatch(r'[+-]?\d+', str(raw).strip()):
            raise ValueError(f"{name} must be an integer, got {raw!r}")
        return int(raw)
    if isinstance(default, float):
        import math
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value
    if isinstance(default, list):
        if not default or not all(type(v) is int for v in default):
            raise ValueError(f"{name}: unsupported list option type")
        return [_typed_option(name, item, 0) for item in str(raw).split(',')]
    if isinstance(default, str):
        return str(raw)
    raise ValueError(f"{name}: unsupported option type {type(default).__name__}")


def update_config_from_env(config, overrides=None):
    """Strictly parse declared options; explicit override mappings reject unknown keys.

    Normal process environments also include unrelated runtime variables, so only
    declared configuration names are consumed in that mode.
    """
    if not isinstance(config, dict):
        raise ValueError('Configuration must be a mapping')
    values = os.environ if overrides is None else overrides
    if overrides is not None and set(values) - set(config):
        raise ValueError(f"Unknown options: {sorted(set(values) - set(config))}")
    for name, default in config.items():
        if name in values:
            config[name] = _typed_option(name, values[name], default)
    return config


def canonical_model_name(model_name):
    name = str(model_name).upper()
    aliases = {'ENB0': 'EN0', 'ENB2': 'EN2', 'ENXL': 'ENX',
               'VITT': 'VTT', 'VITS': 'VTS', 'VITB': 'VTB', 'VITL': 'VTL'}
    name = aliases.get(name, name)
    if name not in _MODEL_SIZES:
        raise ValueError(f"Unknown model architecture: {model_name!r}")
    return name


_MODEL_SIZES = {'EN0': 224, 'EN2': 260, 'ENS': 384, 'ENM': 480, 'ENL': 480,
                'ENX': 512, 'CNP': 288, 'CNN': 288, 'CNT': 384, 'CNS': 384,
                'CNB': 384, 'CNL': 384, 'VTT': 384, 'VTS': 384, 'VTB': 384, 'VTL': 384}


def model_img_size_mapping(model_name):
    return _MODEL_SIZES[canonical_model_name(model_name)]


class NullStrategy:
    def scope(self):
        @contextmanager
        def null_scope():
            yield
        return null_scope()


def setup_strategy():
    from jax import devices
    from keras import distribution
    gpus = devices()
    if any('cuda' in str(device).lower() for device in gpus):
        strategy = distribution.DataParallel(devices=gpus)
        print(str(len(gpus)) + ' x GPU activated\n')
    else:
        strategy = NullStrategy()
        print('CPU-only training activated\n')
    return strategy


def get_mod(s: str) -> str:
    m = Path(s)
    return m.stem[:3] if m.suffix.lower() == '.keras' else m.stem
