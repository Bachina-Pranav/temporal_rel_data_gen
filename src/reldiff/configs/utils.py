from pathlib import Path
from typing import Any, cast

import tomli

RawConfig = dict[str, Any]

_CONFIG_NONE = "__none__"


def _replace(data, condition, value):
    def do(x):
        if isinstance(x, dict):
            return {k: do(v) for k, v in x.items()}
        elif isinstance(x, list):
            return [do(y) for y in x]
        else:
            return value if condition(x) else x

    return do(data)


def unpack_config(config: RawConfig) -> RawConfig:
    config = cast(RawConfig, _replace(config, lambda x: x == _CONFIG_NONE, None))
    return config


def pack_config(config: RawConfig) -> RawConfig:
    config = cast(RawConfig, _replace(config, lambda x: x is None, _CONFIG_NONE))
    return config


def load_config(path: Path | str) -> RawConfig:
    with open(path, "rb") as f:
        return unpack_config(tomli.load(f))


def load_dataset_config(path: Path | str) -> RawConfig:
    path = Path(path)
    defaults: RawConfig = {
        "is_disjoint": False,
        "order_cols": {},
        "dimension_tables": [],
        "n_hops_dataloader": None,
    }
    if not path.exists():
        print(f"Dataset config file {path} not found, using defaults")
        return dict(defaults)
    loaded = load_config(path)
    return {
        "is_disjoint": loaded.get("is_disjoint", False),
        "order_cols": loaded.get("order_cols") or {},
        "dimension_tables": loaded.get("dimension_tables") or [],
        "n_hops_dataloader": loaded.get("n_hops_dataloader"),
    }
