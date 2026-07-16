"""Registry for interaction benchmark adapters."""

from __future__ import annotations

from .base import InteractionDatasetAdapter
from .hm import HMAdapter
from .movielens import MovieLensAdapter
from .retailrocket import RetailRocketAdapter
from .yelp import YelpAdapter


_ADAPTERS: dict[str, type[InteractionDatasetAdapter]] = {
    "movielens": MovieLensAdapter,
    "movielens_100k": MovieLensAdapter,
    "yelp": YelpAdapter,
    "yelp_100k": YelpAdapter,
    "retailrocket": RetailRocketAdapter,
    "retailrocket_100k": RetailRocketAdapter,
    "hm": HMAdapter,
    "hm_100k": HMAdapter,
    "h&m": HMAdapter,
}


def get_adapter(name: str) -> InteractionDatasetAdapter:
    key = str(name).lower().replace("-", "_")
    if key not in _ADAPTERS:
        raise KeyError(f"Unknown interaction dataset {name!r}; available: {list_datasets()}")
    return _ADAPTERS[key]()


def list_datasets() -> list[str]:
    return sorted({"movielens", "yelp", "retailrocket", "hm"})
