# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Shared helpers for the grouped-config constructor pattern used across SAM2.

Several SAM2 components collapse a long parameter list into an optional
dataclass ``config`` plus ``**kwargs``. They all need the same merge logic:
start from the provided config (or the dataclass defaults) and let keyword
arguments matching a config field win. :func:`resolve_config` centralizes that
logic so each call site does not carry its own near-duplicate helper.
"""

from dataclasses import fields, replace
from typing import Any, Dict, Optional, Type, TypeVar

ConfigT = TypeVar("ConfigT")


def config_field_names(config_cls: Type[Any]) -> tuple:
    """Return the field names declared on a config dataclass."""
    return tuple(f.name for f in fields(config_cls))


def resolve_config(
    config: Optional[ConfigT],
    config_cls: Type[ConfigT],
    overrides: Dict[str, Any],
) -> ConfigT:
    """Merge an optional config dataclass with keyword overrides.

    ``config`` (or the ``config_cls`` defaults when it is ``None``) provides the
    base values, and any ``overrides`` matching a config field take precedence.
    This preserves historical keyword-based call sites (e.g. Hydra
    instantiation) while collapsing the long parameter list into a single
    config object.
    """
    base = config if config is not None else config_cls()
    field_overrides = {
        k: overrides[k] for k in config_field_names(config_cls) if k in overrides
    }
    if not field_overrides:
        return base
    return replace(base, **field_overrides)
