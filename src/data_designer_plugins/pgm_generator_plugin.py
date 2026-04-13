# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Data Designer plugin for generating synthetic data via a PGMGenerator.

Any subclass of PGMGenerator can be used by passing its dotted module path as
``generator_class`` in the column config, e.g.::

    PGMGeneratorPluginConfig(
        name="person",
        generator_class="my_generators.MyPersonGenerator",
        evidence={"region": "West"},
    )

The generator is instantiated once and cached for the lifetime of the process.
"""

import importlib
import logging
from typing import Any, Literal

import pandas as pd
from data_designer.config.column_configs import SingleColumnConfig
from data_designer.engine.column_generators.generators.base import (
    ColumnGeneratorFullColumn,
    FromScratchColumnGenerator,
    GenerationStrategy,
)
from data_designer.plugins import Plugin, PluginType

from pgms.generators.base.pgm_generator import PGMGenerator

logger = logging.getLogger(__name__)

_generator_cache: dict[str, PGMGenerator] = {}


def _get_generator(generator_class: str) -> PGMGenerator:
    """Import, validate, instantiate, and cache a PGMGenerator subclass."""
    if generator_class not in _generator_cache:
        try:
            module_name, class_name = generator_class.rsplit(".", 1)
        except ValueError:
            raise ValueError(
                "generator_class must be a fully-qualified dotted path "
                "(e.g. 'mypackage.MyGenerator'), "
                f"got: {generator_class!r}"
            )
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        if not (isinstance(cls, type) and issubclass(cls, PGMGenerator)):
            raise TypeError(f"{generator_class!r} is not a subclass of PGMGenerator")
        logger.info("Initializing %s (this may take a moment)...", generator_class)
        _generator_cache[generator_class] = cls()
    return _generator_cache[generator_class]


class PGMGeneratorPluginConfig(SingleColumnConfig):
    """Configuration for the PGM Generator plugin.

    Each row will contain a dict of all fields produced by the generator.
    Use ExpressionColumnConfig to extract individual fields::

        ExpressionColumnConfig(
            name="full_name",
            expr="{{ person.first_name }} {{ person.last_name }}"
        )

    Attributes:
        generator_class: Fully-qualified dotted path to a PGMGenerator
            subclass, e.g. ``"my_generators.MyPersonGenerator"``.
        evidence: Optional dictionary for rejection sampling.  Keys are
            field names; values can be a single value or a list of allowed
            values. Example: ``{"region": "West"}``
        column_type: Plugin discriminator field (fixed: ``"pgm-generator"``).
    """

    generator_class: str
    evidence: dict[str, Any | list[Any]] | None = None
    column_type: Literal["pgm-generator"] = "pgm-generator"

    @property
    def required_columns(self) -> list[str]:
        return []

    @property
    def side_effect_columns(self) -> list[str]:
        return []


class PGMGeneratorPluginTask(
    FromScratchColumnGenerator[PGMGeneratorPluginConfig],
    ColumnGeneratorFullColumn[PGMGeneratorPluginConfig],
):
    """Column generator backed by any PGMGenerator subclass."""

    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        return GenerationStrategy.FULL_COLUMN

    def _generate_records(self, num_records: int) -> pd.DataFrame:
        generator = _get_generator(self.config.generator_class)
        evidence = self.config.evidence or {}
        logger.info(
            "Generating %d records with generator=%s, evidence=%s",
            num_records,
            self.config.generator_class,
            evidence,
        )
        generated_df = generator.generate_samples(size=num_records, evidence=evidence)
        records = [row.to_dict() for _, row in generated_df.iterrows()]
        return pd.DataFrame({self.config.name: records})

    def generate_from_scratch(self, num_records: int) -> pd.DataFrame:
        return self._generate_records(num_records)

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        generated_df = self._generate_records(len(data))
        data[self.config.name] = generated_df[self.config.name].values
        return data


plugin = Plugin(
    impl_qualified_name=(
        "data_designer_plugins.pgm_generator_plugin.PGMGeneratorPluginTask"
    ),
    config_qualified_name=(
        "data_designer_plugins.pgm_generator_plugin.PGMGeneratorPluginConfig"
    ),
    plugin_type=PluginType.COLUMN_GENERATOR,
)
