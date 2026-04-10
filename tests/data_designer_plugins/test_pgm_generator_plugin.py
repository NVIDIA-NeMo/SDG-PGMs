# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the PGM Generator Data Designer plugin."""

from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from data_designer.engine.column_generators.generators.base import (
    GenerationStrategy,
)
from data_designer.plugins import PluginType

from data_designer_plugins.pgm_generator_plugin import (
    PGMGeneratorPluginConfig,
    PGMGeneratorPluginTask,
    _generator_cache,
    plugin,
)

_FAKE_GENERATOR_CLASS = (
    "tests.data_designer_plugins.test_pgm_generator_plugin.FakePGMGenerator"
)


class FakePGMGenerator:
    """Minimal PGMGenerator stand-in for unit tests (no real data needed)."""

    def generate_samples(
        self,
        size: int,
        evidence: dict[str, Any | list[Any]] | None = None,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [{"name": f"person_{i}", "age": 30 + i} for i in range(size)]
        )


@pytest.fixture(autouse=True)
def clear_generator_cache():
    """Ensure the module-level cache is empty before each test."""
    _generator_cache.clear()
    yield
    _generator_cache.clear()


class TestPGMGeneratorPluginConfig:
    def test_defaults(self):
        config = PGMGeneratorPluginConfig(
            name="person",
            generator_class=_FAKE_GENERATOR_CLASS,
        )
        assert config.evidence is None
        assert config.column_type == "pgm-generator"
        assert config.required_columns == []
        assert config.side_effect_columns == []

    def test_with_evidence(self):
        config = PGMGeneratorPluginConfig(
            name="person",
            generator_class=_FAKE_GENERATOR_CLASS,
            evidence={"region": "West"},
        )
        assert config.evidence == {"region": "West"}

    def test_with_list_evidence(self):
        config = PGMGeneratorPluginConfig(
            name="person",
            generator_class=_FAKE_GENERATOR_CLASS,
            evidence={"region": ["West", "East"]},
        )
        assert config.evidence == {"region": ["West", "East"]}


class TestPGMGeneratorPluginTask:
    def _make_task(self, evidence=None):
        config = PGMGeneratorPluginConfig(
            name="person",
            generator_class=_FAKE_GENERATOR_CLASS,
            evidence=evidence,
        )
        resource_provider = MagicMock()
        with patch(
            "data_designer_plugins.pgm_generator_plugin.PGMGenerator",
            FakePGMGenerator,
        ):
            return PGMGeneratorPluginTask(
                config=config,
                resource_provider=resource_provider,
            )

    def test_generation_strategy(self):
        assert (
            PGMGeneratorPluginTask.get_generation_strategy()
            == GenerationStrategy.FULL_COLUMN
        )

    def test_can_generate_from_scratch(self):
        task = self._make_task()
        assert task.can_generate_from_scratch is True

    def test_generate_from_scratch(self):
        task = self._make_task()
        with patch(
            "data_designer_plugins.pgm_generator_plugin._get_generator",
            return_value=FakePGMGenerator(),
        ):
            result = task.generate_from_scratch(num_records=5)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        assert "person" in result.columns
        record = result["person"].iloc[0]
        assert isinstance(record, dict)
        assert "name" in record and "age" in record

    def test_generate_appends_to_existing_dataframe(self):
        task = self._make_task()
        existing = pd.DataFrame({"other_col": ["a", "b", "c"]})
        with patch(
            "data_designer_plugins.pgm_generator_plugin._get_generator",
            return_value=FakePGMGenerator(),
        ):
            result = task.generate(existing)

        assert len(result) == 3
        assert "other_col" in result.columns
        assert "person" in result.columns


class TestGetGenerator:
    def test_invalid_class_path_raises(self):
        from data_designer_plugins.pgm_generator_plugin import (
            _get_generator,
        )

        with pytest.raises(ValueError, match="fully-qualified dotted path"):
            _get_generator("NotADottedPath")

    def test_non_pgm_generator_raises(self):
        from data_designer_plugins.pgm_generator_plugin import (
            _get_generator,
        )

        with pytest.raises(TypeError, match="not a subclass of PGMGenerator"):
            _get_generator("builtins.str")

    def test_caches_instance(self):
        from data_designer_plugins.pgm_generator_plugin import (
            _get_generator,
        )

        with patch(
            "data_designer_plugins.pgm_generator_plugin.PGMGenerator",
            FakePGMGenerator,
        ):
            g1 = _get_generator(_FAKE_GENERATOR_CLASS)
            g2 = _get_generator(_FAKE_GENERATOR_CLASS)
        assert g1 is g2


class TestPlugin:
    def test_plugin_type(self):
        assert plugin.plugin_type == PluginType.COLUMN_GENERATOR

    def test_plugin_impl_name(self):
        assert "PGMGeneratorPluginTask" in plugin.impl_qualified_name

    def test_plugin_config_name(self):
        assert "PGMGeneratorPluginConfig" in plugin.config_qualified_name
