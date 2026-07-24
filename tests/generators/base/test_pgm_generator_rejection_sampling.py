# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

from pgms.generators.base import pgm_generator as pgm_generator_module
from pgms.generators.base.pgm_generator import PGMGenerator


class _FailingPGMGenerator(PGMGenerator):
    """Minimal generator whose forward sampling always returns an empty DataFrame."""

    def get_data(self):
        return {}

    def get_edges(self):
        return [[]]

    def get_variables(self, data):
        return {"A": ["a1", "a2"]}

    def get_cpds(self, data):
        return [{}]

    def get_postprocessing_steps(self):
        return None

    def get_latents(self):
        return []

    def _forward_sample(self, size, partial_samples=None):
        return pd.DataFrame(columns=["A"])


def test_rejection_sample_raises_on_empty_partial_samples():
    """Empty partial_samples must raise ValueError instead of ZeroDivisionError."""
    gen = _FailingPGMGenerator()
    with pytest.raises(ValueError, match="partial_samples cannot be empty"):
        gen.generate_samples(size=1, partial_samples=pd.DataFrame(columns=["A"]))


def test_rejection_sample_large_fail_guard_uses_total_size():
    """The large-sample failure guard must compare against the requested size.

    Previously the guard compared the dynamic batch size (_size) against
    REJ_SAMP_LARGE_FAIL_SIZE, so small batches never triggered it even when the
    total requested size was huge. This test patches the threshold to a value
    between the dynamic batch size and the total size and verifies the guard
    still fires.
    """
    gen = _FailingPGMGenerator()
    original_threshold = pgm_generator_module.REJ_SAMP_LARGE_FAIL_SIZE
    try:
        pgm_generator_module.REJ_SAMP_LARGE_FAIL_SIZE = 50
        with pytest.raises(TimeoutError):
            gen.generate_samples(size=100, disable_progress_bar=True)
    finally:
        pgm_generator_module.REJ_SAMP_LARGE_FAIL_SIZE = original_threshold
