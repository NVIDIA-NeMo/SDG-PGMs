# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the synthetic SSN generator in us_person_field_utils."""

import random
import re
from collections import Counter

import pytest
from examples.us_person.us_person_field_utils import (
    STATE_TO_AREA_SSN,
    generate_us_national_id,
)

SSN_PATTERN = re.compile(r"^\d{3}-\d{2}-\d{4}$")


@pytest.fixture(autouse=True)
def _seed_random():
    """Make the test deterministic across runs without leaking state."""
    state = random.getstate()
    random.seed(20260527)
    yield
    random.setstate(state)


@pytest.mark.parametrize("state", ["NY", "CA", "TX"])
@pytest.mark.parametrize("birth_date", ["1985-06-15", "2015-01-01"])
def test_format_is_valid(state, birth_date):
    """Output always matches the canonical XXX-XX-XXXX format."""
    ssn = generate_us_national_id(state=state, birth_date=birth_date)
    assert SSN_PATTERN.match(ssn), ssn


@pytest.mark.parametrize("state", ["NY", "CA", "TX"])
def test_area_is_never_invalid(state):
    """Area is never 000, 666, or in 900-999 across many samples."""
    for _ in range(200):
        ssn = generate_us_national_id(state=state, birth_date="1985-06-15")
        area = int(ssn.split("-")[0])
        assert area != 666
        assert area != 0
        assert area < 900


def test_pre_2011_usually_picks_state_range():
    """For pre-2011 births, area should be in the state range most of the
    time. The off-state branch fires with probability 0.3, so over 500 trials
    we should see clear majority-in-range."""
    state = "NY"
    state_lo, state_hi = STATE_TO_AREA_SSN[state]
    in_range = 0
    n = 500
    for _ in range(n):
        ssn = generate_us_national_id(state=state, birth_date="1985-06-15")
        area = int(ssn.split("-")[0])
        if state_lo <= area <= state_hi:
            in_range += 1
    # With 70% in-state plus some fraction of the 30% off-state that
    # randomly lands in NY's range, expect well above 60%.
    assert in_range / n > 0.6


def test_post_2011_uses_full_range():
    """For post-2011 births, state code is ignored; area is drawn from 1-899."""
    areas = Counter()
    for _ in range(500):
        ssn = generate_us_national_id(state="NY", birth_date="2015-01-01")
        areas[int(ssn.split("-")[0])] += 1
    # Confirm we see values well outside NY's [50, 134] range.
    outside_ny = sum(c for a, c in areas.items() if a < 50 or a > 134)
    assert outside_ny > 0


def test_unknown_state_falls_back_to_full_range():
    """An unknown state code (e.g., from a non-US example) does not error."""
    ssn = generate_us_national_id(state="ZZ", birth_date="1985-06-15")
    assert SSN_PATTERN.match(ssn)
