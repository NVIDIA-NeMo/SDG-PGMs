# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import random
from datetime import date

from .us_phone_number_utils import PhoneNumber

SSN_RANDOMIZATION_DATE = date(2011, 6, 25)

# Simplified one-contiguous-range-per-state approximation of historic SSA
# area-number assignments (pre-2011 randomization). The real assignments are
# discontinuous per state (e.g., NM also held 585, AZ also held 600-601, and
# the Caribbean/Pacific territories shared 580-587). See SSA's public
# reference: https://www.ssa.gov/employer/stateweb.htm. These ranges are
# illustrative only and intentionally non-overlapping.
STATE_TO_AREA_SSN = {
    "NH": [1, 3],
    "ME": [4, 7],
    "VT": [8, 9],
    "MA": [10, 34],
    "RI": [35, 39],
    "CT": [40, 49],
    "NY": [50, 134],
    "NJ": [135, 158],
    "PA": [159, 211],
    "MD": [212, 220],
    "DE": [221, 222],
    "VA": [223, 231],
    "WV": [232, 236],
    "NC": [237, 246],
    "SC": [247, 251],
    "GA": [252, 260],
    "FL": [261, 267],
    "OH": [268, 302],
    "IN": [303, 317],
    "IL": [318, 361],
    "MI": [362, 386],
    "WI": [387, 399],
    "KY": [400, 407],
    "TN": [408, 415],
    "AL": [416, 424],
    "MS": [425, 428],
    "AR": [429, 432],
    "LA": [433, 439],
    "OK": [440, 448],
    "TX": [449, 467],
    "MN": [468, 477],
    "IA": [478, 485],
    "MO": [486, 500],
    "ND": [501, 502],
    "SD": [503, 504],
    "NE": [505, 508],
    "KS": [509, 515],
    "MT": [516, 517],
    "ID": [518, 519],
    "WY": [520, 520],
    "CO": [521, 524],
    "NM": [525, 525],
    "AZ": [526, 527],
    "UT": [528, 529],
    "NV": [530, 530],
    "WA": [531, 539],
    "OR": [540, 544],
    "CA": [545, 573],
    "AK": [574, 574],
    "HI": [575, 576],
    "DC": [577, 579],
    "PR": [580, 584],
    "VI": [585, 585],
    "GU": [586, 586],
    "AS": [587, 587],
}


def generate_phone_number(
    age: int,
    zipcode: str,
    style: str = "dash",
) -> str | None:
    """
    Generate a phone number correlated with location (zipcode).
    Phone number is None for children.
    """
    if age < 18:
        return None
    locality_var = random.random()
    if locality_var < 0.6:
        # Exact match to zipcode 60% of the time
        return PhoneNumber.from_zip_prefix(zipcode).format(style=style)
    elif locality_var < 0.8:
        # Nearby zipcodes 20% of the time
        return PhoneNumber.from_zip_prefix(zipcode[:4]).format(style=style)
    elif locality_var < 0.9:
        # More distant zipcodes 10% of the time
        return PhoneNumber.from_zip_prefix(zipcode[:3]).format(style=style)
    # Random (population-weighted) area code 10% of the time
    return PhoneNumber.generate().format(style=style)


def generate_us_national_id(state: str, birth_date: str) -> str:
    """
    Generate a synthetic SSN based on state and birth date.

    The Social Security Administration switched to randomized area numbers
    on June 25, 2011. For births *before* that date, this function picks
    the area number from the state's historic SSA range with probability
    0.7, and from a random other state's range with probability 0.3
    (Americans changed states). For births on or after that date, the
    area number is drawn from the full 1-899 range.

    The output is purely illustrative: it follows the SSN format
    ``XXX-XX-XXXX`` but is not a real SSN and is not validated against
    any government source.

    Args:
        state: Two-letter state code (e.g., "NY", "CA").
        birth_date: Date of birth in ISO format.

    Returns:
        A formatted synthetic SSN in the format "XXX-XX-XXXX".
    """
    birth_date = date.fromisoformat(birth_date)
    if birth_date < SSN_RANDOMIZATION_DATE:
        if random.random() < 0.3:
            area_range = random.choice(list(STATE_TO_AREA_SSN.values()))
        else:
            area_range = STATE_TO_AREA_SSN.get(state, [1, 899])
    else:
        area_range = [1, 899]
    area = 666
    while area == 666:
        area = random.randint(area_range[0], area_range[1])
    group = random.randint(1, 99)
    serial = random.randint(1, 9999)
    return f"{area:03d}-{group:02d}-{serial:04d}"
