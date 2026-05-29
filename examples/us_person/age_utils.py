# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import random
from datetime import date, timedelta


def _shift_year(d: date, year: int) -> date:
    """Return d with its year shifted, falling back to Feb 28 on Feb 29."""
    try:
        return d.replace(year=year)
    except ValueError:
        # Feb 29 in a non-leap target year.
        return d.replace(year=year, day=28)


def convert_age_to_birth_date(age: int) -> str:
    today = date.today()
    start_date = _shift_year(today, today.year - age - 1)
    end_date = _shift_year(today, today.year - age)
    days_between = (end_date - start_date).days
    random_days = random.randint(0, days_between)
    birthdate = start_date + timedelta(days=random_days)
    return birthdate.isoformat()
