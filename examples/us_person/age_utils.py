# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import random
from datetime import date, timedelta


def convert_age_to_birth_date(age: int) -> str:
    today = date.today()
    start_date = today.replace(year=today.year - age - 1)
    end_date = today.replace(year=today.year - age)
    days_between = (end_date - start_date).days
    random_days = random.randint(0, days_between)
    birthdate = start_date + timedelta(days=random_days)
    return birthdate.isoformat()
