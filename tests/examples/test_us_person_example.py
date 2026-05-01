# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
from pathlib import Path


def test_us_person_generator_runs_with_dummy_data(monkeypatch):
    repo_root = Path(__file__).parents[2]
    sys.path.insert(0, str(repo_root))
    us_person = importlib.import_module("examples.us_person")
    us_phone_number_utils = importlib.import_module(
        "examples.us_person.us_phone_number_utils"
    )
    data_dir = repo_root / "examples" / "us_person" / "data"
    monkeypatch.delenv("US_PERSON_DATA_PATH", raising=False)
    monkeypatch.setenv(
        "US_PERSON_AREA_CODE_DATA_PATH",
        str(data_dir / "zip_area_code_map.parquet"),
    )
    us_phone_number_utils.ZIPCODE_AREA_CODE_MAP = None
    us_phone_number_utils.ZIPCODE_POPULATION_MAP = None

    generator = us_person.USPersonGenerator(
        data_path=str(data_dir / "person_data_v4.tar")
    )
    samples = generator.generate_samples(size=5, seed=0, disable_progress_bar=True)

    assert samples.shape[0] == 5
    assert {
        "zipcode",
        "first_name",
        "last_name",
        "street_name",
        "city",
        "phone_number",
        "email_address",
        "national_id",
        "uuid",
    }.issubset(samples.columns)
    assert set(samples["zipcode"]).issubset({"11111", "11112", "22221", "22222"})
