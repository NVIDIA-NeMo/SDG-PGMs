# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from examples.us_person import USPersonGenerator


def test_us_person_generator_runs_with_dummy_data():
    data_dir = Path(__file__).parents[2] / "examples" / "us_person" / "data"
    generator = USPersonGenerator(data_path=str(data_dir / "person_data.tar"))
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
