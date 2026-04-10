# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end test for a cascaded multi-field PGM generator.

Models a toy "employee" domain across two cascaded Bayesian networks
with a postprocessing step and a latent variable in between:

  Model 1 (demographics):
      Department ──► _Seniority
      (independent)   (latent — dropped from final output)

  Postprocessing after Model 1:
      Map _Seniority → ExperienceYears (e.g. "junior" → "0-3", etc.)

  Model 2 (compensation):
      _Seniority ──► SalaryBand

This exercises cascaded models, cross-model variable passing,
inter-model postprocessing, latent variable removal, evidence-based
rejection sampling, and fill_na_and_gen_cpd.
"""

import pandas as pd
import pytest
from pgmpy.factors.discrete import TabularCPD

from pgms.generators.base.pgm_generator import Edge, PGMGenerator

SENIORITY_TO_EXPERIENCE = {
    "junior": "0-3",
    "mid": "4-8",
    "senior": "9+",
}


def _postprocess_experience(df: pd.DataFrame) -> pd.DataFrame:
    df["experience_years"] = df["_seniority"].map(SENIORITY_TO_EXPERIENCE)
    return df


class EmployeeGenerator(PGMGenerator):
    """Small cascaded PGM for testing."""

    def get_data(self):
        return None

    def get_edges(self):
        return [
            [Edge(start="department", end="_seniority")],
            [Edge(start="_seniority", end="salary_band")],
        ]

    def get_variables(self, data):
        return {
            "department": ["engineering", "sales", "marketing"],
            "_seniority": ["junior", "mid", "senior"],
            "salary_band": ["low", "medium", "high"],
        }

    def get_cpds(self, data):
        dept_cpd = TabularCPD(
            variable="department",
            variable_card=3,
            values=[[0.5], [0.3], [0.2]],
            state_names={"department": ["engineering", "sales", "marketing"]},
        )
        # P(_seniority | department) — engineering skews senior,
        # sales skews mid, marketing is roughly uniform.
        seniority_cpd = TabularCPD(
            variable="_seniority",
            variable_card=3,
            values=[
                [0.2, 0.3, 0.35],  # junior
                [0.3, 0.5, 0.35],  # mid
                [0.5, 0.2, 0.30],  # senior
            ],
            evidence=["department"],
            evidence_card=[3],
            state_names={
                "department": ["engineering", "sales", "marketing"],
                "_seniority": ["junior", "mid", "senior"],
            },
        )
        # P(salary_band | _seniority)
        salary_cpd = TabularCPD(
            variable="salary_band",
            variable_card=3,
            values=[
                [0.7, 0.2, 0.05],  # low
                [0.25, 0.6, 0.25],  # medium
                [0.05, 0.2, 0.70],  # high
            ],
            evidence=["_seniority"],
            evidence_card=[3],
            state_names={
                "_seniority": ["junior", "mid", "senior"],
                "salary_band": ["low", "medium", "high"],
            },
        )
        return [
            {"department": dept_cpd, "_seniority": seniority_cpd},
            {"salary_band": salary_cpd},
        ]

    def get_postprocessing_steps(self):
        return [[_postprocess_experience], None]

    def get_latents(self):
        return ["_seniority"]


class EmployeeGeneratorFromCounts(EmployeeGenerator):
    """Variant that builds the seniority CPD via fill_na_and_gen_cpd."""

    def get_cpds(self, data):
        base_cpds = super().get_cpds(data)
        # Replace the hand-written seniority CPD with one built from counts.
        counts = pd.DataFrame(
            {
                "department": pd.Categorical(
                    [
                        "engineering",
                        "engineering",
                        "engineering",
                        "sales",
                        "sales",
                        "sales",
                        "marketing",
                        "marketing",
                        "marketing",
                    ],
                    categories=["engineering", "sales", "marketing"],
                ),
                "_seniority": pd.Categorical(
                    [
                        "junior",
                        "mid",
                        "senior",
                        "junior",
                        "mid",
                        "senior",
                        "junior",
                        "mid",
                        "senior",
                    ],
                    categories=["junior", "mid", "senior"],
                ),
                "count": [20, 30, 50, 30, 50, 20, 35, 35, 30],
            }
        )
        seniority_cpd = self.fill_na_and_gen_cpd(
            counts=counts,
            variable_name="_seniority",
            evidence=["department"],
        )
        base_cpds[0]["_seniority"] = seniority_cpd
        return base_cpds


@pytest.fixture(params=["hand_written", "from_counts"])
def gen(request):
    if request.param == "hand_written":
        return EmployeeGenerator()
    return EmployeeGeneratorFromCounts()


class TestEmployeeGeneratorE2E:
    def test_conditional_structure_holds(self, gen):
        """Engineering skews toward senior → high salary should appear more
        often for engineering than for sales."""
        samples = gen.generate_samples(size=2000, seed=0, disable_progress_bar=True)

        # Latent should be dropped, postprocessed column should be present.
        assert set(samples.columns) == {
            "department",
            "salary_band",
            "experience_years",
        }

        eng = samples[samples["department"] == "engineering"]
        sales = samples[samples["department"] == "sales"]
        eng_high_rate = (eng["salary_band"] == "high").mean()
        sales_high_rate = (sales["salary_band"] == "high").mean()
        assert eng_high_rate > sales_high_rate

    def test_seeded_reproducibility(self, gen):
        a = gen.generate_samples(size=50, seed=123, disable_progress_bar=True)
        b = gen.generate_samples(size=50, seed=123, disable_progress_bar=True)
        pd.testing.assert_frame_equal(a, b)
