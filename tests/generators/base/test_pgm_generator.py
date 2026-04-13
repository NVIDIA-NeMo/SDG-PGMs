# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork

from pgms.generators.base.pgm_generator import Edge, PGMGenerator


class DummyPGMGenerator(PGMGenerator):
    def get_data(self):
        # Return some dummy data
        return {"dummy": True}

    def get_edges(self):
        return [[], [Edge(start="B", end="C")]]

    def get_variables(self, data):
        # Define three variables with varying state counts.
        return {"A": ["a1", "a2"], "B": ["b1", "b2"], "C": ["c1", "c2", "c3", "c4"]}

    def get_cpds(self, data):
        # In _build_network, missing CPDs for nodes will be generated automatically.
        a_cpd = TabularCPD(
            variable="A",
            variable_card=2,
            values=[[0.5], [0.5]],
            state_names={"A": ["a1", "a2"]},
        )
        b_cpd = TabularCPD(
            variable="B",
            variable_card=2,
            values=[[0.5], [0.5]],
            state_names={"B": ["b1", "b2"]},
        )
        c_cpd = TabularCPD(
            variable="C",
            variable_card=4,
            values=[[0.5, 0.0], [0.5, 0.0], [0.0, 0.5], [0.0, 0.5]],
            evidence=["B"],
            evidence_card=[2],
            state_names={"B": ["b1", "b2"], "C": ["c1", "c2", "c3", "c4"]},
        )
        return [{"A": a_cpd, "B": b_cpd}, {"C": c_cpd}]

    def get_postprocessing_steps(self):
        return None

    def get_latents(self):
        return []


@pytest.fixture
def gen():
    return DummyPGMGenerator()


def test_build_network(gen):
    """
    Test that _build_network creates a BayesianNetwork with the expected
    nodes and edges, and that missing CPDs are auto-generated.
    """
    assert isinstance(gen.models, list)
    assert len(gen.models) == 2

    # Check expected nodes and edges
    assert isinstance(gen.models[0], BayesianNetwork)
    assert set(gen.models[0].nodes()) == {"A", "B"}
    assert len(gen.models[0].edges()) == 0
    assert isinstance(gen.models[1], BayesianNetwork)
    assert set(gen.models[1].nodes()) == {"B", "C"}
    assert len(gen.models[1].edges()) == 1
    assert ("B", "C") in gen.models[1].edges()

    # Check expected CPDs, including auto-generated CPDs (flat distributions).
    for var in ["A", "B", "C"]:
        if var == "C":
            model = gen.models[1]
            cardinality = 4
            values = [[0.5, 0.0], [0.5, 0.0], [0.0, 0.5], [0.0, 0.5]]
            evidence = ["B"]
        else:
            model = gen.models[0]
            cardinality = 2
            values = np.ones((cardinality, 1)) / cardinality
            evidence = []
        cpd = model.get_cpds(var)
        assert isinstance(cpd, TabularCPD)
        assert cpd.variable == var
        assert cpd.variable_card == cardinality
        assert cpd.get_evidence() == evidence
        assert np.allclose(cpd.get_values(), values)


def test_fill_na_and_gen_cpd():
    """
    Test that fill_na_and_gen_cpd correctly fills missing combinations in
    a count DataFrame and returns a valid TabularCPD.
    """
    gen = DummyPGMGenerator()

    # Create a counts DataFrame for variable "B" given evidence ["A"].
    # Ensure that the columns for "A" and "B" are categorical with the same
    # categories as defined in get_variables.
    data = {
        "A": pd.Categorical(
            ["a1", "a1", "a2"],
            categories=["a1", "a2"],
        ),
        "B": pd.Categorical(
            ["b1", "b2", "b1"],
            categories=["b1", "b2"],
        ),
        "count": [10, 5, 3],
    }
    counts_df = pd.DataFrame(data)

    # Call fill_na_and_gen_cpd for variable "B" given evidence ["A"].
    cpd = gen.fill_na_and_gen_cpd(
        counts=counts_df, variable_name="B", evidence=["A"], pseudocount=1e-10
    )

    # Verify that the result is a TabularCPD with the expected properties.
    assert isinstance(cpd, TabularCPD)
    assert cpd.variable == "B"
    assert cpd.variable_card == 2
    assert cpd.get_evidence() == ["A"]
    assert (cpd.cardinality == [2, 2]).all()

    # Check that state names match what was defined in get_variables.
    state_names = cpd.state_names
    assert state_names["A"] == ["a1", "a2"]
    assert state_names["B"] == ["b1", "b2"]

    # Verify that the CPD values form a proper probability table:
    # The underlying tensor should have shape (2, 2) and each column should sum to 1.
    values = cpd.get_values()
    assert values.shape == (2, 2)
    col_sums = values.sum(axis=0)
    np.testing.assert_allclose(col_sums, np.ones(2), rtol=1e-5, atol=1e-5)
    # Check normalized values
    np.testing.assert_allclose(
        values[0] / values[1], np.array([10.0 / 5, 3.0 / 1e-10]), rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize(
    "evidence,specific_allowed_values,latents,postprocessing_steps",
    [
        ({}, {}, [], None),
        ({"B": "b1"}, {"B": ["b1"], "C": ["c1", "c2"]}, [], None),
        ({"A": "a1"}, {"A": ["a1"]}, [], None),
        ({"B": "b1"}, {"B": ["b1"], "C": ["c1", "c2"]}, ["B"], None),
        (
            {},
            {"A": ["A1", "A2"]},
            [],
            [[lambda df: df.assign(A=df["A"].str.capitalize())], None],
        ),
    ],
)
def test_generate_samples(
    gen, evidence, specific_allowed_values, latents, postprocessing_steps
):
    """
    Test that generate_samples returns a DataFrame with the expected shape and
    content depending on the evidence, latents and postprocessing provided.
    """
    gen.latents = latents
    gen.postprocessing_steps = postprocessing_steps
    allowed_values = {
        "A": ["a1", "a2"],
        "B": ["b1", "b2"],
        "C": ["c1", "c2", "c3", "c4"],
    }
    allowed_values.update(specific_allowed_values)
    samples = gen.generate_samples(
        size=10, evidence=evidence, disable_progress_bar=True
    )
    # Verify that the returned samples is a DataFrame with 10 rows.
    assert isinstance(samples, pd.DataFrame)
    assert len(samples) == 10
    # Expected columns should match the variables.
    expected_columns = {"A", "B", "C"} - set(latents)
    assert set(samples.columns) == expected_columns
    # Check that values are valid.
    for var in expected_columns:
        assert all(samples[var].isin(allowed_values[var]))
    if "B" in expected_columns and "C" in expected_columns:
        # Check that the samples are consistent with the model.
        for sample in samples.itertuples():
            if sample.B == "b1":
                assert sample.C in ["c1", "c2"]
            elif sample.B == "b2":
                assert sample.C in ["c3", "c4"]


def test_validate_evidence(gen):
    with pytest.raises(ValueError):
        gen._validate_evidence({"D": "d1"})
    with pytest.raises(ValueError):
        gen._validate_evidence({"B": ["b1", "b3"]})
    with pytest.raises(ValueError):
        gen._validate_evidence({"A": "a3"})
    gen._validate_evidence({"B": "b1", "C": "c1"})


def test_partial_samples(gen):
    partial = pd.DataFrame({"A": ["a1"] * 20})
    samples = gen.generate_samples(
        size=20, partial_samples=partial, disable_progress_bar=True
    )
    assert len(samples) == 20
    assert (samples["A"] == "a1").all()
    assert set(samples.columns) == {"A", "B", "C"}


def test_rejection_sampling_failure():
    gen = DummyPGMGenerator()
    # Invalid variable combination
    bad_evidence = {"B": "b2", "C": "c1"}
    with pytest.raises(TimeoutError):
        gen.generate_samples(
            size=1000, evidence=bad_evidence, disable_progress_bar=True
        )
