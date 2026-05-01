# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import random
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import IO

import numpy as np
import pandas as pd
import torch
from pgmpy import config
from pgmpy.factors.discrete import TabularCPD

from examples.us_person import (
    age_utils,
    email_address_utils,
)
from examples.us_person import (
    us_person_field_utils as utils,
)
from examples.us_person.data import USPersonData
from pgms.generators.base.pgm_generator import Edge, PGMGenerator
from pgms.generators.base.utils import tensor_normalize

# Set up logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

USPersonDataPath = str | IO

DEFAULT_PERSON_DATA_FILENAME = "person_data_v4.tar"
DEFAULT_PERSON_DATA_PRIMARY_PATH = os.environ.get(
    "US_PERSON_DATA_PATH",
    str(Path(__file__).parent.joinpath("data", DEFAULT_PERSON_DATA_FILENAME)),
)

PERSON_DATA_TARFILE_PATHS = {
    "total_dist": "census_total_by_zipcode.parquet",
    "age_dist": "census_single_year_of_age.parquet",
    "age_sex_dist": "census_age_sex_by_zipcode.parquet",
    "city_state_dist": "usps_city_state_by_zipcode.parquet",
    "county_dist": "kaggle_county_by_zipcode.parquet",
    "marital_dist": "census_marital_by_zipcode.parquet",
    "education_dist": "census_education_by_zipcode.parquet",
    "bachelors_dist": "census_bachelors_by_zipcode.parquet",
    "occupation_dist": "census_derived_occupation_by_education_age_bachelors.parquet",
    "detailed_occupation_dist": "census_detailed_occupations_national.parquet",
    "race_dist": "census_race_by_zipcode.parquet",
    "ethnicity_dist": "census_ethnicity_by_zipcode.parquet",
    "ethnicity_total_dist": "census_ethnicity_total_by_zipcode.parquet",
    "race_ethnicity_dist": "census_race_ethnicity_by_zipcode.parquet",
    "last_name_dist": "harvard_nemo_surname_by_group.parquet",
    "first_name_dist": "harvard_nemo_first_name_by_group_gender.parquet",
    "middle_name_dist": "harvard_nemo_middle_name_by_group_gender.parquet",
    "street_name_dist": "openaddresses_street_name_by_zipcode.parquet",
    "unit_dist": "openaddresses_unit_by_zipcode.parquet",
    "age_group_map": "age_group_map.parquet",
}


class USPersonGenerator(PGMGenerator):
    def __init__(
        self,
        data_path: USPersonDataPath | None = None,
        data: USPersonData | None = None,
    ):
        assert not (data_path and data), (
            "Only one of data_path or data should be provided."
        )
        if (data is None) and (data_path is None):
            data_path = DEFAULT_PERSON_DATA_PRIMARY_PATH
        super().__init__(data_path=data_path, data=data)

    def get_variables(self, data: USPersonData) -> dict[str, list[str]]:
        return {
            "zipcode": list(data.total_dist["zipcode"].cat.categories),
            "zipcode_category": list(
                data.total_dist["zipcode_category"].cat.categories
            ),
            "zip_prefix4": list(data.total_dist["zip_prefix4"].cat.categories),
            "sex": list(data.age_sex_dist["sex"].cat.categories),
            "age": list(data.age_dist["age"].cat.categories),
            "age_group": list(data.age_sex_dist["age_group"].cat.categories),
            "age_group_marital": list(
                data.age_group_map["age_group_marital"].cat.categories
            ),
            "age_group_education": list(
                data.age_group_map["age_group_education"].cat.categories
            ),
            "age_group_bachelors": list(
                data.age_group_map["age_group_bachelors"].cat.categories
            ),
            "age_group_occupation": list(
                data.age_group_map["age_group_occupation"].cat.categories
            ),
            "marital_status": list(data.marital_dist["marital_status"].cat.categories),
            "education_level": list(
                data.education_dist["education_level"].cat.categories
            ),
            "education_level_bachelors": list(
                data.occupation_dist["education_level_bachelors"].cat.categories
            ),
            "bachelors_field": list(
                data.bachelors_dist["bachelors_field"].cat.categories
            ),
            "occupation": list(data.occupation_dist["occupation"].cat.categories),
            "detailed_occupation": list(
                data.detailed_occupation_dist["detailed_occupation"].cat.categories
            ),
            "race_broad": list(data.race_dist["race_broad"].cat.categories),
            "race_group": list(data.race_dist["race_group"].cat.categories),
            "ethnicity": list(data.ethnicity_dist["ethnicity"].cat.categories),
            "ethnicity_origin": list(data.ethnicity_dist["origin"].cat.categories),
            "ethnic_background": list(data.last_name_dist.columns.categories),
            "last_name": list(data.last_name_dist.index.categories),
            "first_name": list(data.first_name_dist.index.categories),
            "middle_name": list(data.middle_name_dist.index.categories),
            "street_name": list(data.street_name_dist["street_name"].cat.categories),
            "unit": list(data.unit_dist["unit"].cat.categories),
            "city": list(data.city_state_dist["city"].cat.categories),
            "state": list(data.city_state_dist["state"].cat.categories),
            "country": ["USA"],
        }

    def get_latents(self) -> list[str]:
        # Define latent/unobserved variables
        # These will be treated like any other variable,
        # except that they are removed from the final samples
        return [
            "zipcode_category",
            "zip_prefix4",
            "age_group",
            "age_group_education",
            "age_group_bachelors",
            "age_group_marital",
            "age_group_occupation",
            "education_level_bachelors",
            "ethnicity",
            "ethnicity_origin",
            "race_broad",
            "race_group",
        ]

    def get_data(
        self,
        data_path: USPersonDataPath | None = None,
        data: USPersonData | None = None,
    ) -> USPersonData:
        """Load count tables from a tarfile containing parquet pandas dataframes."""
        if data is None:
            data = USPersonData.from_tarfile(data_path, PERSON_DATA_TARFILE_PATHS)
        # Parquet can't handle columns as categoricals; needs to be done on load.
        data.last_name_dist.columns = data.last_name_dist.columns.astype("category")
        # Similarly for ints
        data.age_dist.age = data.age_dist.age.astype("category")
        return data

    def get_edges(self, *args, **kwargs) -> list[list[Edge]]:
        return [
            [
                Edge(start="zipcode", end="sex"),
                Edge(start="zipcode", end="age_group"),
                Edge(start="sex", end="age_group"),
                Edge(start="age_group", end="age"),
                Edge(start="sex", end="age"),
            ],
            [
                Edge(start="zipcode_category", end="street_name"),
                Edge(start="zipcode_category", end="unit"),
                Edge(start="zipcode", end="marital_status"),
                Edge(start="sex", end="marital_status"),
                Edge(start="age_group_marital", end="marital_status"),
                Edge(start="zipcode", end="education_level"),
                Edge(start="sex", end="education_level"),
                Edge(start="age_group_education", end="education_level"),
                Edge(start="zipcode", end="bachelors_field"),
                Edge(start="sex", end="bachelors_field"),
                Edge(start="age_group_bachelors", end="bachelors_field"),
                Edge(start="zip_prefix4", end="ethnicity"),
                Edge(start="zip_prefix4", end="race_broad"),
                Edge(start="ethnicity", end="race_broad"),
                Edge(start="zip_prefix4", end="race_group"),
                Edge(start="race_broad", end="race_group"),
                Edge(start="zip_prefix4", end="ethnicity_origin"),
                Edge(start="ethnicity", end="ethnicity_origin"),
            ],
            [
                Edge(start="age_group_occupation", end="occupation"),
                Edge(start="education_level_bachelors", end="occupation"),
                Edge(start="occupation", end="detailed_occupation"),
                Edge(start="sex", end="detailed_occupation"),
                Edge(start="ethnic_background", end="last_name"),
                Edge(start="ethnic_background", end="first_name"),
                Edge(start="sex", end="first_name"),
                Edge(start="ethnic_background", end="middle_name"),
                Edge(start="sex", end="middle_name"),
            ],
        ]

    def get_mappings(self, data: USPersonData) -> dict[str, pd.DataFrame]:
        """Load deterministic mappings for use in postprocessing."""
        county_dist = (
            data.county_dist.groupby("zipcode", observed=True)["county"]
            .apply(list)
            .reset_index()
        )
        return {
            "zipcode_category": data.total_dist[["zipcode", "zipcode_category"]],
            "city_state_zip": data.city_state_dist[["zipcode", "city", "state"]],
            "county_zip": county_dist,
            "age_group": data.age_group_map,
        }

    def get_cpds(self, data: USPersonData) -> list[dict[str, TabularCPD]]:
        """Create CPDs based on count tables or defaults."""

        logger.info("Building PGM statistics...")

        # Top-level CPDs
        vals = data.total_dist.sort_values("zipcode")["count"].values[:, None]
        zipcode_cpd = TabularCPD(
            variable="zipcode",
            variable_card=len(self.variables["zipcode"]),
            values=tensor_normalize(
                vals,
                config.get_dtype(),
                config.get_device(),
            ),
            state_names={"zipcode": list(self.variables["zipcode"])},
        )

        # Calculate address probabilities
        street_name_cpd = self.fill_na_and_gen_cpd(
            data.street_name_dist, "street_name", ["zipcode_category"], pseudocount=0
        )
        unit_cpd = self.fill_na_and_gen_cpd(
            data.unit_dist, "unit", ["zipcode_category"], pseudocount=0
        )

        # Calculate sex probabilities
        sex_probs = (
            data.age_sex_dist.groupby(["zipcode", "sex"], observed=True)["count"]
            .sum()
            .reset_index()
        )
        sex_cpd = self.fill_na_and_gen_cpd(sex_probs, "sex", ["zipcode"])

        # Calculate age group probabilities
        age_group_cpd = self.fill_na_and_gen_cpd(
            data.age_sex_dist,
            "age_group",
            ["zipcode", "sex"],
        )

        # Calculate age probabilities
        age_cpd = self.fill_na_and_gen_cpd(
            data.age_dist,
            "age",
            ["age_group", "sex"],
        )

        # Calculate marital status probabilities
        marital_evidence = ["zipcode", "sex", "age_group_marital"]
        marital_cpd = self.fill_na_and_gen_cpd(
            data.marital_dist, "marital_status", marital_evidence
        )

        # Calculate education level probabilities
        education_evidence = ["zipcode", "age_group_education", "sex"]
        education_cpd = self.fill_na_and_gen_cpd(
            data.education_dist, "education_level", education_evidence
        )

        # Calculate bachelors probabilities
        bachelors_evidence = ["zipcode", "sex", "age_group_bachelors"]
        bachelors_cpd = self.fill_na_and_gen_cpd(
            data.bachelors_dist, "bachelors_field", bachelors_evidence
        )

        # Calculate occupation probabilities
        occupation_evidence = [
            "education_level_bachelors",
            "age_group_occupation",
        ]
        occupation_cpd = self.fill_na_and_gen_cpd(
            data.occupation_dist, "occupation", occupation_evidence
        )

        # Calculate detailed occupation probabilities
        detailed_occupation_evidence = ["occupation", "sex"]
        detailed_occupation_cpd = self.fill_na_and_gen_cpd(
            data.detailed_occupation_dist,
            "detailed_occupation",
            detailed_occupation_evidence,
            pseudocount=0,
        )

        # Calculate ethnicity probabilities
        ethnicity_cpd = self.fill_na_and_gen_cpd(
            data.ethnicity_total_dist, "ethnicity", ["zip_prefix4"]
        )

        # Calculate broad race probabilities
        race_broad_cpd = self.fill_na_and_gen_cpd(
            data.race_ethnicity_dist, "race_broad", ["zip_prefix4", "ethnicity"]
        )

        # Calculate race probabilities
        race_group_cpd = self.fill_na_and_gen_cpd(
            data.race_dist, "race_group", ["zip_prefix4", "race_broad"]
        )

        ethnicity_origin_probs = data.ethnicity_dist[
            ["zip_prefix4", "ethnicity", "origin", "count"]
        ]
        ethnicity_origin_probs = ethnicity_origin_probs.rename(
            columns={"origin": "ethnicity_origin"}
        )
        ethnicity_origin_cpd = self.fill_na_and_gen_cpd(
            ethnicity_origin_probs, "ethnicity_origin", ["zip_prefix4", "ethnicity"]
        )

        # Treated as uniform since it will be generated from race_group and ethnicity_origin
        vals = torch.tensor([[1.0] * len(self.variables["ethnic_background"])]).T
        ethnic_background_cpd = TabularCPD(
            variable="ethnic_background",
            variable_card=len(self.variables["ethnic_background"]),
            values=tensor_normalize(vals, config.get_dtype(), config.get_device()),
            state_names={
                "ethnic_background": list(self.variables["ethnic_background"])
            },
        )

        last_name_cpd = TabularCPD(
            variable="last_name",
            variable_card=len(self.variables["last_name"]),
            values=tensor_normalize(
                data.last_name_dist.values, config.get_dtype(), config.get_device()
            ),
            evidence=["ethnic_background"],
            evidence_card=[len(self.variables[k]) for k in ["ethnic_background"]],
            state_names={
                k: list(self.variables[k]) for k in ["ethnic_background", "last_name"]
            },
        )

        first_name_cpd = TabularCPD(
            variable="first_name",
            variable_card=len(self.variables["first_name"]),
            values=tensor_normalize(
                data.first_name_dist.values,
                config.get_dtype(),
                config.get_device(),
            ),
            evidence=["ethnic_background", "sex"],
            evidence_card=[
                len(self.variables[k]) for k in ["ethnic_background", "sex"]
            ],
            state_names={
                k: list(self.variables[k])
                for k in ["ethnic_background", "sex", "first_name"]
            },
        )

        middle_name_cpd = TabularCPD(
            variable="middle_name",
            variable_card=len(self.variables["middle_name"]),
            values=tensor_normalize(
                data.middle_name_dist.values,
                config.get_dtype(),
                config.get_device(),
            ),
            evidence=["ethnic_background", "sex"],
            evidence_card=[
                len(self.variables[k]) for k in ["ethnic_background", "sex"]
            ],
            state_names={
                k: list(self.variables[k])
                for k in ["ethnic_background", "sex", "middle_name"]
            },
        )

        cpd_definitions = [
            {
                "zipcode": zipcode_cpd,
                "sex": sex_cpd,
                "age_group": age_group_cpd,
                "age": age_cpd,
            },
            {
                "street_name": street_name_cpd,
                "unit": unit_cpd,
                "marital_status": marital_cpd,
                "education_level": education_cpd,
                "bachelors_field": bachelors_cpd,
                "ethnicity": ethnicity_cpd,
                "race_broad": race_broad_cpd,
                "race_group": race_group_cpd,
                "ethnicity_origin": ethnicity_origin_cpd,
            },
            {
                "occupation": occupation_cpd,
                "detailed_occupation": detailed_occupation_cpd,
                "ethnic_background": ethnic_background_cpd,
                "last_name": last_name_cpd,
                "first_name": first_name_cpd,
                "middle_name": middle_name_cpd,
            },
        ]
        return cpd_definitions

    def generate_zipcode_groups(self, samples: pd.DataFrame) -> pd.DataFrame:
        """
        Postprocess zipcodes to generate 4-digit prefixes and categories.
        """
        samples["zip_prefix4"] = samples["zipcode"].str[:4]
        samples = pd.merge(
            samples,
            self.mappings["zipcode_category"],
            how="left",
            left_on="zipcode",
            right_on="zipcode",
            validate="m:1",
        )
        return samples

    def generate_education_bachelors_field(self, samples: pd.DataFrame) -> pd.DataFrame:
        """
        Combine education_level and bachelors_field into a single column.
        """
        samples["education_level_bachelors"] = samples["education_level"]
        condition = samples["education_level"].isin(["bachelors", "graduate"])
        samples.loc[condition, "education_level_bachelors"] = (
            samples.loc[condition, "education_level"]
            + "-"
            + samples.loc[condition, "bachelors_field"]
        )
        return samples

    def set_pre_college_bachelors_to_na(self, samples: pd.DataFrame) -> pd.DataFrame:
        """
        Zero out bachelor's field for education below bachelor's.
        """
        mask_below_bachelors = ~samples["education_level"].isin(
            ["bachelors", "graduate"]
        )
        samples.loc[mask_below_bachelors, "bachelors_field"] = "no_degree"
        return samples

    def postprocess_occupation(self, samples: pd.DataFrame) -> pd.DataFrame:
        """
        Change occupation to detailed occupation and set to
        'not_in_workforce' for individuals over 85.

        Args:
            samples: DataFrame containing occupation column

        Returns:
            DataFrame with modified occupation values
        """

        samples["occupation"] = samples["detailed_occupation"]
        samples = samples.drop(columns=["detailed_occupation"])
        samples.loc[samples["age_group"] == "85_plus", "occupation"] = (
            "not_in_workforce"
        )

        return samples

    def generate_ethnic_background(self, samples: pd.DataFrame) -> pd.DataFrame:
        """
        Generate ethnic_background column based on ethnicity and race information.
        If ethnicity is "hispanic or latino", use ethnicity_origin.
        Otherwise, combine race_group and race_subgroup.

        Args:
            samples: DataFrame containing ethnicity, ethnicity_origin, race_group, and race_subgroup columns

        Returns:
            DataFrame with additional ethnic_background column
        """

        # For hispanic/latino individuals, use ethnicity_origin
        hispanic_mask = samples["ethnicity"] == "hispanic or latino"

        # Initialize ethnic_background with the race combination
        samples["ethnic_background"] = samples["race_group"]

        # Override with ethnicity_origin for hispanic/latino individuals
        samples.loc[hispanic_mask, "ethnic_background"] = samples.loc[
            hispanic_mask, "ethnicity_origin"
        ]

        return samples

    def convert_names_to_titlecase(self, samples: pd.DataFrame) -> pd.DataFrame:
        """
        Convert first, middle and last names to title case.

        Args:
            samples: DataFrame containing generated records.

        Returns:
            DataFrame with first_name, middle_name and last_name in title case.
        """
        samples["first_name"] = samples["first_name"].str.title()
        samples["middle_name"] = samples["middle_name"].str.title()
        samples["last_name"] = samples["last_name"].str.title()
        return samples

    def add_street_numbers(
        self, samples: pd.DataFrame, mean_street_number: int = 200
    ) -> pd.DataFrame:
        """
        Add a street number sampled independently from a geometic distribution.

        Args:
            samples: DataFrame containing generated records.
            mean_street_number: Mean of the geometric distribution (default: 200).

        Returns:
            DataFrame with additional street number column.
        """
        # For geometric distribution, p = 1/mean
        p = 1 / mean_street_number
        street_numbers = np.random.geometric(p=p, size=len(samples))
        samples["street_number"] = street_numbers
        return samples

    def append_city_state_country(self, samples: pd.DataFrame) -> pd.DataFrame:
        """
        Append the city and state based on zipcode.
        Country is set to "USA".

        Args:
            samples: DataFrame containing generated records

        Returns:
            DataFrame with additional city and state columns
        """
        samples = pd.merge(
            samples,
            self.mappings["city_state_zip"],
            how="left",
            left_on="zipcode",
            right_on="zipcode",
            validate="m:1",
        )
        samples["city"] = samples["city"].astype(str)
        samples["state"] = samples["state"].astype(str)
        samples["country"] = "USA"
        return samples

    def append_county(self, samples: pd.DataFrame) -> pd.DataFrame:
        """
        Append the county based on zipcode.
        Requires sampling one of the counties mapped to the zipcode.

        Args:
            samples: DataFrame containing generated records

        Returns:
            DataFrame with additional county column
        """
        samples = pd.merge(
            samples,
            self.mappings["county_zip"],
            how="left",
            left_on="zipcode",
            right_on="zipcode",
        )
        samples["county"] = samples["county"].apply(lambda x: random.choice(x))
        samples["county"] = samples["county"].astype(str)
        return samples

    def append_age_groups(self, samples: pd.DataFrame) -> pd.DataFrame:
        """
        Append the different granularities of age groups.

        Args:
            samples: DataFrame containing generated records

        Returns:
            DataFrame with additional age_group columns
        """
        samples = pd.merge(
            samples,
            self.mappings["age_group"],
            how="left",
            left_on="age_group",
            right_on="age_group",
            validate="m:1",
        )
        return samples

    def add_birth_date(self, samples: pd.DataFrame) -> pd.DataFrame:
        """
        Add a birth date based on the age column.
        """
        samples["birth_date"] = samples["age"].apply(
            age_utils.convert_age_to_birth_date
        )
        return samples

    def generate_phone_number(self, samples: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a phone number based on the zipcode and age.
        """
        samples["phone_number"] = samples.apply(
            lambda row: utils.generate_phone_number(
                age=row["age"],
                zipcode=row["zipcode"],
            ),
            axis=1,
        )
        return samples

    def generate_email_address(self, samples: pd.DataFrame) -> pd.DataFrame:
        """
        Generate an email address based on the first name, middle name, last name, age and birth date.
        """
        samples["email_address"] = samples.apply(
            lambda row: email_address_utils.EmailAddressGen(
                locale="US"
            ).generate_email_address(
                first_name=row["first_name"],
                middle_name=row["middle_name"],
                last_name=row["last_name"],
                birth_date=row["birth_date"],
            ),
            axis=1,
        )
        return samples

    def generate_us_national_id(self, samples: pd.DataFrame) -> pd.DataFrame:
        """
        Generate an SSN based on the state and birth date.
        """
        samples["national_id"] = samples.apply(
            lambda row: utils.generate_us_national_id(
                state=row["state"],
                birth_date=row["birth_date"],
            ),
            axis=1,
        )
        return samples

    def add_uuid(self, samples: pd.DataFrame) -> pd.DataFrame:
        """
        Add a unique UUID to each record in the DataFrame.

        Args:
            samples: DataFrame containing generated records

        Returns:
            DataFrame with additional uuid column
        """

        # Generate UUIDs
        samples["uuid"] = [str(uuid.uuid4()) for _ in range(len(samples))]

        return samples

    def get_postprocessing_steps(self) -> list[list[Callable]]:
        """Load postprocessing steps for all models."""
        return [
            [
                self.generate_zipcode_groups,
                self.append_age_groups,
            ],
            [
                self.generate_ethnic_background,
                self.set_pre_college_bachelors_to_na,
                self.generate_education_bachelors_field,
            ],
            [
                self.convert_names_to_titlecase,
                self.add_street_numbers,
                self.append_city_state_country,
                self.append_county,
                self.postprocess_occupation,
                self.add_birth_date,
                self.generate_phone_number,
                self.generate_email_address,
                self.generate_us_national_id,
                self.add_uuid,
            ],
        ]
