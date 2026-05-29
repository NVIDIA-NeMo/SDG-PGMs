# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from .data import USPersonData
from .us_person_generator import PERSON_DATA_TARFILE_PATHS

OUTPUT_DIR = Path(__file__).parent / "data"
DUMMY_PERSON_DATA_FILENAME = "person_data.tar"
DUMMY_AREA_CODE_FILENAME = "zip_area_code_map.parquet"
RNG = np.random.default_rng(20260430)

ZIPCODES = ["11111", "11112", "22221", "22222"]
ZIP_PREFIXES = ["1111", "2222"]
ZIPCODE_TO_PREFIX = {
    "11111": "1111",
    "11112": "1111",
    "22221": "2222",
    "22222": "2222",
}
ZIPCODE_CATEGORIES = ["core_alpha", "edge_alpha", "core_beta", "rural_beta"]
ZIPCODE_TO_CATEGORY = dict(zip(ZIPCODES, ZIPCODE_CATEGORIES))
SEXES = ["Female", "Male"]
AGE_GROUPS = ["under_18", "18_24", "25_44", "45_84", "85_plus"]
AGE_BY_GROUP = {
    "under_18": 9,
    "18_24": 21,
    "25_44": 36,
    "45_84": 62,
    "85_plus": 88,
}
AGE_GROUP_MARITAL = AGE_GROUPS
AGE_GROUP_EDUCATION = AGE_GROUPS
AGE_GROUP_BACHELORS = ["under_25", "25_44", "45_84", "85_plus"]
AGE_GROUP_OCCUPATION = ["under_18", "18_24", "25_44", "45_84", "85_plus"]
AGE_GROUP_MAPPINGS = {
    "under_18": ("under_18", "under_18", "under_25", "under_18"),
    "18_24": ("18_24", "18_24", "under_25", "18_24"),
    "25_44": ("25_44", "25_44", "25_44", "25_44"),
    "45_84": ("45_84", "45_84", "45_84", "45_84"),
    "85_plus": ("85_plus", "85_plus", "85_plus", "85_plus"),
}
MARITAL_STATUSES = ["never_married", "married", "previously_married"]
EDUCATION_LEVELS = [
    "less_than_9th",
    "high_school",
    "some_college",
    "bachelors",
    "graduate",
]
BACHELORS_FIELDS = ["no_degree", "stem", "business", "arts_humanities"]
EDUCATION_LEVEL_BACHELORS = [
    "less_than_9th",
    "high_school",
    "some_college",
    *[
        f"{education_level}-{bachelors_field}"
        for education_level in ["bachelors", "graduate"]
        for bachelors_field in BACHELORS_FIELDS
    ],
]
OCCUPATIONS = [
    "not_in_workforce",
    "technical_role",
    "care_role",
    "operations_role",
]
DETAILED_OCCUPATIONS = [
    "not_in_workforce",
    "systems_specialist",
    "care_coordinator",
    "field_operator",
    "service_associate",
]
OCCUPATION_DETAIL_WEIGHTS = {
    "not_in_workforce": [80, 1, 1, 1, 1],
    "technical_role": [1, 80, 5, 10, 4],
    "care_role": [1, 4, 80, 5, 10],
    "operations_role": [1, 8, 5, 75, 11],
}
RACE_BROAD = ["broad_alpha", "broad_beta", "broad_gamma"]
RACE_GROUPS = ["background_alpha", "background_beta", "background_gamma"]
ETHNICITIES = ["hispanic or latino", "not hispanic nor latino"]
ETHNICITY_ORIGINS = ["origin_alpha", "origin_beta", "not_applicable"]
ETHNIC_BACKGROUNDS = [
    "background_alpha",
    "background_beta",
    "background_gamma",
    "origin_alpha",
    "origin_beta",
    "not_applicable",
]
LAST_NAMES = [
    "DUMMYLASTA",
    "DUMMYLASTB",
    "DUMMYLASTC",
    "DUMMYLASTD",
    "DUMMYLASTE",
    "DUMMYLASTF",
    "DUMMYLASTG",
    "DUMMYLASTH",
]
FIRST_NAMES = [
    "DUMMYFIRSTA",
    "DUMMYFIRSTB",
    "DUMMYFIRSTC",
    "DUMMYFIRSTD",
    "DUMMYFIRSTE",
    "DUMMYFIRSTF",
    "DUMMYFIRSTG",
    "DUMMYFIRSTH",
]
MIDDLE_NAMES = [
    "DUMMYMIDDLEA",
    "DUMMYMIDDLEB",
    "DUMMYMIDDLEC",
    "DUMMYMIDDLED",
    "DUMMYMIDDLEE",
    "DUMMYMIDDLEF",
    "DUMMYMIDDLEG",
    "DUMMYMIDDLEH",
]
STREET_NAMES = ["Example Way", "Synthetic St", "Placeholder Rd", "Fixture Ave"]
UNITS = ["", "Unit 1", "Unit 2", "Suite A"]
CITIES = ["Exampleton", "Demoville", "Sample City", "Fixture Falls"]
STATES = ["AA", "AA", "BB", "BB"]
COUNTIES = [
    ["Alpha County", "North Alpha County"],
    ["West Alpha County", "South Alpha County"],
    ["Beta County", "Lake Beta County"],
    ["Rural Beta County", "East Beta County"],
]


def _cat(values: list[str], categories: list[str]) -> pd.Categorical:
    return pd.Categorical(values, categories=categories)


def _counts(size: int, low: int = 25, high: int = 500) -> np.ndarray:
    return RNG.integers(low, high, size=size)


def _weighted_matrix(rows: list[str], columns: pd.Index, scale: float) -> pd.DataFrame:
    matrix = RNG.random((len(rows), len(columns))) + 0.05
    for i in range(len(rows)):
        matrix[i, i % len(columns)] += scale
    return pd.DataFrame(matrix, index=pd.CategoricalIndex(rows), columns=columns)


def _total_dist() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "zipcode": _cat(ZIPCODES, ZIPCODES),
            "zipcode_category": _cat(
                [ZIPCODE_TO_CATEGORY[z] for z in ZIPCODES], ZIPCODE_CATEGORIES
            ),
            "zip_prefix4": _cat([ZIPCODE_TO_PREFIX[z] for z in ZIPCODES], ZIP_PREFIXES),
            "count": _counts(len(ZIPCODES), 2_000, 9_000),
        }
    )


def _age_group_map() -> pd.DataFrame:
    rows = [AGE_GROUP_MAPPINGS[age_group] for age_group in AGE_GROUPS]
    return pd.DataFrame(
        {
            "age_group": AGE_GROUPS,
            "age_group_marital": _cat([r[0] for r in rows], AGE_GROUP_MARITAL),
            "age_group_education": _cat([r[1] for r in rows], AGE_GROUP_EDUCATION),
            "age_group_bachelors": _cat([r[2] for r in rows], AGE_GROUP_BACHELORS),
            "age_group_occupation": _cat([r[3] for r in rows], AGE_GROUP_OCCUPATION),
        }
    )


def _age_dist() -> pd.DataFrame:
    rows = list(product(AGE_GROUPS, SEXES))
    return pd.DataFrame(
        {
            "age_group": _cat([r[0] for r in rows], AGE_GROUPS),
            "sex": _cat([r[1] for r in rows], SEXES),
            "age": [AGE_BY_GROUP[r[0]] for r in rows],
            "count": _counts(len(rows), 100, 1_200),
        }
    )


def _age_sex_dist() -> pd.DataFrame:
    rows = list(product(ZIPCODES, AGE_GROUPS, SEXES))
    return pd.DataFrame(
        {
            "zipcode": _cat([r[0] for r in rows], ZIPCODES),
            "age_group": _cat([r[1] for r in rows], AGE_GROUPS),
            "sex": _cat([r[2] for r in rows], SEXES),
            "count": _counts(len(rows), 20, 900),
        }
    )


def _city_state_dist() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "zipcode": _cat(ZIPCODES, ZIPCODES),
            "city": _cat(CITIES, CITIES),
            "state": _cat(STATES, sorted(set(STATES))),
        }
    )


def _county_dist() -> pd.DataFrame:
    rows = [
        (zipcode, county)
        for zipcode, counties in zip(ZIPCODES, COUNTIES)
        for county in counties
    ]
    counties = sorted({county for _, county in rows})
    return pd.DataFrame(
        {
            "zipcode": [r[0] for r in rows],
            "county": pd.Categorical([r[1] for r in rows], categories=counties),
        }
    )


def _marital_dist() -> pd.DataFrame:
    rows = list(product(ZIPCODES, SEXES, AGE_GROUP_MARITAL, MARITAL_STATUSES))
    return pd.DataFrame(
        {
            "zipcode": _cat([r[0] for r in rows], ZIPCODES),
            "sex": _cat([r[1] for r in rows], SEXES),
            "age_group_marital": _cat([r[2] for r in rows], AGE_GROUP_MARITAL),
            "marital_status": _cat([r[3] for r in rows], MARITAL_STATUSES),
            "count": _counts(len(rows), 1, 400),
        }
    )


def _education_dist() -> pd.DataFrame:
    rows = list(product(ZIPCODES, SEXES, AGE_GROUP_EDUCATION, EDUCATION_LEVELS))
    return pd.DataFrame(
        {
            "zipcode": _cat([r[0] for r in rows], ZIPCODES),
            "sex": _cat([r[1] for r in rows], SEXES),
            "age_group_education": _cat([r[2] for r in rows], AGE_GROUP_EDUCATION),
            "education_level": _cat([r[3] for r in rows], EDUCATION_LEVELS),
            "count": _counts(len(rows), 1, 500),
        }
    )


def _bachelors_dist() -> pd.DataFrame:
    rows = list(product(ZIPCODES, SEXES, AGE_GROUP_BACHELORS, BACHELORS_FIELDS))
    return pd.DataFrame(
        {
            "zipcode": _cat([r[0] for r in rows], ZIPCODES),
            "sex": _cat([r[1] for r in rows], SEXES),
            "age_group_bachelors": _cat([r[2] for r in rows], AGE_GROUP_BACHELORS),
            "bachelors_field": _cat([r[3] for r in rows], BACHELORS_FIELDS),
            "count": _counts(len(rows), 1, 350),
        }
    )


def _occupation_dist() -> pd.DataFrame:
    rows = list(product(EDUCATION_LEVEL_BACHELORS, AGE_GROUP_OCCUPATION, OCCUPATIONS))
    counts = []
    for education, age_group, occupation in rows:
        base = 200
        if age_group == "under_18" and occupation == "not_in_workforce":
            base = 2_000
        elif "bachelors" in education and occupation == "technical_role":
            base = 1_000
        elif (
            education in {"high_school", "some_college"}
            and occupation == "operations_role"
        ):
            base = 900
        counts.append(base + int(RNG.integers(1, 300)))
    return pd.DataFrame(
        {
            "education_level_bachelors": _cat(
                [r[0] for r in rows], EDUCATION_LEVEL_BACHELORS
            ),
            "age_group_occupation": _cat([r[1] for r in rows], AGE_GROUP_OCCUPATION),
            "occupation": _cat([r[2] for r in rows], OCCUPATIONS),
            "count": np.array(counts, dtype=float),
        }
    )


def _detailed_occupation_dist() -> pd.DataFrame:
    rows = list(product(OCCUPATIONS, SEXES, DETAILED_OCCUPATIONS))
    return pd.DataFrame(
        {
            "occupation": _cat([r[0] for r in rows], OCCUPATIONS),
            "sex": _cat([r[1] for r in rows], SEXES),
            "detailed_occupation": _cat([r[2] for r in rows], DETAILED_OCCUPATIONS),
            "count": [
                OCCUPATION_DETAIL_WEIGHTS[occupation][
                    DETAILED_OCCUPATIONS.index(detailed)
                ]
                + int(RNG.integers(0, 10))
                for occupation, _, detailed in rows
            ],
        }
    )


def _race_dist() -> pd.DataFrame:
    rows = list(product(ZIP_PREFIXES, RACE_BROAD, RACE_GROUPS))
    return pd.DataFrame(
        {
            "zip_prefix4": _cat([r[0] for r in rows], ZIP_PREFIXES),
            "race_broad": _cat([r[1] for r in rows], RACE_BROAD),
            "race_group": _cat([r[2] for r in rows], RACE_GROUPS),
            "count": _counts(len(rows), 1, 700),
        }
    )


def _ethnicity_total_dist() -> pd.DataFrame:
    rows = list(product(ZIP_PREFIXES, ETHNICITIES))
    return pd.DataFrame(
        {
            "zip_prefix4": _cat([r[0] for r in rows], ZIP_PREFIXES),
            "ethnicity": _cat([r[1] for r in rows], ETHNICITIES),
            "count": _counts(len(rows), 100, 1_500),
        }
    )


def _ethnicity_dist() -> pd.DataFrame:
    rows = list(product(ZIP_PREFIXES, ETHNICITIES, ETHNICITY_ORIGINS))
    return pd.DataFrame(
        {
            "zip_prefix4": _cat([r[0] for r in rows], ZIP_PREFIXES),
            "ethnicity": _cat([r[1] for r in rows], ETHNICITIES),
            "origin": _cat([r[2] for r in rows], ETHNICITY_ORIGINS),
            "count": _counts(len(rows), 1, 500),
        }
    )


def _race_ethnicity_dist() -> pd.DataFrame:
    rows = list(product(ZIP_PREFIXES, ETHNICITIES, RACE_BROAD))
    return pd.DataFrame(
        {
            "zip_prefix4": _cat([r[0] for r in rows], ZIP_PREFIXES),
            "ethnicity": _cat([r[1] for r in rows], ETHNICITIES),
            "race_broad": _cat([r[2] for r in rows], RACE_BROAD),
            "count": _counts(len(rows), 1, 700),
        }
    )


def _last_name_dist() -> pd.DataFrame:
    columns = pd.Index(ETHNIC_BACKGROUNDS)
    return _weighted_matrix(LAST_NAMES, columns, scale=2.0)


def _first_name_dist() -> pd.DataFrame:
    columns = pd.MultiIndex.from_product(
        [ETHNIC_BACKGROUNDS, SEXES], names=[None, "group"]
    )
    return _weighted_matrix(FIRST_NAMES, columns, scale=1.5)


def _middle_name_dist() -> pd.DataFrame:
    columns = pd.MultiIndex.from_product(
        [ETHNIC_BACKGROUNDS, SEXES], names=[None, "group"]
    )
    return _weighted_matrix(MIDDLE_NAMES, columns, scale=1.2)


def _street_name_dist() -> pd.DataFrame:
    rows = list(product(ZIPCODE_CATEGORIES, STREET_NAMES))
    return pd.DataFrame(
        {
            "zipcode_category": _cat([r[0] for r in rows], ZIPCODE_CATEGORIES),
            "street_name": _cat([r[1] for r in rows], STREET_NAMES),
            "count": _counts(len(rows), 1, 250),
        }
    )


def _unit_dist() -> pd.DataFrame:
    rows = list(product(ZIPCODE_CATEGORIES, UNITS))
    return pd.DataFrame(
        {
            "zipcode_category": _cat([r[0] for r in rows], ZIPCODE_CATEGORIES),
            "unit": _cat([r[1] for r in rows], UNITS),
            "count": _counts(len(rows), 1, 300),
        }
    )


def build_dummy_person_data() -> USPersonData:
    return USPersonData(
        total_dist=_total_dist(),
        age_dist=_age_dist(),
        age_group_map=_age_group_map(),
        age_sex_dist=_age_sex_dist(),
        city_state_dist=_city_state_dist(),
        county_dist=_county_dist(),
        marital_dist=_marital_dist(),
        education_dist=_education_dist(),
        bachelors_dist=_bachelors_dist(),
        detailed_occupation_dist=_detailed_occupation_dist(),
        occupation_dist=_occupation_dist(),
        race_dist=_race_dist(),
        ethnicity_dist=_ethnicity_dist(),
        ethnicity_total_dist=_ethnicity_total_dist(),
        race_ethnicity_dist=_race_ethnicity_dist(),
        last_name_dist=_last_name_dist(),
        first_name_dist=_first_name_dist(),
        middle_name_dist=_middle_name_dist(),
        street_name_dist=_street_name_dist(),
        unit_dist=_unit_dist(),
    )


def write_dummy_data(output_dir: Path = OUTPUT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    person_data = build_dummy_person_data()
    person_data.to_tarfile(
        output_dir / DUMMY_PERSON_DATA_FILENAME, PERSON_DATA_TARFILE_PATHS
    )
    pd.DataFrame(
        {
            "zipcode": ZIPCODES,
            "area_code": ["201", "202", "301", "302"],
            "count": _counts(len(ZIPCODES), 500, 5_000),
        }
    ).to_parquet(output_dir / DUMMY_AREA_CODE_FILENAME)


if __name__ == "__main__":
    write_dummy_data()
