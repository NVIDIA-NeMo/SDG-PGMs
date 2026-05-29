# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
from pydantic import BaseModel, ConfigDict

from pgms.data_ingest.data_banks import TarFileMixin


class USPersonData(BaseModel, TarFileMixin):
    """Container for US person count distributions."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    total_dist: pd.DataFrame
    age_dist: pd.DataFrame
    age_group_map: pd.DataFrame
    age_sex_dist: pd.DataFrame
    city_state_dist: pd.DataFrame
    county_dist: pd.DataFrame
    marital_dist: pd.DataFrame
    education_dist: pd.DataFrame
    bachelors_dist: pd.DataFrame
    detailed_occupation_dist: pd.DataFrame
    occupation_dist: pd.DataFrame
    race_dist: pd.DataFrame
    ethnicity_dist: pd.DataFrame
    ethnicity_total_dist: pd.DataFrame
    race_ethnicity_dist: pd.DataFrame
    last_name_dist: pd.DataFrame
    first_name_dist: pd.DataFrame
    middle_name_dist: pd.DataFrame
    street_name_dist: pd.DataFrame
    unit_dist: pd.DataFrame
