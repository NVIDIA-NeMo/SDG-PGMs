# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch.nn.functional import normalize


def tensor_normalize(
    array: np.ndarray, dtype: torch.dtype, device: str
) -> torch.Tensor:
    """
    Normalize a numpy array and convert it to a PyTorch tensor.

    Args:
        array: Numpy array to normalize
        dtype: PyTorch data type to convert to.
        device: Device to move the tensor to.

    Returns:
        Normalized PyTorch tensor.
    """
    array = torch.Tensor(array).type(dtype).to(device)
    array = normalize(array, p=1, dim=0)
    return array


def load_file(file_info: tuple[str, str]) -> tuple[str, pd.DataFrame]:
    """
    Load a file (either parquet or csv) and return it with its identifier.

    Args:
        file_info: Tuple of (identifier, file_path)

    Returns:
        Tuple of (identifier, DataFrame)
    """
    identifier, file_path = file_info
    if file_path.endswith(".parquet"):
        return identifier, pd.read_parquet(file_path)
    return identifier, pd.read_csv(file_path)


def bernoulli_ucb(successes, n, confidence=0.95):
    """
    Calculate the upper confidence bound of a Bernoulli processs.
    """
    return stats.beta.ppf(confidence, successes + 1, n - successes)
