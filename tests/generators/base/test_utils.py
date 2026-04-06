import numpy as np
import pandas as pd
import pytest
import torch

from pgms.generators.base import utils


def test_tensor_normalize_1d():
    """Test normalization on a 1-D numpy array."""
    arr = np.array([1.0, 2.0, 3.0])
    dtype = torch.float32
    device = "cpu"
    result = utils.tensor_normalize(arr, dtype, device)
    # For a 1D tensor, normalization with p=1 divides by the sum of absolute values.
    total = np.abs(arr).sum()  # 6.0
    expected = torch.tensor([1.0 / total, 2.0 / total, 3.0 / total], dtype=dtype)
    assert torch.allclose(result, expected)
    # Also check that the tensor is on the correct device.
    assert result.device.type == device


def test_tensor_normalize_2d():
    """Test normalization on a 2-D numpy array (normalizing along dim=0)."""
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    dtype = torch.float32
    device = "cpu"
    result = utils.tensor_normalize(arr, dtype, device)
    # Normalization is performed column-wise (dim=0):
    # First column: [1, 3] -> norm = |1|+|3| = 4, so normalized to [1/4, 3/4]
    # Second column: [2, 4] -> norm = 2+4 = 6, so normalized to [2/6, 4/6]
    expected = torch.tensor(
        [[1.0 / 4.0, 2.0 / 6.0], [3.0 / 4.0, 4.0 / 6.0]], dtype=dtype
    )
    assert torch.allclose(result, expected)
    assert result.device.type == device


# --- Tests for load_file --- #


def test_load_file_csv(tmp_path):
    """Test loading a CSV file."""
    # Create a simple CSV file.
    csv_content = "a,b\n1,2\n3,4"
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(csv_content)

    identifier = "csv_test"
    file_info = (identifier, str(csv_file))
    returned_id, df = utils.load_file(file_info)

    # Build the expected DataFrame.
    expected_df = pd.DataFrame({"a": [1, 3], "b": [2, 4]})
    pd.testing.assert_frame_equal(df, expected_df)
    assert returned_id == identifier


def test_load_file_parquet(tmp_path):
    """Test loading a Parquet file."""
    # Ensure the necessary parquet engine is available; skip if not.
    pytest.importorskip("pyarrow")

    # Create a DataFrame and write it to a Parquet file.
    df_expected = pd.DataFrame({"a": [1, 3], "b": [2, 4]})
    parquet_file = tmp_path / "test.parquet"
    df_expected.to_parquet(parquet_file)

    identifier = "parquet_test"
    file_info = (identifier, str(parquet_file))
    returned_id, df = utils.load_file(file_info)

    pd.testing.assert_frame_equal(df, df_expected)
    assert returned_id == identifier
