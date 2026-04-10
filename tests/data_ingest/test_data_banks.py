# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import tarfile
import tempfile

import pandas as pd
import pytest
from pydantic import BaseModel

from pgms.data_ingest.data_banks import TarFileMixin

DUMMY_TARFILE_PATHS = {
    "a": "a.parquet",
    "b": "b.parquet",
}


class DummyTarData(BaseModel, TarFileMixin):
    a: pd.DataFrame
    b: pd.DataFrame

    class Config:
        arbitrary_types_allowed = True


def create_tarfile_with_dfs(tmp_path, df_a, df_b=None):
    tar_path = tmp_path / "test_archive.tar"
    with tarfile.open(tar_path, "w:") as tar:
        # Write df_a to a temporary parquet file and add it.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as tmp_file_a:
            df_a.to_parquet(tmp_file_a.name)
            tar.add(tmp_file_a.name, arcname="a.parquet")

        if df_b is not None:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".parquet"
            ) as tmp_file_b:
                df_b.to_parquet(tmp_file_b.name)
                tar.add(tmp_file_b.name, arcname="b.parquet")
    os.remove(tmp_file_a.name)
    if df_b is not None:
        os.remove(tmp_file_b.name)
    return tar_path


@pytest.fixture
def sample_data():
    df_a = pd.DataFrame({"col1": [1, 2, 3]})
    df_b = pd.DataFrame({"col2": ["a", "b", "c"]})
    instance = DummyTarData(a=df_a, b=df_b)
    return df_a, df_b, instance


def test_from_tarfile_valid(tmp_path, sample_data):
    """Test that from_tarfile correctly reads a valid tar file."""
    df_a, df_b, _ = sample_data
    tar_path = create_tarfile_with_dfs(tmp_path, df_a, df_b)
    loaded = DummyTarData.from_tarfile(str(tar_path), DUMMY_TARFILE_PATHS)
    pd.testing.assert_frame_equal(loaded.a, df_a)
    pd.testing.assert_frame_equal(loaded.b, df_b)


def test_from_tarfile_missing_subfile(tmp_path, sample_data):
    """Test that from_tarfile raises an error when a subfile is missing."""
    df_a, df_b, _ = sample_data
    tar_path = create_tarfile_with_dfs(tmp_path, df_a)
    with pytest.raises(KeyError):
        DummyTarData.from_tarfile(str(tar_path), DUMMY_TARFILE_PATHS)


def test_to_tarfile_valid(tmp_path, sample_data):
    """Test that to_tarfile creates a tar file with the expected contents."""
    _, _, instance = sample_data
    tar_path = tmp_path / "output.tar."
    instance.to_tarfile(str(tar_path), DUMMY_TARFILE_PATHS)
    with tarfile.open(str(tar_path), "r:") as tar:
        names = tar.getnames()
        assert "a.parquet" in names
        assert "b.parquet" in names

        with tar.extractfile("a.parquet") as a_file:
            loaded_a = pd.read_parquet(a_file)
        with tar.extractfile("b.parquet") as b_file:
            loaded_b = pd.read_parquet(b_file)
        pd.testing.assert_frame_equal(loaded_a, instance.a)
        pd.testing.assert_frame_equal(loaded_b, instance.b)


def test_to_tarfile_missing_attribute(tmp_path, sample_data):
    """Test that to_tarfile raises an AttributeError when an expected attribute is missing."""
    _, _, instance = sample_data
    # Remove attribute 'b'
    delattr(instance, "b")
    tar_path = tmp_path / "output_missing.tar."
    with pytest.raises(AttributeError):
        instance.to_tarfile(str(tar_path), DUMMY_TARFILE_PATHS)


def test_from_tarfile_invalid_path():
    """Test that from_tarfile raises an exception when provided an invalid file path."""
    with pytest.raises(Exception):
        DummyTarData.from_tarfile("nonexistent_file.tar.", DUMMY_TARFILE_PATHS)
