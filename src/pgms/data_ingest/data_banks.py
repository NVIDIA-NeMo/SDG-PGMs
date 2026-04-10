# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import tarfile
import tempfile
from typing import IO

import pandas as pd
import smart_open


class TarFileMixin:
    """
    Mixin that provides methods to read from and write to tar files.
    """

    @classmethod
    def from_tarfile(cls, data_file: str | IO, tarfile_subpaths: dict[str, str]):
        try:
            # Open the file using smart_open if a string path is provided.
            file_obj = (
                smart_open.open(data_file, "rb")
                if isinstance(data_file, str)
                else data_file
            )
        except Exception as e:
            print(
                f"Error opening tar file '{data_file}'. Please check the file path and try again."
            )
            raise e
        with file_obj as fd:
            try:
                with tarfile.open("r:", fileobj=fd) as tar:
                    dfs = {}
                    for attr_name, inner_file in tarfile_subpaths.items():
                        try:
                            subfile = tar.extractfile(inner_file)
                        except KeyError:
                            msg = (
                                f"Subfile '{inner_file}' not found in the tar archive. "
                                "Please check the tar file contents."
                            )
                            raise KeyError(msg)
                        dfs[attr_name] = pd.read_parquet(subfile)
            except tarfile.TarError as e:
                print(
                    "Failed to open the tar archive. Please ensure the file is a valid tar.gz archive."
                )
                raise e

        return cls(**dfs)

    def to_tarfile(self, tarfile_path: str, tarfile_subpaths: dict[str, str]):
        try:
            with tarfile.open(tarfile_path, "w:") as tar:
                for attr_name, inner_file in tarfile_subpaths.items():
                    try:
                        df = getattr(self, attr_name)
                    except AttributeError as e:
                        msg = (
                            f"Attribute '{attr_name}' not found in the instance. "
                            "Please ensure it exists before creating the tar archive."
                        )
                        print(msg)
                        raise e

                    # Write the dataframe to a temporary parquet file.
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".parquet"
                    ) as tmp_file:
                        tmp_file_path = tmp_file.name
                        df.to_parquet(tmp_file_path)

                    # Add the file to the tarball under the expected name.
                    tar.add(tmp_file_path, arcname=inner_file)
                    os.remove(tmp_file_path)
        except Exception as e:
            print(
                f"Failed to create tar file at '{tarfile_path}'. Please check the path and try again."
            )
            raise e
