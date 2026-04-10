# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import tarfile
import tempfile
from typing import IO

import pandas as pd
import smart_open

logger = logging.getLogger(__name__)


class TarFileMixin:
    """
    Mixin that provides methods to read from and write to tar files.
    """

    @classmethod
    def from_tarfile(cls, data_file: str | IO, tarfile_subpaths: dict[str, str]):
        try:
            file_obj = (
                smart_open.open(data_file, "rb")
                if isinstance(data_file, str)
                else data_file
            )
        except Exception as e:
            logger.error(
                "Error opening tar file '%s'. Please check the file path and try again.",
                data_file,
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
                logger.error(
                    "Failed to open the tar archive. "
                    "Please ensure the file is a valid tar archive."
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
                        logger.error(
                            "Attribute '%s' not found in the instance. "
                            "Please ensure it exists before creating the tar archive.",
                            attr_name,
                        )
                        raise e

                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".parquet"
                    ) as tmp_file:
                        tmp_file_path = tmp_file.name
                        df.to_parquet(tmp_file_path)

                    tar.add(tmp_file_path, arcname=inner_file)
                    os.remove(tmp_file_path)
        except Exception as e:
            logger.error(
                "Failed to create tar file at '%s'. "
                "Please check the path and try again.",
                tarfile_path,
            )
            raise e
