# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from setuptools import setup, find_packages

VERSION = os.getenv("PACKAGE_VERSION", "0.0.0")
LIB = "libstreamergcs.so"


def assert_lib_exists():
    lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), LIB)
    if os.path.islink(lib_path):
        target_path = os.path.realpath(lib_path)
        if not os.path.exists(target_path):
            raise FileNotFoundError(
                f"{target_path} (target of the symlink) not found. Aborting build."
            )
    else:
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"{lib_path} not found. Aborting build.")


assert_lib_exists()
setup(
    name="runai-model-streamer-gcs",
    version=VERSION,
    license_files=("LICENSE", "THIRD_PARTY_NOTICES.txt"),
    packages=find_packages(),
    install_requires=["google-cloud-storage", "google-auth"],
    data_files=[("/runai_model_streamer/libstreamer/lib/", [LIB])],
)
