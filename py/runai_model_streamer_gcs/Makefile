# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARCH := $(shell uname -m)
PLATFORM := manylinux_2_30_${ARCH}
PACKAGE_VERSION ?= 0.0.0

.PHONY: build clean

build:
	python3 setup.py bdist_wheel --plat-name ${PLATFORM}

test:
	python3 -m unittest discover -b runai_model_streamer_gcs

clean:
	rm -rf build/ dist/ runai_model_streamer_gcs.egg-info/

install:
	pip3 uninstall -y runai_model_streamer_gcs && \
	pip3 install dist/runai_model_streamer_gcs-${PACKAGE_VERSION}-py3-none-${PLATFORM}.whl
