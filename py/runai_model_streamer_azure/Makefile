# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARCH := $(shell uname -m)
PLATFORM := manylinux2014_${ARCH}
PACKAGE_VERSION ?= 0.0.0

.PHONY: build test install_dev_dep clean

install_dev_dep:
	pip3 install azure-storage-blob azure-identity

build:
	python3 setup.py bdist_wheel --plat-name ${PLATFORM}

test: install_dev_dep
	python3 -m unittest discover -b runai_model_streamer_azure

clean:
	rm -rf build/ dist/ runai_model_streamer_azure.egg-info/

install:
	pip3 uninstall -y runai_model_streamer_azure && \
	pip3 install dist/runai_model_streamer_azure-${PACKAGE_VERSION}-py3-none-${PLATFORM}.whl

