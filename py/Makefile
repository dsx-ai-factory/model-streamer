# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

.PHONY: build test clean

build:
	make -C runai_model_streamer build
	make -C runai_model_streamer_gcs build
	make -C runai_model_streamer_s3 build
	make -C runai_model_streamer_azure build

test:
	make -C runai_model_streamer test

clean:
	make -C runai_model_streamer clean
	make -C runai_model_streamer_gcs clean
	make -C runai_model_streamer_s3 clean
	make -C runai_model_streamer_azure clean

install:
	make -C runai_model_streamer install
	make -C runai_model_streamer_gcs install
	make -C runai_model_streamer_s3 install
	make -C runai_model_streamer_azure install
