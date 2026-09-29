# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

RUN_TIMES := 10
MINIO_API_PORT := 9000
MINIO_DATA_DIR := /tmp/minio
MINIO_BUCKET := test-bucket
MINIO_ROOT_USER := minioadmin
MINIO_ROOT_PASSWORD := minioadmin

FAKE_GCS_SERVER_API_PORT := 8000
FAKE_GCS_SERVER_ROOT := /tmp/fake-gcs-server
FAKE_GCS_SERVER_DATA_DIR := ${FAKE_GCS_SERVER_ROOT}/data
FAKE_GCS_SERVER_BUCKET := test-bucket

AZURITE_SERVER_WORKSPACE := /tmp/azurite
AZURITE_API_PORT := 10000
AZURITE_ACCOUNT_NAME := devstoreaccount1
# Well known storage account key https://github.com/Azure/Azurite?tab=readme-ov-file#default-storage-account
AZURITE_ACCOUNT_KEY := Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==
AZURITE_CONTAINER := test-bucket

.PHONY: gcs s3 azure fuzzing all

s3:
	@echo "Starting MinIO server on port ${MINIO_API_PORT} with data directory ${MINIO_DATA_DIR}"; \
	mkdir -p ${MINIO_DATA_DIR}/${MINIO_BUCKET}; \
	MINIO_ROOT_USER=${MINIO_ROOT_USER} MINIO_ROOT_PASSWORD=${MINIO_ROOT_PASSWORD} minio server ${MINIO_DATA_DIR} --address ":${MINIO_API_PORT}" & \
	MINIO_PID=$$!; \
	echo "MinIO PID is $$MINIO_PID"; \
	RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING=0 AWS_BUCKET=${MINIO_BUCKET} AWS_ENDPOINT_URL=http://127.0.0.2:${MINIO_API_PORT} AWS_ACCESS_KEY_ID=${MINIO_ROOT_USER} AWS_SECRET_ACCESS_KEY=${MINIO_ROOT_PASSWORD} pytest s3/; \
	echo "Stopping MinIO..."; \
	kill $$MINIO_PID; \
	wait $$MINIO_PID 2>/dev/null || true; \
	rm -rf ${MINIO_DATA_DIR}

gcs:
	@echo "Starting Fake GCS Server on port ${FAKE_GCS_SERVER_API_PORT} with root directory ${FAKE_GCS_SERVER_ROOT}"; \
	mkdir -p ${FAKE_GCS_SERVER_ROOT}; \
	mkdir -p ${FAKE_GCS_SERVER_DATA_DIR}/${FAKE_GCS_SERVER_BUCKET}; \
	fake-gcs-server -scheme http -port ${FAKE_GCS_SERVER_API_PORT} -data ${FAKE_GCS_SERVER_DATA_DIR} -filesystem-root ${FAKE_GCS_SERVER_ROOT} & \
	FAKE_GCS_SERVER_PID=$$!; \
	echo "Fake GCS Server PID is $$FAKE_GCS_SERVER_PID"; \
	GOOGLE_CLOUD_PROJECT=fake-project GCS_BUCKET=${FAKE_GCS_SERVER_BUCKET} API_ENDPOINT_OVERRIDE=http://localhost:${FAKE_GCS_SERVER_API_PORT} CLOUD_STORAGE_EMULATOR_ENDPOINT=http://localhost:${FAKE_GCS_SERVER_API_PORT} RUNAI_STREAMER_GCS_USE_ANONYMOUS_CREDENTIALS=True pytest gcs/; \
	echo "Stopping Fake GCS Server..."; \
	kill $$FAKE_GCS_SERVER_PID; \
	wait $$FAKE_GCS_SERVER_PID 2>/dev/null || true; \
	rm -rf ${FAKE_GCS_SERVER_ROOT}

azure:
# Developer note: To enable C++ debug logging, add these env vars to the pytest command:
#   RUNAI_STREAMER_LOG_LEVEL=DEBUG RUNAI_STREAMER_LOG_FILE=/tmp/azure_cpp.log
# Build Azure library with AZURITE_TESTING flag enabled
	@echo "Building Azure library with Azurite testing support..."; \
	make -C ../cpp build_azure_test; \
	echo "Starting Azurite server on port ${AZURITE_API_PORT}"; \
	azurite --silent --skipApiVersionCheck --location ${AZURITE_SERVER_WORKSPACE} --blobPort ${AZURITE_API_PORT} & \
	AZURITE_PID=$$!; \
	echo "Azurite PID is $$AZURITE_PID"; \
	LD_LIBRARY_PATH="../cpp/bazel-bin/azure:$${LD_LIBRARY_PATH}" AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=http;AccountName=${AZURITE_ACCOUNT_NAME};AccountKey=${AZURITE_ACCOUNT_KEY};BlobEndpoint=http://127.0.0.1:${AZURITE_API_PORT}/${AZURITE_ACCOUNT_NAME};" AZURE_CONTAINER=${AZURITE_CONTAINER} pytest -v azure/ 2>&1; \
	echo "Stopping Azurite..."; \
	kill $$AZURITE_PID; \
	wait $$AZURITE_PID 2>/dev/null || true; \
	rm -rf ${AZURITE_SERVER_WORKSPACE}

# Which filesystem reader to use. Unset means "leave it to the library", which is what a plain
# `make all` does; the root Makefile's test_strategies target passes each strategy in turn.
#
# Set through this indirection rather than exported directly, because an EMPTY
# RUNAI_STREAMER_FS_STRATEGY is not the same as an unset one - the parser rejects an empty entry
# with InvalidParameterError, so every test would fail at once.
FS_STRATEGY ?=
FS_ENV := $(if $(FS_STRATEGY),RUNAI_STREAMER_FS_STRATEGY=$(FS_STRATEGY),)

fuzzing:
	for i in $(shell seq 1 ${RUN_TIMES}); \
	do \
		echo "Running test iteration $$i"; \
		${FS_ENV} pytest fuzzing/; \
	done

all: fuzzing s3 gcs azure
