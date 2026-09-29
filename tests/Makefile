RUN_TIMES := 10
S3_TEST_API_PORT := 9000
S3_TEST_BUCKET := test-bucket
S3_TEST_ACCESS_KEY := testkey
S3_TEST_SECRET_KEY := testsecret

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
	@set -e; \
	S3_TEST_DATA_DIR=$$(mktemp -d); \
	echo "Starting SeaweedFS (weed mini) on port ${S3_TEST_API_PORT} with data directory $$S3_TEST_DATA_DIR"; \
	mkdir -p $$S3_TEST_DATA_DIR/${S3_TEST_BUCKET}; \
	S3_CONFIG=$$(mktemp); \
	printf '{"identities":[{"name":"admin_user","credentials":[{"accessKey":"${S3_TEST_ACCESS_KEY}","secretKey":"${S3_TEST_SECRET_KEY}"}],"actions":["Admin","Read","Write","List","Tagging"]}]}' > $$S3_CONFIG; \
	cleanup() { \
		rc=$$?; \
		echo "Stopping weed..."; \
		[ -n "$$PYTEST_PID" ] && kill $$PYTEST_PID 2>/dev/null; \
		[ -n "$$PYTEST_PID" ] && wait $$PYTEST_PID 2>/dev/null; \
		if [ -n "$$WEED_PID" ]; then \
			kill -INT $$WEED_PID 2>/dev/null; \
			i=0; while kill -0 $$WEED_PID 2>/dev/null && [ $$i -lt 50 ]; do sleep 0.1; i=$$((i+1)); done; \
			kill -9 $$WEED_PID 2>/dev/null; \
			wait $$WEED_PID 2>/dev/null; \
		fi; \
		rm -f $$S3_CONFIG; \
		rm -rf $$S3_TEST_DATA_DIR; \
		exit $$rc; \
	}; \
	trap cleanup EXIT INT TERM; \
	weed mini -dir=$$S3_TEST_DATA_DIR -bucket=${S3_TEST_BUCKET} -s3.port=${S3_TEST_API_PORT} -s3.config=$$S3_CONFIG & \
	WEED_PID=$$!; \
	echo "weed PID is $$WEED_PID"; \
	RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING=0 AWS_BUCKET=${S3_TEST_BUCKET} AWS_ENDPOINT_URL=http://127.0.0.2:${S3_TEST_API_PORT} AWS_ACCESS_KEY_ID=${S3_TEST_ACCESS_KEY} AWS_SECRET_ACCESS_KEY=${S3_TEST_SECRET_KEY} WEED_PID=$$WEED_PID pytest s3/ & \
	PYTEST_PID=$$!; \
	wait $$PYTEST_PID

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
