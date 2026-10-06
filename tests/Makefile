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

# The C++ device test: real object storage into GPU memory, through the public C API. Built by bazel,
# so it is a path rather than a pytest target, and it SKIPS unless it finds both a GPU and an object.
#
# OVERRIDABLE, because the default rarely works as-is: bazel-bin is a symlink into the build
# container's cache, and that container usually has no GPU, while the host with the GPU has no
# bazel-bin. Build there, copy the binary here, and point at it:
#
#   make s3_device DEVICE_TEST=/path/to/object_device_test
DEVICE_TEST ?= ../cpp/bazel-bin/streamer/impl/device_io/tests/object_device_test
DEVICE_FIXTURE := /tmp/runai-object-fixture.bin
DEVICE_OBJECT := device-fixture.bin

# Where the S3 plugin is. The streamer dlopens it by BARE NAME, so without a directory the
# loader can search, every range fails with S3NotSupported - and gtest then reports the mismatched
# bytes, which reads like data corruption rather than a missing library.
#
# Overridable for the same reason DEVICE_TEST is: on a host with no bazel-bin these point nowhere, so
# copy the plugin out and name its directory.
S3_PLUGIN_DIR ?= ../cpp/bazel-bin/s3

.PHONY: gcs s3 azure fuzzing all s3_device require_device

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
		if [ -n "$$PYTEST_PID" ]; then \
			kill $$PYTEST_PID 2>/dev/null || true; \
			wait $$PYTEST_PID 2>/dev/null || true; \
		fi; \
		if [ -n "$$WEED_PID" ]; then \
			kill -INT $$WEED_PID 2>/dev/null || true; \
			i=0; while kill -0 $$WEED_PID 2>/dev/null && [ $$i -lt 50 ]; do sleep 0.1; i=$$((i+1)); done; \
			kill -9 $$WEED_PID 2>/dev/null || true; \
			wait $$WEED_PID 2>/dev/null || true; \
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
	test_status=$$?; \
	echo "Stopping Fake GCS Server..."; \
	kill $$FAKE_GCS_SERVER_PID; \
	wait $$FAKE_GCS_SERVER_PID 2>/dev/null || true; \
	rm -rf ${FAKE_GCS_SERVER_ROOT}; \
	exit $$test_status

azure:
# Developer note: To enable C++ debug logging, add these env vars to the pytest command:
#   RUNAI_STREAMER_LOG_LEVEL=DEBUG RUNAI_STREAMER_LOG_FILE=/tmp/azure_cpp.log
# Build Azure library with AZURITE_TESTING flag enabled
	@echo "Building Azure library with Azurite testing support..."; \
	make -C ../cpp build_azure_test || exit $$?; \
	echo "Starting Azurite server on port ${AZURITE_API_PORT}"; \
	azurite --silent --skipApiVersionCheck --location ${AZURITE_SERVER_WORKSPACE} --blobPort ${AZURITE_API_PORT} & \
	AZURITE_PID=$$!; \
	echo "Azurite PID is $$AZURITE_PID"; \
	LD_LIBRARY_PATH="../cpp/bazel-bin/azure:$${LD_LIBRARY_PATH}" AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=http;AccountName=${AZURITE_ACCOUNT_NAME};AccountKey=${AZURITE_ACCOUNT_KEY};BlobEndpoint=http://127.0.0.1:${AZURITE_API_PORT}/${AZURITE_ACCOUNT_NAME};" AZURE_CONTAINER=${AZURITE_CONTAINER} pytest -v azure/ 2>&1; \
	test_status=$$?; \
	echo "Stopping Azurite..."; \
	kill $$AZURITE_PID; \
	wait $$AZURITE_PID 2>/dev/null || true; \
	rm -rf ${AZURITE_SERVER_WORKSPACE}; \
	exit $$test_status

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
		${FS_ENV} pytest fuzzing/ || exit $$?; \
	done

all: fuzzing s3 gcs azure

# --- object storage to a device, byte-verified ------------------------------------------------------
#
# The Python suites read object storage to HOST memory and check the tensors. This reads it to DEVICE
# memory through the C API and checks the bytes - the one thing the s3 mock can never answer, because
# it never writes into the destination.
#
# The fixture is written here and uploaded by the same tool the suite already needs, so the expected
# bytes have exactly one definition: the file on disk.
#
# Needs a GPU, which is why this is a separate target rather than part of `make all`.
#
# A PRE-FLIGHT, not the real guard. It fails early so a machine with no GPU does not start an
# emulator and upload a 32 MB fixture before finding out.
#
# It cannot be the guard, because nvidia-smi is a separate NVML binary: it ignores
# CUDA_VISIBLE_DEVICES, and it says nothing about whether the TEST PROCESS can load libcuda or open a
# device. Each of those still skips, and a skipped gtest exits 0. What makes the target honest is
# RUNAI_TEST_REQUIRE_DEVICE=1 in the recipe below, which turns those skips into failures.
require_device:
	@test -x "$(DEVICE_TEST)" || { \
		echo "$(DEVICE_TEST) is missing - build it first: (cd ../cpp && bazel build //streamer/impl/device_io/tests:object_device_test)"; \
		exit 1; }
	@nvidia-smi -L 2>/dev/null | grep -q GPU || { \
		echo "no GPU listed here. This target reads object storage into DEVICE memory, so there is"; \
		echo "nothing for it to do."; \
		exit 1; }

$(DEVICE_FIXTURE):
	head -c 33554432 /dev/urandom > $(DEVICE_FIXTURE)

s3_device: require_device $(DEVICE_FIXTURE)
	@set -e; \
	S3_TEST_DATA_DIR=$$(mktemp -d); \
	echo "Starting SeaweedFS (weed mini) on port ${S3_TEST_API_PORT} with data directory $$S3_TEST_DATA_DIR"; \
	mkdir -p $$S3_TEST_DATA_DIR/${S3_TEST_BUCKET}; \
	S3_CONFIG=$$(mktemp); \
	printf '{"identities":[{"name":"admin_user","credentials":[{"accessKey":"${S3_TEST_ACCESS_KEY}","secretKey":"${S3_TEST_SECRET_KEY}"}],"actions":["Admin","Read","Write","List","Tagging"]}]}' > $$S3_CONFIG; \
	cleanup() { \
		rc=$$?; \
		echo "Stopping weed..."; \
		if [ -n "$$WEED_PID" ]; then \
			kill -INT $$WEED_PID 2>/dev/null || true; \
			i=0; while kill -0 $$WEED_PID 2>/dev/null && [ $$i -lt 50 ]; do sleep 0.1; i=$$((i+1)); done; \
			kill -9 $$WEED_PID 2>/dev/null || true; \
			wait $$WEED_PID 2>/dev/null || true; \
		fi; \
		rm -f $$S3_CONFIG; \
		rm -rf $$S3_TEST_DATA_DIR; \
		exit $$rc; \
	}; \
	trap cleanup EXIT INT TERM; \
	weed mini -dir=$$S3_TEST_DATA_DIR -bucket=${S3_TEST_BUCKET} -s3.port=${S3_TEST_API_PORT} -s3.config=$$S3_CONFIG & \
	WEED_PID=$$!; \
	echo "weed PID is $$WEED_PID"; \
	i=0; \
	until AWS_ENDPOINT_URL=http://127.0.0.2:${S3_TEST_API_PORT} AWS_ACCESS_KEY_ID=${S3_TEST_ACCESS_KEY} AWS_SECRET_ACCESS_KEY=${S3_TEST_SECRET_KEY} \
	      python3 -c "import boto3,os; boto3.client('s3', endpoint_url=os.environ['AWS_ENDPOINT_URL']).upload_file('$(DEVICE_FIXTURE)', '${S3_TEST_BUCKET}', '$(DEVICE_OBJECT)')" 2>/dev/null; do \
		i=$$((i+1)); \
		test $$i -lt 150 || { echo "weed did not accept the fixture within 30s"; exit 1; }; \
		sleep 0.2; \
	done; \
	LD_LIBRARY_PATH="$(S3_PLUGIN_DIR):$${LD_LIBRARY_PATH}" \
	RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING=0 \
	AWS_ENDPOINT_URL=http://127.0.0.2:${S3_TEST_API_PORT} AWS_ACCESS_KEY_ID=${S3_TEST_ACCESS_KEY} AWS_SECRET_ACCESS_KEY=${S3_TEST_SECRET_KEY} \
	RUNAI_TEST_OBJECT_URI=s3://${S3_TEST_BUCKET}/$(DEVICE_OBJECT) RUNAI_TEST_OBJECT_FILE=$(DEVICE_FIXTURE) \
	RUNAI_TEST_REQUIRE_DEVICE=1 \
	$(DEVICE_TEST)
