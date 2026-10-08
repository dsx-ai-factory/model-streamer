.PHONY: build build_mock build_azure_test test test_strategy clean

ARCH := $$(uname -m)

build:
	$(MAKE) build_streamer
	$(MAKE) build_s3
	$(MAKE) build_gcs
	$(MAKE) build_azure

.PHONY: build_streamer build_s3 build_gcs build_azure

# Shared by the full build above and individual CI jobs.
build_streamer:
	bazel build streamer:libstreamer.so "--config=${ARCH}"

build_s3:
	bazel build s3:libstreamers3.so --define USE_SYSTEM_LIBS=${USE_SYSTEM_LIBS} "--config=${ARCH}"

build_gcs:
	bazel build gcs:libstreamergcs.so --define USE_SYSTEM_LIBS=${USE_SYSTEM_LIBS} "--config=${ARCH}"

build_azure:
	bazel build azure:libstreamerazure.so --define USE_SYSTEM_LIBS=${USE_SYSTEM_LIBS} "--config=${ARCH}"

# Build Azure library with Azurite testing support (enables AZURE_STORAGE_ACCOUNT_KEY)
build_azure_test:
	bazel build azure:libstreamerazure.so \
		--define USE_SYSTEM_LIBS=${USE_SYSTEM_LIBS} \
		"--config=${ARCH}" \
		--config=azurite

# --config=${ARCH} matches the `build` target above. Without it this lands in a DIFFERENT output tree and
# repoints the bazel-bin symlink there, where libstreamer.so and the backend plugins are stale or absent -
# every later run then fails with `OSError: ... cannot open shared object file`, which reads like a
# product regression and is not one.
build_mock:
	bazel build mock:libstreamer-mock.so \
		"--config=${ARCH}"

# Extra flags for the test run. Empty by default, because a contributor's host may genuinely have no
# io_uring and their tests should skip rather than fail.
#
# CI passes --test_env=RUNAI_STREAMER_REQUIRE_IO_URING=1 (see .github/workflows/on-pr.yaml). GitHub
# runners DO have io_uring - measured on a PR run: kernel.io_uring_disabled is 0, and the container
# has no seccomp filter at all because devcontainer.json's runArgs reach CI. So the io_uring tests
# already run there, and without this flag a skip and a pass are the same green.
BAZEL_TEST_FLAGS ?=

# Show a failing test's output. Bazel defaults to `summary`, which prints the target name and a path
# to a test.log on a CI runner that is then discarded - so a failure arrives with no evidence.
#
# This does NOT make the library's own logs visible. Those go nowhere during a test:
# RUNAI_STREAMER_LOG_TO_STDERR defaults to false and RUNAI_STREAMER_LOG_FILE is unset (logging.cc),
# so every LOG(ERROR) is discarded. Setting either one here does not work - logging_test asserts the
# behaviour of both when UNSET, and both are read once in a static initialiser, so that test cannot
# override them per-case. Making library logs visible needs a per-target setting or a test-side sink.
BAZEL_TEST_LOGGING := --test_output=errors

test:
	bazel test //...:all $(BAZEL_TEST_LOGGING) $(BAZEL_TEST_FLAGS)

.PHONY: test_streamer test_s3 test_gcs test_azure

# Match each component build's architecture and defines to reuse compiled outputs.
# Core owns every target outside the three backend trees, including shared utilities.
test_streamer:
	bazel test "--config=${ARCH}" $(BAZEL_TEST_LOGGING) $(BAZEL_TEST_FLAGS) \
		-- //...:all -//s3/... -//gcs/... -//azure/...

test_s3:
	bazel test //s3/... "--config=${ARCH}" --define USE_SYSTEM_LIBS=${USE_SYSTEM_LIBS} \
		$(BAZEL_TEST_LOGGING) $(BAZEL_TEST_FLAGS)

test_gcs:
	bazel test //gcs/... "--config=${ARCH}" --define USE_SYSTEM_LIBS=${USE_SYSTEM_LIBS} \
		$(BAZEL_TEST_LOGGING) $(BAZEL_TEST_FLAGS)

test_azure:
	bazel test //azure/... "--config=${ARCH}" --define USE_SYSTEM_LIBS=${USE_SYSTEM_LIBS} \
		$(BAZEL_TEST_LOGGING) $(BAZEL_TEST_FLAGS)

# The C++ half of the strategy sweep - see the root Makefile's test_strategies for the whole story.
#
# These two targets and no others. Every other test either pins its own strategy with
# utils::temp::Env, which overrides --test_env, or drives MockIoEngine and never reaches a real
# reader, so re-running them per strategy would repeat identical work.
test_strategy:
	bazel test //streamer:streamer_test //streamer/impl/streamer:streamer_test \
		--test_env=RUNAI_STREAMER_FS_STRATEGY=$(FS_STRATEGY) \
		$(BAZEL_TEST_LOGGING)
