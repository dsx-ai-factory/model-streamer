X86_64_ARCH := x86_64
AARCH64_ARCH := aarch64

.PHONY: build build_aarch64 build_x86_64 test test_strategies install

build_x86_64:
	make -C cpp build ARCH=${X86_64_ARCH} && \
	make -C py build ARCH=${X86_64_ARCH}

build_aarch64:
	make -C cpp build ARCH=${AARCH64_ARCH} && \
	make -C py build ARCH=${AARCH64_ARCH}

build: 
	make -C py clean && \
	make build_$(shell uname -m)

install: build
	make -C py install

test: install
	make -C cpp test && \
	make -C py test && \
	make -C tests all && \
	make test_strategies

# Filesystem read strategies, every one we support. Each list keeps sync_buffered last as the
# fallback, and never repeats a name - a duplicate is rejected by the parser, and then every test
# that builds a streamer fails together, which looks like a product fault and is not one.
FS_STRATEGIES := sync_buffered \
                 libaio_direct,sync_buffered \
                 io_uring_direct,sync_buffered \
                 io_uring_buffered,sync_buffered

# CI selects the target that reuses its installed wheel.
PYTHON_STRATEGY_TEST ?= test-unit-real

# Run the suites that actually respond to RUNAI_STREAMER_FS_STRATEGY, once per strategy.
#
# Replaces the old `test_iouring` target, which swept only three strategies, never covered
# libaio_direct, and was manual because we believed CI had no io_uring. CI does have it - measured.
#
# Only three suites are swept, because only these three change behaviour with the strategy:
#
#   cpp test_strategy    the two C++ targets that inherit the environment. The rest pin their own
#                        strategy with utils::temp::Env or drive MockIoEngine.
#   test-unit-real       the Python tests against the REAL library. `make -C py test` runs them
#                        against the mock, which stubs the C API and reaches no reader at all.
#   tests fuzzing        the only local-filesystem integration test. tests/s3, tests/gcs and
#                        tests/azure read object storage, which the filesystem strategy never reaches.
#
# Measured at roughly 27s + 21s + 10s for all four strategies together.
test_strategies:
	for strategy in $(FS_STRATEGIES); do \
	    echo "=== filesystem strategy: $$strategy ===" && \
	    make -C cpp test_strategy FS_STRATEGY=$$strategy && \
	    make -C py/runai_model_streamer $(PYTHON_STRATEGY_TEST) FS_STRATEGY=$$strategy && \
	    make -C tests fuzzing FS_STRATEGY=$$strategy RUN_TIMES=3 || exit 1; \
	done

# CI installs wheels downloaded from the architecture build jobs. The local
# `install` and `test` targets above still build from source as before.
COMPONENT ?= streamer
ARCH ?= $(shell uname -m)
CI_PACKAGE_DIR = runai_model_streamer$(if $(filter-out streamer,$(COMPONENT)),_$(COMPONENT))

.PHONY: ci-build ci-build-cpp ci-build-python ci-install ci-test-cpp ci-test-python ci-test-integration

ci-build:
	$(MAKE) ci-build-cpp
	$(MAKE) ci-build-python

ci-build-cpp:
	$(MAKE) -C cpp build_$(COMPONENT) ARCH=$(ARCH)

ci-build-python:
	$(MAKE) -C py/$(CI_PACKAGE_DIR) build ARCH=$(ARCH)

ci-install:
	$(MAKE) -C py install

ci-test-cpp:
	$(MAKE) -C cpp test_$(COMPONENT) ARCH=$(ARCH)

ci-test-python:
	$(MAKE) -C py/runai_model_streamer install
	# Distributed tests use the installed real library; unit tests override it with the mock.
	library="$$(python3 -c 'from runai_model_streamer.libstreamer import DEFAULT_STREAMER_LIBRARY; print(DEFAULT_STREAMER_LIBRARY)')" && \
		STREAMER_LIBRARY="$$library" $(MAKE) -C py test

ci-test-integration: ci-install
	$(MAKE) -C tests all
	library="$$(python3 -c 'from runai_model_streamer.libstreamer import DEFAULT_STREAMER_LIBRARY; print(DEFAULT_STREAMER_LIBRARY)')" && \
		$(MAKE) test_strategies PYTHON_STRATEGY_TEST=test-unit-real-installed STREAMER_REAL_LIB="$$library"
