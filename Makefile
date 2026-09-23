X86_64_ARCH := x86_64
AARCH64_ARCH := aarch64
PACKAGE_VERSION ?= 0.0.0
SDK_ARCHIVE ?=
SDK_NATIVE_ARCHIVE = sdk/dist/runai-model-streamer-sdk-$(patsubst v%,%,$(PACKAGE_VERSION))-linux-$(shell uname -m).tar.gz

.PHONY: build build_aarch64 build_x86_64 test test-python test-sdk test_strategies install

build_x86_64:
	make -C cpp build ARCH=${X86_64_ARCH} && \
	make -C py build ARCH=${X86_64_ARCH} && \
	python3 sdk/package.py --version "$(PACKAGE_VERSION)" --arch $(X86_64_ARCH)

build_aarch64:
	make -C cpp build ARCH=${AARCH64_ARCH} && \
	make -C py build ARCH=${AARCH64_ARCH} && \
	python3 sdk/package.py --version "$(PACKAGE_VERSION)" --arch $(AARCH64_ARCH)

build: 
	make -C py clean && \
	make build_x86_64 && \
	make build_aarch64

install: build
	make -C py install

test: install
	make -C cpp test && \
	$(MAKE) test-python && \
	$(MAKE) -C py/runai_model_streamer test-dist && \
	$(MAKE) test-sdk SDK_ARCHIVE="$(SDK_NATIVE_ARCHIVE)" && \
	make -C tests all && \
	make test_strategies

test-python:
	$(MAKE) -C py/runai_model_streamer test-unit

# Standalone SDK testing builds the native libraries without installing Python
# packages. CI can supply an already packaged archive to test the exact release
# artifact without rebuilding it.
test-sdk:
	python3 -m unittest discover -s sdk -p 'test_*.py'
ifneq ($(SDK_ARCHIVE),)
	bash sdk/verify.sh "$(SDK_ARCHIVE)"
else
	$(MAKE) -C cpp build ARCH=$(shell uname -m)
	python3 sdk/package.py --version "$(PACKAGE_VERSION)" --arch $(shell uname -m)
	bash sdk/verify.sh "$(SDK_NATIVE_ARCHIVE)"
endif

# Filesystem read strategies, every one we support. Each list keeps sync_buffered last as the
# fallback, and never repeats a name - a duplicate is rejected by the parser, and then every test
# that builds a streamer fails together, which looks like a product fault and is not one.
FS_STRATEGIES := sync_buffered \
                 libaio_direct,sync_buffered \
                 io_uring_direct,sync_buffered \
                 io_uring_buffered,sync_buffered

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
	    make -C py/runai_model_streamer test-unit-real FS_STRATEGY=$$strategy && \
	    make -C tests fuzzing FS_STRATEGY=$$strategy RUN_TIMES=3 || exit 1; \
	done
