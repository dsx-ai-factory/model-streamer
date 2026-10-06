ARCH := $(shell uname -m)
PLATFORM := manylinux_2_28_${ARCH}
PACKAGE_VERSION ?= 0.0.0

STREAMER_MOCK_LIB = ../../cpp/bazel-bin/mock/libstreamer-mock.so

# The real library, for test-unit-real. Same bazel output tree as the mock, since both are built with
# --config=${ARCH}, so the two can sit side by side and neither rebuild disturbs the other.
STREAMER_REAL_LIB = ../../cpp/bazel-bin/streamer/libstreamer.so

# Which filesystem reader test-unit-real uses. Always keep sync_buffered LAST as a fallback, and never
# repeat a name: a duplicate is rejected by the parser, and then every test that builds a streamer
# fails at once, which looks like a product fault and is not one.
FS_STRATEGY ?= sync_buffered

.PHONY: install_dev_dep build mock real test test-unit test-unit-real test-unit-real-installed test-dist clean

install_dev_dep:
	python3 -m pip install --require-hashes --only-binary=:all: -r ../../.devcontainer/dependencies/requirements.lock

build:
	python3 setup.py bdist_wheel --plat-name ${PLATFORM}

mock:
	make -C ../../cpp build_mock

real:
	make -C ../../cpp build

test-unit: mock install_dev_dep
	STREAMER_LIBRARY=$(STREAMER_MOCK_LIB) python3 -m unittest discover -b runai_model_streamer

# The same tests against the REAL library instead of the mock.
#
# Not part of `make -C py test`, which runs the same tests against the mock. The repository root runs
# it once per file system strategy, through `make test_strategies` - which `make test` includes, so CI
# reaches it.
#
# Why it earns its place: the mock's runai_set_fs_strategy is a stub that returns 0, and the mock
# never reads RUNAI_STREAMER_FS_STRATEGY and performs no I/O. So `test-unit` cannot reach any reader,
# and a green from it says nothing about io_uring, about O_DIRECT, or about how a read failure is
# reported. A bug where zero-sized ranges on a missing file returned Success was invisible to the
# whole suite until these tests ran against the real library.
test-unit-real: real install_dev_dep

# CI supplies STREAMER_REAL_LIB from its installed wheel and has already installed
# development dependencies via ci-install. Run the identical tests without rebuilding.
test-unit-real test-unit-real-installed:
	STREAMER_LIBRARY=$(STREAMER_REAL_LIB) RUNAI_STREAMER_FS_STRATEGY=$(FS_STRATEGY) \
		python3 -m unittest discover -b runai_model_streamer

test-dist: mock install_dev_dep
	torchrun --nproc_per_node 2 run_dist_tests.py

test: test-unit test-dist

clean:
	rm -rf build/ dist/ runai_model_streamer.egg-info/

install: install_dev_dep
	pip3 uninstall -y runai_model_streamer && \
	pip3 install dist/runai_model_streamer-${PACKAGE_VERSION}-py3-none-${PLATFORM}.whl
