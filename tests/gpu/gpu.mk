# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Entry points of the real-GPU tests (tests/gpu/README.md), included by the root Makefile. They are thin: the options are
# those of tests/gpu/run.sh, and `make gpu-help` lists the targets.

GPU_DIR := $(patsubst %/,%,$(dir $(lastword $(MAKEFILE_LIST))))

# The streamer to test: exactly one of WHEEL (one or more wheel files), WHEEL_DIR and STREAMER_SPEC.
# The rest is optional: SUITES (default streaming), GPUS (1 or 2, default: what the machine has, at most 2), MATRIX
# (core, full or smoke), ROWS, and EXTRA for any other option of run.sh.
GPU_RUN_OPTIONS = \
	$(foreach wheel,$(WHEEL),--wheel '$(wheel)') \
	$(if $(WHEEL_DIR),--wheel-dir '$(WHEEL_DIR)') \
	$(if $(STREAMER_SPEC),--streamer-spec '$(STREAMER_SPEC)') \
	$(if $(SUITES),--suites '$(SUITES)') \
	$(if $(GPUS),--gpus '$(GPUS)') \
	$(if $(MATRIX),--matrix '$(MATRIX)') \
	$(if $(ROWS),--rows '$(ROWS)') \
	$(EXTRA)

.PHONY: gpu-help gpu-test gpu-test-1gpu gpu-test-object-storage gpu-dry-run gpu-unit-tests

gpu-help:
	@echo "make gpu-test WHEEL=PATH            run the real-GPU tests with a wheel (also WHEEL_DIR=DIR or STREAMER_SPEC=SPEC)"
	@echo "make gpu-test-1gpu WHEEL=PATH       the same on one GPU (the rows that need two are left out)"
	@echo "make gpu-test-object-storage ...    the object storage suite only (needs the S3 plugin wheel too: WHEEL='A B')"
	@echo "make gpu-dry-run WHEEL=PATH         show what gpu-test would run, without running it"
	@echo "make gpu-unit-tests                 the CPU unit tests of the GPU tests (no GPU, no container)"
	@echo "Options: SUITES=a,b GPUS=1|2 MATRIX=core|full|smoke ROWS=prefixes EXTRA='--seed 7 ...'; see $(GPU_DIR)/README.md"

gpu-test:
	$(GPU_DIR)/run.sh $(GPU_RUN_OPTIONS)

gpu-test-1gpu: GPUS := 1
gpu-test-1gpu: gpu-test

gpu-test-object-storage: SUITES := object_storage
gpu-test-object-storage: gpu-test

gpu-dry-run:
	$(GPU_DIR)/run.sh --dry-run $(GPU_RUN_OPTIONS)

gpu-unit-tests:
	bash $(GPU_DIR)/harness/unit_tests.sh all
