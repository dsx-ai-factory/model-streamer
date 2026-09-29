# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

runai_cc_auto_library(
    name = "mount_capabilities",
    deps = [
        "//common/exception",
        "//common/response_code",
        # DirectBlockSize: the probe reads one block at the alignment routing assumes, so it must use
        # the same constant rather than writing 4096 again - a disagreement here would be invisible.
        "//posix_io/alignment",
        "//utils/logging",
    ],
)

runai_cc_test(
    name = "mount_capabilities_test",
    srcs = ["mount_capabilities_test.cc"],
    deps = [
        ":mount_capabilities",
        "//posix_io/alignment",
        "//utils/random",
        "//utils/temp/file",
    ],
)
