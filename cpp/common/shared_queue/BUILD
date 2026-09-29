# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

runai_cc_auto_library(
    name = "shared_queue",
    deps = [
        "//utils/semaphore",
        "//utils/logging",
    ],
)

runai_cc_test(
    name = "shared_queue_test",
    srcs = ["shared_queue_test.cc"],
    deps = [":shared_queue",
            "//utils/threadpool",
            "//utils/thread",
            "//utils/random",
            "//common/response",
    ],
)
