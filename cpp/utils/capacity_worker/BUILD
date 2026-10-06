# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

runai_cc_auto_library(
    name = "capacity_worker",
    deps = [
        "//utils/threadpool",
        "//utils/capacity_queue",
    ],
)

runai_cc_test(
    name = "capacity_worker_test",
    srcs = ["capacity_worker_test.cc"],
    deps = [
        ":capacity_worker",
        "//utils/threadpool",
        "//utils/random",
    ],
)
