# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

runai_cc_auto_library(
    name = "semaphore",
    linkopts = ["-ldl"],
    deps = ["//utils/logging"],
)

# Exercise the same timeout and wake-up assertions when the running libc has no
# sem_clockwait, even when CI itself runs on a newer distribution.
runai_cc_test(
    name = "semaphore_fallback_test",
    srcs = ["semaphore_test.cc", "semaphore_fallback_test.cc"],
    linkopts = ["-Wl,--wrap=dlsym"],
    deps = [":semaphore", "//utils/thread", "//utils/random"],
)

runai_cc_test(
    name = "semaphore_test",
    srcs = ["semaphore_test.cc"],
    deps = [
        ":semaphore",
        "//utils/thread",
        "//utils/random",
    ],
)
