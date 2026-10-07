# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

# A blocking FIFO on a mutex and a semaphore. Its own package because it has users that want a
# queue and not a thread pool - ThreadPool and DrainingWorker are both built on it.
runai_cc_auto_library(
    name = "deque",
    deps = [
        "//utils/logging",
        "//utils/semaphore",
    ],
)

runai_cc_test(
    name = "deque_test",
    srcs = ["deque_test.cc"],
    deps = [":deque"],
)
