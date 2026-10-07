# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

runai_cc_auto_library(
    name = "chunk_splitter",
    deps = [
        "//streamer/impl/task",
        "//utils/logging",
    ],
)

runai_cc_test(
    name = "chunk_splitter_test",
    srcs = ["chunk_splitter_test.cc"],
    deps = [
        ":chunk_splitter",
        "//streamer/impl/assigner",
        "//streamer/impl/batches",
        "//streamer/impl/config/config",
        "//utils/random",
        "//utils/temp/env",
        "//utils/temp/file",
    ],
)
