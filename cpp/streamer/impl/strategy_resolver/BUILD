# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

runai_cc_auto_library(
    name = "strategy_resolver",
    deps = [
        "//common/exception",
        "//posix_io/io_uring_probe",
        "//posix_io/libaio_probe",
        "//posix_io/strategy",
        "//common/response_code",
        "//utils/logging",
    ],
)

runai_cc_test(
    name = "strategy_resolver_test",
    srcs = ["strategy_resolver_test.cc"],
    deps = [
        ":strategy_resolver",
        "//utils/temp/env",
    ],
    # Turn a log sink ON for this target, which is what makes the LOG(ERROR) messages in
    # strategy_resolver.cc actually get BUILT.
    #
    # should_process_log() requires __print or __file, so with neither set the arguments of a LOG
    # statement are never evaluated - they are unexecuted code. A `_resolved.value()` in one of these
    # messages threw std::bad_optional_access after a failed resolution, and no test could see it
    # because no test had a sink.
    #
    # Set here rather than through --test_env on the whole suite: //utils/logging:logging_test
    # asserts what these variables do when UNSET, and both are read once in a static initialiser, so
    # a global setting makes that test fail by construction.
    env = {"RUNAI_STREAMER_LOG_TO_STDERR": "1"},
)
