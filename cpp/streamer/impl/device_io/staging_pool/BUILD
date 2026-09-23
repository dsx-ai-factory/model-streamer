load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

runai_cc_auto_library(
    name = "staging_pool",
    # Only the streamer reaches the device path. runai_cc_library defaults to public, so
    # this has to be stated per target - a package() in the parent BUILD does not reach
    # sub-packages.
    visibility = ["//streamer:__subpackages__"],
    deps = [
        "//device",
        "//device/owned",
        "//utils/logging",
        "//utils/scope_guard",
    ],
)

runai_cc_test(
    name = "staging_pool_test",
    srcs = ["staging_pool_test.cc"],
    deps = [
        ":staging_pool",
        "//common/exception",
        "//device/mock",
    ],
    env = {"RUNAI_STREAMER_LOG_TO_STDERR": "1"},
)

runai_cc_test(
    name = "staging_pool_stress_test",
    srcs = ["staging_pool_stress_test.cc"],
    deps = [
        ":staging_pool",
        "//device/mock",
    ],
    # One scenario per run. Repeat it with --runs_per_test=10000 --nocache_test_results: a fresh
    # process each time gives different heap layout and scheduling, which a loop inside one process
    # does not.
    size = "small",
    env = {"RUNAI_STREAMER_LOG_TO_STDERR": "1"},
)
