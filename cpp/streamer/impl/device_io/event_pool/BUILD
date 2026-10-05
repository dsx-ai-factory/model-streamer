load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

runai_cc_auto_library(
    name = "event_pool",
    deps = [
        "//device",
        "//device/owned",
        "//utils/logging",
    ],
    visibility = ["//streamer:__subpackages__"],
)

runai_cc_test(
    name = "event_pool_test",
    srcs = ["event_pool_test.cc"],
    deps = [
        ":event_pool",
        "//device/mock",
    ],
    env = {"RUNAI_STREAMER_LOG_TO_STDERR": "1"},
)
