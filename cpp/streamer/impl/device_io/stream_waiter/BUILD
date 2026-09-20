load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

runai_cc_auto_library(
    name = "stream_waiter",
    # Only the streamer reaches the device path. runai_cc_library defaults to public, so
    # this has to be stated per target - a package() in the parent BUILD does not reach
    # sub-packages.
    visibility = ["//streamer:__subpackages__"],
    deps = [
        "//device",
        "//streamer/impl/device_io/staging_pool",
        "//utils/draining_worker",
        "//utils/logging",
    ],
)

runai_cc_test(
    name = "stream_waiter_test",
    srcs = ["stream_waiter_test.cc"],
    deps = [
        ":stream_waiter",
        "//device/mock",
    ],
    env = {"RUNAI_STREAMER_LOG_TO_STDERR": "1"},
)
