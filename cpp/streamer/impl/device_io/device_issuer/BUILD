load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

runai_cc_auto_library(
    name = "device_issuer",
    deps = [
        "//common/device",
        "//streamer/impl/device_io/device_writer_client",
        "//streamer/impl/device_io/staging_pool",
        "//utils/draining_worker",
        "//utils/logging",
    ],
    visibility = ["//streamer:__subpackages__"],
)

runai_cc_test(
    name = "device_issuer_test",
    srcs = ["device_issuer_test.cc"],
    deps = [
        ":device_issuer",
        "//device/mock",
    ],
    env = {"RUNAI_STREAMER_LOG_TO_STDERR": "1"},
)
