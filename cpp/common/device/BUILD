load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

runai_cc_auto_library(
    name = "device",
    deps = [
        "//streamer/api",
    ]
)

runai_cc_test(
    name = "device_test",
    srcs = ["device_test.cc"],
    deps = [
        ":device",
    ]
)
