load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

# The only place that maps a device type to a vendor. Everything above it names the interface, and
# the interface names no vendor - see backends.h.
runai_cc_auto_library(
    name = "backends",
    deps = [
        "//common/device",
        "//device",
        "//device/cuda",
        "//utils/logging",
    ],
    visibility = ["//visibility:public"],
)

runai_cc_test(
    name = "backends_test",
    srcs = ["backends_test.cc"],
    deps = [
        ":backends",
    ],
)
