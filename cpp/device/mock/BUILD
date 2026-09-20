load("//:rules.bzl", "runai_cc_auto_library")

# TEST ONLY, and its own package so that no production glob can reach it - a mock in the shipped
# library is a defect. Nothing under //streamer may depend on this.
runai_cc_auto_library(
    name = "mock",
    deps = [
        "//device",
    ],
    testonly = True,
    visibility = ["//visibility:public"],
)
