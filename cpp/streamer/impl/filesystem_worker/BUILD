load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

runai_cc_auto_library(
    name = "filesystem_worker",
    deps = [
        "//streamer/impl/workload",
        "//utils/threadpool",
    ],
    visibility = ["//streamer:__subpackages__"],
)

runai_cc_test(
    name = "filesystem_worker_test",
    srcs = ["filesystem_worker_test.cc"],
    deps = [
        ":filesystem_worker",
    ],
)
