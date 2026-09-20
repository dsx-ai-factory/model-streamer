load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

runai_cc_auto_library(
    name = "threadpool",
    deps = [
        "//utils/deque",
        "//utils/thread",
        "//utils/semaphore",
    ]
)

runai_cc_test(
    name = "threadpool_test",
    srcs = ["threadpool_test.cc"],
    deps = [
        ":threadpool",
        "//utils/random",
    ],
)
