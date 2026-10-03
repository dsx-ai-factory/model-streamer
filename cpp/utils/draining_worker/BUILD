load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

runai_cc_auto_library(
    name = "draining_worker",
    deps = [
        "//utils/thread",
        "//utils/deque",
    ],
)

runai_cc_test(
    name = "draining_worker_test",
    srcs = ["draining_worker_test.cc"],
    deps = [":draining_worker"],
)

runai_cc_test(
    name = "draining_worker_stress_test",
    srcs = ["draining_worker_stress_test.cc"],
    deps = [":draining_worker"],
    # One scenario per run. Repeat with --runs_per_test=10000 --nocache_test_results: a fresh
    # process each time gives different scheduling, which a loop inside one process does not.
    size = "small",
)
