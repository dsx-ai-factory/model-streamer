load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

runai_cc_auto_library(
    name = "streamer",
    deps = [
        "//device/cuda",
        "//common/device",
        "//streamer/impl/assigner",
        "//streamer/impl/config/config",
        "//streamer/impl/batches",
        "//streamer/impl/workload",
        "//streamer/impl/filesystem_worker",
        "//streamer/impl/object_storage_worker",
        "//streamer/impl/request",
        "//streamer/impl/submissions:submissions_mgr",
        "//streamer/impl/pools:backend_pools",
        "//streamer/impl/strategy_resolver",
        "//posix_io/alignment",
        "//posix_io/mount_capabilities",
        "//streamer/impl/async_io/async_io_worker",
        "//streamer/impl/async_io/fs_async_router",
        "//streamer/impl/async_io/async_io_stats",
        "//common/responder",
        "//utils/threadpool",
        "//utils/fdlimit",
    ],
)

runai_cc_test(
    name = "streamer_test",
    srcs = ["streamer_test.cc"],
    deps = [
        "//posix_io/mock",
        "//posix_io/io_uring_probe",
        ":streamer",
        "//utils/random",
        "//utils/fd",
        "//utils/temp/env",
        "//utils/temp/file",
    ],
)
