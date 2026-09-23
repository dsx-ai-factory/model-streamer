load("//:rules.bzl", "runai_cc_binary", "runai_cc_test")

# The only test that talks to a real driver. It SKIPS where there is none, so it is safe on CPU CI.
#
# Its own package so that cuda.h can be made visible to exactly one test target - see
# cpp/third_party/cuda.BUILD. The readback needs the driver directly, because the Device interface has
# no device-to-host copy.
runai_cc_test(
    name = "real_device_test",
    srcs = ["real_device_test.cc"],
    deps = [
        "//device/cuda",
        "//streamer/impl/device_io/device_writer",
        "//streamer/impl/device_io/device_writer_client",
        "//streamer/impl/streamer",
        "//utils/temp/file",
        "//utils/random",
        "//utils/temp/env",
        "@cuda//:cuda_headers",
    ],
    env = {"RUNAI_STREAMER_LOG_TO_STDERR": "1"},
)

# Read performance into device memory, with a byte-wise check of every arm. Run by hand on a GPU
# host - see project_benchmarks_run_manually.
runai_cc_binary(
    name = "device_stream_bench",
    srcs = ["device_stream_bench.cc"],
    deps = [
        "//device/cuda",
        "//streamer/impl/streamer",
        "//utils/random",
        "//utils/temp/file",
        "@cuda//:cuda_headers",
    ],
    linkopts = ["-lpthread", "-lm", "-ldl"],
)
