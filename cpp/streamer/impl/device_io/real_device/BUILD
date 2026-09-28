load("//:rules.bzl", "runai_cc_binary", "runai_cc_test")

# The only test that talks to a real driver. It SKIPS where there is none, so it is safe on CPU CI.
#
# Its own package so that cuda.h can be made visible to exactly one test target - see
# cpp/third_party/cuda.BUILD. The readback needs the driver directly, because the Device interface has
# no device-to-host copy.
runai_cc_test(
    name = "real_device_test",
    # Excluded from --config=tsan for the same reason as streamer_s3_test: it dlopens
    # libstreamers3.so, and the sanitized binary does not resolve $ORIGIN in its RUNPATH.
    tags = ["no_tsan"],
    srcs = ["real_device_test.cc"],
    deps = [
        "//device/cuda",
        "//streamer/impl/device_io/device_writer",
        "//streamer/impl/device_io/device_writer_client",
        "//streamer/impl/streamer",
        "//utils/temp/file",
        "//utils/random",
        "//utils/temp/env",
        "//utils/dylib",
        "@cuda//:cuda_headers",
        # The object storage path, driven through the s3 MOCK plugin: a real driver and a real staging
        # pool, without a real bucket.
        "//s3/s3_mock:libstreamers3.so",
    ],
    linkstatic = False,
    data = ["//s3/s3_mock:libstreamers3.so"],
    linkopts = [
        "-Wl,-rpath,$$ORIGIN/../../../../s3/s3_mock",
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
