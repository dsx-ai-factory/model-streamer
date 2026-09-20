load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

runai_cc_auto_library(
    name = "cuda",
    deps = [
        "//device",
        "//common/response_code",
        "//utils/env",
        "//utils/logging",
        "@cuda//:cuda_headers",
    ],
    # libcuda.so is opened with dlopen at run time; nothing links against it.
    linkopts = ["-ldl"],
    visibility = ["//visibility:public"],
)

runai_cc_test(
    name = "cuda_lib_test",
    srcs = ["cuda_lib_test.cc"],
    deps = [":cuda"],
)

runai_cc_test(
    name = "cuda_device_test",
    srcs = ["cuda_device_test.cc"],
    deps = [
        ":cuda",
        "//utils/temp/env",
    ],
    # The failure paths log; without a sink the LOG arguments are never built, so the code that
    # formats them is untested.
    env = {"RUNAI_STREAMER_LOG_TO_STDERR": "1"},
)
