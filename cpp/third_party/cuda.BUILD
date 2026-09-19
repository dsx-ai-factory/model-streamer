# CUDA driver API headers, for compile-time type checking only.
# libcuda.so is not linked here - it is loaded with dlopen at run time, so the built
# library depends on no particular CUDA version and runs where there is no driver.
cc_library(
    name = "cuda_headers",
    hdrs = ["include/cuda.h"],
    includes = ["include"],
    visibility = ["//visibility:public"],
)
