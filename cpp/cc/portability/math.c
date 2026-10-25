/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Ubuntu 20.04's libm defaults to GLIBC_2.29 versions of these functions.
// Keep the native SDKs linkable on glibc 2.28 using the existing libm
// compatibility implementations. --wrap also covers calls from static SDKs.
#if defined(__aarch64__)
#define RUNAI_LIBM_VERSION "GLIBC_2.17"
#elif defined(__x86_64__)
#define RUNAI_LIBM_VERSION "GLIBC_2.2.5"
#else
#error Unsupported architecture
#endif

extern double compatible_log(double);
extern double compatible_log2(double);
extern double compatible_pow(double, double);
__asm__(".symver compatible_log,log@" RUNAI_LIBM_VERSION);
__asm__(".symver compatible_log2,log2@" RUNAI_LIBM_VERSION);
__asm__(".symver compatible_pow,pow@" RUNAI_LIBM_VERSION);

double __wrap_log(double x)
{
    return compatible_log(x);
}

double __wrap_log2(double x)
{
    return compatible_log2(x);
}

double __wrap_pow(double x, double y)
{
    return compatible_pow(x, y);
}
