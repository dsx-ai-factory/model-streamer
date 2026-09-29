/* Copyright (C) 2011-2018 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <http://www.gnu.org/licenses/>.  */

/*
 * NVIDIA-authored portions:
 * Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use these portions except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * The __wrap___fdelt_chk function below derives from glibc's debug/fdelt_chk.c
 * and retains the following upstream notice and LGPL-2.1-or-later terms.
 * Modifications: renamed the function and replaced __chk_fail with exit.
 */

#include <stdlib.h>
#include <string.h>

// by default, the gcc in our development environment uses `memcpy` from glibc 2.14.
// this adds the dependency on `GLIBC_2.14` from libc.so, which is too advance, and
// is not available in older distributions.
// other people encountered this issue and it is was discussed online [1].
// one of the ways to solve this issue is to explicitly tell the linker to use a
// certain version of `memcpy`, using `__asm__(".symver ...")`, as described in [2] and [3].
// unfortunately, this way does not work properly with "gold" linker, which is the
// one Bazel uses, as described in [4].
// therefore, we just implement `memcpy` ourselves (using `memmove` as detailed below)

void * __wrap_memcpy(void * dest, const void * src, size_t n)
{
    // According to https://stackoverflow.com/a/48825106/9540328, `memmove` behaves
    // the same as `memcpy` but in a safer way.
    // this means that we can use it as our implementation of `memcpy`

    return memmove(dest, src, n);
}

long int __wrap___fdelt_chk(long int d)
{
    // copied pretty much as-is from https://github.com/lattera/glibc/blob/master/debug/fdelt_chk.c
    // using `exit` instead of `__chk_fail` because it is not available

    if (d < 0 || d >= FD_SETSIZE)
    {
        exit(EXIT_FAILURE);
    }

    return d / __NFDBITS;
}

// References and useful links:
//   [1] A Stackoverflow article about `memcpy@GLIBC_2.14`      - https://stackoverflow.com/questions/8823267/linking-against-older-symbol-version-in-a-so-file
//   [2] Linking to older versioned symbols (glibc)             - http://web.archive.org/web/20160107032111/http://www.trevorpounds.com/blog/?p=103
//   [3] `__wrap_memcpy` implementation in Protocol Buffers     - https://chromium.googlesource.com/external/github.com/google/protobuf/+/HEAD/ruby/ext/google/protobuf_c/wrap_memcpy.c
//   [4] Gold linker and `__wrap_memcpy`
//       [a] - http://sourceware-org.1504.n7.nabble.com/gold-vs-ld-bfd-mismatch-on-wrap-td413031.html
//       [b] - https://sourceware.org/bugzilla/show_bug.cgi?id=24462
//   [5] Creating portable Linux binaries                       - http://insanecoding.blogspot.com/2012/07/creating-portable-linux-binaries.html
//   [6] Officially distributed packages (glibc in particular)  - https://distrowatch.com/dwres.php?resource=major
//   [7] Using linker flag `--wrap`                             - https://stackoverflow.com/questions/46444052/how-to-wrap-functions-with-the-wrap-option-correctly
