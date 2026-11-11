# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.16)
project(hello_streamer LANGUAGES C)

find_package(runai-streamer CONFIG REQUIRED)
add_executable(hello_streamer main.c)
set_target_properties(hello_streamer PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED YES C_EXTENSIONS NO)
target_link_libraries(hello_streamer PRIVATE runai::streamer)
