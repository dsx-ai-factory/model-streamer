# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT TARGET runai::streamer)
    get_filename_component(_runai_streamer_prefix "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
    add_library(runai::streamer SHARED IMPORTED)
    set_target_properties(runai::streamer PROPERTIES
        IMPORTED_LOCATION "${_runai_streamer_prefix}/lib/libstreamer.so"
        INTERFACE_INCLUDE_DIRECTORIES "${_runai_streamer_prefix}/include"
    )
    unset(_runai_streamer_prefix)
endif()
