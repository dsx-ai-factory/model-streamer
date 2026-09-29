/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/logging/logging.h"

namespace Argument
{

enum
{
    Level   = 1,
    Message = 2,
};

} // namespace Argument

int main(int argc, char* argv[])
{
    const std::string level = argv[Argument::Level];
    const std::string message = argv[Argument::Message];

    if (level == "SPAM")
    {
        LOG(SPAM) << message;
    }
    else if (level == "DEBUG")
    {
        LOG(DEBUG) << message;
    }
    else if (level == "INFO")
    {
        LOG(INFO) << message;
    }
    else if (level == "WARNING")
    {
        LOG(WARNING) << message;
    }
    else if (level == "ERROR")
    {
        LOG(ERROR) << message;
    }
    else
    {
        return 1;
    }

    return 0;
}
