/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// A separate binary from client_configuration_test: the IMDS region is resolved once per process, so
// this needs a process in which nothing has resolved it yet.

#include "s3/client_configuration/client_configuration.h"

#include <gtest/gtest.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <memory>
#include <string>
#include <thread>

#include "s3/s3_init/s3_init.h"
#include "utils/temp/env/env.h"

namespace runai::llm::streamer::impl::s3
{

namespace
{

// Answers IMDSv2 token requests and nothing else, so every region lookup fails as it does off AWS on a
// network that has a metadata endpoint without the data. Counts the region (availability zone) requests.
class FakeImds
{
 public:
    FakeImds()
    {
        _fd = ::socket(AF_INET, SOCK_STREAM, 0);
        int one = 1;
        ::setsockopt(_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = 0;
        EXPECT_EQ(::bind(_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)), 0);
        EXPECT_EQ(::listen(_fd, 16), 0);
        socklen_t len = sizeof(addr);
        ::getsockname(_fd, reinterpret_cast<sockaddr *>(&addr), &len);
        _port = ntohs(addr.sin_port);
        _thread = std::thread([this] { serve(); });
    }

    ~FakeImds()
    {
        _stop = true;
        _thread.join();
        ::close(_fd);
    }

    std::string endpoint() const { return "http://127.0.0.1:" + std::to_string(_port); }
    int region_requests() const { return _region_requests; }

 private:
    void serve()
    {
        while (!_stop)
        {
            pollfd p{_fd, POLLIN, 0};
            if (::poll(&p, 1, 50) <= 0)
            {
                continue;
            }
            const int client = ::accept(_fd, nullptr, nullptr);
            if (client < 0)
            {
                continue;
            }
            std::string request;
            char buffer[4096];
            while (request.find("\r\n\r\n") == std::string::npos)
            {
                const auto n = ::recv(client, buffer, sizeof(buffer), 0);
                if (n <= 0)
                {
                    break;
                }
                request.append(buffer, n);
            }

            std::string response = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
            if (request.rfind("PUT /latest/api/token", 0) == 0)
            {
                response = "HTTP/1.1 200 OK\r\nContent-Length: 5\r\nConnection: close\r\n\r\ntoken";
            }
            else if (request.find("/latest/meta-data/placement/availability-zone") != std::string::npos)
            {
                ++_region_requests;
            }
            ::send(client, response.data(), response.size(), 0);
            ::close(client);
        }
    }

    int _fd = -1;
    int _port = 0;
    std::atomic<bool> _stop{false};
    std::atomic<int> _region_requests{0};
    std::thread _thread;
};

} // namespace

TEST(ClientConfigurationImds, Region_Is_Asked_From_Imds_Once_Per_Process)
{
    FakeImds imds;
    utils::temp::Env endpoint(std::string("AWS_EC2_METADATA_SERVICE_ENDPOINT"), imds.endpoint());
    utils::temp::UnsetEnv imds_disabled(std::string("AWS_EC2_METADATA_DISABLED"));
    utils::temp::UnsetEnv region(std::string("AWS_REGION"));
    utils::temp::UnsetEnv default_region(std::string("AWS_DEFAULT_REGION"));
    utils::temp::Env config_file(std::string("AWS_CONFIG_FILE"), std::string("/nonexistent"));
    S3Init init;

    for (int i = 0; i < 4; ++i)
    {
        EXPECT_EQ(ClientConfiguration().config.region, "us-east-1");
    }
    EXPECT_EQ(imds.region_requests(), 1);

    // a region configured later is used, without asking IMDS
    utils::temp::Env configured(std::string("AWS_REGION"), std::string("eu-west-1"));
    EXPECT_EQ(ClientConfiguration().config.region, "eu-west-1");
    EXPECT_EQ(imds.region_requests(), 1);
}

}; // namespace runai::llm::streamer::impl::s3
