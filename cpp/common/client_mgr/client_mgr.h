/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <set>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

#include "common/backend_api/object_storage/object_storage.h"
#include "utils/logging/logging.h"

namespace runai::llm::streamer::common
{

class IClient
{
public:
    /**
     * @brief Virtual destructor.
     * Essential for any base class that will be managed via a pointer.
     */
    virtual ~IClient() = default;

    /**
     * @brief Verifies that the client's credentials have not changed.
     * @param config The current configuration to check against.
     * @return True if credentials match, false otherwise.
     */
    virtual bool verify_credentials(const common::backend_api::ObjectClientConfig_t& config) const = 0;
};

// Singleton clients manager for reusing clients of the current object storage bucket

template <typename T, const char *ClientType>
struct ClientMgr
{
    // This check is performed at compile-time
    static_assert(std::is_base_of<IClient, T>::value, 
                  "Template parameter T must be a class derived from IClient.");

    ~ClientMgr();

    ClientMgr<T, ClientType> & operator=(const ClientMgr<T, ClientType> &) = delete;

    static T* pop(const common::backend_api::ObjectClientConfig_t & config);
    static void push(T* client);

    static void clear();

    // stop all clients
    static void stop();

    // for testing:
    static unsigned size();
    static unsigned unused();

 private:
    ClientMgr();

    static ClientMgr<T, ClientType> & get();

    mutable std::mutex _mutex;

    // reuse clients of the same bucket
    std::set<T*> _unused_clients;
    std::map<T*, std::unique_ptr<T>> _clients;
};

template <typename T, const char* ClientType>
ClientMgr<T, ClientType>::ClientMgr()
{}

template <typename T, const char* ClientType>
ClientMgr<T, ClientType>::~ClientMgr()
{}

template <typename T, const char* ClientType>
ClientMgr<T, ClientType> & ClientMgr<T, ClientType>::get()
{
    static ClientMgr<T, ClientType> __clients_mgr;
    return __clients_mgr;
}

template <typename T, const char* ClientType>
unsigned ClientMgr<T, ClientType>::size()
{
    auto & mgr = get();

    const auto guard = std::unique_lock<std::mutex>(mgr._mutex);
    return mgr._clients.size();
}

template <typename T, const char* ClientType>
unsigned ClientMgr<T, ClientType>::unused()
{
    auto & mgr = get();

    const auto guard = std::unique_lock<std::mutex>(mgr._mutex);
    return mgr._unused_clients.size();
}

template <typename T, const char* ClientType>
T* ClientMgr<T, ClientType>::pop(const common::backend_api::ObjectClientConfig_t & config)
{
    auto & mgr = get();

    {
        const auto guard = std::unique_lock<std::mutex>(mgr._mutex);

        auto & unused = mgr._unused_clients;
        while (!unused.empty())
        {
            auto ptr = *unused.begin();
            unused.erase(unused.begin());

            // Reuse client only if credentials have not changed
            if (ptr->verify_credentials(config))
            {
                LOG(DEBUG) << "Reusing " << ClientType << " client";
                return ptr;
            }

            // release the stale client
            mgr._clients.erase(ptr);
        }
    }

    // create new client if there are no unused clients for this bucket

    LOG(DEBUG) << "Creating client for endpoint " << config.endpoint_url;
    auto client = std::make_unique<T>(config);

    const auto guard = std::unique_lock<std::mutex>(mgr._mutex);
    auto ptr = client.get();
    mgr._clients[ptr] = std::move(client);
    return ptr;
}

template <typename T, const char* ClientType>
void ClientMgr<T, ClientType>::push(T* client)
{
    LOG(DEBUG) << "Releasing " << ClientType << " client";
    auto & mgr = get();
    const auto guard = std::unique_lock<std::mutex>(mgr._mutex);
    mgr._unused_clients.insert(client);
}

template <typename T, const char* ClientType>
void ClientMgr<T, ClientType>::clear()
{
    LOG(DEBUG) << "Releasing all " << ClientType << " clients";
    auto & mgr = get();

    const auto guard = std::unique_lock<std::mutex>(mgr._mutex);

    // verify that all clients are unused
    unsigned count = mgr._unused_clients.size();

    if (count != mgr._clients.size())
    {
        LOG(ERROR) << "There are used " << ClientType << " clients - number of clients is " << mgr._clients.size() << " while number of unused clients is " << count;
        return;
    }

    mgr._clients.clear();
    mgr._unused_clients.clear();
}

template <typename T, const char* ClientType>
void ClientMgr<T, ClientType>::stop()
{
    LOG(DEBUG) << "Stopping all " << ClientType << " clients";
    auto & mgr = get();

    const auto guard = std::unique_lock<std::mutex>(mgr._mutex);

    for (auto & pair : mgr._clients)
    {
        pair.second->stop();
    }
}

}; //namespace runai::llm::streamer::common
