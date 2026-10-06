/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <limits>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace runai::llm::streamer::impl
{

// RUNAI_STREAMER_FS_QUEUE_DEPTH: how many file system reads may be in flight, per mount.
//
//     <default> [ "," <type> "=" <value> ]*      e.g. "512,nfs=64,virtiofs=256"
//
// Keys are file system types as /proc/self/mountinfo reports them, not paths: a path differs per
// deployment, while "NFS is the slow one" travels with the storage.
//
// A type names the file system, not the device - an NVMe and a spinning disk formatted ext4 both
// report `ext4` - so a key separates network storage from local and says nothing about local speed.
class FsQueueDepth
{
 public:
    struct Entry
    {
        std::string type;
        unsigned    value = 0;
    };

    // Throws InvalidParameterError on a malformed value, so a typo is reported rather than ignored.
    // The leading default is mandatory, which makes a plain integer a complete value.
    static FsQueueDepth parse(const std::string & value);

    explicit FsQueueDepth(unsigned value);

    // The first entry whose type is a prefix of this one, else the default. A prefix, so `nfs` covers
    // `nfs4`; first match rather than longest, so `nfs=64,nfs4=32` answers 64 for an nfs4 mount.
    unsigned for_type(const std::string & fs_type) const;

    unsigned default_value() const;

    // In the order written, which is the order for_type matches in.
    const std::vector<Entry> & entries() const;

 private:
    FsQueueDepth() = default;

    unsigned _default = 0;
    std::vector<Entry> _entries;
};

std::ostream & operator<<(std::ostream &, const FsQueueDepth &);

}; // namespace runai::llm::streamer::impl
