# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where a rank's outgoing bytes come from, and how a slot is filled with them.

A distribution source is ALWAYS a ring of slots on the device the collective sends from. What
differs between devices is only how a slot gets filled:

- DIRECT, where the C++ layer can write to that device: the ring IS the read destination, so there
  is no copy at all;
- STAGED, where it cannot - a CPU device under gloo, or any device type the library does not serve:
  a host ring is read into, and the ranges are copied into the slot.

The loop sees one thing, slots that are ready to send. So a new device type is a new implementation
of this interface, rather than another branch in the loop.

NAMED FOR DISTRIBUTION, NOT FOR BROADCASTING. Broadcasting is one way to distribute: every rank
contributes a slot of the same size, which is also the condition an all-gather needs, and replacing
N broadcasts with one all-gather is an open option. Nothing in this interface should have to change
if the collective does.
"""

from __future__ import annotations
from typing import Any, List, NamedTuple, Optional, Protocol


RANGE_OFFSET_ALIGNMENT = 8
"""The largest element size safetensors has, F64 and I64.

Every range offset is a multiple of it, so a range sits where its dtype can be read from whatever
the sizes before it were. A caller that takes a view instead of a copy then needs no copy after all.
"""


class SlotRange(NamedTuple):
    """One range inside a filled slot, in THE CALLER'S NUMBERING.

    file_id and range_index are the ones the consumer asked with - not the partition's, and not the
    packer's. A receiving rank never saw the sender's partition and cannot interpret its numbering,
    so the conversion belongs on the sending side. The metadata path already works this way:
    prefill() converts through rank_dicts_map before writing a metadata row, so the ids that cross
    the wire today are already the caller's.

    offset is from the START OF THE SLOT, so it means the same thing to a receiver reading its own
    copy at its own address.
    """

    file_id: int
    range_index: int
    offset: int
    size: int


class Filled(NamedTuple):
    """A slot ready to send.

    memory is EXACTLY the bytes to send - the used prefix, not the whole slot. A slot is sized for
    the ring, not for what happens to be in it, so the last slot of a stream is usually part full
    and its tail would cost a collective the same as real data.

    ranges describes what is in those bytes, in layout order.
    """

    memory: Any
    ranges: List[SlotRange]


class DistributionSource(Protocol):
    """What the distribution loop needs from a source, and nothing more.

    FOUR GUARANTEES, which belong to this interface rather than to any implementation:

    1. next_slot() BLOCKS until the slot is ready to send. Readiness differs - a direct source's
       slot is ready when the reads land, a staged source's when its copies are ordered on the
       sending stream - and the loop must not learn which, or every new source becomes a branch
       in the loop.
    2. memory is the USED PREFIX, never the whole slot.
    3. ranges follow ONE LAYOUT RULE: the first starts at offset 0, and each next one starts at the
       previous range's end, rounded up to a multiple of RANGE_OFFSET_ALIGNMENT. No reordering, and
       never a wider pad than the rule asks for.

       The rule reads sizes only, never an address, so every rank computes the same layout. Later
       work computes what a peer will send instead of being told, and two rules would disagree
       silently.
    4. release() takes the EVENT after which the slot may be refilled, or None when nothing is
       enqueued against it. A direct source needs the clone event; a staged source on a host
       device may need nothing.

       AFTER release, the slot is the source's again. A caller must not read `memory` or `ranges`
       once it has released them - a ring hands the same buffer out again, so what it reads then
       belongs to a later slot.

    MORE THAN ONE SLOT MAY BE OUT AT A TIME. A loop with an asynchronous collective wants to fill
    the next slot while the previous is still in flight, which is the overlap the ring exists for.
    So a source must not assume its last slot was released before next_slot() is called again; it
    blocks when no slot is free, which is a different thing.

    WHAT IS NOT GUARANTEED HERE: how many slots exist, or how long a source will block. Both belong
    to the ring an implementation was built with.
    """

    def next_slot(self) -> Optional[Filled]:
        """The next slot to send, in layout order, or None when the stream is done.

        Blocks until it is ready to send."""
        ...

    def release(self, filled: Filled, done: Any = None) -> None:
        """Give the slot back, to be refilled once `done` has completed.

        The caller must not touch `filled` afterwards."""
        ...
