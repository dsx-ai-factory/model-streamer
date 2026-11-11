# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""broadcast() sends the slot it was given, and yields views into THAT slot.

The distribution loop is wired to the staged source, whose slot is a view of the staging buffer. So
sending the staging buffer and sending the slot are the same bytes, and no end-to-end test can tell
them apart. A source that is read into directly has no staging buffer at all, and then the two stop
being the same thing.

These tests give broadcast() a slot that is NOT the staging buffer, which is the only way to see
which of the two it uses."""

import unittest
from unittest.mock import patch

import torch

from runai_model_streamer.distributed_streamer.distribution_source import SlotRange
from runai_model_streamer.distributed_streamer.distributed_streamer import (
    _distributedStreamer,
    write_metadata_rows,
)


GROUP = object()


class Sender(unittest.TestCase):
    """One rank, which is therefore always the broadcasting rank."""

    def setUp(self):
        self.streamer = _distributedStreamer.__new__(_distributedStreamer)
        self.streamer.group_size = 1
        self.streamer.original_group_rank = 0
        self.streamer.local_group_global_ranks = [0]
        self.streamer.distribution_group = GROUP
        self.streamer.rank = 0

        self.sent_tensors = []

    def run_broadcast(self, sent, ranges, chunks_to_read=None, count=None):
        """Drive the sender branch and collect both what went out and what came back.

        The decoy is what the loop used to pass: a buffer the same length as the slot, holding
        different bytes. Nothing here may touch it."""
        self.decoy = torch.zeros(len(sent) if sent is not None else 1, dtype=torch.uint8)

        metadata = torch.zeros(16, 4, dtype=torch.int64)
        write_metadata_rows(metadata, ranges)

        chunks_to_read = [len(ranges)] if chunks_to_read is None else chunks_to_read

        def record(tensor, src=None, group=None):
            self.sent_tensors.append(tensor)

        with patch(
            "runai_model_streamer.distributed_streamer.distributed_streamer.dist.broadcast",
            side_effect=record,
        ):
            return list(
                self.streamer.broadcast(
                    chunks_to_read,
                    metadata,
                    torch.zeros(16, 4, dtype=torch.int64),
                    sent,
                    self.decoy,
                    len(ranges) if count is None else count,
                )
            ), metadata

    def slot(self):
        """Position-sensitive bytes, so a wrong offset cannot read as a right one."""
        return torch.arange(1, 1 + 96, dtype=torch.uint8)

    def test_the_slot_itself_goes_on_the_wire(self):
        sent = self.slot()
        ranges = [SlotRange(4, 0, 0, 40), SlotRange(4, 1, 64, 32)]

        self.run_broadcast(sent, ranges)

        data = self.sent_tensors[1]        # [0] is the metadata
        self.assertIs(data, sent, "the sender passes the source's own tensor through")
        self.assertEqual(data.data_ptr(), sent.data_ptr())
        self.assertNotEqual(data.data_ptr(), self.decoy.data_ptr())

    def test_the_yields_are_views_into_the_slot(self):
        sent = self.slot()
        ranges = [SlotRange(4, 0, 0, 40), SlotRange(4, 1, 64, 32)]

        yielded, _ = self.run_broadcast(sent, ranges)

        self.assertEqual([(f, i) for f, i, _ in yielded], [(4, 0), (4, 1)])
        for buffer, r in zip([b for _, _, b in yielded], ranges):
            self.assertEqual(buffer.data_ptr(), sent.data_ptr() + r.offset)
            self.assertEqual(len(buffer), r.size)
            self.assertEqual(buffer.tolist(), sent[r.offset : r.offset + r.size].tolist())

    def test_a_yield_follows_the_slot_when_it_changes(self):
        """A view, not a copy. The loop releases the slot only after the yields for this reason."""
        sent = self.slot()
        ranges = [SlotRange(4, 0, 0, 40)]

        yielded, _ = self.run_broadcast(sent, ranges)
        buffer = yielded[0][2]
        sent[7] = 200

        self.assertEqual(buffer[7].item(), 200)

    def test_what_is_sent_is_the_size_the_metadata_implies(self):
        """The sender sizes the collective by the slot, the receivers by these rows.

        They are two different numbers in two different places and a mismatch is a hang, not an
        error. The contract makes them equal: the slot is the used prefix, so it ends where the
        last range ends."""
        sent = self.slot()
        ranges = [SlotRange(4, 0, 0, 40), SlotRange(4, 1, 64, 32)]

        _, metadata = self.run_broadcast(sent, ranges)

        last = metadata[len(ranges)]
        self.assertEqual(self.sent_tensors[1].numel(), 96)
        self.assertEqual(last[3].item() + last[2].item(), 96)

    def test_a_rank_with_nothing_to_send_sends_only_metadata(self):
        """sent is None then, and the count guard is what keeps it unread."""
        yielded, _ = self.run_broadcast(None, [], chunks_to_read=[0])

        self.assertEqual(yielded, [])
        self.assertEqual(len(self.sent_tensors), 1, "the metadata, and no data collective")


if __name__ == "__main__":
    unittest.main()
