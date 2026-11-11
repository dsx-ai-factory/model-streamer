# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The contract every distribution source must honour.

DistributionSourceContract is a mixin. A real source gets a TestCase that mixes it in and supplies
make_source() and expected_ranges(), and inherits every test below.

It uses ONLY what the Protocol defines - next_slot() and release(). An earlier version reached for
fake-only state (a slot_size() method, a ready_when_returned list), which meant a source
implementing exactly the Protocol failed three of its own contract tests.
"""

import unittest

from runai_model_streamer.distributed_streamer.distribution_source import (
    RANGE_OFFSET_ALIGNMENT,
    Filled,
    SlotRange,
    advance,
)


class FakeSource:
    """A minimal source that honours the contract.

    Its slots are bytearrays from a fixed-size POOL, not fresh allocations, so it reuses buffers
    the way a ring does. That matters: a contract test that read a slot after releasing it would
    pass against fresh allocations and fail against any real source."""

    SLOT_SIZE = 4096
    DEPTH = 2

    def __init__(self, plan, depth=DEPTH):
        """plan is a list of slots, each a list of (file_id, range_index, size)."""
        self._plan = list(plan)
        self._free = [bytearray(self.SLOT_SIZE) for _ in range(depth)]
        self._out = {}                 # id(memoryview) -> the buffer it came from
        self.released = []

    def next_slot(self):
        if not self._plan:
            return None
        if not self._free:
            raise AssertionError("a real source would block here; this fake was driven too deep")

        entries = self._plan.pop(0)
        buffer = self._free.pop()

        ranges, cursor = [], 0
        for file_id, range_index, size in entries:
            offset, cursor = advance(cursor, size)
            ranges.append(SlotRange(file_id, range_index, offset, size))

        # A real source blocks until the bytes are in the slot. This one has nothing to wait for.
        view = memoryview(buffer)[:cursor]
        self._out[id(view)] = buffer
        return Filled(memory=view, ranges=ranges)

    def release(self, filled, done=None):
        self.released.append(done)
        self._free.append(self._out.pop(id(filled.memory)))


class DistributionSourceContract:
    """Mix into a TestCase that supplies make_source() and expected_ranges()."""

    def make_source(self):
        raise NotImplementedError

    def expected_ranges(self):
        """Every (file_id, range_index) the stream should deliver, as a set."""
        raise NotImplementedError

    def source_depth(self):
        """How many slots this source can hold out at once, or None when it is not known.

        ASKED, not discovered. next_slot() blocks when no slot is free, so a test that called it a
        second time on a depth-one source would hang rather than fail. A depth-one source is legal -
        a rank short of device memory gets one slot."""
        return None

    def release_event(self):
        """An event this source may be released with, or None when it needs none.

        ASKED, because a source may REQUIRE one: a ring whose slots are refilled by another stream
        refuses a release without an event, and nothing generic can invent one it would accept."""
        return None

    def each_slot(self, source):
        """Yield every slot, letting the test assert BEFORE it is released.

        Assertions cannot be made after the release: a ring hands the same buffer out again, so a
        slot read after release describes a later one."""
        while True:
            filled = source.next_slot()
            if filled is None:
                return
            yield filled
            source.release(filled, self.release_event())

    def test_the_memory_is_the_used_prefix(self):
        for filled in self.each_slot(self.make_source()):
            extent = max((r.offset + r.size for r in filled.ranges), default=0)
            self.assertEqual(len(filled.memory), extent,
                             "memory must be exactly the bytes the ranges occupy")

    def test_the_ranges_follow_the_shared_layout_rule(self):
        """Guarantee 3, as a rule rather than as a reason.

        Order alone is not enough. A source that packed tightly and one that padded by its own width
        are both in order, and a peer's computed layout is wrong for either."""
        for filled in self.each_slot(self.make_source()):
            cursor, previous_end = 0, 0
            for r in filled.ranges:
                at, cursor = advance(cursor, r.size)
                self.assertEqual(r.offset, at,
                                 "the previous end, rounded up - no tighter, no wider")

                # Said WITHOUT the helper, so a rule that was wrong in the same way in both places
                # cannot pass: a range with bytes starts where its dtype can read it, and a pad is
                # never wide enough to fit another range before it.
                if r.size:
                    self.assertEqual(r.offset % RANGE_OFFSET_ALIGNMENT, 0)
                    self.assertLess(r.offset - previous_end, RANGE_OFFSET_ALIGNMENT)
                previous_end = r.offset + r.size
            self.assertEqual(cursor, len(filled.memory),
                             "the used prefix ends at the last range, with no pad after it")

    def test_every_range_is_delivered_exactly_once_in_the_callers_numbering(self):
        seen = [(r.file_id, r.range_index)
                for filled in self.each_slot(self.make_source()) for r in filled.ranges]
        self.assertEqual(len(seen), len(set(seen)), "a range must not be delivered twice")
        self.assertEqual(set(seen), self.expected_ranges())

    def test_the_stream_ends_with_none(self):
        source = self.make_source()
        for _ in self.each_slot(source):
            pass
        self.assertIsNone(source.next_slot(), "a finished stream keeps answering None")

    def test_release_takes_the_event_this_source_needs(self):
        """It checks no more than that the call is accepted.

        Whether a source really WAITS for the event before refilling is not observable through this
        interface - the loop hands the event over and has no way to watch what happens to it. That
        part of guarantee 4 belongs to the implementation's own tests, against its own ring."""
        source = self.make_source()
        filled = source.next_slot()
        if filled is None:
            self.skipTest("an empty stream has no slot to release")
        source.release(filled, self.release_event())

    def test_a_source_that_needs_no_event_still_accepts_one(self):
        """Guarantee 4 says `done` may be an event OR None, so neither form may be refused."""
        if self.release_event() is not None:
            self.skipTest("this source is covered by the test above")

        source = self.make_source()
        filled = source.next_slot()
        if filled is None:
            self.skipTest("an empty stream has no slot to release")
        source.release(filled, object())

    def test_a_source_that_needs_no_event_takes_the_one_argument_form(self):
        """Guarantee 4's "or None when nothing is enqueued". Skipped for a source that needs an
        event, which cannot honour it - and that is why the suite asks rather than assumes."""
        if self.release_event() is not None:
            self.skipTest("this source is released with an event")

        source = self.make_source()
        filled = source.next_slot()
        if filled is None:
            self.skipTest("an empty stream has no slot to release")
        source.release(filled)

    def test_two_slots_may_be_out_at_once(self):
        """A loop with an asynchronous collective fills the next slot while the previous is still
        in flight. A source that assumed its last slot was released would break there."""
        depth = self.source_depth()
        if depth is None:
            self.skipTest("this source does not say how deep it is")
        if depth < 2:
            self.skipTest("a depth-one source holds one slot by design")

        source = self.make_source()
        first = source.next_slot()
        if first is None:
            self.skipTest("an empty stream has no slots to hold open")
        second = source.next_slot()
        if second is None:
            self.skipTest("this plan has only one slot")
        self.assertIsNot(first.memory, second.memory, "two slots out at once must not share memory")
        source.release(first, self.release_event())
        source.release(second, self.release_event())


PLAN = [
    [(0, 0, 100), (0, 1, 200)],      # two ranges of file 0
    [(1, 0, 300), (0, 2, 50)],       # a slot may mix files, and need not be in id order
    [(1, 1, 10)],                    # a part full last slot
]


class TestFakeSourceHonoursTheContract(DistributionSourceContract, unittest.TestCase):
    """The contract, run against the fake. A real source mixes in the same suite."""

    def make_source(self):
        return FakeSource(PLAN)

    def expected_ranges(self):
        return {(f, i) for slot in PLAN for f, i, _ in slot}

    def source_depth(self):
        return FakeSource.DEPTH


class TestAnEmptyStreamHonoursTheContract(DistributionSourceContract, unittest.TestCase):
    """A rank whose share is empty is a REAL case, not a degenerate one.

    The distributed path returns early when a rank has nothing to read, and that rank still takes
    part in the collective - it receives from every peer. So its source must answer None at once
    rather than raise, and the contract must survive being run against it."""

    def make_source(self):
        return FakeSource([])

    def expected_ranges(self):
        return set()

    def source_depth(self):
        return FakeSource.DEPTH


class TestTheLayoutRule(unittest.TestCase):
    """The rule as NUMBERS.

    Every other test here compares a source against the constant, so all of them stay green if the
    constant changes. Ranks agree on a value, not on an expression, so the value is pinned."""

    def test_the_alignment_is_512(self):
        self.assertEqual(RANGE_OFFSET_ALIGNMENT, 512)

    def test_an_offset_is_rounded_up_to_it(self):
        self.assertEqual(advance(0, 1)[0], 0)
        self.assertEqual(advance(1, 1)[0], 512)
        self.assertEqual(advance(511, 1)[0], 512)
        self.assertEqual(advance(512, 1)[0], 512, "already on the rule, so not pushed on")
        self.assertEqual(advance(513, 1)[0], 1024)

    def test_a_zero_sized_range_takes_no_pad(self):
        """The ring places them this way, so the rule must say so too: a source that padded them
        would describe a slot the ring laid out differently."""
        self.assertEqual(advance(100, 0), (100, 100))


class TestTheContractRejectsABrokenSource(unittest.TestCase):
    """The suite is only worth something if a broken source fails it.

    Each case names WHICH test must catch it, and requires the rest to pass cleanly - a suite that
    merely 'had a failure somewhere' could be failing for an unrelated reason, or crashing."""

    def contract_result(self, source_factory, expected=None, depth=FakeSource.DEPTH):
        class Case(DistributionSourceContract, unittest.TestCase):
            def make_source(inner):
                return source_factory()

            def expected_ranges(inner):
                return expected if expected is not None else {
                    (f, i) for slot in PLAN for f, i, _ in slot}

            def source_depth(inner):
                return depth

        suite = unittest.TestLoader().loadTestsFromTestCase(Case)
        result = unittest.TestResult()
        suite.run(result)
        return result

    def assert_caught_by(self, result, test_name):
        self.assertEqual([], [f"{t}: {e}" for t, e in result.errors],
                         "the contract must FAIL a broken source, not crash on it")
        failed = {t.id().rsplit(".", 1)[-1] for t, _ in result.failures}
        self.assertIn(test_name, failed)

    def test_sending_the_whole_slot_is_caught(self):
        class WholeSlot(FakeSource):
            def next_slot(self):
                filled = super().next_slot()
                if filled is None:
                    return None
                whole = memoryview(self._out[id(filled.memory)])
                self._out[id(whole)] = self._out.pop(id(filled.memory))
                return Filled(memory=whole, ranges=filled.ranges)

        self.assert_caught_by(self.contract_result(lambda: WholeSlot(PLAN)),
                              "test_the_memory_is_the_used_prefix")

    def test_a_wider_pad_than_the_rule_asks_for_is_caught(self):
        """Still a multiple of the alignment, so only the RULE separates it from a correct source."""
        class Padded(FakeSource):
            def next_slot(self):
                filled = super().next_slot()
                if filled is None:
                    return None
                ranges, cursor = [], 0
                for r in filled.ranges:
                    offset = advance(cursor, r.size)[0] + RANGE_OFFSET_ALIGNMENT
                    ranges.append(SlotRange(r.file_id, r.range_index, offset, r.size))
                    cursor = offset + r.size
                return Filled(memory=filled.memory, ranges=ranges)

        self.assert_caught_by(self.contract_result(lambda: Padded(PLAN)),
                              "test_the_ranges_follow_the_shared_layout_rule")

    def test_tight_packing_is_caught(self):
        """What the staged source did before this rule, and what a new one would reach for."""
        class Tight(FakeSource):
            def next_slot(self):
                filled = super().next_slot()
                if filled is None:
                    return None
                ranges, cursor = [], 0
                for r in filled.ranges:
                    ranges.append(SlotRange(r.file_id, r.range_index, cursor, r.size))
                    cursor += r.size
                return Filled(memory=filled.memory, ranges=ranges)

        self.assert_caught_by(self.contract_result(lambda: Tight(PLAN)),
                              "test_the_ranges_follow_the_shared_layout_rule")

    def test_dropping_a_range_is_caught(self):
        short = [slot[:-1] for slot in PLAN if slot[:-1]]
        self.assert_caught_by(self.contract_result(lambda: FakeSource(short)),
                              "test_every_range_is_delivered_exactly_once_in_the_callers_numbering")

    def test_requiring_a_release_before_the_next_slot_is_caught(self):
        """A source that CLAIMS two slots and then demands a release after one."""
        class DepthOne(FakeSource):
            def next_slot(self):
                if self._out:
                    raise AssertionError("this source demands a release first")
                return super().next_slot()

        result = self.contract_result(lambda: DepthOne(PLAN))
        names = {t.id().rsplit(".", 1)[-1] for t, _ in result.errors + result.failures}
        self.assertIn("test_two_slots_may_be_out_at_once", names)
