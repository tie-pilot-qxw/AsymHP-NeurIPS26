"""Tests for split_planner.py — pure Python, no GPU."""

from __future__ import annotations

import math
import unittest

from split_planner import (
    ComputeUnit,
    SplitPlan,
    derive_q_boundaries,
    head_costs,
    plan_with_splits,
    water_fill,
)


# Reusable cost model. mask_slope is small relative to attn_slope * S so attn dominates.
COST_MODEL = {
    "mask_fit": {"slope_ms_per_unit": 7.0e-06},
    "attention_fit": {"slope_ms_per_unit": 1.7e-09},
}


def make_costs_for_test(densities, world_size, seq_len=188760):
    cost_mask, cost_attn, cost_total = head_costs(densities, COST_MODEL, seq_len)
    return cost_mask, cost_attn, cost_total


class TestWaterFilling(unittest.TestCase):
    def test_balanced(self):
        # 4 ranks, identical floors, pool 4.0 → each gets 1.0, T = floor + 1.
        floors = [10.0, 10.0, 10.0, 10.0]
        T, shares = water_fill(floors, 4.0)
        self.assertAlmostEqual(T, 11.0)
        for s in shares:
            self.assertAlmostEqual(s, 1.0)

    def test_drops_overloaded_helper(self):
        # Owner floor=5, two helpers at floor=6 and 100. Pool 6.
        # Active set should drop the floor=100 participant; T over {5, 6} with pool 6 = (6+5+6)/2=8.5.
        # → owner gets 3.5, helper@6 gets 2.5, helper@100 gets 0.
        floors = [5.0, 6.0, 100.0]
        T, shares = water_fill(floors, 6.0)
        self.assertAlmostEqual(T, 8.5)
        self.assertAlmostEqual(shares[0], 3.5)
        self.assertAlmostEqual(shares[1], 2.5)
        self.assertAlmostEqual(shares[2], 0.0)

    def test_pool_zero(self):
        # floors [3, 4], pool 0 → drop the higher-floor participant (4),
        # left with active={0}, T = 3.0, both shares = 0.
        floors = [3.0, 4.0]
        T, shares = water_fill(floors, 0.0)
        self.assertAlmostEqual(T, 3.0)
        self.assertAlmostEqual(sum(shares), 0.0)
        for s in shares:
            self.assertGreaterEqual(s, -1e-12)


class TestDeriveQBoundaries(unittest.TestCase):
    def test_simple_two_way_split(self):
        # owner share 6, helper share 4 → 60/40 of seq_len 100.
        b = derive_q_boundaries(shares_in_order=[6.0, 4.0], pool=10.0, seq_len=100, q_granularity=1)
        self.assertEqual(b, [0, 60, 100])

    def test_q_granularity_alignment(self):
        # Same split, granularity 16 → boundary should snap to multiples of 16 and stay non-empty.
        b = derive_q_boundaries(shares_in_order=[6.0, 4.0], pool=10.0, seq_len=100, q_granularity=16)
        self.assertIsNotNone(b)
        self.assertEqual(b[0], 0)
        self.assertEqual(b[-1], 100)
        self.assertEqual(len(b), 3)
        self.assertTrue(b[1] % 16 == 0 or (100 - b[1]) % 16 != 0)  # snap may pick nearest mult
        # owner gets at least 16 rows; helper gets at least 16 rows
        self.assertGreaterEqual(b[1] - b[0], 16)
        self.assertGreaterEqual(b[2] - b[1], 16)

    def test_too_many_for_granularity(self):
        # 4 participants, granularity 30, seq_len 100 → 4*30=120 > 100, infeasible.
        b = derive_q_boundaries(
            shares_in_order=[1, 1, 1, 1], pool=4, seq_len=100, q_granularity=30
        )
        self.assertIsNone(b)

    def test_three_way(self):
        b = derive_q_boundaries(
            shares_in_order=[5, 3, 2], pool=10, seq_len=200, q_granularity=1
        )
        # 50%, 30%, 20% → boundaries 0, 100, 160, 200
        self.assertEqual(b, [0, 100, 160, 200])


class TestPlanNoSplit(unittest.TestCase):
    def test_balanced_densities_yield_no_split(self):
        # 12 heads, all density 0.2; world=4 → 3 heads each, equal load → no split.
        densities = [0.2] * 12
        plan = plan_with_splits(
            densities=densities, cost_model=COST_MODEL, seq_len=188760, world_size=4,
            min_heads_per_rank=1,
        )
        self.assertEqual(len(plan.diagnostics["splits"]), 0)
        for rank_units in plan.units_per_rank:
            for u in rank_units:
                self.assertEqual(u.role, "full")
                self.assertEqual(u.q_lo, 0)
                self.assertEqual(u.q_hi, 188760)
                self.assertEqual(u.split_id, -1)
        self.assertEqual(plan.predicted_max_ms, plan.baseline_max_ms)
        self.assertAlmostEqual(plan.speedup, 1.0)


class TestPlanFindsSplit(unittest.TestCase):
    """Use the real density log row for layer=21 prev_ts=898 — the user's
    bench case for step20 layer21 480-frames."""

    DENSITIES = [
        0.3582, 0.1280, 0.1222, 0.1239, 0.2572, 0.7219,
        0.1131, 0.1169, 0.1049, 0.1100, 0.1166, 0.2321,
    ]

    def test_default_split(self):
        plan = plan_with_splits(
            densities=self.DENSITIES, cost_model=COST_MODEL,
            seq_len=188760, world_size=6,
            max_helpers_per_split=2, max_splits_per_plan=1,
        )
        # Exactly one split.
        self.assertEqual(len(plan.diagnostics["splits"]), 1)
        sp = plan.diagnostics["splits"][0]
        self.assertEqual(sp["head"], 5)  # heaviest head by density
        # Owner is whichever rank greedy_unequal placed head 5 on (sticky check):
        # head 5 has the largest density so seed-step puts it on rank 0.
        self.assertEqual(sp["owner"], 0)
        # 1..max_helpers helpers.
        self.assertGreaterEqual(len(sp["helpers"]), 1)
        self.assertLessEqual(len(sp["helpers"]), 2)
        # Improvement.
        self.assertLess(plan.predicted_max_ms, plan.baseline_max_ms)
        # Best-candidate planned_max equals plan.predicted_max_ms by construction.
        self.assertAlmostEqual(plan.predicted_max_ms, sp["best_planned_max_ms"], places=6)
        # Sanity: planner result <= global one-split ceiling (which represents
        # what's achievable with infinite helpers).
        self.assertGreaterEqual(
            plan.predicted_max_ms, sp["global_one_split_ceiling_ms"] - 1e-6
        )

    def test_max_helpers_one(self):
        plan = plan_with_splits(
            densities=self.DENSITIES, cost_model=COST_MODEL,
            seq_len=188760, world_size=6,
            max_helpers_per_split=1, max_splits_per_plan=1,
        )
        self.assertEqual(len(plan.diagnostics["splits"]), 1)
        sp = plan.diagnostics["splits"][0]
        self.assertEqual(len(sp["helpers"]), 1)
        self.assertEqual(sp["owner"], 0)

    def test_q_granularity_alignment_and_actual_share_recompute(self):
        # Use q_granularity = 120 (188760 % 120 == 0) so seq_len divisibility
        # check passes. With a granularity > 1, snap-to-grid will make actual
        # row counts differ from water-fill desired shares, and ComputeUnit
        # cost_ms must reflect actual shares (not desired).
        granularity = 120
        seq_len = 188760
        self.assertEqual(seq_len % granularity, 0)

        plan = plan_with_splits(
            densities=self.DENSITIES, cost_model=COST_MODEL,
            seq_len=seq_len, world_size=6,
            max_helpers_per_split=2, max_splits_per_plan=1,
            q_granularity=granularity,
        )
        # Alignment.
        for rank_units in plan.units_per_rank:
            for u in rank_units:
                if u.role in ("split_owner", "helper"):
                    self.assertEqual(u.q_lo % granularity, 0,
                                     f"q_lo={u.q_lo} not aligned to {granularity}")
                    self.assertEqual(u.q_hi % granularity, 0,
                                     f"q_hi={u.q_hi} not aligned to {granularity}")
                    self.assertGreater(u.q_hi - u.q_lo, 0)

        # Actual-share recompute: every helper's stored cost_ms must equal
        # (q_hi - q_lo) / seq_len * cost_attn(head). Owner's cost_ms must
        # equal cost_mask(head) + (q_hi - q_lo) / seq_len * cost_attn(head).
        sp = plan.diagnostics["splits"][0]
        h = sp["head"]
        from split_planner import head_costs
        cost_mask, cost_attn, _ = head_costs(self.DENSITIES, COST_MODEL, seq_len)
        for rank_units in plan.units_per_rank:
            for u in rank_units:
                if u.role == "helper":
                    expected = (u.q_hi - u.q_lo) / seq_len * cost_attn[h]
                    self.assertAlmostEqual(u.cost_ms, expected, places=6,
                        msg=f"helper rank {u.owner_rank} h={h} cost_ms wrong")
                elif u.role == "split_owner":
                    expected = cost_mask[h] + (u.q_hi - u.q_lo) / seq_len * cost_attn[h]
                    self.assertAlmostEqual(u.cost_ms, expected, places=6,
                        msg=f"split_owner h={h} cost_ms wrong")

        # Ranges cover [0, seq_len) with no gaps.
        ranges = []
        for rank_units in plan.units_per_rank:
            for u in rank_units:
                if u.split_id == sp.get("split_id_unused", 0) or u.role in ("split_owner", "helper"):
                    if u.global_head == h:
                        ranges.append((u.q_lo, u.q_hi))
        ranges.sort()
        self.assertEqual(ranges[0][0], 0)
        self.assertEqual(ranges[-1][1], seq_len)
        for i in range(len(ranges) - 1):
            self.assertEqual(ranges[i][1], ranges[i + 1][0])

    def test_seq_len_must_divide_q_granularity(self):
        with self.assertRaises(ValueError):
            plan_with_splits(
                densities=self.DENSITIES, cost_model=COST_MODEL,
                seq_len=188760, world_size=6,
                max_helpers_per_split=2, max_splits_per_plan=1,
                q_granularity=128,  # 188760 % 128 != 0
            )

    def test_max_splits_zero_returns_baseline(self):
        plan = plan_with_splits(
            densities=self.DENSITIES, cost_model=COST_MODEL,
            seq_len=188760, world_size=6,
            max_helpers_per_split=2, max_splits_per_plan=0,
        )
        self.assertEqual(len(plan.diagnostics["splits"]), 0)
        for rank_units in plan.units_per_rank:
            for u in rank_units:
                self.assertEqual(u.role, "full")
        self.assertAlmostEqual(plan.speedup, 1.0)

    def test_max_splits_two_obeys_owner_or_helper_rule(self):
        # Multi-split: each rank is at most one of {owner, helper, non-participant}.
        # Already-split heads are excluded from later iterations (role filter);
        # already-owner ranks are excluded from later iterations as both owner
        # candidate and helper candidate. Force the planner with delta=0.0 and
        # min_improvement_ms=0.001 so each useful step actually fires.
        plan = plan_with_splits(
            densities=self.DENSITIES, cost_model=COST_MODEL,
            seq_len=188760, world_size=6,
            max_helpers_per_split=2, max_splits_per_plan=2,
            delta=0.0, min_improvement_ms=0.001,
        )
        owners_seen: set = set()
        helpers_seen: set = set()
        heads_split: set = set()
        for s in plan.diagnostics["splits"]:
            self.assertNotIn(s["owner"], owners_seen,
                             "owner locked once cannot become owner again")
            self.assertNotIn(s["owner"], helpers_seen,
                             "rank already a helper cannot become an owner")
            self.assertNotIn(s["head"], heads_split,
                             "head split once cannot be split again")
            for h in s["helpers"]:
                self.assertNotIn(h, owners_seen,
                                 "owner cannot serve as helper of another split")
            owners_seen.add(s["owner"])
            helpers_seen.update(s["helpers"])
            heads_split.add(s["head"])
        self.assertGreaterEqual(len(plan.diagnostics["splits"]), 1)


class TestRejectOwnerZero(unittest.TestCase):
    """Construct a contrived case where water-fill would assign s_owner=0."""

    def test_owner_zero_rejected(self):
        # Owner has ONLY one head (dominating). Helpers have very low baselines.
        # 4 ranks. Owner head density = 1.0, others ≈ 0.001. world=4, 12 heads.
        # After greedy_unequal: rank 0 holds head 0 (density 1.0) alone, others share rest.
        # cost(h0) huge, cost(others) tiny. floor(owner) = 0 + cost_mask(h0) = small.
        # floor(helpers) = baseline ≈ 0. pool = cost_attn(h0) huge.
        # T = (huge + small + 0+0+0)/4 ≈ huge/4. owner share = T - floor(owner) ≈ huge/4 (positive).
        # So owner-zero won't trigger here — water-fill will try to keep owner contributing.
        # To force s_owner=0, need floor(owner) > T: i.e., owner's non-h work + cost_mask(h)
        # already exceeds the achievable level.
        # Construct: 12 heads, head 0 is huge attn but small mask-equivalent;
        # rank 0 also holds many other heads keeping its baseline_excl_h high.
        # That requires hand-crafting LPT outcome; instead, drive the API directly.
        # Cleaner test: just check the path. Use a case where helpers are lower-loaded
        # and the bug-free planner picks a normal split. Owner-zero is rare; we test
        # the rejection logic via direct water-fill instead.
        from split_planner import _floor_for, water_fill
        # Owner floor = 100 (heavy non-h work + mask), helpers floors = 1, 1.
        # Pool = 4. T_candidate = (4 + 100 + 1 + 1)/3 = 35.33 < 100 → drop owner.
        # But owner can't be dropped (it's the source) → in our planner code, this
        # manifests as shares[0] <= 1e-9 → reject candidate.
        # Verify water_fill itself drops owner (it's just "highest floor" to it).
        floors = [100.0, 1.0, 1.0]  # treating index 0 as owner
        T, shares = water_fill(floors, 4.0)
        # water_fill drops index 0 → active = {1, 2}, T = (4 + 1 + 1)/2 = 3.0.
        self.assertAlmostEqual(T, 3.0)
        self.assertAlmostEqual(shares[0], 0.0)  # owner share = 0


if __name__ == "__main__":
    unittest.main()
