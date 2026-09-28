#!/usr/bin/env python
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_sp_all2all_attention import (
    estimate_rank_comm_cost,
    greedy_lpt_assignment_comm_aware,
    validate_comm_cost_model_shape,
)


def make_model(world_size, pull_slopes, push_slopes=None):
    if push_slopes is None:
        push_slopes = [0.0] * world_size

    def fits(slopes):
        return [
            {
                "rank": rank,
                "intercept_ms": 0.0,
                "slope_ms_per_head": float(slopes[rank]),
                "r2": 1.0,
                "num_samples": 2,
            }
            for rank in range(world_size)
        ]

    return {
        "schema_version": 1,
        "world_size": world_size,
        "pull_qkv": {"rank_fits": fits(pull_slopes)},
        "push_out": {"rank_fits": fits(push_slopes)},
    }


class CommunicationAwarePlannerTest(unittest.TestCase):
    def test_communication_model_validates_every_shape_dimension(self):
        shape = {
            "batch": 1,
            "total_heads": 12,
            "s_local": 27900,
            "seq_len": 111600,
            "head_dim": 128,
            "dtype": "torch.bfloat16",
        }
        model = {
            "world_size": 4,
            "shape": shape.copy(),
        }
        kwargs = {
            "world_size": 4,
            **shape,
        }
        validate_comm_cost_model_shape(model, **kwargs)

        for name in ("batch", "total_heads", "s_local", "seq_len", "head_dim"):
            with self.subTest(name=name):
                mismatched = {
                    **model,
                    "shape": {
                        **shape,
                        name: int(shape[name]) + 1,
                    },
                }
                with self.assertRaisesRegex(ValueError, name):
                    validate_comm_cost_model_shape(mismatched, **kwargs)

        mismatched_dtype = {
            **model,
            "shape": {
                **shape,
                "dtype": "torch.float16",
            },
        }
        with self.assertRaisesRegex(ValueError, "dtype"):
            validate_comm_cost_model_shape(mismatched_dtype, **kwargs)

        for name in shape:
            with self.subTest(missing=name):
                missing = {
                    **model,
                    "shape": {
                        key: value
                        for key, value in shape.items()
                        if key != name
                    },
                }
                with self.assertRaisesRegex(ValueError, name):
                    validate_comm_cost_model_shape(missing, **kwargs)

    def test_rank_comm_cost_sums_pull_and_push(self):
        model = make_model(2, [2.0, 3.0], [0.5, 1.0])
        self.assertAlmostEqual(estimate_rank_comm_cost(model, 0, 4), 10.0)
        self.assertAlmostEqual(estimate_rank_comm_cost(model, 1, 2), 8.0)
        self.assertEqual(estimate_rank_comm_cost(model, 0, 0), 0.0)

    def test_high_per_head_comm_prefers_balanced_counts(self):
        # Compute-only LPT leaves the heavy head alone. With communication
        # dominating, three heads per rank is the lower-makespan placement.
        costs = [10.0] + [2.0] * 11
        model = make_model(4, [10.0] * 4)
        assigned, loads = greedy_lpt_assignment_comm_aware(costs, 4, model)
        self.assertEqual(sorted(len(heads) for heads in assigned), [3, 3, 3, 3])
        self.assertEqual(sorted(head for heads in assigned for head in heads), list(range(12)))
        self.assertAlmostEqual(max(loads), 44.0)

    def test_rank_specific_comm_can_use_unequal_counts(self):
        costs = [1.0] * 8
        model = make_model(2, [10.0, 0.0])
        assigned, loads = greedy_lpt_assignment_comm_aware(costs, 2, model)
        self.assertEqual([len(heads) for heads in assigned], [1, 7])
        self.assertEqual(sorted(head for heads in assigned for head in heads), list(range(8)))
        self.assertAlmostEqual(max(loads), 11.0)

    def test_min_heads_per_rank_is_preserved(self):
        costs = [8.0, 7.0, 6.0, 5.0, 1.0, 1.0, 1.0, 1.0]
        model = make_model(4, [1.0] * 4, [1.0] * 4)
        assigned, _ = greedy_lpt_assignment_comm_aware(
            costs,
            4,
            model,
            min_heads_per_rank=2,
        )
        self.assertTrue(all(len(heads) >= 2 for heads in assigned))
        self.assertEqual(sorted(head for heads in assigned for head in heads), list(range(8)))


if __name__ == "__main__":
    unittest.main()
