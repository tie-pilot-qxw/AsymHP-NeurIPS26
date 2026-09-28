import unittest
import sys
import importlib.util
from pathlib import Path

import torch

_CONTEXT_PATH = Path(__file__).resolve().parent / "wan_sp" / "context.py"
_SPEC = importlib.util.spec_from_file_location("_head_index_context", _CONTEXT_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
LayerPlan = _MODULE.LayerPlan


class HeadIndexStagingTest(unittest.TestCase):
    def make_plan(self):
        return LayerPlan(
            head_order=[0, 2, 1, 3],
            real_heads_per_rank=[2, 2],
            max_hpr=2,
            restore_index=[0, 2, 1, 3],
            assigned=[[0, 2], [1, 3]],
            strategy="greedy_unequal",
        )

    def test_caches_int32_and_int64_device_indices(self):
        plan = self.make_plan()
        idx32 = plan.h_idxs_tensor(1, torch.device("cpu"))
        idx64 = plan.h_idxs_tensor(1, torch.device("cpu"), dtype=torch.long)

        self.assertEqual(idx32.dtype, torch.int32)
        self.assertEqual(idx64.dtype, torch.int64)
        self.assertEqual(idx32.tolist(), [1, 3])
        self.assertEqual(idx64.tolist(), [1, 3])
        self.assertEqual(
            idx32.data_ptr(),
            plan.h_idxs_tensor(1, torch.device("cpu")).data_ptr(),
        )
        self.assertEqual(
            idx64.data_ptr(),
            plan.h_idxs_tensor(1, torch.device("cpu"), dtype=torch.long).data_ptr(),
        )

    def test_rejects_unneeded_index_dtype(self):
        plan = self.make_plan()
        with self.assertRaisesRegex(ValueError, "unsupported head-index dtype"):
            plan.h_idxs_tensor(0, torch.device("cpu"), dtype=torch.int16)

    def test_caches_other_plan_indices(self):
        plan = self.make_plan()
        order = plan.head_order_tensor(torch.device("cpu"))
        restore = plan.restore_tensor(torch.device("cpu"))

        self.assertEqual(order.tolist(), [0, 2, 1, 3])
        self.assertEqual(restore.tolist(), [0, 2, 1, 3])
        self.assertEqual(
            order.data_ptr(), plan.head_order_tensor(torch.device("cpu")).data_ptr()
        )
        self.assertEqual(
            restore.data_ptr(), plan.restore_tensor(torch.device("cpu")).data_ptr()
        )


if __name__ == "__main__":
    unittest.main()
