import unittest

import torch

from split_runtime import snap_to_cluster_boundaries


class SnapToClusterBoundariesTest(unittest.TestCase):
    def test_leaves_non_empty_tail_segment(self):
        qc_sz = torch.tensor([3, 2, 5, 10], dtype=torch.int32)

        rows, clusters = snap_to_cluster_boundaries(qc_sz, [0, 18, 20])

        self.assertEqual(clusters, [0, 3, 4])
        self.assertEqual(rows, [0, 10, 20])

    def test_leaves_one_cluster_per_remaining_segment(self):
        qc_sz = torch.tensor([1, 1, 1, 1, 1], dtype=torch.int32)

        rows, clusters = snap_to_cluster_boundaries(qc_sz, [0, 4, 5])

        self.assertEqual(clusters, [0, 4, 5])
        self.assertEqual(rows, [0, 4, 5])


if __name__ == "__main__":
    unittest.main()
