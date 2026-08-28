"""Tests that the Toad adapter returns finite, auditable metrics."""

from __future__ import annotations

import unittest

from ..toad_adapter import resolve_backend, toad_iv_from_bins, toad_ks, toad_psi


class ToadAdapterTest(unittest.TestCase):
    def test_backend_and_iv(self) -> None:
        self.assertEqual(resolve_backend("toad"), "toad")
        iv, by_bin = toad_iv_from_bins(
            [0, 0, 1, 1, 2, 2],
            [0, 1, 0, 1, 1, 1],
        )
        self.assertIsNotNone(iv)
        self.assertEqual(set(by_bin), {0, 1, 2})
        self.assertGreater(float(iv), 0.0)

    def test_ks_and_psi_are_finite_when_a_month_misses_a_bin(self) -> None:
        self.assertEqual(toad_ks([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]), 1.0)
        psi = toad_psi([0, 0], [0, 1, 2], support=[0, 1, 2])
        self.assertIsNotNone(psi)
        self.assertGreater(float(psi), 0.0)


if __name__ == "__main__":
    unittest.main()
