"""Unit and invariant tests of the version 1.1 verification code: python -m mux_s1.tests_v11"""
import unittest

import numpy as np

from . import adapters as AD
from . import core as C
from . import realdata as RD


def tiny(model="M0_markov", n_sess=1, n_pairs=4, n_tr=8, N=80, faces=False, seed=0):
    return AD.synthetic_sessions(model, seed=seed, n_sess=n_sess, n_pairs=n_pairs, n_tr=n_tr,
                                 N=N, with_faces=faces)


class TestCanonical(unittest.TestCase):
    def test_structure_accepted_and_validated(self):
        ss = tiny()
        self.assertTrue(RD.check_sessions(ss))
        self.assertEqual(ss[0].n_neurons, 80)
        bad = tiny()
        bad[0].pairs[0].bino_a = bad[0].pairs[0].bino_a[:, :, :5]
        with self.assertRaises(ValueError):
            RD.check_sessions(bad)

    def test_rebin_conserves_counts(self):
        x = np.random.default_rng(0).integers(0, 4, (3, 70, 5))
        for k in (1, 2, 5, 7):
            self.assertEqual(RD.rebin(x, k).sum(), x[:, : (70 // k) * k].sum())

    def test_template_is_bin_width_invariant(self):
        """A template is a rate, so summing neighbouring bins must leave it unchanged."""
        x = tiny()[0].pairs[0].prime_a
        t1 = RD.template(x, 0.010)
        t5 = RD.template(RD.rebin(x, 5), 0.050)
        self.assertTrue(np.allclose(t1, t5, rtol=1e-10))


class TestPrimitives(unittest.TestCase):
    def test_llr_matches_version_1_0(self):
        s = tiny()[0]
        p = s.pairs[0]
        lamA, lamB = RD.pair_templates(p, 0.010)
        self.assertTrue(np.allclose(RD.llr(p.bino_a, lamA, lamB, 0.010),
                                    C.llr(p.bino_a, lamA, lamB)))

    def test_fit_D_matches_version_1_0_at_10ms(self):
        acf = np.exp(-2 * 10 * np.arange(21) / 80.0) - 0.05
        self.assertAlmostEqual(RD.fit_D_ms(acf, 10.0)[0], C.fit_D(acf)[0], places=6)

    def test_fit_D_is_bin_width_consistent(self):
        """The same underlying exponential read at 10 and at 20 ms must give a similar D."""
        D_true = 90.0
        a10 = np.exp(-2 * 10 * np.arange(21) / D_true) - 0.02
        a20 = np.exp(-2 * 20 * np.arange(11) / D_true) - 0.02
        self.assertLess(abs(RD.fit_D_ms(a10, 10.0)[0] - RD.fit_D_ms(a20, 20.0)[0]), 12.0)


class TestStates(unittest.TestCase):
    def test_states_shape_and_conscious_convention(self):
        s = tiny(n_pairs=3, n_tr=6)
        st = RD.session_states(s[0], bin_ms=10.0)
        self.assertEqual(len(st), 3)
        pr = st[0]
        self.assertEqual(pr["c"].shape, (pr["na"] + pr["nb"], s[0].n_bins))
        self.assertTrue(pr["c"].dtype == bool)

    def test_labels_from_a_subset_use_only_that_subset(self):
        """Two different label subsets must give different state paths; the same subset the same."""
        s = tiny(n_pairs=2, n_tr=6, N=60)[0]
        a = RD.session_states(s, bin_ms=10.0, label_idx=np.arange(0, 30))
        b = RD.session_states(s, bin_ms=10.0, label_idx=np.arange(30, 60))
        c = RD.session_states(s, bin_ms=10.0, label_idx=np.arange(0, 30))
        self.assertTrue(np.array_equal(a[0]["c"], c[0]["c"]))
        self.assertFalse(np.array_equal(a[0]["c"], b[0]["c"]))

    def test_dwell_scales_with_bin_width(self):
        s = tiny(n_pairs=4, n_tr=8, N=200)[0]
        d10 = RD.dwell_stats([RD.session_states(s, bin_ms=10.0)])["median_ms"]
        d20 = RD.dwell_stats([RD.session_states(s, bin_ms=20.0)])["median_ms"]
        self.assertGreaterEqual(d20, d10)


class TestChecks(unittest.TestCase):
    def test_counts_block_runs_and_reports_every_check(self):
        res = RD.run_counts_block(tiny(n_sess=2, n_pairs=4, n_tr=8, N=150))
        for k in ("A1_priming_state1", "A2_acf_first_lag", "A3_bino_state1", "A4_dwell_by_binwidth"):
            self.assertIn(k, res["A"])
        for k in ("B1_dwell_over_D", "B2_D_global_over_D_trial", "B3_acf_over_calibration",
                  "B4_label_threshold", "B5_fig3a_equal_noise", "B11_dwell_cv", "B12_overdispersion"):
            self.assertIn(k, res["B"])

    def test_signal_gate_blanks_the_three_gated_predictions(self):
        res = RD.run_counts_block(tiny(n_sess=1, n_pairs=4, n_tr=8, N=60))   # few neurons: weak signal
        if not res["signal_ok"]:
            for k in RD.SIGNAL_GATED:
                self.assertIsNone(res["B"][k]["passed"])

    def test_fig3a_equal_noise_separates_switching_from_averaging(self):
        """The pre-registered sign of B5: positive for switching, negative for a static mixture."""
        sw = RD.fig3a_equal_noise(tiny("M0_markov", n_sess=2, n_pairs=8, n_tr=12, N=300))
        av = RD.fig3a_equal_noise(tiny("C1_averaging", n_sess=2, n_pairs=8, n_tr=12, N=300))
        self.assertGreater(sw["unequal_noise"], 0)
        self.assertGreater(av["unequal_noise"], 0)          # the authors' version is won by both
        self.assertGreater(sw["equal_noise"], av["equal_noise"])

    def test_probes_leave_the_data_untouched(self):
        s = tiny(n_pairs=3, n_tr=8, N=120)
        before = [p.bino_a.copy() for p in s[0].pairs]
        RD.run_counts_block(s)
        for p, b in zip(s[0].pairs, before):
            self.assertTrue(np.array_equal(p.bino_a, b))


class TestGainBlock(unittest.TestCase):
    def test_skipped_without_a_face_space(self):
        g = RD.gain_block(tiny(n_pairs=3, n_tr=6))
        self.assertFalse(g["available"])

    def test_skipped_when_a_session_has_too_few_faces(self):
        """The coding model needs more faces than features; too few must degrade, not crash."""
        g = RD.gain_block(tiny("M0_markov", n_sess=1, n_pairs=10, n_tr=6, N=100, faces=True))
        self.assertFalse(g["available"])
        self.assertTrue(g["skipped"])

    def test_runs_with_faces_and_reports_the_four_checks(self):
        g = RD.gain_block(tiny("M4_adaptation", n_sess=1, n_pairs=35, n_tr=8, N=150, faces=True))
        self.assertTrue(g["available"])
        for k in ("B6_decoder_without_mean_rate", "B7_gain_early_minus_late",
                  "B9_mean_rate_sign", "B10_gain_vs_r2"):
            self.assertIn(k, g)
        self.assertGreaterEqual(g["decoder"]["raw"], 0.5)   # oriented accuracy


if __name__ == "__main__":
    unittest.main(verbosity=1)
