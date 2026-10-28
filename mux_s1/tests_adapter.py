"""Tests of the reader of the released files: python3 -m mux_s1.tests_adapter

These run without the release. They cover the part of the adapter that can be wrong silently: the
binning arithmetic, the sorting of trials into the canonical containers, the two file vintages and
the axis order of a dense raster. The ground truth is planted by hand, so every expected count
below is arithmetic rather than a recorded output.

The strongest of them, `test_the_two_vintages_agree`, writes the SAME planted data twice, once as
a v5 file of spike times and once as a v7.3 file of a dense raster, and requires the canonical
form to come out identical. A mistake in either reading path breaks it.
"""
import os
import shutil
import tempfile
import unittest

import numpy as np

from . import adapters as A

# ---- the planted session ---------------------------------------------------------------------
N_NEURONS = 6
ONSETS_S = [1.0, 1.25, 2.0, 0.5, 1.75, 3.0]     # a different onset in every trial
BIN_MS, WINDOW_MS = 10.0, (300, 1000)
KINDS = dict(binocular=1, priming=2, unambiguous=3)


def planted_spikes(neuron):
    """Relative spike times in ms: one spike in each bin 0..neuron, plus two outside the window."""
    return [300.0 + 10.0 * k for k in range(neuron + 1)] + [100.0, 1500.0]


def expected_counts(n_bins=70):
    """(T, N) counts every reading path must produce for any trial."""
    out = np.zeros((n_bins, N_NEURONS), np.int32)
    for j in range(N_NEURONS):
        out[:j + 1, j] = 1
    return out


def trial_table():
    """(kind, pair, flashed) per trial: two pairs, three trials of each face, priming for each."""
    rows = []
    for pair, (fa, fb) in enumerate([(0, 1), (2, 3)]):
        for f in (fa, fb):
            rows += [(KINDS["binocular"], pair, f)] * 3
            rows += [(KINDS["priming"], pair, f)] * 2
    return rows


def write_v5(path):
    """A v5 .mat of spike times, cell array of shape (n_trials, n_neurons) as MATLAB stores it."""
    from scipy.io import savemat
    rows = trial_table()
    n = len(rows)
    cells = np.empty((n, N_NEURONS), dtype=object)
    for t in range(n):
        for j in range(N_NEURONS):
            cells[t, j] = np.asarray(ONSETS_S[t % len(ONSETS_S)] + np.asarray(planted_spikes(j)) / 1000.0)
    savemat(path, dict(
        spikes=cells,
        onset=np.asarray([ONSETS_S[t % len(ONSETS_S)] for t in range(n)], float),
        kind=np.asarray([r[0] for r in rows], float),
        pair=np.asarray([r[1] for r in rows], float),
        flashed=np.asarray([r[2] for r in rows], float),
        unit_id=np.arange(N_NEURONS, dtype=float) + 101))
    return path


def write_v73(path, ms_per_sample=1.0, n_samples=4000):
    """A v7.3 (HDF5) .mat of a dense raster, with the axes reversed as h5py exposes them."""
    import h5py
    rows = trial_table()
    n = len(rows)
    mat = np.zeros((n, n_samples, N_NEURONS), np.uint8)       # MATLAB order
    for t in range(n):
        onset_sample = int(round(ONSETS_S[t % len(ONSETS_S)] * 1000.0 / ms_per_sample))
        for j in range(N_NEURONS):
            for rel in planted_spikes(j):
                s = onset_sample + int(round(rel / ms_per_sample))
                if 0 <= s < n_samples:
                    mat[t, s, j] += 1
    with h5py.File(path, "w", userblock_size=512) as f:
        f.create_dataset("raster", data=np.transpose(mat, (2, 1, 0)))   # as MATLAB writes it
        for name, col in (("onset", [ONSETS_S[t % len(ONSETS_S)] * 1000.0 / ms_per_sample
                                     for t in range(n)]),
                          ("kind", [r[0] for r in rows]), ("pair", [r[1] for r in rows]),
                          ("flashed", [r[2] for r in rows])):
            f.create_dataset(name, data=np.asarray(col, float).reshape(1, n))
    with open(path, "r+b") as fh:
        fh.write(b"MATLAB 7.3 MAT-file, Platform: test, Created by mux_s1.tests_adapter")
    return path


FIELDS_V5 = dict(spike_times="spikes", raster=None, onset="onset", trial_kind="kind",
                 pair_id="pair", face_flashed="flashed", face_other=None, neuron_ids="unit_id",
                 faces=None, ms_per_sample=1.0, raster_order="trial,sample,neuron",
                 kind_codes=dict(KINDS))
FIELDS_V73 = dict(FIELDS_V5, spike_times=None, raster="raster", neuron_ids=None)


class _Fields:
    """Fills MAT_FIELDS for one test and puts it back afterwards."""

    def __init__(self, fields):
        self.fields = fields

    def __enter__(self):
        self.saved = dict(A.MAT_FIELDS)
        A.MAT_FIELDS.clear()
        A.MAT_FIELDS.update(self.fields)

    def __exit__(self, *a):
        A.MAT_FIELDS.clear()
        A.MAT_FIELDS.update(self.saved)


# ==============================================================================================
class TestBinning(unittest.TestCase):
    def test_the_window_is_a_whole_number_of_bins(self):
        self.assertEqual(A.bin_edges(10.0, (300, 1000)), (300.0, 70))
        self.assertEqual(A.bin_edges(50.0, (300, 1000)), (300.0, 14))
        with self.assertRaises(ValueError):
            A.bin_edges(30.0, (300, 1000))

    def test_a_spike_lands_in_the_bin_that_starts_at_its_time(self):
        c = A.bin_spike_times(np.array([1.300, 1.3099, 1.310]), 1.0, 10.0, (300, 1000))
        self.assertEqual(c[0], 2)            # [300, 310) holds 300 and 309.9 ms
        self.assertEqual(c[1], 1)            # [310, 320) holds 310 ms
        self.assertEqual(c.sum(), 3)

    def test_a_spike_on_the_first_edge_is_kept_at_every_onset(self):
        """The float error of the subtraction must not thin the first bin, onset by onset."""
        for onset in ONSETS_S + [0.123, 7.77, 12.5]:
            c = A.bin_spike_times(np.array([onset + 0.300]), onset, 10.0, (300, 1000))
            self.assertEqual(c[0], 1, f"the edge spike was dropped at onset {onset}")
            self.assertEqual(c.sum(), 1)

    def test_spikes_outside_the_window_are_dropped(self):
        c = A.bin_spike_times(np.array([1.1, 1.2999, 1.5, 2.0, 1.4]), 1.0, 10.0, (300, 1000))
        self.assertEqual(c.sum(), 2)         # only 1.4 s and 1.5 s are inside 300-1000 ms
        self.assertEqual(c.dtype, np.int32)

    def test_the_planted_pattern_bins_to_the_expected_counts(self):
        for j in range(N_NEURONS):
            t = 1.0 + np.asarray(planted_spikes(j)) / 1000.0
            c = A.bin_spike_times(t, 1.0, BIN_MS, WINDOW_MS)
            self.assertEqual(c.sum(), j + 1)
            self.assertTrue(np.array_equal(c[:j + 1], np.ones(j + 1, np.int32)))
            self.assertEqual(c[j + 1:].sum(), 0)

    def test_a_raster_sums_into_bins(self):
        r = np.zeros((2000, 3), np.uint8)
        r[1300, 0] = 1                       # 300 ms after an onset at sample 1000
        r[1305, 0] = 1                       # the same bin
        r[1310, 1] = 1                       # the next bin
        out = A.bin_raster(r, 1000, 10.0, (300, 1000))
        self.assertEqual(out[0, 0], 2)
        self.assertEqual(out[1, 1], 1)
        self.assertEqual(out.sum(), 3)
        self.assertEqual(out.shape, (70, 3))

    def test_a_trial_that_ends_inside_the_window_is_padded_not_truncated(self):
        r = np.ones((1400, 2), np.uint8)     # the window needs 1700 samples, the trial has 1400
        out = A.bin_raster(r, 1000, 10.0, (300, 1000))
        self.assertEqual(out.shape, (70, 2))
        self.assertEqual(out[:10].sum(), 10 * 10 * 2)      # the samples that exist
        self.assertEqual(out[-1].sum(), 0)                 # the padding

    def test_a_bin_shorter_than_a_sample_is_refused(self):
        with self.assertRaises(ValueError):
            A.bin_raster(np.zeros((100, 1), np.uint8), 0, 0.5, (0, 10), ms_per_sample=1.0)


class TestAssembly(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_an_unfilled_map_raises_instead_of_dropping_a_condition(self):
        with _Fields(dict(FIELDS_V5, trial_kind=None)):
            with self.assertRaises(KeyError):
                A._require_fields()
        with _Fields(dict(FIELDS_V5, kind_codes=dict(binocular=1, priming=2, unambiguous=None))):
            with self.assertRaises(KeyError):
                A._require_fields()

    def test_a_v5_file_of_spike_times_reads_into_the_canonical_form(self):
        p = write_v5(os.path.join(self.dir, "session_01.mat"))
        with _Fields(FIELDS_V5):
            s = A.session_from_mat(p, bin_ms=BIN_MS, window_ms=WINDOW_MS)
        self.assertEqual(len(s.pairs), 2)
        self.assertEqual(s.n_neurons, N_NEURONS)
        self.assertEqual(s.bin_ms, BIN_MS)
        exp = expected_counts()
        for pr in s.pairs:
            self.assertEqual(pr.bino_a.shape, (3, 70, N_NEURONS))
            self.assertEqual(pr.prime_a.shape, (2, 70, N_NEURONS))
            self.assertIsNone(pr.unamb_a)                  # none were planted
            for f in ("bino_a", "bino_b", "prime_a", "prime_b"):
                v = getattr(pr, f)
                self.assertTrue(np.issubdtype(v.dtype, np.integer))
                self.assertGreaterEqual(v.min(), 0)
                for tr in range(v.shape[0]):
                    self.assertTrue(np.array_equal(v[tr], exp))
        self.assertEqual([(p_.face_a, p_.face_b) for p_ in s.pairs], [(0, 1), (2, 3)])

    def test_the_two_vintages_agree(self):
        """The same planted data, as v5 spike times and as a v7.3 raster, must read the same."""
        p5 = write_v5(os.path.join(self.dir, "session_01.mat"))
        p73 = write_v73(os.path.join(self.dir, "session_02.mat"))
        with _Fields(FIELDS_V5):
            a = A.session_from_mat(p5, bin_ms=BIN_MS, window_ms=WINDOW_MS)
        with _Fields(FIELDS_V73):
            b = A.session_from_mat(p73, bin_ms=BIN_MS, window_ms=WINDOW_MS)
        self.assertEqual(len(a.pairs), len(b.pairs))
        self.assertEqual(a.n_neurons, b.n_neurons)
        for pa, pb in zip(a.pairs, b.pairs):
            self.assertEqual((pa.face_a, pa.face_b), (pb.face_a, pb.face_b))
            for f in ("bino_a", "bino_b", "prime_a", "prime_b"):
                self.assertTrue(np.array_equal(getattr(pa, f), getattr(pb, f)), f)

    def test_a_wrong_axis_order_never_passes_for_the_right_answer(self):
        """Declaring the axes wrong must raise or give a wrong shape, never the correct form."""
        p73 = write_v73(os.path.join(self.dir, "session_02.mat"))
        with _Fields(FIELDS_V73):
            good = A.session_from_mat(p73, bin_ms=BIN_MS, window_ms=WINDOW_MS)
        for order in ("neuron,sample,trial", "sample,trial,neuron", "trial,neuron,sample"):
            with _Fields(dict(FIELDS_V73, raster_order=order)):
                try:
                    bad = A.session_from_mat(p73, bin_ms=BIN_MS, window_ms=WINDOW_MS)
                except (IndexError, ValueError):
                    continue                               # it fails loudly, which is enough
            # the neuron count can match by accident, so the counts themselves are what is checked
            same = (bad.n_neurons == good.n_neurons and len(bad.pairs) == len(good.pairs)
                    and all(np.array_equal(getattr(x, f), getattr(y, f))
                            for x, y in zip(bad.pairs, good.pairs)
                            for f in ("bino_a", "bino_b", "prime_a", "prime_b")))
            self.assertFalse(same, f"the order {order} silently gave the right counts")
        with _Fields(dict(FIELDS_V73, raster_order="trial,sample,sample")):
            with self.assertRaises(ValueError):
                A.session_from_mat(p73, bin_ms=BIN_MS, window_ms=WINDOW_MS)

    def test_session_files_sorts_numerically(self):
        for name in ("session_10.mat", "session_2.mat", "session_1.mat", "notes.txt"):
            open(os.path.join(self.dir, name), "w").close()
        got = [os.path.basename(f) for f in A.session_files(self.dir)]
        self.assertEqual(got, ["session_1.mat", "session_2.mat", "session_10.mat"])


class TestCache(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_a_session_survives_the_round_trip_exactly(self):
        s = A.synthetic_sessions("M0_markov", seed=0, n_sess=1, n_pairs=3, n_tr=4, N=20,
                                 with_faces=True)[0]
        p = A.save_session(s, os.path.join(self.dir, "s0.npz"))
        back = A.load_session(p)
        self.assertEqual(back.bin_ms, s.bin_ms)
        self.assertEqual(back.name, s.name)
        self.assertEqual(len(back.pairs), len(s.pairs))
        self.assertTrue(np.allclose(back.faces, s.faces))
        for pa, pb in zip(s.pairs, back.pairs):
            for f in ("bino_a", "bino_b", "prime_a", "prime_b"):
                self.assertTrue(np.array_equal(getattr(pa, f), getattr(pb, f)))
            self.assertEqual((pa.face_a, pa.face_b), (pb.face_a, pb.face_b))

    def test_the_cache_is_far_smaller_than_the_raster_it_came_from(self):
        s = A.synthetic_sessions("M0_markov", seed=0, n_sess=1, n_pairs=5, n_tr=5, N=50)[0]
        p = A.save_session(s, os.path.join(self.dir, "s0.npz"))
        raw = sum(getattr(q, f).nbytes for q in s.pairs
                  for f in ("bino_a", "bino_b", "prime_a", "prime_b"))
        self.assertLess(os.path.getsize(p), raw)


class TestPlumbing(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_the_report_counts_what_was_planted(self):
        p = write_v5(os.path.join(self.dir, "session_01.mat"))
        with _Fields(FIELDS_V5):
            s = A.session_from_mat(p, bin_ms=BIN_MS, window_ms=WINDOW_MS)
        r = A.plumbing_report([s])[0]
        self.assertEqual(r["n_neurons"], N_NEURONS)
        self.assertEqual(r["n_pairs"], 2)
        self.assertEqual(r["n_trials_used"], 20)           # 2 pairs x 2 faces x (3 bino + 2 prime)
        self.assertEqual(r["n_trials_in_file"], 20)
        self.assertEqual(r["bins_per_trial"], 70)
        self.assertEqual(r["skipped_pairs"], 0)
        # every trial holds sum(j+1 for j<6) = 21 spikes over 6 neurons in a 700 ms window
        self.assertAlmostEqual(r["mean_rate_hz"], round(21 / 6 / 0.7, 2), places=2)

    def test_the_table_has_a_row_per_session(self):
        p = write_v5(os.path.join(self.dir, "session_01.mat"))
        with _Fields(FIELDS_V5):
            s = A.session_from_mat(p, bin_ms=BIN_MS, window_ms=WINDOW_MS)
        t = A.plumbing_table(A.plumbing_report([s, s]))
        self.assertEqual(len(t.strip().split("\n")), 4)    # header, rule, two rows


if __name__ == "__main__":
    unittest.main(verbosity=1)
