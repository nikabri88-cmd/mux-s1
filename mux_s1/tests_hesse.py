"""Tests of the reader of the Hesse et al. release: python3 -m mux_s1.tests_hesse

They run without the release, on files built here in the MATLAB 7.3 layout seen in the release
(session struct, cell of paradigm structs in #refs#, uint16 text, transposed axes). Every expected
mask and count is worked out by hand from the authors' rules, so a wrong port fails the test.

Planted spikes: trial t (1-based), unit j gets one spike at 300 + 10*((t + j) % 70) ms after the
onset, plus one at 150 ms and one at 1200 ms that lie outside the window. Its counts are a single
1 in bin (t + j) % 70, so the counts of a row tell which file trial it came from.
"""
import os
import shutil
import tempfile
import unittest

import numpy as np

from . import hesse_reader as H

KIND_POS = dict(prime_red=0, unamb_red_supp=1, bino_red_supp=2, trans_red_supp=3,
                prime_cyan=4, unamb_cyan_supp=5, bino_cyan_supp=6, trans_cyan_supp=7)
SUFFIX = ["red_suppressed_red_only", "red_suppressed_cyan_only", "red_suppressed_binocular",
          "red_suppressed_transparent", "cyan_suppressed_cyan_only", "cyan_suppressed_red_only",
          "cyan_suppressed_binocular", "cyan_suppressed_transparent"]
PSR = np.arange(-200, 1501)


def stim_names(pairs):
    out = []
    for k, (red, cyan) in enumerate(pairs, 1):
        out += [f"image_pair_{k}_red_{red}_cyan_{cyan}_{s}.tif" for s in SUFFIX]
    return out + ["gray.tif"]


def planted_trial(t, n_units, n_samples=None, onset=None):
    """(time, unit) 0/1 raster of trial t on the PSR axis."""
    r = np.zeros((PSR.size, n_units), np.uint8)
    for j in range(n_units):
        for tau in (300 + 10 * ((t + j) % 70), 150, 1200):
            r[tau + 200, j] = 1
    return r


def expected_counts(t, units):
    c = np.zeros((70, len(units)), np.uint8)
    for k, j in enumerate(units):
        c[(t + j) % 70, k] = 1
    return c


# ---------------------------------------------------------------- MATLAB 7.3 writer for tests
class _W:
    def __init__(self, f):
        self.f, self.n = f, 0
        f.create_group("#refs#")

    def _new(self):
        self.n += 1
        return f"#refs#/r{self.n}"

    def _cls(self, d, c):
        d.attrs["MATLAB_class"] = np.bytes_(c)
        return d

    def char(self, g, name, s):
        return self._cls(g.create_dataset(name, data=np.array([ord(x) for x in s] or [0], np.uint16).reshape(-1, 1)), "char")

    def num(self, g, name, a, cls="double"):
        a = np.atleast_2d(np.asarray(a))
        return self._cls(g.create_dataset(name, data=a.T), cls)

    def empty(self, g, name):
        d = self._cls(g.create_dataset(name, data=np.array([0, 0], np.uint64)), "double")
        d.attrs["MATLAB_empty"] = np.uint8(1)
        return d

    def cell(self, g, name, items):
        import h5py
        refs = np.empty((len(items), 1), dtype=h5py.ref_dtype)
        for i, it in enumerate(items):
            refs[i, 0] = it.ref
        return self._cls(g.create_dataset(name, data=refs), "cell")

    def cellstr(self, g, name, strings):
        return self.cell(g, name, [self.char(self.f, self._new(), s) for s in strings])

    def cellnum(self, g, name, values):
        items = []
        for v in values:
            if v is None:
                items.append(self.empty(self.f, self._new()))                       # [] in MATLAB
            else:
                items.append(self.num(self.f, self._new(), [list(np.atleast_1d(v))]))
        return self.cell(g, name, items)

    def struct(self, path=None):
        g = self.f.create_group(path or self._new())
        g.attrs["MATLAB_class"] = np.bytes_("struct")
        return g


def make_fake(path, paradigm_name, pairs, sequence, prev_valid=None, comments=(), unit_probes=(1, 1, 2, 2, 3, 3),
              use_raster=True):
    """sequence: list of (pair index 0-based or None, kind). Returns the stimulus index list."""
    import h5py
    names = stim_names(pairs)
    siv = []
    for p, kind in sequence:
        siv.append(len(names) if kind == "gray" else p * 8 + KIND_POS[kind] + 1)    # 1-based
    n_tr, n_u = len(siv), len(unit_probes)
    prev_valid = np.ones(n_tr, np.uint8) if prev_valid is None else np.asarray(prev_valid, np.uint8)
    with h5py.File(path, "w", userblock_size=512) as f:
        w = _W(f)
        s = w.struct("session")
        w.num(s, "unit_probes_index", [list(unit_probes)])
        fix = w.struct(); w.char(fix, "paradigm_name", "fixation")
        par = w.struct()
        w.char(par, "paradigm_name", paradigm_name)
        w.cellstr(par, "stimulus_list", names)
        w.num(par, "stimulus_index_valid", [siv])
        w.num(par, "b_previous_trial_valid", [prev_valid], "logical")
        w.num(par, "on_durations", [[1000.0] * n_tr])
        if comments:
            w.cellstr(par, "comments", [c for c, _ in comments])
            w.cellnum(par, "comments_succeeding_trial_index", [t for _, t in comments])
        else:
            w.empty(par, "comments")
        dbg = par.create_group("debug"); dbg.attrs["MATLAB_class"] = np.bytes_("struct")
        w.num(dbg, "ab_valid_trials", [np.ones(n_tr, np.uint8)], "logical")
        if use_raster:
            w.num(par, "peri_stimulus_range", [PSR])
            mat = np.stack([planted_trial(t + 1, n_u) for t in range(n_tr)])        # (trial, time, unit)
            w._cls(par.create_dataset("raster_all_units", data=np.transpose(mat, (2, 1, 0))), "logical")
        else:
            gap = 3000
            on = np.array([2000 + gap * t for t in range(n_tr)], float)              # 1-based ms
            L = int(on[-1] + 4000)
            tc = np.zeros((L, n_u), np.uint8)
            for t in range(n_tr):
                r = planted_trial(t + 1, n_u)
                i0 = int(on[t]) - 1 - 200                                            # psr -200 at index 0
                tc[i0:i0 + PSR.size] = np.maximum(tc[i0:i0 + PSR.size], r)
            w._cls(par.create_dataset("timecourse_all_units", data=tc.T), "logical")
            w.num(par, "timecourse_on_times", [on])
        empty_slot = w.empty(f, w._new())
        w.cell(s, "paradigms_data", [empty_slot, fix, par])
    with open(path, "r+b") as fh:
        fh.write(b"MATLAB 7.3 MAT-file, Platform: test")
    return siv


BASE_PAIRS = [("liang_monkey_face_001", "OB45_house"), ("liang_monkey_face_002", "OB46_car")]
# trial (1-based) : what it is, and what the authors' rules make of it
BASE_SEQ = [(0, "prime_red"),        # 1  priming of face_001 (red)
            (0, "bino_red_supp"),    # 2  valid binocular, house dominant -> bino_b of pair 1
            (0, "prime_cyan"),       # 3  priming of house (cyan)
            (0, "bino_cyan_supp"),   # 4  valid binocular, face_001 dominant -> bino_a of pair 1
            (None, "gray"),          # 5  gray
            (0, "bino_cyan_supp"),   # 6  previous trial is gray -> invalid
            (1, "prime_red"),        # 7  priming of face_002
            (1, "bino_red_supp"),    # 8  b_previous_trial_valid false -> invalid
            (1, "prime_cyan"),       # 9  priming of car
            (1, "bino_cyan_supp"),   # 10 valid -> bino_a of pair 2
            (1, "prime_red"),        # 11 priming of face_002
            (1, "bino_red_supp"),    # 12 valid -> bino_b of pair 2
            (0, "prime_red"),        # 13 priming of face_001
            (0, "unamb_red_supp"),   # 14 valid unambiguous, house dominant -> unamb_b of pair 1
            (0, "prime_cyan"),       # 15 priming of house
            (0, "bino_cyan_supp")]   # 16 valid -> bino_a of pair 1
PREV_VALID = [1] * 16
PREV_VALID[7] = 0                    # trial 8


class TestNames(unittest.TestCase):
    def test_the_release_names_parse_as_the_authors_parse_them(self):
        n = "image_pair_1_red_liang_monkey_face_001_cyan_OB45_house_red_suppressed_red_only.tif"
        m, before, after = H.parse_string(n, H.CONTENTS)
        self.assertEqual(H.CONTENTS[m], "_red_only")
        sp = H.parse_stimuli([n,
                              n.replace("red_suppressed_red_only", "red_suppressed_cyan_only"),
                              n.replace("red_suppressed_red_only", "red_suppressed_binocular"),
                              n.replace("red_suppressed_red_only", "cyan_suppressed_cyan_only"),
                              n.replace("red_suppressed_red_only", "cyan_suppressed_transparent"),
                              "gray.tif"])
        self.assertEqual(list(sp["image_red"][:5]), ["liang_monkey_face_001"] * 5)
        self.assertEqual(list(sp["image_cyan"][:5]), ["OB45_house"] * 5)
        self.assertEqual(sp["is_priming"].tolist(), [True, False, False, True, False, False])
        self.assertEqual(sp["is_unambiguous"].tolist(), [False, True, False, False, False, False])
        self.assertEqual(sp["is_binocular"].tolist(), [False, False, True, False, False, False])
        self.assertEqual(sp["is_transparent"].tolist(), [False, False, False, False, True, False])
        self.assertTrue(sp["is_gray"][-1])

    def test_parse_string_takes_the_first_occurrence_and_the_last_matching_pattern(self):
        self.assertEqual(H.parse_string("a_x_b_x_c", "_x_"), (0, "a", "b_x_c"))
        m, _, _ = H.parse_string("ab", ["a", "b"])
        self.assertEqual(m, 1)


class _Tmp(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def fake(self, name="session_250504_Mowgli", paradigm="flash_suppression_optimized", **kw):
        p = os.path.join(self.dir, name + ".mat")
        kw.setdefault("prev_valid", PREV_VALID)
        make_fake(p, paradigm, kw.pop("pairs", BASE_PAIRS), kw.pop("sequence", BASE_SEQ), **kw)
        return p


class TestClassification(_Tmp):
    def test_masks_follow_the_authors_rules_trial_by_trial(self):
        names = stim_names(BASE_PAIRS)
        siv = [len(names) if k == "gray" else p * 8 + KIND_POS[k] + 1 for p, k in BASE_SEQ]
        cl = H.classify_trials("flash_suppression_optimized", names, siv, PREV_VALID, [], [],
                               np.ones(16), np.ones(16, bool))
        rows = lambda m: (np.flatnonzero(m) + 1).tolist()
        self.assertEqual(rows(cl["is_priming"]), [1, 3, 7, 9, 11, 13, 15])
        self.assertEqual(rows(cl["is_binocular"]), [2, 4, 10, 12, 16])     # 6 after gray, 8 prev invalid
        self.assertEqual(rows(cl["is_unambiguous"]), [14])
        self.assertEqual(rows(cl["is_gray"]), [5])
        self.assertEqual(cl["n_binocular_all"], 7)
        pl = cl["priming_list"]
        self.assertEqual(pl, sorted(["liang_monkey_face_001", "OB45_house", "liang_monkey_face_002", "OB46_car"]))
        name = lambda i: pl[i]
        self.assertEqual(name(cl["dom_idx"][0]), "liang_monkey_face_001")    # priming shows the red face
        self.assertEqual(name(cl["dom_idx"][1]), "OB45_house")               # flashed cyan dominates
        self.assertEqual(name(cl["dom_idx"][3]), "liang_monkey_face_001")
        self.assertEqual(name(cl["sup_idx"][3]), "OB45_house")
        self.assertEqual(cl["dom_idx"][4], -1)                              # gray

    def test_the_first_trial_wraps_around_as_circshift_does(self):
        names = stim_names(BASE_PAIRS)
        # trial 1 binocular, last trial priming: circshift makes trial 1 'preceded by priming'
        seq = [(0, "bino_cyan_supp"), (0, "prime_red"), (0, "prime_cyan")]
        siv = [p * 8 + KIND_POS[k] + 1 for p, k in seq]
        cl = H.classify_trials("flash_suppression_optimized", names, siv, [1, 1, 1], [], [], np.ones(3), np.ones(3, bool))
        self.assertTrue(cl["is_binocular"][0])

    def test_throw_away_rule_drops_the_marked_trials(self):
        keep = H.throw_away_mask(16, ["x", "Throw away last 1 blocks"], [3, 12], np.ones(16))
        # ab(9:12) has 4 valid trials, so keep((12-4):12) = trials 8..12 are dropped
        self.assertEqual((np.flatnonzero(~keep) + 1).tolist(), [8, 9, 10, 11, 12])

    def test_the_many_stimuli_exception_keeps_binocular_trials_when_a_stimulus_would_vanish(self):
        names = stim_names(BASE_PAIRS)
        seq = [(0, "prime_red"), (0, "bino_red_supp"), (None, "gray"), (1, "bino_red_supp")]
        siv = [len(names) if k == "gray" else p * 8 + KIND_POS[k] + 1 for p, k in seq]
        for pname, expect in (("flash_suppression_optimized", [2]),
                              ("flash_suppression_many_stimuli", [2, 4])):
            cl = H.classify_trials(pname, names, siv, [1, 1, 1, 1], [], [], np.ones(4), np.ones(4, bool))
            self.assertEqual((np.flatnonzero(cl["is_binocular"]) + 1).tolist(), expect, pname)


class TestRead(_Tmp):
    def check_row(self, rows, trial_numbers, units):
        self.assertEqual(rows.shape[0], len(trial_numbers))
        for r, t in zip(rows, trial_numbers):
            self.assertTrue(np.array_equal(r, expected_counts(t, units)), f"trial {t}")

    def test_a_session_reads_into_the_canonical_form(self):
        s = H.read_session(self.fake())
        units = [0, 1, 2, 3]                            # probes 1 and 2 of (1,1,2,2,3,3)
        self.assertEqual(s.meta["n_units_used"], 4)
        self.assertEqual(s.meta["n_units_by_probe"], {1: 2, 2: 2, 3: 2})
        self.assertEqual(len(s.pairs), 2)
        pl = s.meta["priming_list"]
        p1 = [p for p in s.pairs if pl[p.face_a] == "liang_monkey_face_001"][0]
        p2 = [p for p in s.pairs if pl[p.face_a] == "liang_monkey_face_002"][0]
        self.assertEqual(pl[p1.face_b], "OB45_house")
        self.check_row(p1.bino_a, [4, 16], units)
        self.check_row(p1.bino_b, [2], units)
        self.check_row(p1.prime_a, [1, 13], units)
        self.check_row(p1.prime_b, [3, 15], units)
        self.check_row(p1.unamb_b, [14], units)
        self.assertIsNone(p1.unamb_a)
        self.check_row(p2.bino_a, [10], units)
        self.check_row(p2.bino_b, [12], units)
        self.check_row(p2.prime_a, [7, 11], units)
        self.check_row(p2.prime_b, [9], units)
        self.assertEqual(s.bin_ms, 10.0)
        self.assertEqual(p1.bino_a.dtype, np.uint8)

    def test_the_rebuilt_raster_path_gives_the_same_session(self):
        a = H.read_session(self.fake())
        b = H.read_session(self.fake(name="session_250504_Mowgli", use_raster=False))
        self.assertEqual(a.meta["raster_source"], "raster_all_units")
        self.assertEqual(b.meta["raster_source"], "timecourse_all_units")
        for x, y in zip(a.pairs, b.pairs):
            for f in ("bino_a", "bino_b", "prime_a", "prime_b"):
                self.assertTrue(np.array_equal(getattr(x, f), getattr(y, f)), f)

    def test_throw_away_comment_removes_its_binocular_trials(self):
        s = H.read_session(self.fake(comments=[("Throw away last 1 blocks", 12)]))
        pl = s.meta["priming_list"]
        names = {pl[p.face_a] for p in s.pairs}
        # pair 2 lost both binocular trials (10 and 12), so it cannot enter the analysis
        self.assertEqual(names, {"liang_monkey_face_001"})
        self.assertEqual(s.meta["skipped"][0]["n_bino_a"], 0)
        self.assertEqual(s.meta["n_thrown_away"], 5)

    def test_injection_exclusion_is_off_by_default_and_drops_later_trials_when_on(self):
        com = [("04-May-2025 20:10:48:790 Injected 3 mg psilocybin", 13)]
        on = H.read_session(self.fake(comments=com), exclude_after_injection=True)
        off = H.read_session(self.fake(comments=com))
        self.assertEqual(off.meta["injection_comments"], [com[0][0]])
        self.assertEqual(off.meta["excluded_after_injection"], 0)
        self.assertEqual(on.meta["excluded_after_injection"], 4)          # trials 13..16
        pl = on.meta["priming_list"]
        p1 = [p for p in on.pairs if pl[p.face_a] == "liang_monkey_face_001"][0]
        self.check_row(p1.bino_a, [4], [0, 1, 2, 3])
        self.check_row(p1.prime_a, [1], [0, 1, 2, 3])

    def test_the_secondary_priming_window_changes_only_the_priming_counts(self):
        a = H.read_session(self.fake())
        b = H.read_session(self.fake(), prime_window_ms=(100, 1000))
        self.assertEqual(b.pairs[0].prime_a.shape[1], 90)
        self.assertTrue(np.array_equal(a.pairs[0].bino_a, b.pairs[0].bino_a))

    def test_repeat_only_keeps_pairs_below_sixty_in_fsm_sessions(self):
        pairs = [(f"face_{i:03d}", f"obj_{i:03d}") for i in range(31)]     # 62 priming images
        seq = []
        for i in range(31):
            seq += [(i, "prime_red"), (i, "bino_red_supp"), (i, "prime_cyan"), (i, "bino_cyan_supp")]
        kw = dict(pairs=pairs, sequence=seq, prev_valid=[1] * len(seq), unit_probes=(1, 2))
        p = self.fake(name="session_251212_Alfie", paradigm="flash_suppression_many_stimuli", **kw)
        self.assertEqual(len(H.read_session(p).pairs), 31)
        rep = H.read_session(p, repeat_only=True)
        self.assertEqual(len(rep.pairs), 29)                                 # obj_029, obj_030 have index 60, 61
        self.assertTrue(all(q.face_a < 60 and q.face_b < 60 for q in rep.pairs))

    def test_dandi_sessions_reads_files_and_caches_identically(self):
        from . import adapters as A
        self.fake()
        H.DESIGNS["TEST"] = ["session_250504_Mowgli"]
        try:
            direct = A.dandi_sessions(self.dir, design="TEST")[0]
            cache_dir = os.path.join(self.dir, "cache")
            A.save_session(direct, os.path.join(cache_dir, "session_250504_Mowgli__primary.npz"))
            cached = A.dandi_sessions(cache_dir, design="TEST")[0]
        finally:
            del H.DESIGNS["TEST"]
        self.assertEqual(cached.meta["n_units_used"], 4)
        self.assertEqual(len(cached.pairs), len(direct.pairs))
        for x, y in zip(direct.pairs, cached.pairs):
            for f in ("bino_a", "bino_b", "prime_a", "prime_b", "unamb_a", "unamb_b"):
                if getattr(x, f) is None:
                    self.assertIsNone(getattr(y, f), f)          # pair 2 has no unambiguous trials
                    continue
                self.assertTrue(np.array_equal(getattr(x, f), getattr(y, f)), f)
                self.assertEqual(getattr(y, f).dtype, np.uint8)
            self.assertEqual((x.face_a, x.face_b), (y.face_a, y.face_b))
        with self.assertRaises(ValueError):
            H.DESIGNS["TEST"] = ["session_250504_Mowgli"]
            try:
                A.dandi_sessions(cache_dir, design="TEST", bin_ms=20.0)    # cache made at 10 ms
            finally:
                del H.DESIGNS["TEST"]

    def test_an_empty_comment_index_drops_nothing_and_a_vector_reads_its_first_element(self):
        # session 250518_Mowgli of the release holds such entries
        empty = H.read_session(self.fake(comments=[("Throw away last 1 blocks", None)]))
        self.assertEqual(empty.meta["n_thrown_away"], 0)
        self.assertEqual(empty.meta["comment_index_note"], {"empty": 1, "multi": 0})
        vec = H.read_session(self.fake(comments=[("Throw away last 1 blocks", [12, 14])]))
        scal = H.read_session(self.fake(comments=[("Throw away last 1 blocks", 12)]))
        self.assertEqual(vec.meta["n_thrown_away"], scal.meta["n_thrown_away"])
        # which trials go matters, not only how many: from 12 the unambiguous trial 14 stays,
        # from 14 it would be dropped
        self.assertEqual(len(vec.pairs), len(scal.pairs))
        for x, y in zip(vec.pairs, scal.pairs):
            for f in ("bino_a", "bino_b", "prime_a", "prime_b", "unamb_a", "unamb_b"):
                a, b = getattr(x, f), getattr(y, f)
                self.assertTrue((a is None and b is None) or np.array_equal(a, b), f)
        self.assertIsNotNone(scal.pairs[0].unamb_b)
        self.assertEqual(vec.meta["comment_index_note"], {"empty": 0, "multi": 1})
        inj = H.read_session(self.fake(comments=[("Injected 3 mg psilocybin", None)]), exclude_after_injection=True)
        self.assertEqual(inj.meta["excluded_after_injection"], 0)
        self.assertEqual(len(inj.meta["injection_comments"]), 1)

    def test_comments_as_one_string_and_indices_as_a_plain_vector_are_read(self):
        import h5py
        p = self.fake(comments=[("Throw away last 1 blocks", 12)])
        with h5py.File(p, "r+") as f:
            par = [f[r] for r in f["session"]["paradigms_data"][()].reshape(-1)][2]
            del par["comments"], par["comments_succeeding_trial_index"]
            w = _W.__new__(_W); w.f, w.n = f, 10_000
            w.char(par, "comments", "Throw away last 1 blocks")
            w.num(par, "comments_succeeding_trial_index", [[12.0]])
        s = H.read_session(p)
        self.assertEqual(s.meta["comments"], ["Throw away last 1 blocks"])
        self.assertEqual(s.meta["n_thrown_away"], 5)

    def test_one_read_gives_the_same_variants_as_separate_reads(self):
        p = self.fake(comments=[("Injected 3 mg psilocybin", 13)])
        many = H.read_variants(p)
        self.assertEqual(sorted(many), sorted(H.VARIANTS))
        for key, kw in H.VARIANTS.items():
            one = H.read_session(p, **kw)
            self.assertEqual(len(one.pairs), len(many[key].pairs), key)
            for x, y in zip(one.pairs, many[key].pairs):
                for f in ("bino_a", "bino_b", "prime_a", "prime_b"):
                    self.assertTrue(np.array_equal(getattr(x, f), getattr(y, f)), (key, f))

    def test_a_session_outside_rec_it_is_refused(self):
        with self.assertRaises(KeyError):
            H.read_session(self.fake(name="session_999999_Nobody"))


if __name__ == "__main__":
    unittest.main(verbosity=1)
