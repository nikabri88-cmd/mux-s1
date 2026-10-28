"""Tests of the face-feature step: python3 -m mux_s1.tests_faces

Feature maps are written here as MATLAB containers.Map with mat-io, the same package that reads the
release's maps, so the tests need no file from the release.
"""
import os
import shutil
import tempfile
import unittest

import numpy as np

from . import adapters as A
from . import faces as FC
from . import hesse_reader as H
from . import tests_hesse as TH


def write_map(path, mapping, version="v7"):
    from matio import load_from_mat, save_to_mat
    from matio.utils.matclass import MatlabContainerMap
    save_to_mat(path, {"feat_map": MatlabContainerMap({k: np.asarray(v, float).reshape(1, -1)
                                                       for k, v in mapping.items()})}, version=version)
    return path


class TestFaces(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_the_set_follows_the_xml_name_as_in_the_authors_code(self):
        self.assertEqual(FC.set_for_xml(r"\\kofiko\stimulusset\FlashSuppression_many_faces_Liang_2000_many_pairs\x.xml"),
                         "liang_2000_many_pairs")
        self.assertEqual(FC.set_for_xml(r"\\kofiko\stimulusset\FlashSuppression_many_faces_Liang_2500_many_pairs\x.xml"),
                         "liang_2500_many_pairs")
        self.assertEqual(FC.set_for_xml("anything else"), "liang_2500_many_pairs")

    def test_a_containers_map_reads_into_vectors_with_backslashes_replaced(self):
        p = write_map(os.path.join(self.dir, "m.mat"), {"a\\b": [1, 2, 3], "c": [4, 5, 6]})
        m = FC.load_feature_map(p)
        self.assertEqual(sorted(m), ["a_b", "c"])
        self.assertTrue(np.array_equal(m["c"], [4.0, 5.0, 6.0]))

    def test_the_converted_file_gives_the_same_vectors_as_the_map(self):
        for fs in FC.SETS:
            os.makedirs(os.path.join(self.dir, fs))
        write_map(os.path.join(self.dir, FC.SETS[0], FC.MAP_FILE), {"a\\b": [1, 2, 3], "c": [4, 5, 6]})
        write_map(os.path.join(self.dir, FC.SETS[1], FC.MAP_FILE), {"c": [7, 8, 9]})
        npz = FC.convert_maps(self.dir, os.path.join(self.dir, "fm.npz"))
        for fs in FC.SETS:
            direct = FC.load_feature_map(os.path.join(self.dir, fs, FC.MAP_FILE))
            conv = FC.load_feature_npz(fs, npz)
            self.assertEqual(sorted(direct), sorted(conv))
            for k in direct:
                self.assertTrue(np.array_equal(direct[k], conv[k]))
        self.assertEqual(sorted(FC.source_hashes(npz)), sorted(FC.SETS))

    def test_the_shipped_file_matches_the_release_maps(self):
        src = FC.source_hashes()
        self.assertTrue(src["liang_2000_many_pairs"].startswith("1a95e705ae677114"))
        self.assertTrue(src["liang_2500_many_pairs"].startswith("b97aaec284bf3c2a"))
        self.assertEqual(len(FC.load_feature_npz("liang_2000_many_pairs")), 2000)
        self.assertEqual(len(FC.load_feature_npz("liang_2500_many_pairs")), 1500)

    def test_rows_follow_the_priming_list_and_gaps_are_nan(self):
        NAN = FC.NAN_STIMULI[0]
        fmap = {"f1": np.array([1.0, 1.0]), "f2": np.array([2.0, 2.0]), NAN: np.array([9.0, 9.0])}
        F, missing = FC.faces_matrix(["f2", "zz", "f1", NAN], fmap)
        self.assertTrue(np.array_equal(F[0], [2, 2])); self.assertTrue(np.array_equal(F[2], [1, 1]))
        self.assertTrue(np.isnan(F[1]).all()); self.assertTrue(np.isnan(F[3]).all())
        self.assertEqual(missing, ["zz", NAN])

    def _session_with_faces(self, mapping):
        p = os.path.join(self.dir, "session_250504_Mowgli.mat")
        TH.make_fake(p, "flash_suppression_optimized", TH.BASE_PAIRS, TH.BASE_SEQ, prev_valid=TH.PREV_VALID)
        s = H.read_session(p)
        F, missing = FC.faces_matrix(s.meta["priming_list"], mapping)
        npz = FC.save_faces(os.path.join(self.dir, "session_250504_Mowgli__faces.npz"), F,
                            "liang_2500_many_pairs", "x.xml", missing, s.meta["priming_list"])
        return s, npz

    def test_faces_attach_when_every_face_in_a_pair_has_features(self):
        names = ["liang_monkey_face_001", "OB45_house", "liang_monkey_face_002", "OB46_car"]
        s, npz = self._session_with_faces({n: np.arange(3.0) + i for i, n in enumerate(names)})
        FC.attach_faces(s, npz)
        pl = s.meta["priming_list"]
        for p in s.pairs:
            self.assertTrue(np.array_equal(s.faces[p.face_a], np.arange(3.0) + names.index(pl[p.face_a])))
        self.assertEqual(s.meta["faces"]["n_features"], 3)

    def test_a_face_in_a_pair_without_features_is_refused(self):
        s, npz = self._session_with_faces({"liang_monkey_face_001": np.zeros(3)})
        with self.assertRaises(ValueError):
            FC.attach_faces(s, npz)

    def test_a_faces_file_of_another_session_is_refused(self):
        names = ["liang_monkey_face_001", "OB45_house", "liang_monkey_face_002", "OB46_car"]
        s, npz = self._session_with_faces({n: np.zeros(3) for n in names})
        s.meta["priming_list"] = list(reversed(s.meta["priming_list"]))
        with self.assertRaises(ValueError):
            FC.attach_faces(s, npz)

    def test_dandi_sessions_leaves_a_session_without_faces_when_features_are_missing(self):
        s, npz = self._session_with_faces({"liang_monkey_face_001": np.zeros(3)})
        A.save_session(s, os.path.join(self.dir, "session_250504_Mowgli__primary.npz"))
        H.DESIGNS["TEST"] = ["session_250504_Mowgli"]
        try:
            got = A.dandi_sessions(self.dir, design="TEST")[0]
        finally:
            del H.DESIGNS["TEST"]
        self.assertIsNone(got.faces)
        self.assertIn("no features", got.meta["faces_error"])

    def test_dandi_sessions_attaches_faces_from_the_cache(self):
        names = ["liang_monkey_face_001", "OB45_house", "liang_monkey_face_002", "OB46_car"]
        s, npz = self._session_with_faces({n: np.ones(3) * i for i, n in enumerate(names)})
        A.save_session(s, os.path.join(self.dir, "session_250504_Mowgli__primary.npz"))
        H.DESIGNS["TEST"] = ["session_250504_Mowgli"]
        try:
            got = A.dandi_sessions(self.dir, design="TEST")[0]
        finally:
            del H.DESIGNS["TEST"]
        self.assertIsNotNone(got.faces)
        self.assertEqual(got.faces.shape, (4, 3))


if __name__ == "__main__":
    unittest.main(verbosity=1)
