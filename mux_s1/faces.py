"""Face features for the gain block (B6, B7, B9, B10), from the stimulus feature maps of the release.

Source: data.zip -> data/stimuli/{liang_2000_many_pairs, liang_2500_many_pairs}/
feat_map_priming_reconstructed_0.00.mat. Each holds a MATLAB containers.Map from stimulus name to a
feature vector (120 numbers), read with the package mat-io. These are the descriptions from which
the authors' face model draws the stimuli: inputs of the analysis, none of its results.

Rules, ported from matlab/helpers/flash_suppression_axes_get_stimuli.m:
  - a session whose paradigm xml_filename contains 'FlashSuppression_many_faces_Liang_2000_many_pairs'
    uses liang_2000_many_pairs, every other session uses liang_2500_many_pairs;
  - keys have '\\' replaced by '_';
  - five synthetic stimuli have NaN features and are excluded by name.
The two sets share the names monkey_face_120d_0001..1500 with different vectors, so the set has to
follow the xml_filename; it cannot be guessed from the names.

Rows of `Session.faces` follow the session's priming_list, which is what `Pair.face_a/face_b`
index (hesse_reader). Rows of stimuli without features are NaN; the frozen gain block reads only
the rows of faces that appear in pairs, and `attach_faces` refuses a session where one of those
rows is NaN, so that a gap cannot pass into the regression silently.

This module sits beside hesse_reader.py and leaves it unchanged, so the counts cached in step 2 stay
valid; it reads the xml_filename with the reader's own helpers.
"""
import json
import os

import numpy as np

SET_2000_TAG = "FlashSuppression_many_faces_Liang_2000_many_pairs"
SETS = ("liang_2000_many_pairs", "liang_2500_many_pairs")
MAP_FILE = "feat_map_priming_reconstructed_0.00.mat"
NAN_STIMULI = ("monkey_face_test_syn_120d_mean.tif", "monkey_face_test_syn_120d_training_mark-1_texture0.tif",
               "monkey_face_test_syn_120d_training_mark0_texture-1.tif",
               "monkey_face_test_syn_120d_training_mark0_texture1.tif",
               "monkey_face_test_syn_120d_training_mark1_texture0.tif")


def set_for_xml(xml_filename):
    return SETS[0] if SET_2000_TAG in xml_filename else SETS[1]


def read_xml(path, name):
    """xml_filename of the session's rec_it paradigm, read with the reader's helpers."""
    import h5py
    from . import hesse_reader as HR
    abbr, _ = HR.REC_IT[name]
    with h5py.File(path, "r") as h:
        par = HR._find_paradigm(h, HR.PARADIGMS[abbr])
        return HR._text(par["xml_filename"])


def load_feature_map(path):
    """containers.Map -> {stimulus name: 1-d float vector}."""
    from matio import load_from_mat
    fm = load_from_mat(path)["feat_map"]
    return {str(k).replace("\\", "_"): np.asarray(fm[k], float).reshape(-1) for k in fm.keys()}


FEATURE_NPZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feature_maps.npz")


def convert_maps(stimuli_dir, out_npz=FEATURE_NPZ):
    """The authors' containers.Map files -> one plain .npz (names, features, sha256 of each source).

    Made once, where mat-io is installed (it needs numpy >= 2.2.2, newer than Colab's), and shipped
    with the code; the Colab step checks that the source files in Drive have the recorded hashes.
    """
    import hashlib
    out, src = {}, {}
    for fs in SETS:
        p = os.path.join(stimuli_dir, fs, MAP_FILE)
        src[fs] = hashlib.sha256(open(p, "rb").read()).hexdigest()
        m = load_feature_map(p)
        names = sorted(m)
        out[f"{fs}__names"] = np.array(names)
        out[f"{fs}__features"] = np.stack([m[n] for n in names]).astype(np.float64)
    out["__sources__"] = np.array(json.dumps(src))
    np.savez_compressed(out_npz, **out)
    return out_npz


def source_hashes(npz_path=FEATURE_NPZ):
    return json.loads(str(np.load(npz_path, allow_pickle=False)["__sources__"]))


def load_feature_npz(feature_set, npz_path=FEATURE_NPZ):
    """{stimulus name: vector} of one set, from the converted file."""
    z = np.load(npz_path, allow_pickle=False)
    names = [str(x) for x in z[f"{feature_set}__names"]]
    F = z[f"{feature_set}__features"]
    return {n: F[i] for i, n in enumerate(names)}


def faces_matrix(priming_list, fmap):
    """(len(priming_list), n_features) in priming_list order; NaN rows where no features exist."""
    n_feat = len(next(iter(fmap.values())))
    F = np.full((len(priming_list), n_feat), np.nan)
    missing = []
    for i, name in enumerate(priming_list):
        v = fmap.get(name)
        if v is None or name in NAN_STIMULI or not np.all(np.isfinite(v)):
            missing.append(name)
            continue
        F[i] = v
    return F, missing


def used_faces(session):
    return sorted({f for p in session.pairs for f in (p.face_a, p.face_b)})


def save_faces(out_path, F, feature_set, xml_filename, missing, priming_list):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez_compressed(out_path, faces=F, __meta__=json.dumps(dict(
        feature_set=feature_set, xml_filename=xml_filename, missing=missing,
        n_priming=len(priming_list), priming_list=list(priming_list))))
    return out_path


def attach_faces(session, faces_npz):
    """Session.faces from a saved faces file, after checking it belongs to this session."""
    z = np.load(faces_npz, allow_pickle=False)
    meta = json.loads(str(z["__meta__"]))
    F = z["faces"]
    plist = session.meta.get("priming_list")
    if plist is not None and list(plist) != list(meta["priming_list"]):
        raise ValueError(f"{session.name}: the faces file was made for another priming list")
    used = used_faces(session)
    bad = [i for i in used if not np.all(np.isfinite(F[i]))]
    if bad:
        raise ValueError(f"{session.name}: {len(bad)} faces in pairs have no features "
                         f"(first: {[meta['priming_list'][i] for i in bad[:3]]})")
    session.faces = F
    session.meta = dict(session.meta, faces=dict(feature_set=meta["feature_set"],
                                                 xml_filename=meta["xml_filename"],
                                                 n_features=int(F.shape[1]), n_used=len(used),
                                                 n_missing_unused=len(meta["missing"])))
    return session
