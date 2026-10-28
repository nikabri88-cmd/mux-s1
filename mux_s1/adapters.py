"""Readers that put data into the canonical form of `mux_s1.realdata`.

This file sits OUTSIDE the frozen set of version 1.1 on purpose: the layout of the DANDI files of
Hesse et al. becomes visible only on release day, while the statistics in `realdata.py` do not
depend on it. What an adapter must return is fixed here and in the pre-registration; the body of
`dandi_sessions` is written when the files are in hand, and the change is entered in the deviation
log together with the hash of this file at that moment.

`synthetic_sessions` builds the same canonical form from the simulator of version 1.0 and is what
the verification code is tested on before the release.

Release note, 28 Oct 2026 (supersedes the note below where they differ)
---------------------------------------------------------------------
The release turned out to store trial types only implicitly: the authors derive them at load time
from stimulus file names, previous-trial validity and comments. Filling a field map was not
enough, so the reader is `mux_s1/hesse_reader.py`, a port of the authors' own rules cited by file.
`dandi_sessions` below calls it. The generic MAT_FIELDS machinery stays for its own tests.

Release note, 27 Oct 2026
-------------------------
The release came out on Zenodo instead of DANDI, as 19 `session_*.mat` files of 0.2-2.7 GB each
holding the per-trial spike data, plus a `python_summaries` directory of cross-session summaries.
`dandi_sessions` keeps its name, so that the pre-registered contract and the call in
`verify_real.py` stay as published; only its body is new, and it reads the Zenodo files.

The body is split in two on purpose:

  * the binning and assembly machinery below is general and is covered by unit tests that run
    without the release (`tests_adapter.py`), so its correctness does not rest on the layout;
  * `MAT_FIELDS` is the only place that names fields of the released files. It is filled in from
    the output of `describe_mat`, which walks one file and prints its tree.

What the machinery must not do is decide anything that a prediction reads. It bins counts and
sorts trials, nothing else: no smoothing, no normalization, no exclusion of trials beyond what
the pre-registration already names.
"""
import json
import os
import re

import numpy as np

from . import core as C
from .realdata import Pair, Session

CALIBRATION = dict(kappa=0.13915092914337512,
                   timescale={"M0_markov": 114.0, "M1_rhythm": 157.36363636363637,
                              "M1b_rhythm_k3": 158.88888888888889, "M2_cont_unimodal": 97.10526315789474,
                              "M3_cont_bimodal": 73.33333333333333, "M4_adaptation": 114.0,
                              "M4c_adaptation_tuning": 114.0, "M5_conscious_gain": 114.0,
                              "M6_heterogeneous": 114.0, "M7_ring": 113.21038251366122,
                              "C1_averaging": 114.0, "C2_uncoordinated": 114.0})


def synthetic_sessions(model, seed=0, n_sess=19, n_pairs=25, n_tr=15, N=300, t_prime_ms=1000.0,
                       kappa=None, timescale=None, with_faces=False):
    """Canonical sessions from the MUX-S1 1.0 simulator (the DYN design of part II).

    with_faces=True also returns the 50-dimensional feature vectors of the faces of each pair, so
    the gain block can run; the DYN design has too few faces for the reconstruction analyses.
    """
    spec = C.MODELS[model]
    kappa = CALIBRATION["kappa"] if kappa is None else kappa
    timescale = CALIBRATION["timescale"][model] if timescale is None else timescale
    rng = np.random.default_rng([int(seed), 11, 1])
    out = []
    for si in range(n_sess):
        pop = C.Population(rng, N, kappa)
        F = C.make_faces(rng, 2 * n_pairs)
        D = pop.delta(F)
        pairs = []
        for j in range(n_pairs):
            a, b = 2 * j, 2 * j + 1
            ring_u = None
            if spec["kind"] == "ring":
                u = rng.standard_normal(N)
                ring_u = u / np.linalg.norm(u) * np.linalg.norm(D[a] - D[b]) / 2
            prime_a = C.poisson(rng, C.priming_rates(spec, pop, D[a], n_tr))
            prime_b = C.poisson(rng, C.priming_rates(spec, pop, D[b], n_tr))
            lam0, _ = C.binocular_rates(rng, spec, pop, D[a], D[b], n_tr, timescale, t_prime_ms, ring_u)
            lam1, _ = C.binocular_rates(rng, spec, pop, D[b], D[a], n_tr, timescale, t_prime_ms, ring_u)
            pairs.append(Pair(bino_a=C.poisson(rng, lam0), bino_b=C.poisson(rng, lam1),
                              prime_a=prime_a, prime_b=prime_b, face_a=a, face_b=b))
        out.append(Session(pairs=pairs, bin_ms=C.DT * 1000.0, name=f"{model}_s{si}",
                           faces=(F if with_faces else None),
                           meta=dict(model=model, seed=seed, kappa=kappa, timescale=timescale)))
    return out


# ==============================================================================================
# the released files: naming
# ==============================================================================================
# Every entry is a path inside one session file, written with "/" between levels; a MATLAB struct
# field and an HDF5 group are both a level. None means "not filled in yet" and raises on use, so a
# half-filled map fails loudly instead of quietly dropping a condition.
#
# Fill these in from the output of `describe_mat(<one session file>)`. Nothing else in this module
# names a field of the release.
MAT_FIELDS = dict(
    # --- the spikes, in ONE of the two forms; leave the other None ---
    spike_times=None,        # ragged: spike times per trial and neuron, in SECONDS
    raster=None,             # dense: 0/1 or counts, axes named by `raster_order`
    raster_order="trial,sample,neuron",   # the axes AS MATLAB SEES THEM, in any order
    ms_per_sample=1.0,       # only for `raster`
    # --- per-trial columns, one value per trial ---
    onset=None,              # onset of the flashed face, SECONDS (samples when `raster` is used)
    trial_kind=None,         # condition code, matched against `kind_codes` below
    pair_id=None,            # which stimulus pair the trial belongs to
    face_flashed=None,       # the face that reaches awareness (the flashed one) in that trial
    face_other=None,         # the suppressed face; for priming trials it may be absent
    # --- optional ---
    neuron_ids=None,         # labels of the units, kept in Session.meta
    faces=None,              # (n_faces, n_feat) stimulus features, when the release carries them
    # --- how the condition codes read ---
    kind_codes=dict(binocular=None, priming=None, unambiguous=None),
)

_FIELDS_DOC = """MAT_FIELDS is not filled in. Run

    python -m mux_s1.adapters describe <one session_*.mat>

and copy the field paths into MAT_FIELDS. The analysis refuses to run on a half-filled map, so
that a missing condition cannot pass as an empty one."""


# ==============================================================================================
# reading a .mat of either vintage
# ==============================================================================================
def _is_v73(path):
    with open(path, "rb") as fh:
        return b"MATLAB 7.3" in fh.read(128)


class MatFile:
    """One session file, opened lazily, addressed by "a/b/c" paths.

    v7.3 files are HDF5 and are read with h5py, which keeps them on disk: a 2.7 GB session is
    never held whole in memory. Older files are read with scipy.io.loadmat.
    """

    def __init__(self, path):
        self.path = path
        self.v73 = _is_v73(path)
        if self.v73:
            import h5py
            self._h = h5py.File(path, "r")
        else:
            from scipy.io import loadmat
            self._h = loadmat(path, squeeze_me=True, struct_as_record=False)

    def close(self):
        if self.v73:
            self._h.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    # -- addressing -----------------------------------------------------------------------------
    def get(self, path, default=KeyError):
        if path is None:
            raise KeyError(_FIELDS_DOC)
        node = self._h
        for part in str(path).strip("/").split("/"):
            try:
                node = node[part] if not hasattr(node, "_fieldnames") else getattr(node, part)
            except (KeyError, AttributeError, IndexError):
                if default is KeyError:
                    raise KeyError(f"{path!r} is absent from {os.path.basename(self.path)}")
                return default
        return node

    def array(self, path, default=KeyError):
        """A numeric field as a 1-d or 2-d numpy array, with MATLAB's column order undone."""
        node = self.get(path, default)
        if node is default and default is not KeyError:
            return default
        a = np.asarray(node[()] if hasattr(node, "shape") and self.v73 else node)
        if self.v73 and a.ndim == 2:
            a = a.T                      # h5py hands back the transpose of the MATLAB shape
        return np.squeeze(a)

    def column(self, path, n=None, default=KeyError):
        """A per-trial column as a flat array of length n_trials."""
        a = self.array(path, default)
        if a is default and default is not KeyError:
            return default
        a = np.asarray(a).reshape(-1)
        if n is not None and a.size != n:
            raise ValueError(f"{path!r} has {a.size} entries, expected {n} trials")
        return a

    def ragged(self, path, i, j):
        """Spike times of trial i, neuron j, in seconds, from a cell array of either vintage."""
        node = self.get(path)
        if self.v73:
            import h5py
            ref = node[()]
            ref = ref.T if ref.ndim == 2 else ref
            cell = ref[i, j] if ref.ndim == 2 else ref[i]
            if isinstance(cell, h5py.Reference):
                cell = self._h[cell][()]
            v = np.asarray(cell).reshape(-1)
        else:
            cell = node[i] if np.ndim(node) == 1 else node[i][j]
            v = np.asarray(cell).reshape(-1)
        return v.astype(float)


def describe_mat(path, max_depth=4, max_rows=400):
    """Print the tree of one session file, enough to fill in MAT_FIELDS.

    Reads shapes and dtypes only, so it costs nothing on a 2.7 GB file.
    """
    lines = []
    with MatFile(path) as m:
        if m.v73:
            def walk(name, node):
                if name.count("/") >= max_depth or len(lines) >= max_rows:
                    return
                shape = getattr(node, "shape", None)
                dt = getattr(node, "dtype", None)
                lines.append(f"  {name:<52} {str(shape or ''):<22} {dt or 'group'}")
            m._h.visititems(walk)
        else:
            def walk(node, prefix, depth):
                if depth > max_depth or len(lines) >= max_rows:
                    return
                names = getattr(node, "_fieldnames", None)
                if names:
                    for k in names:
                        walk(getattr(node, k), f"{prefix}/{k}" if prefix else k, depth + 1)
                else:
                    a = np.asarray(node)
                    lines.append(f"  {prefix:<52} {str(a.shape):<22} {a.dtype}")
            for k, v in m._h.items():
                if not k.startswith("__"):
                    walk(v, k, 1)
    print(f"{os.path.basename(path)}  ({'v7.3 / HDF5' if _is_v73(path) else 'v5'})")
    print(f"  {'path':<52} {'shape':<22} dtype")
    print("\n".join(lines))
    return lines


# ==============================================================================================
# binning: the only arithmetic the adapter does
# ==============================================================================================
def bin_edges(bin_ms, window_ms):
    """(lo, n_bins) for the pre-registered window, in ms after the onset."""
    lo, hi = float(window_ms[0]), float(window_ms[1])
    span = hi - lo
    n = int(round(span / float(bin_ms)))
    if abs(n * float(bin_ms) - span) > 1e-9:
        raise ValueError(f"window {window_ms} is not a whole number of {bin_ms} ms bins")
    return lo, n


EPS_MS = 1e-6     # a millionth of a millisecond: finer than any recorded spike time


def bin_spike_times(times_s, onset_s, bin_ms, window_ms):
    """Counts of one neuron in one trial: half-open bins [lo + k*w, lo + (k+1)*w) after onset.

    The relative time is rounded to EPS_MS before the division. Subtracting two float seconds
    leaves an error of order 1e-13 ms, which is enough to push a spike sitting exactly on the
    first edge of the window below it: at an onset of 1.75 s a spike at 300.0 ms comes out as
    299.99999999999983 and would be dropped, while at an onset of 1.25 s the same spike is kept.
    That would thin the first bin by an amount that depends on the onset of each trial, which no
    later step could detect. Rounding at a resolution a millionfold finer than any recorded spike
    time removes it and changes nothing else.
    """
    lo, n = bin_edges(bin_ms, window_ms)
    rel = np.round((np.asarray(times_s, float) - float(onset_s)) * 1000.0, 6)
    idx = np.floor((rel - lo) / float(bin_ms)).astype(np.int64)
    keep = (idx >= 0) & (idx < n)
    return np.bincount(idx[keep], minlength=n).astype(np.int32)


def bin_raster(raster, onset_sample, bin_ms, window_ms, ms_per_sample=1.0):
    """Counts from a dense (sample, neuron) trial, summed into bins of bin_ms."""
    lo, n = bin_edges(bin_ms, window_ms)
    per_bin = int(round(float(bin_ms) / float(ms_per_sample)))
    if per_bin < 1:
        raise ValueError(f"bin of {bin_ms} ms is shorter than one sample of {ms_per_sample} ms")
    start = int(round(float(onset_sample) + lo / float(ms_per_sample)))
    stop = start + n * per_bin
    seg = np.asarray(raster)[max(start, 0):stop]
    if start < 0 or seg.shape[0] < n * per_bin:          # a trial that ends inside the window
        pad = np.zeros((n * per_bin - seg.shape[0], seg.shape[1]), seg.dtype)
        seg = np.concatenate([seg, pad], 0)
    return seg.reshape(n, per_bin, seg.shape[1]).sum(1).astype(np.int32)


# ==============================================================================================
# assembly into the canonical form
# ==============================================================================================
def _require_fields():
    missing = [k for k in ("onset", "trial_kind", "pair_id", "face_flashed") if MAT_FIELDS.get(k) is None]
    if MAT_FIELDS.get("spike_times") is None and MAT_FIELDS.get("raster") is None:
        missing.append("spike_times or raster")
    if any(v is None for v in MAT_FIELDS["kind_codes"].values()):
        missing.append("kind_codes")
    if missing:
        raise KeyError(f"MAT_FIELDS is missing {missing}.\n\n{_FIELDS_DOC}")


def _raster_axes(v73):
    """Where trial, sample and neuron sit in the array as it is actually indexed.

    `raster_order` names the axes as MATLAB sees them. h5py exposes a v7.3 dataset with the axes
    reversed, so the two vintages are indexed differently and the order has to be declared rather
    than guessed: guessing it wrong would read neurons as trials and nothing downstream would
    notice.
    """
    order = [s.strip() for s in str(MAT_FIELDS.get("raster_order")
                                    or "trial,sample,neuron").split(",")]
    if sorted(order) != ["neuron", "sample", "trial"]:
        raise ValueError(f"raster_order must name trial, sample and neuron once each, got {order}")
    return order[::-1] if v73 else order


def _raster_trial(m, node, t):
    """(sample, neuron) of one trial from a dense raster, in either vintage."""
    disk = _raster_axes(m.v73)
    sl = [slice(None)] * 3
    sl[disk.index("trial")] = int(t)
    a = np.asarray(node[tuple(sl)])
    rest = [d for d in disk if d != "trial"]
    return a.T if rest == ["neuron", "sample"] else a


def _trial_counts(m, trials, bin_ms, window_ms, n_neurons):
    """(n_trials, T, N) counts for the given trial indices."""
    lo, n_bins = bin_edges(bin_ms, window_ms)
    onset = m.column(MAT_FIELDS["onset"])
    out = np.zeros((len(trials), n_bins, n_neurons), np.int32)
    if MAT_FIELDS.get("raster") is not None:
        node = m.get(MAT_FIELDS["raster"])
        for k, t in enumerate(trials):
            out[k] = bin_raster(_raster_trial(m, node, t), onset[t], bin_ms, window_ms,
                                MAT_FIELDS.get("ms_per_sample", 1.0))
    else:
        for k, t in enumerate(trials):
            for j in range(n_neurons):
                out[k, :, j] = bin_spike_times(m.ragged(MAT_FIELDS["spike_times"], t, j),
                                               onset[t], bin_ms, window_ms)
    return out


def _n_neurons(m):
    if MAT_FIELDS.get("raster") is not None:
        node = m.get(MAT_FIELDS["raster"])
        shape = np.asarray(node).shape if not m.v73 else node.shape
        return int(shape[_raster_axes(m.v73).index("neuron")])
    ids = m.array(MAT_FIELDS["neuron_ids"], default=None) if MAT_FIELDS.get("neuron_ids") else None
    if ids is not None:
        return int(np.asarray(ids).reshape(-1).size)
    node = m.get(MAT_FIELDS["spike_times"])
    arr = node[()] if hasattr(node, "shape") and m.v73 else node
    arr = np.asarray(arr)
    arr = arr.T if arr.ndim == 2 and m.v73 else arr
    return int(arr.shape[1]) if arr.ndim == 2 else 1


def session_from_mat(path, design="FACE1500", bin_ms=10.0, window_ms=(300, 1000), name=None):
    """One released session file in the canonical form."""
    _require_fields()
    with MatFile(path) as m:
        kind = m.column(MAT_FIELDS["trial_kind"])
        pair = m.column(MAT_FIELDS["pair_id"], n=kind.size)
        flashed = m.column(MAT_FIELDS["face_flashed"], n=kind.size)
        other = (m.column(MAT_FIELDS["face_other"], n=kind.size, default=None)
                 if MAT_FIELDS.get("face_other") else None)
        codes = MAT_FIELDS["kind_codes"]
        N = _n_neurons(m)

        def sel(**kw):
            mask = np.ones(kind.size, bool)
            for k, v in kw.items():
                col = dict(kind=kind, pair=pair, flashed=flashed)[k]
                mask &= (np.asarray([str(x).strip() for x in col]) == str(v).strip()
                         if col.dtype.kind in "US" else col == v)
            return np.flatnonzero(mask)

        pairs, skipped = [], []
        for p in sorted(set(np.asarray(pair).tolist())):
            bino = sel(kind=codes["binocular"], pair=p)
            if bino.size == 0:
                continue
            faces = sorted(set(np.asarray(flashed)[bino].tolist()))
            if len(faces) != 2:
                skipped.append(dict(pair=p, reason=f"{len(faces)} flashed faces, expected 2"))
                continue
            fa, fb = faces
            got = {}
            for tag, f in (("a", fa), ("b", fb)):
                got[f"bino_{tag}"] = _trial_counts(m, sel(kind=codes["binocular"], pair=p, flashed=f),
                                                   bin_ms, window_ms, N)
                pr = sel(kind=codes["priming"], flashed=f)
                got[f"prime_{tag}"] = _trial_counts(m, pr, bin_ms, window_ms, N)
                un = sel(kind=codes["unambiguous"], pair=p, flashed=f) if codes.get("unambiguous") else []
                got[f"unamb_{tag}"] = (_trial_counts(m, un, bin_ms, window_ms, N)
                                       if len(un) else None)
            if min(got["prime_a"].shape[0], got["prime_b"].shape[0]) == 0:
                skipped.append(dict(pair=p, reason="no priming trials for one of the two faces"))
                continue
            pairs.append(Pair(face_a=int(fa), face_b=int(fb), **got))

        F = None
        if MAT_FIELDS.get("faces"):
            F = m.array(MAT_FIELDS["faces"], default=None)
            if F is not None:
                F = np.asarray(F, float)
                F = F if F.ndim == 2 else None
        ids = (m.array(MAT_FIELDS["neuron_ids"], default=None)
               if MAT_FIELDS.get("neuron_ids") else None)
        meta = dict(source=os.path.basename(path), design=design, window_ms=list(window_ms),
                    n_trials_file=int(kind.size), n_neurons=int(N), skipped=skipped,
                    neuron_ids=(np.asarray(ids).reshape(-1).tolist() if ids is not None else None))
    return Session(pairs=pairs, bin_ms=float(bin_ms), faces=F,
                   name=(name or os.path.splitext(os.path.basename(path))[0]), meta=meta)


def session_files(path):
    """The released session files of a directory, in a stable order."""
    if os.path.isfile(path):
        return [path]
    out = [os.path.join(path, f) for f in os.listdir(path)
           if re.match(r"session.*\.mat$", f, re.I)]
    return sorted(out, key=lambda p: [int(x) for x in re.findall(r"\d+", os.path.basename(p))] or [0])


def dandi_sessions(path, design="FACE1500", bin_ms=10.0, window_ms=(300, 1000)):
    """Reader of the released files (written on release day).

    Contract, fixed before the release:
      - returns a list of `realdata.Session`, one per recording session (the same neurons);
      - counts are binned at `bin_ms` inside `window_ms` after the onset of the flashed face for
        binocular trials, and after the onset of the primed face for priming trials;
      - `Pair.bino_a` holds the trials where face a was flashed (conscious = a), `bino_b` the
        trials where face b was; `prime_a`, `prime_b` the priming trials of each face alone;
      - `unamb_a`, `unamb_b` hold the unambiguous-context trials of the first experiment, or None;
      - `Session.faces` holds the feature vectors of the stimuli when the release provides them,
        with `face_a`, `face_b` indexing into it; otherwise None and the gain block is skipped;
      - units are spike counts per bin, integer, non-negative; no smoothing, no normalization.
    """
    # The release came out on Zenodo (record 23604909). Its layout needed the authors' trial rules
    # ported, which lives in `hesse_reader.py`; the generic MAT_FIELDS reader above is kept for its
    # tests only. `path` is a directory holding either the per-session caches written in Colab
    # (<session>__primary.npz) or the session_*.mat files themselves.
    from . import hesse_reader as HR

    if design not in HR.DESIGNS:
        raise KeyError(f"design {design!r}; known: {sorted(HR.DESIGNS)}")
    out = []
    for name in HR.DESIGNS[design]:
        npz = os.path.join(path, f"{name}__primary.npz")
        mat = os.path.join(path, f"{name}.mat")
        if os.path.exists(npz):
            s = load_session(npz)
        elif os.path.exists(mat):
            s = HR.read_session(mat, name=name, bin_ms=bin_ms, window_ms=window_ms)
        else:
            raise FileNotFoundError(f"{name}: neither {npz} nor {mat}")
        if float(s.bin_ms) != float(bin_ms) or list(s.meta.get("window_ms", window_ms)) != list(window_ms):
            raise ValueError(f"{name}: cache made with bin {s.bin_ms} ms, window {s.meta.get('window_ms')}")
        faces_npz = os.path.join(path, f"{name}__faces.npz")
        if os.path.exists(faces_npz):              # written in step 2b from the release's feature maps
            from . import faces as FC
            try:
                FC.attach_faces(s, faces_npz)
            except ValueError as e:
                # pre-registration, reading rule 2: without a feature space the gain block is
                # reported as not computed, nothing is substituted; the counts block is unaffected
                s.meta = dict(s.meta, faces_error=str(e))
        out.append(s)
    return out


# ==============================================================================================
# per-session cache, so that 23 GB never sit on the disk at once
# ==============================================================================================
def save_session(sess, out_path):
    """The canonical form of one session as a compressed .npz (small: counts, not spikes)."""
    d = {"__meta__": json.dumps(dict(bin_ms=sess.bin_ms, name=sess.name, meta=sess.meta,
                                     n_pairs=len(sess.pairs)))}
    if sess.faces is not None:
        d["faces"] = np.asarray(sess.faces)
    for i, p in enumerate(sess.pairs):
        for f in ("bino_a", "bino_b", "prime_a", "prime_b", "unamb_a", "unamb_b"):
            v = getattr(p, f)
            if v is not None:
                v = np.asarray(v)
                if not np.issubdtype(v.dtype, np.integer):
                    raise TypeError(f"pair {i} {f}: counts must be integer, got {v.dtype}")
                d[f"p{i}_{f}"] = v                 # dtype kept: uint8 counts stay uint8
        d[f"p{i}_faces"] = np.asarray([p.face_a, p.face_b], np.int64)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez_compressed(out_path, **d)
    return out_path


def load_session(npz_path):
    z = np.load(npz_path, allow_pickle=False)
    meta = json.loads(str(z["__meta__"]))
    pairs = []
    for i in range(meta["n_pairs"]):
        fa, fb = z[f"p{i}_faces"]
        kw = {f: (z[f"p{i}_{f}"] if f"p{i}_{f}" in z else None)
              for f in ("bino_a", "bino_b", "prime_a", "prime_b", "unamb_a", "unamb_b")}
        pairs.append(Pair(face_a=int(fa), face_b=int(fb), **kw))
    return Session(pairs=pairs, bin_ms=meta["bin_ms"], name=meta["name"], meta=meta["meta"],
                   faces=(z["faces"] if "faces" in z else None))


def load_cached_sessions(cache_dir):
    files = sorted(os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if f.endswith(".npz"))
    return [load_session(f) for f in files]


# ==============================================================================================
# the plumbing check: quantities no prediction reads
# ==============================================================================================
def plumbing_report(sessions):
    """Counts and rates that appear in no prediction of MUX-S1, for checking the reader itself.

    A1-A4 compare the released data against our digitization of the figures, so a failure there
    is ambiguous between a bad digitization and a bad reader. The numbers below separate the two:
    they are properties of the file (how many trials, how many units, how fast they fire) and are
    matched against the methods section of the paper. A reader that reproduces them is reading the
    file correctly, which leaves A1-A4 free to test the model instead of the plumbing.
    """
    rows = []
    for s in sessions:
        counts = [v for p in s.pairs for f in ("bino_a", "bino_b", "prime_a", "prime_b")
                  for v in [getattr(p, f)] if v is not None and v.size]
        tot = sum(int(v.shape[0]) for v in counts)
        spikes = sum(int(v.sum()) for v in counts)
        bins = sum(int(v.shape[0] * v.shape[1]) for v in counts)
        rows.append(dict(
            session=s.name, n_pairs=len(s.pairs), n_neurons=int(s.n_neurons),
            n_trials_used=tot, n_trials_in_file=s.meta.get("n_trials_file"),
            trials_per_pair=round(tot / max(len(s.pairs), 1), 1),
            mean_rate_hz=round(spikes / max(bins, 1) / (s.bin_ms / 1000.0) / max(int(s.n_neurons), 1), 2),
            bins_per_trial=int(counts[0].shape[1]) if counts else 0,
            skipped_pairs=len(s.meta.get("skipped") or []),
            has_faces=s.faces is not None))
    return rows


def plumbing_table(rows):
    head = ["session", "n_pairs", "n_neurons", "n_trials_used", "n_trials_in_file",
            "trials_per_pair", "mean_rate_hz", "bins_per_trial", "skipped_pairs", "has_faces"]
    L = ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
    for r in rows:
        L.append("| " + " | ".join(str(r.get(k, "")) for k in head) + " |")
    return "\n".join(L)


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3 and sys.argv[1] == "describe":
        describe_mat(sys.argv[2])
    else:
        print(__doc__)
        print("usage: python -m mux_s1.adapters describe <session_*.mat>")
