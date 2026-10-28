"""Reader of the Hesse et al. (2026) release into the canonical form of `mux_s1.realdata`.

Release: Zenodo record 23604909 (concept DOI 10.5281/zenodo.23604908), 19 files
`session_<date>_<monkey>.mat`, MATLAB 7.3 (HDF5). Code of the authors:
github.com/miou/conscious_suppressed_alternations (read for the definitions below).

Every rule here is a port of the authors' own code, cited by file, so that nothing about which
trials, which neurons or which stimuli enter the analysis is chosen by us. Where the
pre-registration of MUX-S1 1.1 speaks (window 300-1000 ms, 10-ms bins via the frozen runner,
canonical containers), it wins; where it is silent, the authors' definition is used.

Sources of the rules
--------------------
  config/recording_list.yaml, list `rec_it`      which paradigm and which probes form the IT
                                                 population of each session (the population of
                                                 the authors' Fig. 2 autocorrelation)
  config/switches_config_release.m               sessions 1-10 many_combinations, 11-19 many_faces
  matlab/per_session/parse_flash_suppression_stimuli.m   stimulus names -> content, context, images
  matlab/helpers/parse_string.m                  first occurrence, last matching pattern wins
  matlab/per_session/flash_suppression_get_variables.m   trial masks, validity, 'Throw away' rule,
                                                 the many_stimuli exception, dominant/suppressed
  matlab/per_session/flash_suppression_add_variables_to_session.m   reversed indices for priming
  matlab/helpers/computeRaster.m                 in-range trials when the raster is rebuilt
  python/src/percept_hmm/flash_suppression.py    stimulus pairs (red, cyan) and trial assignment
  python/src/percept_hmm/pipelines/dataio.py     units: all units on the listed probes, no filter

Not ported, on purpose: the random balancing of high-repeat stimuli (`trial_selected`). It is
unseeded and the masks the authors export to their dynamics pipeline are the unbalanced ones
(flash_suppression_add_variables_to_session.m writes trial_is_binocular, not *_selected).
"""
import re

import numpy as np

from .adapters import bin_edges
from .realdata import Pair, Session

# ---- which recording forms the IT population of each session (authors' rec_it) --------------
PARADIGMS = {"fsm": "flash_suppression_many_stimuli",
             "fso": "flash_suppression_optimized",
             "fsor": "flash_suppression_optimized_repeat_subset"}

REC_IT = {
    "session_250225_Alfie": ("fsor", (1, 2)), "session_250312_Alfie": ("fsor", (1, 2)),
    "session_250317_Alfie": ("fsor", (1, 2)), "session_250321_Alfie": ("fsor", (1, 2)),
    "session_250504_Mowgli": ("fso", (1, 2)), "session_250513_Alfie": ("fso", (1,)),
    "session_250518_Mowgli": ("fso", (1, 2)), "session_250604_Alfie": ("fso", (1,)),
    "session_250615_Alfie": ("fso", (1,)), "session_251212_Alfie": ("fsm", (1, 2)),
    "session_251218_Alfie": ("fsm", (1, 2)), "session_260108_Mowgli": ("fsm", (1, 2)),
    "session_260119_Alfie": ("fsm", (1, 2)), "session_260205_Mowgli": ("fsm", (1, 2)),
    "session_260210_Mowgli": ("fso", (1,)), "session_260218_Mowgli": ("fsm", (1, 2)),
    "session_260225_Mowgli": ("fsm", (1, 2)), "session_260309_Alfie": ("fsm", (1,)),
    "session_260322_Alfie": ("fsm", (1,)),
}
MANY_COMBINATIONS = ["session_250225_Alfie", "session_250312_Alfie", "session_250317_Alfie",
                     "session_250321_Alfie", "session_250504_Mowgli", "session_250513_Alfie",
                     "session_250518_Mowgli", "session_250604_Alfie", "session_250615_Alfie",
                     "session_260210_Mowgli"]
MANY_FACES = ["session_251212_Alfie", "session_251218_Alfie", "session_260108_Mowgli",
              "session_260119_Alfie", "session_260205_Mowgli", "session_260218_Mowgli",
              "session_260225_Mowgli", "session_260309_Alfie", "session_260322_Alfie"]
# FACE1500 in MUX-S1 1.0 is "a reduced copy of the authors' second experiment, 1500 faces, 9
# sessions": the many_faces group. ALL is the 19 IT recordings the authors pool for Fig. 2.
DESIGNS = {"FACE1500": MANY_FACES, "ALL": MANY_COMBINATIONS + MANY_FACES}

REPEAT_STIM_MAX = 60         # python/scripts/fs_rivalry/extract_data.py, RSTIM_MAX (0-based)
INJECTION = re.compile(r"\binject", re.I)


# ==============================================================================================
# MATLAB 7.3 access
# ==============================================================================================
def _cls(node):
    v = node.attrs.get("MATLAB_class", b"")
    return v.decode() if isinstance(v, bytes) else str(v)


def _empty(node):
    return "MATLAB_empty" in node.attrs


def _text(ds):
    if _empty(ds):
        return ""
    a = np.asarray(ds[()]).reshape(-1)
    return "".join(chr(int(c)) for c in a)


def _num(ds):
    """A numeric field in MATLAB orientation, squeezed."""
    if _empty(ds):
        return np.zeros(0)
    a = np.asarray(ds[()])
    return np.squeeze(a.T if a.ndim >= 2 else a)


def _cell_refs(h, ds):
    if _empty(ds):
        return []
    refs = np.asarray(ds[()])
    refs = refs.T.reshape(-1, order="F") if refs.ndim >= 2 else refs.reshape(-1)
    return [h[r] for r in refs]


def _cellstr(h, ds):
    """A cell of strings; a single char field is read as a one-element list."""
    if _cls(ds) == "char":
        return [_text(ds)]
    return [_text(x) for x in _cell_refs(h, ds)]


def _cell_or_vector(h, ds):
    """Entries of a field that is a cell of numbers in the release, or a plain numeric vector."""
    import h5py
    if _empty(ds):
        return []
    if ds.dtype == h5py.ref_dtype:
        return [np.asarray(_num(x)).reshape(-1) for x in _cell_refs(h, ds)]
    return [np.array([v]) for v in np.asarray(_num(ds)).reshape(-1)]


def _find_paradigm(h, paradigm_name):
    """The struct in session.paradigms_data whose paradigm_name equals the given name."""
    import h5py
    found = [g for g in _cell_refs(h, h["session"]["paradigms_data"])
             if isinstance(g, h5py.Group) and "paradigm_name" in g and _text(g["paradigm_name"]) == paradigm_name]
    if len(found) != 1:
        raise ValueError(f"{len(found)} paradigms named {paradigm_name!r} in the file, expected one")
    return found[0]


# ==============================================================================================
# the authors' trial definitions, ported
# ==============================================================================================
def parse_string(s, patterns):
    """matlab/helpers/parse_string.m: first occurrence; the LAST matching pattern in the list wins."""
    if isinstance(patterns, str):
        patterns = [patterns]
    match, before, after = None, s, s
    for i, p in enumerate(patterns):
        pos = s.find(p)
        if pos >= 0:
            match = i
            before = s[:pos]
            after = s[pos + len(p):]
    return match, before, after


CONTENTS = ["_red_only", "_cyan_only", "_binocular", "_transparent"]
CONTEXTS = ["_red_suppressed", "_cyan_suppressed"]


def parse_stimuli(stimulus_list):
    """matlab/per_session/parse_flash_suppression_stimuli.m"""
    red, cyan, content, context = [], [], [], []
    for name in stimulus_list:
        if name == "gray.tif":
            red.append("gray"); cyan.append("gray"); content.append("gray"); context.append("gray")
            continue
        m, before, _ = parse_string(name, CONTENTS)
        if m is None:
            raise ValueError(f"stimulus {name!r}: no content suffix")
        content.append(CONTENTS[m])
        m, before, _ = parse_string(before, CONTEXTS)
        if m is None:
            raise ValueError(f"stimulus {name!r}: no context")
        context.append(CONTEXTS[m])
        _, before, after = parse_string(before, "_cyan_")
        cyan.append(after)
        _, before, after = parse_string(before, "_red_")
        red.append(after)
    content, context = np.array(content, object), np.array(context, object)
    has = lambda arr, sub: np.array([sub in x for x in arr])
    is_priming = ((has(context, "red_suppressed") & has(content, "red_only"))
                  | (has(context, "cyan_suppressed") & has(content, "cyan_only")))
    is_unambiguous = (has(content, "red_only") | has(content, "cyan_only")) & ~is_priming
    return dict(image_red=np.array(red, object), image_cyan=np.array(cyan, object),
                contents=content, contexts=context, is_priming=is_priming,
                is_unambiguous=is_unambiguous, is_binocular=has(content, "binocular"),
                is_transparent=has(content, "transparent"),
                is_gray=np.array([c == "gray" for c in context]))


def _first_or_none(a):
    """A cell entry of comments_succeeding_trial_index as MATLAB's colon ranges read it.

    An empty entry (a comment after the last trial) makes every range in the 'Throw away' rule
    empty, so nothing is dropped: None. A vector entry is read by the colon operator through its
    first element: the first element.
    """
    a = np.asarray(a).reshape(-1)
    return float(a[0]) if a.size else None


def throw_away_mask(n_trials, comments, comment_trial_index, ab_valid_trials):
    """The 'Throw away last N blocks' rule of flash_suppression_get_variables.m (goggles forgotten).

    MATLAB indices are 1-based and the ranges a:b are inclusive; the port keeps that arithmetic.
    """
    keep = np.ones(n_trials, bool)
    ab = np.asarray(ab_valid_trials, bool).reshape(-1)
    valid_pos = np.flatnonzero(ab) + 1                         # find(ab_valid_trials), 1-based
    for ci, c in enumerate(comments):
        if "Throw away last " not in c:
            continue
        rest = c.split("Throw away last ")[-1]
        n_blocks = int(float(rest.split(" blocks")[0]))
        if ci >= len(comment_trial_index) or comment_trial_index[ci] is None:
            continue                                           # empty index: MATLAB's ranges are empty, nothing is dropped
        t = int(comment_trial_index[ci])                       # among valid trials, 1-based
        t_all = int(valid_pos[t - 1])                          # among all trials, 1-based
        lo = t_all - 3 * n_blocks
        n_throw = int(ab[lo - 1:t_all].sum())                  # ab(lo:t_all)
        keep[max(t - n_throw, 1) - 1:t] = False                # keep((t - n_throw):t) = false
    return keep


def classify_trials(paradigm_name, stimulus_list, stimulus_index_valid, b_previous_trial_valid,
                    comments, comment_trial_index, ab_valid_trials, b_in_range):
    """Trial masks and dominant/suppressed indices, as flash_suppression_get_variables.m and
    flash_suppression_add_variables_to_session.m compute them. Indices are 0-based."""
    sp = parse_stimuli(stimulus_list)
    b_in_range = np.asarray(b_in_range, bool)
    stim = (np.asarray(stimulus_index_valid).reshape(-1).astype(np.int64) - 1)[b_in_range]

    # priming stimuli and their lists (MATLAB unique sorts)
    n_stim = len(stimulus_list)
    priming_img = np.array([None] * n_stim, object)
    sel_r = sp["is_priming"] & np.array(["red_only" in x for x in sp["contents"]])
    sel_c = sp["is_priming"] & np.array(["cyan_only" in x for x in sp["contents"]])
    priming_img[sel_r] = sp["image_red"][sel_r]
    priming_img[sel_c] = sp["image_cyan"][sel_c]
    plist = sorted({x for x in priming_img if x})
    red_list = set(sp["image_red"][sel_r])
    cyan_list = set(sp["image_cyan"][sel_c])
    pos = {name: i for i, name in enumerate(plist)}

    dominant = np.array([None] * n_stim, object)
    suppressed = np.array([None] * n_stim, object)
    rs = np.array(["red_suppressed" in x for x in sp["contexts"]])
    cs = np.array(["cyan_suppressed" in x for x in sp["contexts"]])
    dominant[rs], suppressed[rs] = sp["image_cyan"][rs], sp["image_red"][rs]
    dominant[cs], suppressed[cs] = sp["image_red"][cs], sp["image_cyan"][cs]

    is_priming = sp["is_priming"][stim]
    is_gray = sp["is_gray"][stim]

    # validity: previous trial valid AND previous trial priming AND not thrown away
    keep = throw_away_mask(len(np.asarray(b_previous_trial_valid).reshape(-1)), comments,
                           comment_trial_index, ab_valid_trials)
    prev_ok = np.asarray(b_previous_trial_valid, bool).reshape(-1)[b_in_range]
    valid = prev_ok & np.roll(is_priming, 1) & keep[b_in_range]       # circshift(.., 1)
    bt = sp["is_binocular"][stim] | sp["is_transparent"][stim]
    exception = False
    if "flash_suppression_many_stimuli" in paradigm_name:
        if len(np.unique(stim[valid & bt])) < len(np.unique(stim[bt])):
            valid = keep[b_in_range].copy()
            exception = True

    is_binocular = sp["is_binocular"][stim] & valid
    is_unambiguous = sp["is_unambiguous"][stim] & valid
    is_transparent = sp["is_transparent"][stim] & valid

    dom_idx = np.full(stim.size, -1, np.int64)
    sup_idx = np.full(stim.size, -1, np.int64)
    for t, s in enumerate(stim):
        if is_priming[t]:                                     # reversed for priming
            d, u = suppressed[s], dominant[s]
        elif not is_gray[t]:
            d, u = dominant[s], suppressed[s]
        else:
            continue
        if d not in pos or u not in pos:
            raise ValueError(f"trial {t}: image {d!r} / {u!r} is not among the priming stimuli")
        dom_idx[t], sup_idx[t] = pos[d], pos[u]

    return dict(stim=stim, is_priming=is_priming, is_binocular=is_binocular,
                is_unambiguous=is_unambiguous, is_transparent=is_transparent, is_gray=is_gray,
                dom_idx=dom_idx, sup_idx=sup_idx, priming_list=plist,
                is_red=np.array([x in red_list for x in plist]),
                is_cyan=np.array([x in cyan_list for x in plist]),
                n_binocular_all=int(sp["is_binocular"][stim].sum()),
                n_unambiguous_all=int(sp["is_unambiguous"][stim].sum()),
                n_thrown_away=int((~keep).sum()), many_stimuli_exception=exception)


def stimulus_pairs(cl):
    """python/src/percept_hmm/flash_suppression.py, get_stim_pairs: (red, cyan) from priming trials."""
    pairs = set()
    for t in np.flatnonzero(cl["is_priming"]):
        s1, s2 = int(cl["dom_idx"][t]), int(cl["sup_idx"][t])
        if cl["is_red"][s1] and cl["is_cyan"][s2]:
            pairs.add((s1, s2))
        elif cl["is_cyan"][s1] and cl["is_red"][s2]:
            pairs.add((s2, s1))
        else:
            raise ValueError("Stimulus pair not red/cyan!")
    return sorted(pairs)


# ==============================================================================================
# spike counts
# ==============================================================================================
def _window_counts(par, units, trials_in_file, b_in_range, bin_ms, window_ms, block=256):
    """(n_trials_in_range, T, N) counts in the window, from raster_all_units or timecourse_all_units.

    raster_all_units is used when present and long enough, as in flash_suppression_get_variables.m;
    otherwise the raster is rebuilt from the timecourse with computeRaster's in-range rule.
    """
    lo, n_bins = bin_edges(bin_ms, window_ms)
    per_bin = int(round(bin_ms))
    t_idx = np.flatnonzero(b_in_range)
    out = np.zeros((t_idx.size, n_bins, units.size), np.uint8)
    if "raster_all_units" in par and not _empty(par["raster_all_units"]):
        psr = _num(par["peri_stimulus_range"]).reshape(-1)
        if not np.all(np.diff(psr) == 1):
            raise ValueError("peri_stimulus_range is not a 1-ms axis")
        i0 = int(np.flatnonzero(psr == lo)[0])
        i1 = i0 + n_bins * per_bin
        if i1 > psr.size:
            raise ValueError("the window runs past the end of raster_all_units")
        ds = par["raster_all_units"]                 # on disk: (unit, time, trial)
        for b0 in range(0, t_idx.size, block):
            tt = t_idx[b0:b0 + block]
            slab = np.asarray(ds[:, i0:i1, int(tt[0]):int(tt[-1]) + 1])[:, :, tt - tt[0]]
            slab = slab[units]                       # (N, time, trials)
            c = slab.reshape(units.size, n_bins, per_bin, tt.size).sum(2)
            out[b0:b0 + tt.size] = np.transpose(c, (2, 1, 0))
    else:
        ds = par["timecourse_all_units"]             # on disk: (unit, time)
        on = _num(par["timecourse_on_times"]).reshape(-1)
        for k, t in enumerate(t_idx):
            s0 = int(round(on[t])) - 1 + int(lo)     # 1-based ms sample of the onset
            seg = np.asarray(ds[:, s0:s0 + n_bins * per_bin])[units]
            out[k] = seg.reshape(units.size, n_bins, per_bin).sum(2).T
    return out


def _in_range(par, n_trials):
    """b_in_range of flash_suppression_get_variables.m."""
    on_dur = _num(par["on_durations"]).reshape(-1)
    need = max(-200.0, float(on_dur.max()) if on_dur.size else 0.0)
    if "raster_all_units" in par and not _empty(par["raster_all_units"]):
        psr = _num(par["peri_stimulus_range"]).reshape(-1)
        if psr[-1] >= need:
            return np.ones(n_trials, bool), "raster_all_units"
    on = _num(par["timecourse_on_times"]).reshape(-1)
    L = int(par["timecourse_all_units"].shape[1])     # disk (unit, time)
    after = float(on_dur.max())
    return (on - 200 > 0) & (on + after <= L), "timecourse_all_units"


# ==============================================================================================
# one session in the canonical form
# ==============================================================================================
def load_release(path, name=None, bin_ms=10.0, span_ms=(300, 1000)):
    """Open a release file once: classify its trials and take counts over span_ms.

    Every variant (primary and secondary) is then cut from this by `build_session`, so a 2.7 GB
    file is read once however many variants are made from it.
    """
    import os
    import h5py

    name = name or os.path.splitext(os.path.basename(path))[0]
    if name not in REC_IT:
        raise KeyError(f"{name} is not in the authors' rec_it list")
    abbr, probes = REC_IT[name]
    with h5py.File(path, "r") as h:
        s = h["session"]
        par = _find_paradigm(h, PARADIGMS[abbr])
        pname = _text(par["paradigm_name"])
        stim_list = _cellstr(h, par["stimulus_list"])
        siv = _num(par["stimulus_index_valid"]).reshape(-1)
        n_trials = siv.size
        prev = _num(par["b_previous_trial_valid"]).reshape(-1)
        comments = _cellstr(h, par["comments"]) if "comments" in par else []
        raw_cidx = (_cell_or_vector(h, par["comments_succeeding_trial_index"])
                    if "comments_succeeding_trial_index" in par else [])
        cidx = [_first_or_none(a) for a in raw_cidx]
        cidx_note = dict(empty=sum(a.size == 0 for a in raw_cidx), multi=sum(a.size > 1 for a in raw_cidx))
        ab = _num(par["debug"]["ab_valid_trials"]).reshape(-1) if "debug" in par else np.ones(0)
        b_in_range, source = _in_range(par, n_trials)
        cl = classify_trials(pname, stim_list, siv, prev, comments, cidx, ab, b_in_range)
        unit_probe = _num(s["unit_probes_index"]).reshape(-1).astype(int)
        units = np.flatnonzero(np.isin(unit_probe, probes))
        counts = _window_counts(par, units, n_trials, b_in_range, bin_ms, span_ms)
    return dict(path=path, name=name, abbr=abbr, probes=probes, pname=pname, source=source,
                bin_ms=float(bin_ms), span_ms=tuple(span_ms), counts=counts, cl=cl,
                b_in_range=b_in_range, n_trials=int(n_trials), comments=comments, cidx=cidx, cidx_note=cidx_note,
                unit_probe=unit_probe, units=units)


def _cut(L, window_ms):
    """Counts of window_ms cut from the span read by load_release."""
    lo_s, _ = bin_edges(L["bin_ms"], L["span_ms"])
    lo, n = bin_edges(L["bin_ms"], window_ms)
    i0 = (lo - lo_s) / L["bin_ms"]
    if i0 != int(i0) or i0 < 0 or int(i0) + n > L["counts"].shape[1]:
        raise ValueError(f"window {window_ms} is not inside the span {L['span_ms']} on the bin grid")
    return L["counts"][:, int(i0):int(i0) + n]


def build_session(L, window_ms=(300, 1000), prime_window_ms=None, exclude_after_injection=False,
                  repeat_only=False):
    """One variant of a loaded release file as realdata.Session.

    prime_window_ms   None: priming counts use window_ms (the pre-registered contract).
                      (100, 1000): the window of the authors' LLR templates (secondary analysis).
    exclude_after_injection   secondary analysis: trials from the first comment that mentions an
                      injection onward are dropped (session 250504_Mowgli records saline, then
                      psilocybin; neither the authors' code nor the preprint mentions it).
    repeat_only       secondary analysis: in fsm sessions keep pairs with both indices below 60,
                      as the authors' Fig. 5 dwell-time pipeline does.
    """
    import os

    cl, abbr = L["cl"], L["abbr"]
    counts = _cut(L, window_ms)
    pcounts = counts if prime_window_ms is None else _cut(L, prime_window_ms)
    comments, cidx, b_in_range = L["comments"], L["cidx"], L["b_in_range"]

    n_tr = cl["stim"].size
    use = np.ones(n_tr, bool)
    inj = [(c, cidx[i] if i < len(cidx) else None) for i, c in enumerate(comments) if INJECTION.search(c)]
    inj_idx = [int(t) for _, t in inj if t is not None]
    if exclude_after_injection and inj_idx:
        first = min(inj_idx)                                            # 1-based, among trials
        in_range_pos = np.flatnonzero(b_in_range)                      # file trial of each row
        use &= (in_range_pos + 1) < first

    pairs_rc = stimulus_pairs(cl)
    if repeat_only and abbr == "fsm":
        pairs_rc = [(r, c) for r, c in pairs_rc if r < REPEAT_STIM_MAX and c < REPEAT_STIM_MAX]

    dom, sup = cl["dom_idx"], cl["sup_idx"]
    # priming trials of a stimulus come from the pairs that are kept (get_stim_priming_trials
    # collects them from the pair table); a priming trial's pair is its (red, cyan) ordering
    kept = set(pairs_rc)
    d0, s0 = np.maximum(dom, 0), np.maximum(sup, 0)
    first_branch = cl["is_red"][d0] & cl["is_cyan"][s0]          # same order as get_stim_pairs
    red_first = np.where(first_branch, dom, sup)
    cyan_second = np.where(first_branch, sup, dom)
    in_kept = np.array([(int(a), int(b)) in kept for a, b in zip(red_first, cyan_second)])
    prime_rows = {sidx: np.flatnonzero(cl["is_priming"] & use & in_kept & (dom == sidx))
                  for sidx in range(len(cl["priming_list"]))}
    pairs, skipped = [], []
    for r, c in pairs_rc:
        in_pair = ((dom == r) & (sup == c)) | ((dom == c) & (sup == r))
        rows = dict(
            bino_a=np.flatnonzero(cl["is_binocular"] & use & in_pair & (dom == r)),
            bino_b=np.flatnonzero(cl["is_binocular"] & use & in_pair & (dom == c)),
            unamb_a=np.flatnonzero(cl["is_unambiguous"] & use & in_pair & (dom == r)),
            unamb_b=np.flatnonzero(cl["is_unambiguous"] & use & in_pair & (dom == c)))
        pa, pb = prime_rows[r], prime_rows[c]
        if min(rows["bino_a"].size, rows["bino_b"].size, pa.size, pb.size) == 0:
            skipped.append(dict(pair=[int(r), int(c)], n_bino_a=int(rows["bino_a"].size),
                                n_bino_b=int(rows["bino_b"].size), n_prime_a=int(pa.size),
                                n_prime_b=int(pb.size)))
            continue
        pairs.append(Pair(bino_a=counts[rows["bino_a"]], bino_b=counts[rows["bino_b"]],
                          prime_a=pcounts[pa], prime_b=pcounts[pb],
                          unamb_a=(counts[rows["unamb_a"]] if rows["unamb_a"].size else None),
                          unamb_b=(counts[rows["unamb_b"]] if rows["unamb_b"].size else None),
                          face_a=int(r), face_b=int(c)))

    unit_probe = L["unit_probe"]
    meta = dict(
        source=os.path.basename(L["path"]), rec_it=dict(paradigm=abbr, probes=list(L["probes"])),
        paradigm_name=L["pname"], raster_source=L["source"], window_ms=list(window_ms),
        prime_window_ms=list(prime_window_ms or window_ms), bin_ms=L["bin_ms"],
        n_trials_file=L["n_trials"], n_trials_in_range=int(n_tr),
        n_units_session=int(unit_probe.size),
        n_units_by_probe={int(p): int((unit_probe == p).sum()) for p in np.unique(unit_probe)},
        n_units_used=int(L["units"].size),
        n_priming=int(cl["is_priming"].sum()), n_binocular_valid=int(cl["is_binocular"].sum()),
        n_binocular_all=cl["n_binocular_all"], n_unambiguous_valid=int(cl["is_unambiguous"].sum()),
        n_thrown_away=cl["n_thrown_away"], many_stimuli_exception=cl["many_stimuli_exception"],
        n_pairs_all=len(stimulus_pairs(cl)), n_pairs_used=len(pairs), skipped=skipped,
        priming_list=cl["priming_list"], comments=comments,
        injection_comments=[c for c, _ in inj], comment_index_note=L.get("cidx_note", {}),
        excluded_after_injection=int((~use).sum()), repeat_only=bool(repeat_only and abbr == "fsm"))
    return Session(pairs=pairs, bin_ms=L["bin_ms"], name=L["name"], faces=None, meta=meta)


def _span(*windows):
    ws = [w for w in windows if w is not None]
    return (min(w[0] for w in ws), max(w[1] for w in ws))


def read_session(path, name=None, bin_ms=10.0, window_ms=(300, 1000), prime_window_ms=None,
                 exclude_after_injection=False, repeat_only=False):
    """One release file -> realdata.Session (the IT population of the authors' rec_it list)."""
    L = load_release(path, name, bin_ms, _span(window_ms, prime_window_ms))
    return build_session(L, window_ms, prime_window_ms, exclude_after_injection, repeat_only)


# the variants written in the Colab run; only 'primary' is read by the frozen runner
VARIANTS = {
    "primary": dict(),
    "prime100": dict(prime_window_ms=(100, 1000)),
    "noinj": dict(exclude_after_injection=True),
    "repeat": dict(repeat_only=True),
}


def read_variants(path, name=None, bin_ms=10.0, window_ms=(300, 1000), variants=None):
    """All variants of one file from a single read."""
    variants = VARIANTS if variants is None else variants
    span = _span(window_ms, *[v.get("prime_window_ms") for v in variants.values()])
    L = load_release(path, name, bin_ms, span)
    return {k: build_session(L, window_ms, **v) for k, v in variants.items()}
