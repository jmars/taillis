#!/usr/bin/env python3
"""COLD-BAND — is there a stable contiguous COLD band in the expert axis?

Deciding measurement for layout.md §10.7 (is `taillis.variant_band` worth having?), asked on
instruments already in the repository: per-expert ROUTING, two populations.

PURE TRACE ANALYSIS. No GPU, no gpu.lock, no model run. Inputs are READ-ONLY:
  * general sample   ~/futaie/aubier/logs/engine/m1-u6/trace-{A1..A6,ctrl_B1..B4}.txt
                     rows `R <layer> <pos> <sel0..3> <w0..w3>`; pos < n_prompt is PREFILL.
                     n_prompt is MEASURED from each trace's own run log (`# tokens = ... :
                     n_prompt=P + n_decode=D = N positions`, trace-*.stdout.txt), not assumed.
  * target corpus    /var/lib/taillis/trace-a++/conversations/dump-conv-*.txt (96 files)
                     rows `R <ubatch> L <layer> T <n_tokens> <4*T expert ids>`;
                     T > 1 is PREFILL, T == 1 is DECODE (this is the split q1-census used; it
                     also absorbs the known L35 dumper artifact — see the validation section).

METRIC (stated, and the trap avoided): SELECTION COUNT per (layer, expert) — each of the 4 ids in
a row counts once — because the two populations do not carry the same fields (aubier has weights,
ours does not). Counts are normalised PER LAYER (each layer sums to 1) so layers are comparable.
A cold window of width w is compared against the uniform share w/128. Weight mass is computed for
aubier ONLY, as a clearly-labelled SECONDARY check, and is never mixed with selection counts.

PREFILL AND DECODE ARE NEVER POOLED. Every statistic is computed per regime.

EXCLUSIONS (each already verified in the brief; re-verified here where free):
  * trace-ctrl_B1-rep2.txt is byte-identical to trace-ctrl_B1.txt  -> EXCLUDED (double count).
  * trace-A5-rebuild.txt is byte-identical to trace-A5.txt          -> EXCLUDED (double count).
  * trace-treat-B{1..4}.txt are the TREATMENT arm: canonical router ids but a DIFFERENT DECODE
    TRAJECTORY (layer-0 prompt phase bit-exact ctrl-vs-treat, layer-0 decode agrees on 12/128,
    3/128, 2/128, 45/128) -> EXCLUDED so one prompt contributes one trajectory.
  * trace-ctrl32_B1.txt is not in the named set (same prompt as ctrl_B1, a re-run at a different
    activation precision) -> EXCLUDED.

PRE-DECLARED RULE (fixed before any data was read; NOT moved after seeing results)
--------------------------------------------------------------------------------
Definitions fixed here:
  * window [lo, lo+w) with 0 <= lo <= 128-w (NON-WRAPPING). Tie-break: LOWEST lo.
  * layer mass m_L(lo,w) = sum of that layer's normalised selection share over the window.
  * uniform window share U(w) = w/128.
  * "coldest-or-near" in a layer  ==  m_L(lo,w) <= 1.25 * min_lo' m_L(lo',w)   (within 25% of
    that layer's own best achievable window). If the layer's minimum is 0, "near" means == 0.
  * ADEQUATE cell (population x regime) == median per-layer token count >= 100 tokens.
  * The candidate band for a regime is the width-4 window chosen on the GENERAL SAMPLE (the
    dual criterion's "select on general, test on target"); the mirror direction (chosen on the
    target corpus) is reported as a symmetric check.
  * Null: per-layer independent random RELABELLING of the 128 experts (shares preserved, so the
    null is exact for the window statistic), NPERM = 2000, seed 20261018.

STABLE BAND iff  STABLE(width-4 candidate window W*, per regime) where
  (i)   materially cold: mean over the 36 layers of m_L(W*,4) <= 0.5 * U(4) = 0.015625;
  (ii)  coldest-or-near in >= 30 of 36 layers;
  (iii) reproduced on held-out halves of BOTH populations: in EACH half of EACH population the
        same W* satisfies (i) and (ii);
  (iv)  outside the permutation null in BOTH populations: p(W*) < 0.01, where
        p(W*) = P_null(S(W*) <= S_obs(W*)), S(lo) = sum over the 36 layers of m_L(lo,4).
NO STABLE BAND otherwise. A regime whose cell is not ADEQUATE is INSUFFICIENT.

OVERALL VERDICT (fixed here): STABLE BAND iff at least one adequate regime PASSES and no adequate
regime returns NO STABLE BAND (i.e. both regimes PASS, or one PASSES and the other is
INSUFFICIENT). Otherwise NO STABLE BAND, or INSUFFICIENT if no regime is adequate.
"""

import hashlib
import json
import os
import sys

import numpy as np

G_DIR = os.path.expanduser("~/futaie/aubier/logs/engine/m1-u6")
T_DIR = "/var/lib/taillis/trace-a++/conversations"
OUT = os.path.expanduser("~/futaie/taillis/results/cold-band-evidence")

G_FILES = [("A1", "A"), ("A2", "A"), ("A3", "A"), ("A4", "A"), ("A5", "A"), ("A6", "A"),
           ("ctrl_B1", "B"), ("ctrl_B2", "B"), ("ctrl_B3", "B"), ("ctrl_B4", "B")]
EXCLUDED = ["trace-ctrl_B1-rep2.txt", "trace-A5-rebuild.txt", "trace-treat-B1.txt",
            "trace-treat-B2.txt", "trace-treat-B3.txt", "trace-treat-B4.txt",
            "trace-ctrl32_B1.txt"]

NL, NE, KUSED = 36, 128, 4
WIDTHS = (3, 4, 5, 8)
NPERM = 2000
NPERM_SPEC = 1000
SEED = 20261018
ADEQ_TOKENS = 100
NEAR = 1.25
COLD_FRAC = 0.5
MIN_LAYERS_NEAR = 30
P_MAX = 0.01


# ---------------------------------------------------------------- validation + loading
def sha256(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def prompt_len(name):
    """MEASURED from the trace's own run log; the brief's B-set numbers are used as a check."""
    with open(os.path.join(G_DIR, "trace-%s.stdout.txt" % name)) as fh:
        for line in fh:
            if line.startswith("# tokens"):
                # '# tokens = <path>: n_prompt=P + n_decode=D = N positions'
                seg = line.split("n_prompt=")[1]
                p = int(seg.split("+")[0])
                d = int(seg.split("n_decode=")[1].split("=")[0])
                return p, d
    raise SystemExit("no '# tokens' line for %s" % name)


VAL = {"excluded": [], "g_files": {}, "t_files": {}, "problems": []}


def load_general():
    """-> (counts, wmass, tokens) per (pop='G', regime), plus per-set ('A','B') counters."""
    acc = {}
    for name, setname in G_FILES:
        path = os.path.join(G_DIR, "trace-%s.txt" % name)
        p_len, d_len = prompt_len(name)
        counts = np.zeros((2, NL, NE))       # [0]=prefill [1]=decode
        wmass = np.zeros((2, NL, NE))
        pos_seen = [set(), set()]
        rows = 0
        maxid = -1
        layerseen = [set(), set()]
        with open(path) as fh:
            for line in fh:
                if line.startswith("#") or not line.strip():
                    continue
                f = line.split()
                if f[0] != "R":
                    raise SystemExit("bad row in %s: %r" % (path, line[:60]))
                layer, pos = int(f[1]), int(f[2])
                ids = [int(x) for x in f[3:7]]
                ws = [float(x) for x in f[7:11]]
                reg = 0 if pos < p_len else 1
                if layer >= NL or any(i < 0 or i >= NE for i in ids):
                    VAL["problems"].append("%s id/layer out of range at pos %d" % (name, pos))
                pos_seen[reg].add(pos)
                layerseen[reg].add(layer)
                for i, w in zip(ids, ws):
                    counts[reg, layer, i] += 1.0
                    wmass[reg, layer, i] += w
                maxid = max(maxid, max(ids))
                rows += 1
        # validation: every layer present in prefill, and the position sets are contiguous
        pre_ok = (len(layerseen[0]) == NL)
        dec_missing = sorted(set(range(NL)) - layerseen[1])
        VAL["g_files"][name] = {
            "set": setname, "prompt_tokens": p_len, "decode_tokens": d_len,
            "rows": rows, "max_expert_id": maxid,
            "layers_prefill": len(layerseen[0]), "layers_decode": len(layerseen[1]),
            "prefill_positions": len(pos_seen[0]), "decode_positions": len(pos_seen[1]),
            "prefill_layers_complete": pre_ok,
        }
        if len(pos_seen[0]) != p_len or len(pos_seen[1]) != d_len:
            VAL["problems"].append(
                "%s: positions %d/%d != n_prompt/n_decode %d/%d"
                % (name, len(pos_seen[0]), len(pos_seen[1]), p_len, d_len))
        if dec_missing:
            VAL["g_files"][name]["decode_missing_layers"] = dec_missing
        for reg in (0, 1):
            key = (setname, reg)
            if key not in acc:
                acc[key] = [np.zeros((NL, NE)), np.zeros((NL, NE))]
            acc[key][0] += counts[reg]
            acc[key][1] += wmass[reg]
    return acc


def load_target():
    files = sorted(f for f in os.listdir(T_DIR) if f.startswith("dump-conv-") and f.endswith(".txt"))
    halves = {"H1": [np.zeros((2, NL, NE)), np.zeros((2, NL, NE))],
              "H2": [np.zeros((2, NL, NE)), np.zeros((2, NL, NE))]}
    l35_absent_prefill = 0     # prefill row-sets whose layer set lacks 35 (dumper artifact)
    pre_row_sets = 0
    dec_row_sets = 0
    dec_row_sets_one_row = 0   # the L35 row reappears alone, inside a prefill ubatch's group
    l35_decode_rows = 0
    tvals = {}
    for idx, fn in enumerate(files):
        counts = np.zeros((2, NL, NE))
        ub_prev = -1
        nrows = 0
        ub_layers = {}
        with open(os.path.join(T_DIR, fn)) as fh:
            for line in fh:
                if line.startswith("#") or not line.strip():
                    continue
                f = line.split()
                if f[0] != "R" or f[2] != "L" or f[4] != "T":
                    raise SystemExit("bad row in %s: %r" % (fn, line[:60]))
                ub, layer, T = int(f[1]), int(f[3]), int(f[5])
                ids = f[6:]
                if len(ids) != KUSED * T:
                    VAL["problems"].append("%s: row R %d L %d T %d has %d ids"
                                           % (fn, ub, layer, T, len(ids)))
                ii = [int(x) for x in ids]
                if any(i < 0 or i >= NE for i in ii):
                    VAL["problems"].append("%s: expert id out of range at ub %d L %d" % (fn, ub, layer))
                if ub < ub_prev:
                    VAL["problems"].append("%s: ubatch went backwards at %d" % (fn, ub))
                ub_prev = ub
                reg = 0 if T > 1 else 1
                ub_layers.setdefault(ub, {"pre": set(), "dec": set()})[
                    "pre" if T > 1 else "dec"].add(layer)
                if layer == 35 and reg == 1:
                    l35_decode_rows += 1
                counts[reg, layer] += np.bincount(np.asarray(ii), minlength=NE)
                tvals[T] = tvals.get(T, 0) + 1
                nrows += 1
        h = "H1" if idx % 2 == 0 else "H2"
        halves[h][0] += counts
        missing35 = sum(1 for v in ub_layers.values() if v["pre"] and 35 not in v["pre"])
        pre_row_sets += sum(1 for v in ub_layers.values() if v["pre"])
        dec_row_sets += sum(1 for v in ub_layers.values() if v["dec"])
        dec_row_sets_one_row += sum(1 for v in ub_layers.values()
                                    if v["dec"] and len(v["dec"]) == 1)
        l35_absent_prefill += missing35
        VAL["t_files"][fn] = {"rows": nrows, "half": h, "prefill_row_sets_lacking_L35": missing35}
    return files, halves, (l35_absent_prefill, pre_row_sets, dec_row_sets,
                           dec_row_sets_one_row, l35_decode_rows), tvals


def main():
    np.random.seed(SEED)
    res = {}

    # ---------------- duplicate check on the excluded files (free, and it is the cited evidence)
    pairs = [("trace-A5.txt", "trace-A5-rebuild.txt"), ("trace-ctrl_B1.txt", "trace-ctrl_B1-rep2.txt")]
    for a, b in pairs:
        ha, hb = sha256(os.path.join(G_DIR, a)), sha256(os.path.join(G_DIR, b))
        VAL["excluded"].append({"pair": [a, b], "identical": ha == hb, "sha256": ha[:16]})
    for name in EXCLUDED:
        p = os.path.join(G_DIR, name)
        VAL["excluded"].append({"excluded": name, "exists": os.path.exists(p)})

    gacc = load_general()
    tfiles, thalves, l35info, tvals = load_target()
    l35_absent_prefill, pre_row_sets, dec_row_sets, dec_row_sets_one_row, l35_decode_rows = l35info
    VAL["t_n_files"] = len(tfiles)
    VAL["t_l35_absent_prefill_row_sets"] = l35_absent_prefill
    VAL["t_prefill_row_sets"] = pre_row_sets
    VAL["t_decode_row_sets"] = dec_row_sets
    VAL["t_decode_row_sets_of_one_row"] = dec_row_sets_one_row
    VAL["t_l35_decode_rows"] = l35_decode_rows
    VAL["t_T_value_hist"] = {str(k): v for k, v in sorted(tvals.items())[:12]}
    VAL["t_T_value_hist_note"] = "capped at 12 entries; T=1 is decode, T>1 prefill"

    # ---------------- assemble populations
    # G pooled = A + B; halves = A set (derivation) and B set (held out)  -- the pre-existing split
    pops = {}
    for reg, rname in ((0, "prefill"), (1, "decode")):
        G = gacc[("A", reg)][0] + gacc[("B", reg)][0]
        Gw = gacc[("A", reg)][1] + gacc[("B", reg)][1]
        T = thalves["H1"][0][reg] + thalves["H2"][0][reg]
        pops[("G", rname)] = {"counts": G, "wmass": Gw,
                              "halves": {"A": gacc[("A", reg)][0], "B": gacc[("B", reg)][0]}}
        pops[("T", rname)] = {"counts": T,
                              "halves": {"H1": thalves["H1"][0][reg],
                                         "H2": thalves["H2"][0][reg]}}
    def share(counts):
        tot = counts.sum(axis=1, keepdims=True)
        return np.divide(counts, tot, out=np.zeros_like(counts), where=tot > 0)

    # A layer with ZERO rows in a cell is ABSENT, not cold: its share vector is a zero vector and
    # would otherwise score as the coldest possible band.  Undefined layers are excluded from every
    # statistic and named in the report.  (MEASURED: the target corpus has no L35 prefill rows in
    # any conversation - the known dumper artifact, which q1-census also records.)
    for k, p in pops.items():
        p["share_full"] = share(p["counts"])
        p["tokens_full"] = p["counts"].sum(axis=1) / KUSED
        p["defined"] = np.where(p["tokens_full"] > 0)[0]
        p["undefined_layers"] = [int(x) for x in np.where(p["tokens_full"] <= 0)[0]]
        p["n_defined"] = int(p["defined"].size)
        p["share"] = p["share_full"][p["defined"]]
        p["tokens"] = p["tokens_full"][p["defined"]]
        p["share_halves"], p["tokens_halves"], p["defined_halves"] = {}, {}, {}
        for h, c in p["halves"].items():
            tk = c.sum(axis=1) / KUSED
            df = np.where(tk > 0)[0]
            p["defined_halves"][h] = df
            p["share_halves"][h] = share(c)[df]
            p["tokens_halves"][h] = tk[df]
    res["adequacy"] = {
        "%s/%s" % k: {"median_tokens_per_layer": float(np.median(p["tokens"])),
                      "min_tokens_per_layer": float(p["tokens"].min()),
                      "n_layers_defined": p["n_defined"],
                      "undefined_layers": p["undefined_layers"],
                      "adequate": bool(np.median(p["tokens"]) >= ADEQ_TOKENS)}
        for k, p in pops.items()}

    # ---------------- window machinery
    def window_mass(share_mat, w):
        """-> (n_layers, 128-w+1) window masses."""
        cs = np.concatenate([np.zeros((share_mat.shape[0], 1)), np.cumsum(share_mat, axis=1)], axis=1)
        return cs[:, w:] - cs[:, :-w]

    def null_min_dist(share_mat, w, nperm, rng):
        """Distribution of min_lo S(lo) over per-layer independent relabellings (returned as an
        array), and the same for a FIXED window position, computed by the caller from the
        permuted matrices.  Here we return the permuted window-mass stacks' minima AND the mean
        permuted stack for the fixed-window test."""
        n = share_mat.shape[0]
        mins = np.empty(nperm)
        sds = np.empty(nperm)
        for i in range(nperm):
            perm = rng.random((n, NE)).argsort(axis=1)
            sp = np.take_along_axis(share_mat, perm, axis=1)
            Sp = window_mass(sp, w).sum(axis=0)
            mins[i] = Sp.min()
            sds[i] = Sp.std()      # spread of the window surface, a cross-layer-agreement diagnostic
        return mins, sds

    def null_fixed_window(share_mat, lo, w, nperm, rng):
        n = share_mat.shape[0]
        out = np.empty(nperm)
        for i in range(nperm):
            perm = rng.random((n, NE)).argsort(axis=1)
            sp = np.take_along_axis(share_mat, perm, axis=1)
            out[i] = window_mass(sp, w)[:, lo].sum()
        return out

    def per_layer_null_p(share_mat, w, nperm, rng):
        """Per-layer test of "this layer HAS a cold contiguous window": the observed per-layer minimum
        window mass against the same minimum under relabelling.  NOTE the null minimum is NOT ~uniform
        -- it is the min of 125 windows of this layer's own count vector, so it already carries the
        selection effect.  That is the whole point: a layer's coldest window is only evidence of
        structure if it beats ITS OWN min-of-125 null."""
        n = share_mat.shape[0]
        obs = window_mass(share_mat, w).min(axis=1)
        cnt = np.zeros(n)
        null_sum = np.zeros(n)
        null_q05 = np.empty((nperm, n))
        for i in range(nperm):
            perm = rng.random((n, NE)).argsort(axis=1)
            sp = np.take_along_axis(share_mat, perm, axis=1)
            mn = window_mass(sp, w).min(axis=1)
            cnt += (mn <= obs)
            null_sum += mn
            null_q05[i] = mn
        return obs, (1.0 + cnt) / (nperm + 1.0), null_sum / nperm, np.percentile(null_q05, 5, axis=0)

    rng = np.random.default_rng(SEED)

    # ---------------- 1. per-layer coldest windows, distinctness
    res["per_layer"] = {}
    for key, p in pops.items():
        tag = "%s/%s" % key
        per = {}
        for w in WIDTHS:
            wm = window_mass(p["share"], w)
            best = wm.argmin(axis=1)
            los = sorted(set(int(x) for x in best))
            per["w%d" % w] = {
                "distinct_windows": len(los), "chosen_lo_hist": {str(k): int((best == k).sum())
                                                                 for k in los},
                "n_windows_available": int(128 - w + 1),
                "mean_layer_min_mass": float(wm.min(axis=1).mean()),
                "uniform_window_share": w / 128.0,
            }
        res["per_layer"][tag] = per

    # ---------------- 2. best global band + 3. permutation null
    res["global_band"] = {}
    for key, p in pops.items():
        tag = "%s/%s" % key
        gb = {}
        for w in WIDTHS:
            wm = window_mass(p["share"], w)
            S = wm.sum(axis=0)                     # (128-w+1,)
            lo = int(S.argmin())
            masses = wm[:, lo]
            uni = w / 128.0
            entry = {"lo": lo, "hi": lo + w,
                     "sum_S": float(S[lo]), "mean_layer_mass": float(masses.mean()),
                     "ratio_to_uniform": float(masses.mean() / uni),
                     "layers_exact_coldest": int((wm.argmin(axis=1) == lo).sum()),
                     "layers_hotter_than_uniform": int((masses > uni * 1.0000001).sum()),
                     "layers_above_half_uniform": int((masses > COLD_FRAC * uni).sum()),
                     "layers_hotter_than_uniform_ids": [int(x) for x in np.where(masses > uni * 1.0000001)[0]],
                     "layers_above_half_uniform_ids": [int(x) for x in np.where(masses > COLD_FRAC * uni)[0]],
                     "per_layer_mass": [float(x) for x in masses],
                     "rank_of_S_among_windows": int((S < S[lo]).sum()) + 1,
                     "n_windows": int(S.size),
                     # context: the SAME statistic across all candidate windows, and the hottest one
                     "S_min": float(S.min()), "S_median": float(np.median(S)),
                     "S_max": float(S.max()), "S_mean": float(S.mean()),
                     "frac_windows_below_uniform": float((S < uni * p["share"].shape[0]).mean()),
                     "hottest_lo": int(S.argmax()),
                     "hottest_mean_layer_mass": float(wm[:, int(S.argmax())].mean()),
                     "hottest_ratio_to_uniform": float(wm[:, int(S.argmax())].mean() / uni)}
            gb["w%d" % w] = entry
        # nulls at w=4 (the rule's width)
        wm4 = window_mass(p["share"], 4)
        S4 = wm4.sum(axis=0)
        lo4 = int(S4.argmin())
        mins, sds = null_min_dist(p["share"], 4, NPERM, rng)
        fixed = null_fixed_window(p["share"], lo4, 4, NPERM, rng)
        S_obs = float(S4[lo4])
        S_fix = float(S4[lo4])
        gb["w4_null"] = {
            "observed_min_S": S_obs,
            "null_min_S_mean": float(mins.mean()), "null_min_S_sd": float(mins.std()),
            "p_min_over_windows": float((1.0 + (mins <= S_obs).sum()) / (NPERM + 1.0)),
            "z_min_over_windows": float((S_obs - mins.mean()) / (mins.std() + 1e-12)),
            "null_fixed_window_mean": float(fixed.mean()),
            "p_candidate_window": float((1.0 + (fixed <= S_fix).sum()) / (NPERM + 1.0)),
            "z_candidate_window": float((S_fix - fixed.mean()) / (fixed.std() + 1e-12)),
            # A cold BAND common to all layers would make the observed window surface LUMPIER than
            # the relabelled null (cross-layer agreement inflates var(S)); a flat surface is the
            # signature of "no window is special".
            "observed_S_sd": float(S4.std()),
            "null_S_sd_mean": float(sds.mean()),
            "ratio_observed_null_S_sd": float(S4.std() / (sds.mean() + 1e-12)),
            "nperm": NPERM, "seed": SEED,
        }
        obs_l, p_l, nl_mean, nl_q05 = per_layer_null_p(p["share"], 4, NPERM, rng)
        gb["w4_per_layer_null"] = {
            "n_layers_p_lt_0.05": int((p_l < 0.05).sum()),
            "n_layers_p_lt_bonferroni": int((p_l < 0.05 / p["share"].shape[0]).sum()),
            "min_p": float(p_l.min()),
            "median_layer_min_mass": float(np.median(obs_l)),
            "mean_layer_min_mass": float(obs_l.mean()),
            "null_mean_per_layer_min_mass": float(nl_mean.mean()),
            "ratio_observed_vs_null": float(obs_l.mean() / (nl_mean.mean() + 1e-12)),
            "n_layers_below_null_mean": int((obs_l < nl_mean).sum()),
            "n_layers_below_null_p05": int((obs_l < nl_q05).sum()),
            "per_layer_p": [float(x) for x in p_l],
        }
        res["global_band"][tag] = gb

    # ---------------- 4. stability
    def rank_1d(x):
        """Average ranks (ties get their mean rank) — vectorised, Python-loop free."""
        uniq, inv = np.unique(x, return_inverse=True)
        counts = np.bincount(inv)
        return (np.cumsum(counts) - counts)[inv] + (counts[inv] - 1) / 2.0

    def ranks_rows(mat):
        return np.stack([rank_1d(row) for row in mat])

    def spearman(a, b):
        ra, rb = rank_1d(a), rank_1d(b)
        ra = ra - ra.mean(); rb = rb - rb.mean()
        d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
        return float((ra * rb).sum() / d) if d > 0 else 0.0

    def layer_rhos(s1, s2):
        r1, r2 = ranks_rows(s1), ranks_rows(s2)
        r1 = r1 - r1.mean(axis=1, keepdims=True)
        r2 = r2 - r2.mean(axis=1, keepdims=True)
        num = (r1 * r2).sum(axis=1)
        den = np.sqrt((r1 * r1).sum(axis=1) * (r2 * r2).sum(axis=1))
        return np.divide(num, den, out=np.zeros_like(num), where=den > 0)

    def spec_stats(s1, s2):
        per = layer_rhos(s1, s2)
        return {"median_layer_rho": float(np.median(per)), "mean_layer_rho": float(per.mean()),
                "pooled_rho": spearman(s1.ravel(), s2.ravel()),
                "per_layer": [float(x) for x in per]}

    def spec_null(s1, s2, nperm, rng):
        out = np.empty(nperm)
        for i in range(nperm):
            perm = rng.random((s2.shape[0], NE)).argsort(axis=1)
            s2p = np.take_along_axis(s2, perm, axis=1)
            out[i] = float(np.median(layer_rhos(s1, s2p)))
        return out

    res["stability"] = {}
    for key, p in pops.items():
        tag = "%s/%s" % key
        hs = sorted(p["share_halves"])
        both = np.intersect1d(p["defined_halves"][hs[0]], p["defined_halves"][hs[1]])
        d0, d1 = p["defined_halves"][hs[0]], p["defined_halves"][hs[1]]
        s1 = p["share_halves"][hs[0]][np.isin(d0, both)]
        s2 = p["share_halves"][hs[1]][np.isin(d1, both)]
        st = {"halves": hs, "n_layers_compared": int(both.size),
              "layer_ids_compared": [int(x) for x in both],
              "half_tokens_median": {h: float(np.median(p["tokens_halves"][h])) for h in hs},
              "spearman": spec_stats(s1, s2)}
        nl = spec_null(s1, s2, NPERM_SPEC, rng)
        st["spearman_null"] = {"mean": float(nl.mean()), "sd": float(nl.std()),
                               "p95": float(np.percentile(nl, 95)),
                               "p_value": float((1.0 + (nl >= st["spearman"]["median_layer_rho"]).sum())
                                                / (NPERM_SPEC + 1.0))}
        # exact-window agreement at w=4, per layer and file-wide
        wm1, wm2 = window_mass(s1, 4), window_mass(s2, 4)
        b1, b2 = wm1.argmin(axis=1), wm2.argmin(axis=1)
        agree = int((b1 == b2).sum())
        g1 = int(wm1.sum(axis=0).argmin()); g2 = int(wm2.sum(axis=0).argmin())
        st["window_agreement_w4"] = {
            "layers_agreeing": agree, "of_layers": int(s1.shape[0]),
            "filewide_lo": [g1, g2], "filewide_agree": bool(g1 == g2),
            "distinct_windows_half1": len(set(int(x) for x in b1)),
            "distinct_windows_half2": len(set(int(x) for x in b2)),
        }
        # null for the agreement count: relabel half 2 independently per layer
        cnt = np.empty(NPERM_SPEC, int)
        for i in range(NPERM_SPEC):
            perm = rng.random((s2.shape[0], NE)).argsort(axis=1)
            s2p = np.take_along_axis(s2, perm, axis=1)
            cnt[i] = int((b1 == window_mass(s2p, 4).argmin(axis=1)).sum())
        st["window_agreement_null"] = {"mean": float(cnt.mean()), "sd": float(cnt.std()),
                                       "p95": float(np.percentile(cnt, 95)),
                                       "p_value": float((1.0 + (cnt >= agree).sum())
                                                        / (NPERM_SPEC + 1.0))}
        res["stability"][tag] = st

    # cross-population, same regime
    res["cross_population"] = {}
    for rname in ("prefill", "decode"):
        # Spearman / per-layer agreement: aligned on the layers BOTH populations define.
        common = np.intersect1d(pops[("G", rname)]["defined"], pops[("T", rname)]["defined"])
        sg = pops[("G", rname)]["share_full"][common]
        stt = pops[("T", rname)]["share_full"][common]
        # Band APPLICATION uses each population's own whole cell, so "the band chosen on G" really is
        # G's own argmin (restricting to the layer intersection can move it).
        sg_own = pops[("G", rname)]["share"]
        st_own = pops[("T", rname)]["share"]
        cs = {"n_layers_compared": int(common.size),
              "layer_ids_compared": [int(x) for x in common],
              "spearman": spec_stats(sg, stt)}
        nl = spec_null(sg, stt, NPERM_SPEC, rng)
        cs["spearman_null"] = {"mean": float(nl.mean()), "sd": float(nl.std()),
                               "p95": float(np.percentile(nl, 95)),
                               "p_value": float((1.0 + (nl >= cs["spearman"]["median_layer_rho"]).sum())
                                                / (NPERM_SPEC + 1.0))}
        wg_own, wt_own = window_mass(sg_own, 4), window_mass(st_own, 4)
        lo_g, lo_t = int(wg_own.sum(axis=0).argmin()), int(wt_own.sum(axis=0).argmin())
        wg, wt = window_mass(sg, 4), window_mass(stt, 4)   # common-layer matrices, for agreement
        for tag, lo, wm_target in (("G_band_on_T", lo_g, wt_own), ("T_band_on_G", lo_t, wg_own)):
            masses = wm_target[:, lo]
            cs[tag] = {"lo": lo, "mean_layer_mass": float(masses.mean()),
                       "ratio_to_uniform": float(masses.mean() / (4 / 128.0)),
                       "rank_among_windows": int((wm_target.sum(axis=0) < wm_target.sum(axis=0)[lo]).sum()) + 1,
                       "layers_exact_coldest": int((wm_target.argmin(axis=1) == lo).sum()),
                       "layers_hotter_than_uniform": int((masses > 4 / 128.0 * 1.0000001).sum()),
                       "layers_above_half_uniform": int((masses > COLD_FRAC * 4 / 128.0).sum())}
        cs["agreement"] = {"G_lo": lo_g, "T_lo": lo_t, "same_window": bool(lo_g == lo_t),
                           "G_lo_on_common_layers": int(wg.sum(axis=0).argmin()),
                           "T_lo_on_common_layers": int(wt.sum(axis=0).argmin()),
                           "distinct_windows_G": len(set(int(x) for x in wg.argmin(axis=1))),
                           "distinct_windows_T": len(set(int(x) for x in wt.argmin(axis=1)))}
        cnt = np.empty(NPERM_SPEC, int)
        bG, bT = wg.argmin(axis=1), wt.argmin(axis=1)
        for i in range(NPERM_SPEC):
            perm = rng.random((stt.shape[0], NE)).argsort(axis=1)
            stp = np.take_along_axis(stt, perm, axis=1)
            cnt[i] = int((bG == window_mass(stp, 4).argmin(axis=1)).sum())
        cs["agreement_null"] = {"mean": float(cnt.mean()), "sd": float(cnt.std()),
                                "p95": float(np.percentile(cnt, 95)),
                                "p_value": float((1.0 + (cnt >= int((bG == bT).sum())).sum())
                                                 / (NPERM_SPEC + 1.0))}
        cs["layers_agreeing"] = int((bG == bT).sum())
        res["cross_population"][rname] = cs

    # ---------------- secondary: aubier WEIGHT MASS (clearly separate from selection counts)
    res["secondary_weight_mass"] = {}
    for reg, rname in ((0, "prefill"), (1, "decode")):
        Gw = gacc[("A", reg)][1] + gacc[("B", reg)][1]
        tot = Gw.sum(axis=1, keepdims=True)
        sw = np.divide(Gw, tot, out=np.zeros_like(Gw), where=tot > 0)
        gsh = pops[("G", rname)]["share"]
        gdef = pops[("G", rname)]["defined"]
        sww = sw[gdef]
        wm = window_mass(sww, 4)
        lo = int(wm.sum(axis=0).argmin())
        lo_sel = int(window_mass(gsh, 4).sum(axis=0).argmin())
        res["secondary_weight_mass"][rname] = {
            "band_by_weight_lo": lo,
            "band_by_selection_lo": lo_sel,
            "same_window": bool(lo == lo_sel),
            "weight_mean_layer_mass": float(wm[:, lo].mean()),
            "weight_ratio_to_uniform": float(wm[:, lo].mean() / (4 / 128.0)),
            "weight_mean_layer_mass_at_selection_lo": float(wm[:, lo_sel].mean()),
            "weight_ratio_at_selection_lo": float(wm[:, lo_sel].mean() / (4 / 128.0)),
            "median_layer_spearman_selection_vs_weight":
                float(np.median(layer_rhos(gsh, sww))),
        }

    # ---------------- 5. the pre-declared rule, applied mechanically
    def evaluate(pop_key, regime, w=4, candidate=None):
        p = pops[(pop_key, regime)]
        wm = window_mass(p["share"], w)
        if candidate is None:
            candidate = int(wm.sum(axis=0).argmin())
        masses = wm[:, candidate]
        uni = w / 128.0
        lay_min = wm.min(axis=1)
        near = masses <= NEAR * lay_min
        near = np.where(lay_min == 0, masses == 0, near)
        mean_mass = float(masses.mean())
        i_ok = mean_mass <= COLD_FRAC * uni
        ii_ok = int(near.sum()) >= MIN_LAYERS_NEAR
        # (iv): p for the candidate window
        fixed = null_fixed_window(p["share"], candidate, w, NPERM, rng)
        p_cand = float((1.0 + (fixed <= float(masses.sum())).sum()) / (NPERM + 1.0))
        iv_ok = p_cand < P_MAX
        return {"lo": int(candidate), "hi": int(candidate + w),
                "mean_layer_mass": mean_mass, "half_uniform": COLD_FRAC * uni,
                "ratio_to_uniform": mean_mass / uni,
                "layers_near_coldest": int(near.sum()), "layers_needed": MIN_LAYERS_NEAR,
                "n_layers": int(p["share"].shape[0]),
                "layers_exact_coldest": int((wm.argmin(axis=1) == candidate).sum()),
                "layers_hotter_than_uniform": int((masses > uni * 1.0000001).sum()),
                "layers_above_half_uniform": int((masses > COLD_FRAC * uni).sum()),
                "p_candidate": p_cand, "z_candidate": float(
                    (masses.sum() - fixed.mean()) / (fixed.std() + 1e-12)),
                "i_materially_cold": bool(i_ok), "ii_near_cold_30of36": bool(ii_ok),
                "iv_outside_null": bool(iv_ok)}

    res["rule"] = {}
    cand_lo = {}
    for rname in ("prefill", "decode"):
        cand_lo[rname] = int(window_mass(pops[("G", rname)]["share"], 4).sum(axis=0).argmin())
    for rname in ("prefill", "decode"):
        ad_G = res["adequacy"]["G/%s" % rname]["adequate"]
        ad_T = res["adequacy"]["T/%s" % rname]["adequate"]
        lo = cand_lo[rname]
        gg = evaluate("G", rname, 4, lo)
        tt = evaluate("T", rname, 4, lo)
        # (iii) each half of each population
        halves = {}
        ok_all = True
        for pop_key in ("G", "T"):
            pp = pops[(pop_key, rname)]
            for h, sh in sorted(pp["share_halves"].items()):
                wm = window_mass(sh, 4)
                masses = wm[:, lo]
                uni = 4 / 128.0
                lay_min = wm.min(axis=1)
                near = masses <= NEAR * lay_min
                near = np.where(lay_min == 0, masses == 0, near)
                h_i = float(masses.mean()) <= COLD_FRAC * uni
                h_ii = int(near.sum()) >= MIN_LAYERS_NEAR
                halves["%s/%s/%s" % (pop_key, rname, h)] = {
                    "mean_layer_mass": float(masses.mean()),
                    "ratio_to_uniform": float(masses.mean() / uni),
                    "layers_near_coldest": int(near.sum()),
                    "n_layers": int(sh.shape[0]),
                    "i_ok": bool(h_i), "ii_ok": bool(h_ii), "ok": bool(h_i and h_ii)}
                ok_all = ok_all and h_i and h_ii
        entry = {"candidate_lo": lo, "candidate_hi": lo + 4,
                 "chosen_on": "G/general-sample", "cells_adequate": {"G": ad_G, "T": ad_T},
                 "G": gg, "T": tt, "halves": halves,
                 "iii_reproduced_on_all_halves": bool(ok_all)}
        if not (ad_G and ad_T):
            verdict = "INSUFFICIENT"
        elif gg["i_materially_cold"] and gg["ii_near_cold_30of36"] and ok_all and \
                gg["iv_outside_null"] and tt["i_materially_cold"] and tt["ii_near_cold_30of36"] and \
                tt["iv_outside_null"]:
            verdict = "STABLE BAND"
        else:
            verdict = "NO STABLE BAND"
        entry["verdict"] = verdict
        res["rule"][rname] = entry

    # mirror direction, descriptive only
    res["rule_mirror_chosen_on_T"] = {}
    for rname in ("prefill", "decode"):
        lo = int(window_mass(pops[("T", rname)]["share"], 4).sum(axis=0).argmin())
        res["rule_mirror_chosen_on_T"][rname] = {"lo": lo,
                                                 "G": evaluate("G", rname, 4, lo),
                                                 "T": evaluate("T", rname, 4, lo)}

    vs = [e["verdict"] for e in res["rule"].values()]
    if all(v == "STABLE BAND" for v in vs):
        overall = "STABLE BAND"
    elif any(v == "STABLE BAND" for v in vs) and all(v in ("STABLE BAND", "INSUFFICIENT") for v in vs):
        overall = "STABLE BAND (single regime; the other regime is INSUFFICIENT)"
    elif all(v == "INSUFFICIENT" for v in vs):
        overall = "INSUFFICIENT"
    else:
        overall = "NO STABLE BAND"
    res["overall_verdict"] = overall
    res["validation"] = VAL

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "cold-band-results.json"), "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)

    # ---------------- human-readable report
    def p(*a):
        print(*a)

    p("=== VALIDATION ===")
    p("general sample: %d traces used; excluded %d" % (len(VAL["g_files"]), len(EXCLUDED)))
    for n, v in sorted(VAL["g_files"].items()):
        p("  %-9s set=%s n_prompt=%3d n_decode=%3d rows=%5d maxid=%3d layers pre/de=%d/%d pos pre/de=%d/%d"
          % (n, v["set"], v["prompt_tokens"], v["decode_tokens"], v["rows"], v["max_expert_id"],
             v["layers_prefill"], v["layers_decode"], v["prefill_positions"], v["decode_positions"]))
    for e in VAL["excluded"]:
        p("  excluded %s" % e)
    p("target corpus: %d conversations; prefill row-sets=%d of which lacking the L35 row=%d; "
      "decode row-sets=%d (of one row, i.e. the lone L35 row=%d)"
      % (VAL["t_n_files"], VAL["t_prefill_row_sets"], VAL["t_l35_absent_prefill_row_sets"],
         VAL["t_decode_row_sets"], VAL["t_decode_row_sets_of_one_row"]))
    p("problems: %d %s" % (len(VAL["problems"]), VAL["problems"][:5]))

    p("\n=== ADEQUACY (median tokens per layer, over DEFINED layers) ===")
    for k, v in sorted(res["adequacy"].items()):
        p("  %-12s median=%8.1f min=%6.1f defined=%2d undefined_layers=%s adequate=%s"
          % (k, v["median_tokens_per_layer"], v["min_tokens_per_layer"],
             v["n_layers_defined"], v["undefined_layers"], v["adequate"]))
    p("\n=== 1. PER-LAYER COLDEST WINDOWS (distinct windows chosen by the 36 layers) ===")
    for tag in sorted(res["per_layer"]):
        for w in ("w3", "w4", "w5", "w8"):
            e = res["per_layer"][tag][w]
            n = pops[tuple(tag.split("/"))]["n_defined"]
            p("  %-12s %s distinct=%3d/%3d of %d layers mean_layer_min=%.5f (uniform %.5f)"
              % (tag, w, e["distinct_windows"], e["n_windows_available"], n,
                 e["mean_layer_min_mass"], e["uniform_window_share"]))
    p("\n=== 2/3. GLOBAL BAND + NULL ===")
    for tag in sorted(res["global_band"]):
        for w in ("w3", "w4", "w5", "w8"):
            e = res["global_band"][tag][w]
            p("  %-12s %s lo=%3d mean_mass=%.5f ratio=%.3f exact_coldest=%2d hotter_than_uniform=%2d >half=%2d rank=%d/%d"
              % (tag, w, e["lo"], e["mean_layer_mass"], e["ratio_to_uniform"],
                 e["layers_exact_coldest"], e["layers_hotter_than_uniform"],
                 e["layers_above_half_uniform"], e["rank_of_S_among_windows"], e["n_windows"]))
            p("     windows below uniform: %d/%d (%.0f%%); S_min/median/max = %.4f/%.4f/%.4f; hottest window lo=%d ratio=%.3f"
              % (int(round(e["frac_windows_below_uniform"] * e["n_windows"])), e["n_windows"],
                 100 * e["frac_windows_below_uniform"], e["S_min"], e["S_median"], e["S_max"],
                 e["hottest_lo"], e["hottest_ratio_to_uniform"]))
        n4 = res["global_band"][tag]["w4_null"]
        pl = res["global_band"][tag]["w4_per_layer_null"]
        p("    window-surface spread: observed sd(S)=%.4f vs null sd=%.4f (ratio %.3f)"
          % (n4["observed_S_sd"], n4["null_S_sd_mean"], n4["ratio_observed_null_S_sd"]))
        p("    null(w4): p_min=%.4f p_candidate=%.4f z_min=%.2f z_cand=%.2f; per-layer p<0.05: %d, bonferroni: %d"
          % (n4["p_min_over_windows"], n4["p_candidate_window"], n4["z_min_over_windows"],
             n4["z_candidate_window"], pl["n_layers_p_lt_0.05"],
             pl["n_layers_p_lt_bonferroni"]))
        p("    per-layer null(w4): observed per-layer min mean=%.5f vs null mean=%.5f (ratio %.3f); layers below null mean=%d, below null p05=%d"
          % (pl["mean_layer_min_mass"], pl["null_mean_per_layer_min_mass"],
             pl["ratio_observed_vs_null"], pl["n_layers_below_null_mean"],
             pl["n_layers_below_null_p05"]))
    p("\n=== 4. STABILITY ===")
    for tag in sorted(res["stability"]):
        s = res["stability"][tag]
        p("  %-12s halves=%s n=%d median_rho=%.4f pooled_rho=%.4f null_mean=%.4f p=%.4f | window agree %d/%d filewide %s p=%.4f"
          % (tag, s["halves"], s["n_layers_compared"], s["spearman"]["median_layer_rho"],
             s["spearman"]["pooled_rho"], s["spearman_null"]["mean"], s["spearman_null"]["p_value"],
             s["window_agreement_w4"]["layers_agreeing"], s["window_agreement_w4"]["of_layers"],
             s["window_agreement_w4"]["filewide_lo"], s["window_agreement_null"]["p_value"]))
    for rname in ("prefill", "decode"):
        c = res["cross_population"][rname]
        p("  cross %-8s n=%d median_rho=%.4f pooled_rho=%.4f null_mean=%.4f p=%.4f | G_lo=%d T_lo=%d layers_agree=%d p=%.4f"
          % (rname, c["n_layers_compared"], c["spearman"]["median_layer_rho"], c["spearman"]["pooled_rho"],
             c["spearman_null"]["mean"], c["spearman_null"]["p_value"],
             c["agreement"]["G_lo"], c["agreement"]["T_lo"], c["layers_agreeing"],
             c["agreement_null"]["p_value"]))
        for t in ("G_band_on_T", "T_band_on_G"):
            e = c[t]
            p("     %s lo=%d mean_mass=%.5f ratio=%.3f exact_coldest=%d hotter=%d >half=%d rank=%d"
              % (t, e["lo"], e["mean_layer_mass"], e["ratio_to_uniform"],
                 e["layers_exact_coldest"], e["layers_hotter_than_uniform"],
                 e["layers_above_half_uniform"], e["rank_among_windows"]))
    p("\n=== SECONDARY: aubier WEIGHT MASS (never mixed with selection counts) ===")
    for rname, e in sorted(res["secondary_weight_mass"].items()):
        p("  %-8s weight_band=%s selection_band=%s same=%s weight_ratio=%.3f weight_ratio_at_sel_lo=%.3f median_rho(sel,w)=%.3f"
          % (rname, e["band_by_weight_lo"], e["band_by_selection_lo"], e["same_window"],
             e["weight_ratio_to_uniform"], e["weight_ratio_at_selection_lo"],
             e["median_layer_spearman_selection_vs_weight"]))
    p("\n=== 5. THE PRE-DECLARED RULE ===")
    for rname in ("prefill", "decode"):
        e = res["rule"][rname]
        p("  %-8s candidate=[%d,%d) chosen_on=%s" % (rname, e["candidate_lo"], e["candidate_hi"],
                                                     e["chosen_on"]))
        for pop_key in ("G", "T"):
            g = e[pop_key]
            nd = pops[(pop_key, rname)]["n_defined"]
            p("    %s mean_mass=%.5f (%.3fx uniform, half=%.5f) near=%d/%d exact=%d hotter=%d >half=%d p=%.4f  i=%s ii=%s iv=%s"
              % (pop_key, g["mean_layer_mass"], g["ratio_to_uniform"], g["half_uniform"],
                 g["layers_near_coldest"], nd, g["layers_exact_coldest"],
                 g["layers_hotter_than_uniform"], g["layers_above_half_uniform"],
                 g["p_candidate"], g["i_materially_cold"], g["ii_near_cold_30of36"],
                 g["iv_outside_null"]))
        for h, hv in sorted(e["halves"].items()):
            p("    %-16s mean=%.5f ratio=%.3f near=%d/%d i=%s ii=%s"
              % (h, hv["mean_layer_mass"], hv["ratio_to_uniform"], hv["layers_near_coldest"],
                 hv["n_layers"], hv["i_ok"], hv["ii_ok"]))
        p("    (iii) all halves reproduced=%s" % e["iii_reproduced_on_all_halves"])
        p("    VERDICT(%s) = %s" % (rname, e["verdict"]))
    p("\n=== MIRROR (band chosen on the TARGET corpus), descriptive ===")
    for rname, e in sorted(res["rule_mirror_chosen_on_T"].items()):
        p("  %-8s lo=%d G: ratio=%.3f near=%d p=%.4f | T: ratio=%.3f near=%d p=%.4f"
          % (rname, e["lo"], e["G"]["ratio_to_uniform"], e["G"]["layers_near_coldest"],
             e["G"]["p_candidate"], e["T"]["ratio_to_uniform"], e["T"]["layers_near_coldest"],
             e["T"]["p_candidate"]))
    p("\nOVERALL = %s" % res["overall_verdict"])
    return res


if __name__ == "__main__":
    main()
