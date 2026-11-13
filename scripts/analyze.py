"""Summarise the raw measurements and draw the figures used in the report.

Inputs : results/raw/measurements.csv, results/raw/runs.csv
Outputs: results/summary/summary.csv   median, quartiles, IQR, memory, threads per configuration
         results/summary/speedup.csv   S(p) = Tseq/Tp and E(p) = S(p)/p (medians), with the
                                       quartile-based range of S, for every (method, n, p)
         results/summary/limits.csv    largest completed size per configuration
         results/summary/tables.tex    LaTeX tables included by the report
         results/figures/*.png         figures

Speed-up uses the median time of the equivalent sequential kernel (blocked_seq for
blocked_omp, ikj_seq for ikj_omp) on the same input, build and placement policy.

    python scripts/analyze.py [--results results]
"""
import argparse
import csv
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FAMILIES = {"blocked": ("blocked_seq", "blocked_omp"), "ikj": ("ikj_seq", "ikj_omp")}
FAMILY_NAME = {"blocked": "blocked (tiled ikj, b = 64)", "ikj": "ikj (loop reorder)"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# Sequential palette for matrix sizes (light -> dark) and categorical colours.
SIZE_COLORS = ["#9ec5f4", "#6aa5ec", "#3987e0", "#2a78d6", "#1d5fb0", "#15467f", "#0c2d52"]
CAT = {"none": "#8a8984", "compact": "#eb6834", "spread": "#2a78d6",
       "novec": "#8a8984", "sse2": "#2a78d6", "avx2": "#4a3aa7",
       "blocked": "#4a3aa7", "ikj": "#eb6834"}
PHYS, LOGICAL = 8, 16


MIN_N_SCALING_FIG = 1024


def style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.set_title(title, color=INK, loc="left", fontsize=11)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)


def thread_axis(ax, threads):
    ax.set_xscale("log", base=2)
    ax.set_xticks(threads)
    ax.set_xticklabels([str(p) for p in threads])
    ax.minorticks_off()
    ax.axvspan(PHYS, LOGICAL, color="#f3f2ee", zorder=0)
    ax.axvspan(LOGICAL, max(threads) * 1.15, color="#fbe9e1", zorder=0)


def fmt_t(x):
    if x >= 1:
        return f"{x:.2f} s"
    if x >= 1e-3:
        return f"{x * 1e3:.1f} ms"
    return f"{x * 1e6:.0f} $\\mu$s"


def stats(values):
    t = np.array(values, dtype=float)
    q1, med, q3 = np.percentile(t, [25, 50, 75])
    return {"reps": len(t), "median_s": med, "q1_s": q1, "q3_s": q3,
            "rel_iqr_pct": 100 * (q3 - q1) / med if med > 0 else 0.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    res = args.results
    sum_dir, fig_dir = os.path.join(res, "summary"), os.path.join(res, "figures")
    os.makedirs(sum_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    plt.rcParams.update({"font.size": 9.5, "figure.dpi": 150, "savefig.bbox": "tight"})

    meas = list(csv.DictReader(open(os.path.join(res, "raw", "measurements.csv"))))
    runs = list(csv.DictReader(open(os.path.join(res, "raw", "runs.csv"))))
    keyf = ("experiment", "method", "build", "n", "threads", "affinity")

    def key_of(r):
        return (r["experiment"], r["method"], r["build"], int(r["n"]), int(r["threads"]),
                r["affinity"])

    run_by = {key_of(r): r for r in runs}

    # ---------- per-configuration summary ----------
    groups, inner = {}, {}
    for r in meas:
        if r["phase"] != "measure":
            continue
        groups.setdefault(key_of(r), []).append(float(r["time_per_mult_s"]))
        inner[key_of(r)] = int(r["inner"])
    S = {}
    for key, vals in groups.items():
        run = run_by[key]
        if run["status"] != "ok":
            continue
        s = stats(vals)
        s.update(dict(zip(keyf, key)))
        s.update({"inner": inner[key], "dtype": "float64", "representation": "dense row-major",
                  "peak_working_set_mb": int(run["peak_working_set_bytes"]) / 1e6,
                  "peak_private_mb": int(run["peak_private_bytes"]) / 1e6,
                  "peak_os_threads": int(run["peak_os_threads"]),
                  "team_size_in_kernel": int(run["team_size_in_kernel"]),
                  "oversubscribed": run["oversubscribed"],
                  "distinct_cores": run["distinct_cores"], "distinct_lp": run["distinct_lp"],
                  "warmup_runs": run["warmup_runs"]})
        S[key] = s
    fields = [*keyf, "representation", "dtype", "reps", "warmup_runs", "inner", "median_s",
              "q1_s", "q3_s", "rel_iqr_pct", "peak_working_set_mb", "peak_private_mb",
              "peak_os_threads", "team_size_in_kernel", "distinct_lp", "distinct_cores",
              "oversubscribed"]
    with open(os.path.join(sum_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in sorted(S.values(), key=lambda s: tuple(s[k] for k in keyf)):
            w.writerow({k: (f"{s[k]:.6g}" if isinstance(s[k], float) else s[k]) for k in fields})

    def get(exp, m, n, p=1, aff="spread", build="sse2"):
        return S.get((exp, m, build, n, p, aff))

    # ---------- limits ----------
    with open(os.path.join(sum_dir, "limits.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "method", "build", "threads", "affinity", "largest_ok_n",
                    "largest_attempted_n", "failures"])
        for k in sorted({(r["experiment"], r["method"], r["build"], int(r["threads"]),
                          r["affinity"]) for r in runs}):
            mine = [r for r in runs if (r["experiment"], r["method"], r["build"],
                                        int(r["threads"]), r["affinity"]) == k]
            ok = [int(r["n"]) for r in mine if r["status"] == "ok"]
            att = [int(r["n"]) for r in mine if r["status"] != "skipped"]
            fail = "; ".join(f"n={r['n']}: {r['status']} ({r['note']})" for r in mine
                             if r["status"] != "ok")
            w.writerow([*k, max(ok) if ok else "", max(att) if att else "", fail or "none"])

    tex = ["% Generated by scripts/analyze.py -- do not edit by hand."]

    # =====================================================================
    # Scaling: speed-up and efficiency
    # =====================================================================
    sc = [s for s in S.values() if s["experiment"] == "scaling"]
    threads = sorted({s["threads"] for s in sc if "_omp" in s["method"]})
    sp_rows = []
    for fam, (seq, par) in FAMILIES.items():
        for n in sorted({s["n"] for s in sc if s["method"] == seq}):
            base = get("scaling", seq, n)
            one = get("scaling", par, n, 1)
            if not base:
                continue
            for p in threads:
                t = get("scaling", par, n, p)
                if not t:
                    continue
                sp = base["median_s"] / t["median_s"]
                sp_rows.append({
                    "family": fam, "method": par, "baseline": seq, "n": n, "threads": p,
                    "tseq_median_s": base["median_s"], "tp_median_s": t["median_s"],
                    "tp_rel_iqr_pct": t["rel_iqr_pct"], "speedup": sp, "efficiency": sp / p,
                    "speedup_low": base["q1_s"] / t["q3_s"], "speedup_high": base["q3_s"] / t["q1_s"],
                    "speedup_vs_omp1": one["median_s"] / t["median_s"] if one else "",
                    "gflops": 2 * n ** 3 / t["median_s"] / 1e9,
                    "oversubscribed": p > LOGICAL})
    with open(os.path.join(sum_dir, "speedup.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sp_rows[0].keys()))
        w.writeheader()
        for r in sp_rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})

    def sprow(fam, n, p):
        for r in sp_rows:
            if r["family"] == fam and r["n"] == n and r["threads"] == p:
                return r
        return None

    sizes_fam = {fam: sorted({r["n"] for r in sp_rows if r["family"] == fam}) for fam in FAMILIES}
    all_sizes = sorted({n for v in sizes_fam.values() for n in v})
    size_color = {n: SIZE_COLORS[i % len(SIZE_COLORS)] for i, n in enumerate(all_sizes)}

    # Figure: speed-up and efficiency vs threads, one panel per kernel family
    for metric, fname, ylab, ttl in (
            ("speedup", "fig_speedup.png", "speed-up S(p) = Tseq / Tp", "Speed-up"),
            ("efficiency", "fig_efficiency.png", "efficiency E(p) = S(p) / p", "Efficiency")):
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), sharey=True)
        for ax, fam in zip(axes, FAMILIES):
            for n in [n for n in sizes_fam[fam] if n >= MIN_N_SCALING_FIG]:
                rows = [sprow(fam, n, p) for p in threads if sprow(fam, n, p)]
                x = [r["threads"] for r in rows]
                y = [r[metric] for r in rows]
                if metric == "speedup":
                    lo = [r["speedup_low"] for r in rows]
                    hi = [r["speedup_high"] for r in rows]
                    ax.fill_between(x, lo, hi, color=size_color[n], alpha=0.15, linewidth=0)
                ax.plot(x, y, marker="o", markersize=3.5, linewidth=1.8, color=size_color[n],
                        label=f"n = {n}")
            if metric == "speedup":
                ax.plot([1, LOGICAL], [1, LOGICAL], linestyle="--", color=INK2, linewidth=1,
                        label="ideal S = p")
                ax.axhline(PHYS, color=INK2, linewidth=0.8, linestyle=":")
                ax.set_yscale("log", base=2)
                ax.set_yticks([0.5, 1, 2, 4, 8, 16])
                ax.set_yticklabels(["0.5", "1", "2", "4", "8", "16"])
            else:
                ax.axhline(1, color=INK2, linewidth=1, linestyle="--")
                ax.set_ylim(0, 1.25)
            thread_axis(ax, threads)
            style(ax, "threads p", ylab if fam == "blocked" else "",
                  f"{ttl}: {FAMILY_NAME[fam]}")
        axes[1].legend(frameon=False, fontsize=8, loc="upper left", ncol=1)
        fig.text(0.5, -0.04, f"n >= {MIN_N_SCALING_FIG} (smaller n: separate granularity figure);  "
                 "grey band: SMT (9-16 threads on 8 physical cores);  "
                 "red band: oversubscription (more threads than 16 logical processors)",
                 ha="center", color=INK2, fontsize=8.5)
        fig.savefig(os.path.join(fig_dir, fname))
        plt.close(fig)

    # Figure: throughput vs n for selected thread counts
    sel = [p for p in (1, 2, 4, 8, 16) if p in threads]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6), sharey=True)
    pc = dict(zip(sel, ["#9ec5f4", "#6aa5ec", "#2a78d6", "#15467f", "#0c2d52"]))
    for ax, fam in zip(axes, FAMILIES):
        seq, par = FAMILIES[fam]
        ns = sizes_fam[fam]
        ax.plot(ns, [2 * n ** 3 / get("scaling", seq, n)["median_s"] / 1e9 for n in ns],
                color=INK2, linestyle="--", marker="s", markersize=3.5, linewidth=1.4,
                label="sequential")
        for p in sel:
            pts = [(n, sprow(fam, n, p)["gflops"]) for n in ns if sprow(fam, n, p)]
            ax.plot(*zip(*pts), color=pc[p], marker="o", markersize=3.5, linewidth=1.8,
                    label=f"OpenMP p = {p}")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns], rotation=40)
        ax.minorticks_off()
        style(ax, "matrix size n (n × n)", "GFLOP/s (2n³ / median time)" if fam == "blocked" else "",
              f"Throughput: {FAMILY_NAME[fam]}")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=6, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.2))
    fig.savefig(os.path.join(fig_dir, "fig_throughput.png"))
    plt.close(fig)

    # Figure: blocked at small n -- tile-count bound of the static schedule
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    bs = 64
    for n in [n for n in sizes_fam["blocked"] if n <= 1024]:
        rows = [sprow("blocked", n, p) for p in threads if sprow("blocked", n, p)]
        ax.plot([r["threads"] for r in rows], [r["speedup"] for r in rows], marker="o",
                markersize=3.5, color=size_color[n], linewidth=1.8, label=f"n = {n} (measured)")
        tiles = math.ceil(n / bs)
        bound = [tiles / math.ceil(tiles / p) for p in threads]
        ax.plot(threads, bound, color=size_color[n], linestyle=":", linewidth=1.3,
                label=f"n = {n}: {tiles} tiles / ⌈{tiles}/p⌉")
    thread_axis(ax, threads)
    ax.set_yscale("log", base=2)
    ax.set_yticks([1, 2, 4, 8, 16])
    ax.set_yticklabels(["1", "2", "4", "8", "16"])
    style(ax, "threads p", "speed-up over blocked_seq", "Row-tile granularity (blocked, small n)")
    ax.legend(frameon=False, fontsize=7.5, ncol=1)
    fig.savefig(os.path.join(fig_dir, "fig_granularity.png"))
    plt.close(fig)

    # ---------- tables: time and speed-up ----------
    def time_table(fam, name):
        seq, par = FAMILIES[fam]
        cols = threads
        tex.append(f"\\newcommand{{\\{name}}}{{")
        tex.append("\\begin{tabular}{r r " + "r " * len(cols) + "}\\toprule")
        tex.append(f"& Sequential & \\multicolumn{{{len(cols)}}}{{c}}{{OpenMP, $p$ threads}} \\\\"
                   f"\\cmidrule(lr){{3-{2 + len(cols)}}}")
        tex.append("$n$ & " + seq.replace("_", "\\_") + " & "
                   + " & ".join(f"{p}" for p in cols) + " \\\\\\midrule")
        for n in sizes_fam[fam]:
            b = get("scaling", seq, n)
            cells = [f"{fmt_t(b['median_s'])} {{\\scriptsize({b['rel_iqr_pct']:.0f})}}"]
            for p in cols:
                t = get("scaling", par, n, p)
                cells.append(f"{fmt_t(t['median_s'])} {{\\scriptsize({t['rel_iqr_pct']:.0f})}}"
                             if t else "--")
            tex.append(f"{n} & " + " & ".join(cells) + " \\\\")
        tex.append("\\bottomrule\\end{tabular}}")

    time_table("blocked", "BlockedTimeTable")
    time_table("ikj", "IkjTimeTable")

    def speedup_table(fam, name):
        tex.append(f"\\newcommand{{\\{name}}}{{")
        tex.append("\\begin{tabular}{r " + "r " * len(threads) + "}\\toprule")
        tex.append("$n$ & " + " & ".join(f"$p={p}$" for p in threads) + " \\\\\\midrule")
        for n in sizes_fam[fam]:
            cells = []
            for p in threads:
                r = sprow(fam, n, p)
                if not r:
                    cells.append("--")
                    continue
                cells.append(f"{r['speedup']:.2f} {{\\scriptsize\\color{{codecomment}}"
                             f"{100 * r['efficiency']:.0f}\\%}}")
            tex.append(f"{n} & " + " & ".join(cells) + " \\\\")
        tex.append("\\bottomrule\\end{tabular}}")

    speedup_table("blocked", "BlockedSpeedupTable")
    speedup_table("ikj", "IkjSpeedupTable")

    # ---------- table: 1-thread overhead ----------
    tex.append("\\newcommand{\\OverheadTable}{")
    tex.append("\\begin{tabular}{r r r r r r r}\\toprule & \\multicolumn{3}{c}{blocked} & "
               "\\multicolumn{3}{c}{ikj} \\\\\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}"
               "$n$ & seq & omp, $p=1$ & ratio & seq & omp, $p=1$ & ratio \\\\\\midrule")
    for n in all_sizes:
        cells = []
        for fam in FAMILIES:
            seq, par = FAMILIES[fam]
            b, o = get("scaling", seq, n), get("scaling", par, n, 1)
            cells += ([fmt_t(b["median_s"]), fmt_t(o["median_s"]),
                       f"{o['median_s'] / b['median_s']:.3f}"] if b and o else ["--"] * 3)
        tex.append(f"{n} & " + " & ".join(cells) + " \\\\")
    tex.append("\\bottomrule\\end{tabular}}")

    # ---------- table: memory and OS threads ----------
    mem_threads = [p for p in (1, 8, 16, 32) if p in threads]
    tex.append("\\newcommand{\\MemoryTable}{")
    tex.append("\\begin{tabular}{r r r " + "r " * len(mem_threads) + "r " * len(mem_threads)
               + "}\\toprule")
    tex.append(f"& Model & \\multicolumn{{{1 + len(mem_threads)}}}{{c}}{{Peak working set (MB), blocked}}"
               f" & \\multicolumn{{{len(mem_threads)}}}{{c}}{{Peak OS threads}} \\\\"
               f"\\cmidrule(lr){{3-{3 + len(mem_threads)}}}"
               f"\\cmidrule(lr){{{4 + len(mem_threads)}-{3 + 2 * len(mem_threads)}}}")
    tex.append("$n$ & $24n^2$ (MB) & seq & " + " & ".join(f"$p={p}$" for p in mem_threads)
               + " & " + " & ".join(f"$p={p}$" for p in mem_threads) + " \\\\\\midrule")
    for n in sizes_fam["blocked"]:
        b = get("scaling", "blocked_seq", n)
        cells = [f"{24 * n * n / 1e6:.1f}", f"{b['peak_working_set_mb']:.1f}"]
        cells += [f"{get('scaling', 'blocked_omp', n, p)['peak_working_set_mb']:.1f}"
                  if get("scaling", "blocked_omp", n, p) else "--" for p in mem_threads]
        cells += [f"{get('scaling', 'blocked_omp', n, p)['peak_os_threads']}"
                  if get("scaling", "blocked_omp", n, p) else "--" for p in mem_threads]
        tex.append(f"{n} & " + " & ".join(cells) + " \\\\")
    tex.append("\\bottomrule\\end{tabular}}")

    # =====================================================================
    # Affinity supplementary
    # =====================================================================
    af = [s for s in S.values() if s["experiment"] == "affinity"]
    if af:
        an = sorted({s["n"] for s in af})[0]
        ath = sorted({s["threads"] for s in af})
        pols = ["none", "compact", "spread"]
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.5), sharey=False)
        for ax, fam in zip(axes, FAMILIES):
            seq, par = FAMILIES[fam]
            base = get("scaling", seq, an)
            width = 0.26
            for i, pol in enumerate(pols):
                vals = []
                for p in ath:
                    s = get("affinity", par, an, p, pol)
                    vals.append(base["median_s"] / s["median_s"] if s and base else np.nan)
                xs = np.arange(len(ath)) + (i - 1) * width
                ax.bar(xs, vals, width=width * 0.92, color=CAT[pol], label=pol)
            ax.set_xticks(np.arange(len(ath)))
            ax.set_xticklabels([f"p = {p}" for p in ath])
            style(ax, "", "speed-up over sequential" if fam == "blocked" else "",
                  f"Thread placement, n = {an}: {FAMILY_NAME[fam]}")
        axes[0].legend(frameon=False, fontsize=8, title="placement", title_fontsize=8)
        fig.savefig(os.path.join(fig_dir, "fig_affinity.png"))
        plt.close(fig)

        tex.append("\\newcommand{\\AffinityTable}{")
        tex.append("\\begin{tabular}{l r " + "r " * len(pols) + "r}\\toprule")
        tex.append("Kernel & $p$ & " + " & ".join(f"\\texttt{{{p}}}" for p in pols)
                   + " & cores used (none) \\\\\\midrule")
        for fam in FAMILIES:
            par = FAMILIES[fam][1]
            for p in ath:
                cells = []
                for pol in pols:
                    s = get("affinity", par, an, p, pol)
                    cells.append(f"{fmt_t(s['median_s'])} {{\\scriptsize({s['rel_iqr_pct']:.0f})}}"
                                 if s else "--")
                sn = get("affinity", par, an, p, "none")
                tex.append(f"{par.replace('_', chr(92) + '_')} & {p} & " + " & ".join(cells)
                           + f" & {sn['distinct_cores'] if sn else '--'} \\\\")
            tex.append("\\addlinespace[2pt]")
        tex.append("\\bottomrule\\end{tabular}}")

    # =====================================================================
    # Vectorisation extension
    # =====================================================================
    vc = [s for s in S.values() if s["experiment"] == "vectorization"]
    if vc:
        vn = sorted({s["n"] for s in vc})[0]
        vth = sorted({s["threads"] for s in vc if "_omp" in s["method"]})
        builds = ["novec", "sse2", "avx2"]
        tex.append("\\newcommand{\\VecTable}{")
        tex.append("\\begin{tabular}{l r r r r r r r}\\toprule")
        tex.append("& & \\multicolumn{3}{c}{Median time} & \\multicolumn{2}{c}{Gain over scalar}"
                   " & Speed-up \\\\\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-8}")
        tex.append("Kernel & $p$ & scalar & SSE2 & AVX2 & SSE2 & AVX2 & AVX2 vs scalar seq \\\\"
                   "\\midrule")
        vec_plot = {}
        for fam in FAMILIES:
            seq, par = FAMILIES[fam]
            seq_scalar = get("vectorization", seq, vn, 1, build="novec")
            for m, plist in ((seq, [1]), (par, vth)):
                for p in plist:
                    t = {b: get("vectorization", m, vn, p, build=b) for b in builds}
                    if not all(t.values()):
                        continue
                    for b in builds:
                        vec_plot[(fam, m, p, b)] = t[b]["median_s"]
                    tex.append(f"{m.replace('_', chr(92) + '_')} & {p} & "
                               + " & ".join(fmt_t(t[b]["median_s"]) for b in builds) + " & "
                               + f"{t['novec']['median_s'] / t['sse2']['median_s']:.2f}$\\times$ & "
                               + f"{t['novec']['median_s'] / t['avx2']['median_s']:.2f}$\\times$ & "
                               + f"{seq_scalar['median_s'] / t['avx2']['median_s']:.1f}$\\times$ \\\\")
            tex.append("\\addlinespace[2pt]")
        tex.append("\\bottomrule\\end{tabular}}")

        fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.5), sharey=True)
        for ax, fam in zip(axes, FAMILIES):
            seq, par = FAMILIES[fam]
            ref = vec_plot.get((fam, seq, 1, "novec"))
            labels = ["seq"] + [f"p = {p}" for p in vth]
            confs = [(seq, 1)] + [(par, p) for p in vth]
            width = 0.26
            for i, b in enumerate(builds):
                vals = [ref / vec_plot[(fam, m, p, b)] if (fam, m, p, b) in vec_plot else np.nan
                        for m, p in confs]
                ax.bar(np.arange(len(confs)) + (i - 1) * width, vals, width=width * 0.92,
                       color=CAT[b], label={"novec": "scalar", "sse2": "SSE2 (16 B)",
                                            "avx2": "AVX2 (32 B)"}[b])
            ax.set_xticks(np.arange(len(confs)))
            ax.set_xticklabels(labels)
            ax.set_yscale("log", base=2)
            ax.set_yticks([1, 2, 4, 8, 16, 32])
            ax.set_yticklabels(["1", "2", "4", "8", "16", "32"])
            style(ax, "", "speed-up over scalar sequential" if fam == "blocked" else "",
                  f"{fam}, n = {vn}")
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle("Vectorisation x threads (bars: median speed-up over the scalar sequential "
                     "build)", fontsize=10, y=1.02)
        fig.savefig(os.path.join(fig_dir, "fig_vectorization.png"))
        plt.close(fig)

    # =====================================================================
    # omp simd directive diagnostic
    # =====================================================================
    sd = [s for s in S.values() if s["experiment"] == "simd_directive"]
    if sd:
        tex.append("\\newcommand{\\SimdTable}{")
        tex.append("\\begin{tabular}{r r r r r r}\\toprule & & \\multicolumn{2}{c}{with \\texttt{omp simd}}"
                   " & & \\\\\\cmidrule(lr){3-4}"
                   "$n$ & $p$ & scaling run & vect.\\ run & without & ratio \\\\\\midrule")
        for s in sorted(sd, key=lambda s: (s["n"], s["threads"])):
            w_ = get("scaling", "blocked_omp", s["n"], s["threads"])
            v_ = get("vectorization", "blocked_omp", s["n"], s["threads"])
            if not w_:
                continue
            tex.append(f"{s['n']} & {s['threads']} & {fmt_t(w_['median_s'])} & "
                       f"{fmt_t(v_['median_s']) if v_ else '--'} & {fmt_t(s['median_s'])} & "
                       f"{s['median_s'] / w_['median_s']:.2f}$\\times$ \\\\")
        tex.append("\\bottomrule\\end{tabular}}")

    # =====================================================================
    # Repeatability: the same configuration measured in independent processes by
    # different experiments (n = 2048, SSE2 build, spread placement)
    # =====================================================================
    rn = 2048
    exps = ["scaling", "affinity", "vectorization"]
    tex.append("\\newcommand{\\RepeatTable}{")
    tex.append("\\begin{tabular}{l r r r r r}\\toprule Kernel & $p$ & scaling & placement & "
               "vectorisation & max/min \\\\\\midrule")
    for fam in FAMILIES:
        seq, par = FAMILIES[fam]
        for m, plist in ((seq, [1]), (par, [1, 2, 4, 8, 16])):
            for p in plist:
                vals = [get(e, m, rn, p) for e in exps]
                got = [v["median_s"] for v in vals if v]
                if len(got) < 2:
                    continue
                tex.append(f"{m.replace('_', chr(92) + '_')} & {p} & "
                           + " & ".join(f"{fmt_t(v['median_s'])} {{\\scriptsize({v['rel_iqr_pct']:.0f})}}"
                                        if v else "--" for v in vals)
                           + f" & {max(got) / min(got):.2f} \\\\")
        tex.append("\\addlinespace[2pt]")
    tex.append("\\bottomrule\\end{tabular}}")

    with open(os.path.join(sum_dir, "tables.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")
    print(f"wrote {sum_dir} and {fig_dir}")


if __name__ == "__main__":
    main()
