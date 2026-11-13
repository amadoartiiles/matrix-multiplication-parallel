"""Correctness of every benchmarked configuration, in two steps.

1. Reference check (as in Assignments 1 and 2). The result of the last repetition of each
   sequential baseline (blocked_seq, ikj_seq) is dumped and compared with NumPy's A @ B
   using the componentwise forward-error bound of a length-n dot product in binary64,
   |C_hat - C| <= gamma_n (|A| |B|), gamma_n = n u / (1 - n u), u = 2^-53 [Higham 2002, 3.5].
   Implementation and reference both satisfy it, so the check is
       |C_impl - C_ref| <= 2 gamma_n (|A| |B|)      (element-wise).
   NumPy is used only for checking.  -> results/validation.csv

2. Bitwise check of every run. Every program prints a 64-bit FNV-1a hash of the bit
   patterns of its result C. All kernels, thread counts, placements and builds must give
   the same hash for the same n as the validated sequential result, and all measured
   repetitions of a configuration must print the same checksum.  -> results/hash_check.csv

    python scripts/validate_outputs.py --data DIR --outputs DIR [--results results]
"""
import argparse
import csv
import os

import numpy as np

U = 2.0 ** -53


def gamma(n):
    return n * U / (1 - n * U)


def read_dense(path, n):
    return np.fromfile(path, dtype="<f8").reshape(n, n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--results", default="results")
    args = parser.parse_args()

    meas = list(csv.DictReader(open(os.path.join(args.results, "raw", "measurements.csv"))))
    runs = [r for r in csv.DictReader(open(os.path.join(args.results, "raw", "runs.csv")))
            if r["status"] == "ok"]
    key_fields = ["experiment", "method", "build", "n", "threads", "affinity"]
    checks = {}
    for r in meas:
        if r["phase"] == "measure":
            checks.setdefault(tuple(r[k] for k in key_fields), set()).add(r["checksum"])

    # ---------- 1. reference check of the dumped sequential results ----------
    rows = []
    ref_hash = {}
    dumped = [r for r in runs if r["experiment"] == "scaling" and "_seq" in r["method"]]
    for n in sorted({int(r["n"]) for r in dumped}):
        A = read_dense(os.path.join(args.data, f"A_{n}.bin"), n)
        B = read_dense(os.path.join(args.data, f"B_{n}.bin"), n)
        ref = A @ B
        bound = 2 * gamma(n) * (np.abs(A) @ np.abs(B))
        results = {}
        for r in [r for r in dumped if int(r["n"]) == n]:
            path = os.path.join(args.outputs, "scaling", f"{r['method']}_{r['build']}_n{n}.bin")
            C = read_dense(path, n)
            err = np.abs(C - ref)
            results[r["method"]] = C
            ok = bool((err <= bound).all())
            rows.append({"n": n, "method": r["method"], "build": r["build"],
                         "reference": "numpy dense A@B", "max_abs_error": f"{err.max():.3e}",
                         "max_error_over_bound": f"{(err / np.where(bound > 0, bound, 1)).max():.3e}",
                         "within_tolerance": ok, "hash_c": r["hash_c"], "equal_to": ""})
            if ok:
                ref_hash.setdefault(n, set()).add(r["hash_c"])
            del err
        for row in rows:
            if row["n"] == n:
                row["equal_to"] = " ".join(sorted(m for m in results if m != row["method"] and
                                                  np.array_equal(results[m], results[row["method"]])))
        print(f"n={n:5d}: " + ", ".join(f"{r['method']}={r['within_tolerance']} "
                                         f"(err/bound {r['max_error_over_bound']})"
                                         for r in rows if r["n"] == n), flush=True)
        del A, B, ref, bound, results
    with open(os.path.join(args.results, "validation.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---------- 2. bitwise check of every configuration ----------
    hrows = []
    for r in runs:
        n = int(r["n"])
        expected = ref_hash.get(n, set())
        cks = checks.get(tuple(r[k] for k in key_fields), set())
        hrows.append({**{k: r[k] for k in key_fields}, "hash_c": r["hash_c"],
                      "validated_hash": " ".join(sorted(expected)),
                      "bitwise_equal_to_validated": len(expected) == 1 and r["hash_c"] in expected,
                      "checksums_consistent_across_reps": len(cks) == 1,
                      "team_size_ok": (r["team_size_in_kernel"] == r["threads"]
                                       if "_omp" in r["method"] else r["team_size_in_kernel"] == "1")})
    with open(os.path.join(args.results, "hash_check.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hrows[0].keys()))
        w.writeheader()
        w.writerows(hrows)
    bad = [h for h in hrows if not (h["bitwise_equal_to_validated"]
                                    and h["checksums_consistent_across_reps"] and h["team_size_ok"])]
    print(f"{len(hrows)} configurations: {len(hrows) - len(bad)} bitwise identical to the validated "
          f"result with consistent checksums and the requested team size")
    for h in bad:
        print("  FAILED:", h)
    failed = [r for r in rows if not r["within_tolerance"]]
    if failed or bad:
        raise SystemExit(f"{len(failed)} reference failures, {len(bad)} bitwise/team failures")


if __name__ == "__main__":
    main()
