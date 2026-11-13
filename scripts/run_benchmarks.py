"""Benchmark driver: runs every configuration of the selected experiments as a separate process.

Same protocol as the Assignment 1 and 2 drivers. For each configuration
(experiment, method, build, n, threads, affinity) it
  * launches build/<exe> bench with OMP_NUM_THREADS=p and OMP_DYNAMIC=FALSE,
  * reads its CSV rows (warm-up and measured repetitions) and "# key,value" comments,
  * samples the peak memory and the number of OS threads of the process with psutil,
  * stops it if one multiplication exceeds the time limit or the process exceeds the
    memory budget, and then skips larger sizes of the same configuration.

Experiments (config/benchmark_config.json): scaling, affinity, vectorization, simd_directive.

Outputs (in --out, default results/):
  raw/measurements.csv  every timing row, labelled with the configuration
  raw/runs.csv          one row per configuration: status, memory, threads, placement, hash
  selftest.txt          output of the known-answer self tests of every build
Result matrices of the sequential baselines go to --outputs (for validate_outputs.py).

    python scripts/run_benchmarks.py --data DIR --outputs DIR [--experiments scaling]
                                     [--sizes 1024] [--methods blocked_omp] [--threads 1 8]
                                     [--reps 3] [--resume]
"""
import argparse
import csv
import json
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

import psutil

EXE = ".exe" if sys.platform == "win32" else ""
CONFIG_FIELDS = ["experiment", "method", "build", "n", "threads", "affinity", "block"]
RUN_FIELDS = ["timestamp", *CONFIG_FIELDS, "status", "exit_code", "warmup_runs", "measured_rows",
              "inner", "wall_time_s", "peak_working_set_bytes", "peak_private_bytes",
              "peak_os_threads", "team_size_in_kernel", "omp_max_threads", "oversubscribed",
              "distinct_lp", "distinct_cores", "placement", "hash_c", "note"]
ROW_FIELDS = ["phase", "rep", "inner", "batch_time_s", "time_per_mult_s", "checksum"]
MEAS_FIELDS = [*CONFIG_FIELDS, *ROW_FIELDS]


def is_parallel(method):
    return "_omp" in method


def build_jobs(cfg, experiments, data, outputs):
    """List of configurations in execution order: n ascending, then method (sequential
    baseline before its parallel version), threads ascending, affinity, build."""
    jobs = []
    for exp_name in experiments:
        exp = cfg["experiments"][exp_name]
        sizes = sorted({n for s in exp["methods"].values() for n in s})
        for n in sizes:
            for m, msizes in exp["methods"].items():
                if n not in msizes:
                    continue
                for p in (exp["threads"] if is_parallel(m) else [1]):
                    for aff in exp["affinity"]:
                        for build in exp["builds"]:
                            dump = None
                            if m in exp.get("dump_methods", []):
                                dump = os.path.join(outputs, exp_name, f"{m}_{build}_n{n}.bin")
                            jobs.append({
                                "experiment": exp_name, "method": m, "build": build, "n": n,
                                "threads": p, "affinity": aff, "block": cfg["block_size"],
                                "exe": os.path.join("build",
                                                    cfg["c"]["builds"][build]["exe"] + EXE),
                                "a": os.path.join(data, f"A_{n}.bin"),
                                "b": os.path.join(data, f"B_{n}.bin"), "dump": dump})
    return jobs


def command(job, cfg, warmup, reps):
    cmd = [job["exe"], "bench", "--method", job["method"], "--n", str(job["n"]),
           "--a", job["a"], "--b", job["b"], "--threads", str(job["threads"]),
           "--affinity", job["affinity"], "--block", str(job["block"]),
           "--warmup", str(warmup), "--reps", str(reps),
           "--min-batch-time", str(cfg["min_batch_time_s"])]
    if job["dump"]:
        cmd += ["--dump", job["dump"]]
    return cmd


def proc_sample(proc):
    """Peak working set, peak private bytes (bytes) and current OS thread count."""
    try:
        mi = proc.memory_info()
        nt = proc.num_threads()
    except psutil.Error:
        return 0, 0, 0
    return (getattr(mi, "peak_wset", mi.rss), getattr(mi, "peak_pagefile", getattr(mi, "vms", 0)),
            nt)


def run_config(job, cfg, warmup, reps):
    cmd = command(job, cfg, warmup, reps)
    limit = cfg["time_limit_per_multiplication_s"]
    budget = cfg["memory_budget_bytes"]
    interval = cfg["memory_sampling_interval_s"]
    if job["dump"]:
        os.makedirs(os.path.dirname(job["dump"]), exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS=str(job["threads"]), OMP_DYNAMIC="FALSE")
    for var in ("OMP_PROC_BIND", "OMP_PLACES", "OMP_WAIT_POLICY", "GOMP_SPINCOUNT"):
        env.pop(var, None)

    start = time.perf_counter()
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             env=env)
    proc = psutil.Process(popen.pid)
    lines = queue.Queue()
    threading.Thread(target=lambda: [lines.put(l) for l in popen.stdout], daemon=True).start()

    rows, comments = [], []

    def drain():
        try:
            while True:
                line = lines.get_nowait().strip()
                if line.startswith("#"):
                    comments.append(line[1:].strip())
                elif line and not line.startswith("method,"):
                    rows.append(line.split(","))
        except queue.Empty:
            pass

    peak_ws = peak_priv = peak_threads = 0
    last_progress = time.perf_counter()
    status, note = "ok", ""
    while True:
        ws, priv, nt = proc_sample(proc)
        peak_ws, peak_priv, peak_threads = max(peak_ws, ws), max(peak_priv, priv), max(peak_threads, nt)
        before = len(rows) + len(comments)
        drain()
        if len(rows) + len(comments) > before:
            last_progress = time.perf_counter()
        if popen.poll() is not None:
            break
        if peak_ws > budget:
            status, note = "memory_limit", f"peak working set exceeded {budget} bytes"
            popen.kill()
            break
        if time.perf_counter() - last_progress > limit:
            status, note = "timeout", f"no multiplication finished within {limit} s"
            popen.kill()
            break
        time.sleep(interval)
    popen.wait()
    wall = time.perf_counter() - start
    time.sleep(0.2)
    drain()
    stderr = popen.stderr.read()
    if status == "ok" and popen.returncode != 0:
        status, note = "error", stderr.strip().replace("\n", " ")[:300]

    extra = {}
    for c in comments:
        parts = c.split(",")
        extra.update(dict(zip(parts[0::2], parts[1::2])))
    measured = [r for r in rows if r[3] == "measure"]
    run = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **{k: job[k] for k in CONFIG_FIELDS},
        "status": status, "exit_code": popen.returncode, "warmup_runs": warmup,
        "measured_rows": len(measured), "inner": measured[0][5] if measured else "",
        "wall_time_s": f"{wall:.3f}", "peak_working_set_bytes": peak_ws,
        "peak_private_bytes": peak_priv, "peak_os_threads": peak_threads,
        "team_size_in_kernel": extra.get("team_size_in_kernel", ""),
        "omp_max_threads": extra.get("omp_max_threads", ""),
        "oversubscribed": job["threads"] > psutil.cpu_count(logical=True),
        "distinct_lp": extra.get("distinct_lp", ""),
        "distinct_cores": extra.get("distinct_cores", ""),
        "placement": extra.get("placement", ""), "hash_c": extra.get("hash_c", ""),
        "note": note,
    }
    # Program rows: method,n,threads,phase,rep,inner,batch,time,checksum -> labelled rows.
    labelled = [[job[k] for k in CONFIG_FIELDS] + r[3:] for r in rows]
    return run, labelled


def append_csv(path, fields, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(fields)
        w.writerows(rows)


def selftests(cfg, out_dir):
    with open(os.path.join(out_dir, "selftest.txt"), "w") as log:
        for name, b in cfg["c"]["builds"].items():
            exe = os.path.join("build", b["exe"] + EXE)
            res = subprocess.run([exe, "selftest"], capture_output=True, text=True)
            log.write(f"== build {name} ({exe}) ==\n{res.stdout}\n")
            print(f"[selftest] {name}: {'PASSED' if res.returncode == 0 else 'FAILED'}")
            if res.returncode != 0:
                sys.exit(f"self test failed for {name}:\n{res.stdout}{res.stderr}")
        res = subprocess.run([os.path.join("build", "parallel" + EXE), "topology"],
                             capture_output=True, text=True)
        log.write(f"== topology ==\n{res.stdout}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/benchmark_config.json")
    parser.add_argument("--data", default="data")
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--out", default="results")
    parser.add_argument("--experiments", nargs="*",
                        default=["scaling", "affinity", "vectorization", "simd_directive"])
    parser.add_argument("--sizes", type=int, nargs="*")
    parser.add_argument("--methods", nargs="*")
    parser.add_argument("--threads", type=int, nargs="*")
    parser.add_argument("--reps", type=int)
    parser.add_argument("--resume", action="store_true",
                        help="skip configurations that already have status ok in runs.csv")
    parser.add_argument("--allow-battery", action="store_true",
                        help="run even if the laptop is not on AC power")
    args = parser.parse_args()
    battery = psutil.sensors_battery()
    if battery and not battery.power_plugged and not args.allow_battery:
        sys.exit("not on AC power: connect the charger (or pass --allow-battery)")

    with open(args.config) as f:
        cfg = json.load(f)
    reps = args.reps or cfg["measured_repetitions"]
    warmup = cfg["warmup_runs"]
    os.makedirs(os.path.join(args.out, "raw"), exist_ok=True)
    selftests(cfg, args.out)

    jobs = build_jobs(cfg, args.experiments, args.data, args.outputs)
    if args.sizes:
        jobs = [j for j in jobs if j["n"] in args.sizes]
    if args.methods:
        jobs = [j for j in jobs if j["method"] in args.methods]
    if args.threads:
        jobs = [j for j in jobs if j["threads"] in args.threads]

    stopped = {}
    meas_path = os.path.join(args.out, "raw", "measurements.csv")
    runs_path = os.path.join(args.out, "raw", "runs.csv")
    done = set()
    if args.resume and os.path.exists(runs_path):
        done = {tuple(r[k] for k in CONFIG_FIELDS)
                for r in csv.DictReader(open(runs_path)) if r["status"] == "ok"}
    print(f"{len(jobs)} configurations", flush=True)
    for job in jobs:
        key = (job["experiment"], job["method"], job["build"], job["threads"], job["affinity"])
        label = (f"{job['experiment']}/{job['method']}[{job['build']}] n={job['n']} "
                 f"p={job['threads']} {job['affinity']}")
        if tuple(str(job[k]) for k in CONFIG_FIELDS) in done:
            print(f"[done] {label}")
            continue
        # Dense A, B and C alone (24 n^2 bytes) must fit in the memory budget.
        if 24 * job["n"] ** 2 > cfg["memory_budget_bytes"]:
            stopped[key] = "memory_limit"
        if key in stopped:
            row = {k: "" for k in RUN_FIELDS}
            row.update({"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        **{k: job[k] for k in CONFIG_FIELDS}, "status": "skipped",
                        "measured_rows": 0,
                        "note": f"not attempted: {stopped[key]} at a smaller size or storage model"})
            append_csv(runs_path, RUN_FIELDS, [[row[k] for k in RUN_FIELDS]])
            print(f"[skip] {label}")
            continue
        print(f"[run ] {label} ...", flush=True)
        run, rows = run_config(job, cfg, warmup, reps)
        battery = psutil.sensors_battery()
        if battery and not battery.power_plugged:
            run["note"] = (run["note"] + " on battery power").strip()
            print("       WARNING: on battery power", flush=True)
        append_csv(meas_path, MEAS_FIELDS, rows)
        append_csv(runs_path, RUN_FIELDS, [[run[k] for k in RUN_FIELDS]])
        med = sorted(float(r[len(CONFIG_FIELDS) + 4]) for r in rows
                     if r[len(CONFIG_FIELDS)] == "measure")
        med_s = f"median {med[len(med) // 2]:.4g} s" if med else ""
        print(f"       {run['status']} {med_s} peakWS {run['peak_working_set_bytes'] / 1e6:.1f} MB "
              f"threads(os) {run['peak_os_threads']} team {run['team_size_in_kernel']} "
              f"cores {run['distinct_cores']}", flush=True)
        if run["status"] != "ok":
            stopped[key] = run["status"]
        time.sleep(cfg["pause_between_runs_s"])


if __name__ == "__main__":
    main()
