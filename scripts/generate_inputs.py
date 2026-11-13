"""Generate the dense input matrices and a manifest with SHA-256 hashes.

The generator, seed and file format are exactly those of Assignment 1 (also reused in
Assignment 2), so A_n.bin and B_n.bin are byte-identical to the earlier inputs for the
shared sizes; --check-earlier compares the hashes with an earlier manifest
(Assignment 1 data/manifest.json uses the key "files", Assignment 2 the key "dense").

    python scripts/generate_inputs.py [--data data] [--check-earlier "<path>/manifest.json"]
"""
import argparse
import hashlib
import json
import os

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def generate(n, seed, low, high):
    rng = np.random.default_rng([seed, n])
    A = rng.uniform(low, high, size=(n, n)).astype("<f8")
    B = rng.uniform(low, high, size=(n, n)).astype("<f8")
    return A, B


def all_sizes(cfg):
    sizes = set()
    for exp in cfg["experiments"].values():
        for s in exp["methods"].values():
            sizes.update(s)
    return sorted(sizes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/benchmark_config.json")
    parser.add_argument("--data", default="data")
    parser.add_argument("--check-earlier", nargs="*", default=[],
                        help="manifest.json of Assignment 1 and/or 2 to compare hashes with")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)
    inp = cfg["input"]
    os.makedirs(args.data, exist_ok=True)
    manifest = {"numpy_version": np.__version__, "input": inp, "files": {}}
    for n in all_sizes(cfg):
        A, B = generate(n, inp["seed"], inp["low"], inp["high"])
        for name, M in (("A", A), ("B", B)):
            path = os.path.join(args.data, f"{name}_{n}.bin")
            M.tofile(path)
            manifest["files"][f"{name}_{n}.bin"] = {"n": n, "bytes": os.path.getsize(path),
                                                    "sha256": sha256(path)}
        print(f"n={n}: wrote A_{n}.bin and B_{n}.bin", flush=True)

    checks = []
    for path in args.check_earlier:
        with open(path) as f:
            old = json.load(f)
        old = old.get("files", old.get("dense", {}))
        shared = [k for k in old if k in manifest["files"]]
        same = [k for k in shared if old[k]["sha256"] == manifest["files"][k]["sha256"]]
        checks.append({"manifest": path.replace("\\", "/"), "files_compared": len(shared),
                       "identical": len(same)})
        print(f"{path}: {len(same)}/{len(shared)} shared files identical")
    manifest["check_against_earlier_assignments"] = checks
    with open(os.path.join(args.data, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
