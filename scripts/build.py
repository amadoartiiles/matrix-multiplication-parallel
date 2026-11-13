"""Compile c/parallel.c once per configured build into build/.

    python scripts/build.py [--config config/benchmark_config.json]

Builds (config "c.builds"):
  build/parallel        main build, -O2 (SSE2 16-byte vectors on x86-64)
  build/parallel_novec  -fno-tree-vectorize -DMM_NO_SIMD (scalar; vectorisation extension)
  build/parallel_avx2   -mavx2 (32-byte vectors, no FMA; vectorisation extension)
and writes the GCC vectorisation report (-fopt-info-vec-optimized) of every build to
results/vectorization_report.txt as evidence of which loops are vectorised.
The C compiler is taken from the CC environment variable, or the configured one.
"""
import argparse
import json
import os
import re
import subprocess
import sys

EXE = ".exe" if sys.platform == "win32" else ""


def run(cmd, **kw):
    print("$", " ".join(cmd))
    return subprocess.run(cmd, check=True, **kw)


def kernel_of_line(src_lines, line):
    """Name of the C function that contains a source line."""
    for l in range(line - 1, -1, -1):
        m = re.search(r"double \*(mm_\w+)\(", src_lines[l])
        if m:
            return m.group(1)
    return "?"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/benchmark_config.json")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)

    os.makedirs("build", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    cc = os.environ.get("CC", cfg["c"]["compiler"])
    flags = cfg["c"]["flags"]
    report = []
    for name, b in cfg["c"]["builds"].items():
        all_flags = [*flags, *b["extra_flags"]]
        run([cc, *all_flags, "-o", f"build/{b['exe']}{EXE}", "c/parallel.c", "-lm"])
        res = subprocess.run([cc, *all_flags, "-fopt-info-vec-optimized", "-c", "c/parallel.c",
                              "-o", os.devnull], capture_output=True, text=True, check=True)
        src = open("c/parallel.c").read().splitlines()
        loops = []
        for l in res.stderr.splitlines():
            m = re.match(r"c/parallel.c:(\d+):", l)
            if "basic block" not in l and m:
                loops.append(f"{l}   [{kernel_of_line(src, int(m.group(1)))}]")
        report.append(f"== build {name}: {' '.join(all_flags)} ==\n"
                      + ("\n".join(loops) if loops else "(no loop vectorised)") + "\n")
    with open("results/vectorization_report.txt", "w") as f:
        f.write("Loop vectorisation reported by GCC (-fopt-info-vec-optimized; basic-block\n"
                "vectorisation lines omitted); the kernel containing each loop is in brackets.\n"
                "A kernel that does not appear has a scalar inner loop.\n\n" + "\n".join(report))
    print("wrote results/vectorization_report.txt")


if __name__ == "__main__":
    main()
