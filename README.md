# Assignment 3 — Parallel Matrix Multiplication and Optional Vectorisation

BIG DATA · Grado en Ciencia e Ingeniería de Datos · ULPGC · 2026-2027

Research question: **how does the number of OpenMP threads change the runtime, speed-up,
efficiency and memory of a shared-memory dense matrix multiplication on one 8-core / 16-thread
laptop, and when does parallel execution help?** Optional extension: how much do SSE2 and AVX2
vectorisation add, separately from the thread count?

The report is in `report/` (`report.tex` → `report.pdf`).

## Reused material from Assignments 1 and 2

* Sequential baselines: the validated `mm_blocked` (cache blocking, tile 64) and `mm_ikj`
  (loop reordering) kernels of `c/dense.c` from
  <https://github.com/amadoartiiles/matrix-multiplication-optimised-sparse> (Assignment 2, commit `d78be14`),
  copied into `c/parallel.c` with the bodies unchanged (`mm_blocked_seq`, `mm_ikj_seq`).
  The Assignment 1 `ijk` kernel is kept only as the reference of the self test.
* Inputs: the Assignment 1 generator, seed (20261012) and file format; `generate_inputs.py
  --check-earlier` verifies that the files are byte-identical to the Assignment 1 and 2 inputs
  for every shared size (the manifest records the result).
* Driver, memory measurement (psutil peak working set), error bound, budgets and repetition
  protocol follow the Assignment 1 and 2 scripts.

## Repository layout

```
c/common.h                    timer, dense file I/O, checksum, FNV-1a hash of the result
c/parallel.c                  kernels blocked_seq/omp, ikj_seq/omp, blocked_omp_nosimd;
                              thread pinning; self test; bench; topology
config/benchmark_config.json  sizes, thread counts, placements, builds, repetitions, budgets
scripts/generate_inputs.py    dense inputs (Assignment 1 generator) + data/manifest.json
scripts/build.py              builds build/parallel{,_novec,_avx2}; vectorisation report
scripts/run_benchmarks.py     driver: self tests, one process per configuration, memory, limits
scripts/validate_outputs.py   NumPy reference check + bitwise (hash) check of every run
scripts/collect_env.py        hardware / software description -> results/environment.json
scripts/analyze.py            summary tables, speed-up/efficiency and figures
scripts/run_campaign.sh       the whole campaign in one command (keeps Windows awake)
scripts/frequency_check.ps1   supplementary: effective CPU clock while blocked_omp runs with p threads
results/raw/                  raw measurements (measurements.csv, runs.csv)
results/summary/              summary.csv, speedup.csv, limits.csv, tables.tex
results/figures/              PNG figures used in the report
results/validation.csv        reference check; results/hash_check.csv bitwise check of every run
results/selftest.txt          known-answer tests of every build + CPU topology
results/frequency_check.csv   output of the supplementary clock check (separate run)
results/vectorization_report.txt  GCC -fopt-info-vec output of every build
report/                       LaTeX source and PDF of the report
```

## Requirements (versions used for the reported results)

| Component | Version |
|---|---|
| OS | Windows 11 Home 10.0.26300 |
| C compiler | GCC 16.2.0 (MinGW-w64 x86_64-ucrt-posix-seh, WinLibs build), libgomp (OpenMP 5.1, `_OPENMP` = 202111) |
| Python (scripts only) | CPython 3.11.9 |
| Python packages | numpy 2.2.5, scipy 1.17.1, matplotlib 3.10.1, psutil 7.2.2 |

The benchmarked program is plain C11 + OpenMP without numerical libraries. NumPy is used only to
generate inputs and as the correctness reference.

## Reproduce everything

From the repository root (Git Bash), **on AC power**:

```bash
pip install -r requirements.txt
bash scripts/run_campaign.sh "$HOME/mm3_work"     # inputs, build, benchmarks, validation, analysis
cd report && pdflatex report.tex && pdflatex report.tex
```

or step by step:

```bash
DATA=$HOME/mm3_work/data; OUT=$HOME/mm3_work/outputs
python scripts/generate_inputs.py --data $DATA        # 1. inputs (+ --check-earlier <manifests>)
python scripts/build.py                               # 2. three builds + vectorisation report
python scripts/run_benchmarks.py --data $DATA --outputs $OUT   # 3. all experiments
python scripts/validate_outputs.py --data $DATA --outputs $OUT # 4. correctness
python scripts/collect_env.py                         # 5. environment
python scripts/analyze.py                             # 6. tables + figures
```

Individual program:

```bash
build/parallel.exe selftest
build/parallel.exe topology
build/parallel.exe bench --method blocked_omp --n 2048 --a $DATA/A_2048.bin --b $DATA/B_2048.bin \
    --threads 8 --affinity spread
```

Supplementary clock check (after the campaign; the sampling perturbs the run, so its times
are not used elsewhere):
`powershell -ExecutionPolicy Bypass -File scripts/frequency_check.ps1 -Data $DATA`.

A quick pilot: `python scripts/run_benchmarks.py --data $DATA --outputs $OUT --out pilot
--sizes 256 1024 --threads 1 8 16 --reps 3`.

## Fixed experimental settings

* Numeric type IEEE 754 binary64; row-major dense storage; tile size 64 (Assignment 2).
* GCC flags `-O2 -std=c11 -Wall -Wextra -fopenmp` (no `-march`, no fast-math, so no FMA).
  Builds of the vectorisation extension add `-fno-tree-vectorize -DMM_NO_SIMD` (scalar) or
  `-mavx2` (32-byte vectors, still no FMA). Every build gives bitwise identical results.
* Parallelisation: `omp parallel` + `omp for schedule(static)` over row tiles (blocked) or rows
  (ikj); `omp simd` on the inner loop of the parallel kernels (without it GCC leaves that loop
  scalar, see `blocked_omp_nosimd`); all kernels `noinline`.
* Thread count p: `OMP_NUM_THREADS=p`, `omp_set_num_threads(p)`, `OMP_DYNAMIC=FALSE`; the team
  size observed inside the kernel and the OS thread count of the process are recorded.
  `OMP_WAIT_POLICY` is left at the libgomp default.
* Placement: libgomp on Windows does not implement `OMP_PLACES`, so the program pins threads
  with `SetThreadAffinityMask`: `spread` (main study: one thread per physical core first, then
  SMT siblings), `compact` (both siblings of a core first) or `none` (OS scheduling).
  Placement is checked after the measurements (`placement`, `distinct_cores` in `runs.csv`).
* Timed region: allocation + zero initialisation of C + the loops (thread creation happens in
  the warm-up run, outside the measured samples). Inputs are loaded before timing.
* 1 warm-up + 5 measured repetitions per configuration, one process per configuration;
  samples shorter than 0.1 s are batched (time per multiplication reported).
* Speed-up S(p) = median Tseq / median Tp with the equivalent sequential kernel, efficiency
  E(p) = S(p)/p; the quartile range of S is reported too.
* Budgets: 300 s per multiplication, 4 GB peak working set per process.
* Memory metric: peak working set of the process (Windows `PeakWorkingSetSize`, psutil).

## Raw data format

`results/raw/measurements.csv`:
`experiment,method,build,n,threads,affinity,block,phase,rep,inner,batch_time_s,time_per_mult_s,checksum`
(`phase` is `warmup` or `measure`). The numeric type (`float64`) and the representation (dense
row-major) are the same in every row; `results/summary/summary.csv` carries them explicitly.

`results/raw/runs.csv`: one row per configuration with status (`ok`, `timeout`, `memory_limit`,
`error`, `skipped`), wall time, peak working set, peak private bytes, peak OS threads, team size
seen inside the kernel, oversubscription flag, logical processors and cores used, and the hash
of C. An interrupted campaign is continued with `run_benchmarks.py --resume`.

## Use of generative AI

Claude Opus (Anthropic), through Claude Code, assisted with writing and debugging the C program
and Python scripts, running the measurement campaign and drafting the report. All measurements
in `results/` come from actually running the code on the machine described in
`results/environment.json`, and every number in the report was checked against them.
