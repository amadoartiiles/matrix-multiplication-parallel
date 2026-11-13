/*
 * Shared-memory parallel dense matrix multiplication with OpenMP, and the equivalent
 * sequential baselines, on contiguous row-major IEEE 754 binary64 buffers.
 *
 *   blocked_seq  cache-blocked i-k-j kernel of Assignment 2 (mm_blocked, body unchanged)
 *   blocked_omp  the same loop nest; the outer loop over row tiles (ii) is shared among
 *                threads with "omp parallel for schedule(static)"
 *   ikj_seq      loop-reordered kernel of Assignment 2 (mm_ikj, body unchanged)
 *   ikj_omp      the same loop nest; the outer loop over rows (i) is shared among threads
 *   blocked_omp_nosimd  blocked_omp without the "omp simd" directive (diagnostic only)
 *
 * Each thread writes only the rows of C that belong to its iterations of the outer loop,
 * and A and B are read-only, so there is no shared write and no reduction. Every C[i][j]
 * still accumulates its products in increasing k order from 0.0, so all kernels and all
 * thread counts return bitwise identical results (checked by the self test and by the
 * hash printed by every benchmark run).
 *
 * Usage:
 *   parallel selftest
 *   parallel bench --method M --n N --a FILE --b FILE [--threads P] [--affinity none|spread|compact]
 *                  [--block BS] [--warmup W] [--reps R] [--min-batch-time T] [--dump FILE]
 *   parallel topology
 *
 * Bench output (stdout): CSV rows "method,n,threads,phase,rep,inner,batch_time_s,
 * time_per_mult_s,checksum" for every warm-up and measured repetition, and comment lines
 * "# key,value" (team size observed inside the kernel, hash of C, thread placement).
 */
#include "common.h"

#include <omp.h>

#define DEFAULT_BLOCK 64
#define MAX_LP 256

/* "omp simd" on the innermost loop of the parallel kernels: its iterations are independent
 * (each updates a different C[i][j]), so vectorising it does not change any result. It is
 * needed because GCC does not vectorise the loop inside the OpenMP outlined function on its
 * own (see mm_blocked_omp_nosimd). Building with -DMM_NO_SIMD removes the directive (used
 * together with -fno-tree-vectorize for the scalar build of the vectorisation extension). */
/* Every kernel is compiled as a separate function. When GCC inlined mm_ikj_seq into the
 * dispatcher multiply(), the same vectorised loop ran 1.5x slower than as a function of its
 * own (code-generation artefact found in the pilot runs), whereas the OpenMP kernels are
 * always separate (outlined) functions. noinline keeps sequential and parallel kernels
 * compiled in the same context. */
#define KERNEL __attribute__((noinline)) static

#ifdef MM_NO_SIMD
#define OMP_SIMD
#else
#define OMP_SIMD _Pragma("omp simd")
#endif

/* Team size observed inside the last parallel kernel (written by thread 0). */
static int team_size_seen = 0;

/* ======================================================================
 * Kernels: the timed region is exactly one of these calls
 * ====================================================================== */

/* Assignment 1 baseline (i-j-k). Used only as the reference of the self test. */
KERNEL double *mm_ijk(const double *A, const double *B, int n) {
    double *C = (double *)calloc((size_t)n * n, sizeof(double));
    if (C == NULL) return NULL;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++) {
                s += A[(size_t)i * n + k] * B[(size_t)k * n + j];
            }
            C[(size_t)i * n + j] = s;
        }
    }
    return C;
}

/* Sequential baseline 1: Assignment 2 mm_ikj, body unchanged. */
KERNEL double *mm_ikj_seq(const double *A, const double *B, int n) {
    double *C = (double *)calloc((size_t)n * n, sizeof(double));
    if (C == NULL) return NULL;
    for (int i = 0; i < n; i++) {
        double *Ci = C + (size_t)i * n;
        for (int k = 0; k < n; k++) {
            const double a = A[(size_t)i * n + k];
            const double *Bk = B + (size_t)k * n;
            for (int j = 0; j < n; j++) {
                Ci[j] += a * Bk[j];
            }
        }
    }
    return C;
}

/* Parallel ikj: rows of C are distributed in contiguous blocks of about n/p rows. */
KERNEL double *mm_ikj_omp(const double *A, const double *B, int n) {
    double *C = (double *)calloc((size_t)n * n, sizeof(double));
    if (C == NULL) return NULL;
#pragma omp parallel
    {
        if (omp_get_thread_num() == 0) team_size_seen = omp_get_num_threads();
#pragma omp for schedule(static)
        for (int i = 0; i < n; i++) {
            double *Ci = C + (size_t)i * n;
            for (int k = 0; k < n; k++) {
                const double a = A[(size_t)i * n + k];
                const double *Bk = B + (size_t)k * n;
                OMP_SIMD
                for (int j = 0; j < n; j++) {
                    Ci[j] += a * Bk[j];
                }
            }
        }
    }
    return C;
}

/* Sequential baseline 2 (main baseline): Assignment 2 mm_blocked, body unchanged. */
KERNEL double *mm_blocked_seq(const double *A, const double *B, int n, int bs) {
    double *C = (double *)calloc((size_t)n * n, sizeof(double));
    if (C == NULL) return NULL;
    for (int ii = 0; ii < n; ii += bs) {
        const int iend = ii + bs < n ? ii + bs : n;
        for (int kk = 0; kk < n; kk += bs) {
            const int kend = kk + bs < n ? kk + bs : n;
            for (int jj = 0; jj < n; jj += bs) {
                const int jend = jj + bs < n ? jj + bs : n;
                for (int i = ii; i < iend; i++) {
                    double *Ci = C + (size_t)i * n;
                    for (int k = kk; k < kend; k++) {
                        const double a = A[(size_t)i * n + k];
                        const double *Bk = B + (size_t)k * n;
                        for (int j = jj; j < jend; j++) {
                            Ci[j] += a * Bk[j];
                        }
                    }
                }
            }
        }
    }
    return C;
}

/* Parallel blocked: the ceil(n/bs) row tiles are distributed in contiguous chunks.
 * Inside a tile the loop nest is identical to mm_blocked_seq. */
KERNEL double *mm_blocked_omp(const double *A, const double *B, int n, int bs) {
    double *C = (double *)calloc((size_t)n * n, sizeof(double));
    if (C == NULL) return NULL;
#pragma omp parallel
    {
        if (omp_get_thread_num() == 0) team_size_seen = omp_get_num_threads();
#pragma omp for schedule(static)
        for (int ii = 0; ii < n; ii += bs) {
            const int iend = ii + bs < n ? ii + bs : n;
            for (int kk = 0; kk < n; kk += bs) {
                const int kend = kk + bs < n ? kk + bs : n;
                for (int jj = 0; jj < n; jj += bs) {
                    const int jend = jj + bs < n ? jj + bs : n;
                    for (int i = ii; i < iend; i++) {
                        double *Ci = C + (size_t)i * n;
                        for (int k = kk; k < kend; k++) {
                            const double a = A[(size_t)i * n + k];
                            const double *Bk = B + (size_t)k * n;
                            OMP_SIMD
                            for (int j = jj; j < jend; j++) {
                                Ci[j] += a * Bk[j];
                            }
                        }
                    }
                }
            }
        }
    }
    return C;
}

/* Diagnostic variant: mm_blocked_omp without the simd directive. Inside the outlined
 * OpenMP function GCC -O2 cannot prove that C does not alias A or B, and its "very cheap"
 * cost model does not version the loop for aliasing, so this inner loop stays scalar
 * (results/vectorization_report.txt). Used only to quantify that effect. */
KERNEL double *mm_blocked_omp_nosimd(const double *A, const double *B, int n, int bs) {
    double *C = (double *)calloc((size_t)n * n, sizeof(double));
    if (C == NULL) return NULL;
#pragma omp parallel
    {
        if (omp_get_thread_num() == 0) team_size_seen = omp_get_num_threads();
#pragma omp for schedule(static)
        for (int ii = 0; ii < n; ii += bs) {
            const int iend = ii + bs < n ? ii + bs : n;
            for (int kk = 0; kk < n; kk += bs) {
                const int kend = kk + bs < n ? kk + bs : n;
                for (int jj = 0; jj < n; jj += bs) {
                    const int jend = jj + bs < n ? jj + bs : n;
                    for (int i = ii; i < iend; i++) {
                        double *Ci = C + (size_t)i * n;
                        for (int k = kk; k < kend; k++) {
                            const double a = A[(size_t)i * n + k];
                            const double *Bk = B + (size_t)k * n;
                            for (int j = jj; j < jend; j++) {
                                Ci[j] += a * Bk[j];
                            }
                        }
                    }
                }
            }
        }
    }
    return C;
}

enum { M_BLOCKED_SEQ, M_BLOCKED_OMP, M_IKJ_SEQ, M_IKJ_OMP, M_BLOCKED_OMP_NOSIMD, M_COUNT };
static const char *METHOD_NAME[M_COUNT] = {"blocked_seq", "blocked_omp", "ikj_seq", "ikj_omp",
                                           "blocked_omp_nosimd"};
static const int METHOD_PARALLEL[M_COUNT] = {0, 1, 0, 1, 1};

static double *multiply(int method, const double *A, const double *B, int n, int bs) {
    switch (method) {
        case M_BLOCKED_SEQ: return mm_blocked_seq(A, B, n, bs);
        case M_BLOCKED_OMP: return mm_blocked_omp(A, B, n, bs);
        case M_IKJ_SEQ: return mm_ikj_seq(A, B, n);
        case M_IKJ_OMP: return mm_ikj_omp(A, B, n);
        case M_BLOCKED_OMP_NOSIMD: return mm_blocked_omp_nosimd(A, B, n, bs);
    }
    return NULL;
}

static int method_from_name(const char *name) {
    for (int m = 0; m < M_COUNT; m++)
        if (strcmp(name, METHOD_NAME[m]) == 0) return m;
    return -1;
}

/* ======================================================================
 * Thread placement (OpenMP places are not implemented by libgomp on Windows,
 * so threads are pinned explicitly with SetThreadAffinityMask)
 *
 *   none     no pinning: the operating system places and migrates threads
 *   spread   one thread per physical core first, SMT siblings only after all cores
 *            are used (thread t -> logical processor order[t mod #LP])
 *   compact  both SMT siblings of a core before the next core
 * ====================================================================== */
static int n_cores = 0, n_lp = 0;
static int core_of_lp[MAX_LP];
static int order_spread[MAX_LP], order_compact[MAX_LP];

static int read_topology(void) {
#ifdef _WIN32
    int lps[MAX_LP][4], nl[MAX_LP];
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &len);
    char *buf = (char *)malloc(len);
    if (buf == NULL ||
        !GetLogicalProcessorInformationEx(RelationProcessorCore,
                                          (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buf, &len)) {
        free(buf);
        return 0;
    }
    n_cores = n_lp = 0;
    for (char *p = buf; p < buf + len;) {
        PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX e = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)p;
        KAFFINITY mask = e->Processor.GroupMask[0].Mask;
        nl[n_cores] = 0;
        for (int b = 0; b < (int)(8 * sizeof(KAFFINITY)); b++)
            if ((mask >> b) & 1) {
                if (nl[n_cores] < 4) lps[n_cores][nl[n_cores]++] = b;
                core_of_lp[b] = n_cores;
                n_lp++;
            }
        n_cores++;
        p += e->Size;
    }
    free(buf);
    int t = 0;
    for (int s = 0; s < 4; s++)
        for (int c = 0; c < n_cores; c++)
            if (s < nl[c]) order_spread[t++] = lps[c][s];
    t = 0;
    for (int c = 0; c < n_cores; c++)
        for (int s = 0; s < nl[c]; s++) order_compact[t++] = lps[c][s];
    return 1;
#else
    return 0;
#endif
}

static int current_lp(void) {
#ifdef _WIN32
    return (int)GetCurrentProcessorNumber();
#else
    return -1;
#endif
}

static int pin_current_thread(int lp) {
#ifdef _WIN32
    return SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << lp) != 0;
#else
    (void)lp;
    return 0;
#endif
}

/* Pins the p threads of the OpenMP pool (and the initial thread) according to the policy.
 * libgomp keeps the same pool threads for later parallel regions with the same team size. */
static int apply_affinity(const char *policy, int p, int parallel) {
    if (strcmp(policy, "none") == 0) return 1;
    if (!read_topology()) {
        fprintf(stderr, "thread pinning is only implemented for Windows\n");
        return 0;
    }
    const int *order = strcmp(policy, "spread") == 0 ? order_spread : order_compact;
    if (!parallel) return pin_current_thread(order[0]);
    int ok = 1;
#pragma omp parallel num_threads(p) reduction(&& : ok)
    ok = pin_current_thread(order[omp_get_thread_num() % n_lp]);
    return ok;
}

/* Logical processor on which each thread of a p-thread team is running now. */
static void report_placement(int p, int parallel) {
    int lp[1024];
    int team = 1;
    if (parallel) {
#pragma omp parallel num_threads(p)
        {
            volatile double x = 0.0; /* keep every thread busy for a moment */
            for (int r = 0; r < 2000000; r++) x += 1.0;
            (void)x;
            if (omp_get_thread_num() < 1024) lp[omp_get_thread_num()] = current_lp();
            if (omp_get_thread_num() == 0) team = omp_get_num_threads();
        }
    } else {
        lp[0] = current_lp();
    }
    printf("# placement,");
    for (int t = 0; t < team && t < 1024; t++) printf("%s%d", t ? ";" : "", lp[t]);
    if (read_topology()) {
        int used[MAX_LP] = {0}, cores[MAX_LP] = {0}, nlp = 0, nc = 0;
        for (int t = 0; t < team && t < 1024; t++) {
            if (lp[t] < 0 || lp[t] >= MAX_LP) continue;
            if (!used[lp[t]]++) nlp++;
            if (!cores[core_of_lp[lp[t]]]++) nc++;
        }
        printf(",distinct_lp,%d,distinct_cores,%d", nlp, nc);
    }
    printf("\n");
}

/* ======================================================================
 * Self test with exact known answers
 * ====================================================================== */
static uint64_t lcg_state = 88172645463325252ULL;
static double lcg_uniform(void) { /* uniform in [-1, 1), deterministic */
    lcg_state = lcg_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(lcg_state >> 11) * (1.0 / 9007199254740992.0) * 2.0 - 1.0;
}

static int selftest(void) {
    int ok = 1;
    char name[128];
    const int threads[6] = {1, 2, 3, 7, 16, 33};
    omp_set_dynamic(0);

    for (int m = 0; m < M_COUNT; m++) {
        for (int t = 0; t < (METHOD_PARALLEL[m] ? 6 : 1); t++) {
            omp_set_num_threads(threads[t]);
            const char *mn = METHOD_NAME[m];
            int p = METHOD_PARALLEL[m] ? threads[t] : 1;
            /* Same known-answer cases as Assignments 1 and 2. */
            double A3[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
            double B3[9] = {9, 8, 7, 6, 5, 4, 3, 2, 1};
            double E3[9] = {30, 24, 18, 84, 69, 54, 138, 114, 90};
            double *C3 = multiply(m, A3, B3, 3, 2);
            snprintf(name, sizeof name, "%s p=%d: known 3x3 product", mn, p);
            ok &= check(name, equal_exact(C3, E3, 3));
            free(C3);

            double A4[16], I4[16], Z4[16];
            for (int i = 0; i < 16; i++) {
                A4[i] = (double)(i * 3 - 7);
                I4[i] = (i % 5 == 0) ? 1.0 : 0.0;
                Z4[i] = 0.0;
            }
            double *AI = multiply(m, A4, I4, 4, 1);
            double *IA = multiply(m, I4, A4, 4, 1);
            double *AZ = multiply(m, A4, Z4, 4, 1);
            snprintf(name, sizeof name, "%s p=%d: A * I == A, I * A == A, A * 0 == 0", mn, p);
            ok &= check(name, equal_exact(AI, A4, 4) && equal_exact(IA, A4, 4) &&
                                  equal_exact(AZ, Z4, 4));
            free(AI);
            free(IA);
            free(AZ);

            double A2[4] = {1, 2, 0, 1}, B2[4] = {1, 0, 3, 1}, E2[4] = {7, 2, 3, 1};
            double *C2 = multiply(m, A2, B2, 2, 1);
            snprintf(name, sizeof name, "%s p=%d: non-commuting 2x2", mn, p);
            ok &= check(name, equal_exact(C2, E2, 2));
            free(C2);
        }
    }

    /* Random inputs whose sizes are not multiples of the tile size, with more threads than
     * tiles or rows in some cases: every kernel and thread count must be bitwise identical
     * to the Assignment 1 i-j-k kernel. */
    const int sizes[6] = {1, 7, 33, 100, 130, 257};
    const int blocks[3] = {DEFAULT_BLOCK, 16, 5};
    for (int s = 0; s < 6; s++) {
        int n = sizes[s];
        double *A = (double *)malloc((size_t)n * n * sizeof(double));
        double *B = (double *)malloc((size_t)n * n * sizeof(double));
        for (int idx = 0; idx < n * n; idx++) A[idx] = lcg_uniform();
        for (int idx = 0; idx < n * n; idx++) B[idx] = lcg_uniform();
        double *R = mm_ijk(A, B, n);
        int all = 1;
        for (int m = 0; m < M_COUNT; m++)
            for (int t = 0; t < (METHOD_PARALLEL[m] ? 6 : 1); t++)
                for (int b = 0; b < 3; b++) {
                    omp_set_num_threads(threads[t]);
                    team_size_seen = 0;
                    double *C = multiply(m, A, B, n, blocks[b]);
                    int same = memcmp(C, R, (size_t)n * n * sizeof(double)) == 0;
                    if (METHOD_PARALLEL[m] && team_size_seen != threads[t]) same = 0;
                    if (!same)
                        printf("  mismatch: %s p=%d bs=%d n=%d\n", METHOD_NAME[m], threads[t],
                               blocks[b], n);
                    all &= same;
                    free(C);
                }
        snprintf(name, sizeof name, "all kernels, p in {1,2,3,7,16,33} == ijk, n=%d", n);
        ok &= check(name, all);
        free(R);
        free(A);
        free(B);
    }

    printf("selftest %s\n", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}

/* ======================================================================
 * Benchmark
 * ====================================================================== */
static int bench(int method, int n, const char *path_a, const char *path_b, int p,
                 const char *affinity, int bs, int warmup, int reps, double min_batch_time,
                 const char *dump_path) {
    double *A = read_dense(path_a, n);
    double *B = read_dense(path_b, n);
    if (A == NULL || B == NULL) return 2;
    const char *mname = METHOD_NAME[method];
    const int parallel = METHOD_PARALLEL[method];
    if (!parallel) p = 1;

    omp_set_dynamic(0);
    omp_set_num_threads(p);
    if (!apply_affinity(affinity, p, parallel)) {
        fprintf(stderr, "cannot apply affinity %s\n", affinity);
        return 5;
    }

    printf("# omp_max_threads,%d,affinity,%s,block,%d\n", omp_get_max_threads(), affinity, bs);
    printf("method,n,threads,phase,rep,inner,batch_time_s,time_per_mult_s,checksum\n");
    fflush(stdout);

    double last_single = 0.0;
    for (int w = 0; w < warmup; w++) {
        double t0 = now_seconds();
        double *C = multiply(method, A, B, n, bs);
        double t1 = now_seconds();
        if (C == NULL) { fprintf(stderr, "allocation failed\n"); return 3; }
        last_single = t1 - t0;
        printf("%s,%d,%d,warmup,%d,1,%.9e,%.9e,%.17g\n", mname, n, p, w, last_single,
               last_single, checksum_dense(C, n));
        fflush(stdout);
        free(C);
    }

    /* Batch very short multiplications so each sample lasts >= min_batch_time. */
    int inner = 1;
    if (last_single > 0.0 && last_single < min_batch_time)
        inner = (int)ceil(min_batch_time / last_single);

    double *C = NULL;
    for (int r = 0; r < reps; r++) {
        double t0 = now_seconds();
        for (int b = 0; b < inner; b++) {
            if (C != NULL) free(C);
            C = multiply(method, A, B, n, bs);
            if (C == NULL) { fprintf(stderr, "allocation failed\n"); return 3; }
            sink = C[(size_t)b % ((size_t)n * n)]; /* every result is consumed */
        }
        double t1 = now_seconds();
        double batch = t1 - t0;
        printf("%s,%d,%d,measure,%d,%d,%.9e,%.9e,%.17g\n", mname, n, p, r, inner, batch,
               batch / inner, checksum_dense(C, n));
        fflush(stdout);
    }

    printf("# team_size_in_kernel,%d\n", parallel ? team_size_seen : 1);
    printf("# hash_c,%016llx\n", (unsigned long long)hash_dense(C, n));
    report_placement(p, parallel);
    fflush(stdout);

    if (dump_path != NULL && C != NULL && !write_dense(dump_path, C, n)) {
        fprintf(stderr, "cannot write %s\n", dump_path);
        return 4;
    }
    free(C);
    free(A);
    free(B);
    return 0;
}

static int topology(void) {
    if (!read_topology()) {
        printf("topology unavailable\n");
        return 1;
    }
    printf("physical_cores,%d,logical_processors,%d\n", n_cores, n_lp);
    printf("spread_order");
    for (int t = 0; t < n_lp; t++) printf(",%d", order_spread[t]);
    printf("\ncompact_order");
    for (int t = 0; t < n_lp; t++) printf(",%d", order_compact[t]);
    printf("\nopenmp,%d,omp_num_procs,%d,omp_max_threads,%d\n", _OPENMP, omp_get_num_procs(),
           omp_get_max_threads());
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 2 && strcmp(argv[1], "selftest") == 0) return selftest();
    if (argc >= 2 && strcmp(argv[1], "topology") == 0) return topology();
    if (argc >= 2 && strcmp(argv[1], "bench") == 0) {
        int n = 0, warmup = 1, reps = 5, bs = DEFAULT_BLOCK, method = -1, p = 1;
        double min_batch_time = 0.1;
        const char *path_a = NULL, *path_b = NULL, *dump = NULL, *affinity = "none";
        for (int i = 2; i + 1 < argc; i += 2) {
            if (strcmp(argv[i], "--method") == 0) method = method_from_name(argv[i + 1]);
            else if (strcmp(argv[i], "--n") == 0) n = atoi(argv[i + 1]);
            else if (strcmp(argv[i], "--a") == 0) path_a = argv[i + 1];
            else if (strcmp(argv[i], "--b") == 0) path_b = argv[i + 1];
            else if (strcmp(argv[i], "--threads") == 0) p = atoi(argv[i + 1]);
            else if (strcmp(argv[i], "--affinity") == 0) affinity = argv[i + 1];
            else if (strcmp(argv[i], "--block") == 0) bs = atoi(argv[i + 1]);
            else if (strcmp(argv[i], "--warmup") == 0) warmup = atoi(argv[i + 1]);
            else if (strcmp(argv[i], "--reps") == 0) reps = atoi(argv[i + 1]);
            else if (strcmp(argv[i], "--min-batch-time") == 0) min_batch_time = atof(argv[i + 1]);
            else if (strcmp(argv[i], "--dump") == 0) dump = argv[i + 1];
        }
        if (n <= 0 || method < 0 || path_a == NULL || path_b == NULL || bs <= 0 || p <= 0 ||
            (strcmp(affinity, "none") && strcmp(affinity, "spread") && strcmp(affinity, "compact"))) {
            fprintf(stderr, "invalid arguments\n");
            return 1;
        }
        return bench(method, n, path_a, path_b, p, affinity, bs, warmup, reps, min_batch_time,
                     dump);
    }
    fprintf(stderr, "usage: parallel selftest | topology | bench --method "
                    "blocked_seq|blocked_omp|ikj_seq|ikj_omp|blocked_omp_nosimd --n N --a FILE --b FILE "
                    "[--threads P] [--affinity none|spread|compact] [--block BS] [--warmup W] "
                    "[--reps R] [--min-batch-time T] [--dump FILE]\n");
    return 1;
}
