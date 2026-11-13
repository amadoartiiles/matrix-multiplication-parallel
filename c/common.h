/*
 * Shared helpers for parallel.c: monotonic timer, dense matrix file I/O,
 * checksums, a 64-bit hash of the result and the self-test reporter.
 * Dense-matrix subset of c/common.h of Assignment 2 (same timer and file format).
 *
 * File format
 *   .bin  n*n little-endian IEEE 754 binary64 values, row-major (Assignment 1 format).
 */
#ifndef MM_COMMON_H
#define MM_COMMON_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

/* Results are written here so the compiler cannot discard batched products. */
static volatile double sink;

/* ---------- monotonic timer ---------- */
static inline double now_seconds(void) {
#ifdef _WIN32
    static LARGE_INTEGER freq = {0};
    LARGE_INTEGER t;
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

/* ---------- file I/O ---------- */
static inline double *read_dense(const char *path, int n) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        fprintf(stderr, "cannot open %s\n", path);
        return NULL;
    }
    size_t count = (size_t)n * n;
    double *M = (double *)malloc(count * sizeof(double));
    if (M == NULL || fread(M, sizeof(double), count, f) != count) {
        fprintf(stderr, "cannot read %zu doubles from %s\n", count, path);
        fclose(f);
        free(M);
        return NULL;
    }
    fclose(f);
    return M;
}

static inline int write_dense(const char *path, const double *M, int n) {
    FILE *f = fopen(path, "wb");
    if (f == NULL) return 0;
    size_t count = (size_t)n * n;
    size_t written = fwrite(M, sizeof(double), count, f);
    fclose(f);
    return written == count;
}

/* ---------- checksums ---------- */
/* Row-major sequential sum of a dense matrix, used to consume the result. */
static inline double checksum_dense(const double *C, int n) {
    double s = 0.0;
    for (size_t idx = 0; idx < (size_t)n * n; idx++) s += C[idx];
    return s;
}

/* 64-bit FNV-1a over the bit patterns of the n*n doubles: two results have the same
 * hash if they are bitwise identical (collisions are possible but improbable). */
static inline uint64_t hash_dense(const double *C, int n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t idx = 0; idx < (size_t)n * n; idx++) {
        uint64_t bits;
        memcpy(&bits, &C[idx], sizeof bits);
        h ^= bits;
        h *= 1099511628211ULL;
    }
    return h;
}

/* ---------- self-test reporting ---------- */
static inline int equal_exact(const double *X, const double *Y, int n) {
    for (size_t idx = 0; idx < (size_t)n * n; idx++)
        if (X[idx] != Y[idx]) return 0;
    return 1;
}

static inline int check(const char *name, int ok) {
    printf("%-52s %s\n", name, ok ? "PASS" : "FAIL");
    return ok;
}

#endif
