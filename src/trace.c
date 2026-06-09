// Trace ring + leveled logger -- implementation. See inc/trace.h.
// Standalone C; depends only on the C standard library plus the platform
// monotonic clock (QueryPerformanceCounter on Windows, clock_gettime on
// POSIX). No rt_* dependency.

#include "trace/trace.h"

#include <locale.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ----------------------------------------------------------------------------
// monotonic time, seconds (double)
// ----------------------------------------------------------------------------

#if defined(_WIN32)

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

static double trace_seconds_now(void) {
    // Frequency is fixed for the life of the process; the unsynchronized
    // first-call init races only to write the same constant.
    static LARGE_INTEGER freq;
    if (freq.QuadPart == 0) { QueryPerformanceFrequency(&freq); }
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)freq.QuadPart;
}

#else

#include <time.h>

static double trace_seconds_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

#endif

static double trace_since_start(void) {
    static double start_time = -1.0;
    double now = trace_seconds_now();
    if (start_time < 0.0) { start_time = now; }
    return now - start_time;
}

// Idempotent; first call enables `%'d` thousand-separator output where the
// platform's printf supports it (glibc/macOS; MSVC ignores the flag).
static void trace_set_numeric_locale(void) {
    static bool done = false;
    if (!done) {
        setlocale(LC_NUMERIC, "");
        setlocale(LC_NUMERIC, "en_US.UTF-8");
        done = true;
    }
}

// ----------------------------------------------------------------------------
// ring + logger
// ----------------------------------------------------------------------------

// Single-writer ring; release-store on g_trace_head makes entry contents
// visible to readers on other threads before the new head.
static struct trace_entry g_trace_ring[TRACE_RING_CAPACITY];
static _Atomic uint64_t   g_trace_head = 0;
static _Atomic int        g_trace_min_level = trace_level_error;
static _Atomic(struct trace_observer *) g_trace_observer = NULL;
static struct trace_observer g_trace_observer_slot;

// Writers run on several threads (Swift main, the turn queue, the C
// engine), so the load-fill-publish sequence in _trace_ must be
// serialized: two writers that claim the same slot would double-free
// e->message. A C11 atomic_flag spinlock keeps the ring
// single-writer-at-a-time without a pthread dependency, so the Windows
// build still compiles, and it preserves the head-as-publish-barrier
// the lock-free readers rely on (head advances only after the slot is
// filled).
static atomic_flag g_trace_lock = ATOMIC_FLAG_INIT;

void trace_set_min_level(enum trace_level lvl) {
    atomic_store_explicit(&g_trace_min_level, (int)lvl, memory_order_release);
}

enum trace_level trace_min_level(void) {
    return (enum trace_level)atomic_load_explicit(&g_trace_min_level,
                                                  memory_order_acquire);
}

static const char * trace_level_prefix(enum trace_level level) {
    static const char * names[] = { "DEBUG", "INFO ", "WARN ", "ERROR" };
    static const int n  = (int)(sizeof(names) / sizeof(names[0]));
    const int ix = (int)level - (int)trace_level_debug;
    return 0 <= ix && ix < n ? names[ix] : "?????";
}

void _trace_(enum trace_level level,
             const char * filename, int32_t line, const char * func,
             const char * format, ...) {
    trace_set_numeric_locale();
    const char * file  = filename;
    const char * slash = strrchr(file, '/');
    if (slash != NULL) { file = slash + 1; }
    // also handle Windows path separators
    const char * back = strrchr(file, '\\');
    if (back != NULL) { file = back + 1; }
    while (atomic_flag_test_and_set_explicit(&g_trace_lock,
                                             memory_order_acquire)) {
    }
    uint64_t idx = atomic_load_explicit(&g_trace_head,
                                        memory_order_relaxed);
    struct trace_entry * e = &g_trace_ring[idx % TRACE_RING_CAPACITY];
    free(e->message);
    e->message   = NULL;
    e->message_n = 0;
    e->timestamp = trace_since_start();
    e->level     = level;
    e->file      = file;
    e->function  = func;
    e->line      = line;
    // Probe length, allocate exact, format. malloc failure here is silent
    // (trace must not abort the program); readers see NULL.
    va_list ap;
    va_start(ap, format);
    va_list cp;
    va_copy(cp, ap);
    int n = vsnprintf(NULL, 0, format, cp);
    va_end(cp);
    if (n > 0) {
        e->message = (char *)malloc((size_t)n + 1);
        if (e->message != NULL) {
            vsnprintf(e->message, (size_t)n + 1, format, ap);
            e->message_n = (size_t)n;
        }
    }
    va_end(ap);
    if ((int)level >= atomic_load_explicit(&g_trace_min_level,
                                           memory_order_acquire)) {
        const char * msg = (e->message != NULL) ? e->message : "";
        fprintf(stderr, "%8.3f [%.1s] %s:%d %s %s",
                e->timestamp, trace_level_prefix(level),
                e->file, e->line, e->function,
                msg);
        if (e->message_n == 0 || e->message[e->message_n - 1] != '\n') {
            fputc('\n', stderr);
        }
    }
    atomic_store_explicit(&g_trace_head, idx + 1, memory_order_release);
    atomic_flag_clear_explicit(&g_trace_lock, memory_order_release);
    // Notify outside the lock: re-entrant tracing from an observer would
    // deadlock the spinlock, and our slot stays ours until CAPACITY more
    // writes wrap onto it.
    struct trace_observer * obs =
        atomic_load_explicit(&g_trace_observer, memory_order_acquire);
    if (obs != NULL && obs->on_trace != NULL) {
        obs->on_trace(obs, e);
    }
}

const char * trace_message(const struct trace_entry * e, size_t * out_n) {
    const char * r = NULL;
    if (e != NULL && e->message != NULL) {
        r = e->message;
        if (out_n != NULL) { *out_n = e->message_n; }
    } else if (out_n != NULL) {
        *out_n = 0;
    }
    return r;
}

void trace_subscribe(const struct trace_observer * observer) {
    if (observer == NULL) {
        atomic_store_explicit(&g_trace_observer, NULL,
                              memory_order_release);
    } else {
        g_trace_observer_slot = *observer;
        atomic_store_explicit(&g_trace_observer,
                              &g_trace_observer_slot,
                              memory_order_release);
    }
}

uint64_t trace_head(void) {
    return atomic_load_explicit(&g_trace_head, memory_order_acquire);
}

const struct trace_entry * trace_at(uint64_t index) {
    uint64_t head = atomic_load_explicit(&g_trace_head,
                                         memory_order_acquire);
    const struct trace_entry * r = NULL;
    if (index < head) {
        bool wrapped = head > TRACE_RING_CAPACITY
                    && index < head - TRACE_RING_CAPACITY;
        if (!wrapped) {
            r = &g_trace_ring[index % TRACE_RING_CAPACITY];
        }
    }
    return r;
}
