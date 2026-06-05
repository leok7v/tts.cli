// Public surface of the trace ring + leveled logger.
// Implementation lives in src/trace.c. Standalone C (no rt_* dependency);
// compiles on Windows (MSVC /std:c17 /experimental:c11atomics), macOS and
// Linux.

#ifndef TRACE_H
#define TRACE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define TRACE_RING_CAPACITY 8192 // power of two; head % CAPACITY lowers to a mask

enum trace_level {
    trace_level_debug = 0,
    trace_level_info  = 1,
    trace_level_warn  = 2,
    trace_level_error = 3,
};

// One ring slot. `file` / `function` point at literal __FILE__ / __func__
// in the read-only segment; `message` is heap-owned (NUL-terminated, no
// fixed cap). Slot reuse on wrap-around frees the prior `message` first.
struct trace_entry {
    double             timestamp; // seconds since first trace() call
    enum trace_level   level;
    const char *       file;      // basename (after last '/')
    const char *       function;  // __func__ at the call site
    int32_t            line;
    char *             message;   // owned; NUL-terminated; may be NULL
    size_t             message_n; // byte length excluding NUL
};

const char * trace_message(const struct trace_entry * e, size_t * out_n);

struct trace_observer {
    void * that;
    void (*on_trace)(const struct trace_observer * o,
                     const struct trace_entry * e);
};

#if defined(__GNUC__) || defined(__clang__)
#define TRACE_PRINTF_ATTR __attribute__((format(printf, 5, 6)))
#else
#define TRACE_PRINTF_ATTR
#endif

void _trace_(enum trace_level level,
             const char * filename, int32_t line, const char * func,
             const char * format, ...) TRACE_PRINTF_ATTR;

// Per-process minimum level for stderr mirroring. Ring captures
// everything regardless. Default trace_level_warn.
void             trace_set_min_level(enum trace_level lvl);
enum trace_level trace_min_level(void);

// trace(info, "loaded %d weights", n); — `level` is a bareword
// (debug | info | warn | error).
#define trace(level, format, ...) \
    _trace_(trace_level_##level, __FILE__, __LINE__, __func__, (format), ##__VA_ARGS__)

// abort() (not exit()) so we skip atexit/fflush of half-baked pipes;
// trace(error, ...) already flushed stderr.
#define fatal(format, ...) do { \
    trace(error, format, ##__VA_ARGS__); \
    abort(); \
} while (0)

// NULL = detach. Update is atomic; an in-flight trace() may fire the
// previous callback once more.
void trace_subscribe(const struct trace_observer * observer);

uint64_t trace_head(void);
const struct trace_entry * trace_at(uint64_t index);

#ifdef __cplusplus
}
#endif
#endif // TRACE_H
