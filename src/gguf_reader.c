// gguf_reader.c -- one GGUF v3 reader shared by both engines.
//
// General GGUF processing (magic/version, KV table, tensor table, the
// aligned data section) lives here once. Per-tensor *processing* is the
// caller's job, handed in as a functor — so the reader stays free of any
// engine-specific tensor representation (no tensor.c, no Metal upload):
//
//   - tts/kittens.c passes a functor that wraps F32 in place and
//     dequantizes F16 into its arena (struct tensor).
//   - llm/src/llm.c passes a functor that uploads each tensor's bytes
//     to the Metal/SIMD backend, dispatching on the quant type.
//
// The functor receives the tensor's bytes either as a zero-copy pointer
// into the full-file mmap (GGUF_SRC_SHARED) or via a page-aligned slice
// the reader maps for just that tensor and releases afterward
// (GGUF_SRC_REGION). REGION lets a client upload weights without keeping
// the whole ~1 GB file resident — pair it with gguf_release_mmap to drop
// the full mapping once the tensors are consumed.
//
// Single-file library: every symbol is `static`, so the file is pulled
// into each consumer via `#include "gguf_reader.c"` (one TU each) without
// link-time collisions. Depends only on libc + POSIX mmap (macOS).

#ifndef GGUF_READER_C
#define GGUF_READER_C

#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define GGUF_MAGIC   0x46554747u // 'G','G','U','F' little-endian
#define GGUF_VERSION 3

enum gguf_value_type {
    GGUF_VT_U8    = 0,  GGUF_VT_I8    = 1,
    GGUF_VT_U16   = 2,  GGUF_VT_I16   = 3,
    GGUF_VT_U32   = 4,  GGUF_VT_I32   = 5,
    GGUF_VT_F32   = 6,  GGUF_VT_BOOL  = 7,
    GGUF_VT_STR   = 8,  GGUF_VT_ARRAY = 9,
    GGUF_VT_U64   = 10, GGUF_VT_I64   = 11,
    GGUF_VT_F64   = 12,
};

// ggml tensor type ids (subset). Callers interpret t->type themselves;
// these name the ones the engines care about.
enum gguf_tensor_type {
    GGUF_TT_F32  = 0,
    GGUF_TT_F16  = 1,
    GGUF_TT_Q8_0 = 8,
    GGUF_TT_Q4_K = 12,
    GGUF_TT_Q5_K = 13,
    GGUF_TT_Q6_K = 14,
};

struct gguf_str { uint64_t n; const char * s; };

struct gguf_value {
    uint32_t        type;
    const uint8_t * raw;       // points into mmap; type-specific layout
    uint64_t        raw_bytes; // for arrays/strings, total bytes
    // for arrays only:
    uint32_t        arr_type;
    uint64_t        arr_n;
    const uint8_t * arr_data;
};

struct gguf_kv {
    struct gguf_str   key;
    struct gguf_value v;
};

struct gguf_tensor {
    struct gguf_str name;
    uint32_t        n_dims;
    uint64_t        shape[4];
    uint32_t        type;      // ggml type id; interpreted by the caller
    uint64_t        offset;    // from tensor data section start
    const uint8_t * data;      // into full-file mmap; resolved at open
};

struct gguf {
    const uint8_t *      base;        // mmap base
    size_t               bytes;
    uint64_t             n_kv;
    uint64_t             n_tensors;
    struct gguf_kv *     kvs;
    struct gguf_tensor * tensors;
    const uint8_t *      tensor_data; // base + (aligned) end-of-info section
    uint32_t             alignment;   // from general.alignment KV; default 32
    // Owned heap copy of the model path, kept alive for the gguf's
    // lifetime so per-tensor region mmap can reopen the file after the
    // full mapping has been released.
    char *               path;
    // File offset where the tensor data section starts. Recorded so
    // region mmap survives gguf_release_mmap.
    uint64_t             data_offset;
};

// Per-tensor processing functor. `data` covers `nbytes` bytes of tensor
// `t`. With GGUF_SRC_REGION the pointer is valid only for the call.
typedef void (*gguf_tensor_fn)(void * ctx, const struct gguf_tensor * t,
                               const void * data, size_t nbytes);

enum gguf_src {
    GGUF_SRC_SHARED = 0,   // zero-copy pointer into the full-file mmap
    GGUF_SRC_REGION = 1,   // page-aligned per-tensor mmap, freed after fn
};

static int  gguf_open(struct gguf * g, const char * path);
static void gguf_close(struct gguf * g);
static void gguf_release_mmap(struct gguf * g);
static const struct gguf_kv * gguf_find_kv(const struct gguf * g, const char * k);
static const struct gguf_tensor * gguf_find_tensor(const struct gguf * g,
                                                   const char * name);

static inline uint64_t gguf_n_tensors(const struct gguf * g) {
    return g->n_tensors;
}

static inline const struct gguf_tensor * gguf_tensor_at(const struct gguf * g,
                                                        uint64_t i) {
    return &g->tensors[i];
}

// Abort on allocation failure — a truncated model load is unrecoverable.
static void * gg_oom(void * p) {
    if (p == NULL) { fprintf(stderr, "[gguf] out of memory\n"); abort(); }
    return p;
}

// Each gr_* helper reads a fixed-width value from `b` at offset *c and
// advances *c past it. memcpy because values within the mmap aren't
// aligned in general even though the mmap base is.
static uint32_t gr_u32(const uint8_t * b, size_t * c) {
    uint32_t v;
    memcpy(&v, b + *c, 4);
    *c += 4;
    return v;
}

static uint64_t gr_u64(const uint8_t * b, size_t * c) {
    uint64_t v;
    memcpy(&v, b + *c, 8);
    *c += 8;
    return v;
}

static float gr_f32(const uint8_t * b, size_t * c) {
    float v;
    memcpy(&v, b + *c, 4);
    *c += 4;
    return v;
}

static struct gguf_str gr_str(const uint8_t * b, size_t * c) {
    struct gguf_str s;
    s.n = gr_u64(b, c);
    s.s = (const char *)(b + *c);
    *c += s.n;
    return s;
}

// True when `want` more bytes are readable at cursor c in a buffer of
// `bytes`. The model file is mmap'd from a network download, so every
// length and offset it carries is untrusted; the parse below gates each
// read on this. Overflow-safe: never forms c + want, which could wrap.
static bool gr_have(size_t c, size_t want, size_t bytes) {
    return c <= bytes && want <= bytes - c;
}

// Cursor just past a value of `type` at `c`, or a value > `bytes` if the
// value (any nested length included) would run off the end or the type is
// unknown. Callers treat any returned cursor > bytes as a parse failure.
static size_t gguf_value_skip(const uint8_t * b, size_t c, uint32_t type,
                              size_t bytes) {
    size_t end  = c;
    size_t fail = bytes + 1;
    switch (type) {
        case GGUF_VT_U8:
        case GGUF_VT_I8:
        case GGUF_VT_BOOL:
            end = gr_have(c, 1, bytes) ? c + 1 : fail;
            break;
        case GGUF_VT_U16:
        case GGUF_VT_I16:
            end = gr_have(c, 2, bytes) ? c + 2 : fail;
            break;
        case GGUF_VT_U32:
        case GGUF_VT_I32:
        case GGUF_VT_F32:
            end = gr_have(c, 4, bytes) ? c + 4 : fail;
            break;
        case GGUF_VT_U64:
        case GGUF_VT_I64:
        case GGUF_VT_F64:
            end = gr_have(c, 8, bytes) ? c + 8 : fail;
            break;
        case GGUF_VT_STR: {
            if (gr_have(c, 8, bytes)) {
                uint64_t n = gr_u64(b, &end);
                end = gr_have(end, (size_t)n, bytes) ? end + (size_t)n : fail;
            } else {
                end = fail;
            }
            break;
        }
        case GGUF_VT_ARRAY: {
            if (gr_have(c, 12, bytes)) {
                uint32_t at = gr_u32(b, &end);
                uint64_t an = gr_u64(b, &end);
                uint64_t i  = 0;
                while (i < an && end <= bytes) {
                    end = gguf_value_skip(b, end, at, bytes);
                    i++;
                }
            } else {
                end = fail;
            }
            break;
        }
        default:
            end = fail;
            break;
    }
    return end;
}

// Full-file read-only mmap. The fd is closed immediately — the mapping
// keeps the file alive until munmap. `path` is copied so region mmap can
// reopen later.
static int gguf_mmap_file(struct gguf * g, const char * path) {
    int ok = 0;
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "[gguf] open(%s): errno %d\n", path, errno);
    } else {
        struct stat st;
        if (fstat(fd, &st) == 0 && st.st_size > 0) {
            void * m = mmap(NULL, (size_t)st.st_size, PROT_READ,
                            MAP_PRIVATE, fd, 0);
            if (m != MAP_FAILED) {
                g->base  = (const uint8_t *)m;
                g->bytes = (size_t)st.st_size;
                size_t plen = strlen(path) + 1;
                g->path = (char *)gg_oom(malloc(plen));
                memcpy(g->path, path, plen);
                ok = 1;
            } else {
                fprintf(stderr, "[gguf] mmap(%s): errno %d\n", path, errno);
            }
        } else {
            fprintf(stderr, "[gguf] fstat(%s): errno %d\n", path, errno);
        }
        close(fd);
    }
    return ok;
}

// Map a page-aligned slice covering just this tensor's `bytes`, returning
// the (unaligned) pointer to the tensor body. The slice origin/length go
// out via *out_slice / *out_slice_len so the caller can munmap exactly
// that range. Falls back to the shared pointer (t->data) on any failure.
static const uint8_t * gguf_region_map(const struct gguf * g,
                                       const struct gguf_tensor * t,
                                       size_t bytes,
                                       void ** out_slice,
                                       size_t * out_slice_len) {
    *out_slice     = NULL;
    *out_slice_len = 0;
    const uint8_t * body = t->data;
    long page = sysconf(_SC_PAGESIZE);
    if (g->path != NULL && bytes != 0 && page > 0) {
        uint64_t file_off = g->data_offset + t->offset;
        uint64_t page_off = file_off & ~((uint64_t)page - 1);
        size_t   slack    = (size_t)(file_off - page_off);
        size_t   mapped   = (slack + bytes + (size_t)page - 1)
                          & ~((size_t)page - 1);
        int fd = open(g->path, O_RDONLY);
        if (fd >= 0) {
            void * p = mmap(NULL, mapped, PROT_READ, MAP_PRIVATE, fd,
                            (off_t)page_off);
            close(fd);
            if (p != MAP_FAILED) {
                *out_slice     = p;
                *out_slice_len = mapped;
                body = (const uint8_t *)p + slack;
            }
        }
    }
    return body;
}

// Locate the tensor's bytes (shared pointer, or a region mmap'd just for
// this call) and hand them to `fn`. With GGUF_SRC_REGION the slice is
// released after `fn` returns. `nbytes` is supplied by the caller because
// quant/ternary block sizing (I2_S, TQ2_0, Q*_K) is engine-specific.
static void gguf_load_tensor(const struct gguf * g,
                             const struct gguf_tensor * t,
                             size_t nbytes, enum gguf_src src,
                             gguf_tensor_fn fn, void * ctx) {
    // gguf_open validated the tensor START; the byte extent depends on the
    // caller's quant sizing, so check it here. An out-of-range tensor is
    // skipped (fn not called) rather than read past EOF -- the consumer
    // then sees a missing weight and fails the load cleanly.
    bool fits = gr_have(g->data_offset + t->offset, nbytes, g->bytes);
    if (!fits) {
        fprintf(stderr, "[gguf] tensor extent out of bounds "
                "(data %llu + %zu > %zu); skipping\n",
                (unsigned long long)(g->data_offset + t->offset),
                nbytes, g->bytes);
    } else if (src == GGUF_SRC_REGION) {
        void *  slice     = NULL;
        size_t  slice_len = 0;
        const uint8_t * body = gguf_region_map(g, t, nbytes,
                                               &slice, &slice_len);
        fn(ctx, t, body, nbytes);
        if (slice != NULL && slice_len != 0) { munmap(slice, slice_len); }
    } else {
        const uint8_t * body = (t->data != NULL)
                             ? t->data : g->tensor_data + t->offset;
        fn(ctx, t, body, nbytes);
    }
}

// 1 if the file is large enough and magic + version match; advances *c
// past the 8-byte header.
static int gguf_check_header(const struct gguf * g, size_t * c) {
    bool     have_hdr = gr_have(*c, 8, g->bytes);
    uint32_t magic    = have_hdr ? gr_u32(g->base, c) : 0;
    uint32_t ver      = (have_hdr && magic == GGUF_MAGIC)
                      ? gr_u32(g->base, c) : 0;
    int ok = have_hdr && magic == GGUF_MAGIC && ver == GGUF_VERSION;
    if (!ok) {
        if (!have_hdr) {
            fprintf(stderr, "[gguf] file too small for header\n");
        } else if (magic != GGUF_MAGIC) {
            fprintf(stderr, "[gguf] bad magic %08x\n", magic);
        } else {
            fprintf(stderr, "[gguf] version %u not supported (need %u)\n",
                    ver, GGUF_VERSION);
        }
    }
    return ok;
}

// Skip a KV value at `c` (kv->v.type already read), populating the array
// fields when it is one. Returns the cursor past the value, or > bytes if
// the value runs off the end.
static size_t gguf_parse_kv_value(struct gguf * g, struct gguf_kv * kv,
                                  size_t c) {
    size_t end = g->bytes + 1;
    if (kv->v.type != GGUF_VT_ARRAY) {
        end = gguf_value_skip(g->base, c, kv->v.type, g->bytes);
    } else if (gr_have(c, 12, g->bytes)) {
        size_t hc = c;
        kv->v.arr_type = gr_u32(g->base, &hc);
        kv->v.arr_n    = gr_u64(g->base, &hc);
        kv->v.arr_data = g->base + hc;
        end = gguf_value_skip(g->base, c, GGUF_VT_ARRAY, g->bytes);
    }
    return (end <= g->bytes) ? end : g->bytes + 1;
}

// Parse one KV starting at *cursor. On any out-of-bounds read *cursor is
// left > g->bytes so the parse loop stops.
static void gguf_parse_one_kv(struct gguf * g, struct gguf_kv * kv,
                              size_t * cursor) {
    size_t c      = *cursor;
    bool   key_ok = gr_have(c, 8, g->bytes);
    if (key_ok) { kv->key = gr_str(g->base, &c); }
    bool head_ok = key_ok && c <= g->bytes && gr_have(c, 4, g->bytes);
    if (head_ok) {
        kv->v.type = gr_u32(g->base, &c);
        kv->v.raw  = g->base + c;
        c = gguf_parse_kv_value(g, kv, c);
        if (c <= g->bytes) {
            kv->v.raw_bytes = (uint64_t)((g->base + c) - kv->v.raw);
        }
    }
    *cursor = (head_ok && c <= g->bytes) ? c : g->bytes + 1;
}

// Parse the KV section starting at c; allocate + fill g->kvs[0..n_kv).
// Returns the cursor past the section, or > g->bytes on malformed input.
static size_t gguf_parse_kvs(struct gguf * g, size_t c) {
    g->kvs = (struct gguf_kv *)gg_oom(
        calloc((size_t)g->n_kv, sizeof(struct gguf_kv)));
    uint64_t i = 0;
    while (i < g->n_kv && c <= g->bytes) {
        gguf_parse_one_kv(g, &g->kvs[i], &c);
        i++;
    }
    return c;
}

static void gguf_apply_alignment_kv(struct gguf * g) {
    const char * key = "general.alignment";
    size_t kn = strlen(key);
    for (uint64_t i = 0; i < g->n_kv; i++) {
        const struct gguf_kv * kv = &g->kvs[i];
        if (kv->key.n == kn
            && memcmp(kv->key.s, key, kn) == 0
            && kv->v.type == GGUF_VT_U32) {
            size_t cc = 0;
            g->alignment = gr_u32(kv->v.raw, &cc);
        }
    }
}

// Parse one tensor descriptor starting at *cursor. On any out-of-bounds
// read, or an n_dims that would overrun shape[4], *cursor is left
// > g->bytes so the parse loop stops.
static void gguf_parse_one_tensor(struct gguf * g, struct gguf_tensor * t,
                                  size_t * cursor) {
    size_t c       = *cursor;
    bool   name_ok = gr_have(c, 8, g->bytes);
    if (name_ok) { t->name = gr_str(g->base, &c); }
    bool dimc_ok = name_ok && c <= g->bytes && gr_have(c, 4, g->bytes);
    if (dimc_ok) { t->n_dims = gr_u32(g->base, &c); }
    // GGUF tensors are at most 4-D; a larger count is corrupt and would
    // write past shape[4].
    bool dims_ok = dimc_ok && t->n_dims <= 4
                && gr_have(c, (size_t)t->n_dims * 8 + 12, g->bytes);
    if (dims_ok) {
        uint32_t d = 0;
        while (d < t->n_dims) {
            t->shape[d] = gr_u64(g->base, &c);
            d++;
        }
        t->type   = gr_u32(g->base, &c);
        t->offset = gr_u64(g->base, &c);
    }
    *cursor = dims_ok ? c : g->bytes + 1;
}

// Parse the tensor descriptor section starting at c; returns the cursor
// past the last descriptor (= start of the data section, before padding),
// or > g->bytes on malformed input.
static size_t gguf_parse_tensors(struct gguf * g, size_t c) {
    g->tensors = (struct gguf_tensor *)gg_oom(
        calloc((size_t)g->n_tensors, sizeof(struct gguf_tensor)));
    uint64_t i = 0;
    while (i < g->n_tensors && c <= g->bytes) {
        gguf_parse_one_tensor(g, &g->tensors[i], &c);
        i++;
    }
    return c;
}

static int gguf_open(struct gguf * g, const char * path) {
    memset(g, 0, sizeof(*g));
    g->alignment = 32;
    size_t c  = 0;
    int    ok = gguf_mmap_file(g, path)
             && gguf_check_header(g, &c)
             && gr_have(c, 16, g->bytes);
    if (ok) {
        g->n_tensors = gr_u64(g->base, &c);
        g->n_kv      = gr_u64(g->base, &c);
        c  = gguf_parse_kvs(g, c);
        ok = c <= g->bytes;
    }
    if (ok) {
        gguf_apply_alignment_kv(g);
        c  = gguf_parse_tensors(g, c);
        ok = c <= g->bytes;
    }
    if (ok) {
        size_t a          = g->alignment ? g->alignment : 32;
        size_t data_start = (c + a - 1) & ~(a - 1);
        ok = data_start <= g->bytes;
        if (ok) {
            g->tensor_data = g->base + data_start;
            g->data_offset = (uint64_t)data_start;
            size_t avail   = g->bytes - data_start;
            uint64_t i     = 0;
            while (ok && i < g->n_tensors) {
                // Only the start is checked here; the per-tensor byte
                // extent depends on the caller's quant-aware sizing and is
                // validated in gguf_load_tensor.
                ok = g->tensors[i].offset <= avail;
                if (ok) {
                    g->tensors[i].data = g->tensor_data + g->tensors[i].offset;
                }
                i++;
            }
        }
    }
    if (!ok) {
        gguf_close(g);
    }
    return ok ? 0 : -1;
}

// Release just the file-backed mmap, keeping the parsed metadata (kvs,
// tensors arrays) intact. Lets a client drop ~1 GB of resident pages
// once weights are uploaded and any KV strings it needs are copied out.
// macOS-relevant because madvise(MADV_DONTNEED)/MADV_FREE are no-ops on
// Darwin for read-only file-backed mappings; releasing the whole vma is
// the only way to actually drop those pages.
static void gguf_release_mmap(struct gguf * g) {
    if (g->base) {
        munmap((void *)g->base, g->bytes);
        g->base        = NULL;
        g->tensor_data = NULL;
        g->bytes       = 0;
    }
}

static void gguf_close(struct gguf * g) {
    if (g->kvs) {
        free(g->kvs);
        g->kvs = NULL;
    }
    if (g->tensors) {
        free(g->tensors);
        g->tensors = NULL;
    }
    gguf_release_mmap(g);
    if (g->path) {
        free(g->path);
        g->path = NULL;
    }
}

static const struct gguf_kv * gguf_find_kv(const struct gguf * g,
                                           const char * k) {
    const struct gguf_kv * r = NULL;
    size_t n = strlen(k);
    for (uint64_t i = 0; i < g->n_kv && r == NULL; i++) {
        if (g->kvs[i].key.n == n
            && memcmp(g->kvs[i].key.s, k, n) == 0) {
            r = &g->kvs[i];
        }
    }
    return r;
}

static const struct gguf_tensor * gguf_find_tensor(const struct gguf * g,
                                                   const char * name) {
    const struct gguf_tensor * r = NULL;
    size_t n = strlen(name);
    for (uint64_t i = 0; i < g->n_tensors && r == NULL; i++) {
        if (g->tensors[i].name.n == n
            && memcmp(g->tensors[i].name.s, name, n) == 0) {
            r = &g->tensors[i];
        }
    }
    return r;
}

static uint32_t gguf_kv_u32(const struct gguf * g, const char * k,
                            uint32_t fallback) {
    uint32_t r = fallback;
    const struct gguf_kv * kv = gguf_find_kv(g, k);
    if (kv && kv->v.type == GGUF_VT_U32) {
        size_t c = 0;
        r = gr_u32(kv->v.raw, &c);
    }
    return r;
}

static float gguf_kv_f32(const struct gguf * g, const char * k,
                         float fallback) {
    float r = fallback;
    const struct gguf_kv * kv = gguf_find_kv(g, k);
    if (kv && kv->v.type == GGUF_VT_F32) {
        size_t c = 0;
        r = gr_f32(kv->v.raw, &c);
    }
    return r;
}

#endif // GGUF_READER_C
