// gguf.c -- minimal GGUF v3 reader for kittens-tts.
//
// Opens kitten_full.gguf as mmap, parses header + KV table + tensor
// table once, then offers:
//   - keyed scalar lookup (u32 / f32 / bool / string) for the arch KVs
//   - tensor lookup by name, returning a struct tensor that either wraps
//     the mmap'd data directly (F32 tensors) or holds a freshly-
//     allocated arena buffer with the F16 -> F32 dequant already done.
//
// The reader supports only what kittens-tts's GGUF writer
// (scripts/convert_to_gguf.py) emits: GGUF_TYPE_UINT32 /
// GGUF_TYPE_FLOAT32 / GGUF_TYPE_BOOL / GGUF_TYPE_STRING for KVs;
// GGML_TYPE_F32 / GGML_TYPE_F16 for tensors. Other types trip an
// assert at load time.
//
// Why a custom binary at all instead of loading upstream safetensors
// directly? See scripts/README.md > "Why GGUF, not load-from-
// safetensors directly?" for the full trade-off. Short version: the
// converter does ~60 name remaps, ~30 transposes, LSTM gate-order
// permutes and bias merges, fp16/fp32 per-tensor dispatch, and a
// handful of ONNX-export rename fixups — all offline, in Python.
// Moving any of that into the runtime trades cheap build-time bugs
// for expensive user-facing ones, and couples runtime code to
// upstream PyTorch's layout decisions forever. The on-disk format
// is small, mmap-friendly, and project-controlled, which is worth
// the reader's line count.
//
// Wrapped in a header guard so the only consumer (kittens.c) pulls
// this whole file in via `#include "gguf.c"` without producing a
// double-definition. See node.c.dev for the single-file-library
// convention.

#ifndef GGUF_C
#define GGUF_C

#include "tensor.c"

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>


// ---------------------------------------------------------------------------
// Public surface (was cpu/include/kt_gguf.h; folded in to keep gguf a
// single-file library that consumers pull via `#include "gguf.c"`).
// ---------------------------------------------------------------------------

struct gguf;

// Open and parse a GGUF v3 file. NULL on failure; caller can read
// gguf_last_error() to find out why.
struct gguf * gguf_open(const char * path);
void          gguf_close(struct gguf * g);
const char *  gguf_last_error(void);

// Scalar key lookups. Return 1 on found, 0 on missing.
int gguf_get_u32  (const struct gguf * g, const char * key, uint32_t * out);
int gguf_get_f32  (const struct gguf * g, const char * key, float    * out);
int gguf_get_bool (const struct gguf * g, const char * key, int      * out);

// Number of tensors in the file; mostly for iteration.
int gguf_n_tensors(const struct gguf * g);

// Name of the i'th tensor, owned by g.
const char * gguf_tensor_name(const struct gguf * g, int i);

// Look up a tensor by name and materialize it inside `arena` as a
// fully-realized fp32 struct tensor. F32 tensors are wrapped without copy
// (the underlying bytes stay in the mmap region; the struct tensor itself
// is allocated from `arena`). F16 tensors are dequantized into a fresh
// `arena`-owned buffer. Returns NULL if the tensor isn't present;
// asserts on unsupported dtypes.
struct tensor * gguf_load_tensor(const struct gguf * g, struct arena * arena,
                             const char * name);


// ---------------------------------------------------------------------------
// GGUF on-disk format (v3) summary
// ---------------------------------------------------------------------------
//
// uint32_t  magic        = 'G''G''U''F' = 0x46554747 LE
// uint32_t  version      = 3
// uint64_t  n_tensors
// uint64_t  n_kv
// kv[n_kv]                               // key/value metadata
// ti[n_tensors]                          // tensor info: name, ne[], type, off
// (pad to general.alignment, default 32)
// uint8_t   data[]                       // tensor data section
//
// Each KV:
//   uint64_t key_len; char key[key_len];
//   uint32_t type;
//   value (depends on type; ARRAY adds nested type + count + items)
//
// Each tensor info:
//   uint64_t name_len; char name[name_len];
//   uint32_t n_dims;
//   uint64_t ne[n_dims];                    // shape, innermost first
//   uint32_t dtype;                         // ggml_type
//   uint64_t offset;                        // relative to data section

#define GGUF_MAGIC    0x46554747u
#define GGUF_VERSION  3

enum gguf_type {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

enum ggml_type_min {
    GGML_TYPE_F32 = 0,
    GGML_TYPE_F16 = 1,
};


// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

struct gguf_kv {
    char *  key;          // owned (strndup'd from mmap)
    int     vtype;        // gguf_type
    // Scalar value union (used for u32 / f32 / bool / string only).
    uint32_t u32;
    float    f32;
    int      bv;
    char *   str;         // owned for STRING; NULL otherwise
};

struct gguf_ti {
    char *   name;        // owned
    int32_t  dtype;       // ggml_type (we only read F32/F16)
    int      ndim;
    int64_t  ne[4];
    size_t   offset;      // from start of data section
    size_t   nbytes;
} ;

struct gguf {
    int              fd;
    const char *     map_base;      // const view into the mmap
    size_t           map_size;
    size_t           data_off;      // absolute byte offset of data section
    struct gguf_kv * kv;
    int              n_kv;
    struct gguf_ti * ti;
    int              n_ti;
};


// ---------------------------------------------------------------------------
// Error reporting (single TLS-less slot; this loader is not thread-safe)
// ---------------------------------------------------------------------------

static char g_err[256] = "";

static void set_err(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_err, sizeof(g_err), fmt, ap);
    va_end(ap);
}

const char * gguf_last_error(void) {
    return g_err;
}

// ---------------------------------------------------------------------------
// Byte-stream cursor over the mmap region
// ---------------------------------------------------------------------------

struct cur {
    const char * base;
    size_t       size;
    size_t       pos;
    int          ok;          // 0 = ran off end, 1 = valid
};

static void cur_init(struct cur * c, const char * base, size_t size) {
    c->base = base; c->size = size; c->pos = 0; c->ok = 1;
}

static void cur_read(struct cur * c, void * dst, size_t n) {
    if (c->pos + n > c->size) { c->ok = 0; return; }
    memcpy(dst, c->base + c->pos, n);
    c->pos += n;
}

static void cur_skip(struct cur * c, size_t n) {
    if (c->pos + n > c->size) { c->ok = 0; return; }
    c->pos += n;
}

static uint32_t cur_u32(struct cur * c) {
    uint32_t v = 0; cur_read(c, &v, sizeof(v)); return v;
}

static uint64_t cur_u64(struct cur * c) {
    uint64_t v = 0; cur_read(c, &v, sizeof(v)); return v;
}

static int32_t cur_i32(struct cur * c) {
    int32_t v = 0; cur_read(c, &v, sizeof(v)); return v;
}

__attribute__((unused))
static int64_t cur_i64(struct cur * c) {
    int64_t v = 0; cur_read(c, &v, sizeof(v)); return v;
}

static float cur_f32(struct cur * c) {
    float v = 0; cur_read(c, &v, sizeof(v)); return v;
}

__attribute__((unused))
static double cur_f64(struct cur * c) {
    double v = 0; cur_read(c, &v, sizeof(v)); return v;
}

static char * cur_str_dup(struct cur * c) {
    uint64_t n = cur_u64(c);
    char * s = NULL;
    if (c->ok && c->pos + n <= c->size) {
        s = (char *)malloc(n + 1);
        if (s != NULL) {
            memcpy(s, c->base + c->pos, n);
            s[n] = '\0';
        }
        c->pos += n;
    } else {
        c->ok = 0;
    }
    return s;
}

// Size in bytes of a gguf scalar value (no length prefix). For STRING
// the length is part of the value; we return the actual size by
// peeking the u64 length prefix. The cursor is NOT advanced.
// Returns 0 on unrecognized vtype or when peeking a STRING off the
// end of the buffer. Zero is never a valid byte size for any vtype,
// so caller checks `result == 0` for "unknown / stream truncated".
static size_t gguf_scalar_size(int vtype, struct cur * peek_cur) {
    size_t bytes = 0;
    switch (vtype) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:    bytes = 1; break;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:   bytes = 2; break;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32: bytes = 4; break;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64: bytes = 8; break;
        case GGUF_TYPE_STRING: {
            if (peek_cur->pos + 8 <= peek_cur->size) {
                uint64_t n = 0;
                memcpy(&n, peek_cur->base + peek_cur->pos, 8);
                bytes = 8 + n;
            }
            break;
        }
        default: break;
    }
    return bytes;
}


// ---------------------------------------------------------------------------
// File open / close
// ---------------------------------------------------------------------------

// Skip an ARRAY value's [inner_type, count, items] payload from the
// cursor. Sets c->ok=0 on unknown inner type or stream underflow.
static void gguf_skip_array(struct cur * c) {
    uint32_t inner = cur_u32(c);
    uint64_t count = cur_u64(c);
    if (inner == GGUF_TYPE_STRING) {
        uint64_t k = 0;
        while (k < count && c->ok) {
            free(cur_str_dup(c));
            k++;
        }
    } else {
        struct cur peek = *c;
        size_t per = gguf_scalar_size((int)inner, &peek);
        if (per == 0)  {
            c->ok = 0;
        } else {
            cur_skip(c, per * count);
        }
    }
}

// Read the value portion of one KV given kv->vtype. Sets c->ok=0 on
// unknown vtype or any cursor underflow inside an ARRAY.
static void gguf_read_kv_value(struct cur * c, struct gguf_kv * kv) {
    switch (kv->vtype) {
        case GGUF_TYPE_UINT32:  kv->u32 = cur_u32(c); break;
        case GGUF_TYPE_INT32:   kv->u32 = (uint32_t)cur_i32(c); break;
        case GGUF_TYPE_FLOAT32: kv->f32 = cur_f32(c); break;
        case GGUF_TYPE_BOOL: {
            uint8_t b = 0;
            cur_read(c, &b, 1);
            kv->bv = b ? 1 : 0;
            break;
        }
        case GGUF_TYPE_STRING:  kv->str = cur_str_dup(c); break;
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:    cur_skip(c, 1); break;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:   cur_skip(c, 2); break;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:   cur_skip(c, 8); break;
        case GGUF_TYPE_FLOAT64: cur_skip(c, 8); break;
        case GGUF_TYPE_ARRAY:   gguf_skip_array(c); break;
        default:                c->ok = 0; break;
    }
}

// Parse one KV entry into kv. Sets c->ok=0 on any failure: key alloc,
// unknown vtype, ARRAY with unknown inner type, cursor underflow.
// Post-condition: kv populated and c->ok true iff the read succeeded.
static void gguf_parse_one_kv(struct cur * c, struct gguf_kv * kv) {
    kv->key   = cur_str_dup(c);
    kv->vtype = (int)cur_u32(c);
    if (c->ok && kv->key != NULL) {
        gguf_read_kv_value(c, kv);
    } else {
        c->ok = 0;
    }
}

// Walks g->n_kv KV records; piggybacks on c->ok for both stream-end
// and validation errors. Post-condition: i == g->n_kv && c->ok.
static int gguf_parse_kvs(struct gguf * g, struct cur * c) {
    int i = 0;
    while (i < g->n_kv && c->ok) {
        gguf_parse_one_kv(c, &g->kv[i]);
        i++;
    }
    int done = (i == g->n_kv && c->ok);
    if (!done) { set_err("KV parse failed (idx %d/%d)", i, g->n_kv); }
    return done;
}

// Parse one tensor-info record into t. Sets c->ok=0 on any failure.
static void gguf_parse_one_ti(struct cur * c, struct  gguf_ti * t) {
    t->name = cur_str_dup(c);
    uint32_t n_dims = cur_u32(c);
    if (c->ok && t->name != NULL && n_dims <= 4) {
        t->ndim = (int)n_dims;
        for (int d = 0; d < 4; d++) { t->ne[d] = 1; }
        for (uint32_t d = 0; d < n_dims; d++) {
            t->ne[d] = (int64_t)cur_u64(c);
        }
        t->dtype  = cur_i32(c);
        t->offset = (size_t)cur_u64(c);
    } else {
        c->ok = 0;
    }
}

static int gguf_parse_tis(struct gguf * g, struct cur * c) {
    int i = 0;
    while (i < g->n_ti && c->ok) {
        gguf_parse_one_ti(c, &g->ti[i]);
        i++;
    }
    int done = (i == g->n_ti && c->ok);
    if (!done) { set_err("tensor info parse failed (idx %d/%d)",
                         i, g->n_ti); }
    return done;
}

// Byte size of one tensor's data. Returns 0 on unsupported dtype
// (the natural sentinel - no valid tensor has 0 bytes here).
static size_t gguf_tensor_nbytes(const struct gguf_ti * t) {
    int64_t nel = 1;
    for (int d = 0; d < t->ndim; d++) { nel *= t->ne[d]; }
    size_t per = (t->dtype == GGML_TYPE_F32) ? 4u
               : (t->dtype == GGML_TYPE_F16) ? 2u : 0u;
    return (size_t)nel * per;
}

// Find the general.alignment KV; default 32 when absent.
static uint32_t gguf_general_alignment(const struct gguf * g) {
    uint32_t align = 32;
    for (int i = 0; i < g->n_kv; i++) {
        if (strcmp(g->kv[i].key, "general.alignment") == 0
            && g->kv[i].vtype == GGUF_TYPE_UINT32) {
            align = g->kv[i].u32;
        }
    }
    return align;
}

// Align the cursor to general.alignment, record g->data_off, fill
// t->nbytes for every tensor. Two passes: compute every nbytes first
// (cheap, all-or-nothing), then scan for the first 0 to surface an
// error. Post-condition: j == g->n_ti iff every dtype was supported.
static int gguf_compute_sizes(struct gguf * g, struct cur * c) {
    size_t mask = (size_t)gguf_general_alignment(g) - 1;
    g->data_off = (c->pos + mask) & ~mask;
    for (int i = 0; i < g->n_ti; i++) {
        g->ti[i].nbytes = gguf_tensor_nbytes(&g->ti[i]);
    }
    int j = 0;
    while (j < g->n_ti && g->ti[j].nbytes > 0) { j++; }
    if (j < g->n_ti) {
        set_err("tensor %s: unsupported dtype %d",
                g->ti[j].name, g->ti[j].dtype);
    }
    return j == g->n_ti;
}

static int gguf_parse(struct gguf * g) {
    struct cur c; cur_init(&c, g->map_base, g->map_size);
    uint32_t magic   = cur_u32(&c);
    uint32_t version = cur_u32(&c);
    uint64_t n_ti    = cur_u64(&c);
    uint64_t n_kv    = cur_u64(&c);
    int result = 0;
    if (!c.ok || magic != GGUF_MAGIC || version != GGUF_VERSION) {
        set_err("bad GGUF magic/version (got %08x v%u)",
                magic, version);
    } else if (n_ti > 100000 || n_kv > 10000) {
        set_err("absurd GGUF counts (%llu tensors, %llu kvs)",
                (unsigned long long)n_ti, (unsigned long long)n_kv);
    } else {
        g->n_kv = (int)n_kv;
        g->n_ti = (int)n_ti;
        g->kv = (struct gguf_kv *)calloc((size_t)g->n_kv,
                                     sizeof(struct gguf_kv));
        g->ti = (struct gguf_ti *)calloc((size_t)g->n_ti,
                                     sizeof(struct gguf_ti));
        result = gguf_parse_kvs(g, &c)
              && gguf_parse_tis(g, &c)
              && gguf_compute_sizes(g, &c);
    }
    return result;
}

struct gguf * gguf_open(const char * path) {
    struct gguf * result = NULL;
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        set_err("open(%s): errno %d", path, errno);
    } else {
        struct stat st;
        if (fstat(fd, &st) != 0) {
            set_err("fstat: errno %d", errno);
            close(fd);
        } else {
            size_t size = (size_t)st.st_size;
            void * m = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (m == MAP_FAILED) {
                set_err("mmap: errno %d", errno);
                close(fd);
            } else {
                struct gguf * g = (struct gguf *)calloc(1, sizeof(struct gguf));
                g->fd = fd;
                g->map_base = (const char *)m;
                g->map_size = size;
                if (gguf_parse(g)) {
                    result = g;
                } else {
                    // Hand off to the destructor so any kv->key /
                    // kv->str / ti->name allocated before parse died
                    // gets freed too (the manual cleanup did not).
                    gguf_close(g);
                }
            }
        }
    }
    return result;
}

void gguf_close(struct gguf * g) {
    if (g != NULL) {
        for (int i = 0; i < g->n_kv; i++) {
            free(g->kv[i].key);
            free(g->kv[i].str);
        }
        free(g->kv);
        for (int i = 0; i < g->n_ti; i++) { free(g->ti[i].name); }
        free(g->ti);
        if (g->map_base != NULL) {
            munmap((void *)g->map_base, g->map_size);
        }
        if (g->fd >= 0) { close(g->fd); }
        free(g);
    }
}


// ---------------------------------------------------------------------------
// KV lookups
// ---------------------------------------------------------------------------

static const struct gguf_kv * gguf_find_kv(const struct gguf * g,
        const char * key, int vtype) {
    const struct gguf_kv * found = NULL;
    for (int i = 0; i < g->n_kv; i++) {
        if (g->kv[i].vtype == vtype
            && strcmp(g->kv[i].key, key) == 0) {
            found = &g->kv[i];
        }
    }
    return found;
}

int gguf_get_u32(const struct gguf * g, const char * key,
                    uint32_t * out) {
    const struct gguf_kv * kv = gguf_find_kv(g, key, GGUF_TYPE_UINT32);
    int got = 0;
    if (kv != NULL) { *out = kv->u32; got = 1; }
    return got;
}

int gguf_get_f32(const struct gguf * g, const char * key, float * out) {
    const struct gguf_kv * kv = gguf_find_kv(g, key, GGUF_TYPE_FLOAT32);
    int got = 0;
    if (kv != NULL) { *out = kv->f32; got = 1; }
    return got;
}

int gguf_get_bool(const struct gguf * g, const char * key, int * out) {
    const struct gguf_kv * kv = gguf_find_kv(g, key, GGUF_TYPE_BOOL);
    int got = 0;
    if (kv != NULL) { *out = kv->bv; got = 1; }
    return got;
}


// ---------------------------------------------------------------------------
// Tensor lookup
// ---------------------------------------------------------------------------

int gguf_n_tensors(const struct gguf * g) { return g->n_ti; }

const char * gguf_tensor_name(const struct gguf * g, int i) {
    assert(i >= 0 && i < g->n_ti);
    return g->ti[i].name;
}

static const struct gguf_ti * find_tensor_info(const struct gguf * g,
                                                  const char * name) {
    const struct gguf_ti * found = NULL;
    for (int i = 0; i < g->n_ti; i++) {
        if (strcmp(g->ti[i].name, name) == 0) { found = &g->ti[i]; }
    }
    return found;
}

// Dequantize F16 source into a fresh fp32 arena buffer.
static struct tensor * gguf_dequant_f16(struct arena * a,
        const struct gguf_ti * ti, const uint16_t * src16) {
    struct tensor * t = tensor_new_nd(a, ti->ndim, ti->ne);
    int64_t total = tensor_nelements(t);
    float * dst = t->data;
    for (int64_t i = 0; i < total; i++) {
        _Float16 h;
        memcpy(&h, &src16[i], 2);
        dst[i] = (float)h;
    }
    return t;
}

struct tensor * gguf_load_tensor(const struct gguf * g, struct arena * arena,
                                const char * name) {
    const struct gguf_ti * ti = find_tensor_info(g, name);
    struct tensor * out = NULL;
    if (ti != NULL) {
        const char * src = g->map_base + g->data_off + ti->offset;
        if (ti->dtype == GGML_TYPE_F32) {
            out = tensor_wrap_nd(arena, ti->ndim, (float *)src, ti->ne);
        } else if (ti->dtype == GGML_TYPE_F16) {
            out = gguf_dequant_f16(arena, ti, (const uint16_t *)src);
        } else {
            assert(0 && "gguf: unsupported tensor dtype");
        }
        if (out != NULL) { tensor_set_name(out, name); }
    }
    return out;
}

#endif /* GGUF_READ_C */
