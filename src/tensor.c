// tensor.c -- minimal CPU-only fp32 tensor library for kittens-tts.
//
// Designed as a drop-in replacement for the subset of ggml that
// kittens-tts uses (73 distinct ggml_* symbols, of which ~30 are real
// ops; the rest is graph/backend plumbing that vanishes under eager
// evaluation).
//
// Conventions
// -----------
//   - Single dtype: float32. The kitten-tts model is fp32 end-to-end;
//     quantization is a separate concern for a v2 of this library.
//   - Shape stored in ne[4], ne[0] is the FASTEST-VARYING axis
//     (matches ggml; reverses PyTorch). For NCL activations:
//     ne[0]=L, ne[1]=C, ne[2]=N (=1), ne[3]=1.
//   - Strides stored in nb[4] in BYTES. nb[0] == sizeof(float) when
//     packed.
//   - Tensors are arena-allocated. The arena owns both the tensor
//     header structs and their data slabs. A weights arena lives for
//     the model's lifetime; a scratch arena is reset at the top of
//     every kittens_synthesize call.
//   - All ops are EAGER: each call computes immediately and returns a
//     fresh tensor owned by the same arena as its first input. No
//     graph builder, no two-phase compute.
//   - Broadcasting is restricted to the three cases kittens-tts uses:
//     scalar (1-element tensor) | channel-vector (length C along
//     ne[1]) | matching shape. Anything else trips an assert.
//
// Build
// -----
//   On Apple: link with -framework Accelerate, define
//     ACCELERATE_NEW_LAPACK to silence the macOS 13.3+ deprecation
//     warning on the legacy ILP32 cblas symbols.
//   On Linux/Windows: link with -lopenblas (or any cblas-providing
//     BLAS) and -lm.
//
// Wrapped in a header guard so the only consumer (gguf.c, which is
// itself included from kittens.c) pulls this whole file in via
// `#include "tensor.c"` without producing a double-definition. See
// node.c.dev for the single-file-library convention.

#ifndef TENSOR_C
#define TENSOR_C

// Must be defined BEFORE any include that could transitively pull in
// Accelerate's cblas headers - otherwise the legacy LP64 cblas_sgemm
// declarations (deprecated in macOS 13.3) are picked up and the
// compiler issues deprecation warnings on every BLAS call site.
#if defined(__APPLE__) && !defined(ACCELERATE_NEW_LAPACK)
    #define ACCELERATE_NEW_LAPACK
#endif

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__APPLE__)
    #include <Accelerate/Accelerate.h>
#else
    #include <cblas.h>
#endif


// ---------------------------------------------------------------------------
// Public surface (was cpu/include/kt_tensor.h; folded in to keep tensor
// a single-file library that consumers pull via `#include "tensor.c"`).
// ---------------------------------------------------------------------------

#define TENSOR_MAX_DIMS 4

struct arena;

struct tensor {
    int64_t        ne[TENSOR_MAX_DIMS]; // logical extents, ne[0] inner
    int64_t        nb[TENSOR_MAX_DIMS]; // byte strides
    int            ndim;                // 1..4
    float *        data;                // 64-byte aligned, arena-owned
    struct arena * arena;               // for output allocation
    char           name[32];            // debug only; "" if unnamed
};

// Arena lifecycle
struct arena * arena_new(size_t initial_bytes);
void           arena_free(struct arena * a);
void           arena_reset(struct arena * a);
size_t         arena_used(const struct arena * a);
size_t         arena_capacity(const struct arena * a);
void           arena_set_active(struct arena * a);
struct arena * arena_get_active(void);

// Tensor creation
struct tensor * tensor_new_1d(struct arena * a, int64_t n0);
struct tensor * tensor_new_2d(struct arena * a, int64_t n0, int64_t n1);
struct tensor * tensor_new_3d(struct arena * a,
                              int64_t n0, int64_t n1, int64_t n2);
struct tensor * tensor_new_4d(struct arena * a,
                              int64_t n0, int64_t n1, int64_t n2,
                              int64_t n3);
struct tensor * tensor_new_nd(struct arena * a, int ndim,
                              const int64_t ne[TENSOR_MAX_DIMS]);
struct tensor * tensor_wrap_1d(struct arena * a, float * data,
                               int64_t n0);
struct tensor * tensor_wrap_2d(struct arena * a, float * data,
                               int64_t n0, int64_t n1);
struct tensor * tensor_wrap_3d(struct arena * a, float * data,
                               int64_t n0, int64_t n1, int64_t n2);
struct tensor * tensor_wrap_4d(struct arena * a, float * data,
                               int64_t n0, int64_t n1, int64_t n2,
                               int64_t n3);
struct tensor * tensor_wrap_nd(struct arena * a, int ndim,
                               float * data,
                               const int64_t ne[TENSOR_MAX_DIMS]);
void tensor_set_name(struct tensor * t, const char * name);

// Layout ops. `offset` is a byte offset into src->data.
struct tensor * tensor_view_1d(struct tensor * t, int64_t n0, size_t offset);
struct tensor * tensor_view_2d(struct tensor * t, int64_t n0, int64_t n1,
                               size_t nb1, size_t offset);
struct tensor * tensor_view_3d(struct tensor * t,
                               int64_t n0, int64_t n1, int64_t n2,
                               size_t nb1, size_t nb2, size_t offset);
struct tensor * tensor_reshape_2d(struct tensor * t, int64_t n0, int64_t n1);
struct tensor * tensor_reshape_3d(struct tensor * t,
                                  int64_t n0, int64_t n1, int64_t n2);
struct tensor * tensor_reshape_4d(struct tensor * t,
                                  int64_t n0, int64_t n1,
                                  int64_t n2, int64_t n3);
struct tensor * tensor_permute(struct tensor * t,
                               int p0, int p1, int p2, int p3);
struct tensor * tensor_transpose(struct tensor * t);
struct tensor * tensor_cont(struct tensor * t);
struct tensor * tensor_cont_2d(struct tensor * t, int64_t n0, int64_t n1);
struct tensor * tensor_concat(struct tensor * a, struct tensor * b,
                              int axis);
struct tensor * tensor_repeat(struct tensor * t,
                              const struct tensor * shape_like);
struct tensor * tensor_repeat_to(struct tensor * t, int ndim,
                                 int64_t n0, int64_t n1,
                                 int64_t n2, int64_t n3);
void tensor_cpy(const struct tensor * src, struct tensor * dst);
struct tensor * tensor_get_rows(struct tensor * data,
                                const int32_t * ids, int n_ids);

// Elementwise
struct tensor * tensor_add(struct tensor * x, struct tensor * y);
struct tensor * tensor_sub(struct tensor * x, struct tensor * y);
struct tensor * tensor_mul(struct tensor * x, struct tensor * y);
struct tensor * tensor_div(struct tensor * x, struct tensor * y);
struct tensor * tensor_scale(struct tensor * x, float s);
struct tensor * tensor_sigmoid(struct tensor * x);
struct tensor * tensor_tanh(struct tensor * x);
struct tensor * tensor_leaky_relu(struct tensor * x, float slope);
struct tensor * tensor_gelu_erf(struct tensor * x);
struct tensor * tensor_step(struct tensor * x);
struct tensor * tensor_sin(struct tensor * x);
struct tensor * tensor_cos(struct tensor * x);
struct tensor * tensor_exp(struct tensor * x);
struct tensor * tensor_sqrt(struct tensor * x);
struct tensor * tensor_atan2(struct tensor * y, struct tensor * x);

// Reductions / norm
struct tensor * tensor_norm(struct tensor * x, int axis, float eps);
struct tensor * tensor_softmax(struct tensor * x, int axis,
                               float scale);
struct tensor * tensor_cumsum(struct tensor * x, int axis);

// Linear algebra
struct tensor * tensor_mul_mat(struct tensor * w, struct tensor * x);
void tensor_set_sgemm_impl(int impl);
int  tensor_get_sgemm_impl(void);

// Memory diagnostics — phys_footprint is the field iOS jetsam reads.
// Peak is high-water-mark sampled at every arena_pages_alloc (mmap).
uint64_t tensor_current_phys_footprint(void);
uint64_t tensor_peak_phys_footprint(void);
void     tensor_reset_peak_phys_footprint(void);

// Conv1d family
struct tensor * tensor_conv_1d(struct tensor * w, struct tensor * x,
                               int stride, int pad, int dilation);
struct tensor * tensor_conv_1d_dw(struct tensor * w, struct tensor * x,
                                  int stride, int pad, int dilation);
struct tensor * tensor_conv_transpose_1d(struct tensor * w,
                                         struct tensor * x,
                                         int stride, int pad);
struct tensor * tensor_im2col(struct tensor * x,
                              int kernel, int stride, int pad,
                              int dilation);

// Shape inspection
int64_t tensor_nelements(const struct tensor * t);
size_t  tensor_nbytes(const struct tensor * t);
bool    tensor_is_packed(const struct tensor * t);
bool    tensor_same_shape(const struct tensor * a,
                          const struct tensor * b);


// ---------------------------------------------------------------------------
// Platform: aligned alloc (slab pages)
// ---------------------------------------------------------------------------
//
// Slabs are allocated via mmap directly (not posix_memalign / free) so
// that munmap returns the pages to the kernel IMMEDIATELY. macOS's
// libsystem_malloc caches medium/large allocations on a private free
// list, which means `posix_memalign(N) ... free()` looks like a leak
// to the OS RSS counter even after we've correctly freed it. A single
// long-sentence stage-4 peak (hundreds of MB of scratch slabs) would
// show up as multi-GB resident memory until process exit.
//
// mmap is page-aligned (16 KB on Apple Silicon); we round the request
// up to a page boundary and store the rounded size alongside the slab
// so munmap can pass the same length back. Linux/macOS both support
// MAP_ANON for anonymous private pages.

#include <sys/mman.h>
#include <unistd.h>

#define TENSOR_ALIGN 64

static size_t arena_page_size(void) {
    static size_t cached = 0;
    if (cached == 0) {
        long ps = sysconf(_SC_PAGESIZE);
        cached = ps > 0 ? (size_t)ps : 4096;
    }
    return cached;
}

// Peak phys_footprint tracker. Sampled at every arena_pages_alloc
// (mmap) — the only call that grows the process's virtual commit on
// our hot path. Updates a high-water-mark counter so callers see the
// peak that happened DURING a synthesize call, not just whatever
// task_info would return after arena_reset has freed the slabs.
// Read via tensor_peak_phys_footprint(); reset to current via
// tensor_reset_peak_phys_footprint().
#if defined(__APPLE__)
#include <mach/mach.h>
static uint64_t g_peak_phys_footprint = 0;
static uint64_t tensor_phys_footprint_now(void) {
    struct task_vm_info info;
    mach_msg_type_number_t cnt = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO,
                  (task_info_t)&info, &cnt) != KERN_SUCCESS) {
        return 0;
    }
    return info.phys_footprint;
}
static inline void tensor_sample_peak(void) {
    uint64_t cur = tensor_phys_footprint_now();
    if (cur > g_peak_phys_footprint) { g_peak_phys_footprint = cur; }
}
uint64_t tensor_peak_phys_footprint(void) { return g_peak_phys_footprint; }
uint64_t tensor_current_phys_footprint(void) {
    return tensor_phys_footprint_now();
}
void tensor_reset_peak_phys_footprint(void) {
    g_peak_phys_footprint = tensor_phys_footprint_now();
}
#else
uint64_t tensor_peak_phys_footprint(void) { return 0; }
uint64_t tensor_current_phys_footprint(void) { return 0; }
void tensor_reset_peak_phys_footprint(void) { }
static inline void tensor_sample_peak(void) { }
#endif

static void * arena_pages_alloc(size_t bytes, size_t * out_mapped) {
    const size_t pg = arena_page_size();
    size_t rounded = (bytes + pg - 1) & ~(pg - 1);
    void * result = mmap(NULL, rounded, PROT_READ | PROT_WRITE,
                         MAP_ANON | MAP_PRIVATE, -1, 0);
    if (result == MAP_FAILED) { result = NULL; rounded = 0; }
    *out_mapped = rounded;
    // Sample immediately after the mmap. This is the moment
    // phys_footprint can jump on the slab-growth path; sampling here
    // catches the new high-water mark before anything else runs.
    tensor_sample_peak();
    return result;
}

static void arena_pages_free(void * p, size_t mapped) {
    if (p != NULL && mapped > 0) {
        munmap(p, mapped);
    }
}


// ---------------------------------------------------------------------------
// Arena: chained slabs, bump cursor inside each slab.
// ---------------------------------------------------------------------------

typedef struct arena_slab {
    char *           data;       // page-aligned (mmap'd)
    size_t           capacity;   // bytes the user asked for
    size_t           mapped;     // bytes actually mmap'd (>= capacity)
    size_t           used;
    struct arena_slab * next;
} arena_slab;

struct arena {
    arena_slab * head;              // active slab, has free space
    arena_slab * first;             // never freed by reset
    size_t    initial_bytes;
};

static arena_slab * arena_slab_new(size_t bytes) {
    arena_slab * s = (arena_slab *)calloc(1, sizeof(arena_slab));
    assert(s != NULL);
    size_t mapped = 0;
    s->data = (char *)arena_pages_alloc(bytes, &mapped);
    assert(s->data != NULL);
    s->capacity = bytes;
    s->mapped = mapped;
    s->used = 0;
    s->next = NULL;
    return s;
}

static void arena_slab_free(arena_slab * s) {
    if (s != NULL) {
        arena_pages_free(s->data, s->mapped);
        free(s);
    }
}

struct arena * arena_new(size_t initial_bytes) {
    struct arena * a = (struct arena *)calloc(1, sizeof(struct arena));
    assert(a != NULL);
    size_t sz = initial_bytes < 4096 ? 4096 : initial_bytes;
    a->initial_bytes = sz;
    a->first = arena_slab_new(sz);
    a->head = a->first;
    return a;
}

void arena_free(struct arena * a) {
    if (a != NULL) {
        arena_slab * s = a->first;
        while (s != NULL) {
            arena_slab * next = s->next;
            arena_slab_free(s);
            s = next;
        }
        free(a);
    }
}

void arena_reset(struct arena * a) {
    assert(a != NULL);
    // Drop every slab past the first; reset cursor of the first.
    arena_slab * s = a->first->next;
    while (s != NULL) {
        arena_slab * next = s->next;
        arena_slab_free(s);
        s = next;
    }
    a->first->next = NULL;
    a->first->used = 0;
    a->head = a->first;
}

size_t arena_used(const struct arena * a) {
    assert(a != NULL);
    size_t total = 0;
    const arena_slab * s = a->first;
    while (s != NULL) {
        total += s->used;
        s = s->next;
    }
    return total;
}

size_t arena_capacity(const struct arena * a) {
    assert(a != NULL);
    size_t total = 0;
    const arena_slab * s = a->first;
    while (s != NULL) {
        total += s->capacity;
        s = s->next;
    }
    return total;
}

// "Active arena" target for allocating ops. See struct tensor.h.
static struct arena * g_active_arena = NULL;

void arena_set_active(struct arena * a) { g_active_arena = a; }
struct arena * arena_get_active(void)   { return g_active_arena; }

// Helper used at every op alloc site: route output to the active arena
// when one is set, otherwise to the input's own arena. Critical for
// avoiding leaks into weights_arena when ops compute on weight inputs.
static inline struct arena * arena_aout(struct arena * fallback) {
    return g_active_arena != NULL ? g_active_arena : fallback;
}

// Carve `bytes` from the arena, 64-byte aligned. Grows on demand.
//
// Growth policy: allocate a new slab sized to fit just this request
// (rounded up to a 1 MB minimum so follow-up small allocations
// coalesce into the same slab). The previous "double the capacity"
// policy nearly doubled peak memory — for a 600 MB peak need the slab
// chain was 1+2+4+...+512 = 1023 MB, which OOM-killed the app on iOS
// (jetsam at ~2 GB phys_footprint). Linear growth means peak slab
// total is ~1x actual usage.
static void * arena_alloc(struct arena * a, size_t bytes) {
    assert(a != NULL);
    size_t rounded = (bytes + (TENSOR_ALIGN - 1)) & ~((size_t)(TENSOR_ALIGN - 1));
    arena_slab * s = a->head;
    size_t aligned_used = (s->used + (TENSOR_ALIGN - 1))
                          & ~((size_t)(TENSOR_ALIGN - 1));
    if (aligned_used + rounded > s->capacity) {
        const size_t MIN_NEW_SLAB = (size_t)1 << 20;   // 1 MB
        size_t new_cap = rounded + TENSOR_ALIGN;
        if (new_cap < MIN_NEW_SLAB) { new_cap = MIN_NEW_SLAB; }
        arena_slab * ns = arena_slab_new(new_cap);
        s->next = ns;
        a->head = ns;
        s = ns;
        aligned_used = 0;
    }
    void * out = s->data + aligned_used;
    s->used = aligned_used + rounded;
    return out;
}

// ---------------------------------------------------------------------------
// Shape helpers
// ---------------------------------------------------------------------------

static void tensor_set_packed_strides(struct tensor * t) {
    t->nb[0] = sizeof(float);
    for (int i = 1; i < TENSOR_MAX_DIMS; i++) {
        t->nb[i] = t->nb[i - 1] * t->ne[i - 1];
    }
}

int64_t tensor_nelements(const struct tensor * t) {
    assert(t != NULL);
    int64_t n = 1;
    for (int i = 0; i < TENSOR_MAX_DIMS; i++) {
        n *= t->ne[i];
    }
    return n;
}

size_t tensor_nbytes(const struct tensor * t) {
    return (size_t)tensor_nelements(t) * sizeof(float);
}

bool tensor_is_packed(const struct tensor * t) {
    bool packed = true;
    int64_t expected = sizeof(float);
    for (int i = 0; i < TENSOR_MAX_DIMS; i++) {
        if (t->ne[i] > 1 && t->nb[i] != expected) {
            packed = false;
        }
        expected *= t->ne[i];
    }
    return packed;
}

bool tensor_same_shape(const struct tensor * a, const struct tensor * b) {
    bool same = (a->ndim == b->ndim);
    for (int i = 0; i < TENSOR_MAX_DIMS; i++) {
        if (a->ne[i] != b->ne[i]) { same = false; }
    }
    return same;
}


// ---------------------------------------------------------------------------
// Tensor creation
// ---------------------------------------------------------------------------

static struct tensor * tensor_alloc_header(struct arena * a) {
    struct tensor * t = (struct tensor *)arena_alloc(a, sizeof(struct tensor));
    memset(t, 0, sizeof(*t));
    t->arena = a;
    for (int i = 0; i < TENSOR_MAX_DIMS; i++) { t->ne[i] = 1; }
    return t;
}

static struct tensor * tensor_alloc_with_data(struct arena * a,
                                      int ndim,
                                      int64_t n0, int64_t n1,
                                      int64_t n2, int64_t n3) {
    struct tensor * t = tensor_alloc_header(a);
    t->ndim = ndim;
    t->ne[0] = n0; t->ne[1] = n1; t->ne[2] = n2; t->ne[3] = n3;
    tensor_set_packed_strides(t);
    size_t bytes = (size_t)tensor_nelements(t) * sizeof(float);
    t->data = (float *)arena_alloc(a, bytes);
    return t;
}

struct tensor * tensor_new_1d(struct arena * a, int64_t n0) {
    return tensor_alloc_with_data(a, 1, n0, 1, 1, 1);
}

struct tensor * tensor_new_2d(struct arena * a, int64_t n0, int64_t n1) {
    return tensor_alloc_with_data(a, 2, n0, n1, 1, 1);
}

struct tensor * tensor_new_3d(struct arena * a, int64_t n0, int64_t n1,
                              int64_t n2) {
    return tensor_alloc_with_data(a, 3, n0, n1, n2, 1);
}

struct tensor * tensor_new_4d(struct arena * a, int64_t n0, int64_t n1,
                              int64_t n2, int64_t n3) {
    return tensor_alloc_with_data(a, 4, n0, n1, n2, n3);
}

struct tensor * tensor_new_nd(struct arena * a, int ndim,
                              const int64_t ne[TENSOR_MAX_DIMS]) {
    int64_t n0 = ne[0], n1 = ne[1], n2 = ne[2], n3 = ne[3];
    struct tensor * t = NULL;
    if      (ndim == 1) { t = tensor_alloc_with_data(a, 1, n0, 1,  1,  1);  }
    else if (ndim == 2) { t = tensor_alloc_with_data(a, 2, n0, n1, 1,  1);  }
    else if (ndim == 3) { t = tensor_alloc_with_data(a, 3, n0, n1, n2, 1);  }
    else                { t = tensor_alloc_with_data(a, 4, n0, n1, n2, n3); }
    return t;
}

struct tensor * tensor_wrap_1d(struct arena * a, float * data, int64_t n0) {
    struct tensor * t = tensor_alloc_header(a);
    t->ndim = 1;
    t->ne[0] = n0;
    tensor_set_packed_strides(t);
    t->data = data;
    return t;
}

struct tensor * tensor_wrap_2d(struct arena * a, float * data,
                       int64_t n0, int64_t n1) {
    struct tensor * t = tensor_alloc_header(a);
    t->ndim = 2;
    t->ne[0] = n0; t->ne[1] = n1;
    tensor_set_packed_strides(t);
    t->data = data;
    return t;
}

struct tensor * tensor_wrap_3d(struct arena * a, float * data,
                       int64_t n0, int64_t n1, int64_t n2) {
    struct tensor * t = tensor_alloc_header(a);
    t->ndim = 3;
    t->ne[0] = n0; t->ne[1] = n1; t->ne[2] = n2;
    tensor_set_packed_strides(t);
    t->data = data;
    return t;
}

struct tensor * tensor_wrap_4d(struct arena * a, float * data,
                       int64_t n0, int64_t n1, int64_t n2, int64_t n3) {
    struct tensor * t = tensor_alloc_header(a);
    t->ndim = 4;
    t->ne[0] = n0; t->ne[1] = n1; t->ne[2] = n2; t->ne[3] = n3;
    tensor_set_packed_strides(t);
    t->data = data;
    return t;
}

struct tensor * tensor_wrap_nd(struct arena * a, int ndim, float * data,
                       const int64_t ne[TENSOR_MAX_DIMS]) {
    int64_t n0 = ne[0], n1 = ne[1], n2 = ne[2], n3 = ne[3];
    struct tensor * t = NULL;
    if      (ndim == 1) { t = tensor_wrap_1d(a, data, n0); }
    else if (ndim == 2) { t = tensor_wrap_2d(a, data, n0, n1); }
    else if (ndim == 3) { t = tensor_wrap_3d(a, data, n0, n1, n2); }
    else                { t = tensor_wrap_4d(a, data, n0, n1, n2, n3); }
    return t;
}

void tensor_set_name(struct tensor * t, const char * name) {
    assert(t != NULL && name != NULL);
    size_t n = strlen(name);
    if (n > sizeof(t->name) - 1) { n = sizeof(t->name) - 1; }
    memcpy(t->name, name, n);
    t->name[n] = '\0';
}

// ---------------------------------------------------------------------------
// Layout ops
// ---------------------------------------------------------------------------

// Build a header sharing src's data buffer (view). src and out are
// distinct here: src is the input we view into, out the new header.
static struct tensor * tensor_make_view(struct tensor * src, int ndim,
                                        int64_t n0, int64_t n1,
                                        int64_t n2, int64_t n3,
                                        int64_t nb0, int64_t nb1,
                                        int64_t nb2, int64_t nb3,
                                        size_t offset) {
    struct tensor * out = tensor_alloc_header(arena_aout(src->arena));
    out->ndim = ndim;
    out->ne[0] = n0;  out->ne[1] = n1;  out->ne[2] = n2;  out->ne[3] = n3;
    out->nb[0] = nb0; out->nb[1] = nb1; out->nb[2] = nb2; out->nb[3] = nb3;
    out->data = (float *)((char *)src->data + offset);
    return out;
}

struct tensor * tensor_view_1d(struct tensor * t, int64_t n0, size_t offset) {
    return tensor_make_view(t, 1, n0, 1, 1, 1,
                            t->nb[0], t->nb[0] * n0,
                            t->nb[0] * n0, t->nb[0] * n0,
                            offset);
}

struct tensor * tensor_view_2d(struct tensor * t, int64_t n0, int64_t n1,
                               size_t nb1, size_t offset) {
    return tensor_make_view(t, 2, n0, n1, 1, 1,
                            t->nb[0], (int64_t)nb1,
                            (int64_t)nb1 * n1, (int64_t)nb1 * n1,
                            offset);
}

struct tensor * tensor_view_3d(struct tensor * t,
                               int64_t n0, int64_t n1, int64_t n2,
                               size_t nb1, size_t nb2, size_t offset) {
    return tensor_make_view(t, 3, n0, n1, n2, 1,
                            t->nb[0], (int64_t)nb1, (int64_t)nb2,
                            (int64_t)nb2 * n2,
                            offset);
}

struct tensor * tensor_reshape_2d(struct tensor * src, int64_t n0, int64_t n1) {
    assert(tensor_is_packed(src));
    assert(n0 * n1 == tensor_nelements(src));
    struct tensor * t = tensor_alloc_header(arena_aout(src->arena));
    t->ndim = 2;
    t->ne[0] = n0; t->ne[1] = n1;
    tensor_set_packed_strides(t);
    t->data = src->data;
    return t;
}

struct tensor * tensor_reshape_3d(struct tensor * src,
                          int64_t n0, int64_t n1, int64_t n2) {
    assert(tensor_is_packed(src));
    assert(n0 * n1 * n2 == tensor_nelements(src));
    struct tensor * t = tensor_alloc_header(arena_aout(src->arena));
    t->ndim = 3;
    t->ne[0] = n0; t->ne[1] = n1; t->ne[2] = n2;
    tensor_set_packed_strides(t);
    t->data = src->data;
    return t;
}

struct tensor * tensor_reshape_4d(struct tensor * src,
                                  int64_t n0, int64_t n1,
                                  int64_t n2, int64_t n3) {
    assert(tensor_is_packed(src));
    assert(n0 * n1 * n2 * n3 == tensor_nelements(src));
    struct tensor * t = tensor_alloc_header(arena_aout(src->arena));
    t->ndim = 4;
    t->ne[0] = n0; t->ne[1] = n1; t->ne[2] = n2; t->ne[3] = n3;
    tensor_set_packed_strides(t);
    t->data = src->data;
    return t;
}

struct tensor * tensor_permute(struct tensor * src, int p0, int p1, int p2,
                               int p3) {
    assert(p0 >= 0 && p0 < 4);
    assert(p1 >= 0 && p1 < 4);
    assert(p2 >= 0 && p2 < 4);
    assert(p3 >= 0 && p3 < 4);
    struct tensor * t = tensor_alloc_header(arena_aout(src->arena));
    t->ndim = src->ndim;
    int perm[4] = { p0, p1, p2, p3 };
    for (int i = 0; i < TENSOR_MAX_DIMS; i++) {
        t->ne[i] = src->ne[perm[i]];
        t->nb[i] = src->nb[perm[i]];
    }
    t->data = src->data;
    return t;
}

struct tensor * tensor_transpose(struct tensor * src) {
    return tensor_permute(src, 1, 0, 2, 3);
}

// Copy a strided tensor into a fresh packed buffer.
struct tensor * tensor_cont(struct tensor * src) {
    struct tensor * t = tensor_alloc_with_data(arena_aout(src->arena),
                                       src->ndim,
                                       src->ne[0], src->ne[1],
                                       src->ne[2], src->ne[3]);
    const int64_t n0 = src->ne[0];
    const int64_t n1 = src->ne[1];
    const int64_t n2 = src->ne[2];
    const int64_t n3 = src->ne[3];
    const int64_t s0 = src->nb[0];
    const int64_t s1 = src->nb[1];
    const int64_t s2 = src->nb[2];
    const int64_t s3 = src->nb[3];
    const char * sb = (const char *)src->data;
    float * dst = t->data;
    for (int64_t i3 = 0; i3 < n3; i3++) {
        for (int64_t i2 = 0; i2 < n2; i2++) {
            for (int64_t i1 = 0; i1 < n1; i1++) {
                const char * row = sb + i3 * s3 + i2 * s2 + i1 * s1;
                if (s0 == (int64_t)sizeof(float)) {
                    memcpy(dst, row, (size_t)n0 * sizeof(float));
                    dst += n0;
                } else {
                    for (int64_t i0 = 0; i0 < n0; i0++) {
                        *dst++ = *(const float *)(row + i0 * s0);
                    }
                }
            }
        }
    }
    return t;
}

struct tensor * tensor_cont_2d(struct tensor * src, int64_t n0, int64_t n1) {
    struct tensor * packed = tensor_cont(src);
    return tensor_reshape_2d(packed, n0, n1);
}

void tensor_cpy(const struct tensor * src, struct tensor * dst) {
    assert(tensor_same_shape(src, dst));
    const int64_t n0 = src->ne[0];
    const int64_t n1 = src->ne[1];
    const int64_t n2 = src->ne[2];
    const int64_t n3 = src->ne[3];
    for (int64_t i3 = 0; i3 < n3; i3++) {
        for (int64_t i2 = 0; i2 < n2; i2++) {
            for (int64_t i1 = 0; i1 < n1; i1++) {
                const char * srow = (const char *)src->data
                                  + i3 * src->nb[3]
                                  + i2 * src->nb[2]
                                  + i1 * src->nb[1];
                char * drow = (char *)dst->data
                            + i3 * dst->nb[3]
                            + i2 * dst->nb[2]
                            + i1 * dst->nb[1];
                for (int64_t i0 = 0; i0 < n0; i0++) {
                    *(float *)(drow + i0 * dst->nb[0]) =
                        *(const float *)(srow + i0 * src->nb[0]);
                }
            }
        }
    }
}

// Concat along axis. Both inputs must agree on all other axes and be
// packed (for v1; strided concat is harder and not needed yet).
struct tensor * tensor_concat(struct tensor * a, struct tensor * b, int axis) {
    assert(axis >= 0 && axis < TENSOR_MAX_DIMS);
    assert(a->ndim == b->ndim);
    for (int i = 0; i < TENSOR_MAX_DIMS; i++) {
        if (i != axis) { assert(a->ne[i] == b->ne[i]); }
    }
    assert(tensor_is_packed(a) && tensor_is_packed(b));
    int64_t out_ne[TENSOR_MAX_DIMS];
    for (int i = 0; i < TENSOR_MAX_DIMS; i++) {
        out_ne[i] = (i == axis) ? (a->ne[i] + b->ne[i]) : a->ne[i];
    }
    struct tensor * t = tensor_alloc_with_data(arena_aout(a->arena), a->ndim,
                                       out_ne[0], out_ne[1],
                                       out_ne[2], out_ne[3]);
    // Iterate over the "outer" axes (> axis), then within each slab
    // copy A's row then B's row along `axis`.
    int64_t outer = 1;
    for (int i = axis + 1; i < TENSOR_MAX_DIMS; i++) { outer *= a->ne[i]; }
    int64_t inner = 1;
    for (int i = 0; i < axis; i++) { inner *= a->ne[i]; }
    size_t row_a = (size_t)inner * (size_t)a->ne[axis] * sizeof(float);
    size_t row_b = (size_t)inner * (size_t)b->ne[axis] * sizeof(float);
    char * dst = (char *)t->data;
    const char * sa = (const char *)a->data;
    const char * sb = (const char *)b->data;
    for (int64_t k = 0; k < outer; k++) {
        memcpy(dst, sa + k * row_a, row_a);
        dst += row_a;
        memcpy(dst, sb + k * row_b, row_b);
        dst += row_b;
    }
    return t;
}

// Same as tensor_repeat but takes the target shape as ints. Avoids the
// caller having to materialize a full template tensor (which would
// allocate the full output buffer just to be ignored).
struct tensor * tensor_repeat_to(struct tensor * src, int ndim,
                         int64_t n0, int64_t n1,
                         int64_t n2, int64_t n3) {
    struct tensor template;
    memset(&template, 0, sizeof(template));
    template.ndim = ndim;
    template.ne[0] = n0; template.ne[1] = n1;
    template.ne[2] = n2; template.ne[3] = n3;
    return tensor_repeat(src, &template);
}

// repeat: tile src to match shape_like. Each src axis must be 1 or
// equal to shape_like's. (For v1: trivial broadcast tiling.)
struct tensor * tensor_repeat(struct tensor * src,
                              const struct tensor * shape_like) {
    for (int i = 0; i < TENSOR_MAX_DIMS; i++) {
        assert(src->ne[i] == 1 || src->ne[i] == shape_like->ne[i]);
    }
    struct tensor * t = tensor_alloc_with_data(arena_aout(src->arena),
                                       shape_like->ndim,
                                       shape_like->ne[0],
                                       shape_like->ne[1],
                                       shape_like->ne[2],
                                       shape_like->ne[3]);
    const int64_t n0 = t->ne[0], n1 = t->ne[1];
    const int64_t n2 = t->ne[2], n3 = t->ne[3];
    const int64_t r0 = src->ne[0] == 1 ? 0 : src->nb[0];
    const int64_t r1 = src->ne[1] == 1 ? 0 : src->nb[1];
    const int64_t r2 = src->ne[2] == 1 ? 0 : src->nb[2];
    const int64_t r3 = src->ne[3] == 1 ? 0 : src->nb[3];
    float * dst = t->data;
    const char * sb = (const char *)src->data;
    for (int64_t i3 = 0; i3 < n3; i3++) {
        for (int64_t i2 = 0; i2 < n2; i2++) {
            for (int64_t i1 = 0; i1 < n1; i1++) {
                const char * row = sb + i3 * r3 + i2 * r2 + i1 * r1;
                for (int64_t i0 = 0; i0 < n0; i0++) {
                    *dst++ = *(const float *)(row + i0 * r0);
                }
            }
        }
    }
    return t;
}

struct tensor * tensor_get_rows(struct tensor * data,
                                const int32_t * ids, int n_ids) {
    assert(data->ndim == 2);
    assert(tensor_is_packed(data));
    const int64_t embed = data->ne[0];
    const int64_t vocab = data->ne[1];
    struct tensor * t = tensor_new_2d(arena_aout(data->arena), embed,
                              (int64_t)n_ids);
    for (int i = 0; i < n_ids; i++) {
        int32_t row = ids[i];
        assert(row >= 0 && (int64_t)row < vocab);
        memcpy(t->data + (size_t)i * (size_t)embed,
               data->data + (size_t)row * (size_t)embed,
               (size_t)embed * sizeof(float));
    }
    return t;
}

// ---------------------------------------------------------------------------
// Broadcasting kernels (ggml-style: y.ne[i] must be 1 or equal to x.ne[i]).
// ---------------------------------------------------------------------------

// `y` is broadcastable against `x` iff for every axis i, y.ne[i] is 1
// (broadcast along that axis) or y.ne[i] equals x.ne[i]. The loop's
// final `i` value IS the post-condition: i == TENSOR_MAX_DIMS iff every
// axis was compatible.
static int tensor_broadcastable(const struct tensor * x,
                                const struct tensor * y) {
    int i = 0;
    while (i < TENSOR_MAX_DIMS
           && (y->ne[i] == 1 || y->ne[i] == x->ne[i])) {
        i++;
    }
    return i == TENSOR_MAX_DIMS;
}

enum tensor_bin {
    TENSOR_BIN_ADD, TENSOR_BIN_SUB, TENSOR_BIN_MUL, TENSOR_BIN_DIV
};

// Vector-vector for two contiguous buffers via vDSP (Apple's hand-tuned
// SIMD kernels — fast even at -O0 because the work is in libBLAS).
static void tensor_vec_vv(enum tensor_bin op, const float * x, const float * y,
                      float * out, int64_t n) {
    const vDSP_Length N = (vDSP_Length)n;
    switch (op) {
        case TENSOR_BIN_ADD: vDSP_vadd(x, 1, y, 1, out, 1, N); break;
        // vDSP_vsub computes A - B with arg order (B, IB, A, IA, ...).
        case TENSOR_BIN_SUB: vDSP_vsub(y, 1, x, 1, out, 1, N); break;
        case TENSOR_BIN_MUL: vDSP_vmul(x, 1, y, 1, out, 1, N); break;
        // vDSP_vdiv computes A / B with arg order (B, IB, A, IA, ...).
        case TENSOR_BIN_DIV: vDSP_vdiv(y, 1, x, 1, out, 1, N); break;
    }
}

// Vector-scalar (y is broadcast as a single value) via vDSP.
static void tensor_vec_vs(enum tensor_bin op, const float * x, float s,
                      float * out, int64_t n) {
    const vDSP_Length N = (vDSP_Length)n;
    float arg;
    switch (op) {
        case TENSOR_BIN_ADD: vDSP_vsadd(x, 1, &s, out, 1, N); break;
        case TENSOR_BIN_SUB: arg = -s;
                         vDSP_vsadd(x, 1, &arg, out, 1, N); break;
        case TENSOR_BIN_MUL: vDSP_vsmul(x, 1, &s, out, 1, N); break;
        case TENSOR_BIN_DIV: arg = 1.0f / s;
                         vDSP_vsmul(x, 1, &arg, out, 1, N); break;
    }
}

// Scalar fallback for the strided / broadcast path. Switch dispatch
// (no function pointer) so -O0 doesn't pay a call per element.
static inline float tensor_scalar_op(enum tensor_bin op, float a, float b) {
    switch (op) {
        case TENSOR_BIN_ADD: return a + b;
        case TENSOR_BIN_SUB: return a - b;
        case TENSOR_BIN_MUL: return a * b;
        case TENSOR_BIN_DIV: return a / b;
    }
    return 0.0f;
}

static struct tensor * tensor_apply_binop(struct tensor * x, struct tensor * y,
                                  enum tensor_bin op) {
    assert(tensor_broadcastable(x, y));
    struct tensor * out = tensor_alloc_with_data(arena_aout(x->arena), x->ndim,
                                         x->ne[0], x->ne[1],
                                         x->ne[2], x->ne[3]);
    const int64_t n0 = x->ne[0], n1 = x->ne[1];
    const int64_t n2 = x->ne[2], n3 = x->ne[3];
    const float * xb = x->data;
    const float * yb = y->data;
    float * ob = out->data;
    int64_t total = tensor_nelements(x);
    // Fast path 1: identical packed shape -> vDSP vector-vector kernel.
    if (tensor_same_shape(x, y) && tensor_is_packed(x) && tensor_is_packed(y)) {
        tensor_vec_vv(op, xb, yb, ob, total);
    } else if (tensor_nelements(y) == 1) {
        // Fast path 2: y is scalar -> vDSP vector-scalar kernel.
        if (tensor_is_packed(x)) {
            tensor_vec_vs(op, xb, yb[0], ob, total);
        } else {
            float s = yb[0];
            for (int64_t i3 = 0; i3 < n3; i3++) {
              for (int64_t i2 = 0; i2 < n2; i2++) {
                for (int64_t i1 = 0; i1 < n1; i1++) {
                  for (int64_t i0 = 0; i0 < n0; i0++) {
                    const float * xp = (const float *)
                        ((const char *)xb + i3 * x->nb[3]
                                          + i2 * x->nb[2]
                                          + i1 * x->nb[1]
                                          + i0 * x->nb[0]);
                    int64_t oi = ((i3 * n2 + i2) * n1 + i1) * n0 + i0;
                    ob[oi] = tensor_scalar_op(op, *xp, s);
                  }
                }
              }
            }
        }
    } else {
        // Fast path 3: general broadcast where both x and y are packed
        // along ne[0] (the inner axis). For each row, dispatch the
        // inner kernel through vDSP — vector-vector when y.ne[0] equals
        // x.ne[0], vector-scalar when y broadcasts along ne[0].
        const int64_t ys0 = (y->ne[0] == 1) ? 0 : y->nb[0];
        const int64_t ys1 = (y->ne[1] == 1) ? 0 : y->nb[1];
        const int64_t ys2 = (y->ne[2] == 1) ? 0 : y->nb[2];
        const int64_t ys3 = (y->ne[3] == 1) ? 0 : y->nb[3];
        const int x_inner_packed = (x->nb[0] == (int64_t)sizeof(float));
        const int y_inner_packed_or_scalar =
            (ys0 == 0) || (ys0 == (int64_t)sizeof(float));
        for (int64_t i3 = 0; i3 < n3; i3++) {
          for (int64_t i2 = 0; i2 < n2; i2++) {
            for (int64_t i1 = 0; i1 < n1; i1++) {
              const char * xrow = (const char *)xb
                                + i3 * x->nb[3]
                                + i2 * x->nb[2]
                                + i1 * x->nb[1];
              const char * yrow = (const char *)yb
                                + i3 * ys3 + i2 * ys2 + i1 * ys1;
              float * orow = ob
                           + ((i3 * n2 + i2) * n1 + i1) * n0;
              if (x_inner_packed && y_inner_packed_or_scalar) {
                  if (ys0 == 0) {
                      tensor_vec_vs(op, (const float *)xrow,
                                *(const float *)yrow, orow, n0);
                  } else {
                      tensor_vec_vv(op, (const float *)xrow,
                                (const float *)yrow, orow, n0);
                  }
              } else if (ys0 == 0) {
                  float yv = *(const float *)yrow;
                  for (int64_t i0 = 0; i0 < n0; i0++) {
                      float xv = *(const float *)(xrow + i0 * x->nb[0]);
                      orow[i0] = tensor_scalar_op(op, xv, yv);
                  }
              } else {
                  for (int64_t i0 = 0; i0 < n0; i0++) {
                      float xv = *(const float *)(xrow + i0 * x->nb[0]);
                      float yv = *(const float *)(yrow + i0 * ys0);
                      orow[i0] = tensor_scalar_op(op, xv, yv);
                  }
              }
            }
          }
        }
    }
    return out;
}

struct tensor * tensor_add(struct tensor * x, struct tensor * y) {
    return tensor_apply_binop(x, y, TENSOR_BIN_ADD);
}
struct tensor * tensor_sub(struct tensor * x, struct tensor * y) {
    return tensor_apply_binop(x, y, TENSOR_BIN_SUB);
}
struct tensor * tensor_mul(struct tensor * x, struct tensor * y) {
    return tensor_apply_binop(x, y, TENSOR_BIN_MUL);
}
struct tensor * tensor_div(struct tensor * x, struct tensor * y) {
    return tensor_apply_binop(x, y, TENSOR_BIN_DIV);
}

// ---------------------------------------------------------------------------
// Elementwise unary
// ---------------------------------------------------------------------------

enum tensor_unop {
    TENSOR_U_SCALE, TENSOR_U_SIGMOID, TENSOR_U_TANH, TENSOR_U_LRELU,
    TENSOR_U_GELU,  TENSOR_U_STEP,    TENSOR_U_SIN,  TENSOR_U_COS,
    TENSOR_U_EXP,   TENSOR_U_SQRT
};

// Vector-only fast paths via vForce (transcendentals) and vDSP (linear
// ops). These are SIMD ASM kernels in Accelerate, so they hit memory
// bandwidth even when the calling code is built at -O0.
static void tensor_vec_unary(enum tensor_unop op, const float * x, float * out,
                             int64_t n, float param) {
    int len = (int)n;
    const vDSP_Length N = (vDSP_Length)n;
    switch (op) {
    case TENSOR_U_SCALE: {
        float s = param;
        vDSP_vsmul(x, 1, &s, out, 1, N);
        break;
    }
    case TENSOR_U_SIGMOID: {
        // sigmoid(x) = 1 / (1 + exp(-x))
        // Implemented as: out = -x; out = exp(out); out = 1+out;
        // out = 1/out — three vForce/vDSP passes; still beats per-elem
        // function calls at -O0.
        vDSP_vneg(x, 1, out, 1, N);
        vvexpf(out, out, &len);
        float one = 1.0f;
        vDSP_vsadd(out, 1, &one, out, 1, N);
        vvrecf(out, out, &len);
        break;
    }
    case TENSOR_U_TANH: vvtanhf(out, x, &len); break;
    case TENSOR_U_LRELU: {
        // leaky_relu(x, s) = max(x, x*s) when s in (0, 1).
        // out = x*s, then out = max(x, x*s).
        vDSP_vsmul(x, 1, &param, out, 1, N);
        vDSP_vmax(x, 1, out, 1, out, 1, N);
        break;
    }
    case TENSOR_U_GELU: {
        // 0.5 * x * (1 + erf(x / sqrt(2)))
        const float inv_sqrt2 = (float)M_SQRT1_2;
        vDSP_vsmul(x, 1, &inv_sqrt2, out, 1, N);
        // No vForce vverf — fall back to scalar erff per element. Still
        // a single pass.
        for (int64_t i = 0; i < n; i++) {
            float v = out[i];
            out[i] = 0.5f * x[i] * (1.0f + erff(v));
        }
        break;
    }
    case TENSOR_U_STEP:
        for (int64_t i = 0; i < n; i++) out[i] = x[i] > 0.0f ? 1.0f : 0.0f;
        break;
    case TENSOR_U_SIN:  vvsinf (out, x, &len); break;
    case TENSOR_U_COS:  vvcosf (out, x, &len); break;
    case TENSOR_U_EXP:  vvexpf (out, x, &len); break;
    case TENSOR_U_SQRT: vvsqrtf(out, x, &len); break;
    }
}

// Scalar fallback for the strided path. Switch dispatch (no function
// pointer call per element).
static inline float tensor_scalar_unary(enum tensor_unop op, float v, float p) {
    switch (op) {
    case TENSOR_U_SCALE:   return v * p;
    case TENSOR_U_SIGMOID: return 1.0f / (1.0f + expf(-v));
    case TENSOR_U_TANH:    return tanhf(v);
    case TENSOR_U_LRELU:   return v > 0.0f ? v : v * p;
    case TENSOR_U_GELU:    return 0.5f * v *
                              (1.0f + erff(v * (float)M_SQRT1_2));
    case TENSOR_U_STEP:    return v > 0.0f ? 1.0f : 0.0f;
    case TENSOR_U_SIN:     return sinf(v);
    case TENSOR_U_COS:     return cosf(v);
    case TENSOR_U_EXP:     return expf(v);
    case TENSOR_U_SQRT:    return sqrtf(v);
    }
    return 0.0f;
}

static struct tensor * tensor_apply_unary(struct tensor * x,
                                          enum tensor_unop op, float param) {
    struct tensor * out = tensor_alloc_with_data(arena_aout(x->arena), x->ndim,
                                         x->ne[0], x->ne[1],
                                         x->ne[2], x->ne[3]);
    int64_t n = tensor_nelements(x);
    if (tensor_is_packed(x)) {
        tensor_vec_unary(op, x->data, out->data, n, param);
    } else {
        // Strided input: indexed walk. Per-row vector path would need
        // packed inner stride; we can still handle that inline if it
        // matters later.
        const int64_t n0 = x->ne[0], n1 = x->ne[1];
        const int64_t n2 = x->ne[2], n3 = x->ne[3];
        for (int64_t i3 = 0; i3 < n3; i3++) {
            for (int64_t i2 = 0; i2 < n2; i2++) {
                for (int64_t i1 = 0; i1 < n1; i1++) {
                    for (int64_t i0 = 0; i0 < n0; i0++) {
                        const float * xp = (const float *)
                            ((const char *)x->data + i3 * x->nb[3]
                                                   + i2 * x->nb[2]
                                                   + i1 * x->nb[1]
                                                   + i0 * x->nb[0]);
                        int64_t oi = ((i3 * n2 + i2) * n1 + i1)
                                         * n0 + i0;
                        out->data[oi] = tensor_scalar_unary(op, *xp, param);
                    }
                }
            }
        }
    }
    return out;
}

struct tensor * tensor_scale(struct tensor * x, float s) {
    return tensor_apply_unary(x, TENSOR_U_SCALE, s);
}

struct tensor * tensor_sigmoid(struct tensor * x) {
    return tensor_apply_unary(x, TENSOR_U_SIGMOID, 0.0f);
}

struct tensor * tensor_tanh(struct tensor * x) {
    return tensor_apply_unary(x, TENSOR_U_TANH, 0.0f);
}

struct tensor * tensor_leaky_relu(struct tensor * x, float slope) {
    return tensor_apply_unary(x, TENSOR_U_LRELU, slope);
}

struct tensor * tensor_gelu_erf(struct tensor * x) {
    return tensor_apply_unary(x, TENSOR_U_GELU, 0.0f);
}

struct tensor * tensor_step(struct tensor * x) {
    return tensor_apply_unary(x, TENSOR_U_STEP, 0.0f);
}

struct tensor * tensor_sin(struct tensor * x) {
    return tensor_apply_unary(x, TENSOR_U_SIN, 0.0f);
}

struct tensor * tensor_cos(struct tensor * x) {
    return tensor_apply_unary(x, TENSOR_U_COS, 0.0f);
}

struct tensor * tensor_exp(struct tensor * x) {
    return tensor_apply_unary(x, TENSOR_U_EXP, 0.0f);
}

struct tensor * tensor_sqrt(struct tensor * x) {
    return tensor_apply_unary(x, TENSOR_U_SQRT, 0.0f);
}

struct tensor * tensor_atan2(struct tensor * y, struct tensor * x) {
    assert(tensor_same_shape(x, y));
    assert(tensor_is_packed(x) && tensor_is_packed(y));
    struct tensor * out = tensor_alloc_with_data(arena_aout(x->arena), x->ndim,
                                         x->ne[0], x->ne[1],
                                         x->ne[2], x->ne[3]);
    int64_t n = tensor_nelements(x);
    for (int64_t i = 0; i < n; i++) {
        out->data[i] = atan2f(y->data[i], x->data[i]);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Reductions / normalization
// ---------------------------------------------------------------------------

// Reduce along ne[axis], keepdim (write per-outer scalar into out).
// LayerNorm: subtract mean, divide by sqrt(var + eps). No scale/bias.
struct tensor * tensor_norm(struct tensor * x, int axis, float eps) {
    // For kittens-tts the only axis we ever normalize over is the
    // innermost (ne[0]) — matches ggml_norm semantics.
    assert(axis == 0);
    assert(tensor_is_packed(x));
    struct tensor * out = tensor_alloc_with_data(arena_aout(x->arena), x->ndim,
                                         x->ne[0], x->ne[1],
                                         x->ne[2], x->ne[3]);
    const int64_t n0 = x->ne[0];
    int64_t outer = tensor_nelements(x) / n0;
    const float * xb = x->data;
    float * ob = out->data;
    const vDSP_Length N = (vDSP_Length)n0;
    for (int64_t r = 0; r < outer; r++) {
        const float * row = xb + r * n0;
        float * orow = ob + r * n0;
        // mean(x) and mean(x^2) via vDSP — single SIMD pass each.
        float mean = 0.0f;
        float meanSq = 0.0f;
        vDSP_meanv (row, 1, &mean,   N);
        vDSP_measqv(row, 1, &meanSq, N);
        const float var = meanSq - mean * mean;
        const float invstd = 1.0f / sqrtf(var + eps);
        // out = (x - mean) * invstd  ==  x * invstd + (-mean * invstd)
        // Encoded as a single fused multiply-add via vDSP_vsmsa.
        const float bias = -mean * invstd;
        vDSP_vsmsa((float *)row, 1, (float *)&invstd, (float *)&bias,
                   orow, 1, N);
    }
    return out;
}

struct tensor * tensor_softmax(struct tensor * x, int axis, float scale) {
    assert(axis == 0);
    assert(tensor_is_packed(x));
    struct tensor * out = tensor_alloc_with_data(arena_aout(x->arena), x->ndim,
                                         x->ne[0], x->ne[1],
                                         x->ne[2], x->ne[3]);
    const int64_t n0 = x->ne[0];
    int64_t outer = tensor_nelements(x) / n0;
    const float * xb = x->data;
    float * ob = out->data;
    const vDSP_Length N = (vDSP_Length)n0;
    int Nint = (int)n0;
    for (int64_t r = 0; r < outer; r++) {
        const float * row = xb + r * n0;
        float * orow = ob + r * n0;
        // 1) orow = row * scale
        vDSP_vsmul(row, 1, (float *)&scale, orow, 1, N);
        // 2) max(orow)
        float mx;
        vDSP_maxv(orow, 1, &mx, N);
        // 3) orow -= mx
        float negmx = -mx;
        vDSP_vsadd(orow, 1, &negmx, orow, 1, N);
        // 4) orow = exp(orow)
        vvexpf(orow, orow, &Nint);
        // 5) sum and divide
        float sum;
        vDSP_sve(orow, 1, &sum, N);
        float inv = 1.0f / sum;
        vDSP_vsmul(orow, 1, &inv, orow, 1, N);
    }
    return out;
}

struct tensor * tensor_cumsum(struct tensor * x, int axis) {
    assert(axis == 0);
    assert(tensor_is_packed(x));
    struct tensor * out = tensor_alloc_with_data(arena_aout(x->arena), x->ndim,
                                         x->ne[0], x->ne[1],
                                         x->ne[2], x->ne[3]);
    const int64_t n0 = x->ne[0];
    int64_t outer = tensor_nelements(x) / n0;
    const float * xb = x->data;
    float * ob = out->data;
    for (int64_t r = 0; r < outer; r++) {
        const float * row = xb + r * n0;
        float * orow = ob + r * n0;
        float acc = 0.0f;
        for (int64_t i = 0; i < n0; i++) {
            acc += row[i];
            orow[i] = acc;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Linear algebra: tensor_mul_mat dispatches through a function pointer so the
// app can A/B-switch between the Accelerate BLAS (default, fastest on
// Apple) and a hand-rolled portable C kernel (research/wasm path; see
// rnd/blas/ for the comparison bench).
//
// Convention (matches ggml_mul_mat):
//   w: ne[0]=K (contraction, innermost), ne[1]=Nw (output rows)
//   x: ne[0]=K, ne[1]=Nx
//   out: ne[0]=Nw, ne[1]=Nx
// Math: out[n, m] = sum_k w[m, k] * x[n, k] = dot(w[m, :], x[n, :]).
// Both inputs must be packed for v1.
//
// The kernel interface treats A=x and Bt=w (both row-major (rows, K))
// and computes C = A @ Bt^T row-major (Nx, Nw).
// ---------------------------------------------------------------------------

typedef void (*tensor_sgemm_fn)(int M, int N, int K,
                            const float * A,  int lda,
                            const float * Bt, int ldb,
                            float * C, int ldc);

static void tensor_sgemm_accelerate(int M, int N, int K,
                                const float * A,  int lda,
                                const float * Bt, int ldb,
                                float * C, int ldc) {
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans, CblasTrans,
                M, N, K,
                1.0f, A, lda, Bt, ldb, 0.0f, C, ldc);
}

// Hand-rolled 4x8 register-tiled kernel for C(M,N) = A(M,K) @ Bt^T (Bt
// stored (N,K) row-major). Pure C99 so the same source compiles for
// wasm with emcc -msimd128 letting the auto-vectorizer do the SIMD.
//
// For-K innermost: 4 broadcast loads from A's 4 rows, 8 broadcast loads
// from Bt's 8 rows, 32 FMAs accumulated into stack-resident scalars
// (clang allocates these to NEON regs at -O3).
static void tensor_sgemm_kernel_4x8(int K,
                                    const float * A,  int lda,
                                    const float * Bt, int ldb,
                                    float * C, int ldc) {
    // slight deviation of code style in this function is acceptable:
    float c00 = 0, c01 = 0, c02 = 0, c03 = 0, c04 = 0, c05 = 0, c06 = 0, c07 = 0;
    float c10 = 0, c11 = 0, c12 = 0, c13 = 0, c14 = 0, c15 = 0, c16 = 0, c17 = 0;
    float c20 = 0, c21 = 0, c22 = 0, c23 = 0, c24 = 0, c25 = 0, c26 = 0, c27 = 0;
    float c30 = 0, c31 = 0, c32 = 0, c33 = 0, c34 = 0, c35 = 0, c36 = 0, c37 = 0;
    for (int k = 0; k < K; k++) {
        float a0 = A[0 * lda + k];
        float a1 = A[1 * lda + k];
        float a2 = A[2 * lda + k];
        float a3 = A[3 * lda + k];
        float b0 = Bt[0 * ldb + k];
        float b1 = Bt[1 * ldb + k];
        float b2 = Bt[2 * ldb + k];
        float b3 = Bt[3 * ldb + k];
        float b4 = Bt[4 * ldb + k];
        float b5 = Bt[5 * ldb + k];
        float b6 = Bt[6 * ldb + k];
        float b7 = Bt[7 * ldb + k];
        c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3;
        c04 += a0 * b4; c05 += a0 * b5; c06 += a0 * b6; c07 += a0 * b7;
        c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3;
        c14 += a1 * b4; c15 += a1 * b5; c16 += a1 * b6; c17 += a1 * b7;
        c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3;
        c24 += a2 * b4; c25 += a2 * b5; c26 += a2 * b6; c27 += a2 * b7;
        c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3;
        c34 += a3 * b4; c35 += a3 * b5; c36 += a3 * b6; c37 += a3 * b7;
    }
    C[0 * ldc + 0] = c00; C[0 * ldc + 1] = c01; C[0 * ldc + 2] = c02; C[0 * ldc + 3] = c03;
    C[0 * ldc + 4] = c04; C[0 * ldc + 5] = c05; C[0 * ldc + 6] = c06; C[0 * ldc + 7] = c07;
    C[1 * ldc + 0] = c10; C[1 * ldc + 1] = c11; C[1 * ldc + 2] = c12; C[1 * ldc + 3] = c13;
    C[1 * ldc + 4] = c14; C[1 * ldc + 5] = c15; C[1 * ldc + 6] = c16; C[1 * ldc + 7] = c17;
    C[2 * ldc + 0] = c20; C[2 * ldc + 1] = c21; C[2 * ldc + 2] = c22; C[2 * ldc + 3] = c23;
    C[2 * ldc + 4] = c24; C[2 * ldc + 5] = c25; C[2 * ldc + 6] = c26; C[2 * ldc + 7] = c27;
    C[3 * ldc + 0] = c30; C[3 * ldc + 1] = c31; C[3 * ldc + 2] = c32; C[3 * ldc + 3] = c33;
    C[3 * ldc + 4] = c34; C[3 * ldc + 5] = c35; C[3 * ldc + 6] = c36; C[3 * ldc + 7] = c37;
}

static void tensor_sgemm_edge(int M, int N, int K,
                              const float * A,  int lda,
                              const float * Bt, int ldb,
                              float * C, int ldc) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += A[m * lda + k] * Bt[n * ldb + k];
            }
            C[m * ldc + n] = acc;
        }
    }
}

static void tensor_sgemm_tiled(int M, int N, int K,
                               const float * A,  int lda,
                               const float * Bt, int ldb,
                               float * C, int ldc) {
    const int MR = 4, NR = 8;
    int M_main = (M / MR) * MR;
    int N_main = (N / NR) * NR;
    for (int m = 0; m < M_main; m += MR) {
        for (int n = 0; n < N_main; n += NR) {
            tensor_sgemm_kernel_4x8(K,
                                A  + m * lda, lda,
                                Bt + n * ldb, ldb,
                                C  + m * ldc + n, ldc);
        }
        if (N_main < N) {
            tensor_sgemm_edge(MR, N - N_main, K,
                          A  + m * lda, lda,
                          Bt + N_main * ldb, ldb,
                          C  + m * ldc + N_main, ldc);
        }
    }
    if (M_main < M) {
        tensor_sgemm_edge(M - M_main, N, K,
                      A  + M_main * lda, lda,
                      Bt, ldb,
                      C  + M_main * ldc, ldc);
    }
}

// Process-global single-threaded selector. Default is Accelerate.
static tensor_sgemm_fn tensor_sgemm_active = tensor_sgemm_accelerate;

void tensor_set_sgemm_impl(int impl) {
    if (impl == 1) {
        tensor_sgemm_active = tensor_sgemm_tiled;
    } else {
        tensor_sgemm_active = tensor_sgemm_accelerate;
    }
}

int tensor_get_sgemm_impl(void) {
    return (tensor_sgemm_active == tensor_sgemm_tiled) ? 1 : 0;
}


struct tensor * tensor_mul_mat(struct tensor * w, struct tensor * x) {
    // Convention exactly matches ggml_mul_mat(a, b):
    //   a (= w) stored ne[0]=K, ne[1]=Nw  (PyTorch (Nw, K) row-major)
    //   b (= x) stored ne[0]=K, ne[1]=Nx  (PyTorch (Nx, K) row-major)
    //   c (= out) stored ne[0]=Nw, ne[1]=Nx
    // Batched: ne[2] and ne[3] are batch dims; w and x must agree on
    // them. Output preserves the batch dims.
    if (!tensor_is_packed(w)) { w = tensor_cont(w); }
    if (!tensor_is_packed(x)) { x = tensor_cont(x); }
    assert(w->ne[0] == x->ne[0]);
    assert(w->ne[2] == x->ne[2]);
    assert(w->ne[3] == x->ne[3]);
    const int64_t K  = w->ne[0];
    const int64_t Nw = w->ne[1];
    const int64_t Nx = x->ne[1];
    const int64_t B2 = w->ne[2];
    const int64_t B3 = w->ne[3];
    struct arena * oa = arena_aout(w->arena);
    int ndim = (B3 > 1) ? 4 : (B2 > 1) ? 3 : 2;
    int64_t ne[TENSOR_MAX_DIMS] = { Nw, Nx, B2, B3 };
    struct tensor * out = tensor_new_nd(oa, ndim, ne);
    const size_t w_stride = (size_t)Nw * (size_t)K;
    const size_t x_stride = (size_t)Nx * (size_t)K;
    const size_t o_stride = (size_t)Nw * (size_t)Nx;
    for (int64_t b3 = 0; b3 < B3; b3++) {
        for (int64_t b2 = 0; b2 < B2; b2++) {
            const size_t b = (size_t)b3 * (size_t)B2 + (size_t)b2;
            const float * w_b = w->data + b * w_stride;
            const float * x_b = x->data + b * x_stride;
            float * o_b = out->data + b * o_stride;
            tensor_sgemm_active((int)Nx, (int)Nw, (int)K,
                            x_b, (int)K,
                            w_b, (int)K,
                            o_b, (int)Nw);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// 1D convolution.
//
// Layout: input  (B, Cin, L) with B in ne[2], Cin in ne[1], L in ne[0].
//         weight (Cout, Cin, K) with K in ne[0], Cin in ne[1],
//                Cout in ne[2].
//         output (B, Cout, Lout).
// ---------------------------------------------------------------------------

static int64_t tensor_conv_out_len(int64_t L_in, int K, int stride, int pad,
                                   int dilation) {
    return (L_in + 2 * pad - dilation * (K - 1) - 1) / stride + 1;
}

// Produce (B, Cin*K, Lout) im2col matrix. Input (B, Cin, L).
struct tensor * tensor_im2col(struct tensor * x, int kernel, int stride,
                              int pad, int dilation) {
    assert(tensor_is_packed(x));
    const int64_t L_in = x->ne[0];
    const int64_t Cin  = x->ne[1];
    const int64_t B    = x->ne[2];
    const int64_t Lout = tensor_conv_out_len(L_in, kernel, stride, pad,
                                             dilation);
    struct tensor * out = tensor_new_3d(arena_aout(x->arena), Lout,
                                        Cin * (int64_t)kernel, B);
    const float * xb = x->data;
    float * ob = out->data;
    for (int64_t b = 0; b < B; b++) {
        for (int64_t c = 0; c < Cin; c++) {
            for (int k = 0; k < kernel; k++) {
                int64_t off = b * Cin * (int64_t)kernel * Lout
                            + (c * (int64_t)kernel + k) * Lout;
                for (int64_t lo = 0; lo < Lout; lo++) {
                    int64_t li = lo * stride - pad + k * dilation;
                    float v = 0.0f;
                    if (li >= 0 && li < L_in) {
                        v = xb[b * Cin * L_in + c * L_in + li];
                    }
                    ob[off + lo] = v;
                }
            }
        }
    }
    return out;
}

struct tensor * tensor_conv_1d(struct tensor * w, struct tensor * x,
                       int stride, int pad, int dilation) {
    // w stored (K, Cin, Cout) — ne[0]=K, ne[1]=Cin, ne[2]=Cout.
    //   Memory offset for w[co, ci, k] is k + ci*K + co*Cin*K.
    //   As a row-major matrix it's (Cout, Cin*K), with each "row" co
    //   holding Cin*K elements ordered (ci outer, k inner).
    // x stored (L, Cin, B). Memory offset b*Cin*L + ci*L + l.
    // im2col output cols stored (Lout, Cin*K, B); memory base
    //   b*Cin*K*Lout + z*Lout + lo, where z = ci*K + k.
    //   As row-major per batch: cols[b] is (Cin*K, Lout) with lda=Lout.
    // For each batch b, sgemm:
    //   out[b] row-major (Cout, Lout) = W (Cout, Cin*K)
    //                                  @ cols[b] (Cin*K, Lout).
    assert(tensor_is_packed(w));
    assert(tensor_is_packed(x));
    const int64_t K    = w->ne[0];
    const int64_t Cin  = w->ne[1];
    const int64_t Cout = w->ne[2];
    const int64_t L_in = x->ne[0];
    const int64_t B    = x->ne[2];
    assert(x->ne[1] == Cin);
    const int64_t Lout = tensor_conv_out_len(L_in, (int)K,
                                         stride, pad, dilation);
    struct tensor * cols = tensor_im2col(x, (int)K, stride, pad, dilation);
    struct tensor * out  = tensor_new_3d(arena_aout(w->arena), Lout, Cout, B);
    for (int64_t b = 0; b < B; b++) {
        const float * c_b = cols->data
                          + b * (size_t)(Cin * K) * (size_t)Lout;
        float * o_b = out->data + b * (size_t)Cout * (size_t)Lout;
        cblas_sgemm(CblasRowMajor,
                    CblasNoTrans, CblasNoTrans,
                    (int)Cout, (int)Lout, (int)(Cin * K),
                    1.0f,
                    w->data, (int)(Cin * K),
                    c_b,     (int)Lout,
                    0.0f,
                    o_b,     (int)Lout);
    }
    return out;
}

struct tensor * tensor_conv_1d_dw(struct tensor * w, struct tensor * x,
                          int stride, int pad, int dilation) {
    // Depthwise: w (K, 1, C), x (L, C, B), out (Lout, C, B).
    assert(tensor_is_packed(w));
    assert(tensor_is_packed(x));
    const int64_t K = w->ne[0];
    const int64_t C = w->ne[2];
    assert(w->ne[1] == 1);
    assert(x->ne[1] == C);
    const int64_t L_in = x->ne[0];
    const int64_t B    = x->ne[2];
    const int64_t Lout = tensor_conv_out_len(L_in, (int)K,
                                         stride, pad, dilation);
    struct tensor * out = tensor_new_3d(arena_aout(w->arena), Lout, C, B);
    const float * xb = x->data;
    const float * wb = w->data;
    float * ob = out->data;
    for (int64_t b = 0; b < B; b++) {
        for (int64_t c = 0; c < C; c++) {
            const float * xrow = xb + b * C * L_in + c * L_in;
            const float * wrow = wb + c * K;
            float * orow = ob + b * C * Lout + c * Lout;
            for (int64_t lo = 0; lo < Lout; lo++) {
                float acc = 0.0f;
                for (int k = 0; k < K; k++) {
                    int64_t li = lo * stride - pad + k * dilation;
                    if (li >= 0 && li < L_in) {
                        acc += wrow[k] * xrow[li];
                    }
                }
                orow[lo] = acc;
            }
        }
    }
    return out;
}

struct tensor * tensor_conv_transpose_1d(struct tensor * w, struct tensor * x,
                                 int stride, int pad) {
    // w (K, Cout, Cin) — ne[0]=K, ne[1]=Cout, ne[2]=Cin (PyTorch order).
    // x (Lin, Cin, B), out (Lout, Cout, B).
    assert(tensor_is_packed(w));
    assert(tensor_is_packed(x));
    const int64_t K    = w->ne[0];
    const int64_t Cout = w->ne[1];
    const int64_t Cin  = w->ne[2];
    const int64_t Lin  = x->ne[0];
    const int64_t B    = x->ne[2];
    assert(x->ne[1] == Cin);
    int64_t Lfull = (Lin - 1) * stride + K;
    int64_t Lout = Lfull - 2 * pad;
    if (Lout < 0) { Lout = 0; }
    struct tensor * out = tensor_new_3d(arena_aout(w->arena), Lout, Cout, B);
    memset(out->data, 0, (size_t)tensor_nbytes(out));
    // Scatter form.
    for (int64_t b = 0; b < B; b++) {
        for (int64_t co = 0; co < Cout; co++) {
            float * orow = out->data + b * Cout * Lout + co * Lout;
            for (int64_t ci = 0; ci < Cin; ci++) {
                const float * xrow = x->data
                                   + b * Cin * Lin + ci * Lin;
                const float * wrow = w->data
                                   + ci * Cout * K + co * K;
                for (int64_t li = 0; li < Lin; li++) {
                    float xv = xrow[li];
                    int64_t base = li * stride - pad;
                    for (int64_t k = 0; k < K; k++) {
                        int64_t lo = base + k;
                        if (lo >= 0 && lo < Lout) {
                            orow[lo] += wrow[k] * xv;
                        }
                    }
                }
            }
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Smoke tests (compile as standalone with -DTENSOR_TESTS).
// Was tests/test_kt_basic.c; folded in to match the node.c.dev
// single-file-library + bottom-of-file smoke-test convention.
// ---------------------------------------------------------------------------

#ifdef TENSOR_TESTS

static int g_failures = 0;

static void check_close(const char * name,
                        float got, float want, float tol) {
    float diff = fabsf(got - want);
    if (diff > tol) {
        fprintf(stderr,
                "FAIL %s: got %.6f want %.6f (diff %.6g, tol %.6g)\n",
                name, got, want, diff, tol);
        g_failures++;
    }
}

static void check_array(const char * name,
                        const float * got, const float * want,
                        int64_t n, float tol) {
    int bad = 0;
    for (int64_t i = 0; i < n; i++) {
        if (fabsf(got[i] - want[i]) > tol) { bad++; }
    }
    if (bad > 0) {
        fprintf(stderr, "FAIL %s: %d/%lld elements outside tol %.6g\n",
                name, bad, (long long)n, tol);
        for (int64_t i = 0; i < n && i < 8; i++) {
            fprintf(stderr, "  [%lld] got %.6f want %.6f\n",
                    (long long)i, got[i], want[i]);
        }
        g_failures++;
    }
}

static void test_arena(void) {
    struct arena * a = arena_new(1024);
    size_t cap0 = arena_capacity(a);
    struct tensor * t = tensor_new_2d(a, 8, 8);
    assert(t != NULL);
    assert(arena_used(a) > 0);
    arena_reset(a);
    assert(arena_used(a) == 0);
    assert(arena_capacity(a) == cap0);
    arena_free(a);
}

static void test_elementwise(void) {
    struct arena * a = arena_new(4096);
    struct tensor * x = tensor_new_1d(a, 4);
    struct tensor * y = tensor_new_1d(a, 4);
    float xv[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    float yv[4] = { 10.0f, 20.0f, 30.0f, 40.0f };
    memcpy(x->data, xv, sizeof(xv));
    memcpy(y->data, yv, sizeof(yv));
    struct tensor * sum = tensor_add(x, y);
    float esum[4] = { 11, 22, 33, 44 };
    check_array("add", sum->data, esum, 4, 1e-6f);
    struct tensor * prod = tensor_mul(x, y);
    float eprod[4] = { 10, 40, 90, 160 };
    check_array("mul", prod->data, eprod, 4, 1e-6f);
    struct tensor * scl = tensor_scale(x, 2.0f);
    float escl[4] = { 2, 4, 6, 8 };
    check_array("scale", scl->data, escl, 4, 1e-6f);
    arena_free(a);
}

static void test_broadcast_channel(void) {
    struct arena * a = arena_new(4096);
    struct tensor * x = tensor_new_3d(a, 3, 2, 1);
    struct tensor * b = tensor_new_3d(a, 1, 2, 1);
    float xv[6] = { 1, 2, 3, 10, 20, 30 };
    float bv[2] = { 100.0f, 1000.0f };
    memcpy(x->data, xv, sizeof(xv));
    memcpy(b->data, bv, sizeof(bv));
    struct tensor * out = tensor_add(x, b);
    float want[6] = { 101, 102, 103, 1010, 1020, 1030 };
    check_array("add+channel-bcast", out->data, want, 6, 1e-6f);
    arena_free(a);
}

static void test_unary(void) {
    struct arena * a = arena_new(4096);
    struct tensor * x = tensor_new_1d(a, 4);
    float xv[4] = { -1.0f, 0.0f, 1.0f, 2.0f };
    memcpy(x->data, xv, sizeof(xv));
    struct tensor * sg = tensor_sigmoid(x);
    check_close("sigmoid(-1)", sg->data[0], 1.0f / (1.0f + expf(1)),
                1e-6f);
    check_close("sigmoid(0)",  sg->data[1], 0.5f, 1e-6f);
    struct tensor * th = tensor_tanh(x);
    check_close("tanh(0)",     th->data[1], 0.0f, 1e-6f);
    check_close("tanh(1)",     th->data[2], tanhf(1), 1e-6f);
    struct tensor * lr = tensor_leaky_relu(x, 0.2f);
    float elr[4] = { -0.2f, 0.0f, 1.0f, 2.0f };
    check_array("leaky_relu", lr->data, elr, 4, 1e-6f);
    struct tensor * st = tensor_step(x);
    float est[4] = { 0, 0, 1, 1 };
    check_array("step", st->data, est, 4, 1e-6f);
    arena_free(a);
}

static void test_get_rows(void) {
    struct arena * a = arena_new(4096);
    struct tensor * d = tensor_new_2d(a, 3, 4);
    float dv[12] = { 1,2,3, 10,20,30, 100,200,300, 1000,2000,3000 };
    memcpy(d->data, dv, sizeof(dv));
    int32_t ids[3] = { 0, 2, 1 };
    struct tensor * out = tensor_get_rows(d, ids, 3);
    float want[9] = { 1,2,3, 100,200,300, 10,20,30 };
    check_array("get_rows", out->data, want, 9, 1e-6f);
    arena_free(a);
}

static void test_layout(void) {
    struct arena * a = arena_new(4096);
    struct tensor * t = tensor_new_2d(a, 3, 2);
    float v[6] = { 1, 2, 3,  4, 5, 6 };
    memcpy(t->data, v, sizeof(v));
    struct tensor * tr = tensor_transpose(t);
    struct tensor * trc = tensor_cont(tr);
    float want[6] = { 1, 4,  2, 5,  3, 6 };
    check_array("transpose+cont", trc->data, want, 6, 1e-6f);
    arena_free(a);
}

static void test_concat(void) {
    struct arena * a = arena_new(4096);
    struct tensor * x = tensor_new_2d(a, 2, 3);
    struct tensor * y = tensor_new_2d(a, 2, 2);
    float xv[6] = { 1,2, 3,4, 5,6 };
    float yv[4] = { 7,8, 9,10 };
    memcpy(x->data, xv, sizeof(xv));
    memcpy(y->data, yv, sizeof(yv));
    struct tensor * cat = tensor_concat(x, y, 1);
    float want[10] = { 1,2, 3,4, 5,6,  7,8, 9,10 };
    check_array("concat axis 1", cat->data, want, 10, 1e-6f);
    arena_free(a);
}

static void test_norm(void) {
    struct arena * a = arena_new(4096);
    struct tensor * x = tensor_new_1d(a, 4);
    float v[4] = { 1, 2, 3, 4 };
    memcpy(x->data, v, sizeof(v));
    struct tensor * n = tensor_norm(x, 0, 1e-5f);
    float mean = 2.5f;
    float invstd = 1.0f / sqrtf(1.25f + 1e-5f);
    float want[4];
    for (int i = 0; i < 4; i++) { want[i] = (v[i] - mean) * invstd; }
    check_array("norm", n->data, want, 4, 1e-5f);
    arena_free(a);
}

static void test_softmax(void) {
    struct arena * a = arena_new(4096);
    struct tensor * x = tensor_new_1d(a, 3);
    float v[3] = { 1.0f, 2.0f, 3.0f };
    memcpy(x->data, v, sizeof(v));
    struct tensor * sm = tensor_softmax(x, 0, 1.0f);
    float e1 = expf(1 - 3), e2 = expf(2 - 3), e3 = 1.0f;
    float s = e1 + e2 + e3;
    float want[3] = { e1 / s, e2 / s, e3 / s };
    check_array("softmax", sm->data, want, 3, 1e-6f);
    arena_free(a);
}

static void test_cumsum(void) {
    struct arena * a = arena_new(4096);
    struct tensor * x = tensor_new_1d(a, 5);
    float v[5] = { 1, 2, 3, 4, 5 };
    memcpy(x->data, v, sizeof(v));
    struct tensor * cs = tensor_cumsum(x, 0);
    float want[5] = { 1, 3, 6, 10, 15 };
    check_array("cumsum", cs->data, want, 5, 1e-6f);
    arena_free(a);
}

static void test_mul_mat_identity(void) {
    struct arena * a = arena_new(4096);
    struct tensor * w = tensor_new_2d(a, 3, 3);
    struct tensor * x = tensor_new_2d(a, 3, 2);
    float wv[9] = { 1,0,0,  0,1,0,  0,0,1 };
    float xv[6] = { 1,2,3,  4,5,6 };
    memcpy(w->data, wv, sizeof(wv));
    memcpy(x->data, xv, sizeof(xv));
    struct tensor * out = tensor_mul_mat(w, x);
    check_array("mul_mat identity", out->data, xv, 6, 1e-6f);
    arena_free(a);
}

static void test_mul_mat_simple(void) {
    struct arena * a = arena_new(4096);
    struct tensor * w = tensor_new_2d(a, 2, 3);
    struct tensor * x = tensor_new_2d(a, 2, 1);
    float wv[6] = { 1, 2,  3, 4,  5, 6 };
    float xv[2] = { 10, 20 };
    memcpy(w->data, wv, sizeof(wv));
    memcpy(x->data, xv, sizeof(xv));
    struct tensor * out = tensor_mul_mat(w, x);
    float want[3] = { 50, 110, 170 };
    check_array("mul_mat simple", out->data, want, 3, 1e-6f);
    arena_free(a);
}

static void test_conv_1d_simple(void) {
    struct arena * a = arena_new(8192);
    struct tensor * w = tensor_new_3d(a, 3, 1, 1);
    struct tensor * x = tensor_new_3d(a, 5, 1, 1);
    float wv[3] = { 1, 0, -1 };
    float xv[5] = { 1, 2, 3, 4, 5 };
    memcpy(w->data, wv, sizeof(wv));
    memcpy(x->data, xv, sizeof(xv));
    struct tensor * out = tensor_conv_1d(w, x, 1, 1, 1);
    float want[5] = { -2, -2, -2, -2, 4 };
    check_array("conv_1d", out->data, want, 5, 1e-5f);
    arena_free(a);
}

static void test_conv_transpose_1d_simple(void) {
    struct arena * a = arena_new(4096);
    struct tensor * w = tensor_new_3d(a, 3, 1, 1);
    struct tensor * x = tensor_new_3d(a, 2, 1, 1);
    float wv[3] = { 1, 2, 3 };
    float xv[2] = { 10, 20 };
    memcpy(w->data, wv, sizeof(wv));
    memcpy(x->data, xv, sizeof(xv));
    struct tensor * out = tensor_conv_transpose_1d(w, x, 2, 0);
    float want[5] = { 10, 20, 50, 40, 60 };
    check_array("conv_transpose_1d simple", out->data, want, 5, 1e-6f);
    arena_free(a);
}

static void test_conv_1d_dw(void) {
    struct arena * a = arena_new(8192);
    struct tensor * w = tensor_new_3d(a, 2, 1, 2);
    struct tensor * x = tensor_new_3d(a, 3, 2, 1);
    float wv[4] = { 1, 1, 2, 3 };
    float xv[6] = { 1, 2, 3,  10, 20, 30 };
    memcpy(w->data, wv, sizeof(wv));
    memcpy(x->data, xv, sizeof(xv));
    struct tensor * out = tensor_conv_1d_dw(w, x, 1, 0, 1);
    float want[4] = { 3, 5, 80, 130 };
    check_array("conv_1d_dw", out->data, want, 4, 1e-5f);
    arena_free(a);
}

int main(void) {
    test_arena();
    test_elementwise();
    test_broadcast_channel();
    test_unary();
    test_get_rows();
    test_layout();
    test_concat();
    test_norm();
    test_softmax();
    test_cumsum();
    test_mul_mat_identity();
    test_mul_mat_simple();
    test_conv_1d_simple();
    test_conv_1d_dw();
    test_conv_transpose_1d_simple();
    if (g_failures == 0) {
        printf("struct tensor basic: ALL PASS\n");
    } else {
        printf("struct tensor basic: %d failures\n", g_failures);
    }
    return g_failures == 0 ? 0 : 1;
}

#endif /* TENSOR_TESTS */

#endif /* TENSOR_C */
