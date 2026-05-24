// kittens.c -- pure-C / cblas KittenTTS backend.
//
// Public API in cpu/include/kittens.h. The pipeline runs four stages
// (textstage / genfront / decoder / generator) on struct tensor and an
// arena allocator. Eager evaluation - no graph builder, no two-phase
// compute. Single-file library: this TU pulls gguf.c (which pulls
// tensor.c) via #include, so the whole stack compiles as one TU.

#ifndef KITTENS_C
#define KITTENS_C

#include "kittens.h"
#include "gguf.c"           // transitively #includes "tensor.c"

#include <assert.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// OOM_TRACE: phys_footprint markers at each pipeline stage and inside
// build_noise_contribs / build_hifi_block. Reads task_info's
// phys_footprint, the field iOS jetsam uses. Compile with -DOOM_TRACE
// to enable; default builds are silent. See PLAN/TODO for what the
// numbers mean (macOS compressor lag inflates reading; instantaneous
// working set is the per-stage delta).
#ifdef OOM_TRACE
#include <mach/mach.h>
static double oom_mb(void) {
    struct task_vm_info info; mach_msg_type_number_t n = TASK_VM_INFO_COUNT;
    return task_info(mach_task_self(), TASK_VM_INFO,
                     (task_info_t)&info, &n) == KERN_SUCCESS
        ? (double)info.phys_footprint / (1024.0 * 1024.0) : 0.0;
}
#define OOM_MARK(stage) fprintf(stderr, \
    "  [OOM %-26s L=%-4d F=%-5d phys=%7.1f MB]\n", \
    stage, L, F, oom_mb())
#else
#define OOM_MARK(stage) ((void)0)
#endif

// ---------------------------------------------------------------------------
// Arch + weights (internal; loaded from GGUF at ctx-create time)
// ---------------------------------------------------------------------------

struct arch {
    int vocab, max_pos, token_types;
    int embd_dim, hidden, n_layers, n_heads, head_dim, ffn_dim;
    float ln_eps;
    int bert_enc_dim, style_dim, lstm_hidden, dur_logits;
    int audio_per_frame, istft_hop, istft_trim;
};

// Bound weight pointers. Loaded by bind_weights() into weights_arena.
struct weights {
    // Albert / BERT
    struct tensor * e_word, * e_pos, * e_type;
    struct tensor * e_ln_w, * e_ln_b;
    struct tensor * proj_w, * proj_b;
    struct tensor * q_w, * q_b, * k_w, * k_b, * v_w, * v_b, * o_w, * o_b;
    struct tensor * attn_ln_w, * attn_ln_b;
    struct tensor * ffn_w, * ffn_b, * ffn_out_w, * ffn_out_b;
    struct tensor * full_ln_w, * full_ln_b;
    // post-BERT projection
    struct tensor * bert_enc_w, * bert_enc_b;
    // PredictorTextEncoder
    struct tensor * pt_l0_fW, * pt_l0_fR, * pt_l0_fb;
    struct tensor * pt_l0_bW, * pt_l0_bR, * pt_l0_bb;
    struct tensor * pt_fc1_w, * pt_fc1_b;
    struct tensor * pt_l2_fW, * pt_l2_fR, * pt_l2_fb;
    struct tensor * pt_l2_bW, * pt_l2_bR, * pt_l2_bb;
    struct tensor * pt_fc3_w, * pt_fc3_b;
    // Duration head
    struct tensor * dur_l_fW, * dur_l_fR, * dur_l_fb;
    struct tensor * dur_l_bW, * dur_l_bR, * dur_l_bb;
    struct tensor * dur_w, * dur_b;
    // Acoustic text encoder
    struct tensor * ac_embd;
    struct tensor * ac_c0_w, * ac_c0_b, * ac_ln0_g, * ac_ln0_b;
    struct tensor * ac_c1_w, * ac_c1_b, * ac_ln1_g, * ac_ln1_b;
    struct tensor * ac_l_fW, * ac_l_fR, * ac_l_fb;
    struct tensor * ac_l_bW, * ac_l_bR, * ac_l_bb;
    // GenFront shared LSTM
    struct tensor * sh_fW, * sh_fR, * sh_fb;
    struct tensor * sh_bW, * sh_bR, * sh_bb;
    // F0 / N / decoder / generator blocks are looked up dynamically by
    // formatted name (e.g. "f0.0.c1.weight", "dec.decode.3.pool.weight",
    // "gen.r0.c1.0.weight"). See named() / named_fmt() below.
};

struct named_entry {
    char        name[64];
    struct tensor * t;
};

struct kittens_ctx {
    struct gguf *         gguf;
    struct arena *        weights_arena;  // model-lifetime
    struct arena *        scratch_arena;  // reset each kittens_synthesize
    struct arch           arch;
    struct weights        W;
    // Lazy name->tensor cache for dynamically named block weights
    // (e.g. "f0.0.c1.weight", "gen.r0.c1.0.weight"). Populated on first
    // lookup; entries live in weights_arena so they survive arena resets.
    struct named_entry *  cache;
    int                   cache_count;
    int                   cache_cap;
    char                  err[256];
};

static void set_ctx_err(struct kittens_ctx * ctx, const char * fmt, ...) {
    va_list ap; va_start(ap, fmt);
    vsnprintf(ctx->err, sizeof(ctx->err), fmt, ap);
    va_end(ap);
}

const char * kittens_last_error(const struct kittens_ctx * ctx) {
    return ctx != NULL ? ctx->err : gguf_last_error();
}

// ---------------------------------------------------------------------------
// Weight binding
// ---------------------------------------------------------------------------

// Looked up by name; returns NULL if not present, caller decides.
static struct tensor * bind(struct kittens_ctx * ctx, const char * name) {
    struct tensor * t = gguf_load_tensor(ctx->gguf, ctx->weights_arena, name);
    if (t == NULL) {
        set_ctx_err(ctx, "missing GGUF tensor: %s", name);
    }
    return t;
}

// Required: assert it exists.
static struct tensor * bind_req(struct kittens_ctx * ctx, const char * name) {
    struct tensor * t = bind(ctx, name);
    if (t == NULL) {
        fprintf(stderr, "kittens: required tensor missing: %s\n", name);
        abort();
    }
    return t;
}

static int load_arch(struct kittens_ctx * ctx) {
    uint32_t v;
    int loaded = gguf_get_u32(ctx->gguf,
                                 "kittens-tts.vocab_size", &v);
    if (!loaded) {
        set_ctx_err(ctx, "missing arch KV: kittens-tts.vocab_size");
    } else {
        ctx->arch.vocab = (int)v;
        gguf_get_u32(ctx->gguf, "kittens-tts.max_position", &v);
        ctx->arch.max_pos = (int)v;
        gguf_get_u32(ctx->gguf, "kittens-tts.token_types", &v);
        ctx->arch.token_types = (int)v;
        gguf_get_u32(ctx->gguf, "kittens-tts.embedding_dim", &v);
        ctx->arch.embd_dim = (int)v;
        gguf_get_u32(ctx->gguf, "kittens-tts.hidden_size", &v);
        ctx->arch.hidden = (int)v;
        gguf_get_u32(ctx->gguf, "kittens-tts.num_layers", &v);
        ctx->arch.n_layers = (int)v;
        gguf_get_u32(ctx->gguf, "kittens-tts.num_heads", &v);
        ctx->arch.n_heads = (int)v;
        gguf_get_u32(ctx->gguf, "kittens-tts.head_dim", &v);
        ctx->arch.head_dim = (int)v;
        gguf_get_u32(ctx->gguf, "kittens-tts.ffn_dim", &v);
        ctx->arch.ffn_dim = (int)v;
        gguf_get_f32(ctx->gguf, "kittens-tts.layer_norm_eps",
                        &ctx->arch.ln_eps);
        gguf_get_u32(ctx->gguf, "kittens-tts.bert_enc_dim", &v);
        ctx->arch.bert_enc_dim = (int)v;
        gguf_get_u32(ctx->gguf, "kittens-tts.style_dim", &v);
        ctx->arch.style_dim = (int)v;
        gguf_get_u32(ctx->gguf, "kittens-tts.lstm_hidden", &v);
        ctx->arch.lstm_hidden = (int)v;
        gguf_get_u32(ctx->gguf, "kittens-tts.dur_logits", &v);
        ctx->arch.dur_logits = (int)v;
        gguf_get_u32(ctx->gguf, "kittens-tts.audio_per_frame", &v);
        ctx->arch.audio_per_frame = (int)v;
        gguf_get_u32(ctx->gguf, "kittens-tts.istft_hop", &v);
        ctx->arch.istft_hop = (int)v;
        gguf_get_u32(ctx->gguf, "kittens-tts.istft_trim", &v);
        ctx->arch.istft_trim = (int)v;
    }
    return loaded;
}

static void bind_weights(struct kittens_ctx * ctx) {
    struct weights * W = &ctx->W;
    // Albert
    W->e_word   = bind_req(ctx, "embd.word.weight");
    W->e_pos    = bind_req(ctx, "embd.pos.weight");
    W->e_type   = bind_req(ctx, "embd.type.weight");
    W->e_ln_w   = bind_req(ctx, "embd.ln.weight");
    W->e_ln_b   = bind_req(ctx, "embd.ln.bias");
    W->proj_w   = bind_req(ctx, "embd_to_hidden.weight");
    W->proj_b   = bind_req(ctx, "embd_to_hidden.bias");
    W->q_w      = bind_req(ctx, "layer.attn_q.weight");
    W->q_b      = bind_req(ctx, "layer.attn_q.bias");
    W->k_w      = bind_req(ctx, "layer.attn_k.weight");
    W->k_b      = bind_req(ctx, "layer.attn_k.bias");
    W->v_w      = bind_req(ctx, "layer.attn_v.weight");
    W->v_b      = bind_req(ctx, "layer.attn_v.bias");
    W->o_w      = bind_req(ctx, "layer.attn_out.weight");
    W->o_b      = bind_req(ctx, "layer.attn_out.bias");
    W->attn_ln_w= bind_req(ctx, "layer.attn_ln.weight");
    W->attn_ln_b= bind_req(ctx, "layer.attn_ln.bias");
    W->ffn_w    = bind_req(ctx, "layer.ffn.weight");
    W->ffn_b    = bind_req(ctx, "layer.ffn.bias");
    W->ffn_out_w= bind_req(ctx, "layer.ffn_out.weight");
    W->ffn_out_b= bind_req(ctx, "layer.ffn_out.bias");
    W->full_ln_w= bind_req(ctx, "layer.full_ln.weight");
    W->full_ln_b= bind_req(ctx, "layer.full_ln.bias");
    // post-BERT
    W->bert_enc_w = bind_req(ctx, "bert_enc.weight");
    W->bert_enc_b = bind_req(ctx, "bert_enc.bias");
    // PredictorTextEncoder
    W->pt_l0_fW = bind_req(ctx, "pred_text.lstm0.fwd.W");
    W->pt_l0_fR = bind_req(ctx, "pred_text.lstm0.fwd.R");
    W->pt_l0_fb = bind_req(ctx, "pred_text.lstm0.fwd.b");
    W->pt_l0_bW = bind_req(ctx, "pred_text.lstm0.bwd.W");
    W->pt_l0_bR = bind_req(ctx, "pred_text.lstm0.bwd.R");
    W->pt_l0_bb = bind_req(ctx, "pred_text.lstm0.bwd.b");
    W->pt_fc1_w = bind_req(ctx, "pred_text.fc1.weight");
    W->pt_fc1_b = bind_req(ctx, "pred_text.fc1.bias");
    W->pt_l2_fW = bind_req(ctx, "pred_text.lstm2.fwd.W");
    W->pt_l2_fR = bind_req(ctx, "pred_text.lstm2.fwd.R");
    W->pt_l2_fb = bind_req(ctx, "pred_text.lstm2.fwd.b");
    W->pt_l2_bW = bind_req(ctx, "pred_text.lstm2.bwd.W");
    W->pt_l2_bR = bind_req(ctx, "pred_text.lstm2.bwd.R");
    W->pt_l2_bb = bind_req(ctx, "pred_text.lstm2.bwd.b");
    W->pt_fc3_w = bind_req(ctx, "pred_text.fc3.weight");
    W->pt_fc3_b = bind_req(ctx, "pred_text.fc3.bias");
    // Duration head
    W->dur_l_fW = bind_req(ctx, "dur.lstm.fwd.W");
    W->dur_l_fR = bind_req(ctx, "dur.lstm.fwd.R");
    W->dur_l_fb = bind_req(ctx, "dur.lstm.fwd.b");
    W->dur_l_bW = bind_req(ctx, "dur.lstm.bwd.W");
    W->dur_l_bR = bind_req(ctx, "dur.lstm.bwd.R");
    W->dur_l_bb = bind_req(ctx, "dur.lstm.bwd.b");
    W->dur_w    = bind_req(ctx, "dur_proj.weight");
    W->dur_b    = bind_req(ctx, "dur_proj.bias");
    // Acoustic text encoder
    W->ac_embd  = bind_req(ctx, "acoustic.embd.weight");
    W->ac_c0_w  = bind_req(ctx, "acoustic.cnn0.weight");
    W->ac_c0_b  = bind_req(ctx, "acoustic.cnn0.bias");
    W->ac_ln0_g = bind_req(ctx, "acoustic.ln0.gamma");
    W->ac_ln0_b = bind_req(ctx, "acoustic.ln0.beta");
    W->ac_c1_w  = bind_req(ctx, "acoustic.cnn1.weight");
    W->ac_c1_b  = bind_req(ctx, "acoustic.cnn1.bias");
    W->ac_ln1_g = bind_req(ctx, "acoustic.ln1.gamma");
    W->ac_ln1_b = bind_req(ctx, "acoustic.ln1.beta");
    W->ac_l_fW  = bind_req(ctx, "acoustic.lstm.fwd.W");
    W->ac_l_fR  = bind_req(ctx, "acoustic.lstm.fwd.R");
    W->ac_l_fb  = bind_req(ctx, "acoustic.lstm.fwd.b");
    W->ac_l_bW  = bind_req(ctx, "acoustic.lstm.bwd.W");
    W->ac_l_bR  = bind_req(ctx, "acoustic.lstm.bwd.R");
    W->ac_l_bb  = bind_req(ctx, "acoustic.lstm.bwd.b");
    // Shared LSTM (GenFront)
    W->sh_fW    = bind_req(ctx, "shared.lstm.fwd.W");
    W->sh_fR    = bind_req(ctx, "shared.lstm.fwd.R");
    W->sh_fb    = bind_req(ctx, "shared.lstm.fwd.b");
    W->sh_bW    = bind_req(ctx, "shared.lstm.bwd.W");
    W->sh_bR    = bind_req(ctx, "shared.lstm.bwd.R");
    W->sh_bb    = bind_req(ctx, "shared.lstm.bwd.b");
    // The F0/N/decoder/generator block tensors are looked up by
    // formatted name inside the inference path; not bound up-front.
}

// ---------------------------------------------------------------------------
// Public lifecycle
// ---------------------------------------------------------------------------

struct kittens_ctx * kittens_create(const char * gguf_path) {
    struct kittens_ctx * ctx = (struct kittens_ctx *)calloc(1, sizeof(*ctx));
    int success = 0;
    if (ctx != NULL) {
        ctx->gguf = gguf_open(gguf_path);
        if (ctx->gguf == NULL) {
            snprintf(ctx->err, sizeof(ctx->err),
                     "gguf_open(%s): %s",
                     gguf_path, gguf_last_error());
        } else {
            ctx->weights_arena = arena_new(64 * 1024 * 1024);
            // Scratch starts tiny (1 MB) and doubles on demand. Each
            // synthesize call ends with arena_reset, so resting
            // size is just the first slab. With initial=64MB the app
            // sat on 64MB of mostly-empty scratch even at idle; 1MB
            // is plenty for the smallest stages and grows as needed.
            ctx->scratch_arena = arena_new(1 * 1024 * 1024);
            if (load_arch(ctx)) {
                bind_weights(ctx);
                success = 1;
            }
        }
    }
    struct kittens_ctx * result = NULL;
    if (success) {
        result = ctx;
    } else if (ctx != NULL) {
        kittens_destroy(ctx);
    }
    return result;
}

void kittens_destroy(struct kittens_ctx * ctx) {
    if (ctx != NULL) {
        if (ctx->scratch_arena != NULL) {
            arena_free(ctx->scratch_arena);
        }
        if (ctx->weights_arena != NULL) {
            arena_free(ctx->weights_arena);
        }
        if (ctx->gguf != NULL) {
            gguf_close(ctx->gguf);
        }
        free(ctx->cache);
        free(ctx);
    }
}

// ---------------------------------------------------------------------------
// Lazy lookup of block weights by formatted name.
// ---------------------------------------------------------------------------

static struct tensor * named(struct kittens_ctx * ctx, const char * name) {
    // Search-loop post-condition: i == cache_count means not found.
    int i = 0;
    while (i < ctx->cache_count
           && strcmp(ctx->cache[i].name, name) != 0) {
        i++;
    }
    struct tensor * result;
    if (i < ctx->cache_count) {
        result = ctx->cache[i].t;
    } else {
        struct tensor * t = gguf_load_tensor(ctx->gguf,
                                            ctx->weights_arena, name);
        if (t != NULL) {
            if (ctx->cache_count == ctx->cache_cap) {
                int nc = ctx->cache_cap == 0
                    ? 128 : ctx->cache_cap * 2;
                ctx->cache = (struct named_entry *)realloc(
                    ctx->cache,
                    (size_t)nc * sizeof(struct named_entry));
                ctx->cache_cap = nc;
            }
            snprintf(ctx->cache[ctx->cache_count].name,
                     sizeof(ctx->cache[0].name), "%s", name);
            ctx->cache[ctx->cache_count].t = t;
            ctx->cache_count++;
        }
        result = t;
    }
    return result;
}

static struct tensor * named_fmt(struct kittens_ctx * ctx,
                                    const char * fmt, ...) {
    char buf[64];
    va_list ap; va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    return named(ctx, buf);
}

void kittens_audio_free(struct kittens_audio a) {
    free(a.samples);
}

void kittens_set_sgemm_impl(int impl) { tensor_set_sgemm_impl(impl); }
int  kittens_get_sgemm_impl(void)     { return tensor_get_sgemm_impl(); }

uint64_t kittens_current_phys_footprint(void) {
    return tensor_current_phys_footprint();
}
uint64_t kittens_peak_phys_footprint(void) {
    return tensor_peak_phys_footprint();
}
void kittens_reset_peak_phys_footprint(void) {
    tensor_reset_peak_phys_footprint();
}

// ---------------------------------------------------------------------------
// Primitive helpers (mirror ggml/kittens-tts.c)
// ---------------------------------------------------------------------------

// LayerNorm over ne[0] with optional gamma/beta. eps configurable.
static struct tensor * layer_norm(struct tensor * x, struct tensor * w,
                                 struct tensor * b, float eps) {
    struct tensor * h = tensor_norm(x, 0, eps);
    if (w != NULL) { h = tensor_mul(h, w); }
    if (b != NULL) { h = tensor_add(h, b); }
    return h;
}

// AdaLayerNorm on (C, L) tensors. style:(style_dim,), fcW:(style_dim, 2C),
// fcB:(2C,). Splits the projected style into gamma|beta and applies
// out = normed * (1 + gamma) + beta with broadcast over L.
static struct tensor * ada_layer_norm(struct tensor * x, struct tensor * style,
                                     struct tensor * fcW, struct tensor * fcB,
                                     int C) {
    struct tensor * h = tensor_mul_mat(fcW, style);     // (2C,)
    h = tensor_add(h, fcB);
    const size_t fsz = sizeof(float);
    struct tensor * gamma = tensor_view_1d(h, C, 0);
    struct tensor * beta  = tensor_view_1d(h, C, (size_t)C * fsz);
    struct tensor * n   = tensor_norm(x, 0, 1e-5f);
    struct tensor * n_g = tensor_mul(n, gamma);
    struct tensor * out = tensor_add(n_g, n);
    out = tensor_add(out, beta);
    return out;
}

// AdaIN1D on NLC tensors x:(C, L). style:(style_dim,), fcW/fcB project
// to 2C (gamma|beta). nW/nB are per-channel multiplicative norm
// (may be NULL).
//
// Instance norm normalizes over the time axis. With ne[0]=C and ne[1]=L
// we transpose to put L innermost, normalize, then transpose back.
static struct tensor * ada_in_1d(struct tensor * x, struct tensor * style,
                                 struct tensor * fcW, struct tensor * fcB,
                                 struct tensor * nW, struct tensor * nB,
                                 int C) {
    struct tensor * h = tensor_mul_mat(fcW, style);
    h = tensor_add(h, fcB);
    const size_t fsz = sizeof(float);
    struct tensor * gamma = tensor_view_1d(h, C, 0);
    struct tensor * beta  = tensor_view_1d(h, C, (size_t)C * fsz);
    struct tensor * gamma_c = tensor_cont(gamma);
    struct tensor * beta_c  = tensor_cont(beta);
    struct tensor * x_t = tensor_cont(tensor_transpose(x));   // (L, C)
    struct tensor * n_t = tensor_norm(x_t, 0, 1e-5f);
    struct tensor * n   = tensor_cont(tensor_transpose(n_t)); // (C, L)
    if (nW != NULL) { n = tensor_mul(n, nW); }
    if (nB != NULL) { n = tensor_add(n, nB); }
    struct tensor * n_g = tensor_mul(n, gamma_c);
    struct tensor * out = tensor_add(n_g, n);
    out = tensor_add(out, beta_c);
    return out;
}

// Snake activation: x + (1/alpha) * sin(alpha*x)^2. x:(C, L), alpha:(C,)
static struct tensor * snake_1d(struct tensor * x, struct tensor * alpha) {
    struct tensor * ax = tensor_mul(x, alpha);
    struct tensor * s  = tensor_sin(ax);
    struct tensor * s2 = tensor_mul(s, s);
    struct tensor * s2_over_a = tensor_div(s2, alpha);
    return tensor_add(x, s2_over_a);
}

// Conv1d on NLC tensor x:(C, L). Returns NLC (Cout, Lout). Kernel kw
// stored ne=(K, Cin, Cout). pad < 0 means SAME ((K-1)/2).
static struct tensor * conv1d_nlc(struct tensor * x, struct tensor * kw,
                                 struct tensor * kb,
                                 int stride, int pad, int dilation) {
    const int K = (int)kw->ne[0];
    if (pad < 0) { pad = (K - 1) / 2; }
    struct tensor * x_ncl = tensor_cont(tensor_transpose(x));   // (L, C)
    struct tensor * x_3d = tensor_reshape_3d(x_ncl,
                                     x_ncl->ne[0], x_ncl->ne[1], 1);
    struct tensor * y_3d = tensor_conv_1d(kw, x_3d, stride, pad, dilation);
    if (kb != NULL) {
        // bias (Cout,) broadcast along L via ne[1] of y_3d which is Cout.
        // y_3d ne=(Lout, Cout, 1). kb ne=(Cout,). We want to add per
        // channel: build a (1, Cout, 1) view.
        struct tensor * b3 = tensor_reshape_3d(kb, 1, kb->ne[0], 1);
        y_3d = tensor_add(y_3d, b3);
    }
    struct tensor * y_2d = tensor_reshape_2d(y_3d, y_3d->ne[0], y_3d->ne[1]);
    return tensor_cont(tensor_transpose(y_2d));             // (Cout, Lout) NLC
}

// Repeat-interleave 2x along the L axis: (C, L) -> (C, 2L).
// output[c, 2l]   = x[c, l]
// output[c, 2l+1] = x[c, l]
static struct tensor * repeat_interleave_2x_nlc(struct tensor * x) {
    const int64_t C = x->ne[0];
    const int64_t L = x->ne[1];
    struct tensor * x4 = tensor_reshape_4d(x, C, L, 1, 1);
    struct tensor * stacked = tensor_concat(x4, x4, 2);     // (C, L, 2, 1)
    struct tensor * p = tensor_permute(stacked, 0, 2, 1, 3); // (C, 2, L, 1)
    struct tensor * pc = tensor_cont(p);
    return tensor_reshape_2d(pc, C, 2 * L);
}

// Insert one zero between every pair of adjacent L-positions:
// (C, L) -> (C, 2L), output[c, 2t]=x[c,t], output[c, 2t+1]=0.
static struct tensor * insert_zeros_2x_nlc(struct tensor * x) {
    const int64_t C = x->ne[0];
    const int64_t L = x->ne[1];
    struct tensor * zeros = tensor_scale(x, 0.0f);
    struct tensor * x4 = tensor_reshape_4d(x, C, L, 1, 1);
    struct tensor * z4 = tensor_reshape_4d(zeros, C, L, 1, 1);
    struct tensor * stacked = tensor_concat(x4, z4, 2);
    struct tensor * p = tensor_permute(stacked, 0, 2, 1, 3);
    struct tensor * pc = tensor_cont(p);
    return tensor_reshape_2d(pc, C, 2 * L);
}

// Depthwise conv-transpose-1d 2x (stride=2, padding=1, K=3) implemented
// via insert-zeros + depthwise conv on the K-flipped kernel. pool_w
// stored ne=(K, 1, C) with K-axis pre-flipped at convert time.
static struct tensor * upsample_2x_dwT(struct tensor * x,
                                      struct tensor * pool_w,
                                      struct tensor * pool_b) {
    const int K = (int)pool_w->ne[0];
    assert(K == 3);
    struct tensor * xup = insert_zeros_2x_nlc(x);
    struct tensor * xup_ncl = tensor_cont(tensor_transpose(xup));
    struct tensor * xup_3d  = tensor_reshape_3d(xup_ncl,
                                        xup_ncl->ne[0],
                                        xup_ncl->ne[1], 1);
    struct tensor * y_3d = tensor_conv_1d_dw(pool_w, xup_3d, 1, 1, 1);
    if (pool_b != NULL) {
        struct tensor * b3 = tensor_reshape_3d(pool_b, 1, pool_b->ne[0], 1);
        y_3d = tensor_add(y_3d, b3);
    }
    struct tensor * y_2d = tensor_reshape_2d(y_3d, y_3d->ne[0], y_3d->ne[1]);
    return tensor_cont(tensor_transpose(y_2d));
}

// ConvTranspose1d on NLC: x:(Cin, L) -> (Cout, Lout). Kernel kw stored
// ne=(K, Cout, Cin) (PyTorch (Cin, Cout, K) packed by the converter).
// tensor_conv_transpose_1d handles symmetric padding internally.
static struct tensor * conv_transpose_1d_nlc(struct tensor * x,
                                            struct tensor * kw,
                                            struct tensor * kb,
                                            int stride, int pad) {
    struct tensor * b = tensor_cont(tensor_transpose(x));   // (L, Cin)
    struct tensor * b_3d = tensor_reshape_3d(b, b->ne[0], b->ne[1], 1);
    struct tensor * y_3d = tensor_conv_transpose_1d(kw, b_3d, stride, pad);
    if (kb != NULL) {
        struct tensor * bb = tensor_reshape_3d(kb, 1, kb->ne[0], 1);
        y_3d = tensor_add(y_3d, bb);
    }
    struct tensor * y_2d = tensor_reshape_2d(y_3d, y_3d->ne[0], y_3d->ne[1]);
    return tensor_cont(tensor_transpose(y_2d));
}

// Reflection-pad LEFT only by 1 sample on NLC tensor x:(C, L).
// Prepends x[:, 1:2] to x -> (C, L+1). Only n=1 supported (only use
// in the model is iSTFT n=1).
static struct tensor * reflection_pad_left(struct tensor * x, int n) {
    struct tensor * result = x;
    if (n > 0) {
        assert(n == 1);
        const int64_t C = x->ne[0];
        struct tensor * slice = tensor_view_2d(x, C, 1, (size_t)x->nb[1],
                                       (size_t)x->nb[1]);
        struct tensor * slice_c = tensor_cont(slice);
        result = tensor_concat(slice_c, x, 1);
    }
    return result;
}

// Broadcast a 1D style (C,) to a (C, L) tensor — every column a copy
// of style. Output goes to the active arena set in kittens_synthesize.
static struct tensor * style_bcast_CxL(struct tensor * style, int C, int L) {
    struct tensor * s2 = tensor_reshape_2d(style, C, 1);
    return tensor_repeat_to(s2, 2, C, L, 1, 1);
}

// ---------------------------------------------------------------------------
// LSTM helpers (eager — no graph plumbing)
// ---------------------------------------------------------------------------
//
// One direction of a bidir LSTM. ifgo gate order.
//   x:  (in_size, T)
//   W:  (in_size, 4H)
//   R:  (H,       4H)
//   b:  (4H,)
//   h0, c0: (H,) — initial states (zero tensors)
//   returns (H, T) packed
static struct tensor * lstm_dir(struct arena * a,
                                struct tensor * x, struct tensor * W,
                                struct tensor * R, struct tensor * b,
                                struct tensor * h0, struct tensor * c0,
                                int H, int T, int reverse) {
    struct tensor * Wx_full = tensor_mul_mat(W, x);     // (4H, T)
    Wx_full = tensor_add(Wx_full, b);                   // (4H,) over T
    struct tensor * out = tensor_new_2d(a, H, T);
    struct tensor * h_prev = h0;
    struct tensor * c_prev = c0;
    const size_t fsz = sizeof(float);
    for (int step = 0; step < T; step++) {
        const int t = reverse ? (T - 1 - step) : step;
        struct tensor * Wx_t = tensor_view_1d(Wx_full, 4 * H,
                                              (size_t)t * 4 * H * fsz);
        struct tensor * Rh   = tensor_mul_mat(R, h_prev);
        struct tensor * z    = tensor_add(Wx_t, Rh);
        struct tensor * zi = tensor_view_1d(z, H, 0);
        struct tensor * zf = tensor_view_1d(z, H, (size_t)1 * H * fsz);
        struct tensor * zg = tensor_view_1d(z, H, (size_t)2 * H * fsz);
        struct tensor * zo = tensor_view_1d(z, H, (size_t)3 * H * fsz);
        struct tensor * gi  = tensor_sigmoid(zi);
        struct tensor * gf_ = tensor_sigmoid(zf);
        struct tensor * gg  = tensor_tanh   (zg);
        struct tensor * go  = tensor_sigmoid(zo);
        struct tensor * fc  = tensor_mul(gf_, c_prev);
        struct tensor * ig  = tensor_mul(gi,  gg);
        struct tensor * c_t = tensor_add(fc, ig);
        struct tensor * h_t = tensor_mul(go, tensor_tanh(c_t));
        // out is packed (H, T); write h_t into column t at byte t*H*4.
        memcpy((char *)out->data + (size_t)t * H * fsz,
               h_t->data, (size_t)H * fsz);
        h_prev = h_t;
        c_prev = c_t;
    }
    return out;
}

// Bidirectional LSTM: returns (2H, T), forward concatenated with
// backward along ne[0].
static struct tensor * bidir_lstm(struct arena * a,
                                 struct tensor * x,
                                 struct tensor * fW, struct tensor * fR,
                                 struct tensor * fb,
                                 struct tensor * bW, struct tensor * bR,
                                 struct tensor * bb,
                                 struct tensor * h0, struct tensor * c0,
                                 int H, int T) {
    struct tensor * fwd = lstm_dir(a, x, fW, fR, fb, h0, c0, H, T, 0);
    struct tensor * bwd = lstm_dir(a, x, bW, bR, bb, h0, c0, H, T, 1);
    return tensor_concat(fwd, bwd, 0);
}

// ---------------------------------------------------------------------------
// AdaINResBlock1D builder (used in F0/N paths and decoder)
// ---------------------------------------------------------------------------

static struct tensor * build_ada_block_1d(struct kittens_ctx * ctx, const char * prefix,
                                      struct tensor * x, struct tensor * style,
                                      struct tensor * shortcut_in, int divide) {
    struct tensor * n1_fcW = named_fmt(ctx, "%s.n1.fcW", prefix);
    struct tensor * n1_fcB = named_fmt(ctx, "%s.n1.fcB", prefix);
    struct tensor * n1_nW  = named_fmt(ctx, "%s.n1.nW",  prefix);
    struct tensor * n1_nB  = named_fmt(ctx, "%s.n1.nB",  prefix);
    struct tensor * n2_fcW = named_fmt(ctx, "%s.n2.fcW", prefix);
    struct tensor * n2_fcB = named_fmt(ctx, "%s.n2.fcB", prefix);
    struct tensor * n2_nW  = named_fmt(ctx, "%s.n2.nW",  prefix);
    struct tensor * n2_nB  = named_fmt(ctx, "%s.n2.nB",  prefix);
    struct tensor * c1_w   = named_fmt(ctx, "%s.c1.weight", prefix);
    struct tensor * c1_b   = named_fmt(ctx, "%s.c1.bias",   prefix);
    struct tensor * c2_w   = named_fmt(ctx, "%s.c2.weight", prefix);
    struct tensor * c2_b   = named_fmt(ctx, "%s.c2.bias",   prefix);
    struct tensor * sv_w   = named_fmt(ctx, "%s.sv.weight", prefix);
    struct tensor * sv_b   = named_fmt(ctx, "%s.sv.bias",   prefix);
    struct tensor * pool_w = named_fmt(ctx, "%s.pool.weight", prefix);
    struct tensor * pool_b = named_fmt(ctx, "%s.pool.bias",   prefix);

    assert(n1_fcW != NULL && n2_fcW != NULL
           && c1_w != NULL && c2_w != NULL);

    const int upsample    = (pool_w != NULL);
    const int has_conv1x1 = (sv_w   != NULL);
    const int Cin = (int)x->ne[0];

    struct tensor * h = ada_in_1d(x, style, n1_fcW, n1_fcB,
                                 n1_nW, n1_nB, Cin);
    h = tensor_leaky_relu(h, 0.2f);

    if (upsample) { h = upsample_2x_dwT(h, pool_w, pool_b); }

    h = conv1d_nlc(h, c1_w, c1_b, 1, -1, 1);

    const int Cmid = (int)c1_w->ne[2];
    h = ada_in_1d(h, style, n2_fcW, n2_fcB,
                     n2_nW, n2_nB, Cmid);
    h = tensor_leaky_relu(h, 0.2f);

    h = conv1d_nlc(h, c2_w, c2_b, 1, -1, 1);

    struct tensor * shortcut = shortcut_in != NULL ? shortcut_in : x;
    struct tensor * res;
    if (upsample) {
        struct tensor * sup = repeat_interleave_2x_nlc(shortcut);
        res = conv1d_nlc(sup, sv_w, sv_b, 1, 0, 1);
    } else if (has_conv1x1) {
        res = conv1d_nlc(shortcut, sv_w, sv_b, 1, 0, 1);
    } else {
        res = shortcut;
    }

    struct tensor * out = tensor_add(h, res);
    if (divide) { out = tensor_scale(out, 1.0f / sqrtf(2.0f)); }
    return out;
}

// ---------------------------------------------------------------------------
// AdaINResBlockHiFiGAN (3 dilations, snake activation, residual)
// ---------------------------------------------------------------------------
//
// x:(C, L). Returns (C, L). The block has 3 sub-iterations with
// dilations (1, 3, 5); each sub-iteration is two AdaIN+snake+conv
// passes plus a residual add.
// Checkpoint helpers — copy tensor data to malloc'd host memory so we
// can reset the scratch arena between sub-stages without losing the
// state we need next. Without this, every intermediate tensor inside
// build_hifi_block (~21 per dilation × 3 dilations) accumulates in
// scratch and we OOM on iOS for long sentences.
struct savept {
    int      ndim;
    int64_t  ne[4];
    float *  data;
};

static struct savept save(const struct tensor * t) {
    struct savept p;
    p.ndim = t->ndim;
    for (int i = 0; i < 4; i++) { p.ne[i] = t->ne[i]; }
    int64_t n = p.ne[0] * p.ne[1] * p.ne[2] * p.ne[3];
    p.data = (float *)malloc((size_t)n * sizeof(float));
    memcpy(p.data, t->data, (size_t)n * sizeof(float));
    return p;
}

static struct tensor * restore(struct arena * a, const struct savept * p) {
    struct tensor * t = tensor_new_nd(a, p->ndim, p->ne);
    int64_t n = p->ne[0] * p->ne[1] * p->ne[2] * p->ne[3];
    memcpy(t->data, p->data, (size_t)n * sizeof(float));
    return t;
}

static void savept_free(struct savept * p) {
    if (p && p->data) { free(p->data); p->data = NULL; }
}

static struct tensor * build_hifi_block(struct kittens_ctx * ctx, const char * prefix,
                                    struct tensor * x, struct tensor * style) {
    struct arena * sa = ctx->scratch_arena;
    const int C = (int)x->ne[0];
    static const int dilations[3] = { 1, 3, 5 };
    // Save x AND style to host THEN arena_reset BEFORE k=0. Without
    // this, k=0's ~350 MB working set stacks on top of whatever the
    // caller left in arena (typically ~600 MB during noise-contribs)
    // and per-call peak balloons past iOS's process limit. The k>0
    // resets inside the loop already follow this pattern; the missing
    // one was at entry.
    struct savept ps = save(style);
    struct savept px0 = save(x);
    arena_reset(sa);
    struct tensor * out = restore(sa, &px0);
    style = restore(sa, &ps);
    savept_free(&px0);
#ifdef OOM_TRACE
    { const int L = 0, F = 0;
      fprintf(stderr, "    [OOM hifi-entry  %-9s     phys=%7.1f MB]\n",
              prefix, oom_mb()); (void)L; (void)F; }
#endif
    for (int k = 0; k < 3; k++) {
        if (k > 0) {
            struct savept po = save(out);
            arena_reset(sa);
            out   = restore(sa, &po);  savept_free(&po);
            style = restore(sa, &ps);
        }
        const int d = dilations[k];
        struct tensor * a1_fcW = named_fmt(ctx, "%s.a1.%d.fcW", prefix, k);
        struct tensor * a1_fcB = named_fmt(ctx, "%s.a1.%d.fcB", prefix, k);
        struct tensor * a1_nW  = named_fmt(ctx, "%s.a1.%d.nW",  prefix, k);
        struct tensor * a1_nB  = named_fmt(ctx, "%s.a1.%d.nB",  prefix, k);
        struct tensor * a2_fcW = named_fmt(ctx, "%s.a2.%d.fcW", prefix, k);
        struct tensor * a2_fcB = named_fmt(ctx, "%s.a2.%d.fcB", prefix, k);
        struct tensor * a2_nW  = named_fmt(ctx, "%s.a2.%d.nW",  prefix, k);
        struct tensor * a2_nB  = named_fmt(ctx, "%s.a2.%d.nB",  prefix, k);
        struct tensor * al1    = named_fmt(ctx, "%s.al1.%d",    prefix, k);
        struct tensor * al2    = named_fmt(ctx, "%s.al2.%d",    prefix, k);
        struct tensor * c1_w   = named_fmt(ctx, "%s.c1.%d.weight", prefix, k);
        struct tensor * c1_b   = named_fmt(ctx, "%s.c1.%d.bias",   prefix, k);
        struct tensor * c2_w   = named_fmt(ctx, "%s.c2.%d.weight", prefix, k);
        struct tensor * c2_b   = named_fmt(ctx, "%s.c2.%d.bias",   prefix, k);
        const int K1 = (int)c1_w->ne[0];
        const int K2 = (int)c2_w->ne[0];
        struct tensor * h = ada_in_1d(out, style, a1_fcW, a1_fcB,
                                     a1_nW, a1_nB, C);
        h = snake_1d(h, al1);
        h = conv1d_nlc(h, c1_w, c1_b, 1, d * (K1 - 1) / 2, d);
        h = ada_in_1d(h, style, a2_fcW, a2_fcB, a2_nW, a2_nB, C);
        h = snake_1d(h, al2);
        h = conv1d_nlc(h, c2_w, c2_b, 1, (K2 - 1) / 2, 1);
        out = tensor_add(out, h);
#ifdef OOM_TRACE
        { const int L = 0, F = 0;
          fprintf(stderr, "    [OOM hifi-k%d-end %-9s     phys=%7.1f MB]\n",
                  k, prefix, oom_mb()); (void)L; (void)F; }
#endif
    }
    savept_free(&ps);
    return out;
}

// ---------------------------------------------------------------------------
// BERT/Albert encoder
// ---------------------------------------------------------------------------

static struct tensor * build_albert(struct kittens_ctx * ctx, int L,
                                const int32_t * ids,
                                const int32_t * pos,
                                const int32_t * type) {
    const struct arch * a = &ctx->arch;
    const struct weights * W = &ctx->W;

    struct tensor * h = tensor_get_rows(W->e_word, ids,  L);
    struct tensor * p = tensor_get_rows(W->e_pos,  pos,  L);
    struct tensor * t = tensor_get_rows(W->e_type, type, L);
    h = tensor_add(h, p);
    h = tensor_add(h, t);
    h = layer_norm(h, W->e_ln_w, W->e_ln_b, a->ln_eps);
    h = tensor_mul_mat(W->proj_w, h);
    h = tensor_add(h, W->proj_b);
    const float kq_scale = 1.0f / sqrtf((float)a->head_dim);
    for (int il = 0; il < a->n_layers; il++) {
        struct tensor * residual = h;
        struct tensor * q = tensor_add(tensor_mul_mat(W->q_w, h), W->q_b);
        struct tensor * k = tensor_add(tensor_mul_mat(W->k_w, h), W->k_b);
        struct tensor * v = tensor_add(tensor_mul_mat(W->v_w, h), W->v_b);
        q = tensor_reshape_4d(q, a->head_dim, a->n_heads, L, 1);
        k = tensor_reshape_4d(k, a->head_dim, a->n_heads, L, 1);
        v = tensor_reshape_4d(v, a->head_dim, a->n_heads, L, 1);
        q = tensor_permute(q, 0, 2, 1, 3);     // (head_dim, L, n_heads, 1)
        k = tensor_permute(k, 0, 2, 1, 3);
        v = tensor_permute(v, 0, 2, 1, 3);
        v = tensor_cont(tensor_transpose(v));  // (L, head_dim, n_heads, 1)
        struct tensor * kq = tensor_mul_mat(k, q);     // (L, L, n_heads, 1)
        kq = tensor_softmax(kq, 0, kq_scale);
        struct tensor * kqv = tensor_mul_mat(v, kq);   // (head_dim, L, n_heads)
        kqv = tensor_permute(kqv, 0, 2, 1, 3);         // (head_dim, n_heads, L)
        kqv = tensor_cont_2d(kqv, a->hidden, L);       // (hidden, L)
        struct tensor * att_out =
            tensor_add(tensor_mul_mat(W->o_w, kqv), W->o_b);
        h = tensor_add(att_out, residual);
        h = layer_norm(h, W->attn_ln_w, W->attn_ln_b, a->ln_eps);
        struct tensor * mid = h;
        struct tensor * ffn =
            tensor_add(tensor_mul_mat(W->ffn_w, h), W->ffn_b);
        ffn = tensor_gelu_erf(ffn);
        ffn = tensor_add(tensor_mul_mat(W->ffn_out_w, ffn), W->ffn_out_b);
        h = tensor_add(ffn, mid);
        h = layer_norm(h, W->full_ln_w, W->full_ln_b, a->ln_eps);
    }
    return h;
}

// ---------------------------------------------------------------------------
// PredictorTextEncoder
// ---------------------------------------------------------------------------

static struct tensor * build_pred_text(struct kittens_ctx * ctx,
                                       struct tensor * bert_out,
                                       struct tensor * style,
                                       struct tensor * h0,
                                       struct tensor * c0, int L) {
    const struct weights * W = &ctx->W;
    struct arena * sa = ctx->scratch_arena;
    const int C = 128;
    const int H = ctx->arch.lstm_hidden;
    struct tensor * s_bcast = style_bcast_CxL(style, C, L);
    struct tensor * x = tensor_concat(bert_out, s_bcast, 0);   // (384, L)
    struct tensor * y = bidir_lstm(sa, x,
                                   W->pt_l0_fW, W->pt_l0_fR, W->pt_l0_fb,
                                   W->pt_l0_bW, W->pt_l0_bR, W->pt_l0_bb,
                                   h0, c0, H, L);
    struct tensor * y1 = ada_layer_norm(y, style, W->pt_fc1_w,
                                        W->pt_fc1_b, C);
    struct tensor * x2 = tensor_concat(y1, s_bcast, 0);        // (256, L)
    struct tensor * y2 = bidir_lstm(sa, x2,
                                    W->pt_l2_fW, W->pt_l2_fR, W->pt_l2_fb,
                                    W->pt_l2_bW, W->pt_l2_bR, W->pt_l2_bb,
                                    h0, c0, H, L);
    return ada_layer_norm(y2, style, W->pt_fc3_w, W->pt_fc3_b, C);
}

// ---------------------------------------------------------------------------
// AcousticTextEncoder
// ---------------------------------------------------------------------------

static struct tensor * build_acoustic(struct kittens_ctx * ctx,
                                      const int32_t * ids,
                                      struct tensor * h0,
                                      struct tensor * c0, int L) {
    const struct weights * W = &ctx->W;
    struct arena * sa = ctx->scratch_arena;
    const int H = ctx->arch.lstm_hidden;
    struct tensor * x = tensor_get_rows(W->ac_embd, ids, L);  // (128, L) NLC
    for (int i = 0; i < 2; i++) {
        struct tensor * cnnW = (i == 0) ? W->ac_c0_w : W->ac_c1_w;
        struct tensor * cnnB = (i == 0) ? W->ac_c0_b : W->ac_c1_b;
        struct tensor * lnG  = (i == 0) ? W->ac_ln0_g : W->ac_ln1_g;
        struct tensor * lnB  = (i == 0) ? W->ac_ln0_b : W->ac_ln1_b;
        const int K = (int)cnnW->ne[0];
        const int pad = (K - 1) / 2;
        struct tensor * x_ncl = tensor_cont(tensor_transpose(x));
        struct tensor * x_ncl_3d = tensor_reshape_3d(x_ncl,
                                                    x_ncl->ne[0],
                                                    x_ncl->ne[1], 1);
        struct tensor * y_3d = tensor_conv_1d(cnnW, x_ncl_3d, 1, pad, 1);
        struct tensor * b3 = tensor_reshape_3d(cnnB, 1, cnnB->ne[0], 1);
        y_3d = tensor_add(y_3d, b3);

        struct tensor * y_2d = tensor_reshape_2d(y_3d, y_3d->ne[0], y_3d->ne[1]);
        x = tensor_cont(tensor_transpose(y_2d));
        x = layer_norm(x, lnG, lnB, 1e-5f);
        x = tensor_leaky_relu(x, 0.2f);
    }

    struct tensor * y = bidir_lstm(sa, x,
                                  W->ac_l_fW, W->ac_l_fR, W->ac_l_fb,
                                  W->ac_l_bW, W->ac_l_bR, W->ac_l_bb,
                                  h0, c0, H, L);
    return y;
}

// ---------------------------------------------------------------------------
// TextStage orchestrator
// ---------------------------------------------------------------------------

struct textstage_outs {
    struct tensor * prosody256;
    struct tensor * text;
    struct tensor * dur_sig;
};

static struct textstage_outs build_textstage(struct kittens_ctx * ctx,
                                             int L,
                                             const int32_t * ids,
                                             const int32_t * pos,
                                             const int32_t * type,
                                             struct tensor * style_pr,
                                             struct tensor * h0,
                                             struct tensor * c0) {
    const struct weights * W = &ctx->W;
    struct arena * sa = ctx->scratch_arena;
    struct tensor * bert = build_albert(ctx, L, ids, pos, type);
    struct tensor * bert_proj = tensor_add(
        tensor_mul_mat(W->bert_enc_w, bert), W->bert_enc_b);   // (256, L)
    struct tensor * prosody = build_pred_text(ctx, bert_proj, style_pr,
                                              h0, c0, L);      // (128, L)
    struct tensor * s_bcast = style_bcast_CxL(style_pr,
                                              ctx->arch.style_dim, L);
    struct tensor * prosody256 = tensor_concat(prosody, s_bcast, 0);
    struct tensor * dlstm = bidir_lstm(sa, prosody256,
                                       W->dur_l_fW, W->dur_l_fR, W->dur_l_fb,
                                       W->dur_l_bW, W->dur_l_bR, W->dur_l_bb,
                                       h0, c0, ctx->arch.lstm_hidden, L);
    struct tensor * dur_logits = tensor_add(
        tensor_mul_mat(W->dur_w, dlstm), W->dur_b);
    struct tensor * dur_sig = tensor_sigmoid(dur_logits);
    struct tensor * text = build_acoustic(ctx, ids, h0, c0, L);
    struct textstage_outs r = { prosody256, text, dur_sig };
    return r;
}

// ---------------------------------------------------------------------------
// GenFront: shared LSTM + 6 AdaINResBlock1D + f0_proj / n_proj
// ---------------------------------------------------------------------------

struct genfront_outs {
    struct tensor * f0_proj;   // (1, 2F)
    struct tensor * n_proj;    // (1, 2F)
};

static struct genfront_outs build_genfront(struct kittens_ctx * ctx,
                                           struct tensor * prosody_lr_nlc,
                                           struct tensor * style,
                                           struct tensor * h0,
                                           struct tensor * c0, int F) {
    const struct weights * W = &ctx->W;
    struct arena * sa = ctx->scratch_arena;
    const int H = ctx->arch.lstm_hidden;
    struct tensor * sh = bidir_lstm(sa, prosody_lr_nlc,
                                    W->sh_fW, W->sh_fR, W->sh_fb,
                                    W->sh_bW, W->sh_bR, W->sh_bb,
                                    h0, c0, H, F);
    struct tensor * f0 = build_ada_block_1d(ctx, "f0.0", sh, style, sh, 1);
    f0 = build_ada_block_1d(ctx, "f0.1", f0, style, f0, 1);
    f0 = build_ada_block_1d(ctx, "f0.2", f0, style, f0, 1);
    struct tensor * f0_proj_w = named_fmt(ctx, "f0_proj.weight");
    struct tensor * f0_proj_b = named_fmt(ctx, "f0_proj.bias");
    struct tensor * f0p = conv1d_nlc(f0, f0_proj_w, f0_proj_b, 1, -1, 1);
    struct tensor * nx = build_ada_block_1d(ctx, "n.0", sh, style, sh, 1);
    nx = build_ada_block_1d(ctx, "n.1", nx, style, nx, 1);
    nx = build_ada_block_1d(ctx, "n.2", nx, style, nx, 1);
    struct tensor * n_proj_w = named_fmt(ctx, "n_proj.weight");
    struct tensor * n_proj_b = named_fmt(ctx, "n_proj.bias");
    struct tensor * np = conv1d_nlc(nx, n_proj_w, n_proj_b, 1, -1, 1);
    struct genfront_outs r = { f0p, np };
    return r;
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

static struct tensor * build_decoder(struct kittens_ctx * ctx,
                                     struct tensor * text_lr,
                                     struct tensor * f0_proj,
                                     struct tensor * n_proj,
                                     struct tensor * style_aco) {
    struct tensor * asrW = named_fmt(ctx, "dec.asr.weight");
    struct tensor * asrB = named_fmt(ctx, "dec.asr.bias");
    struct tensor * f0W  = named_fmt(ctx, "dec.f0_conv.weight");
    struct tensor * f0B  = named_fmt(ctx, "dec.f0_conv.bias");
    struct tensor * nW   = named_fmt(ctx, "dec.n_conv.weight");
    struct tensor * nB   = named_fmt(ctx, "dec.n_conv.bias");
    struct tensor * asr   = conv1d_nlc(text_lr, asrW, asrB, 1, 0, 1);
    struct tensor * f0_dn = conv1d_nlc(f0_proj, f0W, f0B, 2, 1, 1);
    struct tensor * n_dn  = conv1d_nlc(n_proj,  nW,  nB,  2, 1, 1);
    struct tensor * enc_in = tensor_concat(text_lr, f0_dn, 0);
    enc_in = tensor_concat(enc_in, n_dn, 0);
    struct tensor * x = build_ada_block_1d(ctx, "dec.encode", enc_in,
                                           style_aco, enc_in, 1);
    for (int i = 0; i < 4; i++) {
        char prefix[32];
        snprintf(prefix, sizeof(prefix), "dec.decode.%d", i);
        struct tensor * x_cat = tensor_concat(x, asr, 0);
        x_cat = tensor_concat(x_cat, f0_dn, 0);
        x_cat = tensor_concat(x_cat, n_dn,  0);
        x = build_ada_block_1d(ctx, prefix, x_cat, style_aco, x_cat, 1);
    }
    return x;
}

// ---------------------------------------------------------------------------
// Noise contributions (sine excitation + STFT analysis + noise_res blocks)
// ---------------------------------------------------------------------------

struct noise_outs { struct tensor * nr0; struct tensor * nr1; };

static struct noise_outs build_noise_contribs(struct kittens_ctx * ctx,
                                              struct tensor * f0_proj,
                                              struct tensor * style_aco,
                                              struct tensor * harmonics,
                                              struct tensor * s_range,
                                              struct tensor * eps_t,
                                              int F) {
    struct arena * sa = ctx->scratch_arena;
    const int T_frames = 2 * F;
    const int hop      = 300;
    const int T_audio  = T_frames * hop;
    const float sr     = 24000.0f;
    const float two_pi = 2.0f * (float)M_PI;
#ifdef OOM_TRACE
    const int L = 0; (void)T_audio;  // OOM_MARK uses L; not relevant here
#endif
    OOM_MARK("noise:entry");
    // f0_audio: nearest-neighbor upsample (1, 2F) -> (1, T_audio).
    struct tensor * f0_3d  = tensor_reshape_3d(f0_proj, 1, 1, T_frames);
    struct tensor * f0_audio_3d = tensor_repeat_to(f0_3d, 3,
                                                   1, hop, T_frames, 1);
    struct tensor * f0_audio = tensor_reshape_2d(f0_audio_3d, 1, T_audio);
    struct tensor * voiced = tensor_step(f0_audio);
    OOM_MARK("noise:f0+voiced");
    // f0_per_frame[h, t] = f0_proj[t] * (h+1). Repeat (1, 2F) -> (9, 2F).
    struct tensor * f0_repeated = tensor_repeat_to(f0_proj, 2,
                                                   9, T_frames, 1, 1);
    struct tensor * harm_2d = tensor_reshape_2d(harmonics, 9, 1);
    struct tensor * f0_per_frame = tensor_mul(f0_repeated, harm_2d);
    // step = f0_per_frame * (hop/sr); phase_start = (cumsum-step)*2π.
    struct tensor * step_nlc = tensor_scale(f0_per_frame, (float)hop / sr);
    struct tensor * step_ncl = tensor_cont(tensor_transpose(step_nlc));
    struct tensor * cs       = tensor_cumsum(step_ncl, 0);
    struct tensor * ps_ncl   = tensor_sub(cs, step_ncl);
    ps_ncl = tensor_scale(ps_ncl, two_pi);
    struct tensor * phase_start_nlc = tensor_cont(tensor_transpose(ps_ncl));
    OOM_MARK("noise:phase_start");
    // phase_within[h, t, s] = f0_per_frame[h, t] * s * (2π/sr)
    struct tensor * fpf_3d = tensor_reshape_3d(f0_per_frame, 9, T_frames, 1);
    struct tensor * s_3d   = tensor_reshape_3d(s_range, 1, 1, hop);
    struct tensor * fpf_x  = tensor_repeat_to(fpf_3d, 3, 9, T_frames, hop, 1);
    struct tensor * s_x    = tensor_repeat_to(s_3d,   3, 9, T_frames, hop, 1);
    struct tensor * within = tensor_mul(fpf_x, s_x);
    within = tensor_scale(within, two_pi / sr);
    OOM_MARK("noise:within(9,2F,hop)");
    struct tensor * ps_3d = tensor_reshape_3d(phase_start_nlc,
                                              9, T_frames, 1);
    struct tensor * ps_expanded = tensor_repeat_to(ps_3d, 3,
                                                   9, T_frames, hop, 1);
    struct tensor * phase = tensor_add(ps_expanded, within);
    OOM_MARK("noise:phase");
    phase = tensor_permute(phase, 0, 2, 1, 3);     // (9, hop, 2F)
    phase = tensor_cont(phase);
    phase = tensor_reshape_2d(phase, 9, T_audio);
    OOM_MARK("noise:phase-perm");
    struct tensor * sines = tensor_scale(tensor_sin(phase), 0.1f);
    struct tensor * sin_gen = tensor_mul(sines, voiced);
    OOM_MARK("noise:sin_gen");
    struct tensor * l_lin_w = named_fmt(ctx, "l_lin.weight");
    struct tensor * l_lin_b = named_fmt(ctx, "l_lin.bias");
    struct tensor * mixed = tensor_add(
        tensor_mul_mat(l_lin_w, sin_gen), l_lin_b);
    struct tensor * excitation = tensor_tanh(mixed);
    OOM_MARK("noise:excitation");
    struct tensor * stft_fr = named_fmt(ctx, "stft_fwd.real");
    struct tensor * stft_fi = named_fmt(ctx, "stft_fwd.imag");
    struct tensor * stft_real = conv1d_nlc(excitation, stft_fr,
                                           NULL, 5, 10, 1);
    struct tensor * stft_imag = conv1d_nlc(excitation, stft_fi,
                                           NULL, 5, 10, 1);
    OOM_MARK("noise:stft");
    struct tensor * re2 = tensor_mul(stft_real, stft_real);
    struct tensor * im2 = tensor_mul(stft_imag, stft_imag);
    struct tensor * mag2 = tensor_add(re2, im2);
    mag2 = tensor_add(mag2, eps_t);
    struct tensor * mag = tensor_sqrt(mag2);
    struct tensor * phi = tensor_atan2(stft_imag, stft_real);
    struct tensor * stft_out = tensor_concat(mag, phi, 0);
    struct tensor * nc0_w = named_fmt(ctx, "nc0.weight");
    struct tensor * nc0_b = named_fmt(ctx, "nc0.bias");
    struct tensor * nc1_w = named_fmt(ctx, "nc1.weight");
    struct tensor * nc1_b = named_fmt(ctx, "nc1.bias");
    struct tensor * nc0 = conv1d_nlc(stft_out, nc0_w, nc0_b, 6, 3, 1);
    struct tensor * nc1 = conv1d_nlc(stft_out, nc1_w, nc1_b, 1, 0, 1);
    OOM_MARK("noise:nc0+nc1");
    // build_hifi_block now does save+arena_reset at entry, so we only
    // need to preserve what we need AFTER its first call: nc1 (input
    // to second hifi block) and style_aco (input to both). nc0 is
    // saved by build_hifi_block itself.
    struct savept p_nc1 = save(nc1);
    struct savept p_sty = save(style_aco);
    struct tensor * nr0 = build_hifi_block(ctx, "nr0", nc0, style_aco);
    OOM_MARK("noise:hifi-nr0");
    struct savept p_nr0 = save(nr0);
    nc1       = restore(sa, &p_nc1);
    style_aco = restore(sa, &p_sty);
    struct tensor * nr1 = build_hifi_block(ctx, "nr1", nc1, style_aco);
    OOM_MARK("noise:hifi-nr1");
    nr0 = restore(sa, &p_nr0);
    savept_free(&p_nc1);
    savept_free(&p_sty);
    savept_free(&p_nr0);
    struct noise_outs r = { nr0, nr1 };
    return r;
}

// ---------------------------------------------------------------------------
// Generator (HiFi-GAN-style upsamplers + ResBlocks + iSTFT head)
// ---------------------------------------------------------------------------

static struct tensor * build_generator(struct kittens_ctx * ctx,
                                   struct tensor * dec_out,
                                   struct tensor * nr0, struct tensor * nr1,
                                   struct tensor * style_aco) {
    struct tensor * u0_w = named_fmt(ctx, "gen.u0.weight");
    struct tensor * u0_b = named_fmt(ctx, "gen.u0.bias");
    struct tensor * u1_w = named_fmt(ctx, "gen.u1.weight");
    struct tensor * u1_b = named_fmt(ctx, "gen.u1.bias");
    struct tensor * cp_w = named_fmt(ctx, "gen.cp.weight");
    struct tensor * cp_b = named_fmt(ctx, "gen.cp.bias");
    struct tensor * sb_r = named_fmt(ctx, "stft_bwd.real");
    struct tensor * sb_i = named_fmt(ctx, "stft_bwd.imag");
    struct arena * sa = ctx->scratch_arena;
    // build_hifi_block resets the arena internally; every input the
    // caller still needs after a hifi call must be snapshotted to host
    // memory FIRST, then restored. Snapshot style_aco and nr1 once at
    // entry - both are referenced multiple times after the first reset.
    // nr0 is used before any reset and doesn't need saving.
    struct savept p_sty = save(style_aco);
    struct savept p_nr1 = save(nr1);
    struct tensor * x = tensor_leaky_relu(dec_out, 0.1f);
    x = conv_transpose_1d_nlc(x, u0_w, u0_b, 10, 5);
    x = tensor_add(x, nr0);
    struct savept px, pr;
    px = save(x);
    struct tensor * r0 = build_hifi_block(ctx, "gen.r0", x, style_aco);
    pr = save(r0);
    x         = restore(sa, &px);
    style_aco = restore(sa, &p_sty);
    struct tensor * r1 = build_hifi_block(ctx, "gen.r1", x, style_aco);
    r0 = restore(sa, &pr);
    x = tensor_scale(tensor_add(r0, r1), 0.5f);
    x = tensor_leaky_relu(x, 0.1f);
    savept_free(&px); savept_free(&pr);
    x = conv_transpose_1d_nlc(x, u1_w, u1_b, 6, 3);
    x = reflection_pad_left(x, 1);
    nr1 = restore(sa, &p_nr1);
    style_aco = restore(sa, &p_sty);
    x = tensor_add(x, nr1);
    px = save(x);
    struct tensor * r2 = build_hifi_block(ctx, "gen.r2", x, style_aco);
    pr = save(r2);
    x         = restore(sa, &px);
    style_aco = restore(sa, &p_sty);
    struct tensor * r3 = build_hifi_block(ctx, "gen.r3", x, style_aco);
    r2 = restore(sa, &pr);
    x = tensor_scale(tensor_add(r2, r3), 0.5f);
    x = tensor_leaky_relu(x, 0.1f);
    savept_free(&px); savept_free(&pr);
    savept_free(&p_nr1);
    savept_free(&p_sty);
    x = conv1d_nlc(x, cp_w, cp_b, 1, -1, 1);
    // iSTFT head: mag = exp(x[0:11, :]); inner = sin(x[11:22, :]);
    // real-and-imag conv-transposes; audio = audio_r - audio_i; trim.
    const int64_t L = x->ne[1];
    struct tensor * mag_log = tensor_view_2d(x, 11, L, (size_t)x->nb[1], 0);
    struct tensor * phase   = tensor_view_2d(x, 11, L,
                                             (size_t)x->nb[1],
                                             (size_t)11 * x->nb[0]);
    mag_log = tensor_cont(mag_log);
    phase   = tensor_cont(phase);
    struct tensor * mag = tensor_exp(mag_log);
    struct tensor * inner = tensor_sin(phase);
    struct tensor * re = tensor_mul(mag, tensor_cos(inner));
    struct tensor * im = tensor_mul(mag, tensor_sin(inner));
    struct tensor * audio_r = conv_transpose_1d_nlc(re, sb_r, NULL, 5, 0);
    struct tensor * audio_i = conv_transpose_1d_nlc(im, sb_i, NULL, 5, 0);
    struct tensor * audio = tensor_sub(audio_r, audio_i);    // (1, T_audio)
    const int trim = ctx->arch.istft_trim;
    const int64_t T = audio->ne[1];
    assert(T > 2 * trim);
    return tensor_cont(tensor_view_1d(audio, T - 2 * trim,
                                      (size_t)trim * sizeof(float)));
}

// ---------------------------------------------------------------------------
// Fade helpers (host-side)
// ---------------------------------------------------------------------------

static void fade_in(float * x, int n, int fade) {
    if (fade > 0 && fade <= n) {
        for (int i = 0; i < fade; i++) {
            const float t = (float)i / (float)(fade - 1 > 0 ? fade - 1 : 1);
            x[i] *= 0.5f - 0.5f * cosf((float)M_PI * t);
        }
    }
}

static void fade_out(float * x, int n, int fade) {
    if (fade > 0 && fade <= n) {
        const int start = n - fade;
        for (int i = 0; i < fade; i++) {
            const float t = (float)i / (float)(fade - 1 > 0 ? fade - 1 : 1);
            x[start + i] *= 0.5f + 0.5f * cosf((float)M_PI * t);
        }
    }
}

// ---------------------------------------------------------------------------
// Public kittens_synthesize: full pipeline with length regulation
// ---------------------------------------------------------------------------

struct kittens_audio kittens_synthesize(struct kittens_ctx * ctx,
                                        const int32_t * phonemes,
                                        int n_phonemes,
                                        const float * style256,
                                        float speed) {
    struct kittens_audio out = { NULL, 0 };
    if (ctx == NULL || phonemes == NULL || n_phonemes <= 0
        || style256 == NULL || speed <= 0.0f) {
        return out;
    }
    const int L = n_phonemes;
    int F = 0;
    const struct arch * A = &ctx->arch;
    struct arena * sa = ctx->scratch_arena;
    OOM_MARK("entry");
    // CRITICAL: redirect ALL op outputs into the scratch arena.
    // Without this, ops whose first input is a model weight (q_w,
    // ffn_w, embedding tables, ...) allocate their output in
    // weights_arena, which is never reset and grows by ~100-200 MB
    // per synthesize call until the process is OOM-killed. See
    // tensor.c::arena_set_active.
    arena_set_active(sa);
    // Host buffers we hold across arena resets.
    int   * durs = NULL;
    float * prosody_h = NULL, * text_h = NULL, * dur_h = NULL;
    float * prosody_lr_h = NULL, * text_lr_h = NULL;
    float * f0p_h = NULL, * np_h = NULL, * dec_h = NULL;
    float * audio_buf = NULL;
    // ---- Stage 1: TextStage ----
    arena_reset(sa);
    {
        int32_t * pos_ids  = (int32_t *)malloc(sizeof(int32_t) * L);
        int32_t * type_ids = (int32_t *)malloc(sizeof(int32_t) * L);
        // Position embeddings only have max_pos rows. Long paragraphs
        // (heavy with "…", "—", etc.) can phonemize past that ceiling;
        // clamp to max_pos-1 so the model degrades gracefully rather
        // than asserting in tensor_get_rows. Quality past max_pos drops
        // (every later token shares the last position embedding) but
        // playback continues. Swift-side chunking should prevent us
        // from getting here in the first place.
        const int max_p = A->max_pos > 0 ? A->max_pos - 1 : 0;
        for (int i = 0; i < L; i++) {
            pos_ids[i]  = i <= max_p ? i : max_p;
            type_ids[i] = 0;
        }
        struct tensor * style_pr = tensor_new_1d(sa, A->style_dim);
        memcpy(style_pr->data, style256 + A->style_dim,
               sizeof(float) * A->style_dim);
        struct tensor * h0 = tensor_new_1d(sa, A->lstm_hidden);
        struct tensor * c0 = tensor_new_1d(sa, A->lstm_hidden);
        memset(h0->data, 0, sizeof(float) * A->lstm_hidden);
        memset(c0->data, 0, sizeof(float) * A->lstm_hidden);
        struct textstage_outs ts = build_textstage(ctx, L,
                                                   phonemes,
                                                   pos_ids, type_ids,
                                                   style_pr, h0, c0);
        free(pos_ids); free(type_ids);
        prosody_h = (float *)malloc(sizeof(float) * 256 * L);
        text_h    = (float *)malloc(sizeof(float) * 128 * L);
        dur_h     = (float *)malloc(sizeof(float) *  50 * L);
        memcpy(prosody_h, ts.prosody256->data, sizeof(float) * 256 * L);
        memcpy(text_h,    ts.text->data,       sizeof(float) * 128 * L);
        memcpy(dur_h,     ts.dur_sig->data,    sizeof(float) *  50 * L);
    }
    OOM_MARK("after-stage1");
    // ---- Length regulation ----
    durs = (int *)malloc(sizeof(int) * L);
    F = 0;
    for (int i = 0; i < L; i++) {
        float sum = 0.0f;
        for (int j = 0; j < 50; j++) {
            sum += dur_h[i * 50 + j];
        }
        int d = (int)lrintf(sum / speed);
        if (d < 1) { d = 1; }
        durs[i] = d;
        F += d;
    }
    prosody_lr_h = (float *)calloc((size_t)256 * F, sizeof(float));
    text_lr_h    = (float *)calloc((size_t)128 * F, sizeof(float));
    {
        int t = 0;
        for (int l = 0; l < L; l++) {
            const int d = durs[l];
            for (int k = 0; k < d; k++, t++) {
                memcpy(prosody_lr_h + (size_t)t * 256,
                       prosody_h    + (size_t)l * 256,
                       sizeof(float) * 256);
                memcpy(text_lr_h    + (size_t)t * 128,
                       text_h       + (size_t)l * 128,
                       sizeof(float) * 128);
            }
        }
    }
    free(prosody_h); prosody_h = NULL;
    free(text_h);    text_h    = NULL;
    free(dur_h);     dur_h     = NULL;
    free(durs);      durs      = NULL;
    // ---- Stage 2: GenFront ----
    arena_reset(sa);
    {
        struct tensor * prosody_lr = tensor_new_2d(sa, 256, F);
        memcpy(prosody_lr->data, prosody_lr_h,
               sizeof(float) * 256 * F);
        struct tensor * style_pr = tensor_new_1d(sa, A->style_dim);
        memcpy(style_pr->data, style256 + A->style_dim,
               sizeof(float) * A->style_dim);
        struct tensor * h0 = tensor_new_1d(sa, A->lstm_hidden);
        struct tensor * c0 = tensor_new_1d(sa, A->lstm_hidden);
        memset(h0->data, 0, sizeof(float) * A->lstm_hidden);
        memset(c0->data, 0, sizeof(float) * A->lstm_hidden);
        struct genfront_outs g = build_genfront(ctx, prosody_lr,
                                                style_pr, h0, c0, F);
        f0p_h = (float *)malloc(sizeof(float) * 2 * F);
        np_h  = (float *)malloc(sizeof(float) * 2 * F);
        memcpy(f0p_h, g.f0_proj->data, sizeof(float) * 2 * F);
        memcpy(np_h,  g.n_proj->data,  sizeof(float) * 2 * F);
    }
    free(prosody_lr_h); prosody_lr_h = NULL;
    OOM_MARK("after-stage2");
    // ---- Stage 3: Decoder ----
    arena_reset(sa);
    {
        struct tensor * text_lr = tensor_new_2d(sa, 128, F);
        struct tensor * f0p_t   = tensor_new_2d(sa, 1,   2 * F);
        struct tensor * np_t    = tensor_new_2d(sa, 1,   2 * F);
        struct tensor * style_a = tensor_new_1d(sa, 128);
        memcpy(text_lr->data, text_lr_h, sizeof(float) * 128 * F);
        memcpy(f0p_t->data,   f0p_h,     sizeof(float) * 2 * F);
        memcpy(np_t->data,    np_h,      sizeof(float) * 2 * F);
        memcpy(style_a->data, style256,  sizeof(float) * 128);
        struct tensor * dec_out = build_decoder(ctx, text_lr, f0p_t,
                                                np_t, style_a);
        dec_h = (float *)malloc(sizeof(float) * 256 * 2 * F);
        memcpy(dec_h, dec_out->data, sizeof(float) * 256 * 2 * F);
    }
    free(text_lr_h); text_lr_h = NULL;
    free(np_h);      np_h      = NULL;
    OOM_MARK("after-stage3");
    // ---- Stage 4a: Noise contributions ----
    // Run noise_contribs to completion, copy nr0/nr1 to host buffers,
    // then reset the arena so all the noise intermediates (sine
    // generator, STFT, magnitude, phase, etc.) die before the
    // generator's HiFi blocks start. Cuts peak by ~150-200 MB at
    // long F.
    float * nr0_h = NULL, * nr1_h = NULL;
    int64_t nr0_ne[4] = {0,0,0,0}, nr1_ne[4] = {0,0,0,0};
    int     nr0_nd = 0, nr1_nd = 0;
    arena_reset(sa);
    {
        struct tensor * f0_t  = tensor_new_2d(sa, 1,   2 * F);
        struct tensor * sty_a = tensor_new_1d(sa, 128);
        struct tensor * harm  = tensor_new_1d(sa, 9);
        struct tensor * s_rng = tensor_new_1d(sa, 300);
        struct tensor * eps_t = tensor_new_1d(sa, 1);
        memcpy(f0_t->data,  f0p_h,     sizeof(float) * 2 * F);
        memcpy(sty_a->data, style256,  sizeof(float) * 128);
        for (int i = 0; i < 9;   i++) { harm->data[i]  = (float)(i + 1); }
        for (int i = 0; i < 300; i++) { s_rng->data[i] = (float)i; }
        eps_t->data[0] = 1e-9f;
        struct noise_outs nz = build_noise_contribs(ctx, f0_t, sty_a,
                                                    harm, s_rng,
                                                    eps_t, F);
        nr0_nd = nz.nr0->ndim;
        nr1_nd = nz.nr1->ndim;
        for (int i = 0; i < 4; i++) {
            nr0_ne[i] = nz.nr0->ne[i];
            nr1_ne[i] = nz.nr1->ne[i];
        }
        int64_t nr0_n = nr0_ne[0]*nr0_ne[1]*nr0_ne[2]*nr0_ne[3];
        int64_t nr1_n = nr1_ne[0]*nr1_ne[1]*nr1_ne[2]*nr1_ne[3];
        nr0_h = (float *)malloc((size_t)nr0_n * sizeof(float));
        nr1_h = (float *)malloc((size_t)nr1_n * sizeof(float));
        memcpy(nr0_h, nz.nr0->data, (size_t)nr0_n * sizeof(float));
        memcpy(nr1_h, nz.nr1->data, (size_t)nr1_n * sizeof(float));
    }
    OOM_MARK("after-stage4a");
    // ---- Stage 4b: Generator + iSTFT ----
    int n = 0;
    arena_reset(sa);
    {
        struct tensor * dec_t = tensor_new_2d(sa, 256, 2 * F);
        struct tensor * sty_a = tensor_new_1d(sa, 128);
        memcpy(dec_t->data, dec_h,     sizeof(float) * 256 * 2 * F);
        memcpy(sty_a->data, style256,  sizeof(float) * 128);
        // Recreate nr0 / nr1 as tensors in the (now-fresh) arena.
        struct tensor * nr0 = tensor_new_nd(sa, nr0_nd, nr0_ne);
        struct tensor * nr1 = tensor_new_nd(sa, nr1_nd, nr1_ne);
        size_t nr0_n = (size_t)(nr0_ne[0] * nr0_ne[1]
                              * nr0_ne[2] * nr0_ne[3]);
        size_t nr1_n = (size_t)(nr1_ne[0] * nr1_ne[1]
                              * nr1_ne[2] * nr1_ne[3]);
        memcpy(nr0->data, nr0_h, nr0_n * sizeof(float));
        memcpy(nr1->data, nr1_h, nr1_n * sizeof(float));
        struct tensor * audio_t = build_generator(ctx, dec_t, nr0, nr1,
                                                  sty_a);
        const int T_audio = (int)audio_t->ne[0];
        // Tail-drop 3 frames × 600 samples.
        n = T_audio;
        const int tail_drop = 3 * 600;
        if (n > tail_drop) { n -= tail_drop; }
        audio_buf = (float *)malloc(sizeof(float) * (size_t)n);
        memcpy(audio_buf, audio_t->data, sizeof(float) * (size_t)n);
    }
    OOM_MARK("after-stage4b");
    free(nr0_h); free(nr1_h);
    free(f0p_h); f0p_h = NULL;
    free(dec_h); dec_h = NULL;
    // Fade in 3 ms, fade out 40 ms.
    fade_in (audio_buf, n,  72);
    fade_out(audio_buf, n, 960);
    out.samples   = audio_buf;
    out.n_samples = (uint64_t)n;
    arena_set_active(NULL);
    // Release the per-call peak. Without this, the doubling slab chain
    // (1+2+4+...+512+1024 MB for a long-sentence stage-4 peak) stays
    // resident until the next synthesize call's stage-1 reset - which
    // the OS reports as ~2-3 GB of RSS between sentences.
    arena_reset(sa);
    return out;
}

// ---------------------------------------------------------------------------
// Smoke test (compile as standalone with -DKITTENS_TESTS). Folded in
// to match the single-file-library + bottom-of-file smoke convention.
// ---------------------------------------------------------------------------

#ifdef KITTENS_TESTS

int main(int argc, char ** argv) {
    const char * path = argc > 1
        ? argv[1]
        : "app/Resources/nano/kitten_full.gguf";
    struct kittens_ctx * ctx = kittens_create(path);
    int rc = 1;
    if (ctx == NULL) {
        fprintf(stderr, "kittens_create failed: %s\n",
                kittens_last_error(NULL));
    } else {
        const int32_t ids[] = { 0, 10, 0 };
        float style[256];
        for (int i = 0; i < 256; i++) { style[i] = 0.0f; }
        struct kittens_audio a = kittens_synthesize(ctx, ids, 3, style, 1.0f);
        if (a.samples == NULL) {
            fprintf(stderr, "kittens_synthesize returned NULL: %s\n",
                    kittens_last_error(ctx));
        } else {
            printf("kittens link smoke: ctx OK, audio %llu samples\n",
                   (unsigned long long)a.n_samples);
            rc = 0;
            kittens_audio_free(a);
        }
        kittens_destroy(ctx);
    }
    return rc;
}

#endif /* KITTENS_TESTS */

#endif /* KITTENS_C */
