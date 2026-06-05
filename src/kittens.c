#ifndef KITTENS_C
#define KITTENS_C

#include "tts/kittens.h"
#include "tensor.c"
#include "gguf_reader.c"
#include "trace/trace.h"

#include <assert.h>
#include <mach/mach.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double rss_mb(void) {
    struct task_vm_info info; mach_msg_type_number_t n = TASK_VM_INFO_COUNT;
    return task_info(mach_task_self(), TASK_VM_INFO,
                     (task_info_t)&info, &n) == KERN_SUCCESS
        ? (double)info.phys_footprint / (1024.0 * 1024.0) : 0.0;
}

static bool is_trace_rss_enabled(void) {
    static int enabled = -1;
    if (enabled == -1) {
        const char * v = getenv("TTS_TRACE_RSS");
        enabled = (v != NULL && v[0] != 0 && v[0] != '0') ? 1 : 0;
    }
    return enabled;
}

static int g_carry_fp16 = 0;

#define trace_rss(tag) do {                                          \
    if (is_trace_rss_enabled()) {                                    \
        trace(info, "[RSS %-24s] %.2f MB", tag, rss_mb());           \
    }                                                                \
} while(0)

struct arch {
    int vocab, max_pos, token_types;
    int embd_dim, hidden, n_layers, n_heads, head_dim, ffn_dim;
    float ln_eps;
    int bert_enc_dim, style_dim, lstm_hidden, dur_logits;
    int audio_per_frame, istft_hop, istft_trim;
};

struct weights {
    struct tensor * e_word, * e_pos, * e_type;
    struct tensor * e_ln_w, * e_ln_b;
    struct tensor * proj_w, * proj_b;
    struct tensor * q_w, * q_b, * k_w, * k_b, * v_w, * v_b, * o_w, * o_b;
    struct tensor * attn_ln_w, * attn_ln_b;
    struct tensor * ffn_w, * ffn_b, * ffn_out_w, * ffn_out_b;
    struct tensor * full_ln_w, * full_ln_b;
    struct tensor * bert_enc_w, * bert_enc_b;
    struct tensor * pt_l0_fW, * pt_l0_fR, * pt_l0_fb;
    struct tensor * pt_l0_bW, * pt_l0_bR, * pt_l0_bb;
    struct tensor * pt_fc1_w, * pt_fc1_b;
    struct tensor * pt_l2_fW, * pt_l2_fR, * pt_l2_fb;
    struct tensor * pt_l2_bW, * pt_l2_bR, * pt_l2_bb;
    struct tensor * pt_fc3_w, * pt_fc3_b;
    struct tensor * dur_l_fW, * dur_l_fR, * dur_l_fb;
    struct tensor * dur_l_bW, * dur_l_bR, * dur_l_bb;
    struct tensor * dur_w, * dur_b;
    struct tensor * ac_embd;
    struct tensor * ac_c0_w, * ac_c0_b, * ac_ln0_g, * ac_ln0_b;
    struct tensor * ac_c1_w, * ac_c1_b, * ac_ln1_g, * ac_ln1_b;
    struct tensor * ac_l_fW, * ac_l_fR, * ac_l_fb;
    struct tensor * ac_l_bW, * ac_l_bR, * ac_l_bb;
    struct tensor * sh_fW, * sh_fR, * sh_fb;
    struct tensor * sh_bW, * sh_bR, * sh_bb;
};

struct named_entry {
    char        name[64];
    struct tensor * t;
};

struct kittens_ctx {
    struct gguf           gguf;
    struct arena *        weights_arena;
    struct arena *        scratch_arena;
    struct arch           arch;
    struct weights        W;
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
    return ctx != NULL ? ctx->err : "";
}

static int kt_u32(const struct gguf * g, const char * key, uint32_t * out) {
    const struct gguf_kv * kv = gguf_find_kv(g, key);
    int found = (kv != NULL && kv->v.type == GGUF_VT_U32);
    if (found) { *out = gguf_kv_u32(g, key, 0); }
    return found;
}

static int kt_f32(const struct gguf * g, const char * key, float * out) {
    const struct gguf_kv * kv = gguf_find_kv(g, key);
    int found = (kv != NULL && kv->v.type == GGUF_VT_F32);
    if (found) { *out = gguf_kv_f32(g, key, 0.0f); }
    return found;
}

struct kt_build_ctx {
    struct arena *  arena;
    const char *    name;
    struct tensor * out;
};

static void kt_build_tensor(void * vctx, const struct gguf_tensor * t,
                            const void * data, size_t nbytes) {
    (void)nbytes;
    struct kt_build_ctx * c = (struct kt_build_ctx *)vctx;
    int64_t ne[4] = { 1, 1, 1, 1 };
    for (uint32_t d = 0; d < t->n_dims; d++) { ne[d] = (int64_t)t->shape[d]; }
    struct tensor * out = NULL;
    if (t->type == GGUF_TT_F32) {
        out = tensor_wrap_nd(c->arena, (int)t->n_dims, (float *)data, ne);
    } else if (t->type == GGUF_TT_F16) {
        out = tensor_new_nd(c->arena, (int)t->n_dims, ne);
        int64_t total = tensor_nelements(out);
        const uint16_t * src16 = (const uint16_t *)data;
        float * dst = out->data;
        for (int64_t i = 0; i < total; i++) {
            _Float16 h;
            memcpy(&h, &src16[i], 2);
            dst[i] = (float)h;
        }
    } else {
        assert(0 && "gguf: unsupported tensor dtype");
    }
    if (out != NULL) { tensor_set_name(out, c->name); }
    c->out = out;
}

static struct tensor * kt_load_tensor(const struct gguf * g,
                                      struct arena * arena,
                                      const char * name) {
    const struct gguf_tensor * t = gguf_find_tensor(g, name);
    struct tensor * out = NULL;
    if (t != NULL) {
        uint64_t nel = 1;
        for (uint32_t d = 0; d < t->n_dims; d++) { nel *= t->shape[d]; }
        size_t per = (t->type == GGUF_TT_F32) ? 4u
                   : (t->type == GGUF_TT_F16) ? 2u : 0u;
        struct kt_build_ctx c = { arena, name, NULL };
        gguf_load_tensor(g, t, (size_t)nel * per, GGUF_SRC_SHARED,
                         kt_build_tensor, &c);
        out = c.out;
    }
    return out;
}

static struct tensor * bind(struct kittens_ctx * ctx, const char * name) {
    struct tensor * t = kt_load_tensor(&ctx->gguf, ctx->weights_arena, name);
    if (t == NULL) {
        set_ctx_err(ctx, "missing GGUF tensor: %s", name);
    }
    return t;
}

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
    int loaded = kt_u32(&ctx->gguf,
                                 "kittens-tts.vocab_size", &v);
    if (!loaded) {
        set_ctx_err(ctx, "missing arch KV: kittens-tts.vocab_size");
    } else {
        ctx->arch.vocab = (int)v;
        kt_u32(&ctx->gguf, "kittens-tts.max_position", &v);
        ctx->arch.max_pos = (int)v;
        kt_u32(&ctx->gguf, "kittens-tts.token_types", &v);
        ctx->arch.token_types = (int)v;
        kt_u32(&ctx->gguf, "kittens-tts.embedding_dim", &v);
        ctx->arch.embd_dim = (int)v;
        kt_u32(&ctx->gguf, "kittens-tts.hidden_size", &v);
        ctx->arch.hidden = (int)v;
        kt_u32(&ctx->gguf, "kittens-tts.num_layers", &v);
        ctx->arch.n_layers = (int)v;
        kt_u32(&ctx->gguf, "kittens-tts.num_heads", &v);
        ctx->arch.n_heads = (int)v;
        kt_u32(&ctx->gguf, "kittens-tts.head_dim", &v);
        ctx->arch.head_dim = (int)v;
        kt_u32(&ctx->gguf, "kittens-tts.ffn_dim", &v);
        ctx->arch.ffn_dim = (int)v;
        kt_f32(&ctx->gguf, "kittens-tts.layer_norm_eps",
                        &ctx->arch.ln_eps);
        kt_u32(&ctx->gguf, "kittens-tts.bert_enc_dim", &v);
        ctx->arch.bert_enc_dim = (int)v;
        kt_u32(&ctx->gguf, "kittens-tts.style_dim", &v);
        ctx->arch.style_dim = (int)v;
        kt_u32(&ctx->gguf, "kittens-tts.lstm_hidden", &v);
        ctx->arch.lstm_hidden = (int)v;
        kt_u32(&ctx->gguf, "kittens-tts.dur_logits", &v);
        ctx->arch.dur_logits = (int)v;
        kt_u32(&ctx->gguf, "kittens-tts.audio_per_frame", &v);
        ctx->arch.audio_per_frame = (int)v;
        kt_u32(&ctx->gguf, "kittens-tts.istft_hop", &v);
        ctx->arch.istft_hop = (int)v;
        kt_u32(&ctx->gguf, "kittens-tts.istft_trim", &v);
        ctx->arch.istft_trim = (int)v;
    }
    return loaded;
}

static void bind_weights(struct kittens_ctx * ctx) {
    struct weights * W = &ctx->W;
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
    W->bert_enc_w = bind_req(ctx, "bert_enc.weight");
    W->bert_enc_b = bind_req(ctx, "bert_enc.bias");
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
    W->dur_l_fW = bind_req(ctx, "dur.lstm.fwd.W");
    W->dur_l_fR = bind_req(ctx, "dur.lstm.fwd.R");
    W->dur_l_fb = bind_req(ctx, "dur.lstm.fwd.b");
    W->dur_l_bW = bind_req(ctx, "dur.lstm.bwd.W");
    W->dur_l_bR = bind_req(ctx, "dur.lstm.bwd.R");
    W->dur_l_bb = bind_req(ctx, "dur.lstm.bwd.b");
    W->dur_w    = bind_req(ctx, "dur_proj.weight");
    W->dur_b    = bind_req(ctx, "dur_proj.bias");
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
    W->sh_fW    = bind_req(ctx, "shared.lstm.fwd.W");
    W->sh_fR    = bind_req(ctx, "shared.lstm.fwd.R");
    W->sh_fb    = bind_req(ctx, "shared.lstm.fwd.b");
    W->sh_bW    = bind_req(ctx, "shared.lstm.bwd.W");
    W->sh_bR    = bind_req(ctx, "shared.lstm.bwd.R");
    W->sh_bb    = bind_req(ctx, "shared.lstm.bwd.b");
}

struct kittens_ctx * kittens_create(const char * gguf_path) {
    trace_set_min_level(trace_level_info);
    struct kittens_ctx * ctx = (struct kittens_ctx *)calloc(1, sizeof(*ctx));
    int success = 0;
    if (ctx != NULL) {
        if (gguf_open(&ctx->gguf, gguf_path) != 0) {
            snprintf(ctx->err, sizeof(ctx->err),
                     "gguf_open(%s) failed", gguf_path);
        } else {
            ctx->weights_arena = arena_new(64 * 1024 * 1024);
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
        gguf_close(&ctx->gguf);
        free(ctx->cache);
        free(ctx);
    }
}

static struct tensor * named(struct kittens_ctx * ctx, const char * name) {
    int i = 0;
    while (i < ctx->cache_count
           && strcmp(ctx->cache[i].name, name) != 0) {
        i++;
    }
    struct tensor * result;
    if (i < ctx->cache_count) {
        result = ctx->cache[i].t;
    } else {
        struct tensor * t = kt_load_tensor(&ctx->gguf,
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

void kittens_set_carry_fp16(int mode) { g_carry_fp16 = mode; }
int  kittens_get_carry_fp16(void)     { return g_carry_fp16; }

uint64_t kittens_current_phys_footprint(void) {
    return tensor_current_phys_footprint();
}
uint64_t kittens_peak_phys_footprint(void) {
    return tensor_peak_phys_footprint();
}
void kittens_reset_peak_phys_footprint(void) {
    tensor_reset_peak_phys_footprint();
}

static struct tensor * layer_norm(struct tensor * x, struct tensor * w,
                                 struct tensor * b, float eps) {
    struct tensor * h = tensor_norm(x, 0, eps);
    if (w != NULL) { h = tensor_mul(h, w); }
    if (b != NULL) { h = tensor_add(h, b); }
    return h;
}

static struct tensor * ada_layer_norm(struct tensor * x, struct tensor * style,
                                     struct tensor * fcW, struct tensor * fcB,
                                     int C) {
    struct tensor * h = tensor_mul_mat(fcW, style);
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
    struct tensor * x_t = tensor_cont(tensor_transpose(x));
    struct tensor * n_t = tensor_norm(x_t, 0, 1e-5f);
    struct tensor * n   = tensor_cont(tensor_transpose(n_t));
    if (nW != NULL) { n = tensor_mul(n, nW); }
    if (nB != NULL) { n = tensor_add(n, nB); }
    struct tensor * n_g = tensor_mul(n, gamma_c);
    struct tensor * out = tensor_add(n_g, n);
    out = tensor_add(out, beta_c);
    return out;
}

static struct tensor * snake_1d(struct tensor * x, struct tensor * alpha) {
    struct tensor * ax = tensor_mul(x, alpha);
    struct tensor * s  = tensor_sin(ax);
    struct tensor * s2 = tensor_mul(s, s);
    struct tensor * s2_over_a = tensor_div(s2, alpha);
    return tensor_add(x, s2_over_a);
}

static struct tensor * conv1d_nlc(struct tensor * x, struct tensor * kw,
                                 struct tensor * kb,
                                 int stride, int pad, int dilation) {
    const int K = (int)kw->ne[0];
    if (pad < 0) { pad = (K - 1) / 2; }
    struct tensor * x_ncl = tensor_cont(tensor_transpose(x));
    struct tensor * x_3d = tensor_reshape_3d(x_ncl,
                                     x_ncl->ne[0], x_ncl->ne[1], 1);
    struct tensor * y_3d = tensor_conv_1d(kw, x_3d, stride, pad, dilation);
    if (kb != NULL) {
        struct tensor * b3 = tensor_reshape_3d(kb, 1, kb->ne[0], 1);
        y_3d = tensor_add(y_3d, b3);
    }
    struct tensor * y_2d = tensor_reshape_2d(y_3d, y_3d->ne[0], y_3d->ne[1]);
    return tensor_cont(tensor_transpose(y_2d));
}

static struct tensor * repeat_interleave_2x_nlc(struct tensor * x) {
    const int64_t C = x->ne[0];
    const int64_t L = x->ne[1];
    struct tensor * x4 = tensor_reshape_4d(x, C, L, 1, 1);
    struct tensor * stacked = tensor_concat(x4, x4, 2);
    struct tensor * p = tensor_permute(stacked, 0, 2, 1, 3);
    struct tensor * pc = tensor_cont(p);
    return tensor_reshape_2d(pc, C, 2 * L);
}

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

static struct tensor * conv_transpose_1d_nlc(struct tensor * x,
                                            struct tensor * kw,
                                            struct tensor * kb,
                                            int stride, int pad) {
    struct tensor * b = tensor_cont(tensor_transpose(x));
    struct tensor * b_3d = tensor_reshape_3d(b, b->ne[0], b->ne[1], 1);
    struct tensor * y_3d = tensor_conv_transpose_1d(kw, b_3d, stride, pad);
    if (kb != NULL) {
        struct tensor * bb = tensor_reshape_3d(kb, 1, kb->ne[0], 1);
        y_3d = tensor_add(y_3d, bb);
    }
    struct tensor * y_2d = tensor_reshape_2d(y_3d, y_3d->ne[0], y_3d->ne[1]);
    return tensor_cont(tensor_transpose(y_2d));
}

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

static struct tensor * style_bcast_CxL(struct tensor * style, int C, int L) {
    struct tensor * s2 = tensor_reshape_2d(style, C, 1);
    return tensor_repeat_to(s2, 2, C, L, 1, 1);
}

static struct tensor * lstm_dir(struct arena * a,
                                struct tensor * x, struct tensor * W,
                                struct tensor * R, struct tensor * b,
                                struct tensor * h0, struct tensor * c0,
                                int H, int T, int reverse) {
    struct tensor * Wx_full = tensor_mul_mat(W, x);
    Wx_full = tensor_add(Wx_full, b);
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
        memcpy((char *)out->data + (size_t)t * H * fsz,
               h_t->data, (size_t)H * fsz);
        h_prev = h_t;
        c_prev = c_t;
    }
    return out;
}

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

struct savept {
    int      ndim;
    int      fp16;
    int64_t  ne[4];
    float *  data;
};

static void f32_to_f16(_Float16 * dst, const float * src, size_t n) {
    for (size_t i = 0; i < n; i++) { dst[i] = (_Float16)src[i]; }
}

static void f16_to_f32(float * dst, const _Float16 * src, size_t n) {
    for (size_t i = 0; i < n; i++) { dst[i] = (float)src[i]; }
}

static void carry_store(void * host, int fp16, size_t off,
                        const float * src, size_t n) {
    if (fp16) { f32_to_f16((_Float16 *)host + off, src, n); }
    else      { memcpy((float *)host + off, src, n * sizeof(float)); }
}

static void carry_load(float * dst, const void * host, int fp16,
                       size_t off, size_t n) {
    if (fp16) { f16_to_f32(dst, (const _Float16 *)host + off, n); }
    else      { memcpy(dst, (const float *)host + off, n * sizeof(float)); }
}

static float * host_alloc(size_t bytes) {
    size_t mapped = 0;
    float * p = (float *)arena_pages_alloc(bytes, &mapped);
    assert(p != NULL);
    return p;
}

static void host_free(float * p, size_t bytes) {
    if (p != NULL) {
        const size_t pg = arena_page_size();
        arena_pages_free(p, (bytes + pg - 1) & ~(pg - 1));
    }
}

static void host_retire_prefix(float * base, size_t * retired,
                               size_t upto) {
    const size_t pg = arena_page_size();
    const size_t new_end = upto & ~(pg - 1);
    if (new_end > *retired) {
        munmap((char *)base + *retired, new_end - *retired);
        *retired = new_end;
    }
}

static void host_free_rest(float * base, size_t retired, size_t bytes) {
    if (base != NULL) {
        const size_t pg = arena_page_size();
        const size_t total = (bytes + pg - 1) & ~(pg - 1);
        if (total > retired) {
            munmap((char *)base + retired, total - retired);
        }
    }
}

static struct savept save(const struct tensor * t) {
    struct savept p;
    p.ndim = t->ndim;
    p.fp16 = 0;
    for (int i = 0; i < 4; i++) { p.ne[i] = t->ne[i]; }
    int64_t n = p.ne[0] * p.ne[1] * p.ne[2] * p.ne[3];
    p.data = host_alloc((size_t)n * sizeof(float));
    memcpy(p.data, t->data, (size_t)n * sizeof(float));
    return p;
}

static struct savept save16(const struct tensor * t) {
    struct savept p;
    p.ndim = t->ndim;
    p.fp16 = g_carry_fp16 >= 1;
    for (int i = 0; i < 4; i++) { p.ne[i] = t->ne[i]; }
    int64_t n = p.ne[0] * p.ne[1] * p.ne[2] * p.ne[3];
    if (p.fp16) {
        p.data = host_alloc((size_t)n * 2);
        f32_to_f16((_Float16 *)p.data, t->data, (size_t)n);
    } else {
        p.data = host_alloc((size_t)n * sizeof(float));
        memcpy(p.data, t->data, (size_t)n * sizeof(float));
    }
    return p;
}

static struct savept savept_wrap(float * data, int64_t n0, int64_t n1,
                                 int fp16) {
    struct savept p;
    p.ndim = 2;
    p.fp16 = fp16;
    p.ne[0] = n0; p.ne[1] = n1; p.ne[2] = 1; p.ne[3] = 1;
    p.data = data;
    return p;
}

static struct tensor * restore(struct arena * a, const struct savept * p) {
    struct tensor * t = tensor_new_nd(a, p->ndim, p->ne);
    int64_t n = p->ne[0] * p->ne[1] * p->ne[2] * p->ne[3];
    carry_load(t->data, p->data, p->fp16, 0, (size_t)n);
    return t;
}

struct hifi_out {
    float * data;
    int     C;
    int64_t L;
    int     fp16;
};

static size_t hifi_out_bytes(const struct hifi_out * h) {
    return (size_t)h->C * (size_t)h->L * (h->fp16 ? 2u : 4u);
}

static void hifi_out_compress(struct hifi_out * h) {
    if (g_carry_fp16 >= 1 && !h->fp16 && h->data != NULL) {
        const size_t n = (size_t)h->C * (size_t)h->L;
        _Float16 * dst = (_Float16 *)h->data;
        for (size_t i = 0; i < n; i++) { dst[i] = (_Float16)h->data[i]; }
        const size_t pg = arena_page_size();
        const size_t total = (n * sizeof(float) + pg - 1) & ~(pg - 1);
        const size_t keep  = (n * 2 + pg - 1) & ~(pg - 1);
        if (total > keep) { munmap((char *)h->data + keep, total - keep); }
        h->fp16 = 1;
    }
}

static void savept_free(struct savept * p) {
    if (p && p->data) {
        int64_t n = p->ne[0] * p->ne[1] * p->ne[2] * p->ne[3];
        host_free(p->data, (size_t)n * (p->fp16 ? 2u : 4u));
        p->data = NULL;
    }
}

#ifndef HIFI_INTRA_K_ONLY

#define HIFI_SEG_L         1024
#define HIFI_SEG_OVERLAP    128

static void hifi_compute_stats_host(const void * host, int fp16,
                                    int C, int64_t L,
                                    float * mean, float * istd) {
    float * tmp = (float *)malloc((size_t)L * sizeof(float));
    assert(tmp != NULL);
    const int64_t Cs = (int64_t)C;
    const float * h32 = (const float *)host;
    const _Float16 * h16 = (const _Float16 *)host;
    int c = 0;
    while (c < C) {
        int64_t t = 0;
        if (fp16) {
            while (t < L) { tmp[t] = (float)h16[t * Cs + c]; t++; }
        } else {
            while (t < L) { tmp[t] = h32[t * Cs + c]; t++; }
        }
        float m, msq;
        vDSP_meanv (tmp, 1, &m,   (vDSP_Length)L);
        vDSP_measqv(tmp, 1, &msq, (vDSP_Length)L);
        const float var = msq - m * m;
        mean[c] = m;
        istd[c] = 1.0f / sqrtf(var + 1e-5f);
        c++;
    }
    free(tmp);
}

static struct tensor * ada_in_1d_with_stats(struct tensor * x,
                                            struct tensor * style,
                                            struct tensor * fcW,
                                            struct tensor * fcB,
                                            struct tensor * nW,
                                            struct tensor * nB,
                                            int C,
                                            const float * mean,
                                            const float * istd) {
    struct arena * sa = arena_get_active();
    struct tensor * h = tensor_mul_mat(fcW, style);
    h = tensor_add(h, fcB);
    const size_t fsz = sizeof(float);
    struct tensor * gamma   = tensor_view_1d(h, C, 0);
    struct tensor * beta    = tensor_view_1d(h, C, (size_t)C * fsz);
    struct tensor * gamma_c = tensor_cont(gamma);
    struct tensor * beta_c  = tensor_cont(beta);
    struct tensor * neg_mean = tensor_new_1d(sa, C);
    struct tensor * istd_t   = tensor_new_1d(sa, C);
    int c = 0;
    while (c < C) {
        ((float *)neg_mean->data)[c] = -mean[c];
        ((float *)istd_t->data)[c]   =  istd[c];
        c++;
    }
    struct tensor * n = tensor_add(x, neg_mean);
    n = tensor_mul(n, istd_t);
    if (nW != NULL) { n = tensor_mul(n, nW); }
    if (nB != NULL) { n = tensor_add(n, nB); }
    struct tensor * n_g = tensor_mul(n, gamma_c);
    struct tensor * out = tensor_add(n_g, n);
    out = tensor_add(out, beta_c);
    return out;
}

static struct hifi_out build_hifi_block(struct kittens_ctx * ctx,
                                        const char * prefix,
                                        struct savept * px,
                                        struct tensor * style,
                                        int consume) {
    struct arena * sa = ctx->scratch_arena;
    const int     C = (int)px->ne[0];
    const int64_t L = px->ne[1];
    static const int dilations[3] = { 1, 3, 5 };
    arena_set_active(sa);
    struct savept ps = save(style);
    arena_reset(sa);
    trace(info, "[RSS hifi2-entry %-9s L=%-6lld phys=%7.1f MB]",
          prefix, (long long)L, rss_mb());
    struct hifi_out result = { NULL, C, L, 0 };
    if (L <= (int64_t)HIFI_SEG_L + 2 * HIFI_SEG_OVERLAP) {
        struct tensor * out = restore(sa, px);
        if (consume) { savept_free(px); }
        style = restore(sa, &ps);
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
        }
        const size_t total_floats = (size_t)C * (size_t)L;
        result.fp16 = g_carry_fp16 >= 1;
        if (result.fp16) {
            result.data = host_alloc(total_floats * 2);
            f32_to_f16((_Float16 *)result.data, out->data, total_floats);
        } else {
            result.data = host_alloc(total_floats * sizeof(float));
            memcpy(result.data, out->data, total_floats * sizeof(float));
        }
    } else {
        const size_t total_floats = (size_t)C * (size_t)L;
        const int    t2   = g_carry_fp16 >= 2;
        const size_t elem = t2 ? 2u : 4u;
        float * out_host = NULL;
        if (consume && px->fp16 == t2) {
            out_host = px->data;
            px->data = NULL;
        } else {
            out_host = host_alloc(total_floats * elem);
            if (t2) {
                if (px->fp16) {
                    memcpy(out_host, px->data, total_floats * 2);
                } else {
                    f32_to_f16((_Float16 *)out_host, px->data,
                               total_floats);
                }
            } else {
                carry_load(out_host, px->data, px->fp16, 0, total_floats);
            }
            if (consume) { savept_free(px); }
        }
        float * h3_host = (float *)malloc(total_floats * elem);
        assert(h3_host != NULL);
        float *  mean_in   = (float *)malloc((size_t)C * sizeof(float));
        float *  istd_in   = (float *)malloc((size_t)C * sizeof(float));
        float *  mean_mid  = (float *)malloc((size_t)C * sizeof(float));
        float *  istd_mid  = (float *)malloc((size_t)C * sizeof(float));
        assert(mean_in && istd_in && mean_mid && istd_mid);
        for (int k = 0; k < 3; k++) {
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
            hifi_compute_stats_host(out_host, t2, C, L,
                                    mean_in, istd_in);
            int64_t a = 0;
            while (a < L) {
                int64_t b  = a + (int64_t)HIFI_SEG_L;
                if (b > L) { b = L; }
                int64_t lo = a - (int64_t)HIFI_SEG_OVERLAP;
                if (lo < 0) { lo = 0; }
                int64_t hi = b + (int64_t)HIFI_SEG_OVERLAP;
                if (hi > L) { hi = L; }
                const int64_t seg_W  = hi - lo;
                const int     cm_off = (int)(a - lo);
                const int     cm_n   = (int)(b - a);
                arena_reset(sa);
                struct tensor * seg_in = tensor_new_2d(sa, C, seg_W);
                carry_load(seg_in->data, out_host, t2,
                           (size_t)lo * (size_t)C,
                           (size_t)seg_W * (size_t)C);
                struct tensor * sty = restore(sa, &ps);
                struct tensor * h = ada_in_1d_with_stats(
                    seg_in, sty, a1_fcW, a1_fcB, a1_nW, a1_nB, C,
                    mean_in, istd_in);
                h = snake_1d(h, al1);
                h = conv1d_nlc(h, c1_w, c1_b, 1, d * (K1 - 1) / 2, d);
                carry_store(h3_host, t2, (size_t)a * (size_t)C,
                            (const float *)h->data
                                + (size_t)cm_off * (size_t)C,
                            (size_t)cm_n * (size_t)C);
                a = b;
            }
            hifi_compute_stats_host(h3_host, t2, C, L,
                                    mean_mid, istd_mid);
            a = 0;
            while (a < L) {
                int64_t b  = a + (int64_t)HIFI_SEG_L;
                if (b > L) { b = L; }
                int64_t lo = a - (int64_t)HIFI_SEG_OVERLAP;
                if (lo < 0) { lo = 0; }
                int64_t hi = b + (int64_t)HIFI_SEG_OVERLAP;
                if (hi > L) { hi = L; }
                const int64_t seg_W  = hi - lo;
                const int     cm_off = (int)(a - lo);
                const int     cm_n   = (int)(b - a);
                arena_reset(sa);
                struct tensor * seg_h3 = tensor_new_2d(sa, C, seg_W);
                carry_load(seg_h3->data, h3_host, t2,
                           (size_t)lo * (size_t)C,
                           (size_t)seg_W * (size_t)C);
                struct tensor * sty = restore(sa, &ps);
                struct tensor * h = ada_in_1d_with_stats(
                    seg_h3, sty, a2_fcW, a2_fcB, a2_nW, a2_nB, C,
                    mean_mid, istd_mid);
                h = snake_1d(h, al2);
                h = conv1d_nlc(h, c2_w, c2_b, 1, (K2 - 1) / 2, 1);
                const float * src = (const float *)h->data
                                  + (size_t)cm_off * (size_t)C;
                const int total_n = cm_n * C;
                if (t2) {
                    _Float16 * dst = (_Float16 *)out_host
                                   + (size_t)a * (size_t)C;
                    int i = 0;
                    while (i < total_n) {
                        dst[i] = (_Float16)((float)dst[i] + src[i]);
                        i++;
                    }
                } else {
                    float * dst = out_host + (size_t)a * (size_t)C;
                    int i = 0;
                    while (i < total_n) { dst[i] += src[i]; i++; }
                }
                a = b;
            }
            trace(info, "[RSS hifi2-k%d-end %-9s     phys=%7.1f MB]",
                  k, prefix, rss_mb());
        }
        arena_reset(sa);
        result.data = out_host;
        result.fp16 = t2;
        hifi_out_compress(&result);
        free(h3_host);
        free(mean_in);   free(istd_in);
        free(mean_mid);  free(istd_mid);
    }
    savept_free(&ps);
    return result;
}

#else

static void hifi_reset_after_op(struct arena * sa,
                                struct tensor ** out_p,
                                struct tensor ** h_p,
                                const struct savept * ps,
                                struct tensor ** style_p) {
    struct savept p_out = save(*out_p);
    struct savept p_h   = save(*h_p);
    arena_reset(sa);
    *out_p   = restore(sa, &p_out);  savept_free(&p_out);
    *h_p     = restore(sa, &p_h);    savept_free(&p_h);
    *style_p = restore(sa, ps);
}

static struct hifi_out build_hifi_block(struct kittens_ctx * ctx, const char * prefix,
                                    struct savept * px, struct tensor * style,
                                    int consume) {
    struct arena * sa = ctx->scratch_arena;
    const int     C = (int)px->ne[0];
    const int64_t L = px->ne[1];
    static const int dilations[3] = { 1, 3, 5 };
    struct savept ps = save(style);
    arena_reset(sa);
    struct tensor * out = restore(sa, px);
    if (consume) { savept_free(px); }
    style = restore(sa, &ps);
    trace(info, "[RSS hifi-entry  %-9s     phys=%7.1f MB]",
          prefix, rss_mb());
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
        hifi_reset_after_op(sa, &out, &h, &ps, &style);
        h = snake_1d(h, al1);
        h = conv1d_nlc(h, c1_w, c1_b, 1, d * (K1 - 1) / 2, d);
        hifi_reset_after_op(sa, &out, &h, &ps, &style);
        h = ada_in_1d(h, style, a2_fcW, a2_fcB, a2_nW, a2_nB, C);
        hifi_reset_after_op(sa, &out, &h, &ps, &style);
        h = snake_1d(h, al2);
        h = conv1d_nlc(h, c2_w, c2_b, 1, (K2 - 1) / 2, 1);
        out = tensor_add(out, h);
        trace(info, "[RSS hifi-k%d-end %-9s     phys=%7.1f MB]",
              k, prefix, rss_mb());
    }
    savept_free(&ps);
    const size_t total_floats = (size_t)C * (size_t)L;
    struct hifi_out result = { NULL, C, L, 0 };
    result.data = host_alloc(total_floats * sizeof(float));
    memcpy(result.data, out->data, total_floats * sizeof(float));
    hifi_out_compress(&result);
    return result;
}

#endif

static struct tensor * build_albert(struct kittens_ctx * ctx, int L,
                                const int32_t * ids,
                                const int32_t * pos,
                                const int32_t * type) {
    const struct arch * a = &ctx->arch;
    const struct weights * W = &ctx->W;
    struct arena * sa = ctx->scratch_arena;
    struct tensor * h = tensor_get_rows(W->e_word, ids,  L);
    struct tensor * p = tensor_get_rows(W->e_pos,  pos,  L);
    struct tensor * t = tensor_get_rows(W->e_type, type, L);
    h = tensor_add(h, p);
    h = tensor_add(h, t);
    h = layer_norm(h, W->e_ln_w, W->e_ln_b, a->ln_eps);
    h = tensor_mul_mat(W->proj_w, h);
    h = tensor_add(h, W->proj_b);
    assert(tensor_is_packed(h));
    {
        struct savept p_h = save(h);
        arena_reset(sa);
        h = restore(sa, &p_h);
        savept_free(&p_h);
    }
    const float kq_scale = 1.0f / sqrtf((float)a->head_dim);
    for (int il = 0; il < a->n_layers; il++) {
        if (il > 0) {
            assert(tensor_is_packed(h));
            struct savept p_h = save(h);
            arena_reset(sa);
            h = restore(sa, &p_h);
            savept_free(&p_h);
        }
        struct tensor * residual = h;
        struct tensor * q = tensor_add(tensor_mul_mat(W->q_w, h), W->q_b);
        struct tensor * k = tensor_add(tensor_mul_mat(W->k_w, h), W->k_b);
        struct tensor * v = tensor_add(tensor_mul_mat(W->v_w, h), W->v_b);
        q = tensor_reshape_4d(q, a->head_dim, a->n_heads, L, 1);
        k = tensor_reshape_4d(k, a->head_dim, a->n_heads, L, 1);
        v = tensor_reshape_4d(v, a->head_dim, a->n_heads, L, 1);
        q = tensor_permute(q, 0, 2, 1, 3);
        k = tensor_permute(k, 0, 2, 1, 3);
        v = tensor_permute(v, 0, 2, 1, 3);
        v = tensor_cont(tensor_transpose(v));
        struct tensor * kq = tensor_mul_mat(k, q);
        kq = tensor_softmax(kq, 0, kq_scale);
        struct tensor * kqv = tensor_mul_mat(v, kq);
        kqv = tensor_permute(kqv, 0, 2, 1, 3);
        kqv = tensor_cont_2d(kqv, a->hidden, L);
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
    struct tensor * x = tensor_concat(bert_out, s_bcast, 0);
    struct tensor * y = bidir_lstm(sa, x,
                                   W->pt_l0_fW, W->pt_l0_fR, W->pt_l0_fb,
                                   W->pt_l0_bW, W->pt_l0_bR, W->pt_l0_bb,
                                   h0, c0, H, L);
    struct tensor * y1 = ada_layer_norm(y, style, W->pt_fc1_w,
                                        W->pt_fc1_b, C);
    struct tensor * x2 = tensor_concat(y1, s_bcast, 0);
    struct tensor * y2 = bidir_lstm(sa, x2,
                                    W->pt_l2_fW, W->pt_l2_fR, W->pt_l2_fb,
                                    W->pt_l2_bW, W->pt_l2_bR, W->pt_l2_bb,
                                    h0, c0, H, L);
    return ada_layer_norm(y2, style, W->pt_fc3_w, W->pt_fc3_b, C);
}

static struct tensor * build_acoustic(struct kittens_ctx * ctx,
                                      const int32_t * ids,
                                      struct tensor * h0,
                                      struct tensor * c0, int L) {
    const struct weights * W = &ctx->W;
    struct arena * sa = ctx->scratch_arena;
    const int H = ctx->arch.lstm_hidden;
    struct tensor * x = tensor_get_rows(W->ac_embd, ids, L);
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
    struct savept p_sty = save(style_pr);
    struct savept p_h0  = save(h0);
    struct savept p_c0  = save(c0);
    struct tensor * bert = build_albert(ctx, L, ids, pos, type);
    style_pr = restore(sa, &p_sty);
    h0       = restore(sa, &p_h0);
    c0       = restore(sa, &p_c0);
    savept_free(&p_sty);
    savept_free(&p_h0);
    savept_free(&p_c0);
    struct tensor * bert_proj = tensor_add(
        tensor_mul_mat(W->bert_enc_w, bert), W->bert_enc_b);
    struct tensor * prosody = build_pred_text(ctx, bert_proj, style_pr,
                                              h0, c0, L);
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

struct genfront_outs {
    struct tensor * f0_proj;
    struct tensor * n_proj;
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

struct noise_outs { struct hifi_out nr0; struct hifi_out nr1; };

#ifndef NOISE_NO_SEG

#define NOISE_SEG_F   256
#define NOISE_SEG_U 16384
#define NOISE_SEG_V  4096

struct noise_hosts {
    const float * fpf;
    const float * ps;
    const float * f0;
    const float * srange;
    int64_t       Tf;
};

static void noise_singen_segment(struct kittens_ctx * ctx,
                                 const struct noise_hosts * h,
                                 int64_t fa, int64_t fb,
                                 void * out_h) {
    struct arena * sa = ctx->scratch_arena;
    const int   hop    = 300;
    const float sr     = 24000.0f;
    const float two_pi = 2.0f * (float)M_PI;
    const int64_t nf = fb - fa;
    const int64_t nt = nf * hop;
    arena_reset(sa);
    struct tensor * fpf_t = tensor_new_2d(sa, 9, nf);
    memcpy(fpf_t->data, h->fpf + 9 * fa, (size_t)(9 * nf) * sizeof(float));
    struct tensor * ps_t = tensor_new_2d(sa, 9, nf);
    memcpy(ps_t->data, h->ps + 9 * fa, (size_t)(9 * nf) * sizeof(float));
    struct tensor * s_t = tensor_new_1d(sa, hop);
    memcpy(s_t->data, h->srange, (size_t)hop * sizeof(float));
    struct tensor * fpf_3d = tensor_reshape_3d(fpf_t, 9, nf, 1);
    struct tensor * s_3d   = tensor_reshape_3d(s_t, 1, 1, hop);
    struct tensor * fpf_x  = tensor_repeat_to(fpf_3d, 3, 9, nf, hop, 1);
    struct tensor * s_x    = tensor_repeat_to(s_3d,   3, 9, nf, hop, 1);
    struct tensor * within = tensor_mul(fpf_x, s_x);
    within = tensor_scale(within, two_pi / sr);
    struct tensor * ps_3d = tensor_reshape_3d(ps_t, 9, nf, 1);
    struct tensor * ps_x  = tensor_repeat_to(ps_3d, 3, 9, nf, hop, 1);
    struct tensor * phase = tensor_add(ps_x, within);
    phase = tensor_cont(tensor_permute(phase, 0, 2, 1, 3));
    phase = tensor_reshape_2d(phase, 9, nt);
    struct tensor * sines = tensor_scale(tensor_sin(phase), 0.1f);
    struct tensor * voiced = tensor_new_2d(sa, 1, nt);
    for (int64_t f = fa; f < fb; f++) {
        const float v = h->f0[f] > 0.0f ? 1.0f : 0.0f;
        float * dst = voiced->data + (f - fa) * hop;
        for (int s = 0; s < hop; s++) { dst[s] = v; }
    }
    struct tensor * sin_gen = tensor_mul(sines, voiced);
    memcpy((float *)out_h + 9 * fa * hop, sin_gen->data,
           (size_t)(9 * nt) * sizeof(float));
}

static void noise_stft_segment(struct kittens_ctx * ctx,
                               const void * exc_h, int64_t T_audio,
                               int64_t ua, int64_t ub, float eps,
                               void * stft_h) {
    struct arena * sa = ctx->scratch_arena;
    const int c16 = g_carry_fp16 >= 1;
    const int64_t nu = ub - ua;
    const int64_t W  = (nu - 1) * 5 + 20;
    const int64_t T0 = ua * 5 - 10;
    arena_reset(sa);
    struct tensor * slice = tensor_new_2d(sa, 1, W);
    const float * e32 = (const float *)exc_h;
    for (int64_t i = 0; i < W; i++) {
        const int64_t t = T0 + i;
        slice->data[i] = (t >= 0 && t < T_audio) ? e32[t] : 0.0f;
    }
    struct tensor * fr = named_fmt(ctx, "stft_fwd.real");
    struct tensor * fi = named_fmt(ctx, "stft_fwd.imag");
    struct tensor * st_r = conv1d_nlc(slice, fr, NULL, 5, 0, 1);
    struct tensor * st_i = conv1d_nlc(slice, fi, NULL, 5, 0, 1);
    struct tensor * eps_seg = tensor_new_1d(sa, 1);
    eps_seg->data[0] = eps;
    struct tensor * re2 = tensor_mul(st_r, st_r);
    struct tensor * im2 = tensor_mul(st_i, st_i);
    struct tensor * mag2 = tensor_add(re2, im2);
    mag2 = tensor_add(mag2, eps_seg);
    struct tensor * mag = tensor_sqrt(mag2);
    struct tensor * phi = tensor_atan2(st_i, st_r);
    struct tensor * out = tensor_concat(mag, phi, 0);
    carry_store(stft_h, c16, (size_t)(22 * ua), out->data,
                (size_t)(22 * nu));
}

static void noise_conv_blocks(struct kittens_ctx * ctx,
                              const void * in_h, int Cin, int64_t Lin,
                              const char * w_name, const char * b_name,
                              int stride, int pad,
                              void * out_h, int Cout, int64_t Lout) {
    struct arena * sa = ctx->scratch_arena;
    const int c16 = g_carry_fp16 >= 1;
    struct tensor * w = named_fmt(ctx, "%s", w_name);
    struct tensor * b = named_fmt(ctx, "%s", b_name);
    const int K = (int)w->ne[0];
    int64_t va = 0;
    while (va < Lout) {
        int64_t vb = va + (int64_t)NOISE_SEG_V;
        if (vb > Lout) { vb = Lout; }
        const int64_t nv = vb - va;
        const int64_t W  = (nv - 1) * stride + K;
        const int64_t U0 = va * stride - pad;
        arena_reset(sa);
        struct tensor * slice = tensor_new_2d(sa, Cin, W);
        for (int64_t i = 0; i < W; i++) {
            const int64_t u = U0 + i;
            float * dst = slice->data + i * Cin;
            if (u >= 0 && u < Lin) {
                carry_load(dst, in_h, c16, (size_t)(u * Cin),
                           (size_t)Cin);
            } else {
                memset(dst, 0, (size_t)Cin * sizeof(float));
            }
        }
        struct tensor * y = conv1d_nlc(slice, w, b, stride, 0, 1);
        carry_store(out_h, c16, (size_t)(va * Cout), y->data,
                    (size_t)(nv * Cout));
        va = vb;
    }
}

static struct noise_outs build_noise_contribs(struct kittens_ctx * ctx,
                                              struct tensor * f0_proj,
                                              struct tensor * style_aco,
                                              struct tensor * harmonics,
                                              struct tensor * s_range,
                                              struct tensor * eps_t,
                                              int F) {
    struct arena * sa = ctx->scratch_arena;
    const int     T_frames = 2 * F;
    const int     hop      = 300;
    const int64_t T_audio  = (int64_t)T_frames * hop;
    const float   sr       = 24000.0f;
    const float   two_pi   = 2.0f * (float)M_PI;
    trace_rss("noise:entry");
    struct tensor * f0_repeated = tensor_repeat_to(f0_proj, 2,
                                                   9, T_frames, 1, 1);
    struct tensor * harm_2d = tensor_reshape_2d(harmonics, 9, 1);
    struct tensor * f0_per_frame = tensor_mul(f0_repeated, harm_2d);
    struct tensor * step_nlc = tensor_scale(f0_per_frame, (float)hop / sr);
    struct tensor * step_ncl = tensor_cont(tensor_transpose(step_nlc));
    struct tensor * cs       = tensor_cumsum(step_ncl, 0);
    struct tensor * ps_ncl   = tensor_sub(cs, step_ncl);
    ps_ncl = tensor_scale(ps_ncl, two_pi);
    struct tensor * phase_start_nlc = tensor_cont(tensor_transpose(ps_ncl));
    trace_rss("noise:phase_start");
    struct savept p_fpf = save(f0_per_frame);
    struct savept p_ps  = save(phase_start_nlc);
    struct savept p_f0  = save(f0_proj);
    struct savept p_sr  = save(s_range);
    struct savept p_sty = save(style_aco);
    const float eps = eps_t->data[0];
    const struct noise_hosts h = {
        p_fpf.data, p_ps.data, p_f0.data, p_sr.data, T_frames
    };
    const int c16 = g_carry_fp16 >= 1;
    float * singen_h = (float *)malloc((size_t)(9 * T_audio)
                                       * sizeof(float));
    assert(singen_h != NULL);
    int64_t fa = 0;
    while (fa < T_frames) {
        int64_t fb = fa + (int64_t)NOISE_SEG_F;
        if (fb > T_frames) { fb = T_frames; }
        noise_singen_segment(ctx, &h, fa, fb, singen_h);
        fa = fb;
    }
    arena_reset(sa);
    struct tensor * sin_gen = tensor_wrap_2d(sa, singen_h, 9, T_audio);
    struct tensor * l_lin_w = named_fmt(ctx, "l_lin.weight");
    struct tensor * l_lin_b = named_fmt(ctx, "l_lin.bias");
    struct tensor * mixed = tensor_add(
        tensor_mul_mat(l_lin_w, sin_gen), l_lin_b);
    struct tensor * excitation = tensor_tanh(mixed);
    float * exc_h = (float *)malloc((size_t)T_audio * sizeof(float));
    assert(exc_h != NULL);
    memcpy(exc_h, excitation->data, (size_t)T_audio * sizeof(float));
    free(singen_h);
    trace_rss("noise:excitation");
    const int64_t L1 = tensor_conv_out_len(T_audio, 20, 5, 10, 1);
    void * stft_h = malloc((size_t)(22 * L1) * (c16 ? 2u : 4u));
    assert(stft_h != NULL);
    int64_t ua = 0;
    while (ua < L1) {
        int64_t ub = ua + (int64_t)NOISE_SEG_U;
        if (ub > L1) { ub = L1; }
        noise_stft_segment(ctx, exc_h, T_audio, ua, ub, eps, stft_h);
        ua = ub;
    }
    free(exc_h);
    trace_rss("noise:stft");
    struct tensor * nc0_w = named_fmt(ctx, "nc0.weight");
    struct tensor * nc1_w = named_fmt(ctx, "nc1.weight");
    const int     C0 = (int)nc0_w->ne[2];
    const int     C1 = (int)nc1_w->ne[2];
    const int64_t L0 = tensor_conv_out_len(L1, (int)nc0_w->ne[0], 6, 3, 1);
    float * nc0_h = host_alloc((size_t)(C0 * L0) * (c16 ? 2u : 4u));
    noise_conv_blocks(ctx, stft_h, 22, L1, "nc0.weight", "nc0.bias",
                      6, 3, nc0_h, C0, L0);
    trace_rss("noise:nc0");
    arena_reset(sa);
    struct savept p_nc0 = savept_wrap(nc0_h, C0, L0, c16);
    struct tensor * sty = restore(sa, &p_sty);
    struct hifi_out nr0 = build_hifi_block(ctx, "nr0", &p_nc0, sty, 1);
    trace_rss("noise:hifi-nr0");
    float * nc1_h = host_alloc((size_t)(C1 * L1) * (c16 ? 2u : 4u));
    noise_conv_blocks(ctx, stft_h, 22, L1, "nc1.weight", "nc1.bias",
                      1, 0, nc1_h, C1, L1);
    free(stft_h);
    trace_rss("noise:nc1");
    arena_reset(sa);
    struct savept p_nc1 = savept_wrap(nc1_h, C1, L1, c16);
    sty = restore(sa, &p_sty);
    struct hifi_out nr1 = build_hifi_block(ctx, "nr1", &p_nc1, sty, 1);
    trace_rss("noise:hifi-nr1");
    savept_free(&p_fpf);
    savept_free(&p_ps);
    savept_free(&p_f0);
    savept_free(&p_sr);
    savept_free(&p_sty);
    struct noise_outs r = { nr0, nr1 };
    return r;
}

#else

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
    (void)T_audio;
    trace_rss("noise:entry");
    struct tensor * f0_3d  = tensor_reshape_3d(f0_proj, 1, 1, T_frames);
    struct tensor * f0_audio_3d = tensor_repeat_to(f0_3d, 3,
                                                   1, hop, T_frames, 1);
    struct tensor * f0_audio = tensor_reshape_2d(f0_audio_3d, 1, T_audio);
    struct tensor * voiced = tensor_step(f0_audio);
    trace_rss("noise:f0+voiced");
    struct tensor * f0_repeated = tensor_repeat_to(f0_proj, 2,
                                                   9, T_frames, 1, 1);
    struct tensor * harm_2d = tensor_reshape_2d(harmonics, 9, 1);
    struct tensor * f0_per_frame = tensor_mul(f0_repeated, harm_2d);
    struct tensor * step_nlc = tensor_scale(f0_per_frame, (float)hop / sr);
    struct tensor * step_ncl = tensor_cont(tensor_transpose(step_nlc));
    struct tensor * cs       = tensor_cumsum(step_ncl, 0);
    struct tensor * ps_ncl   = tensor_sub(cs, step_ncl);
    ps_ncl = tensor_scale(ps_ncl, two_pi);
    struct tensor * phase_start_nlc = tensor_cont(tensor_transpose(ps_ncl));
    trace_rss("noise:phase_start");
    struct tensor * fpf_3d = tensor_reshape_3d(f0_per_frame, 9, T_frames, 1);
    struct tensor * s_3d   = tensor_reshape_3d(s_range, 1, 1, hop);
    struct tensor * fpf_x  = tensor_repeat_to(fpf_3d, 3, 9, T_frames, hop, 1);
    struct tensor * s_x    = tensor_repeat_to(s_3d,   3, 9, T_frames, hop, 1);
    struct tensor * within = tensor_mul(fpf_x, s_x);
    within = tensor_scale(within, two_pi / sr);
    trace_rss("noise:within(9,2F,hop)");
    struct tensor * ps_3d = tensor_reshape_3d(phase_start_nlc,
                                              9, T_frames, 1);
    struct tensor * ps_expanded = tensor_repeat_to(ps_3d, 3,
                                                   9, T_frames, hop, 1);
    struct tensor * phase = tensor_add(ps_expanded, within);
    trace_rss("noise:phase");
    phase = tensor_permute(phase, 0, 2, 1, 3);
    phase = tensor_cont(phase);
    phase = tensor_reshape_2d(phase, 9, T_audio);
    trace_rss("noise:phase-perm");
    struct tensor * sines = tensor_scale(tensor_sin(phase), 0.1f);
    struct tensor * sin_gen = tensor_mul(sines, voiced);
    trace_rss("noise:sin_gen");
    struct tensor * l_lin_w = named_fmt(ctx, "l_lin.weight");
    struct tensor * l_lin_b = named_fmt(ctx, "l_lin.bias");
    struct tensor * mixed = tensor_add(
        tensor_mul_mat(l_lin_w, sin_gen), l_lin_b);
    struct tensor * excitation = tensor_tanh(mixed);
    trace_rss("noise:excitation");
    struct tensor * stft_fr = named_fmt(ctx, "stft_fwd.real");
    struct tensor * stft_fi = named_fmt(ctx, "stft_fwd.imag");
    struct tensor * stft_real = conv1d_nlc(excitation, stft_fr,
                                           NULL, 5, 10, 1);
    struct tensor * stft_imag = conv1d_nlc(excitation, stft_fi,
                                           NULL, 5, 10, 1);
    trace_rss("noise:stft");
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
    trace_rss("noise:nc0+nc1");
    struct savept p_nc0 = save16(nc0);
    struct savept p_nc1 = save16(nc1);
    struct savept p_sty = save(style_aco);
    struct hifi_out nr0 = build_hifi_block(ctx, "nr0", &p_nc0, style_aco, 1);
    trace_rss("noise:hifi-nr0");
    style_aco = restore(sa, &p_sty);
    struct hifi_out nr1 = build_hifi_block(ctx, "nr1", &p_nc1, style_aco, 1);
    trace_rss("noise:hifi-nr1");
    savept_free(&p_sty);
    struct noise_outs r = { nr0, nr1 };
    return r;
}

#endif

#ifndef ISTFT_NO_SEG

#define ISTFT_SEG_L 4096

struct istft_stream {
    const float * r2;
    const float * r3;
    int           C;
    int64_t       L;
    int           fp16;
};

static struct tensor * istft_mix_segment(struct arena * sa,
                                         const struct istft_stream * s,
                                         int64_t lo, int64_t hi) {
    struct tensor * seg = tensor_new_2d(sa, s->C, hi - lo);
    const size_t n   = (size_t)(hi - lo) * (size_t)s->C;
    const size_t off = (size_t)lo * (size_t)s->C;
    float * dst = seg->data;
    if (s->fp16) {
        const _Float16 * r2 = (const _Float16 *)s->r2 + off;
        const _Float16 * r3 = (const _Float16 *)s->r3 + off;
        for (size_t i = 0; i < n; i++) {
            const float v = 0.5f * ((float)r2[i] + (float)r3[i]);
            const float w = 0.1f * v;
            dst[i] = v > w ? v : w;
        }
    } else {
        const float * r2 = s->r2 + off;
        const float * r3 = s->r3 + off;
        for (size_t i = 0; i < n; i++) {
            const float v = 0.5f * (r2[i] + r3[i]);
            const float w = 0.1f * v;
            dst[i] = v > w ? v : w;
        }
    }
    return seg;
}

struct istft_spectra { struct tensor * re; struct tensor * im; };

static struct istft_spectra istft_segment_spectra(struct tensor * seg,
                                                  struct tensor * cp_w,
                                                  struct tensor * cp_b) {
    struct tensor * y = conv1d_nlc(seg, cp_w, cp_b, 1, -1, 1);
    const int64_t W = y->ne[1];
    struct tensor * mag_log = tensor_cont(
        tensor_view_2d(y, 11, W, (size_t)y->nb[1], 0));
    struct tensor * phase = tensor_cont(
        tensor_view_2d(y, 11, W, (size_t)y->nb[1], (size_t)(11 * y->nb[0])));
    struct tensor * mag   = tensor_exp(mag_log);
    struct tensor * inner = tensor_sin(phase);
    struct istft_spectra r;
    r.re = tensor_mul(mag, tensor_cos(inner));
    r.im = tensor_mul(mag, tensor_sin(inner));
    return r;
}

static struct tensor * istft_frame_window(struct tensor * spec,
                                          int64_t lo, int64_t wf,
                                          int64_t b) {
    return tensor_cont(tensor_view_2d(spec, 11, b - wf,
                                      (size_t)spec->nb[1],
                                      (size_t)((wf - lo) * spec->nb[1])));
}

static struct tensor * build_istft_tail(struct kittens_ctx * ctx,
                                        struct hifi_out * r2,
                                        struct hifi_out * r3) {
    struct arena * sa = ctx->scratch_arena;
    struct tensor * cp_w = named_fmt(ctx, "gen.cp.weight");
    struct tensor * cp_b = named_fmt(ctx, "gen.cp.bias");
    struct tensor * sb_r = named_fmt(ctx, "stft_bwd.real");
    struct tensor * sb_i = named_fmt(ctx, "stft_bwd.imag");
    const int     pad_cp = ((int)cp_w->ne[0] - 1) / 2;
    const int     Kb     = (int)sb_r->ne[0];
    const int     fr_ov  = (Kb + 3) / 5;
    assert(r2->fp16 == r3->fp16);
    const struct istft_stream s = {
        r2->data, r3->data, r2->C, r2->L, r2->fp16
    };
    const size_t row = (size_t)s.C * (s.fp16 ? 2u : 4u);
    size_t r2_retired = 0;
    size_t r3_retired = 0;
    const int64_t Tfull = (s.L - 1) * 5 + Kb;
    float * audio_h = (float *)malloc((size_t)Tfull * sizeof(float));
    assert(audio_h != NULL);
    int64_t a = 0;
    while (a < s.L) {
        int64_t b = a + (int64_t)ISTFT_SEG_L;
        if (b > s.L) { b = s.L; }
        int64_t lo = a - (int64_t)(fr_ov + pad_cp);
        if (lo < 0) { lo = 0; }
        int64_t hi = b + (int64_t)pad_cp;
        if (hi > s.L) { hi = s.L; }
        int64_t wf = a - (int64_t)fr_ov;
        if (wf < 0) { wf = 0; }
        host_retire_prefix(r2->data, &r2_retired, (size_t)lo * row);
        host_retire_prefix(r3->data, &r3_retired, (size_t)lo * row);
        arena_reset(sa);
        struct tensor * seg = istft_mix_segment(sa, &s, lo, hi);
        struct istft_spectra sp = istft_segment_spectra(seg, cp_w, cp_b);
        struct tensor * re_w = istft_frame_window(sp.re, lo, wf, b);
        struct tensor * im_w = istft_frame_window(sp.im, lo, wf, b);
        struct tensor * ar = conv_transpose_1d_nlc(re_w, sb_r, NULL, 5, 0);
        struct tensor * ai = conv_transpose_1d_nlc(im_w, sb_i, NULL, 5, 0);
        struct tensor * chunk = tensor_sub(ar, ai);
        const int64_t g0 = a * 5;
        const int64_t g1 = (b == s.L) ? Tfull : b * 5;
        memcpy(audio_h + g0, chunk->data + (a - wf) * 5,
               (size_t)(g1 - g0) * sizeof(float));
        a = b;
    }
    host_free_rest(r2->data, r2_retired, hifi_out_bytes(r2));
    r2->data = NULL;
    host_free_rest(r3->data, r3_retired, hifi_out_bytes(r3));
    r3->data = NULL;
    const int trim = ctx->arch.istft_trim;
    assert(Tfull > 2 * trim);
    arena_reset(sa);
    struct tensor * out = tensor_new_1d(sa, Tfull - 2 * trim);
    memcpy(out->data, audio_h + trim,
           (size_t)(Tfull - 2 * trim) * sizeof(float));
    free(audio_h);
    return out;
}

#endif

static struct tensor * hifi_materialize(struct arena * sa,
                                        struct hifi_out * h) {
    struct tensor * t = tensor_new_2d(sa, h->C, h->L);
    carry_load(t->data, h->data, h->fp16, 0,
               (size_t)h->C * (size_t)h->L);
    host_free(h->data, hifi_out_bytes(h));
    h->data = NULL;
    return t;
}

static struct tensor * build_generator(struct kittens_ctx * ctx,
                                   struct tensor * dec_out,
                                   struct hifi_out * nr0,
                                   struct hifi_out * nr1,
                                   struct tensor * style_aco) {
    struct tensor * u0_w = named_fmt(ctx, "gen.u0.weight");
    struct tensor * u0_b = named_fmt(ctx, "gen.u0.bias");
    struct tensor * u1_w = named_fmt(ctx, "gen.u1.weight");
    struct tensor * u1_b = named_fmt(ctx, "gen.u1.bias");
    struct arena * sa = ctx->scratch_arena;
    struct savept p_sty = save(style_aco);
    struct tensor * x = tensor_leaky_relu(dec_out, 0.1f);
    x = conv_transpose_1d_nlc(x, u0_w, u0_b, 10, 5);
    x = tensor_add(x, hifi_materialize(sa, nr0));
    struct savept px = save16(x);
    struct hifi_out r0 = build_hifi_block(ctx, "gen.r0", &px, style_aco, 0);
    struct tensor * sty = restore(sa, &p_sty);
    struct hifi_out r1 = build_hifi_block(ctx, "gen.r1", &px, sty, 1);
    arena_reset(sa);
    x = tensor_add(hifi_materialize(sa, &r0), hifi_materialize(sa, &r1));
    x = tensor_scale(x, 0.5f);
    x = tensor_leaky_relu(x, 0.1f);
    x = conv_transpose_1d_nlc(x, u1_w, u1_b, 6, 3);
    px = save(x);
    arena_reset(sa);
    x = restore(sa, &px);
    savept_free(&px);
    x = reflection_pad_left(x, 1);
    x = tensor_add(x, hifi_materialize(sa, nr1));
    sty = restore(sa, &p_sty);
    px = save16(x);
    struct hifi_out r2 = build_hifi_block(ctx, "gen.r2", &px, sty, 0);
    sty = restore(sa, &p_sty);
    struct hifi_out r3 = build_hifi_block(ctx, "gen.r3", &px, sty, 1);
    savept_free(&px);
    savept_free(&p_sty);
#ifndef ISTFT_NO_SEG
    trace_rss("gen:istft-tail");
    struct tensor * out = build_istft_tail(ctx, &r2, &r3);
    trace_rss("gen:audio");
    return out;
#else
    struct tensor * cp_w = named_fmt(ctx, "gen.cp.weight");
    struct tensor * cp_b = named_fmt(ctx, "gen.cp.bias");
    struct tensor * sb_r = named_fmt(ctx, "stft_bwd.real");
    struct tensor * sb_i = named_fmt(ctx, "stft_bwd.imag");
    arena_reset(sa);
    x = tensor_add(hifi_materialize(sa, &r2), hifi_materialize(sa, &r3));
    x = tensor_scale(x, 0.5f);
    x = tensor_leaky_relu(x, 0.1f);
    x = conv1d_nlc(x, cp_w, cp_b, 1, -1, 1);
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
    struct tensor * audio = tensor_sub(audio_r, audio_i);
    const int trim = ctx->arch.istft_trim;
    const int64_t T = audio->ne[1];
    assert(T > 2 * trim);
    return tensor_cont(tensor_view_1d(audio, T - 2 * trim,
                                      (size_t)trim * sizeof(float)));
#endif
}

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
    trace_rss("entry");
    arena_set_active(sa);
    int   * durs = NULL;
    float * prosody_h = NULL, * text_h = NULL, * dur_h = NULL;
    float * prosody_lr_h = NULL, * text_lr_h = NULL;
    float * f0p_h = NULL, * np_h = NULL, * dec_h = NULL;
    float * audio_buf = NULL;
    arena_reset(sa);
    {
        int32_t * pos_ids  = (int32_t *)malloc(sizeof(int32_t) * L);
        int32_t * type_ids = (int32_t *)malloc(sizeof(int32_t) * L);
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
    trace_rss("after-stage1");
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
    trace_rss("after-stage2");
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
    trace_rss("after-stage3");
    struct noise_outs nz = { { NULL, 0, 0, 0 }, { NULL, 0, 0, 0 } };
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
        nz = build_noise_contribs(ctx, f0_t, sty_a, harm, s_rng,
                                  eps_t, F);
    }
    free(f0p_h); f0p_h = NULL;
    trace_rss("after-stage4a");
    int n = 0;
    arena_reset(sa);
    {
        struct tensor * dec_t = tensor_new_2d(sa, 256, 2 * F);
        struct tensor * sty_a = tensor_new_1d(sa, 128);
        memcpy(dec_t->data, dec_h,     sizeof(float) * 256 * 2 * F);
        memcpy(sty_a->data, style256,  sizeof(float) * 128);
        free(dec_h); dec_h = NULL;
        struct tensor * audio_t = build_generator(ctx, dec_t,
                                                  &nz.nr0, &nz.nr1,
                                                  sty_a);
        const int T_audio = (int)audio_t->ne[0];
        n = T_audio;
        const int tail_drop = 3 * 600;
        if (n > tail_drop) { n -= tail_drop; }
        audio_buf = (float *)malloc(sizeof(float) * (size_t)n);
        memcpy(audio_buf, audio_t->data, sizeof(float) * (size_t)n);
    }
    trace_rss("after-stage4b");
    fade_in (audio_buf, n,  72);
    fade_out(audio_buf, n, 960);
    out.samples   = audio_buf;
    out.n_samples = (uint64_t)n;
    arena_set_active(NULL);
    arena_reset(sa);
    return out;
}

#ifdef KITTENS_TESTS

int main(int argc, char ** argv) {
    const char * path = argc > 1
        ? argv[1]
        : "Resources/nano/kitten_full.gguf";
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

#endif

#endif
