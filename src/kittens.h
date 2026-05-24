// kittens.h -- public C API for the pure-C / cblas KittenTTS backend.
//
// Lifecycle:
//   struct kittens_ctx * ctx = kittens_create("path/to/kitten_full.gguf");
//   struct kittens_audio a   = kittens_synthesize(ctx, ids, n_ids, style, 1.0f);
//   // ... do something with a.samples (f32 PCM, 24 kHz mono) ...
//   kittens_audio_free(a);
//   kittens_destroy(ctx);
//
// All inputs are caller-owned; outputs returned by kittens_synthesize
// must be released with kittens_audio_free. Thread-unsafe - use one ctx
// per thread, or add your own locking.
//
// Backend characteristic: pure C99 + a portable cblas. On Apple this is
// Accelerate; on Linux/Windows OpenBLAS or BLIS. No ggml linkage.

#pragma once
#ifndef KITTENS_H
#define KITTENS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct kittens_ctx;

// Create a context that owns the loaded GGUF model and inference state.
// Returns NULL on failure; kittens_last_error() returns the message.
struct kittens_ctx * kittens_create(const char * gguf_path);

// Destroy a context and free all associated resources.
void kittens_destroy(struct kittens_ctx * ctx);

// Audio output buffer. `samples` is owned by the library; release via
// kittens_audio_free.
struct kittens_audio {
    float *   samples;
    uint64_t  n_samples;
};

// Synthesize a single chunk:
//   phonemes:    n_phonemes int32 phoneme IDs (caller-owned)
//   n_phonemes:  L
//   style256:    256 floats - concatenated [acoustic[0:128], prosodic[128:256]]
//   speed:       1.0 = normal speech rate
//
// Output: 24 kHz f32 PCM mono. Returned `samples` is NULL only on error.
struct kittens_audio kittens_synthesize(struct kittens_ctx * ctx,
                                        const int32_t * phonemes,
                                        int             n_phonemes,
                                        const float *   style256,
                                        float           speed);

// Free a kittens_audio returned by kittens_synthesize.
void kittens_audio_free(struct kittens_audio a);

// Last-error message for a context (or for global init failures if
// ctx is NULL). Pointer is valid until the next kittens_* call on the
// same context.
const char * kittens_last_error(const struct kittens_ctx * ctx);

// Select the sgemm implementation tensor_mul_mat dispatches to. Process-
// global, single-threaded; caller must serialize. Default 0.
//   0 = Accelerate cblas_sgemm (fast on Apple)
//   1 = portable hand-rolled 4x8 tiled C kernel (research / wasm path)
// See rnd/blas/README.md for the comparison bench informing this choice.
void kittens_set_sgemm_impl(int impl);
int  kittens_get_sgemm_impl(void);

// Memory diagnostics. `phys_footprint` is the field iOS jetsam reads
// — wired + dirty + compressed pages attributed to this process. The
// peak is sampled at every internal mmap (arena slab growth), which
// is the only place virtual commit grows on the synthesize hot path,
// so the peak reflects the true intra-synthesize high-water mark
// rather than whatever task_info would return after arena_reset has
// already freed the slabs. `reset_peak` rebases the peak to the
// current footprint — call from the UI's "clear" affordance.
uint64_t kittens_current_phys_footprint(void);
uint64_t kittens_peak_phys_footprint(void);
void     kittens_reset_peak_phys_footprint(void);

#ifdef __cplusplus
}
#endif

#endif // KITTENS_H
