#pragma once
#ifndef KITTENS_H
#define KITTENS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct kittens_ctx;

struct kittens_ctx * kittens_create(const char * gguf_path);

void kittens_destroy(struct kittens_ctx * ctx);

struct kittens_audio {
    float *   samples;
    uint64_t  n_samples;
};

struct kittens_audio kittens_synthesize(struct kittens_ctx * ctx,
                                        const int32_t * phonemes,
                                        int             n_phonemes,
                                        const float *   style256,
                                        float           speed);

void kittens_audio_free(struct kittens_audio a);

const char * kittens_last_error(const struct kittens_ctx * ctx);

void kittens_set_sgemm_impl(int impl);
int  kittens_get_sgemm_impl(void);

void kittens_set_carry_fp16(int mode);
int  kittens_get_carry_fp16(void);

uint64_t kittens_current_phys_footprint(void);
uint64_t kittens_peak_phys_footprint(void);
void     kittens_reset_peak_phys_footprint(void);

#ifdef __cplusplus
}
#endif

#endif
