// kittens_tts_cli.c -- self-contained command-line driver for the
// pure-C KittenTTS backend. No Swift. The model (kitten_full.gguf),
// the eight voice embeddings (voices.safetensors), and the
// phonemizer data (en_rules, en_list) are all linked INTO the
// executable via C23 #embed, so the single binary is fully portable.
//
// Build (see Makefile):
//   clang -std=c23 -DKITTENS_TTS_CLI_MAIN ... kittens_tts_cli.c \
//         src/kittens.c src/phonemizer.c -framework Accelerate -lm
//
// Usage:
//   kittens-tts-cli -p "Text. More text." [-s Voice] [-o out.wav]
//   kittens-tts-cli -f input.txt -s Kiki -o out.mp3
//   kittens-tts-cli --voices
//
// Pipeline (a deliberately simple subset of the Swift app's):
//   text -> split into paragraphs (one per '\n') -> sentences
//   (split on . ! ?) -> phonemize each (phonemizer.c) -> map IPA
//   code points to model phoneme ids (the table generated from the
//   Swift PhonemeVocab) -> kittens_synthesize per sentence -> splice
//   with inter-sentence / inter-paragraph silence -> WAV (or pipe
//   raw PCM through ffmpeg for .mp3/.m4a/etc.).
//
// NOT ported from the Swift TextPreprocessor: currency / percent /
// ordinal / decimal expansion. The phonemizer speaks plain cardinals
// ("1969" -> "one thousand nine hundred sixty nine") on its own; the
// rest ("3rd", "50%", "$5") read literally. Good enough for prose.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <stdint.h>
#include <ctype.h>
#include <unistd.h>

#include "kittens.h"
#include "kitten_symbols.h"

// ---- phonemizer surface (implemented in src/phonemizer.c) ----------
typedef void * PhonemizerHandle;
PhonemizerHandle phonemizer_create(const char * rules_path,
                                   const char * list_path,
                                   const char * dialect);
char * phonemizer_phonemize(PhonemizerHandle h, const char * text);
void   phonemizer_free_string(char * s);
void   phonemizer_destroy(PhonemizerHandle h);

// ---- embedded resources (C23 #embed) -------------------------------
static const unsigned char EMB_GGUF[] = {
#embed "resources/kitten_full.gguf"
};
static const unsigned char EMB_VOICES[] = {
#embed "resources/voices.safetensors"
};
static const unsigned char EMB_RULES[] = {
#embed "resources/en_rules"
};
static const unsigned char EMB_LIST[] = {
#embed "resources/en_list"
};

// ===========================================================================
// Audio model constants (mirror the Swift Backend / C synth contract).
// ===========================================================================
enum { SAMPLE_RATE   = 24000 };
enum { STYLE_DIM     = 256 };
enum { VOICE_ROWS    = 400 };
enum { PARA_SIL_MS   = 180 };   // silence before a new paragraph
enum { SENT_SIL_MS   = 80 };    // silence before a new sentence

// ===========================================================================
// Voice catalog (friendly name <-> internal id <-> per-voice speed
// prior), copied verbatim from cpu/Backend.swift so the CLI speaks
// identically to the app.
// ===========================================================================
struct voice_entry {
    const char * friendly;
    const char * internal;
    float        speed_prior;
};
// Display order matches Backend.voiceDisplayOrder.
static const struct voice_entry VOICES[] = {
    { "Hugo",   "expr-voice-4-m", 0.9f },
    { "Luna",   "expr-voice-3-f", 0.8f },
    { "Kiki",   "expr-voice-5-f", 0.8f },
    { "Leo",    "expr-voice-5-m", 0.8f },
    { "Bella",  "expr-voice-2-f", 0.8f },
    { "Jasper", "expr-voice-2-m", 0.8f },
    { "Bruno",  "expr-voice-3-m", 0.8f },
    { "Rosie",  "expr-voice-4-f", 0.8f },
};
enum { N_VOICES = (int)(sizeof(VOICES) / sizeof(VOICES[0])) };

// Resolve a -s argument to a voice entry. Accepts either the friendly
// name ("Kiki") or the internal id ("expr-voice-5-f"), case-
// insensitive on the friendly side. Returns NULL if unknown.
static const struct voice_entry * voice_lookup(const char * name) {
    const struct voice_entry * found = NULL;
    int i = 0;
    while (i < N_VOICES && found == NULL) {
        if (strcasecmp(name, VOICES[i].friendly) == 0
            || strcmp(name, VOICES[i].internal) == 0) {
            found = &VOICES[i];
        }
        i++;
    }
    return found;
}

// ===========================================================================
// IPA code-point -> phoneme id. KITTEN_SYMBOL_CP[i] is the code point
// for id i (generated header). Duplicate code points resolve
// last-wins, matching the Swift Dictionary(uniquingKeysWith:). We
// build a flat lookup array once (max code point is U+2192).
// ===========================================================================
enum { SYM_MAP_SIZE = 0x2300 };
static int g_sym_id[SYM_MAP_SIZE];

static void symbol_map_build(void) {
    int i = 0;
    while (i < SYM_MAP_SIZE) { g_sym_id[i] = -1; i++; }
    i = 0;
    while (i < KITTEN_SYMBOL_N) {
        unsigned cp = KITTEN_SYMBOL_CP[i];
        if (cp < SYM_MAP_SIZE) { g_sym_id[cp] = i; }   // later wins
        i++;
    }
}

static int symbol_id(unsigned cp) {
    int id = -1;
    if (cp < SYM_MAP_SIZE) { id = g_sym_id[cp]; }
    return id;
}

// Decode one UTF-8 sequence starting at s[*i] (within [0,n)). On
// success advances *i past the sequence and writes the code point to
// *cp; returns 1. On a malformed lead byte advances by one and
// returns 0 (caller skips it).
static int utf8_decode(const char * s, size_t n, size_t * i,
                       unsigned * cp) {
    int ok = 0;
    unsigned char c0 = (unsigned char)s[*i];
    if (c0 < 0x80) {
        *cp = c0; *i += 1; ok = 1;
    } else if ((c0 & 0xE0) == 0xC0 && *i + 1 < n) {
        *cp = ((c0 & 0x1Fu) << 6)
            | ((unsigned char)s[*i + 1] & 0x3Fu);
        *i += 2; ok = 1;
    } else if ((c0 & 0xF0) == 0xE0 && *i + 2 < n) {
        *cp = ((c0 & 0x0Fu) << 12)
            | (((unsigned char)s[*i + 1] & 0x3Fu) << 6)
            | ((unsigned char)s[*i + 2] & 0x3Fu);
        *i += 3; ok = 1;
    } else if ((c0 & 0xF8) == 0xF0 && *i + 3 < n) {
        *cp = ((c0 & 0x07u) << 18)
            | (((unsigned char)s[*i + 1] & 0x3Fu) << 12)
            | (((unsigned char)s[*i + 2] & 0x3Fu) << 6)
            | ((unsigned char)s[*i + 3] & 0x3Fu);
        *i += 4; ok = 1;
    } else {
        *i += 1;   // resync past a bad byte
    }
    return ok;
}

// True for sentence-final / pause punctuation the model uses for
// prosody. Mirrors the Swift cePhonemize punct-preservation set.
static int is_terminal_punct(char c) {
    return c == '.' || c == '!' || c == '?'
        || c == ',' || c == ';' || c == ':';
}

// Build the model id sequence for one phonemized sentence. Token shape
// matches Swift PhonemeVocab.phonemize: [0, <ipa ids>, <term?>, 10, 0]
// where 0 is pad and 10 is the ellipsis marker the model expects as a
// terminator. `term` is the sentence's trailing punctuation char (or 0)
// — appended (mapped through the symbol table) when the phonemizer
// dropped it, so intonation survives. Returns the id count written.
static int sentence_to_ids(const char * ipa, char term,
                           int32_t * ids, int cap) {
    int n = 0;
    ids[n++] = 0;
    size_t i = 0, len = strlen(ipa);
    int last_emitted = -1;
    while (i < len && n < cap - 3) {
        unsigned cp = 0;
        if (utf8_decode(ipa, len, &i, &cp)) {
            int id = symbol_id(cp);
            if (id >= 0) { ids[n++] = id; last_emitted = id; }
        }
    }
    if (term != 0 && n < cap - 3) {
        int tid = symbol_id((unsigned char)term);
        if (tid >= 0 && tid != last_emitted) { ids[n++] = tid; }
    }
    ids[n++] = 10;
    ids[n++] = 0;
    return n;
}

// ===========================================================================
// Voice embedding lookup from the embedded safetensors blob. Minimal
// reader: u64 header length, JSON header, packed F32 data. We only
// need one tensor (the named voice) as a flat [400 * 256] float row-
// major buffer. Returns a pointer into EMB_VOICES (no copy) or NULL.
// ===========================================================================
static const float * voice_embedding(const char * internal_name) {
    const float * rows = NULL;
    uint64_t hlen = 0;
    memcpy(&hlen, EMB_VOICES, 8);              // u64 LE on arm64
    const char * json = (const char *)EMB_VOICES + 8;
    size_t data_base = 8 + (size_t)hlen;
    // Find "name": { ... "data_offsets":[start,end] ... } by locating
    // the quoted key, then the data_offsets array after it.
    char key[64];
    int klen = snprintf(key, sizeof key, "\"%s\"", internal_name);
    const char * at = memmem(json, (size_t)hlen, key, (size_t)klen);
    if (at != NULL) {
        const char * doff = strstr(at, "\"data_offsets\"");
        const char * lb = (doff != NULL) ? strchr(doff, '[') : NULL;
        if (lb != NULL) {
            long start = 0, end = 0;
            if (sscanf(lb + 1, "%ld , %ld", &start, &end) == 2) {
                rows = (const float *)(EMB_VOICES
                        + data_base + (size_t)start);
                (void)end;
            }
        }
    }
    return rows;
}

// ===========================================================================
// Growable f32 audio buffer.
// ===========================================================================
struct audio_buf {
    float * data;
    size_t  n;
    size_t  cap;
};

static int audio_reserve(struct audio_buf * b, size_t extra) {
    int ok = 1;
    if (b->n + extra > b->cap) {
        size_t want = (b->cap == 0) ? (size_t)SAMPLE_RATE : b->cap;
        while (want < b->n + extra) { want *= 2; }
        float * p = (float *)realloc(b->data, want * sizeof(float));
        if (p == NULL) {
            ok = 0;
        } else {
            b->data = p;
            b->cap = want;
        }
    }
    return ok;
}

static void audio_push_silence(struct audio_buf * b, int samples) {
    if (samples > 0 && audio_reserve(b, (size_t)samples)) {
        memset(b->data + b->n, 0, (size_t)samples * sizeof(float));
        b->n += (size_t)samples;
    }
}

static void audio_push(struct audio_buf * b, const float * src,
                       size_t count) {
    if (count > 0 && audio_reserve(b, count)) {
        memcpy(b->data + b->n, src, count * sizeof(float));
        b->n += count;
    }
}

// ===========================================================================
// Synthesis: walk paragraphs (split on '\n') then sentences (split on
// . ! ?), phonemize + synth each, splice with silence. `emitted`
// tracks whether anything has been spoken yet so the very first chunk
// carries no leading silence.
// ===========================================================================
struct synth_ctx {
    struct kittens_ctx * tts;
    PhonemizerHandle     phon;
    const float *        voice;     // [400 * 256]
    float                speed;     // already × speed prior
};

// Synthesize one trimmed, non-empty sentence and append it (with the
// supplied leading silence) to `out`. `term` is its trailing punct.
static void synth_sentence(struct synth_ctx * s, const char * text,
                           char term, int silence_ms,
                           struct audio_buf * out) {
    char * ipa = phonemizer_phonemize(s->phon, text);
    if (ipa != NULL) {
        int32_t ids[4096];
        int n_ids = sentence_to_ids(ipa, term, ids, 4096);
        phonemizer_free_string(ipa);
        int ref = (int)strlen(text);
        if (ref > VOICE_ROWS - 1) { ref = VOICE_ROWS - 1; }
        const float * style = s->voice + (size_t)ref * STYLE_DIM;
        int sil = (int)((double)silence_ms * (SAMPLE_RATE / 1000.0)
                        / (double)s->speed);
        audio_push_silence(out, sil);
        struct kittens_audio a = kittens_synthesize(
            s->tts, ids, n_ids, style, s->speed);
        if (a.samples != NULL) {
            audio_push(out, a.samples, (size_t)a.n_samples);
            kittens_audio_free(a);
        }
    }
}

// Split one paragraph into sentences on . ! ? (terminator kept with
// the sentence), trim, synth each. `first_para` selects the leading
// silence grade for the paragraph's first sentence.
static void synth_paragraph(struct synth_ctx * s, const char * para,
                            size_t len, int first_para,
                            struct audio_buf * out, int * emitted) {
    size_t i = 0;
    while (i < len) {
        size_t start = i;
        while (i < len && para[i] != '.' && para[i] != '!'
               && para[i] != '?') {
            i++;
        }
        char term = (i < len) ? para[i] : 0;
        size_t end = (i < len) ? i + 1 : i;   // include terminator
        // Trim leading/trailing whitespace of [start, end).
        size_t a = start, b = end;
        while (a < b && isspace((unsigned char)para[a])) { a++; }
        while (b > a && isspace((unsigned char)para[b - 1])) { b--; }
        // Drop a lone terminator (b-a==1 and it's the punct).
        int has_text = (b > a)
            && !(b - a == 1 && para[a] == term && term != 0);
        if (has_text) {
            char buf[2048];
            size_t m = b - a;
            if (m > sizeof(buf) - 1) { m = sizeof(buf) - 1; }
            memcpy(buf, para + a, m);
            buf[m] = '\0';
            int first_sentence = (start == 0);
            int sil = (*emitted == 0) ? 0
                    : ((first_para && first_sentence) ? PARA_SIL_MS
                                                      : SENT_SIL_MS);
            synth_sentence(s, buf, term, sil, out);
            *emitted = 1;
        }
        i = end;
    }
}

static void synth_document(struct synth_ctx * s, const char * text,
                           struct audio_buf * out) {
    int emitted = 0;
    size_t i = 0, len = strlen(text);
    int para_idx = 0;
    while (i < len) {
        size_t start = i;
        while (i < len && text[i] != '\n') { i++; }
        synth_paragraph(s, text + start, i - start,
                        para_idx == 0 ? 1 : 1, out, &emitted);
        // (Every paragraph boundary uses the paragraph silence grade
        // for its first sentence; first_para flag stays 1 so the
        // grade selection in synth_paragraph picks PARA_SIL_MS.)
        if (i < len) { i++; }   // skip the '\n'
        para_idx++;
    }
}

// ===========================================================================
// Output: native WAV writer, or pipe raw s16le PCM through ffmpeg for
// any other extension (mp3 / m4a / aac / ...).
// ===========================================================================
static int16_t clamp_i16(float v) {
    float s = v;
    if (s > 1.0f) { s = 1.0f; }
    if (s < -1.0f) { s = -1.0f; }
    return (int16_t)(s * 32767.0f);
}

static void put_u32(FILE * f, uint32_t v) {
    unsigned char b[4] = { (unsigned char)(v),
                           (unsigned char)(v >> 8),
                           (unsigned char)(v >> 16),
                           (unsigned char)(v >> 24) };
    fwrite(b, 1, 4, f);
}
static void put_u16(FILE * f, uint16_t v) {
    unsigned char b[2] = { (unsigned char)(v),
                           (unsigned char)(v >> 8) };
    fwrite(b, 1, 2, f);
}

static int write_wav_stream(FILE * f, const float * pcm, size_t n) {
    uint32_t data_bytes = (uint32_t)(n * 2);
    fwrite("RIFF", 1, 4, f);
    put_u32(f, 36 + data_bytes);
    fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f);
    put_u32(f, 16);
    put_u16(f, 1);                         // PCM
    put_u16(f, 1);                         // mono
    put_u32(f, SAMPLE_RATE);
    put_u32(f, SAMPLE_RATE * 2);           // byte rate
    put_u16(f, 2);                         // block align
    put_u16(f, 16);                        // bits
    fwrite("data", 1, 4, f);
    put_u32(f, data_bytes);
    size_t i = 0;
    while (i < n) {
        int16_t s = clamp_i16(pcm[i]);
        put_u16(f, (uint16_t)s);
        i++;
    }
    return 1;
}

static int write_wav_file(const char * path, const float * pcm,
                          size_t n) {
    int ok = 0;
    FILE * f = fopen(path, "wb");
    if (f != NULL) {
        ok = write_wav_stream(f, pcm, n);
        fclose(f);
    }
    return ok;
}

// Pipe raw little-endian s16 PCM into ffmpeg, which infers the target
// container/codec from the output path's extension.
static int write_via_ffmpeg(const char * path, const float * pcm,
                            size_t n) {
    int ok = 0;
    char cmd[2048];
    snprintf(cmd, sizeof cmd,
        "ffmpeg -loglevel error -y -f s16le -ar %d -ac 1 -i - \"%s\"",
        SAMPLE_RATE, path);
    FILE * pipe = popen(cmd, "w");
    if (pipe != NULL) {
        size_t i = 0;
        while (i < n) {
            int16_t s = clamp_i16(pcm[i]);
            unsigned char b[2] = { (unsigned char)s,
                                   (unsigned char)((uint16_t)s >> 8) };
            fwrite(b, 1, 2, pipe);
            i++;
        }
        int rc = pclose(pipe);
        ok = (rc == 0);
    }
    return ok;
}

static int has_suffix_ci(const char * s, const char * suf) {
    size_t ls = strlen(s), lf = strlen(suf);
    return ls >= lf && strcasecmp(s + ls - lf, suf) == 0;
}

static int write_output(const char * path, const float * pcm,
                        size_t n) {
    int ok = 0;
    if (has_suffix_ci(path, ".wav")) {
        ok = write_wav_file(path, pcm, n);
    } else {
        ok = write_via_ffmpeg(path, pcm, n);
    }
    return ok;
}

// ===========================================================================
// Temp-file plumbing. The gguf loader mmaps a path and the phonemizer
// reads paths via fopen, so embedded blobs are spilled to temp files
// at startup. gguf keeps its fd after mmap and the phonemizer reads
// its files fully during create, so we unlink immediately after the
// loaders consume them — nothing lingers on disk.
// ===========================================================================
static int spill_temp(const char * tag, const unsigned char * data,
                      size_t n, char * out_path, size_t cap) {
    int ok = 0;
    const char * tmp = getenv("TMPDIR");
    if (tmp == NULL) { tmp = "/tmp"; }
    snprintf(out_path, cap, "%s/kittens_cli_%d_%s", tmp,
             (int)getpid(), tag);
    FILE * f = fopen(out_path, "wb");
    if (f != NULL) {
        size_t wrote = fwrite(data, 1, n, f);
        fclose(f);
        ok = (wrote == n);
    }
    return ok;
}

// ===========================================================================
// Text input + argument parsing.
// ===========================================================================
static char * slurp_file(const char * path) {
    char * out = NULL;
    FILE * f = fopen(path, "rb");
    if (f != NULL) {
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        if (sz >= 0) {
            out = (char *)malloc((size_t)sz + 1);
            if (out != NULL) {
                size_t got = fread(out, 1, (size_t)sz, f);
                out[got] = '\0';
            }
        }
        fclose(f);
    }
    return out;
}

struct args {
    const char * prompt;     // -p
    const char * file;       // -f
    const char * voice;      // -s
    const char * out;        // -o
    float        speed;      // --speed (default 1.0)
    int          list_voices;// --voices
    int          help;       // -h / --help
    int          bad;        // unknown / missing-value flag
};

static struct args parse_args(int argc, char ** argv) {
    struct args a = { NULL, NULL, "Kiki", NULL, 1.0f, 0, 0, 0 };
    int i = 1;
    while (i < argc) {
        const char * f = argv[i];
        int has_next = (i + 1 < argc);
        if (strcmp(f, "-p") == 0 && has_next) {
            a.prompt = argv[++i];
        } else if (strcmp(f, "-f") == 0 && has_next) {
            a.file = argv[++i];
        } else if (strcmp(f, "-s") == 0 && has_next) {
            a.voice = argv[++i];
        } else if (strcmp(f, "-o") == 0 && has_next) {
            a.out = argv[++i];
        } else if (strcmp(f, "--speed") == 0 && has_next) {
            a.speed = (float)atof(argv[++i]);
        } else if (strcmp(f, "--voices") == 0) {
            a.list_voices = 1;
        } else if (strcmp(f, "-h") == 0
                   || strcmp(f, "--help") == 0) {
            a.help = 1;
        } else {
            a.bad = 1;
        }
        i++;
    }
    return a;
}

static void print_usage(const char * argv0) {
    fprintf(stderr,
        "kittens-tts-cli -- offline neural TTS, single binary\n\n"
        "Usage:\n"
        "  %s -p \"Text. Sentences split on . ! ?\\n"
        "Paragraphs split on newline.\" [options]\n"
        "  %s -f input.txt [options]\n"
        "  %s --voices\n\n"
        "Options:\n"
        "  -p TEXT       text to speak (sentences end with . ! ?,\n"
        "                paragraphs separated by a single newline)\n"
        "  -f FILE       read text from FILE instead of -p\n"
        "  -s VOICE      voice name (default Kiki); see --voices\n"
        "  -o FILE       output file. .wav written natively;\n"
        "                any other extension (.mp3/.m4a/...) is\n"
        "                produced by piping through ffmpeg.\n"
        "                If omitted, writes out.wav\n"
        "  --speed X     speech rate multiplier (default 1.0)\n"
        "  --voices      list available voices and exit\n"
        "  -h, --help    this help\n",
        argv0, argv0, argv0);
}

static void print_voices(void) {
    printf("Available voices (use with -s):\n");
    int i = 0;
    while (i < N_VOICES) {
        printf("  %-8s (%s)\n", VOICES[i].friendly,
               VOICES[i].internal);
        i++;
    }
}

// ===========================================================================
// run -- all driver work. Single entry, single exit (rc).
// ===========================================================================
static int run(int argc, char ** argv, char ** env) {
    (void)env;
    struct args a = parse_args(argc, argv);
    int rc = 0;
    if (a.help) {
        print_usage(argv[0]);
    } else if (a.list_voices) {
        print_voices();
    } else if (a.bad || (a.prompt == NULL && a.file == NULL)) {
        print_usage(argv[0]);
        rc = 2;
    } else {
        const struct voice_entry * ve = voice_lookup(a.voice);
        char * text = (a.file != NULL) ? slurp_file(a.file) : NULL;
        const char * prompt = (a.file != NULL) ? text : a.prompt;
        if (ve == NULL) {
            fprintf(stderr, "unknown voice '%s' (try --voices)\n",
                    a.voice);
            rc = 2;
        } else if (prompt == NULL) {
            fprintf(stderr, "could not read input '%s'\n",
                    a.file ? a.file : "(none)");
            rc = 1;
        } else {
            symbol_map_build();
            // Spill the phonemizer data + model to temp files, load,
            // then unlink (loaders retain what they need).
            char p_rules[1024], p_list[1024], p_gguf[1024];
            int spilled =
                spill_temp("en_rules", EMB_RULES, sizeof EMB_RULES,
                           p_rules, sizeof p_rules)
                && spill_temp("en_list", EMB_LIST, sizeof EMB_LIST,
                              p_list, sizeof p_list)
                && spill_temp("model.gguf", EMB_GGUF, sizeof EMB_GGUF,
                              p_gguf, sizeof p_gguf);
            PhonemizerHandle phon = spilled
                ? phonemizer_create(p_rules, p_list, "en-us") : NULL;
            struct kittens_ctx * tts = spilled
                ? kittens_create(p_gguf) : NULL;
            unlink(p_rules); unlink(p_list); unlink(p_gguf);
            const float * voice = voice_embedding(ve->internal);
            if (!spilled || phon == NULL || tts == NULL
                || voice == NULL) {
                fprintf(stderr, "init failed (phonemizer=%p "
                        "model=%p voice=%p)\n",
                        (void *)phon, (void *)tts, (void *)voice);
                rc = 1;
            } else {
                struct synth_ctx s = {
                    tts, phon, voice, a.speed * ve->speed_prior };
                struct audio_buf out = { NULL, 0, 0 };
                synth_document(&s, prompt, &out);
                const char * dst = (a.out != NULL) ? a.out : "out.wav";
                if (out.n == 0) {
                    fprintf(stderr, "no audio produced\n");
                    rc = 1;
                } else if (!write_output(dst, out.data, out.n)) {
                    fprintf(stderr, "failed to write '%s'\n", dst);
                    rc = 1;
                } else {
                    fprintf(stderr,
                        "wrote %s  (%.2fs, voice %s, speed %.2f)\n",
                        dst, (double)out.n / SAMPLE_RATE,
                        ve->friendly, (double)a.speed);
                }
                free(out.data);
            }
            if (tts != NULL) { kittens_destroy(tts); }
            if (phon != NULL) { phonemizer_destroy(phon); }
        }
        free(text);
    }
    return rc;
}

#ifdef KITTENS_TTS_CLI_MAIN
int main(int argc, char ** argv, char ** env) {
    return run(argc, argv, env);
}
#endif
