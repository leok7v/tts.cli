// phonemizer.c -- pure C99 port of the original C++ phonemizer.
// The .cpp original + the port plan + the proofdiff pair map are
// archived under rnd/archive/phonemizer/; this file is the
// production engine compiled into KittensCPU. The .cpp/.c byte-
// equal parity harness still runs via `make compare` against the
// archived sources.
//
// Single-file library: all primitives (arr / chars / map) are
// inlined at the top of this TU; no #include of maps.c / chars.c
// / arrays.c. The Swift bridge reaches phonemizer_create /
// _destroy / _phonemize / _free_string / _get_error via the
// declarations in app/bridge.h, the only consumer.

#ifndef PHONEMIZER_C
#define PHONEMIZER_C

#include <assert.h>
#include <ctype.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Inlined primitives. The originals live in node.c.dev (arrays.c +
// chars.c + maps.c). We carry only the symbols phonemizer.c uses,
// pruned of dead code (no printf, no for_each, no int-keyed
// accessors). The full versions are upstream if you want them.
// ---------------------------------------------------------------------------

struct arr {
    void * data;
    size_t count;
    size_t capacity;
};

static inline void * oom(void * a) {
    if (a == NULL) { fprintf(stderr, "OOM"); abort(); }
    return a;
}

static inline void arr_grow(struct arr * a, size_t esize, size_t need) {
    if (a->data == NULL) {
        a->capacity = need;
        a->data = oom(malloc(need * esize));
    } else if (need > a->capacity) {
        a->capacity = need * 2;
        a->data = oom(realloc(a->data, a->capacity * esize));
    }
}

#define define_array(T, name)                                          \
struct name { T * data; size_t count; size_t capacity; };              \
static inline void name##_grow(struct name * a, size_t need) {         \
    arr_grow((struct arr *)a, sizeof(T), need);                        \
}                                                                      \
static inline void name##_put(struct name * a, T v) {                  \
    name##_grow(a, a->count + 1);                                      \
    a->data[a->count++] = v;                                           \
}                                                                      \
static inline void name##_free(struct name * a) {                      \
    free(a->data);                                                     \
    a->data = NULL;                                                    \
    a->count = 0;                                                      \
    a->capacity = 0;                                                   \
}                                                                      \
struct name##_swallow_semicolon

struct chars { // always zero terminated array of bytes
    char * data;
    size_t count;
    size_t capacity;
};

static inline void chars_grow(struct chars * s, size_t need) {
    arr_grow((struct arr *)s, 1, need);
}

static inline void chars_put(struct chars * s, const char * d,
                             size_t count) {
    chars_grow(s, s->count + count + 1);
    if (s->data) {
        memcpy(s->data + s->count, d, count);
        s->count += count;
        s->data[s->count] = '\0';
    }
}

static inline void chars_free(struct chars * s) {
    free(s->data);
    s->data = NULL;
    s->count = 0;
    s->capacity = 0;
}

static inline void chars_puts(struct chars * s, const char * a) {
    chars_put(s, a, strlen(a));
}

enum map_key { MAP_KEY_INT, MAP_KEY_CHARS };

#define MAP_EMPTY     0
#define MAP_LIVE      1
#define MAP_TOMBSTONE 2

struct map {
    uint8_t *           states;
    void *              keys;
    void *              values;
    size_t              count;
    size_t              capacity;
    size_t              key_size;
    size_t              value_size;
    enum map_key        key_kind;
    void              (*value_free)(void *);
};

static inline uint64_t map_hash_i(int64_t k) {
    uint64_t x = (uint64_t)k;
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

static inline uint64_t map_hash_s(const struct chars * k) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < k->count; i++) {
        h ^= (uint8_t)k->data[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

static inline uint64_t map_hash(const struct map * m, const void * k) {
    uint64_t h = 0;
    if (m->key_kind == MAP_KEY_INT) {
        h = map_hash_i(*(const int64_t *)k);
    } else {
        h = map_hash_s((const struct chars *)k);
    }
    return h;
}

static inline int map_eq(const struct map * m, const void * a,
                         const void * b) {
    int r = 0;
    if (m->key_kind == MAP_KEY_INT) {
        r = *(const int64_t *)a == *(const int64_t *)b;
    } else {
        const struct chars * x = a;
        const struct chars * y = b;
        r = x->count == y->count
            && memcmp(x->data, y->data, x->count) == 0;
    }
    return r;
}

static inline void * map_k(struct map * m, size_t i) {
    return (char *)m->keys + i * m->key_size;
}

static inline void * map_v(struct map * m, size_t i) {
    return (char *)m->values + i * m->value_size;
}

static inline void map_key_copy(struct map * m, void * dst,
                                const void * src) {
    if (m->key_kind == MAP_KEY_INT) {
        memcpy(dst, src, sizeof(int64_t));
    } else {
        const struct chars * s = src;
        struct chars * d = dst;
        *d = (struct chars){0};
        chars_put(d, s->data, s->count);
    }
}

static inline void map_key_free(struct map * m, void * k) {
    if (m->key_kind == MAP_KEY_CHARS) { chars_free(k); }
}

static void map_init(struct map * m, enum map_key kk,
                     size_t ks, size_t vs, void (*vf)(void *)) {
    m->states = NULL;
    m->keys = NULL;
    m->values = NULL;
    m->count = 0;
    m->capacity = 0;
    m->key_size = ks;
    m->value_size = vs;
    m->key_kind = kk;
    m->value_free = vf;
}

static void map_grow(struct map * m, size_t new_cap) {
    uint8_t * ns = oom(calloc(new_cap, 1));
    void * nk = oom(malloc(new_cap * m->key_size));
    void * nv = oom(malloc(new_cap * m->value_size));
    size_t mask = new_cap - 1;
    for (size_t i = 0; i < m->capacity; i++) {
        if (m->states[i] == MAP_LIVE) {
            void * ok = map_k(m, i);
            size_t j = (size_t)map_hash(m, ok) & mask;
            while (ns[j] != MAP_EMPTY) { j = (j + 1) & mask; }
            ns[j] = MAP_LIVE;
            memcpy((char *)nk + j * m->key_size, ok, m->key_size);
            memcpy((char *)nv + j * m->value_size,
                   map_v(m, i), m->value_size);
        }
    }
    free(m->states);
    free(m->keys);
    free(m->values);
    m->states = ns;
    m->keys = nk;
    m->values = nv;
    m->capacity = new_cap;
}

static void * map_put(struct map * m, const void * k, const void * v) {
    void * r = NULL;
    if (m->capacity == 0) {
        map_grow(m, 16);
    } else if ((m->count + 1) * 4 > m->capacity * 3) {
        map_grow(m, m->capacity * 2);
    }
    size_t mask = m->capacity - 1;
    size_t j = (size_t)map_hash(m, k) & mask;
    size_t tomb = (size_t)-1;
    int done = 0;
    while (!done && m->states[j] != MAP_EMPTY) {
        if (m->states[j] == MAP_TOMBSTONE) {
            if (tomb == (size_t)-1) { tomb = j; }
            j = (j + 1) & mask;
        } else if (map_eq(m, map_k(m, j), k)) {
            void * vs = map_v(m, j);
            if (m->value_free) { m->value_free(vs); }
            memcpy(vs, v, m->value_size);
            r = vs;
            done = 1;
        } else {
            j = (j + 1) & mask;
        }
    }
    if (!done) {
        if (tomb != (size_t)-1) { j = tomb; }
        m->states[j] = MAP_LIVE;
        map_key_copy(m, map_k(m, j), k);
        memcpy(map_v(m, j), v, m->value_size);
        m->count++;
        r = map_v(m, j);
    }
    return r;
}

static void * map_get(struct map * m, const void * k) {
    void * r = NULL;
    if (m->capacity > 0) {
        size_t mask = m->capacity - 1;
        size_t j = (size_t)map_hash(m, k) & mask;
        int done = 0;
        while (!done && m->states[j] != MAP_EMPTY) {
            if (m->states[j] == MAP_LIVE
                && map_eq(m, map_k(m, j), k)) {
                r = map_v(m, j);
                done = 1;
            } else {
                j = (j + 1) & mask;
            }
        }
    }
    return r;
}

static void map_remove(struct map * m, const void * k) {
    if (m->capacity > 0) {
        size_t mask = m->capacity - 1;
        size_t j = (size_t)map_hash(m, k) & mask;
        int done = 0;
        while (!done && m->states[j] != MAP_EMPTY) {
            if (m->states[j] == MAP_LIVE
                && map_eq(m, map_k(m, j), k)) {
                if (m->value_free) { m->value_free(map_v(m, j)); }
                map_key_free(m, map_k(m, j));
                m->states[j] = MAP_TOMBSTONE;
                m->count--;
                done = 1;
            } else {
                j = (j + 1) & mask;
            }
        }
    }
}

static void map_free(struct map * m) {
    if (m->capacity > 0) {
        for (size_t i = 0; i < m->capacity; i++) {
            if (m->states[i] == MAP_LIVE) {
                if (m->value_free) { m->value_free(map_v(m, i)); }
                map_key_free(m, map_k(m, i));
            }
        }
        free(m->states);
        free(m->keys);
        free(m->values);
    }
    m->states = NULL;
    m->keys = NULL;
    m->values = NULL;
    m->count = 0;
    m->capacity = 0;
}

static inline void chars_free_v(void * s) {
    chars_free((struct chars *)s);
}

// ---------------------------------------------------------------------------
// Small project-local helpers used throughout (extensions to chars.c).
// ---------------------------------------------------------------------------

// Append a single byte to a chars. Wraps the (data, count=1) form of
// chars_put. Used by every per-byte accumulator (split_ws, tokenize,
// the UTF-8 walk, the abbreviation chain).
static inline void chars_put_byte(struct chars * s, char c) {
    chars_put(s, &c, 1);
}

// ---------------------------------------------------------------------------
// Leaf helpers (no state). Mirrors the file-scope static helpers at the
// top of phonemizer.cpp.
// ---------------------------------------------------------------------------

static bool is_vowel_letter(char c) {
    c = (char)tolower((unsigned char)c);
    return c == 'a' || c == 'e' || c == 'i' || c == 'o'
        || c == 'u' || c == 'y';
}

static bool has_any_vowel_letter(const char * s, size_t n) {
    bool found = false;
    for (size_t i = 0; i < n; i++) {
        if (!found && is_vowel_letter(s[i])) { found = true; }
    }
    return found;
}

// True iff `s` contains any phoneme vowel-code char (a/A/e/E/i/I/o/O/
// u/U/V/0/3/@). Used to validate that stem phonemes have a syllable.
static bool has_any_vowel_code(const char * s, size_t n) {
    static const char vc[] = "aAeEiIoOuUV03@";
    bool found = false;
    for (size_t i = 0; i < n; i++) {
        if (!found && strchr(vc, s[i]) != NULL) { found = true; }
    }
    return found;
}

// Lowercase `s[0..n)` into `*out`. Caller-owned buffer; reset on entry.
static void to_lower(const char * s, size_t n, struct chars * out) {
    out->count = 0;
    chars_grow(out, n + 1);
    for (size_t i = 0; i < n; i++) {
        out->data[i] = (char)tolower((unsigned char)s[i]);
    }
    out->count = n;
    out->data[n] = '\0';
}

// Strip leading/trailing ASCII whitespace; write the trimmed view to
// `*out`. If `s[0..n)` is all whitespace, `*out` is left empty.
static void trim(const char * s, size_t n, struct chars * out) {
    out->count = 0;
    size_t start = 0;
    while (start < n
           && (s[start] == ' ' || s[start] == '\t'
               || s[start] == '\r' || s[start] == '\n')) {
        start++;
    }
    size_t end = n;
    while (end > start
           && (s[end - 1] == ' ' || s[end - 1] == '\t'
               || s[end - 1] == '\r' || s[end - 1] == '\n')) {
        end--;
    }
    if (end > start) {
        chars_put(out, s + start, end - start);
    }
}

// Replace the first occurrence of `from` in `s` with `to`; no-op when
// `from` is absent. SESE search loop; `i == s->count` means not found.
static void replace_first_char(struct chars * s, char from, char to) {
    size_t i = 0;
    while (i < s->count && s->data[i] != from) { i++; }
    if (i < s->count) { s->data[i] = to; }
}

// Parse "$1".."$6" stress-position flag. Returns 0 when `flag` is not
// in that form. `len` is the byte length of the flag (no NUL needed).
static int parse_stress_n(const char * flag, size_t len) {
    int n = 0;
    if (len == 2 && flag[0] == '$'
        && flag[1] >= '1' && flag[1] <= '6') {
        n = flag[1] - '0';
    }
    return n;
}

// ---------------------------------------------------------------------------
// String-array container (vector<std::string> counterpart).
// ---------------------------------------------------------------------------

define_array(struct chars, charsv);

// Walks each element, frees its chars heap, then frees the array
// storage. Inverse of charsv_put + the chars_put calls done by
// split_ws.
static void charsv_clear(struct charsv * arr) {
    for (size_t i = 0; i < arr->count; i++) {
        chars_free(&arr->data[i]);
    }
    charsv_free(arr);
}

// ---------------------------------------------------------------------------
// Per-entry flag bundle for load_dictionary (the en_list reader).
// Populated by parse_entry_flags from parts[2..] + phonemes_str.
// Mirrors `struct EntryFlags` in phonemizer.h.
// ---------------------------------------------------------------------------

struct entry_flags {
    bool noun, verb, past;
    bool pastf, nounf, verbf;
    bool atend, capital, atstart;
    bool onlys, only;
    bool grammar;
    bool strend2, u2;
    int  stress_n;
};

// ---------------------------------------------------------------------------
// File-scope parse helpers (used by the loaders in phase D-7+).
// ---------------------------------------------------------------------------

// Whitespace-tokenize `s[0..n)` into `*out`. Mirrors both the
// `splitWS` (std::istringstream) and `tokenizeRuleLine` (char-by-char)
// helpers in phonemizer.cpp -- behaviourally identical for the inputs
// the loaders feed (no embedded NULs, standard ASCII whitespace).
// Caller-owned out; charsv_clear runs at entry to drop any previous
// contents.
static void split_ws(const char * s, size_t n, struct charsv * out) {
    charsv_clear(out);
    struct chars tok = {0};
    for (size_t i = 0; i < n; i++) {
        if (isspace((unsigned char)s[i])) {
            if (tok.count > 0) {
                charsv_put(out, tok);
                tok = (struct chars){0};
            }
        } else {
            chars_put_byte(&tok, s[i]);
        }
    }
    if (tok.count > 0) {
        charsv_put(out, tok);
    }
}

// Strip leading "?N" / "?!N" dialect prefix from `s[0..n)`. On match,
// sets *cond / *negated and writes the trimmed remainder to *out.
// On no match, *cond = 0, *negated = false, and *out gets a copy of
// the original line. Caller owns out (charsv_clear-style reset at
// entry).
static void parse_leading_dialect(const char * s, size_t n,
                                  int * cond, bool * negated,
                                  struct chars * out) {
    *cond = 0;
    *negated = false;
    out->count = 0;
    if (out->data != NULL) { out->data[0] = '\0'; }
    bool matched = false;
    if (n > 0 && s[0] == '?') {
        size_t space = 0;
        while (space < n && s[space] != ' ' && s[space] != '\t') {
            space++;
        }
        if (space < n) {
            size_t cs = 1;
            size_t ce = space;
            if (cs < ce && s[cs] == '!') {
                *negated = true;
                cs++;
            }
            int v = 0;
            bool any_digit = false;
            while (cs < ce && isdigit((unsigned char)s[cs])) {
                v = v * 10 + (s[cs] - '0');
                cs++;
                any_digit = true;
            }
            // .cpp swallows std::stoi throws (cond stays 0) but
            // ALWAYS strips the "?<x>" prefix. Match that: gate the
            // *cond assignment on a valid digit run, but always trim
            // and emit the remainder.
            if (any_digit && cs == ce) { *cond = v; }
            trim(s + space, n - space, out);
            matched = true;
        }
    }
    if (!matched) {
        chars_put(out, s, n);
    }
}

// Strip trailing "//" comment + trim whitespace. Caller-owned out.
static void strip_comment_and_trim(const char * raw, size_t n,
                                   struct chars * out) {
    size_t cut = n;
    for (size_t i = 0; i + 1 < n; i++) {
        if (cut == n && raw[i] == '/' && raw[i + 1] == '/') {
            cut = i;
        }
    }
    trim(raw, cut, out);
}

// ---------------------------------------------------------------------------
// Phoneme-code utilities (no state). Read-only inspection of phoneme
// strings; used by suffix / prefix decision helpers in later phases.
// ---------------------------------------------------------------------------

// True iff `code[0..n)` starts with a vowel phoneme code. Multi-char
// codes (e.g. "eI", "aU") are classified by their first byte; this
// matches the C++ original which only looks at code[0].
static bool is_vowel_code(const char * code, size_t n) {
    bool r = false;
    if (n > 0) {
        char c = code[0];
        r = c == '@' || c == 'a' || c == 'A' || c == 'E' || c == 'I'
         || c == 'i' || c == 'O' || c == 'U' || c == 'V' || c == '0'
         || c == '3' || c == 'e' || c == 'o' || c == 'u';
    }
    return r;
}

// Count vowel-letter groups in `s[0..n)` (consecutive vowels = 1
// group). A silent trailing magic-e ('e' after a non-vowel) doesn't
// add a syllable. Matches countSuffixSyllables in phonemizer.cpp.
static int count_suffix_syllables(const char * s, size_t n) {
    static const char vow_let[] = "aeiouAEIOU";
    int r = 0;
    bool in_vowel = false;
    for (size_t i = 0; i < n; i++) {
        if (strchr(vow_let, s[i]) != NULL) {
            if (!in_vowel) { r++; in_vowel = true; }
        } else {
            in_vowel = false;
        }
    }
    if (n > 0 && (s[n - 1] == 'e' || s[n - 1] == 'E')
        && n >= 2 && strchr(vow_let, s[n - 2]) == NULL) {
        r = r - 1 > 1 ? r - 1 : 1;
    }
    return r;
}

// Multi-char phoneme codes considered to contain a "full" (stressable
// or schwa-final) vowel. Shared by prefix_has_full_vowel, count_prefix_
// vowels, prefix_ends_in_schwa (mirrors the three local-static tables
// in phonemizer.cpp; merged here because they're byte-identical).
static const char * const vowel_codes_multi[] = {
    "aI@3", "aU@r", "i@3r",
    "aI@",  "aI3",  "aU@",  "i@3", "3:r", "A:r",
    "o@r",  "A@r",  "e@r",
    "eI", "aI", "aU", "OI", "oU", "IR", "VR", "U@",
    "A@", "e@", "i@", "O@", "o@", "3:", "A:", "i:", "u:",
    "O:", "e:", "a:",
    "aa", "@L", "@2", "@5", "I2", "I#", "E2", "E#", "e#",
    "a#", "a2", "0#", "02", "O2", "A#",
    NULL
};

// True iff `phonemes[0..n)` contains any "full" (stressable) vowel.
// Local table here differs from vowel_codes_multi: it's the
// fully-stressable subset (no "@" schwa series).
static bool prefix_has_full_vowel(const char * phonemes, size_t n) {
    static const char * const full_vowels[] = {
        "eI", "aI", "aU", "OI", "oU", "3:", "A:", "i:", "u:",
        "O:", "e:", "a:",
        "aI@", "aU@", "oU#", "i@", "e@", "A@", "U@", "O@",
        "a", "A", "E", "V", "0", "o",
        NULL
    };
    bool found = false;
    for (size_t pi = 0; pi < n && !found; pi++) {
        for (int fi = 0; full_vowels[fi] != NULL && !found; fi++) {
            const char * fv = full_vowels[fi];
            size_t fvlen = strlen(fv);
            found = pi + fvlen <= n && memcmp(phonemes + pi, fv, fvlen) == 0;
        }
    }
    return found;
}

// Walk `phonemes[0..n)` as a sequence of phoneme codes (multi-char
// first, single-char fallback) and return the vowel-code count.
static int count_prefix_vowels(const char * phonemes, size_t n) {
    int r = 0;
    size_t pi = 0;
    while (pi < n) {
        char c = phonemes[pi];
        if (c == '\'' || c == ',' || c == '%' || c == '=') {
            pi++;
        } else {
            bool mch = false;
            for (int mi = 0; vowel_codes_multi[mi] != NULL && !mch; mi++) {
                const char * mc = vowel_codes_multi[mi];
                size_t ml = strlen(mc);
                mch = pi + ml <= n && memcmp(phonemes + pi, mc, ml) == 0;
                if (mch) {
                    if (is_vowel_code(mc, ml)) { r++; }
                    pi += ml;
                }
            }
            if (!mch) {
                if (is_vowel_code(&phonemes[pi], 1)) { r++; }
                pi++;
            }
        }
    }
    return r;
}

// True iff the LAST vowel phoneme code in `phonemes[0..n)` is a schwa-
// type (3, 3:, @, @2, @5, @L, I2, I#, a#). Used by the prefix-stress
// heuristic ("super" + "nova/market" -> keep prefix primary).
static bool prefix_ends_in_schwa(const char * phonemes, size_t n) {
    const char * last_v = NULL;
    size_t last_v_len = 0;
    size_t pi = 0;
    while (pi < n) {
        char c = phonemes[pi];
        if (c == '\'' || c == ',' || c == '%' || c == '=') {
            pi++;
        } else {
            bool mch = false;
            for (int mi = 0; vowel_codes_multi[mi] != NULL && !mch; mi++) {
                const char * mc = vowel_codes_multi[mi];
                size_t ml = strlen(mc);
                mch = pi + ml <= n && memcmp(phonemes + pi, mc, ml) == 0;
                if (mch) {
                    if (is_vowel_code(mc, ml)) {
                        last_v = mc;
                        last_v_len = ml;
                    }
                    pi += ml;
                }
            }
            if (!mch) {
                if (is_vowel_code(&phonemes[pi], 1)) {
                    last_v = &phonemes[pi];
                    last_v_len = 1;
                }
                pi++;
            }
        }
    }
    bool r = false;
    if (last_v != NULL) {
        r = (last_v_len == 1 && last_v[0] == '3')
         || (last_v_len == 2 && last_v[0] == '3' && last_v[1] == ':')
         || (last_v_len == 1 && last_v[0] == '@')
         || (last_v_len == 2 && last_v[0] == '@'
             && (last_v[1] == '2' || last_v[1] == '5' || last_v[1] == 'L'))
         || (last_v_len == 2 && last_v[0] == 'I'
             && (last_v[1] == '2' || last_v[1] == '#'))
         || (last_v_len == 2 && last_v[0] == 'a' && last_v[1] == '#');
    }
    return r;
}

// ---------------------------------------------------------------------------
// Map-value support: strpair, free-wrappers, init shorthands.
// ---------------------------------------------------------------------------

// Two-string value type for phrase_split_dict (the one map whose value
// is a `pair<string, string>` in the C++ original).
struct strpair {
    struct chars a;
    struct chars b;
};

// Adapter: maps.c value_free is `void(*)(void*)`. `chars_free_v`
// already exists in maps.c (typed wrapper around chars_free); add
// the matching wrapper for the only project-local value type.
static void strpair_free_v(void * v) {
    struct strpair * p = v;
    chars_free(&p->a);
    chars_free(&p->b);
}

// vector<pair<string,string>> -> struct strpairs (define_array gives
// data/count/capacity + put/grow/free; per-element heap cleanup needs
// the explicit walk in strpairs_clear).
define_array(struct strpair, strpairs);

static void strpairs_clear(struct strpairs * arr) {
    for (size_t i = 0; i < arr->count; i++) {
        chars_free(&arr->data[i].a);
        chars_free(&arr->data[i].b);
    }
    strpairs_free(arr);
}

// Map init shorthands. All keys are chars (MAP_KEY_CHARS); value
// shape varies:
//   smap = unordered_map<string, string>
//   set  = unordered_set<string>   (value is a 1-byte presence flag)
//   imap = unordered_map<string, int>
//   pmap = unordered_map<string, pair<string, string>>

static void smap_init(struct map * m) {
    map_init(m, MAP_KEY_CHARS, sizeof(struct chars),
             sizeof(struct chars), chars_free_v);
}

static void set_init(struct map * m) {
    map_init(m, MAP_KEY_CHARS, sizeof(struct chars), 1, NULL);
}

static void imap_init(struct map * m) {
    map_init(m, MAP_KEY_CHARS, sizeof(struct chars), sizeof(int), NULL);
}

static void pmap_init(struct map * m) {
    map_init(m, MAP_KEY_CHARS, sizeof(struct chars),
             sizeof(struct strpair), strpair_free_v);
}

// ---------------------------------------------------------------------------
// State: struct phonemizer
// ---------------------------------------------------------------------------
//
// Mirrors the private members of `class IPAPhonemizer` (see
// phonemizer.h). Field comments are short — see the C++ header for
// the full intent of each map / set. Maps are zero-init friendly;
// phonemizer_state_init below wires the key_kind / value_size /
// value_free for each one before the first put / get.

// ---------------------------------------------------------------------------
// Rule-parser structures (port of rule_parser.h).
// ---------------------------------------------------------------------------

// Letter-class bitsets indexed by char. Mirrors the std::set<char>
// groupA..groupK in rule_parser.h. plus a user-defined L-groups
// table. SetLetterBits semantics from tr_languages.c.
struct letter_groups {
    bool groupA[256];  // vowels: aeiou
    bool groupB[256];  // hard consonants: bcdfgjklmnpqstvxz
    bool groupC[256];  // all consonants: bcdfghjklmnpqrstvwxz
    bool groupF[256];  // voiceless: cfhkpqstx
    bool groupG[256];  // voiced: bdgjlmnrvwyz
    bool groupH[256];  // sonorants: hlmnr
    bool groupY[256];  // front vowels for English: aeiouy
    bool groupK[256];  // non-vowels: bcdfghjklmnpqrstvwxyz
    struct charsv lgroups[100];  // L01..L99 user-defined groups
};

// A single phoneme rule (one line from the en_rules file).
struct phoneme_rule {
    int             condition;          // 0=always, 3=en-us, ...
    bool            condition_negated;
    struct chars    left_ctx;
    struct chars    match;
    struct chars    right_ctx;
    struct chars    phonemes;
    int             del_fwd;
    bool            is_prefix;
    bool            is_suffix;
    int             suffix_strip_len;
    int             suffix_flags;
};

define_array(struct phoneme_rule, rules);

// Free heap inside every rule then free the array storage.
static void rules_clear(struct rules * arr) {
    for (size_t i = 0; i < arr->count; i++) {
        struct phoneme_rule * r = &arr->data[i];
        chars_free(&r->left_ctx);
        chars_free(&r->match);
        chars_free(&r->right_ctx);
        chars_free(&r->phonemes);
    }
    rules_free(arr);
}

static void rules_free_v(void * v) { rules_clear((struct rules *)v); }

struct replace_rule {
    struct chars from;
    struct chars to;
};

define_array(struct replace_rule, replaces);

static void replaces_clear(struct replaces * arr) {
    for (size_t i = 0; i < arr->count; i++) {
        chars_free(&arr->data[i].from);
        chars_free(&arr->data[i].to);
    }
    replaces_free(arr);
}

struct ruleset {
    struct letter_groups groups;
    struct replaces      replacements;
    struct map           rule_groups; // chars -> struct rules
};

// SetLetterBits equivalents. Sets `mask[(unsigned char)c]` for each
// byte of `letters` (NUL-terminated literal).
static void letter_bits(bool mask[256], const char * letters) {
    for (size_t i = 0; letters[i] != '\0'; i++) {
        mask[(unsigned char)letters[i]] = true;
    }
}

// Initialise the ruleset to its default (empty) state. Letter groups
// follow SetLetterBits() from tr_languages.c.
static void ruleset_init(struct ruleset * rs) {
    memset(&rs->groups, 0, sizeof(rs->groups));
    letter_bits(rs->groups.groupA, "aeiou");
    letter_bits(rs->groups.groupB, "bcdfgjklmnpqstvxz");
    letter_bits(rs->groups.groupC, "bcdfghjklmnpqrstvwxz");
    letter_bits(rs->groups.groupF, "cfhkpqstx");
    letter_bits(rs->groups.groupG, "bdgjlmnrvwyz");
    letter_bits(rs->groups.groupH, "hlmnr");
    letter_bits(rs->groups.groupY, "aeiouy");
    letter_bits(rs->groups.groupK, "bcdfghjklmnpqrstvwxyz");
    rs->replacements = (struct replaces){0};
    map_init(&rs->rule_groups, MAP_KEY_CHARS,
             sizeof(struct chars), sizeof(struct rules), rules_free_v);
}

static void ruleset_free(struct ruleset * rs) {
    for (size_t i = 0; i < 100; i++) {
        charsv_clear(&rs->groups.lgroups[i]);
    }
    replaces_clear(&rs->replacements);
    map_free(&rs->rule_groups);
}

struct phonemizer {
    char err[256];
    struct chars dialect;       // "en-us" or "en-gb"
    bool         loaded;

    // String -> string dictionaries (10).
    struct map dict;            // word -> raw phoneme code string
    struct map verb_dict;       // $verb pronunciation
    struct map past_dict;       // $past pronunciation
    struct map noun_dict;       // $noun pronunciation
    struct map ipa_overrides;   // explicit IPA overrides
    struct map atstart_dict;    // $atstart
    struct map atend_dict;      // $atend
    struct map capital_dict;    // $capital
    struct map onlys_bare_dict; // $onlys bare-word override
    struct map phrase_dict;     // bigram phrase -> phoneme string

    // String sets (15). Presence-only; value_size = 1.
    struct map pastf_words;             // $pastf
    struct map nounf_words;             // $nounf
    struct map verbf_words;             // $verbf
    struct map unstressed_words;        // $u
    struct map unstressend_words;       // $unstressend
    struct map abbrev_words;            // $abbrev
    struct map onlys_words;             // $onlys
    struct map only_words;              // $only
    struct map noun_form_stress;        // $N $onlys (noun-form stress)
    struct map verb_flag_words;         // bare $verb flag
    struct map strend_words;            // $strend2 + bare phoneme
    struct map u2_strend2_words;        // $u2 + $strend2
    struct map comma_strend2_words;     // $strend2 + comma-prefix phoneme
    struct map u_plus_secondary_words;  // $u+ + secondary stress
    struct map keep_sec_phrase_keys;    // $u2+ on phrase entries

    // String -> int maps (2).
    struct map stress_pos;        // $N flag -> 1..6
    struct map word_alt_flags;    // $altN bitmask

    // String -> pair<string, string> map (1).
    struct map phrase_split_dict; // (a||b) split-phrase entries

    // vector<pair<string,string>> sorted by length desc (1).
    struct strpairs compound_prefixes;

    // Rule set: letter groups, replacements, rule_groups map.
    struct ruleset rules;
};

// Initialise all maps to their correct (key_kind, value_size,
// value_free) triples. Idempotent on already-zeroed memory; safe
// to call once per phonemizer instance before any put / get.
static void phonemizer_state_init(struct phonemizer * p) {
    smap_init(&p->dict);
    smap_init(&p->verb_dict);
    smap_init(&p->past_dict);
    smap_init(&p->noun_dict);
    smap_init(&p->ipa_overrides);
    smap_init(&p->atstart_dict);
    smap_init(&p->atend_dict);
    smap_init(&p->capital_dict);
    smap_init(&p->onlys_bare_dict);
    smap_init(&p->phrase_dict);
    set_init(&p->pastf_words);
    set_init(&p->nounf_words);
    set_init(&p->verbf_words);
    set_init(&p->unstressed_words);
    set_init(&p->unstressend_words);
    set_init(&p->abbrev_words);
    set_init(&p->onlys_words);
    set_init(&p->only_words);
    set_init(&p->noun_form_stress);
    set_init(&p->verb_flag_words);
    set_init(&p->strend_words);
    set_init(&p->u2_strend2_words);
    set_init(&p->comma_strend2_words);
    set_init(&p->u_plus_secondary_words);
    set_init(&p->keep_sec_phrase_keys);
    imap_init(&p->stress_pos);
    imap_init(&p->word_alt_flags);
    pmap_init(&p->phrase_split_dict);
    ruleset_init(&p->rules);
}

// Free every owned heap reachable from `*p`. Doesn't free `p` itself
// (caller owns the storage).
static void phonemizer_state_free(struct phonemizer * p) {
    chars_free(&p->dialect);
    map_free(&p->dict);
    map_free(&p->verb_dict);
    map_free(&p->past_dict);
    map_free(&p->noun_dict);
    map_free(&p->ipa_overrides);
    map_free(&p->atstart_dict);
    map_free(&p->atend_dict);
    map_free(&p->capital_dict);
    map_free(&p->onlys_bare_dict);
    map_free(&p->phrase_dict);
    map_free(&p->pastf_words);
    map_free(&p->nounf_words);
    map_free(&p->verbf_words);
    map_free(&p->unstressed_words);
    map_free(&p->unstressend_words);
    map_free(&p->abbrev_words);
    map_free(&p->onlys_words);
    map_free(&p->only_words);
    map_free(&p->noun_form_stress);
    map_free(&p->verb_flag_words);
    map_free(&p->strend_words);
    map_free(&p->u2_strend2_words);
    map_free(&p->comma_strend2_words);
    map_free(&p->u_plus_secondary_words);
    map_free(&p->keep_sec_phrase_keys);
    map_free(&p->stress_pos);
    map_free(&p->word_alt_flags);
    map_free(&p->phrase_split_dict);
    strpairs_clear(&p->compound_prefixes);
    ruleset_free(&p->rules);
}

// ---------------------------------------------------------------------------
// State-map convenience helpers (used everywhere from D-5 onward).
// `chars_view` builds a non-owning struct chars over `(k, kn)` so we
// can call map_get / map_put without heap-allocating a key buffer.
// map_put deep-copies the key; the view's data pointer never escapes.
// ---------------------------------------------------------------------------

static inline struct chars chars_view(const char * k, size_t kn) {
    struct chars v = { .data = (char *)k, .count = kn, .capacity = 0 };
    return v;
}

// Set: idempotent presence insert / remove / has.
static void set_add(struct map * s, const char * k, size_t kn) {
    struct chars view = chars_view(k, kn);
    char one = 1;
    map_put(s, &view, &one);
}

static bool set_has(struct map * s, const char * k, size_t kn) {
    struct chars view = chars_view(k, kn);
    return map_get(s, &view) != NULL;
}

// String map: overwrite, insert-if-absent, lookup, remove.
static void smap_set(struct map * m, const char * k, size_t kn,
                     const char * v, size_t vn) {
    struct chars view = chars_view(k, kn);
    struct chars tmp = {0};
    chars_put(&tmp, v, vn);
    map_put(m, &view, &tmp);
}

static bool smap_emplace(struct map * m, const char * k, size_t kn,
                         const char * v, size_t vn) {
    struct chars view = chars_view(k, kn);
    bool inserted = false;
    if (map_get(m, &view) == NULL) {
        struct chars tmp = {0};
        chars_put(&tmp, v, vn);
        map_put(m, &view, &tmp);
        inserted = true;
    }
    return inserted;
}

static struct chars * smap_get(struct map * m, const char * k, size_t kn) {
    struct chars view = chars_view(k, kn);
    return map_get(m, &view);
}

static void smap_erase(struct map * m, const char * k, size_t kn) {
    struct chars view = chars_view(k, kn);
    map_remove(m, &view);
}

// Int map: overwrite, insert-if-absent, lookup, OR-into.
static void imap_set(struct map * m, const char * k, size_t kn, int v) {
    struct chars view = chars_view(k, kn);
    map_put(m, &view, &v);
}

static bool imap_emplace(struct map * m, const char * k, size_t kn, int v) {
    struct chars view = chars_view(k, kn);
    bool inserted = false;
    if (map_get(m, &view) == NULL) {
        map_put(m, &view, &v);
        inserted = true;
    }
    return inserted;
}

static int * imap_get(struct map * m, const char * k, size_t kn) {
    struct chars view = chars_view(k, kn);
    return map_get(m, &view);
}

// OR-into: imap[k] |= bit (insert with bit if absent).
static void imap_or(struct map * m, const char * k, size_t kn, int bit) {
    int * v = imap_get(m, k, kn);
    if (v != NULL) {
        *v |= bit;
    } else {
        imap_set(m, k, kn, bit);
    }
}

// Pair map: insert-if-absent of split-phrase value.
static bool pmap_emplace(struct map * m, const char * k, size_t kn,
                         const char * a, size_t an,
                         const char * b, size_t bn) {
    struct chars view = chars_view(k, kn);
    bool inserted = false;
    if (map_get(m, &view) == NULL) {
        struct strpair sp = {0};
        chars_put(&sp.a, a, an);
        chars_put(&sp.b, b, bn);
        map_put(m, &view, &sp);
        inserted = true;
    }
    return inserted;
}

// ---------------------------------------------------------------------------
// Suffix-decision helpers (used by the -ing / -ed family in phase 6).
// ---------------------------------------------------------------------------

// Resolve word_alt_flags: explicit param if >= 0, else look up the
// word's own $altN bitmask from word_alt_flags state map.
static int determine_alt_flags(struct phonemizer * p,
                               const char * word, size_t n,
                               int explicit_flags) {
    int r = 0;
    if (explicit_flags >= 0) {
        r = explicit_flags;
    } else {
        struct chars wl = {0};
        to_lower(word, n, &wl);
        int * v = map_get(&p->word_alt_flags, &wl);
        if (v != NULL) { r = *v; }
        chars_free(&wl);
    }
    return r;
}

// Voicing assimilation for the -ed suffix "d#" placeholder. If the
// last "real" phoneme of the stem is unvoiced, rewrite "d#" -> "t";
// if it's t/d, rewrite -> "I#d" (syllabic ɪd). Other endings (voiced
// or vowel) keep "d#" (maps to /d/). The "real" search skips the
// internal stress / boundary markers.
static void devoice_ed_suffix(const char * stem_ph, size_t stem_n,
                              struct chars * suffix) {
    bool is_dhash = suffix->count == 2
                 && memcmp(suffix->data, "d#", 2) == 0;
    if (is_dhash && stem_n > 0) {
        char last_ph = 0;
        for (int sj = (int)stem_n - 1; sj >= 0 && last_ph == 0; sj--) {
            char c = stem_ph[sj];
            if (c != '\'' && c != ',' && c != '%' && c != '='
                && c != '\x01') {
                last_ph = c;
            }
        }
        static const char voiceless_last[] = "ptkfTSCxXhs";
        if (last_ph == 't' || last_ph == 'd') {
            suffix->count = 0;
            chars_puts(suffix, "I#d");
        } else if (strchr(voiceless_last, last_ph) != NULL) {
            suffix->count = 0;
            chars_puts(suffix, "t");
        }
    }
}

// Magic-e applicability for CVC stems before -ing or -ed. Looks at
// the last vowel-group in `base_ph`: magic-e applies if that vowel
// already carries primary or secondary stress, or no primary stress
// exists yet (monosyllabic), or the last vowel is a "full" vowel
// (not in `weak_vowels`). For -ing weak_vowels = "I@"; for -ed
// weak_vowels = "I@3" (rhotic ɚ is already a complete vowel).
// Empty base_ph defaults to true — caller's outer flow decides.
static bool should_use_magic_e_for_cvc_stem(const char * base_ph,
                                            size_t bn,
                                            const char * weak_vowels,
                                            size_t wn) {
    bool use_magic_e = true;
    if (bn > 0) {
        static const char vc[] = "aAeEiIoOuUV03@";
        int last_v = -1;
        for (int k = (int)bn - 1; k >= 0 && last_v < 0; k--) {
            if (strchr(vc, base_ph[k]) != NULL) { last_v = k; }
        }
        if (last_v > 0) {
            int vstart = last_v;
            while (vstart > 0
                   && (strchr(vc, base_ph[vstart - 1]) != NULL
                       || base_ph[vstart - 1] == ':'
                       || base_ph[vstart - 1] == '#')) {
                vstart--;
            }
            bool stressed_at_end = vstart > 0
                && (base_ph[vstart - 1] == '\''
                    || base_ph[vstart - 1] == ',');
            bool no_explicit_stress = memchr(base_ph, '\'', bn) == NULL;
            bool last_vowel_is_full =
                memchr(weak_vowels, base_ph[last_v], wn) == NULL;
            use_magic_e = stressed_at_end || no_explicit_stress
                       || last_vowel_is_full;
        }
    }
    return use_magic_e;
}

// Syllabic-L collapse for -ing stems. If stem phonemes end in "@L"
// (syllabic L, used for word-final 'l' after a consonant) and the
// orthographic base ends in 'l', the syllabic context is lost when
// -ing follows. Drop "@L" -> "@l" (vowel+l base) or "l" (other
// consonant+l). Exceptions: 't' before 'l' ("bottling"), and "-ngl"
// endings ("tingling" handled elsewhere as "@-lI2N").
static void simplify_syllabic_l_for_ing(const char * base, size_t bn,
                                        struct chars * sph) {
    bool ends_in_at_L = sph->count >= 2
                     && sph->data[sph->count - 2] == '@'
                     && sph->data[sph->count - 1] == 'L';
    bool base_ends_in_l = bn >= 2 && base[bn - 1] == 'l';
    bool base_prev_is_t = bn >= 2 && base[bn - 2] == 't';
    bool base_ends_in_ngl = bn >= 3
                         && base[bn - 3] == 'n'
                         && base[bn - 2] == 'g'
                         && base[bn - 1] == 'l';
    if (ends_in_at_L && base_ends_in_l
        && !base_prev_is_t && !base_ends_in_ngl) {
        char penult = base[bn - 2];
        bool vowel_before_l = penult == 'a' || penult == 'e'
                           || penult == 'i' || penult == 'o'
                           || penult == 'u';
        sph->count -= 2;        // drop trailing "@L"
        sph->data[sph->count] = '\0';
        chars_puts(sph, vowel_before_l ? "@l" : "l");
    }
}

// ---------------------------------------------------------------------------
// Dictionary loader (en_list reader).
// ---------------------------------------------------------------------------

// Storage dispatch for a single parsed en_list entry. Mirrors the
// big if/else-if chain in IPAPhonemizer::storeDictionaryEntry.
static void store_dictionary_entry(struct phonemizer * p,
                                   const char * w, size_t wn,
                                   const char * ph, size_t pn,
                                   int dialect_cond,
                                   const struct entry_flags * f) {
    if (f->pastf) { set_add(&p->pastf_words, w, wn); }
    if (f->nounf) { set_add(&p->nounf_words, w, wn); }
    if (f->verbf) { set_add(&p->verbf_words, w, wn); }
    bool is_flag_only = pn > 0 && ph[0] == '$';
    if (f->stress_n > 0 && !f->noun && !f->verb
        && (!f->grammar || is_flag_only)) {
        imap_emplace(&p->stress_pos, w, wn, f->stress_n);
        if (is_flag_only && f->onlys) {
            set_add(&p->noun_form_stress, w, wn);
        }
    }
    if (is_flag_only) {
        bool is_alt = pn == 5 && ph[0] == '$'
                   && ph[1] == 'a' && ph[2] == 'l' && ph[3] == 't'
                   && ph[4] >= '1' && ph[4] <= '6';
        if (is_alt) { smap_erase(&p->dict, w, wn); }
        if (f->verb) { set_add(&p->verb_flag_words, w, wn); }
    } else if (f->noun) {
        smap_emplace(&p->noun_dict, w, wn, ph, pn);
    } else if (f->verb) {
        smap_emplace(&p->verb_dict, w, wn, ph, pn);
    } else if (f->atend) {
        if (!f->atstart && ph[0] != '$') {
            smap_set(&p->atend_dict, w, wn, ph, pn);
        }
    } else if (f->capital) {
        if (ph[0] != '$') {
            smap_set(&p->capital_dict, w, wn, ph, pn);
        }
    } else if (f->atstart) {
        smap_set(&p->atstart_dict, w, wn, ph, pn);
    } else if (f->past) {
        smap_emplace(&p->past_dict, w, wn, ph, pn);
    } else if (f->onlys) {
        if (dialect_cond != 0) {
            smap_set(&p->dict, w, wn, ph, pn);
            set_add(&p->onlys_words, w, wn);
        } else {
            bool inserted = smap_emplace(&p->dict, w, wn, ph, pn);
            if (inserted) {
                set_add(&p->onlys_words, w, wn);
            } else if (ph[0] != '$') {
                smap_set(&p->onlys_bare_dict, w, wn, ph, pn);
            }
        }
    } else {
        smap_set(&p->dict, w, wn, ph, pn);
        if (f->only) { set_add(&p->only_words, w, wn); }
        if (f->strend2 && wn >= 2 && pn > 0
            && ph[0] != ',' && ph[0] != '\'' && ph[0] != '%') {
            struct strpair sp = {0};
            chars_put(&sp.a, w, wn);
            chars_put(&sp.b, ph, pn);
            strpairs_put(&p->compound_prefixes, sp);
            set_add(&p->strend_words, w, wn);
        }
        if (f->strend2 && pn > 0 && ph[0] == ',') {
            set_add(&p->comma_strend2_words, w, wn);
        }
        if (f->u2 && f->strend2) {
            set_add(&p->u2_strend2_words, w, wn);
        }
    }
}

// Phrase entry parser. Reads "(word1 word2) phonemes [flags]" and
// stores into phrase_dict / phrase_split_dict / keep_sec_phrase_keys.
// Returns true iff `line` was a phrase entry (consumed); false means
// caller continues normal word-entry processing.
static bool parse_phrase_entry(struct phonemizer * p,
                               const char * line, size_t n) {
    bool is_phrase = n > 0 && line[0] == '(';
    if (is_phrase) {
        const char * close = memchr(line, ')', n);
        if (close != NULL && close > line + 1) {
            size_t words_len = (size_t)(close - line) - 1;
            size_t rest_off = (size_t)(close - line) + 1;
            size_t rest_len = n - rest_off;
            struct chars words_str = {0};
            struct chars rest_str = {0};
            trim(line + 1, words_len, &words_str);
            trim(line + rest_off, rest_len, &rest_str);
            if (rest_str.count > 0 && rest_str.data[0] != '$') {
                struct charsv rp = {0};
                split_ws(rest_str.data, rest_str.count, &rp);
                if (rp.count > 0 && rp.data[0].count > 0
                    && rp.data[0].data[0] != '$') {
                    struct charsv words = {0};
                    split_ws(words_str.data, words_str.count, &words);
                    bool has_atend = false;
                    bool has_pause = false;
                    bool has_u2_plus = false;
                    for (size_t ri = 1; ri < rp.count; ri++) {
                        const struct chars * r = &rp.data[ri];
                        if (r->count == 6
                            && memcmp(r->data, "$atend", 6) == 0) {
                            has_atend = true;
                        }
                        if (r->count == 6
                            && memcmp(r->data, "$pause", 6) == 0) {
                            has_pause = true;
                        }
                        if (r->count == 4
                            && memcmp(r->data, "$u2+",  4) == 0) {
                            has_u2_plus = true;
                        }
                    }
                    bool storable = words.count == 2
                        && !has_atend && !has_pause
                        && memchr(words.data[0].data, '.',
                                  words.data[0].count) == NULL
                        && memchr(words.data[1].data, '.',
                                  words.data[1].count) == NULL;
                    if (storable) {
                        struct chars w0lo = {0};
                        struct chars w1lo = {0};
                        to_lower(words.data[0].data,
                                 words.data[0].count, &w0lo);
                        to_lower(words.data[1].data,
                                 words.data[1].count, &w1lo);
                        struct chars key = {0};
                        chars_put(&key, w0lo.data, w0lo.count);
                        chars_put_byte(&key, ' ');
                        chars_put(&key, w1lo.data, w1lo.count);
                        chars_free(&w0lo);
                        chars_free(&w1lo);
                        const struct chars * phs = &rp.data[0];
                        const char * pipe = NULL;
                        for (size_t k = 0;
                             pipe == NULL && k + 1 < phs->count; k++) {
                            if (phs->data[k] == '|'
                                && phs->data[k + 1] == '|') {
                                pipe = phs->data + k;
                            }
                        }
                        if (pipe != NULL) {
                            size_t pa = (size_t)(pipe - phs->data);
                            size_t pb_off = pa + 2;
                            size_t pb_len = phs->count - pb_off;
                            pmap_emplace(&p->phrase_split_dict,
                                         key.data, key.count,
                                         phs->data, pa,
                                         phs->data + pb_off, pb_len);
                        } else {
                            bool has_prime = memchr(phs->data, '\'',
                                                    phs->count) != NULL;
                            bool starts_pct = phs->count > 0
                                           && phs->data[0] == '%';
                            struct chars phon = {0};
                            if (!has_prime && !starts_pct) {
                                chars_put_byte(&phon, '%');
                            }
                            chars_put(&phon, phs->data, phs->count);
                            smap_emplace(&p->phrase_dict,
                                         key.data, key.count,
                                         phon.data, phon.count);
                            chars_free(&phon);
                            if (has_u2_plus) {
                                set_add(&p->keep_sec_phrase_keys,
                                        key.data, key.count);
                            }
                        }
                        chars_free(&key);
                    }
                    charsv_clear(&words);
                }
                charsv_clear(&rp);
            }
            chars_free(&words_str);
            chars_free(&rest_str);
        }
    }
    return is_phrase;
}

// Per-entry flag scanner. Reads parts[2..] + phonemes_str (parts[1])
// and side-effects unstressed_words / u_plus_secondary_words /
// unstressend_words / abbrev_words / word_alt_flags. Populates
// `*flags` with the per-entry bits.
static void parse_entry_flags(struct phonemizer * p,
                              const struct charsv * parts,
                              const char * w, size_t wn,
                              const char * ph, size_t pn,
                              struct entry_flags * flags) {
    if (pn > 0 && ph[0] == '$') {
        flags->stress_n = parse_stress_n(ph, pn);
    }
    for (size_t fi = 2; fi < parts->count; fi++) {
        const struct chars * fl = &parts->data[fi];
        const char * fd = fl->data;
        size_t fc = fl->count;
        bool eq_noun  = fc == 5 && memcmp(fd, "$noun",  5) == 0;
        bool eq_verb  = fc == 5 && memcmp(fd, "$verb",  5) == 0;
        bool eq_past  = fc == 5 && memcmp(fd, "$past",  5) == 0;
        bool eq_pastf = fc == 6 && memcmp(fd, "$pastf", 6) == 0;
        bool eq_nounf = fc == 6 && memcmp(fd, "$nounf", 6) == 0;
        bool eq_verbf = fc == 6 && memcmp(fd, "$verbf", 6) == 0;
        bool eq_atend = fc == 6 && memcmp(fd, "$atend", 6) == 0;
        bool eq_allcaps  = fc == 8 && memcmp(fd, "$allcaps",  8) == 0;
        bool eq_sentence = fc == 9 && memcmp(fd, "$sentence", 9) == 0;
        bool eq_capital = fc == 8 && memcmp(fd, "$capital", 8) == 0;
        bool eq_atstart = fc == 8 && memcmp(fd, "$atstart", 8) == 0;
        bool eq_strend2 = fc == 8 && memcmp(fd, "$strend2", 8) == 0;
        bool eq_alt2    = fc == 5 && memcmp(fd, "$alt2",    5) == 0;
        bool eq_alt3    = fc == 5 && memcmp(fd, "$alt3",    5) == 0;
        bool eq_only    = fc == 5 && memcmp(fd, "$only",    5) == 0;
        bool eq_onlys   = fc == 6 && memcmp(fd, "$onlys",   6) == 0;
        bool eq_u2      = fc == 3 && memcmp(fd, "$u2",      3) == 0;
        bool eq_uplus   = fc == 3 && memcmp(fd, "$u+",      3) == 0;
        bool eq_u       = fc == 2 && memcmp(fd, "$u",       2) == 0;
        bool eq_unstressend = fc == 12
            && memcmp(fd, "$unstressend", 12) == 0;
        bool eq_abbrev  = fc == 7 && memcmp(fd, "$abbrev", 7) == 0;
        bool is_altN = fc == 5 && fd[0] == '$' && fd[1] == 'a'
                    && fd[2] == 'l' && fd[3] == 't'
                    && fd[4] >= '1' && fd[4] <= '6';
        if (eq_noun)  { flags->noun = true; }
        if (eq_verb)  { flags->verb = true; flags->grammar = true; }
        if (eq_past)  { flags->past = true; }
        if (eq_pastf) { flags->pastf = true; }
        if (eq_nounf) { flags->nounf = true; flags->grammar = true; }
        if (eq_verbf) { flags->verbf = true; flags->grammar = true; }
        if (eq_atend || eq_allcaps || eq_sentence) { flags->atend = true; }
        if (eq_capital) { flags->capital = true; }
        if (eq_atstart) { flags->atstart = true; }
        if (eq_verbf || eq_strend2 || eq_alt2 || eq_alt3 || eq_only) {
            flags->grammar = true;
        }
        if (eq_only)  { flags->only = true; }
        if (eq_onlys) { flags->onlys = true; }
        if (eq_strend2) { flags->strend2 = true; }
        if (eq_u2) { flags->u2 = true; }
        if (eq_uplus) {
            set_add(&p->unstressed_words, w, wn);
            bool has_comma = memchr(ph, ',', pn) != NULL;
            bool has_prime = memchr(ph, '\'', pn) != NULL;
            if (has_comma && !has_prime) {
                set_add(&p->u_plus_secondary_words, w, wn);
            }
        }
        if (eq_u) { set_add(&p->unstressed_words, w, wn); }
        if (eq_unstressend) { set_add(&p->unstressend_words, w, wn); }
        if (eq_abbrev) { set_add(&p->abbrev_words, w, wn); }
        if (is_altN) {
            imap_or(&p->word_alt_flags, w, wn, 1 << (fd[4] - '1'));
        }
        if (flags->stress_n == 0) {
            flags->stress_n = parse_stress_n(fd, fc);
        }
    }
    // phonemes_str-as-flag fallback (e.g. "gi $abbrev" pattern).
    bool ph_eq_abbrev = pn == 7 && memcmp(ph, "$abbrev", 7) == 0;
    bool ph_eq_verb   = pn == 5 && memcmp(ph, "$verb",   5) == 0;
    bool ph_eq_verbf  = pn == 6 && memcmp(ph, "$verbf",  6) == 0;
    bool ph_eq_nounf  = pn == 6 && memcmp(ph, "$nounf",  6) == 0;
    bool ph_eq_pastf  = pn == 6 && memcmp(ph, "$pastf",  6) == 0;
    bool ph_eq_only   = pn == 5 && memcmp(ph, "$only",   5) == 0;
    bool ph_eq_u      = pn == 2 && memcmp(ph, "$u",      2) == 0;
    bool ph_eq_uplus  = pn == 3 && memcmp(ph, "$u+",     3) == 0;
    bool ph_is_altN = pn == 5 && ph[0] == '$' && ph[1] == 'a'
                   && ph[2] == 'l' && ph[3] == 't'
                   && ph[4] >= '1' && ph[4] <= '6';
    if (ph_eq_abbrev) { set_add(&p->abbrev_words, w, wn); }
    if (ph_is_altN) {
        imap_or(&p->word_alt_flags, w, wn, 1 << (ph[4] - '1'));
    }
    if (ph_eq_verb || ph_eq_verbf || ph_eq_nounf
        || ph_eq_pastf || ph_eq_only) {
        flags->grammar = true;
    }
    if (ph_eq_pastf) { flags->pastf = true; }
    if (ph_eq_nounf) { flags->nounf = true; }
    if (ph_eq_verbf) { flags->verbf = true; }
    if (ph_eq_u || ph_eq_uplus) {
        set_add(&p->unstressed_words, w, wn);
    }
    if (ph_eq_u)    { flags->grammar = true; }
    if (ph_eq_verb) { flags->verb    = true; }
}

// Comparator for qsort: order strpairs by .a.count descending so the
// loader can do longest-match-first lookup on compound_prefixes.
static int strpair_len_desc(const void * pa, const void * pb) {
    const struct strpair * a = pa;
    const struct strpair * b = pb;
    int r = 0;
    if (a->a.count < b->a.count) { r = 1; }
    else if (a->a.count > b->a.count) { r = -1; }
    return r;
}

// Load the en_list dictionary file. Returns true on success. On
// failure writes the error message to p->err and returns false.
static bool load_dictionary(struct phonemizer * p, const char * path) {
    FILE * f = fopen(path, "r");
    bool ok = true;
    if (f == NULL) {
        snprintf(p->err, sizeof(p->err),
                 "Cannot open dictionary file: %s", path);
        ok = false;
    } else {
        bool is_en_us = (p->dialect.count == 5
            && (memcmp(p->dialect.data, "en-us", 5) == 0
             || memcmp(p->dialect.data, "en_us", 5) == 0));
        char * raw = NULL;
        size_t cap = 0;
        ssize_t n_read = 0;
        while ((n_read = getline(&raw, &cap, f)) != -1) {
            size_t n = (size_t)n_read;
            if (n > 0 && raw[n - 1] == '\n') { n--; }
            if (n > 0 && raw[n - 1] == '\r') { n--; }
            struct chars line = {0};
            strip_comment_and_trim(raw, n, &line);
            bool live = line.count > 0;
            if (live && parse_phrase_entry(p, line.data, line.count)) {
                live = false;
            }
            int dialect_cond = 0;
            bool cond_negated = false;
            struct chars after_dialect = {0};
            if (live) {
                parse_leading_dialect(line.data, line.count,
                                      &dialect_cond, &cond_negated,
                                      &after_dialect);
                line.count = 0;
                chars_put(&line, after_dialect.data, after_dialect.count);
                if (dialect_cond != 0) {
                    bool match = (dialect_cond == 3 || dialect_cond == 6)
                               && is_en_us;
                    bool applies = cond_negated ? !match : match;
                    if (!applies) { live = false; }
                }
            }
            if (live) {
                struct charsv parts = {0};
                split_ws(line.data, line.count, &parts);
                if (parts.count >= 2) {
                    const struct chars * w_part = &parts.data[0];
                    const struct chars * ph_part = &parts.data[1];
                    struct chars norm = {0};
                    to_lower(w_part->data, w_part->count, &norm);
                    struct entry_flags flags = {0};
                    parse_entry_flags(p, &parts,
                                      norm.data, norm.count,
                                      ph_part->data, ph_part->count,
                                      &flags);
                    store_dictionary_entry(p, norm.data, norm.count,
                                           ph_part->data, ph_part->count,
                                           dialect_cond, &flags);
                    chars_free(&norm);
                }
                charsv_clear(&parts);
            }
            chars_free(&after_dialect);
            chars_free(&line);
        }
        free(raw);
        fclose(f);
        // Sort compound_prefixes longest-first for greedy matching.
        qsort(p->compound_prefixes.data, p->compound_prefixes.count,
              sizeof(struct strpair), strpair_len_desc);
        // Post-load: remove "made" from unstressed_words (content
        // word with $u+ that the reference still stresses in sentence
        // context).
        struct chars made = chars_view("made", 4);
        map_remove(&p->unstressed_words, &made);
    }
    return ok;
}

// ---------------------------------------------------------------------------
// Rule loader (en_rules reader).
// ---------------------------------------------------------------------------

// Parse a ".LNN items..." line. `line` is the trimmed input;
// `line[0]` is '.'. Appends items (up to a "//" comment) into the
// matching lgroup slot.
static void parse_lgroup_def(const char * line, size_t n,
                             struct ruleset * rs) {
    if (n >= 3 && line[1] == 'L') {
        int id = 0;
        size_t i = 2;
        while (i < n && isdigit((unsigned char)line[i])) {
            id = id * 10 + (line[i] - '0');
            i++;
        }
        if (id > 0 && id < 100) {
            struct charsv items = {0};
            split_ws(line + i, n - i, &items);
            struct charsv * g = &rs->groups.lgroups[id];
            size_t taken = items.count;
            for (size_t k = 0; k < items.count; k++) {
                const struct chars * s = &items.data[k];
                if (taken == items.count && s->count >= 2
                    && s->data[0] == '/' && s->data[1] == '/') {
                    taken = k;
                }
            }
            for (size_t k = 0; k < taken; k++) {
                charsv_put(g, items.data[k]);
                items.data[k] = (struct chars){0};
            }
            charsv_clear(&items);
        }
    }
}

// Detect 'P' prefix-boundary marker in the rule's right context.
// 'P' is the prefix marker when followed by a digit / end / '_' /
// '+' / '<'. A 'P' preceded by 'L' is an L-group reference, not
// the prefix marker. `rule->is_prefix` is the natural search-done
// state.
static void detect_prefix_marker(struct phoneme_rule * rule) {
    const struct chars * rc = &rule->right_ctx;
    for (size_t k = 0; k < rc->count && !rule->is_prefix; k++) {
        if (rc->data[k] == 'P') {
            bool followed_by_marker = (k + 1 >= rc->count)
                || (rc->data[k + 1] >= '1' && rc->data[k + 1] <= '9')
                || rc->data[k + 1] == '_'
                || rc->data[k + 1] == '+'
                || rc->data[k + 1] == '<';
            bool preceded_by_L = k > 0 && rc->data[k - 1] == 'L';
            if (followed_by_marker && !preceded_by_L) {
                rule->is_prefix = true;
            }
        }
    }
}

// Detect 'S<N>[flags]' RULE_ENDING marker in the rule's right
// context. 'S' (not preceded by 'L') begins a suffix directive:
// <N> chars to strip + flag letters i/m/v/e/d/q/p.
static void detect_suffix_marker(struct phoneme_rule * rule) {
    static const int SUFX_I_BIT = 0x200;
    static const int SUFX_M_BIT = 0x80000;
    static const int SUFX_V_BIT = 0x800;
    static const int SUFX_E_BIT = 0x100;
    static const int SUFX_D_BIT = 0x1000;
    static const int SUFX_Q_BIT = 0x4000;
    static const int SUFX_P_BIT = 0x400;
    const struct chars * rc = &rule->right_ctx;
    for (size_t k = 0; k < rc->count && !rule->is_suffix; k++) {
        if (rc->data[k] == 'S' && (k == 0 || rc->data[k - 1] != 'L')) {
            size_t k2 = k + 1;
            int n = 0;
            int sflags = 0;
            while (k2 < rc->count
                   && isdigit((unsigned char)rc->data[k2])) {
                n = n * 10 + (rc->data[k2] - '0');
                k2++;
            }
            while (k2 < rc->count
                   && isalpha((unsigned char)rc->data[k2])) {
                char fc = rc->data[k2];
                k2++;
                if      (fc == 'i') { sflags |= SUFX_I_BIT; }
                else if (fc == 'm') { sflags |= SUFX_M_BIT; }
                else if (fc == 'v') { sflags |= SUFX_V_BIT; }
                else if (fc == 'e') { sflags |= SUFX_E_BIT; }
                else if (fc == 'd') { sflags |= SUFX_D_BIT; }
                else if (fc == 'q') { sflags |= SUFX_Q_BIT; }
                else if (fc == 'p') { sflags |= SUFX_P_BIT; }
            }
            if (n > 0) {
                rule->is_suffix = true;
                rule->suffix_strip_len = n;
                rule->suffix_flags = sflags;
            }
        }
    }
}

// Load the en_rules file. Returns true on success; on failure
// writes the message to p->err and returns false.
static bool load_rules(struct phonemizer * p, const char * path) {
    FILE * f = fopen(path, "r");
    bool ok = true;
    if (f == NULL) {
        snprintf(p->err, sizeof(p->err),
                 "Cannot open rules file: %s", path);
        ok = false;
    } else {
        bool is_en_us = (p->dialect.count == 5
            && (memcmp(p->dialect.data, "en-us", 5) == 0
             || memcmp(p->dialect.data, "en_us", 5) == 0));
        struct chars current_group = {0};
        bool in_replace_section = false;
        char * raw = NULL;
        size_t cap = 0;
        ssize_t n_read = 0;
        while ((n_read = getline(&raw, &cap, f)) != -1) {
            size_t n = (size_t)n_read;
            if (n > 0 && raw[n - 1] == '\n') { n--; }
            if (n > 0 && raw[n - 1] == '\r') { n--; }
            struct chars line = {0};
            strip_comment_and_trim(raw, n, &line);
            bool live = line.count > 0;
            // Directive lines (".LNN", ".replace", ".group X")
            if (live && line.data[0] == '.') {
                if (line.count >= 2 && line.data[1] == 'L') {
                    parse_lgroup_def(line.data, line.count, &p->rules);
                    live = false;
                } else if (line.count == 8
                    && memcmp(line.data, ".replace", 8) == 0) {
                    in_replace_section = true;
                    current_group.count = 0;
                    if (current_group.data != NULL) {
                        current_group.data[0] = '\0';
                    }
                    live = false;
                } else if (line.count >= 6
                    && memcmp(line.data, ".group", 6) == 0) {
                    in_replace_section = false;
                    struct chars rest = {0};
                    trim(line.data + 6, line.count - 6, &rest);
                    current_group.count = 0;
                    chars_put(&current_group, rest.data, rest.count);
                    chars_free(&rest);
                    live = false;
                }
            }
            if (live && in_replace_section) {
                struct charsv parts = {0};
                split_ws(line.data, line.count, &parts);
                if (parts.count >= 2) {
                    struct replace_rule rr = {0};
                    chars_put(&rr.from, parts.data[0].data,
                              parts.data[0].count);
                    chars_put(&rr.to,   parts.data[1].data,
                              parts.data[1].count);
                    replaces_put(&p->rules.replacements, rr);
                }
                charsv_clear(&parts);
                live = false;
            }
            if (live && current_group.count == 0) { live = false; }
            if (live) {
                int dialect_cond = 0;
                bool cond_negated = false;
                struct chars rule_line = {0};
                parse_leading_dialect(line.data, line.count,
                                      &dialect_cond, &cond_negated,
                                      &rule_line);
                bool applies = true;
                if (dialect_cond != 0) {
                    bool match = (dialect_cond == 3) && is_en_us;
                    applies = cond_negated ? !match : match;
                }
                if (applies) {
                    struct charsv tokens = {0};
                    split_ws(rule_line.data, rule_line.count, &tokens);
                    if (tokens.count > 0) {
                        struct phoneme_rule rule = {0};
                        rule.condition = dialect_cond;
                        rule.condition_negated = cond_negated;
                        size_t ti = 0;
                        // Left context: token ending with ')'.
                        if (ti < tokens.count
                            && tokens.data[ti].count > 0
                            && tokens.data[ti]
                                .data[tokens.data[ti].count - 1] == ')') {
                            const struct chars * t = &tokens.data[ti];
                            chars_put(&rule.left_ctx, t->data,
                                      t->count - 1);
                            ti++;
                        }
                        // Match string: next token not starting '('.
                        if (ti < tokens.count
                            && tokens.data[ti].count > 0
                            && tokens.data[ti].data[0] != '(') {
                            const struct chars * t = &tokens.data[ti];
                            chars_put(&rule.match, t->data, t->count);
                            ti++;
                        } else {
                            chars_put(&rule.match,
                                      current_group.data,
                                      current_group.count);
                        }
                        // Right context: token starting with '('.
                        if (ti < tokens.count
                            && tokens.data[ti].count > 0
                            && tokens.data[ti].data[0] == '(') {
                            const struct chars * t = &tokens.data[ti];
                            chars_put(&rule.right_ctx,
                                      t->data + 1, t->count - 1);
                            ti++;
                        }
                        // Phonemes: rest of tokens, joined (no sep).
                        for (size_t j = ti; j < tokens.count; j++) {
                            chars_put(&rule.phonemes,
                                      tokens.data[j].data,
                                      tokens.data[j].count);
                        }
                        bool flag_only = rule.phonemes.count > 0
                            && rule.phonemes.data[0] == '$';
                        bool match_empty = rule.match.count == 0;
                        if (!flag_only && !match_empty) {
                            detect_prefix_marker(&rule);
                            detect_suffix_marker(&rule);
                            // Get-or-create rules vec for this group.
                            struct chars view = chars_view(
                                current_group.data,
                                current_group.count);
                            struct rules * rvec = map_get(
                                &p->rules.rule_groups, &view);
                            if (rvec == NULL) {
                                struct rules empty = {0};
                                rvec = map_put(&p->rules.rule_groups,
                                               &view, &empty);
                            }
                            rules_put(rvec, rule);
                            // rule heap moved into the vec; reset.
                            rule = (struct phoneme_rule){0};
                        } else {
                            chars_free(&rule.left_ctx);
                            chars_free(&rule.match);
                            chars_free(&rule.right_ctx);
                            chars_free(&rule.phonemes);
                        }
                    }
                    charsv_clear(&tokens);
                }
                chars_free(&rule_line);
            }
            chars_free(&line);
        }
        free(raw);
        chars_free(&current_group);
        fclose(f);
    }
    return ok;
}

// ---------------------------------------------------------------------------
// Rule-application core: leaves (apply_replacements, match_lgroup_at).
// ---------------------------------------------------------------------------

// Apply every replacement rule in `*reps` in-place on `*out`.
// For each rule, walks the buffer finding occurrences of `from`
// and substituting `to`. The buffer grows / shrinks as needed via
// memmove + chars_grow. Empty `from` rules are no-ops. SESE: the
// inner loop's `pos < out->count - from->count + 1` is the natural
// terminator; the substitution branch advances `pos` by to->count
// (post-substitution position), the no-match branch by 1.
static void apply_replacements(struct chars * out,
                               const struct replaces * reps) {
    for (size_t ri = 0; ri < reps->count; ri++) {
        const struct chars * from = &reps->data[ri].from;
        const struct chars * to   = &reps->data[ri].to;
        if (from->count > 0) {
            size_t pos = 0;
            while (pos + from->count <= out->count) {
                bool hit = memcmp(out->data + pos, from->data,
                                  from->count) == 0;
                if (hit && to->count == from->count) {
                    memcpy(out->data + pos, to->data, to->count);
                    pos += to->count;
                } else if (hit && to->count < from->count) {
                    size_t shrink = from->count - to->count;
                    memcpy(out->data + pos, to->data, to->count);
                    memmove(out->data + pos + to->count,
                            out->data + pos + from->count,
                            out->count - (pos + from->count));
                    out->count -= shrink;
                    out->data[out->count] = '\0';
                    pos += to->count;
                } else if (hit) {
                    size_t grow = to->count - from->count;
                    chars_grow(out, out->count + grow + 1);
                    memmove(out->data + pos + to->count,
                            out->data + pos + from->count,
                            out->count - (pos + from->count));
                    memcpy(out->data + pos, to->data, to->count);
                    out->count += grow;
                    out->data[out->count] = '\0';
                    pos += to->count;
                } else {
                    pos++;
                }
            }
        }
    }
}

// Check if `word[pos..)` starts with any item in `lgroup`. Returns
// the longest matching item length, or 0 if no match. Lower-case
// comparison on both sides (matches the .cpp).
static int match_lgroup_at(const struct charsv * lgroup,
                           const char * word, size_t wn, int pos) {
    int best = 0;
    if (pos >= 0 && pos < (int)wn) {
        for (size_t i = 0; i < lgroup->count; i++) {
            const struct chars * item = &lgroup->data[i];
            int ilen = (int)item->count;
            if (ilen > 0 && pos + ilen <= (int)wn) {
                bool ok = true;
                for (int j = 0; j < ilen && ok; j++) {
                    char wc = (char)tolower((unsigned char)word[pos + j]);
                    char ic = (char)tolower((unsigned char)item->data[j]);
                    if (wc != ic) { ok = false; }
                }
                if (ok && ilen > best) { best = ilen; }
            }
        }
    }
    return best;
}

// ---------------------------------------------------------------------------
// Rule-application core: context scoring.
// ---------------------------------------------------------------------------

// Result of a left/right context match: score iff `matched` is true.
struct ctx_score {
    int  score;
    bool matched;
};

// matchGroup equivalent: indexes one of the bool[256] tables in
// `groups` selected by `g`. Returns false on invalid pos or
// unknown group.
static bool match_group(const struct letter_groups * groups,
                        char g, const char * word,
                        size_t wn, int pos) {
    bool r = false;
    if (pos >= 0 && pos < (int)wn) {
        char c = (char)tolower((unsigned char)word[pos]);
        unsigned char u = (unsigned char)c;
        switch (g) {
            case 'A': r = groups->groupA[u]; break;
            case 'B': r = groups->groupB[u]; break;
            case 'C': r = groups->groupC[u]; break;
            case 'F': r = groups->groupF[u]; break;
            case 'G': r = groups->groupG[u]; break;
            case 'H': r = groups->groupH[u]; break;
            case 'Y': r = groups->groupY[u]; break;
            case 'K': r = groups->groupK[u]; break;
            default: break;
        }
    }
    return r;
}

// "&" stressed-state scan: walk phonemes_so_far (multi-char-aware,
// using STRESSED_MC) and decide whether any vowel code in there is
// stressable (not preceded by '%'/'='/',' and not inherently
// unstressed). Replaces the inner-loop scan inside matchLeftContextScore.
static const char * const STRESSED_MC[] = {
    "aI@3","aU@r","i@3r","aI@","aI3","aU@","i@3",
    "3:r","A:r","o@r","A@r","e@r",
    "eI","aI","aU","OI","oU","IR","VR",
    "e@","i@","U@","A@","O@","o@",
    "3:","A:","i:","u:","O:","e:","a:","aa",
    "@L","@2","@5",
    "I2","I#","E2","E#","e#","a#","a2","0#","02",
    "O2","A~","O~","A#",
    NULL
};

static bool stressed_in_phonemes(const char * ph, size_t pn) {
    static const char VOWEL_PH_STRESSABLE[] = "aAeEiIoOuUV03";
    bool found = false;
    size_t pi = 0;
    bool prev_unstressed = false;
    while (pi < pn && !found) {
        char pc = ph[pi];
        if (pc == '%' || pc == '=' || pc == ',') {
            prev_unstressed = true;
            pi++;
        } else if (pc == '\'') {
            prev_unstressed = false;
            pi++;
        } else {
            const char * code = NULL;
            size_t code_len = 0;
            for (int mi = 0;
                 STRESSED_MC[mi] != NULL && code == NULL; mi++) {
                size_t mcl = strlen(STRESSED_MC[mi]);
                if (pi + mcl <= pn
                    && memcmp(ph + pi, STRESSED_MC[mi], mcl) == 0) {
                    code = STRESSED_MC[mi];
                    code_len = mcl;
                }
            }
            if (code == NULL) {
                code = ph + pi;
                code_len = 1;
            }
            bool is_vowel = code_len > 0
                && strchr(VOWEL_PH_STRESSABLE, code[0]) != NULL;
            if (is_vowel) {
                bool inherently_unstressed = code[0] == '@'
                    || (code_len == 1 && code[0] == 'i')
                    || (code_len == 2 && code[0] == 'a' && code[1] == '#')
                    || (code_len == 2 && code[0] == 'I' && code[1] == '#')
                    || (code_len == 2 && code[0] == 'I' && code[1] == '2');
                if (!inherently_unstressed && !prev_unstressed) {
                    found = true;
                }
            }
            if (!found) {
                prev_unstressed = false;
                pi += code_len;
            }
        }
    }
    return found;
}

// Match left context: scans ctx right-to-left starting at pos-1
// in word. Token types: literal (case-fold), '_' word-bound, '&'
// stressed, '@' syllable, '!' capital (skip), '%' double, '+'/'<'
// inc/dec score, A/B/C/F/G/H/Y letter group, K not-vowel, X no-
// vowels, D digit, Z non-alpha, LNN L-group ref, 'E' replaced-e.
// Returns (score, matched). SESE: every former early-return
// becomes `ok = false` and the loop's `&& ok` predicate exits.
static struct ctx_score match_left_context_score(
        const char * ctx_str, size_t ctx_n,
        const char * word, size_t wn, int pos,
        const struct ruleset * rs,
        const char * ph_so_far, size_t ph_n) {
    struct ctx_score result = { 0, false };
    int score = 0;
    bool ok = true;
    assert(ctx_n > 0);
    int word_pos = pos - 1;
    int ci = (int)ctx_n - 1;
    int distance_left = -2;
    char prev_char = (pos > 0 && pos < (int)wn) ? word[pos] : 0;
    while (ci >= 0 && ok) {
        char cc = ctx_str[ci];
        if (cc == '_') {
            if (word_pos >= 0) {
                ok = false;
            } else {
                ci--;
                score += 4;
            }
        } else if (cc == '&') {
            bool found_stressed = false;
            if (ph_n > 0) {
                found_stressed = stressed_in_phonemes(ph_so_far, ph_n);
            } else {
                for (int k = 0; k < pos && !found_stressed; k++) {
                    if (is_vowel_letter(word[k])) {
                        found_stressed = true;
                    }
                }
            }
            if (!found_stressed) {
                ok = false;
            } else {
                ci--;
                score += 19;
            }
        } else if (cc == '@') {
            int syllable_count = 0;
            while (ci >= 0 && ctx_str[ci] == '@') {
                syllable_count++;
                ci--;
            }
            int vowel_groups = 0;
            if (ph_n > 0) {
                static const char VOWEL_PH[] = "aAeEIiOUVu03@o";
                bool in_v2 = false;
                for (size_t k = 0; k < ph_n; k++) {
                    bool v = strchr(VOWEL_PH, ph_so_far[k]) != NULL;
                    if (v && !in_v2) {
                        vowel_groups++;
                        in_v2 = true;
                    } else if (!v) {
                        in_v2 = false;
                    }
                }
            } else {
                bool in_v2 = false;
                for (int wp = 0; wp < pos; wp++) {
                    bool v = is_vowel_letter(word[wp]);
                    if (v && !in_v2) {
                        vowel_groups++;
                        in_v2 = true;
                    } else if (!v) {
                        in_v2 = false;
                    }
                }
            }
            if (syllable_count > vowel_groups) {
                ok = false;
            } else {
                int dist = distance_left + 2;
                if (dist > 19) { dist = 19; }
                score += 18 + syllable_count - dist;
            }
        } else if (cc == '!') {
            ci--;
        } else if (cc == '%') {
            if (word_pos < 0) {
                ok = false;
            } else {
                char cur = (char)tolower((unsigned char)word[word_pos]);
                char nxt = (char)tolower((unsigned char)prev_char);
                if (cur != nxt) {
                    ok = false;
                } else {
                    distance_left += 2;
                    if (distance_left > 19) { distance_left = 19; }
                    prev_char = word[word_pos];
                    word_pos--;
                    ci--;
                    score += 21 - distance_left;
                }
            }
        } else if (cc == '+') {
            score += 20;
            ci--;
        } else if (cc == '<') {
            score -= 20;
            ci--;
        } else if (cc == 'A' || cc == 'B' || cc == 'C'
                || cc == 'F' || cc == 'G' || cc == 'H' || cc == 'Y') {
            if (!match_group(&rs->groups, cc, word, wn, word_pos)) {
                ok = false;
            } else {
                distance_left += 2;
                if (distance_left > 19) { distance_left = 19; }
                int lg_pts = (cc == 'C') ? 19 : 20;
                prev_char = word[word_pos];
                word_pos--;
                ci--;
                score += lg_pts - distance_left;
            }
        } else if (cc == 'K') {
            if (word_pos < 0 || is_vowel_letter(word[word_pos])) {
                ok = false;
            } else {
                distance_left += 2;
                if (distance_left > 19) { distance_left = 19; }
                prev_char = word[word_pos];
                word_pos--;
                ci--;
                score += 20 - distance_left;
            }
        } else if (cc == 'X') {
            bool found_vowel = false;
            for (int k = 0; k <= word_pos && !found_vowel; k++) {
                if (is_vowel_letter(word[k])) { found_vowel = true; }
            }
            if (found_vowel) {
                ok = false;
            } else {
                ci--;
                score += 3;
            }
        } else if (cc == 'D') {
            if (word_pos < 0
                || !isdigit((unsigned char)word[word_pos])) {
                ok = false;
            } else {
                distance_left += 2;
                if (distance_left > 19) { distance_left = 19; }
                prev_char = word[word_pos];
                word_pos--;
                ci--;
                score += 21 - distance_left;
            }
        } else if (cc == 'Z') {
            if (word_pos < 0
                || isalpha((unsigned char)word[word_pos])) {
                ok = false;
            } else {
                distance_left += 2;
                if (distance_left > 19) { distance_left = 19; }
                prev_char = word[word_pos];
                word_pos--;
                ci--;
                score += 21 - distance_left;
            }
        } else if (cc >= '0' && cc <= '9') {
            int gid = cc - '0';
            int ci2 = ci - 1;
            if (ci2 >= 0
                && ctx_str[ci2] >= '0' && ctx_str[ci2] <= '9') {
                gid += (ctx_str[ci2] - '0') * 10;
                ci2--;
            }
            if (ci2 >= 0 && ctx_str[ci2] == 'L') {
                if (gid > 0 && gid < 100) {
                    int matched = match_lgroup_at(
                        &rs->groups.lgroups[gid], word, wn,
                        word_pos);
                    if (matched == 0) {
                        ok = false;
                    } else {
                        distance_left += 2;
                        if (distance_left > 19) {
                            distance_left = 19;
                        }
                        if (matched > 0) {
                            prev_char = word[word_pos - matched + 1];
                        }
                        word_pos -= matched;
                        score += 20 - distance_left;
                    }
                }
                ci = ci2 - 1;
            } else {
                ci--;
            }
        } else if (cc == 'E') {
            ok = false;
        } else {
            if (word_pos < 0) {
                ok = false;
            } else {
                char wc = (char)tolower((unsigned char)word[word_pos]);
                char mc = (char)tolower((unsigned char)cc);
                if (wc != mc) {
                    ok = false;
                } else {
                    distance_left += 2;
                    if (distance_left > 19) { distance_left = 19; }
                    prev_char = word[word_pos];
                    word_pos--;
                    ci--;
                    score += 21 - distance_left;
                }
            }
        }
    }
    if (ok) {
        result.score = score;
        result.matched = true;
    }
    return result;
}

// Result of a right-context match. Adds the RULE_DEL_FWD bookkeeping
// (start position + count of chars to mark silent) on top of the
// shared (score, matched) shape.
struct right_ctx_score {
    int  score;
    int  del_fwd_start;
    int  del_fwd_count;
    bool matched;
};

// Match right context: scans ctx left-to-right starting at `pos`
// in word. Token types: literal, '_' word-end, '#' RULE_DEL_FWD,
// A/B/C/F/G/H/Y letter group, K (matches null at word end), X (no
// vowels remaining), D digit, Z non-alpha, '%' double, '+'/'<'
// inc/dec, '@' syllable, '&' fails, '!' skipped, '$w_altN'
// word-alt condition, 'N' / Nn suffix-removed condition, 'P' skip
// to whitespace, 'S<N>[flags]' suffix directive (no consume),
// LNN L-group ref, 'E' replaced-e match.
static struct right_ctx_score match_right_context_score(
        const char * ctx_str, size_t ctx_n,
        const char * word, size_t wn, int pos,
        const struct ruleset * rs,
        char initial_prev_char,
        int word_alt_flags,
        const bool * replaced_e_arr, size_t re_n,
        bool suffix_removed) {
    struct right_ctx_score result = { 0, -1, 0, false };
    int score = 0;
    int del_fwd_pos = -1;
    bool ok = true;
    assert(ctx_n > 0);
    int word_pos = pos;
    int ci = 0;
    int clen = (int)ctx_n;
    int distance_right = -6;
    char prev_char = initial_prev_char;
    while (ci < clen && ok) {
        char cc = ctx_str[ci];
        if (cc == '_') {
            if (word_pos < (int)wn) {
                ok = false;
            } else {
                distance_right += 6;
                if (distance_right > 19) { distance_right = 19; }
                ci++;
                score += 21 - distance_right;
            }
        } else if (cc == '#') {
            if (del_fwd_pos < 0) {
                for (int sp = pos;
                     sp < word_pos && del_fwd_pos < 0; sp++) {
                    if (word[sp] == 'e') { del_fwd_pos = sp; }
                }
            }
            ci++;
        } else if (cc == 'A' || cc == 'B' || cc == 'C'
                || cc == 'F' || cc == 'G' || cc == 'H' || cc == 'Y') {
            if (!match_group(&rs->groups, cc, word, wn, word_pos)) {
                ok = false;
            } else {
                distance_right += 6;
                if (distance_right > 19) { distance_right = 19; }
                int lg_pts = (cc == 'C') ? 19 : 20;
                prev_char = word[word_pos];
                word_pos++;
                ci++;
                score += lg_pts - distance_right;
            }
        } else if (cc == 'K') {
            // K matches non-vowel; null at word end also non-vowel.
            if (word_pos < (int)wn && is_vowel_letter(word[word_pos])) {
                ok = false;
            } else {
                distance_right += 6;
                if (distance_right > 19) { distance_right = 19; }
                if (word_pos < (int)wn) { prev_char = word[word_pos]; }
                word_pos++;
                ci++;
                score += 20 - distance_right;
            }
        } else if (cc == 'X') {
            bool found = false;
            for (int k = word_pos; k < (int)wn && !found; k++) {
                if (is_vowel_letter(word[k])) { found = true; }
            }
            if (found) {
                ok = false;
            } else {
                distance_right += 6;
                if (distance_right > 19) { distance_right = 19; }
                ci++;
                score += 19 - distance_right;
            }
        } else if (cc == 'D') {
            if (word_pos >= (int)wn
                || !isdigit((unsigned char)word[word_pos])) {
                ok = false;
            } else {
                distance_right += 6;
                if (distance_right > 19) { distance_right = 19; }
                prev_char = word[word_pos];
                word_pos++;
                ci++;
                score += 21 - distance_right;
            }
        } else if (cc == 'Z') {
            if (word_pos >= (int)wn
                || isalpha((unsigned char)word[word_pos])) {
                ok = false;
            } else {
                distance_right += 6;
                if (distance_right > 19) { distance_right = 19; }
                prev_char = word[word_pos];
                word_pos++;
                ci++;
                score += 21 - distance_right;
            }
        } else if (cc == '%') {
            if (word_pos >= (int)wn) {
                ok = false;
            } else {
                char cur = (char)tolower((unsigned char)word[word_pos]);
                char prv = (char)tolower((unsigned char)prev_char);
                if (cur != prv) {
                    ok = false;
                } else {
                    distance_right += 6;
                    if (distance_right > 19) { distance_right = 19; }
                    prev_char = word[word_pos];
                    word_pos++;
                    ci++;
                    score += 21 - distance_right;
                }
            }
        } else if (cc == '+') {
            score += 20;
            ci++;
        } else if (cc == '<') {
            score -= 20;
            ci++;
        } else if (cc == '@') {
            int syllable_count = 0;
            while (ci < clen && ctx_str[ci] == '@') {
                syllable_count++;
                ci++;
            }
            int vowel_groups = 0;
            bool in_v = false;
            for (int wp = word_pos; wp < (int)wn; wp++) {
                bool v = is_vowel_letter(word[wp]);
                if (v && !in_v) {
                    vowel_groups++;
                    in_v = true;
                } else if (!v) {
                    in_v = false;
                }
            }
            if (syllable_count > vowel_groups) {
                ok = false;
            } else {
                distance_right += 6;
                if (distance_right > 19) { distance_right = 19; }
                score += 18 + syllable_count - distance_right;
            }
        } else if (cc == '&') {
            ok = false;
        } else if (cc == '!') {
            ci++;
        } else if (cc == '$') {
            if (ci + 6 < clen && ctx_str[ci + 1] == 'w'
                && ctx_str[ci + 2] == '_' && ctx_str[ci + 3] == 'a'
                && ctx_str[ci + 4] == 'l' && ctx_str[ci + 5] == 't'
                && ctx_str[ci + 6] >= '1' && ctx_str[ci + 6] <= '6') {
                int alt_n = ctx_str[ci + 6] - '0';
                int alt_bit = 1 << (alt_n - 1);
                if (!(word_alt_flags & alt_bit)) {
                    ok = false;
                } else {
                    ci += 7;
                }
            } else {
                ok = false;
            }
        } else if (cc == 'N') {
            if (ci + 1 < clen && isdigit((unsigned char)ctx_str[ci + 1])) {
                ci += 2;
            } else if (suffix_removed) {
                ok = false;
            } else {
                score += 1;
                ci++;
            }
        } else if (cc == 'P') {
            while (ci < clen
                   && !isspace((unsigned char)ctx_str[ci])) {
                ci++;
            }
        } else if (cc == 'S') {
            ci++;
            while (ci < clen && isdigit((unsigned char)ctx_str[ci])) {
                ci++;
            }
            while (ci < clen && isalpha((unsigned char)ctx_str[ci])) {
                ci++;
            }
        } else if (cc == 'L' && ci + 1 < clen
                   && isdigit((unsigned char)ctx_str[ci + 1])) {
            int gid = 0;
            ci++;
            while (ci < clen && isdigit((unsigned char)ctx_str[ci])) {
                gid = gid * 10 + (ctx_str[ci] - '0');
                ci++;
            }
            if (gid > 0 && gid < 100
                && rs->groups.lgroups[gid].count > 0) {
                int matched = match_lgroup_at(
                    &rs->groups.lgroups[gid], word, wn, word_pos);
                if (matched == 0) {
                    ok = false;
                } else {
                    distance_right += 6;
                    if (distance_right > 19) { distance_right = 19; }
                    if (matched > 0) {
                        prev_char = word[word_pos + matched - 1];
                    }
                    word_pos += matched;
                    score += 20 - distance_right;
                }
            }
        } else if (isdigit((unsigned char)cc)) {
            ci++;
        } else if (cc == 'E') {
            bool re_marked = replaced_e_arr != NULL
                && word_pos < (int)wn
                && word_pos < (int)re_n
                && replaced_e_arr[word_pos];
            if (re_marked) {
                distance_right += 6;
                if (distance_right > 19) { distance_right = 19; }
                prev_char = word[word_pos];
                word_pos++;
                ci++;
                score += 21 - distance_right;
            } else {
                ok = false;
            }
        } else {
            if (word_pos >= (int)wn) {
                ok = false;
            } else {
                char wc = (char)tolower((unsigned char)word[word_pos]);
                char mc = (char)tolower((unsigned char)cc);
                if (wc != mc) {
                    ok = false;
                } else {
                    distance_right += 6;
                    if (distance_right > 19) { distance_right = 19; }
                    prev_char = word[word_pos];
                    word_pos++;
                    ci++;
                    score += 21 - distance_right;
                }
            }
        }
    }
    if (ok) {
        result.score = score;
        result.matched = true;
        if (del_fwd_pos >= 0) {
            result.del_fwd_start = del_fwd_pos;
            result.del_fwd_count = 1;
        }
    }
    return result;
}

// Best-match record returned by find_best_rule. Mirrors
// IPAPhonemizer::RuleMatchResult in phonemizer.h.
struct rule_match {
    int          score;
    struct chars phonemes;
    int          advance;
    int          del_start;
    int          del_count;
    bool         is_prefix;
    bool         is_suffix;
    int          suffix_strip_len;
    int          suffix_flags;
};

static void rule_match_free(struct rule_match * rm) {
    chars_free(&rm->phonemes);
}

// Match a single rule at `pos` in `word`. Returns the rule's score
// (>= 0) or -1 if it doesn't match. The phonemes are written to
// `*out_phonemes` and the advance + del_fwd info via out-pointers.
// The caller resets out_phonemes->count = 0 BEFORE calling; the
// function appends to it.
static int match_rule(const struct ruleset * rs,
                      const struct phoneme_rule * rule,
                      const char * word, size_t wn, int pos,
                      struct chars * out_phonemes,
                      int * advance,
                      int * del_fwd_start,
                      int * del_fwd_count,
                      int group_length,
                      const char * ph_so_far, size_t ph_n,
                      int word_alt_flags,
                      const bool * replaced_e_arr, size_t re_n,
                      bool suffix_removed) {
    int result = -1;
    const struct chars * match = &rule->match;
    int mlen = (int)match->count;
    if (pos + mlen <= (int)wn) {
        bool key_ok = true;
        for (int i = 0; i < mlen && key_ok; i++) {
            char wc = (char)tolower((unsigned char)word[pos + i]);
            char mc = (char)tolower((unsigned char)match->data[i]);
            if (wc != mc) { key_ok = false; }
        }
        if (key_ok) {
            int lscore = 0;
            bool lmatch = true;
            if (rule->left_ctx.count > 0) {
                struct ctx_score ls = match_left_context_score(
                    rule->left_ctx.data, rule->left_ctx.count,
                    word, wn, pos, rs, ph_so_far, ph_n);
                lscore = ls.score;
                lmatch = ls.matched;
            }
            if (lmatch) {
                char last_match_char = (mlen > 0)
                    ? word[pos + mlen - 1] : 0;
                struct right_ctx_score rresult = { 0, -1, 0, true };
                if (rule->right_ctx.count > 0) {
                    rresult = match_right_context_score(
                        rule->right_ctx.data, rule->right_ctx.count,
                        word, wn, pos + mlen, rs,
                        last_match_char, word_alt_flags,
                        replaced_e_arr, re_n, suffix_removed);
                }
                if (rresult.matched) {
                    int additional = mlen - group_length;
                    if (additional < 0) { additional = 0; }
                    int dialect_bonus = (rule->condition != 0) ? 1 : 0;
                    int total = 1 + additional * 21 + lscore
                              + rresult.score + dialect_bonus;
                    // Phonemes: strip everything from '$' onwards
                    // (flag annotations like "$verb").
                    size_t phn = rule->phonemes.count;
                    const char * pd = rule->phonemes.data;
                    size_t dollar = phn;
                    for (size_t k = 0; k < phn && dollar == phn; k++) {
                        if (pd[k] == '$') { dollar = k; }
                    }
                    struct chars trimmed = {0};
                    trim(pd, dollar, &trimmed);
                    chars_put(out_phonemes,
                              trimmed.data, trimmed.count);
                    chars_free(&trimmed);
                    *advance       = mlen;
                    *del_fwd_start = rresult.del_fwd_start;
                    *del_fwd_count = rresult.del_fwd_count;
                    result = total;
                }
            }
        }
    }
    return result;
}

// Try every rule under `key` against `word[pos..)`, updating
// `*best` with the highest scorer (ties: last-rule-wins). `bonus`
// is added to each rule's score (2-char group gets +35). Replaces
// the [&] try_group lambda inside findBestRule.
static void try_group(const struct ruleset * rs,
                      const char * key, size_t kn, int bonus,
                      int group_length,
                      const char * word, size_t wn, int pos,
                      int len, int word_alt_flags,
                      const bool * replaced_e, size_t re_n,
                      bool allow_suffix_strip,
                      bool suffix_phoneme_only,
                      bool suffix_removed,
                      const char * ph_so_far, size_t ph_n,
                      struct rule_match * best) {
    struct chars view = chars_view(key, kn);
    // map_get isn't const-correct in maps.c; cast away const for
    // the lookup. We don't mutate the map via the returned pointer.
    struct rules * rv = map_get(
        (struct map *)&rs->rule_groups, &view);
    if (rv != NULL) {
        for (size_t i = 0; i < rv->count; i++) {
            const struct phoneme_rule * rule = &rv->data[i];
            struct chars ph = {0};
            int adv = 0, dfs = -1, dfc = 0;
            int sc = match_rule(rs, rule, word, wn, pos,
                                &ph, &adv, &dfs, &dfc,
                                group_length, ph_so_far, ph_n,
                                word_alt_flags,
                                replaced_e, re_n, suffix_removed);
            bool valid = sc >= 0;
            bool is_ending_skip = !allow_suffix_strip
                && !suffix_phoneme_only && rule->is_suffix
                && pos + adv == len;
            if (valid && !is_ending_skip
                && sc + bonus >= best->score) {
                rule_match_free(best);
                best->score            = sc + bonus;
                best->phonemes         = ph;       // moved in
                ph = (struct chars){0};            // detach
                best->advance          = adv;
                best->del_start        = dfs;
                best->del_count        = dfc;
                best->is_prefix        = rule->is_prefix;
                best->is_suffix        = rule->is_suffix;
                best->suffix_strip_len = rule->suffix_strip_len;
                best->suffix_flags     = rule->suffix_flags;
            }
            chars_free(&ph);
        }
    }
}

// Try the 2-char group (with +35 bonus) and the 1-char group at
// `pos`. Returns the best-scoring match across both groups (ties:
// last-rule-wins via the `>=` comparison in try_group).
static struct rule_match find_best_rule(const struct ruleset * rs,
                                        const char * word, size_t wn,
                                        int pos, int len,
                                        char pos_char,
                                        int word_alt_flags,
                                        const bool * replaced_e,
                                        size_t re_n,
                                        bool allow_suffix_strip,
                                        bool suffix_phoneme_only,
                                        bool suffix_removed,
                                        const char * ph_so_far,
                                        size_t ph_n) {
    struct rule_match best = {
        -1, {0}, 1, -1, 0, false, false, 0, 0
    };
    if (pos + 1 < len) {
        char k2[2];
        k2[0] = pos_char;
        k2[1] = (char)tolower((unsigned char)word[pos + 1]);
        try_group(rs, k2, 2, 35, 2,
                  word, wn, pos, len, word_alt_flags,
                  replaced_e, re_n, allow_suffix_strip,
                  suffix_phoneme_only, suffix_removed,
                  ph_so_far, ph_n, &best);
    }
    char k1 = pos_char;
    try_group(rs, &k1, 1, 0, 1,
              word, wn, pos, len, word_alt_flags,
              replaced_e, re_n, allow_suffix_strip,
              suffix_phoneme_only, suffix_removed,
              ph_so_far, ph_n, &best);
    return best;
}

// ---------------------------------------------------------------------------
// Stress-promotion helpers (used by apply_rules).
// ---------------------------------------------------------------------------

// Search ph[0..from) backwards for the nearest non-rule-boundary
// byte. Returns the byte (or 0 if none) via *prev_ch, its index
// via *prev_pos (-1 if none). Replaces the C++ prevNonBnd `[&]`
// lambda (called 3 times in findLastStressableVowel).
static void prev_non_bnd(const char * ph, int from,
                         char * prev_ch, int * prev_pos) {
    *prev_ch = 0;
    *prev_pos = -1;
    for (int pi = from - 1; pi >= 0 && *prev_pos < 0; pi--) {
        if (ph[pi] != '\x01') {
            *prev_ch = ph[pi];
            *prev_pos = pi;
        }
    }
}

// Locate the last fully-stressable vowel in `phonemes`. Returns the
// byte index of the vowel code's first char, or -1 if none found.
// Skips rule-boundary markers ('\x01'), reduced vowels ('2'/'#'),
// and codes already marked unstressed ('%' or '=' immediately
// preceding). Steps back through multi-char diphthong codes (eI,
// aU, oU, aa, A@, e@, ...) so the index lands at the START byte.
static int find_last_stressable_vowel(const char * phonemes, size_t pn) {
    static const char sp_vowels[]  = "aAeEiIoOuUV03@";
    static const char diph_second[] = "IU";
    static const char diph_start[]  = "eaOoUAE";
    static const char at_starters[] = "AeioOU";
    int insert_at = -1;
    int slen = (int)pn;
    for (int si = slen - 1; si >= 0 && insert_at < 0; si--) {
        char sc = phonemes[si];
        if (sc != '\x01' && strchr(sp_vowels, sc) != NULL) {
            int ni = si + 1;
            while (ni < slen && phonemes[ni] == '\x01') { ni++; }
            bool is_reduced = ni < slen
                && (phonemes[ni] == '2' || phonemes[ni] == '#');
            if (!is_reduced) {
                char prev_ch = 0;
                int  prev_pos = -1;
                prev_non_bnd(phonemes, si, &prev_ch, &prev_pos);
                // Step back through multi-char vowel codes.
                if (strchr(diph_second, sc) != NULL
                    && strchr(diph_start, prev_ch) != NULL) {
                    si = prev_pos;
                    prev_non_bnd(phonemes, si, &prev_ch, &prev_pos);
                } else if (sc == 'a' && prev_ch == 'a') {
                    si = prev_pos;
                    prev_non_bnd(phonemes, si, &prev_ch, &prev_pos);
                } else if (sc == '@'
                           && strchr(at_starters, prev_ch) != NULL) {
                    si = prev_pos;
                    prev_non_bnd(phonemes, si, &prev_ch, &prev_pos);
                }
                if (prev_ch != '%' && prev_ch != '=') {
                    insert_at = si;
                }
            }
        }
    }
    return insert_at;
}

// Handle phonSTRESS_PREV: when `*emit` starts with '=' (the code-8
// stress-prev byte), retroactively promote the last preceding
// stressable vowel in `*phonemes` to primary stress. The leading
// '=' is dropped from `*emit`. Earlier '\'' marks are demoted to
// '\x02' (protected secondary) so the prosody step still runs.
static void apply_stress_prev(struct chars * emit,
                              struct chars * phonemes) {
    if (emit->count > 0 && emit->data[0] == '=') {
        // Drop leading '=' from emit.
        memmove(emit->data, emit->data + 1, emit->count - 1);
        emit->count -= 1;
        emit->data[emit->count] = '\0';
        int insert_at = find_last_stressable_vowel(
            phonemes->data, phonemes->count);
        if (insert_at >= 0) {
            char before_vowel = 0;
            for (int pi = insert_at - 1;
                 pi >= 0 && before_vowel == 0; pi--) {
                if (phonemes->data[pi] != '\x01') {
                    before_vowel = phonemes->data[pi];
                }
            }
            if (before_vowel != '\'') {
                // Insert '\'' at `insert_at`. Shift right + write.
                chars_grow(phonemes, phonemes->count + 2);
                memmove(phonemes->data + insert_at + 1,
                        phonemes->data + insert_at,
                        phonemes->count - (size_t)insert_at);
                phonemes->data[insert_at] = '\'';
                phonemes->count += 1;
                phonemes->data[phonemes->count] = '\0';
                for (int di = 0; di < insert_at; di++) {
                    if (phonemes->data[di] == '\'') {
                        phonemes->data[di] = '\x02';
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Suffix-rule + stem-lookup chain (called from apply_rules on a
// terminal RULE_ENDING match). apply_rules is forward-declared
// because the chain recurses back into it for stem rephonemization.
// ---------------------------------------------------------------------------

static void apply_rules(struct phonemizer * p,
                        const char * word_orig, size_t wn,
                        bool allow_suffix_strip,
                        int word_alt_flags_param,
                        bool suffix_phoneme_only,
                        bool suffix_removed,
                        bool * out_replaced_e, size_t out_re_cap,
                        bool * out_pos_visited, size_t out_pv_cap,
                        struct chars * out);

// Find the multi-char phoneme code starting at `ph[pos..)` in the
// big S_MC2 table; return the matched code via *out_data / *out_len
// (or single-byte fallback when nothing matches).
static const char * const S_MC2[] = {
    "aI@3","aU@r","i@3r","aI@","aI3","aU@","i@3","3:r","A:r",
    "o@r","A@r","e@r",
    "eI","aI","aU","OI","oU","tS","dZ","IR","VR",
    "e@","i@","U@","A@","O@","o@",
    "3:","A:","i:","u:","O:","e:","a:","aa",
    "@L","@2","@5",
    "I2","I#","E2","E#","e#","a#","a2","0#","02","O2","A~","O~","A#",
    "r-","w#","t#","d#","z#","t2","d2","n-","m-","l/","z/",
    NULL
};

static void find_phoneme_code(const char * ph, size_t pn, size_t pos,
                              const char ** out_data, size_t * out_len) {
    *out_data = NULL;
    *out_len = 0;
    for (int mi = 0; S_MC2[mi] != NULL && *out_data == NULL; mi++) {
        size_t mclen = strlen(S_MC2[mi]);
        if (pos + mclen <= pn
            && memcmp(ph + pos, S_MC2[mi], mclen) == 0) {
            *out_data = S_MC2[mi];
            *out_len = mclen;
        }
    }
    if (*out_data == NULL) {
        *out_data = ph + pos;
        *out_len = 1;
    }
}

// Apply Nth-vowel primary-stress placement to raw phoneme string.
// Strips ALL existing '\'' / ',' first, then inserts '\'' before
// the n-th vowel code (vowel-aware multi-char scan). Caller-owned
// out; reset to count=0 on entry not required (the function does
// its own).
static void apply_stress_position(const char * raw, size_t rn,
                                  int n, struct chars * out) {
    out->count = 0;
    // Pass 1: copy raw -> out, dropping all '\'' and ','.
    chars_grow(out, rn + 1);
    for (size_t i = 0; i < rn; i++) {
        if (raw[i] != '\'' && raw[i] != ',') {
            out->data[out->count++] = raw[i];
        }
    }
    out->data[out->count] = '\0';
    // Pass 2: scan multi-char-aware, count vowel codes, insert
    // '\'' before the n-th.
    int vowel_count = 0;
    size_t pi = 0;
    bool inserted = false;
    while (pi < out->count && !inserted) {
        char c = out->data[pi];
        if (c == '%' || c == '=' || c == '|') {
            pi++;
        } else {
            const char * code = NULL;
            size_t code_len = 0;
            find_phoneme_code(out->data, out->count, pi,
                              &code, &code_len);
            if (is_vowel_code(code, code_len)) {
                vowel_count++;
                if (vowel_count == n) {
                    chars_grow(out, out->count + 2);
                    memmove(out->data + pi + 1,
                            out->data + pi,
                            out->count - pi);
                    out->data[pi] = '\'';
                    out->count++;
                    out->data[out->count] = '\0';
                    inserted = true;
                }
            }
            if (!inserted) { pi += code_len; }
        }
    }
}

// SUFX_E: conditionally append 'e' to stem so dict / magic-e rules
// fire on the original verb form. Mutates `*stem` in place (count +
// data may grow by one byte).
static void append_magic_e_if_needed(struct phonemizer * p,
                                     struct chars * stem,
                                     const char * suffix_ph,
                                     size_t sn,
                                     int suffix_flags) {
    const int SUFX_E_BIT = 0x100;
    const int SUFX_V_BIT = 0x800;
    bool entering = (suffix_flags & SUFX_E_BIT)
                 && stem->count > 0
                 && stem->data[stem->count - 1] != 'e';
    if (entering) {
        struct chars stem_norm = {0};
        to_lower(stem->data, stem->count, &stem_norm);
        bool sfx_is_s_bare =
            (sn == 1 && suffix_ph[0] == 's')
            || (sn == 1 && suffix_ph[0] == 'z')
            || (sn == 3 && memcmp(suffix_ph, "I#z", 3) == 0)
            || (sn == 4 && memcmp(suffix_ph, "%I#z", 4) == 0);
        if (!sfx_is_s_bare && (suffix_flags & SUFX_V_BIT)) {
            // Try stem+"e" lookup in verb_dict.
            struct chars stem_e = {0};
            chars_put(&stem_e, stem_norm.data, stem_norm.count);
            chars_put_byte(&stem_e, 'e');
            if (smap_get(&p->verb_dict, stem_e.data, stem_e.count)
                != NULL) {
                chars_put_byte(stem, 'e');
            }
            chars_free(&stem_e);
        }
        if (stem->data[stem->count - 1] != 'e') {
            bool stem_bare_in_dict = false;
            if (!sfx_is_s_bare) {
                stem_bare_in_dict = smap_get(
                    &p->verb_dict, stem_norm.data,
                    stem_norm.count) != NULL;
            }
            bool blocked_by_onlys = set_has(
                &p->onlys_words, stem_norm.data, stem_norm.count)
                || set_has(&p->only_words,
                           stem_norm.data, stem_norm.count);
            if (!stem_bare_in_dict && !blocked_by_onlys) {
                stem_bare_in_dict = smap_get(
                    &p->dict, stem_norm.data, stem_norm.count) != NULL;
            }
            if (!stem_bare_in_dict) {
                static const char * const add_e_endings[] = {
                    "c", "rs", "ir", "ur", "ath", "ns", "u",
                    "spong", "rang", "larg", NULL
                };
                static const char vowels_incl_y[] = "aeiouy";
                static const char hard_cons[] = "bcdfgjklmnpqstvxz";
                bool add_e = false;
                for (int ai = 0;
                     add_e_endings[ai] != NULL && !add_e; ai++) {
                    size_t plen = strlen(add_e_endings[ai]);
                    if (stem->count >= plen
                        && memcmp(stem->data + stem->count - plen,
                                  add_e_endings[ai], plen) == 0) {
                        add_e = true;
                    }
                }
                if (!add_e && stem->count >= 2) {
                    char last = (char)tolower(
                        (unsigned char)stem->data[stem->count - 1]);
                    char prev = (char)tolower(
                        (unsigned char)stem->data[stem->count - 2]);
                    bool last_hard = strchr(hard_cons, last) != NULL;
                    bool prev_vowel = strchr(vowels_incl_y, prev)
                        != NULL;
                    bool ion_exc = stem->count >= 3
                        && memcmp(stem->data + stem->count - 3,
                                  "ion", 3) == 0;
                    add_e = last_hard && prev_vowel && !ion_exc;
                }
                if (add_e) { chars_put_byte(stem, 'e'); }
            }
        }
        chars_free(&stem_norm);
    }
}

// Result of looking up a stem in the dict chain. matched_stem may
// differ from the input stem_norm when the magic-e-stripped fallback
// fired (e.g. "tornadoe" -> "tornado"). stem_ph carries the dict
// phonemes when matched.
struct stem_lookup {
    struct chars stem_ph;
    struct chars matched_stem;
    bool         found_dict_entry;
    bool         used_onlys_bare;
};

static void stem_lookup_free(struct stem_lookup * sl) {
    chars_free(&sl->stem_ph);
    chars_free(&sl->matched_stem);
}

// Step 1-3 of stem_phoneme_from_dict: walk verb_dict / onlys_bare_
// dict / dict / magic-e fallback (dt_noe) in priority order.
// Returns a populated stem_lookup; the matched_stem field reflects
// which key won (relevant for the dt_noe path).
static struct stem_lookup lookup_stem_in_dicts(
        struct phonemizer * p,
        const char * stem_norm, size_t snn,
        bool stem_is_onlys, bool suffix_is_s,
        int suffix_flags) {
    const int SUFX_V_BIT = 0x800;
    struct stem_lookup r = { {0}, {0}, false, false };
    chars_put(&r.matched_stem, stem_norm, snn);
    // Step 1: verb_dict (when SUFX_V is set and the suffix isn't -s).
    if (!suffix_is_s && (suffix_flags & SUFX_V_BIT) != 0) {
        struct chars * v = smap_get(&p->verb_dict, stem_norm, snn);
        if (v != NULL) {
            chars_put(&r.stem_ph, v->data, v->count);
            r.found_dict_entry = true;
        }
    }
    // Step 2: onlys_bare_dict (for -s) or dict.
    if (!r.found_dict_entry) {
        if (suffix_is_s) {
            struct chars * o = smap_get(
                &p->onlys_bare_dict, stem_norm, snn);
            if (o != NULL) {
                chars_put(&r.stem_ph, o->data, o->count);
                r.used_onlys_bare = true;
            }
        }
        if (!r.used_onlys_bare) {
            struct chars * d = smap_get(&p->dict, stem_norm, snn);
            if (d != NULL && !stem_is_onlys) {
                chars_put(&r.stem_ph, d->data, d->count);
                r.found_dict_entry = true;
            }
        }
    }
    // Step 3: magic-e-stripped fallback ("tornadoe" -> "tornado").
    if (!r.found_dict_entry && !r.used_onlys_bare
        && snn > 1 && stem_norm[snn - 1] == 'e') {
        size_t snn2 = snn - 1;
        struct chars * d = smap_get(&p->dict, stem_norm, snn2);
        bool blocked = set_has(&p->onlys_words, stem_norm, snn2)
                    || set_has(&p->only_words,  stem_norm, snn2);
        if (d != NULL && !blocked) {
            chars_put(&r.stem_ph, d->data, d->count);
            r.matched_stem.count = 0;
            chars_put(&r.matched_stem, stem_norm, snn2);
            r.found_dict_entry = true;
        }
    }
    return r;
}

// Apply $N stress override on a dict hit, otherwise re-phonemize
// via apply_rules with combined $altN flags. Writes the result to
// *out (caller-owned, reset at entry).
static void apply_stem_stress_or_rules_fallback(
        struct phonemizer * p,
        const char * stem, size_t stem_n,
        const char * stem_norm, size_t snn,
        const struct stem_lookup * lookup,
        int word_alt_flags,
        const struct rule_match * match,
        struct chars * out) {
    const int SUFX_V_BIT = 0x800;
    out->count = 0;
    if (out->data != NULL) { out->data[0] = '\0'; }
    if (lookup->used_onlys_bare || lookup->found_dict_entry) {
        const struct chars * key = &lookup->matched_stem;
        if (lookup->used_onlys_bare) {
            // The onlys_bare path uses the original stem_norm key
            // for stress lookup (matched_stem == stem_norm here).
            key = &lookup->matched_stem;
        }
        chars_put(out, lookup->stem_ph.data, lookup->stem_ph.count);
        int * sp = imap_get(&p->stress_pos, key->data, key->count);
        if (sp != NULL) {
            struct chars stressed = {0};
            apply_stress_position(out->data, out->count, *sp,
                                  &stressed);
            out->count = 0;
            chars_put(out, stressed.data, stressed.count);
            chars_free(&stressed);
        }
    } else {
        // Combine stem's own $altN with parent word's; try stem+"e"
        // too ("fertil"->"fertile" inheritance).
        int * stem_own = imap_get(&p->word_alt_flags,
                                  stem_norm, snn);
        if (stem_own == NULL) {
            struct chars try_e = {0};
            chars_put(&try_e, stem_norm, snn);
            chars_put_byte(&try_e, 'e');
            stem_own = imap_get(&p->word_alt_flags,
                                try_e.data, try_e.count);
            chars_free(&try_e);
        }
        int stem_alt = stem_own != NULL ? *stem_own : 0;
        int combined = stem_alt | word_alt_flags;
        apply_rules(p, stem, stem_n,
                    true, combined,
                    false, true,
                    NULL, 0, NULL, 0,
                    out);
        // $N stress override on the result; skip noun-form-only
        // when suffix is verbal, skip $verb-flag words entirely.
        int * sp2 = imap_get(&p->stress_pos, stem_norm, snn);
        bool noun_form = set_has(&p->noun_form_stress,
                                 stem_norm, snn);
        bool verb_flag = set_has(&p->verb_flag_words,
                                 stem_norm, snn);
        bool apply_stress = sp2 != NULL
            && (!noun_form || !(match->suffix_flags & SUFX_V_BIT))
            && !verb_flag;
        if (apply_stress) {
            struct chars stressed = {0};
            apply_stress_position(out->data, out->count, *sp2,
                                  &stressed);
            out->count = 0;
            chars_put(out, stressed.data, stressed.count);
            chars_free(&stressed);
        }
    }
}

// Top-level stem-phoneme lookup. Returns true if a phoneme string
// was produced (out non-empty on success).
static bool stem_phoneme_from_dict(struct phonemizer * p,
                                   const char * stem, size_t stem_n,
                                   const struct rule_match * match,
                                   int word_alt_flags,
                                   struct chars * out) {
    bool produced = false;
    if (stem_n > 0) {
        struct chars stem_norm = {0};
        to_lower(stem, stem_n, &stem_norm);
        bool stem_is_onlys = set_has(
            &p->onlys_words, stem_norm.data, stem_norm.count);
        // -s/-es suffix detection: $onlys IS valid for -s suffix.
        bool suffix_is_s =
            (match->phonemes.count == 1
             && (match->phonemes.data[0] == 's'
              || match->phonemes.data[0] == 'z'))
            || (match->phonemes.count == 3
                && memcmp(match->phonemes.data, "I#z", 3) == 0)
            || (match->phonemes.count == 4
                && memcmp(match->phonemes.data, "%I#z", 4) == 0);
        if (suffix_is_s) { stem_is_onlys = false; }
        if (set_has(&p->only_words, stem_norm.data, stem_norm.count)
            && !suffix_is_s) {
            stem_is_onlys = true;
        }
        struct stem_lookup lookup = lookup_stem_in_dicts(
            p, stem_norm.data, stem_norm.count,
            stem_is_onlys, suffix_is_s, match->suffix_flags);
        apply_stem_stress_or_rules_fallback(
            p, stem, stem_n,
            stem_norm.data, stem_norm.count,
            &lookup, word_alt_flags, match, out);
        stem_lookup_free(&lookup);
        chars_free(&stem_norm);
        produced = true;
    }
    return produced;
}

// SUFFIX rule terminal path: strip N chars, handle SUFX_I/E flags,
// re-phonemize stem (via dict / rules), apply 'd#' devoicing,
// return stem_ph + suffix_ph via *out.
static void process_suffix_rule(struct phonemizer * p,
                                const char * word, size_t wn,
                                int word_alt_flags,
                                const struct rule_match * match,
                                struct chars * out) {
    int strip = match->suffix_strip_len;
    if (strip <= 0 || strip > (int)wn) { strip = match->advance; }
    size_t stem_n = wn - (size_t)strip;
    struct chars stem = {0};
    chars_put(&stem, word, stem_n);
    const int SUFX_I_BIT = 0x200;
    if ((match->suffix_flags & SUFX_I_BIT)
        && stem.count > 0
        && stem.data[stem.count - 1] == 'i') {
        stem.data[stem.count - 1] = 'y';
    }
    append_magic_e_if_needed(p, &stem,
                             match->phonemes.data,
                             match->phonemes.count,
                             match->suffix_flags);
    struct chars stem_ph = {0};
    stem_phoneme_from_dict(p, stem.data, stem.count, match,
                           word_alt_flags, &stem_ph);
    struct chars suffix_ph = {0};
    chars_put(&suffix_ph, match->phonemes.data, match->phonemes.count);
    devoice_ed_suffix(stem_ph.data, stem_ph.count, &suffix_ph);
    out->count = 0;
    chars_put(out, stem_ph.data, stem_ph.count);
    chars_put(out, suffix_ph.data, suffix_ph.count);
    chars_free(&stem);
    chars_free(&stem_ph);
    chars_free(&suffix_ph);
}

// ---------------------------------------------------------------------------
// Stubs for word_to_phonemes and process_phoneme_string. The real
// versions arrive in later phases (the dispatch chain + prosody
// pass). For now, word_to_phonemes routes straight through
// apply_rules + process_phoneme_string, and process_phoneme_string
// is a pass-through. This is enough to make apply_rules callable
// end-to-end for the rule-only path.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Prosody sub-steps. Each operates in-place on a `struct chars *ph`
// (the working phoneme-code buffer). Ordered by their position in
// processPhonemeString.
// ---------------------------------------------------------------------------

// Step 1: Velar nasal assimilation. n+k/g -> N+k/g (ŋ before velar
// stops). E.g. "income" -> ˈɪŋkʌm.
static void apply_velar_nasal_assimilation(struct chars * ph) {
    for (size_t i = 0; i + 1 < ph->count; i++) {
        if (ph->data[i] == 'n'
            && (ph->data[i + 1] == 'k' || ph->data[i + 1] == 'g')) {
            ph->data[i] = 'N';
        }
    }
}

// Step 2: Happy tensing -- word-final unstressed ɪ -> i (American
// English). Applies to %I, I2, and bare-I at word end (unstressed
// and not part of a diphthong).
static void apply_happy_tensing(struct chars * ph) {
    if (ph->count >= 2
        && ph->data[ph->count - 2] == '%'
        && ph->data[ph->count - 1] == 'I') {
        ph->count -= 2;
        chars_put_byte(ph, 'i');
    } else if (ph->count >= 2
               && ph->data[ph->count - 2] == 'I'
               && ph->data[ph->count - 1] == '2') {
        ph->count -= 2;
        chars_put_byte(ph, 'i');
    } else if (ph->count > 0 && ph->data[ph->count - 1] == 'I') {
        char prev = ph->count >= 2 ? ph->data[ph->count - 2] : 0;
        static const char diphthong_before[] = "eaOoU";
        bool part_of_diph = prev != 0
            && strchr(diphthong_before, prev) != NULL;
        if (prev != '\'' && prev != ',' && !part_of_diph) {
            static const char all_vowels[] = "aAeEiIOUV03o";
            int vowel_count = 0;
            for (size_t i = 0; i < ph->count; i++) {
                if (strchr(all_vowels, ph->data[i]) != NULL) {
                    vowel_count++;
                }
            }
            if (vowel_count > 1) {
                ph->data[ph->count - 1] = 'i';
            }
        }
    }
}

// Step 3: Vowel reduction -- unstressed A: / A -> @ (American
// English; preserve %A@ rhotic diphthong).
static void apply_vowel_reduction(struct chars * ph) {
    struct repl {
        const char * from;
        size_t       from_n;
        const char * to;
        size_t       to_n;
        char         not_followed_by;
    };
    static const struct repl reductions[] = {
        { "%A:", 3, "%@", 2, 0   },
        { "=A:", 3, "%@", 2, 0   },
        { "%A",  2, "%@", 2, '@' },
        { "=A",  2, "%@", 2, '@' },
    };
    for (size_t ri = 0;
         ri < sizeof(reductions) / sizeof(reductions[0]); ri++) {
        const struct repl * r = &reductions[ri];
        size_t rpos = 0;
        while (rpos + r->from_n <= ph->count) {
            bool hit = memcmp(ph->data + rpos, r->from, r->from_n)
                       == 0;
            size_t after = rpos + r->from_n;
            bool blocked = hit && r->not_followed_by != 0
                && after < ph->count
                && ph->data[after] == r->not_followed_by;
            if (hit && !blocked) {
                if (r->to_n == r->from_n) {
                    memcpy(ph->data + rpos, r->to, r->to_n);
                    rpos += r->to_n;
                } else if (r->to_n < r->from_n) {
                    memcpy(ph->data + rpos, r->to, r->to_n);
                    memmove(ph->data + rpos + r->to_n,
                            ph->data + rpos + r->from_n,
                            ph->count - rpos - r->from_n);
                    ph->count -= r->from_n - r->to_n;
                    ph->data[ph->count] = '\0';
                    rpos += r->to_n;
                } else {
                    size_t grow = r->to_n - r->from_n;
                    chars_grow(ph, ph->count + grow + 1);
                    memmove(ph->data + rpos + r->to_n,
                            ph->data + rpos + r->from_n,
                            ph->count - rpos - r->from_n);
                    memcpy(ph->data + rpos, r->to, r->to_n);
                    ph->count += grow;
                    ph->data[ph->count] = '\0';
                    rpos += r->to_n;
                }
            } else if (blocked) {
                rpos = after;
            } else {
                rpos++;
            }
        }
    }
}

// Step 3b: LOT+R -> THOUGHT+R in American English. "0r" (ɑːɹ)
// becomes "O:r" (ɔːɹ).
static void apply_lot_plus_r_merge(struct chars * ph) {
    size_t rpos = 0;
    while (rpos + 2 <= ph->count) {
        if (ph->data[rpos] == '0' && ph->data[rpos + 1] == 'r') {
            chars_grow(ph, ph->count + 2);
            memmove(ph->data + rpos + 3,
                    ph->data + rpos + 2,
                    ph->count - rpos - 2);
            ph->data[rpos]     = 'O';
            ph->data[rpos + 1] = ':';
            ph->data[rpos + 2] = 'r';
            ph->count += 1;
            ph->data[ph->count] = '\0';
            rpos += 3;
        } else {
            rpos++;
        }
    }
}

// Step 3c: Strip morpheme-boundary schwa before r: @-r -> r. The
// '@' and '-' are elided; the 'r' stays.
static void strip_morpheme_schwa_r(struct chars * ph) {
    size_t i = 0;
    while (i + 2 < ph->count) {
        if (ph->data[i] == '@'
            && ph->data[i + 1] == '-'
            && ph->data[i + 2] == 'r') {
            memmove(ph->data + i, ph->data + i + 2,
                    ph->count - i - 2);
            ph->count -= 2;
            ph->data[ph->count] = '\0';
        } else {
            i++;
        }
    }
}

// Step 4: Bare schwa '@' before 'r' -> r-colored schwa '3' (ɚ).
// Pre-pass: 'a#r' -> '@r' (unstressed 'a' before 'r' behaves like
// schwa-before-r). The 'r' is absorbed into '3' when followed by
// a consonant / end-of-word, OR when '@' was unstressed-prefixed
// ('%'/'=' immediately before).
static void apply_bare_schwa_to_rhotic(struct chars * ph) {
    // Pre-pass: a#r -> @r.
    for (size_t i = 0; i + 2 < ph->count; i++) {
        if (ph->data[i] == 'a'
            && ph->data[i + 1] == '#'
            && ph->data[i + 2] == 'r') {
            ph->data[i] = '@';
            // Erase '#' (index i+1).
            memmove(ph->data + i + 1, ph->data + i + 2,
                    ph->count - i - 2);
            ph->count -= 1;
            ph->data[ph->count] = '\0';
        }
    }
    // Main pass.
    for (size_t rpos = 0; rpos + 1 < ph->count; rpos++) {
        if (ph->data[rpos] == '@') {
            // Find 'r' after '@', skipping stress/modifier marks.
            size_t r_pos = rpos + 1;
            while (r_pos < ph->count
                   && (ph->data[r_pos] == '\''
                    || ph->data[r_pos] == ','
                    || ph->data[r_pos] == '%'
                    || ph->data[r_pos] == '=')) {
                r_pos++;
            }
            if (r_pos < ph->count && ph->data[r_pos] == 'r') {
                bool is_diphthong = rpos > 0 && (
                    ph->data[rpos - 1] == 'o'
                 || ph->data[rpos - 1] == 'A'
                 || ph->data[rpos - 1] == 'U'
                 || ph->data[rpos - 1] == 'O'
                 || ph->data[rpos - 1] == 'e'
                 || ph->data[rpos - 1] == 'i'
                 || ph->data[rpos - 1] == 'I'
                 || ph->data[rpos - 1] == 'a');
                if (!is_diphthong) {
                    ph->data[rpos] = '3';
                    // Absorb 'r' if followed by a consonant / end,
                    // OR if '@' was unstressed-prefixed.
                    size_t after_r = r_pos + 1;
                    while (after_r < ph->count
                           && (ph->data[after_r] == '%'
                            || ph->data[after_r] == '='
                            || ph->data[after_r] == '\''
                            || ph->data[after_r] == ',')) {
                        after_r++;
                    }
                    static const char vowel_starts[] =
                        "aAeEiIoOuUV03@";
                    bool next_is_vowel = after_r < ph->count
                        && strchr(vowel_starts,
                                  ph->data[after_r]) != NULL;
                    bool unstressed_pre = rpos > 0
                        && (ph->data[rpos - 1] == '='
                         || ph->data[rpos - 1] == '%');
                    if (!next_is_vowel || unstressed_pre) {
                        // Absorb 'r' into '3'.
                        memmove(ph->data + r_pos,
                                ph->data + r_pos + 1,
                                ph->count - r_pos - 1);
                        ph->count -= 1;
                        ph->data[ph->count] = '\0';
                    }
                }
            }
        }
    }
}

// Step 4b: Linking-R. After '3', '3:', 'U@', or 'A@', when followed
// (across stress markers) by a vowel-starting code, insert 'r'.
static void apply_linking_r(struct chars * ph) {
    static const char vowel_starts[] = "aAeEiIoOuUV03@";
    for (size_t rpos = 0; rpos < ph->count; rpos++) {
        int code_len = 0;
        if (ph->data[rpos] == '3') {
            code_len = 1;
            if (rpos + 1 < ph->count && ph->data[rpos + 1] == ':') {
                code_len = 2;
            }
        } else if (ph->data[rpos] == 'U'
                   && rpos + 1 < ph->count
                   && ph->data[rpos + 1] == '@') {
            code_len = 2;
        } else if (ph->data[rpos] == 'A'
                   && rpos + 1 < ph->count
                   && ph->data[rpos + 1] == '@') {
            code_len = 2;
        }
        if (code_len > 0) {
            size_t after_code = rpos + (size_t)code_len;
            bool already_r = after_code < ph->count
                && ph->data[after_code] == 'r';
            if (!already_r) {
                size_t after = after_code;
                while (after < ph->count
                       && (ph->data[after] == '\''
                        || ph->data[after] == ','
                        || ph->data[after] == '%'
                        || ph->data[after] == '=')) {
                    after++;
                }
                bool next_vowel = after < ph->count
                    && strchr(vowel_starts,
                              ph->data[after]) != NULL;
                if (next_vowel) {
                    // Insert 'r' at after_code.
                    chars_grow(ph, ph->count + 2);
                    memmove(ph->data + after_code + 1,
                            ph->data + after_code,
                            ph->count - after_code);
                    ph->data[after_code] = 'r';
                    ph->count += 1;
                    ph->data[ph->count] = '\0';
                    rpos += (size_t)code_len;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Step 4c/d/e: -tion / -ology / -ic stress rebalances.
// ---------------------------------------------------------------------------

// Tail after the -tion 'n' is terminal iff it consists only of
// modifier chars (%/=/-/'/,), terminal 'i' (%i / =i), or plural
// 'z'. Distinguishes "-tion"/"-tions" from "-ciency".
static bool is_tion_terminal_after_n(const char * ph, size_t pn,
                                     size_t after_N) {
    bool truly_terminal = true;
    size_t ki = after_N;
    while (ki < pn && truly_terminal) {
        char c = ph[ki];
        bool is_modifier = c == '%' || c == '=' || c == '-'
                        || c == '\'' || c == ',';
        if (!is_modifier && c != 'i' && c != 'z') {
            truly_terminal = false;
        }
        ki++;
    }
    return truly_terminal;
}

// Scan up to 10 chars after S for the -tion '@n' pattern with a
// terminal tail.
static bool has_tion_after_s(const char * ph, size_t pn,
                             size_t after_S) {
    bool found_tion = false;
    bool stop = false;
    size_t ki = after_S;
    while (ki < pn && ki < after_S + 10 && !found_tion && !stop) {
        char c = ph[ki];
        bool is_modifier = (c == '=' || c == '%' || c == '-'
                            || c == '\'' || c == ',');
        if (c == '@') {
            // Inner scan up to 4 chars (ki2 < ki + 4) for 'n'/'N'
            // with modifier skip-set {-, =, %} (matches .cpp).
            size_t ki2 = ki + 1;
            bool n_found = false;
            bool n_stop = false;
            while (ki2 < pn && ki2 < ki + 4
                   && !n_found && !n_stop) {
                char c2 = ph[ki2];
                bool c2_modifier = (c2 == '-' || c2 == '='
                                    || c2 == '%');
                if (c2 == 'n' || c2 == 'N') {
                    if (is_tion_terminal_after_n(ph, pn,
                                                 ki2 + 1)) {
                        found_tion = true;
                    }
                    n_found = true;
                } else if (!c2_modifier) {
                    n_stop = true;
                }
                ki2++;
            }
            stop = true;  // saw '@'; stop outer scan whether matched
        } else if (!is_modifier) {
            stop = true;  // non-modifier, non-@: not a -tion suffix
        }
        ki++;
    }
    return found_tion;
}

static size_t find_tion_suffix_s_pos(const char * ph, size_t pn) {
    size_t s_pos = (size_t)-1;
    size_t k = pn;
    while (k > 0 && s_pos == (size_t)-1) {
        size_t idx = k - 1;
        if (ph[idx] == 'S' && has_tion_after_s(ph, pn, idx + 1)) {
            s_pos = idx;
        }
        k--;
    }
    return s_pos;
}

static bool primary_already_on_vowel(const char * ph, size_t pn,
                                     size_t vowel_pos) {
    (void)pn;
    static const char vowel_chars[] = "aAeEiIoOuUV03@";
    bool primary = false;
    bool stop = false;
    int bi = (int)vowel_pos - 1;
    while (bi >= 0 && !primary && !stop) {
        char bc = ph[bi];
        if (bc == '\'') {
            primary = true;
        } else if (strchr(vowel_chars, bc) != NULL) {
            stop = true;
        }
        bi--;
    }
    return primary;
}

static void move_stress_to_vowel(struct chars * ph, size_t vowel_pos) {
    char * prime = memchr(ph->data, '\'', ph->count);
    size_t prime_pos = prime != NULL
        ? (size_t)(prime - ph->data) : (size_t)-1;
    bool primary_after = prime_pos != (size_t)-1
        && prime_pos >= vowel_pos;
    if (!primary_after) {
        if (prime_pos != (size_t)-1) {
            ph->data[prime_pos] = ','; // demote
        }
        if (vowel_pos > 0 && ph->data[vowel_pos - 1] == ',') {
            memmove(ph->data + vowel_pos - 1,
                    ph->data + vowel_pos,
                    ph->count - vowel_pos);
            ph->count -= 1;
            ph->data[ph->count] = '\0';
            vowel_pos--;
        }
        // Insert '\'' at vowel_pos.
        chars_grow(ph, ph->count + 2);
        memmove(ph->data + vowel_pos + 1,
                ph->data + vowel_pos,
                ph->count - vowel_pos);
        ph->data[vowel_pos] = '\'';
        ph->count += 1;
        ph->data[ph->count] = '\0';
    }
}

// Multi-char vowel-code-aware vowel count in ph[0..end). Writes
// the count + last-vowel-position out.
static void count_vowels_before_unit_aware(const char * ph, size_t pn,
                                           size_t end,
                                           int * out_count,
                                           size_t * out_last_pos) {
    static const char * const mc[] = {
        "aI@3","aU@r","i@3r","aI@","aI3","aU@","i@3",
        "3:r","A:r","o@r","A@r","e@r",
        "eI","aI","aU","OI","oU","IR","VR",
        "e@","i@","U@","A@","O@","o@",
        "3:","A:","i:","u:","O:","e:","a:","aa",
        "@L","@2","@5",
        "I2","I#","E2","E#","e#","a#","a2","0#","02","O2",
        "A~","O~","A#",
        NULL
    };
    (void)pn;
    static const char vowel_c[] = "aAeEiIoOuUV03@";
    *out_count = 0;
    *out_last_pos = (size_t)-1;
    size_t vi = 0;
    while (vi < end) {
        char c = ph[vi];
        if (c == '\'' || c == ',' || c == '%' || c == '=') {
            vi++;
        } else {
            const char * code = NULL;
            size_t code_len = 0;
            for (int mi = 0; mc[mi] != NULL && code == NULL; mi++) {
                size_t mcl = strlen(mc[mi]);
                if (vi + mcl <= end
                    && memcmp(ph + vi, mc[mi], mcl) == 0) {
                    code = mc[mi];
                    code_len = mcl;
                }
            }
            if (code == NULL) {
                code = ph + vi;
                code_len = 1;
            }
            if (strchr(vowel_c, code[0]) != NULL) {
                (*out_count)++;
                *out_last_pos = vi;
            }
            vi += code_len;
        }
    }
}

static void apply_tion_stress_fix(struct chars * ph) {
    size_t s_pos = find_tion_suffix_s_pos(ph->data, ph->count);
    if (s_pos != (size_t)-1) {
        int v_count = 0;
        size_t vowel_pos = (size_t)-1;
        count_vowels_before_unit_aware(ph->data, ph->count, s_pos,
                                       &v_count, &vowel_pos);
        if (vowel_pos != (size_t)-1
            && !primary_already_on_vowel(ph->data, ph->count,
                                         vowel_pos)) {
            move_stress_to_vowel(ph, vowel_pos);
        }
    }
}

// "-ology" pattern at position k: '0l' + optional modifiers + '@dZ'.
static bool matches_ology_at(const char * ph, size_t pn, size_t k) {
    bool matched = false;
    if (k + 2 < pn && ph[k] == '0' && ph[k + 1] == 'l') {
        size_t a = k + 2;
        while (a < pn
               && (ph[a] == '=' || ph[a] == '%' || ph[a] == ',')) {
            a++;
        }
        matched = a + 2 < pn
            && ph[a] == '@' && ph[a + 1] == 'd' && ph[a + 2] == 'Z';
    }
    return matched;
}

static void apply_ology_stress_fix(struct chars * ph) {
    size_t ol_pos = (size_t)-1;
    for (size_t k = 0;
         k + 2 < ph->count && ol_pos == (size_t)-1; k++) {
        if (matches_ology_at(ph->data, ph->count, k)) {
            ol_pos = k;
        }
    }
    if (ol_pos != (size_t)-1) {
        bool primary_on_ol = ol_pos > 0
            && ph->data[ol_pos - 1] == '\'';
        if (!primary_on_ol) {
            char * prime = memchr(ph->data, '\'', ph->count);
            if (prime != NULL
                && (size_t)(prime - ph->data) < ol_pos) {
                *prime = ',';
            }
            if (ol_pos > 0 && ph->data[ol_pos - 1] == ',') {
                memmove(ph->data + ol_pos - 1,
                        ph->data + ol_pos,
                        ph->count - ol_pos);
                ph->count -= 1;
                ph->data[ph->count] = '\0';
                ol_pos--;
            }
            chars_grow(ph, ph->count + 2);
            memmove(ph->data + ol_pos + 1,
                    ph->data + ol_pos,
                    ph->count - ol_pos);
            ph->data[ol_pos] = '\'';
            ph->count += 1;
            ph->data[ph->count] = '\0';
        }
    }
}

// "-ic"/"-ical"/"-ics" stress fix.
static void apply_ic_stress_fix(struct chars * ph) {
    size_t ik_pos = (size_t)-1;
    size_t k_pos = (size_t)-1;
    for (size_t i = ph->count; i > 0; i--) {
        if (ph->data[i - 1] == 'k') {
            k_pos = i - 1;
            break;
        }
    }
    if (k_pos != (size_t)-1 && k_pos >= 1
        && ph->data[k_pos - 1] == 'I') {
        bool i_unstressed = (k_pos >= 2
            && (ph->data[k_pos - 2] == '='
             || ph->data[k_pos - 2] == '%'))
            || k_pos == 1;
        if (i_unstressed) {
            ik_pos = k_pos - 2;
        }
    }
    if (ik_pos != (size_t)-1) {
        int vowels_before = 0;
        size_t last_vowel_pos = (size_t)-1;
        count_vowels_before_unit_aware(ph->data, ph->count, ik_pos,
                                       &vowels_before,
                                       &last_vowel_pos);
        bool already_on_last = last_vowel_pos != (size_t)-1
            && last_vowel_pos > 0
            && ph->data[last_vowel_pos - 1] == '\'';
        if (vowels_before >= 2
            && last_vowel_pos != (size_t)-1
            && !already_on_last) {
            move_stress_to_vowel(ph, last_vowel_pos);
        }
    }
}

// ---------------------------------------------------------------------------
// strip_rule_boundary_markers: removes '\x01' bytes from `*ph` and
// writes a parallel `bool *rba` array of size *out_n marking
// positions that had a rule boundary right after them. *out_n is
// the new ph length on return (or 0 when no \x01 was present, in
// which case *rba is left untouched).
// ---------------------------------------------------------------------------

static void strip_rule_boundary_markers(struct chars * ph,
                                        bool * rba, size_t rba_cap,
                                        size_t * out_n) {
    *out_n = 0;
    bool any = memchr(ph->data, '\x01', ph->count) != NULL;
    if (any) {
        size_t w = 0;
        for (size_t r = 0; r < ph->count; r++) {
            if (ph->data[r] == '\x01') {
                if (w > 0 && w - 1 < rba_cap) {
                    rba[w - 1] = true;
                }
            } else {
                if (w < rba_cap) { rba[w] = false; }
                ph->data[w++] = ph->data[r];
            }
        }
        ph->count = w;
        ph->data[ph->count] = '\0';
        *out_n = w;
    }
}

// ---------------------------------------------------------------------------
// Late-phase reductions (steps 5.5b2 .. 6f). All operate in-place
// on *ph. Most read 1-2 stress markers and walk a substring; the
// big ones (reduceEUnstressed, reduceABeforePrimary,
// reduceZeroDiminished) need the multi-char-aware vowel scanner.
// ---------------------------------------------------------------------------

// Step 5.5: bare '0' -> '@' only when ph contains "u:S" (the
// 4c -ution pattern fired).
static void reduce_bare_zero_after_ution(struct chars * ph) {
    bool has_uS = false;
    for (size_t i = 0; i + 2 < ph->count && !has_uS; i++) {
        if (ph->data[i] == 'u' && ph->data[i + 1] == ':'
            && ph->data[i + 2] == 'S') {
            has_uS = true;
        }
    }
    if (has_uS) {
        for (size_t pi = 0; pi < ph->count; pi++) {
            if (ph->data[pi] == '0') {
                bool stressed = pi > 0
                    && (ph->data[pi - 1] == '\''
                     || ph->data[pi - 1] == ',');
                if (!stressed) { ph->data[pi] = '@'; }
            }
        }
    }
}

// Step 5.5b2: bare 'V' -> '@' between ',' and '\'' (in that order).
static void reduce_v_between_sec_and_primary(struct chars * ph) {
    char * prim = memchr(ph->data, '\'', ph->count);
    char * sec  = memchr(ph->data, ',',  ph->count);
    if (prim != NULL && sec != NULL && sec < prim) {
        size_t pp = (size_t)(prim - ph->data);
        size_t sp = (size_t)(sec  - ph->data);
        for (size_t pi = sp + 1; pi < pp; pi++) {
            if (ph->data[pi] == 'V') {
                bool stressed = pi > 0
                    && (ph->data[pi - 1] == '\''
                     || ph->data[pi - 1] == ',');
                if (!stressed) { ph->data[pi] = '@'; }
            }
        }
    }
}

// Step 5.5c2: bare 'a' -> 'a#' between '\'' and ',' (rule-derived).
static void reduce_a_between_primary_and_sec(struct chars * ph,
                                             bool rule_derived) {
    if (!rule_derived) { return; }
    char * prim = memchr(ph->data, '\'', ph->count);
    char * sec  = memchr(ph->data, ',',  ph->count);
    if (prim != NULL && sec != NULL && prim < sec) {
        size_t pp = (size_t)(prim - ph->data);
        size_t sp = (size_t)(sec  - ph->data);
        for (size_t pi = pp + 1; pi < sp; pi++) {
            if (ph->data[pi] == 'a') {
                bool is_diphthong_start = pi + 1 < ph->count
                    && (ph->data[pi + 1] == 'I'
                     || ph->data[pi + 1] == 'U'
                     || ph->data[pi + 1] == ':'
                     || ph->data[pi + 1] == '@'
                     || ph->data[pi + 1] == '#');
                if (!is_diphthong_start) {
                    bool stressed = pi > 0
                        && (ph->data[pi - 1] == '\''
                         || ph->data[pi - 1] == ',');
                    if (!stressed) {
                        chars_grow(ph, ph->count + 2);
                        memmove(ph->data + pi + 2,
                                ph->data + pi + 1,
                                ph->count - pi - 1);
                        ph->data[pi + 1] = '#';
                        ph->count += 1;
                        ph->data[ph->count] = '\0';
                        sp++;
                        pi++;
                    }
                }
            }
        }
    }
}

// Step 5.5d: bare '0' -> '@' between ',' and '\''.
static void reduce_zero_between_sec_and_primary(struct chars * ph) {
    char * prim = memchr(ph->data, '\'', ph->count);
    char * sec  = memchr(ph->data, ',',  ph->count);
    if (prim != NULL && sec != NULL && sec < prim) {
        size_t pp = (size_t)(prim - ph->data);
        size_t sp = (size_t)(sec  - ph->data);
        for (size_t pi = sp + 1; pi < pp; pi++) {
            if (ph->data[pi] == '0') {
                bool is_variant = pi + 1 < ph->count
                    && (ph->data[pi + 1] == '#'
                     || ph->data[pi + 1] == '2');
                if (is_variant) {
                    pi++;
                } else {
                    bool stressed = pi > 0
                        && (ph->data[pi - 1] == '\''
                         || ph->data[pi - 1] == ',');
                    if (!stressed) { ph->data[pi] = '@'; }
                }
            }
        }
    }
}

// Step 5.5d2: mirror of 5.5d. '\'' before the first ',' AFTER it.
static void reduce_zero_between_primary_and_sec(struct chars * ph) {
    char * prim = memchr(ph->data, '\'', ph->count);
    if (prim != NULL) {
        size_t pp = (size_t)(prim - ph->data);
        char * sec = memchr(ph->data + pp + 1, ',',
                            ph->count - pp - 1);
        if (sec != NULL) {
            size_t sp = (size_t)(sec - ph->data);
            for (size_t pi = pp + 1; pi < sp; pi++) {
                if (ph->data[pi] == '0') {
                    bool is_variant = pi + 1 < ph->count
                        && (ph->data[pi + 1] == '#'
                         || ph->data[pi + 1] == '2');
                    if (is_variant) {
                        pi++;
                    } else {
                        bool stressed = pi > 0
                            && (ph->data[pi - 1] == '\''
                             || ph->data[pi - 1] == ',');
                        if (!stressed) { ph->data[pi] = '@'; }
                    }
                }
            }
        }
    }
}

// Step 5.5e helper: returns true iff the FIRST vowel code in
// ph[start..end) is unstressed.
static bool first_inter_vowel_is_unstressed(const char * ph,
                                            size_t start, size_t end) {
    static const char vow[] = "aAeEiIoOuUV03@";
    bool unstressed = false;
    bool found = false;
    size_t si = start;
    while (si < end && !found) {
        char c = ph[si];
        if (strchr(vow, c) != NULL) {
            bool sv = si > 0
                && (ph[si - 1] == '\'' || ph[si - 1] == ',');
            if (!sv) { unstressed = true; }
            found = true;
        }
        si++;
    }
    return unstressed;
}

// Step 5.5e: '0#' -> '@' before primary, unless heavy-syllable.
static void reduce_zero_hash_before_primary(struct chars * ph) {
    char * prim = memchr(ph->data, '\'', ph->count);
    if (prim != NULL) {
        size_t pp = (size_t)(prim - ph->data);
        for (size_t pi = 0; pi < pp; pi++) {
            if (ph->data[pi] == '0') {
                bool stressed = pi > 0
                    && (ph->data[pi - 1] == '\''
                     || ph->data[pi - 1] == ',');
                bool has_hash = pi + 1 < ph->count
                    && ph->data[pi + 1] == '#';
                bool is_02 = pi + 1 < ph->count
                    && ph->data[pi + 1] == '2';
                if (!stressed && is_02) {
                    pi++;
                } else if (!stressed && has_hash
                    && !first_inter_vowel_is_unstressed(
                        ph->data, pi + 2, pp)) {
                    // '0#' -> '@'.
                    memmove(ph->data + pi + 1,
                            ph->data + pi + 2,
                            ph->count - pi - 2);
                    ph->data[pi] = '@';
                    ph->count -= 1;
                    ph->data[ph->count] = '\0';
                    pp -= 1;
                }
            }
        }
    }
}

// Step 5.5f: bare '0' -> '@' pre-tonic, standalone rule output.
static void reduce_bare_zero_before_primary(struct chars * ph,
                                            const bool * rba,
                                            size_t rba_n) {
    if (rba_n == 0) { return; }
    char * prim = memchr(ph->data, '\'', ph->count);
    if (prim != NULL) {
        size_t pp = (size_t)(prim - ph->data);
        static const char vow[] = "aAeEiIoOuUV03@";
        for (size_t pi = 0; pi < pp; pi++) {
            if (ph->data[pi] == '0') {
                bool is_variant = pi + 1 < ph->count
                    && (ph->data[pi + 1] == '#'
                     || ph->data[pi + 1] == '2');
                if (is_variant) {
                    pi++;
                } else {
                    bool marked = pi > 0
                        && (ph->data[pi - 1] == '\''
                         || ph->data[pi - 1] == ','
                         || ph->data[pi - 1] == '%'
                         || ph->data[pi - 1] == '=');
                    bool is_standalone = pi < rba_n && rba[pi];
                    bool has_prior_vowel = false;
                    for (size_t k = 0; k < pi && !has_prior_vowel; k++) {
                        if (strchr(vow, ph->data[k]) != NULL) {
                            has_prior_vowel = true;
                        }
                    }
                    if (!marked && is_standalone && has_prior_vowel) {
                        ph->data[pi] = '@';
                    }
                }
            }
        }
    }
}

// Step 5.5g: bare 'E' -> '@' pre-tonic after '%'-syll + nasal 'n'.
static void reduce_e_pre_tonic_after_pct_nasal(struct chars * ph) {
    static const char vow[] = "aAeEiIoOuUV03@";
    char * prim = memchr(ph->data, '\'', ph->count);
    if (prim != NULL) {
        size_t pp = (size_t)(prim - ph->data);
        for (size_t pi = 1; pi < pp; pi++) {
            if (ph->data[pi] == 'E') {
                bool is_variant = pi + 1 < ph->count
                    && (ph->data[pi + 1] == '2'
                     || ph->data[pi + 1] == '#');
                if (is_variant) {
                    pi++;
                } else if (ph->data[pi - 1] != '\''
                    && ph->data[pi - 1] != ','
                    && pi + 1 < ph->count
                    && ph->data[pi + 1] == 'n') {
                    // Last '%' before this 'E'.
                    size_t pct_pos = (size_t)-1;
                    for (size_t j = 0; j < pi; j++) {
                        if (ph->data[j] == '%') { pct_pos = j; }
                    }
                    if (pct_pos != (size_t)-1) {
                        bool has_vowel_between = false;
                        for (size_t k = pct_pos + 1;
                             k < pi && !has_vowel_between; k++) {
                            if (strchr(vow, ph->data[k]) != NULL) {
                                has_vowel_between = true;
                            }
                        }
                        if (has_vowel_between) { ph->data[pi] = '@'; }
                    }
                }
            }
        }
    }
}

// Step 5b: post-stress 'I' -> 'i' before '@L' (syllabic L).
static void post_stress_i_before_syllabic_l(struct chars * ph) {
    char * stress = memchr(ph->data, '\'', ph->count);
    if (stress == NULL) { return; }
    size_t sp = (size_t)(stress - ph->data);
    // Find @L.
    size_t al_pos = (size_t)-1;
    for (size_t i = 0; i + 1 < ph->count; i++) {
        if (ph->data[i] == '@' && ph->data[i + 1] == 'L') {
            al_pos = i;
            break;
        }
    }
    if (al_pos == (size_t)-1 || al_pos <= sp) { return; }
    static const char step5b_vowels[] = "aAeEiIoOuUV03@";
    static const char diph_before[] = "aAeEoOuU";
    for (size_t pi = sp + 1; pi < al_pos; pi++) {
        bool is_plain_I = ph->data[pi] == 'I'
            && !(pi + 1 < ph->count
                 && (ph->data[pi + 1] == '2'
                  || ph->data[pi + 1] == '#'));
        if (is_plain_I) {
            bool seen_stressed_vowel = false;
            for (size_t k = sp + 1; k < pi && !seen_stressed_vowel; k++) {
                if (strchr(step5b_vowels, ph->data[k]) != NULL) {
                    seen_stressed_vowel = true;
                }
            }
            if (seen_stressed_vowel) {
                bool directly_before_al = pi + 2 < ph->count
                    && ph->data[pi + 1] == '@'
                    && ph->data[pi + 2] == 'L';
                bool part_of_diph = pi > 0
                    && strchr(diph_before, ph->data[pi - 1]) != NULL;
                if (directly_before_al && !part_of_diph) {
                    ph->data[pi] = 'i';
                }
            }
        }
    }
}

// Step 5c: 3: -> 3 in pre-tonic / inter-stress positions.
static void reduce_3_long_to_short(struct chars * ph) {
    char * stress = memchr(ph->data, '\'', ph->count);
    if (stress == NULL) { return; }
    size_t prime_pos = (size_t)(stress - ph->data);
    static const char vow[] = "aAeEiIoOuUV03@";
    size_t pi = 0;
    while (pi + 1 < ph->count) {
        if (ph->data[pi] == '3' && ph->data[pi + 1] == ':') {
            bool is_stressed = pi > 0
                && (ph->data[pi - 1] == '\''
                 || ph->data[pi - 1] == ',');
            bool has_explicit_unstress = pi > 0
                && ph->data[pi - 1] == '%';
            bool is_pretonic = pi + 2 < prime_pos;
            bool has_secondary_after = memchr(
                ph->data + pi + 2, ',', ph->count - pi - 2) != NULL;
            bool pct_before_primary = prime_pos > 0
                && ph->data[prime_pos - 1] == '%';
            bool has_vowel_before = false;
            for (size_t k = 0; k < pi && !has_vowel_before; k++) {
                if (strchr(vow, ph->data[k]) != NULL) {
                    has_vowel_before = true;
                }
            }
            if (!is_stressed && !has_explicit_unstress
                && has_vowel_before
                && (is_pretonic
                 || (has_secondary_after && !pct_before_primary))) {
                memmove(ph->data + pi + 1,
                        ph->data + pi + 2,
                        ph->count - pi - 2);
                ph->count -= 1;
                ph->data[ph->count] = '\0';
                if (prime_pos > 0) { prime_pos--; }
            } else {
                pi += 2;
            }
        } else {
            pi++;
        }
    }
}

// Step 6: American English flap rule (/t/ -> [*] between a vowel
// and an unstressed vowel). De-flap '*3n/m/N' -> 't3n/m/N' first.
static void apply_flap_rule(struct chars * ph) {
    // De-tap pre-pass.
    for (size_t pi = 0; pi + 2 < ph->count; pi++) {
        if (ph->data[pi] == '*' && ph->data[pi + 1] == '3'
            && (ph->data[pi + 2] == 'n'
             || ph->data[pi + 2] == 'm'
             || ph->data[pi + 2] == 'N')) {
            ph->data[pi] = 't';
        }
    }
    static const char vowel_chars[] = "aAeEIiOUVu03@oY";
    for (size_t pi = 1; pi + 1 < ph->count; pi++) {
        if (ph->data[pi] == 't') {
            char prev = ph->data[pi - 1];
            bool prev_vowel = prev == ':' || prev == '#' || prev == 'r'
                || strchr(vowel_chars, prev) != NULL
                || (prev == '2' && pi >= 2
                    && strchr(vowel_chars, ph->data[pi - 2]) != NULL)
                || (prev == 'L' && pi >= 2 && ph->data[pi - 2] == '@');
            if (prev_vowel) {
                size_t nxt_pos = ph->data[pi + 1] == '#'
                    ? pi + 2 : pi + 1;
                char nxt = nxt_pos < ph->count
                    ? ph->data[nxt_pos] : 0;
                bool next_unstressed_vowel = false;
                if (nxt == '%' || nxt == '=') {
                    if (nxt_pos + 1 < ph->count
                        && strchr(vowel_chars,
                                  ph->data[nxt_pos + 1]) != NULL) {
                        bool v2_long = nxt_pos + 2 < ph->count
                            && ph->data[nxt_pos + 2] == ':';
                        int n2p = v2_long ? (int)(nxt_pos + 3)
                                          : (int)(nxt_pos + 2);
                        bool n2_n = n2p < (int)ph->count
                            && ph->data[n2p] == 'n';
                        bool v2_3c = ph->data[nxt_pos + 1] == '3'
                            && v2_long;
                        next_unstressed_vowel = !(n2_n && !v2_3c);
                    }
                } else if (nxt != 0 && nxt != '\'' && nxt != ','
                    && strchr(vowel_chars, nxt) != NULL) {
                    bool is_3colon = nxt == '3'
                        && nxt_pos + 1 < ph->count
                        && ph->data[nxt_pos + 1] == ':';
                    bool vowel_is_long = nxt_pos + 1 < ph->count
                        && ph->data[nxt_pos + 1] == ':';
                    int next2_pos = vowel_is_long
                        ? (int)(nxt_pos + 2) : (int)(nxt_pos + 1);
                    if (nxt == '@' && next2_pos < (int)ph->count
                        && ph->data[next2_pos] == 'L') {
                        next2_pos++;
                    }
                    bool next2_is_n = next2_pos < (int)ph->count
                        && ph->data[next2_pos] == 'n';
                    next_unstressed_vowel = !(next2_is_n && !is_3colon);
                }
                if (next_unstressed_vowel) { ph->data[pi] = '*'; }
            }
        }
    }
}

// Step 6b: '-ness' suffix word-end normalisation.
static void reduce_ness_suffix(struct chars * ph) {
    size_t plen = ph->count;
    if (plen >= 4 && ph->data[plen - 1] == 's'
        && ph->data[plen - 2] == 'E'
        && ph->data[plen - 3] == ','
        && ph->data[plen - 4] == 'n') {
        // 'n,Es' -> 'n@s' (count shrinks by 1)
        ph->data[plen - 3] = '@';
        ph->data[plen - 2] = 's';
        ph->count -= 1;
        ph->data[ph->count] = '\0';
    } else if (plen >= 3 && ph->data[plen - 1] == 's'
        && ph->data[plen - 2] == 'E'
        && ph->data[plen - 3] == 'n') {
        ph->data[plen - 2] = '@';
        ph->data[plen - 1] = 's';
    }
}

// Helpers for step 6c.
static bool prev_syllable_is_primary(const char * ph, size_t before) {
    bool found = false;
    bool stop = false;
    int j = (int)before - 1;
    while (j >= 0 && !found && !stop) {
        char c = ph[j];
        if (c == '\'') { found = true; }
        else if (c == ',' || c == '%') { stop = true; }
        j--;
    }
    return found;
}

static bool next_syllable_is_primary(const char * ph, size_t pn,
                                     size_t after) {
    static const char vow[] = "aAeEiIoOuUV03@";
    bool found = false;
    bool stop = false;
    size_t j = after;
    while (j < pn && !found && !stop) {
        char c = ph[j];
        if (c == '\'') { found = true; }
        else if (c == ',' || strchr(vow, c) != NULL) { stop = true; }
        j++;
    }
    return found;
}

// Step 6c: 'oU#' compound prefix -> '0' / '@' / 'oU' by context.
static void reduce_ou_hash_compound(struct chars * ph) {
    for (size_t i = 0; i + 2 < ph->count; i++) {
        if (ph->data[i] == 'o' && ph->data[i + 1] == 'U'
            && ph->data[i + 2] == '#') {
            bool self_primary = i > 0 && ph->data[i - 1] == '\'';
            if (self_primary) {
                // Replace 'oU#' (3) with '0' (1). Shrink by 2.
                memmove(ph->data + i + 1,
                        ph->data + i + 3,
                        ph->count - i - 3);
                ph->data[i] = '0';
                ph->count -= 2;
                ph->data[ph->count] = '\0';
            } else if (prev_syllable_is_primary(ph->data, i)) {
                memmove(ph->data + i + 1,
                        ph->data + i + 3,
                        ph->count - i - 3);
                ph->data[i] = '@';
                ph->count -= 2;
                ph->data[ph->count] = '\0';
            } else if (next_syllable_is_primary(ph->data, ph->count,
                                                i + 3)) {
                memmove(ph->data + i + 1,
                        ph->data + i + 3,
                        ph->count - i - 3);
                ph->data[i] = '@';
                ph->count -= 2;
                ph->data[ph->count] = '\0';
            } else {
                // Default: keep 'oU' (drop '#').
                memmove(ph->data + i + 2,
                        ph->data + i + 3,
                        ph->count - i - 3);
                ph->count -= 1;
                ph->data[ph->count] = '\0';
            }
        }
    }
}

// Step 6e: '-ically' schwa elision. Word-final 'k@li' -> 'kli'.
static void elide_ically_schwa(struct chars * ph) {
    if (ph->count >= 4
        && ph->data[ph->count - 4] == 'k'
        && ph->data[ph->count - 3] == '@'
        && ph->data[ph->count - 2] == 'l'
        && ph->data[ph->count - 1] == 'i') {
        ph->data[ph->count - 3] = 'l';
        ph->data[ph->count - 2] = 'i';
        ph->count -= 1;
        ph->data[ph->count] = '\0';
    }
}

// Step 6f: Add secondary stress before syllabic-n in compounds.
static void add_compound_syllabic_n_stress(struct chars * ph) {
    if (ph->count < 2
        || ph->data[ph->count - 1] != '-'
        || ph->data[ph->count - 2] != 'n') {
        return;
    }
    if (memchr(ph->data, '\'', ph->count) == NULL) { return; }
    static const char vc[] = "aAeEiIoOuUV03@";
    size_t sn_start = ph->count - 2;
    if (sn_start > 0 && ph->data[sn_start - 1] == '?') { sn_start--; }
    int vgroups = 0;
    bool in_v = false;
    for (size_t vi = 0; vi < sn_start; vi++) {
        char c = ph->data[vi];
        if (c != '\'' && c != ',' && c != '%' && c != '=') {
            if (strchr(vc, c) != NULL) {
                if (!in_v) { vgroups++; in_v = true; }
            } else {
                in_v = false;
            }
        }
    }
    char * prime = memchr(ph->data, '\'', ph->count);
    size_t prime_pos = (size_t)(prime - ph->data);
    bool has_unstressed_prefix = false;
    for (size_t k = 0;
         k < prime_pos && !has_unstressed_prefix; k++) {
        if (ph->data[k] == '%' || ph->data[k] == '=') {
            has_unstressed_prefix = true;
        }
    }
    if (vgroups >= 2 && !has_unstressed_prefix
        && (sn_start == 0 || ph->data[sn_start - 1] != ',')) {
        chars_grow(ph, ph->count + 2);
        memmove(ph->data + sn_start + 1,
                ph->data + sn_start,
                ph->count - sn_start);
        ph->data[sn_start] = ',';
        ph->count += 1;
        ph->data[ph->count] = '\0';
    }
}

// Step 5.5-dim0: '0' -> '@' when middle vowel and <= UNSTRESSED.
// Walks ph as a syllable list (multi-char-aware vowel scan), then
// reduces right-to-left so insertion positions stay valid.
static void reduce_zero_diminished(struct chars * ph) {
    if (memchr(ph->data, '0', ph->count) == NULL) { return; }
    struct syl0d {
        size_t pos;
        const char * code;
        size_t code_len;
        int level;
    };
    enum { MAX_SYL = 256 };
    struct syl0d syls[MAX_SYL];
    int n = 0;
    size_t pi = 0;
    int cur_level = -1;
    while (pi < ph->count && n < MAX_SYL) {
        char c = ph->data[pi];
        if (c == '\'') { cur_level = 4; pi++; }
        else if (c == ',') { cur_level = 2; pi++; }
        else if (c == '%' || c == '=') { cur_level = 1; pi++; }
        else {
            const char * code = NULL;
            size_t code_len = 0;
            find_phoneme_code(ph->data, ph->count, pi,
                              &code, &code_len);
            if (is_vowel_code(code, code_len)) {
                // Skip '@-' (morpheme-boundary non-syllabic).
                if (code_len == 1 && code[0] == '@'
                    && pi + 1 < ph->count
                    && ph->data[pi + 1] == '-') {
                    pi += code_len;
                } else {
                    syls[n].pos = pi;
                    syls[n].code = code;
                    syls[n].code_len = code_len;
                    syls[n].level = cur_level;
                    n++;
                    cur_level = -1;
                    pi += code_len;
                }
            } else if (ph->data[pi] == '-') {
                pi++;
            } else {
                pi += code_len;
            }
        }
    }
    // Right-to-left reduction.
    for (int v = n - 1; v >= 0; v--) {
        int vnum = v + 1;
        bool eligible = syls[v].code_len == 1
            && syls[v].code[0] == '0'
            && syls[v].level <= 1
            && vnum != 1 && vnum != n;
        if (eligible && vnum == n - 1 && syls[n - 1].level <= 1) {
            eligible = false;
        }
        if (eligible) {
            ph->data[syls[v].pos] = '@';
        }
    }
}

// Step 5a-prime: adjacent primary demotion. Demote the EARLIER of
// two consecutive primaries (at syllable distance 1) to ','.
static void demote_adjacent_primaries(struct chars * ph,
                                      const char * ph_in,
                                      size_t in_n) {
    if (memchr(ph_in, '\'', in_n) == NULL) { return; }
    struct sylp { size_t pos; bool is_primary; };
    enum { MAX_SYL2 = 256 };
    struct sylp syls[MAX_SYL2];
    int n = 0;
    size_t pi = 0;
    bool prim = false;
    while (pi < ph->count && n < MAX_SYL2) {
        char c = ph->data[pi];
        if (c == '\'') { prim = true; pi++; }
        else if (c == ',' || c == '%' || c == '=') {
            prim = false; pi++;
        }
        else {
            const char * code = NULL;
            size_t code_len = 0;
            find_phoneme_code(ph->data, ph->count, pi,
                              &code, &code_len);
            if (is_vowel_code(code, code_len)) {
                syls[n].pos = pi;
                syls[n].is_primary = prim;
                n++;
                prim = false;
            }
            pi += code_len;
        }
    }
    // Demote earlier in each adjacent-primary pair (right-to-left).
    for (int si = n - 1; si >= 1; si--) {
        if (syls[si].is_primary && syls[si - 1].is_primary) {
            if (syls[si - 1].pos > 0) {
                size_t cp = syls[si - 1].pos - 1;
                if (cp < ph->count && ph->data[cp] == '\'') {
                    ph->data[cp] = ',';
                    syls[si - 1].is_primary = false;
                }
            }
        }
    }
}

// Step 5.5c: bare 'a' -> 'a#' before primary, after the first
// (protected) vowel. Rule-derived only.
static void reduce_a_before_primary(struct chars * ph,
                                    bool rule_derived) {
    if (!rule_derived) { return; }
    char * prim = memchr(ph->data, '\'', ph->count);
    if (prim == NULL) { return; }
    size_t primary_pos = (size_t)(prim - ph->data);
    char * sec = memchr(ph->data, ',', ph->count);
    char * pct = memchr(ph->data, '%', ph->count);
    size_t scan_start = (size_t)-1;
    if (sec != NULL && (size_t)(sec - ph->data) < primary_pos) {
        scan_start = (size_t)(sec - ph->data) + 1;
    }
    if (pct != NULL && (size_t)(pct - ph->data) < primary_pos) {
        size_t pct_scan = (size_t)(pct - ph->data) + 1;
        while (pct_scan < primary_pos) {
            const char * code = NULL;
            size_t code_len = 0;
            find_phoneme_code(ph->data, ph->count, pct_scan,
                              &code, &code_len);
            if (is_vowel_code(code, code_len)) {
                pct_scan += code_len;
                break;
            }
            pct_scan++;
        }
        if (pct_scan < primary_pos) {
            if (scan_start == (size_t)-1 || pct_scan < scan_start) {
                scan_start = pct_scan;
            }
        }
    }
    if (scan_start == (size_t)-1) {
        size_t gen_scan = 0;
        bool found_first = false;
        while (gen_scan < primary_pos && !found_first) {
            if (ph->data[gen_scan] == '\''
                || ph->data[gen_scan] == ','
                || ph->data[gen_scan] == '%') {
                gen_scan++;
            } else {
                const char * code = NULL;
                size_t code_len = 0;
                find_phoneme_code(ph->data, ph->count, gen_scan,
                                  &code, &code_len);
                gen_scan += code_len;
                if (is_vowel_code(code, code_len)) {
                    found_first = true;
                }
            }
        }
        if (gen_scan < primary_pos) { scan_start = gen_scan; }
    }
    if (scan_start != (size_t)-1) {
        for (size_t pi = scan_start; pi < primary_pos; pi++) {
            if (ph->data[pi] == 'a') {
                bool is_diphthong_start = pi + 1 < ph->count
                    && (ph->data[pi + 1] == 'I'
                     || ph->data[pi + 1] == 'U'
                     || ph->data[pi + 1] == ':'
                     || ph->data[pi + 1] == '@'
                     || ph->data[pi + 1] == '#');
                if (!is_diphthong_start) {
                    bool protected_vowel = pi > 0
                        && (ph->data[pi - 1] == '\''
                         || ph->data[pi - 1] == ','
                         || ph->data[pi - 1] == '%');
                    if (!protected_vowel) {
                        chars_grow(ph, ph->count + 2);
                        memmove(ph->data + pi + 2,
                                ph->data + pi + 1,
                                ph->count - pi - 1);
                        ph->data[pi + 1] = '#';
                        ph->count += 1;
                        ph->data[ph->count] = '\0';
                        primary_pos++;
                        pi++;
                    }
                }
            }
        }
    }
}

// Step 5.5b: bare 'E' -> 'I2' (or '@' before nasal 'n') across 4
// contexts. The most complex of the late-phase reductions.
static bool is_unstressed_code(const char * c, size_t cn) {
    return (cn == 1 && c[0] == '@')
        || (cn == 2 && c[0] == '@' && c[1] == '2')
        || (cn == 2 && c[0] == '@' && c[1] == '5')
        || (cn == 2 && c[0] == '@' && c[1] == 'L')
        || (cn == 1 && c[0] == '3')
        || (cn == 2 && c[0] == 'I' && c[1] == '#');
}

static void reduce_e_unstressed(struct chars * ph,
                                const bool * rba, size_t rba_n) {
    bool rule_derived = rba_n > 0;
    char * prim = memchr(ph->data, '\'', ph->count);
    if (prim == NULL) { return; }
    size_t primary_pos = (size_t)(prim - ph->data);
    char * sec = memchr(ph->data, ',', ph->count);
    size_t secondary_pos = sec != NULL
        ? (size_t)(sec - ph->data) : (size_t)-1;
    // Context 1: ',' before '\''.
    if (rule_derived && secondary_pos != (size_t)-1
        && secondary_pos < primary_pos) {
        for (size_t pi = secondary_pos + 1;
             pi < primary_pos; pi++) {
            if (ph->data[pi] == 'E') {
                bool is_variant = pi + 1 < ph->count
                    && (ph->data[pi + 1] == '#'
                     || ph->data[pi + 1] == '2');
                if (is_variant) { pi++; }
                else {
                    bool stressed = pi > 0
                        && (ph->data[pi - 1] == '\''
                         || ph->data[pi - 1] == ',');
                    if (!stressed) {
                        bool before_n = pi + 1 < primary_pos
                            && ph->data[pi + 1] == 'n';
                        if (before_n) {
                            ph->data[pi] = '@';
                        } else {
                            chars_grow(ph, ph->count + 2);
                            memmove(ph->data + pi + 2,
                                    ph->data + pi + 1,
                                    ph->count - pi - 1);
                            ph->data[pi] = 'I';
                            ph->data[pi + 1] = '2';
                            ph->count += 1;
                            ph->data[ph->count] = '\0';
                            primary_pos++;
                            pi++;
                        }
                    }
                }
            }
        }
    }
    // Context 3: '\'' before ','.
    if (rule_derived && secondary_pos != (size_t)-1
        && secondary_pos > primary_pos) {
        for (size_t pi = primary_pos + 1;
             pi < secondary_pos; pi++) {
            if (ph->data[pi] == 'E') {
                bool is_variant = pi + 1 < ph->count
                    && (ph->data[pi + 1] == '#'
                     || ph->data[pi + 1] == '2');
                if (is_variant) { pi++; }
                else {
                    bool stressed = pi > 0
                        && (ph->data[pi - 1] == '\''
                         || ph->data[pi - 1] == ',');
                    if (!stressed) {
                        bool before_n = pi + 1 < secondary_pos
                            && ph->data[pi + 1] == 'n';
                        if (before_n) {
                            ph->data[pi] = '@';
                        } else {
                            chars_grow(ph, ph->count + 2);
                            memmove(ph->data + pi + 2,
                                    ph->data + pi + 1,
                                    ph->count - pi - 1);
                            ph->data[pi] = 'I';
                            ph->data[pi + 1] = '2';
                            ph->count += 1;
                            ph->data[ph->count] = '\0';
                            secondary_pos++;
                            pi++;
                        }
                    }
                }
            }
        }
    }
    // Context 2: '%' initial unstressed syllable.
    if (ph->count > 0 && ph->data[0] == '%') {
        size_t scan_start = 1;
        while (scan_start < ph->count
               && (ph->data[scan_start] == '\''
                || ph->data[scan_start] == ','
                || ph->data[scan_start] == '%'
                || ph->data[scan_start] == '=')) {
            scan_start++;
        }
        if (scan_start < ph->count) {
            const char * code = NULL;
            size_t code_len = 0;
            find_phoneme_code(ph->data, ph->count, scan_start,
                              &code, &code_len);
            if (is_vowel_code(code, code_len)) {
                scan_start += code_len;
            }
        }
        for (size_t pi = scan_start; pi < primary_pos; pi++) {
            if (ph->data[pi] == 'E') {
                bool is_variant = pi + 1 < ph->count
                    && (ph->data[pi + 1] == '#'
                     || ph->data[pi + 1] == '2');
                if (is_variant) { pi++; }
                else {
                    bool stressed = pi > 0
                        && (ph->data[pi - 1] == '\''
                         || ph->data[pi - 1] == ',');
                    if (!stressed) {
                        chars_grow(ph, ph->count + 2);
                        memmove(ph->data + pi + 2,
                                ph->data + pi + 1,
                                ph->count - pi - 1);
                        ph->data[pi] = 'I';
                        ph->data[pi + 1] = '2';
                        ph->count += 1;
                        ph->data[ph->count] = '\0';
                        primary_pos++;
                        pi++;
                    }
                }
            }
        }
    }
    // Context 4: bare 'E' after first explicit stress, middle pos.
    if (rule_derived) {
        size_t first_stress = (size_t)-1;
        for (size_t pi = 0;
             pi < ph->count && first_stress == (size_t)-1; pi++) {
            if (ph->data[pi] == '\'' || ph->data[pi] == ',') {
                first_stress = pi;
            }
        }
        if (first_stress != (size_t)-1) {
            size_t pi = first_stress + 1;
            while (pi < ph->count) {
                if (ph->data[pi] == 'E') {
                    bool is_variant = pi + 1 < ph->count
                        && (ph->data[pi + 1] == '#'
                         || ph->data[pi + 1] == '2');
                    if (is_variant) { pi += 2; }
                    else {
                        bool stressed = pi > 0
                            && (ph->data[pi - 1] == '\''
                             || ph->data[pi - 1] == ',');
                        bool reduce_to_i2 = false;
                        if (!stressed) {
                            // Count vowels after this E.
                            int vowels_after = 0;
                            const char * last_code = NULL;
                            size_t last_code_len = 0;
                            size_t pj = pi + 1;
                            while (pj < ph->count) {
                                const char * code = NULL;
                                size_t code_len = 0;
                                find_phoneme_code(ph->data, ph->count,
                                                  pj, &code, &code_len);
                                if (is_vowel_code(code, code_len)) {
                                    vowels_after++;
                                    last_code = code;
                                    last_code_len = code_len;
                                }
                                pj += code_len;
                            }
                            bool should_reduce = vowels_after >= 2
                                || (vowels_after == 1
                                    && !is_unstressed_code(
                                        last_code, last_code_len));
                            if (should_reduce) {
                                bool before_n = pi + 1 < ph->count
                                    && ph->data[pi + 1] == 'n';
                                if (before_n) {
                                    ph->data[pi] = '@';
                                } else {
                                    chars_grow(ph, ph->count + 2);
                                    memmove(ph->data + pi + 2,
                                            ph->data + pi + 1,
                                            ph->count - pi - 1);
                                    ph->data[pi] = 'I';
                                    ph->data[pi + 1] = '2';
                                    ph->count += 1;
                                    ph->data[ph->count] = '\0';
                                    reduce_to_i2 = true;
                                }
                            }
                        }
                        pi += reduce_to_i2 ? 2 : 1;
                    }
                } else {
                    pi++;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Stress-phase helpers (step 5.0 + 5a auxiliaries).
// ---------------------------------------------------------------------------

// Step 5.0: '=' suffix stress shift. When ph has '=' AFTER the
// primary '\'', and no second primary, move primary to the last
// stressable vowel between them. Returns true if it fired (caller
// uses this to suppress backward secondary in step 5a).
static bool apply_equals_suffix_stress_shift(struct chars * ph) {
    bool fired = false;
    char * prim = memchr(ph->data, '\'', ph->count);
    if (prim == NULL) { return false; }
    size_t prim_pos = (size_t)(prim - ph->data);
    size_t eq_pos = (size_t)-1;
    for (size_t i = ph->count; i > 0; i--) {
        if (ph->data[i - 1] == '=') { eq_pos = i - 1; break; }
    }
    if (eq_pos == (size_t)-1 || prim_pos >= eq_pos) { return false; }
    bool has_second_primary = false;
    for (size_t i = prim_pos + 1; i < eq_pos; i++) {
        if (ph->data[i] == '\'') { has_second_primary = true; break; }
    }
    if (has_second_primary) { return false; }
    size_t last_sv_pos = (size_t)-1;
    size_t scan = prim_pos + 1;
    while (scan < eq_pos) {
        char c = ph->data[scan];
        if (c == '\'' || c == ',' || c == '%' || c == '=' || c == '*') {
            scan++;
        } else {
            const char * code = NULL;
            size_t code_len = 0;
            find_phoneme_code(ph->data, ph->count, scan,
                              &code, &code_len);
            if (is_vowel_code(code, code_len)) {
                bool is_weak =
                    (code_len == 1 && code[0] == '@')
                 || (code_len == 2 && code[0] == '@'
                     && (code[1] == '2' || code[1] == '5'
                      || code[1] == 'L'))
                 || (code_len == 1 && code[0] == '3')
                 || (code_len == 2 && code[0] == 'I'
                     && (code[1] == '#' || code[1] == '2'))
                 || (code_len == 2 && code[0] == 'a' && code[1] == '#')
                 || (code_len == 1 && code[0] == 'i');
                bool is_stressed = scan > 0
                    && (ph->data[scan - 1] == '\''
                     || ph->data[scan - 1] == ',');
                if (!is_weak && !is_stressed) {
                    last_sv_pos = scan;
                }
            }
            scan += code_len;
        }
    }
    if (last_sv_pos != (size_t)-1) {
        size_t prim_vowel_pos = prim_pos + 1;
        bool found_vowel = false;
        while (prim_vowel_pos < ph->count && !found_vowel) {
            const char * code = NULL;
            size_t code_len = 0;
            find_phoneme_code(ph->data, ph->count, prim_vowel_pos,
                              &code, &code_len);
            if (is_vowel_code(code, code_len)) {
                found_vowel = true;
            } else {
                prim_vowel_pos += code_len;
            }
        }
        if (prim_vowel_pos != last_sv_pos) {
            ph->data[prim_pos] = ',';
            if (last_sv_pos > 0
                && ph->data[last_sv_pos - 1] == '*') {
                ph->data[last_sv_pos - 1] = 't';
            }
            chars_grow(ph, ph->count + 2);
            memmove(ph->data + last_sv_pos + 1,
                    ph->data + last_sv_pos,
                    ph->count - last_sv_pos);
            ph->data[last_sv_pos] = '\'';
            ph->count += 1;
            ph->data[ph->count] = '\0';
            fired = true;
        }
    }
    return fired;
}

// Step 5a-cleanup: remove ',' at syllable distance 1 from primary
// (rule-derived only, when 5a did NOT run, and the ',' isn't
// inside the same rule as the primary).
static void cleanup_adjacent_secondary(struct chars * ph,
                                       bool step5a_ran,
                                       bool is_rule_leading_comma,
                                       const bool * rba,
                                       size_t rba_n) {
    if (step5a_ran || is_rule_leading_comma || rba_n == 0
        || memchr(ph->data, '\'', ph->count) == NULL
        || memchr(ph->data, ',', ph->count) == NULL) {
        return;
    }
    struct syle { size_t pos; bool is_primary; bool is_secondary; };
    enum { MAX_SYL3 = 256 };
    struct syle syls[MAX_SYL3];
    int n = 0;
    size_t pi = 0;
    bool prim = false;
    bool sec = false;
    while (pi < ph->count && n < MAX_SYL3) {
        char c = ph->data[pi];
        if (c == '\'') { prim = true; pi++; }
        else if (c == ',') { sec = true; pi++; }
        else if (c == '%' || c == '=') { pi++; }
        else {
            const char * code = NULL;
            size_t code_len = 0;
            find_phoneme_code(ph->data, ph->count, pi,
                              &code, &code_len);
            if (is_vowel_code(code, code_len)) {
                syls[n].pos = pi;
                syls[n].is_primary = prim;
                syls[n].is_secondary = sec;
                n++;
                prim = false; sec = false;
            }
            pi += code_len;
        }
    }
    int prim_idx = -1;
    for (int si = 0; si < n && prim_idx < 0; si++) {
        if (syls[si].is_primary) { prim_idx = si; }
    }
    if (prim_idx < 0) { return; }
    char * prim_marker = memchr(ph->data, '\'', ph->count);
    size_t prim_ph_pos = (size_t)(prim_marker - ph->data);
    // Collect commas to remove (descending order).
    size_t to_remove[MAX_SYL3];
    int n_remove = 0;
    for (int si = 0; si < n; si++) {
        if (syls[si].is_secondary
            && (si == prim_idx + 1 || si == prim_idx - 1)) {
            // Find comma_pos by scanning backward.
            size_t syl_pos = syls[si].pos;
            size_t comma_pos = (size_t)-1;
            bool stop = false;
            int bp = (int)syl_pos - 1;
            while (bp >= 0 && !stop) {
                if (ph->data[bp] == ',') {
                    comma_pos = (size_t)bp;
                    stop = true;
                } else if (ph->data[bp] == '\''
                    || ph->data[bp] == '%') {
                    stop = true;
                }
                bp--;
            }
            if (comma_pos != (size_t)-1) {
                size_t lo = comma_pos < prim_ph_pos
                    ? comma_pos : prim_ph_pos;
                size_t hi = comma_pos > prim_ph_pos
                    ? comma_pos : prim_ph_pos;
                bool same_rule = true;
                for (size_t rp = lo;
                     rp < hi && rp < rba_n && same_rule; rp++) {
                    if (rba[rp]) { same_rule = false; }
                }
                if (!same_rule) {
                    to_remove[n_remove++] = comma_pos;
                }
            }
        }
    }
    // Sort descending.
    for (int i = 0; i < n_remove; i++) {
        for (int j = i + 1; j < n_remove; j++) {
            if (to_remove[j] > to_remove[i]) {
                size_t t = to_remove[i];
                to_remove[i] = to_remove[j];
                to_remove[j] = t;
            }
        }
    }
    for (int i = 0; i < n_remove; i++) {
        size_t pos = to_remove[i];
        memmove(ph->data + pos, ph->data + pos + 1,
                ph->count - pos - 1);
        ph->count -= 1;
        ph->data[ph->count] = '\0';
    }
}

// Step 5a-trochaic: insert ',' on the first vowel V with no marker
// AND both neighbours have stress level <= 1. When ph_in had no
// primary, the first trochaic assignment becomes the PRIMARY and
// the pick_last primary is removed.
static void trochaic_compound_prefix(struct chars * ph,
                                     const char * ph_in, size_t in_n,
                                     bool step5a_ran,
                                     bool starts_with_secondary) {
    if (step5a_ran || starts_with_secondary
        || memchr(ph->data, '\'', ph->count) == NULL
        || memchr(ph_in, ',', in_n) == NULL) {
        return;
    }
    struct sylt {
        size_t pos;
        const char * code;
        size_t code_len;
        int level;
    };
    enum { MAX_SYL4 = 256 };
    struct sylt syls[MAX_SYL4];
    int n = 0;
    size_t pi = 0;
    int cur_level = -1;
    while (pi < ph->count && n < MAX_SYL4) {
        char c = ph->data[pi];
        if (c == '\'') { cur_level = 4; pi++; }
        else if (c == ',') { cur_level = 2; pi++; }
        else if (c == '%' || c == '=') { cur_level = 1; pi++; }
        else {
            const char * code = NULL;
            size_t code_len = 0;
            find_phoneme_code(ph->data, ph->count, pi,
                              &code, &code_len);
            if (is_vowel_code(code, code_len)) {
                syls[n].pos = pi;
                syls[n].code = code;
                syls[n].code_len = code_len;
                syls[n].level = cur_level;
                n++;
                cur_level = -1;
            } else {
                cur_level = -1;
            }
            pi += code_len;
        }
    }
    bool input_has_primary = memchr(ph_in, '\'', in_n) != NULL
        || memchr(ph_in, '=', in_n) != NULL;
    bool trochaic_primary_done = false;
    bool stop_outer = false;
    for (int sv = 0; sv < n && !stop_outer; sv++) {
        if (syls[sv].level == -1) {
            const char * vcode = syls[sv].code;
            size_t vcl = syls[sv].code_len;
            bool is_unstressed_phone =
                (vcl == 1 && (vcode[0] == '@' || vcode[0] == '3'
                           || vcode[0] == 'i'))
             || (vcl == 2 && vcode[0] == '@'
                 && (vcode[1] == '2' || vcode[1] == '5'
                  || vcode[1] == 'L'))
             || (vcl == 2 && vcode[0] == 'I'
                 && (vcode[1] == '#' || vcode[1] == '2'))
             || (vcl == 2 && vcode[0] == 'a' && vcode[1] == '#');
            if (!is_unstressed_phone) {
                // effectiveLv: scan in direction, skipping schwas.
                int prev_lv = -1, next_lv = -1;
                for (int nv = sv - 1; nv >= 0; nv--) {
                    const char * c2 = syls[nv].code;
                    size_t cl = syls[nv].code_len;
                    bool is_schwa = (cl == 1 && (c2[0] == '@' || c2[0] == '3'))
                        || (cl == 2 && c2[0] == '@'
                            && (c2[1] == '2' || c2[1] == '5'
                             || c2[1] == 'L'));
                    if (!is_schwa) { prev_lv = syls[nv].level; break; }
                }
                for (int nv = sv + 1; nv < n; nv++) {
                    const char * c2 = syls[nv].code;
                    size_t cl = syls[nv].code_len;
                    bool is_schwa = (cl == 1 && (c2[0] == '@' || c2[0] == '3'))
                        || (cl == 2 && c2[0] == '@'
                            && (c2[1] == '2' || c2[1] == '5'
                             || c2[1] == 'L'));
                    if (!is_schwa) { next_lv = syls[nv].level; break; }
                }
                if (prev_lv <= 1 && next_lv <= 1) {
                    if (!input_has_primary && !trochaic_primary_done) {
                        // Move pick_last primary to this position.
                        char * pm = memchr(ph->data, '\'',
                                           ph->count);
                        if (pm != NULL) {
                            size_t pp = (size_t)(pm - ph->data);
                            memmove(ph->data + pp,
                                    ph->data + pp + 1,
                                    ph->count - pp - 1);
                            ph->count -= 1;
                            ph->data[ph->count] = '\0';
                            for (int k = 0; k < n; k++) {
                                if (syls[k].pos > pp) {
                                    syls[k].pos--;
                                }
                                if (syls[k].level == 4) {
                                    syls[k].level = -1;
                                }
                            }
                        }
                        chars_grow(ph, ph->count + 2);
                        memmove(ph->data + syls[sv].pos + 1,
                                ph->data + syls[sv].pos,
                                ph->count - syls[sv].pos);
                        ph->data[syls[sv].pos] = '\'';
                        ph->count += 1;
                        ph->data[ph->count] = '\0';
                        for (int nv = sv + 1; nv < n; nv++) {
                            syls[nv].pos++;
                        }
                        syls[sv].level = 4;
                        trochaic_primary_done = true;
                    } else {
                        chars_grow(ph, ph->count + 2);
                        memmove(ph->data + syls[sv].pos + 1,
                                ph->data + syls[sv].pos,
                                ph->count - syls[sv].pos);
                        ph->data[syls[sv].pos] = ',';
                        ph->count += 1;
                        ph->data[ph->count] = '\0';
                        for (int nv = sv + 1; nv < n; nv++) {
                            syls[nv].pos++;
                        }
                        stop_outer = true;
                    }
                }
            }
        }
    }
}

// Step 5a-final: add secondary stress to last stressable vowel
// when its direct predecessor is a schwa/phUNSTRESSED phoneme.
static void final_syllable_secondary(struct chars * ph) {
    if (memchr(ph->data, '\'', ph->count) == NULL) { return; }
    struct sylf {
        size_t pos;
        const char * code;
        size_t code_len;
        int level;
    };
    enum { MAX_SYL5 = 256 };
    struct sylf syls[MAX_SYL5];
    int n = 0;
    size_t pi = 0;
    int cur_level = -1;
    while (pi < ph->count && n < MAX_SYL5) {
        char c = ph->data[pi];
        if (c == '\'') { cur_level = 4; pi++; }
        else if (c == ',') { cur_level = 2; pi++; }
        else if (c == '%' || c == '=') { cur_level = 1; pi++; }
        else {
            const char * code = NULL;
            size_t code_len = 0;
            find_phoneme_code(ph->data, ph->count, pi,
                              &code, &code_len);
            if (is_vowel_code(code, code_len)) {
                if (code_len == 1 && code[0] == '@'
                    && pi + 1 < ph->count
                    && ph->data[pi + 1] == '-') {
                    pi += code_len;
                } else {
                    syls[n].pos = pi;
                    syls[n].code = code;
                    syls[n].code_len = code_len;
                    syls[n].level = cur_level;
                    n++;
                    cur_level = -1;
                    pi += code_len;
                }
            } else if (ph->data[pi] == '-') {
                pi++;
            } else {
                pi += code_len;
            }
        }
    }
    if (n < 2) { return; }
    struct sylf * last = &syls[n - 1];
    struct sylf * prev = &syls[n - 2];
    // Is the last code itself unstressed/phNONSYLLABIC?
    bool last_unstressed =
        (last->code_len == 1 && (last->code[0] == '@'
                               || last->code[0] == '3'
                               || last->code[0] == 'i'))
     || (last->code_len == 2 && last->code[0] == '@'
         && (last->code[1] == '2' || last->code[1] == '5'
          || last->code[1] == 'L'))
     || (last->code_len == 2 && last->code[0] == 'I'
         && (last->code[1] == '#' || last->code[1] == '2'))
     || (last->code_len == 2 && last->code[0] == 'a'
         && last->code[1] == '#');
    bool last_is_stressable = !last_unstressed;
    if (last->level == -1 && last_is_stressable) {
        bool prev_unstressed =
            (prev->code_len == 1 && (prev->code[0] == '@'
                                  || prev->code[0] == '3'))
         || (prev->code_len == 2 && prev->code[0] == '@'
             && (prev->code[1] == '2' || prev->code[1] == '5'
              || prev->code[1] == 'L'))
         || (prev->code_len == 2 && prev->code[0] == 'I'
             && (prev->code[1] == '#' || prev->code[1] == '2'))
         || (prev->code_len == 2 && prev->code[0] == 'a'
             && prev->code[1] == '#')
         || prev->level == 1;
        if (prev_unstressed) {
            chars_grow(ph, ph->count + 2);
            memmove(ph->data + last->pos + 1,
                    ph->data + last->pos,
                    ph->count - last->pos);
            ph->data[last->pos] = ',';
            ph->count += 1;
            ph->data[ph->count] = '\0';
        }
    }
}

// ---------------------------------------------------------------------------
// Step 5 (insert primary) + 5a (place secondary). The two big
// stress-placement passes.
// ---------------------------------------------------------------------------

// Helper for insert_primary_stress: returns true iff a stressable
// (non-schwa) vowel exists after `pi`, with no primary stress
// marker in between.
static bool has_strong_after(const char * ph, size_t pn, size_t pi) {
    bool unst = false;
    bool found = false;
    bool stop = false;
    while (pi < pn && !found && !stop) {
        char c = ph[pi];
        if (c == '%' || c == '=' || c == ',') { unst = true; pi++; }
        else if (c == '\'') { stop = true; }
        else {
            const char * code = NULL;
            size_t code_len = 0;
            find_phoneme_code(ph, pn, pi, &code, &code_len);
            if (is_vowel_code(code, code_len)) {
                bool is_weak =
                    (code_len == 1 && code[0] == '@')
                 || (code_len == 2 && code[0] == '@'
                     && (code[1] == '2' || code[1] == '5'
                      || code[1] == 'L'))
                 || (code_len == 1 && code[0] == '3');
                if (!unst && !is_weak) { found = true; }
                unst = false;
            }
            pi += code_len;
        }
    }
    return found;
}

// Step 5: insert primary stress.
static void insert_primary_stress(struct chars * ph,
                                  bool force_final_stress,
                                  const bool * rba, size_t rba_n) {
    bool starts_with_secondary = ph->count > 0
        && ph->data[0] == ',';
    bool is_rule_leading_comma = starts_with_secondary && rba_n > 0;
    bool active = memchr(ph->data, '\'', ph->count) == NULL
        && (!starts_with_secondary || is_rule_leading_comma);
    if (!active) { return; }
    // Determine pick_last.
    bool pick_last = force_final_stress;
    size_t last_eq = (size_t)-1;
    for (size_t i = ph->count; i > 0; i--) {
        if (ph->data[i - 1] == '=') { last_eq = i - 1; break; }
    }
    if (last_eq != (size_t)-1) {
        bool has_strong = false;
        size_t pi2 = last_eq + 1;
        while (pi2 < ph->count && !has_strong) {
            const char * code = NULL;
            size_t code_len = 0;
            find_phoneme_code(ph->data, ph->count, pi2,
                              &code, &code_len);
            if (is_vowel_code(code, code_len)) {
                bool is_weak =
                    (code_len == 2 && code[0] == '@'
                     && (code[1] == '2' || code[1] == '5'
                      || code[1] == 'L'))
                 || (code_len == 2 && code[0] == 'I'
                     && (code[1] == '#' || code[1] == '2'))
                 || (code_len == 1 && code[0] == '@')
                 || (code_len == 1 && code[0] == '3')
                 || (code_len == 2 && code[0] == 'a' && code[1] == '#')
                 || (code_len == 2 && code[0] == 'i' && code[1] == '@')
                 || (code_len == 1 && code[0] == 'i');
                if (!is_weak) { has_strong = true; }
            }
            pi2 += code_len;
        }
        pick_last = !has_strong;
    }
    bool ignore_comma_for_primary = rba_n > 0;
    // Strong-diphthong-after-leading-secondary exception.
    bool use_pick_last_for_secondary = false;
    if (ignore_comma_for_primary) {
        size_t comma_pi = (size_t)-1;
        bool stop_search = false;
        for (size_t spi = 0; spi < ph->count && !stop_search; spi++) {
            if (ph->data[spi] == '\'') { stop_search = true; }
            else if (ph->data[spi] == ',') {
                if (spi > 0) { comma_pi = spi; }
                stop_search = true;
            }
        }
        if (comma_pi != (size_t)-1) {
            size_t scan = comma_pi + 1;
            while (scan < ph->count
                   && (ph->data[scan] == '\''
                    || ph->data[scan] == ','
                    || ph->data[scan] == '%'
                    || ph->data[scan] == '=')) {
                scan++;
            }
            if (scan < ph->count) {
                const char * code = NULL;
                size_t code_len = 0;
                find_phoneme_code(ph->data, ph->count, scan,
                                  &code, &code_len);
                if (is_vowel_code(code, code_len)) {
                    scan += code_len;
                }
            }
            static const char * const strong_diph[] = {
                "oU","aI","eI","aU","OI","aI@","aI3","aU@","i:",
                "u:","A:","E:","3:","o:","U:", NULL
            };
            bool stop_strong = false;
            while (scan < ph->count && !stop_strong) {
                if (ph->data[scan] == '\'') { stop_strong = true; }
                else {
                    const char * code = NULL;
                    size_t code_len = 0;
                    find_phoneme_code(ph->data, ph->count, scan,
                                      &code, &code_len);
                    for (int si = 0;
                         strong_diph[si] != NULL
                         && !use_pick_last_for_secondary; si++) {
                        size_t sdl = strlen(strong_diph[si]);
                        if (code_len == sdl
                            && memcmp(code, strong_diph[si], sdl) == 0) {
                            use_pick_last_for_secondary = true;
                        }
                    }
                    if (use_pick_last_for_secondary) {
                        stop_strong = true;
                    } else {
                        scan += code_len;
                    }
                }
            }
        }
    }
    if (use_pick_last_for_secondary) {
        pick_last = true;
        ignore_comma_for_primary = false;
    }
    // Main scan.
    bool unstressed = false;
    bool secondary_next = false;
    size_t last_strong_pos = (size_t)-1;
    size_t last_schwa_pos = (size_t)-1;
    size_t secondary_vowel_pos = (size_t)-1;
    size_t insert_pos = (size_t)-1;
    size_t pi = 0;
    bool stop_main = false;
    while (pi < ph->count && !stop_main) {
        char c = ph->data[pi];
        if (c == '\'') { stop_main = true; }
        else if (c == ',') {
            if (!ignore_comma_for_primary || pi == 0) {
                secondary_next = true;
            }
            pi++;
        } else if (c == '%' || c == '=') {
            unstressed = true; pi++;
        } else {
            const char * code = NULL;
            size_t code_len = 0;
            find_phoneme_code(ph->data, ph->count, pi,
                              &code, &code_len);
            if (is_vowel_code(code, code_len)) {
                if (secondary_next) {
                    secondary_next = false;
                    unstressed = false;
                    if (secondary_vowel_pos == (size_t)-1) {
                        secondary_vowel_pos = pi;
                    }
                    pi += code_len;
                } else if (unstressed) {
                    unstressed = false;
                    pi += code_len;
                } else if ((code_len == 2 && code[0] == '@'
                            && (code[1] == '2' || code[1] == '5'
                             || code[1] == 'L'))
                        || (code_len == 2 && code[0] == 'I'
                            && (code[1] == '#' || code[1] == '2'))
                        || (code_len == 2 && code[0] == 'a'
                            && code[1] == '#')
                        || (code_len == 1 && code[0] == 'i')) {
                    pi += code_len;
                } else if ((code_len == 1 && code[0] == '@')
                        || (code_len == 1 && code[0] == '3')) {
                    if (!has_strong_after(ph->data, ph->count,
                                          pi + code_len)) {
                        last_schwa_pos = pi;
                        if (!pick_last && insert_pos == (size_t)-1) {
                            insert_pos = pi;
                        }
                    }
                    pi += code_len;
                } else {
                    if (pick_last) {
                        if (use_pick_last_for_secondary
                            && code_len == 1 && code[0] == 'I') {
                            last_schwa_pos = pi;
                        } else {
                            last_strong_pos = pi;
                        }
                        pi += code_len;
                    } else {
                        insert_pos = pi;
                        stop_main = true;
                    }
                }
            } else {
                pi += code_len;
            }
        }
    }
    if (pick_last) {
        insert_pos = last_strong_pos != (size_t)-1
            ? last_strong_pos : last_schwa_pos;
    }
#if 0
    // Centering/initial-diphthong skip: if primary landed on a
    // centering or initial diphthong, look forward for a better
    // candidate. Ported as dead code (matches the .cpp's disabled
    // branch — both is_diphthong and is_centering are hardcoded
    // false; there's no path to flip them). Preserved here so the
    // logic can be recovered if a future dialect / lemma override
    // needs it. The .cpp original lived around phonemizer.cpp:
    // 4913-4983.
    else if (insert_pos != (size_t)-1) {
        const char * found_code = NULL;
        size_t found_len = 0;
        find_phoneme_code(ph->data, ph->count, insert_pos,
                          &found_code, &found_len);
        // disabled: initial diphthong IS the primary stress target
        // (e.g. "apricot" eIprIk0t -> ˈeɪpɹɪkˌɑːt).
        bool is_diphthong = false;
        // disabled: 'o@'/'e@' are primary stress targets in en-us.
        bool is_centering = false;
        if (is_diphthong || is_centering) {
            static const char * const CENTERING_DIPHS[] = {
                "aI@3","aU@r","i@3r","aI@","aI3","aU@","i@3",
                "3:r","A:r","o@r","e@r","e@","i@","U@","o@",
                "3:","A:","i:","u:","O:","e:","a:","aa",
                NULL
            };
            static const char * const ALL_DIPHS[] = {
                "aI@3","aU@r","i@3r","aI@","aI3","aU@","i@3",
                "aI","aU","eI","OI","oU",
                "3:r","A:r","o@r","e@r","e@","i@","U@","o@",
                "3:","A:","i:","u:","O:","e:","a:","aa",
                NULL
            };
            const char * const * skip_list = is_centering
                ? CENTERING_DIPHS : ALL_DIPHS;
            size_t pi2 = insert_pos + found_len;
            bool better_found = false;
            size_t better_pos = (size_t)-1;
            bool unst2 = false;
            bool sec2 = false;
            while (pi2 < ph->count && !better_found) {
                char c2 = ph->data[pi2];
                if (c2 == ',') {
                    sec2 = true;
                    pi2++;
                } else if (c2 == '%' || c2 == '=') {
                    unst2 = true;
                    pi2++;
                } else if (c2 == '\'') {
                    pi2++;
                } else {
                    const char * code2 = NULL;
                    size_t code2_len = 0;
                    find_phoneme_code(ph->data, ph->count, pi2,
                                      &code2, &code2_len);
                    if (is_vowel_code(code2, code2_len)) {
                        bool is_skip = false;
                        for (int di = 0;
                             skip_list[di] != NULL && !is_skip;
                             di++) {
                            size_t sl = strlen(skip_list[di]);
                            if (sl == code2_len
                                && memcmp(code2, skip_list[di],
                                          sl) == 0) {
                                is_skip = true;
                            }
                        }
                        bool weak_v = (code2_len == 1
                            && (code2[0] == '@' || code2[0] == '3'
                                || code2[0] == 'i'))
                            || (code2_len == 2
                                && ((code2[0] == '@' && (code2[1] == '2'
                                    || code2[1] == '5' || code2[1] == 'L'))
                                    || (code2[0] == 'I'
                                        && (code2[1] == '#' || code2[1] == '2'))
                                    || (code2[0] == 'a' && code2[1] == '#')));
                        if (sec2 || unst2 || weak_v || is_skip) {
                            sec2 = false;
                            unst2 = false;
                            pi2 += code2_len;
                        } else {
                            better_pos = pi2;
                            better_found = true;
                        }
                    } else {
                        pi2 += code2_len;
                    }
                }
            }
            if (better_found) { insert_pos = better_pos; }
        }
    }
#endif
    // Suppress primary on schwa when ph starts with '%' OR there's
    // a secondary-marked vowel.
    bool at_schwa_or_r = insert_pos != (size_t)-1
        && insert_pos < ph->count
        && (ph->data[insert_pos] == '@'
         || (ph->data[insert_pos] == '3'
             && (insert_pos + 1 >= ph->count
              || ph->data[insert_pos + 1] != ':')));
    bool pct_lead = ph->count > 0 && ph->data[0] == '%';
    bool has_secondary = secondary_vowel_pos != (size_t)-1;
    bool suppress_schwa = at_schwa_or_r
        && (pct_lead || has_secondary);
    if (insert_pos != (size_t)-1 && !suppress_schwa) {
        chars_grow(ph, ph->count + 2);
        memmove(ph->data + insert_pos + 1,
                ph->data + insert_pos,
                ph->count - insert_pos);
        ph->data[insert_pos] = '\'';
        ph->count += 1;
        ph->data[ph->count] = '\0';
    } else {
        // Last resort: stress 'a#' as 'a'.
        size_t pi2 = 0;
        bool done = false;
        while (pi2 < ph->count && !done) {
            const char * code = NULL;
            size_t code_len = 0;
            find_phoneme_code(ph->data, ph->count, pi2,
                              &code, &code_len);
            if (code_len == 2 && code[0] == 'a' && code[1] == '#') {
                chars_grow(ph, ph->count + 2);
                memmove(ph->data + pi2 + 1,
                        ph->data + pi2,
                        ph->count - pi2);
                ph->data[pi2] = '\'';
                ph->count += 1;
                ph->data[ph->count] = '\0';
                bool has_variant_digit = pi2 + 3 < ph->count
                    && ph->data[pi2 + 3] >= '1'
                    && ph->data[pi2 + 3] <= '9'
                    && ph->data[pi2 + 3] != '3'
                    && ph->data[pi2 + 3] != '8';
                if (has_variant_digit) {
                    // Remove '#' at pi2+2.
                    memmove(ph->data + pi2 + 2,
                            ph->data + pi2 + 3,
                            ph->count - pi2 - 3);
                    ph->count -= 1;
                    ph->data[ph->count] = '\0';
                    // Strip trailing variant digits.
                    while (pi2 + 2 < ph->count) {
                        char dc = ph->data[pi2 + 2];
                        if (dc >= '1' && dc <= '9' && dc != '3'
                            && dc != '8') {
                            memmove(ph->data + pi2 + 2,
                                    ph->data + pi2 + 3,
                                    ph->count - pi2 - 3);
                            ph->count -= 1;
                            ph->data[ph->count] = '\0';
                        } else {
                            break;
                        }
                    }
                }
                done = true;
            } else if (is_vowel_code(code, code_len)) {
                done = true;
            } else {
                pi2 += code_len;
            }
        }
    }
}

// Step 5a: place secondary stress at even syllable distances from
// primary.
static bool place_secondary_stress(struct chars * ph,
                                   const char * ph_in, size_t in_n,
                                   bool step50_fired,
                                   const bool * rba,
                                   size_t rba_n) {
    if (memchr(ph->data, '\'', ph->count) == NULL
        || memchr(ph_in, ',', in_n) != NULL) {
        return false;
    }
    struct sylinfo {
        size_t pos;
        const char * code;
        size_t code_len;
        bool stressable;
        bool is_primary;
        bool already_secondary;
    };
    enum { MAX_SYL6 = 256 };
    struct sylinfo syls[MAX_SYL6];
    int n = 0;
    size_t pi = 0;
    bool unstressed_prefix = false;
    bool secondary_prefix = false;
    bool primary_prefix = false;
    while (pi < ph->count && n < MAX_SYL6) {
        char c = ph->data[pi];
        if (c == '\'') { primary_prefix = true; pi++; }
        else if (c == ',') { secondary_prefix = true; pi++; }
        else if (c == '%' || c == '=') {
            unstressed_prefix = true; pi++;
        }
        else {
            const char * code = NULL;
            size_t code_len = 0;
            find_phoneme_code(ph->data, ph->count, pi,
                              &code, &code_len);
            // Centering-diphthong split via rule boundaries.
            if (code_len >= 3 && !primary_prefix && !secondary_prefix) {
                char last = code[code_len - 1];
                if (last == '@' || last == '3') {
                    size_t pi_orig = pi;
                    for (size_t q = 0; q < pi; q++) {
                        char qc = ph->data[q];
                        if (qc == '\'' || qc == ',' || qc == '%'
                            || qc == '=') { pi_orig--; }
                    }
                    bool has_boundary = false;
                    for (size_t k = 0;
                         k + 1 < code_len && !has_boundary; k++) {
                        size_t check_pos = pi_orig + k;
                        if (check_pos < rba_n && rba[check_pos]) {
                            has_boundary = true;
                        }
                    }
                    if (has_boundary) {
                        code_len--;
                    }
                }
            }
            if (is_vowel_code(code, code_len)) {
                if (code_len == 1 && code[0] == '@'
                    && pi + 1 < ph->count
                    && ph->data[pi + 1] == '-') {
                    pi += code_len;
                } else {
                    bool is_schwa =
                        (code_len == 1 && (code[0] == '@'
                                        || code[0] == '3'))
                     || (code_len == 2 && code[0] == '@'
                         && (code[1] == '2' || code[1] == '5'
                          || code[1] == 'L'));
                    bool is_reduced =
                        (code_len == 2 && code[0] == 'I'
                         && (code[1] == '#' || code[1] == '2'))
                     || (code_len == 1 && code[0] == 'i')
                     || (code_len == 2 && code[0] == 'a'
                         && code[1] == '#');
                    syls[n].pos = pi;
                    syls[n].code = code;
                    syls[n].code_len = code_len;
                    syls[n].is_primary = primary_prefix;
                    syls[n].already_secondary = secondary_prefix;
                    syls[n].stressable = !is_schwa && !is_reduced
                        && !unstressed_prefix;
                    n++;
                    primary_prefix = false;
                    secondary_prefix = false;
                    unstressed_prefix = false;
                    pi += code_len;
                }
            } else {
                pi += code_len;
            }
        }
    }
    int primary_idx = -1;
    for (int si = 0; si < n && primary_idx < 0; si++) {
        if (syls[si].is_primary) { primary_idx = si; }
    }
    if (primary_idx >= 0 && n >= 3) {
        int to_mark[MAX_SYL6];
        int n_mark = 0;
        // Backward scan: leftmost stressable at dist >= 2 from
        // primary, then cascade rightward every 2 syllables.
        if (primary_idx >= 2 && !step50_fired
            && memchr(ph->data, ',', ph->count) == NULL) {
            int first_sec = -1;
            for (int idx = 0;
                 idx <= primary_idx - 2 && first_sec < 0; idx++) {
                if (syls[idx].stressable
                    && !syls[idx].already_secondary
                    && !syls[idx].is_primary) {
                    first_sec = idx;
                }
            }
            if (first_sec >= 0) {
                to_mark[n_mark++] = first_sec;
                for (int idx = first_sec + 2;
                     idx <= primary_idx - 2; idx += 2) {
                    if (syls[idx].stressable
                        && !syls[idx].already_secondary
                        && !syls[idx].is_primary) {
                        to_mark[n_mark++] = idx;
                    }
                }
                // Even-distance pass from primary backwards.
                for (int dist = 2;
                     primary_idx - dist >= 0; dist += 2) {
                    int idx = primary_idx - dist;
                    if (syls[idx].stressable
                        && !syls[idx].already_secondary
                        && !syls[idx].is_primary) {
                        bool too_close = false;
                        for (int m = 0; m < n_mark; m++) {
                            int diff = to_mark[m] - idx;
                            if (diff < 0) { diff = -diff; }
                            if (diff < 2) { too_close = true; }
                        }
                        if (!too_close) { to_mark[n_mark++] = idx; }
                    }
                }
            }
        }
        // Forward scan: distance 2, 4, ... from primary.
        int stressable_after = 0;
        for (int idx = primary_idx + 1; idx < n; idx++) {
            if (syls[idx].stressable) { stressable_after++; }
        }
        if (stressable_after >= 1) {
            bool stop_fwd = false;
            for (int dist = 2;
                 primary_idx + dist < n && !stop_fwd; dist += 2) {
                int idx = primary_idx + dist;
                bool later_primary = false;
                for (int k = idx + 1; k < n && !later_primary; k++) {
                    if (syls[k].is_primary) { later_primary = true; }
                }
                if (later_primary) {
                    stop_fwd = true;
                } else if (syls[idx].stressable
                    && !syls[idx].already_secondary
                    && !syls[idx].is_primary) {
                    to_mark[n_mark++] = idx;
                } else if (!syls[idx].stressable
                    && !syls[idx].is_primary) {
                    int idx2 = primary_idx + dist + 1;
                    if (idx2 < n && syls[idx2].stressable
                        && !syls[idx2].already_secondary
                        && !syls[idx2].is_primary) {
                        bool later_p2 = false;
                        for (int k = idx2 + 1;
                             k < n && !later_p2; k++) {
                            if (syls[k].is_primary) {
                                later_p2 = true;
                            }
                        }
                        if (!later_p2) { to_mark[n_mark++] = idx2; }
                    }
                }
            }
        }
        // Sort descending so insertions don't shift earlier indices.
        for (int i = 0; i < n_mark; i++) {
            for (int j = i + 1; j < n_mark; j++) {
                if (to_mark[j] > to_mark[i]) {
                    int t = to_mark[i];
                    to_mark[i] = to_mark[j];
                    to_mark[j] = t;
                }
            }
        }
        for (int i = 0; i < n_mark; i++) {
            int idx = to_mark[i];
            chars_grow(ph, ph->count + 2);
            memmove(ph->data + syls[idx].pos + 1,
                    ph->data + syls[idx].pos,
                    ph->count - syls[idx].pos);
            ph->data[syls[idx].pos] = ',';
            ph->count += 1;
            ph->data[ph->count] = '\0';
        }
    }
    return true;
}

// process_phoneme_string: full prosody pipeline. Steps 1..6f all
// land in this single orchestrator. `is_strend` becomes
// force_final_stress in the stress phases. Dialect is hard-coded
// to en-us (the only one this repo ships).
static void process_phoneme_string(const char * in, size_t in_n,
                                   bool is_strend,
                                   struct chars * out) {
    out->count = 0;
    // Snapshot of input before \x01 strip; stress phases consult
    // it to detect pre-existing ',' / '\''.
    struct chars ph_in = {0};
    chars_put(&ph_in, in, in_n);
    chars_put(out, in, in_n);
    enum { MAX_PH = 512 };
    bool rba[MAX_PH] = {0};
    size_t rba_n = 0;
    strip_rule_boundary_markers(out, rba, MAX_PH, &rba_n);
    // \x02 (phonSTRESS_PREV-demoted secondary) -> ','.
    for (size_t i = 0; i < out->count; i++) {
        if (out->data[i] == '\x02') { out->data[i] = ','; }
    }
    bool is_en_us = true;
    // Steps 1-3c.
    apply_velar_nasal_assimilation(out);
    if (is_en_us) {
        apply_happy_tensing(out);
        apply_vowel_reduction(out);
        apply_lot_plus_r_merge(out);
    }
    strip_morpheme_schwa_r(out);
    // Steps 4 + 4b.
    if (is_en_us) {
        apply_bare_schwa_to_rhotic(out);
        apply_linking_r(out);
    }
    // Steps 4c/d/e.
    if (is_en_us) {
        apply_tion_stress_fix(out);
        apply_ology_stress_fix(out);
        apply_ic_stress_fix(out);
    }
    // Step 5 (primary) + 5.0 (= suffix shift) + 5a (secondary)
    // + cleanup / trochaic / final-syllable secondaries.
    bool starts_with_secondary = out->count > 0
        && out->data[0] == ',';
    bool is_rule_leading_comma = starts_with_secondary && rba_n > 0;
    insert_primary_stress(out, is_strend, rba, rba_n);
    bool step50_fired = apply_equals_suffix_stress_shift(out);
    bool step5a_ran = place_secondary_stress(out,
                                             ph_in.data, ph_in.count,
                                             step50_fired,
                                             rba, rba_n);
    cleanup_adjacent_secondary(out, step5a_ran,
                               is_rule_leading_comma, rba, rba_n);
    trochaic_compound_prefix(out, ph_in.data, ph_in.count,
                             step5a_ran, starts_with_secondary);
    final_syllable_secondary(out);
    // Late phases (5.5+ / 6+).
    if (is_en_us) { reduce_zero_diminished(out); }
    demote_adjacent_primaries(out, ph_in.data, ph_in.count);
    if (is_en_us) { reduce_bare_zero_after_ution(out); }
    if (is_en_us) { reduce_e_unstressed(out, rba, rba_n); }
    if (is_en_us) { reduce_v_between_sec_and_primary(out); }
    if (is_en_us) { reduce_a_before_primary(out, rba_n > 0); }
    if (is_en_us) { reduce_a_between_primary_and_sec(out, rba_n > 0); }
    if (is_en_us) { reduce_zero_between_sec_and_primary(out); }
    if (is_en_us) { reduce_zero_between_primary_and_sec(out); }
    if (is_en_us) { reduce_zero_hash_before_primary(out); }
    if (is_en_us) { reduce_bare_zero_before_primary(out, rba, rba_n); }
    if (is_en_us) { reduce_e_pre_tonic_after_pct_nasal(out); }
    if (is_en_us) { post_stress_i_before_syllabic_l(out); }
    if (is_en_us) { reduce_3_long_to_short(out); }
    if (is_en_us) { apply_flap_rule(out); }
    if (is_en_us) { reduce_ness_suffix(out); }
    if (is_en_us) { reduce_ou_hash_compound(out); }
    if (is_en_us) { elide_ically_schwa(out); }
    add_compound_syllabic_n_stress(out);
    chars_free(&ph_in);
}

static void word_to_phonemes(struct phonemizer * p,
                             const char * word, size_t wn,
                             struct chars * out);

// ---------------------------------------------------------------------------
// Dict-lookup arms of the wordToPhonemes dispatch chain (D-15).
// Each returns the phoneme string in *out (non-empty) iff the arm
// claimed the word; out stays empty when the arm passes.
// ---------------------------------------------------------------------------

// $capital: when word starts with an uppercase letter, look it up
// in capital_dict by its lowercased form.
static void check_capital_dict(struct phonemizer * p,
                               const char * word, size_t wn,
                               const char * norm, size_t nn,
                               struct chars * out) {
    out->count = 0;
    bool is_capital = wn > 0
        && (unsigned char)word[0] >= 'A'
        && (unsigned char)word[0] <= 'Z';
    if (is_capital) {
        struct chars * v = smap_get(&p->capital_dict, norm, nn);
        if (v != NULL) {
            process_phoneme_string(v->data, v->count, false, out);
        }
    }
}

// Try full word in dictionary. $onlys bare-word entries take
// priority over plain entries.
static void check_main_dict(struct phonemizer * p,
                            const char * norm, size_t nn,
                            struct chars * out) {
    out->count = 0;
    struct chars * raw = smap_get(&p->onlys_bare_dict, norm, nn);
    if (raw == NULL) {
        raw = smap_get(&p->dict, norm, nn);
    }
    if (raw != NULL) {
        struct chars temp = {0};
        chars_put(&temp, raw->data, raw->count);
        int * sp = imap_get(&p->stress_pos, norm, nn);
        if (sp != NULL) {
            struct chars stressed = {0};
            apply_stress_position(temp.data, temp.count, *sp,
                                  &stressed);
            temp.count = 0;
            chars_put(&temp, stressed.data, stressed.count);
            chars_free(&stressed);
        }
        bool is_strend = set_has(&p->strend_words, norm, nn);
        process_phoneme_string(temp.data, temp.count, is_strend, out);
        chars_free(&temp);
    }
}

// Hyphenated compound: split on '-', phonemize each segment via
// word_to_phonemes, concatenate. Returns empty if any segment is
// empty or contains no letters or fails to phonemize.
static void check_hyphenated(struct phonemizer * p,
                             const char * norm, size_t nn,
                             struct chars * out) {
    out->count = 0;
    const char * first_hyphen = memchr(norm, '-', nn);
    bool has_hyphen = first_hyphen != NULL
        && first_hyphen > norm
        && (size_t)(first_hyphen - norm) + 1 < nn;
    if (has_hyphen) {
        struct chars accum = {0};
        size_t seg_start = 0;
        bool processing = true;
        while (seg_start < nn && processing) {
            const char * next_h = memchr(
                norm + seg_start, '-', nn - seg_start);
            size_t seg_end = next_h != NULL
                ? (size_t)(next_h - norm) : nn;
            size_t seg_len = seg_end - seg_start;
            if (seg_len == 0) {
                processing = false;
            } else {
                bool has_letter = false;
                for (size_t k = 0; k < seg_len && !has_letter; k++) {
                    if (isalpha((unsigned char)norm[seg_start + k])) {
                        has_letter = true;
                    }
                }
                if (!has_letter) {
                    processing = false;
                } else {
                    struct chars seg_ipa = {0};
                    word_to_phonemes(p, norm + seg_start, seg_len,
                                     &seg_ipa);
                    if (seg_ipa.count == 0) {
                        processing = false;
                    } else {
                        chars_put(&accum, seg_ipa.data,
                                  seg_ipa.count);
                        seg_start = next_h != NULL
                            ? (size_t)(next_h - norm) + 1
                            : nn;
                    }
                    chars_free(&seg_ipa);
                }
            }
        }
        if (processing && accum.count > 0) {
            chars_put(out, accum.data, accum.count);
        }
        chars_free(&accum);
    }
}

// Find the last phoneme code in `rc[0..rcn)` (skipping stress and
// boundary markers). Writes it to *out (caller-owned).
static void last_phoneme_code(const char * rc, size_t rcn,
                              struct chars * out) {
    out->count = 0;
    size_t ri = rcn;
    bool found = false;
    while (ri > 0 && !found) {
        ri--;
        char c = rc[ri];
        bool is_marker = c == '\'' || c == ',' || c == '%'
                      || c == '=' || c == '-';
        if (!is_marker) {
            chars_put_byte(out, c);
            if (ri >= 1) {
                char two[2] = { rc[ri - 1], c };
                bool is_two_char =
                    (two[0] == 't' && two[1] == 'S')
                 || (two[0] == 'd' && two[1] == 'Z')
                 || (two[0] == 'O' && two[1] == ':')
                 || (two[0] == 'A' && two[1] == ':')
                 || (two[0] == 'i' && two[1] == ':')
                 || (two[0] == 'u' && two[1] == ':')
                 || (two[0] == 'e' && two[1] == ':')
                 || (two[0] == 'I' && two[1] == '#')
                 || (two[0] == 'I' && two[1] == '2')
                 || (two[0] == '@' && two[1] == 'L')
                 || (two[0] == '3' && two[1] == ':')
                 || (two[0] == 'e' && two[1] == 'I')
                 || (two[0] == 'a' && two[1] == 'I')
                 || (two[0] == 'a' && two[1] == 'U')
                 || (two[0] == 'o' && two[1] == 'U')
                 || (two[0] == 'O' && two[1] == 'I');
                if (is_two_char) {
                    out->count = 0;
                    chars_put(out, two, 2);
                }
            }
            found = true;
        }
    }
}

// Possessive "'s" suffix. Word ends in apostrophe-s; phonemize the
// base + a voiced/unvoiced/syllabic 's' allomorph based on the
// orthographic + phonemic context.
static void check_possessive(struct phonemizer * p,
                             const char * norm, size_t nn,
                             struct chars * out) {
    out->count = 0;
    bool is_possessive = nn >= 3
        && norm[nn - 2] == '\''
        && norm[nn - 1] == 's';
    if (is_possessive) {
        size_t base_n = nn - 2;
        struct chars base_ipa = {0};
        word_to_phonemes(p, norm, base_n, &base_ipa);
        if (base_ipa.count > 0) {
            struct chars raw_code = {0};
            // Look up base in dict; else apply rules.
            struct chars * d = smap_get(&p->dict, norm, base_n);
            if (d != NULL) {
                process_phoneme_string(d->data, d->count, false,
                                       &raw_code);
            } else {
                struct chars rules_out = {0};
                apply_rules(p, norm, base_n, true, -1,
                            false, false,
                            NULL, 0, NULL, 0, &rules_out);
                process_phoneme_string(rules_out.data,
                                       rules_out.count, false,
                                       &raw_code);
                chars_free(&rules_out);
            }
            // Determine the suffix allomorph from base orthography
            // + last phoneme code.
            const char * poss = "z";
            size_t poss_n = 1;
            struct chars last_ph = {0};
            last_phoneme_code(raw_code.data, raw_code.count, &last_ph);
            bool ends_och = base_n >= 3
                && memcmp(norm + base_n - 3, "och", 3) == 0;
            bool ends_ch = base_n >= 2
                && memcmp(norm + base_n - 2, "ch", 2) == 0;
            bool ends_se = base_n >= 2
                && memcmp(norm + base_n - 2, "se", 2) == 0;
            bool ends_ce = base_n >= 2
                && memcmp(norm + base_n - 2, "ce", 2) == 0;
            bool ends_sh = base_n >= 2
                && memcmp(norm + base_n - 2, "sh", 2) == 0;
            if (ends_och) {
                poss = "s"; poss_n = 1;
            } else if (ends_ch) {
                // Sibilant phonemes (tS / dZ / s / z / S / Z) -> I2z.
                bool is_sib =
                    (last_ph.count == 2 && memcmp(last_ph.data, "tS", 2) == 0)
                 || (last_ph.count == 2 && memcmp(last_ph.data, "dZ", 2) == 0)
                 || (last_ph.count == 1 && last_ph.data[0] == 's')
                 || (last_ph.count == 1 && last_ph.data[0] == 'z')
                 || (last_ph.count == 1 && last_ph.data[0] == 'S')
                 || (last_ph.count == 1 && last_ph.data[0] == 'Z');
                static const char unvoiced[] = "ptkfsTSx";
                if (is_sib) {
                    poss = "I2z"; poss_n = 3;
                } else if (last_ph.count > 0
                    && strchr(unvoiced, last_ph.data[0]) != NULL) {
                    poss = "s"; poss_n = 1;
                } else {
                    poss = "z"; poss_n = 1;
                }
            } else if (ends_se || ends_ce || ends_sh) {
                poss = "I#z"; poss_n = 3;
            } else if (base_n > 0) {
                char lc = norm[base_n - 1];
                if (lc == 's' || lc == 'z' || lc == 'x') {
                    poss = "I#z"; poss_n = 3;
                } else if (lc == 'f' || lc == 'p'
                        || lc == 't' || lc == 'k') {
                    poss = "s"; poss_n = 1;
                }
            }
            struct chars combined = {0};
            chars_put(&combined, raw_code.data, raw_code.count);
            chars_put(&combined, poss, poss_n);
            process_phoneme_string(combined.data, combined.count,
                                   false, out);
            chars_free(&combined);
            chars_free(&last_ph);
            chars_free(&raw_code);
        }
        chars_free(&base_ipa);
    }
}

// Single-letter word (not "a" article): try "_X" key in dict.
static void check_single_letter(struct phonemizer * p,
                                const char * norm, size_t nn,
                                struct chars * out) {
    out->count = 0;
    bool is_single = nn == 1
        && !(nn == 1 && norm[0] == 'a');
    if (is_single) {
        char key[2] = { '_', norm[0] };
        struct chars * v = smap_get(&p->dict, key, 2);
        if (v != NULL) {
            process_phoneme_string(v->data, v->count, false, out);
        }
    }
}

// Compound-prefix decomposition: find a prefix from
// compound_prefixes that matches the word, recurse on the suffix,
// combine with stress demotion.
static void check_compound_prefixes(struct phonemizer * p,
                                    const char * norm, size_t nn,
                                    struct chars * out) {
    out->count = 0;
    if (nn < 5 || p->compound_prefixes.count == 0) { return; }
    bool found = false;
    for (size_t i = 0;
         i < p->compound_prefixes.count && !found; i++) {
        const struct strpair * cp = &p->compound_prefixes.data[i];
        const struct chars * pref = &cp->a;
        const struct chars * pref_ph = &cp->b;
        bool live = pref->count >= 4 && pref->count < nn;
        size_t sfx_len = 0;
        if (live) {
            sfx_len = nn - pref->count;
            if (sfx_len < 2) { live = false; }
            if (live && memcmp(norm, pref->data, pref->count) != 0) {
                live = false;
            }
            if (live && !has_any_vowel_letter(
                    norm + pref->count, sfx_len)) {
                live = false;
            }
            // Suffix must be >=4 chars OR be a recognized dict word.
            if (live && sfx_len < 4) {
                bool in_dict = smap_get(&p->dict,
                                        norm + pref->count, sfx_len)
                                != NULL
                            || smap_get(&p->verb_dict,
                                        norm + pref->count, sfx_len)
                                != NULL;
                if (!in_dict) { live = false; }
            }
        }
        if (live) {
            // Process prefix phonemes (currently pass-through).
            struct chars pfx_ph = {0};
            process_phoneme_string(pref_ph->data, pref_ph->count,
                                   false, &pfx_ph);
            // Count vowels in pfx_ph (multi-char-aware).
            static const char * const MC_VOWELS[] = {
                "O@","o@","U@","A@","e@","i@","aI@3","aI3",
                "aU@","aI@","i@3","3:r","A:r","o@r","A@r","e@r",
                "eI","aI","aU","OI","oU","IR","VR","3:","A:",
                "i:","u:","O:","e:","a:","aa","@L","@2","@5",
                "I2","I#","E2","E#","e#","a#","a2","0#","02",
                "O2","A#", NULL
            };
            int nvowels = 0;
            size_t pi = 0;
            while (pi < pfx_ph.count) {
                char c = pfx_ph.data[pi];
                if (c == '\'' || c == ','
                    || c == '%' || c == '=') {
                    pi++;
                } else {
                    bool matched = false;
                    for (int mi = 0;
                         MC_VOWELS[mi] != NULL && !matched; mi++) {
                        size_t ml = strlen(MC_VOWELS[mi]);
                        if (pi + ml <= pfx_ph.count
                            && memcmp(pfx_ph.data + pi,
                                      MC_VOWELS[mi], ml) == 0) {
                            nvowels++;
                            pi += ml;
                            matched = true;
                        }
                    }
                    if (!matched) {
                        if (is_vowel_code(&pfx_ph.data[pi], 1)) {
                            nvowels++;
                        }
                        pi++;
                    }
                }
            }
            if (nvowels >= 2) {
                replace_first_char(&pfx_ph, '\'', ',');
            } else {
                // Strip all stress markers.
                size_t w = 0;
                for (size_t r = 0; r < pfx_ph.count; r++) {
                    if (pfx_ph.data[r] != '\''
                        && pfx_ph.data[r] != ',') {
                        pfx_ph.data[w++] = pfx_ph.data[r];
                    }
                }
                pfx_ph.count = w;
                if (pfx_ph.data != NULL) {
                    pfx_ph.data[pfx_ph.count] = '\0';
                }
            }
            // Recurse on suffix via full word_to_phonemes.
            struct chars sfx_ph = {0};
            word_to_phonemes(p, norm + pref->count, sfx_len,
                             &sfx_ph);
            struct chars combined = {0};
            chars_put(&combined, pfx_ph.data, pfx_ph.count);
            chars_put(&combined, sfx_ph.data, sfx_ph.count);
            // $N stress override on the full word.
            int * sp = imap_get(&p->stress_pos, norm, nn);
            if (sp != NULL) {
                struct chars stressed = {0};
                apply_stress_position(combined.data, combined.count,
                                      *sp, &stressed);
                struct chars processed = {0};
                process_phoneme_string(stressed.data,
                                       stressed.count, false,
                                       &processed);
                combined.count = 0;
                chars_put(&combined, processed.data, processed.count);
                chars_free(&stressed);
                chars_free(&processed);
            }
            chars_put(out, combined.data, combined.count);
            chars_free(&combined);
            chars_free(&sfx_ph);
            chars_free(&pfx_ph);
            found = true;
        }
    }
}

// applyRulesFallback: terminal rules path (used when nothing else
// in the dispatch chain claims the word). apply_rules + $N stress
// override + process_phoneme_string.
static void apply_rules_fallback(struct phonemizer * p,
                                 const char * norm, size_t nn,
                                 struct chars * out) {
    struct chars raw_ph = {0};
    apply_rules(p, norm, nn, true, -1, false, false,
                NULL, 0, NULL, 0, &raw_ph);
    int * sp = imap_get(&p->stress_pos, norm, nn);
    if (sp != NULL) {
        struct chars stressed = {0};
        apply_stress_position(raw_ph.data, raw_ph.count, *sp,
                              &stressed);
        raw_ph.count = 0;
        chars_put(&raw_ph, stressed.data, stressed.count);
        chars_free(&stressed);
    }
    process_phoneme_string(raw_ph.data, raw_ph.count, false, out);
    chars_free(&raw_ph);
}

// ---------------------------------------------------------------------------
// Morphological-suffix arms (D-25). Each tries to claim the word by
// stripping a known suffix and recursing into get_stem_phonemes for
// the stem. First non-empty result wins in check_morphological_-
// suffixes (the dispatcher). All arms write into a caller-owned
// struct chars * out; "out->count == 0" means "this arm passed".
// ---------------------------------------------------------------------------

// Stem phonemizer. Tries verb_dict, dict, magic-e variants, and a
// rules + stress-position fallback. Returns raw dict phonemes (with
// \x01 boundaries) for dict hits, or processed phonemes from the
// rules path. Empty out when stem is too short, has no vowel letter,
// or the result has no vowel phoneme code.
static void get_stem_phonemes(struct phonemizer * p,
                              const char * stem, size_t sn,
                              struct chars * out) {
    out->count = 0;
    if (sn >= 2 && has_any_vowel_letter(stem, sn)) {
        struct chars * vt = smap_get(&p->verb_dict, stem, sn);
        if (vt != NULL) {
            chars_put(out, vt->data, vt->count);
        } else {
            bool is_onlys = set_has(&p->onlys_words, stem, sn)
                         || set_has(&p->only_words, stem, sn);
            struct chars * jt = smap_get(&p->dict, stem, sn);
            if (jt == NULL || is_onlys) {
                struct chars sx = {0};
                chars_put(&sx, stem, sn);
                chars_put_byte(&sx, 'e');
                struct chars * je = smap_get(&p->dict,
                                             sx.data, sx.count);
                bool je_onlys = je != NULL && (
                    set_has(&p->onlys_words, sx.data, sx.count) ||
                    set_has(&p->only_words, sx.data, sx.count));
                if (je != NULL && !je_onlys) {
                    chars_put(out, je->data, je->count);
                } else {
                    struct chars * ve = smap_get(&p->verb_dict,
                                                 sx.data, sx.count);
                    if (ve != NULL) {
                        chars_put(out, ve->data, ve->count);
                    }
                }
                chars_free(&sx);
            }
            if (out->count == 0 && jt != NULL && !is_onlys) {
                chars_put(out, jt->data, jt->count);
            }
            if (out->count == 0) {
                int stem_alt_flags =
                    set_has(&p->verb_flag_words, stem, sn) ? 1 : -1;
                struct chars raw = {0};
                apply_rules(p, stem, sn, true, stem_alt_flags,
                            false, false, NULL, 0, NULL, 0, &raw);
                int * sp = imap_get(&p->stress_pos, stem, sn);
                if (sp != NULL
                    && !set_has(&p->noun_form_stress, stem, sn)
                    && !set_has(&p->verb_flag_words, stem, sn)) {
                    struct chars stressed = {0};
                    apply_stress_position(raw.data, raw.count,
                                          *sp, &stressed);
                    raw.count = 0;
                    chars_put(&raw, stressed.data, stressed.count);
                    chars_free(&stressed);
                }
                process_phoneme_string(raw.data, raw.count, false,
                                       out);
                chars_free(&raw);
            }
        }
        if (out->count > 0
            && !has_any_vowel_code(out->data, out->count)) {
            out->count = 0;
        }
    }
}

// -ing non-magic-e stem fallbacks. -ns/-rs stems prefer magic-e
// because the bare-stem 's' fires as 'z'. Each step is skipped when
// sph is already populated.
static void try_ing_non_magic_e_stem_fallbacks(
        struct phonemizer * p,
        const char * base, size_t bn,
        bool base_has_stress_override,
        struct chars * sph) {
    bool try_magic_e_first = !base_has_stress_override && bn >= 2
        && (memcmp(base + bn - 2, "ns", 2) == 0
            || memcmp(base + bn - 2, "rs", 2) == 0);
    if (try_magic_e_first) {
        struct chars sx = {0};
        chars_put(&sx, base, bn);
        chars_put_byte(&sx, 'e');
        get_stem_phonemes(p, sx.data, sx.count, sph);
        chars_free(&sx);
    }
    if (sph->count == 0) { get_stem_phonemes(p, base, bn, sph); }
    if (sph->count == 0 && bn > 0
        && !is_vowel_letter(base[bn - 1])
        && has_any_vowel_letter(base, bn)) {
        struct chars sx = {0};
        chars_put(&sx, base, bn);
        chars_put_byte(&sx, 'e');
        get_stem_phonemes(p, sx.data, sx.count, sph);
        chars_free(&sx);
    }
}

// -ed non-magic-e stem fallbacks. Doubled-consonant CVC stems with
// 2+ vowel groups prefer the undoubled form (e.g. "controlled" ->
// "control"). Otherwise: -rs/-ns magic-e first, then bare stem,
// then secondary fallbacks.
static void try_ed_non_magic_e_stem_fallbacks(
        struct phonemizer * p,
        const char * base, size_t bn,
        bool base_has_stress_override,
        struct chars * sph) {
    static const char CVC_DOUBLE_CONS[] = "lptmnrgdb";
    bool base_has_double = bn >= 2
        && base[bn - 1] == base[bn - 2]
        && strchr(CVC_DOUBLE_CONS, base[bn - 1]) != NULL;
    bool prefer_undoubled = false;
    if (base_has_double) {
        int vowel_groups = 0;
        bool in_v = false;
        size_t prefix_n = bn - 2;
        for (size_t i = 0; i < prefix_n; i++) {
            if (is_vowel_letter(base[i])) {
                if (!in_v) { vowel_groups++; in_v = true; }
            } else {
                in_v = false;
            }
        }
        prefer_undoubled = (vowel_groups >= 2);
    }
    if (sph->count == 0 && base_has_double && prefer_undoubled) {
        get_stem_phonemes(p, base, bn - 1, sph);
    }
    bool try_e_first = sph->count == 0 && !base_has_stress_override
        && bn >= 2 && (memcmp(base + bn - 2, "ns", 2) == 0
                       || memcmp(base + bn - 2, "rs", 2) == 0);
    if (try_e_first) {
        struct chars sx = {0};
        chars_put(&sx, base, bn);
        chars_put_byte(&sx, 'e');
        get_stem_phonemes(p, sx.data, sx.count, sph);
        chars_free(&sx);
    }
    if (sph->count == 0) { get_stem_phonemes(p, base, bn, sph); }
    if (sph->count == 0 && base_has_double && !prefer_undoubled) {
        get_stem_phonemes(p, base, bn - 1, sph);
    }
    if (sph->count == 0 && bn > 0
        && !is_vowel_letter(base[bn - 1])
        && has_any_vowel_letter(base, bn)) {
        struct chars sx = {0};
        chars_put(&sx, base, bn);
        chars_put_byte(&sx, 'e');
        get_stem_phonemes(p, sx.data, sx.count, sph);
        chars_free(&sx);
    }
}

// -ed candidate pre-flight: bundles 7 skip checks. Returns false
// when the word should NOT be treated as a regular -ed suffix
// (silent-e stem, 'u'-ending stem, soft-c/g without dict entry,
// "nged"/"eted" without dict, "mented" without prefix or stress,
// or 3+ trailing consonants on the base).
static bool is_ed_suffix_candidate(struct phonemizer * p,
                                   const char * norm, size_t nn,
                                   const char * base, size_t bn) {
    bool processing = true;
    if (bn > 0 && base[bn - 1] == 'e') { processing = false; }
    if (processing && bn > 0 && base[bn - 1] == 'u') {
        processing = false;
    }
    if (processing && nn >= 4) {
        char penult = norm[nn - 3];
        bool is_soft_c = (penult == 'c' || penult == 'g');
        bool is_ng_ged = (penult == 'g' && nn >= 6
                          && norm[nn - 4] == 'n');
        if (is_soft_c && !is_ng_ged
            && smap_get(&p->dict, base, bn) == NULL
            && smap_get(&p->verb_dict, base, bn) == NULL) {
            processing = false;
        }
    }
    if (processing && nn >= 5
        && memcmp(norm + nn - 4, "nged", 4) == 0
        && smap_get(&p->dict, base, bn) == NULL
        && smap_get(&p->verb_dict, base, bn) == NULL) {
        processing = false;
    }
    if (processing && nn >= 5
        && memcmp(norm + nn - 4, "eted", 4) == 0
        && smap_get(&p->dict, base, bn) == NULL
        && smap_get(&p->verb_dict, base, bn) == NULL) {
        processing = false;
    }
    if (processing && nn >= 7
        && memcmp(norm + nn - 6, "mented", 6) == 0) {
        char before_m = norm[nn - 7];
        bool stem_has_stress = imap_get(&p->stress_pos,
                                        norm, nn - 2) != NULL;
        if (!is_vowel_letter(before_m) && !stem_has_stress) {
            processing = false;
        }
    }
    if (processing) {
        int trail_cons = 0;
        int bi = (int)bn - 1;
        while (bi >= 0 && !is_vowel_letter(base[bi])) {
            trail_cons++;
            bi--;
        }
        if (trail_cons >= 3) { processing = false; }
    }
    return processing;
}

// -ed allomorph voicing: t/d -> I#d (ɪd, syllabic), unvoiced ->
// 't', voiced/vowel -> 'd'. Override: if the regular choice is 't'
// but a full-word rule fires a 'd' (e.g. "tied"/"tried"), use the
// full-word phonemes instead. Writes the final raw phonemes into
// *out (caller calls process_phoneme_string).
static void compute_ed_suffix_voicing(struct phonemizer * p,
                                      const char * sph, size_t sn,
                                      const char * norm, size_t nn,
                                      struct chars * out) {
    out->count = 0;
    static const char UNVOICED[] = "ptkfTSshx";
    char last = sph[sn - 1];
    const char * ed_ph;
    size_t ed_n;
    if (last == 't' || last == 'd') {
        ed_ph = "I#d"; ed_n = 3;
    } else if (strchr(UNVOICED, last) != NULL) {
        ed_ph = "t"; ed_n = 1;
    } else {
        ed_ph = "d"; ed_n = 1;
    }
    chars_put(out, sph, sn);
    chars_put(out, ed_ph, ed_n);
    if (ed_n == 1 && ed_ph[0] == 't' && nn >= 2) {
        bool * fw_re = calloc(nn, sizeof(bool));
        bool * fw_pv = calloc(nn, sizeof(bool));
        struct chars fw_ph = {0};
        apply_rules(p, norm, nn, false, -1, false, false,
                    fw_re, nn, fw_pv, nn, &fw_ph);
        int e_pos = (int)nn - 2;
        bool e_was_visited = (e_pos >= 0
                              && (size_t)e_pos < nn
                              && fw_pv[e_pos]);
        char fw_last = 0;
        for (int ri = (int)fw_ph.count - 1;
             ri >= 0 && fw_last == 0; ri--) {
            if (fw_ph.data[ri] != '\x01') {
                fw_last = fw_ph.data[ri];
            }
        }
        if (!e_was_visited && fw_last == 'd') {
            out->count = 0;
            chars_put(out, fw_ph.data, fw_ph.count);
        }
        chars_free(&fw_ph);
        free(fw_re);
        free(fw_pv);
    }
}

// Scan the rules for a prefix-marker rule under group `key` that
// matches at word start with non-zero advance shorter than the
// whole word. Used by has_prefix_at_start as a compound-guard.
static bool prefix_match_in_group(const struct ruleset * rs,
                                  const char * key, size_t kn,
                                  const char * w, size_t wn,
                                  int glen) {
    bool matched = false;
    struct chars view = chars_view(key, kn);
    struct rules * rv = map_get((struct map *)&rs->rule_groups,
                                &view);
    if (rv != NULL) {
        for (size_t i = 0; i < rv->count && !matched; i++) {
            const struct phoneme_rule * rule = &rv->data[i];
            if (rule->is_prefix) {
                struct chars ph = {0};
                int adv = 0, dfs = -1, dfc = 0;
                int sc = match_rule(rs, rule, w, wn, 0, &ph, &adv,
                                    &dfs, &dfc, glen, "", 0, 0,
                                    NULL, 0, false);
                matched = (sc >= 0 && adv > 0 && adv < (int)wn);
                chars_free(&ph);
            }
        }
    }
    return matched;
}

// True when the word starts with a PREFIX-marker rule whose match
// covers > 0 and < wn characters (i.e. a real compound prefix, not
// the whole word). Used by check_suffix_ed to skip the magic-e
// branch on compound words like "infrared".
static bool has_prefix_at_start(struct phonemizer * p,
                                const char * w, size_t wn) {
    bool r = false;
    if (wn > 0) {
        char c0 = (char)tolower((unsigned char)w[0]);
        if (wn >= 2) {
            char k2[2];
            k2[0] = c0;
            k2[1] = (char)tolower((unsigned char)w[1]);
            r = prefix_match_in_group(&p->rules, k2, 2, w, wn, 2);
        }
        if (!r) {
            r = prefix_match_in_group(&p->rules, &c0, 1, w, wn, 1);
        }
    }
    return r;
}

// Detect CVC / CVRC / -nc patterns for -ing/-ed magic-e candidacy.
// CVC: vowel + consonant. CVRC (when include_cvrc): vowel + 'r' +
// 'g'/'c' (soft-g/-c via magic-e). Suppress CVC for -en/-an/-in/
// -on/-un when phonemized base ends in syllabic n, for -el, -w,
// and -er endings (rhotic schwa or semi-vowel, not magic-e).
static void detect_stem_pattern_for_suffix(struct phonemizer * p,
                                           const char * base,
                                           size_t bn,
                                           bool include_cvrc,
                                           bool * out_cvc,
                                           bool * out_nc) {
    bool cvc = bn >= 2 && !is_vowel_letter(base[bn - 1])
               && is_vowel_letter(base[bn - 2]);
    if (include_cvrc && !cvc && bn >= 3
        && (base[bn - 1] == 'g' || base[bn - 1] == 'c')
        && base[bn - 2] == 'r'
        && is_vowel_letter(base[bn - 3])) {
        cvc = true;
    }
    if (cvc && bn >= 2 && base[bn - 1] == 'n') {
        char prev_vowel = base[bn - 2];
        if (prev_vowel == 'e' || prev_vowel == 'a'
            || prev_vowel == 'i' || prev_vowel == 'o'
            || prev_vowel == 'u') {
            struct chars base_ph = {0};
            get_stem_phonemes(p, base, bn, &base_ph);
            if (base_ph.count >= 2) {
                char last2 = base_ph.data[base_ph.count - 1];
                char last1 = base_ph.data[base_ph.count - 2];
                bool ends_in_schwa_n = (last1 == '@'
                    && (last2 == 'n' || last2 == 'N'));
                bool ends_in_syllabic_n = (last1 == 'n'
                                           && last2 == '-');
                if (ends_in_schwa_n || ends_in_syllabic_n) {
                    cvc = false;
                }
            }
            chars_free(&base_ph);
        }
    }
    bool nc = bn >= 2 && base[bn - 1] == 'c' && base[bn - 2] == 'n';
    if (cvc && bn >= 2 && base[bn - 1] == 'l'
        && base[bn - 2] == 'e') {
        cvc = false;
    }
    if (cvc && bn > 0 && base[bn - 1] == 'w') { cvc = false; }
    if (cvc && bn >= 2 && base[bn - 1] == 'r'
        && tolower((unsigned char)base[bn - 2]) == 'e') {
        cvc = false;
    }
    *out_cvc = cvc;
    *out_nc = nc;
}

// -ing suffix handler. Strips "ing", phonemizes the stem (with
// magic-e CVC/CVRC/-nc detection, $strend2 prefer-rules override,
// and 4 non-magic-e fallbacks), appends "%IN", runs prosody.
static void check_suffix_ing(struct phonemizer * p,
                             const char * norm, size_t nn,
                             struct chars * out) {
    out->count = 0;
    bool is_ing = (nn >= 3 && memcmp(norm + nn - 3, "ing", 3) == 0);
    if (is_ing) {
        const char * base = norm;
        size_t bn = nn - 3;
        struct chars sph = {0};
        bool processing = true;
        if (bn >= 2 && memcmp(base + bn - 2, "ng", 2) == 0
            && smap_get(&p->dict, base, bn) == NULL
            && smap_get(&p->verb_dict, base, bn) == NULL) {
            processing = false;
        }
        if (processing) {
            bool cvc_pattern = false;
            bool nc_pattern = false;
            detect_stem_pattern_for_suffix(p, base, bn, true,
                                           &cvc_pattern,
                                           &nc_pattern);
            bool base_has_stress_override =
                imap_get(&p->stress_pos, base, bn) != NULL
                || (smap_get(&p->dict, base, bn) != NULL
                    && !set_has(&p->onlys_words, base, bn)
                    && !set_has(&p->only_words, base, bn))
                || smap_get(&p->verb_dict, base, bn) != NULL;
            if ((cvc_pattern || nc_pattern)
                && !base_has_stress_override) {
                struct chars magic_e_ing = {0};
                chars_put(&magic_e_ing, base, bn);
                chars_put_byte(&magic_e_ing, 'e');
                bool use_magic_e = true;
                struct chars base_only_ph = {0};
                if (cvc_pattern && !nc_pattern) {
                    get_stem_phonemes(p, base, bn, &base_only_ph);
                    use_magic_e = should_use_magic_e_for_cvc_stem(
                        base_only_ph.data, base_only_ph.count,
                        "I@", 2);
                }
                if (use_magic_e) {
                    if (set_has(&p->strend_words,
                                magic_e_ing.data,
                                magic_e_ing.count)) {
                        struct chars rules_ph = {0};
                        apply_rules(p, magic_e_ing.data,
                                    magic_e_ing.count, true, -1,
                                    false, false,
                                    NULL, 0, NULL, 0, &rules_ph);
                        if (rules_ph.count > 0) {
                            process_phoneme_string(rules_ph.data,
                                rules_ph.count, false, &sph);
                        }
                        chars_free(&rules_ph);
                    }
                    if (sph.count == 0) {
                        get_stem_phonemes(p, magic_e_ing.data,
                                          magic_e_ing.count, &sph);
                    }
                    if (sph.count == 0) {
                        if (base_only_ph.count > 0) {
                            chars_put(&sph, base_only_ph.data,
                                      base_only_ph.count);
                        } else {
                            get_stem_phonemes(p, base, bn, &sph);
                        }
                    }
                }
                chars_free(&base_only_ph);
                chars_free(&magic_e_ing);
            } else {
                try_ing_non_magic_e_stem_fallbacks(p, base, bn,
                    base_has_stress_override, &sph);
            }
            if (sph.count == 0 && bn >= 2
                && base[bn - 1] == base[bn - 2]) {
                get_stem_phonemes(p, base, bn - 1, &sph);
            }
            if (sph.count == 0 && bn > 0 && base[bn - 1] == 'i') {
                struct chars sx = {0};
                chars_put(&sx, base, bn - 1);
                chars_put_byte(&sx, 'y');
                get_stem_phonemes(p, sx.data, sx.count, &sph);
                chars_free(&sx);
            }
        }
        if (sph.count > 0) {
            simplify_syllabic_l_for_ing(base, bn, &sph);
            bool stem_is_strend = set_has(&p->strend_words,
                                          base, bn);
            if (!stem_is_strend) {
                struct chars sx = {0};
                chars_put(&sx, base, bn);
                chars_put_byte(&sx, 'e');
                stem_is_strend = set_has(&p->strend_words,
                                         sx.data, sx.count);
                chars_free(&sx);
            }
            chars_put(&sph, "%IN", 3);
            process_phoneme_string(sph.data, sph.count,
                                   stem_is_strend, out);
        }
        chars_free(&sph);
    }
}

// -ed suffix handler. Strips "ed", phonemizes the stem (with same
// CVC/-nc magic-e logic as -ing minus CVRC, plus a y_stem path for
// 'i'-ending bases), applies allomorph voicing.
static void check_suffix_ed(struct phonemizer * p,
                            const char * norm, size_t nn,
                            struct chars * out) {
    out->count = 0;
    bool processing = (nn >= 4
        && memcmp(norm + nn - 2, "ed", 2) == 0);
    size_t bn = 0;
    const char * base = norm;
    if (processing) {
        bn = nn - 2;
        processing = is_ed_suffix_candidate(p, norm, nn, base, bn);
    }
    if (processing) {
        struct chars sph = {0};
        bool cvc_pattern = false;
        bool nc_pattern = false;
        detect_stem_pattern_for_suffix(p, base, bn, false,
                                       &cvc_pattern, &nc_pattern);
        bool base_has_stress_override2 =
            imap_get(&p->stress_pos, base, bn) != NULL
            || (smap_get(&p->dict, base, bn) != NULL
                && !set_has(&p->onlys_words, base, bn)
                && !set_has(&p->only_words, base, bn))
            || smap_get(&p->verb_dict, base, bn) != NULL;
        if (bn > 0 && base[bn - 1] == 'i' && bn >= 2) {
            struct chars sx = {0};
            chars_put(&sx, base, bn - 1);
            chars_put_byte(&sx, 'y');
            get_stem_phonemes(p, sx.data, sx.count, &sph);
            chars_free(&sx);
        }
        if ((cvc_pattern || nc_pattern)
            && !base_has_stress_override2) {
            struct chars magic_e = {0};
            chars_put(&magic_e, base, bn);
            chars_put_byte(&magic_e, 'e');
            if (has_prefix_at_start(p, magic_e.data,
                                    magic_e.count)) {
                processing = false;
            } else {
                bool use_magic_e_ed = true;
                if (cvc_pattern && !nc_pattern) {
                    struct chars base_ph_ed = {0};
                    get_stem_phonemes(p, base, bn, &base_ph_ed);
                    use_magic_e_ed = should_use_magic_e_for_cvc_stem(
                        base_ph_ed.data, base_ph_ed.count,
                        "I@3", 3);
                    chars_free(&base_ph_ed);
                }
                if (use_magic_e_ed) {
                    if (sph.count == 0) {
                        get_stem_phonemes(p, magic_e.data,
                                          magic_e.count, &sph);
                    }
                    if (sph.count == 0) {
                        get_stem_phonemes(p, base, bn, &sph);
                    }
                }
            }
            chars_free(&magic_e);
        } else {
            try_ed_non_magic_e_stem_fallbacks(p, base, bn,
                base_has_stress_override2, &sph);
        }
        if (processing && sph.count == 0 && bn >= 2
            && base[bn - 1] == base[bn - 2]) {
            get_stem_phonemes(p, base, bn - 1, &sph);
            if (sph.count == 0) {
                get_stem_phonemes(p, base, bn, &sph);
            }
        }
        if (processing && sph.count > 0) {
            struct chars final_ph = {0};
            compute_ed_suffix_voicing(p, sph.data, sph.count,
                                      norm, nn, &final_ph);
            process_phoneme_string(final_ph.data, final_ph.count,
                                   false, out);
            chars_free(&final_ph);
        }
        chars_free(&sph);
    }
}

// Stem phonemizer shared by checkSuffixDictS (called with stem_s
// and stem_es). Inlines $onlys_bare / $only / dict / stress_pos
// priority, returns processed phonemes in *out (or empty when the
// stem is too short, has no vowel letter, or processing yields no
// vowel phoneme).
static void do_stem_ph_s(struct phonemizer * p,
                         const char * stem, size_t sn,
                         struct chars * out) {
    out->count = 0;
    if (sn >= 2 && has_any_vowel_letter(stem, sn)) {
        struct chars ph2 = {0};
        struct chars * obit = smap_get(&p->onlys_bare_dict, stem, sn);
        if (obit != NULL) {
            process_phoneme_string(obit->data, obit->count, false,
                                   &ph2);
        } else if (set_has(&p->only_words, stem, sn)) {
            struct chars * vt = smap_get(&p->verb_dict, stem, sn);
            if (vt != NULL) {
                process_phoneme_string(vt->data, vt->count, false,
                                       &ph2);
            }
        } else {
            struct chars * jt = smap_get(&p->dict, stem, sn);
            if (jt != NULL) {
                process_phoneme_string(jt->data, jt->count, false,
                                       &ph2);
            } else {
                int * sp = imap_get(&p->stress_pos, stem, sn);
                if (sp != NULL) {
                    struct chars raw = {0};
                    apply_rules(p, stem, sn, true, 0, false, false,
                                NULL, 0, NULL, 0, &raw);
                    if (raw.count > 0) {
                        struct chars stressed = {0};
                        apply_stress_position(raw.data, raw.count,
                                              *sp, &stressed);
                        process_phoneme_string(stressed.data,
                            stressed.count, false, &ph2);
                        chars_free(&stressed);
                    }
                    chars_free(&raw);
                }
            }
        }
        if (ph2.count > 0
            && has_any_vowel_code(ph2.data, ph2.count)) {
            chars_put(out, ph2.data, ph2.count);
        }
        chars_free(&ph2);
    }
}

// -s / -es dictionary-stem suffix. Two passes: strip 's', else
// strip "es" (only when stem phonemes end sibilant). Voicing per
// English plural/3rd-person allomorphs.
static void check_suffix_dict_s(struct phonemizer * p,
                                const char * norm, size_t nn,
                                struct chars * out) {
    out->count = 0;
    if (nn >= 3 && norm[nn - 1] == 's'
        && !(nn >= 2 && norm[nn - 2] == 's')) {
        static const char UNVOICED_S[] = "ptkfTSCxXhs";
        const char * stem_s = norm;
        size_t ssn = nn - 1;
        bool skip_s_strip = (ssn >= 3 && stem_s[ssn - 1] == 'u');
        if (!skip_s_strip) {
            struct chars sph_s = {0};
            do_stem_ph_s(p, stem_s, ssn, &sph_s);
            if (sph_s.count > 0) {
                static const char SIBILANTS_PH[] = "SZsz";
                char last = sph_s.data[sph_s.count - 1];
                bool last_sibilant =
                    strchr(SIBILANTS_PH, last) != NULL;
                const char * s_ph;
                size_t s_n;
                if (last_sibilant) { s_ph = "I#z"; s_n = 3; }
                else if (strchr(UNVOICED_S, last) != NULL) {
                    s_ph = "s"; s_n = 1;
                } else { s_ph = "z"; s_n = 1; }
                chars_put(&sph_s, s_ph, s_n);
                process_phoneme_string(sph_s.data, sph_s.count,
                                       false, out);
            }
            chars_free(&sph_s);
        }
        if (out->count == 0 && nn >= 4 && norm[nn - 2] == 'e') {
            const char * stem_es = norm;
            size_t esn = nn - 2;
            struct chars sph_es = {0};
            do_stem_ph_s(p, stem_es, esn, &sph_es);
            if (sph_es.count > 0) {
                static const char SIBILANTS_ES[] = "SZszC";
                char last = sph_es.data[sph_es.count - 1];
                bool stem_sibilant =
                    strchr(SIBILANTS_ES, last) != NULL;
                if (stem_sibilant) {
                    chars_put(&sph_es, "I#z", 3);
                    process_phoneme_string(sph_es.data,
                        sph_es.count, false, out);
                }
            }
            chars_free(&sph_es);
        }
    }
}

// -ies plural/3rd-person handler. Strip "ies", restore "y",
// phonemize the stem, append voiced 'z'. The direct-rules override
// fires when a stem-vs-fullword vowel disagrees (e.g. "species").
static void check_suffix_ies(struct phonemizer * p,
                             const char * norm, size_t nn,
                             struct chars * out) {
    out->count = 0;
    if (nn >= 4 && memcmp(norm + nn - 3, "ies", 3) == 0) {
        struct chars base = {0};
        chars_put(&base, norm, nn - 3);
        chars_put_byte(&base, 'y');
        struct chars sph = {0};
        get_stem_phonemes(p, base.data, base.count, &sph);
        if (sph.count > 0) {
            static const char VOWELS_IES[] = "aAeEIiOUVu03@o";
            int sv_pos = -1;
            for (size_t k = 0; k < sph.count && sv_pos < 0; k++) {
                if (strchr(VOWELS_IES, sph.data[k]) != NULL) {
                    sv_pos = (int)k;
                }
            }
            bool stem_has_diphthong = (sv_pos >= 0
                && sv_pos + 1 < (int)sph.count
                && (sph.data[sv_pos + 1] == 'I'
                    || sph.data[sv_pos + 1] == 'U'));
            bool emitted_direct = false;
            if (!stem_has_diphthong) {
                char sv = (sv_pos >= 0) ? sph.data[sv_pos] : 0;
                struct chars full_raw = {0};
                apply_rules(p, norm, nn, false, 0, false, false,
                            NULL, 0, NULL, 0, &full_raw);
                int dv_pos = -1;
                for (size_t k = 0;
                     k < full_raw.count && dv_pos < 0; k++) {
                    if (strchr(VOWELS_IES,
                               full_raw.data[k]) != NULL) {
                        dv_pos = (int)k;
                    }
                }
                char dv = (dv_pos >= 0) ? full_raw.data[dv_pos] : 0;
                bool direct_fv_unstressed = (dv_pos > 0
                    && full_raw.data[dv_pos - 1] == '%');
                bool magic_e_strut = (sv == 'V'
                    && (dv == 'u' || dv == 'U'));
                if (dv != 0 && sv != dv && !direct_fv_unstressed
                    && !magic_e_strut) {
                    process_phoneme_string(full_raw.data,
                        full_raw.count, false, out);
                    emitted_direct = true;
                }
                chars_free(&full_raw);
            }
            if (!emitted_direct) {
                chars_put(&sph, "z", 1);
                process_phoneme_string(sph.data, sph.count, false,
                                       out);
            }
        }
        chars_free(&sph);
        chars_free(&base);
    }
}

// -[Ce]s magic-e suffix. Strip 's', then check that the base ends
// in consonant+e where the consonant is not a sibilant. Voicing
// matches plural/3rd-person allomorphs.
static void check_suffix_magic_es(struct phonemizer * p,
                                  const char * norm, size_t nn,
                                  struct chars * out) {
    out->count = 0;
    if (nn >= 4 && norm[nn - 1] == 's'
        && !(nn >= 2 && norm[nn - 2] == 's')) {
        size_t bn = nn - 1;
        const char * base = norm;
        if (bn >= 2 && base[bn - 1] == 'e') {
            char c_before_e = base[bn - 2];
            bool is_consonant = !is_vowel_letter(c_before_e);
            bool is_sibilant = (c_before_e == 's'
                || c_before_e == 'z' || c_before_e == 'x'
                || c_before_e == 'c');
            bool is_digraph_sibilant = (c_before_e == 'h'
                && bn >= 3 && (base[bn - 3] == 'c'
                               || base[bn - 3] == 's'));
            if (is_consonant && !is_sibilant
                && !is_digraph_sibilant) {
                struct chars sph = {0};
                if (has_any_vowel_letter(base, bn)) {
                    struct chars ph2 = {0};
                    struct chars * jt = smap_get(&p->dict,
                                                 base, bn);
                    if (jt != NULL) {
                        chars_put(&ph2, jt->data, jt->count);
                        int * sp = imap_get(&p->stress_pos,
                                            base, bn);
                        if (sp != NULL) {
                            struct chars stressed = {0};
                            apply_stress_position(ph2.data,
                                ph2.count, *sp, &stressed);
                            ph2.count = 0;
                            chars_put(&ph2, stressed.data,
                                      stressed.count);
                            chars_free(&stressed);
                        }
                    } else {
                        static const char GROUP_B_CHARS[] =
                            "bcdfgjklmnpqstvxz";
                        static const char DELFWD_VOWELS[] = "aioy";
                        bool use_spo = false;
                        if (bn >= 3 && base[bn - 1] == 'e') {
                            char c_cons = (char)tolower(
                                (unsigned char)base[bn - 2]);
                            char c_prev = (char)tolower(
                                (unsigned char)base[bn - 3]);
                            use_spo =
                                strchr(GROUP_B_CHARS,
                                       c_cons) != NULL
                                && strchr(DELFWD_VOWELS,
                                          c_prev) != NULL;
                        }
                        struct chars raw = {0};
                        apply_rules(p, base, bn, true, -1,
                                    use_spo, false,
                                    NULL, 0, NULL, 0, &raw);
                        int * sp = imap_get(&p->stress_pos,
                                            base, bn);
                        if (sp != NULL) {
                            struct chars stressed = {0};
                            apply_stress_position(raw.data,
                                raw.count, *sp, &stressed);
                            raw.count = 0;
                            chars_put(&raw, stressed.data,
                                      stressed.count);
                            chars_free(&stressed);
                        }
                        process_phoneme_string(raw.data,
                            raw.count, false, &ph2);
                        chars_free(&raw);
                    }
                    if (ph2.count > 0
                        && has_any_vowel_code(ph2.data,
                                              ph2.count)) {
                        chars_put(&sph, ph2.data, ph2.count);
                    }
                    chars_free(&ph2);
                }
                if (sph.count > 0) {
                    static const char UNVOICED[] = "ptkfTSCxXhs";
                    static const char SIBILANTS_PH[] = "SZsz";
                    char last = sph.data[sph.count - 1];
                    bool last_sib =
                        strchr(SIBILANTS_PH, last) != NULL;
                    const char * s_ph;
                    size_t s_n;
                    if (last_sib) { s_ph = "I#z"; s_n = 3; }
                    else if (strchr(UNVOICED, last) != NULL) {
                        s_ph = "s"; s_n = 1;
                    } else { s_ph = "z"; s_n = 1; }
                    chars_put(&sph, s_ph, s_n);
                    process_phoneme_string(sph.data, sph.count,
                                           false, out);
                }
                chars_free(&sph);
            }
        }
    }
}

// -ches / -shes digraph-sibilant suffix.
static void check_suffix_ch_sh_es(struct phonemizer * p,
                                  const char * norm, size_t nn,
                                  struct chars * out) {
    out->count = 0;
    if (nn >= 5 && norm[nn - 1] == 's'
        && norm[nn - 2] == 'e' && norm[nn - 3] == 'h'
        && (norm[nn - 4] == 'c' || norm[nn - 4] == 's')) {
        const char * stem = norm;
        size_t sn2 = nn - 2;
        if (has_any_vowel_letter(stem, sn2) && sn2 >= 2) {
            struct chars sph = {0};
            struct chars * jt = smap_get(&p->dict, stem, sn2);
            if (jt != NULL) {
                chars_put(&sph, jt->data, jt->count);
            } else {
                struct chars raw = {0};
                apply_rules(p, stem, sn2, true, -1, false, false,
                            NULL, 0, NULL, 0, &raw);
                process_phoneme_string(raw.data, raw.count, false,
                                       &sph);
                chars_free(&raw);
            }
            if (has_any_vowel_code(sph.data, sph.count)) {
                chars_put(&sph, "I#z", 3);
                process_phoneme_string(sph.data, sph.count, false,
                                       out);
            }
            chars_free(&sph);
        }
    }
}

// -xes suffix.
static void check_suffix_xes(struct phonemizer * p,
                             const char * norm, size_t nn,
                             struct chars * out) {
    out->count = 0;
    if (nn >= 4 && memcmp(norm + nn - 3, "xes", 3) == 0) {
        const char * stem = norm;
        size_t sn2 = nn - 2;
        if (has_any_vowel_letter(stem, sn2) && sn2 >= 2) {
            struct chars sph = {0};
            struct chars * jt = smap_get(&p->dict, stem, sn2);
            if (jt != NULL) {
                chars_put(&sph, jt->data, jt->count);
            } else {
                struct chars raw = {0};
                apply_rules(p, stem, sn2, true, -1, false, false,
                            NULL, 0, NULL, 0, &raw);
                process_phoneme_string(raw.data, raw.count, false,
                                       &sph);
                chars_free(&raw);
            }
            if (has_any_vowel_code(sph.data, sph.count)) {
                chars_put(&sph, "I#z", 3);
                process_phoneme_string(sph.data, sph.count, false,
                                       out);
            }
            chars_free(&sph);
        }
    }
}

// -arily suffix. Two cases on the stem+"ari" phonemes: stressed
// "'A@ri" (5-char strip) vs schwa-r "3ri"/"@ri" (3-char strip,
// demote primary stress to secondary). Combined stem + "'e@rI#l%i".
static void check_suffix_arily(struct phonemizer * p,
                               const char * norm, size_t nn,
                               struct chars * out) {
    out->count = 0;
    if (nn >= 8 && memcmp(norm + nn - 5, "arily", 5) == 0) {
        size_t stem_arily_n = nn - 5;
        const char * stem_arily = norm;
        if (has_any_vowel_letter(stem_arily, stem_arily_n)
            && stem_arily_n >= 2) {
            struct chars stem_with_ari = {0};
            chars_put(&stem_with_ari, stem_arily, stem_arily_n);
            chars_put(&stem_with_ari, "ari", 3);
            struct chars sph_arily = {0};
            struct chars * jt_ar = smap_get(&p->dict,
                stem_with_ari.data, stem_with_ari.count);
            if (jt_ar != NULL) {
                chars_put(&sph_arily, jt_ar->data, jt_ar->count);
            } else {
                apply_rules(p, stem_with_ari.data,
                            stem_with_ari.count, true, -1,
                            false, false, NULL, 0, NULL, 0,
                            &sph_arily);
            }
            if (sph_arily.count > 0) {
                struct chars sph_stem = {0};
                bool ends_in_stressed = sph_arily.count >= 5
                    && memcmp(sph_arily.data + sph_arily.count - 5,
                              "'A@ri", 5) == 0;
                bool ends_in_schwa_r = !ends_in_stressed
                    && sph_arily.count >= 3
                    && (memcmp(sph_arily.data + sph_arily.count - 3,
                               "3ri", 3) == 0
                        || memcmp(sph_arily.data
                                  + sph_arily.count - 3,
                                  "@ri", 3) == 0);
                if (ends_in_stressed) {
                    chars_put(&sph_stem, sph_arily.data,
                              sph_arily.count - 5);
                } else if (ends_in_schwa_r) {
                    chars_put(&sph_stem, sph_arily.data,
                              sph_arily.count - 3);
                    replace_first_char(&sph_stem, '\'', ',');
                }
                if (sph_stem.count > 0) {
                    chars_put(&sph_stem, "'e@rI#l%i", 9);
                    process_phoneme_string(sph_stem.data,
                        sph_stem.count, false, out);
                }
                chars_free(&sph_stem);
            }
            chars_free(&sph_arily);
            chars_free(&stem_with_ari);
        }
    }
}

// Dispatcher: first non-empty arm wins. Order mirrors the .cpp.
static void check_morphological_suffixes(struct phonemizer * p,
                                         const char * norm,
                                         size_t nn,
                                         struct chars * out) {
    out->count = 0;
    if (nn >= 5) {
        check_suffix_ing(p, norm, nn, out);
        if (out->count == 0) {
            check_suffix_ed(p, norm, nn, out);
        }
    }
    if (out->count == 0) { check_suffix_ies(p, norm, nn, out); }
    if (out->count == 0) { check_suffix_dict_s(p, norm, nn, out); }
    if (out->count == 0) {
        check_suffix_magic_es(p, norm, nn, out);
    }
    if (out->count == 0) {
        check_suffix_ch_sh_es(p, norm, nn, out);
    }
    if (out->count == 0) { check_suffix_xes(p, norm, nn, out); }
    if (out->count == 0) { check_suffix_arily(p, norm, nn, out); }
}

// Word -> phoneme codes. Dispatch chain: each arm tries to claim
// the word; first non-empty result wins. Order mirrors the .cpp
// `wordToPhonemes` exactly.
static void word_to_phonemes(struct phonemizer * p,
                             const char * word, size_t wn,
                             struct chars * out) {
    out->count = 0;
    struct chars norm = {0};
    to_lower(word, wn, &norm);
    check_capital_dict(p, word, wn, norm.data, norm.count, out);
    if (out->count == 0) {
        check_main_dict(p, norm.data, norm.count, out);
    }
    if (out->count == 0) {
        check_hyphenated(p, norm.data, norm.count, out);
    }
    if (out->count == 0) {
        check_possessive(p, norm.data, norm.count, out);
    }
    if (out->count == 0) {
        check_single_letter(p, norm.data, norm.count, out);
    }
    if (out->count == 0) {
        check_morphological_suffixes(p, norm.data, norm.count, out);
    }
    if (out->count == 0) {
        check_compound_prefixes(p, norm.data, norm.count, out);
    }
    if (out->count == 0) {
        apply_rules_fallback(p, norm.data, norm.count, out);
    }
    chars_free(&norm);
}

// ---------------------------------------------------------------------------
// process_prefix_rule: PREFIX-marker terminal path inside apply_rules.
// Re-translates the suffix as a new word and combines stem + suffix
// phonemes, with stress demotion when both carry primary stress.
// ---------------------------------------------------------------------------

static void process_prefix_rule(struct phonemizer * p,
                                const char * word, size_t wn,
                                int pos,
                                const struct rule_match * match,
                                struct chars * phonemes,
                                struct chars * out) {
    out->count = 0;
    size_t suffix_off = (size_t)(pos + match->advance);
    const char * suffix = word + suffix_off;
    size_t sn = wn - suffix_off;
    bool prefix_has_stress = memchr(phonemes->data, '\'',
                                    phonemes->count) != NULL
                          || memchr(phonemes->data, ',',
                                    phonemes->count) != NULL
                          || memchr(phonemes->data, '%',
                                    phonemes->count) != NULL;
    bool has_full_vowel = prefix_has_full_vowel(
        phonemes->data, phonemes->count);
    bool skip = !prefix_has_stress && has_full_vowel;
    if (!skip) {
        struct chars sfx_ph = {0};
        // Lowercase suffix once for the set/map lookups.
        struct chars sfx_lo = {0};
        to_lower(suffix, sn, &sfx_lo);
        bool onlys = set_has(&p->onlys_words, suffix, sn);
        bool noun_form = set_has(&p->noun_form_stress,
                                 sfx_lo.data, sfx_lo.count);
        bool verb_flag = set_has(&p->verb_flag_words,
                                 sfx_lo.data, sfx_lo.count);
        if (onlys || noun_form || verb_flag) {
            // Force rule-based phonemes (skip dict).
            struct chars rules_out = {0};
            apply_rules(p, suffix, sn, true, -1, false, false,
                        NULL, 0, NULL, 0, &rules_out);
            process_phoneme_string(rules_out.data,
                                   rules_out.count, false, &sfx_ph);
            chars_free(&rules_out);
        } else {
            word_to_phonemes(p, suffix, sn, &sfx_ph);
        }
        chars_free(&sfx_lo);
        // Stress demotion when both prefix and suffix carry '\''.
        bool pfx_has_prime = memchr(phonemes->data, '\'',
                                    phonemes->count) != NULL;
        bool sfx_has_prime = memchr(sfx_ph.data, '\'',
                                    sfx_ph.count) != NULL;
        if (pfx_has_prime && sfx_has_prime) {
            int sfx_syllables = count_suffix_syllables(suffix, sn);
            int pfx_vowels = count_prefix_vowels(
                phonemes->data, phonemes->count);
            bool pfx_ends_schwa = false;
            if (sfx_syllables == 2 && pfx_vowels >= 2) {
                pfx_ends_schwa = prefix_ends_in_schwa(
                    phonemes->data, phonemes->count);
            }
            if (sfx_syllables >= 2 && pfx_vowels >= 2
                && !pfx_ends_schwa) {
                // Demote prefix primary -> secondary.
                char * pp = memchr(phonemes->data, '\'',
                                   phonemes->count);
                if (pp != NULL) { *pp = ','; }
            } else if (pfx_vowels == 1) {
                // Remove suffix primary entirely.
                char * sp = memchr(sfx_ph.data, '\'', sfx_ph.count);
                if (sp != NULL) {
                    size_t at = (size_t)(sp - sfx_ph.data);
                    memmove(sp, sp + 1, sfx_ph.count - at - 1);
                    sfx_ph.count--;
                    sfx_ph.data[sfx_ph.count] = '\0';
                }
            } else {
                // Demote suffix primary -> secondary.
                char * sp = memchr(sfx_ph.data, '\'', sfx_ph.count);
                if (sp != NULL) { *sp = ','; }
            }
            // Post-demotion: drop ',' immediately before a
            // phUNSTRESSED vowel (@, I#, I2, a#).
            char * cp = memchr(sfx_ph.data, ',', sfx_ph.count);
            if (cp != NULL) {
                size_t at = (size_t)(cp - sfx_ph.data);
                if (at + 1 < sfx_ph.count) {
                    char nc = sfx_ph.data[at + 1];
                    bool phU = nc == '@'
                        || (nc == 'I' && at + 2 < sfx_ph.count
                            && (sfx_ph.data[at + 2] == '#'
                             || sfx_ph.data[at + 2] == '2'))
                        || (nc == 'a' && at + 2 < sfx_ph.count
                            && sfx_ph.data[at + 2] == '#');
                    if (phU) {
                        memmove(cp, cp + 1,
                                sfx_ph.count - at - 1);
                        sfx_ph.count--;
                        sfx_ph.data[sfx_ph.count] = '\0';
                    }
                }
            }
        }
        // Strip '\x01' rule-boundary markers from prefix phonemes
        // before combining.
        size_t w = 0;
        for (size_t r = 0; r < phonemes->count; r++) {
            if (phonemes->data[r] != '\x01') {
                phonemes->data[w++] = phonemes->data[r];
            }
        }
        phonemes->count = w;
        if (phonemes->data != NULL) {
            phonemes->data[phonemes->count] = '\0';
        }
        chars_put(out, phonemes->data, phonemes->count);
        chars_put(out, sfx_ph.data, sfx_ph.count);
        chars_free(&sfx_ph);
    }
}

// ---------------------------------------------------------------------------
// apply_rules: rule-scan main loop. Mirrors IPAPhonemizer::applyRules.
// Caller-owned out; resets count = 0 at entry. out_replaced_e and
// out_pos_visited may be NULL; if non-NULL the function writes up to
// out_re_cap / out_pv_cap entries.
// ---------------------------------------------------------------------------

static void apply_rules(struct phonemizer * p,
                        const char * word_orig, size_t wn,
                        bool allow_suffix_strip,
                        int word_alt_flags_param,
                        bool suffix_phoneme_only,
                        bool suffix_removed,
                        bool * out_replaced_e, size_t out_re_cap,
                        bool * out_pos_visited, size_t out_pv_cap,
                        struct chars * out) {
    out->count = 0;
    if (out->data != NULL) { out->data[0] = '\0'; }
    // Apply replacements into a working `word` buffer.
    struct chars word = {0};
    chars_put(&word, word_orig, wn);
    apply_replacements(&word, &p->rules.replacements);
    int len = (int)word.count;
    int word_alt_flags = determine_alt_flags(
        p, word.data, word.count, word_alt_flags_param);
    bool * replaced_e = calloc((size_t)(len > 0 ? len : 1),
                               sizeof(bool));
    bool * pos_visited = calloc((size_t)(len > 0 ? len : 1),
                                sizeof(bool));
    struct chars phonemes = {0};
    struct chars final_result = {0};
    bool finished = false;
    int pos = 0;
    while (pos < len && !finished) {
        pos_visited[pos] = true;
        char pos_char = replaced_e[pos]
            ? 'E'
            : (char)tolower((unsigned char)word.data[pos]);
        struct rule_match match = find_best_rule(
            &p->rules, word.data, word.count, pos, len,
            pos_char, word_alt_flags,
            replaced_e, (size_t)len,
            allow_suffix_strip, suffix_phoneme_only,
            suffix_removed,
            phonemes.data, phonemes.count);
        if (match.score < 0) {
            pos++;
        } else {
            apply_stress_prev(&match.phonemes, &phonemes);
            chars_put(&phonemes, match.phonemes.data,
                      match.phonemes.count);
            chars_put_byte(&phonemes, '\x01');
            bool terminal_suffix =
                (allow_suffix_strip || suffix_phoneme_only)
                && match.is_suffix
                && pos + match.advance == len;
            if (terminal_suffix && !suffix_phoneme_only) {
                process_suffix_rule(p, word.data, word.count,
                                    word_alt_flags, &match,
                                    &final_result);
                finished = true;
            } else if (terminal_suffix && suffix_phoneme_only) {
                pos += match.advance;
            } else if (match.is_prefix
                       && pos + match.advance < len) {
                struct chars pfx_res = {0};
                process_prefix_rule(p, word.data, word.count,
                                    pos, &match, &phonemes,
                                    &pfx_res);
                if (pfx_res.count > 0) {
                    final_result.count = 0;
                    chars_put(&final_result, pfx_res.data,
                              pfx_res.count);
                    finished = true;
                } else {
                    pos += match.advance;
                }
                chars_free(&pfx_res);
            } else {
                if (match.del_count > 0 && match.del_start >= 0) {
                    for (int d = 0;
                         d < match.del_count
                         && match.del_start + d < len; d++) {
                        replaced_e[match.del_start + d] = true;
                    }
                }
                pos += match.advance;
            }
        }
        rule_match_free(&match);
    }
    // Copy out the replaced_e / pos_visited if caller asked.
    if (out_replaced_e != NULL && out_re_cap > 0) {
        size_t n = (size_t)len < out_re_cap
            ? (size_t)len : out_re_cap;
        memcpy(out_replaced_e, replaced_e, n * sizeof(bool));
    }
    if (out_pos_visited != NULL && out_pv_cap > 0) {
        size_t n = (size_t)len < out_pv_cap
            ? (size_t)len : out_pv_cap;
        memcpy(out_pos_visited, pos_visited, n * sizeof(bool));
    }
    if (finished) {
        chars_put(out, final_result.data, final_result.count);
    } else {
        chars_put(out, phonemes.data, phonemes.count);
    }
    chars_free(&phonemes);
    chars_free(&final_result);
    chars_free(&word);
    free(replaced_e);
    free(pos_visited);
}

// ---------------------------------------------------------------------------
// Text tokenization
// ---------------------------------------------------------------------------

// One token from tokenize(): a word or single-char punctuation. `text`
// is heap-owned by the surrounding `struct tokens`. needs_space is
// true iff the token had a preceding word in the same utterance (used
// by phonemizeText to insert a separator before this token's output).
struct token {
    struct chars text;
    bool         is_word;
    bool         needs_space;
};

define_array(struct token, tokens);

// Walks each token, frees its per-element chars heap, then frees the
// array storage itself. Inverse of tokens_put + the per-element
// chars_put done by flush_word / emit_punct.
static void tokens_clear(struct tokens * arr) {
    for (size_t i = 0; i < arr->count; i++) {
        chars_free(&arr->data[i].text);
    }
    tokens_free(arr);
}

// Flush accumulated `cur` as a word token (is_word=true,
// needs_space = out already has tokens). Resets `cur` for the next
// word. No-op when cur is empty. Replaces the C++ `flush_word`
// [&]-lambda by passing the two captured locals explicitly.
static void flush_word(struct tokens * out, struct chars * cur) {
    if (cur->count > 0) {
        struct token tk = {
            .text = {0}, .is_word = true,
            .needs_space = out->count > 0
        };
        chars_put(&tk.text, cur->data, cur->count);
        tokens_put(out, tk);
        cur->count = 0;
        if (cur->data != NULL) { cur->data[0] = '\0'; }
    }
}

// Flush any accumulated word, then append `c` as a 1-char non-word
// token (is_word=false, needs_space=false). Replaces the C++
// `emit_punct` [&]-lambda the same way.
static void emit_punct(struct tokens * out, struct chars * cur, char c) {
    flush_word(out, cur);
    struct token tk = {
        .text = {0}, .is_word = false, .needs_space = false
    };
    chars_put(&tk.text, &c, 1);
    tokens_put(out, tk);
}

// Split `text[0..n)` into word and punctuation tokens. UTF-8 multi-
// byte runs are accumulated into a single word. Abbreviations like
// "U.S." are kept as one word (period mid-word). Hyphens between
// letters stay in the word; sentence-terminating periods, commas,
// and other punctuation become standalone non-word tokens.
//
// The caller owns `out`; `tokens_clear(out)` runs at entry so any
// previous contents are freed before the new tokens are emitted.
static void tokenize(const char * text, size_t n, struct tokens * out) {
    tokens_clear(out);
    struct chars cur = {0};
    size_t i = 0;
    while (i < n) {
        unsigned char c = (unsigned char)text[i];
        if (c >= 0x80) {
            // UTF-8 multi-byte: take lead byte then all continuation
            // bytes (0x80..0xBF). Mirrors the .cpp loop.
            chars_put_byte(&cur, text[i]);
            i++;
            while (i < n
                   && ((unsigned char)text[i] & 0xC0) == 0x80) {
                chars_put_byte(&cur, text[i]);
                i++;
            }
        } else if (isalpha(c) || c == '\'') {
            chars_put_byte(&cur, (char)c);
            i++;
        } else if (c == '.' && cur.count == 1
                   && isupper((unsigned char)cur.data[0])
                   && i + 1 < n
                   && isupper((unsigned char)text[i + 1])) {
            // Abbreviation start: single uppercase letter + '.' +
            // next char is uppercase. E.g. "U.S." -- accumulate
            // "U." into cur and keep collecting.
            chars_put_byte(&cur, '.');
            i++;
        } else if (c == '.' && cur.count > 0) {
            // Trailing-period dispatch. If cur already contains a
            // '.', we're mid-abbreviation ("U.S" -> "U.S.") --
            // append the period and flush. Otherwise the period is
            // plain punctuation. memchr is the find('.') equivalent.
            if (memchr(cur.data, '.', cur.count) != NULL) {
                chars_put_byte(&cur, '.');
                i++;
                flush_word(out, &cur);
            } else {
                emit_punct(out, &cur, (char)c);
                i++;
            }
        } else if (c == '-' && cur.count > 0
                   && i + 1 < n
                   && isalpha((unsigned char)text[i + 1])) {
            chars_put_byte(&cur, (char)c);
            i++;
        } else if (isdigit(c)) {
            chars_put_byte(&cur, (char)c);
            i++;
        } else if (isspace(c)) {
            flush_word(out, &cur);
            i++;
        } else {
            emit_punct(out, &cur, (char)c);
            i++;
        }
    }
    flush_word(out, &cur);
    chars_free(&cur);
}

// ---------------------------------------------------------------------------
// Phoneme code -> IPA UTF-8 conversion (D-26a). Ported from
// ipa_table.h + phonemizer.cpp's singleCodeToIPA / emitPhonemeCode /
// phonemesToIPA. The ASCII_TO_IPA[] table and the per-dialect
// IPA_EN_US / IPA_EN_GB tables are pure data carried verbatim.
// ---------------------------------------------------------------------------

// ipa1[] table mirror. Maps each ASCII byte 0x20..0x7F to a Unicode
// codepoint. Most entries are identity; the differences encode the
// X-SAMPA-like phoneme alphabet (a -> a, A -> ɑ, 3 -> ɜ, ...).
static const uint32_t ASCII_TO_IPA[96] = {
    0x0020, 0x0021, 0x0022, 0x02b0, 0x0024, 0x0025, 0x00e6, 0x02c8,
    0x0028, 0x0029, 0x027e, 0x002b, 0x02cc, 0x002d, 0x002e, 0x002f,
    0x0252, 0x0031, 0x0032, 0x025c, 0x0034, 0x0035, 0x0036, 0x0037,
    0x0275, 0x0039, 0x02d0, 0x02b2, 0x003c, 0x003d, 0x003e, 0x0294,
    0x0259, 0x0251, 0x03b2, 0x00e7, 0x00f0, 0x025b, 0x0046, 0x0262,
    0x0127, 0x026a, 0x025f, 0x004b, 0x026b, 0x0271, 0x014b, 0x0254,
    0x03a6, 0x0263, 0x0280, 0x0283, 0x03b8, 0x028a, 0x028c, 0x0153,
    0x03c7, 0x00f8, 0x0292, 0x032a, 0x005c, 0x005d, 0x005e, 0x005f,
    0x0060, 0x0061, 0x0062, 0x0063, 0x0064, 0x0065, 0x0066, 0x0261,
    0x0068, 0x0069, 0x006a, 0x006b, 0x006c, 0x006d, 0x006e, 0x006f,
    0x0070, 0x0071, 0x0072, 0x0073, 0x0074, 0x0075, 0x0076, 0x0077,
    0x0078, 0x0079, 0x007a, 0x007b, 0x007c, 0x007d, 0x0303, 0x007f,
};

// Append a Unicode codepoint as UTF-8 bytes to a struct chars.
static void append_utf8(struct chars * out, uint32_t cp) {
    if (cp < 0x80) {
        chars_put_byte(out, (char)cp);
    } else if (cp < 0x800) {
        chars_put_byte(out, (char)(0xC0 | (cp >> 6)));
        chars_put_byte(out, (char)(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        chars_put_byte(out, (char)(0xE0 | (cp >> 12)));
        chars_put_byte(out, (char)(0x80 | ((cp >> 6) & 0x3F)));
        chars_put_byte(out, (char)(0x80 | (cp & 0x3F)));
    } else {
        chars_put_byte(out, (char)(0xF0 | (cp >> 18)));
        chars_put_byte(out, (char)(0x80 | ((cp >> 12) & 0x3F)));
        chars_put_byte(out, (char)(0x80 | ((cp >> 6) & 0x3F)));
        chars_put_byte(out, (char)(0x80 | (cp & 0x3F)));
    }
}

// Per-byte ipa1 walk. Skips '|', '/'; breaks on '#' (variant marker
// handled via overrides), '_' at start (pause), '/' (variant). is_-
// vowel is unused (kept for sig parity with the .cpp).
static void phoneme_code_to_ipa_table(const char * code, size_t cn,
                                      bool is_vowel,
                                      struct chars * out) {
    (void)is_vowel;
    bool first = true;
    bool stop = false;
    for (size_t i = 0; i < cn && !stop; i++) {
        unsigned char uc = (unsigned char)code[i];
        char c = code[i];
        if (c == '/') {
            stop = true;
        } else if (!first && c >= '0' && c <= '9') {
            /* skip variant digit after first char */
        } else if (c == '#') {
            stop = true;
        } else if (c == '|') {
            /* skip pipe separator */
        } else if (c == '_' && first) {
            stop = true;
        } else {
            if (uc >= 0x20 && uc < 0x80) {
                append_utf8(out, ASCII_TO_IPA[uc - 0x20]);
            } else if (uc >= 0x80) {
                chars_put_byte(out, c);
            }
            first = false;
        }
    }
}

// Phoneme-code -> IPA UTF-8 string with explicit overrides and the
// ipa1-table fallback. Stress markers (', and ,) render as the
// modifier letters U+02C8 / U+02CC. '%', '=', '|', '||', '==' have
// no rendering — they are bookkeeping markers stripped here.
static void single_code_to_ipa(struct phonemizer * p,
                               const char * code, size_t cn,
                               struct chars * out) {
    out->count = 0;
    if (cn == 1 && code[0] == '\'') {
        chars_put(out, "\xcb\x88", 2);          // ˈ
    } else if (cn == 1 && code[0] == ',') {
        chars_put(out, "\xcb\x8c", 2);          // ˌ
    } else if (cn == 1 &&
        (code[0] == '%' || code[0] == '=' || code[0] == '|')) {
        /* unstressed / boundary marker: nothing emitted */
    } else if (cn == 2 &&
        ((code[0] == '=' && code[1] == '=')
         || (code[0] == '|' && code[1] == '|'))) {
        /* word-end / double-unstressed marker: nothing emitted */
    } else if (cn > 0) {
        struct chars * over = smap_get(&p->ipa_overrides, code, cn);
        if (over != NULL) {
            chars_put(out, over->data, over->count);
        } else {
            bool is_vowel = is_vowel_code(code, cn);
            phoneme_code_to_ipa_table(code, cn, is_vowel, out);
        }
    }
}

// Multi-char phoneme codes recognised by emit_phoneme_code. Longest-
// first so the greedy match prefers (e.g.) "aU@r" over "aU@" or "aU".
static const char * const MULTI_CODES[] = {
    "aI@3", "aU@r", "i@3r",
    "aI@", "aI3", "aU@", "i@3", "3:r", "A:r", "I2#",
    "o@r", "O@r", "e@r",
    "eI", "aI", "aU", "OI", "oU", "tS", "dZ", "IR", "VR",
    "e@", "i@", "U@", "A@", "O@", "o@",
    "3:", "A:", "i:", "u:", "O:", "e:", "a:",
    "aa",
    "@L", "@2", "@5",
    "r-", "w#", "t#", "d#", "z#", "t2", "d2", "n-", "m-",
    "l/", "z/",
    "I2", "I#", "E2", "E#", "e#", "a#", "a2", "0#", "02",
    "O2", "A~", "O~", "A#",
    NULL
};

// Greedy multi-char match starting at pstr[*i], skipping "false
// diphthongs" i@/U@ after %/=. Absorbs variant-marker digits whose
// IPA mapping is identity. Emits any pending stress before vowel or
// syllabic-consonant codes (both act as syllable nuclei).
static void emit_phoneme_code(struct phonemizer * p,
                              const char * pstr, size_t plen,
                              size_t * i,
                              struct chars * pending_stress,
                              bool * last_was_unstress,
                              bool * last_code_was_vowel,
                              struct chars * result) {
    const char * code = NULL;
    size_t code_len = 0;
    bool found = false;
    for (int mi = 0; MULTI_CODES[mi] != NULL && !found; mi++) {
        const char * mc = MULTI_CODES[mi];
        size_t mclen = strlen(mc);
        bool skip = *last_was_unstress && mclen == 2
            && (strcmp(mc, "i@") == 0 || strcmp(mc, "U@") == 0);
        if (!skip && *i + mclen <= plen
            && memcmp(pstr + *i, mc, mclen) == 0) {
            code = mc;
            code_len = mclen;
            *i += mclen;
            found = true;
        }
    }
    if (!found) {
        code = pstr + *i;
        code_len = 1;
        (*i)++;
    }
    *last_was_unstress = false;
    while (*i < plen && pstr[*i] >= '0' && pstr[*i] <= '9'
           && ASCII_TO_IPA[(unsigned char)pstr[*i] - 0x20]
              == (unsigned char)pstr[*i]) {
        (*i)++;
    }
    bool is_syllabic = (code_len == 2
        && ((code[0] == 'n' && code[1] == '-')
            || (code[0] == 'm' && code[1] == '-')
            || (code[0] == '@' && code[1] == 'L')
            || (code[0] == 'r' && code[1] == '-')
            || (code[0] == 'l' && code[1] == '/')));
    bool emit_stress = pending_stress->count > 0
        && (is_vowel_code(code, code_len) || is_syllabic);
    if (emit_stress) {
        chars_put(result, pending_stress->data,
                  pending_stress->count);
        pending_stress->count = 0;
    }
    struct chars one = {0};
    single_code_to_ipa(p, code, code_len, &one);
    chars_put(result, one.data, one.count);
    chars_free(&one);
    *last_code_was_vowel = is_vowel_code(code, code_len);
}

// Phoneme-code string -> IPA UTF-8. Top-level driver: strips
// trailing $-flags, walks the trimmed string, emits stress markers
// lazily (pending until the next vowel/syllabic-consonant code).
static void phonemes_to_ipa(struct phonemizer * p,
                            const char * ph, size_t pn,
                            struct chars * out) {
    out->count = 0;
    if (pn > 0) {
        size_t dollar = pn;
        for (size_t i = 0; i < pn && dollar == pn; i++) {
            if (ph[i] == '$') { dollar = i; }
        }
        struct chars pstr = {0};
        trim(ph, dollar, &pstr);
        size_t i = 0;
        size_t len = pstr.count;
        struct chars pending_stress = {0};
        bool last_was_unstress = false;
        bool last_code_was_vowel = false;
        while (i < len) {
            unsigned char c = (unsigned char)pstr.data[i];
            if (isspace(c)) {
                i++;
            } else if (pstr.data[i] == '|' || pstr.data[i] == '-') {
                i++;
            } else if (pstr.data[i] == ';') {
                if (!last_code_was_vowel) {
                    chars_put(out, "\xca\xb2", 2);   // ʲ
                }
                i++;
            } else if (pstr.data[i] == '\''
                       || pstr.data[i] == ',') {
                pending_stress.count = 0;
                single_code_to_ipa(p, pstr.data + i, 1,
                                   &pending_stress);
                last_was_unstress = false;
                i++;
            } else if (pstr.data[i] == '%'
                       || pstr.data[i] == '=') {
                pending_stress.count = 0;
                last_was_unstress = true;
                i++;
            } else {
                emit_phoneme_code(p, pstr.data, len, &i,
                                  &pending_stress,
                                  &last_was_unstress,
                                  &last_code_was_vowel, out);
            }
        }
        chars_free(&pending_stress);
        chars_free(&pstr);
    }
}

// Per-dialect IPA override tables. Loaded into p->ipa_overrides by
// build_ipa_overrides at init time. The common table applies to
// every dialect; the per-dialect table adds dialect-specific
// overrides on top.
struct ipa_override_entry {
    const char * code;
    const char * ipa;
};

static const struct ipa_override_entry IPA_COMMON[] = {
    {"r",    "\xc9\xb9"},
    {"r-",   "\xc9\xb9"},
    {"n-",   "n\xcc\xa9"},
    {"m-",   "m\xcc\xa9"},
    {"3:r",  "\xc9\x9c\xcb\x90\xc9\xb9"},
    {"3:",   "\xc9\x9c\xcb\x90"},
    {"@L",   "\xc9\x99l"},
    {"a#",   "\xc9\x90"},
    {"e#",   "\xc9\x9b"},
    {"I#",   "\xe1\xb5\xbb"},
    {"I2#",  "\xe1\xb5\xbb"},
    {"w#",   "\xca\x8d"},
    {"@2",   "\xc9\x99"},
    {"@5",   "\xc9\x99"},
    {"I2",   "\xc9\xaa"},
    {NULL,   NULL},
};

static const struct ipa_override_entry IPA_EN_US[] = {
    {"3",    "\xc9\x9a"},
    {"a",    "\xc3\xa6"},
    {"aa",   "\xc3\xa6"},
    {"0",    "\xc9\x91\xcb\x90"},
    {"0#",   "\xc9\x91\xcb\x90"},
    {"A#",   "\xc9\x91\xcb\x90"},
    {"A@",   "\xc9\x91\xcb\x90\xc9\xb9"},
    {"A:r",  "\xc9\x91\xcb\x90\xc9\xb9"},
    {"e@",   "\xc9\x9b\xc9\xb9"},
    {"e@r",  "\xc9\x9b\xc9\xb9"},
    {"U@",   "\xca\x8a\xc9\xb9"},
    {"O@",   "\xc9\x94\xcb\x90\xc9\xb9"},
    {"O@r",  "\xc9\x94\xcb\x90\xc9\xb9"},
    {"o@",   "o\xcb\x90\xc9\xb9"},
    {"o@r",  "o\xcb\x90\xc9\xb9"},
    {"i@",   "i\xc9\x99"},
    {"i@3",  "\xc9\xaa\xc9\xb9"},
    {"i@3r", "\xc9\xaa\xc9\xb9"},
    {"aI@",  "a\xc9\xaa\xc9\x99"},
    {"aI3",  "a\xc9\xaa\xc9\x9a"},
    {"aU@",  "a\xc9\xaa\xca\x8a\xc9\xb9"},
    {"IR",   "\xc9\x99\xc9\xb9"},
    {"VR",   "\xca\x8c\xc9\xb9"},
    {"02",   "\xca\x8c"},
    {"i",    "i"},
    {NULL,   NULL},
};

static const struct ipa_override_entry IPA_EN_GB[] = {
    {"3",    "\xc9\x9a"},
    {"a",    "a"},
    {"aa",   "a"},
    {"0",    "\xc9\x92"},
    {"oU",   "\xc9\x99\xca\x8a"},
    {"A@",   "\xc9\x91\xcb\x90"},
    {"IR",   "\xc9\x99\xc9\xb9"},
    {NULL,   NULL},
};

static void load_override_table(struct phonemizer * p,
                                const struct ipa_override_entry * t) {
    for (int i = 0; t[i].code != NULL; i++) {
        smap_set(&p->ipa_overrides,
                 t[i].code, strlen(t[i].code),
                 t[i].ipa, strlen(t[i].ipa));
    }
}

// Populate p->ipa_overrides from the dialect string. Call once at
// the start of phonemizer_create, after dialect is set.
static void build_ipa_overrides(struct phonemizer * p) {
    bool is_en_us = (p->dialect.count == 5
        && (memcmp(p->dialect.data, "en-us", 5) == 0
            || memcmp(p->dialect.data, "en_us", 5) == 0));
    load_override_table(p, IPA_COMMON);
    load_override_table(p, is_en_us ? IPA_EN_US : IPA_EN_GB);
}

// ---------------------------------------------------------------------------
// IPA-level utilities (D-26b). Stress-marker detection, UTF-8
// vowel-start probe, default-stress insertion. Used by number
// expansion + acronym spelling + processWordToken's default-stress
// step.
// ---------------------------------------------------------------------------

// True when the IPA byte string already contains a U+02C8 (ˈ) or
// U+02CC (ˌ) stress marker.
static bool contains_stress_marker(const char * ipa, size_t n) {
    bool found = false;
    for (size_t i = 0; i + 1 < n && !found; i++) {
        if ((unsigned char)ipa[i] == 0xCB
            && ((unsigned char)ipa[i + 1] == 0x88
                || (unsigned char)ipa[i + 1] == 0x8C)) {
            found = true;
        }
    }
    return found;
}

// True iff `s[i..)` starts with an IPA vowel character. Covers the
// ASCII vowels plus the UTF-8 ranges used by the override tables
// (æ ø œ in C3-XX, ɐ ɑ ɒ ɔ ə ɚ ɛ ɜ ɞ ɪ ɵ in C9-XX, ʊ ʌ in CA-XX,
// ᵻ in E1 B5 BB).
static bool is_ipa_vowel_start(const char * s, size_t n, size_t i) {
    bool r = false;
    if (i < n) {
        unsigned char c = (unsigned char)s[i];
        if (c < 0x80) {
            char lc = (char)tolower((unsigned char)c);
            r = (lc == 'a' || lc == 'e' || lc == 'i'
                 || lc == 'o' || lc == 'u');
        } else if (i + 1 < n) {
            unsigned char c2 = (unsigned char)s[i + 1];
            if (c == 0xC3) {
                // æ=0xA6, ø=0xB8.
                // (Historical: this branch also listed c2==0x93
                // and the comment claimed it was œ, but U+0153 œ
                // encodes as 0xC5 0x93, not 0xC3 0x93 — see the
                // 0xC5 branch below.)
                r = (c2 == 0xA6 || c2 == 0xB8);
            } else if (c == 0xC5) {
                // œ = U+0153 = 0xC5 0x93.
                r = (c2 == 0x93);
            } else if (c == 0xC9) {
                r = (c2 == 0x90 || c2 == 0x91 || c2 == 0x92
                     || c2 == 0x94 || c2 == 0x99 || c2 == 0x9A
                     || c2 == 0x9B || c2 == 0x9C || c2 == 0x9E
                     || c2 == 0xAA || c2 == 0xB5);
            } else if (c == 0xCA) {
                r = (c2 == 0x8A || c2 == 0x8C);
            } else if (c == 0xE1 && i + 2 < n) {
                unsigned char c3 = (unsigned char)s[i + 2];
                r = (c2 == 0xB5 && c3 == 0xBB);
            }
        }
    }
    return r;
}

// True iff `s[i..)` starts with U+0259 (ə, schwa).
static bool is_ipa_schwa(const char * s, size_t n, size_t i) {
    return i + 1 < n
        && (unsigned char)s[i] == 0xC9
        && (unsigned char)s[i + 1] == 0x99;
}

// UTF-8 byte width of the codepoint starting at byte `c`.
static size_t utf8_step(unsigned char c) {
    size_t step = 4;
    if (c < 0x80) { step = 1; }
    else if (c < 0xE0) { step = 2; }
    else if (c < 0xF0) { step = 3; }
    return step;
}

// First-IPA-vowel scan. Sets *is_schwa for the found vowel.
// Returns (size_t)-1 when no vowel exists in s[start..n).
static size_t find_first_ipa_vowel(const char * s, size_t n,
                                   size_t start, bool * is_schwa) {
    size_t pos = (size_t)-1;
    size_t i = start;
    while (i < n && pos == (size_t)-1) {
        if (is_ipa_vowel_start(s, n, i)) {
            pos = i;
            *is_schwa = is_ipa_schwa(s, n, i);
        } else {
            i += utf8_step((unsigned char)s[i]);
        }
    }
    return pos;
}

// First non-schwa IPA vowel scan from `start`. Used by
// add_default_stress to skip a leading schwa.
static size_t find_first_non_schwa_vowel(const char * s, size_t n,
                                         size_t start) {
    size_t pos = (size_t)-1;
    size_t j = start;
    while (j < n && pos == (size_t)-1) {
        if (is_ipa_vowel_start(s, n, j)
            && !is_ipa_schwa(s, n, j)) {
            pos = j;
        } else {
            j += utf8_step((unsigned char)s[j]);
        }
    }
    return pos;
}

// Insert a primary-stress marker (U+02C8 ˈ) before the first vowel
// of an unstressed IPA word. Skip an initial schwa if a non-schwa
// vowel follows (so "h@loU" -> "həlˈoʊ", not "ˈhəloʊ"). No-op when
// the input already has a stress marker.
static void add_default_stress(const char * ipa, size_t n,
                               struct chars * out) {
    static const char STRESS[] = "\xcb\x88";
    out->count = 0;
    if (n == 0) {
        /* empty -> empty */
    } else if (contains_stress_marker(ipa, n)) {
        chars_put(out, ipa, n);
    } else {
        bool first_is_schwa = false;
        size_t first_vowel = find_first_ipa_vowel(ipa, n, 0,
                                                  &first_is_schwa);
        if (first_vowel == (size_t)-1) {
            chars_put(out, ipa, n);
        } else {
            size_t target = first_vowel;
            if (first_is_schwa) {
                size_t non_schwa = find_first_non_schwa_vowel(
                    ipa, n, first_vowel + 2);
                if (non_schwa != (size_t)-1) { target = non_schwa; }
            }
            chars_put(out, ipa, target);
            chars_put(out, STRESS, 2);
            chars_put(out, ipa + target, n - target);
        }
    }
}

// ---------------------------------------------------------------------------
// Number-to-words conversion (D-26b). Cardinal, American English.
// Spoken form for any integer 0..9999999.
// ---------------------------------------------------------------------------

static const char * const NUM_ONES[] = {
    "", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "eleven", "twelve", "thirteen",
    "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen"
};

static const char * const NUM_TENS[] = {
    "", "", "twenty", "thirty", "forty", "fifty",
    "sixty", "seventy", "eighty", "ninety"
};

// Recursive cardinal-number speller. Writes lower-case words
// separated by single spaces into *out (caller-owned).
static void int_to_words(int n, struct chars * out) {
    if (n == 0) {
        chars_puts(out, "zero");
    } else {
        if (n >= 1000000) {
            int_to_words(n / 1000000, out);
            chars_puts(out, " million");
            n %= 1000000;
            if (n > 0) { chars_put_byte(out, ' '); }
        }
        if (n >= 1000) {
            int_to_words(n / 1000, out);
            chars_puts(out, " thousand");
            n %= 1000;
            if (n > 0) { chars_put_byte(out, ' '); }
        }
        if (n >= 100) {
            chars_puts(out, NUM_ONES[n / 100]);
            chars_puts(out, " hundred");
            n %= 100;
            if (n > 0) { chars_put_byte(out, ' '); }
        }
        if (n >= 20) {
            chars_puts(out, NUM_TENS[n / 10]);
            n %= 10;
            if (n > 0) {
                chars_put_byte(out, ' ');
                chars_puts(out, NUM_ONES[n]);
            }
        } else if (n > 0) {
            chars_puts(out, NUM_ONES[n]);
        }
    }
}

// Pure-digit token -> spoken IPA (via word_to_phonemes ->
// phonemes_to_ipa -> add_default_stress on each sub-word). Returns
// true and appends to *result iff the token's digits parse to a
// non-negative value in [0, 9999999]; false leaves *result alone.
static bool expand_number_token(struct phonemizer * p,
                                const struct token * t,
                                struct chars * result,
                                bool * first_word) {
    bool all_digits = t->text.count > 0;
    for (size_t i = 0; i < t->text.count; i++) {
        if (!isdigit((unsigned char)t->text.data[i])) {
            all_digits = false;
        }
    }
    bool consumed = false;
    if (all_digits) {
        long long num_val = strtoll(t->text.data, NULL, 10);
        if (num_val >= 0 && num_val <= 9999999LL) {
            struct chars num_words = {0};
            int_to_words((int)num_val, &num_words);
            // Walk space-separated sub-words.
            size_t i = 0;
            bool is_first_sub = true;
            while (i < num_words.count) {
                size_t j = i;
                while (j < num_words.count
                       && num_words.data[j] != ' ') {
                    j++;
                }
                if (j > i) {
                    struct chars wph = {0};
                    word_to_phonemes(p, num_words.data + i, j - i,
                                     &wph);
                    struct chars wipa_raw = {0};
                    phonemes_to_ipa(p, wph.data, wph.count,
                                    &wipa_raw);
                    struct chars wipa = {0};
                    add_default_stress(wipa_raw.data,
                                       wipa_raw.count, &wipa);
                    if (is_first_sub) {
                        if (t->needs_space && !*first_word) {
                            chars_put_byte(result, ' ');
                        }
                        is_first_sub = false;
                    } else {
                        chars_put_byte(result, ' ');
                    }
                    chars_put(result, wipa.data, wipa.count);
                    *first_word = false;
                    chars_free(&wipa);
                    chars_free(&wipa_raw);
                    chars_free(&wph);
                }
                i = j + 1;
            }
            chars_free(&num_words);
            consumed = true;
        }
    }
    return consumed;
}

// ---------------------------------------------------------------------------
// Acronym spelling (D-26b). All-upper or mixed-case-no-vowel tokens
// (DNA, PhD, BSc, ...) -> letter-name spelling with comma demotion
// for non-final letters. CamelCase groups split before each lower-
// upper boundary: "PhD" -> ["Ph", "D"].
// ---------------------------------------------------------------------------

// Spell one CamelCase group as letter names + apply stress demotion
// (non-final letter codes get a leading ',', final letter codes
// get a leading '\'' iff they don't already carry one). Renders to
// IPA via process_phoneme_string + phonemes_to_ipa.
static void spell_group(struct phonemizer * p,
                        const char * grp, size_t gn,
                        struct chars * out) {
    out->count = 0;
    if (gn > 0) {
        struct chars codes = {0};
        // letter -> ph code (one entry per letter).
        for (size_t li = 0; li < gn; li++) {
            char lc_lower = (char)tolower((unsigned char)grp[li]);
            char uk[2] = { '_', lc_lower };
            struct chars * uit = smap_get(&p->dict, uk, 2);
            struct chars letter_ph = {0};
            if (uit != NULL) {
                chars_put(&letter_ph, uit->data, uit->count);
            } else {
                word_to_phonemes(p, &grp[li], 1, &letter_ph);
            }
            // Demote non-final, promote final.
            if (li + 1 < gn) {
                // Strip primary marker, prepend ','. Find first
                // \'' or ','; if absent prepend ','.
                size_t first = letter_ph.count;
                for (size_t k = 0; k < letter_ph.count
                                   && first == letter_ph.count; k++) {
                    if (letter_ph.data[k] == '\''
                        || letter_ph.data[k] == ',') {
                        first = k;
                    }
                }
                if (first == letter_ph.count) {
                    chars_put_byte(&codes, ',');
                    chars_put(&codes, letter_ph.data,
                              letter_ph.count);
                } else {
                    chars_put(&codes, letter_ph.data, first);
                    chars_put_byte(&codes, ',');
                    for (size_t k = first + 1; k < letter_ph.count;
                         k++) {
                        char c2 = letter_ph.data[k];
                        if (c2 != '\'' && c2 != ',') {
                            chars_put_byte(&codes, c2);
                        }
                    }
                }
            } else {
                bool has_prime = memchr(letter_ph.data, '\'',
                                        letter_ph.count) != NULL;
                if (!has_prime) {
                    chars_put_byte(&codes, '\'');
                }
                chars_put(&codes, letter_ph.data, letter_ph.count);
            }
            chars_free(&letter_ph);
        }
        if (codes.count > 0) {
            struct chars processed = {0};
            process_phoneme_string(codes.data, codes.count, false,
                                   &processed);
            phonemes_to_ipa(p, processed.data, processed.count,
                            out);
            chars_free(&processed);
        }
        chars_free(&codes);
    }
}

// Acronym / mixed-case-no-vowel handler. Returns true iff the token
// matched and IPA was appended; false leaves *result alone.
static bool spell_acronym_token(struct phonemizer * p,
                                const struct token * t,
                                struct chars * result,
                                bool * first_word) {
    struct chars lower = {0};
    to_lower(t->text.data, t->text.count, &lower);
    bool all_upper = t->text.count >= 2;
    for (size_t i = 0; i < t->text.count; i++) {
        if (!isupper((unsigned char)t->text.data[i])) {
            all_upper = false;
        }
    }
    bool unknown_word = smap_get(&p->dict,
                                 lower.data, lower.count) == NULL;
    bool mixed_case_no_vowel_abbrev = false;
    if (!all_upper && t->text.count >= 2 && unknown_word) {
        bool any_upper = false, any_lower = false, any_vowel = false;
        bool all_nasal = true;
        for (size_t i = 0; i < t->text.count; i++) {
            unsigned char ch = (unsigned char)t->text.data[i];
            if (isupper(ch)) { any_upper = true; }
            if (islower(ch)) { any_lower = true; }
            char lc = (char)tolower(ch);
            if (lc == 'a' || lc == 'e' || lc == 'i'
                || lc == 'o' || lc == 'u') {
                any_vowel = true;
            }
            // `all_nasal` carves out hum-interjection shapes from
            // the letter-spell trigger below: tokens whose letters
            // are ALL drawn from {m, h, n} are the canonical shape
            // of nasal interjections ("Mmm", "Mhm", "Hmm", "Mhmm",
            // "Nn", "Mn"). Real abbreviations like "Ph", "PhD",
            // "BSc" include other consonants so they fall through
            // this gate unchanged. Mixed-case + no-vowel + non-
            // nasal is still treated as a letter-spelled
            // abbreviation. The phonemizer's rules engine handles
            // the lowercased nasal form correctly (e.g. "mmm" ->
            // /məm/, "mhm" -> /məm/), so we just want the
            // capitalized form to fall through to it.
            if (lc != 'm' && lc != 'h' && lc != 'n') {
                all_nasal = false;
            }
        }
        mixed_case_no_vowel_abbrev =
            any_upper && any_lower && !any_vowel && !all_nasal;
    }
    bool consumed = false;
    bool trigger = (all_upper || mixed_case_no_vowel_abbrev)
        && (set_has(&p->abbrev_words, lower.data, lower.count)
            || unknown_word);
    if (trigger) {
        // CamelCase split for mixed-case abbreviations.
        // Walk text; new group starts whenever prev is lower and
        // current is upper.
        bool first_grp = true;
        size_t gs = 0;
        for (size_t ci = 0; ci <= t->text.count; ci++) {
            bool boundary = false;
            if (mixed_case_no_vowel_abbrev && ci > 0
                && ci < t->text.count
                && islower((unsigned char)t->text.data[ci - 1])
                && isupper((unsigned char)t->text.data[ci])) {
                boundary = true;
            }
            if (boundary || ci == t->text.count) {
                if (ci > gs) {
                    struct chars ipa = {0};
                    spell_group(p, t->text.data + gs, ci - gs,
                                &ipa);
                    if (ipa.count > 0) {
                        if (first_grp) {
                            if (t->needs_space && !*first_word) {
                                chars_put_byte(result, ' ');
                            }
                        } else {
                            chars_put_byte(result, ' ');
                        }
                        chars_put(result, ipa.data, ipa.count);
                        *first_word = false;
                        first_grp = false;
                    }
                    chars_free(&ipa);
                }
                gs = ci;
            }
        }
        consumed = true;
    }
    chars_free(&lower);
    return consumed;
}

// ---------------------------------------------------------------------------
// Cross-word probes + override / r-linking / t-flap helpers (D-26c).
// All operate on token streams + the per-token IPA / ph_codes
// strings. Each helper has a tight single-purpose contract; the
// processWordToken pipeline composes them in fixed order.
// ---------------------------------------------------------------------------

// True iff ipa ends in ɚ (U+025A: 0xC9 0x9A) or ɜː (U+025C U+02D0:
// 0xC9 0x9C 0xCB 0x90). Used to detect candidates for r-sandhi.
static bool ends_in_rhotic_r(const char * ipa, size_t n) {
    bool ends_in_schwar = (n >= 2
        && (unsigned char)ipa[n - 2] == 0xC9
        && (unsigned char)ipa[n - 1] == 0x9A);
    bool ends_in_long_er = (n >= 4
        && (unsigned char)ipa[n - 4] == 0xC9
        && (unsigned char)ipa[n - 3] == 0x9C
        && (unsigned char)ipa[n - 2] == 0xCB
        && (unsigned char)ipa[n - 1] == 0x90);
    return ends_in_schwar || ends_in_long_er;
}

// Walk forward from `start` until either a word token is hit (then
// returns its index) or a non-word (punctuation) token is hit
// (returns (size_t)-1, meaning the punctuation breaks the link).
static size_t find_next_word_stop_on_punct(const struct tokens * ts,
                                           size_t start) {
    size_t found = (size_t)-1;
    bool stop = false;
    size_t tj = start;
    while (tj < ts->count && found == (size_t)-1 && !stop) {
        if (ts->data[tj].is_word) {
            if (ts->data[tj].text.count > 0) { found = tj; }
        } else {
            stop = true;
        }
        tj++;
    }
    return found;
}

// Walk forward, skipping non-word tokens, returns the first
// non-empty word's index ((size_t)-1 if none).
static size_t find_next_non_empty_word(const struct tokens * ts,
                                       size_t start) {
    size_t found = (size_t)-1;
    size_t tk = start;
    while (tk < ts->count && found == (size_t)-1) {
        if (ts->data[tk].is_word && ts->data[tk].text.count > 0) {
            found = tk;
        }
        tk++;
    }
    return found;
}

// True iff the next token's word starts with a vowel letter AND
// doesn't have a /j/ phoneme onset (so "u" in "utility" -> jˈuː
// blocks linking-r). Caller picks the index.
static bool word_is_vowel_initial_non_yod(struct phonemizer * p,
                                          const struct tokens * ts,
                                          size_t idx) {
    bool result = false;
    if (ts->data[idx].text.count > 0) {
        char fc = (char)tolower(
            (unsigned char)ts->data[idx].text.data[0]);
        bool is_vowel = (fc == 'a' || fc == 'e' || fc == 'i'
                         || fc == 'o' || fc == 'u');
        if (is_vowel) {
            bool j_onset = false;
            if (fc == 'u' || fc == 'e') {
                struct chars lower = {0};
                to_lower(ts->data[idx].text.data,
                         ts->data[idx].text.count, &lower);
                struct chars nph = {0};
                word_to_phonemes(p, lower.data, lower.count, &nph);
                size_t npi = 0;
                while (npi < nph.count
                       && (nph.data[npi] == '\''
                           || nph.data[npi] == ','
                           || nph.data[npi] == '%'
                           || nph.data[npi] == '=')) {
                    npi++;
                }
                j_onset = (npi < nph.count && nph.data[npi] == 'j');
                chars_free(&nph);
                chars_free(&lower);
            }
            result = !j_onset;
        }
    }
    return result;
}

// True iff the next word starts with a vowel phoneme code. Handles
// abbreviation-style first-letter spelling when the token contains
// '.', is all-upper-unknown, or is in abbrev_words.
static bool next_word_is_vowel_initial(struct phonemizer * p,
                                       const struct tokens * ts,
                                       size_t tj) {
    bool use_first_letter = (memchr(ts->data[tj].text.data, '.',
                                    ts->data[tj].text.count)
                             != NULL);
    if (!use_first_letter && ts->data[tj].text.count >= 2) {
        bool all_upper = ts->data[tj].text.count > 0;
        for (size_t i = 0; i < ts->data[tj].text.count; i++) {
            if (!isupper(
                (unsigned char)ts->data[tj].text.data[i])) {
                all_upper = false;
            }
        }
        if (all_upper) {
            struct chars lower_nxt = {0};
            to_lower(ts->data[tj].text.data,
                     ts->data[tj].text.count, &lower_nxt);
            bool unknown = smap_get(&p->dict, lower_nxt.data,
                                    lower_nxt.count) == NULL;
            bool is_abbrev = set_has(&p->abbrev_words,
                                     lower_nxt.data,
                                     lower_nxt.count);
            if (unknown || is_abbrev) { use_first_letter = true; }
            chars_free(&lower_nxt);
        }
    }
    struct chars next_ph = {0};
    if (use_first_letter) {
        bool found = false;
        for (size_t i = 0;
             i < ts->data[tj].text.count && !found; i++) {
            char lc = ts->data[tj].text.data[i];
            if (isalpha((unsigned char)lc)) {
                word_to_phonemes(p, &lc, 1, &next_ph);
                found = true;
            }
        }
    } else {
        word_to_phonemes(p, ts->data[tj].text.data,
                         ts->data[tj].text.count, &next_ph);
    }
    size_t pi = 0;
    while (pi < next_ph.count
           && (next_ph.data[pi] == '\''
               || next_ph.data[pi] == ','
               || next_ph.data[pi] == '%'
               || next_ph.data[pi] == '=')) {
        pi++;
    }
    bool r = pi < next_ph.count
        && is_vowel_code(next_ph.data + pi, 1);
    chars_free(&next_ph);
    return r;
}

// "or"/"and" between vowels suppresses linking-r (cross-conjunction
// pauses break the chain).
static bool conjunction_suppresses_linking(const struct tokens * ts,
                                           size_t tj,
                                           bool first_word) {
    bool suppress = false;
    struct chars lc = {0};
    to_lower(ts->data[tj].text.data, ts->data[tj].text.count, &lc);
    bool is_conjunction =
        (lc.count == 2 && memcmp(lc.data, "or", 2) == 0)
        || (lc.count == 3 && memcmp(lc.data, "and", 3) == 0);
    if (is_conjunction && !first_word) {
        size_t after = find_next_word_stop_on_punct(ts, tj + 1);
        suppress = (after != (size_t)-1);
    }
    chars_free(&lc);
    return suppress;
}

// R-linking (r-sandhi). Inserts ɹ (U+0279, 0xC9 0xB9) at the end of
// ipa when the current word ends in ɚ or ɜː and the next word
// begins with a vowel phoneme. Suppressed before mid-utterance
// conjunctions and across punctuation.
static void apply_r_linking(struct phonemizer * p,
                            bool first_word, size_t ti,
                            const struct tokens * ts,
                            struct chars * ipa) {
    if (ends_in_rhotic_r(ipa->data, ipa->count)) {
        size_t tj = find_next_word_stop_on_punct(ts, ti + 1);
        if (tj != (size_t)-1
            && !conjunction_suppresses_linking(ts, tj, first_word)
            && next_word_is_vowel_initial(p, ts, tj)) {
            chars_put(ipa, "\xC9\xB9", 2);
        }
    }
}

// Inter-word t-flap on ph_codes. "t#" tail (function-word
// final-t marker like at#/it#/but#) becomes "*" (flap) when the
// next word starts with a vowel phoneme. No-op otherwise.
static void apply_inter_word_t_flap(struct phonemizer * p,
                                    size_t ti,
                                    const struct tokens * ts,
                                    struct chars * ph_codes) {
    if (ph_codes->count >= 2
        && ph_codes->data[ph_codes->count - 2] == 't'
        && ph_codes->data[ph_codes->count - 1] == '#') {
        size_t tnext = ti + 1;
        while (tnext < ts->count && !ts->data[tnext].is_word) {
            tnext++;
        }
        if (tnext < ts->count
            && ts->data[tnext].text.count > 0) {
            struct chars lower = {0};
            to_lower(ts->data[tnext].text.data,
                     ts->data[tnext].text.count, &lower);
            struct chars nph = {0};
            word_to_phonemes(p, lower.data, lower.count, &nph);
            size_t npi = 0;
            while (npi < nph.count
                   && (nph.data[npi] == '\''
                       || nph.data[npi] == ','
                       || nph.data[npi] == '%'
                       || nph.data[npi] == '=')) {
                npi++;
            }
            static const char VOWEL_STARTS[] = "aAeEIiOUVu03@oY";
            bool next_vowel_onset = (npi < nph.count
                && strchr(VOWEL_STARTS, nph.data[npi]) != NULL);
            if (next_vowel_onset) {
                ph_codes->data[ph_codes->count - 2] = '*';
                ph_codes->count--;
                ph_codes->data[ph_codes->count] = '\0';
            }
            chars_free(&nph);
            chars_free(&lower);
        }
    }
}

// Cross-word @->3 rhotacization. Standalone trailing '@' (schwa)
// promotes to '3' (ɚ) before an r-initial next word. Skipped when
// '@' is the tail of a diphthong digraph (i@/e@/A@/O@/o@/U@).
static void apply_cross_word_schwa_rhotic(struct phonemizer * p,
                                          size_t ti,
                                          bool is_en_us,
                                          bool is_isolated_word,
                                          const struct tokens * ts,
                                          struct chars * ph_codes) {
    if (is_en_us && !is_isolated_word
        && ph_codes->count > 0
        && ph_codes->data[ph_codes->count - 1] == '@') {
        char prev = ph_codes->count >= 2
            ? ph_codes->data[ph_codes->count - 2] : 0;
        bool standalone = !(prev == 'A' || prev == 'e' || prev == 'O'
                            || prev == 'o' || prev == 'U'
                            || prev == 'i');
        if (standalone) {
            size_t tj = ti + 1;
            if (tj < ts->count && ts->data[tj].is_word
                && ts->data[tj].text.count > 0) {
                struct chars nph = {0};
                word_to_phonemes(p, ts->data[tj].text.data,
                                 ts->data[tj].text.count, &nph);
                size_t pi = 0;
                while (pi < nph.count
                       && (nph.data[pi] == '\''
                           || nph.data[pi] == ','
                           || nph.data[pi] == '%'
                           || nph.data[pi] == '=')) {
                    pi++;
                }
                if (pi < nph.count && nph.data[pi] == 'r') {
                    ph_codes->data[ph_codes->count - 1] = '3';
                }
                chars_free(&nph);
            }
        }
    }
}

// Cross-word /t/ flap on "it" before vowel-initial next word.
// Operates on the post-IPA string. Restricted to "it"; "that" /
// "but" / "not" do NOT flap cross-word.
static void apply_cross_word_t_flap(struct phonemizer * p,
                                    const struct token * t,
                                    size_t ti,
                                    bool is_en_us,
                                    bool is_isolated_word,
                                    const struct tokens * ts,
                                    struct chars * ipa) {
    bool token_is_it = (t->text.count == 2
        && tolower((unsigned char)t->text.data[0]) == 'i'
        && tolower((unsigned char)t->text.data[1]) == 't');
    if (is_en_us && !is_isolated_word
        && ipa->count > 0 && ipa->data[ipa->count - 1] == 't'
        && token_is_it) {
        size_t tj = ti + 1;
        while (tj < ts->count && !ts->data[tj].is_word) { tj++; }
        if (tj < ts->count && ts->data[tj].text.count > 0) {
            struct chars nph = {0};
            word_to_phonemes(p, ts->data[tj].text.data,
                             ts->data[tj].text.count, &nph);
            size_t pi2 = 0;
            while (pi2 < nph.count
                   && (nph.data[pi2] == '\''
                       || nph.data[pi2] == ','
                       || nph.data[pi2] == '%'
                       || nph.data[pi2] == '=')) {
                pi2++;
            }
            if (pi2 < nph.count) {
                char fc = nph.data[pi2];
                static const char NEXT_VOWELS[] = "aAeEiIoOuUV03@";
                bool nv = strchr(NEXT_VOWELS, fc) != NULL;
                if (nv) {
                    ipa->count--;          // drop trailing 't'
                    chars_put(ipa, "\xC9\xBE", 2); // ɾ
                }
            }
            chars_free(&nph);
        }
    }
}

// Step D: shift '\'' out of diphthong digraphs (e.g. "e'I" -> "'eI"
// so the full digraph gets primary stress).
static void fix_diphthong_stress_position(struct chars * ph_codes) {
    static const char * const DIPHS[] = {
        "eI", "aI", "aU", "OI", "oU", NULL
    };
    for (size_t si = 1; si + 1 < ph_codes->count; si++) {
        if (ph_codes->data[si] == '\'') {
            char prev = ph_codes->data[si - 1];
            char next = ph_codes->data[si + 1];
            bool matched = false;
            for (int di = 0; DIPHS[di] != NULL && !matched; di++) {
                matched = (DIPHS[di][0] == prev
                           && DIPHS[di][1] == next);
                if (matched) {
                    memmove(ph_codes->data + si,
                            ph_codes->data + si + 1,
                            ph_codes->count - si - 1);
                    ph_codes->count--;
                    ph_codes->data[ph_codes->count] = '\0';
                    // Re-insert '\'' BEFORE prev (position si - 1).
                    chars_grow(ph_codes, ph_codes->count + 2);
                    memmove(ph_codes->data + si,
                            ph_codes->data + si - 1,
                            ph_codes->count - (si - 1));
                    ph_codes->data[si - 1] = '\'';
                    ph_codes->count++;
                    ph_codes->data[ph_codes->count] = '\0';
                }
            }
        }
    }
}

// Pre-vowel "the" allophone fixup. Swap trailing ə (0xC9 0x99) to
// ɪ (0xC9 0xAA) when the phrase ends in "the" and the next word
// is vowel-initial non-yod.
static void apply_pre_vowel_the_fixup(bool phrase_pre_vowel_the,
                                      struct chars * ipa) {
    if (phrase_pre_vowel_the && ipa->count >= 2
        && (unsigned char)ipa->data[ipa->count - 2] == 0xc9
        && (unsigned char)ipa->data[ipa->count - 1] == 0x99) {
        ipa->data[ipa->count - 1] = (char)0xaa;
    }
}

// Default-stress decision. Inserts ˈ before the first non-schwa
// vowel UNLESS the token is inherently unstressed (% function word,
// $u-flagged in context, weak-schwa @2/@5 with no stress, article
// a/an in context).
static void maybe_add_default_stress(struct phonemizer * p,
                                     const struct token * t,
                                     bool phrase_matched,
                                     bool is_isolated_word,
                                     bool is_unstressed_word,
                                     const char * ph_codes,
                                     size_t pn,
                                     struct chars * ipa) {
    bool has_pct = memchr(ph_codes, '%', pn) != NULL;
    bool has_prime = memchr(ph_codes, '\'', pn) != NULL;
    bool has_comma = memchr(ph_codes, ',', pn) != NULL;
    bool pct_unstressed = has_pct && !has_prime
        && (!has_comma || phrase_matched) && !is_isolated_word;
    bool u_unstressed = is_unstressed_word && !is_isolated_word;
    bool has_at2 = pn >= 2 && memmem(ph_codes, pn, "@2", 2) != NULL;
    bool has_at5 = pn >= 2 && memmem(ph_codes, pn, "@5", 2) != NULL;
    bool weak_schwa = (has_at2 || has_at5)
        && !has_prime && !has_comma;
    struct chars tl = {0};
    to_lower(t->text.data, t->text.count, &tl);
    bool is_a = tl.count == 1 && tl.data[0] == 'a';
    bool is_an = tl.count == 2 && memcmp(tl.data, "an", 2) == 0;
    bool is_a_sharp = pn == 2 && memcmp(ph_codes, "a#", 2) == 0;
    bool is_a_sharp_n = pn == 3 && memcmp(ph_codes, "a#n", 3) == 0;
    bool article_a = !is_isolated_word && (is_a || is_an)
        && (is_a_sharp || is_a_sharp_n);
    chars_free(&tl);
    bool no_stress = pct_unstressed || weak_schwa || u_unstressed
        || article_a;
    if (!no_stress) {
        struct chars adjusted = {0};
        add_default_stress(ipa->data, ipa->count, &adjusted);
        ipa->count = 0;
        chars_put(ipa, adjusted.data, adjusted.count);
        chars_free(&adjusted);
    }
    (void)p;
}

// Update expect_past/noun/verb POS-context counters. $pastf/$nounf/
// $verbf flags set a fresh window; the always-decrement at the end
// realises the 2/1 schedule.
static void update_pos_context_counters(struct phonemizer * p,
                                        const struct token * t,
                                        int * expect_past,
                                        int * expect_noun,
                                        int * expect_verb) {
    struct chars lw = {0};
    to_lower(t->text.data, t->text.count, &lw);
    if (set_has(&p->pastf_words, lw.data, lw.count)) {
        *expect_past = 3;
        *expect_noun = 0;
        *expect_verb = 0;
    } else if (set_has(&p->nounf_words, lw.data, lw.count)) {
        *expect_noun = 2;
        *expect_past = 0;
        *expect_verb = 0;
    } else if (set_has(&p->verbf_words, lw.data, lw.count)) {
        *expect_verb = 2;
        *expect_past = 0;
        *expect_noun = 0;
    }
    if (*expect_past > 0) { (*expect_past)--; }
    if (*expect_noun > 0) { (*expect_noun)--; }
    if (*expect_verb > 0) { (*expect_verb)--; }
    chars_free(&lw);
}

// POS-context override. After $pastf/$nounf/$verbf bumped a
// counter, swap ph_codes for the side-dict entry if it exists.
// Gated off for isolated words and phrase-matched bigrams.
static void apply_pos_context_override(struct phonemizer * p,
                                       const struct token * t,
                                       bool is_isolated_word,
                                       bool phrase_matched,
                                       int expect_past,
                                       int expect_noun,
                                       int expect_verb,
                                       struct chars * ph_codes) {
    if (!is_isolated_word && !phrase_matched) {
        struct chars lw = {0};
        to_lower(t->text.data, t->text.count, &lw);
        struct chars * hit = NULL;
        if (expect_past > 0) {
            hit = smap_get(&p->past_dict, lw.data, lw.count);
        } else if (expect_noun > 0) {
            hit = smap_get(&p->noun_dict, lw.data, lw.count);
        } else if (expect_verb > 0) {
            hit = smap_get(&p->verb_dict, lw.data, lw.count);
        }
        if (hit != NULL) {
            struct chars tmp = {0};
            process_phoneme_string(hit->data, hit->count, false,
                                   &tmp);
            ph_codes->count = 0;
            chars_put(ph_codes, tmp.data, tmp.count);
            chars_free(&tmp);
        }
        chars_free(&lw);
    }
}

// $atstart: first-word dict override (atstart_dict).
static void apply_atstart_override(struct phonemizer * p,
                                   const struct token * t,
                                   bool first_word,
                                   bool phrase_matched,
                                   struct chars * ph_codes) {
    if (first_word && !phrase_matched) {
        struct chars lw = {0};
        to_lower(t->text.data, t->text.count, &lw);
        struct chars * ait = smap_get(&p->atstart_dict,
                                      lw.data, lw.count);
        if (ait != NULL) {
            ph_codes->count = 0;
            chars_put(ph_codes, ait->data, ait->count);
        }
        chars_free(&lw);
    }
}

// $atend: last-word-of-utterance dict override (atend_dict).
// Skipped for isolated words and phrase-matched tokens.
static void apply_atend_override(struct phonemizer * p,
                                 const struct token * t,
                                 size_t ti, size_t last_word_ti,
                                 bool is_isolated_word,
                                 bool phrase_matched,
                                 struct chars * ph_codes) {
    if (ti == last_word_ti && !is_isolated_word && !phrase_matched) {
        struct chars lw = {0};
        to_lower(t->text.data, t->text.count, &lw);
        struct chars * aeit = smap_get(&p->atend_dict,
                                       lw.data, lw.count);
        if (aeit != NULL) {
            ph_codes->count = 0;
            chars_put(ph_codes, aeit->data, aeit->count);
        }
        chars_free(&lw);
    }
}

// Hand-coded function-word allophone overrides. Each branch is
// mutually exclusive (lemma equality gate). See .cpp comment for
// the linguistic justification of each branch.
static void apply_lemma_override(struct phonemizer * p,
                                 const struct token * t, size_t ti,
                                 const struct tokens * ts,
                                 bool is_isolated_word,
                                 size_t last_word_ti,
                                 struct chars * ph_codes) {
    struct chars lw = {0};
    to_lower(t->text.data, t->text.count, &lw);
    bool eq_the = lw.count == 3 && memcmp(lw.data, "the", 3) == 0;
    bool eq_a = lw.count == 1 && lw.data[0] == 'a';
    bool eq_an = lw.count == 2 && memcmp(lw.data, "an", 2) == 0;
    bool eq_to = lw.count == 2 && memcmp(lw.data, "to", 2) == 0;
    bool eq_use = lw.count == 3 && memcmp(lw.data, "use", 3) == 0;
    if (eq_the && ph_codes->count > 0) {
        size_t tj = find_next_word_stop_on_punct(ts, ti + 1);
        if (tj != (size_t)-1
            && word_is_vowel_initial_non_yod(p, ts, tj)) {
            ph_codes->count = 0;
            chars_put(ph_codes, "%DI", 3);
        }
    }
    if (eq_a) {
        ph_codes->count = 0;
        if (is_isolated_word) {
            chars_put(ph_codes, "eI", 2);
        } else {
            chars_put(ph_codes, "a#", 2);
        }
    }
    if (eq_an && !is_isolated_word) {
        size_t tj2 = find_next_non_empty_word(ts, ti + 1);
        bool next_vowel_initial = false;
        if (tj2 != (size_t)-1) {
            char fc = (char)tolower(
                (unsigned char)ts->data[tj2].text.data[0]);
            next_vowel_initial = (fc == 'a' || fc == 'e'
                || fc == 'i' || fc == 'o' || fc == 'u');
        }
        ph_codes->count = 0;
        if (next_vowel_initial) {
            chars_put(ph_codes, "a#n", 3);
        } else {
            chars_put(ph_codes, "an", 2);
        }
    }
    if (eq_to) {
        ph_codes->count = 0;
        if (is_isolated_word || ti == last_word_ti) {
            chars_put(ph_codes, "tu:", 3);
        } else {
            size_t tj2 = find_next_non_empty_word(ts, ti + 1);
            bool use_tU = (tj2 != (size_t)-1
                && word_is_vowel_initial_non_yod(p, ts, tj2));
            if (use_tU) {
                chars_put(ph_codes, "tU", 2);
            } else {
                chars_put(ph_codes, "t@5", 3);
            }
        }
    }
    if (eq_use && !is_isolated_word) {
        static const char * const PRONOUNS[] = {
            "i", "we", "you", "they", "he", "she", "who", NULL
        };
        struct chars prev_word = {0};
        if (ti > 0) {
            bool found = false;
            int tj = (int)ti - 1;
            while (tj >= 0 && !found) {
                if (ts->data[tj].is_word) {
                    to_lower(ts->data[tj].text.data,
                             ts->data[tj].text.count, &prev_word);
                    found = true;
                }
                tj--;
            }
        }
        bool is_pronoun = false;
        for (int i = 0; PRONOUNS[i] != NULL && !is_pronoun; i++) {
            size_t pl = strlen(PRONOUNS[i]);
            is_pronoun = prev_word.count == pl
                && memcmp(prev_word.data, PRONOUNS[i], pl) == 0;
        }
        if (is_pronoun) {
            ph_codes->count = 0;
            chars_put(ph_codes, "ju:z", 4);
        }
        chars_free(&prev_word);
    }
    chars_free(&lw);
}

// ---------------------------------------------------------------------------
// Step B + Step C: function-word stress assignment (D-26d). Step B
// strips '\'' from $u-flagged words (suppress primary). Step C
// promotes/demotes stress on function-words based on sentence
// position + keep-secondary tables.
// ---------------------------------------------------------------------------

// Hand-coded keep-secondary table: function words that get their
// '\'' demoted to ',' in sentence context. Same list as the .cpp
// STEP_B_KEEP_SECONDARY_WORDS and the step-C KEEP_SECONDARY (the
// step-B table is a subset; step-C also includes "make"/"makes"
// and other forms).
static const char * const STEP_B_KEEP_SECONDARY_WORDS[] = {
    "within", "without", "about", "across", "above", "among",
    "amongst", "before", "upon", "below", "beside", "between",
    "beyond", "despite", "except", "inside", "outside", "toward",
    "towards", "along", "around", "behind", "beneath",
    "underneath", "over", "under",
    NULL
};

static const char * const STEP_C_KEEP_SECONDARY[] = {
    "on", "onto", "multiple", "multiples", "going",
    "into", "any", "how", "where", "why",
    "being", "while", "but",
    "across", "above", "among", "amongst", "before",
    "within", "without", "upon", "below", "beside", "between",
    "beyond", "underneath", "behind", "beneath",
    "over", "under",
    "about",
    "make", "makes",
    NULL
};

static const char * const STEP_C_NEEDS_SECONDARY[] = {
    "our", NULL
};

static bool word_list_has(const char * const * list,
                          const char * w, size_t wn) {
    bool found = false;
    for (int i = 0; list[i] != NULL && !found; i++) {
        size_t li = strlen(list[i]);
        if (li == wn && memcmp(list[i], w, wn) == 0) {
            found = true;
        }
    }
    return found;
}

// "-ing forms of $strend2 stems with secondary-stressed dict
// phoneme". E.g. "making" from "make" (m,eIk). Returns true iff
// the stem (bare or magic-e restored) is in strend_words AND
// that stem's dict_ entry contains ','.
static bool is_ing_of_strend_secondary(struct phonemizer * p,
                                       const char * w, size_t wn) {
    bool result = false;
    if (wn > 3 && memcmp(w + wn - 3, "ing", 3) == 0) {
        const char * base = w;
        size_t bn = wn - 3;
        struct chars sk = {0};
        if (set_has(&p->strend_words, base, bn)) {
            chars_put(&sk, base, bn);
        } else {
            struct chars magic_e = {0};
            chars_put(&magic_e, base, bn);
            chars_put_byte(&magic_e, 'e');
            if (set_has(&p->strend_words,
                        magic_e.data, magic_e.count)) {
                chars_put(&sk, magic_e.data, magic_e.count);
            }
            chars_free(&magic_e);
        }
        if (sk.count > 0) {
            struct chars * sit = smap_get(&p->dict,
                                          sk.data, sk.count);
            result = sit != NULL
                && memchr(sit->data, ',', sit->count) != NULL;
        }
        chars_free(&sk);
    }
    return result;
}

// keep_sec -> primary promotion: a keep_sec word gets promoted to
// primary stress when no following stressed content remains.
static void try_promote_keep_sec_to_primary(struct phonemizer * p,
                                            const char * tl,
                                            size_t tn,
                                            size_t ti,
                                            const struct tokens * ts,
                                            bool * keep_sec,
                                            struct chars * ph_codes) {
    if (*keep_sec && !set_has(&p->u_plus_secondary_words, tl, tn)) {
        bool has_following_stressed = false;
        for (size_t tj = ti + 1;
             tj < ts->count && !has_following_stressed; tj++) {
            if (ts->data[tj].is_word) {
                struct chars fw = {0};
                to_lower(ts->data[tj].text.data,
                         ts->data[tj].text.count, &fw);
                if (!set_has(&p->unstressed_words,
                             fw.data, fw.count)) {
                    bool fw_is_strend_sec = set_has(
                        &p->comma_strend2_words, fw.data, fw.count);
                    bool fw_is_ing_strend =
                        is_ing_of_strend_secondary(p,
                            fw.data, fw.count);
                    bool fw_in_secondary_set =
                        word_list_has(STEP_C_KEEP_SECONDARY,
                                      fw.data, fw.count)
                        || set_has(&p->u2_strend2_words,
                                   fw.data, fw.count)
                        || fw_is_strend_sec || fw_is_ing_strend;
                    struct chars * dit = smap_get(&p->dict,
                                                  fw.data, fw.count);
                    bool fw_weak = !fw_in_secondary_set
                        && dit != NULL && dit->count > 0
                        && (dit->data[0] == ',' || dit->data[0] == '%');
                    if (!fw_weak) { has_following_stressed = true; }
                }
                chars_free(&fw);
            }
        }
        if (!has_following_stressed) {
            *keep_sec = false;
            if (memchr(ph_codes->data, '\'', ph_codes->count)
                == NULL) {
                replace_first_char(ph_codes, ',', '\'');
            }
        }
    }
}

// Leading-comma gated ',' -> '\'' promotion.
static void apply_comma_to_primary_promotion(bool keep_sec,
                                             bool needs_sec,
                                             bool phrase_matched,
                                             bool is_isolated_word,
                                             struct chars * ph_codes) {
    size_t comma_pos = ph_codes->count;
    for (size_t i = 0;
         i < ph_codes->count && comma_pos == ph_codes->count; i++) {
        if (ph_codes->data[i] == ',') { comma_pos = i; }
    }
    bool comma_at_start = (comma_pos == 0
        || (comma_pos == 1 && ph_codes->count > 0
            && ph_codes->data[0] == '%'));
    static const char VOWEL_CHARS[] = "aAeEiIoOuUV03@";
    bool no_real_vowel_before = true;
    if (!comma_at_start && comma_pos < ph_codes->count) {
        for (size_t k = 0; k < comma_pos && no_real_vowel_before;
             k++) {
            char cc = ph_codes->data[k];
            if (cc != '\'' && cc != '%' && cc != '='
                && strchr(VOWEL_CHARS, cc) != NULL) {
                no_real_vowel_before = false;
            }
        }
    } else {
        no_real_vowel_before = false;
    }
    bool effectively_leading = !comma_at_start
        && comma_pos < ph_codes->count
        && no_real_vowel_before;
    bool leading_comma = comma_at_start || effectively_leading;
    bool has_prime = memchr(ph_codes->data, '\'',
                            ph_codes->count) != NULL;
    bool is_pct_phrase = phrase_matched && ph_codes->count > 0
        && ph_codes->data[0] == '%' && !is_isolated_word
        && !has_prime;
    bool should_promote = !keep_sec && !needs_sec && !has_prime
        && comma_pos < ph_codes->count && !is_pct_phrase
        && (is_isolated_word || !leading_comma);
    if (should_promote) {
        replace_first_char(ph_codes, ',', '\'');
    }
}

// Step B. Strip '\'' from $u-flagged words (unless they're in the
// step-B keep-secondary set), and from %-prefix words whose only
// primary is the step-5 last-resort before 'a#'. Returns
// is_unstressed_word for use by maybe_add_default_stress.
static bool apply_step_b(struct phonemizer * p,
                         const struct token * t,
                         bool phrase_matched,
                         bool is_isolated_word,
                         struct chars * ph_codes) {
    struct chars token_lower = {0};
    to_lower(t->text.data, t->text.count, &token_lower);
    // unstress_check = token_lower without trailing 's clitic.
    size_t apos_pos = token_lower.count;
    for (size_t i = 0;
         i < token_lower.count && apos_pos == token_lower.count;
         i++) {
        if (token_lower.data[i] == '\'') { apos_pos = i; }
    }
    bool is_unstressed_word = !phrase_matched
        && (set_has(&p->unstressed_words,
                    token_lower.data, token_lower.count)
            || (apos_pos != token_lower.count
                && set_has(&p->unstressed_words,
                           token_lower.data, apos_pos)));
    size_t prime_pos = ph_codes->count;
    for (size_t i = 0;
         i < ph_codes->count && prime_pos == ph_codes->count; i++) {
        if (ph_codes->data[i] == '\'') { prime_pos = i; }
    }
    size_t hash_a_pos = ph_codes->count;
    if (ph_codes->count >= 3) {
        for (size_t i = 0;
             i + 2 < ph_codes->count && hash_a_pos == ph_codes->count;
             i++) {
            if (ph_codes->data[i] == '\''
                && ph_codes->data[i + 1] == 'a'
                && ph_codes->data[i + 2] == '#') {
                hash_a_pos = i;
            }
        }
    }
    bool is_pct_word = !is_isolated_word
        && ph_codes->count > 0 && ph_codes->data[0] == '%'
        && prime_pos < ph_codes->count
        && hash_a_pos < ph_codes->count
        && prime_pos == hash_a_pos;
    bool is_step_b_keep_sec = word_list_has(
        STEP_B_KEEP_SECONDARY_WORDS,
        token_lower.data, token_lower.count);
    if ((is_unstressed_word || is_pct_word)
        && !is_isolated_word && !is_step_b_keep_sec) {
        size_t wi = 0;
        for (size_t ri = 0; ri < ph_codes->count; ri++) {
            if (ph_codes->data[ri] != '\'') {
                ph_codes->data[wi++] = ph_codes->data[ri];
            }
        }
        ph_codes->count = wi;
        ph_codes->data[ph_codes->count] = '\0';
    }
    chars_free(&token_lower);
    return is_unstressed_word;
}

// Step C. Function-word stress assignment in sentence context.
// Computes keep_sec/needs_sec/is_strend_secondary/is_ing_of_strend/
// is_keep_sec_phrase from token + state, then runs promotion,
// demotion, leading-comma-gated promotion, and $unstressend demotion.
static void apply_step_c(struct phonemizer * p,
                         const struct token * t, size_t ti,
                         const struct tokens * ts,
                         bool is_isolated_word, bool phrase_matched,
                         size_t last_word_ti,
                         const char * matched_phrase_key,
                         size_t mpk_n,
                         struct chars * ph_codes) {
    struct chars tl = {0};
    to_lower(t->text.data, t->text.count, &tl);
    bool has_prime = memchr(ph_codes->data, '\'',
                            ph_codes->count) != NULL;
    bool is_strend_secondary =
        set_has(&p->comma_strend2_words, tl.data, tl.count)
        && ph_codes->count > 0 && ph_codes->data[0] == ','
        && !has_prime;
    bool is_ing_of_strend = is_ing_of_strend_secondary(p,
        tl.data, tl.count);
    bool is_keep_sec_phrase = mpk_n > 0
        && set_has(&p->keep_sec_phrase_keys,
                   matched_phrase_key, mpk_n);
    bool keep_sec = !is_isolated_word
        && (word_list_has(STEP_C_KEEP_SECONDARY,
                          tl.data, tl.count)
            || set_has(&p->u2_strend2_words, tl.data, tl.count)
            || set_has(&p->u_plus_secondary_words,
                       tl.data, tl.count)
            || is_strend_secondary || is_ing_of_strend
            || is_keep_sec_phrase);
    bool needs_sec = !is_isolated_word
        && word_list_has(STEP_C_NEEDS_SECONDARY, tl.data, tl.count);
    try_promote_keep_sec_to_primary(p, tl.data, tl.count, ti, ts,
                                    &keep_sec, ph_codes);
    if (keep_sec) { replace_first_char(ph_codes, '\'', ','); }
    bool has_comma = memchr(ph_codes->data, ',',
                            ph_codes->count) != NULL;
    has_prime = memchr(ph_codes->data, '\'',
                       ph_codes->count) != NULL;
    if (needs_sec && !has_prime && !has_comma) {
        static const char STRONG_VOWELS[] = "aAeEiIoOuUV3";
        size_t pos = ph_codes->count;
        for (size_t i = 0;
             i < ph_codes->count && pos == ph_codes->count; i++) {
            if (strchr(STRONG_VOWELS, ph_codes->data[i]) != NULL) {
                pos = i;
            }
        }
        if (pos < ph_codes->count) {
            chars_grow(ph_codes, ph_codes->count + 2);
            memmove(ph_codes->data + pos + 1,
                    ph_codes->data + pos,
                    ph_codes->count - pos);
            ph_codes->data[pos] = ',';
            ph_codes->count++;
            ph_codes->data[ph_codes->count] = '\0';
        }
    }
    apply_comma_to_primary_promotion(keep_sec, needs_sec,
                                     phrase_matched,
                                     is_isolated_word, ph_codes);
    if (!is_isolated_word && ti == last_word_ti
        && set_has(&p->unstressend_words, tl.data, tl.count)) {
        replace_first_char(ph_codes, '\'', ',');
    }
    chars_free(&tl);
}

// ---------------------------------------------------------------------------
// Bigram cliticization + phrase lookup + period-abbrev (D-26e). Tried
// in 4 sub-paths before fall-through to wordToPhonemes; the direct-
// clitic and split-phrase paths emit IPA themselves, the static and
// loaded paths fill ph_codes for downstream processing.
// ---------------------------------------------------------------------------

// Direct (bigram -> IPA) clitic substitutions. The .cpp's
// cliticIpaTable() singleton, ported verbatim. Strings are UTF-8
// IPA literals.
static const struct ipa_override_entry CLITIC_IPA[] = {
    {"of a",    "\xc9\x99v\xc9\x99"},                     // əvə
    {"of the",  "\xca\x8cv\xc3\xb0\xc9\x99"},             // ʌvðə
    {"in the",  "\xc9\xaan\xc3\xb0\xc9\x99"},             // ɪnðə
    {"on the",  "\xc9\x94n\xc3\xb0\xc9\x99"},             // ɔnðə
    {"from the",
        "\x66\xc9\xb9\xca\x8cm\xc3\xb0\xc9\x99"},        // fɹʌmðə
    {"that a",
        "\xc3\xb0\xcb\x8c\xc3\xa6\xc9\xbe\xc9\x99"},     // ðˌæɾə
    {"i am",    "a\xc9\xaa\xc9\x90m"},                    // aɪɐm
    {"was a",   "w\xca\x8cz\xc9\x90"},                    // wʌzɐ
    {"to be",   "t\xc9\x99" "bi"},                        // təbi
    {"out of",
        "\xcb\x8c" "a\xca\x8a\xc9\xbe\xc9\x99v"},        // ˌaʊɾəv
    {NULL, NULL},
};

// Static phrase code dict. The .cpp's staticPhraseDict() singleton.
static const struct ipa_override_entry STATIC_PHRASE_CODES[] = {
    {"has been", "h'azbi:n"},
    {NULL, NULL},
};

// Linear lookup in the const ipa_override_entry table (small enough
// that hashing is overkill).
static const char * lookup_const_table(
        const struct ipa_override_entry * t,
        const char * k, size_t kn) {
    const char * result = NULL;
    for (int i = 0; t[i].code != NULL && result == NULL; i++) {
        size_t li = strlen(t[i].code);
        if (li == kn && memcmp(t[i].code, k, kn) == 0) {
            result = t[i].ipa;
        }
    }
    return result;
}

struct clitic_or_phrase_result {
    struct chars ph_codes;
    bool         phrase_matched;
    bool         clitic_matched;
    bool         phrase_pre_vowel_the;
    struct chars matched_phrase_key;
    size_t       advance_to;
};

// Build "wordA wordB" lower-cased bigram into *out.
static void build_lower_bigram(const struct token * t1,
                               const struct token * t2,
                               struct chars * out) {
    out->count = 0;
    struct chars lo1 = {0};
    struct chars lo2 = {0};
    to_lower(t1->text.data, t1->text.count, &lo1);
    to_lower(t2->text.data, t2->text.count, &lo2);
    chars_put(out, lo1.data, lo1.count);
    chars_put_byte(out, ' ');
    chars_put(out, lo2.data, lo2.count);
    chars_free(&lo2);
    chars_free(&lo1);
}

// Direct (bigram -> IPA) clitic.
static bool try_emit_direct_clitic(struct phonemizer * p,
                                   const char * bigram, size_t bn,
                                   const struct token * t,
                                   const struct tokens * ts,
                                   size_t tj,
                                   struct chars * result,
                                   bool * first_word) {
    const char * raw = lookup_const_table(CLITIC_IPA, bigram, bn);
    bool matched = (raw != NULL);
    if (matched) {
        struct chars clitic = {0};
        chars_puts(&clitic, raw);
        // "the"-ending bigram: swap trailing ə to ɪ before
        // vowel-initial non-yod next word.
        struct chars t2lo = {0};
        to_lower(ts->data[tj].text.data,
                 ts->data[tj].text.count, &t2lo);
        bool t2_is_the = (t2lo.count == 3
            && memcmp(t2lo.data, "the", 3) == 0);
        if (t2_is_the) {
            size_t tk = find_next_non_empty_word(ts, tj + 1);
            if (tk != (size_t)-1
                && word_is_vowel_initial_non_yod(p, ts, tk)
                && clitic.count >= 2
                && (unsigned char)clitic.data[clitic.count - 2]
                    == 0xc9
                && (unsigned char)clitic.data[clitic.count - 1]
                    == 0x99) {
                clitic.data[clitic.count - 1] = (char)0xaa;
            }
        }
        if (t->needs_space && !*first_word) {
            chars_put_byte(result, ' ');
        }
        chars_put(result, clitic.data, clitic.count);
        *first_word = false;
        chars_free(&t2lo);
        chars_free(&clitic);
    }
    return matched;
}

// Static phrase code lookup. Fills r->ph_codes with processed
// phonemes when matched.
static bool try_match_static_phrase(const char * bigram, size_t bn,
                                    struct clitic_or_phrase_result * r) {
    const char * raw = lookup_const_table(STATIC_PHRASE_CODES,
                                          bigram, bn);
    bool matched = (raw != NULL);
    if (matched) {
        struct chars tmp = {0};
        chars_puts(&tmp, raw);
        process_phoneme_string(tmp.data, tmp.count, false,
                               &r->ph_codes);
        r->phrase_matched = true;
        chars_free(&tmp);
    }
    return matched;
}

// Loaded phrase dict lookup (phrase_dict, populated by load_dict
// for $phrase entries). Detects pre-vowel "the" suffix for the
// downstream apply_pre_vowel_the_fixup.
static bool try_match_loaded_phrase(struct phonemizer * p,
                                    const char * bigram, size_t bn,
                                    size_t tj,
                                    const struct tokens * ts,
                                    struct clitic_or_phrase_result * r) {
    struct chars * pit = smap_get(&p->phrase_dict, bigram, bn);
    bool matched = (pit != NULL);
    if (matched) {
        bool phrase_ends_the = pit->count >= 3
            && memcmp(pit->data + pit->count - 3, "D@2", 3) == 0;
        if (phrase_ends_the) {
            size_t tk = find_next_non_empty_word(ts, tj + 1);
            r->phrase_pre_vowel_the = (tk != (size_t)-1
                && word_is_vowel_initial_non_yod(p, ts, tk));
        }
        process_phoneme_string(pit->data, pit->count, false,
                               &r->ph_codes);
        r->phrase_matched = true;
        chars_put(&r->matched_phrase_key, bigram, bn);
    }
    return matched;
}

// One half of a split-phrase: prefix '%' when no stress markers
// (and not the first half of a phrase with no primary anywhere).
// Hoisted from the .cpp's doSplitPart lambda (called twice).
static void render_split_part(struct phonemizer * p,
                              const struct chars * ph,
                              bool is_first, bool phrase_has_primary,
                              struct chars * out) {
    out->count = 0;
    bool has_stress =
        memchr(ph->data, '\'', ph->count) != NULL
        || memchr(ph->data, ',', ph->count) != NULL;
    struct chars ph_proc = {0};
    if (!has_stress && !(is_first && !phrase_has_primary)) {
        chars_put_byte(&ph_proc, '%');
    }
    chars_put(&ph_proc, ph->data, ph->count);
    struct chars processed = {0};
    process_phoneme_string(ph_proc.data, ph_proc.count, false,
                           &processed);
    phonemes_to_ipa(p, processed.data, processed.count, out);
    chars_free(&processed);
    chars_free(&ph_proc);
}

// Split-phrase emit (phrase_split_dict; "a||b" entries).
static bool try_emit_split_phrase(struct phonemizer * p,
                                  const char * bigram, size_t bn,
                                  const struct token * t,
                                  struct chars * result,
                                  bool * first_word) {
    struct chars view = chars_view(bigram, bn);
    struct strpair * psit = map_get(&p->phrase_split_dict, &view);
    bool matched = (psit != NULL);
    if (matched) {
        bool phrase_has_primary =
            memchr(psit->a.data, '\'', psit->a.count) != NULL
            || memchr(psit->b.data, '\'', psit->b.count) != NULL;
        struct chars ipa1 = {0};
        render_split_part(p, &psit->a, true, phrase_has_primary,
                          &ipa1);
        if (t->needs_space && !*first_word) {
            chars_put_byte(result, ' ');
        }
        chars_put(result, ipa1.data, ipa1.count);
        *first_word = false;
        struct chars ipa2 = {0};
        render_split_part(p, &psit->b, false, phrase_has_primary,
                          &ipa2);
        chars_put_byte(result, ' ');
        chars_put(result, ipa2.data, ipa2.count);
        chars_free(&ipa2);
        chars_free(&ipa1);
    }
    return matched;
}

// Top-level bigram dispatch. Tries the 4 sub-paths in order; the
// caller advances ti to r.advance_to and uses the flags to gate
// later steps.
static struct clitic_or_phrase_result
try_clitic_or_phrase(struct phonemizer * p,
                     const struct token * t, size_t ti,
                     const struct tokens * ts,
                     struct chars * result,
                     bool * first_word) {
    struct clitic_or_phrase_result r = {{0}, false, false, false,
                                        {0}, ti};
    size_t tj = ti + 1;
    while (tj < ts->count && !ts->data[tj].is_word) { tj++; }
    if (tj < ts->count && ts->data[tj].is_word) {
        struct chars bigram = {0};
        build_lower_bigram(t, &ts->data[tj], &bigram);
        if (try_emit_direct_clitic(p, bigram.data, bigram.count,
                                   t, ts, tj, result, first_word)) {
            r.clitic_matched = true;
            r.advance_to = tj;
        } else if (try_match_static_phrase(bigram.data, bigram.count,
                                           &r)) {
            r.advance_to = tj;
        } else if (try_match_loaded_phrase(p, bigram.data,
                                           bigram.count, tj, ts,
                                           &r)) {
            r.advance_to = tj;
        } else if (try_emit_split_phrase(p, bigram.data, bigram.count,
                                         t, result, first_word)) {
            r.clitic_matched = true;
            r.advance_to = tj;
        }
        chars_free(&bigram);
    }
    return r;
}

// Period-abbrev expansion. "U.S." / "U.K." / "N.Y." -> letter-spell
// with secondary stress on all but the last letter and primary on
// the last. Returns true and appends IPA when expanded; otherwise
// fills *ph_codes with either the single letter's phonemes or the
// raw token's word_to_phonemes result.
static bool expand_period_abbreviation(struct phonemizer * p,
                                       const struct token * t,
                                       struct chars * result,
                                       bool * first_word,
                                       struct chars * ph_codes) {
    bool consumed = false;
    bool has_dot = memchr(t->text.data, '.', t->text.count) != NULL;
    if (has_dot) {
        struct charsv letter_ipa = {0};
        for (size_t ci = 0; ci < t->text.count; ci++) {
            char lc = t->text.data[ci];
            if (isalpha((unsigned char)lc)) {
                char lc_lower = (char)tolower((unsigned char)lc);
                char uk[2] = { '_', lc_lower };
                struct chars * uit = smap_get(&p->dict, uk, 2);
                struct chars entry = {0};
                if (uit != NULL) {
                    chars_put(&entry, uit->data, uit->count);
                } else {
                    word_to_phonemes(p, &lc, 1, &entry);
                }
                charsv_put(&letter_ipa, entry);
                entry = (struct chars){0};
            }
        }
        if (letter_ipa.count >= 2) {
            struct chars combined_codes = {0};
            for (size_t li = 0; li < letter_ipa.count; li++) {
                struct chars * code = &letter_ipa.data[li];
                if (li + 1 < letter_ipa.count) {
                    size_t first = code->count;
                    for (size_t k = 0;
                         k < code->count && first == code->count;
                         k++) {
                        if (code->data[k] == '\''
                            || code->data[k] == ',') {
                            first = k;
                        }
                    }
                    if (first == code->count) {
                        chars_put_byte(&combined_codes, ',');
                        chars_put(&combined_codes,
                                  code->data, code->count);
                    } else {
                        chars_put(&combined_codes,
                                  code->data, first);
                        chars_put_byte(&combined_codes, ',');
                        for (size_t k = first + 1;
                             k < code->count; k++) {
                            char c2 = code->data[k];
                            if (c2 != '\'' && c2 != ',') {
                                chars_put_byte(&combined_codes, c2);
                            }
                        }
                    }
                } else {
                    bool has_prime = memchr(code->data, '\'',
                                            code->count) != NULL;
                    if (!has_prime) {
                        chars_put_byte(&combined_codes, '\'');
                    }
                    chars_put(&combined_codes, code->data,
                              code->count);
                }
            }
            struct chars combined_ipa = {0};
            phonemes_to_ipa(p, combined_codes.data,
                            combined_codes.count, &combined_ipa);
            if (t->needs_space && !*first_word) {
                chars_put_byte(result, ' ');
            }
            chars_put(result, combined_ipa.data,
                      combined_ipa.count);
            *first_word = false;
            consumed = true;
            chars_free(&combined_ipa);
            chars_free(&combined_codes);
        } else if (letter_ipa.count == 1) {
            ph_codes->count = 0;
            chars_put(ph_codes, letter_ipa.data[0].data,
                      letter_ipa.data[0].count);
        } else {
            word_to_phonemes(p, t->text.data, t->text.count,
                             ph_codes);
        }
        charsv_clear(&letter_ipa);
    } else {
        word_to_phonemes(p, t->text.data, t->text.count, ph_codes);
    }
    return consumed;
}

// ---------------------------------------------------------------------------
// process_word_token + phonemize_text + phonemize (D-26f). The
// top-level per-token pipeline composing all D-26 helpers in fixed
// order, then the per-utterance and per-batch drivers.
// ---------------------------------------------------------------------------

// Run the full per-token pipeline. Updates ti to skip over a
// consumed bigram. Appends the resulting IPA (with leading space
// when needed) to *result. Updates *first_word.
static void process_word_token(struct phonemizer * p,
                               const struct token * t,
                               size_t * ti,
                               const struct tokens * ts,
                               bool is_isolated_word,
                               size_t last_word_ti, bool is_en_us,
                               int * expect_past,
                               int * expect_noun,
                               int * expect_verb,
                               struct chars * result,
                               bool * first_word) {
    bool consumed = false;
    if (!consumed && expand_number_token(p, t, result, first_word)) {
        consumed = true;
    }
    if (!consumed && spell_acronym_token(p, t, result, first_word)) {
        consumed = true;
    }
    struct chars ph_codes = {0};
    bool phrase_matched = false;
    bool clitic_matched = false;
    bool phrase_pre_vowel_the = false;
    struct chars matched_phrase_key = {0};
    if (!consumed) {
        struct clitic_or_phrase_result cr =
            try_clitic_or_phrase(p, t, *ti, ts, result, first_word);
        ph_codes = cr.ph_codes;
        cr.ph_codes = (struct chars){0};
        phrase_matched = cr.phrase_matched;
        clitic_matched = cr.clitic_matched;
        phrase_pre_vowel_the = cr.phrase_pre_vowel_the;
        matched_phrase_key = cr.matched_phrase_key;
        cr.matched_phrase_key = (struct chars){0};
        *ti = cr.advance_to;
        if (clitic_matched) { consumed = true; }
    }
    if (!consumed && !phrase_matched) {
        if (expand_period_abbreviation(p, t, result, first_word,
                                       &ph_codes)) {
            consumed = true;
        }
    }
    if (!consumed) {
        apply_pos_context_override(p, t, is_isolated_word,
                                   phrase_matched,
                                   *expect_past, *expect_noun,
                                   *expect_verb, &ph_codes);
        apply_atstart_override(p, t, *first_word, phrase_matched,
                               &ph_codes);
        apply_atend_override(p, t, *ti, last_word_ti,
                             is_isolated_word, phrase_matched,
                             &ph_codes);
        apply_lemma_override(p, t, *ti, ts, is_isolated_word,
                             last_word_ti, &ph_codes);
        bool is_unstressed_word = apply_step_b(p, t,
            phrase_matched, is_isolated_word, &ph_codes);
        apply_step_c(p, t, *ti, ts, is_isolated_word,
                     phrase_matched, last_word_ti,
                     matched_phrase_key.data,
                     matched_phrase_key.count, &ph_codes);
        fix_diphthong_stress_position(&ph_codes);
        apply_inter_word_t_flap(p, *ti, ts, &ph_codes);
        apply_cross_word_schwa_rhotic(p, *ti, is_en_us,
            is_isolated_word, ts, &ph_codes);
        struct chars ipa = {0};
        phonemes_to_ipa(p, ph_codes.data, ph_codes.count, &ipa);
        maybe_add_default_stress(p, t, phrase_matched,
                                 is_isolated_word,
                                 is_unstressed_word,
                                 ph_codes.data, ph_codes.count,
                                 &ipa);
        apply_pre_vowel_the_fixup(phrase_pre_vowel_the, &ipa);
        apply_r_linking(p, *first_word, *ti, ts, &ipa);
        apply_cross_word_t_flap(p, t, *ti, is_en_us,
            is_isolated_word, ts, &ipa);
        if (ipa.count > 0) {
            if (!*first_word) { chars_put_byte(result, ' '); }
            chars_put(result, ipa.data, ipa.count);
            *first_word = false;
        }
        chars_free(&ipa);
    }
    update_pos_context_counters(p, t, expect_past, expect_noun,
                                expect_verb);
    chars_free(&matched_phrase_key);
    chars_free(&ph_codes);
}

// Per-utterance pipeline: tokenize, find the last-word token (for
// $atend / $unstressend gates), then run process_word_token over
// each word token. Punctuation tokens are not emitted in the IPA.
static void phonemize_text(struct phonemizer * p,
                           const char * text, size_t tn,
                           struct chars * out) {
    out->count = 0;
    if (p->loaded) {
        bool is_en_us = (p->dialect.count == 5
            && (memcmp(p->dialect.data, "en-us", 5) == 0
                || memcmp(p->dialect.data, "en_us", 5) == 0));
        struct tokens ts = {0};
        tokenize(text, tn, &ts);
        bool first_word = true;
        int word_token_count = 0;
        size_t last_word_ti = (size_t)-1;
        for (size_t tii = 0; tii < ts.count; tii++) {
            if (ts.data[tii].is_word) {
                word_token_count++;
                last_word_ti = tii;
            }
        }
        bool is_isolated_word = (word_token_count == 1);
        int expect_past = 0;
        int expect_noun = 0;
        int expect_verb = 0;
        for (size_t ti = 0; ti < ts.count; ti++) {
            if (ts.data[ti].is_word) {
                process_word_token(p, &ts.data[ti], &ti, &ts,
                    is_isolated_word, last_word_ti, is_en_us,
                    &expect_past, &expect_noun, &expect_verb,
                    out, &first_word);
            }
        }
        tokens_clear(&ts);
    }
}

// ---------------------------------------------------------------------------
// Public C API. Matches phonemizer/include/CEPhonemizer.h. Swift
// bridges through these names.
// ---------------------------------------------------------------------------

typedef void * PhonemizerHandle;

PhonemizerHandle phonemizer_create(const char * rules_path,
                                   const char * list_path,
                                   const char * dialect) {
    struct phonemizer * p = malloc(sizeof(struct phonemizer));
    if (p != NULL) {
        memset(p, 0, sizeof(*p));
        phonemizer_state_init(p);
        const char * d = (dialect != NULL) ? dialect : "en-us";
        chars_puts(&p->dialect, d);
        build_ipa_overrides(p);
        bool ok = load_dictionary(p, list_path)
                  && load_rules(p, rules_path);
        if (ok) {
            p->loaded = true;
        } else {
            phonemizer_state_free(p);
            free(p);
            p = NULL;
        }
    }
    return p;
}

void phonemizer_destroy(PhonemizerHandle h) {
    if (h != NULL) {
        struct phonemizer * p = h;
        phonemizer_state_free(p);
        free(p);
    }
}

char * phonemizer_phonemize(PhonemizerHandle h, const char * text) {
    char * result = NULL;
    if (h != NULL && text != NULL) {
        struct phonemizer * p = h;
        struct chars out = {0};
        phonemize_text(p, text, strlen(text), &out);
        result = malloc(out.count + 1);
        if (result != NULL) {
            memcpy(result, out.data, out.count);
            result[out.count] = '\0';
        }
        chars_free(&out);
    }
    return result;
}

void phonemizer_free_string(char * s) { free(s); }

const char * phonemizer_get_error(PhonemizerHandle h) {
    return h != NULL ? ((struct phonemizer *)h)->err : "null handle";
}

// ---------------------------------------------------------------------------
// Smoke test (compile as standalone with -DPHONEMIZER_TESTS)
// ---------------------------------------------------------------------------

#ifdef PHONEMIZER_TESTS

static void test_leaf_helpers(void) {
    // is_vowel_letter: case-insensitive aeiouy.
    assert(is_vowel_letter('a') && is_vowel_letter('A'));
    assert(is_vowel_letter('Y') && is_vowel_letter('e'));
    assert(!is_vowel_letter('b') && !is_vowel_letter('Z'));
    assert(!is_vowel_letter('1') && !is_vowel_letter(' '));
    // has_any_vowel_letter / has_any_vowel_code on edge cases.
    assert(!has_any_vowel_letter("", 0));
    assert(!has_any_vowel_letter("bcd", 3));
    assert(has_any_vowel_letter("bcdY", 4));
    assert(has_any_vowel_letter("aeiou", 5));
    assert(!has_any_vowel_code("", 0));
    assert(!has_any_vowel_code("bcd", 3));
    assert(has_any_vowel_code("bcd@", 4));
    assert(has_any_vowel_code("V", 1));
    // to_lower: empty + mixed.
    struct chars lo = {0};
    to_lower("", 0, &lo);
    assert(lo.count == 0 && lo.data != NULL && lo.data[0] == '\0');
    to_lower("AbC.dE-1", 8, &lo);
    assert(lo.count == 8 && strcmp(lo.data, "abc.de-1") == 0);
    chars_free(&lo);
    // trim: all whitespace, leading/trailing, internal preserved.
    struct chars tr = {0};
    trim(" \t\r\n", 4, &tr);
    assert(tr.count == 0);
    trim("  abc  ", 7, &tr);
    assert(tr.count == 3 && strcmp(tr.data, "abc") == 0);
    tr.count = 0; tr.data[0] = '\0';
    trim("a b\tc", 5, &tr);
    assert(tr.count == 5 && strcmp(tr.data, "a b\tc") == 0);
    chars_free(&tr);
    // replace_first_char: hits, miss.
    struct chars rp = {0};
    chars_puts(&rp, "abca");
    replace_first_char(&rp, 'a', 'X');
    assert(strcmp(rp.data, "Xbca") == 0);
    replace_first_char(&rp, 'z', 'Y');
    assert(strcmp(rp.data, "Xbca") == 0);
    chars_free(&rp);
    // parse_stress_n: in-range, out-of-range, wrong shape.
    assert(parse_stress_n("$1", 2) == 1);
    assert(parse_stress_n("$6", 2) == 6);
    assert(parse_stress_n("$0", 2) == 0);
    assert(parse_stress_n("$7", 2) == 0);
    assert(parse_stress_n("$1x", 3) == 0);
    assert(parse_stress_n("1", 1) == 0);
}

static void test_phoneme_utils(void) {
    // is_vowel_code: first-byte classification.
    assert(!is_vowel_code("", 0));
    assert(is_vowel_code("a", 1) && is_vowel_code("@", 1));
    assert(is_vowel_code("aI", 2) && is_vowel_code("3:", 2));
    assert(!is_vowel_code("p", 1) && !is_vowel_code("'", 1));
    assert(!is_vowel_code("k", 1));
    // count_suffix_syllables: groups + magic-e.
    assert(count_suffix_syllables("", 0) == 0);
    assert(count_suffix_syllables("ing", 3) == 1);
    assert(count_suffix_syllables("ed", 2) == 1);
    // Trailing 'e' after a non-vowel is treated as silent magic-e:
    // "able" sees a/e groups (2) then clamps via max(1, n-1) to 1;
    // "ize" likewise. "ical" has no trailing e -> stays at 2.
    assert(count_suffix_syllables("able", 4) == 1);
    assert(count_suffix_syllables("ize", 3) == 1);
    assert(count_suffix_syllables("ical", 4) == 2);
    // prefix_has_full_vowel: "kat" has 'a'; "p" has none.
    assert(!prefix_has_full_vowel("p", 1));
    assert(prefix_has_full_vowel("kat", 3));
    assert(prefix_has_full_vowel("'eI", 3));     // "eI" is in the table
    assert(prefix_has_full_vowel("k3:", 3));     // "3:" too
    assert(!prefix_has_full_vowel("'b", 2));     // stress mark + consonant
    // count_prefix_vowels: counts vowel CODES (multi-char first).
    assert(count_prefix_vowels("", 0) == 0);
    assert(count_prefix_vowels("p", 1) == 0);
    assert(count_prefix_vowels("a", 1) == 1);
    assert(count_prefix_vowels("aI", 2) == 1);    // single multi-char vowel
    assert(count_prefix_vowels("a@", 2) == 2);    // two single-char vowels
    assert(count_prefix_vowels("'a", 2) == 1);    // stress mark skipped
    assert(count_prefix_vowels("k3:t", 4) == 1);  // 3: is one vowel
    // prefix_ends_in_schwa: last vowel-code identity.
    assert(!prefix_ends_in_schwa("", 0));
    assert(prefix_ends_in_schwa("s'u:p3", 6));    // ends in "3"
    assert(prefix_ends_in_schwa("k@", 2));         // ends in "@"
    assert(prefix_ends_in_schwa("kI2", 3));        // ends in "I2"
    assert(!prefix_ends_in_schwa("kat", 3));       // ends in "a" (not schwa)
    assert(!prefix_ends_in_schwa("ki:", 3));       // ends in "i:" (not schwa)
}

static void put_chars_key(struct map * m, const char * k,
                          const void * v) {
    struct chars key = {0};
    chars_puts(&key, k);
    map_put(m, &key, v);
    chars_free(&key);
}

static void * get_chars_key(struct map * m, const char * k) {
    struct chars key = {0};
    chars_puts(&key, k);
    void * r = map_get(m, &key);
    chars_free(&key);
    return r;
}

static void test_state_init_and_roundtrip(void) {
    struct phonemizer p = {0};
    phonemizer_state_init(&p);

    // smap put/get: dict["hello"] = "hElO"
    struct chars val = {0};
    chars_puts(&val, "hElO");
    put_chars_key(&p.dict, "hello", &val);
    val = (struct chars){0};  // ownership moved into map
    struct chars * got = get_chars_key(&p.dict, "hello");
    assert(got != NULL && strcmp(got->data, "hElO") == 0);
    assert(get_chars_key(&p.dict, "missing") == NULL);

    // set membership: pastf_words += "was"
    char one = 1;
    put_chars_key(&p.pastf_words, "was", &one);
    assert(get_chars_key(&p.pastf_words, "was") != NULL);
    assert(get_chars_key(&p.pastf_words, "ran") == NULL);

    // imap: stress_pos["construct"] = 2
    int two = 2;
    put_chars_key(&p.stress_pos, "construct", &two);
    int * sp = get_chars_key(&p.stress_pos, "construct");
    assert(sp != NULL && *sp == 2);

    // pmap: phrase_split_dict["most of"] = {"moUst", "@v"}
    struct strpair sp_val = {0};
    chars_puts(&sp_val.a, "moUst");
    chars_puts(&sp_val.b, "@v");
    put_chars_key(&p.phrase_split_dict, "most of", &sp_val);
    sp_val = (struct strpair){0};
    struct strpair * pg = get_chars_key(&p.phrase_split_dict, "most of");
    assert(pg != NULL);
    assert(strcmp(pg->a.data, "moUst") == 0);
    assert(strcmp(pg->b.data, "@v") == 0);

    // strpairs (compound_prefixes) round-trip.
    struct strpair cp = {0};
    chars_puts(&cp.a, "under");
    chars_puts(&cp.b, "Vnd3");
    strpairs_put(&p.compound_prefixes, cp);
    cp = (struct strpair){0};
    assert(p.compound_prefixes.count == 1);
    assert(strcmp(p.compound_prefixes.data[0].a.data, "under") == 0);
    assert(strcmp(p.compound_prefixes.data[0].b.data, "Vnd3") == 0);

    // dialect chars round-trip.
    chars_puts(&p.dialect, "en-us");
    assert(strcmp(p.dialect.data, "en-us") == 0);

    phonemizer_state_free(&p);
    // After free, the maps + strpairs are all empty.
    assert(p.dict.capacity == 0 && p.dict.count == 0);
    assert(p.pastf_words.capacity == 0);
    assert(p.phrase_split_dict.capacity == 0);
    assert(p.compound_prefixes.data == NULL);
    assert(p.compound_prefixes.count == 0);
}

static void check_tok(const struct token * tk, const char * text,
                      bool is_word, bool needs_space) {
    assert(tk->text.count == strlen(text));
    assert(strcmp(tk->text.data, text) == 0);
    assert(tk->is_word == is_word);
    assert(tk->needs_space == needs_space);
}

static void test_tokenize(void) {
    struct tokens out = {0};
    // Simple two-word sentence with trailing period.
    tokenize("hello world.", 12, &out);
    assert(out.count == 3);
    check_tok(&out.data[0], "hello", true,  false);
    check_tok(&out.data[1], "world", true,  true);
    check_tok(&out.data[2], ".",     false, false);
    // Re-tokenize into the same buffer (tokens_clear must free old).
    tokenize("a, b!", 5, &out);
    assert(out.count == 4);
    check_tok(&out.data[0], "a", true,  false);
    check_tok(&out.data[1], ",", false, false);
    check_tok(&out.data[2], "b", true,  true);
    check_tok(&out.data[3], "!", false, false);
    // Two-letter abbreviation collapses: the second '.' lands while
    // cur is mid-abbreviation ("U.S"), so it appends + flushes
    // "U.S." as one word token.
    tokenize("U.S.", 4, &out);
    assert(out.count == 1);
    check_tok(&out.data[0], "U.S.", true, false);
    // Three-letter abbreviations don't fully collapse: the .cpp
    // algorithm only seeds via abbrev-start when cur.count == 1, so
    // "U.S.A." flushes as "U.S." + "A" + ".".
    tokenize("U.S.A. is", 9, &out);
    assert(out.count == 4);
    check_tok(&out.data[0], "U.S.", true,  false);
    check_tok(&out.data[1], "A",    true,  true);
    check_tok(&out.data[2], ".",    false, false);
    check_tok(&out.data[3], "is",   true,  true);
    // Lone period after a multi-char word goes to a punct token.
    tokenize("hi.", 3, &out);
    assert(out.count == 2);
    check_tok(&out.data[0], "hi", true,  false);
    check_tok(&out.data[1], ".",  false, false);
    // Hyphenated word stays together when surrounded by letters.
    tokenize("well-known", 10, &out);
    assert(out.count == 1);
    check_tok(&out.data[0], "well-known", true, false);
    // Apostrophe inside word.
    tokenize("don't go", 8, &out);
    assert(out.count == 2);
    check_tok(&out.data[0], "don't", true, false);
    check_tok(&out.data[1], "go",    true, true);
    // Digits glue into the current word.
    tokenize("a1b2", 4, &out);
    assert(out.count == 1);
    check_tok(&out.data[0], "a1b2", true, false);
    // UTF-8 multi-byte (em-dash U+2014 = E2 80 94, 3 bytes) is
    // accumulated into the current word.
    tokenize("a\xe2\x80\x94" "b", 5, &out);
    assert(out.count == 1);
    check_tok(&out.data[0], "a\xe2\x80\x94" "b", true, false);
    tokens_clear(&out);
    assert(out.count == 0 && out.data == NULL);
}

static void test_suffix_decision_helpers(void) {
    // determine_alt_flags: explicit > 0 short-circuits the state lookup.
    struct phonemizer p = {0};
    phonemizer_state_init(&p);
    assert(determine_alt_flags(&p, "any", 3, 7) == 7);  // explicit wins
    assert(determine_alt_flags(&p, "any", 3, 0) == 0);  // explicit wins even when 0
    assert(determine_alt_flags(&p, "missing", 7, -1) == 0); // lookup miss
    // Seed word_alt_flags["compact"] = 4 (bit-mask, e.g. $alt3).
    struct chars k = {0};
    chars_puts(&k, "compact");
    int v = 4;
    map_put(&p.word_alt_flags, &k, &v);
    chars_free(&k);
    assert(determine_alt_flags(&p, "compact", 7, -1) == 4);
    // Case-folding: "COMPACT" lowercases to "compact".
    assert(determine_alt_flags(&p, "COMPACT", 7, -1) == 4);
    phonemizer_state_free(&p);

    // devoice_ed_suffix: "d#" + stem ending in unvoiced -> "t";
    // "d#" + stem ending in t/d -> "I#d"; "d#" + voiced/vowel -> unchanged.
    struct chars sfx = {0};
    chars_puts(&sfx, "d#");
    devoice_ed_suffix("kik", 3, &sfx);     // ends in 'k' (unvoiced)
    assert(strcmp(sfx.data, "t") == 0);
    sfx.count = 0; sfx.data[0] = '\0';
    chars_puts(&sfx, "d#");
    devoice_ed_suffix("wAnt", 4, &sfx);    // ends in 't' (alveolar) -> "I#d"
    assert(strcmp(sfx.data, "I#d") == 0);
    sfx.count = 0; sfx.data[0] = '\0';
    chars_puts(&sfx, "d#");
    devoice_ed_suffix("plei", 4, &sfx);    // ends in vowel -> unchanged
    assert(strcmp(sfx.data, "d#") == 0);
    sfx.count = 0; sfx.data[0] = '\0';
    chars_puts(&sfx, "d#");
    devoice_ed_suffix("rVn", 3, &sfx);     // ends in 'n' (voiced) -> unchanged
    assert(strcmp(sfx.data, "d#") == 0);
    // Non-"d#" suffix: function is a no-op.
    sfx.count = 0; sfx.data[0] = '\0';
    chars_puts(&sfx, "z");
    devoice_ed_suffix("kik", 3, &sfx);
    assert(strcmp(sfx.data, "z") == 0);
    chars_free(&sfx);

    // should_use_magic_e_for_cvc_stem: tests cover the three OR-arms
    // (stressed_at_end, no_explicit_stress, last_vowel_is_full).
    // Empty base_ph -> true (caller's outer flow decides).
    assert(should_use_magic_e_for_cvc_stem("", 0, "I@", 2) == true);
    // Last vowel "@" is in weak_vowels, no explicit stress marker
    // means no_explicit_stress is TRUE -> still true.
    assert(should_use_magic_e_for_cvc_stem("k@t", 3, "I@", 2) == true);
    // Last vowel "@" weak AND there's an explicit stress earlier ->
    // last_vowel_is_full is FALSE, no_explicit_stress is FALSE, but
    // the stress is on 'a' (vstart>0 and bp[vstart-1]=='\''), so
    // stressed_at_end measures the LAST vowel, which is unstressed
    // -> all three arms FALSE -> result FALSE.
    assert(should_use_magic_e_for_cvc_stem("'ak@t", 5, "I@", 2) == false);
    // Full last vowel ("a" not in "I@") -> last_vowel_is_full TRUE.
    assert(should_use_magic_e_for_cvc_stem("'kAt", 4, "I@", 2) == true);

    // simplify_syllabic_l_for_ing: "@L" + vowel-then-l base -> "@l".
    struct chars sph = {0};
    chars_puts(&sph, "h@L");
    simplify_syllabic_l_for_ing("heal", 4, &sph);
    assert(strcmp(sph.data, "h@l") == 0);
    sph.count = 0; sph.data[0] = '\0';
    // "@L" + consonant-then-l base -> "l".
    chars_puts(&sph, "p@L");
    simplify_syllabic_l_for_ing("peopl", 5, &sph);
    assert(strcmp(sph.data, "pl") == 0);
    sph.count = 0; sph.data[0] = '\0';
    // 't' before 'l' is the exception -> unchanged.
    chars_puts(&sph, "b0t@L");
    simplify_syllabic_l_for_ing("bottl", 5, &sph);
    assert(strcmp(sph.data, "b0t@L") == 0);
    sph.count = 0; sph.data[0] = '\0';
    // "ngl" ending exception -> unchanged.
    chars_puts(&sph, "tIN@L");
    simplify_syllabic_l_for_ing("tingl", 5, &sph);
    assert(strcmp(sph.data, "tIN@L") == 0);
    chars_free(&sph);
}

static void test_parse_helpers(void) {
    // split_ws: trims leading/trailing, splits on runs of whitespace.
    struct charsv out = {0};
    split_ws("  hello world  ", 15, &out);
    assert(out.count == 2);
    assert(strcmp(out.data[0].data, "hello") == 0);
    assert(strcmp(out.data[1].data, "world") == 0);
    // Multiple internal whitespace + tabs.
    split_ws("a\t b\t\tc", 7, &out);
    assert(out.count == 3);
    assert(strcmp(out.data[0].data, "a") == 0);
    assert(strcmp(out.data[1].data, "b") == 0);
    assert(strcmp(out.data[2].data, "c") == 0);
    // Empty + all-whitespace -> empty result.
    split_ws("", 0, &out);
    assert(out.count == 0);
    split_ws("   \t\n", 5, &out);
    assert(out.count == 0);
    // Re-split into same buffer: charsv_clear at entry drops old.
    split_ws("foo bar", 7, &out);
    assert(out.count == 2);
    charsv_clear(&out);
    assert(out.count == 0 && out.data == NULL);

    // parse_leading_dialect: "?3 rest"  -> cond=3, negated=false.
    int cond = 99;
    bool neg = true;
    struct chars rem = {0};
    parse_leading_dialect("?3 a b", 6, &cond, &neg, &rem);
    assert(cond == 3 && neg == false);
    assert(strcmp(rem.data, "a b") == 0);
    // Negated: "?!3 rest" -> cond=3, negated=true.
    parse_leading_dialect("?!6 zzz", 7, &cond, &neg, &rem);
    assert(cond == 6 && neg == true);
    assert(strcmp(rem.data, "zzz") == 0);
    // No dialect prefix: cond=0, negated=false, remainder=line.
    parse_leading_dialect("plain", 5, &cond, &neg, &rem);
    assert(cond == 0 && neg == false);
    assert(strcmp(rem.data, "plain") == 0);
    // Malformed (no space) -> no match.
    parse_leading_dialect("?3", 2, &cond, &neg, &rem);
    assert(cond == 0 && neg == false);
    assert(strcmp(rem.data, "?3") == 0);
    // Non-digit cond: .cpp's std::stoi throws and is swallowed,
    // leaving *cond = 0 but the prefix IS still stripped. The .c
    // matches that post-condition.
    parse_leading_dialect("?ab xyz", 7, &cond, &neg, &rem);
    assert(cond == 0 && neg == false);
    assert(strcmp(rem.data, "xyz") == 0);
    chars_free(&rem);

    // strip_comment_and_trim: cuts at first "//", trims result.
    struct chars cleaned = {0};
    strip_comment_and_trim("hello // comment", 16, &cleaned);
    assert(strcmp(cleaned.data, "hello") == 0);
    strip_comment_and_trim("  no comment  ", 14, &cleaned);
    assert(strcmp(cleaned.data, "no comment") == 0);
    strip_comment_and_trim("// only comment", 15, &cleaned);
    assert(cleaned.count == 0);
    strip_comment_and_trim("", 0, &cleaned);
    assert(cleaned.count == 0);
    chars_free(&cleaned);
}

static void write_fixture(const char * path, const char * content) {
    FILE * f = fopen(path, "w");
    assert(f != NULL);
    fputs(content, f);
    fclose(f);
}

static void test_load_dictionary(void) {
    const char * path = "../tmp/phon_build/test_en_list.txt";
    // Minimal en_list with representative shapes. The dialect is
    // en-us; ?6 entries should apply, ?!3 should not, "?3" should
    // apply, "?!3" should not.
    write_fixture(path,
        "// a comment-only line\n"
        "hello h@'loU\n"
        "world w'3:ld\n"
        "the D@ $u\n"
        "elaborate I#lab3@t $noun\n"
        "construct k@n'strVkt $verb\n"
        "?6 organize '0g@naIz\n"        // en-us conditional
        "?!3 colour k'Vl3\n"             // skipped on en-us
        "U.S. ju:'es $abbrev\n"
        "(of course) @v'kO:s\n"          // phrase entry
        "made m,eId $u+\n"               // u_plus_secondary (then erased
                                         // since "made" -> unstressed
                                         // removed at end)
        "under Vnd3 $strend2\n"          // compound_prefix
        "construct $alt3\n"              // flag-only, erases dict
    );
    struct phonemizer p = {0};
    phonemizer_state_init(&p);
    chars_puts(&p.dialect, "en-us");
    bool ok = load_dictionary(&p, path);
    assert(ok);
    // Basic dict lookup.
    struct chars * v = smap_get(&p.dict, "hello", 5);
    assert(v != NULL && strcmp(v->data, "h@'loU") == 0);
    v = smap_get(&p.dict, "world", 5);
    assert(v != NULL && strcmp(v->data, "w'3:ld") == 0);
    // $u flag sets dict + adds to unstressed_words.
    v = smap_get(&p.dict, "the", 3);
    assert(v != NULL && strcmp(v->data, "D@") == 0);
    assert(set_has(&p.unstressed_words, "the", 3));
    // $noun routes to noun_dict, not dict.
    assert(smap_get(&p.dict, "elaborate", 9) == NULL);
    v = smap_get(&p.noun_dict, "elaborate", 9);
    assert(v != NULL && strcmp(v->data, "I#lab3@t") == 0);
    // $verb routes to verb_dict.
    v = smap_get(&p.verb_dict, "construct", 9);
    assert(v != NULL && strcmp(v->data, "k@n'strVkt") == 0);
    // ?6 applies on en-us.
    v = smap_get(&p.dict, "organize", 8);
    assert(v != NULL && strcmp(v->data, "'0g@naIz") == 0);
    // ?!3 skipped on en-us -> "colour" should be absent.
    assert(smap_get(&p.dict, "colour", 6) == NULL);
    // $abbrev adds to abbrev_words but no dict entry (flag-only with
    // a real phoneme also stores the phoneme in dict).
    assert(set_has(&p.abbrev_words, "u.s.", 4));
    // Phrase entry -> phrase_dict. The "%" prefix is added only when
    // the phoneme has NO '\'' primary-stress marker. "@v'kO:s"
    // contains a "'" -> stored as-is, no "%" prepended.
    v = smap_get(&p.phrase_dict, "of course", 9);
    assert(v != NULL && strcmp(v->data, "@v'kO:s") == 0);
    // "made" was added to unstressed_words by $u+ but post-load
    // erased -- the loader's special-case.
    assert(!set_has(&p.unstressed_words, "made", 4));
    // u_plus_secondary_words: "made" had ',' but no '\''.
    assert(set_has(&p.u_plus_secondary_words, "made", 4));
    // Compound prefix.
    bool found_under = false;
    for (size_t i = 0; i < p.compound_prefixes.count; i++) {
        const struct strpair * sp = &p.compound_prefixes.data[i];
        if (sp->a.count == 5 && memcmp(sp->a.data, "under", 5) == 0
            && sp->b.count == 4 && memcmp(sp->b.data, "Vnd3", 4) == 0) {
            found_under = true;
        }
    }
    assert(found_under);
    assert(set_has(&p.strend_words, "under", 5));
    // $alt3 (flag-only) erased "construct" from dict. The $verb
    // entry stored it in verb_dict, so that survives; dict has no
    // construct entry.
    assert(smap_get(&p.dict, "construct", 9) == NULL);
    int * af = imap_get(&p.word_alt_flags, "construct", 9);
    assert(af != NULL && (*af & (1 << 2)) != 0);  // alt3 bit
    phonemizer_state_free(&p);
}

static void test_load_rules(void) {
    const char * path = "../tmp/phon_build/test_en_rules.txt";
    // Real en_rules format: tokens are space-separated. Left context
    // is a token ending in ')'. Right context is a token starting
    // with '('. So "Z) a ae" means left_ctx="Z", match="a", phon="ae".
    // "a (P prefix" means match="a", right_ctx="P", is_prefix=true.
    write_fixture(path,
        "// rules header comment\n"
        ".L01 ab cd ef // L-group with comment\n"
        ".replace\n"
        "foo bar\n"
        "x y\n"
        ".group a\n"
        "       a              a\n"
        "    Z) a              ae\n"
        "       a (P           prefix\n"
        "       a (S2          suffix\n"
        "?3     a              en_us_a\n"
        "?!3    a              not_en_us\n"
        "       a              $verb\n"
        ".group b\n"
        "       b              b\n"
    );
    struct phonemizer p = {0};
    phonemizer_state_init(&p);
    chars_puts(&p.dialect, "en-us");
    bool ok = load_rules(&p, path);
    assert(ok);

    // L-group: "ab", "cd", "ef" (the "//" item is dropped).
    struct charsv * l1 = &p.rules.groups.lgroups[1];
    assert(l1->count == 3);
    assert(l1->data[0].count == 2 && memcmp(l1->data[0].data, "ab", 2) == 0);
    assert(l1->data[1].count == 2 && memcmp(l1->data[1].data, "cd", 2) == 0);
    assert(l1->data[2].count == 2 && memcmp(l1->data[2].data, "ef", 2) == 0);

    // Replacement section.
    assert(p.rules.replacements.count == 2);
    assert(strcmp(p.rules.replacements.data[0].from.data, "foo") == 0);
    assert(strcmp(p.rules.replacements.data[0].to.data,   "bar") == 0);
    assert(strcmp(p.rules.replacements.data[1].from.data, "x") == 0);
    assert(strcmp(p.rules.replacements.data[1].to.data,   "y") == 0);

    // Group "a": plain rule + left-ctx + prefix + suffix +
    // en_us-conditional. Total 5 (the ?!3 and $verb are skipped).
    struct chars va = chars_view("a", 1);
    struct rules * ra = map_get(&p.rules.rule_groups, &va);
    assert(ra != NULL);
    assert(ra->count == 5);
    // Rule 0: plain "a -> a"
    assert(ra->data[0].left_ctx.count == 0);
    assert(strcmp(ra->data[0].match.data, "a") == 0);
    assert(strcmp(ra->data[0].phonemes.data, "a") == 0);
    assert(!ra->data[0].is_prefix && !ra->data[0].is_suffix);
    // Rule 1: "(Z)a -> ae" (left context "Z")
    assert(strcmp(ra->data[1].left_ctx.data, "Z") == 0);
    assert(strcmp(ra->data[1].match.data, "a") == 0);
    assert(strcmp(ra->data[1].phonemes.data, "ae") == 0);
    // Rule 2: "a(P) -> prefix" -> is_prefix true
    assert(ra->data[2].is_prefix);
    // Rule 3: "a(S2) -> suffix" -> is_suffix true, strip=2
    assert(ra->data[3].is_suffix);
    assert(ra->data[3].suffix_strip_len == 2);
    // Rule 4: ?3 en-us conditional fires on en-us.
    assert(ra->data[4].condition == 3);
    assert(strcmp(ra->data[4].phonemes.data, "en_us_a") == 0);

    // Group "b" should have 1 rule.
    struct chars vb = chars_view("b", 1);
    struct rules * rb = map_get(&p.rules.rule_groups, &vb);
    assert(rb != NULL && rb->count == 1);
    assert(strcmp(rb->data[0].phonemes.data, "b") == 0);

    // Letter groups (SetLetterBits).
    assert(p.rules.groups.groupA['a'] && p.rules.groups.groupA['e']);
    assert(!p.rules.groups.groupA['b'] && !p.rules.groups.groupA['y']);
    assert(p.rules.groups.groupY['y']);   // English includes 'y'
    assert(p.rules.groups.groupC['b'] && !p.rules.groups.groupC['a']);

    phonemizer_state_free(&p);
}

static void test_rule_core_leaves(void) {
    // apply_replacements: equal / shorter / longer / multiple rules.
    struct replaces reps = {0};
    struct replace_rule r = {0};
    // "foo" -> "bar" (equal length)
    chars_puts(&r.from, "foo"); chars_puts(&r.to, "bar");
    replaces_put(&reps, r); r = (struct replace_rule){0};
    // "abc" -> "X" (shorter)
    chars_puts(&r.from, "abc"); chars_puts(&r.to, "X");
    replaces_put(&reps, r); r = (struct replace_rule){0};
    // "z" -> "ZZZ" (longer)
    chars_puts(&r.from, "z"); chars_puts(&r.to, "ZZZ");
    replaces_put(&reps, r); r = (struct replace_rule){0};

    struct chars buf = {0};
    chars_puts(&buf, "foo abc z");
    apply_replacements(&buf, &reps);
    assert(strcmp(buf.data, "bar X ZZZ") == 0);
    // Multiple consecutive hits.
    buf.count = 0; buf.data[0] = '\0';
    chars_puts(&buf, "abcabc");
    apply_replacements(&buf, &reps);
    assert(strcmp(buf.data, "XX") == 0);
    // No matches: buffer unchanged.
    buf.count = 0; buf.data[0] = '\0';
    chars_puts(&buf, "qqq");
    apply_replacements(&buf, &reps);
    assert(strcmp(buf.data, "qqq") == 0);
    chars_free(&buf);
    replaces_clear(&reps);

    // match_lgroup_at: longest-match across items, case-insensitive.
    struct charsv lg = {0};
    struct chars t = {0};
    chars_puts(&t, "ab"); charsv_put(&lg, t); t = (struct chars){0};
    chars_puts(&t, "abc"); charsv_put(&lg, t); t = (struct chars){0};
    chars_puts(&t, "x");  charsv_put(&lg, t); t = (struct chars){0};
    assert(match_lgroup_at(&lg, "abcdef", 6, 0) == 3);  // "abc" wins
    assert(match_lgroup_at(&lg, "ABCdef", 6, 0) == 3);  // case-fold
    assert(match_lgroup_at(&lg, "abdef",  5, 0) == 2);  // "ab"
    assert(match_lgroup_at(&lg, "xy",     2, 0) == 1);  // "x"
    assert(match_lgroup_at(&lg, "qqq",    3, 0) == 0);  // no match
    assert(match_lgroup_at(&lg, "abc",    3, 5) == 0);  // pos OOB
    charsv_clear(&lg);
}

static void test_left_context_score(void) {
    struct ruleset rs = {0};
    ruleset_init(&rs);
    // Seed L01 for the L-group test below.
    {
        struct chars t = {0};
        chars_puts(&t, "ab");
        charsv_put(&rs.groups.lgroups[1], t);
    }
    // Literal match: ctx "a", word "abc", pos=1 (matching the 'b').
    // word_pos starts at pos-1 = 0, ctx is "a" -> word[0]='a'.
    // distance_left -> 0, score = 21 - 0 = 21.
    struct ctx_score s = match_left_context_score("a", 1, "abc", 3, 1,
                                                  &rs, NULL, 0);
    assert(s.matched && s.score == 21);
    // Literal mismatch.
    s = match_left_context_score("z", 1, "abc", 3, 1, &rs, NULL, 0);
    assert(!s.matched && s.score == 0);
    // Word-boundary '_': ctx "_", pos=0 -> word_pos=-1 -> matches,
    // score=+4.
    s = match_left_context_score("_", 1, "abc", 3, 0, &rs, NULL, 0);
    assert(s.matched && s.score == 4);
    // Word-boundary fails when word_pos >= 0.
    s = match_left_context_score("_", 1, "abc", 3, 1, &rs, NULL, 0);
    assert(!s.matched);
    // INC_SCORE: ctx "+", any pos. Score +20.
    s = match_left_context_score("+", 1, "abc", 3, 1, &rs, NULL, 0);
    assert(s.matched && s.score == 20);
    // DEC_SCORE: ctx "<". Score -20 (still matched).
    s = match_left_context_score("<", 1, "abc", 3, 1, &rs, NULL, 0);
    assert(s.matched && s.score == -20);
    // Letter group A (vowel): ctx "A", word "abc", pos=1.
    // word[0]='a' is in groupA -> matched, score=20-0=20.
    s = match_left_context_score("A", 1, "abc", 3, 1, &rs, NULL, 0);
    assert(s.matched && s.score == 20);
    // Letter group C (consonant): "C" at pos=2 of "abc",
    // word_pos=1, word[1]='b' is in groupC -> 19-0=19.
    s = match_left_context_score("C", 1, "abc", 3, 2, &rs, NULL, 0);
    assert(s.matched && s.score == 19);
    // K (non-vowel): word_pos<0 -> fail.
    s = match_left_context_score("K", 1, "abc", 3, 0, &rs, NULL, 0);
    assert(!s.matched);
    // K matching consonant: pos=2, word[1]='b' is not vowel ->
    // matched, score=20-0=20.
    s = match_left_context_score("K", 1, "abc", 3, 2, &rs, NULL, 0);
    assert(s.matched && s.score == 20);
    // X (no vowels in prefix): "X" at pos=2 of "bbc" -> no vowels
    // in word[0..1] -> matched, score=3.
    s = match_left_context_score("X", 1, "bbc", 3, 2, &rs, NULL, 0);
    assert(s.matched && s.score == 3);
    // X fails when vowel present.
    s = match_left_context_score("X", 1, "abc", 3, 2, &rs, NULL, 0);
    assert(!s.matched);
    // D (digit): pos=2 of "1bc", word[1]='b' is not digit -> fail.
    s = match_left_context_score("D", 1, "1bc", 3, 2, &rs, NULL, 0);
    assert(!s.matched);
    s = match_left_context_score("D", 1, "1bc", 3, 1, &rs, NULL, 0);
    assert(s.matched && s.score == 21);
    // L-group ref "L01": pos=2 of "ab", word_pos=1, but "ab" needs
    // pos+ilen<=wn so we need pos that allows matching. With pos=2
    // we look BACK from word_pos=1; the lgroup match attempts at
    // word_pos=1 with item "ab" (count 2): pos+ilen=1+2=3 > wn=2,
    // skipped; with item "ab" no... Hmm match_lgroup_at takes
    // pos directly; let's use a word with the L-group at word_pos.
    s = match_left_context_score("L01", 3, "ab", 2, 2, &rs, NULL, 0);
    // word_pos starts at 1. matchLGroupAt at pos=1 with item "ab"
    // count 2 -- pos+ilen=3 > wn=2, skipped. So matched=0 -> fail.
    assert(!s.matched);
    // 'E' (REPLACED_E) always fails per the C++ comment.
    s = match_left_context_score("E", 1, "abc", 3, 2, &rs, NULL, 0);
    assert(!s.matched);
    ruleset_free(&rs);
}

static void test_right_context_score(void) {
    struct ruleset rs = {0};
    ruleset_init(&rs);
    // Literal at right: ctx "b", word "ab", pos=1 -> word[1]='b'
    // distance_right = 0 after first consume, score = 21 - 0 = 21.
    struct right_ctx_score s = match_right_context_score(
        "b", 1, "ab", 2, 1, &rs, 'a', 0, NULL, 0, false);
    assert(s.matched && s.score == 21);
    // Word-end '_': ctx "_" at pos=2 (end of "ab") matches.
    s = match_right_context_score("_", 1, "ab", 2, 2, &rs, 0, 0,
                                  NULL, 0, false);
    assert(s.matched && s.score == 21);
    s = match_right_context_score("_", 1, "ab", 2, 1, &rs, 0, 0,
                                  NULL, 0, false);
    assert(!s.matched);
    // INC/DEC score.
    s = match_right_context_score("+", 1, "ab", 2, 1, &rs, 0, 0,
                                  NULL, 0, false);
    assert(s.matched && s.score == 20);
    s = match_right_context_score("<", 1, "ab", 2, 1, &rs, 0, 0,
                                  NULL, 0, false);
    assert(s.matched && s.score == -20);
    // K matches non-vowel (or end-of-word null).
    s = match_right_context_score("K", 1, "abx", 3, 2, &rs, 0, 0,
                                  NULL, 0, false);
    assert(s.matched && s.score == 20);  // 'x' is non-vowel
    s = match_right_context_score("K", 1, "ab", 2, 2, &rs, 0, 0,
                                  NULL, 0, false);
    assert(s.matched && s.score == 20);  // word-end as null
    s = match_right_context_score("K", 1, "abe", 3, 2, &rs, 0, 0,
                                  NULL, 0, false);
    assert(!s.matched);                  // 'e' is vowel
    // RULE_STRESSED '&' in right context always fails.
    s = match_right_context_score("&", 1, "ab", 2, 1, &rs, 0, 0,
                                  NULL, 0, false);
    assert(!s.matched);
    // RULE_DEL_FWD '#': finds 'e' between pos and word_pos.
    // ctx "be#" at pos=0 of "abe": consumes 'a' first (mismatch
    // with 'b'), so this should fail. Use "#" alone (no consume).
    s = match_right_context_score("#", 1, "abe", 3, 3, &rs, 0, 0,
                                  NULL, 0, false);
    // del_fwd searches [pos=3, word_pos=3) -> empty -> no del.
    assert(s.matched && s.del_fwd_count == 0);
    ruleset_free(&rs);
}

static void test_match_rule_and_find_best(void) {
    // Build a fixture-loaded ruleset to drive find_best_rule.
    const char * path = "../tmp/phon_build/test_en_rules2.txt";
    write_fixture(path,
        ".group a\n"
        "        a              a\n"
        "        a (_           A_\n"            // higher score at word-end
        ".group b\n"
        "        b              b\n"
    );
    struct phonemizer p = {0};
    phonemizer_state_init(&p);
    chars_puts(&p.dialect, "en-us");
    bool ok = load_rules(&p, path);
    assert(ok);
    // match_rule on the simple "a -> a" rule.
    struct chars va = chars_view("a", 1);
    struct rules * ra = map_get(&p.rules.rule_groups, &va);
    assert(ra != NULL && ra->count == 2);
    struct chars phon = {0};
    int adv = 0, dfs = -1, dfc = 0;
    int sc = match_rule(&p.rules, &ra->data[0], "abc", 3, 0,
                        &phon, &adv, &dfs, &dfc, 1, NULL, 0, 0,
                        NULL, 0, false);
    assert(sc >= 0);
    assert(strcmp(phon.data, "a") == 0);
    assert(adv == 1);
    chars_free(&phon);
    // find_best_rule at pos=0 of "abc" should match "a -> a"
    // (the "_" right-context rule wants word-end which fails).
    struct rule_match best = find_best_rule(
        &p.rules, "abc", 3, 0, 3, 'a', 0, NULL, 0,
        true, false, false, NULL, 0);
    assert(best.score >= 0);
    assert(strcmp(best.phonemes.data, "a") == 0);
    assert(best.advance == 1);
    rule_match_free(&best);
    // find_best_rule at pos=0 of "a" (single-char) should prefer
    // the "_-ended" rule (higher score at word end).
    best = find_best_rule(&p.rules, "a", 1, 0, 1, 'a', 0,
                          NULL, 0, true, false, false, NULL, 0);
    assert(best.score >= 0);
    assert(strcmp(best.phonemes.data, "A_") == 0);
    rule_match_free(&best);
    phonemizer_state_free(&p);
}

static void test_stress_helpers(void) {
    // find_last_stressable_vowel: simple vowel.
    assert(find_last_stressable_vowel("kat", 3) == 1);  // 'a' at 1
    // Diphthong eI: should point at 'e', not 'I'.
    assert(find_last_stressable_vowel("seI", 3) == 1);  // 'e' at 1
    // aU diphthong: same.
    assert(find_last_stressable_vowel("aU", 2) == 0);
    // Reduced vowel marker '2' or '#' immediately after skips it.
    assert(find_last_stressable_vowel("kI2", 3) == -1);
    // Marked unstressed by '%': skipped.
    assert(find_last_stressable_vowel("%a", 2) == -1);
    // Rule-boundary '\x01' is invisible.
    assert(find_last_stressable_vowel("a\x01" "b", 3) == 0);
    // Empty input.
    assert(find_last_stressable_vowel("", 0) == -1);

    // apply_stress_prev: emit "=foo" + phonemes "ka" -> emit "foo",
    // phonemes "k'a".
    struct chars emit = {0};
    chars_puts(&emit, "=foo");
    struct chars ph = {0};
    chars_puts(&ph, "ka");
    apply_stress_prev(&emit, &ph);
    assert(strcmp(emit.data, "foo") == 0);
    assert(strcmp(ph.data, "k'a") == 0);
    chars_free(&emit);
    chars_free(&ph);
    // No leading '=' on emit -> no change.
    struct chars emit2 = {0};
    chars_puts(&emit2, "foo");
    struct chars ph2 = {0};
    chars_puts(&ph2, "ka");
    apply_stress_prev(&emit2, &ph2);
    assert(strcmp(emit2.data, "foo") == 0);
    assert(strcmp(ph2.data, "ka") == 0);
    chars_free(&emit2);
    chars_free(&ph2);
    // Existing '\'' before stressable vowel demoted to '\x02'.
    struct chars emit3 = {0};
    chars_puts(&emit3, "=x");
    struct chars ph3 = {0};
    chars_puts(&ph3, "'ka");          // 'a' is stressable, 'k pre
    apply_stress_prev(&emit3, &ph3);
    // The 'k' between '\'' and 'a' protects 'a' as the search
    // target; before_vowel is 'k' (non-stress-marker), so '
    // is inserted before 'a'. Earlier '\'' becomes '\x02'.
    assert(ph3.count >= 4);
    assert(ph3.data[0] == '\x02');     // demoted
    chars_free(&emit3);
    chars_free(&ph3);
}

static void test_apply_rules_end_to_end(void) {
    const char * path = "../tmp/phon_build/test_en_rules3.txt";
    write_fixture(path,
        ".group a\n"
        "        a              a\n"
        ".group b\n"
        "        b              b\n"
        ".group c\n"
        "        c              k\n"
        ".group t\n"
        "        t              t\n"
        ".group d\n"
        "        d              d\n"
        ".group o\n"
        "        o              0\n"
        ".group g\n"
        "        g              g\n"
    );
    struct phonemizer p = {0};
    phonemizer_state_init(&p);
    chars_puts(&p.dialect, "en-us");
    bool ok = load_rules(&p, path);
    assert(ok);
    // Phonemize "cat": c->k, a->a, t->t. Each rule emits its
    // phoneme + a '\x01' rule-boundary marker.
    struct chars out = {0};
    apply_rules(&p, "cat", 3, true, -1, false, false,
                NULL, 0, NULL, 0, &out);
    // Expected raw: "k\x01" "a\x01" "t\x01" (3 codes + 3 boundaries).
    assert(out.count == 6);
    assert(out.data[0] == 'k' && out.data[1] == '\x01');
    assert(out.data[2] == 'a' && out.data[3] == '\x01');
    assert(out.data[4] == 't' && out.data[5] == '\x01');
    // Phonemize "dog": d->d, o->0, g->g.
    out.count = 0;
    apply_rules(&p, "dog", 3, true, -1, false, false,
                NULL, 0, NULL, 0, &out);
    assert(out.count == 6);
    assert(out.data[0] == 'd' && out.data[2] == '0'
           && out.data[4] == 'g');
    // Phonemize with no rule for the position: silent skip.
    out.count = 0;
    apply_rules(&p, "xyz", 3, true, -1, false, false,
                NULL, 0, NULL, 0, &out);
    // No rules for x/y/z -> all chars skipped -> empty output.
    assert(out.count == 0);
    // word_to_phonemes routes through apply_rules +
    // process_phoneme_string. The full prosody pipeline strips
    // '\x01' rule boundaries and may insert stress markers; just
    // check that 'k', 'a', 't' all appear in the output.
    out.count = 0;
    word_to_phonemes(&p, "cat", 3, &out);
    assert(out.count > 0);
    assert(memchr(out.data, 'k', out.count) != NULL);
    assert(memchr(out.data, 'a', out.count) != NULL);
    assert(memchr(out.data, 't', out.count) != NULL);
    chars_free(&out);
    phonemizer_state_free(&p);
}

static void test_word_to_phonemes_real(void) {
    // Real dict + rules from the bundled nano model.
    const char * list_path = "../app/Resources/nano/en_list";
    const char * rules_path = "../app/Resources/nano/en_rules";
    struct phonemizer p = {0};
    phonemizer_state_init(&p);
    chars_puts(&p.dialect, "en-us");
    bool ok = load_dictionary(&p, list_path);
    assert(ok);
    ok = load_rules(&p, rules_path);
    assert(ok);
    // "hello" should produce a non-empty phoneme string (dict hit).
    struct chars out = {0};
    word_to_phonemes(&p, "hello", 5, &out);
    assert(out.count > 0);
    fprintf(stderr, "[real] hello -> %s\n", out.data);
    // A few more probes -- check we get non-empty output (the
    // actual phoneme string match against .cpp comes via the
    // comparison test in a later phase).
    out.count = 0; if (out.data) out.data[0] = '\0';
    word_to_phonemes(&p, "world", 5, &out);
    assert(out.count > 0);
    fprintf(stderr, "[real] world -> %s\n", out.data);
    out.count = 0; if (out.data) out.data[0] = '\0';
    word_to_phonemes(&p, "cat", 3, &out);
    assert(out.count > 0);
    fprintf(stderr, "[real] cat -> %s\n", out.data);
    out.count = 0; if (out.data) out.data[0] = '\0';
    // Out-of-dict word: should still produce output via rules.
    word_to_phonemes(&p, "blorp", 5, &out);
    assert(out.count > 0);
    fprintf(stderr, "[real] blorp -> %s\n", out.data);
    // Morphological suffix arms (D-25). These exercise -ing, -ed,
    // -ies, -s, -arily handlers via word_to_phonemes' dispatch.
    out.count = 0; if (out.data) out.data[0] = '\0';
    word_to_phonemes(&p, "walking", 7, &out);
    assert(out.count > 0);
    fprintf(stderr, "[real] walking -> %s\n", out.data);
    out.count = 0; if (out.data) out.data[0] = '\0';
    word_to_phonemes(&p, "walked", 6, &out);
    assert(out.count > 0);
    fprintf(stderr, "[real] walked -> %s\n", out.data);
    out.count = 0; if (out.data) out.data[0] = '\0';
    word_to_phonemes(&p, "studies", 7, &out);
    assert(out.count > 0);
    fprintf(stderr, "[real] studies -> %s\n", out.data);
    out.count = 0; if (out.data) out.data[0] = '\0';
    word_to_phonemes(&p, "cats", 4, &out);
    assert(out.count > 0);
    fprintf(stderr, "[real] cats -> %s\n", out.data);
    out.count = 0; if (out.data) out.data[0] = '\0';
    word_to_phonemes(&p, "necessarily", 11, &out);
    assert(out.count > 0);
    fprintf(stderr, "[real] necessarily -> %s\n", out.data);
    // Phoneme codes -> IPA UTF-8 (D-26a smoke test). Build the IPA
    // overrides from dialect, then render a handful of words.
    build_ipa_overrides(&p);
    static const char * const sample_words[] = {
        "hello", "world", "cat", "walking", "studies"
    };
    struct chars ipa = {0};
    for (int wi = 0; wi < 5; wi++) {
        out.count = 0; if (out.data) out.data[0] = '\0';
        word_to_phonemes(&p, sample_words[wi],
                         strlen(sample_words[wi]), &out);
        ipa.count = 0;
        phonemes_to_ipa(&p, out.data, out.count, &ipa);
        if (ipa.data) {
            fprintf(stderr, "[ipa]  %-10s -> %s\n",
                    sample_words[wi], ipa.data);
        }
        assert(ipa.count > 0);
    }
    chars_free(&ipa);
    chars_free(&out);
    phonemizer_state_free(&p);
}

static void test_prosody_steps(void) {
    // Velar nasal: n+k/g -> N+k/g.
    struct chars ph = {0};
    chars_puts(&ph, "ankgxn");
    apply_velar_nasal_assimilation(&ph);
    assert(strcmp(ph.data, "aNkgxn") == 0);   // "Ng" stays as is
    ph.count = 0; ph.data[0] = '\0';
    chars_puts(&ph, "inkmen");
    apply_velar_nasal_assimilation(&ph);
    assert(strcmp(ph.data, "iNkmen") == 0);

    // Happy tensing: word-final %I -> i.
    ph.count = 0; if (ph.data) ph.data[0] = '\0';
    chars_puts(&ph, "h@p%I");
    apply_happy_tensing(&ph);
    assert(strcmp(ph.data, "h@pi") == 0);
    // I2 -> i.
    ph.count = 0; if (ph.data) ph.data[0] = '\0';
    chars_puts(&ph, "h@pI2");
    apply_happy_tensing(&ph);
    assert(strcmp(ph.data, "h@pi") == 0);
    // Bare I at word end with another vowel before -> i.
    // The all-vowels list for this check EXCLUDES '@' (schwa);
    // multi-syllabic test needs another non-schwa vowel.
    ph.count = 0; if (ph.data) ph.data[0] = '\0';
    chars_puts(&ph, "hapI");
    apply_happy_tensing(&ph);
    assert(strcmp(ph.data, "hapi") == 0);
    // Monosyllabic ("only one non-schwa vowel"): bare I stays.
    ph.count = 0; if (ph.data) ph.data[0] = '\0';
    chars_puts(&ph, "h@pI");
    apply_happy_tensing(&ph);
    assert(strcmp(ph.data, "h@pI") == 0);
    // Stress-marker before I -> unchanged.
    ph.count = 0; if (ph.data) ph.data[0] = '\0';
    chars_puts(&ph, "h'I");
    apply_happy_tensing(&ph);
    assert(strcmp(ph.data, "h'I") == 0);
    // Diphthong eI -> unchanged.
    ph.count = 0; if (ph.data) ph.data[0] = '\0';
    chars_puts(&ph, "heI");
    apply_happy_tensing(&ph);
    assert(strcmp(ph.data, "heI") == 0);

    // Vowel reduction: %A: -> %@.
    ph.count = 0; if (ph.data) ph.data[0] = '\0';
    chars_puts(&ph, "k%A:nt");
    apply_vowel_reduction(&ph);
    assert(strcmp(ph.data, "k%@nt") == 0);
    // %A -> %@ (but %A@ is protected).
    ph.count = 0; if (ph.data) ph.data[0] = '\0';
    chars_puts(&ph, "k%A@nt");
    apply_vowel_reduction(&ph);
    assert(strcmp(ph.data, "k%A@nt") == 0);
    ph.count = 0; if (ph.data) ph.data[0] = '\0';
    chars_puts(&ph, "k%Ant");
    apply_vowel_reduction(&ph);
    assert(strcmp(ph.data, "k%@nt") == 0);

    // LOT+R merge: 0r -> O:r.
    ph.count = 0; if (ph.data) ph.data[0] = '\0';
    chars_puts(&ph, "fo0rst");
    apply_lot_plus_r_merge(&ph);
    assert(strcmp(ph.data, "foO:rst") == 0);
    // Multiple occurrences.
    ph.count = 0; if (ph.data) ph.data[0] = '\0';
    chars_puts(&ph, "0r0r");
    apply_lot_plus_r_merge(&ph);
    assert(strcmp(ph.data, "O:rO:r") == 0);

    // Strip morpheme-schwa-r: @-r -> r.
    ph.count = 0; if (ph.data) ph.data[0] = '\0';
    chars_puts(&ph, "av@-rIdZ");
    strip_morpheme_schwa_r(&ph);
    assert(strcmp(ph.data, "avrIdZ") == 0);

    // apply_bare_schwa_to_rhotic: a#r pre-pass + @r -> 3 absorb.
    ph.count = 0; if (ph.data) ph.data[0] = '\0';
    chars_puts(&ph, "a#raUnd");
    apply_bare_schwa_to_rhotic(&ph);
    // a#r -> @r -> @r at end of word (next is 'a' which IS a
    // vowel) -> NOT absorbed (no preceding %/= either).
    // Result: @raUnd with @ -> 3, r kept because next is vowel.
    assert(strcmp(ph.data, "3raUnd") == 0);
    // @r at end of word (no vowel after) -> absorbed.
    ph.count = 0; if (ph.data) ph.data[0] = '\0';
    chars_puts(&ph, "h@r");
    apply_bare_schwa_to_rhotic(&ph);
    assert(strcmp(ph.data, "h3") == 0);  // r absorbed
    // Diphthong protection: 'or' (or-pre-r) stays.
    ph.count = 0; if (ph.data) ph.data[0] = '\0';
    chars_puts(&ph, "ho@r");          // o-@-r where @ follows o
    apply_bare_schwa_to_rhotic(&ph);
    assert(strcmp(ph.data, "ho@r") == 0);  // is_diphthong protects

    // apply_linking_r: 3 + vowel -> insert r.
    ph.count = 0; if (ph.data) ph.data[0] = '\0';
    chars_puts(&ph, "f3a");
    apply_linking_r(&ph);
    assert(strcmp(ph.data, "f3ra") == 0);
    // 3 already has r -> no change.
    ph.count = 0; if (ph.data) ph.data[0] = '\0';
    chars_puts(&ph, "f3ra");
    apply_linking_r(&ph);
    assert(strcmp(ph.data, "f3ra") == 0);
    // 3 + non-vowel -> no change.
    ph.count = 0; if (ph.data) ph.data[0] = '\0';
    chars_puts(&ph, "f3t");
    apply_linking_r(&ph);
    assert(strcmp(ph.data, "f3t") == 0);
    chars_free(&ph);
}

static void test_public_api_end_to_end(void) {
    const char * list_path = "../app/Resources/nano/en_list";
    const char * rules_path = "../app/Resources/nano/en_rules";
    PhonemizerHandle h = phonemizer_create(rules_path, list_path,
                                           "en-us");
    assert(h != NULL);
    static const char * const sentences[] = {
        "hello world",
        "the cat",
        "I have walked 10 miles.",
        "U.S.A.",
        NULL
    };
    for (int i = 0; sentences[i] != NULL; i++) {
        char * ipa = phonemizer_phonemize(h, sentences[i]);
        assert(ipa != NULL);
        fprintf(stderr, "[api]  %-25s -> %s\n", sentences[i], ipa);
        phonemizer_free_string(ipa);
    }
    phonemizer_destroy(h);
}

int main(int argc, char ** argv) {
    (void)argc; (void)argv;
    test_leaf_helpers();
    test_phoneme_utils();
    test_state_init_and_roundtrip();
    test_tokenize();
    test_suffix_decision_helpers();
    test_parse_helpers();
    test_load_dictionary();
    test_load_rules();
    test_rule_core_leaves();
    test_left_context_score();
    test_right_context_score();
    test_match_rule_and_find_best();
    test_stress_helpers();
    test_apply_rules_end_to_end();
    test_prosody_steps();
    test_word_to_phonemes_real();
    test_public_api_end_to_end();
    printf("phonemizer all + word_to_phonemes + prosody + api: ALL PASS\n");
    return 0;
}

#endif /* PHONEMIZER_TESTS */

#endif /* PHONEMIZER_C */
