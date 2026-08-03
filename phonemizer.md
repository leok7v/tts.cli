# phonemizer.c

Read this next to the source. It explains what the file is, how it is
layered, and what each layer owes the one above it. It does not repeat
what the code already says clearly; it says the things that are true
across files and functions and therefore live nowhere in particular.

`src/phonemizer.c` is a single translation unit, about 12k lines, no
floating point, no dependencies beyond libc. It turns English text into
an IPA string. It is a C99 port of a C++ original; the original is
archived and is not worth reading.

## 1. The contract

Five functions, declared in `bridge.h`, defined at the very bottom of
the file:

```c
PhonemizerHandle phonemizer_create(rules_path, list_path, dialect);
char *           phonemizer_phonemize(handle, text);
void             phonemizer_free_string(char *);
void             phonemizer_destroy(handle);
const char *     phonemizer_get_error(handle);
```

`phonemizer_create` reads two data files and returns an opaque handle,
or NULL. `phonemizer_phonemize` returns a malloc'd, NUL-terminated
UTF-8 string that the caller frees with `phonemizer_free_string`. That
is the whole surface. Everything else in the file is `static`.

Two things to know before you trust the API:

The handle is not thread-safe for concurrent `phonemize` calls. Nothing
inside mutates the loaded dictionaries, but the code was never audited
for it and there is no lock. One handle per thread.

`phonemizer_get_error` is nearly unreachable. When a load fails,
`phonemizer_create` writes the message into `p->err`, then frees `p` and
returns NULL, taking the message with it. Passing NULL back in returns
the literal `"null handle"`. If you need the real reason a load failed,
you have to break the create path open. This is deliberate in the sense
that nobody has needed it, not in the sense that it is good.

## 2. The two data files

Both are espeak-ng format, both are read once at create time, neither is
ever written.

### en_list, the dictionary

One entry per line:

```
word  phonemes  [flags...]
```

Tokens are whitespace separated. `//` starts a comment. A leading `?N`
or `?!N` is a dialect condition: `?3` and `?6` mean en-us, `?!3` means
"not en-us", and a condition that does not apply makes the whole line
vanish. The rules file honours only `?3`; the dictionary honours `3` and
`6` both. That asymmetry is in `dialect_condition_applies` and
`parse_rule_line`, and it is not a typo.

The phonemes column is not always phonemes. `gi $abbrev` puts a flag
where the pronunciation would go, and `phoneme_field_as_flag` handles
that case separately from the trailing flag list. A `$altN` in that
column actively retracts an earlier plain entry for the same word
(`store_flag_only_entry`).

Flags fall into three groups, and the split is what
`parse_entry_flags` is organised around:

- part of speech and context bits, folded into a `struct entry_flags`
  that the caller reads: `$noun $verb $past $atend $capital $atstart
  $onlys $only $strend2 $u2` and the stress position `$1` .. `$6`.
- set memberships, applied immediately as side effects on the state:
  `$u $u+ $unstressend $abbrev $altN`.
- flags that only exist to set `grammar`, which gates the stress
  position bit.

`store_dictionary_entry` is the dispatch: given the flags, which of the
ten dictionaries does this entry land in. It is an if/else-if chain, and
the order is load bearing. `$noun` beats `$verb` beats `$atend` beats
`$capital`, and a bare entry with none of them lands in `dict`.

A line starting with `(` is a phrase entry, `(word1 word2) phonemes`,
handled by `parse_phrase_entry` before anything else looks at it. Only
two-word phrases with no `.` in either word are stored. A `||` inside
the phonemes splits the pronunciation across the two words and sends it
to `phrase_split_dict` instead of `phrase_dict`.

Two things happen after the file is fully read, both in
`load_dictionary`:

```c
qsort(compound_prefixes, ..., strpair_len_desc);
map_remove(&p->unstressed_words, &made);
```

The sort puts longest prefixes first so the compound scan can take the
first fit. The comparator returns 0 for equal lengths, and `qsort` is
not stable, so the relative order of same-length prefixes is whatever
the platform's qsort produces. It has never mattered, but it is the one
place where output could in principle differ across libc
implementations.

The second line hand-deletes "made" from the unstressed set. It is a
patch, it is one word, and it is documented in place.

### en_rules, the rules

Sections and directives:

```
.LNN item item item     letter group NN, referenced as LNN in contexts
.replace                 everything after this is "from to" pairs
.group X                 everything after this belongs to key X
```

Inside a group, one rule per line:

```
left)  match  (right   phonemes
```

Every part is optional. `build_rule_fields` reads them positionally: a
first token ending in `)` is the left context, a following token not
starting with `(` is the match string (absent means the rule matches the
group letter itself), a token starting with `(` is the right context,
and everything after that is concatenated into the phoneme output.

Two markers hide inside the right context and are pulled out at load
time rather than at match time:

- `P` not preceded by `L` and followed by a digit, end, `_`, `+` or `<`
  makes the rule a prefix rule (`detect_prefix_marker`).
- `S<N><flags>` makes it a suffix rule that strips N characters, with
  letter flags i/m/v/e/d/q/p mapped to bits (`detect_suffix_marker`,
  `suffix_flag_bit`).

Both markers still get scanned again as ordinary tokens during matching,
where they consume their own characters and score nothing.

Rules whose phonemes start with `$` and rules with an empty match are
dropped rather than stored.

## 3. The primitives, and why the buffer is not a string

The top of the file inlines three containers. They are small and you can
read them in five minutes, but one of them shapes everything above it.

`struct arr` and the `define_array` macro give a typed growable array
with `put` / `grow` / `free`. Used for `charsv` (array of strings),
`rules`, `replaces`, `strpairs`, `tokens`.

`struct map` is an open-addressed hash table with tombstones, keyed
either by int64 or by `struct chars`, with a per-map `value_free`
callback. Four init shorthands name the four shapes actually used:
`smap` (string to string), `set` (presence only, 1-byte value), `imap`
(string to int32), `pmap` (string to pair of strings).

`chars_view` is worth pausing on:

```c
static inline struct chars chars_view(const char * k, size_t kn) {
    struct chars v = { .data = (char *)k, .count = kn, .capacity = 0 };
    return v;
}
```

It builds a non-owning key so a lookup does not have to heap-allocate.
`map_put` deep-copies the key, so the borrowed pointer never escapes.
Every `smap_get`, `set_has`, `imap_get` in the file goes through it.
`capacity == 0` is the tell that a `chars` is a view rather than an
owner; nothing enforces it, so do not free a view.

And then `struct chars` itself:

```c
struct chars { // always zero terminated array of bytes
    char * data;
    size_t count;
    size_t capacity;
};
```

Read the comment literally. It is a byte buffer that happens to keep a
NUL for the convenience of `strchr` and `memcmp`. `count` is
authoritative, not `strlen`. The buffer routinely holds bytes that are
not text, which is the subject of the next section.

## 4. Two sentinels in the byte stream

The phoneme representation is an ASCII alphabet, roughly X-SAMPA:
letters and digits are segments, and a handful of punctuation bytes are
markers.

```
'   primary stress, precedes its vowel
,   secondary stress, precedes its vowel
%   explicitly unstressed
=   unstressed, and a compound boundary
-   morpheme boundary
:   length  (3: A: i: u:)
#   2   5   variant modifiers  (I# I2 a# @2 @5 @L 0#)
```

On top of that alphabet the code smuggles two bytes that never appear in
the data files and never reach the output.

**`\x01`, the rule boundary.** `apply_rules` appends one after every
rule's phonemes:

```c
chars_put(&sc->phonemes, match->phonemes.data, match->phonemes.count);
chars_put_byte(&sc->phonemes, '\x01');
```

So a rule-engine result carries a record of which bytes came from which
rule. `strip_rule_boundary_markers` removes them at the top of
`process_phoneme_string` and turns them into a parallel
`bool rba[]` array: `rba[i]` is true when a boundary followed byte i.
Later passes use it to answer "did these two stress marks come from the
same rule" (`same_rule_span`, in `cleanup_adjacent_secondary`) and "is
this vowel a rule's entire output" (`bare_zero_reduces`).

There is a second, quieter use. `rba_n > 0` is the signal that this
phoneme string came out of the rule engine at all, rather than being
copied from a dictionary entry. Half a dozen reductions gate on it under
the name `rule_derived`:

```c
static void reduce_e_unstressed(struct chars * ph,
                                const bool * rba, size_t rba_n) {
    bool rule_derived = rba_n > 0;
```

That is why the array is threaded through so many signatures. It is
carrying two facts, not one.

**`\x02`, the demoted primary.** `apply_stress_prev` implements the
`=` stress-prev directive: it retroactively promotes an earlier vowel to
primary, and rewrites any `'` before that point to `\x02` so the later
prosody passes do not see two primaries. The first thing
`process_phoneme_string` does after stripping `\x01` is turn every
`\x02` back into `,`.

The practical consequence: if you print an intermediate buffer to a
terminal while debugging, you will see nothing where those bytes are.
Print bytes, not strings.

There is one fixed bound to know. `rba` is a 512-entry stack array in
`process_phoneme_string`, and `strip_rule_boundary_markers` clamps its
writes to that, but sets `*out_n` to the full stripped length. Every
reader guards with `pi < rba_n`, so safety rests on phoneme strings
staying under 512 bytes. Every English word does, by a wide margin. If
you ever feed this synthetic input, that is the first invariant to
check. The same shape applies to the `MAX_SYL` = 256 syllable arrays:
the collectors stop at 256 and the callers do not notice.

## 5. The rule engine

This is the heart of the file. It answers one question repeatedly: at
position `pos` in this word, which rule applies?

### Contexts are a small language

A rule's left and right contexts are strings in a token language with
about fifteen operators. The two scanners read in opposite directions:
the left one from `pos - 1` walking backwards, the right one from
`pos + match_len` walking forwards.

Both share a shape. A cursor struct holds the running state, one small
function handles each token, and a single `ok` flag is the loop
predicate:

```c
struct lctx_scan {
    int32_t score;
    int32_t word_pos;
    int32_t ci;
    int32_t distance_left;
    char    prev_char;
    bool    ok;
};
```

A handler that cannot match sets `s->ok = false` and the loop stops
where it stands. Nothing returns early, so the scan has exactly one
exit, and every token is one readable function.

Tokens split into two disjoint sets, and the split is why the dispatch
is two functions rather than one long chain:

- **boundary / directive tokens** move the context cursor but not the
  word cursor: `_ & @ ! + < X E` on the left; `# + < & ! $ N P S` and
  digits on the right.
- **consuming tokens** advance the word cursor: literals, the letter
  groups `A B C F G H Y`, `K` (non-vowel), `D` (digit), `Z`
  (non-alpha), `%` (doubled letter), `LNN` (letter-group reference).

`lctx_boundary_token` returns false when the byte is not one of its
own, and the caller falls through to `lctx_consuming_token`. Because
the sets are disjoint, that two-step dispatch preserves the original
single-chain order exactly.

Naming is by role, not by character, so you can read a context string
by reading function names:

```
lctx_word_bound      _   word boundary to the left
lctx_stressed        &   some earlier syllable carries stress
lctx_syllables       @   count of syllables to the left
lctx_no_vowels_left  X
lctx_double_letter   %
lctx_letter_group    A B C F G H Y
lctx_lgroup          LNN
lctx_literal         anything else
```

Right-context handlers mirror them under `rctx_`, with four extra:
`rctx_scan_del_fwd` (`#`, remember an `e` to silence),
`rctx_word_alt` (`$w_altN`, a per-word variant condition),
`rctx_suffix_removed_cond` (`N`), and `rctx_replaced_e` (`E`, matches a
position an earlier rule silenced).

One asymmetry that surprises people: `&` in a left context asks whether
stress has already been assigned, and `&` in a right context always
fails. `E` in a left context always fails, and in a right context it is
a real matcher.

### Scoring

`rule_total_score` is four lines and worth memorising:

```
1                                   base
+ 21 * (match_len - group_length)   every char matched beyond the group key
+ left context score
+ right context score
+ 1 if the rule is dialect-conditional
```

Plus a `+35` bonus applied by `find_best_rule` to every rule found under
a two-character group key. So a two-letter group beats a one-letter
group by default, and a longer literal match beats a shorter one, and a
richer context beats a barer one.

Context tokens are worth roughly 19 to 21 points each, minus a distance
penalty that grows as the scan walks away from the match:

```c
static void lctx_advance(struct lctx_scan * s, int32_t pts) {
    s->distance_left += 2;
    if (s->distance_left > 19) { s->distance_left = 19; }
    s->score += pts - s->distance_left;
}
```

Left contexts decay by 2 per token, right contexts by 6
(`rctx_advance`), both capped at 19. Nearby context matters more than
distant context, and the right side decays faster. Those constants come
from espeak-ng and are not derived from anything.

`try_group` keeps the best scorer with `>=`, which means ties go to the
last rule in file order. That is not incidental. It is how en_rules
expresses "this is the general case, and this later line is the
exception".

### The scan

`apply_rules` walks the word left to right:

```c
while (sc->pos < sc->len && !sc->finished) {
    pos_char = sc->replaced_e[sc->pos] ? 'E' : tolower(word[sc->pos]);
    match = find_best_rule(...);
    if (match.score < 0) { sc->pos++; }      // no rule: silent skip
    else                 { apply_matched_rule(...); }
}
```

A position with no matching rule contributes nothing and is skipped in
silence. That is why a word of unknown letters phonemizes to the empty
string rather than to an error.

`apply_matched_rule` has four outcomes, and they are the whole control
flow of the engine:

1. **terminal suffix** (a suffix rule whose match ends the word, and
   suffix stripping is allowed): hand off to `process_suffix_rule`,
   which strips, re-phonemizes the stem, and finishes the scan.
2. **terminal suffix, phoneme-only mode**: take the phonemes, advance,
   do not recurse.
3. **prefix rule with word left over**: hand off to
   `process_prefix_rule`, which re-phonemizes the remainder as a whole
   word. If it produces output the scan finishes; if it declines, the
   cursor just advances.
4. **ordinary**: mark any `del_fwd` deletion into `replaced_e` and
   advance.

Two arrays follow the scan. `replaced_e` marks positions a rule silenced
(they will be re-read as the letter `E`). `pos_visited` records which
positions the scan actually touched, and exists for exactly one caller:
`compute_ed_suffix_voicing` asks "did the scan visit the `e` of this
`-ed`", which distinguishes "tied" (where a whole-word rule already
produced the `d`) from a mechanical suffix strip.

The three boolean parameters on `apply_rules` are worth a note because
their names read alike:

```
allow_suffix_strip   false: terminal suffix rules are skipped entirely
suffix_phoneme_only  true:  take a terminal suffix's phonemes, never recurse
suffix_removed       feeds the right-context 'N' token
```

and `word_alt_flags_param` is `-1` for "look the word up in
`word_alt_flags`" or a non-negative explicit mask.

## 6. Prosody: process_phoneme_string

Everything the rule engine or the dictionary produces goes through this
function before it becomes IPA. It is an orchestrator over three phases,
and the phases are strictly ordered.

```c
strip_rule_boundary_markers(out, rba, MAX_PH, &rba_n);
// \x02 -> ','
apply_segmental_phases(out, is_en_us);
apply_stress_phases(out, &ph_in, is_strend, rba, rba_n);
apply_reduction_phases(out, &ph_in, is_en_us, rba, rba_n);
```

`ph_in` is a snapshot of the input taken before the `\x01` strip. The
stress phases consult it, not the working buffer, to ask "did the input
already carry a `'` or a `,`". That distinction matters: a dictionary
entry that already marks its stress must not be re-stressed, and the
working buffer is being mutated underneath.

`is_en_us` is hardcoded `true` here, even though the dialect string
exists on the state and is consulted elsewhere. This repo ships en-us
only. The en-gb branches are all still present and all still dead in
this path.

### Segmental phases (steps 1 to 4e)

In order, and the order is the linguistics:

```
apply_velar_nasal_assimilation   n before k/g becomes N
apply_happy_tensing              word-final unstressed I becomes i
apply_vowel_reduction            unstressed A: and A become @
apply_lot_plus_r_merge           0r becomes O:r
strip_morpheme_schwa_r           @-r becomes r
apply_bare_schwa_to_rhotic       @ before r becomes 3, r often absorbed
apply_linking_r                  insert r after a rhotic before a vowel
apply_tion_stress_fix            -tion pulls stress onto the syllable before
apply_ology_stress_fix           -ology likewise
apply_ic_stress_fix              -ic likewise
```

The last three are stress moves that live in the segmental phase because
they depend on the segments being final. Each finds an anchor
(`find_tion_suffix_s_pos`, `matches_ology_at`, `unstressed_ik_pos`),
counts vowels before it with the multi-character-aware scanner, and
calls `move_stress_to_vowel`.

### Stress phases (steps 5, 5.0, 5a)

```c
insert_primary_stress(out, is_strend, rba, rba_n);
bool step50_fired = apply_equals_suffix_stress_shift(out);
bool step5a_ran   = place_secondary_stress(out, ph_in, ..., step50_fired, ...);
cleanup_adjacent_secondary(out, step5a_ran, ...);
trochaic_compound_prefix(out, ph_in, ..., step5a_ran, ...);
final_syllable_secondary(out);
```

Note the two return values. `apply_equals_suffix_stress_shift` and
`place_secondary_stress` report whether they fired, and the three passes
after them read those flags. This is a small state machine, not a
pipeline: whether cleanup runs depends on whether secondary placement
already ran.

`insert_primary_stress` is the single most intricate function in the
file. Its shape:

- it is a no-op if a `'` already exists.
- `compute_pick_last` decides whether the primary goes on the first
  eligible vowel or the last one. A compound tail after `=` with no
  strong vowel in it forces "last".
- `scan_for_primary` walks the string once with a `struct primary_scan`
  cursor, tracking the last strong vowel, the last schwa, and the
  position of a leading secondary's vowel. `primary_scan_vowel` is the
  per-vowel decision.
- `suppress_primary_on_schwa` vetoes a primary that would land on a
  schwa when the word leads with `%` or already carries a secondary.
- if everything is vetoed, `stress_a_hash_last_resort` stresses an `a#`.
  A word that reaches this path leaves a fingerprint (`'a#`) that step B
  later recognises and undoes, in `is_pct_last_resort_word`.

`place_secondary_stress` collects syllables into `sec_syl` records, finds
the primary, then marks at even syllable distances: backwards from the
primary (`mark_secondary_backward`) and forwards
(`mark_secondary_forward`). Marks are sorted descending and inserted
right to left, so earlier positions stay valid. That descending-insert
pattern recurs throughout the file; `sort_positions_desc` and
`sort_marks_desc` exist only to enforce it.

`split_centering_diphthong` deserves a mention because it is the one
place `rba` is used for something structural rather than for gating: a
rule boundary inside a three-character centering diphthong means the two
halves came from different rules, so the trailing `@` or `3` counts as
its own syllable.

### Reduction phases (steps 5.5 to 6f)

Sixteen small passes, all en-us, listed in `apply_en_us_reductions` in
execution order. They share a family resemblance: find a stress marker,
walk the span it defines, and rewrite one vowel class. Half of them are
variations on "bare 0 becomes @ in this particular position":

```
reduce_bare_zero_after_ution          only when "u:S" is present
reduce_zero_between_sec_and_primary
reduce_zero_between_primary_and_sec
reduce_zero_hash_before_primary
reduce_bare_zero_before_primary       gated on rba
reduce_zero_diminished                middle syllable, level <= unstressed
```

They are not one function with parameters because the conditions differ
in ways that do not factor: the span endpoints, the variant-digit
skipping, and the extra guard each pass carries. Reading them as a group
is easier than reading any one alone.

`reduce_e_unstressed` is the largest and is split by context into
`reduce_e_between_stresses`, `reduce_e_after_pct_lead`, and
`reduce_e_after_first_stress`, each with the same core rewrite
(`reduce_e_at`: `E` becomes `@` before `n`, `I2` otherwise) applied over
a different span.

`apply_flap_rule` is the American t-flap: `t` between a vowel and an
unstressed vowel becomes `*`. The three predicates
(`flap_prev_is_vowel`, `flap_after_marker`, `flap_before_vowel`) carry
the syllabic-n exceptions, and `deflap_before_syllabic_nasal` runs first
to undo a flap that a rule already emitted where it should not have.

## 7. Word dispatch: word_to_phonemes

Eight arms, first non-empty wins, order copied from the original:

```c
check_capital_dict          capitalised word, capital_dict
check_main_dict             onlys_bare_dict then dict, plus $N stress
check_hyphenated            split on '-', recurse per segment
check_possessive            trailing 's, base plus an allomorph
check_single_letter         "_x" key in dict
check_morphological_suffixes
check_compound_prefixes
apply_rules_fallback        the rule engine, unconditionally
```

The convention throughout is that an arm writes into a caller-owned
`struct chars *out` and leaves it empty to mean "not mine". There is no
return value to check and no sentinel to confuse.

`check_morphological_suffixes` is the same pattern one level down, with
eight arms of its own:

```
check_suffix_ing      check_suffix_ed       check_suffix_ies
check_suffix_dict_s   check_suffix_magic_es check_suffix_ch_sh_es
check_suffix_xes      check_suffix_arily
```

Each strips a suffix, phonemizes the stem, and appends a suffix
pronunciation whose voicing depends on the stem's last phoneme. The
stem side is where the complexity is, and it is shared:
`get_stem_phonemes` tries verb_dict, then dict, then the magic-e
spelling, then the rule engine, and rejects any result with no vowel
phoneme in it.

The `-ing` and `-ed` arms then layer on spelling-recovery fallbacks,
because English spelling deletes information the stem lookup needs back:

```
ing:  magic-e (hoping -> hope), doubled consonant (running -> run),
      i-for-y (studying -> study)
ed:   the same, plus the undoubled preference for two-syllable stems
      (controlled -> control), plus -ns/-rs wanting their e back
```

`should_use_magic_e_for_cvc_stem` is the arbiter for whether the magic-e
spelling is even tried, and it is called with different weak-vowel sets
for `-ing` (`"I@"`) and `-ed` (`"I@3"`).

### The recursion

Six call sites recurse, and two forward declarations exist to let them:

```
word_to_phonemes -> check_hyphenated       -> word_to_phonemes
word_to_phonemes -> check_possessive       -> word_to_phonemes
word_to_phonemes -> check_compound_prefixes-> word_to_phonemes
word_to_phonemes -> apply_rules -> process_prefix_rule -> word_to_phonemes
word_to_phonemes -> apply_rules -> process_suffix_rule
                                -> stem_phoneme_from_dict -> apply_rules
```

Every recursive call is on a strictly shorter string than the one that
made it: a hyphen segment, a base without `'s`, a suffix after a prefix,
a stem after a strip. That is the entire termination argument, and it is
the invariant to protect if you add an arm.

## 8. Codes to IPA

`phonemes_to_ipa` is the boundary. Above it the buffer holds ASCII
phoneme codes; below it, UTF-8 IPA bytes. Passes on either side operate
in different alphabets, and confusing which side you are on is the
easiest mistake to make in the sentence layer.

The rendering is three-layered:

1. **Explicit overrides**, `p->ipa_overrides`, built at create time from
   `IPA_COMMON` plus one of `IPA_EN_US` / `IPA_EN_GB`. Consulted first.
2. **The ipa1 table**, `ASCII_TO_IPA[96]`, mapping every byte 0x20 to
   0x7F to a codepoint. Mostly identity; the differences are the
   phoneme alphabet (`A` to U+0251, `3` to U+025C, and so on).
3. **Structural handling** in `phoneme_code_to_ipa_table`: `#` and `/`
   stop the walk, `|` is skipped, a leading `_` stops, and variant
   digits after the first character are dropped.

`emit_phoneme_code` does a greedy longest-match against `MULTI_CODES`,
which is ordered longest-first so `aU@r` wins over `aU@` over `aU`. Two
details:

- `i@` and `U@` are not matched as pairs when the previous marker was
  `%` or `=`, because in that position they are two separate vowels, not
  a diphthong.
- stress is emitted lazily. A `'` or `,` is held in `pending_stress`
  and flushed just before the next vowel or syllabic consonant, because
  those are the syllable nuclei. `is_syllabic_code` is what makes
  `n-`, `m-`, `@L`, `r-`, `l/` count as nuclei.

## 9. The sentence layer

`phonemize_text` is the per-utterance driver. It tokenizes, finds the
index of the last word token, decides whether the utterance is a single
isolated word, and runs `process_word_token` over each word.
Punctuation tokens are counted but never emitted.

`tokenize` splits into words and single-character punctuation. Letters,
apostrophes and digits accumulate; UTF-8 sequences accumulate whole;
a hyphen between letters stays in the word; a period is punctuation
unless it is mid-abbreviation. That last rule is why "U.S." survives as
one token but "U.S.A." does not (the abbreviation seed only fires when
exactly one character has accumulated, so it becomes "U.S." + "A" +
"."). It is faithful to the original and the tests pin it.

Each token carries `needs_space`, meaning "a word preceded me in this
utterance", which the emitters use to decide on a separator.

### The per-token pipeline

`process_word_token` composes about twenty helpers in a fixed order.
Reading it top to bottom is reading the design:

```
expand_number_token         digits become spelled words, each phonemized
spell_acronym_token         all-caps or mixed-case-no-vowel, letter by letter
try_clitic_or_phrase        bigram lookahead, four sub-paths
expand_period_abbreviation  "U.S." style
--- from here on, ph_codes holds phoneme codes ---
apply_pos_context_override  a recent $pastf/$nounf/$verbf swaps the entry
apply_atstart_override      first word of the utterance
apply_atend_override        last word of the utterance
apply_lemma_override        hand-written cases: the, a, an, to, use
apply_step_b                strip primaries from $u words
apply_step_c                function-word stress: keep, need, promote, demote
fix_diphthong_stress_position
apply_inter_word_t_flap
apply_cross_word_schwa_rhotic
--- phonemes_to_ipa: from here on, ipa holds UTF-8 ---
maybe_add_default_stress
apply_pre_vowel_the_fixup
apply_r_linking
apply_cross_word_t_flap
update_pos_context_counters
```

Four of those steps consume the *next* token, not the current one, and
that is what makes this a sentence phonemizer rather than a word one:
`try_clitic_or_phrase` (bigram), `apply_r_linking`,
`apply_cross_word_schwa_rhotic`, `apply_cross_word_t_flap`. All four
ask the same kind of question, "does the next word start with a vowel",
and they answer it by actually phonemizing the next word and looking at
its first code. The next word therefore gets phonemized twice. That is
the cost of the design, and it is why phonemizing a long sentence is not
linear in its length.

Two of those look-ahead helpers stop at punctuation and two do not:
`find_next_word_stop_on_punct` returns nothing across a comma, while
`find_next_non_empty_word` skips punctuation. Which one a caller uses
encodes whether the effect survives a pause.

The `is_isolated_word` flag threads through nearly every helper. A word
phonemized alone is treated differently from the same word in a
sentence: articles keep their stress, function words are not reduced,
`to` is "tu:" rather than "t@5". If you are debugging a discrepancy
between a word alone and the same word in context, this flag is almost
always the reason.

`expect_past` / `expect_noun` / `expect_verb` are the only mutable state
that survives across tokens. A `$pastf` word sets a counter to 3, and
the always-decrement at the end of `update_pos_context_counters` gives
the effect a two-token or one-token window.

### Step B and step C

These two are worth reading together, because they are where a word's
lexical stress meets its position in the sentence.

Step B (`apply_step_b`) strips every `'` from a word that is flagged
unstressed, unless the word is in `STEP_B_KEEP_SECONDARY_WORDS`. It also
undoes the `'a#` fingerprint that `insert_primary_stress` leaves when it
runs out of options.

Step C (`apply_step_c`) is the promotion and demotion logic:

- `step_c_keeps_secondary` decides whether this word should carry a
  secondary rather than a primary, from a static list plus four state
  sets.
- `try_promote_keep_sec_to_primary` overrides that when no stressed
  content follows, which is how "what are you *on*" gets its stress.
- `apply_comma_to_primary_promotion` promotes a lone `,` to `'` unless
  it is a leading comma, which `comma_is_leading` defines as "at the
  start, or after a `%`, or with no real vowel before it".
- a `$unstressend` word at the end of an utterance gets demoted again.

The static lists are near their use, `STEP_B_KEEP_SECONDARY_WORDS` and
`STEP_C_KEEP_SECONDARY`. The second is a superset of the first plus a
few extras. Both are hand-written and both are the kind of table that
grows when someone hears a bad sentence.

## 10. Things that look like bugs and are not

If you go fixing these, the byte-equality harness will tell you
immediately. Here is why not to.

**A dangling pointer that is read on purpose.** `struct troch_syl` holds
`const char * code` pointing into the phoneme buffer.
`promote_trochaic_primary` and `insert_trochaic_secondary` then
`memmove` that buffer under it. The scan reads whatever the buffer holds
now, which is what the original does and what the output depends on.
The comment above the struct says so. Do not snapshot the code.

**A `strchr` that finds the NUL.** In `devoice_ed_suffix`, `last_ph`
stays 0 when the stem is nothing but markers. Then
`strchr("ptkfTSCxXhs", 0)` returns a pointer to the terminator, which is
not NULL, so the all-marker stem takes the voiceless arm and the suffix
becomes `t`. That is behaviour, not an accident of C, and the Swift port
reproduces it explicitly.

**A deliberate size_t underflow.** In `unstressed_ik_pos`, `k_pos == 1`
makes `k_pos - 2` wrap to `(size_t)-1`, which is exactly the sentinel
the caller checks for. The wrap is the skip.

**Dead code that is kept.** `skip_to_better_primary` sits inside
`#if 0`, with both of its conditions hardcoded false in the original
too. It is preserved because the logic is not recoverable from anywhere
else if a future dialect needs it.

**A redundant condition.** `check_single_letter` tests
`nn == 1 && !(nn == 1 && norm[0] == 'a')`. The inner `nn == 1` is
already known. It is harmless and it matches the original's shape.

**Silent skips.** A position with no matching rule contributes nothing.
A word with no matching rules phonemizes to the empty string and is
simply not emitted. There is no error path for "I cannot pronounce
this".

## 11. Where to look for what

By concern, with the entry point to read first.

```
public API                phonemizer_create, phonemizer_phonemize
containers                struct chars, struct map, chars_view
state layout              struct phonemizer, phonemizer_state_init

loading en_list           load_dictionary
  per-line                load_dictionary_line
  flags                   parse_entry_flags -> entry_flag_*
  routing                 store_dictionary_entry -> store_*_entry
  phrases                 parse_phrase_entry, store_phrase_entry

loading en_rules          load_rules
  directives              parse_rules_directive, parse_lgroup_def
  one rule                parse_rule_line, build_rule_fields
  markers                 detect_prefix_marker, detect_suffix_marker

rule matching             find_best_rule -> try_group -> match_rule
  left context            match_left_context_score -> lctx_*
  right context           match_right_context_score -> rctx_*
  scoring                 rule_total_score, lctx_advance, rctx_advance
  the scan                apply_rules -> run_rule_scan -> apply_matched_rule
  suffix path             process_suffix_rule -> stem_phoneme_from_dict
  prefix path             process_prefix_rule

prosody                   process_phoneme_string
  segmental               apply_segmental_phases
  stress                  apply_stress_phases
    primary               insert_primary_stress -> scan_for_primary
    secondary             place_secondary_stress -> mark_secondary_*
  reductions              apply_reduction_phases, apply_en_us_reductions
  the flap                apply_flap_rule -> flap_*

word dispatch             word_to_phonemes
  suffixes                check_morphological_suffixes -> check_suffix_*
  stems                   get_stem_phonemes, stem_*_phonemes
  compounds               check_compound_prefixes

IPA rendering             phonemes_to_ipa -> emit_phoneme_code
  tables                  ASCII_TO_IPA, IPA_COMMON, IPA_EN_US, MULTI_CODES

sentence layer            phonemize_text -> process_word_token
  tokenizing              tokenize
  numbers                 expand_number_token, int_to_words
  acronyms                spell_acronym_token, spell_group
  bigrams                 try_clitic_or_phrase
  function words          apply_step_b, apply_step_c
  cross-word              apply_r_linking, apply_cross_word_t_flap,
                          apply_cross_word_schwa_rhotic
  lemma overrides         apply_lemma_override -> lemma_override_*

tests                     #ifdef PHONEMIZER_TESTS at the bottom
```

## 12. Conventions the file keeps

Worth knowing before you edit, because they are enforced by review and
by the byte-equality harness respectively.

Single entry, single exit. No early `return`, no `continue`, and the only
`break` statements are the nine switch arms in `match_group`. A loop that
must stop early carries the reason in its predicate, which is usually a
field of the cursor struct rather than an invented flag. `s->ok`,
`sc->finished`, `found`, `done` and `processing` are all real
post-conditions, not bookkeeping.

A pass that is a no-op on some inputs wraps its body rather than
returning early:

```c
if (memchr(ph_in, '\'', in_n) != NULL) {
    ... the whole body ...
}
```

That is one point of cyclomatic complexity for one fewer exit, and it is
the shape `demote_adjacent_primaries` uses.

Output parameters are caller-owned and reset by the callee. A function
taking `struct chars * out` sets `out->count = 0` on entry. Callers do
not pre-clear.

Ownership moves are explicit. When a `struct chars` is handed to a
container, the source is zeroed on the next line:

```c
best->phonemes = ph;          // moved in
ph = (struct chars){0};       // detach
```

Comments say why, not what. There are few of them and each one is
either a linguistic reason, a deliberate quirk, or a bound.

Mutation happens right to left. Anything that inserts or erases at
recorded positions sorts descending first, so earlier positions stay
valid. If you add a pass that mutates in a loop, follow the pattern or
recompute the positions.

Fixed-width integers throughout: `int32_t`, not `int`.
