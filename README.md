# kittens-tts-cli

Offline neural text-to-speech as a **single self-contained binary**.
No network, no external model files, no runtime config — the
model, all eight voices, and the phonemizer data are linked *into* the
executable. Built from the pure-C KittenTTS backend (the same C core
the macOS/iOS app drives).

```
$ ./kittens-tts-cli -p "Hello from the offline kittens engine." -s Kiki -o hello.wav
wrote hello.wav  (2.1s, voice Kiki, speed 1.00)
```

## Build

Requires Apple `clang` (C23 `#embed`) and macOS Accelerate. `ffmpeg`
is needed only for non-WAV output.

```
make            # -> ./kittens-tts-cli   (~39 MB, fully self-contained)
make run        # build + speak a demo line to out.wav
make clean
```

`otool -L` shows it links only `Accelerate` and `libSystem` — both
always present on macOS. Copy the one file anywhere and it runs.

## Usage

```
kittens-tts-cli -p "Text. Sentences split on . ! ?
Paragraphs split on a single newline." [options]
kittens-tts-cli -f input.txt [options]
kittens-tts-cli --voices
```

| flag         | meaning                                                       |
|--------------|---------------------------------------------------------------|
| `-p TEXT`    | text to speak                                                 |
| `-f FILE`    | read text from FILE instead of `-p`                           |
| `-s VOICE`   | voice (default `Kiki`); accepts friendly or internal name     |
| `-o FILE`    | output. `.wav` written natively; any other extension          |
|              | (`.mp3`, `.m4a`, `.aac`, …) is produced via `ffmpeg`. Default  |
|              | `out.wav`                                                     |
| `--speed X`  | speech-rate multiplier (default `1.0`)                        |
| `--voices`   | list voices and exit                                          |
| `-h`         | help                                                          |

### Text format

- **Sentences** are split on `.` `!` `?` — each becomes one synthesis
  unit. The terminator is kept so the model renders the right
  intonation.
- **Paragraphs** are split on a single newline (`\n`); a paragraph
  boundary gets a longer pause than a sentence boundary.

### Voices

```
Hugo  Luna  Kiki  Leo  Bella  Jasper  Bruno  Rosie
```

Each maps to an internal `expr-voice-*` embedding with the same
per-voice speed prior the app uses, so the CLI sounds identical to the
app. Output is 24 kHz mono.

## Exit codes

`0` success · `2` usage error (bad/missing args, unknown voice) ·
`1` runtime failure (model init, write error).

## What this does NOT do

The Swift app's `TextPreprocessor` (currency / percent / ordinal /
decimal expansion) is **not** ported — the CLI is deliberately minimal.
The phonemizer speaks plain cardinals on its own
(`"1969"` → "one thousand nine hundred sixty nine"), but `"3rd"`,
`"50%"`, and `"$5"` read literally. For prose this is fine; for
number-heavy text, expand to words in your input.

## Layout

Engine sources are synced verbatim from the
[kittens.cpu](https://github.com/leok7v/kittens.cpu) upstream
(`tts/src`, `tts/include`, `gguf/`, `src/trace` there).

```
kittens_tts_cli.c        single-file CLI driver (#ifdef KITTENS_TTS_CLI_MAIN)
src/kittens.c            synth core (transitively #includes tensor.c +
                         gguf_reader.c; time-segmented HiFi-GAN / iSTFT /
                         noise stages keep peak RSS bounded)
src/tensor.c             fp32 tensor lib + arena allocator
src/gguf_reader.c        shared GGUF v3 reader
src/trace.c              leveled trace ring (stderr mirror)
src/phonemizer.c         text -> IPA (C99 port)
src/kitten_symbols.h     IPA code-point -> phoneme-id table (generated)
include/tts/kittens.h    public C API
include/trace/trace.h    trace ring API
resources/               kitten_full.gguf, voices.safetensors, en_rules, en_list
```

