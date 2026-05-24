# kittens-tts-cli -- offline neural TTS, single self-contained binary.
#
#   make            build ./kittens-tts-cli
#   make run        build + speak a demo line to out.wav
#   make clean
#
# The driver TU needs C23 (#embed pulls the model + voices + phonemizer
# data into the executable). The C cores compile as gnu17, matching the
# Xcode build they were exported from. Apple Accelerate provides cblas.

CC       ?= clang
BIN      := kittens-tts-cli
SRC      := src
WARN     := -Wall -Wextra -Wno-unused-parameter -Wno-unused-function
OPT      := -O2
INC      := -I$(SRC)
LDFLAGS  := -framework Accelerate -lm

RES := resources/kitten_full.gguf resources/voices.safetensors \
       resources/en_rules resources/en_list

OBJS := build/driver.o build/kittens.o build/phonemizer.o

$(BIN): $(OBJS)
	$(CC) $(OBJS) $(LDFLAGS) -o $@

# Driver: C23 for #embed; depends on the embedded resources + the
# generated symbol table so edits to either force a rebuild.
build/driver.o: kittens_tts_cli.c $(SRC)/kitten_symbols.h $(RES) | build
	$(CC) -std=c23 $(OPT) $(WARN) $(INC) -DKITTENS_TTS_CLI_MAIN \
	      -c kittens_tts_cli.c -o $@

# kittens.c transitively #includes gguf.c + tensor.c (guarded single-
# file libraries), so this one TU covers the whole synth core.
build/kittens.o: $(SRC)/kittens.c | build
	$(CC) -std=gnu17 $(OPT) $(WARN) $(INC) -c $(SRC)/kittens.c -o $@

build/phonemizer.o: $(SRC)/phonemizer.c | build
	$(CC) -std=gnu17 $(OPT) $(WARN) $(INC) -c $(SRC)/phonemizer.c -o $@

build:
	@mkdir -p build

run: $(BIN)
	./$(BIN) -p "Hello from the offline kittens speech engine. It runs entirely on the C P U." -s Kiki -o out.wav

clean:
	@rm -rf build $(BIN)

.PHONY: run clean
