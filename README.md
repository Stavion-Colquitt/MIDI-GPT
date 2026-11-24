# MIDI GPT (LM 2 and LM 3)

> AI-assisted with Claude; creative and technical decisions are Stavion's.

Two GPTs trained from scratch on the Lakh MIDI dataset for my "instrumental maker" project: **LM 2** writes
multi-instrument MIDI from nothing, and **LM 3** is LM 2 fine-tuned to write it to a short request (tempo, time
signature, key, instrument families, note density), with twice the context.

Models on Hugging Face:

- [Stavion-Colquitt/MIDI-GPT-LM2](https://huggingface.co/Stavion-Colquitt/MIDI-GPT-LM2)
- [Stavion-Colquitt/MIDI-GPT-LM3](https://huggingface.co/Stavion-Colquitt/MIDI-GPT-LM3)

Both use the same from-scratch transformer family as my [Tiny GPT (LM 1)](https://github.com/Stavion-Colquitt/Tiny-GPT)
and the same idea: every track of a song goes into one token stream (instrument tokens say who plays), so the model
learns whole arrangements: drums, bass, guitars, keys, strings. The goal is an instrumental maker that writes
editable MIDI straight into an Ableton session.

## Quick start

```
py -3.12 -m pip install torch numpy miditok symusic mlflow safetensors huggingface_hub
# download lmd_full.tar.gz from https://colinraffel.com/projects/lmd/ and extract it to data/lmd_full
# LM 2
py -3.12 prep_midi.py                  # tokenizer + data/train.bin, data/val.bin, data/meta.json
py -3.12 train.py --hours 12           # run 1
py -3.12 train.py --resume --extend_hours 12 --seed 2   # extension
py -3.12 generate.py --n 4 --temp 0.85 # new songs in generated/
py -3.12 check_midi.py "generated/*.mid"
py -3.12 eval_lm2.py                   # full-pass val loss + music stats (lm2_eval/)
py -3.12 export_hf.py                  # safetensors + config + card for Hugging Face
# LM 3 (starts from LM 2: needs the tokenizer and data/ files built by prep_midi.py)
py -3.12 prep_lm3.py
py -3.12 train_lm3.py --hours 12
py -3.12 eval_lm3.py
py -3.12 export_hf_lm3.py
```

`run_lm2.ps1` and `run_lm3.ps1` pin each run to one GPU on a multi-GPU machine (set `LAKH_TOKENS` if the token files
live elsewhere).

## LM 2: unconditional, 17.7M parameters

A 17.7M-parameter GPT that writes multi-instrument MIDI from scratch. Same from-scratch transformer as my
[Tiny GPT (LM 1)](https://github.com/Stavion-Colquitt/Tiny-GPT), retrained on music instead of text: 1.24 billion
tokens of the Lakh MIDI dataset, on a single **RTX 2060 (6 GB)** in about 24 hours. It learns whole arrangements
from one token stream: drums, bass, guitars, keys, strings. It is step 1 of the instrumental maker.

### Results

Validation loss over **all 19.8M validation tokens** (Lakh files whose MD5 name starts with 00 to 03, never trained
on), one deterministic pass with non-overlapping 1,024-token windows:

| Run | Steps | Tokens seen | Val loss | Perplexity |
|-----|-------|-------------|----------|------------|
| Run 1 (cosine, peak LR 1e-3), best checkpoint | 57,000 | 3.7B | 1.3420 | 3.83 |
| Extension (re-warm to 3e-4, cosine to 3e-5, new seed), best checkpoint | 112,000 | 7.3B | **1.3055** | **3.69** |

The extension lowered loss on identical data by about 0.037 nats (2.7%). During training the "best" value looked
better (1.254) because it was picked from hundreds of noisy 40-batch estimates; the full pass above is the number
to trust.

### What the music looks like

8 songs generated from scratch at 3,072 tokens each, compared with the first 3,072 tokens of 30 real validation
songs (medians, interquartile range in brackets). All 8 are in [`examples/`](examples), good and bad, unfiltered.

| | LM 2 | Real Lakh songs |
|---|---|---|
| Valid MIDI | 8 / 8 | 30 / 30 |
| Instrument tracks | 1.5 (1 to 3.5) | 7 (5 to 8.75) |
| Notes per second | 4.8 (2.6 to 9.0) | 21.7 (14.2 to 26.3) |
| Pitch range (semitones) | 39.5 (3.8 to 46) | 57 (50.5 to 61.5) |
| Pitch-class entropy (bits) | 2.16 (1.19 to 2.81) | 2.81 (2.72 to 2.96) |
| Has drums | 6 / 8 | 28 / 30 |

| Example | Temp | Tracks | Notes | Length | Tempo |
|---|---|---|---|---|---|
| `lm2_t0.85_0.mid` | 0.85 | drums, muted + overdriven guitar, finger bass, strings | 1,739 | 0:40 | 135 BPM |
| `lm2_t0.85_1.mid` | 0.85 | drums, piano, steel + clean guitar, acoustic bass | 1,631 | 3:28 | 67 BPM |
| `lm2_t0.85_2.mid` | 0.85 | drums, overdriven guitar, finger bass | 1,843 | 2:27 | 121 BPM |
| `lm2_t0.85_3.mid` | 0.85 | nylon guitar | 439 | 12:52 (mostly gaps) | 121 BPM |
| `lm2_t1.00_4.mid` | 1.0 | piano | 1,238 | 3:27 | 60 BPM |
| `lm2_t1.00_5.mid` .. `_7` | 1.0 | ended early (drums only or nearly empty) | 2 to 275 | 0:01 to 1:15 | |

At temperature 0.85 it usually writes a band arrangement; at 1.0 it often ends the song almost immediately. Output
is thinner than real songs (fewer instruments, far fewer notes), which is the main thing to improve.

### Model

| | |
|---|---|
| Architecture | decoder-only transformer, pre-norm, 8 layers, 6 heads, d_model 384, GELU MLP (4x) |
| Context | 1,024 tokens (about half a minute of a typical song), learned positional embeddings |
| Tokenizer | [MidiTok](https://github.com/Natooz/MidiTok) REMI + BPE, 8,192 tokens, trained on 10,000 training songs (`tokenizer.json`): one stream for all instruments, 16 velocity levels, tempo and time-signature tokens |
| Parameters | 17.7M (token embedding tied to the output head) |
| Attention | PyTorch `scaled_dot_product_attention` (causal) |

### Data

[Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/) `lmd_full` (Colin Raffel, CC BY 4.0): 173,283 songs kept
(files up to 1 MB that encode to at least 64 tokens), 1.236B training tokens and 19.8M validation tokens. The split
is by file name (MD5 hash): names starting 00 to 03 (about 1.6%) are validation.

### Training

- AdamW (0.9, 0.95), weight decay 0.1 on matrices only, gradient clipping 1.0
- 16 x 4 micro-batches of 1,024 tokens = 65,536 tokens per step; FP16 autocast + GradScaler (Turing has no BF16)
- Run 1: 500 warmup steps, cosine from 1e-3 to 1e-4, sized to a 12-hour budget: 60,763 steps
- Extension: resume from the last state, 300-step re-warm to 3e-4, cosine to 3e-5 over 12 hours, new data seed:
  112,767 steps total (7.4B tokens, about 6 passes over the data)
- Evaluated every 500 steps (val loss + 3 sample songs written as .mid), best checkpoint kept by val loss, MLflow

### Limitations

A 17.7M-parameter model with a short context: it is a learning project, not a production composer. Arrangements
are sparse, structure rarely lasts beyond the context window, and LM 2 has no conditioning (you can't ask it for a
genre, key or tempo; that is what LM 3 adds). Lakh is a crowd-sourced collection of MIDI transcriptions of real
songs, so generations can echo familiar patterns from it; check anything before using it in a release.

## LM 3: controllable, 18.2M parameters

### What it is

An 18.2M-parameter GPT that writes multi-instrument MIDI to a request: tempo, time signature, key, instrument
families and note density. It is LM 2 fine-tuned to follow a short control prefix, with the context stretched from
1,024 to 2,048 tokens (about a minute of music).

- Base model: LM 2, 17.7M parameters trained from scratch
  ([Stavion-Colquitt/MIDI-GPT-LM2](https://huggingface.co/Stavion-Colquitt/MIDI-GPT-LM2)).
- LM 3 keeps 8 layers, 6 heads and d_model 384, grows the vocabulary to 8,256 for the controls, and grows the
  context to 2,048 tokens.

### How the control prefix works

The controls are 64 new tokens with ids 8192 to 8255, defined in `lm3_vocab.py`: 8 tempo bins, 6 time signatures,
24 keys plus "none", 17 instrument families, 6 density bins, a control-end token at id 8254 and one reserved style
token at id 8255.

A prefix is one tempo token, one time signature token, one key token, one token per instrument family present
(always in the fixed order of `lm3_vocab.FAMILIES`), one density token, then control-end. Generation starts from that
prefix.

During training the prefix was pinned in front of every window: each window was the song's own control prefix plus a
slice of that song, so the model always saw the request exactly where it sits at generation time. No loss was taken
on the prefix, because the prefix is given, not predicted.

Weights were initialised from LM 2 step 112000. The new embedding rows start at the mean of the old rows, and LM 2's
1,024-row position table was stretched to 2,048 rows by linear interpolation before fine-tuning.

### Training

- Data: Lakh MIDI Dataset, lmd_full split. 170,570 training songs, about 1.24B training tokens.
- 57,289 steps, about 12 hours on one RTX 4060, fp16, 2,048-token windows.
- Best checkpoint: step 53,000, validation loss 1.029. That checkpoint is the published LM 3 model.

### Evaluation

60 random but seeded control requests, 1,536 tokens each, temperature 0.9, top_k 50, seed = request index,
generated with the published checkpoint. Every output was decoded back to MIDI and its controls were measured from
the MIDI file itself with `lm3_vocab.controls_from_score`, then compared with what had been asked for. Full
breakdown, including by tempo range: [`lm3_eval/summary.md`](lm3_eval/summary.md).

All 60 requests (59 decoded), and split by the mode of the key that was asked for, which is where control is
weakest:

| metric | all requests | major keys (34) | minor keys (26) |
|---|---|---|---|
| decoded | 98.3% | 100.0% | 96.2% |
| tempo within 10% | 55.9% | 61.8% | 48.0% |
| exact key | 37.3% | 52.9% | 16.0% |
| key allowing relative major/minor | 40.7% | 55.9% | 20.0% |
| all requested instruments present | 42.4% | 55.9% | 24.0% |
| instrument recall (mean) | 74.2% | 81.1% | 64.7% |
| instrument precision (mean) | 93.7% | 95.8% | 90.7% |
| density bin exact | 42.4% | 35.3% | 52.0% |
| density bin within 1 | 86.4% | 82.4% | 92.0% |
| mean beats generated | 136.1 | 137.4 | 134.3 |
| mean tracks | 3.0 | 3.2 | 2.7 |

Six of the 60 songs are in `examples/`, with every requested instrument present: `req_31.mid`, `req_41.mid` and
`req_56.mid`, the clearest of the set, plus `req_11.mid`, `req_24.mid` and `req_32.mid`. The file names are the
request ids from the evaluation, and nothing else was filtered.

### Usage

From the Hugging Face repo (or any folder holding `model.safetensors`, `config.json` and `tokenizer.json`):

```
pip install torch miditok symusic safetensors
python generate.py --bpm 120 --key "A minor" --instruments DRUMS,Bass,Guitar,Piano --density 3 --n 4 --tokens 1536 --temp 0.9 --seed 0
python check_midi.py "generated/*.mid"
```

In this repo `generate_lm3.py` does the same job: it loads `model.safetensors`, `config.json` and `tokenizer.json`
from the folder it sits in (so run it next to an export), takes the same `--bpm`, `--key`, `--time_sig`,
`--instruments`, `--density`, `--n`, `--tokens`, `--temp`, `--top_k`, `--seed` and `--out` flags, and prints the
request next to what was measured in the generated file. It runs on CPU or CUDA.

`--density` is a bin from 0 to 5 (edges 1, 2, 4, 8, 16 notes per beat across all tracks). `--key` is one of the
24 keys with "major" or "minor", for example `"Eb major"` or `"A minor"`. `--instruments` is a comma separated
subset of Piano, ChromPerc, Organ, Guitar, Bass, Strings, Ensemble, Brass, Reed, Pipe, SynthLead, SynthPad, SynthFX,
Ethnic, Percussive, SFX and DRUMS.

| | |
|---|---|
| Architecture | decoder-only transformer, 8 layers, 6 heads, d_model 384, context 2,048, tied embeddings |
| Tokenizer | MidiTok REMI + BPE, 8,192 tokens plus 64 control tokens (`tokenizer.json`); end-of-song id 2 |
| Files | `model.safetensors` (fp32), `config.json`, `tokenizer.json`, `model.py`, `lm3_vocab.py`, `generate.py`, `check_midi.py`, `examples/` |

### Limitations

- Minor keys drift to major: 16.0% of minor requests came back in the exact key, against 52.9% for major requests,
  and allowing the relative key only lifts that to 20.0% against 55.9%.
- Tempo is coarse: 55.9% within 10%, and generated tempos tend to clump near 121 and 148 BPM whatever was asked.
- Bass is often dropped. Instrument recall averages 74.2% with precision 93.7%, so when the model does add an
  instrument it usually is the right one, but it leaves parts out and bass is the usual casualty.
- About a minute of music per generation (2,048-token context), and sparse arrangements: about 3 tracks and
  136 beats on average, well under real Lakh songs.
- No genre control in this release, and the reserved style token at id 8255 is unused.

### Data and license

Code and weights: MIT.

Training data: Lakh MIDI Dataset (CC BY 4.0), credit Colin Raffel, *Learning-Based Methods for Comparing Sequences,
with Applications to Audio-to-MIDI Alignment and Matching*, PhD thesis, Columbia University, 2016.

Lakh is a crowd-sourced collection of MIDI transcriptions of real songs, so outputs can echo familiar material.
Check anything before releasing it commercially.

## License

Code in this repository: MIT (see [`LICENSE`](LICENSE)). Both models were trained on the Lakh MIDI Dataset
(CC BY 4.0), credit: Colin Raffel, *Learning-Based Methods for Comparing Sequences, with Applications to
Audio-to-MIDI Alignment and Matching*, PhD thesis, Columbia University, 2016. Since Lakh is a crowd-sourced
collection of MIDI transcriptions of real songs, check generated material before releasing it commercially.
