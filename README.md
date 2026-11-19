# MIDI GPT (LM 2)

> **AI disclosure:** This project was built with AI assistance (Claude, by Anthropic) for writing code and
> documentation. All creative and technical decisions (model design, training data, evaluation) are mine, and I
> reviewed everything before publishing.

🤗 [Model on Hugging Face](https://huggingface.co/Stavion-Colquitt/MIDI-GPT-LM2)

A 17.7M-parameter GPT that writes multi-instrument MIDI from scratch. Same from-scratch transformer as my
[Tiny GPT (LM 1)](https://github.com/Stavion-Colquitt/Tiny-GPT), retrained on music instead of text: 1.24 billion
tokens of the Lakh MIDI dataset, on a single **RTX 2060 (6 GB)** in about 24 hours. Every track of a song goes into one
token stream (instrument tokens say who plays), so it learns whole arrangements: drums, bass, guitars, keys,
strings. It's step 1 of an "instrumental maker" that will write editable MIDI straight into an Ableton session.

---

## Results

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

## Model

| | |
|---|---|
| Architecture | decoder-only transformer, pre-norm, 8 layers, 6 heads, d_model 384, GELU MLP (4x) |
| Context | 1,024 tokens (about half a minute of a typical song), learned positional embeddings |
| Tokenizer | [MidiTok](https://github.com/Natooz/MidiTok) REMI + BPE, 8,192 tokens, trained on 10,000 training songs (`tokenizer.json`): one stream for all instruments, 16 velocity levels, tempo and time-signature tokens |
| Parameters | 17.7M (token embedding tied to the output head) |
| Attention | PyTorch `scaled_dot_product_attention` (causal) |

## Data

[Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/) `lmd_full` (Colin Raffel, CC BY 4.0): 173,283 songs kept
(files up to 1 MB that encode to at least 64 tokens), 1.236B training tokens and 19.8M validation tokens. The split
is by file name (MD5 hash): names starting 00 to 03 (about 1.6%) are validation.

## Training

- AdamW (0.9, 0.95), weight decay 0.1 on matrices only, gradient clipping 1.0
- 16 x 4 micro-batches of 1,024 tokens = 65,536 tokens per step; FP16 autocast + GradScaler (Turing has no BF16)
- Run 1: 500 warmup steps, cosine from 1e-3 to 1e-4, sized to a 12-hour budget: 60,763 steps
- Extension: resume from the last state, 300-step re-warm to 3e-4, cosine to 3e-5 over 12 hours, new data seed:
  112,767 steps total (7.4B tokens, about 6 passes over the data)
- Evaluated every 500 steps (val loss + 3 sample songs written as .mid), best checkpoint kept by val loss, MLflow

## Run it

```
py -3.12 -m pip install torch numpy miditok symusic mlflow safetensors huggingface_hub
# download lmd_full.tar.gz from https://colinraffel.com/projects/lmd/ and extract it to data/lmd_full
py -3.12 prep_midi.py                  # tokenizer + data/train.bin, data/val.bin, data/meta.json
py -3.12 train.py --hours 12           # run 1
py -3.12 train.py --resume --extend_hours 12 --seed 2   # extension
py -3.12 generate.py --n 4 --temp 0.85 # new songs in generated/
py -3.12 check_midi.py "generated/*.mid"
py -3.12 eval_lm2.py                   # full-pass val loss + music stats (lm2_eval/)
py -3.12 export_hf.py                  # safetensors + config + card for Hugging Face
```

`run_lm2.ps1` pins the run to one GPU on a multi-GPU machine (set `LAKH_TOKENS` if the token files live elsewhere).

## Limitations

A 17.7M-parameter model with a short context: it is a learning project, not a production composer. Arrangements
are sparse, structure rarely lasts beyond the context window, and there is no conditioning yet (you can't ask for a
genre, key or tempo). Lakh is a crowd-sourced collection of MIDI transcriptions of real songs, so generations can
echo familiar patterns from it; check anything before using it in a release.

## License

Code: MIT. The model was trained on the Lakh MIDI Dataset (CC BY 4.0), credit: Colin Raffel, *Learning-Based
Methods for Comparing Sequences, with Applications to Audio-to-MIDI Alignment and Matching*, PhD thesis, Columbia
University, 2016.
