---
license: mit
library_name: pytorch
pipeline_tag: other
tags:
  - midi
  - music-generation
  - symbolic-music
  - gpt
  - from-scratch
  - miditok
---

# MIDI GPT (LM 2)

> **AI disclosure:** This project was built with AI assistance (Claude, by Anthropic) for writing code and
> documentation. All creative and technical decisions (model design, training data, evaluation) are mine, and I
> reviewed everything before publishing.

A 17.7M-parameter GPT trained from scratch to write multi-instrument MIDI. All tracks of a song are one token
stream (MidiTok REMI + BPE, 8,192 tokens, instrument tokens say who plays), so it writes whole arrangements.
Trained on the Lakh MIDI Dataset (1.24B tokens) on a single RTX 2060 in about 24 hours.
Code and training details: [github.com/Stavion-Colquitt/MIDI-GPT](https://github.com/Stavion-Colquitt/MIDI-GPT).

## Results

Full pass over all 19.8M validation tokens (Lakh files never trained on):

| Checkpoint | Val loss | Perplexity |
|---|---|---|
| Run 1 best (step 57,000) | 1.3420 | 3.83 |
| **This model**, extension best (step 112,000) | **1.3055** | **3.69** |

Generated songs vs real Lakh songs (3,072 tokens each, medians): 1.5 vs 7 instrument tracks, 4.8 vs 21.7 notes per
second, 6/8 vs 28/30 with drums, 8/8 valid MIDI. At temperature 0.85 it usually writes a band arrangement (drums,
bass, guitars, keys or strings); at 1.0 songs often end early. All 8 evaluation songs are in `examples/`, unfiltered.

## Use

```
pip install torch miditok symusic safetensors
python generate.py --n 4 --temp 0.85          # loads model.safetensors + config.json, writes generated/*.mid
python check_midi.py "generated/*.mid"        # tracks, notes, length, tempo of each file
```

| | |
|---|---|
| Architecture | decoder-only transformer, 8 layers, 6 heads, d_model 384, context 1,024, tied embeddings |
| Tokenizer | MidiTok REMI + BPE, 8,192 tokens (`tokenizer.json`); end-of-song token id 2 |
| Files | `model.safetensors` (fp32), `config.json`, `tokenizer.json`, `model.py`, `generate.py`, `check_midi.py`, `examples/` |

## Limitations

Small model, short context (about half a minute of a typical song), no conditioning (no genre, key or tempo
control). Arrangements are sparser than real songs. Lakh is a crowd-sourced collection of MIDI transcriptions of real
songs, so outputs can echo familiar patterns; check anything before releasing it.

## License and data

Code and weights: MIT. Training data: Lakh MIDI Dataset (CC BY 4.0), credit Colin Raffel, *Learning-Based Methods
for Comparing Sequences, with Applications to Audio-to-MIDI Alignment and Matching*, PhD thesis, Columbia
University, 2016.
