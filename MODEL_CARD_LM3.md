---
license: mit
library_name: pytorch
pipeline_tag: other
tags:
  - midi
  - music-generation
  - symbolic-music
  - transformer
  - gpt
  - miditok
---

# MIDI GPT (LM 3)

AI-assisted with Claude; creative and technical decisions are Stavion's.

An 18.2M-parameter GPT that writes multi-instrument MIDI to a request: tempo, time signature, key, instrument
families and note density. It is LM 2 fine-tuned to follow a short control prefix, with the context stretched from
1,024 to 2,048 tokens.

## What it is

You ask for something like "120 BPM, A minor, drums + bass + guitar + piano, medium density" and it writes the
arrangement. All tracks of a song are one token stream (MidiTok REMI + BPE, 8,192 base tokens, instrument tokens say
who plays), and the request is a short prefix of 64 extra control tokens in front of that stream.

- Base model: LM 2, 17.7M parameters trained from scratch
  ([Stavion-Colquitt/MIDI-GPT-LM2](https://huggingface.co/Stavion-Colquitt/MIDI-GPT-LM2)).
- LM 3 keeps 8 layers, 6 heads and d_model 384, grows the vocabulary to 8,256 for the controls, and grows the
  context from 1,024 to 2,048 tokens, which is about a minute of music.

## How it works

The controls are 64 new tokens with ids 8192-8255, defined in `lm3_vocab.py`: 8 tempo bins, 6 time signatures,
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

## Training

- Data: Lakh MIDI Dataset, lmd_full split. 170,570 training songs, about 1.24B training tokens.
- 57,289 steps, about 12 hours on one RTX 4060, fp16, 2,048-token windows.
- Best checkpoint: step 53,000, validation loss 1.029. That checkpoint is the model published here.

## Evaluation

60 random but seeded control requests, 1,536 tokens each, temperature 0.9, top_k 50, seed = request index, generated
with the published checkpoint. Every output was decoded back to MIDI and its controls were measured from the MIDI
file itself with `lm3_vocab.controls_from_score`, then compared with what had been asked for.

All 60 requests (59 decoded):

| metric | rate |
|---|---|
| decoded | 98.3% |
| tempo within 10% | 55.9% |
| exact key | 37.3% |
| key allowing relative major/minor | 40.7% |
| all requested instruments present | 42.4% |
| instrument recall (mean) | 74.2% |
| instrument precision (mean) | 93.7% |
| density bin exact | 42.4% |
| density bin within 1 | 86.4% |
| mean beats generated | 136.1 |
| mean tracks | 3.0 |

Split by the mode of the key that was asked for, which is where control is weakest:

| metric | major keys (34 requests) | minor keys (26 requests) |
|---|---|---|
| decoded | 100.0% | 96.2% |
| tempo within 10% | 61.8% | 48.0% |
| exact key | 52.9% | 16.0% |
| key allowing relative major/minor | 55.9% | 20.0% |
| all requested instruments present | 55.9% | 24.0% |
| instrument recall (mean) | 81.1% | 64.7% |
| instrument precision (mean) | 95.8% | 90.7% |
| density bin exact | 35.3% | 52.0% |
| density bin within 1 | 82.4% | 92.0% |
| mean beats generated | 137.4 | 134.3 |
| mean tracks | 3.2 | 2.7 |

Six of the 60 songs are in `examples/`, with every requested instrument present: `req_31.mid`, `req_41.mid` and
`req_56.mid`, the clearest of the set, plus `req_11.mid`, `req_24.mid` and `req_32.mid`. The file names are the
request ids from the evaluation, and nothing else was filtered.

## Usage

```
pip install torch miditok symusic safetensors
python generate.py --bpm 120 --key "A minor" --instruments DRUMS,Bass,Guitar,Piano --density 3 --n 4 --tokens 1536 --temp 0.9 --seed 0
python check_midi.py "generated/*.mid"
```

`--density` is a bin from 0 to 5 (edges 1, 2, 4, 8, 16 notes per beat across all tracks). `--key` is one of the
24 keys with "major" or "minor", for example `"Eb major"` or `"A minor"`. `--instruments` is a comma separated
subset of Piano, ChromPerc, Organ, Guitar, Bass, Strings, Ensemble, Brass, Reed, Pipe, SynthLead, SynthPad, SynthFX,
Ethnic, Percussive, SFX and DRUMS. The script prints the request next to what was measured in the generated file and
runs on CPU or CUDA.

| | |
|---|---|
| Architecture | decoder-only transformer, 8 layers, 6 heads, d_model 384, context 2,048, tied embeddings |
| Tokenizer | MidiTok REMI + BPE, 8,192 tokens plus 64 control tokens (`tokenizer.json`); end-of-song id 2 |
| Files | `model.safetensors` (fp32), `config.json`, `tokenizer.json`, `model.py`, `lm3_vocab.py`, `generate.py`, `check_midi.py`, `examples/` |

## Limitations

- Minor keys drift to major: 16.0% of minor requests came back in the exact key, against 52.9% for major requests,
  and allowing the relative key only lifts that to 20.0% against 55.9%.
- Tempo is coarse: 55.9% within 10%, and generated tempos tend to clump near 121 and 148 BPM whatever was asked.
- Bass is often dropped. Instrument recall averages 74.2% with precision 93.7%, so when the model does add an
  instrument it usually is the right one, but it leaves parts out and bass is the usual casualty.
- About a minute of music per generation (2,048-token context), and sparse arrangements: about 3 tracks and
  136 beats on average, well under real Lakh songs.
- No genre control in this release, and the reserved style token at id 8255 is unused.

## License and data

Code and weights: MIT.

Training data: Lakh MIDI Dataset (CC BY 4.0), credit Colin Raffel, *Learning-Based Methods for Comparing Sequences,
with Applications to Audio-to-MIDI Alignment and Matching*, PhD thesis, Columbia University, 2016.

Lakh is a crowd-sourced collection of MIDI transcriptions of real songs, so outputs can echo familiar material.
Check anything before releasing it commercially.

## Links

- LM 2: [huggingface.co/Stavion-Colquitt/MIDI-GPT-LM2](https://huggingface.co/Stavion-Colquitt/MIDI-GPT-LM2)
- Code and training details: [github.com/Stavion-Colquitt/MIDI-GPT](https://github.com/Stavion-Colquitt/MIDI-GPT)
