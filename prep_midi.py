"""LM 2 data prep: Lakh MIDI (lmd_full, CC BY 4.0) -> REMI + BPE tokens -> data/train.bin, data/val.bin, data/meta.json.

All tracks of a song go into one token stream (Program tokens say which instrument plays), so the model learns full
arrangements, not single parts. Split is by file (Lakh files are named by MD5): names starting 00-03 -> val (~1.6%).
    py -3.12 prep_midi.py --test 300      # time a few hundred files, write nothing
    py -3.12 prep_midi.py                 # full run (train BPE on 10k train files, then encode everything)
"""
import os, sys, json, time, random, argparse, numpy as np
from pathlib import Path
from multiprocessing import Pool

H = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(H, "data")
SRC = os.environ.get("LAKH_MIDI", os.path.join(D, "lmd_full"))   # extracted lmd_full folder (set LAKH_MIDI to keep it elsewhere)
TOK = os.path.join(H, "tokenizer.json")
VOCAB = 8192; MAX_BYTES = 1_000_000; MIN_TOKENS = 64


def make_tokenizer():
    from miditok import REMI, TokenizerConfig
    cfg = TokenizerConfig(num_velocities=16, use_programs=True, one_token_stream_for_programs=True,
                          use_tempos=True, use_time_signatures=True, use_chords=False, use_rests=False,
                          special_tokens=["PAD", "BOS", "EOS", "MASK"])
    return REMI(cfg)


_tok = None
def _init():
    global _tok
    from miditok import REMI
    _tok = REMI(params=Path(TOK))


def encode(path):
    try:
        if os.path.getsize(path) > MAX_BYTES: return path, None
        from symusic import Score
        seq = _tok.encode(Score(path))
        if isinstance(seq, list): seq = seq[0]
        ids = seq.ids
        return path, (ids if len(ids) >= MIN_TOKENS else None)
    except Exception:
        return path, None


def is_val(p): return int(os.path.basename(p)[:2], 16) < 4


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--test", type=int, default=0); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    files = sorted(str(p) for p in Path(SRC).rglob("*.mid")); print(f"{len(files):,} MIDI files", flush=True)
    random.seed(0)
    if not os.path.exists(TOK):
        tok = make_tokenizer(); t = time.time()
        sample = random.sample([f for f in files if not is_val(f) and os.path.getsize(f) <= MAX_BYTES], 10000)
        tok.train(vocab_size=VOCAB, model="BPE", files_paths=[Path(f) for f in sample])
        tok.save(Path(TOK)); print(f"BPE tokenizer trained ({VOCAB}) in {time.time() - t:.0f} s", flush=True)
    if a.test: files = random.sample(files, a.test)
    t = time.time(); n_ok = 0; counts = {"train": 0, "val": 0}
    out = None if a.test else {s: open(os.path.join(D, f"{s}.bin"), "wb") for s in ("train", "val")}
    _init(); eos = _tok.vocab["EOS_None"]
    with Pool(a.workers, initializer=_init) as pool:
        for i, (p, ids) in enumerate(pool.imap_unordered(encode, files, chunksize=16), 1):
            if ids is not None:
                n_ok += 1; s = "val" if is_val(p) else "train"; counts[s] += len(ids) + 1
                if out: out[s].write(np.asarray(list(ids) + [eos], dtype=np.uint16).tobytes())
            if i % 5000 == 0 or i == len(files):
                el = time.time() - t
                print(f"{i:,}/{len(files):,} files, {n_ok:,} kept, {counts['train'] / 1e6:.0f}M train tokens, "
                      f"{el / 60:.1f} min, ETA {el / i * (len(files) - i) / 60:.0f} min", flush=True)
    if a.test:
        el = time.time() - t
        print(f"TEST: {n_ok}/{len(files)} kept, {sum(counts.values()) / max(n_ok, 1):.0f} tokens/song, "
              f"{el / len(files) * 1000:.1f} ms/file -> full set ~{el / len(files) * 178000 / 60:.0f} min", flush=True)
        return
    for f in out.values(): f.close()
    meta = {"vocab": len(_tok), "eot": eos, "train_tokens": counts["train"], "val_tokens": counts["val"],
            "songs": n_ok, "tokenizer": "MidiTok REMI + BPE", "source": "Lakh MIDI lmd_full (CC BY 4.0)"}
    json.dump(meta, open(os.path.join(D, "meta.json"), "w"), indent=1); print("meta:", meta, flush=True)


if __name__ == "__main__":
    main()
