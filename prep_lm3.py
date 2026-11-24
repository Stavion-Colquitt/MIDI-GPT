"""LM 3 data prep: Lakh MIDI -> [control tokens, CTRL_END, REMI+BPE song tokens, EOS] per song, plus a song index.

Uses LM 2's tokenizer.json unchanged (ids 0-8191); control tokens are new ids 8192-8255 (see lm3_vocab.py).
Same file split as LM 2 (MD5 names 00-03 -> val), so val songs were never trained on by LM 2 either.
    py -3.12 prep_lm3.py --test 200                 # time + sanity check, writes nothing
    py -3.12 prep_lm3.py --workers 6                # full run -> $LM3_TOKENS (default D:\\Datasets\\lakh\\tokens_lm3)
Outputs: {train,val}.bin (uint16 records), {train,val}_idx.npy (int64 rows: start, prefix_len, song_len),
         controls_{train,val}.jsonl (readable controls per song), meta.json
"""
import os, sys, json, time, random, argparse, collections, numpy as np
from pathlib import Path
from multiprocessing import Pool

H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
from lm3_vocab import controls_from_score, control_ids, VOCAB, CTRL_END
SRC = os.environ.get("LAKH_MIDI", r"D:\Datasets\lakh\lmd_full")
OUT = os.environ.get("LM3_TOKENS", r"D:\Datasets\lakh\tokens_lm3")
TOK = os.path.join(H, "tokenizer.json"); MAX_BYTES = 1_000_000; MIN_TOKENS = 64

_tok = None
def _init():
    global _tok
    from miditok import REMI
    _tok = REMI(params=Path(TOK))


def work(path):
    try:
        if os.path.getsize(path) > MAX_BYTES: return path, None, None
        from symusic import Score
        s = Score(path); c = controls_from_score(s)
        if c is None: return path, None, None
        seq = _tok.encode(s)
        if isinstance(seq, list): seq = seq[0]
        ids = seq.ids
        return (path, ids, c) if len(ids) >= MIN_TOKENS else (path, None, None)
    except Exception:
        return path, None, None


def is_val(p): return int(os.path.basename(p)[:2], 16) < 4


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--test", type=int, default=0); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    files = sorted(str(p) for p in Path(SRC).rglob("*.mid")); print(f"{len(files):,} MIDI files", flush=True)
    random.seed(0)
    if a.test: files = random.sample(files, a.test)
    if a.limit: files = random.sample(files, a.limit)      # small real output (for smoke-testing the trainer)
    _init(); eos = _tok.vocab["EOS_None"]; assert len(_tok) == 8192, len(_tok)
    os.makedirs(OUT, exist_ok=True)
    bins = None if a.test else {s: open(os.path.join(OUT, f"{s}.bin"), "wb") for s in ("train", "val")}
    ctrl_f = None if a.test else {s: open(os.path.join(OUT, f"controls_{s}.jsonl"), "w", encoding="utf-8") for s in ("train", "val")}
    idx = {"train": [], "val": []}; pos = {"train": 0, "val": 0}; plens = collections.Counter()
    t = time.time(); n_ok = 0
    with Pool(a.workers, initializer=_init) as pool:
        for i, (p, ids, c) in enumerate(pool.imap_unordered(work, files, chunksize=16), 1):
            if ids is not None:
                n_ok += 1; sp = "val" if is_val(p) else "train"
                pre = control_ids(c) + [CTRL_END]; rec = pre + list(ids) + [eos]
                idx[sp].append((pos[sp], len(pre), len(ids) + 1)); pos[sp] += len(rec); plens[len(pre)] += 1
                if bins:
                    bins[sp].write(np.asarray(rec, dtype=np.uint16).tobytes())
                    ctrl_f[sp].write(json.dumps({"file": os.path.basename(p), **c}) + "\n")
            if i % 5000 == 0 or i == len(files):
                el = time.time() - t
                print(f"{i:,}/{len(files):,} files, {n_ok:,} kept, {pos['train'] / 1e6:.0f}M train tokens, "
                      f"{el / 60:.1f} min, ETA {el / i * (len(files) - i) / 60:.0f} min", flush=True)
    el = time.time() - t
    if a.test:
        print(f"TEST: {n_ok}/{len(files)} kept, {el / len(files) * 1000:.1f} ms/file -> full set ~"
              f"{el / len(files) * 178561 / 60:.0f} min at {a.workers} workers; prefix lengths {dict(sorted(plens.items()))}", flush=True)
        return
    for f in list(bins.values()) + list(ctrl_f.values()): f.close()
    for sp in idx: np.save(os.path.join(OUT, f"{sp}_idx.npy"), np.asarray(idx[sp], dtype=np.int64))
    meta = {"vocab": VOCAB, "base_vocab": 8192, "eot": eos, "ctrl_end": CTRL_END, "train_tokens": pos["train"],
            "val_tokens": pos["val"], "train_songs": len(idx["train"]), "val_songs": len(idx["val"]),
            "prefix_len_counts": dict(sorted(plens.items())), "tokenizer": "MidiTok REMI + BPE (LM 2) + 64 control tokens",
            "source": "Lakh MIDI lmd_full (CC BY 4.0)", "minutes": round(el / 60, 1)}
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1); print("meta:", meta, flush=True); print("PREP DONE", flush=True)


if __name__ == "__main__":
    main()
