"""LM 2 evaluation: full deterministic pass over val.bin (all tokens, non-overlapping 1024 windows) for best.pt and
best_run1.pt, then MIDI samples from best.pt with simple music stats compared to real Lakh val excerpts.
    py -3.12 eval_lm2.py [--data DIR] [--n_samples 8] [--sample_tokens 3072]
Outputs go to lm2_eval/ (results.json, report.md, samples/*.mid, reference/*.mid)."""
import os, json, math, time, argparse, numpy as np, torch
from pathlib import Path
from model import GPT, Config

H = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(H, "lm2_eval")
ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.environ.get("LAKH_TOKENS", os.path.join(H, "data")))
ap.add_argument("--n_samples", type=int, default=8); ap.add_argument("--sample_tokens", type=int, default=3072)
ap.add_argument("--n_ref", type=int, default=30); ap.add_argument("--batch", type=int, default=16)
a = ap.parse_args()
DEV = torch.device("cuda"); os.makedirs(os.path.join(OUT, "samples"), exist_ok=True); os.makedirs(os.path.join(OUT, "reference"), exist_ok=True)
meta = json.load(open(os.path.join(a.data, "meta.json"))); eot = meta["eot"]
val = np.memmap(os.path.join(a.data, "val.bin"), dtype=np.uint16, mode="r")


def load(name):
    ck = torch.load(os.path.join(H, "checkpoints", name), map_location=DEV, weights_only=False)
    m = GPT(Config(**ck["cfg"])).to(DEV); m.load_state_dict(ck["model"]); m.eval()
    return m, ck


@torch.no_grad()
def full_val(m, block):
    n_win = (len(val) - 1) // block; tot, cnt = 0.0, 0
    for s in range(0, n_win, a.batch):
        ix = range(s, min(s + a.batch, n_win))
        x = torch.from_numpy(np.stack([val[i * block:i * block + block] for i in ix]).astype(np.int64)).to(DEV)
        y = torch.from_numpy(np.stack([val[i * block + 1:i * block + 1 + block] for i in ix]).astype(np.int64)).to(DEV)
        with torch.autocast("cuda", dtype=torch.float16): _, loss = m(x, y)
        tot += loss.item() * y.numel(); cnt += y.numel()
    return tot / cnt, cnt


res = {"val_tokens": int(len(val)), "models": {}}
for name in ["best.pt", "best_run1.pt"]:
    if not os.path.exists(os.path.join(H, "checkpoints", name)): continue
    m, ck = load(name); t = time.time(); vl, n = full_val(m, ck["cfg"]["block"])
    res["models"][name] = {"step": ck.get("step"), "val_loss_at_save": ck.get("val_loss"), "full_val_loss": vl,
                           "full_val_ppl": math.exp(vl), "tokens_scored": n}
    print(f"{name}: step {ck.get('step')} | full-pass val loss {vl:.4f} | ppl {math.exp(vl):.2f} | {n:,} tokens | {time.time()-t:.0f}s", flush=True)
    del m; torch.cuda.empty_cache()

from miditok import REMI
tok = REMI(params=Path(os.path.join(H, "tokenizer.json")))


def stats(ids, path):
    sc = tok.decode(ids); sc.dump_midi(path); s = sc.to("second")
    notes = [(n.pitch, n.time, n.duration) for tr in s.tracks if not tr.is_drum for n in tr.notes]
    dur = max((n.time + n.duration for tr in s.tracks for n in tr.notes), default=0.0)
    pcs = np.bincount([p % 12 for p, _, _ in notes], minlength=12).astype(float)
    ent = float(-(pcs[pcs > 0] / pcs.sum() * np.log2(pcs[pcs > 0] / pcs.sum())).sum()) if pcs.sum() else 0.0
    return {"seconds": round(dur, 1), "tracks": len(s.tracks), "drums": any(tr.is_drum for tr in s.tracks),
            "notes_per_s": round(sum(len(tr.notes) for tr in s.tracks) / max(dur, 1e-6), 2),
            "pitch_range": (max(p for p, _, _ in notes) - min(p for p, _, _ in notes)) if notes else 0,
            "pc_entropy_bits": round(ent, 2)}


def summarize(rows):
    ok = [r for r in rows if "error" not in r]
    agg = {"decoded": f"{len(ok)}/{len(rows)}"}
    for k in ["seconds", "tracks", "notes_per_s", "pitch_range", "pc_entropy_bits"]:
        v = [r[k] for r in ok]; agg[k] = f"{np.median(v):.2f} (IQR {np.percentile(v,25):.2f}-{np.percentile(v,75):.2f})" if v else "-"
    agg["with_drums"] = f"{sum(r['drums'] for r in ok)}/{len(ok)}"; return agg


# Real reference excerpts: val songs with at least sample_tokens tokens, first sample_tokens tokens each.
ends = np.flatnonzero(np.asarray(val) == eot); starts = np.concatenate([[0], ends[:-1] + 1])
long = [(s, e) for s, e in zip(starts, ends) if e - s >= a.sample_tokens]
rng = np.random.default_rng(0); pick = rng.choice(len(long), size=min(a.n_ref, len(long)), replace=False)
ref = []
for j, i in enumerate(pick):
    s, _ = long[i]
    try: ref.append(stats(val[s:s + a.sample_tokens].astype(int).tolist(), os.path.join(OUT, "reference", f"ref_{j:02d}.mid")))
    except Exception as ex: ref.append({"error": str(ex)})

m, ck = load("best.pt"); gen = []
for k in range(a.n_samples):
    temp = 0.85 if k < a.n_samples // 2 else 1.0; torch.manual_seed(100 + k); t = time.time()
    with torch.autocast("cuda", dtype=torch.float16):
        g = m.generate(torch.tensor([[eot]], device=DEV), a.sample_tokens, temperature=temp, eot=eot)[0, 1:].tolist()
    if g and g[-1] == eot: g = g[:-1]
    name = f"lm2_t{temp:.2f}_{k}.mid"
    try: r = stats(g, os.path.join(OUT, "samples", name)); r.update(file=name, temp=temp, tokens=len(g))
    except Exception as ex: r = {"file": name, "temp": temp, "tokens": len(g), "error": str(ex)}
    gen.append(r); print(f"sample {name}: {r} ({time.time()-t:.0f}s)", flush=True)

res["samples"] = gen; res["reference"] = ref
res["summary"] = {"generated": summarize(gen), "real_lakh_val": summarize(ref)}
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)
with open(os.path.join(OUT, "report.md"), "w", encoding="utf-8") as f:
    f.write("# LM 2 evaluation\n\n| checkpoint | step | full-pass val loss | ppl |\n|---|---|---|---|\n")
    for k, v in res["models"].items(): f.write(f"| {k} | {v['step']} | {v['full_val_loss']:.4f} | {v['full_val_ppl']:.2f} |\n")
    f.write(f"\nScored {res['val_tokens']:,} val tokens (Lakh files with MD5 prefix 00-03, never trained on).\n\n")
    f.write(f"## Generated vs real ({a.sample_tokens} tokens each)\n\n| stat | generated | real Lakh val |\n|---|---|---|\n")
    for k in res["summary"]["generated"]: f.write(f"| {k} | {res['summary']['generated'][k]} | {res['summary']['real_lakh_val'][k]} |\n")
print("SUMMARY", json.dumps(res["summary"], indent=1)); print("DONE")
