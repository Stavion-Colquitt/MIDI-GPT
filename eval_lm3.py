"""LM 3 control-following eval: 60 seeded requests, generate, decode back, measure.

Reuses lm3_vocab.py (request(), controls_from_score(), KEYS, FAMILIES, TEMPO_NAMES) and model.py (GPT, Config).
Generation loop copied from samples() in train_lm3.py: control tokens blocked, no end-of-song in the first 512
tokens, top_k 50, temperature 0.9, sliding window of 2048. Loads checkpoints_lm3/best.pt.

Writes:  lm3_eval/req_XX.mid, lm3_eval/results.json, lm3_eval/summary.md
    py -3.12 eval_lm3.py               # all 60 requests
    py -3.12 eval_lm3.py --limit 2     # smoke test on the first 2
"""
import os, sys, json, time, random, argparse, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
from model import GPT, Config
import lm3_vocab as V

EVAL = os.path.join(H, "lm3_eval")
CK = os.path.join(H, "checkpoints_lm3", "best.pt")
BLOCK, TEMPERATURE, TOP_K, EOS = 2048, 0.9, 50, 2
OTHERS = ["Piano", "Guitar", "Strings", "SynthPad", "SynthLead", "Organ", "Brass"]
N_REQ, N_TOKENS = 60, 1536

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=N_REQ, help="only the first N requests (smoke test)")
ap.add_argument("--tokens", type=int, default=N_TOKENS)
ap.add_argument("--ckpt", default=CK)
args = ap.parse_args()

DEV = torch.device("cuda")
os.makedirs(EVAL, exist_ok=True)


def build_requests(n=N_REQ):
    """60 seeded requests: tempo 70-175, one of the 24 keys, 4/4, DRUMS+Bass+1-2 others, density_bin 1-4."""
    random.seed(0); reqs = []
    for i in range(n):
        tempo = round(random.uniform(70.0, 175.0), 1)
        key = random.choice([k for k in V.KEYS if k != "none"])
        extra = random.sample(OTHERS, random.choice([1, 2]))
        reqs.append({"index": i, "tempo_bpm": tempo, "key": key, "time_sig": "4/4",
                     "instruments": ["DRUMS", "Bass"] + extra, "density_bin": random.choice([1, 2, 3, 4])})
    return reqs


@torch.no_grad()
def generate(model, prefix, n_tokens, seed):
    """The samples() generation loop from train_lm3.py, one request at a time."""
    idx = torch.tensor([prefix], device=DEV); body = []; ended = False
    torch.manual_seed(seed)
    for i in range(n_tokens):
        with torch.autocast("cuda", dtype=torch.float16):
            logits, _ = model(idx[:, -BLOCK:])
        lg = logits[:, -1].float() / TEMPERATURE
        lg[:, V.BASE:] = -float("inf")                       # never generate control tokens
        if i < 512: lg[:, EOS] = -float("inf")               # no instant endings
        v, _ = torch.topk(lg, TOP_K); lg[lg < v[:, [-1]]] = -float("inf")
        nxt = torch.multinomial(F.softmax(lg, -1), 1)
        if nxt.item() == EOS: ended = True; break
        body.append(nxt.item()); idx = torch.cat([idx, nxt], 1)
    return body, ended


def rel_key(key):
    """Relative major/minor of a key: A minor <-> C major."""
    name, mode = key.rsplit(" ", 1); pc = V.NAMES.index(name)
    return f"{V.NAMES[(pc + 3) % 12]} major" if mode == "minor" else f"{V.NAMES[(pc + 9) % 12]} minor"


def score(req, got, beats, tracks):
    s = {"decoded": got is not None, "beats": beats, "tracks": tracks}
    if got is None:
        s.update(score_val=0.0)
        return s
    want, have = set(req["instruments"]), set(got["instruments"])
    hit = want & have
    s["tempo_err_bpm"] = round(got["tempo"] - req["tempo_bpm"], 1)
    s["tempo_ok_10pct"] = abs(s["tempo_err_bpm"]) <= 0.10 * req["tempo_bpm"]
    s["key_exact"] = got["key"] == req["key"]
    s["key_relative_ok"] = s["key_exact"] or got["key"] == rel_key(req["key"])
    s["inst_hit"] = [f for f in V.FAMILIES if f in hit]
    s["inst_missing"] = [f for f in V.FAMILIES if f in want - have]
    s["inst_extra"] = [f for f in V.FAMILIES if f in have - want]
    s["inst_all_present"] = not (want - have)
    s["inst_recall"] = round(len(hit) / len(want), 3)
    s["inst_precision"] = round(len(hit) / len(have), 3) if have else 0.0
    s["density_exact"] = got["density_bin"] == req["density_bin"]
    s["density_within1"] = abs(got["density_bin"] - req["density_bin"]) <= 1
    s["score_val"] = round(sum([s["tempo_ok_10pct"], s["key_exact"], s["key_relative_ok"], s["inst_recall"],
                                s["inst_precision"], s["density_exact"], s["density_within1"]]), 3)
    return s


def pct(x): return f"{100.0 * x:.1f}%"
def mean(v): return sum(v) / len(v) if v else 0.0


def summarise(reqs, res):
    dec = [r for r in res if r["scores"]["decoded"]]
    def block(rows, title):
        n = len(rows); g = [r["scores"] for r in rows if r["scores"]["decoded"]]; nd = len(g)
        if n == 0: return f"\n**{title}**: no requests\n"
        out = [f"\n**{title}** ({n} requests, {nd} decoded)\n",
               "| metric | rate |", "|---|---|",
               f"| decoded | {pct(nd / n)} |"]
        if nd:
            out += [f"| tempo within 10% | {pct(mean([x['tempo_ok_10pct'] for x in g]))} |",
                    f"| exact key | {pct(mean([x['key_exact'] for x in g]))} |",
                    f"| key allowing relative major/minor | {pct(mean([x['key_relative_ok'] for x in g]))} |",
                    f"| all requested instruments present | {pct(mean([x['inst_all_present'] for x in g]))} |",
                    f"| instrument recall (mean) | {pct(mean([x['inst_recall'] for x in g]))} |",
                    f"| instrument precision (mean) | {pct(mean([x['inst_precision'] for x in g]))} |",
                    f"| density_bin exact | {pct(mean([x['density_exact'] for x in g]))} |",
                    f"| density_bin within 1 | {pct(mean([x['density_within1'] for x in g]))} |",
                    f"| mean beats generated | {mean([x['beats'] for x in g]):.1f} |",
                    f"| mean tracks | {mean([x['tracks'] for x in g]):.1f} |"]
        return "\n".join(out) + "\n"

    md = [f"# LM 3 control-following eval\n",
          f"- checkpoint: `{os.path.basename(str(args.ckpt))}`",
          f"- requests: {len(reqs)}, {args.tokens} tokens each, seed = request index, top_k {TOP_K}, "
          f"temperature {TEMPERATURE}, context {BLOCK}\n",
          "## Overall\n", block(res, "All requests")]
    by_tempo = {}
    for r in res: by_tempo.setdefault(V.TEMPO_NAMES[r["asked_tempo_bin"]], []).append(r)
    md.append("\n## By requested tempo range\n")
    for name in V.TEMPO_NAMES:
        if name in by_tempo: md.append(block(by_tempo[name], f"{name} BPM"))
    for mode in ("major", "minor"):
        rows = [r for r in res if r["asked"]["key"].endswith(mode)]
        md.append("\n## By key mode\n" if mode == "major" else "")
        md.append(block(rows, f"{mode} keys"))
    return "\n".join(md)


def main():
    from miditok import REMI
    from symusic import Score
    t0 = time.time()
    ck = torch.load(args.ckpt, map_location=DEV, weights_only=False)
    cfg = Config(**ck["cfg"]); model = GPT(cfg); model.load_state_dict(ck["model"]); model.to(DEV); model.eval()
    tok = REMI(params=Path(os.path.join(H, "tokenizer.json")))
    print(f"device: {torch.cuda.get_device_name(0)} | vocab {cfg.vocab} | ctx {cfg.block} | "
          f"best.pt step {ck.get('step')} val {ck.get('val_loss')}", flush=True)
    reqs = build_requests()[:args.limit]
    res = []
    for rq in reqs:
        pre = V.request(tempo_bpm=rq["tempo_bpm"], time_sig=rq["time_sig"], key=rq["key"],
                        instruments=tuple(rq["instruments"]), density_bin=rq["density_bin"])
        body, ended = generate(model, pre, args.tokens, rq["index"])
        path = os.path.join(EVAL, f"req_{rq['index']:02d}.mid"); got = None; beats = tracks = 0
        try:
            tok.decode(body).dump_midi(path)
            sc = Score(path); c = V.controls_from_score(sc)
            if sc is not None: beats = round(sc.end() / sc.ticks_per_quarter, 2)
            tracks = sum(1 for tr in sc.tracks if len(tr.notes))
            got = c
        except Exception as e:
            print(f"  req_{rq['index']:02d}: could not decode ({e})", flush=True)
        s = score(rq, got, beats, tracks)
        res.append({"id": f"req_{rq['index']:02d}", "asked": rq, "asked_tempo_bin": int(np.searchsorted(
            V.TEMPO_EDGES, rq["tempo_bpm"], side="right")), "got": got, "tokens": len(body), "ended": ended,
            "scores": s})
        print(f"[{len(res)}/{len(reqs)}] {rq['index']:02d} asked {rq['tempo_bpm']}BPM {rq['key']} "
              f"{'+'.join(rq['instruments'])} d{rq['density_bin']} | got "
              f"{(got or {}).get('tempo')}BPM {(got or {}).get('key')} {(got or {}).get('instruments')} "
              f"d{(got or {}).get('density_bin')} | score {s['score_val']} | {len(body)} tok, {beats} beats, "
              f"{tracks} tracks | {time.time() - t0:.0f}s", flush=True)

    json.dump({"checkpoint": os.path.basename(str(args.ckpt)), "step": ck.get("step"),
               "val_loss": ck.get("val_loss"), "tokens_per_request": args.tokens, "top_k": TOP_K,
               "temperature": TEMPERATURE, "block": BLOCK, "n_requests": len(res),
               "elapsed_s": round(time.time() - t0, 1), "results": res},
              open(os.path.join(EVAL, "results.json"), "w", encoding="utf-8"), indent=1)
    md = summarise(reqs, res); open(os.path.join(EVAL, "summary.md"), "w", encoding="utf-8").write(md)
    print("\n=== summary.md ===\n" + md, flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
