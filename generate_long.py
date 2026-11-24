"""LM 3 step 1: longer LM 2 songs (generate.py left unchanged).

    py -3.12 generate_long.py --n 4 --tokens 6144 --min_tokens 2048

Fixes vs generate.py:
  * --min_tokens: the end-of-song token is blocked until this many tokens exist (no songs that end instantly).
  * sliding-window generation: past block size the model sees the last --ctx tokens (default = block),
    so songs can run longer than the 1024-token context. Window is re-cut every step (same as model.generate).
Written by the check-in 2026-11-20; uses checkpoints/best.pt like generate.py.
"""
import os, sys, json, argparse, torch, torch.nn.functional as F
from pathlib import Path
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
from model import GPT, Config

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=4); ap.add_argument("--temp", type=float, default=0.85)
ap.add_argument("--tokens", type=int, default=6144); ap.add_argument("--min_tokens", type=int, default=2048)
ap.add_argument("--top_k", type=int, default=50); ap.add_argument("--ctx", type=int, default=0)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default=os.path.join(H, "generated_long"))
a = ap.parse_args()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KEYS = ("vocab", "block", "n_layer", "n_head", "n_embd", "dropout")
c = torch.load(os.path.join(H, "checkpoints", "best.pt"), map_location="cpu", weights_only=False)
model = GPT(Config(**{k: c["cfg"][k] for k in KEYS})); model.load_state_dict(c["model"]); model.to(dev).eval()
eot = 2; block = c["cfg"]["block"]; ctx = min(a.ctx or block, block)


@torch.no_grad()
def gen(seed):
    torch.manual_seed(seed); idx = torch.tensor([[eot]], device=dev); out = []
    for i in range(a.tokens):
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            logits, _ = model(idx[:, -ctx:])
        logits = logits[:, -1].float() / a.temp
        if i < a.min_tokens: logits[:, eot] = -float("inf")
        v, _ = torch.topk(logits, a.top_k); logits[logits < v[:, [-1]]] = -float("inf")
        nxt = torch.multinomial(F.softmax(logits, -1), 1)
        if nxt.item() == eot: break
        out.append(nxt.item()); idx = torch.cat([idx, nxt], 1)
    return out


from miditok import REMI
tok = REMI(params=Path(os.path.join(H, "tokenizer.json"))); os.makedirs(a.out, exist_ok=True)
for k in range(a.n):
    g = gen(a.seed + k); path = os.path.join(a.out, f"long_t{a.temp:.2f}_s{a.seed + k}.mid")
    try:
        sc = tok.decode(g); sc.dump_midi(path)
        print(f"{path}: {len(g)} tokens, {sc.end() / sc.ticks_per_quarter:.0f} beats, {len(sc.tracks)} tracks")
    except Exception as e:
        print(f"seed {a.seed + k}: {len(g)} tokens, could not decode ({e})")
