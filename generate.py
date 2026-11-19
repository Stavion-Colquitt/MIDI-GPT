"""Generate MIDI songs with LM 2 (each song starts from the end-of-song token, so it's written from scratch).

    py -3.12 generate.py                          # 4 songs, temperature 0.85, 3072 tokens each -> generated/
    py -3.12 generate.py --n 8 --temp 1.0 --tokens 4096 --seed 7

Loads checkpoints/best.pt if it exists, otherwise the Hugging Face files next to this script
(model.safetensors + config.json). Temperature 0.85 gave the most complete arrangements in our tests; at 1.0 songs
often end almost immediately.
"""
import os, json, argparse, torch
from pathlib import Path
from model import GPT, Config

H = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=4); ap.add_argument("--temp", type=float, default=0.85)
ap.add_argument("--tokens", type=int, default=3072); ap.add_argument("--top_k", type=int, default=50)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default=os.path.join(H, "generated"))
a = ap.parse_args()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KEYS = ("vocab", "block", "n_layer", "n_head", "n_embd", "dropout")


def load():
    ck = os.path.join(H, "checkpoints", "best.pt")
    if os.path.exists(ck):
        c = torch.load(ck, map_location="cpu", weights_only=False)
        m = GPT(Config(**{k: c["cfg"][k] for k in KEYS})); m.load_state_dict(c["model"]); eot = 2
    else:
        from safetensors.torch import load_file
        cfg = json.load(open(os.path.join(H, "config.json")))
        m = GPT(Config(**{k: cfg[k] for k in KEYS}))
        sd = load_file(os.path.join(H, "model.safetensors")); sd["head.weight"] = sd["tok.weight"]   # tied
        m.load_state_dict(sd); eot = cfg.get("eot", 2)
    return m.to(dev).eval(), eot


from miditok import REMI
tok = REMI(params=Path(os.path.join(H, "tokenizer.json")))
model, eot = load(); os.makedirs(a.out, exist_ok=True)
for k in range(a.n):
    torch.manual_seed(a.seed + k)
    with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
        g = model.generate(torch.tensor([[eot]], device=dev), a.tokens, temperature=a.temp, top_k=a.top_k, eot=eot)[0, 1:].tolist()
    if g and g[-1] == eot: g = g[:-1]
    path = os.path.join(a.out, f"lm2_t{a.temp:.2f}_s{a.seed + k}.mid")
    try:
        tok.decode(g).dump_midi(path); print(f"{path}: {len(g)} tokens")
    except Exception as e:
        print(f"seed {a.seed + k}: {len(g)} tokens, could not decode ({e})")
