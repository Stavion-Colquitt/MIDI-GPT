"""Generate MIDI songs with MIDI GPT (LM 3) from a control request: tempo, key, instruments and note density.

    py generate.py                                                # 4 songs, 120 BPM, A minor, band, density 3
    py generate.py --bpm 96 --key "Eb major" --instruments DRUMS,Bass,Guitar --density 2 --n 2
    py generate.py --n 1 --tokens 256 --temp 1.0 --seed 7

Loads model.safetensors + config.json from the folder next to this script (CPU or CUDA). Each song starts from the
control prefix built by lm3_vocab.request(); control tokens are never generated, end-of-song is blocked for the
first 512 tokens, sampling is top_k 50 in a sliding window of the config block size. Writes <out>/lm3_*.mid and
prints the controls that were asked for next to the ones measured in the generated file.
"""
import os, json, argparse, torch, torch.nn.functional as F
from pathlib import Path
from model import GPT, Config
import lm3_vocab as V

H = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--bpm", type=float, default=120.0)
ap.add_argument("--key", default="A minor", help="one of lm3_vocab.KEYS, e.g. 'A minor', 'Eb major'")
ap.add_argument("--time_sig", default="4/4", help="4/4, 3/4, 6/8, 2/4, 12/8 or other")
ap.add_argument("--instruments", default="DRUMS,Bass,Guitar,Piano",
                help="comma separated families from lm3_vocab.FAMILIES")
ap.add_argument("--density", type=int, default=3, help="note density bin 0-5 (edges 1,2,4,8,16 notes per beat)")
ap.add_argument("--n", type=int, default=4, help="how many songs")
ap.add_argument("--tokens", type=int, default=1536, help="max tokens per song (2048-token context)")
ap.add_argument("--temp", type=float, default=0.9)
ap.add_argument("--top_k", type=int, default=50)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=os.path.join(H, "generated"))
a = ap.parse_args()

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KEYS = ("vocab", "block", "n_layer", "n_head", "n_embd", "dropout")
from safetensors.torch import load_file
cfg = json.load(open(os.path.join(H, "config.json")))
model = GPT(Config(**{k: cfg[k] for k in KEYS}))
sd = load_file(os.path.join(H, "model.safetensors"))
sd["head.weight"] = sd["tok.weight"]                       # tied embeddings
model.load_state_dict(sd); model.to(dev).eval()
BLOCK, EOS = cfg["block"], cfg.get("eot", 2)
print(f"MIDI GPT (LM 3) step {cfg.get('step')}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params "
      f"on {dev.type}, context {BLOCK}, vocab {cfg['vocab']}")
from miditok import REMI
from symusic import Score
tok = REMI(params=Path(os.path.join(H, "tokenizer.json")))
instruments = tuple(s.strip() for s in a.instruments.split(",") if s.strip())
prefix = V.request(tempo_bpm=a.bpm, time_sig=a.time_sig, key=a.key, instruments=instruments, density_bin=a.density)
os.makedirs(a.out, exist_ok=True)


@torch.no_grad()
def generate(seed):
    """The samples() loop from train_lm3.py, for one request."""
    idx = torch.tensor([prefix], device=dev); body = []
    torch.manual_seed(seed)
    for i in range(a.tokens):
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            logits, _ = model(idx[:, -BLOCK:])
        lg = logits[:, -1].float() / a.temp
        lg[:, V.BASE:] = -float("inf")                     # never generate control tokens
        if i < 512: lg[:, EOS] = -float("inf")              # no instant endings
        v, _ = torch.topk(lg, a.top_k); lg[lg < v[:, [-1]]] = -float("inf")
        nxt = torch.multinomial(F.softmax(lg, -1), 1)
        if nxt.item() == EOS: break
        body.append(nxt.item()); idx = torch.cat([idx, nxt], 1)
    return body


for k in range(a.n):
    body = generate(a.seed + k)
    path = os.path.join(a.out, f"lm3_bpm{int(round(a.bpm))}_{a.key.replace(' ', '')}_d{a.density}_s{a.seed + k}.mid")
    try:
        tok.decode(body).dump_midi(path)
    except Exception as e:
        print(f"seed {a.seed + k}: {len(body)} tokens, could not decode ({e})"); continue
    s = Score(path)
    got = V.controls_from_score(s) or {}
    print(f"{path}: {len(body)} tokens, {s.end() / max(s.ticks_per_quarter, 1):.0f} beats")
    print(f"  asked:    {a.bpm} BPM, {a.time_sig}, {a.key}, {'+'.join(instruments)}, density bin {a.density}")
    print(f"  measured: {got.get('tempo')} BPM, {got.get('time_sig')}, {got.get('key')}, "
          f"{'+'.join(got.get('instruments', []))}, density {got.get('density')} (bin {got.get('density_bin')})")
