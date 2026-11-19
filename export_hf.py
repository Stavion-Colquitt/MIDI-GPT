"""Package checkpoints/best.pt for Hugging Face: model.safetensors + config.json + tokenizer.json + code + card + examples.

    py -3.12 export_hf.py            -> writes hf_export/ (fp32 weights, no optimizer state)
"""
import os, json, glob, shutil, torch
from safetensors.torch import save_file

H = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(H, "hf_export")
ck = torch.load(os.path.join(H, "checkpoints", "best.pt"), map_location="cpu", weights_only=False)
sd = {k: v.contiguous() for k, v in ck["model"].items() if k != "head.weight"}   # head is tied to tok.weight
os.makedirs(os.path.join(OUT, "examples"), exist_ok=True)
save_file(sd, os.path.join(OUT, "model.safetensors"), metadata={"format": "pt"})
cfg = dict(ck["cfg"]); cfg.update({"architecture": "decoder-only GPT, tied embeddings", "eot": 2, "step": ck["step"],
                                   "tokenizer": "MidiTok REMI + BPE (tokenizer.json)",
                                   "val_loss_full_pass": 1.3055, "val_ppl_full_pass": 3.69})
json.dump(cfg, open(os.path.join(OUT, "config.json"), "w"), indent=2)
for f in ("tokenizer.json", "model.py", "generate.py", "check_midi.py"): shutil.copy(os.path.join(H, f), os.path.join(OUT, f))
shutil.copy(os.path.join(H, "MODEL_CARD.md"), os.path.join(OUT, "README.md"))
for f in glob.glob(os.path.join(H, "examples", "*.mid")): shutil.copy(f, os.path.join(OUT, "examples"))
n = sum(v.numel() for v in sd.values())
print(f"exported step {ck['step']} (val loss at save {ck['val_loss']:.3f}), {n / 1e6:.1f}M params -> {OUT}")
