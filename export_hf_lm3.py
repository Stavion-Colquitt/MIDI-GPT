"""Package checkpoints_lm3/best.pt (LM 3) for Hugging Face, the same way export_hf.py did LM 2.

Builds hf_export_lm3/ with:
  model.safetensors  fp32 weights (head.weight dropped, it is tied to tok.weight)
  config.json        vocab 8256, block 2048, control tokens 8192-8255
  tokenizer.json     the LM 1/LM 2 MidiTok REMI + BPE tokenizer, unchanged
  model.py           the GPT definition, unchanged
  lm3_vocab.py       control vocabulary, request() and controls_from_score(), unchanged
  generate_lm3.py    standalone generator (same file, published as generate.py)
  check_midi.py      copied from hf_export/
  README.md          model card (copy of MODEL_CARD_LM3.md)
  examples/          6 eval songs, all with every requested instrument present

    py -3.12 export_hf_lm3.py
"""
import os, json, shutil, torch
from safetensors.torch import save_file

H = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(H, "hf_export_lm3"); EX = os.path.join(H, "lm3_eval")
EXAMPLES = ["req_31", "req_41", "req_56",    # the eval's best
            "req_11", "req_24", "req_32"]    # three more with every requested instrument present

ck = torch.load(os.path.join(H, "checkpoints_lm3", "best.pt"), map_location="cpu", weights_only=False)
sd = {k: v.contiguous() for k, v in ck["model"].items() if k != "head.weight"}   # head is tied to tok.weight
os.makedirs(os.path.join(OUT, "examples"), exist_ok=True)
save_file(sd, os.path.join(OUT, "model.safetensors"), metadata={"format": "pt"})

cfg = {"vocab": 8256, "block": 2048, "n_layer": 8, "n_head": 6, "n_embd": 384, "dropout": 0.0,
       "architecture": "decoder-only GPT, tied embeddings, control-prefix conditioned (LM 3)",
       "eot": 2, "ctrl_end": 8254, "base_vocab": 8192, "step": int(ck["step"]),
       "val_loss": round(float(ck["val_loss"]), 4), "init": "MIDI-GPT-LM2 best (step 112000)",
       "tokenizer": "MidiTok REMI + BPE (tokenizer.json)"}
assert sd["tok.weight"].shape == (8256, 384) and sd["pos.weight"].shape == (2048, 384), "train_lm3.py's config"
json.dump(cfg, open(os.path.join(OUT, "config.json"), "w"), indent=2)

for f in ("tokenizer.json", "model.py", "lm3_vocab.py"):
    shutil.copy(os.path.join(H, f), os.path.join(OUT, f))
shutil.copy(os.path.join(H, "hf_export", "check_midi.py"), os.path.join(OUT, "check_midi.py"))
shutil.copy(os.path.join(H, "generate_lm3.py"), os.path.join(OUT, "generate.py"))
shutil.copy(os.path.join(H, "MODEL_CARD_LM3.md"), os.path.join(OUT, "README.md"))
for name in EXAMPLES:
    shutil.copy(os.path.join(EX, name + ".mid"), os.path.join(OUT, "examples", name + ".mid"))

n = sum(v.numel() for v in sd.values())
print(f"exported LM 3 step {ck['step']} (val loss {ck['val_loss']:.3f}), {n / 1e6:.1f}M params stored -> {OUT}")
for f in sorted(os.listdir(OUT)):
    p = os.path.join(OUT, f)
    print(f"  {f:<20} {'<dir>' if os.path.isdir(p) else f'{os.path.getsize(p) / 1e6:8.2f} MB'}")
