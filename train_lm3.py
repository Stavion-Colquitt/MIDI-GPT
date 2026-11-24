"""LM 3: fine-tune LM 2 (checkpoints/best.pt) into a controllable, longer-context MIDI model.

Changes vs LM 2 (train.py is untouched):
  * vocab 8192 -> 8256: 64 control tokens (lm3_vocab.py). New embedding rows start at the mean of the old ones.
  * context 1024 -> 2048: learned position table stretched by linear interpolation, then fine-tuned.
  * every training window = [song's control prefix] + a slice of that song (prefix pinned, like generation will be);
    25% of windows start at the song's first token; loss is not taken on the prefix (it's given, not predicted).
  * eval: val loss + 3 conditioned samples per eval, each with its requested vs measured tempo/key/instruments.
Pause: create pause_lm3.flag -> finishes the step, saves latest, exits. Resume: --resume (exactly where it stopped).
    py -3.12 train_lm3.py --hours 12
    py -3.12 train_lm3.py --steps 40 --smoke --data %TEMP%\\lm3_smoke_tokens
    py -3.12 train_lm3.py --resume
"""
import os, sys, time, math, json, argparse, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
from model import GPT, Config, count_params
import lm3_vocab as V

CK = os.path.join(H, "checkpoints_lm3"); os.makedirs(CK, exist_ok=True)
PAUSE = os.path.join(H, "pause_lm3.flag"); SD = os.path.join(H, "samples_lm3"); os.makedirs(SD, exist_ok=True)
DEV = torch.device("cuda")
ap = argparse.ArgumentParser()
ap.add_argument("--hours", type=float, default=None); ap.add_argument("--steps", type=int, default=None)
ap.add_argument("--micro", type=int, default=8); ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--block", type=int, default=2048)
ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--floor", type=float, default=3e-5)
ap.add_argument("--warmup", type=int, default=300); ap.add_argument("--eval_every", type=int, default=500)
ap.add_argument("--resume", action="store_true"); ap.add_argument("--seed", type=int, default=3)
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--data", default=os.environ.get("LM3_TOKENS", r"D:\Datasets\lakh\tokens_lm3"))
ap.add_argument("--init", default=os.path.join(H, "checkpoints", "best.pt"))
args = ap.parse_args()

# Fixed requests used for the eval samples (same every eval, so they're comparable over training)
REQUESTS = [dict(tempo_bpm=120, key="A minor", instruments=("DRUMS", "Bass", "Guitar", "Piano"), density_bin=3),
            dict(tempo_bpm=90, key="Eb major", instruments=("DRUMS", "Bass", "SynthPad", "SynthLead"), density_bin=2),
            dict(tempo_bpm=160, key="E minor", instruments=("DRUMS", "Bass", "Guitar"), density_bin=4)]


class Songs:
    def __init__(self, split):
        self.data = np.memmap(os.path.join(args.data, f"{split}.bin"), dtype=np.uint16, mode="r")
        self.idx = np.load(os.path.join(args.data, f"{split}_idx.npy"))       # start, prefix_len, song_len
        self.cum = np.cumsum(self.idx[:, 2])                                   # sample songs in proportion to length

    def batch(self, n, block):
        X = np.full((n, block), 2, np.int64); Y = np.full((n, block), -100, np.int64)
        for r in range(n):
            s = int(np.searchsorted(self.cum, np.random.randint(self.cum[-1]), side="right"))
            st, pl, sl = (int(v) for v in self.idx[s])
            body = block + 1 - pl
            o = 0 if (np.random.rand() < 0.25 or sl <= body) else np.random.randint(0, sl - body + 1)
            seq = np.concatenate([self.data[st:st + pl], self.data[st + pl + o: st + pl + min(o + body, sl)]]).astype(np.int64)
            L = len(seq) - 1
            X[r, :L] = seq[:-1]; Y[r, :L] = seq[1:]; Y[r, :pl - 1] = -100          # no loss on predicting the prefix
        return (torch.from_numpy(X).pin_memory().to(DEV, non_blocking=True),
                torch.from_numpy(Y).pin_memory().to(DEV, non_blocking=True))


def lm3_loss(model, x, y):
    logits, _ = model(x)
    return F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)


def init_from_lm2(cfg):
    ck = torch.load(args.init, map_location="cpu", weights_only=False); sd = ck["model"]
    old_tok, old_pos = sd["tok.weight"], sd["pos.weight"]
    new_tok = torch.empty(cfg.vocab, old_tok.shape[1]); new_tok[:old_tok.shape[0]] = old_tok
    new_tok[old_tok.shape[0]:] = old_tok.mean(0) + 0.02 * torch.randn(cfg.vocab - old_tok.shape[0], old_tok.shape[1])
    new_pos = F.interpolate(old_pos.T.unsqueeze(0), size=cfg.block, mode="linear", align_corners=True)[0].T.contiguous()
    sd = dict(sd); sd["tok.weight"] = new_tok; sd["head.weight"] = new_tok; sd["pos.weight"] = new_pos
    print(f"init from {os.path.basename(args.init)} (LM 2 step {ck.get('step')}, val {ck.get('val_loss')}): "
          f"vocab {old_tok.shape[0]}->{cfg.vocab}, ctx {old_pos.shape[0]}->{cfg.block}", flush=True)
    return sd


@torch.no_grad()
def evaluate(model, songs, n):
    model.eval(); tot = 0.0
    for _ in range(n):
        x, y = songs.batch(args.micro, args.block)
        with torch.autocast("cuda", dtype=torch.float16): tot += lm3_loss(model, x, y).item()
    model.train(); return tot / n


@torch.no_grad()
def samples(model, tok, step, n_tokens):
    from symusic import Score
    model.eval(); out = []
    for k, rq in enumerate(REQUESTS):
        pre = V.request(**rq); idx = torch.tensor([pre], device=DEV); body = []
        torch.manual_seed(1000 + k)
        for i in range(n_tokens):
            with torch.autocast("cuda", dtype=torch.float16): logits, _ = model(idx[:, -args.block:])
            lg = logits[:, -1].float() / 0.9
            lg[:, V.BASE:] = -float("inf")                       # never generate control tokens
            if i < 512: lg[:, 2] = -float("inf")                 # no instant endings
            v, _ = torch.topk(lg, 50); lg[lg < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(lg, -1), 1)
            if nxt.item() == 2: break
            body.append(nxt.item()); idx = torch.cat([idx, nxt], 1)
        name = f"step{step:06d}_{k}.mid"; path = os.path.join(SD, name)
        try:
            tok.decode(body).dump_midi(path); got = V.controls_from_score(Score(path)) or {}
            want = f"{rq['tempo_bpm']} BPM, {rq['key']}, {'+'.join(rq['instruments'])}"
            have = f"{got.get('tempo')} BPM, {got.get('key')}, {'+'.join(got.get('instruments', []))}"
            out.append(f"{name}: {len(body)} tokens | asked: {want} | got: {have}")
        except Exception as e:
            out.append(f"{name}: {len(body)} tokens, could not decode ({e})")
    model.train(); return out


def keep_awake():
    if os.name == "nt":
        import ctypes; ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def main():
    import mlflow
    from miditok import REMI
    keep_awake(); torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    meta = json.load(open(os.path.join(args.data, "meta.json"))); assert meta["vocab"] == V.VOCAB
    tok = REMI(params=Path(os.path.join(H, "tokenizer.json")))
    cfg = Config(vocab=V.VOCAB, block=args.block); model = GPT(cfg)
    resumed = args.resume and os.path.exists(os.path.join(CK, "latest.pt"))
    if not resumed: model.load_state_dict(init_from_lm2(cfg))
    model.to(DEV)
    print(f"device: {torch.cuda.get_device_name(0)} | params {count_params(model) / 1e6:.1f}M | "
          f"{args.micro * args.accum * cfg.block:,} tokens/step", flush=True)
    decay = [p for p in model.parameters() if p.dim() >= 2]; nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": nodecay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), fused=True)
    scaler = torch.amp.GradScaler("cuda")
    step, best, total, run_id = 0, float("inf"), args.steps, None
    if resumed:
        ck = torch.load(os.path.join(CK, "latest.pt"), map_location=DEV, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
        step, best, total, run_id = ck["step"], ck["best"], ck["total"], ck.get("run_id")
        if total is None and args.hours is None: args.hours = ck.get("hours")
        print(f"resumed at step {step}/{total}", flush=True)
    tr, va = Songs("train"), Songs("val")
    mlflow.set_tracking_uri("sqlite:///" + os.path.join(H, "mlflow.db").replace("\\", "/")); mlflow.set_experiment("lm3-midi-controls")

    def save_latest(run):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "step": step,
                    "best": best, "total": total, "run_id": run.info.run_id, "hours": args.hours, "cfg": cfg.__dict__},
                   os.path.join(CK, "latest.pt"))

    t0 = time.time(); start = step; tpm = None
    with mlflow.start_run(run_id=run_id, run_name="smoke" if args.smoke else None) as run:
        if run_id is None:
            mlflow.log_params({**cfg.__dict__, "params_M": round(count_params(model) / 1e6, 2), "lr": args.lr, "floor": args.floor,
                               "tokens_per_step": args.micro * args.accum * cfg.block, "hours": args.hours, "init": "LM 2 best.pt",
                               "train_tokens": meta["train_tokens"], "precision": "fp16+GradScaler"})
        if not resumed and not args.smoke:                   # baseline: the freshly expanded LM 2 before any LM 3 training
            vl0 = evaluate(model, va, 40); mlflow.log_metric("val_loss", vl0, step=0)
            print(f"EVAL step 0 (expanded LM 2, untrained on controls): val loss {vl0:.3f}", flush=True)
        while total is None or step < total:
            if step < args.warmup: lr = args.lr * (step + 1) / args.warmup
            elif total is not None:
                pr = min((step - args.warmup) / max(total - args.warmup, 1), 1.0)
                lr = args.floor + 0.5 * (args.lr - args.floor) * (1 + math.cos(math.pi * pr))
            else: lr = args.lr
            for g in opt.param_groups: g["lr"] = lr
            for _ in range(args.accum):
                x, y = tr.batch(args.micro, cfg.block)
                with torch.autocast("cuda", dtype=torch.float16): loss = lm3_loss(model, x, y)
                scaler.scale(loss / args.accum).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); step += 1
            if total is None and step == start + 30: torch.cuda.synchronize(); tpm = time.time()
            if total is None and step == start + 130:
                torch.cuda.synchronize(); per = (time.time() - tpm) / 100
                total = int((args.hours * 3600 - (time.time() - t0)) * 0.93 / per) + step
                print(f"{per:.3f} s/step -> {total} steps ({total * args.micro * args.accum * cfg.block / 1e6:.0f}M tokens) "
                      f"for {args.hours} h", flush=True); mlflow.log_param("total_steps", total)
            if step % 25 == 0: print(f"step {step}/{total}  loss {loss.item():.3f}  lr {lr:.2e}", flush=True)
            if step % args.eval_every == 0 or step == total:
                vl = evaluate(model, va, 5 if args.smoke else 40)
                mlflow.log_metrics({"train_loss": loss.item(), "val_loss": vl, "val_ppl": math.exp(vl), "lr": lr}, step=step)
                s = samples(model, tok, step, 256 if args.smoke else 2048)
                with open(os.path.join(H, "samples_lm3.txt"), "a", encoding="utf-8") as f:
                    f.write(f"\n===== step {step} | val loss {vl:.3f} | ppl {math.exp(vl):.1f} =====\n" + "\n".join(s) + "\n")
                print(f"EVAL step {step}: val loss {vl:.3f} | ppl {math.exp(vl):.1f} | {(time.time() - t0) / 3600:.2f} h", flush=True)
                if vl < best:
                    best = vl; torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "step": step, "val_loss": vl},
                                          os.path.join(CK, "best.pt")); print("  saved best", flush=True)
                save_latest(run)
            if os.path.exists(PAUSE):
                save_latest(run); print(f"paused at step {step}/{total} (pause_lm3.flag)", flush=True); return
        mlflow.log_metric("best_val_loss", best)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
