"""LM 3 step 2 (data prep, CPU, no training): per-song control tokens read from each MIDI file itself.
Controls: tempo bin, time signature, key (Krumhansl-Schmuckler on duration-weighted pitch classes, drums excluded),
instrument set (GM family per non-drum program + DRUMS), note density (notes per beat, binned). Genre: not yet
(needs an openly licensed label source; see Training Queue 2c).
    py -3.12 extract_controls.py --test 300            # sample, writes data/controls_test.jsonl + summary
    py -3.12 extract_controls.py --workers 4           # all of lmd_full -> data/controls.jsonl (not run yet)
"""
import os, sys, json, random, argparse, collections
import numpy as np
from pathlib import Path
from multiprocessing import Pool

H = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(H, "data")
SRC = os.environ.get("LAKH_MIDI", os.path.join(D, "lmd_full")); MAX_BYTES = 1_000_000
MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
FAMILIES = ["Piano", "ChromPerc", "Organ", "Guitar", "Bass", "Strings", "Ensemble", "Brass", "Reed", "Pipe",
            "SynthLead", "SynthPad", "SynthFX", "Ethnic", "Percussive", "SFX"]
TEMPO_EDGES = [70, 85, 100, 115, 130, 150, 175]          # bins: <70, 70-84, ..., >=175
DENSITY_EDGES = [1, 2, 4, 8, 16]                        # notes per beat (all tracks)


def key_of(pc):
    if pc.sum() <= 0: return None, 0.0
    best = (-2, None)
    for mode, prof in (("major", MAJ), ("minor", MIN)):
        for k in range(12):
            r = np.corrcoef(pc, np.roll(prof, k))[0, 1]
            if r > best[0]: best = (r, f"{NAMES[k]} {mode}")
    return best[1], float(best[0])


def controls(path):
    try:
        if os.path.getsize(path) > MAX_BYTES: return None
        from symusic import Score
        s = Score(path); tpq = s.ticks_per_quarter
        qpm = [t.qpm for t in s.tempos] or [120.0]
        tempo = float(np.median(qpm))
        ts = f"{s.time_signatures[0].numerator}/{s.time_signatures[0].denominator}" if len(s.time_signatures) else "4/4"
        pc = np.zeros(12); fams = set(); drums = False; n = 0
        for tr in s.tracks:
            if len(tr.notes) == 0: continue
            n += len(tr.notes)
            if tr.is_drum: drums = True; continue
            fams.add(FAMILIES[tr.program // 8])
            for nt in tr.notes: pc[nt.pitch % 12] += nt.duration
        if n == 0: return None
        beats = max(s.end() / tpq, 1.0); dens = n / beats
        key, conf = key_of(pc)
        return {"file": os.path.basename(path), "tempo": round(tempo, 1), "tempo_bin": int(np.searchsorted(TEMPO_EDGES, tempo, side="right")),
                "time_sig": ts, "key": key, "key_conf": round(conf, 3), "instruments": sorted(fams) + (["DRUMS"] if drums else []),
                "density": round(dens, 2), "density_bin": int(np.searchsorted(DENSITY_EDGES, dens, side="right")),
                "n_tempo_changes": len(qpm), "beats": round(beats)}
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--test", type=int, default=0); ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    files = sorted(str(p) for p in Path(SRC).rglob("*.mid")); random.seed(0)
    if a.test: files = random.sample(files, a.test)
    out = os.path.join(D, "controls_test.jsonl" if a.test else "controls.jsonl"); rows = []
    with Pool(a.workers) as p, open(out, "w", encoding="utf-8") as f:
        for r in p.imap_unordered(controls, files, chunksize=16):
            if r: rows.append(r); f.write(json.dumps(r) + "\n")
    C = collections.Counter
    summ = {"files": len(files), "ok": len(rows),
            "tempo_bin": dict(sorted(C(r["tempo_bin"] for r in rows).items())),
            "time_sig_top": C(r["time_sig"] for r in rows).most_common(6),
            "key_top": C(r["key"] for r in rows).most_common(8),
            "mode": dict(C((r["key"] or "none").split()[-1] for r in rows)),
            "key_conf_median": float(np.median([r["key_conf"] for r in rows])) if rows else None,
            "instr_top": C(i for r in rows for i in r["instruments"]).most_common(10),
            "density_bin": dict(sorted(C(r["density_bin"] for r in rows).items())),
            "multi_tempo_frac": round(sum(r["n_tempo_changes"] > 1 for r in rows) / max(len(rows), 1), 3)}
    json.dump(summ, open(out.replace(".jsonl", "_summary.json"), "w"), indent=1); print(json.dumps(summ))


if __name__ == "__main__":
    main()
