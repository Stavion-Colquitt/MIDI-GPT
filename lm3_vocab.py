"""LM 3 control vocabulary: 64 new token ids on top of LM 2's 8,192 (ids 8192-8255).

A song's prefix is: TEMPO, TIMESIG, KEY, one INSTR token per instrument family present (fixed order), DENSITY, CTRL_END.
Controls are read from the MIDI file itself (no labels needed). Genre is not here yet (no openly licensed source).
"""
import numpy as np

BASE = 8192
TEMPO_EDGES = [70, 85, 100, 115, 130, 150, 175]                 # 8 bins: <70, 70-84, 85-99, ..., >=175 BPM
TEMPO_NAMES = ["<70", "70-84", "85-99", "100-114", "115-129", "130-149", "150-174", "175+"]
TIMESIGS = ["4/4", "3/4", "6/8", "2/4", "12/8", "other"]
NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
KEYS = [f"{n} major" for n in NAMES] + [f"{n} minor" for n in NAMES] + ["none"]
FAMILIES = ["Piano", "ChromPerc", "Organ", "Guitar", "Bass", "Strings", "Ensemble", "Brass", "Reed", "Pipe",
            "SynthLead", "SynthPad", "SynthFX", "Ethnic", "Percussive", "SFX", "DRUMS"]
DENSITY_EDGES = [1, 2, 4, 8, 16]                                 # notes per beat, all tracks: 6 bins
KEY_MIN_CONF = 0.6                                               # below this the key token is "none"
MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

T0 = BASE                       # 8192-8199 tempo
S0 = T0 + len(TEMPO_NAMES)      # 8200-8205 time signature
K0 = S0 + len(TIMESIGS)         # 8206-8230 key
I0 = K0 + len(KEYS)             # 8231-8247 instrument families
D0 = I0 + len(FAMILIES)         # 8248-8253 density
CTRL_END = D0 + 6               # 8254
STYLE_STAVION = CTRL_END + 1    # 8255 (reserved for the step 5 fine-tune on his own sets)
VOCAB = STYLE_STAVION + 1       # 8256 (multiple of 64)
assert VOCAB == 8256


def key_of(pc):
    if pc.sum() <= 0: return "none", 0.0
    best = (-2.0, "none")
    for mode, prof in (("major", MAJ), ("minor", MIN)):
        for k in range(12):
            r = float(np.corrcoef(pc, np.roll(prof, k))[0, 1])
            if r > best[0]: best = (r, f"{NAMES[k]} {mode}")
    return best[1], best[0]


def controls_from_score(s):
    """Controls of a symusic Score, or None if it has no notes."""
    tpq = s.ticks_per_quarter
    qpm = [t.qpm for t in s.tempos] or [120.0]; tempo = float(np.median(qpm))
    ts = f"{s.time_signatures[0].numerator}/{s.time_signatures[0].denominator}" if len(s.time_signatures) else "4/4"
    pc = np.zeros(12); fams = set(); n = 0
    for tr in s.tracks:
        if len(tr.notes) == 0: continue
        n += len(tr.notes)
        if tr.is_drum: fams.add("DRUMS"); continue
        fams.add(FAMILIES[tr.program // 8])
        for nt in tr.notes: pc[nt.pitch % 12] += nt.duration
    if n == 0: return None
    dens = n / max(s.end() / tpq, 1.0); key, conf = key_of(pc)
    return {"tempo": round(tempo, 1), "tempo_bin": int(np.searchsorted(TEMPO_EDGES, tempo, side="right")),
            "time_sig": ts if ts in TIMESIGS else "other", "key": key if conf >= KEY_MIN_CONF else "none",
            "key_conf": round(conf, 3), "instruments": [f for f in FAMILIES if f in fams],
            "density": round(dens, 2), "density_bin": int(np.searchsorted(DENSITY_EDGES, dens, side="right"))}


def control_ids(c):
    return ([T0 + c["tempo_bin"], S0 + TIMESIGS.index(c["time_sig"]), K0 + KEYS.index(c["key"])]
            + [I0 + FAMILIES.index(f) for f in c["instruments"]] + [D0 + c["density_bin"]])


def request(tempo_bpm=120, time_sig="4/4", key="A minor", instruments=("DRUMS", "Bass", "Guitar", "Piano"), density_bin=3):
    """Prefix for generation from a human request."""
    c = {"tempo_bin": int(np.searchsorted(TEMPO_EDGES, tempo_bpm, side="right")), "time_sig": time_sig, "key": key,
         "instruments": [f for f in FAMILIES if f in instruments], "density_bin": density_bin}
    return control_ids(c) + [CTRL_END]
