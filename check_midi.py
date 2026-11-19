"""Read MIDI files back and print what's in them (validity check for generated songs).
    py -3.12 check_midi.py generated/*.mid
"""
import sys, glob
from symusic import Score

files = sorted(f for p in sys.argv[1:] for f in glob.glob(p))
ok = 0
for f in files:
    try:
        s = Score(f).to("second")
        notes = [n for t in s.tracks for n in t.notes]
        dur = max((n.time + n.duration for n in notes), default=0.0)
        progs = sorted({("drums" if t.is_drum else f"prog {t.program}") for t in s.tracks if len(t.notes)})
        tempo = s.tempos[0].qpm if len(s.tempos) else None
        print(f"{f}: VALID | {len(s.tracks)} tracks ({', '.join(progs)}) | {len(notes)} notes | {dur:.1f} s"
              + (f" | {tempo:.0f} BPM" if tempo else ""))
        ok += len(notes) > 0
    except Exception as e:
        print(f"{f}: INVALID ({e})")
print(f"{ok}/{len(files)} files are valid MIDI with notes")
