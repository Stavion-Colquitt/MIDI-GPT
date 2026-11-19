# LM 2 evaluation

| checkpoint | step | full-pass val loss | ppl |
|---|---|---|---|
| best.pt | 112000 | 1.3055 | 3.69 |
| best_run1.pt | 57000 | 1.3420 | 3.83 |

Scored 19,767,726 val tokens (Lakh files with MD5 prefix 00-03, never trained on).

## Generated vs real (3072 tokens each)

| stat | generated | real Lakh val |
|---|---|---|
| decoded | 8/8 | 30/30 |
| seconds | 110.70 (IQR 35.70-206.98) | 76.20 (IQR 67.60-114.70) |
| tracks | 1.50 (IQR 1.00-3.50) | 7.00 (IQR 5.00-8.75) |
| notes_per_s | 4.84 (IQR 2.58-9.03) | 21.69 (IQR 14.23-26.26) |
| pitch_range | 39.50 (IQR 3.75-46.00) | 57.00 (IQR 50.50-61.50) |
| pc_entropy_bits | 2.16 (IQR 1.19-2.81) | 2.81 (IQR 2.72-2.96) |
| with_drums | 6/8 | 28/30 |
