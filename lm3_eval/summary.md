# LM 3 control-following eval

- checkpoint: `best.pt`
- requests: 60, 1536 tokens each, seed = request index, top_k 50, temperature 0.9, context 2048

## Overall


**All requests** (60 requests, 59 decoded)

| metric | rate |
|---|---|
| decoded | 98.3% |
| tempo within 10% | 55.9% |
| exact key | 37.3% |
| key allowing relative major/minor | 40.7% |
| all requested instruments present | 42.4% |
| instrument recall (mean) | 74.2% |
| instrument precision (mean) | 93.7% |
| density_bin exact | 42.4% |
| density_bin within 1 | 86.4% |
| mean beats generated | 136.1 |
| mean tracks | 3.0 |


## By requested tempo range


**70-84 BPM** (13 requests, 12 decoded)

| metric | rate |
|---|---|
| decoded | 92.3% |
| tempo within 10% | 50.0% |
| exact key | 41.7% |
| key allowing relative major/minor | 50.0% |
| all requested instruments present | 25.0% |
| instrument recall (mean) | 61.1% |
| instrument precision (mean) | 93.8% |
| density_bin exact | 58.3% |
| density_bin within 1 | 83.3% |
| mean beats generated | 139.0 |
| mean tracks | 2.6 |


**85-99 BPM** (3 requests, 3 decoded)

| metric | rate |
|---|---|
| decoded | 100.0% |
| tempo within 10% | 33.3% |
| exact key | 0.0% |
| key allowing relative major/minor | 0.0% |
| all requested instruments present | 0.0% |
| instrument recall (mean) | 47.2% |
| instrument precision (mean) | 83.3% |
| density_bin exact | 66.7% |
| density_bin within 1 | 100.0% |
| mean beats generated | 159.3 |
| mean tracks | 2.0 |


**100-114 BPM** (10 requests, 10 decoded)

| metric | rate |
|---|---|
| decoded | 100.0% |
| tempo within 10% | 30.0% |
| exact key | 50.0% |
| key allowing relative major/minor | 50.0% |
| all requested instruments present | 70.0% |
| instrument recall (mean) | 88.3% |
| instrument precision (mean) | 88.3% |
| density_bin exact | 20.0% |
| density_bin within 1 | 80.0% |
| mean beats generated | 111.8 |
| mean tracks | 3.8 |


**115-129 BPM** (4 requests, 4 decoded)

| metric | rate |
|---|---|
| decoded | 100.0% |
| tempo within 10% | 75.0% |
| exact key | 50.0% |
| key allowing relative major/minor | 50.0% |
| all requested instruments present | 50.0% |
| instrument recall (mean) | 70.8% |
| instrument precision (mean) | 93.8% |
| density_bin exact | 25.0% |
| density_bin within 1 | 100.0% |
| mean beats generated | 169.2 |
| mean tracks | 2.8 |


**130-149 BPM** (16 requests, 16 decoded)

| metric | rate |
|---|---|
| decoded | 100.0% |
| tempo within 10% | 81.2% |
| exact key | 43.8% |
| key allowing relative major/minor | 50.0% |
| all requested instruments present | 50.0% |
| instrument recall (mean) | 82.8% |
| instrument precision (mean) | 97.9% |
| density_bin exact | 37.5% |
| density_bin within 1 | 75.0% |
| mean beats generated | 131.1 |
| mean tracks | 3.4 |


**150-174 BPM** (14 requests, 14 decoded)

| metric | rate |
|---|---|
| decoded | 100.0% |
| tempo within 10% | 50.0% |
| exact key | 21.4% |
| key allowing relative major/minor | 21.4% |
| all requested instruments present | 35.7% |
| instrument recall (mean) | 72.0% |
| instrument precision (mean) | 94.8% |
| density_bin exact | 50.0% |
| density_bin within 1 | 100.0% |
| mean beats generated | 142.1 |
| mean tracks | 2.7 |


## By key mode


**major keys** (34 requests, 34 decoded)

| metric | rate |
|---|---|
| decoded | 100.0% |
| tempo within 10% | 61.8% |
| exact key | 52.9% |
| key allowing relative major/minor | 55.9% |
| all requested instruments present | 55.9% |
| instrument recall (mean) | 81.1% |
| instrument precision (mean) | 95.8% |
| density_bin exact | 35.3% |
| density_bin within 1 | 82.4% |
| mean beats generated | 137.4 |
| mean tracks | 3.2 |



**minor keys** (26 requests, 25 decoded)

| metric | rate |
|---|---|
| decoded | 96.2% |
| tempo within 10% | 48.0% |
| exact key | 16.0% |
| key allowing relative major/minor | 20.0% |
| all requested instruments present | 24.0% |
| instrument recall (mean) | 64.7% |
| instrument precision (mean) | 90.7% |
| density_bin exact | 52.0% |
| density_bin within 1 | 92.0% |
| mean beats generated | 134.3 |
| mean tracks | 2.7 |
