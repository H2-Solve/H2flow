# Benchmark report

| Run | Arm | Dataset | Model | Problems × seeds | Accuracy (mean ± sd) | 95% CI | Calls/problem | Latency/problem |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `math500-base-gemma3-4b-20261107-115144` | base | math500 | gemma3:4b (Q4_K_M) | 500 × 3 | 73.7 ± 2.4 | 70.2–77.1 | 1.00 | 8.5s |

## math500-base-gemma3-4b-20261107-115144

Options: `{"temperature": 0.2, "num_ctx": 8192, "num_predict": 4096}`, prompt `base_v1`, digest `68ceab7e468a31c988b`

Per seed: seed 0: 369/500 = 73.8 (Wilson 69.8–77.5), seed 1: 380/500 = 76.0 (Wilson 72.1–79.5), seed 2: 356/500 = 71.2 (Wilson 67.1–75.0)

Outcomes: correct 1105, wrong_answer 364, extraction_failure 0, truncated 28, error 3
Calibration: published Gemma 3 4B IT, MATH 0-shot (Google model card) = 75.6; this run is within ±10.0 points (a sanity check, not a comparison).

| subject | accuracy | n |
| --- | --- | --- |
| Algebra | 93.8 | 372 |
| Counting & Probability | 76.3 | 114 |
| Geometry | 58.5 | 123 |
| Intermediate Algebra | 52.2 | 291 |
| Number Theory | 83.9 | 186 |
| Prealgebra | 79.7 | 246 |
| Precalculus | 55.4 | 168 |

| level | accuracy | n |
| --- | --- | --- |
| 1 | 94.6 | 129 |
| 2 | 87.4 | 270 |
| 3 | 87.0 | 315 |
| 4 | 71.4 | 384 |
| 5 | 49.5 | 402 |

