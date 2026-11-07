# Benchmark report

| Run | Arm | Dataset | Model | Problems × seeds | Accuracy (mean ± sd) | 95% CI | Calls/problem | Latency/problem |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `calculus-base-gemma3-4b-20261107-155720` | base | calculus | gemma3:4b (Q4_K_M) | 27 × 3 | 100.0 ± 0.0 | 100.0–100.0 | 1.00 | 4.5s |

## calculus-base-gemma3-4b-20261107-155720

Options: `{"temperature": 0.2, "num_ctx": 8192, "num_predict": 4096}`, prompt `base_v1`, digest `68ceab7e468a31c988b`

Per seed: seed 0: 27/27 = 100.0 (Wilson 87.5–100.0), seed 1: 27/27 = 100.0 (Wilson 87.5–100.0), seed 2: 27/27 = 100.0 (Wilson 87.5–100.0)

Outcomes: correct 81, wrong_answer 0, extraction_failure 0, truncated 0, error 0

| topic | accuracy | n |
| --- | --- | --- |
| algebra | 100.0 | 6 |
| applications | 100.0 | 18 |
| derivatives | 100.0 | 21 |
| integrals | 100.0 | 18 |
| limits | 100.0 | 18 |

