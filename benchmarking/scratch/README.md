# Scratch: temporary, remove before merging into main

One-off check scripts from setting up the rag/full arms, plus (force-added) raw runs in
`benchmarking/results/raw/`, kept on this branch only to move the work to another machine.
Before the PR into main: `git rm -r --cached benchmarking/results/raw` and `git rm -r benchmarking/scratch`.

Run from the repo root with Ollama and Elasticsearch up (on Windows, set
`OLLAMA_HOST=http://127.0.0.1:11434` and `H2FLOW_ES_URL=http://127.0.0.1:9201`).

| Script | What it checks |
| --- | --- |
| `trace_rag.py` | Live trace of the rag and full arms: every model call goes through the harness, one ES search per problem, all 5 chunk texts appear verbatim in the prompt, attributes restored, and an empty-retrieval ablation. |
| `validate_extractor.py <base run> [n]` | Extractor faithfulness: unboxes base-arm answers and checks wrong answers are never "fixed" (116/116 stayed wrong; 92/100 correct kept). |
| `repeat_lim01.py` | Repeats rag on lim-01 to show temperature-0 run-to-run variation in Ollama. |
| `rag_wrong.py <attempts.jsonl>` | Prints unboxed counts and the tail of every non-correct attempt, for hand grading. |
| `inspect_runs.py <run dirs...>` | Dumps per-attempt calls, retrieval, intent and response tails. |
| `full_errors.py <attempts.jsonl>` | Groups a `full` run's failures by problem and route, and dumps the intent, SymPy truth, calls and last response for each. |
| `repro_full_errors.py <attempts.jsonl>` | Replays each failure offline through `verification.py` (no LLM), showing where SymPy crashes and why verification rejects answers. |

See [HANDOFF.md](HANDOFF.md) for where the work stands.
