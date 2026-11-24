"""Run the rag arm on lim-01 several times; show what the model answered and what the grader extracted."""
import hashlib
import json
import os
import sys

os.environ.setdefault("OLLAMA_HOST", "http://127.0.0.1:11434")
os.environ.setdefault("H2FLOW_ES_URL", "http://127.0.0.1:9201")
sys.path.insert(0, os.getcwd())


def main():
    from benchmarking.arms import RunContext, get_arm
    from benchmarking.datasets import load_dataset
    from benchmarking.grading import grade
    from benchmarking.ollama_client import OllamaClient

    orig = OllamaClient._request
    sent = []
    def logged(self, path, payload=None):
        if path == "/api/chat":
            sent.append(payload)
        return orig(self, path, payload)
    OllamaClient._request = logged

    ctx = RunContext(client=OllamaClient(), model="gemma3:4b",
                     options={"temperature": 0.0, "num_ctx": 8192, "num_predict": 4096},
                     prompt_template="{problem}", prompt_name="base_v1")
    p = {x.id: x for x in load_dataset("calculus").problems}["lim-01"]
    arm = get_arm("rag")
    arm.setup(ctx)
    smoke = json.loads(open("benchmarking/results/raw/calculus-rag-gemma3-4b-20261124-173050/attempts.jsonl",
                            encoding="utf-8").readline())
    print("smoke test response sha:", hashlib.sha256(smoke["response"].encode()).hexdigest()[:12],
          "outcome", smoke["outcome"], "extracted", smoke["extracted"])
    for i in range(4):
        a = arm.solve(p, 0, ctx)
        g = grade(p.answer_latex, a.response)
        prompt_sha = hashlib.sha256(json.dumps(sent[-1]["messages"]).encode()).hexdigest()[:12]
        print(f"run {i}: prompt sha {prompt_sha}, response sha {hashlib.sha256(a.response.encode()).hexdigest()[:12]}, "
              f"outcome {g['outcome']}, extracted {g['extracted']!r}, tokens {a.completion_tokens}")
        print("   tail:", repr(a.response[-260:]))


if __name__ == "__main__":
    main()
