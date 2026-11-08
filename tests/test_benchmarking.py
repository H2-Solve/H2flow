"""Benchmark harness: grading, statistics, datasets, and an end-to-end run against a fake Ollama."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("math_verify")

from benchmarking import report, run
from benchmarking.arms import ArmUnavailable, RunContext, get_arm
from benchmarking.datasets import load_dataset, load_jsonl_dataset
from benchmarking.grading import grade, parse_gold
from benchmarking.ollama_client import OllamaClient
from benchmarking.stats import mcnemar_exact, paired_bootstrap_diff, wilson_ci


# ---------- grading ----------

@pytest.mark.parametrize("gold, text, outcome", [
    (r"\frac{\sqrt{2}}{2}", r"So the answer is \boxed{\frac{1}{\sqrt{2}}}.", "correct"),
    (r"\left( 3, \frac{\pi}{2} \right)", r"\boxed{(3, \frac{\pi}{2})}", "correct"),
    (r"\frac{14}{3}", "FINAL_ANSWER: 14/3", "correct"),
    (r"\left\{-1, 1\right\}", r"\boxed{x = 1 \text{ or } x = -1}", "correct"),
    (r"x^{2} \left(x \cos{\left(x \right)} + 3 \sin{\left(x \right)}\right)",
     r"\boxed{3x^2\sin x + x^3\cos x}", "correct"),
    (r"\frac{1}{2}", r"\boxed{\frac{1}{3}}", "wrong_answer"),
    ("5", "I am not sure how to do this.", "extraction_failure"),
])
def test_grade_outcomes(gold, text, outcome):
    assert grade(gold, text)["outcome"] == outcome


def test_truncated_and_error_win_over_content():
    assert grade("5", r"\boxed{5}", truncated=True)["outcome"] == "truncated"
    assert grade("5", r"\boxed{5}", error="timeout")["outcome"] == "error"


def test_calculus_gold_answers_all_parse():
    ds = load_dataset("calculus")
    assert len(ds.problems) >= 25
    assert len({p.id for p in ds.problems}) == len(ds.problems)
    for p in ds.problems:
        assert parse_gold(p.answer_latex), p.id
        assert grade(p.answer_latex, rf"\boxed{{{p.answer_latex}}}")["outcome"] == "correct", p.id


def test_natural_log_gold_matches_ln():
    # golds are written with \ln (sp.latex(..., ln_notation=True)); Math-Verify reads bare \log as base 10
    assert grade(r"\ln{\left(2 \right)}", r"\boxed{\ln 2}")["outcome"] == "correct"


# ---------- stats ----------

def test_wilson_ci_brackets_the_estimate():
    lo, hi = wilson_ci(75, 100)
    assert lo < 0.75 < hi and 0.65 < lo and hi < 0.83


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(10, 0) == pytest.approx(2 / 1024)
    assert mcnemar_exact(5, 5) == 1.0


def test_paired_bootstrap_diff():
    a = [0.0] * 50 + [1.0] * 50
    res = paired_bootstrap_diff(a, [1.0] * 100)
    assert res["diff"] == pytest.approx(0.5)
    assert res["ci"][0] > 0.3 and res["p"] < 0.01


# ---------- arms ----------

def test_pending_arms_refuse_to_run():
    ctx = RunContext(client=None, model="m", options={}, prompt_template="{problem}", prompt_name="p")
    for name in ("rag", "full"):
        arm = get_arm(name)
        assert not arm.ready
        with pytest.raises(ArmUnavailable):
            arm.setup(ctx)


# ---------- end to end with a fake Ollama ----------

class FakeOllama(BaseHTTPRequestHandler):
    answers = {}   # problem text -> reply
    calls = []

    def log_message(self, *args):
        pass

    def _send(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/version":
            self._send({"version": "0.0-fake"})
        elif self.path == "/api/tags":
            self._send({"models": [{"name": "fake:1b", "model": "fake:1b", "digest": "sha256:abc"}]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/api/show":
            self._send({"details": {"family": "fake", "parameter_size": "1B", "quantization_level": "Q4_K_M"},
                        "parameters": "temperature 1"})
            return
        FakeOllama.calls.append(req)
        prompt = req["messages"][0]["content"]
        reply = next((a for q, a in self.answers.items() if q in prompt), "no idea")
        self._send({"message": {"content": reply}, "done_reason": "stop",
                    "prompt_eval_count": 10, "eval_count": 20})


@pytest.fixture
def fake_ollama():
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    FakeOllama.calls = []
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()


@pytest.fixture
def tiny_set(tmp_path):
    rows = [
        {"id": "p1", "problem": "What is 2+3?", "answer_latex": "5", "topic": "arith"},
        {"id": "p2", "problem": "What is 1/2 + 1/4?", "answer_latex": r"\frac{3}{4}", "topic": "arith"},
        {"id": "p3", "problem": "Name a prime.", "answer_latex": "7", "topic": "other"},
    ]
    path = tmp_path / "tiny.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    FakeOllama.answers = {"2+3": r"so \boxed{5}", "1/2 + 1/4": r"\boxed{0.75}", "prime": r"\boxed{2}"}
    return path


def test_run_and_report_end_to_end(fake_ollama, tiny_set, tmp_path):
    out = tmp_path / "runs"
    run.main(["--arm", "base", "--dataset", str(tiny_set), "--model", "fake:1b", "--seeds", "0", "1",
              "--ollama-host", fake_ollama, "--out", str(out), "--run-name", "r1"])
    run_dir = out / "r1"
    rows = [json.loads(line) for line in (run_dir / "attempts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 6
    assert {r["outcome"] for r in rows if r["problem_id"] in ("p1", "p2")} == {"correct"}
    assert {r["outcome"] for r in rows if r["problem_id"] == "p3"} == {"wrong_answer"}

    # every call carries the shared options and the seed; the prompt keeps \boxed{} intact
    assert {c["options"]["seed"] for c in FakeOllama.calls} == {0, 1}
    assert all(c["options"]["num_ctx"] == run.DEFAULTS["num_ctx"] for c in FakeOllama.calls)
    assert r"\boxed{}" in FakeOllama.calls[0]["messages"][0]["content"]

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_digest"] == "sha256:abc"
    assert manifest["model"]["quantization_level"] == "Q4_K_M"
    assert manifest["dataset"]["n_run"] == 3 and "finished_at" in manifest

    # resume does no new work
    n_calls = len(FakeOllama.calls)
    run.main(["--resume", str(run_dir), "--ollama-host", fake_ollama])
    assert len(FakeOllama.calls) == n_calls

    s = report.summarize_run(report.load_run(run_dir))
    assert s["accuracy_mean"] == pytest.approx(2 / 3)
    assert s["cost"]["llm_calls"] == 1 and s["groups"]["topic"]["arith"]["accuracy"] == 1.0

    # a second run where p3 is fixed -> paired comparison shows 1 fixed, 0 broken
    FakeOllama.answers["prime"] = r"\boxed{7}"
    run.main(["--arm", "base", "--dataset", str(tiny_set), "--model", "fake:1b", "--seeds", "0", "1",
              "--ollama-host", fake_ollama, "--out", str(out), "--run-name", "r2"])
    prefix = tmp_path / "summary"
    report.main([str(run_dir), str(out / "r2"), "--save", str(prefix)])
    saved = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    cmp = saved["comparisons"][0]
    assert (cmp["fixed"], cmp["broken"]) == (1, 0)
    assert "Paired comparisons" in prefix.with_suffix(".md").read_text(encoding="utf-8")


def test_sympy_arm_routes_to_sympy_or_falls_back(fake_ollama, monkeypatch):
    pytest.importorskip("sympy")
    pytest.importorskip("langchain_ollama")
    from benchmarking.arms.sympy_only import SympyArm
    from benchmarking.datasets import Problem

    intents = {
        "integral": {"operation": "integral", "expression": "x**2", "variable": "x",
                     "lower_bound": "0", "upper_bound": "3"},
        "prime": {"operation": "unsupported", "expression": None},
    }

    class StubVerification:
        @staticmethod
        def extract_math_intent(question, prompt_path):
            return next(v for k, v in intents.items() if k in question)

    arm = SympyArm()
    ctx = RunContext(client=OllamaClient(fake_ollama), model="gemma3:4b", options={},
                     prompt_template="{problem}", prompt_name="p")
    arm.setup(ctx)
    monkeypatch.setattr(arm, "_verification", StubVerification)
    FakeOllama.answers = {"prime": r"\boxed{7}"}

    solved = arm.solve(Problem("a", "integral of x^2 from 0 to 3", "9"), 0, ctx)
    assert solved.route == "sympy" and grade("9", solved.response)["outcome"] == "correct"

    fallback = arm.solve(Problem("b", "Name a prime.", "7"), 0, ctx)
    assert fallback.route == "fallback" and fallback.llm_calls == 2
    assert grade("7", fallback.response)["outcome"] == "correct"


def test_unreachable_ollama_is_a_clear_error(tiny_set, tmp_path):
    with pytest.raises(SystemExit, match="cannot reach Ollama"):
        run.main(["--dataset", str(tiny_set), "--ollama-host", "http://127.0.0.1:1", "--out", str(tmp_path)])


def test_math500_loader_reads_the_hf_format(tmp_path):
    from benchmarking.datasets import load_math500

    path = tmp_path / "math500.jsonl"
    path.write_text(json.dumps({"problem": "p", "solution": "s", "answer": "5", "subject": "Algebra",
                                "level": 2, "unique_id": "test/algebra/1.json"}) + "\n", encoding="utf-8")
    ds = load_math500(path, download=False)
    assert ds.problems[0].id == "test/algebra/1.json"
    assert ds.problems[0].meta == {"subject": "Algebra", "level": 2}


def test_jsonl_dataset_keeps_metadata(tiny_set):
    ds = load_jsonl_dataset("tiny", tiny_set)
    assert ds.problems[0].meta == {"topic": "arith"} and len(ds.sha256) == 64


def test_client_reports_truncation(fake_ollama):
    res = OllamaClient(fake_ollama).chat("fake:1b", [{"role": "user", "content": "x"}], {})
    assert not res.truncated and res.completion_tokens == 20
