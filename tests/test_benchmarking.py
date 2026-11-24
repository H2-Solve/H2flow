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

def test_system_arms_need_elasticsearch(monkeypatch):
    pytest.importorskip("langchain_ollama")
    pytest.importorskip("elasticsearch")
    monkeypatch.setenv("H2FLOW_ES_URL", "http://127.0.0.1:1")
    ctx = RunContext(client=None, model="gemma3:4b", options={}, prompt_template="{problem}", prompt_name="p")
    for name in ("rag", "full"):
        arm = get_arm(name)
        assert arm.ready
        with pytest.raises(ArmUnavailable, match="Elasticsearch"):
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
        prompt = "\n".join(m["content"] for m in req["messages"])
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


# ---------- rag and full arms: the system code on main, with a fake Ollama and fake retrieval ----------

DOCS = [{"text": "The power rule: d/dx x^n = n x^(n-1).", "book_id": "calculus-volume-1", "module_id": "m1",
         "section": "3.3 Differentiation Rules", "title": "Power rule", "chapter": "3"}]


@pytest.fixture
def system_arm(fake_ollama, monkeypatch):
    """Returns (make(arm_name) -> (arm, ctx)) with Elasticsearch replaced by DOCS."""
    pytest.importorskip("langchain_ollama")
    pytest.importorskip("elasticsearch")
    import RAG.main as rag_main
    from benchmarking.arms import full, rag

    for module in (rag, full):
        monkeypatch.setattr(module, "check_index", lambda: {"index": "fake"})
    monkeypatch.setattr(rag_main, "retrieve_documents", lambda query, top_k, index: DOCS[:top_k])
    options = {"temperature": 0.2, "num_ctx": 8192, "num_predict": 4096}

    def make(name):
        arm = get_arm(name)
        ctx = RunContext(client=OllamaClient(fake_ollama), model="gemma3:4b", options=options,
                         prompt_template="{problem}", prompt_name="p")
        notes = arm.setup(ctx)
        assert notes["index"] == "fake"
        return arm, ctx
    return make


def test_rag_arm_runs_system_rag_through_the_harness(system_arm):
    import RAG.main as rag_main
    from benchmarking.datasets import Problem

    original_llm = rag_main.llm
    arm, ctx = system_arm("rag")
    FakeOllama.answers = {"derivative of x^3": r"By the power rule, \boxed{3x^2}"}
    a = arm.solve(Problem("d", "What is the derivative of x^3?", "3x^{2}"), 1, ctx)

    assert a.error is None and a.route == "rag" and a.llm_calls == 1
    assert grade("3x^{2}", a.to_dict()["final_text"])["outcome"] == "correct"
    assert a.prompt_tokens == 10 and a.extra["retrieved"][0]["module_id"] == "m1"
    call = FakeOllama.calls[-1]
    # system temperature (0), harness num_ctx and seed; the retrieved chunk is in the prompt
    assert call["options"]["temperature"] == 0 and call["options"]["num_ctx"] == 8192
    assert call["options"]["seed"] == 1
    assert "power rule: d/dx" in call["messages"][0]["content"]
    assert rag_main.llm is original_llm  # restored


def test_full_arm_verified_route_grades_the_final_answer(system_arm):
    from benchmarking.datasets import Problem

    arm, ctx = system_arm("full")
    FakeOllama.answers = {
        "CRITICAL INSTRUCTION": "Use the power rule.\nFINAL_ANSWER: 3*x**2",
        "derivative of x^3": '{"operation": "derivative", "expression": "x**3", "variable": "x"}',
    }
    a = arm.solve(Problem("d", "Find the derivative of x^3.", "3x^{2}"), 0, ctx)

    assert a.error is None and a.route == "verified", a.error
    assert a.llm_calls == 2 and a.extra["solver_calls"] == 1
    assert a.extra["intent"]["operation"] == "derivative" and a.extra["sympy_truth"] == "3*x**2"
    assert a.final_text == r"\boxed{3 x^{2}}"
    assert grade("3x^{2}", a.final_text)["outcome"] == "correct"
    intent_call, solver_call = FakeOllama.calls[-2:]
    assert intent_call["format"] == "json" and intent_call["options"]["temperature"] == 0
    assert solver_call["options"]["temperature"] == 0.2


def test_full_arm_unverified_after_retries_is_an_error(system_arm):
    from benchmarking.datasets import Problem

    arm, ctx = system_arm("full")
    FakeOllama.answers = {
        "CRITICAL INSTRUCTION": "FINAL_ANSWER: 8",
        "derivative of x^3": '{"operation": "derivative", "expression": "x**3", "variable": "x"}',
    }
    a = arm.solve(Problem("d", "Find the derivative of x^3.", "3x^{2}"), 2, ctx)

    assert a.route == "unverified" and "Unable to verify" in a.error
    assert a.extra["solver_calls"] == 3 and a.response == "FINAL_ANSWER: 8"
    assert grade("3x^{2}", a.final_text, error=a.error)["outcome"] == "error"
    # each retry gets its own seed, so retries are not word-for-word repeats
    assert [c["seed"] for c in a.extra["calls"]] == [2, 1002, 2002, 3002]


def test_full_arm_unsupported_route_uses_plain_rag(system_arm):
    from benchmarking.datasets import Problem

    arm, ctx = system_arm("full")
    FakeOllama.answers = {
        "look for any formulas": r"A prime has two divisors, e.g. \boxed{7}",
        "Name a prime": '{"operation": "unsupported", "expression": null}',
    }
    a = arm.solve(Problem("p", "Name a prime.", "7"), 0, ctx)

    assert a.error is None and a.route == "unsupported" and a.llm_calls == 2
    assert grade("7", a.to_dict()["final_text"])["outcome"] == "correct"


@pytest.mark.parametrize("text, gold", [
    ("Steps...\nFINAL_ANSWER: x**2*exp(x)", r"x^{2} e^{x}"),
    ("FINAL_ANSWER: `sqrt(2)/2`.", r"\frac{\sqrt{2}}{2}"),
    ("FINAL_ANSWER: log(3) + C", r"\ln 3"),
    ("FINAL_ANSWER: [-2, 2]", r"\left\{-2, 2\right\}"),
])
def test_final_answer_is_converted_for_the_grader(text, gold):
    from benchmarking.arms._system import final_answer_boxed

    assert grade(gold, final_answer_boxed(text))["outcome"] == "correct"


# ---------- answer extraction (grading aid for unboxed answers) ----------

def test_needs_extraction_only_for_plain_text_answers():
    from benchmarking.extract import needs_extraction

    assert needs_extraction({"final_text": "the answer is 2x/(x^2+1)", "response": "..."})
    assert not needs_extraction({"final_text": r"so \boxed{3}", "response": "..."})
    assert not needs_extraction({"final_text": "cut off", "response": "cut off", "truncated": True})
    assert not needs_extraction({"final_text": "", "response": "", "error": "unverified"})


def test_extract_then_report_extracted_grading(fake_ollama, tiny_set, tmp_path):
    from benchmarking import extract

    # p3's answer is plain text: strict grading reads the last number (2), not the answer (7).
    # p1/p2 are boxed and keep their strict grade. The extractor prompt contains the response,
    # so the "(p3resp)" key answers the extraction call; the "prime" key answers the base call.
    FakeOllama.answers = {"(p3resp)": r"\boxed{7}", "2+3": r"so \boxed{5}", "1/2 + 1/4": r"\boxed{0.75}",
                          "prime": "My answer is the number seven. Two is even, so 2 is excluded. (p3resp)"}
    out = tmp_path / "runs"
    run.main(["--arm", "base", "--dataset", str(tiny_set), "--model", "fake:1b", "--seeds", "0",
              "--ollama-host", fake_ollama, "--out", str(out), "--run-name", "r"])
    run_dir = out / "r"
    n_before = len(FakeOllama.calls)
    extract.main([str(run_dir), "--model", "fake:1b", "--ollama-host", fake_ollama])

    calls = FakeOllama.calls[n_before:]
    assert len(calls) == 1                          # only the unboxed attempt
    sent = calls[0]["messages"][0]["content"]
    assert "(p3resp)" in sent and "Name a prime." not in sent and "7" not in sent.split("Response:")[0]
    assert calls[0]["options"]["temperature"] == 0.0

    strict = report.summarize_run(report.load_run(run_dir))
    graded = report.summarize_run(report.load_run(run_dir, grading="extracted"))
    assert graded["accuracy_mean"] == pytest.approx(1.0)
    assert graded["grading"]["changed_to_correct"] == 1 and graded["grading"]["missing"] == 0
    assert strict["accuracy_mean"] == pytest.approx(2 / 3) == pytest.approx(graded["grading"]["strict_accuracy"])
    assert "Strict" in report.to_markdown([graded], [])

    # re-running does no new work
    extract.main([str(run_dir), "--model", "fake:1b", "--ollama-host", fake_ollama])
    assert len(FakeOllama.calls) == n_before + 1


def test_extractor_none_is_an_extraction_failure():
    from benchmarking.extract import _sha, extracted_grade

    row = {"problem_id": "p", "seed": 0, "gold": "7", "outcome": "wrong_answer", "extracted": "2",
           "response": "I cannot answer this from the documents.", "final_text": None}
    rec = {"reply": r"\boxed{NONE}", "response_sha256": _sha(row["response"])}
    assert extracted_grade(row, {("p", 0): rec})["outcome"] == "extraction_failure"
    # a record made for a different response is stale and ignored
    stale = {**rec, "response_sha256": "0" * 64}
    assert extracted_grade(row, {("p", 0): stale})["source"] == "missing"


@pytest.mark.parametrize("reply, gold", [
    (r"\boxed{3x^2 sin(x) + x^3 cos(x)}", r"x^{2} \left(x \cos{\left(x \right)} + 3 \sin{\left(x \right)}\right)"),
    (r"\boxed{\frac{2}{1+4x^2}}", r"\frac{2}{4 x^{2} + 1}"),
    (r"\boxed{ln(2)}", r"\ln 2"),
])
def test_extracted_replies_with_bare_function_names_still_grade(reply, gold):
    from benchmarking.extract import latexify

    assert grade(gold, latexify(reply))["outcome"] == "correct"
    assert latexify(r"\boxed{\sin x}") == r"\boxed{\sin x}"   # already LaTeX: unchanged
