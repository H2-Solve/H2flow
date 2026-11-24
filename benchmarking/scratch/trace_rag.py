"""Live trace of the rag and full arms: what actually reaches Elasticsearch and Ollama.

Checks, per problem:
  1. every model call goes over HTTP through the harness client (count at the HTTP layer == recorded llm_calls)
  2. the system's own ChatOllama objects are never called (they are made to raise if anything bypasses the swap)
  3. exactly one Elasticsearch search per problem, and the chunks recorded == the chunks ES returned
  4. the full text of every retrieved chunk appears verbatim in the prompt sent to the model
  5. the module attributes are restored afterwards
  6. ablation: with retrieval forced empty, the prompt loses the context (proves retrieval feeds the prompt)
"""


def main():
    import json
    import os
    import sys

    os.environ.setdefault("OLLAMA_HOST", "http://127.0.0.1:11434")
    os.environ.setdefault("H2FLOW_ES_URL", "http://127.0.0.1:9201")
    sys.path.insert(0, os.getcwd())

    from elasticsearch import Elasticsearch
    from langchain_ollama import ChatOllama

    import RAG.main as rag_main
    import verification
    from benchmarking.arms import RunContext, get_arm
    from benchmarking.datasets import Problem, load_dataset
    from benchmarking.grading import grade
    from benchmarking.ollama_client import OllamaClient

    print("RAG.main loaded from:", rag_main.__file__)
    print("verification loaded from:", verification.__file__)

    # --- instrumentation ---
    http_chats, es_searches, bypass = [], [], []

    orig_request = OllamaClient._request
    def logged_request(self, path, payload=None):
        res = orig_request(self, path, payload)
        if path == "/api/chat":
            http_chats.append({"payload": payload, "reply": (res.get("message") or {}).get("content", "")})
        return res
    OllamaClient._request = logged_request

    orig_search = Elasticsearch.search
    def logged_search(self, *a, **kw):
        res = orig_search(self, *a, **kw)
        es_searches.append({"index": kw.get("index"), "query": kw.get("query"),
                            "hits": [h["_source"] for h in res["hits"]["hits"]]})
        return res
    Elasticsearch.search = logged_search

    def no_bypass(self, *a, **kw):
        bypass.append(self.model)
        raise RuntimeError("BYPASS: the system's own ChatOllama was called")
    ChatOllama.invoke = no_bypass

    originals = {"llm": rag_main.llm, "solver_llm": rag_main.solver_llm, "get_documents": rag_main.get_documents,
                 "solve_with_sympy": rag_main.solve_with_sympy, "extractor_llm": verification.extractor_llm}

    client = OllamaClient()
    ctx = RunContext(client=client, model="gemma3:4b",
                     options={"temperature": 0.0, "num_ctx": 8192, "num_predict": 4096},
                     prompt_template="{problem}", prompt_name="base_v1")

    calc = {p.id: p for p in load_dataset("calculus").problems}
    math = load_dataset("math500").problems
    problems = [calc["lim-01"], calc[next(k for k in calc if k.startswith("int"))], math[0]]
    unsupported = Problem("concept", "Explain in words why the derivative of a constant is zero.", "0")

    failures = []
    def check(cond, msg):
        print(("  PASS " if cond else "  FAIL ") + msg)
        if not cond:
            failures.append(msg)


    def run(arm_name, problem):
        arm = get_arm(arm_name)
        arm.setup(ctx)
        n_http, n_es = len(http_chats), len(es_searches)
        a = arm.solve(problem, 0, ctx)
        calls, searches = http_chats[n_http:], es_searches[n_es:]
        g = grade(problem.answer_latex, a.to_dict()["final_text"], truncated=a.truncated, error=a.error)
        print(f"\n[{arm_name}] {problem.id}: route={a.route} outcome={g['outcome']} error={a.error}")
        check(not bypass, "no call reached the system's own ChatOllama")
        check(len(calls) == a.llm_calls, f"HTTP chat calls ({len(calls)}) == recorded llm_calls ({a.llm_calls})")
        check(len(searches) == 1, f"exactly one Elasticsearch search ({len(searches)})")
        if searches:
            s = searches[0]
            check(s["index"] == "textbook-chunks", f"searched index {s['index']!r}")
            check(len(s["hits"]) == 5, f"ES returned {len(s['hits'])} hits")
            check([h.get("module_id") for h in s["hits"]] == [d["module_id"] for d in a.extra["retrieved"]],
                  "recorded chunks == chunks ES returned")
            empty = [h.get("module_id") for h in s["hits"] if not (h.get("text") or "").strip()]
            check(not empty, f"no retrieved chunk has empty text {empty}")
            # the prompt that carries context: the answer/solver call (not the intent call)
            ctx_calls = [c for c in calls if "Context:" in "\n".join(m["content"] for m in c["payload"]["messages"])]
            check(len(ctx_calls) == sum(c["role"] != "intent" for c in a.extra["calls"]),
                  f"every answer/solver call carries a Context block ({len(ctx_calls)})")
            for c in ctx_calls:
                sent = "\n".join(m["content"] for m in c["payload"]["messages"])
                missing = [h.get("module_id") for h in s["hits"] if h.get("text", "") not in sent]
                check(not missing, f"all 5 chunk texts appear verbatim in the prompt (missing: {missing})")
                check(problem.problem in sent, "the question is in the prompt")
                check(c["payload"]["options"]["num_ctx"] == 8192, "num_ctx 8192 sent")
                print(f"    prompt: {len(sent)} chars, {sum(len(h.get('text','')) for h in s['hits'])} of them chunk text; "
                      f"roles {[m['role'] for m in c['payload']['messages']]}")
            intent_calls = [c for c in calls if c["payload"].get("format") == "json"]
            if arm_name == "full":
                check(len(intent_calls) == 1, f"one JSON intent call ({len(intent_calls)})")
        restored = (rag_main.llm is originals["llm"] and rag_main.solver_llm is originals["solver_llm"]
                    and rag_main.get_documents is originals["get_documents"]
                    and rag_main.solve_with_sympy is originals["solve_with_sympy"]
                    and verification.extractor_llm is originals["extractor_llm"])
        check(restored, "system module attributes restored after solve")
        return a, calls


    for p in problems:
        run("rag", p)
    for p in problems[:2] + [unsupported]:
        run("full", p)

    # --- ablation: same problem, retrieval forced empty ---
    print("\n[ablation] rag on lim-01 with retrieval returning no chunks")
    real = rag_main.retrieve_documents
    rag_main.retrieve_documents = lambda *a, **k: []
    arm = get_arm("rag"); arm.setup(ctx)
    n = len(http_chats)
    a = arm.solve(calc["lim-01"], 0, ctx)
    rag_main.retrieve_documents = real
    sent = "\n".join(m["content"] for m in http_chats[n]["payload"]["messages"])
    print(f"  prompt tokens without context: {a.prompt_tokens}; prompt: {sent[-250:]!r}")
    check("Document 1:" not in sent, "no documents in the prompt when retrieval is empty")
    check(a.extra["retrieved"] == [], "empty retrieval is visible in the attempt record")

    print("\nSAMPLE PROMPT (rag, lim-01), first 1200 chars:")
    first = next(c for c in http_chats if "Context:" in c["payload"]["messages"][0]["content"])
    print(first["payload"]["messages"][0]["content"][:1200])

    print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILED: {failures}"))


if __name__ == "__main__":  # the SymPy child process re-imports this file
    main()
