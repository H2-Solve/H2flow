"""Scoring helpers for the retrieval evaluation. No Elasticsearch needed."""

import json

from search.evaluate import first_rank, run_eval, summarize


def hits(*module_ids):
    return [{"module_id": m, "section": "", "title": ""} for m in module_ids]


def test_first_rank_is_one_based_and_accepts_alternatives():
    assert first_rank(hits("a", "b", "c"), ["c"]) == 3
    assert first_rank(hits("a", "b", "c"), ["x", "b"]) == 2
    assert first_rank(hits("a", "b"), ["z"]) is None
    assert first_rank([], ["a"]) is None


def test_summarize_counts_and_mrr():
    s = summarize([1, 3, None, 5, 2])
    assert (s["hit@1"], s["hit@3"], s["hit@5"], s["n"]) == (1, 3, 4, 5)
    assert abs(s["mrr"] - (1 + 1 / 3 + 0 + 1 / 5 + 1 / 2) / 5) < 1e-9
    assert summarize([])["mrr"] == 0.0


def test_run_eval_uses_the_search_function():
    questions = [
        {"question": "q1", "expect": ["m2"]},
        {"question": "q2", "expect": ["m9"]},
    ]
    fake = lambda query, k: hits("m1", "m2") if query == "q1" else hits("m1")
    results = run_eval(questions, fake, top_k=5)
    assert [r["rank"] for r in results] == [2, None]


def test_starter_questions_file_is_well_formed():
    with open("eval/questions.json", encoding="utf-8") as f:
        questions = json.load(f)
    assert len(questions) >= 10
    assert all(
        q["question"] and q["expect"] and all(m.startswith("m") for m in q["expect"])
        for q in questions
    )
