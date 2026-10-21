"""Exercise the real load/search/evaluation commands with isolated indexes."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("connection", ["environment", "flag"])
def test_commands_share_the_configured_endpoint(review_index, tmp_path, connection):
    client, index = review_index
    endpoint = os.environ.get("H2FLOW_TEST_ES_URL", "http://localhost:9200")
    env = os.environ.copy()
    env["H2FLOW_ES_URL"] = endpoint if connection == "environment" else "http://127.0.0.1:1"
    flags = ["--es-url", endpoint] if connection == "flag" else []
    repo = Path(__file__).resolve().parents[1]

    def run(module, *arguments):
        result = subprocess.run(
            [sys.executable, "-m", module, *arguments, "--index", index, *flags],
            cwd=repo,
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout

    loaded = run("search.index", "--fixtures", "--reset")
    assert f"chunks into {index}" in loaded
    assert client.count(index=index)["count"] > 0

    retrieved = run("search.retrieve", "exponential population growth")
    assert "m53481" in retrieved

    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps([
        {"question": "exponential population growth", "expect": ["m53481"]},
    ]))
    evaluated = run("search.evaluate", "--questions", str(questions))
    assert "hit@1 1/1" in evaluated
