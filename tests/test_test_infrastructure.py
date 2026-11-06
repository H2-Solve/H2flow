"""Check test setup through actual pytest subprocesses and service boundaries."""

import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys


TESTS = Path(__file__).resolve().parent


def run_tests(directory, endpoint, *arguments, extra_env=None):
    shutil.copy2(TESTS / "conftest.py", directory / "conftest.py")
    env = os.environ.copy()
    env.update({
        "H2FLOW_TEST_ES_URL": endpoint,
        "PYTHONPATH": str(TESTS.parent),
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    env.update(extra_env or {})
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=short", "-p", "no:cacheprovider", *arguments],
        cwd=directory,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_explicit_unavailable_server_is_an_error_for_original_integrations(tmp_path):
    for name in ("test_analysis.py", "test_query.py", "test_search.py"):
        shutil.copy2(TESTS / name, tmp_path / name)
    with socket.socket() as unused:
        unused.bind(("127.0.0.1", 0))
        endpoint = f"http://127.0.0.1:{unused.getsockname()[1]}"
        result = run_tests(
            tmp_path,
            endpoint,
            "test_analysis.py::test_elasticsearch_accepts_the_analyzer",
            "test_query.py::test_section_field_finds_chunks_the_text_alone_misses",
            "test_query.py::test_demoted_sections_rank_lower",
            "test_search.py::test_search_integration",
        )

    assert result.returncode == 1, result.stdout + result.stderr
    assert "Configured Elasticsearch test instance is unreachable" in result.stdout
    assert "4 errors" in result.stdout


def test_temporary_index_is_removed_after_an_assertion_failure(review_es_client, tmp_path):
    index_file = tmp_path / "created-index.txt"
    (tmp_path / "test_failure.py").write_text(
        "import os\nfrom pathlib import Path\n"
        "def test_failure(review_index):\n"
        "    _, index = review_index\n"
        "    Path(os.environ['H2FLOW_CREATED_INDEX_FILE']).write_text(index)\n"
        "    assert False, 'intentional cleanup probe'\n"
    )
    result = run_tests(
        tmp_path,
        os.environ.get("H2FLOW_TEST_ES_URL", "http://localhost:9200"),
        "test_failure.py",
        extra_env={"H2FLOW_CREATED_INDEX_FILE": str(index_file)},
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert "intentional cleanup probe" in result.stdout
    assert not review_es_client.indices.exists(index=index_file.read_text())
