import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.models import AnswerResponse
from backend.services import ServiceUnavailable, get_answer_service


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_demo_answer_is_labeled_and_has_sources(client):
    response = client.post("/api/v1/questions", json={"question": "Solve 2x + 3 = 7"})
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "demo"
    assert data["answer"] == "x = 2"
    assert len(data["steps"]) == 3
    assert "synthetic" in data["citations"][0]["title"]


def test_demo_does_not_fabricate_answers_for_other_questions(client):
    data = client.post("/api/v1/questions", json={"question": "Integrate x squared"}).json()
    assert data["status"] == "unsupported_demo_question"
    assert data["citations"] == []


@pytest.mark.parametrize("payload", [
    {}, {"question": ""}, {"question": "   "}, {"question": "x" * 4001},
    {"question": 123}, {"question": "x", "top_k": 0},
    {"question": "x", "top_k": 11}, {"question": "x", "top_k": True},
    {"question": "x", "unexpected": "value"},
])
def test_invalid_input(client, payload):
    assert client.post("/api/v1/questions", json=payload).status_code == 422


def test_service_can_be_replaced_and_receives_validated_request(client):
    class RagStub:
        def answer(self, request):
            assert request.question == "Find x"
            assert request.top_k == 3
            return AnswerResponse(status="no_sources", mode="live", answer="No relevant passages found.")

    app.dependency_overrides[get_answer_service] = lambda: RagStub()
    response = client.post("/api/v1/questions", json={"question": "  Find x  ", "top_k": 3})
    assert response.status_code == 200
    assert response.json()["status"] == "no_sources"
    assert response.json()["mode"] == "live"


def test_service_failure_is_a_safe_503(client):
    class UnavailableService:
        def answer(self, request):
            raise ServiceUnavailable("private upstream credentials")

    app.dependency_overrides[get_answer_service] = lambda: UnavailableService()
    response = client.post("/api/v1/questions", json={"question": "Find x"})
    assert response.status_code == 503
    assert "private" not in response.text


def test_openapi_documents_request_and_response(client):
    schema = client.get("/openapi.json").json()
    endpoint = schema["paths"]["/api/v1/questions"]["post"]
    assert "requestBody" in endpoint
    assert {"200", "422", "503"} <= endpoint["responses"].keys()
