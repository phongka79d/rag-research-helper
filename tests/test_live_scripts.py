from types import SimpleNamespace

import scripts.live_test_embeddings as live_test_embeddings
import scripts.live_test_responses as live_test_responses


def configured_settings():
    return SimpleNamespace(
        OPENAI_BASE_URL="http://endpoint.test/v1",
        OPENAI_API_KEY="test-key",
        OPENAI_MODEL="test-chat",
        OPENAI_GRAPH_MODEL="",
        OPENAI_EMBEDDING_MODEL="test-embedding",
    )


def test_responses_check_uses_structured_request_and_reports_non_secret_result(
    monkeypatch, capsys
):
    captured = {"requests": []}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.responses = SimpleNamespace(create=self.create)

        def create(self, **kwargs):
            captured["requests"].append(kwargs)
            return SimpleNamespace(output_text='{"status":"ok"}', model="served-model")

    monkeypatch.setattr(live_test_responses, "Settings", configured_settings)
    monkeypatch.setattr(live_test_responses, "OpenAI", FakeOpenAI)

    assert live_test_responses.main() == 0
    output = capsys.readouterr().out
    assert captured["client"] == {
        "api_key": "test-key",
        "base_url": "http://endpoint.test/v1",
    }
    assert [request["model"] for request in captured["requests"]] == ["test-chat"]
    assert "reasoning" not in captured["requests"][0]
    assert captured["requests"][0]["text"] == {"format": {"type": "json_object"}}
    assert "role=text, requested_model=test-chat" in output
    assert "text_model=test-chat, graph_model=test-chat, requests=1" in output
    assert "test-key" not in output


def test_responses_check_tests_distinct_graph_model(monkeypatch, capsys):
    captured = []

    def configured_with_graph():
        settings = configured_settings()
        settings.OPENAI_GRAPH_MODEL = "graph-model"
        return settings

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.responses = SimpleNamespace(create=self.create)

        def create(self, **kwargs):
            captured.append(kwargs)
            return SimpleNamespace(output_text='{"status":"ok"}', model="served")

    monkeypatch.setattr(live_test_responses, "Settings", configured_with_graph)
    monkeypatch.setattr(live_test_responses, "OpenAI", FakeOpenAI)

    assert live_test_responses.main() == 0
    output = capsys.readouterr().out
    assert [request["model"] for request in captured] == ["test-chat", "graph-model"]
    assert "role=graph, requested_model=graph-model" in output
    assert "text_model=test-chat, graph_model=graph-model, requests=2" in output


def test_embeddings_check_validates_vector_and_reports_size(monkeypatch, capsys):
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.embeddings = SimpleNamespace(create=self.create)

        def create(self, **kwargs):
            captured["request"] = kwargs
            return SimpleNamespace(
                data=[SimpleNamespace(embedding=[0.0, 1.5])],
                model="served-embedding",
            )

    monkeypatch.setattr(live_test_embeddings, "Settings", configured_settings)
    monkeypatch.setattr(live_test_embeddings, "OpenAI", FakeOpenAI)

    assert live_test_embeddings.main() == 0
    output = capsys.readouterr().out
    assert captured["client"] == {
        "api_key": "test-key",
        "base_url": "http://endpoint.test/v1",
    }
    assert captured["request"] == {
        "model": "test-embedding",
        "input": "OpenAI-compatible embedding compatibility check.",
    }
    assert "requested_model=test-embedding" in output
    assert "vector_size=2" in output
    assert "response_model=served-embedding" in output
    assert "test-key" not in output


def test_responses_check_redacts_provider_error(monkeypatch, capsys):
    class FailingOpenAI:
        def __init__(self, **kwargs):
            self.responses = SimpleNamespace(create=self.create)

        def create(self, **kwargs):
            raise RuntimeError("provider rejected test-key")

    monkeypatch.setattr(live_test_responses, "Settings", configured_settings)
    monkeypatch.setattr(live_test_responses, "OpenAI", FailingOpenAI)

    assert live_test_responses.main() == 1
    error = capsys.readouterr().err
    assert "Responses check failed" in error
    assert "test-key" not in error
    assert "[redacted]" in error


def test_embeddings_check_redacts_provider_error(monkeypatch, capsys):
    class FailingOpenAI:
        def __init__(self, **kwargs):
            self.embeddings = SimpleNamespace(create=self.create)

        def create(self, **kwargs):
            raise RuntimeError("provider rejected test-key")

    monkeypatch.setattr(live_test_embeddings, "Settings", configured_settings)
    monkeypatch.setattr(live_test_embeddings, "OpenAI", FailingOpenAI)

    assert live_test_embeddings.main() == 1
    error = capsys.readouterr().err
    assert "Embeddings check failed" in error
    assert "test-key" not in error
    assert "[redacted]" in error
