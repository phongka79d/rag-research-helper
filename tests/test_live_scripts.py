from types import SimpleNamespace

import scripts.live_test_embeddings as live_test_embeddings
import scripts.live_test_responses as live_test_responses


def configured_settings():
    return SimpleNamespace(
        OPENAI_BASE_URL="http://endpoint.test/v1",
        OPENAI_API_KEY="test-key",
        OPENAI_MODEL="test-chat",
        OPENAI_EMBEDDING_MODEL="test-embedding",
    )


def test_responses_check_uses_structured_request_and_reports_non_secret_result(
    monkeypatch, capsys
):
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.responses = SimpleNamespace(create=self.create)

        def create(self, **kwargs):
            captured["request"] = kwargs
            return SimpleNamespace(output_text='{"status":"ok"}', model="served-model")

    monkeypatch.setattr(live_test_responses, "Settings", configured_settings)
    monkeypatch.setattr(live_test_responses, "OpenAI", FakeOpenAI)

    assert live_test_responses.main() == 0
    output = capsys.readouterr().out
    assert captured["client"] == {
        "api_key": "test-key",
        "base_url": "http://endpoint.test/v1",
    }
    assert captured["request"]["model"] == "test-chat"
    assert "reasoning" not in captured["request"]
    assert captured["request"]["text"] == {"format": {"type": "json_object"}}
    assert "requested_model=test-chat" in output
    assert "test-key" not in output


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
