import base64

from sensei.vlm_ollama import OLLAMA_SCHEMA, OllamaClient


def test_request_body_structure():
    client = OllamaClient(model="qwen3-vl:8b")
    body = client._request_body("hello", [b"fakepng"])
    assert body["model"] == "qwen3-vl:8b"
    assert body["stream"] is False
    assert body["format"] == OLLAMA_SCHEMA
    assert body["messages"][0]["role"] == "user"
    assert body["messages"][0]["content"] == "hello"
    assert body["messages"][0]["images"] == [base64.b64encode(b"fakepng").decode()]
    assert body["logprobs"] is True
    assert body["options"]["num_ctx"] > 0


def test_extract_level_probs_softmax():
    payload = {
        "message": {
            "logprobs": [
                {
                    "token": "full",
                    "logprob": -0.1,
                    "top_logprobs": [
                        {"token": "empty", "logprob": -2.0},
                        {"token": "half", "logprob": -1.0},
                        {"token": "full", "logprob": -0.1},
                    ],
                }
            ]
        }
    }
    probs = OllamaClient.extract_level_probs(payload)
    assert probs is not None
    assert set(probs) == {"empty", "half", "full"}
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    assert probs["full"] > probs["half"] > probs["empty"]


def test_extract_level_probs_from_first_token_when_not_in_top():
    payload = {
        "message": {
            "logprobs": [
                {
                    "token": "empty",
                    "logprob": -0.05,
                    "top_logprobs": [
                        {"token": "half", "logprob": -1.5},
                        {"token": "full", "logprob": -2.5},
                        {"token": "the", "logprob": -3.0},
                    ],
                }
            ]
        }
    }
    probs = OllamaClient.extract_level_probs(payload)
    assert probs is not None
    assert probs["empty"] == max(probs.values())


def test_extract_level_probs_missing_tokens_returns_none():
    payload = {
        "message": {
            "logprobs": [
                {
                    "token": "I",
                    "logprob": 0.0,
                    "top_logprobs": [{"token": "I", "logprob": 0.0}],
                }
            ]
        }
    }
    assert OllamaClient.extract_level_probs(payload) is None


def test_extract_level_probs_empty_payload():
    assert OllamaClient.extract_level_probs({}) is None
    assert OllamaClient.extract_level_probs({"message": {}}) is None
