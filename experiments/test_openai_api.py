import unittest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from local_speculative_runtime.openai_api import create_app, format_messages_to_prompt, ChatMessage
from local_speculative_runtime.session_cache import SessionCacheAPI

class TestOpenAIAPI(unittest.TestCase):
    def setUp(self):
        # Create a mock SessionCacheAPI
        self.mock_api = MagicMock(spec=SessionCacheAPI)
        
        # Setup mock return values
        self.mock_api.create_session.return_value = {"ok": True, "error": None}
        self.mock_api.generate.return_value = {
            "ok": True,
            "text": "This is a fake completion.",
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "elapsed_sec": 0.5,
            "metadata": {"test_key": "test_value"},
            "error": None
        }
        self.mock_api.clear_session.return_value = {"ok": True}
        
        # Create app with the mock API
        self.app = create_app(api=self.mock_api)
        self.client = TestClient(self.app)

    def test_format_messages_to_prompt(self):
        messages = [
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="Hello!")
        ]
        prompt = format_messages_to_prompt(messages)
        self.assertIn("System: You are a helpful assistant.", prompt)
        self.assertIn("User: Hello!", prompt)
        self.assertTrue(prompt.endswith("Assistant:"))

    def test_chat_completions_success(self):
        payload = {
            "model": "fake-model",
            "messages": [
                {"role": "user", "content": "Tell me a joke."}
            ],
            "max_tokens": 50,
            "temperature": 0.0,
            "stream": False
        }
        
        response = self.client.post("/v1/chat/completions", json=payload)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["object"], "chat.completion")
        self.assertEqual(data["model"], "fake-model")
        self.assertIn("id", data)
        self.assertIn("created", data)
        
        # Check choices
        self.assertEqual(len(data["choices"]), 1)
        choice = data["choices"][0]
        self.assertEqual(choice["message"]["role"], "assistant")
        self.assertEqual(choice["message"]["content"], "This is a fake completion.")
        self.assertEqual(choice["finish_reason"], "stop")
        
        # Check usage
        self.assertEqual(data["usage"]["prompt_tokens"], 10)
        self.assertEqual(data["usage"]["completion_tokens"], 5)
        self.assertEqual(data["usage"]["total_tokens"], 15)
        
        # Verify API calls
        self.mock_api.create_session.assert_called_once()
        self.mock_api.generate.assert_called_once()
        self.mock_api.clear_session.assert_called_once()

    def test_chat_completions_streaming_unsupported(self):
        payload = {
            "model": "fake-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True
        }
        response = self.client.post("/v1/chat/completions", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("Streaming is not supported", response.json()["detail"])

if __name__ == "__main__":
    unittest.main()
