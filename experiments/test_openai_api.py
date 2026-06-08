import unittest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from local_speculative_runtime.openai_api import create_app, format_messages_to_prompt, split_messages_for_session, ChatMessage
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
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content="Hello!")
        ]
        prompt = format_messages_to_prompt(messages)
        self.assertIn("System: You are helpful.", prompt)
        self.assertIn("User: Hello!", prompt)
        self.assertTrue(prompt.endswith("Assistant:"))

    def test_split_messages_for_session_single(self):
        messages = [ChatMessage(role="user", content="Hi")]
        prefix, suffix = split_messages_for_session(messages)
        self.assertEqual(prefix, "")
        self.assertEqual(suffix, "User: Hi\n\nAssistant:")
        
    def test_split_messages_for_session_multiple(self):
        messages = [
            ChatMessage(role="system", content="Sys"),
            ChatMessage(role="user", content="U1"),
            ChatMessage(role="assistant", content="A1"),
            ChatMessage(role="user", content="U2"),
        ]
        prefix, suffix = split_messages_for_session(messages)
        self.assertEqual(prefix, "System: Sys\n\nUser: U1\n\nAssistant: A1\n\n")
        self.assertEqual(suffix, "User: U2\n\nAssistant:")
        
    def test_split_messages_for_session_empty(self):
        with self.assertRaises(ValueError):
            split_messages_for_session([])

    def test_list_models(self):
        # We did not set LSR_MODEL, so app.state.model_id defaults to "local-model"
        response = self.client.get("/v1/models")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["object"], "list")
        self.assertIsInstance(data["data"], list)
        self.assertEqual(len(data["data"]), 1)
        
        model_obj = data["data"][0]
        self.assertEqual(model_obj["object"], "model")
        self.assertEqual(model_obj["id"], "local-model")
        self.assertEqual(model_obj["owned_by"], "local-speculative-runtime")
        self.assertIn("created", model_obj)

    def test_chat_completions_success(self):
        payload = {
            "model": "fake-model",
            "messages": [
                {"role": "system", "content": "Sys"},
                {"role": "user", "content": "Hi"}
            ],
            "max_tokens": 16
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
        
        # Verify header is absent
        self.assertNotIn("X-LSR-Warning", response.headers)
        
        # Verify API calls
        self.mock_api.create_session.assert_called_once()
        create_args, create_kwargs = self.mock_api.create_session.call_args
        self.assertEqual(create_kwargs.get("prefix_text"), "System: Sys\n\n")
        
        self.mock_api.generate.assert_called_once()
        generate_args, generate_kwargs = self.mock_api.generate.call_args
        self.assertEqual(generate_kwargs.get("suffix_text"), "User: Hi\n\nAssistant:")
        
        self.mock_api.clear_session.assert_called_once()

    def test_chat_completions_streaming_fallback(self):
        payload = {
            "model": "fake-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True
        }
        response = self.client.post("/v1/chat/completions", json=payload)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["object"], "chat.completion")
        self.assertIn("choices", data)
        self.assertEqual(data["choices"][0]["message"]["content"], "This is a fake completion.")
        
        # Verify header is present
        self.assertIn("X-LSR-Warning", response.headers)
        self.assertIn("stream=true is not supported", response.headers["X-LSR-Warning"])

    def test_chat_completions_empty_messages(self):
        payload = {
            "model": "fake-model",
            "messages": [],
            "max_tokens": 16
        }
        response = self.client.post("/v1/chat/completions", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("messages list cannot be empty", response.json()["detail"])

if __name__ == "__main__":
    unittest.main()
