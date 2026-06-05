from session_cache_package import SessionCacheAPI

class AgentMockRuntime:
    def __init__(self, safe_token_limit: int = 120000, step_size: int = 512, target_model: str = None, candidate_json_path: str = None):
        self.api = SessionCacheAPI(
            model_path=target_model or "mlx-community/gemma-4-26b-a4b-it-8bit",
            candidate_json_path=candidate_json_path or "experiments/template_candidates.json",
            safe_token_limit=safe_token_limit,
            step_size=step_size
        )
        self.context = {}

    def initialize_context(self, session_id: str, static_context: str) -> dict:
        self.context[session_id] = static_context
        return self.api.create_session(session_id, static_context)

    def handle_task(self, session_id: str, task_text: str, max_tokens: int = 16) -> dict:
        return self.api.generate(session_id, task_text, max_tokens=max_tokens)

    def clear(self, session_id: str) -> dict:
        if session_id in self.context:
            del self.context[session_id]
        return self.api.clear_session(session_id)

    def stats(self) -> dict:
        return self.api.stats()
