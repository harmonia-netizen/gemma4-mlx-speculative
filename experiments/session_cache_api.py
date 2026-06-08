import importlib

import mlx.core as mx
import template_draft_engine as engine
from template_draft_runtime import LongInputGuard, PrefixCacheManager, CandidateRegistry
from session_cache_runtime import SessionCacheRuntime

from local_speculative_runtime.backends import BaseInferenceBackend
from local_speculative_runtime.mlx_backend import MLXBackend
from local_speculative_runtime.llama_cpp_backend import LlamaCppBackend

class SessionCacheAPI:
    def __init__(self, backend_impl: BaseInferenceBackend):
        self.backend = backend_impl

    @classmethod
    def load(cls, model_path: str = engine.DEFAULT_TARGET_MODEL_PATH, candidate_json_path: str = "experiments/template_candidates.json", backend: str = "mlx", **kwargs):
        if backend == "mlx":
            safe_token_limit = kwargs.get("safe_token_limit", 120000)
            step_size = kwargs.get("step_size", 512)
            max_kv_size = kwargs.get("max_kv_size", None)
            impl = MLXBackend(
                model_path=model_path,
                candidate_json_path=candidate_json_path,
                safe_token_limit=safe_token_limit,
                step_size=step_size,
                max_kv_size=max_kv_size
            )
        elif backend in ["llama_cpp", "gguf"]:
            n_ctx = kwargs.get("n_ctx", 2048)
            n_gpu_layers = kwargs.get("n_gpu_layers", -1)
            n_threads = kwargs.get("n_threads", None)
            seed = kwargs.get("seed", 1337)
            verbose = kwargs.get("verbose", False)
            chat_format = kwargs.get("chat_format", None)
            generation_mode = kwargs.get("generation_mode", "low-level")
            impl = LlamaCppBackend(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                seed=seed,
                verbose=verbose,
                chat_format=chat_format,
                candidate_json_path=candidate_json_path,
                generation_mode=generation_mode
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")
            
        return cls(impl)

    def create_session(self, session_id: str, prefix_text: str) -> dict:
        return self.backend.create_session(session_id, prefix_text)

    def generate(self, session_id: str, suffix_text: str, max_tokens: int = 16, draft_block_size: int = 8, template_min_tokens: int = 1, trace: bool = False, **kwargs) -> dict:
        # Generate on backend
        res = self.backend.generate(
            session_id=session_id, 
            prompt_or_suffix=suffix_text, 
            max_tokens=max_tokens,
            draft_block_size=draft_block_size,
            template_min_tokens=template_min_tokens,
            trace=trace,
            **kwargs
        )
        
        # Format response to match legacy dictionary shape
        return {
            "ok": res.ok,
            "session_id": session_id,
            "text": res.text,
            "token_ids": res.token_ids,
            "suffix_tokens": res.prompt_tokens,
            "suffix_prefill_sec": res.metadata.get("suffix_prefill_sec", 0.0),
            "decode_sec": res.metadata.get("decode_sec", 0.0),
            "elapsed_sec": res.elapsed_sec,
            "accepted": res.metadata.get("accepted", 0),
            "drafted": res.metadata.get("drafted", 0),
            "rejected": res.metadata.get("rejected", 0),
            "candidate_name": res.metadata.get("candidate_name", None),
            "fallback_used": res.metadata.get("fallback_used", False),
            "error": res.error,
            "backend": res.backend,
            "metadata": res.metadata,
            "prompt_tokens": res.prompt_tokens,
            "completion_tokens": res.completion_tokens
        }

    def clear_session(self, session_id: str, drop_cache: bool = False) -> dict:
        return self.backend.clear_session(session_id, drop_cache=drop_cache)

    def stats(self) -> dict:
        return self.backend.stats()

