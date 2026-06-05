from typing import Optional, Dict, Any, List
import importlib

import mlx.core as mx

from .backends import BaseInferenceBackend, BackendCapabilities, GenerationResult
import experiments.template_draft_engine as engine
from experiments.template_draft_runtime import LongInputGuard, PrefixCacheManager, CandidateRegistry
from experiments.session_cache_runtime import SessionCacheRuntime

d = importlib.import_module("mlx_vlm.generate.dispatch")

class MLXBackend(BaseInferenceBackend):
    backend_name = "mlx"

    def __init__(self, model_path: str = engine.DEFAULT_TARGET_MODEL_PATH, candidate_json_path: str = "experiments/template_candidates.json", safe_token_limit: int = 120000, step_size: int = 512, max_kv_size: Optional[int] = None):
        target_model, processor = d.load(model_path)
        tokenizer = getattr(processor, "tokenizer", processor)

        registry = CandidateRegistry(json_path=candidate_json_path)
        prefix_manager = PrefixCacheManager(max_entries=2, max_total_tokens=safe_token_limit)
        guard = LongInputGuard(safe_token_limit=safe_token_limit)
        
        self.processor = processor
        self.tokenizer = tokenizer
        self.runtime = SessionCacheRuntime(
            target_model=target_model,
            tokenizer=tokenizer,
            candidate_registry=registry,
            prefix_cache_manager=prefix_manager,
            guard=guard,
            step_size=step_size,
            max_kv_size=max_kv_size
        )

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name=self.backend_name,
            supports_prefix_cache=True,
            supports_template_verify=True,
            supports_snapshot_restore=True,
            supports_token_logprobs=False,
            backend_family="mlx",
            prefix_cache_mode="snapshot",
            state_restore_status="supported",
            template_verify_status="supported",
            tested_models=["mlx-community/gemma-4-26b-a4b-it-8bit"],
            notes=[
                "Primary MLX backend",
                "Full support for template verification and exact KV rollback via MLX state cloning."
            ]
        )

    def load(self, **kwargs) -> None:
        # Loading is handled in __init__ for MLX
        pass

    def tokenize(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids: List[int]) -> str:
        return engine.decode_text(self.tokenizer, token_ids)

    def create_session(self, session_id: str, prefix_text: str) -> Dict[str, Any]:
        formatted_prefix = engine.format_prompt(self.processor, prefix_text)
        prefix_ids = self.tokenize(formatted_prefix)
        
        res = self.runtime.create_session(session_id, formatted_prefix, prefix_ids)
        return {
            "ok": res.ok,
            "session_id": res.session_id,
            "prefix_tokens": res.prefix_tokens,
            "prefix_prefill_sec": res.prefix_prefill_sec,
            "cache_key": res.cache_key,
            "guard_allowed": res.guard_allowed,
            "guard_reason": res.guard_reason,
            "evicted_keys": res.evicted_keys
        }

    def generate(self, session_id: Optional[str], prompt_or_suffix: str, max_tokens: int = 16, **kwargs) -> GenerationResult:
        if session_id is None:
            return GenerationResult(
                ok=False, text="", token_ids=[], elapsed_sec=0.0, prompt_tokens=0, completion_tokens=0, error="session_id cannot be None for MLX backend", backend=self.backend_name, metadata={}
            )

        draft_block_size = kwargs.get("draft_block_size", 8)
        template_min_tokens = kwargs.get("template_min_tokens", 1)
        trace = kwargs.get("trace", False)

        if session_id not in self.runtime.sessions:
            return GenerationResult(
                ok=False, text="", token_ids=[], elapsed_sec=0.0, prompt_tokens=None, completion_tokens=None, error=f"Session {session_id} not found", backend=self.backend_name, metadata={}
            )
            
        session_state = self.runtime.sessions[session_id]
        if session_state.prefix_key not in self.runtime.prefix_manager.entries:
            return GenerationResult(
                ok=False, text="", token_ids=[], elapsed_sec=0.0, prompt_tokens=None, completion_tokens=None, error=f"Prefix cache for session {session_id} has been evicted", backend=self.backend_name, metadata={}
            )
            
        entry = self.runtime.prefix_manager.entries[session_state.prefix_key]
        prefix_ids = entry.token_ids
        prefix_text = self.detokenize(prefix_ids)
        
        formatted_full = engine.format_prompt(self.processor, prefix_text + "\n" + prompt_or_suffix)
        full_ids = self.tokenize(formatted_full)
        suffix_ids = full_ids[len(prefix_ids):]
        
        res = self.runtime.generate_with_suffix(
            session_id=session_id,
            suffix_text=prompt_or_suffix,
            suffix_ids=suffix_ids,
            max_tokens=max_tokens,
            draft_block_size=draft_block_size,
            template_min_tokens=template_min_tokens,
            trace=trace
        )
        
        mx.clear_cache()
        
        return GenerationResult(
            ok=res.ok,
            text=res.text,
            token_ids=res.token_ids,
            elapsed_sec=res.elapsed_sec,
            prompt_tokens=res.suffix_tokens,
            completion_tokens=len(res.token_ids),
            error=res.error,
            backend=self.backend_name,
            metadata={
                "suffix_prefill_sec": res.suffix_prefill_sec,
                "decode_sec": res.decode_sec,
                "accepted": res.accepted,
                "drafted": res.drafted,
                "rejected": res.rejected,
                "candidate_name": res.candidate_name,
                "fallback_used": res.fallback_used
            }
        )

    def clear_session(self, session_id: str, drop_cache: bool = False) -> Dict[str, Any]:
        return self.runtime.clear_session(session_id, drop_cache=drop_cache)

    def stats(self) -> Dict[str, Any]:
        s = self.runtime.stats()
        return {
            "entries": s.entries,
            "current_total_tokens": s.current_total_tokens,
            "max_entries": s.max_entries,
            "max_total_tokens": s.max_total_tokens,
            "keys": s.keys,
            "sessions": len(self.runtime.sessions)
        }
