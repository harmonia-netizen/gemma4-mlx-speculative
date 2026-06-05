import importlib

import mlx.core as mx
import template_draft_engine as engine
from template_draft_runtime import LongInputGuard, PrefixCacheManager, CandidateRegistry
from session_cache_runtime import SessionCacheRuntime

d = importlib.import_module("mlx_vlm.generate.dispatch")

class SessionCacheAPI:
    def __init__(self, model_path: str = engine.DEFAULT_TARGET_MODEL_PATH, candidate_json_path: str = "experiments/template_candidates.json", safe_token_limit: int = 120000, step_size: int = 512, max_kv_size: int = None):
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

    def create_session(self, session_id: str, prefix_text: str) -> dict:
        formatted_prefix = engine.format_prompt(self.processor, prefix_text)
        prefix_ids = self.tokenizer.encode(formatted_prefix)
        
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

    def generate(self, session_id: str, suffix_text: str, max_tokens: int = 16, draft_block_size: int = 8, template_min_tokens: int = 1, trace: bool = False) -> dict:
        if session_id not in self.runtime.sessions:
            return {"ok": False, "error": f"Session {session_id} not found"}
            
        session_state = self.runtime.sessions[session_id]
        if session_state.prefix_key not in self.runtime.prefix_manager.entries:
            return {"ok": False, "error": f"Prefix cache for session {session_id} has been evicted"}
            
        entry = self.runtime.prefix_manager.entries[session_state.prefix_key]
        prefix_ids = entry.token_ids
        prefix_text = engine.decode_text(self.tokenizer, prefix_ids)
        
        formatted_full = engine.format_prompt(self.processor, prefix_text + "\n" + suffix_text)
        full_ids = self.tokenizer.encode(formatted_full)
        suffix_ids = full_ids[len(prefix_ids):]
        
        res = self.runtime.generate_with_suffix(
            session_id=session_id,
            suffix_text=suffix_text,
            suffix_ids=suffix_ids,
            max_tokens=max_tokens,
            draft_block_size=draft_block_size,
            template_min_tokens=template_min_tokens,
            trace=trace
        )
        
        mx.clear_cache()
        
        return {
            "ok": res.ok,
            "session_id": res.session_id,
            "text": res.text,
            "token_ids": res.token_ids,
            "suffix_tokens": res.suffix_tokens,
            "suffix_prefill_sec": res.suffix_prefill_sec,
            "decode_sec": res.decode_sec,
            "elapsed_sec": res.elapsed_sec,
            "accepted": res.accepted,
            "drafted": res.drafted,
            "rejected": res.rejected,
            "candidate_name": res.candidate_name,
            "fallback_used": res.fallback_used,
            "error": res.error
        }

    def clear_session(self, session_id: str) -> dict:
        self.runtime.clear_session(session_id)
        return {"ok": True, "session_id": session_id}

    def stats(self) -> dict:
        s = self.runtime.stats()
        return {
            "entries": s.entries,
            "current_total_tokens": s.current_total_tokens,
            "max_entries": s.max_entries,
            "max_total_tokens": s.max_total_tokens,
            "keys": s.keys
        }
