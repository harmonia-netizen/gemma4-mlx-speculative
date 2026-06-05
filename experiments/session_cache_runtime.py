import time
from dataclasses import dataclass
from typing import Dict, Any

import mlx.core as mx
import template_draft_engine as engine
from template_draft_runtime import LongInputGuard, PrefixCacheManager, CandidateRegistry

@dataclass
class SessionState:
    session_id: str
    prefix_key: str
    turn_count: int
    created_at: float
    last_used_at: float

class SessionCacheRuntime:
    def __init__(self, target_model, tokenizer, candidate_registry: CandidateRegistry, prefix_cache_manager: PrefixCacheManager, guard: LongInputGuard, step_size: int = 512, max_kv_size: int = None):
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.candidate_registry = candidate_registry
        self.prefix_manager = prefix_cache_manager
        self.guard = guard
        self.step_size = step_size
        self.max_kv_size = max_kv_size
        self.sessions: Dict[str, SessionState] = {}
        self.stop_ids = engine.build_stop_ids(tokenizer)

    def create_session(self, session_id: str, prefix_text: str, prefix_ids: list[int]) -> bool:
        prompt_tokens = len(prefix_ids)
        guard_result = self.guard.validate(prompt_tokens, use_chunked_prefill=True)
        if not guard_result.allowed:
            print(f"skipped_by_guard: {guard_result.reason}")
            return False

        # This will internally do a chunked prefill (step_size=512) and take a snapshot, and evict if needed
        entry = self.prefix_manager.get_or_create(prefix_text, prefix_ids, self.target_model, self.max_kv_size)
        
        now = time.time()
        self.sessions[session_id] = SessionState(
            session_id=session_id,
            prefix_key=entry.text_hash,
            turn_count=0,
            created_at=now,
            last_used_at=now
        )
        return True

    def generate_with_suffix(self, session_id: str, suffix_text: str, suffix_ids: list[int], max_tokens: int = 16, draft_block_size: int = 8, template_min_tokens: int = 1, trace: bool = False):
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
            
        session = self.sessions[session_id]
        session.turn_count += 1
        session.last_used_at = time.time()
        
        prefix_key = session.prefix_key
        if prefix_key not in self.prefix_manager.entries:
            raise RuntimeError(f"Prefix cache for session {session_id} has been evicted")
            
        entry = self.prefix_manager.entries[prefix_key]
        entry.last_used_at = time.time()
        entry.hit_count += 1
        
        lm = engine.get_lm(self.target_model)
        total_start = time.perf_counter()
        
        engine.restore_full(entry.cache, entry.snapshot)
        
        if len(suffix_ids) == 0:
            raise ValueError("suffix_ids cannot be empty")
            
        input_arr = mx.array([suffix_ids])
        emb = self.target_model.get_input_embeddings(input_arr, None, mask=None)
        inputs_embeds = emb.inputs_embeds
        extra = {k: v for k, v in emb.to_dict().items() if k != "inputs_embeds" and v is not None}
        
        if input_arr.shape[1] > 1:
            n = input_arr.shape[1] - 1
            lm(
                input_arr[:, :n],
                inputs_embeds=inputs_embeds[:, :n],
                cache=entry.cache,
                n_to_process=n,
                **extra,
            )
            engine.eval_cache(entry.cache)
            
        cur = input_arr[:, -1:]
        prefill_sec = time.perf_counter() - total_start
        
        decode_start = time.perf_counter()
        out = []
        
        first_logits = engine.forward_one(lm, cur, entry.cache)
        tok = engine.argmax_token(first_logits)
        first_id = int(tok.item())
        
        if first_id in self.stop_ids:
            decode_sec = time.perf_counter() - decode_start
            return engine.DecodeResult(
                "", [], time.perf_counter() - total_start, prefill_sec, decode_sec, 0.0
            )

        out.append(first_id)
        target_next_logits = engine.forward_one(lm, tok[:, None], entry.cache)
        
        accepted = 0
        drafted = 0
        rejected = 0
        
        candidate = self.candidate_registry.select_candidate(suffix_text, self.tokenizer, template_min_tokens, trace)
        candidate_ids = engine.encode_candidate(self.tokenizer, candidate) if candidate else []
        
        if candidate_ids and candidate_ids[0] == first_id:
            candidate_ids = candidate_ids[1:]
            
        if len(candidate_ids) < template_min_tokens:
            candidate_ids = []
            
        cursor = 0
        template_disabled = False
        
        while len(out) < max_tokens:
            remaining = max_tokens - len(out)
            if remaining <= 0:
                break
                
            proposal_ids = []
            if not template_disabled and cursor < len(candidate_ids):
                proposal_ids = candidate_ids[cursor : cursor + min(draft_block_size, remaining)]
                
            if not proposal_ids:
                tok = engine.argmax_token(target_next_logits)
                tid = int(tok.item())
                if tid in self.stop_ids:
                    break
                out.append(tid)
                target_next_logits = engine.forward_one(lm, tok[:, None], entry.cache)
                continue
                
            drafted += len(proposal_ids)
            
            if any(p in self.stop_ids for p in proposal_ids):
                if trace:
                    print("trace: reject due to stop token in proposal")
                template_disabled = True
                candidate_ids = []
                cursor = 0
                continue
                
            target_tok = engine.argmax_token(target_next_logits)
            target_id = int(target_tok.item())
            
            if target_id != proposal_ids[0]:
                if trace:
                    print(f"trace: reject at block start. target={target_id}, proposed={proposal_ids[0]}")
                rejected += 1
                template_disabled = True
                candidate_ids = []
                cursor = 0
                continue
                
            snap = engine.full_snapshot(entry.cache)
            verify_logits = engine.forward_many(
                lm, mx.array([proposal_ids], dtype=mx.int32), entry.cache
            )
            
            block_matches = True
            for i in range(1, len(proposal_ids)):
                target_tok = engine.argmax_token(verify_logits[:, i - 1, :])
                target_id = int(target_tok.item())
                if target_id != proposal_ids[i]:
                    if trace:
                        print(f"trace: reject inside block at {i}. target={target_id}, proposed={proposal_ids[i]}")
                    block_matches = False
                    break
                    
            if not block_matches:
                engine.restore_full(entry.cache, snap)
                rejected += 1
                template_disabled = True
                candidate_ids = []
                cursor = 0
                continue
                
            out.extend(proposal_ids)
            accepted += len(proposal_ids)
            cursor += len(proposal_ids)
            target_next_logits = verify_logits[:, -1, :]
            
        decode_sec = time.perf_counter() - decode_start
        return engine.DecodeResult(
            engine.decode_text(self.tokenizer, out),
            out,
            time.perf_counter() - total_start,
            prefill_sec,
            decode_sec,
            len(out) / decode_sec if decode_sec > 0 else float("inf"),
            accepted,
            drafted,
            rejected
        )

    def clear_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]

    def evict_sessions_if_needed(self):
        # Already handled by PrefixCacheManager internally, but we could sync self.sessions if needed
        pass
