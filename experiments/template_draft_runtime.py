import time
import hashlib
import json
import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import mlx.core as mx
import template_draft_engine as engine


@dataclass
class PrefixCacheEntry:
    text_hash: str
    token_ids: List[int]
    snapshot: Any
    cache: Any
    prefill_sec: float
    created_at: float
    last_used_at: float
    hit_count: int
    evicted_keys: list[str] | None = None

class PrefixCacheManager:
    def __init__(self, max_entries: int = 2, max_total_tokens: int = 120000):
        self.entries: Dict[str, PrefixCacheEntry] = {}
        self.max_entries = max_entries
        self.max_total_tokens = max_total_tokens
        self.current_total_tokens = 0

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get_or_create(self, prefix_text: str, prefix_ids: List[int], target_model, max_kv_size) -> PrefixCacheEntry:
        if len(prefix_ids) > self.max_total_tokens:
            raise ValueError(f"prefix length {len(prefix_ids)} exceeds max_total_tokens {self.max_total_tokens}")

        h = self._hash(prefix_text)
        if h in self.entries:
            entry = self.entries[h]
            entry.last_used_at = time.time()
            entry.hit_count += 1
            return entry
            
        evicted = self.evict_if_needed(len(prefix_ids))
        
        lm = engine.get_lm(target_model)
        start = time.perf_counter()
        prompt_cache = engine.make_cache(lm, max_kv_size)
        
        if len(prefix_ids) > 0:
            input_arr = mx.array([prefix_ids])
            emb = target_model.get_input_embeddings(input_arr, None, mask=None)
            inputs_embeds = emb.inputs_embeds
            extra = {k: v for k, v in emb.to_dict().items() if k != "inputs_embeds" and v is not None}
            n = input_arr.shape[1]
            step_size = 512
            for i in range(0, n, step_size):
                chunk_len = min(step_size, n - i)
                lm(
                    input_arr[:, i:i+chunk_len],
                    inputs_embeds=inputs_embeds[:, i:i+chunk_len] if inputs_embeds is not None else None,
                    cache=prompt_cache,
                    n_to_process=chunk_len,
                    **extra,
                )
                engine.eval_cache(prompt_cache)
        prefill_sec = time.perf_counter() - start
        snapshot = engine.full_snapshot(prompt_cache)
        
        now = time.time()
        entry = PrefixCacheEntry(h, prefix_ids, snapshot, prompt_cache, prefill_sec, now, now, 0)
        entry.evicted_keys = evicted
        self.entries[h] = entry
        self.current_total_tokens += len(prefix_ids)
        return entry

    def evict_if_needed(self, additional_tokens: int = 0) -> List[str]:
        evicted = []
        while self.entries and (len(self.entries) >= self.max_entries or self.current_total_tokens + additional_tokens > self.max_total_tokens):
            oldest_key = min(self.entries.keys(), key=lambda k: self.entries[k].last_used_at)
            entry = self.entries.pop(oldest_key)
            self.current_total_tokens -= len(entry.token_ids)
            evicted.append(oldest_key)
        return evicted


@dataclass
class GuardResult:
    allowed: bool
    prompt_tokens: int
    safe_token_limit: int
    reason: str

class LongInputGuard:
    def __init__(self, safe_token_limit: int = 120000, require_chunked_prefill: bool = True):
        self.safe_token_limit = safe_token_limit
        self.require_chunked_prefill = require_chunked_prefill

    def validate(self, prompt_tokens: int, use_chunked_prefill: bool = True) -> GuardResult:
        if prompt_tokens > self.safe_token_limit:
            return GuardResult(False, prompt_tokens, self.safe_token_limit, f"Prompt tokens {prompt_tokens} exceeds safe limit {self.safe_token_limit}")
        if self.require_chunked_prefill and not use_chunked_prefill:
            return GuardResult(False, prompt_tokens, self.safe_token_limit, "Chunked prefill is required but not enabled")
        return GuardResult(True, prompt_tokens, self.safe_token_limit, "")


class CandidateRegistry:
    def __init__(self, json_path: str = "experiments/template_candidates.json"):
        self.json_path = json_path
        self.entries = []
        self._load()

    def _load(self):
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"Candidate JSON not found: {self.json_path}")
        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        self.entries = []
        for item in data:
            c = engine.Candidate(
                name=item["name"],
                text=item["text"],
                confidence=item["confidence"],
                min_tokens=item["min_tokens"],
                tags=tuple(item["tags"])
            )
            self.entries.append({
                "candidate": c,
                "match_keywords": item["match_keywords"]
            })

    def get_candidates(self, user_prompt: str) -> List[engine.Candidate]:
        p = user_prompt.lower()
        matched = []
        for entry in self.entries:
            keywords = [kw.lower() for kw in entry["match_keywords"]]
            if all(kw in p for kw in keywords):
                matched.append(entry["candidate"])
        return matched
        
    def select_candidate(self, user_prompt: str, tokenizer, min_tokens: int = 1, trace: bool = False) -> Optional[engine.Candidate]:
        candidates = self.get_candidates(user_prompt)
        return engine.select_candidate(user_prompt, candidates, tokenizer, min_tokens, trace)


class TemplateDraftRuntime:
    def __init__(self, target_model, tokenizer, processor, max_kv_size=None):
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_kv_size = max_kv_size
        self.prefix_manager = PrefixCacheManager()
        self.candidate_registry = CandidateRegistry()
        self.stop_ids = engine.build_stop_ids(tokenizer)

    def _chunked_prefill(self, prefix_ids, suffix_ids, lm, prompt_cache):
        input_ids = prefix_ids + suffix_ids
        input_arr = mx.array([input_ids])
        emb = self.target_model.get_input_embeddings(input_arr, None, mask=None)
        inputs_embeds = emb.inputs_embeds
        extra = {k: v for k, v in emb.to_dict().items() if k != "inputs_embeds" and v is not None}
        
        step_size = 512
        p = len(prefix_ids)
        if input_arr.shape[1] > 1:
            if p > 0:
                for i in range(0, p, step_size):
                    chunk_len = min(step_size, p - i)
                    lm(
                        input_arr[:, i:i+chunk_len],
                        inputs_embeds=inputs_embeds[:, i:i+chunk_len] if inputs_embeds is not None else None,
                        cache=prompt_cache,
                        n_to_process=chunk_len,
                        **extra,
                    )
                    engine.eval_cache(prompt_cache)
                
            rem = input_arr.shape[1] - 1
            if rem > p:
                for i in range(p, rem, step_size):
                    chunk_len = min(step_size, rem - i)
                    lm(
                        input_arr[:, i:i+chunk_len],
                        inputs_embeds=inputs_embeds[:, i:i+chunk_len] if inputs_embeds is not None else None,
                        cache=prompt_cache,
                        n_to_process=chunk_len,
                        **extra,
                    )
                    engine.eval_cache(prompt_cache)

        cur = input_arr[:, -1:]
        return cur
        
    def baseline_chunked_greedy(self, prefix_ids: List[int], suffix_ids: List[int], max_tokens: int) -> engine.DecodeResult:
        lm = engine.get_lm(self.target_model)
        total_start = time.perf_counter()
        
        prompt_cache = engine.make_cache(lm, self.max_kv_size)
        cur = self._chunked_prefill(prefix_ids, suffix_ids, lm, prompt_cache)
        prefill_sec = time.perf_counter() - total_start
        
        decode_start = time.perf_counter()
        out = []
        
        first_logits = engine.forward_one(lm, cur, prompt_cache)
        tok = engine.argmax_token(first_logits)
        first_id = int(tok.item())
        
        if first_id not in self.stop_ids:
            out.append(first_id)
            next_logits = engine.forward_one(lm, tok[:, None], prompt_cache)
            while len(out) < max_tokens:
                tok = engine.argmax_token(next_logits)
                tid = int(tok.item())
                if tid in self.stop_ids:
                    break
                out.append(tid)
                next_logits = engine.forward_one(lm, tok[:, None], prompt_cache)
                
        decode_sec = time.perf_counter() - decode_start
        
        return engine.DecodeResult(
            engine.decode_text(self.tokenizer, out),
            out,
            time.perf_counter() - total_start,
            prefill_sec,
            decode_sec,
            len(out) / decode_sec if decode_sec > 0 else float("inf")
        )

    def prefix_reuse_greedy(self, prefix_text: str, prefix_ids: List[int], suffix_ids: List[int], max_tokens: int) -> engine.DecodeResult:
        entry = self.prefix_manager.get_or_create(prefix_text, prefix_ids, self.target_model, self.max_kv_size)
        
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
        
        if first_id not in self.stop_ids:
            out.append(first_id)
            next_logits = engine.forward_one(lm, tok[:, None], entry.cache)
            while len(out) < max_tokens:
                tok = engine.argmax_token(next_logits)
                tid = int(tok.item())
                if tid in self.stop_ids:
                    break
                out.append(tid)
                next_logits = engine.forward_one(lm, tok[:, None], entry.cache)
                
        decode_sec = time.perf_counter() - decode_start
        
        return engine.DecodeResult(
            engine.decode_text(self.tokenizer, out),
            out,
            time.perf_counter() - total_start,
            prefill_sec,
            decode_sec,
            len(out) / decode_sec if decode_sec > 0 else float("inf")
        )

    def prefix_reuse_template_draft(
        self, 
        prefix_text: str, 
        prefix_ids: List[int], 
        suffix_ids: List[int], 
        user_prompt: str,
        max_tokens: int,
        block_size: int,
        template_min_tokens: int,
        trace: bool = False
    ) -> engine.DecodeResult:
        entry = self.prefix_manager.get_or_create(prefix_text, prefix_ids, self.target_model, self.max_kv_size)
        
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
        
        candidate = self.candidate_registry.select_candidate(user_prompt, self.tokenizer, template_min_tokens, trace)
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
                proposal_ids = candidate_ids[cursor : cursor + min(block_size, remaining)]
                
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
