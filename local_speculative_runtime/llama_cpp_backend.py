from typing import Optional, Dict, Any, List
import time

from .backends import BaseInferenceBackend, BackendCapabilities, GenerationResult

class LlamaCppBackend(BaseInferenceBackend):
    backend_name = "llama_cpp"

    def __init__(self, model_path: str, n_ctx: int = 2048, n_gpu_layers: int = -1, n_threads: Optional[int] = None, seed: int = 1337, verbose: bool = False, chat_format: Optional[str] = None, auto_load: bool = True, candidate_json_path: str = "experiments/template_candidates.json", generation_mode: str = "low-level"):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads
        self.seed = seed
        self.verbose = verbose
        self.chat_format = chat_format
        self.candidate_json_path = candidate_json_path
        if generation_mode not in {"low-level", "high-level"}:
            raise ValueError(f"Unknown generation_mode: {generation_mode}")
        self.generation_mode = generation_mode
        
        self.llm = None
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.load_error = None
        self.available = False
        
        try:
            import llama_cpp
            self.available = True
        except ImportError as e:
            self.load_error = f"llama-cpp-python is not installed: {e}"
            return
            
        if auto_load:
            self.load()

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name=self.backend_name,
            supports_prefix_cache=True,
            supports_template_verify=True,
            supports_snapshot_restore=True,
            supports_token_logprobs=False,
            backend_family="llama_cpp",
            prefix_cache_mode="highlevel-concat" if self.generation_mode == "high-level" else "lowlevel-state",
            state_restore_status="unsupported" if self.generation_mode == "high-level" else "supported",
            template_verify_status="unsupported" if self.generation_mode == "high-level" else "supported",
            tested_models=["Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-ggml-model-Q4_K.gguf"],
            limitations=[
                "llama-cpp-python create_completion forces full state reset for recurrent models on branch, so we use low-level eval/sample.",
                "Only greedy sampling is supported in this custom loop for Template Draft / prefix reuse."
            ],
            notes=[
                "GGUF backend via llama-cpp-python.",
                "Prefix cache reuse is supported via low-level eval/sample and prefix save_state; Template Draft rollback uses lightweight kv_cache_seq_rm plus n_tokens rollback.",
                "Template verification is fully supported."
            ]
        )

    def load(self, **kwargs) -> None:
        if not self.available:
            return
            
        import llama_cpp
        try:
            self.llm = llama_cpp.Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                n_threads=self.n_threads,
                seed=self.seed,
                verbose=self.verbose,
                chat_format=self.chat_format
            )
        except Exception as e:
            self.load_error = str(e)

    def tokenize(self, text: str, add_bos: bool = False) -> List[int]:
        if not self.llm:
            return []
        try:
            return self.llm.tokenize(text.encode("utf-8"), add_bos=add_bos, special=True)
        except Exception:
            return []

    def detokenize(self, token_ids: List[int]) -> str:
        if not self.llm:
            return ""
        try:
            return self.llm.detokenize(token_ids).decode("utf-8", errors="replace")
        except Exception:
            return ""


    def _clear_context(self) -> None:
        if self.llm:
            self.llm.reset()
            if hasattr(self.llm._ctx, "kv_cache_clear"):
                self.llm._ctx.kv_cache_clear()

    def _eval_chunked(self, tokens: List[int]) -> None:
        if not tokens: return
        n_batch = getattr(self.llm, "n_batch", 512)
        for i in range(0, len(tokens), n_batch):
            self.llm.eval(tokens[i : i + n_batch])

    def _greedy_sample(self) -> int:
        import numpy as np
        import llama_cpp
        last_idx = self.llm._batch.n_tokens() - 1
        logits_ptr = llama_cpp.llama_get_logits_ith(self.llm.ctx, last_idx)
        logits = np.ctypeslib.as_array(logits_ptr, shape=(self.llm.n_vocab(),))
        
        return int(np.argmax(logits))

    def _load_session_state(self, state: Any, full_prompt: str, prompt_or_suffix: str) -> List[int]:
        if state is not None:
            self.llm.load_state(state)
            return self.tokenize(prompt_or_suffix, add_bos=False)
        else:
            self._clear_context()
            return self.tokenize(full_prompt, add_bos=True)

    def _generate_greedy_tokens(self, out_tokens: List[int], max_tokens: int, stop_id: int) -> None:
        import llama_cpp
        has_is_eog = hasattr(llama_cpp, "llama_vocab_is_eog")
        vocab = self.llm._model.vocab if has_is_eog else None
        
        while len(out_tokens) < max_tokens:
            tid = self._greedy_sample()
            
            is_stop = False
            if tid == stop_id:
                is_stop = True
            elif has_is_eog and llama_cpp.llama_vocab_is_eog(vocab, tid):
                is_stop = True
                
            if is_stop:
                break
            out_tokens.append(tid)
            self.llm.eval([tid])

    def _verify_candidate_block(self, block: List[int], trace: bool, metrics: Dict[str, float]) -> List[int]:
        import numpy as np
        t0 = time.perf_counter()
        target_id = self._greedy_sample()
        metrics["sample_count"] = metrics.get("sample_count", 0) + 1
        if target_id != block[0]:
            if trace: print(f"trace: mismatch at block[0]: target={target_id} block[0]={block[0]}")
            metrics["C_verify_sec"] += time.perf_counter() - t0
            return []
            
        t1 = time.perf_counter()
        rollback_n_tokens = self.llm.n_tokens
        metrics["C_save_state_sec"] += time.perf_counter() - t1
        
        t2 = time.perf_counter()
        self.llm.eval(block)
        metrics["C_eval_block_sec"] += time.perf_counter() - t2
        
        for i in range(len(block) - 1):
            pred = int(np.argmax(self.llm.scores[self.llm.n_tokens - len(block) + i, :]))
            if pred != block[i+1]:
                if trace: print(f"trace: mismatch inside block at i={i}: pred={pred} block[{i+1}]={block[i+1]}")
                t3 = time.perf_counter()
                self.llm.n_tokens = rollback_n_tokens
                if hasattr(self.llm._ctx, "kv_cache_seq_rm"):
                    self.llm._ctx.kv_cache_seq_rm(-1, rollback_n_tokens, -1)
                metrics["C_load_state_sec"] += time.perf_counter() - t3
                accepted_tokens = block[:i+1]
                t4 = time.perf_counter()
                self.llm.eval(accepted_tokens)
                metrics["C_eval_block_sec"] += time.perf_counter() - t4
                metrics["C_verify_sec"] += time.perf_counter() - t0
                return accepted_tokens
                
        metrics["C_verify_sec"] += time.perf_counter() - t0
        return block


    def create_session(self, session_id: str, prefix_text: str) -> Dict[str, Any]:
        if self.load_error:
            return {
                "ok": False,
                "session_id": session_id,
                "prefix_tokens": 0,
                "prefix_prefill_sec": 0.0,
                "cache_key": None,
                "guard_allowed": False,
                "guard_reason": self.load_error,
                "evicted_keys": []
            }
            
        start_time = time.perf_counter()
        
        if self.llm:
            model_type = str(self.llm.metadata.get("tokenizer.ggml.model", "")).lower()
            if "gemma" in model_type:
                if "System:" not in prefix_text and "<start_of_turn>" not in prefix_text:
                    prefix_text = f"System: {prefix_text.strip()}\n\n"
        
        tokens_prefix = self.tokenize(prefix_text, add_bos=True)
        prompt_tokens = len(tokens_prefix)
        
        if self.llm:
            self._clear_context()
            self._eval_chunked(tokens_prefix)
            state = self.llm.save_state()
        else:
            state = None
            
        prefix_prefill_sec = time.perf_counter() - start_time
        
        self.sessions[session_id] = {
            "prefix_text": prefix_text,
            "state": state,
            "created_at": time.time(),
            "last_used_at": time.time(),
            "turn_count": 0
        }
        
        return {
            "ok": True,
            "session_id": session_id,
            "prefix_tokens": prompt_tokens,
            "prefix_prefill_sec": prefix_prefill_sec,
            "cache_key": session_id,
            "guard_allowed": True,
            "guard_reason": "",
            "evicted_keys": [],
            "metadata": {
                "backend_capabilities": self.capabilities().__dict__,
                "prefix_cache_mode": "lowlevel-state"
            }
        }

    def generate(self, session_id: Optional[str], prompt_or_suffix: str, max_tokens: int = 16, **kwargs) -> GenerationResult:
        if self.load_error:
            return GenerationResult(False, "", [], 0.0, None, None, self.load_error, self.backend_name, {})
        if not self.llm:
            return GenerationResult(False, "", [], 0.0, 0, 0, "Model not loaded", self.backend_name, {})
            
        if self.llm:
            model_type = str(self.llm.metadata.get("tokenizer.ggml.model", "")).lower()
            if "gemma" in model_type:
                if "User:" not in prompt_or_suffix and "Assistant:" not in prompt_or_suffix and "<start_of_turn>" not in prompt_or_suffix:
                    prompt_or_suffix = f"User: {prompt_or_suffix.strip()}\n\nAssistant:"
            
        temperature = kwargs.get("temperature", 0.0)
        draft_block_size = kwargs.get("draft_block_size", 8)
        template_min_tokens = kwargs.get("template_min_tokens", 1)
        trace = kwargs.get("trace", False)
        
        full_prompt = prompt_or_suffix
        session_turn_count = 0
        state = None
        
        if session_id:
            if session_id not in self.sessions:
                return GenerationResult(False, "", [], 0.0, None, None, f"Session {session_id} not found", self.backend_name, {})
            session_state = self.sessions[session_id]
            session_state["turn_count"] += 1
            session_state["last_used_at"] = time.time()
            session_turn_count = session_state["turn_count"]
            full_prompt = session_state["prefix_text"] + prompt_or_suffix
            state = session_state["state"]
            
        if self.generation_mode == "high-level":
            start_time = time.perf_counter()
            try:
                # high-level llm API tokenizes and evaluates all tokens, so we use full_prompt.
                # temperature 0.0 forces greedy decoding.
                res = self.llm(full_prompt, max_tokens=max_tokens, temperature=temperature or 0.0, echo=False)
                
                text = res["choices"][0]["text"]
                elapsed_sec = time.perf_counter() - start_time
                
                # Approximate token usage since the high-level API might not give exact IDs if not requested
                prompt_tokens = res["usage"]["prompt_tokens"]
                completion_tokens = res["usage"]["completion_tokens"]
                # We return empty token_ids list to signify high-level output.
                out_tokens = []
                
                return GenerationResult(
                    ok=True,
                    text=text,
                    token_ids=out_tokens,
                    elapsed_sec=elapsed_sec,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    error=None,
                    backend=self.backend_name,
                    metadata={
                        "finish_reason": res["choices"][0].get("finish_reason", "stop"),
                        "temperature": temperature,
                        "n_ctx": self.n_ctx,
                        "n_gpu_layers": self.n_gpu_layers,
                        "session_turn_count": session_turn_count,
                        "prefix_cache_mode": "highlevel-concat",
                        "template_verify_enabled": False,
                        "snapshot_restore_enabled": False,
                        "suffix_prefill_sec": 0.0,
                        "decode_sec": elapsed_sec,
                        "accepted": 0,
                        "drafted": 0,
                        "rejected": 0,
                        "sample_count": 0,
                        "candidate_name": None,
                        "fallback_used": False,
                        "generation_mode": "high-level"
                    }
                )
            except Exception as e:
                elapsed_sec = time.perf_counter() - start_time
                return GenerationResult(False, "", [], elapsed_sec, None, None, str(e), self.backend_name, {})
                
        start_time = time.perf_counter()
        
        try:
            stop_id = self.llm.token_eos()
            suffix_tokens = self._load_session_state(state, full_prompt, prompt_or_suffix)
            self._eval_chunked(suffix_tokens)
                
            suffix_prefill_sec = time.perf_counter() - start_time
            decode_start = time.perf_counter()
            
            out_tokens = []
            accepted = 0
            drafted = 0
            rejected = 0
            sample_count = 0
            fallback_used = False
            candidate_name = None
            
            is_gemma = False
            if self.llm:
                model_type = str(self.llm.metadata.get("tokenizer.ggml.model", "")).lower()
                is_gemma = "gemma" in model_type
                
            inside_channel = False
            # Gemma4 GGUF may emit internal channel markers such as
            # <|channel>thought<channel|> before the final answer.
            # Keep these tokens in the model context via eval(), but do not
            # expose them in returned text. Token IDs verified from Gemma4 GGUF.
            CHANNEL_START_ID = 100
            CHANNEL_END_ID = 101
            
            import llama_cpp
            has_is_eog = hasattr(llama_cpp, "llama_vocab_is_eog")
            vocab = self.llm._model.vocab if has_is_eog else None
            
            def is_stop_token(tid: int) -> bool:
                if tid == stop_id:
                    return True
                if has_is_eog and llama_cpp.llama_vocab_is_eog(vocab, tid):
                    return True
                return False
            
            metrics = {
                "C_verify_sec": 0.0,
                "C_save_state_sec": 0.0,
                "C_load_state_sec": 0.0,
                "C_eval_block_sec": 0.0,
                "sample_count": 0,
            }
            
            first_id = self._greedy_sample()
            metrics["sample_count"] += 1
            
            if not is_stop_token(first_id):
                if is_gemma:
                    if first_id == CHANNEL_START_ID:
                        inside_channel = True
                    elif first_id == CHANNEL_END_ID:
                        inside_channel = False
                    elif not inside_channel:
                        out_tokens.append(first_id)
                else:
                    out_tokens.append(first_id)
                    
                self.llm.eval([first_id])
                
                candidate_ids = []
                if template_min_tokens > 0 and draft_block_size > 0:
                    try:
                        from experiments.template_draft_runtime import CandidateRegistry
                        from experiments import template_draft_engine as engine
                        template_json_path = kwargs.get("candidate_json_path", self.candidate_json_path)
                        registry = CandidateRegistry(json_path=template_json_path)
                        
                        class DummyTokenizer:
                            def encode(self, text): return self.backend.tokenize(text)
                            def decode(self, ids): return self.backend.detokenize(ids)
                        dt = DummyTokenizer()
                        dt.backend = self
                        
                        candidate = registry.select_candidate(prompt_or_suffix, dt, template_min_tokens, trace)
                        if candidate:
                            candidate_name = candidate.name
                            c_ids = engine.encode_candidate(dt, candidate)
                            if c_ids and c_ids[0] == first_id:
                                candidate_ids = c_ids[1:]
                            elif not c_ids:
                                candidate_ids = []
                            else:
                                candidate_ids = c_ids
                    except ImportError:
                        pass
                
                cursor = 0
                template_disabled = False
                
                while len(out_tokens) < max_tokens:
                    remaining = max_tokens - len(out_tokens)
                    block = []
                    
                    if not template_disabled and cursor < len(candidate_ids):
                        block = candidate_ids[cursor : cursor + min(draft_block_size, remaining)]
                        
                    if not block:
                        while len(out_tokens) < max_tokens:
                            tid = self._greedy_sample()
                            metrics["sample_count"] += 1
                            if is_stop_token(tid):
                                break
                                
                            if is_gemma:
                                if tid == CHANNEL_START_ID:
                                    inside_channel = True
                                elif tid == CHANNEL_END_ID:
                                    inside_channel = False
                                elif not inside_channel:
                                    out_tokens.append(tid)
                            else:
                                out_tokens.append(tid)
                                
                            self.llm.eval([tid])
                        break
                        
                    drafted += len(block)
                    accepted_tokens = self._verify_candidate_block(block, trace, metrics)
                    
                    if len(accepted_tokens) != len(block):
                        rejected += 1
                        fallback_used = True
                        template_disabled = True
                        candidate_ids = []
                        if accepted_tokens:
                            if is_gemma:
                                for tid in accepted_tokens:
                                    if tid == CHANNEL_START_ID:
                                        inside_channel = True
                                    elif tid == CHANNEL_END_ID:
                                        inside_channel = False
                                    elif not inside_channel:
                                        out_tokens.append(tid)
                            else:
                                out_tokens.extend(accepted_tokens)
                            accepted += len(accepted_tokens)
                        continue
                        
                    if is_gemma:
                        for tid in block:
                            if tid == CHANNEL_START_ID:
                                inside_channel = True
                            elif tid == CHANNEL_END_ID:
                                inside_channel = False
                            elif not inside_channel:
                                out_tokens.append(tid)
                    else:
                        out_tokens.extend(block)
                    accepted += len(block)
                    cursor += len(block)
                    
            elapsed_sec = time.perf_counter() - start_time
            decode_sec = time.perf_counter() - decode_start
            text = self.detokenize(out_tokens)
            
            return GenerationResult(
                ok=True,
                text=text,
                token_ids=out_tokens,
                elapsed_sec=elapsed_sec,
                prompt_tokens=len(suffix_tokens),
                completion_tokens=len(out_tokens),
                error=None,
                backend=self.backend_name,
                metadata={
                    "finish_reason": "stop" if len(out_tokens) < max_tokens else "length",
                    "temperature": temperature,
                    "n_ctx": self.n_ctx,
                    "n_gpu_layers": self.n_gpu_layers,
                    "session_turn_count": session_turn_count,
                    "prefix_cache_mode": "lowlevel-state",
                    "template_verify_enabled": template_min_tokens > 0 and draft_block_size > 0,
                    "snapshot_restore_enabled": True,
                    "suffix_prefill_sec": suffix_prefill_sec,
                    "decode_sec": decode_sec,
                    "accepted": accepted,
                    "drafted": drafted,
                    "rejected": rejected,
                    "sample_count": metrics["sample_count"],
                    "candidate_name": candidate_name,
                    "fallback_used": fallback_used,
                    **metrics
                }
            )
            
        except Exception as e:
            elapsed_sec = time.perf_counter() - start_time
            return GenerationResult(False, "", [], elapsed_sec, None, None, str(e), self.backend_name, {})

    def clear_session(self, session_id: str, drop_cache: bool = False) -> Dict[str, Any]:
        if session_id in self.sessions:
            del self.sessions[session_id]
            return {
                "ok": True, 
                "session_id": session_id, 
                "dropped_cache": drop_cache, 
                "cache_key": None, 
                "error": None,
                "metadata": {
                    "note": "drop_cache is session bookkeeping only for llama_cpp backend"
                }
            }
        return {"ok": False, "session_id": session_id, "dropped_cache": False, "cache_key": None, "error": f"Session {session_id} not found"}

    def stats(self) -> Dict[str, Any]:
        return {
            "backend": self.backend_name,
            "sessions": len(self.sessions),
            "loaded": self.llm is not None,
            "available": self.available,
            "load_error": self.load_error,
            "model_path": self.model_path,
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
            "n_threads": self.n_threads,
            "capabilities": self.capabilities().__dict__,
            "session_ids": list(self.sessions.keys())
        }
