from typing import Optional, Dict, Any, List
import time

from .backends import BaseInferenceBackend, BackendCapabilities, GenerationResult

class LlamaCppBackend(BaseInferenceBackend):
    backend_name = "llama_cpp"

    def __init__(self, model_path: str, n_ctx: int = 2048, n_gpu_layers: int = -1, n_threads: Optional[int] = None, seed: int = 1337, verbose: bool = False, chat_format: Optional[str] = None, auto_load: bool = True, candidate_json_path: str = "experiments/template_candidates.json"):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads
        self.seed = seed
        self.verbose = verbose
        self.chat_format = chat_format
        self.candidate_json_path = candidate_json_path
        
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
            prefix_cache_mode="lowlevel-state",
            state_restore_status="supported",
            template_verify_status="supported",
            tested_models=["Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-ggml-model-Q4_K.gguf"],
            limitations=[
                "llama-cpp-python create_completion forces full state reset for recurrent models on branch, so we use low-level eval/sample.",
                "Only greedy sampling is supported in this custom loop for Template Draft / prefix reuse."
            ],
            notes=[
                "GGUF backend via llama-cpp-python.",
                "Prefix cache reuse and exact KV snapshot restore are supported via low-level eval/sample and save_state/load_state.",
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

    def tokenize(self, text: str) -> List[int]:
        if not self.llm:
            return []
        try:
            return self.llm.tokenize(text.encode("utf-8"), add_bos=False, special=True)
        except Exception:
            return []

    def detokenize(self, token_ids: List[int]) -> str:
        if not self.llm:
            return ""
        try:
            return self.llm.detokenize(token_ids).decode("utf-8", errors="replace")
        except Exception:
            return ""

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
        
        # Tokenize prefix
        tokens_prefix = self.tokenize(prefix_text)
        prompt_tokens = len(tokens_prefix)
        
        if self.llm:
            self.llm.reset()
            if hasattr(self.llm._ctx, "kv_cache_clear"):
                self.llm._ctx.kv_cache_clear()
            
            # Chunk eval to avoid llama_decode returning 1
            n_batch = getattr(self.llm, "n_batch", 512)
            for i in range(0, prompt_tokens, n_batch):
                self.llm.eval(tokens_prefix[i : i + n_batch])
                
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
            
        start_time = time.perf_counter()
        
        try:
            import numpy as np
            
            def greedy_sample():
                return int(np.argmax(self.llm.scores[self.llm.n_tokens - 1, :]))
                
            stop_id = self.llm.token_eos()
            
            # Load state and eval suffix
            suffix_tokens = self.tokenize(prompt_or_suffix)
            if state is not None:
                self.llm.load_state(state)
            else:
                self.llm.reset()
                if hasattr(self.llm._ctx, "kv_cache_clear"):
                    self.llm._ctx.kv_cache_clear()
                suffix_tokens = self.tokenize(full_prompt)
                
            n_batch = getattr(self.llm, "n_batch", 512)
            for i in range(0, len(suffix_tokens), n_batch):
                self.llm.eval(suffix_tokens[i : i + n_batch])
                
            suffix_prefill_sec = time.perf_counter() - start_time
            decode_start = time.perf_counter()
            
            out_tokens = []
            accepted = 0
            drafted = 0
            rejected = 0
            fallback_used = False
            candidate_name = None
            
            first_id = greedy_sample()
            
            if first_id != stop_id:
                out_tokens.append(first_id)
                self.llm.eval([first_id])
                
                # Setup Template Draft
                candidate_ids = []
                if template_min_tokens > 0 and draft_block_size > 0:
                    try:
                        from experiments.template_draft_runtime import CandidateRegistry
                        from experiments import template_draft_engine as engine
                        registry = CandidateRegistry(json_path=self.candidate_json_path)
                        
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
                        # Greedy fallback
                        tid = greedy_sample()
                        if tid == stop_id:
                            break
                        out_tokens.append(tid)
                        self.llm.eval([tid])
                        continue
                        
                    drafted += len(block)
                    target_id = greedy_sample()
                    
                    if target_id != block[0]:
                        if trace: print(f"trace: mismatch at block[0]: target={target_id} block[0]={block[0]}")
                        rejected += 1
                        fallback_used = True
                        template_disabled = True
                        candidate_ids = []
                        continue
                        
                    # Save state for rollback
                    block_state = self.llm.save_state()
                    self.llm.eval(block)
                    
                    match = True
                    for i in range(len(block) - 1):
                        pred = int(np.argmax(self.llm.scores[self.llm.n_tokens - len(block) + i, :]))
                        if pred != block[i+1]:
                            if trace: print(f"trace: mismatch inside block at i={i}: pred={pred} block[{i+1}]={block[i+1]}")
                            match = False
                            self.llm.load_state(block_state)
                            accepted_tokens = block[:i+1]
                            self.llm.eval(accepted_tokens)
                            out_tokens.extend(accepted_tokens)
                            accepted += len(accepted_tokens)
                            rejected += 1
                            fallback_used = True
                            template_disabled = True
                            candidate_ids = []
                            break
                            
                    if not match:
                        continue
                        
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
                    "finish_reason": "stop" if out_tokens and out_tokens[-1] == stop_id else "length",
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
                    "candidate_name": candidate_name,
                    "fallback_used": fallback_used
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
