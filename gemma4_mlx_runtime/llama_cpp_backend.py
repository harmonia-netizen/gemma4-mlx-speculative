from typing import Optional, Dict, Any, List
import time

from .backends import BaseInferenceBackend, BackendCapabilities, GenerationResult

class LlamaCppBackend(BaseInferenceBackend):
    backend_name = "llama_cpp"

    def __init__(self, model_path: str, n_ctx: int = 2048, n_gpu_layers: int = -1, n_threads: Optional[int] = None, seed: int = 1337, verbose: bool = False, chat_format: Optional[str] = None, auto_load: bool = True):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads
        self.seed = seed
        self.verbose = verbose
        self.chat_format = chat_format
        
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
            supports_prefix_cache=False,
            supports_template_verify=False,
            supports_snapshot_restore=False,
            supports_token_logprobs=False,
            backend_family="llama_cpp",
            prefix_cache_mode="llama-prefix-match",
            state_restore_status="failed_for_tested_model",
            template_verify_status="disabled_until_state_restore_passes",
            tested_models=["Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-ggml-model-Q4_K.gguf"],
            limitations=[
                "llama-cpp-python forces full state reset for recurrent/hybrid models on branch",
                "Tested model fails state restore due to hybrid model reset logic"
            ],
            notes=[
                "Initial GGUF backend via llama-cpp-python",
                "Prefix cache reuse and exact KV snapshot restore are unsupported because llama-cpp-python forces full state reset for recurrent/hybrid models on branch.",
                "Template verification is not implemented due to lack of safe rollback support."
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
            return self.llm.tokenize(text.encode("utf-8"), add_bos=False)
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
            
        prompt_tokens = len(self.tokenize(prefix_text))
        
        # Simple storage of prefix text for now.
        # Future enhancement: populate llama.cpp slot or KV cache.
        self.sessions[session_id] = {
            "prefix_text": prefix_text,
            "created_at": time.time(),
            "last_used_at": time.time(),
            "turn_count": 0
        }
        
        return {
            "ok": True,
            "session_id": session_id,
            "prefix_tokens": prompt_tokens,
            "prefix_prefill_sec": 0.0,
            "cache_key": session_id,
            "guard_allowed": True,
            "guard_reason": "",
            "evicted_keys": [],
            "metadata": {
                "backend_capabilities": self.capabilities().__dict__,
                "prefix_cache_mode": "text-concat"
            }
        }

    def generate(self, session_id: Optional[str], prompt_or_suffix: str, max_tokens: int = 16, **kwargs) -> GenerationResult:
        if self.load_error:
            return GenerationResult(False, "", [], 0.0, None, None, self.load_error, self.backend_name, {})
            
        if not self.llm:
            return GenerationResult(False, "", [], 0.0, 0, 0, "Model not loaded", self.backend_name, {})
            
        temperature = kwargs.get("temperature", 0.0)
        
        full_prompt = prompt_or_suffix
        session_turn_count = 0
        if session_id:
            if session_id not in self.sessions:
                return GenerationResult(False, "", [], 0.0, None, None, f"Session {session_id} not found", self.backend_name, {})
                
            session_state = self.sessions[session_id]
            session_state["turn_count"] += 1
            session_state["last_used_at"] = time.time()
            session_turn_count = session_state["turn_count"]
            full_prompt = session_state["prefix_text"] + "\n" + prompt_or_suffix
            
        start_time = time.perf_counter()
        
        try:
            response = self.llm.create_completion(
                prompt=full_prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            elapsed_sec = time.perf_counter() - start_time
            text = response["choices"][0]["text"]
            token_ids = self.tokenize(text)
            
            usage = response.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", len(token_ids))
            
            return GenerationResult(
                ok=True,
                text=text,
                token_ids=token_ids,
                elapsed_sec=elapsed_sec,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                error=None,
                backend=self.backend_name,
                metadata={
                    "finish_reason": response["choices"][0].get("finish_reason"),
                    "temperature": temperature,
                    "n_ctx": self.n_ctx,
                    "n_gpu_layers": self.n_gpu_layers,
                    "session_turn_count": session_turn_count,
                    "prefix_cache_mode": "text-concat",
                    "template_verify_enabled": False,
                    "snapshot_restore_enabled": False
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
