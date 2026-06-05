from dataclasses import dataclass
from typing import Optional, Dict, Any, Protocol, List, runtime_checkable

@dataclass
class BackendCapabilities:
    name: str
    supports_prefix_cache: bool
    supports_template_verify: bool
    supports_snapshot_restore: bool
    supports_token_logprobs: bool
    notes: List[str]
    backend_family: Optional[str] = None
    prefix_cache_mode: Optional[str] = None
    state_restore_status: Optional[str] = None
    template_verify_status: Optional[str] = None
    tested_models: Optional[List[str]] = None
    limitations: Optional[List[str]] = None

@dataclass
class GenerationResult:
    ok: bool
    text: str
    token_ids: List[int]
    elapsed_sec: float
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    error: Optional[str]
    backend: str
    metadata: Dict[str, Any]

@runtime_checkable
class BaseInferenceBackend(Protocol):
    backend_name: str

    def capabilities(self) -> BackendCapabilities:
        ...

    def load(self, **kwargs) -> None:
        ...

    def tokenize(self, text: str) -> List[int]:
        ...

    def detokenize(self, token_ids: List[int]) -> str:
        ...

    def create_session(self, session_id: str, prefix_text: str) -> Dict[str, Any]:
        ...

    def generate(self, session_id: Optional[str], prompt_or_suffix: str, max_tokens: int = 16, **kwargs) -> GenerationResult:
        ...

    def clear_session(self, session_id: str, drop_cache: bool = False) -> Dict[str, Any]:
        ...

    def stats(self) -> Dict[str, Any]:
        ...
