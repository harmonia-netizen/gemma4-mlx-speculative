from dataclasses import dataclass
from typing import Optional

@dataclass
class SessionCreateResult:
    ok: bool
    session_id: str
    prefix_tokens: int
    prefix_prefill_sec: float
    cache_key: Optional[str]
    guard_allowed: bool
    guard_reason: str
    evicted_keys: list[str]

@dataclass
class SessionGenerateResult:
    ok: bool
    session_id: str
    text: str
    token_ids: list[int]
    suffix_tokens: int
    suffix_prefill_sec: float
    decode_sec: float
    elapsed_sec: float
    accepted: int
    drafted: int
    rejected: int
    candidate_name: Optional[str]
    fallback_used: bool
    error: Optional[str]

@dataclass
class CacheStats:
    entries: int
    current_total_tokens: int
    max_entries: int
    max_total_tokens: int
    keys: list[str]

@dataclass
class SessionState:
    session_id: str
    prefix_key: str
    turn_count: int
    created_at: float
    last_used_at: float
