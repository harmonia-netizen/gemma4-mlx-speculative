"""
Backward compatibility shim for experiments.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gemma4_mlx_runtime import (
    SessionCacheAPI,
    SessionCacheRuntime,
    PrefixCacheManager,
    LongInputGuard,
    CandidateRegistry,
    CacheStats,
    SessionCreateResult,
    SessionGenerateResult,
    SessionState,
    get_memory_stats,
    format_memory_stats,
    MemoryStats,
    GuardResult
)

__all__ = [
    "SessionCacheAPI",
    "SessionCacheRuntime",
    "PrefixCacheManager",
    "LongInputGuard",
    "CandidateRegistry",
    "CacheStats",
    "SessionCreateResult",
    "SessionGenerateResult",
    "SessionState",
    "get_memory_stats",
    "format_memory_stats",
    "MemoryStats",
    "GuardResult"
]
