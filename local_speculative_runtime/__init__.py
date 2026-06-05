from .package_info import __version__, PACKAGE_STATUS
from .memory import (
    MemoryStats,
    get_memory_stats,
    format_memory_stats,
    memory_stats_to_dict,
    maybe_reset_mlx_peak_memory,
    maybe_clear_mlx_cache
)
from .candidates import CandidateRegistry
from .session_cache import (
    SessionCacheAPI,
    SessionCacheRuntime,
    PrefixCacheManager,
    LongInputGuard,
    GuardResult,
    SessionCreateResult,
    SessionGenerateResult,
    CacheStats,
    SessionState
)

__all__ = [
    "__version__",
    "PACKAGE_STATUS",
    "MemoryStats",
    "get_memory_stats",
    "format_memory_stats",
    "memory_stats_to_dict",
    "maybe_reset_mlx_peak_memory",
    "maybe_clear_mlx_cache",
    "CandidateRegistry",
    "SessionCacheAPI",
    "SessionCacheRuntime",
    "PrefixCacheManager",
    "LongInputGuard",
    "GuardResult",
    "SessionCreateResult",
    "SessionGenerateResult",
    "CacheStats",
    "SessionState"
]
