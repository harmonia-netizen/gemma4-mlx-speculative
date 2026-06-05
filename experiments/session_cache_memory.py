import os
from dataclasses import dataclass
from typing import Optional

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import tracemalloc
    HAS_TRACEMALLOC = True
except ImportError:
    HAS_TRACEMALLOC = False

@dataclass
class MemoryStats:
    rss_mb: Optional[float]
    python_tracemalloc_current_mb: Optional[float]
    python_tracemalloc_peak_mb: Optional[float]
    note: str

def get_memory_stats() -> MemoryStats:
    rss_mb = None
    note = "Metal/GPU actual memory not included. "
    
    if HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        rss_bytes = process.memory_info().rss
        rss_mb = rss_bytes / (1024 * 1024)
    else:
        try:
            import resource
            rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # getrusage returns KB on Linux but bytes on Mac. We assume Mac (bytes) based on context.
            rss_mb = rss_kb / (1024 * 1024)
            note += "(using getrusage) "
        except ImportError:
            note += "(memory profiling unavailable) "

    current_mb = None
    peak_mb = None
    if HAS_TRACEMALLOC and tracemalloc.is_tracing():
        current, peak = tracemalloc.get_traced_memory()
        current_mb = current / (1024 * 1024)
        peak_mb = peak / (1024 * 1024)

    return MemoryStats(
        rss_mb=rss_mb,
        python_tracemalloc_current_mb=current_mb,
        python_tracemalloc_peak_mb=peak_mb,
        note=note.strip()
    )

def format_memory_stats(stats: MemoryStats) -> str:
    parts = []
    if stats.rss_mb is not None:
        parts.append(f"RSS: {stats.rss_mb:.1f} MB")
    if stats.python_tracemalloc_current_mb is not None:
        parts.append(f"Trace (Cur/Peak): {stats.python_tracemalloc_current_mb:.1f}/{stats.python_tracemalloc_peak_mb:.1f} MB")
    
    if not parts:
        return f"Memory stats unavailable [{stats.note}]"
        
    return " | ".join(parts) + f" [{stats.note}]"
