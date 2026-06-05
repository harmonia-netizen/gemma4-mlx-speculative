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

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

@dataclass
class MemoryStats:
    rss_mb: Optional[float]
    peak_rss_mb: Optional[float]
    tracemalloc_current_mb: Optional[float]
    tracemalloc_peak_mb: Optional[float]
    mlx_cache_note: str
    metal_note: str
    source: str
    note: str

def get_memory_stats() -> MemoryStats:
    rss_mb = None
    peak_rss_mb = None
    source = ""
    note = "Metal/GPU actual memory not included in RSS. "
    
    if HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        rss_bytes = process.memory_info().rss
        rss_mb = rss_bytes / (1024 * 1024)
        source = "psutil"
    else:
        try:
            import resource
            rusage = resource.getrusage(resource.RUSAGE_SELF)
            rss_kb = rusage.ru_maxrss
            # getrusage returns KB on Linux but bytes on Mac. We assume Mac (bytes) based on context.
            rss_mb = rss_kb / (1024 * 1024)
            peak_rss_mb = rss_kb / (1024 * 1024) # ru_maxrss is the peak
            source = "getrusage"
        except ImportError:
            source = "unavailable"
            note += "(memory profiling unavailable) "

    current_mb = None
    peak_mb = None
    if HAS_TRACEMALLOC and tracemalloc.is_tracing():
        current, peak = tracemalloc.get_traced_memory()
        current_mb = current / (1024 * 1024)
        peak_mb = peak / (1024 * 1024)

    mlx_cache_note = "unavailable"
    metal_note = "unavailable"

    if HAS_MLX:
        parts = []
        if hasattr(mx, "metal"):
            # backwards compatibility fallback or new API
            if hasattr(mx, "metal") and hasattr(mx.metal, "get_active_memory"):
                active = getattr(mx, "get_active_memory", mx.metal.get_active_memory)() / (1024 * 1024)
                parts.append(f"active={active:.1f}MB")
            if hasattr(mx, "metal") and hasattr(mx.metal, "get_peak_memory"):
                peak_m = getattr(mx, "get_peak_memory", mx.metal.get_peak_memory)() / (1024 * 1024)
                parts.append(f"peak={peak_m:.1f}MB")
            if hasattr(mx, "metal") and hasattr(mx.metal, "get_cache_memory"):
                cache = getattr(mx, "get_cache_memory", mx.metal.get_cache_memory)() / (1024 * 1024)
                parts.append(f"cache={cache:.1f}MB")
        elif hasattr(mx, "metal"):
            metal_note = "mx APIs not found"
        
        if parts:
            metal_note = ", ".join(parts)

    return MemoryStats(
        rss_mb=rss_mb,
        peak_rss_mb=peak_rss_mb,
        tracemalloc_current_mb=current_mb,
        tracemalloc_peak_mb=peak_mb,
        mlx_cache_note=mlx_cache_note,
        metal_note=metal_note,
        source=source,
        note=note.strip()
    )

def format_memory_stats(stats: MemoryStats) -> str:
    parts = []
    if stats.rss_mb is not None:
        parts.append(f"RSS: {stats.rss_mb:.1f} MB")
    if stats.peak_rss_mb is not None:
        parts.append(f"Peak RSS: {stats.peak_rss_mb:.1f} MB")
    if stats.tracemalloc_current_mb is not None:
        parts.append(f"Trace (Cur/Peak): {stats.tracemalloc_current_mb:.1f}/{stats.tracemalloc_peak_mb:.1f} MB")
    
    parts.append(f"Metal: [{stats.metal_note}]")
    
    if len(parts) == 1: # only Metal info
        return f"Memory stats unavailable [{stats.note}]"
        
    return " | ".join(parts) + f" [source={stats.source}]"
