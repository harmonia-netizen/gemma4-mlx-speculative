import os
import sys
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

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
    mlx_active_mb: Optional[float]
    mlx_peak_mb: Optional[float]
    mlx_cache_mb: Optional[float]
    mlx_available: bool
    mlx_source: str
    metal_note: str
    source: str
    note: str

def maybe_reset_mlx_peak_memory() -> None:
    if not HAS_MLX:
        return
    try:
        if hasattr(mx, "metal") and hasattr(mx.metal, "reset_peak_memory"):
            mx.metal.reset_peak_memory()
    except Exception:
        pass

def maybe_clear_mlx_cache() -> None:
    if not HAS_MLX:
        return
    try:
        if hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
        elif hasattr(mx, "clear_cache"):
            mx.clear_cache()
    except Exception:
        pass

def get_memory_stats(label: Optional[str] = None) -> MemoryStats:
    rss_mb = None
    peak_rss_mb = None
    source = ""
    note = "Metal/GPU actual memory not fully included in RSS. "
    
    if HAS_PSUTIL:
        try:
            process = psutil.Process(os.getpid())
            rss_bytes = process.memory_info().rss
            rss_mb = rss_bytes / (1024 * 1024)
            source = "psutil"
        except Exception:
            source = "psutil (error)"
    
    if not rss_mb:
        try:
            import resource
            rusage = resource.getrusage(resource.RUSAGE_SELF)
            rss_kb = rusage.ru_maxrss
            # MacOS returns bytes, Linux returns KB
            if sys.platform == "darwin":
                rss_mb = rss_kb / (1024 * 1024)
                peak_rss_mb = rss_kb / (1024 * 1024)
            else:
                rss_mb = rss_kb / 1024
                peak_rss_mb = rss_kb / 1024
            source = "getrusage"
        except ImportError:
            source = "unavailable"
            note += "(memory profiling unavailable) "
        except Exception:
            source = "error"

    current_mb = None
    peak_mb = None
    if HAS_TRACEMALLOC and tracemalloc.is_tracing():
        try:
            current, peak = tracemalloc.get_traced_memory()
            current_mb = current / (1024 * 1024)
            peak_mb = peak / (1024 * 1024)
        except Exception:
            pass

    mlx_active_mb = None
    mlx_peak_mb = None
    mlx_cache_mb = None
    mlx_available = False
    mlx_source = "unavailable"
    metal_note = "unavailable"

    if HAS_MLX:
        try:
            if hasattr(mx, "metal"):
                mlx_available = True
                mlx_source = "mx.metal"
                
                if hasattr(mx.metal, "get_active_memory"):
                    mlx_active_mb = mx.metal.get_active_memory() / (1024 * 1024)
                if hasattr(mx.metal, "get_peak_memory"):
                    mlx_peak_mb = mx.metal.get_peak_memory() / (1024 * 1024)
                if hasattr(mx.metal, "get_cache_memory"):
                    mlx_cache_mb = mx.metal.get_cache_memory() / (1024 * 1024)
            else:
                mlx_source = "mx APIs not found"
                metal_note = "mx.metal not available"
        except Exception as e:
            mlx_source = f"error: {str(e)}"
            metal_note = "Exception during MLX memory check"

    parts = []
    if mlx_active_mb is not None:
        parts.append(f"active={mlx_active_mb:.1f}MB")
    if mlx_peak_mb is not None:
        parts.append(f"peak={mlx_peak_mb:.1f}MB")
    if mlx_cache_mb is not None:
        parts.append(f"cache={mlx_cache_mb:.1f}MB")
        
    if parts:
        metal_note = ", ".join(parts)

    if label:
        note = f"[{label}] {note}"

    return MemoryStats(
        rss_mb=rss_mb,
        peak_rss_mb=peak_rss_mb,
        tracemalloc_current_mb=current_mb,
        tracemalloc_peak_mb=peak_mb,
        mlx_active_mb=mlx_active_mb,
        mlx_peak_mb=mlx_peak_mb,
        mlx_cache_mb=mlx_cache_mb,
        mlx_available=mlx_available,
        mlx_source=mlx_source,
        metal_note=metal_note,
        source=source,
        note=note.strip()
    )

def memory_stats_to_dict(stats: MemoryStats) -> Dict[str, Any]:
    return asdict(stats)

def format_memory_stats(stats: MemoryStats) -> str:
    parts = []
    if stats.rss_mb is not None:
        parts.append(f"RSS: {stats.rss_mb:.1f} MB")
    if stats.peak_rss_mb is not None:
        parts.append(f"Peak RSS: {stats.peak_rss_mb:.1f} MB")
    if stats.tracemalloc_current_mb is not None:
        parts.append(f"Trace (Cur/Peak): {stats.tracemalloc_current_mb:.1f}/{stats.tracemalloc_peak_mb:.1f} MB")
    
    parts.append(f"Metal: [{stats.metal_note}]")
    
    if not parts:
        return f"Memory stats unavailable [{stats.note}]"
        
    return " | ".join(parts) + f" [source={stats.source}, mlx={stats.mlx_source}]"
