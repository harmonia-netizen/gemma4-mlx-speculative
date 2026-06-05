import subprocess
import sys

def run_cmd(cmd: str):
    print(f"\n========================================")
    print(f"Running: {cmd}")
    print(f"========================================")
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    res = subprocess.run(cmd, shell=True, env=env)
    if res.returncode != 0:
        print(f"\n[ERROR] Command failed with exit code {res.returncode}: {cmd}")
        sys.exit(res.returncode)
    print(f"[OK] {cmd}\n")

def main():
    commands = [
        # 1. Compile checks
        ".venv/bin/python -m py_compile gemma4_mlx_runtime/__init__.py gemma4_mlx_runtime/session_cache.py gemma4_mlx_runtime/memory.py gemma4_mlx_runtime/candidates.py gemma4_mlx_runtime/package_info.py gemma4_mlx_runtime/backends.py gemma4_mlx_runtime/mlx_backend.py gemma4_mlx_runtime/llama_cpp_backend.py experiments/template_draft_runtime.py experiments/session_cache_runtime.py experiments/session_cache_api.py experiments/session_cache_core.py experiments/session_cache_memory.py experiments/session_cache_package.py experiments/agent_mock_runtime.py experiments/test_memory_stats.py experiments/benchmark_memory_stats.py experiments/benchmark_llama_cpp_backend.py experiments/benchmark_llama_cpp_multiturn.py experiments/test_backend_interface.py experiments/test_llama_cpp_backend_light.py",
        
        # 2. Unit tests
        ".venv/bin/python experiments/test_prefix_cache_manager.py",
        ".venv/bin/python experiments/test_candidate_registry.py",
        ".venv/bin/python experiments/test_session_cache_api.py",
        ".venv/bin/python experiments/test_agent_mock_runtime.py",
        ".venv/bin/python experiments/test_memory_stats.py",
        ".venv/bin/python experiments/test_backend_interface.py",
        ".venv/bin/python experiments/test_llama_cpp_backend_light.py",
        
        # 3. Benchmarks
        ".venv/bin/python experiments/benchmark_session_cache_api.py --repeat-lines 200 --max-tokens 16",
        ".venv/bin/python experiments/benchmark_agent_mock.py --repeat-lines 200 --max-tokens 16",
        ".venv/bin/python experiments/benchmark_agent_multiturn.py --repeat-lines 200 --max-tokens 16",
        ".venv/bin/python experiments/benchmark_memory_stats.py",
        
        # 4. Probes
        ".venv/bin/python experiments/session_100k_probe.py --target-tokens 100000 --safe-token-limit 32000 --max-tokens 16",
        ".venv/bin/python experiments/prompt_100k_reuse_path_probe.py --target-tokens 32000 --max-tokens 16 --step-size 512 --case exact_pytest_plan"
    ]

    for cmd in commands:
        run_cmd(cmd)

    print("\nOK: completion checks passed")

if __name__ == "__main__":
    main()
