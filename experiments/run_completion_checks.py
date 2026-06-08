import subprocess
import sys
import os

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
        ".venv/bin/python -m py_compile local_speculative_runtime/__init__.py local_speculative_runtime/session_cache.py local_speculative_runtime/memory.py local_speculative_runtime/candidates.py local_speculative_runtime/package_info.py local_speculative_runtime/backends.py local_speculative_runtime/mlx_backend.py local_speculative_runtime/llama_cpp_backend.py local_speculative_runtime/cli.py local_speculative_runtime/openai_api.py experiments/template_draft_runtime.py experiments/session_cache_runtime.py experiments/session_cache_api.py experiments/session_cache_core.py experiments/session_cache_memory.py experiments/session_cache_package.py experiments/agent_mock_runtime.py experiments/test_memory_stats.py experiments/benchmark_memory_stats.py experiments/benchmark_llama_cpp_backend.py experiments/benchmark_llama_cpp_multiturn.py experiments/benchmark_llama_cpp_prefix_reuse.py experiments/benchmark_llama_cpp_state_restore.py experiments/benchmark_llama_cpp_template_draft.py experiments/test_backend_interface.py experiments/test_llama_cpp_backend_light.py experiments/test_cli_args.py experiments/test_openai_api.py experiments/smoke_openai_api.py experiments/smoke_openai_sdk.py experiments/run_model_checks.py",
        
        # 2. Unit tests
        ".venv/bin/python experiments/test_openai_api.py",
        ".venv/bin/python experiments/test_cli_args.py",
        ".venv/bin/python experiments/test_prefix_cache_manager.py",
        ".venv/bin/python experiments/test_candidate_registry.py",
        ".venv/bin/python experiments/test_session_cache_api.py",
        ".venv/bin/python experiments/test_memory_stats.py",
        ".venv/bin/python experiments/test_backend_interface.py",
        ".venv/bin/python experiments/test_llama_cpp_backend_light.py",
        
        # 3. Benchmarks
        ".venv/bin/python experiments/benchmark_memory_stats.py",
    ]

    for cmd in commands:
        run_cmd(cmd)

    print("\nOK: completion checks passed")

if __name__ == "__main__":
    main()
