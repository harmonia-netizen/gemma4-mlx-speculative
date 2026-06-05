import argparse
import time

from session_cache_api import SessionCacheAPI
from session_cache_memory import get_memory_stats, format_memory_stats

CASES = {
    "exact_pytest_plan": """次の確認手順をbashブロックだけで出してください。
前提:
- pytestの失敗内容を短く確認したい
- gitの差分も確認したい
- 出力は次の3行だけ
- 説明文は不要
- コマンドはこの順番にする:
  1. git status --short
  2. pytest --tb=short
  3. git diff
""",
    "git_status_command": "作業ツリーの状態を確認するため、git status を実行してください。",
    "git_diff_command": "現在のgitの差分を確認したいので、コマンドを教えてください。",
    "python_compile_check": "pythonコードの構文エラーを確認するため、experiments/session_cache_runtime.py を py_compile してください。",
    "safe_check_plan": "現在の状態を安全に確認したいです。pwdとlsだけを出してください。",
    "medium_pytest_plan": """pytest失敗の原因を安全に確認するため、次に実行する確認手順を3行のbashブロックで出してください。
前提:
- repo=gemma4-mlx-speculative
- destructive commandは禁止
- 3コマンドだけ出す
- 説明文は不要
"""
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat-lines", type=int, default=500)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--safe-token-limit", type=int, default=120000)
    parser.add_argument("--step-size", type=int, default=512)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()

    print("--- 1. Init API ---")
    api = SessionCacheAPI.load(safe_token_limit=args.safe_token_limit, step_size=args.step_size)

    base_line = "User: We need to debug the speculative decoding issue.\nAsst: I'll check the current status and run tests.\n"
    prefix_text = base_line * args.repeat_lines

    session_id = "agent_session_1"

    print("\n--- 2. Create Session ---")
    start = time.perf_counter()
    create_res = api.create_session(session_id, prefix_text)
    print(f"create result ok: {create_res['ok']}")
    if not create_res["ok"]:
        print(f"Failed to create session: {create_res.get('guard_reason')}")
        return
        
    print(f"prefix_prefill_sec: {create_res['prefix_prefill_sec']:.3f}s")
    print(format_memory_stats(get_memory_stats()))

    print("\n--- 3. Multi-turn Generate ---")
    for case_name, user_prompt in CASES.items():
        print(f"\nTurn: {case_name}")
        gen_res = api.generate(session_id, user_prompt, max_tokens=args.max_tokens, trace=args.trace)
        
        if not gen_res["ok"]:
            print(f"Error: {gen_res.get('error')}")
            continue
            
        print(f"suffix_tokens: {gen_res['suffix_tokens']}")
        print(f"suffix_prefill_sec: {gen_res['suffix_prefill_sec']:.3f}s")
        print(f"decode_sec: {gen_res['decode_sec']:.3f}s")
        print(f"elapsed_sec: {gen_res['elapsed_sec']:.3f}s")
        print(f"candidate_name: {gen_res['candidate_name']}")
        print(f"accepted/drafted/rejected: {gen_res['accepted']}/{gen_res['drafted']}/{gen_res['rejected']}")
        print(f"fallback_used: {gen_res['fallback_used']}")
        print(f"output snippet: {repr(gen_res['text'])}")
        print(format_memory_stats(get_memory_stats()))

    print("\n--- 4. Stats ---")
    print(api.stats())

    print("\n--- 5. Clear Session ---")
    api.clear_session(session_id)
    print(api.stats())
    print(format_memory_stats(get_memory_stats()))

    print("\nOK: agent multiturn benchmark completed")

if __name__ == "__main__":
    main()
