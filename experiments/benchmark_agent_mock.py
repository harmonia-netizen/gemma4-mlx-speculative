import argparse
import time

from agent_mock_runtime import AgentMockRuntime
from session_cache_memory import get_memory_stats, format_memory_stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat-lines", type=int, default=500)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--safe-token-limit", type=int, default=120000)
    parser.add_argument("--step-size", type=int, default=512)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()

    print("--- 1. Init Agent Mock ---")
    runtime = AgentMockRuntime(safe_token_limit=args.safe_token_limit, step_size=args.step_size)

    base_line = "Context: Project directory is ~/mlx, speculative decoding engine is in experiments.\nUser: I am ready.\n"
    prefix_text = base_line * args.repeat_lines

    session_id = "agent_mock_1"

    print("\n--- 2. Initialize Context ---")
    start = time.perf_counter()
    init_res = runtime.initialize_context(session_id, prefix_text)
    print(f"ok: {init_res['ok']}")
    if not init_res["ok"]:
        print(f"Failed to initialize context: {init_res.get('guard_reason')}")
        return
        
    print(f"prefix_prefill_sec: {init_res['prefix_prefill_sec']:.3f}s")
    print(format_memory_stats(get_memory_stats()))

    tasks = [
        ("pytest failed", "次の確認手順をbashブロックだけで出してください\npytestの失敗内容を短く確認したい\ngit status --short\npytest --tb=short"),
        ("check git status", "作業ツリーの状態を確認するため、git status --shortを実行"),
        ("check diff stat", "git diff --stat を出して"),
        ("compile runtime files", "pythonのコンパイル確認のため py_compile して"),
        ("ambiguous medium plan", "pytest失敗の原因を安全に確認するため、次に実行する確認手順を3行のbashブロックで出してください。\n前提:\n- destructive commandは禁止\n- 3コマンドだけ出す")
    ]

    print("\n--- 3. Handle Tasks ---")
    for task_name, task_text in tasks:
        print(f"\nTask: {task_name}")
        res = runtime.handle_task(session_id, task_text, max_tokens=args.max_tokens, trace=args.trace)
        
        if not res["ok"]:
            print(f"Error: {res.get('error')}")
            continue
            
        print(f"suffix_tokens: {res['suffix_tokens']}")
        print(f"suffix_prefill_sec: {res['suffix_prefill_sec']:.3f}s")
        print(f"decode_sec: {res['decode_sec']:.3f}s")
        print(f"elapsed_sec: {res['elapsed_sec']:.3f}s")
        print(f"candidate_name: {res['candidate_name']}")
        print(f"accepted/drafted/rejected: {res['accepted']}/{res['drafted']}/{res['rejected']}")
        print(f"fallback_used: {res['fallback_used']}")
        print(f"output snippet: {repr(res['text'])}")
        print(format_memory_stats(get_memory_stats()))

    print("\n--- 4. Stats ---")
    print(runtime.stats())

    print("\n--- 5. Clear ---")
    runtime.clear(session_id)
    print(runtime.stats())
    print(format_memory_stats(get_memory_stats()))

    print("\nOK: agent mock benchmark completed")

if __name__ == "__main__":
    main()
