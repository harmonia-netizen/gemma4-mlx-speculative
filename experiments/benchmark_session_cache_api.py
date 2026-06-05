import argparse
import time
from session_cache_api import SessionCacheAPI
from session_cache_memory import get_memory_stats, format_memory_stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat-lines", type=int, default=500)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()

    print("--- 1. Init API ---")
    api = SessionCacheAPI.load(safe_token_limit=120000)

    base_line = "User: Can you show me the test plan?\nAsst: I will check the tests and then show you the git diff.\n"
    prefix_text = base_line * args.repeat_lines

    session_id = "api_session_1"

    print("\n--- 2. Create Session ---")
    start = time.perf_counter()
    create_res = api.create_session(session_id, prefix_text)
    print(f"create result: {create_res}")
    print(format_memory_stats(get_memory_stats()))
    
    if not create_res["ok"]:
        return

    print("\n--- 3. Stats Before Generate ---")
    print(f"stats: {api.stats()}")

    print("\n--- 4. Generate ---")
    suffix_text = """次の確認手順をbashブロックだけで出してください。
前提:
- pytestの失敗内容を短く確認したい
- gitの差分も確認したい
- 出力は次の3行だけ
- 説明文は不要
- コマンドはこの順番にする:
  1. git status --short
  2. pytest --tb=short
  3. git diff
"""
    gen_res = api.generate(session_id, suffix_text, max_tokens=args.max_tokens, trace=args.trace)
    
    print(f"ok: {gen_res['ok']}")
    print(f"candidate_name: {gen_res.get('candidate_name')}")
    print(f"accepted: {gen_res.get('accepted')} / drafted: {gen_res.get('drafted')} / rejected: {gen_res.get('rejected')}")
    print(f"fallback_used: {gen_res.get('fallback_used')}")
    print(f"decode_sec: {gen_res.get('decode_sec'):.3f}s")
    print(f"text: {repr(gen_res.get('text'))}")
    print(format_memory_stats(get_memory_stats()))

    print("\n--- 5. Clear Session ---")
    clear_res = api.clear_session(session_id)
    print(f"clear result: {clear_res}")
    
    print("\n--- 6. Final Stats ---")
    print(f"stats: {api.stats()}")
    print(format_memory_stats(get_memory_stats()))

    print("\nOK: benchmark_session_cache_api completed")

if __name__ == "__main__":
    main()
