from template_draft_runtime import CandidateRegistry

def test_registry():
    print("test_registry...")
    registry = CandidateRegistry("experiments/template_candidates.json")
    assert len(registry.entries) > 0, "Failed to load candidates"

    class DummyTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [0] * len(text)
            
    tokenizer = DummyTokenizer()

    # 1. exact_pytest_plan
    user_prompt = """次の確認手順をbashブロックだけで出してください
pytestの失敗内容を短く確認したい
git status
git diff
3行で出して"""
    c = registry.select_candidate(user_prompt, tokenizer, min_tokens=1)
    assert c is not None
    assert c.name in ["exact_pytest_plan", "exact_pytest_plan_qwen"], f"Expected exact_pytest_plan, got {c.name}"

    # 2. git status (case insensitive test)
    user_prompt_git = "作業ツリーの状態を確認するため、GIT STATUS を実行"
    c2 = registry.select_candidate(user_prompt_git, tokenizer, min_tokens=1)
    assert c2 is not None, "Expected git_status_short, but got None"
    assert c2.name == "git_status_short", f"Expected git_status_short, got {c2.name}"

    # 3. no_candidate_medium_plan / medium_pytest_plan
    user_prompt_medium = """pytest失敗の原因を安全に確認するため、次に実行する確認手順を3行のbashブロックで出してください。
前提:
- repo=local-speculative-runtime
- destructive commandは禁止
- 3コマンドだけ出す
- 説明文は不要"""
    c3 = registry.select_candidate(user_prompt_medium, tokenizer, min_tokens=1)
    assert c3 is None or c3.name == "safe_runtime_check_plan"

    # 4. confidence < 0.8
    user_prompt_low = "たぶんpytest 自信ない"
    c4 = registry.select_candidate(user_prompt_low, tokenizer, min_tokens=1)
    assert c4 is None, "Expected None due to confidence < 0.8"

    # 5. safe_check_plan
    user_prompt_safe = "安全な確認手順 check ls pwd"
    c5 = registry.select_candidate(user_prompt_safe, tokenizer, min_tokens=1)
    assert c5 is not None
    assert c5.name == "safe_runtime_check_plan"

    # 6. python_compile_check
    user_prompt_compile = "pythonのコンパイル確認のため py_compile"
    c6 = registry.select_candidate(user_prompt_compile, tokenizer, min_tokens=1)
    assert c6 is not None
    assert c6.name == "py_compile_runtime"

    # 7. git_add_status_commit_plan (low confidence -> None)
    user_prompt_commit = "変更をコミットしたいので git add と git commit して"
    c7 = registry.select_candidate(user_prompt_commit, tokenizer, min_tokens=1)
    assert c7 is None, "Expected None due to lowered confidence for git_add_commit"

    # 8. negative_keywords test: rm should reject py_compile_runtime
    user_prompt_rm = "pythonのコンパイル確認のため py_compile して、不要なファイルは rm で消して"
    c8 = registry.select_candidate(user_prompt_rm, tokenizer, min_tokens=1)
    assert c8 is None, "Expected None because 'rm' is in negative keywords"

    # 9. score threshold test: safe_runtime_check_plan needs 3 any_keywords matched
    # "安全" (1), "確認" (2), "check" (3) -> total 3
    user_prompt_safe_threshold = "安全な確認check"
    c9 = registry.select_candidate(user_prompt_safe_threshold, tokenizer, min_tokens=1)
    assert c9 is not None
    assert c9.name == "safe_runtime_check_plan"

    user_prompt_safe_fail = "安全な処理" # only 1 any_keyword ("安全")
    c10 = registry.select_candidate(user_prompt_safe_fail, tokenizer, min_tokens=1)
    assert c10 is None, "Expected None because score threshold is not met"

    print("  OK")

if __name__ == "__main__":
    test_registry()
    print("All tests passed.")
