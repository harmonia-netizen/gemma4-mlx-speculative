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
git status --short
pytest --tb=short"""
    c = registry.select_candidate(user_prompt, tokenizer, min_tokens=1)
    assert c is not None
    assert c.name == "exact_pytest_plan", f"Expected exact_pytest_plan, got {c.name}"

    # 2. git status (case insensitive test)
    user_prompt_git = "作業ツリーの状態を確認するため、GIT STATUS を実行"
    c2 = registry.select_candidate(user_prompt_git, tokenizer, min_tokens=1)
    assert c2 is not None, "Expected git_status_command, but got None"
    assert c2.name == "git_status_command", f"Expected git_status_command, got {c2.name}"

    # 3. no_candidate_medium_plan / medium_pytest_plan
    user_prompt_medium = """pytest失敗の原因を安全に確認するため、次に実行する確認手順を3行のbashブロックで出してください。
前提:
- repo=gemma4-mlx-speculative
- destructive commandは禁止
- 3コマンドだけ出す
- 説明文は不要"""
    c3 = registry.select_candidate(user_prompt_medium, tokenizer, min_tokens=1)
    assert c3 is None

    # 4. confidence < 0.8
    user_prompt_low = "たぶんpytest 自信ない"
    c4 = registry.select_candidate(user_prompt_low, tokenizer, min_tokens=1)
    assert c4 is None, "Expected None due to confidence < 0.8"

    # 5. safe_check_plan
    user_prompt_safe = "安全な確認手順 ls pwd"
    c5 = registry.select_candidate(user_prompt_safe, tokenizer, min_tokens=1)
    assert c5 is not None
    assert c5.name == "safe_check_plan"

    print("  OK")

if __name__ == "__main__":
    test_registry()
    print("All tests passed.")
