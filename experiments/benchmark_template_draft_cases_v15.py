import argparse
import importlib.util
import statistics
from dataclasses import dataclass
from pathlib import Path

V15_PATH = Path(__file__).with_name("run_gemma4_template_draft_v15.py")


@dataclass
class Case:
    name: str
    prompt: str
    max_tokens: int


CASES = [
    Case(
        "short_pytest_command",
        """あなたはローカル常駐エージェントです。
次に実行すべき確認コマンドを1つだけ出してください。
前提:
- repo=local-agent
- pytestが失敗している
- destructive commandは禁止
""",
        64,
    ),
    Case(
        "git_status_command",
        """あなたはローカル常駐エージェントです。
現在の作業ツリーの状態を確認するため、次に実行すべき確認コマンドを1つだけ出してください。
前提:
- repo=gemma4-mlx-speculative
- 変更ファイルの有無を確認したい
- destructive commandは禁止
""",
        64,
    ),
    Case(
        "git_diff_command",
        """あなたはローカル常駐エージェントです。
直前の修正内容を確認するため、次に実行すべき確認コマンドを1つだけ出してください。
前提:
- repo=gemma4-mlx-speculative
- 差分を確認したい
- destructive commandは禁止
""",
        64,
    ),
    Case(
        "medium_pytest_plan",
        """あなたはローカル常駐エージェントです。
pytest失敗の原因を安全に確認するため、次に実行する確認手順を3行のbashブロックで出してください。
前提:
- repo=gemma4-mlx-speculative
- destructive commandは禁止
- 3コマンドだけ出す
- 説明文は不要
""",
        128,
    ),
    Case(
        "exact_pytest_command",
        """あなたはローカル常駐エージェントです。
次に実行すべき確認コマンドを1つだけ出してください。
前提:
- pytestが失敗している
- 短い失敗ログを確認したい
- 出力は `pytest --tb=short` だけ
- 説明文は不要
""",
        64,
    ),
    Case(
        "exact_pytest_plan",
        """あなたはローカル常駐エージェントです。
次の確認手順をbashブロックだけで出してください。
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
        128,
    ),
]


def load_v15_module():
    spec = importlib.util.spec_from_file_location("template_draft_v15", V15_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def median(values):
    return statistics.median(values) if values else float("nan")


def summarize(case_name, name, results):
    decode_secs = [r.decode_sec for r in results]
    tok_s = [r.tok_s for r in results]
    elapsed_secs = [r.elapsed_sec for r in results]
    prefill_secs = [r.prefill_sec for r in results]

    print(f"========== summary: {case_name}: {name} ==========")
    print("runs:", len(results))
    print("tokens:", [len(r.token_ids) for r in results])
    print("median_elapsed_sec:", f"{median(elapsed_secs):.6f}")
    print("median_prefill_sec:", f"{median(prefill_secs):.6f}")
    print("median_decode_sec:", f"{median(decode_secs):.6f}")
    print("median_decode_tok_s:", f"{median(tok_s):.3f}")

    drafted = sum(getattr(r, "drafted", 0) for r in results)
    accepted = sum(getattr(r, "accepted", 0) for r in results)
    rejected = sum(getattr(r, "rejected", 0) for r in results)

    if drafted:
        print("accepted:", accepted, "/", drafted, f"({accepted / drafted:.1%})")
        print("rejected:", rejected)

    return {
        "median_decode_sec": median(decode_secs),
        "median_decode_tok_s": median(tok_s),
        "accepted": accepted,
        "drafted": drafted,
        "rejected": rejected,
    }


def run_once(v9, target_model, tokenizer, target_prompt, user_prompt, stop_ids, args, max_tokens):
    greedy = v9.run_target_greedy(
        target_model,
        tokenizer,
        target_prompt,
        stop_ids,
        max_tokens,
        args.max_kv_size,
    )

    template = v9.run_speculative(
        target_model,
        tokenizer,
        target_prompt,
        user_prompt,
        stop_ids,
        max_tokens,
        args.draft_block_size,
        args.template_min_tokens,
        args.max_kv_size,
        trace_template=args.trace_template,
    )

    if template.token_ids != greedy.token_ids:
        print("MISMATCH")
        print("greedy ids:", greedy.token_ids)
        print("template ids:", template.token_ids)
        print("greedy text:", greedy.text)
        print("template text:", template.text)
        raise SystemExit(2)

    return greedy, template


def selected_cases(case_names):
    if not case_names:
        return CASES

    names = set(case_names)
    out = [c for c in CASES if c.name in names]
    missing = names - {c.name for c in out}
    if missing:
        raise ValueError(f"unknown case(s): {sorted(missing)}")
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--draft-block-size", type=int, default=4)
    parser.add_argument("--template-min-tokens", type=int, default=16)
    parser.add_argument("--max-kv-size", type=int, default=None)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--trace-template", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.draft_block_size <= 0:
        raise ValueError("--draft-block-size must be positive")
    if args.template_min_tokens < 0:
        raise ValueError("--template-min-tokens must be >= 0")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be >= 0")
    if args.runs <= 0:
        raise ValueError("--runs must be positive")
    if args.max_kv_size is not None and args.max_kv_size <= 0:
        raise ValueError("--max-kv-size must be positive")

    v15 = load_v15_module()

    model_path = args.model or v15.DEFAULT_TARGET_MODEL_PATH

    print("loading target:", model_path)
    target_model, processor = v15.d.load(model_path)

    tokenizer = getattr(processor, "tokenizer", processor)
    stop_ids = v15.build_stop_ids(tokenizer)
    print("stop_ids:", sorted(stop_ids))

    summaries = []

    for case in selected_cases(args.case):
        print(f"========== case: {case.name} ==========")

        target_prompt = v15.format_prompt(processor, case.prompt)
        print("prompt_tokens:", len(tokenizer.encode(target_prompt)))

        for i in range(args.warmup_runs):
            print(f"warmup {i + 1}/{args.warmup_runs}")
            run_once(
                v15,
                target_model,
                tokenizer,
                target_prompt,
                case.prompt,
                stop_ids,
                args,
                case.max_tokens,
            )

        greedy_results = []
        template_results = []

        for i in range(args.runs):
            print(f"run {i + 1}/{args.runs}")
            greedy, template = run_once(
                v15,
                target_model,
                tokenizer,
                target_prompt,
                case.prompt,
                stop_ids,
                args,
                case.max_tokens,
            )
            greedy_results.append(greedy)
            template_results.append(template)

            print(
                "  greedy_decode_sec=",
                f"{greedy.decode_sec:.6f}",
                "template_decode_sec=",
                f"{template.decode_sec:.6f}",
                "template_accept=",
                f"{template.accepted}/{template.drafted}",
            )

        greedy_summary = summarize(case.name, "target greedy", greedy_results)
        template_summary = summarize(case.name, "template draft", template_results)

        speedup = (
            greedy_summary["median_decode_sec"] / template_summary["median_decode_sec"]
            if template_summary["median_decode_sec"] > 0
            else float("inf")
        )

        print(f"decode_sec_speedup: {speedup:.3f}x")

        summaries.append(
            (
                case.name,
                greedy_summary["median_decode_sec"],
                template_summary["median_decode_sec"],
                speedup,
                template_summary["accepted"],
                template_summary["drafted"],
            )
        )

    print("========== all cases ==========")
    for name, greedy_sec, template_sec, speedup, accepted, drafted in summaries:
        rate = f"{accepted / drafted:.1%}" if drafted else "n/a"
        print(
            name,
            "greedy_decode_sec=",
            f"{greedy_sec:.6f}",
            "template_decode_sec=",
            f"{template_sec:.6f}",
            "speedup=",
            f"{speedup:.3f}x",
            "accept=",
            f"{accepted}/{drafted}",
            rate,
        )

    print("OK: case benchmark completed with token match")


if __name__ == "__main__":
    main()
