# Template Draft Engine 実験録

## 目的
v15 までの実験で成立した「Candidate dataclass / confidence gating / fast path + restore fallback」の仕組みを、実験用のバージョン付きスクリプトから再利用可能な形に分離・整理すること。
これにより、アプリケーションや今後の別の実験において、推論ロジックの中核（エンジン部分）を容易に利用できるようにする。

## 分離・設計したコンポーネント (template_draft_engine.py)
実験スクリプト内に混在していたロジックを以下の責務に分割・整理した：

1. **データ構造 (Dataclasses)**
   - `Candidate`: 候補のメタデータ (text, confidence, min_tokens, tags など) を保持。
   - `DecodeResult`: 生成結果とパフォーマンス指標をカプセル化。
2. **キャッシュ・スナップショット管理**
   - 安全性の高い `full_snapshot` および `restore_full` を提供。
   - Mismatch発生時のロールバック処理を担う。
3. **候補の生成と選択 (Registry/Provider)**
   - `draft_candidates`: ユーザープロンプトに応じてCandidateリストを生成。
   - `select_candidate`: `confidence >= 0.8` や `min_tokens` 制約によるGatingを行い、最適な候補を1つ選択。
4. **デコードエンジン (Generation/Inference)**
   - `run_target_greedy`: 比較用の標準的なGreedyデコード。
   - `run_template_draft`: 予測候補を用いたSpeculative Decodingの実行本体 (旧 `run_speculative` から改名)。

## 実行結果
分離後のエンジン (`benchmark_template_draft_engine.py`) を用いた検証結果は以下の通りであり、v15 と同等の正確性とパフォーマンスを維持していることを確認した。

- **exact_pytest_plan**
  - **Token Match:** OK
  - **Accepted:** 100%
  - **Speedup:** 約2.19x〜2.21x (期待値通り)
- **medium_pytest_plan**
  - 意図的に候補が排除(Gated out)されるため、無駄なフォールバックを回避。
  - **Speedup:** 約0.996x〜1.00x (Greedy同等の速度)
- **すべてのケース**
  - 完全なToken Matchを達成 (Mismatch = 0)。

## 既存実験版との関係
v10 から v15 までのファイルは、推論アルゴリズムやキャッシング戦略の進化過程を記録する「スナップショット」としてそのまま保持している。
今回の `template_draft_engine.py` は、それらの実験の集大成となる「実用版」の基礎である。

## 次の課題
1. **Candidate Registry の外部化**
   - 現在はエンジン内にハードコードされた候補リストがあるが、これを外部から注入 (Inject) できるような設計に変更する。
2. **API・パッケージ化**
   - Agentや他のシステムからインポートして使いやすいようにクラスベースのインターフェース (例: `TemplateDraftGenerator`) に昇華する。
3. **実運用候補セットの追加とREADME更新**
   - 実際の開発ワークフローで頻出するコマンドパターン（Lint, Test, Statusなど）のテンプレートを拡充する。
