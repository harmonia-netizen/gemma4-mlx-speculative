# Long Context Template Draft Benchmark

## 目的
これまでの実験で、`exact_pytest_plan` のような短〜中プロンプト下においては、Template Draft Engineがデコードフェーズを約2.2倍高速化できることが確認された。
しかし、ローカル常駐エージェントの実運用では、ログファイルや過去の推論履歴など大規模なコンテキスト (Long Context) を伴うプロンプト入力が想定される。
このような状況下では、推論時間全体の大部分を **Prefill (プロンプトの初期処理)** が占めることが予想される。
本実験の目的は以下の通りである。
1. 大規模な入力下でも Template Draft Engine (fast path + restore fallback) の Token Match が正しく保たれることを証明する。
2. `decode_sec` (デコード時間) の Speedup と `elapsed_sec` (全体時間) の Speedup の乖離を測定する。
3. Prefill支配時における投機的デコード (Speculative Decoding) の実効速度向上限界を明らかにする。

## 実験方法
- 新規作成した `benchmark_long_context_engine.py` を用いて、Target Greedy と Template Draft Engine を比較。
- `--repeat-lines` を用いて、ダミーのログ行を複製して長文コンテキストを生成。
- プロンプトの最後には Template Candidate が必ず当たるように、定型のコマンド要求 (`exact_pytest_plan`相当) を配置。

## 実験結果

### 1. Repeat Lines: 500 (~14,600 tokens)
- **Token Match**: OK
- **Target Greedy**:
  - `median_elapsed_sec`: 46.202s
  - `median_prefill_sec`: 45.217s
  - `median_decode_sec`: 0.534s
  - Prefill Share: 97.9%
- **Template Draft Engine**:
  - `median_elapsed_sec`: 46.107s
  - `median_prefill_sec`: 45.377s
  - `median_decode_sec`: 0.317s
  - Prefill Share: 98.4%
  - **Accepted**: 51/51 (100.0%)
- **Speedup**:
  - `decode_sec_speedup`: **1.683x**
  - `elapsed_sec_speedup`: **1.002x**

### 2. Repeat Lines: 2000 (~58,000 tokens)
- **Token Match**: 測定不能 (OOM)
- **結果**: 実行時エラー `RuntimeError: [metal::malloc] Attempting to allocate 108030675488 bytes...` により強制終了。
- 現在の MLX 実装 (キャッシュのフル計算) では、約58kトークンの入力に対して約100GB以上のメモリ要求が発生し、Apple Silicon (実行環境) のリソース上限を突破して OOM (Out Of Memory) となることが確認された。

## 考察とPrefill支配時の限界
- 長文コンテキスト (~14.6k tokens) においても、Template Draft の検証パスおよび Token Match の正しさは完全に維持されることが証明された。
- デコード速度 (`decode_sec_speedup`) は 1.68x 程度の高速化を達成しているが、**全体時間 (`elapsed_sec_speedup`) の向上は 1.002x に留まる**。
- これは推論全体の約98%以上を Prefill が占有しているため、デコードをどれほど高速化してもアムダールの法則により全体のパフォーマンス向上にはほとんど寄与しないことを示している。
- したがって、Long Context なエージェント環境において全体の応答速度（Time to First Token 含む）を改善するには、投機的デコードとは別のアプローチ（Prefillの高速化・キャッシュ化）が必須である。

## 次の課題
1. **Prefill Cache Reuse (KV Cache の再利用)**
   - 毎回最初から Prefill を計算するのではなく、過去の実行状態 (KV Cache) をディスクやメモリからリストアする機構の検討。
2. **Prompt Prefix Caching**
   - 共通のシステムプロンプトや不変のコンテキスト部分をキャッシュ化し、動的な差分のみを計算する技術の導入。
3. **Multi-turn Agent Context での再利用**
   - 常駐エージェントの対話ループ内で、不要な Prefill 計算を省くアーキテクチャの設計。
4. **出力長が長いケースでの追加評価**
   - ターゲット側の出力 (デコードトークン数) が非常に長いケースにおいて、Prefill の割合が相対的に低下した際の実効 Speedup の検証。
