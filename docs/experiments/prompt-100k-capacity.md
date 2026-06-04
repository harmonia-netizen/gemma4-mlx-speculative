# Prompt 100k Capacity Probe

## 目的
本実験は、`prompt_100k.txt` (100,323 トークン) を用いて、現在の Template Draft Runtime アーキテクチャがこの規模の大規模コンテキスト (100K 級) を処理できるか、および Snapshot の保持によってメモリ制約 (OOM) が発生しないかを安全に再検証することを目的とする。

## 前提と歴史
過去の実験において、この 100K プロンプトは以下の構成で検証されていた。
- `run_100k_vlm.py`: Target model (Gemma 4) 単体での推論 (Target-only)。
- `run_100k_vlm_mtp2.py` / `mtp6.py`: MTP (Multi-Token Prediction) モデルによるドラフト推論。
- `run_lm_100k_dense_draft.py` 等: 別途のドラフトモデルを用いた Dense Draft 構成。

過去の MTP 検証ではメモリ負荷がより大きかったが、現在の Runtime は「Target-only による Prefix Prefill」と「Snapshot によるキャッシュ保持」を行うため、これらが 100K トークンに対して正常に動作するかを単独で切り分ける必要がある。

## 検証スクリプト
- `experiments/prompt_100k_capacity_probe.py`
  - 100,323 トークンのプロンプトをチャンク単位 (`step_size=512`) で Prefill する機構を実装。
  - `--mode` により Count / Prefill / Prefill-Snapshot / Greedy の段階的な検証を実行。

## 実験結果

### 1. Count Mode
- **Token Count**: 100,335 トークン (Chat Template のオーバーヘッドを含む)
- 対象モデル: `mlx-community/gemma-4-26b-a4b-it-8bit`

### 2. Target-only Prefill Mode
- **結果**: 正常終了 (OOM なし)
- **Prefill Time**: 約 **314〜322秒**
- **挙動**: キャッシュ (30 layers) の生成が正常に行われ、100K スケールの Attention が 64GB 以上の統合メモリ（あるいはスワップ込み）で破綻せずに処理されることを確認した。

### 3. Prefill Snapshot Mode
- **結果**: 正常終了 (OOM なし)
- **Snapshot Time**: **0.000秒**
- **考察**: 100K prefill後に snapshot object を作成しても、追加の大きなメモリ確保は発生しませんでした。ただし、このmodeではrestore correctnessまでは検証していないため、100Kでのrestore安全性は別途restore probeが必要です。

### 4. Greedy Decode Mode
- **結果**: 正常終了 (OOM なし)
- **Prefill Time**: 約 **340秒**
- **Decode Time**: **1.270秒** (16 tokens)
- **Output Snippet**: `### 1. 結論\n**採用すべきではない。**\n10`
- **考察**: 100K のコンテキストをロードした状態で推論を継続できることを確認した。デコード自体の速度は約 12.6 tokens/sec であり、長大なキャッシュを抱えた状態でも推論自体は実行可能。

## 制限と今後の方針 (Safe Limit)
- **Prefill 時間の壁**: M2/M3 等の Apple Silicon 環境において、100K の Prefill は約 5 分を要する。これは対話型エージェントのレスポンスとしては実用的ではない。
- **Prefix Cache Reuse の必然性**: この結果は「Prefix Cache Reuse 機構」がいかに不可欠であるかを物語っている。1度 5分 かけて Prefill を行った後に同一prefixを再利用できる可能性はあるが、100K状態でのrestore correctnessと長時間運用時のメモリ管理は別途検証が必要である。
- **Safe Token Limit**: 現状の MLX / Gemma 4 の 8-bit 量子化モデルにおいて、100K は処理可能であるものの、実行ハードウェアの物理メモリ (64GB〜128GB) によっては Swapping が発生し、著しい速度低下を招く。当面の推奨上限 (`safe-token-limit`) は 120,000 トークン程度とする。
