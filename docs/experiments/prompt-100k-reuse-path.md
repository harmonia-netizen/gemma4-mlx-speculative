# Prompt 100K Reuse Path Probe

## 目的
100K 級の長大な共通プロンプト (Prefix) を処理する際、Baseline 比較を含む複数回のフルプレフィル直列実行は、統合メモリ上で深刻な Swapping を引き起こし実用的ではない (`prompt_100k_runtime_probe.py` の結果より)。
しかし実運用シナリオにおいては、「一度の 100K Chunked Prefill と Snapshot 取得」の後に、「Snapshot を Restore しながら複数ターンの短い Suffix 推論を行う (Prefix Cache Reuse Path)」というアプローチのみが用いられる。
本実験では、この **実運用に近い Reuse Path 単独での 100K トークン処理の成立性と実行タイミング (Prefill / Suffix / Decode の各所要時間)** を計測する。

## 検証スクリプト
- `experiments/prompt_100k_reuse_path_probe.py`

### 実行フェーズ
1. **Prefix Prefill & Snapshot**: 
   - `step_size=512` による Chunked Prefill で対象トークン (32K / 100K) 分のキャッシュを構築する。
   - 直後に `full_snapshot()` を行い、状態を退避する。
2. **Prefix Reuse + Template Draft**:
   - キャッシュを Restore する。
   - `exact_pytest_plan` (短い検証用 Suffix) などの追加プロンプトを Prefill する。
   - Template Draft Engine を用いて推測デコード (Target Verify) を実行し、生成トークンと Accept Rate、および各フェーズのタイミングを記録する。

*(※本 Probe では full baseline との Token Match 比較は行わない。100K Restore Probe では、今回条件において restore 後の生成トークン列が baseline と一致することを確認済み。)*

## 結果一覧

### 32K Tokens
- **Target Tokens**: 32,000
- **Prefix Prefill**: 86.369s
- **Snapshot Time**: ~0.000s
- **Case**: `exact_pytest_plan` (Suffix tokens: 92)
  - **Suffix Prefill**: 0.496s
  - **Decode (16 tokens)**: 0.398s
  - **Elapsed (Excluding Prefix)**: 0.894s
  - **Template Draft**: Accepted 15 / Drafted 15 (100% Accept)

### 100K Tokens
- **Target Tokens**: 100,000
- **Prefix Prefill**: 357.344s
- **Snapshot Time**: ~0.000s
- **Case**: `exact_pytest_plan` (Suffix tokens: 92)
  - **Suffix Prefill**: 0.736s
  - **Decode (16 tokens)**: 1.514s
  - **Elapsed (Excluding Prefix)**: 2.250s
  - **Template Draft**: Accepted 0 / Drafted 8 / Rejected 1
  - **挙動**: プロンプトの微小な揺れによって Target Greedy が `python -m pytest...` を選択し、Template ( `pytest...` ) との不一致 (Mismatch) を正確に検知。巨大な 100K キャッシュ上でも安全に `restore_full()` によるロールバックと Greedy フォールバックが発動し、2.25秒 で生成が完了することを確認した。

## 考察と意味

### 1. 100K Reuse Path の実用性
100K 級のプロンプトを 1 回ロードするのには約 7〜8分 の時間 (Prefix Prefill) を要するが、その後 Snapshot を Restore して利用する「Reuse Path」は、**わずか 2.6秒程度 (Suffix Prefill + Decode)** で完了する。
これは 100K という極限の長文コンテキストにおいても、Prefix Cache Reuse 機構が「実用的な対話エージェント・レスポンスタイム」を実現するための要石であることを示している。

### 2. 100K 環境下での Template Draft の成立
巨大な 100K キャッシュを背負った状態においても、Template Draft の検証 (Target Verify / `forward_many`) 機構は問題なく機能した。
32K の `exact_pytest_plan` では 15 トークンが 1 ブロックで accept され、template draft fast path が成立した。一方、100K の `exact_pytest_plan` では候補が一致せず accepted=0/8 となり、mismatch rollback 後に greedy fallback で完走した。

## 次の課題
1. **LongInputGuard 正式化**: 1回の Prefill に 数百秒 かかるため、本番環境の Runtime では、プロンプトのトークン数が `safe-token-limit` を超えないよう事前に弾くガード機構を正式に実装する必要がある。
2. **PrefixCacheManager の上限管理**: 1つの 100K Snapshot が保持する参照ツリーは極めて大きいため、複数の Session Cache を同時に抱えた場合の OOM リスクに対する上限管理 (LRU 等) の実装が急務である。
3. **Reuse Path の実アプリ統合**: 対話ループの中核として、今回の Reuse Path アーキテクチャを実際の Agent System へ組み込む。
