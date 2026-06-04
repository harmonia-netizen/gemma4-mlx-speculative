# Prefix Cache Reuse 実験録

## 目的
前回の Long Context Benchmark 実験において、大規模コンテキスト (例: 14.6k トークン) の推論では処理時間全体の約 98% を Prefill が占めることが判明した。
これにより、Target Verify 方式の Speculative Decoding (Template Draft Engine) を用いて Decode をいくら高速化しても、全体の実効速度 (`elapsed_sec`) はほとんど改善されない (Speedup ~1.002x) という限界が明らかになった。

この課題を解決するため、本実験では **「共通する長文コンテキスト (Prefix) の Prefill Cache を Snapshot として保存し、後続の異なる指示 (Suffix) の推論で再利用する」** アプローチを検証する。

目標は以下の通りである。
1. Prefix Cache を復元・再利用することで、2回目以降の推論において Prefix 分の Prefill 時間を削減できるかを測定する。
2. その際に、ベースライン (毎回 Full Prompt を Prefill する方式) と比較して Token Match が完全に維持されるかを証明する。

## 実験 (Probe) 設計
- **スクリプト**: `experiments/prefix_cache_reuse_probe.py`
- **Prefix**: `[INFO] ...` といったダミーのログ行を指定回数 (`repeat-lines`) 生成した長文テキスト。
- **Suffix**: 推論ごとに切り替わる短い要求 (例: `exact_pytest_plan`, `short_pytest`, `git_status`)。計3ケース。
- **ベースライン手法 (Baseline)**: Prefix と Suffix を結合した完全な Prompt を毎ケース最初から Prefill し、Greedy Decode を行う。
- **再利用手法 (Prefix Reuse)**:
  1. Prefix のみで一度だけ Prefill を行い、`full_snapshot` でキャッシュを保存。
  2. 各ケースにおいて `restore_full` で Prefix Cache を復元し、Suffix のみを Forward して Greedy Decode に繋げる。

## 実験結果

### 1. Repeat Lines: 200 (~5,800 tokens)
- **Token Match**: OK (Mismatch 0)
- **Baseline (総経過時間)**: 272.818s ※初期のメモリキャッシュ割り当てに伴う遅延を含む
- **Prefix Reuse (Amortized)**: 26.576s
  - Prefix Prefill: 17.189s
  - Suffix 1~3 (Prefill + Decode): 計 9.387s
- **Amortized Speedup (全体)**: **10.266x**
- **Per-case Speedup (Prefix時間を除外した場合)**: 26x 〜 65x

### 2. Repeat Lines: 500 (~14,500 tokens)
- **Token Match**: OK (Mismatch 0)
- **Baseline (総経過時間)**: 128.738s
  - Case 0: ~43.3s
  - Case 1: ~42.8s
  - Case 2: ~42.7s
- **Prefix Reuse (Amortized)**: 43.842s
  - Prefix Prefill: **41.930s**
  - Case 0 (Suffix Prefill + Decode): **1.190s**
  - Case 1 (Suffix Prefill + Decode): **0.412s**
  - Case 2 (Suffix Prefill + Decode): **0.310s**
- **Amortized Speedup (全体)**: **2.936x**
- **Per-case Speedup (Prefix時間を除外した場合)**: **36x 〜 137x**

## 考察
- Prefix Cache の Snapshot と Restore は完璧に機能し、ベースラインと**完全に一致する Token Output** を出力した。
- 約14.5kトークンの共通コンテキストがある場合、毎回 42 秒かかっていた Prefill が、再利用パスでは 2回目以降 **0.1秒〜0.6秒** の Suffix Prefill だけで完了するようになった。
- **Amortized Speedup が 2.93x** ということは、3つの分岐推論を行った全体時間が約 1/3 に短縮されたことを意味する。分岐数が増えれば増えるほど、この効果はリニアに増大する。

## 制限事項
1. **同一 Prefix が必須**: 入力文字列の Prefix トークン列が厳密に一致している必要がある。
2. **Snapshot / Restore コスト**: 現状は `full_snapshot` によって CPU/RAM 領域に Python オブジェクトとして退避させているため、キャッシュのシリアライズやメモリコピーのオーバーヘッドが存在する。
3. **Multi-user / Multi-session 管理未実装**: 本検証はシングルプロセスの逐次実行を前提とした Probe であり、実システムに組み込むには KV Cache のライフサイクル管理が必要。
4. **Memory 使用量**: MLX のアーキテクチャ上、58,000 トークン (Repeat Lines: 2000) 規模になると約 100GB 以上の割り当てが発生し OOM となる問題は、Prefix Reuse を以てしても解決しない。

## 次の課題
1. **Prefix Cache Manager**
   - キャッシュの永続化、識別 (Hash key)、および自動 Restore を担う専用の Manager コンポーネントの開発。
2. **LRU Cache**
   - 限られたメモリリソースの中で、使用頻度の高い Prefix Snapshot を保持するための LRU 機構の導入。
3. **Session-based Prefix Reuse**
   - ローカルエージェントの実際の「対話履歴 (Conversation)」において、過去ターンの KV Cache を再利用しつつ差分 (新規ターン) のみを Prefill する仕組みの構築。
4. **Template Draft Engine との統合**
   - Prefix Cache の恩恵と Template Draft (Decode 高速化) の恩恵を同時に受けるための統合検証。
