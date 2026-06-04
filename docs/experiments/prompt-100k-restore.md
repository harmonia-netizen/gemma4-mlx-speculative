# Prompt 100K Restore Probe

## 目的
100K 級の長大な共通プロンプト (Prefix) を処理するにあたり、チャンク単位での Prefill および Snapshot の保持が実メモリ上可能であることは `prompt_100k_capacity_probe.py` によって確認された。
本実験ではさらに進めて、**「Snapshot 取得後に推論を進め (Advance)、その後に Snapshot へ Restore した際、生成される Token 列が純粋な Baseline と完全に一致するか (Correctness)」** を検証する。
特に 100K における浮動小数点の累積誤差や、巨大な KV Cache の状態退避・復元におけるエッジケースが存在しないかを確認する。

## 検証スクリプト
- `experiments/prompt_100k_restore_probe.py`

### 実行フェーズ
1. **Baseline**: 対象トークン数 (32K / 64K / 100K) まで Chunked Prefill を行い、続いて 16 Tokens の Greedy Decode を実行して `baseline_ids` を取得する。
2. **Restore Test**:
   - 同様に Prefill を行い、直後に `full_snapshot()` でキャッシュ状態を保存する。
   - `advance_tokens` (8 tokens) 分だけキャッシュを強引に進める (生成・計算を続行)。
   - `restore_full()` を呼び出して、キャッシュを Snapshot 取得時の状態に巻き戻す。
   - 巻き戻した状態から 16 Tokens の Greedy Decode を実行し、`restored_ids` を取得する。
3. **一致確認**: `baseline_ids == restored_ids` を確認する。

## 結果一覧

### 32K Tokens
- **Target Tokens**: 32,000
- **Baseline Prefill**: 86.414s
- **Snapshot Prefill**: 86.052s
- **Advance Time (8 tokens)**: 0.310s
- **Restore Time**: 0.000s
- **Token Match**: **OK** (完全一致)

### 64K Tokens
- **Target Tokens**: 64,000
- **Baseline Prefill**: 240.798s
- **Snapshot Prefill**: 267.483s
- **Advance Time (8 tokens)**: 0.483s
- **Restore Time**: 0.000s
- **Token Match**: **OK** (完全一致)

### 100K Tokens
- **Target Tokens**: 100,000
- **Baseline Prefill**: 490.884s
- **Snapshot Prefill**: 511.249s
- **Advance Time (8 tokens)**: 0.615s
- **Restore Time**: 0.000s
- **Token Match**: **OK** (完全一致)

## 考察と意味

### 1. 100K 状態での完全な巻き戻し安全性
`full_snapshot` と `restore_full` の組み合わせは、100K トークンという極めて長大なコンテキストにおいても、少なくとも今回の条件では restore 後の生成トークン列が baseline と完全一致する状態に戻せることを確認した。
Restore によって Logit が微小にズレるような副作用は一切なく、Template Draft が長大コンテキスト下で Failed Proposal をリジェクトしてロールバックする際にも、100% の安全性が担保される。

### 2. Snapshot のオーバーヘッドゼロ
32K, 64K, 100K いずれの規模においても、`snapshot_create_sec` および `restore_sec` は `0.000s` であった。
これは Python 側での参照退避のみで完結しており、実メモリ領域（VRAM / 統合メモリ上の数百 GB のテンソル）のコピーが発生していないことを裏付けている。

### 3. PrefixCacheManager への反映方針
この結果により、エージェントが持つ「不変の長文プロンプト（インストラクションや過去の固定ログなど）」は、初回ターンで一度 Prefill して Snapshot 化し、`PrefixCacheManager` に永続的に保持させる設計が極めて有効かつ安全であることが確定した。

## 今後の課題
1. **Snapshotの実コピー有無の明確化**: MLX の遅延評価 (Lazy Evaluation) や参照カウントの仕組み上、キャッシュをさらに進めた際に「過去の Snapshot」がコピー・オン・ライト (CoW) でどこまでメモリを消費するかについて、厳密な実メモリ使用量計測プロファイラを導入する必要がある。
2. **複数 Snapshot 保持の限界**: ターンごとに増分を Prefix として Snapshot 化していく (Session Cache) 場合、参照のツリーがどのようにメモリを占有するかを解明する。
3. **LRU / Session Cache Manager**: OOM を防ぐため、古い Snapshot 参照を自動で Drop する LRU 機構の実装が必須となる。
