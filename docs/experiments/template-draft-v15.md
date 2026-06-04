# Template Draft v15 実験録

## 目的
v14 で分離された Candidate Provider をさらに実用化に向けて構造化し、推論の「安全かつ効率的」な楽観的推論（Speculative Decoding）をシステムレベルで整理する。
具体的には、候補をただの文字列ではなく構造化データ (`Candidate` dataclass) として扱い、`confidence` や `min_tokens` に基づく **Gating** 機構を導入した。

## 変更点 (v14との差異)
1. **Candidate Dataclass の導入**
   - 候補生成を `Candidate(name, text, confidence, min_tokens, tags, reason)` 構造体として定義し、候補の識別名やメタデータを持たせた。
2. **Confidence Gating の追加**
   - 候補が生成されても、`confidence < 0.8` の場合は一律に Reject するよう実装した。
   - トークン数が `max(template_min_tokens, c.min_tokens)` を満たさない場合も Reject する安全装置を追加した。
3. **候補のソートと選択**
   - 複数の候補が適格と判断された場合、`(confidence降順, token_count降順, name昇順)` にソートして最も確信度とトークン数が大きい候補を自動選択するようにした。

## 結果
### exact_pytest_plan の速度維持
- **Token Match:** OK
- **Accepted:** 100%
- **Speedup:** 約2.20x〜2.26x
- v14同等の高い高速化率を維持できている。

### medium_pytest_plan の候補抑制
- `medium_pytest_plan`（不完全なBashブロックが途中まで合致するが外れるケース）においては、`draft_candidates` で最初から候補として挙げない（gated out）設計としたため、
- 不要な Draft 評価や Snapshot Restore によるオーバーヘッドがなくなり、Greedy 同等の速度 (Speedup: ~1.01x) で実行されることを確認した。

## 考察とデフォルトのLow-Confidence棄却理由
- Speculative Decoding（特に v13以降の `forward_many` + `restore_full` 方式）では、一度 Mismatch が起きるとキャッシュ巻き戻しのペナルティが発生する。
- したがって、「当たるか分からない」Low-confidence 候補（例：75%）を無作為に投入するより、確実なパターン（Confidence 90% 以上など）に限定して Fast Path を回す方が、総合的な推論スループット（Expected Speedup）は遥かに高くなる。
- このため、今回の実装では `confidence < 0.8` の候補をデフォルトで採用しない設定としている。

## 次の課題 (v16以降)
- **プロンプト解析による Gating/Confidence 計算の動的化**
  - 現在は文字列一致に基づく固定の Confidence を割り当てているが、これを軽量な Router/Classifier で算出し、より複雑なプロンプトでも適切に Gating できるようにする。
- **実運用環境の構築**
  - エージェントのログや過去の対話履歴から「実環境でよく出現するパターン」を抽出し、事前テンプレート辞書に自動追加するパイプラインの構築。
