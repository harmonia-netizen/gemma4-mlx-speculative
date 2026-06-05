# Template Draft Engine & Prefix Reuse: Final Summary

## Overall Conclusion
本プロジェクトでは、MLX における Gemma 4 モデルの推論を高速化するため、「予測可能な出力に対する Speculative Decoding (Template Draft Engine)」と「大規模コンテキストに対する Prefix Cache Reuse」という2つのアプローチを検証・統合した。
結果として、**エージェント用途などの「共通の長文履歴を持ち、短く予測可能なコマンドを出力する」シナリオ** において、有望な高速化アーキテクチャ (Template Draft Runtime) を構築できる見込みが得られた。

## What Worked
1. **Template Draft Engine (Target Verification)**
   - 予測可能な出力 (例: 決まった形式のbashコマンド群) をあらかじめ Candidate Registry に登録しておくことで、推論時に小規模モデルを使うことなく直接 Target Model 自身で Verify (forward_many) できる。
   - Mismatch 時の高速な Snapshot Restore および Fallback 機構により、安全性と Token の完全一致 (Token Match) を100%維持しながら、対象ケースで **Decode 速度を約 2.2 倍** に引き上げることに成功した。
   - Confidence に基づく Candidate Gating により、予測困難なケースでのペナルティを排除できた。

2. **Prefix Cache Reuse**
   - 長大な共通プロンプト (Prefix) を持つ場合、毎回の Prefill が全体時間の 98% を支配するボトルネックとなっていた。
   - Prefix のみを一度 Prefill してキャッシュの Snapshot を保存し、各分岐 (Suffix) 推論の前にそれをリストアする機構を実装。
   - これにより、2回目以降の Suffix 推論における Prefill 時間を 数十秒 から 0.1〜0.4秒 に劇的に削減し、14.6k トークン規模の推論において **全体の実効速度を約 2.9倍 (Amortized Elapsed Speedup)** に改善した。

3. **統合ランタイム (Template Draft Runtime)**
   - 上記2つを統合し、Prefix Reuse による Prefill の削減と Template Draft による Decode の短縮を同時に達成できた。
   - 統合による副作用 (微小な Logit の差異による Token Mismatch) は、少なくとも既存の15K/32K検証ではチャンク処理の境界を Baseline と統一することで回避できた。

## What Did Not Work (Limits & Bottlenecks)
- **Small Model Speculative Decoding の見送り**: Gemma 4 向けの適切なアライメントを持つ小規模 Draft Model が不足していること、および推論エンジン側に余分なメモリ/ロード負荷がかかることから、今回の用途では採用を見送った。
- **純粋な Decode 高速化の限界**: Long Context 下では全体の実行時間が Prefill に完全に支配されるため、Template Draft だけを最適化しても `elapsed_sec` にはほぼ寄与しない (Speedup ~1.002x 程度) ことが判明した。これが Prefix Reuse への移行の決定打となった。
- **100K トークンの限界と成功**: 単一バッチでの巨大プレフィルは OOM を引き起こすが、Chunked Prefill を用いることで 100K トークンの target-only prefill は今回条件で通った。さらに、直列の Baseline フル比較は重すぎて未完走であったが、**「Prefix Reuse Path (Snapshot Restore + Suffix Prefill + Template Draft/Fallback)」単独の計測は完走** し、約100Kのprefixロード後にsuffix処理が約2秒台で完了することを確認した。ただし100Kではtemplate候補がmismatchし、fast path acceptは未成立である。
- **OOM の回避限界**: キャッシュ再利用をしても、元の Prefix 自体が物理メモリの許容上限 (例: 120k超のトークン) に達した場合、実行自体が不可能になる制約は残る。そのため、Runtime 側での `safe-token-limit` 等の Long Input Guard の導入が必要不可欠である。

## Final Architecture
以下の3つのコアコンポーネントで構成される Runtime Prototype を実装した。
- `CandidateRegistry`: 予測可能なプロンプトパターンと出力のペアを管理し、推論時に Gating 基準を満たした候補を提案する。
- `PrefixCacheManager`: 任意の Prefix テキストをハッシュ化して管理し、初回は Prefill して Snapshot を保存、2回目以降は瞬時にキャッシュをリストアする。
- `TemplateDraftRuntime`: これらを統合し、Prefix Restore -> Suffix Prefill -> Candidate Proposal -> Target Verify -> Fallback の完全なライフサイクルを実行する。

## Results
- **Decode Speedup**: `exact_pytest_plan` にて約 **2.2x** (単体検証), 統合環境下で約 **1.89x**
- **Long Context Amortized Speedup**: 15k トークン規模の共通 Prefix 複数ケース推論にて、**約 2.9x**
- **Token Match**: 15K/32K のA/B/C検証では、浮動小数点の計算順序差異を補正するためにチャンク境界を揃えることで token match を確認した。100K のA/B/C統合比較は未完走であり、別枠で capacity / restore / reuse path を検証した。

## Next Steps
1. **CandidateRegistry の外部化**: ハードコードされた候補を JSON/YAML 等の外部設定ファイルに分離する。
2. **PrefixCacheManager の LRU 化**: 実メモリ制約に合わせて保持する Snapshot のライフサイクルを管理する機構。
3. **Session 単位の Cache 管理**: エージェントの「対話履歴 (Conversation)」に対し、ターンごとの差分のみを Append/Prefill する Multi-turn Session Manager への拡張。
4. **API / Package 化**: 今回の Prototype を再利用可能な独立パッケージまたはサーバーエンドポイントとして提供する。

- **100K統合Runtimeの注意**: 100Kのcapacity/restoreは今回条件で成立したが、A/B/C統合Runtime比較は未完走であり、100Kでの統合speedupは未確定。
