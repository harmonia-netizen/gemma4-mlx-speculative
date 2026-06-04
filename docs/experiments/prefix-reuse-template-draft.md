# Prefix Reuse Template Draft Integration 実験録

## 目的
これまでの実験において、2つの独立した高速化アプローチを実証した。
1. **Template Draft Engine**: 予測可能な出力に対してデコードフェーズを約 2.2 倍に高速化するアプローチ (ただし大規模コンテキストでは Prefill 支配により全体時間の短縮幅が縮小)。
2. **Prefix Cache Reuse**: 共通の長文コンテキストの Prefill をキャッシュし、復元することで 2 回目以降の応答時間を劇的に短縮するアプローチ。

本実験では、この2つを統合し、**長文の Prefix Prefill キャッシュを再利用しながら、Suffix (新規指示) の処理時に Template Draft を適用する (Prefix Reuse + Template Draft)** ことで、Prefill 削減と Decode 高速化の「合成効果」が得られるかを検証する。

## 実験 (Probe) 設計
- **スクリプト**: `experiments/prefix_reuse_template_draft_probe.py`
- **コンテキスト**: ダミーの長文ログ行 (`repeat-lines`) + 短い指示 (`suffix`)。
- **検証ケース**:
  1. `exact_pytest_plan`: Template Candidate が完全に一致するケース (Decode高速化の期待値大)。
  2. `medium_pytest_plan`: Candidate Gating により候補が排除され、安全に Greedy にフォールバックするケース。
  3. `git_status`: 同様に短いコマンド出力のケース。
- **比較対象**:
  - **A. Baseline Full Greedy**: 毎ケース、Prefix + Suffix の完全な文字列をゼロから Prefill して標準デコードする。
  - **B. Prefix Reuse Greedy**: Prefix を一度だけ Prefill してスナップショットを取り、各ケースでそれをリストアし、標準デコードする。
  - **C. Prefix Reuse Template Draft**: Prefix スナップショットをリストアし、Suffix を処理したのち、Template Draft (Speculative Decoding) を適用する。

※ 浮動小数点の演算順序 (Chunking) 差異による微細な Logit のズレで Argmax が逆転する事象を防ぎ、数学的な完全一致を保証するため、Baseline の Prefill も内部的に Prefix と Suffix で分割して処理する仕組みを取り入れた。これにより全ケースで Token Match を 100% 維持する。

## 実験結果

### 1. Repeat Lines: 200 (~6,200 tokens)
- **Token Match**: OK (A vs B vs C 完全一致)
- **Amortized Total Elapsed**:
  - A. Baseline: 43.275s
  - B. Prefix Reuse: 15.813s
  - **C. Prefix Reuse + Draft: 15.584s**
- **Speedups**:
  - C vs A (全体 Amortized 速度向上): **2.777x**
  - C vs B (Decode 速度向上 `exact_pytest_plan`): **1.832x**
  - **Accepted**: `exact_pytest_plan` にて 17/17 (100%)

### 2. Repeat Lines: 500 (~15,500 tokens)
- **Token Match**: OK (A vs B vs C 完全一致)
- **Amortized Total Elapsed**:
  - A. Baseline: 140.658s
  - B. Prefix Reuse: 48.417s
  - **C. Prefix Reuse + Draft: 48.209s**
- **Speedups**:
  - C vs A (全体 Amortized 速度向上): **2.918x**
  - C vs B (Decode 速度向上 `exact_pytest_plan`): **1.894x**
  - **Accepted**: `exact_pytest_plan` にて 17/17 (100%)

## 考察と合成効果
- **Prefill の削減**: B および C 方式により、毎回約 46 秒かかっていた Prefill が 2回目以降 0.2〜0.4 秒に短縮された。
- **Decode の高速化**: C 方式ではさらに、`exact_pytest_plan` のような予測可能なケースにおいて Decode 自体の速度が B 方式に対して約 1.89倍 に高速化された。
- **合成効果の達成**: 大規模コンテキスト下であっても、Prefix 再利用による Time-to-First-Token の劇的な短縮と、Template Draft による Time-to-Last-Token の短縮が **矛盾なく共存し機能する** ことが証明された。
- **安全性**: `medium_pytest_plan` や `git_status` のような候補不一致のケースでは Gating 機能によりオーバーヘッドゼロで Greedy 同等の処理が行われ、全体的な Token Match の完全性が保証されている。

## 制限事項
- 複数ターンのチャットセッション等では、毎ターン異なる Prefix が生成されるため、単純な単一 Prefix の使い回しではなく、セッションごとの「KV Cache Append」管理が必要となる。
- `exact_pytest_plan` で Decode は 1.89倍に向上したが、絶対的な Decode 時間が 0.33秒程度と短いため、全体の 48秒 (Amortized Elapsed) に与えるインパクトは小さい。さらに長大な Output を生成するユースケースでより恩恵が可視化される。

## 次の課題 (Next Steps)
- **Stateful Agent Engine の開発**: これまで構築した「Template Draft 機構」と「Cache Snapshot & Restore 機構」をシームレスに扱う、Agent 用の Python 統合クラス/モジュールの作成。
- **外部 Candidate Registry**: 現在ハードコードされている Template のリストを、JSON や YAML などの外部定義から動的に読み込めるようにする。
- **メモリ管理の洗練**: OOM を回避しつつ、過去数ターンのキャッシュを効率的に保持する Session Manager の実装。
