---
sidebar_position: 6
title: "06. クロマチン設計と考察"
---

# 06. クロマチン設計と考察

最終ページです。Evo 2 の **設計（design）** 能力——ただ生成するのではなく、**狙った機能をもつ配列を作る** 手法を見たあと、論文の **Discussion**（オープンソース・安全性・展望）をまとめ、セクション全体を締めくくります。

## 1. 生成から「設計」へ：推論時ガイダンス

[05. ゲノムスケール生成](./05-generation.md)の生成は、学習した分布から自然に近い配列をサンプリングするものでした。しかし「**特定の機能をもつ配列がほしい**」場合、自然なサンプリングだけでは狙った性質に当たりません。

そこで使うのが **推論時ガイダンス（inference-time guidance）** です。生成モデルとは別の **スコア関数（報酬関数）** で生成を誘導し、目的の性質をもつ出力に寄せます。

:::tip[LLM とのつながり：test-time compute と同じ発想]

「推論時に計算を増やし、外部のスコア関数で出力を選別して品質を上げる」というのは、コード生成・アルゴリズム設計・数学推論で使われる **test-time compute**（推論時計算）そのものです（論文も AlphaCode や "large language monkeys" を引用）。OpenAI o1 系の推論時スケーリングや、best-of-N サンプリング、報酬モデルによる選別と地続きの考え方を、Evo 2 は配列設計に持ち込んでいます。

:::

題材は **クロマチンアクセシビリティ（chromatin accessibility）** の設計です。

:::note[クロマチンアクセシビリティとは]

ゲノム DNA は普段ヒストンに巻き取られています。転写因子などが結合できる「**開いた（accessible）**」領域では遺伝子発現が起きやすく、「閉じた」領域では起きにくい。この開き具合は **ATAC-seq** で測定でき、ゲノム配列上の「**ピーク**」として可視化されます。Evo 2 で、このピークの **位置と長さを狙って設計** しよう、というのがこのタスクです。

:::

## 2. ビームサーチによる設計

ここで鍵になるのが **Enformer** と **Borzoi** です。これらは DNA 配列からクロマチンアクセシビリティを予測する **配列→機能（sequence-to-function）モデル** で、ヒトとマウスの細胞型ごとに予測できます。ただし **生成モデルではなく**、自然ゲノムだけで訓練されています。

Evo 2 自身はクロマチンアクセシビリティを明示的に学習していません。しかし、**Enformer と Borzoi のアンサンブルをスコア関数** に使えば、推論時ガイダンスで「開き具合」を条件づけられます。論文は次の **ビームサーチ（beam search）** を採用しました（Fig. 6b）。

1. Evo 2 で **128 bp** のDNAチャンクを複数候補生成する。
2. 各候補について、Enformer と Borzoi で予測したアクセシビリティが **目標パターンにどれだけ合うか** を採点する。
3. 上位の候補だけを残し（accept）、残りは捨てる（reject）。
4. 残した候補をプロンプトに足して、次の 128 bp の生成に進む。

フルの長い配列を一度に作って評価するのではなく、**128 bp ごとに評価して有望なものだけ伸ばす** のがポイントです。

<figure style={{margin: '1.5rem auto', textAlign: 'center', maxWidth: '600px'}}>
  <svg viewBox="0 0 580 250" width="100%" role="img" aria-label="ビームサーチによる設計。128bp候補を生成し、Enformer/Borzoiで採点して上位を選び伸ばす">
    <rect x="230" y="16" width="120" height="30" rx="5" fill="#3B82F6" fillOpacity="0.1" stroke="#3B82F6" strokeWidth="1.4" />
    <text x="290" y="36" fontSize="11" fill="currentColor" textAnchor="middle">プロンプト</text>
    <line x1="290" y1="46" x2="70" y2="74" stroke="currentColor" strokeOpacity="0.35" strokeWidth="1.1" />
    <line x1="290" y1="46" x2="215" y2="74" stroke="currentColor" strokeOpacity="0.35" strokeWidth="1.1" />
    <line x1="290" y1="46" x2="365" y2="74" stroke="currentColor" strokeOpacity="0.35" strokeWidth="1.1" />
    <line x1="290" y1="46" x2="510" y2="74" stroke="currentColor" strokeOpacity="0.35" strokeWidth="1.1" />
    <rect x="15" y="76" width="110" height="40" rx="5" fill="#10B981" fillOpacity="0.12" stroke="#10B981" strokeWidth="1.4" />
    <text x="70" y="92" fontSize="9.5" fill="currentColor" textAnchor="middle">候補 128 bp</text>
    <text x="70" y="108" fontSize="10.5" fill="#10B981" textAnchor="middle" fontWeight="600">Accept ✓</text>
    <rect x="160" y="76" width="110" height="40" rx="5" fill="currentColor" fillOpacity="0.05" stroke="currentColor" strokeOpacity="0.4" strokeWidth="1.3" />
    <text x="215" y="92" fontSize="9.5" fill="currentColor" fillOpacity="0.7" textAnchor="middle">候補 128 bp</text>
    <text x="215" y="108" fontSize="10.5" fill="#EF4444" textAnchor="middle">Reject ✗</text>
    <rect x="310" y="76" width="110" height="40" rx="5" fill="#10B981" fillOpacity="0.12" stroke="#10B981" strokeWidth="1.4" />
    <text x="365" y="92" fontSize="9.5" fill="currentColor" textAnchor="middle">候補 128 bp</text>
    <text x="365" y="108" fontSize="10.5" fill="#10B981" textAnchor="middle" fontWeight="600">Accept ✓</text>
    <rect x="455" y="76" width="110" height="40" rx="5" fill="currentColor" fillOpacity="0.05" stroke="currentColor" strokeOpacity="0.4" strokeWidth="1.3" />
    <text x="510" y="92" fontSize="9.5" fill="currentColor" fillOpacity="0.7" textAnchor="middle">候補 128 bp</text>
    <text x="510" y="108" fontSize="10.5" fill="#EF4444" textAnchor="middle">Reject ✗</text>
    <line x1="70" y1="116" x2="220" y2="150" stroke="currentColor" strokeOpacity="0.35" strokeWidth="1.1" />
    <line x1="365" y1="116" x2="360" y2="150" stroke="currentColor" strokeOpacity="0.35" strokeWidth="1.1" />
    <rect x="150" y="152" width="280" height="40" rx="5" fill="#EF4444" fillOpacity="0.08" stroke="#EF4444" strokeWidth="1.4" />
    <text x="290" y="168" fontSize="11" fill="currentColor" textAnchor="middle" fontWeight="600">Enformer + Borzoi で採点</text>
    <text x="290" y="183" fontSize="9.5" fill="currentColor" fillOpacity="0.85" textAnchor="middle">目標のアクセシビリティ・パターンと照合</text>
    <text x="290" y="222" fontSize="10.5" fill="currentColor" fillOpacity="0.8" textAnchor="middle">上位を選んでプロンプトに追加 → 128 bp ごとに繰り返す（ビームサーチ）</text>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>ビームサーチによる設計。Evo 2 が候補を提案し、外部の予測器が採点して上位を伸ばす（論文 Fig. 6b に対応）</figcaption>
</figure>

## 3. 推論時スケーリング：計算を増やすほど良くなる

自然言語の結果と同様に、論文は **ビーム幅を広げる（＝推論時の計算を増やす）ほど設計品質が上がる** という **log-linear な関係** を観測しました（Fig. 6c）。具体的には、各ステップで **30 個以上のチャンクを生成し上位 2 個を選ぶ** だけで、最終設計の AUROC が **0.9 超** に達します。

<figure style={{margin: '1.5rem auto', textAlign: 'center', maxWidth: '360px'}}>
  <svg viewBox="0 0 300 180" width="100%" role="img" aria-label="推論時スケーリング。計算量を増やすほど設計成功度AUROCが対数線形に向上">
    <line x1="40" y1="150" x2="280" y2="150" stroke="currentColor" strokeOpacity="0.3" strokeWidth="1" />
    <line x1="46" y1="20" x2="46" y2="156" stroke="currentColor" strokeOpacity="0.3" strokeWidth="1" />
    <text x="270" y="166" fontSize="9" fill="currentColor" fillOpacity="0.7" textAnchor="end">推論時コスト（log）</text>
    <text x="14" y="28" fontSize="9" fill="currentColor" fillOpacity="0.7">AUROC</text>
    <line x1="46" y1="58" x2="280" y2="58" stroke="currentColor" strokeOpacity="0.3" strokeWidth="1" strokeDasharray="4 3" />
    <text x="30" y="61" fontSize="8.5" fill="currentColor" fillOpacity="0.6">0.9</text>
    <polyline points="56,138 90,120 124,102 158,84 192,70 226,60 260,53" fill="none" stroke="#3B82F6" strokeWidth="2.4" />
    <circle cx="192" cy="70" r="3.4" fill="#3B82F6" />
    <text x="150" y="118" fontSize="9.5" fill="#3B82F6">計算 ↑ → 品質 ↑</text>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>推論時スケーリング。ビーム幅（計算量）を増やすほど設計成功度が対数線形に向上する</figcaption>
</figure>

## 4. エピゲノムにモールス信号を書く

設計能力を示すため、研究チームは **ピークの幅と位置でモールス信号** を書きました。**狭いピーク＝トン（dot）、広いピーク＝ツー（dash）、閉じた領域＝空白** です。書いたメッセージは、

- **"LO"** — インターネットで最初に送信されたメッセージ（"LOGIN" を打とうとしてクラッシュ）であり、叙事詩『神仙女王（The Faerie Queene）』の最初の単語。
- **"ARC"** — この研究が行われた研究所（Arc Institute）の名。
- **"EVO2"** — モデル自身の名。

これらは机上の予測にとどまりません。**設計した DNA を合成・組み立て → マウス胚性幹細胞（mESC）にサイト特異的に組み込み → ATAC-seq でクロマチンアクセシビリティを測定** という実験で検証されました（Fig. 6e–h）。結果、予測と実測のパターンはよく一致し、**実験での AUROC は 0.92〜0.95**。単純なピークパターンでも AUROC > 0.89 を達成しています。

## 5. なぜ Evo 2 を提案分布に使うのか

「生成は Evo 2 でなくランダムでもいいのでは？」という疑問に、論文は明快に答えます。Evo 2 を、一様ランダムや bigram（2塩基統計）の提案に置き換えると、**品質が劣化** しました（Extended Data Fig. 10b–e）。

- Evo 2 はネイティブ配列を文脈に与えると **自然な dinucleotide 頻度** を生成し、Enformer/Borzoi のアンサンブル予測が **コンセンサス（一致）** する。
- 一様・bigram 提案は **敵対的サンプル（adversarial sample）** を生みやすく、予測器が誤誘導されやすい。

さらに、Evo 2 の設計はピーク領域で **転写因子モチーフ密度が有意に高く**（$P = 3.6\times10^{-7}$）、mESC で発現する転写因子に **有意に富む**（$P = 2.0\times10^{-4}$）——明示的にそう条件づけていないのにです。**強力な生成モデルを提案分布に使うこと自体が、設計の質を高めている** わけです。

## 6. ヒト細胞株への展開と汎用性

同じ手法を **ヒト細胞株（HEK293T・K562）** の 1〜4 kb 設計にも適用しました。アクセシビリティの強弱を1配列内で変える設計では、**36 個中 33 個（92%）が AUROC > 0.8**。一方、2つの細胞型で開き具合を変える **細胞型特異的** な設計はより難しく、2 倍以上の差が出たのは 24 個中 4 個（17%）でした（Fig. 6i–k）。差の出た設計では、ピーク内に K562 発現転写因子のモチーフが有意に富んでいました。

そしてこの枠組みは **クロマチンに限りません**。Fig. 6l が示すように、**有能な予測器が存在する任意の機能** で Evo 2 をガイドできます（タンパク質構造、エピゲノム状態など）。「**強力な生成モデル ＋ 正確なスコア関数**」という組み合わせは、生物設計の汎用レシピになります。

## 7. 設計の追加検証（Extended Data Fig. 10–11）

本編の設計結果を、論文はさらに細かく検証しています。

**提案分布の比較（Extended Data Fig. 10c）** — 推論時スケーリングを、提案分布を変えて比べています。**一様ランダム**は Evo 2 に一貫して劣り、**bigram**（2塩基統計）は初期こそ良いがビーム幅を増やすと頭打ち。**Evo 2 が最初に AUROC > 0.95** に到達し、ここを超えると設計が質的に明確に成功します。

**アンサンブル一致度 ICC（Extended Data Fig. 10d）** — Enformer と4つの Borzoi トラックの予測一致度を **級内相関係数（ICC）** で測ると、Evo 2 生成配列は **最低のビーム幅でも ICC ≈ 0.95** と高い。uniform/bigram はずっと低い。アンサンブルの不一致は予測の不確実性（≒敵対的入力）に対応する、と論文は解釈します。

**dinucleotide 頻度（Extended Data Fig. 10b）** — Evo 2 はマウスゲノムを文脈に与えると **自然な2塩基頻度** を生成します（明示的に強制していないのに）。uniform/bigram はベースライン頻度から逸脱します。

**細胞型特異性の詳細（Extended Data Fig. 11）** — K562 で開き HEK293T で閉じる設計 24 個のうち、**2倍以上の差が 4 個、3倍以上が 1 個**（4–17% の成功率）。設計は5カテゴリ（両細胞で開く／閉じる、K562 の前半／後半にピーク、4 kb 複数ピーク）に整理され、差の出た設計では **K562 で発現する転写因子のモチーフが有意に富んで** いました。

## 8. Discussion：オープンソース・安全性・展望

論文の考察を整理します。

### 徹底したオープンソース化

Evo 2 は **オープンソースモデルとして最大級** で、次をすべて公開しています。

| リソース | 内容・リンク |
| --- | --- |
| モデルパラメータ | `evo2_40b` / `evo2_7b` / 各 base / `evo2_1b_base`（[Hugging Face](https://huggingface.co/arcinstitute/evo2_40b)） |
| 学習コード | savanna（[github](https://github.com/zymrael/savanna)） |
| 推論コード | vortex（[github](https://github.com/zymrael/vortex)） |
| トップレベル | [github.com/arcinstitute/evo2](https://github.com/arcinstitute/evo2) |
| データセット | OpenGenome2（[Hugging Face](https://huggingface.co/datasets/arcinstitute/opengenome2)） |
| Web ツール | [Evo Designer](https://arcinstitute.org/tools/evo/evo-designer) / [Evo Mech Interp Visualizer](https://arcinstitute.org/tools/evo/evo-mech-interp) |
| SAE | Goodfire/Evo-2-Layer-26-Mixed |
| NVIDIA | Evo 2 NIM / BioNeMo 版 |

性能は **40B（1M 文脈）が最良**、**7B（1M）も軽量推論に有用**。実験的な 1B short-context 版は性能が弱いため非推奨とされています。

### 安全性・セキュリティ・倫理

新しいバイオテクノロジーには相応のリスクがあります。チームは **Responsible AI × Biodesign** のコミットメントに沿い、公開前にリスクを評価・緩和しました。

- **データ除外** — 真核（ヒト含む）に感染するウイルスを学習データから除外し、危険なヒト病原体ウイルスの操作・設計能力を持たせない。
- **レッドチーミング** — ウイルスタンパク質を直接引き出そうとする試験で、生成が **ほぼランダム** であることを確認。
- **集団バイアスの緩和** — 変異効果予測における **ancestry bias（祖先集団による偏り）** を、集団に依存しない設計で他手法並みに抑制。
- **注意点** — タスク特化の **追加学習（post-training）でこの安全策は回避され得る** ため、慎重な取り扱いが必要、と明記されています。

これらの安全性評価の詳細（ウイルスの perplexity・レッドチーミング・集団バイアス）は、付録 [07. バイオセーフティと責任ある公開](./07-biosafety.md) で掘り下げています。

:::warning[オープン化とデュアルユース]

完全オープンなモデルは、研究の再現・検証・発展を加速する一方、**想定外の使われ方（事故や悪用）** のリスクも伴います。Evo 2 の取り組みは、**「予防」と「アクセス」を両立** させようとする、生物基盤モデルとしては最も包括的な評価のひとつと位置づけられています。今後さらに評価とリスク緩和手法の拡充が必要、というのが著者らの立場です。

:::

### 今後の展望

- 集団規模の遺伝的変異や、配列→機能の実験データを統合し、さらに広いタスクへ。
- 機構的解釈可能性を **ゲノムマイニング**（新規生物要素の発見）に活用。
- 実験フィードバックによる **教師ありファインチューニング・強化学習** で生成品質を向上。
- ゲノム配列と他モダリティの統合により、**健康と疾患における複雑な表現型のシミュレーション** へ。

## 9. このセクションのまとめ

Evo 2 を題材に、ゲノムという「生命の言語」へ LLM の技術がどう適用されるかを見てきました。

- **アーキテクチャ** — $O(n^2)$ の壁を越える StripedHyena 2 で 100 万塩基の長文脈を実現（[02](./02-architecture.md)）。
- **予測** — 尤度の変化だけで、BRCA1 を含む変異の影響をゼロショット予測（[03](./03-variant-prediction.md)）。
- **解釈** — SAE で、生物学的概念に対応する単義的特徴を抽出（[04](./04-interpretability.md)）。
- **生成** — 自己回帰でゲノム規模の配列を生成（[05](./05-generation.md)）。
- **設計** — 推論時ガイダンスで狙った機能をもつ配列を設計し、実験で検証（本ページ）。

自己回帰・Transformer 代替アーキテクチャ・スパースオートエンコーダ・推論時スケーリング——[基礎知識](../../foundations/chapter1.md)や[LLM をゼロから作る](../../llm-from-scratch/chapter1.md)で学ぶ技術が、そのまま **生命科学の最前線** で生きていることが伝わったのではないでしょうか。

:::tip[このセクションは今後も拡張予定]

「生物学における LLM の応用事例」は、Evo 2 を第1弾として、今後さらに事例を追加していく予定です。タンパク質言語モデル（ESM 系）や、構造予測（AlphaFold 系）など、LLM の発想が広がる領域は数多くあります。

:::

最後に、原論文をぜひ参照してください：**Brixi, Durrant, Ku et al., "Genome modelling and design across all domains of life with Evo 2", *Nature* 652, 1349–1361 (2026).** [doi:10.1038/s41586-026-10176-5](https://doi.org/10.1038/s41586-026-10176-5)
