---
sidebar_position: 1
title: "01. 概要と全体像"
---

# 01. 概要と全体像

このページでは、論文 **「Advancing regulatory variant effect prediction with AlphaGenome」** の全体像をつかみます。書誌情報、要旨（Abstract）の精読、論文が解こうとした2つのトレードオフ、モデルの貢献、そして以降のページで詳しく見るトピックの見取り図を示します。

:::tip[Evo 2 との違いを先に押さえる]

同じ「生物学 × 配列モデル」でも、[Evo 2](../evo2/01-overview.md) と AlphaGenome は **設計思想がまったく異なります**。Evo 2 は「次の塩基を当てる」自己回帰の **ゲノム言語モデル** で、進化が刻んだ尤度を学びます。一方 AlphaGenome は、DNA 配列を入力すると **実験データ（遺伝子発現・クロマチンの開き具合など）を直接予測する** 教師あり回帰モデル（sequence-to-function model）です。両者の関係は本ページ末尾でまとめます。

:::

## 1. 書誌情報

| 項目 | 内容 |
| --- | --- |
| タイトル | Advancing regulatory variant effect prediction with AlphaGenome |
| 著者（筆頭・equal contribution） | Žiga Avsec, Natasha Latysheva, Jun Cheng, Guido Novati, Kyle R. Taylor, Tom Ward, Clare Bycroft, Lauren Nicolaisen, Eirini Arvaniti, Joshua Pan ほか |
| 監督 | **Demis Hassabis**, **Pushmeet Kohli** |
| 所属 | Google DeepMind（London, UK） |
| 掲載誌 | *Nature* **649**, 1206–1218（2026年1月29日号、オンライン公開 2026年1月28日） |
| 投稿 / 受理 | 受理 2025年5月16日 / 受理 2025年12月4日 |
| DOI | [10.1038/s41586-025-10014-0](https://doi.org/10.1038/s41586-025-10014-0) |
| ライセンス | CC BY 4.0（オープンアクセス） |
| モデル提供 | 非商用 API（[deepmind.google.com/science/alphagenome](https://deepmind.google.com/science/alphagenome)）＋ Python SDK |

:::note[著作権について]

論文本体は CC BY 4.0 ライセンスです。本ドキュメントの図は、論文の図版を転載したものではなく、**内容を理解するために独自に描き起こした概念図** です。論文中の図を指すときは「Fig. 1」「Extended Data Fig. 1」のように参照番号で示します。原図は[論文ページ](https://doi.org/10.1038/s41586-025-10014-0)で確認してください。

:::

## 2. ひとことで言うと

**AlphaGenome は、1 Mb（100 万塩基対）の DNA 配列を入力すると、数千種類の機能ゲノミクス・トラックを最大「単一塩基解像度」で同時に予測する統合モデル** です。

「機能ゲノミクス・トラック（genome track）」とは、DNA の各塩基対に1つの値（実験で測った読み取り深さ・シグナル強度など）を対応づけたデータ形式です。たとえば「この位置はどれくらい RNA に転写されているか」「クロマチンがどれくらい開いているか」といった測定値が、塩基ごとに並んだ折れ線になります。AlphaGenome は **配列だけからこの折れ線を当てる** ことを学びます。

特筆すべきは次の3点です。

- **統合性（unified）** — 遺伝子発現・スプライシング・クロマチン状態・転写因子結合・3次元接触マップなど、従来は専用モデルに分かれていた **11 種類のモダリティを1つのモデルで** 予測。ヒト **5,930** トラック、マウス **1,128** トラックを同時に出力します。
- **長文脈 × 高解像度の両立** — 入力 **1 Mb** の広い文脈を見ながら、出力は最大 **1 bp（単一塩基）解像度**。後述するように、従来モデルはこの両立ができませんでした。
- **変異効果予測の SOTA** — トラック予測 24 タスク中 **22 で**、変異効果予測 26 タスク中 **25 で**、その時点で最強の外部モデルに匹敵または凌駕。

## 3. 要旨（Abstract）の精読

論文の要旨を分解しながら読みます。

> Deep learning models that predict functional genomic measurements from DNA sequences are powerful tools for deciphering the genetic regulatory code.

出発点は、**DNA 配列から機能ゲノミクスの測定値を予測する深層学習モデル**（sequence-to-function model）が、遺伝子制御コードを解読する強力な道具だ、という認識です。

> Existing methods involve a trade-off between input sequence length and prediction resolution, thereby limiting their modality scope and performance.

ここが論文の核心の問題提起です。既存手法は **「入力配列の長さ」と「予測の解像度」がトレードオフ** の関係にあり、扱えるモダリティの幅や性能が制限されてきた、と述べます（詳細は次節）。

> We present AlphaGenome, a unified DNA sequence model, which takes as input 1 Mb of DNA sequence and predicts thousands of functional genomic tracks up to single-base-pair resolution across diverse modalities.

モデルの宣言。キーワードは「**統合（unified）**」「**1 Mb 入力**」「**数千トラック**」「**単一塩基解像度**」「**多様なモダリティ**」。

> Trained on human and mouse genomes, AlphaGenome matches or exceeds the strongest available external models in 25 of 26 evaluations of variant effect prediction.

**ヒトとマウスのゲノム** で学習し、変異効果予測の **26 評価中 25** で最強の外部モデルに匹敵・凌駕した、という結果の要約です（→ [03](./03-track-prediction.md) 以降）。

> The ability of AlphaGenome to simultaneously score variant effects across all modalities accurately recapitulates the mechanisms of clinically relevant variants near the TAL1 oncogene.

**全モダリティを横断して同時にスコアリングできる** という統合性が、白血病に関わる **TAL1 がん遺伝子** 近傍の臨床的に重要な変異の機構を正確に再現した、という応用例です（→ [07](./07-multimodal-ablations.md)）。

> To facilitate broader use, we provide tools for making genome track and variant effect predictions from sequence.

最後に、配列からトラック予測・変異効果予測を行う **ツール（API・SDK）を公開** したことが述べられます。

## 4. 論文が解いた2つのトレードオフ

AlphaGenome の新規性は、従来の sequence-to-function モデルが抱えていた **2つの根本的トレードオフ** を同時に解消した点にあります。

<figure style={{margin: '1.5rem auto', textAlign: 'center', maxWidth: '560px'}}>
  <svg viewBox="0 0 540 300" width="100%" role="img" aria-label="入力配列長と出力解像度のトレードオフ。短入力高解像度モデルと長入力低解像度モデルの間に、AlphaGenome が長入力かつ高解像度の領域を占める">
    <line x1="60" y1="250" x2="510" y2="250" stroke="currentColor" strokeOpacity="0.4" strokeWidth="1.4" />
    <line x1="60" y1="250" x2="60" y2="30" stroke="currentColor" strokeOpacity="0.4" strokeWidth="1.4" />
    <text x="285" y="282" fontSize="11.5" fill="currentColor" textAnchor="middle">入力配列長（長いほど右）→ 遠位の制御要素まで見える</text>
    <text x="28" y="140" fontSize="11.5" fill="currentColor" textAnchor="middle" transform="rotate(-90 28 140)">出力解像度（高いほど上）</text>
    <circle cx="120" cy="70" r="8" fill="#EF4444" fillOpacity="0.7" />
    <text x="120" y="56" fontSize="10" fill="currentColor" textAnchor="middle">SpliceAI / BPNet</text>
    <text x="120" y="92" fontSize="9" fill="currentColor" fillOpacity="0.7" textAnchor="middle">短入力・高解像度</text>
    <circle cx="380" cy="200" r="8" fill="#F59E0B" fillOpacity="0.8" />
    <text x="380" y="186" fontSize="10" fill="currentColor" textAnchor="middle">Enformer / Borzoi</text>
    <text x="392" y="222" fontSize="9" fill="currentColor" fillOpacity="0.7" textAnchor="middle">長入力・低解像度（128/32 bp）</text>
    <rect x="430" y="42" width="76" height="40" rx="8" fill="#10B981" fillOpacity="0.16" stroke="#10B981" strokeWidth="1.8" />
    <text x="468" y="60" fontSize="11" fill="currentColor" textAnchor="middle" fontWeight="700">AlphaGenome</text>
    <text x="468" y="75" fontSize="8.5" fill="currentColor" fillOpacity="0.85" textAnchor="middle">1 Mb × 1 bp</text>
    <line x1="128" y1="74" x2="426" y2="58" stroke="#10B981" strokeOpacity="0.35" strokeWidth="1.2" strokeDasharray="4 3" />
    <line x1="388" y1="196" x2="430" y2="74" stroke="#10B981" strokeOpacity="0.35" strokeWidth="1.2" strokeDasharray="4 3" />
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>第1のトレードオフ。従来は「短入力・高解像度」か「長入力・低解像度」の二択だったが、AlphaGenome は右上（長入力かつ高解像度）を実現する</figcaption>
</figure>

**第1のトレードオフ：入力配列長 ⇄ 予測解像度。** 計算資源の制約から、モデルは「長距離の相互作用を捉える」か「塩基レベルの細かい解像度を出す」かを選ばざるをえませんでした。

- **SpliceAI・BPNet・ProCapNet** … 塩基解像度を出せるが、入力が短い（10 kb 以下）。離れたエンハンサーなどの **遠位制御要素を見逃す**。
- **Enformer・Borzoi** … 長い配列（約 200–500 kb）を扱えるが、出力解像度を **128 bp / 32 bp に粗く** する代償を払う。スプライス部位・転写因子のフットプリント・ポリアデニル化部位といった細かい特徴がぼやける。

**第2のトレードオフ：モダリティの多様性 ⇄ 専門特化。** 多くの SOTA モデルは単一モダリティに特化していました（SpliceAI＝スプライス部位、ChromBPNet＝局所クロマチンアクセシビリティ、Orca＝3次元ゲノム構造）。汎用的なマルチモーダルモデル（Enformer・Borzoi など）もありますが、スプライシングのような特定タスクでは専用モデルに劣ったり、接触マップなど一部モダリティを欠いたりしていました。

AlphaGenome は、**マルチモーダル予測・長文脈・塩基解像度を1つの枠組みに統合** することで、この2つのトレードオフを同時に乗り越えます。

:::note[なぜ非コード変異の解釈が重要なのか]

ヒトで観測される遺伝的変異の **98% 以上は非コード領域**（タンパク質をコードしない領域）にあります。これらはクロマチンの開き具合・エピジェネティック修飾・3次元構造・mRNA 量・スプライシングなど、**多様な分子メカニズム** を介して機能に影響します。実験ですべてを調べるのは不可能なため、配列から影響を予測する計算手法が不可欠です。AlphaGenome は、まさにこの「非コード変異の機能解釈」を主戦場にしています。

:::

## 5. 全体像：AlphaGenome が予測する 11 モダリティ

AlphaGenome は 1 Mb の DNA と「種（ヒト/マウス）」を入力に、11 種類の出力タイプを各々の解像度で予測します（Fig. 1a）。

| 大分類 | 出力タイプ（モダリティ） | 解像度 | 測るもの |
| --- | --- | --- | --- |
| 遺伝子発現 | RNA-seq | 1 bp | 転写量（読み取り深さ） |
|  | CAGE | 1 bp | 転写開始点の活性 |
|  | PRO-cap | 1 bp | 転写開始（nascent RNA） |
| クロマチン状態 | DNase | 1 bp | DNA アクセシビリティ |
|  | ATAC | 1 bp | DNA アクセシビリティ |
|  | ヒストン修飾（ChIP-seq） | 128 bp | エピジェネティック標識 |
|  | 転写因子結合（ChIP-seq） | 128 bp | TF がどこに結合するか |
| スプライシング | スプライス部位 | 1 bp | ドナー/アクセプター部位 |
|  | スプライス部位使用率 | 1 bp | 各部位がどれだけ使われるか |
|  | スプライスジャンクション | 1 bp | どのイントロンが除去されるか（**新規**） |
| 3次元構造 | DNA 接触マップ | 2,048 bp | ゲノム領域間の空間的相互作用 |

:::tip[なぜ 1 Mb なのか]

入力長 1 Mb は、関連する遠位制御領域の大部分を含むように設計されています。論文によれば、検証済みのエンハンサー–遺伝子ペアの **99%（471 中 465）が 1 Mb 以内** に収まります。つまり 1 Mb は「ほとんどの制御関係をカバーできる最小限の文脈長」として選ばれています。

:::

## 6. モデルの貢献（何が新しいのか）

論文の貢献を整理すると、次の5点です。

1. **統合アーキテクチャ** — U-Net 風のバックボーン（畳み込み＋Transformer）で、長文脈と塩基解像度を両立（→ [02](./02-architecture.md)）。
2. **新しいスプライスジャンクション予測** — スプライス部位・使用率に加え、「どのイントロンが除去されるか」を直接予測する仕組みを導入（→ [04](./04-splicing-variants.md)）。
3. **包括的なベンチマーク** — トラック予測 24 タスク（22 で SOTA）、変異効果予測 26 タスク（25 で SOTA）。
4. **広範なアブレーション** — 解像度・配列長・蒸留・モダリティ組合せの寄与を分解（→ [07](./07-multimodal-ablations.md)）。
5. **アクセシブルなツール提供** — 公開 API と Python SDK、ゲノム解釈スイート。

## 7. AlphaGenome と Evo 2 ——2つのパラダイム

本セクションの第1弾 [Evo 2](../evo2/01-overview.md) と AlphaGenome は、しばしば一緒に語られますが、立ち位置が異なります。

| 観点 | Evo 2（ゲノム言語モデル） | AlphaGenome（sequence-to-function） |
| --- | --- | --- |
| 学習の枠組み | 自己教師あり（次の塩基を予測） | 教師あり（実験トラックを回帰予測） |
| 何を出力するか | 配列の尤度・新規配列の生成 | 機能ゲノミクス・トラック（発現・クロマチン等） |
| 学習データ | 全生物ドメインの生 DNA | ヒト/マウスの **実験測定データ**（ENCODE, GTEx ほか） |
| 主な強み | 進化的制約・配列生成・全ドメイン汎用性 | 非コード変異の **機能的影響** を高精度・多モダリティで予測 |
| 文脈長 | 100 万トークン | 1 Mb（100 万塩基） |
| アーキテクチャ | StripedHyena 2（畳み込みハイブリッド） | U-Net ＋ Transformer |

:::info[補足：AlphaGenome は「言語モデル」か？]

厳密には、AlphaGenome は GPT 系のような **自己回帰言語モデルではありません**。次のトークンを生成するのではなく、配列全体から各位置の機能値を回帰で当てる教師ありモデルです。それでも本セクションで扱うのは、(1) 長距離依存の捕捉に **Transformer ブロック** を中核として使い、(2) 多様なタスクを単一の汎用表現でこなす **基盤モデル的アプローチ** をとり、(3) LLM で発展した知見（蒸留・スケーリング・表現学習）が色濃く活きているためです。論文の Discussion でも、Evo 2 のような DNA 言語モデルと **相補的に組み合わせる** 将来像が語られています（→ [07](./07-multimodal-ablations.md)）。

:::

## 8. このあとの章

| ページ | 論文セクション | 主な内容 |
| --- | --- | --- |
| [02. アーキテクチャと学習](./02-architecture.md) | Unifying DNA sequence-to-function model | U-Net＋Transformer・配列並列・事前学習＋蒸留 |
| [03. トラック予測性能](./03-track-prediction.md) | Performance overview / Improved track prediction | 24 評価・Pearson r・スプライシング/接触マップ |
| [04. スプライシング変異](./04-splicing-variants.md) | Improved splicing variant predictions | スプライスジャンクション・複合スコアラー・sQTL・ClinVar |
| [05. 発現・遠位制御の変異](./05-expression-variants.md) | gene expression / enhancer–gene / polyadenylation | eQTL・GWAS・エンハンサー連結・paQTL |
| [06. クロマチン変異と MPRA](./06-chromatin-variants.md) | chromatin accessibility, DNase, binding QTLs | caQTL/dsQTL/bQTL・ISM モチーフ・CAGI5 |
| [07. 多モダリティ統合・アブレーション・考察](./07-multimodal-ablations.md) | Multimodal view / ablations / Discussion | TAL1・形質変異・設計選択の寄与・限界と展望 |

それでは、AlphaGenome を支える土台——[アーキテクチャと学習](./02-architecture.md)から見ていきましょう。
