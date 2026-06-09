---
sidebar_position: 1
title: "01. 概要と全体像"
---

# 01. 概要と全体像

このページでは、論文 **「Genome modelling and design across all domains of life with Evo 2」** の全体像をつかみます。書誌情報、要旨（Abstract）の精読、モデルの貢献、そして以降のページで詳しく見る6つのトピックの見取り図を示します。

## 1. 書誌情報

| 項目 | 内容 |
| --- | --- |
| タイトル | Genome modelling and design across all domains of life with Evo 2 |
| 著者（筆頭・equal contribution） | Garyk Brixi, Matthew G. Durrant, Jerome Ku, Mohsen Naghipourfar, Michael Poli, Gwanggyu Sun ほか |
| 監督（jointly supervised） | Dave P. Burke, Hani Goodarzi, **Patrick D. Hsu**, **Brian L. Hie** |
| 主な所属 | Arc Institute / Stanford University / UC Berkeley / NVIDIA / Liquid AI / Goodfire |
| 掲載誌 | *Nature* **652**, 1349–1361（2026年4月30日号、オンライン公開 2026年3月4日） |
| DOI | [10.1038/s41586-026-10176-5](https://doi.org/10.1038/s41586-026-10176-5) |
| プレプリント | bioRxiv [10.1101/2025.02.18.638918](https://doi.org/10.1101/2025.02.18.638918) |
| ライセンス | CC BY-NC-ND 4.0（完全オープン：モデル・コード・データを公開） |

:::note[著作権について]

論文本体は CC BY-NC-ND 4.0 ライセンスです。本ドキュメントの図は、論文の図版を転載したものではなく、**内容を理解するために独自に描き起こした概念図** です。論文中の図を指すときは「Fig. 1」のように参照番号で示します。原図は[論文ページ](https://doi.org/10.1038/s41586-026-10176-5)で確認してください。

:::

## 2. ひとことで言うと

**Evo 2 は、全生物の DNA を学習した「ゲノムの基盤モデル（biological foundation model）」** です。自然言語の GPT 系モデルが大量のテキストから言葉を学ぶように、Evo 2 は **9.3 兆塩基対** の DNA から「生命の言語」を学び、追加学習なしで多様な予測・生成タスクをこなします。

特筆すべきは次の3点です。

- **スケール** — 最大 **400 億パラメータ（40B）**、**100 万トークン** の文脈長、**単一ヌクレオチド解像度**（1 塩基＝1 トークン）。
- **網羅性** — 細菌・古細菌・真核生物・バクテリオファージという **生命のすべてのドメイン** を1つのモデルでカバー。
- **汎用性** — タスクごとのファインチューニングなしに、変異の影響予測からゲノム規模の配列生成までを **ゼロショット** でこなす generalist モデル。

## 3. 要旨（Abstract）の精読

論文の要旨を分解しながら読みます。

> All of life encodes information with DNA.（すべての生命は DNA で情報をコードしている）

出発点はシンプルな事実です。あらゆる生物の情報は DNA という共通の「記号列」で書かれている——だからこそ1つのモデルで全生物を扱う動機が生まれます。

> ……we still lack sufficient understanding of the immense complexity encoded by genomes to predict the effects of many classes of genomic changes or to intelligently compose new biological systems.

ゲノム編集や合成の技術は進んだものの、**「この変異がどんな影響を及ぼすか」を予測したり、「新しい生物システムを設計」したりするには、ゲノムの複雑さの理解が足りない**、という課題設定です。Evo 2 はこの2つ（**予測** と **設計**）を正面から狙います。

> Here we introduce Evo 2, a biological foundation model trained on 9 trillion DNA base pairs from a highly curated genomic atlas spanning all domains of life to have a 1 million token context window with single-nucleotide resolution.

モデルの核となる宣言です。キーワードは「**9 兆塩基対**」「**全生物ドメインを網羅したキュレーション済みゲノムアトラス**」「**100 万トークン文脈**」「**単一ヌクレオチド解像度**」。

> Evo 2 learns to accurately predict the functional impacts of genetic variation—from noncoding pathogenic mutations to clinically significant BRCA1 variants—without task-specific fine-tuning.

**予測能力**。非コード領域の病原性変異から、乳がん・卵巣がんに関わる **BRCA1** の臨床的に重要な変異まで、**タスク特化のファインチューニングなし**で当てる、と主張しています（→ [03. 変異効果予測](./03-variant-prediction.md)）。

> Mechanistic interpretability analyses reveal that Evo 2 learns representations associated with biological features, including exon–intron boundaries, transcription factor binding sites, protein structural elements and prophage genomic regions.

**解釈可能性**。エクソン–イントロン境界、転写因子結合部位、タンパク質の構造要素、プロファージ領域といった **生物学的特徴に対応する内部表現** を、モデルが自発的に学んでいた、という発見です（→ [04. 機構的解釈可能性](./04-interpretability.md)）。

> The generative abilities of Evo 2 produce mitochondrial, prokaryotic and eukaryotic sequences at genome scale ……Evo 2 also generates experimentally validated chromatin accessibility patterns when guided by predictive models and inference-time search.

**生成・設計能力**。ミトコンドリア・原核・真核の配列を **ゲノム規模** で生成し（→ [05. ゲノムスケール生成](./05-generation.md)）、さらに予測モデルと推論時探索で誘導すると、**実験的に検証されたクロマチンアクセシビリティのパターン** を設計できる（→ [06. クロマチン設計](./06-chromatin-design.md)）。

> We have made Evo 2 fully open, including model parameters, training code, inference code and the OpenGenome2 dataset……

最後に、**モデルパラメータ・学習コード・推論コード・OpenGenome2 データセットをすべて公開** したことが述べられます。これは大規模モデルとしては異例の徹底した開放性です。

## 4. モデルの貢献（何が新しいのか）

論文の導入部は、Evo 2 の新規性を **「タスク特化の最適化よりも generalist な能力を重視した」** 点に置き、それを支える5つの技術的前進を挙げています。

1. **データキュレーション** — 真核ゲノムを含む全ドメインを網羅した OpenGenome2 の構築
2. **モデルアーキテクチャ** — 長文脈を効率的に扱う StripedHyena 2
3. **大規模事前学習** — 9.3 兆トークン・100 万文脈までの段階的学習
4. **高度な解釈可能性手法** — スパースオートエンコーダ（SAE）による特徴抽出
5. **推論時の予測・生成手法** — inference-time guidance による設計

:::tip[「generalist 志向」は LLM のトレンドと同じ]

タスクごとに専用モデルを作るのではなく、**1つの大きなモデルを多様なタスクにゼロショットで使い回す** という思想は、まさに GPT 以降の自然言語 LLM がたどった道です。Evo 2 はその哲学をゲノムに持ち込んだ、と理解すると位置づけが明確になります。

:::

## 5. 全体像：Evo 2 の4つの能力

Evo 2 は、セントラルドグマ（分子）からゲノムまでのスケールを横断し、大きく **予測・解釈・生成・設計** の4方向に展開します（論文 Fig. 1a に対応）。

<figure style={{margin: '1.5rem auto', textAlign: 'center', maxWidth: '620px'}}>
  <svg viewBox="0 0 600 340" width="100%" role="img" aria-label="Evo 2 を中心とした4つの能力：予測・生成・設計・解釈">
    <line x1="300" y1="170" x2="300" y2="84" stroke="currentColor" strokeOpacity="0.35" strokeWidth="1.4" />
    <line x1="300" y1="170" x2="425" y2="170" stroke="currentColor" strokeOpacity="0.35" strokeWidth="1.4" />
    <line x1="300" y1="170" x2="300" y2="256" stroke="currentColor" strokeOpacity="0.35" strokeWidth="1.4" />
    <line x1="300" y1="170" x2="175" y2="170" stroke="currentColor" strokeOpacity="0.35" strokeWidth="1.4" />
    <rect x="225" y="20" width="150" height="64" rx="6" fill="#3B82F6" fillOpacity="0.1" stroke="#3B82F6" strokeWidth="1.5" />
    <text x="300" y="46" fontSize="13" fill="currentColor" textAnchor="middle" fontWeight="600">予測 Prediction</text>
    <text x="300" y="66" fontSize="9.5" fill="currentColor" fillOpacity="0.8" textAnchor="middle">変異の有害性・機能をゼロショット予測</text>
    <rect x="425" y="138" width="160" height="64" rx="6" fill="#10B981" fillOpacity="0.1" stroke="#10B981" strokeWidth="1.5" />
    <text x="505" y="164" fontSize="13" fill="currentColor" textAnchor="middle" fontWeight="600">生成 Generation</text>
    <text x="505" y="184" fontSize="9.5" fill="currentColor" fillOpacity="0.8" textAnchor="middle">ゲノム規模の新規配列を生成</text>
    <rect x="225" y="256" width="150" height="64" rx="6" fill="#EF4444" fillOpacity="0.1" stroke="#EF4444" strokeWidth="1.5" />
    <text x="300" y="282" fontSize="13" fill="currentColor" textAnchor="middle" fontWeight="600">設計 Design</text>
    <text x="300" y="302" fontSize="9.5" fill="currentColor" fillOpacity="0.8" textAnchor="middle">狙った機能をもつ配列を設計</text>
    <rect x="15" y="138" width="160" height="64" rx="6" fill="#8B5CF6" fillOpacity="0.1" stroke="#8B5CF6" strokeWidth="1.5" />
    <text x="95" y="164" fontSize="13" fill="currentColor" textAnchor="middle" fontWeight="600">解釈 Interpretation</text>
    <text x="95" y="184" fontSize="9.5" fill="currentColor" fillOpacity="0.8" textAnchor="middle">学習した生物学的概念を抽出</text>
    <ellipse cx="300" cy="170" rx="74" ry="42" fill="currentColor" fillOpacity="0.06" stroke="currentColor" strokeOpacity="0.7" strokeWidth="1.8" />
    <text x="300" y="166" fontSize="17" fill="currentColor" textAnchor="middle" fontWeight="700">Evo 2</text>
    <text x="300" y="186" fontSize="9.5" fill="currentColor" fillOpacity="0.85" textAnchor="middle">全生物ドメインの DNA</text>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>Evo 2 の4つの能力。1つの基盤モデルが、予測・解釈・生成・設計をゼロショットで担う</figcaption>
</figure>

## 6. Evo 1 からの進化

Evo 2 は、同じ研究グループによる前世代モデル **Evo 1**（Nguyen et al., *Science* 2024）を大きく拡張したものです。最大の飛躍は、**原核生物中心から真核生物を含む全ドメインへ**、そして **文脈長の桁違いの拡張** です。

| 観点 | Evo 1 | Evo 2 |
| --- | --- | --- |
| 対象生物 | 主に原核生物・ファージ | **全ドメイン**（細菌・古細菌・真核・ファージ） |
| パラメータ数 | 7B | **7B / 40B** |
| 文脈長 | 131k トークン | **1M トークン** |
| アーキテクチャ | StripedHyena | **StripedHyena 2** |
| 学習トークン数 | — | 7B: 2.4 兆 / 40B: **9.3 兆** |

真核ゲノムは原核ゲノムよりはるかに長く複雑（イントロン、長い制御領域、エクソン-イントロン構造など）です。これを扱うために文脈長を 131k から **1M** へ拡張したことが、Evo 2 の中核的な進歩です。

## 7. このあとの章

| ページ | 論文セクション | 主な内容 |
| --- | --- | --- |
| [02. アーキテクチャ](./02-architecture.md) | Architecture, training, and data | StripedHyena 2・OpenGenome2・2段階学習 |
| [03. 変異効果予測](./03-variant-prediction.md) | Evolutionary constraint / Variant effect prediction | ゼロショット予測・BRCA1・臨床応用 |
| [04. 機構的解釈可能性](./04-interpretability.md) | Feature interpretation | SAE による特徴抽出 |
| [05. ゲノムスケール生成](./05-generation.md) | Genome-scale generation | ミトコンドリア・細菌・酵母ゲノム生成 |
| [06. クロマチン設計と考察](./06-chromatin-design.md) | Chromatin design / Discussion | 推論時ガイダンス・安全性・展望 |
| [07. バイオセーフティと責任ある公開（付録）](./07-biosafety.md) | Extended Data Fig. 2 / Discussion | ウイルス除外の検証・レッドチーミング・集団バイアス |

:::info[公開リソース]

- コード: [github.com/arcinstitute/evo2](https://github.com/arcinstitute/evo2)
- モデル（Hugging Face）: [arcinstitute/evo2_40b](https://huggingface.co/arcinstitute/evo2_40b) ほか
- データセット: [arcinstitute/opengenome2](https://huggingface.co/datasets/arcinstitute/opengenome2)
- Web ツール: [Evo Designer](https://arcinstitute.org/tools/evo/evo-designer)（生成・スコアリング）

各リソースの詳細は [06. クロマチン設計と考察](./06-chromatin-design.md) の Discussion で扱います。

:::

次は、Evo 2 を支える土台——[アーキテクチャと学習・データ](./02-architecture.md)を見ていきましょう。
