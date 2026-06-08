---
sidebar_position: 1
title: "なぜ生物学で LLM なのか"
---

# なぜ生物学で LLM なのか

「大規模言語モデル（LLM）」と聞くと、ChatGPT のように **人間の言葉** を扱う技術を思い浮かべます。しかし LLM の本質は「言葉」そのものではなく、**記号の並び（配列）を読み、次に来る記号を予測する** という、もっと抽象的な仕組みにあります。

そして、私たちの体の設計図である **DNA もまた「記号の並び」** です。だとすれば、自然言語で学んだ LLM の技術は、そっくりそのまま **生命の言語** にも適用できるはずです。この発想から生まれたのが **ゲノム言語モデル（genomic language model）** であり、このセクションではその最前線を論文ベースで詳しく追っていきます。

:::tip[このセクションの位置づけ]

[基礎知識](../foundations/chapter1.md) と [LLM をゼロから作る](../llm-from-scratch/chapter1.md) では、Transformer や自己回帰学習といった **LLM の土台技術** を学びました。本セクションは、その技術が **実世界の科学（生物学）でどう応用されているか** を示す応用編です。第1弾として、Nature 2026 に掲載されたゲノム基盤モデル **[Evo 2](./evo2/01-overview.md)** を、論文のセクションごとに極めて詳細に解説します。

:::

## 1. 生命は「配列」で書かれている

分子生物学の中心原理である **セントラルドグマ（central dogma）** は、生命の情報が次の流れで処理されることを示します。

<figure style={{margin: '1.5rem auto', textAlign: 'center', maxWidth: '600px'}}>
  <svg viewBox="0 0 560 190" width="100%" role="img" aria-label="セントラルドグマ：DNA から転写で RNA、翻訳でタンパク質へ。すべて配列である">
    <rect x="28" y="52" width="132" height="74" rx="6" fill="#3B82F6" fillOpacity="0.08" stroke="#3B82F6" strokeWidth="1.6" />
    <text x="94" y="80" fontSize="14" fill="currentColor" textAnchor="middle" fontWeight="600">DNA</text>
    <text x="94" y="104" fontSize="11" fill="currentColor" fillOpacity="0.8" textAnchor="middle">…A T G C G T…</text>
    <rect x="214" y="52" width="132" height="74" rx="6" fill="#10B981" fillOpacity="0.08" stroke="#10B981" strokeWidth="1.6" />
    <text x="280" y="80" fontSize="14" fill="currentColor" textAnchor="middle" fontWeight="600">RNA</text>
    <text x="280" y="104" fontSize="11" fill="currentColor" fillOpacity="0.8" textAnchor="middle">…A U G C G U…</text>
    <rect x="400" y="52" width="132" height="74" rx="6" fill="#EF4444" fillOpacity="0.08" stroke="#EF4444" strokeWidth="1.6" />
    <text x="466" y="80" fontSize="14" fill="currentColor" textAnchor="middle" fontWeight="600">タンパク質</text>
    <text x="466" y="104" fontSize="11" fill="currentColor" fillOpacity="0.8" textAnchor="middle">…Met-Arg…</text>
    <line x1="164" y1="89" x2="208" y2="89" stroke="currentColor" strokeOpacity="0.6" strokeWidth="1.6" />
    <polygon points="214,89 205,84 205,94" fill="currentColor" fillOpacity="0.6" />
    <text x="186" y="44" fontSize="10.5" fill="currentColor" textAnchor="middle">転写</text>
    <line x1="350" y1="89" x2="394" y2="89" stroke="currentColor" strokeOpacity="0.6" strokeWidth="1.6" />
    <polygon points="400,89 391,84 391,94" fill="currentColor" fillOpacity="0.6" />
    <text x="372" y="44" fontSize="10.5" fill="currentColor" textAnchor="middle">翻訳</text>
    <text x="280" y="160" fontSize="12" fill="currentColor" textAnchor="middle" fontStyle="italic">DNA も RNA もタンパク質も、すべて「記号の並び（＝配列）」</text>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>セントラルドグマ。3つの分子はいずれも有限種類の記号からなる配列であり、LLM が扱う「トークン列」と同じ構造をもつ</figcaption>
</figure>

ここで重要なのは、**3つの分子がすべて有限種類の記号からなる「配列」だ** という点です。

| 分子 | 記号の種類（語彙） | 記号の正体 |
| --- | --- | --- |
| DNA | **4** 種類 | 塩基 A・T・G・C（ヌクレオチド） |
| RNA | **4** 種類 | 塩基 A・U・G・C |
| タンパク質 | **20** 種類 | アミノ酸 |

自然言語の文章が「単語の並び」であるように、ゲノムは「塩基の並び」です。語彙の大きさこそ違いますが、**「有限の記号を一列に並べて意味（機能）を表現する」** という構造は完全に共通しています。

## 2. 自然言語モデルとゲノム言語モデルの対応

この構造の共通性のおかげで、自然言語 LLM の枠組みは、ほぼ読み替えるだけでゲノムに移植できます。

| 観点 | 自然言語 LLM | ゲノム言語モデル |
| --- | --- | --- |
| トークン | 単語・サブワード | 塩基（ヌクレオチド） |
| 語彙サイズ | 数万 | 4（DNA） |
| 1つの「文」 | 文章・段落 | 遺伝子・染色体・ゲノム |
| 学習データ | Web 上の大量テキスト | 公共 DB の大量ゲノム配列 |
| 学習タスク | 次の単語を予測 | 次の塩基を予測 |
| 文脈長 | 数千〜数十万トークン | 数千〜**100 万**塩基 |

学習の目的関数も同じです。自己回帰言語モデルは、配列 $x_1, x_2, \dots, x_n$ の同時確率を、**「これまでの記号から次の記号を予測する」** 条件付き確率の積に分解して学習します。

$$
p(x_1, x_2, \dots, x_n) = \prod_{i=1}^{n} p(x_i \mid x_1, \dots, x_{i-1})
$$

$x_i$ が「単語」なら自然言語モデル、「塩基」ならゲノム言語モデルになる——ただそれだけの違いです。Evo 2 はまさにこの式を、**全生物の DNA** に対して巨大なスケールで学習したモデルです。

:::note[進化は「教師なしのアノテーション」]

ゲノム言語モデルが強力な理由の一つは、**進化（自然選択）が学習データに意味を刻んでいる** ことです。生存に重要な配列は変異が淘汰されて世代を超えて保存され、重要でない配列は変化しやすい。つまり「どの配列が機能的に重要か」という情報が、何十億年もの進化を通じて DNA そのものに織り込まれています。モデルが大量のゲノムから学んだ尤度（配列のもっともらしさ）は、この **進化的制約（evolutionary constraint）** を反映します。これが後で見る「ゼロショット変異効果予測」の土台になります。

:::

## 3. ゲノム言語モデルで何ができるのか

「次の塩基を予測する」だけのモデルが、実際には驚くほど多彩な応用を生みます。本セクションで扱う Evo 2 を例にとると、大きく次の3つです。

- **予測（Prediction）** — ある変異（塩基の変化）が有害かどうか、遺伝子のどこがエクソンか、といった機能を、追加学習なし（ゼロショット）で予測する。臨床的に重要な BRCA1 変異の判定にまで踏み込みます。
- **解釈（Interpretation）** — モデルが内部で何を「理解」したのかを、機構的解釈可能性（mechanistic interpretability）の手法で覗き込む。LLM 研究で発展した **スパースオートエンコーダ（SAE）** がそのまま使われます。
- **生成・設計（Generation / Design）** — 学習した分布から、ミトコンドリアや細菌ゲノム規模の **新しい DNA 配列** を生成する。さらに推論時ガイダンスで、狙った機能（クロマチンの開き具合など）をもつ配列を設計します。

## 4. このセクションの読み方

第1弾の題材は、**Arc Institute・Stanford・NVIDIA** らが開発し Nature 2026 に発表した **Evo 2** です。Evo 2 の解説は、論文の構成に沿って次の6ページに分けています。

1. [概要と全体像](./evo2/01-overview.md) — 論文の要旨・貢献・モデルの全体像
2. [アーキテクチャと学習・データ](./evo2/02-architecture.md) — StripedHyena 2・OpenGenome2・2段階学習
3. [変異効果予測](./evo2/03-variant-prediction.md) — 進化的制約の学習からヒト臨床変異の予測まで
4. [機構的解釈可能性](./evo2/04-interpretability.md) — SAE でモデルの「概念」を取り出す
5. [ゲノムスケール生成](./evo2/05-generation.md) — 細菌・ミトコンドリア・酵母ゲノムの生成
6. [クロマチン設計と考察](./evo2/06-chromatin-design.md) — 推論時ガイダンスによる設計と Discussion

:::info[対象読者と方針]

本ドキュメントは、**LLM の基礎（自己回帰・Transformer など）をある程度知っている読者** を想定し、生物学の用語は初出時に最小限だけ補足します。そのうえで、**モデル設計・評価手法・実験結果といった技術的詳細** に重点を置き、随所で「これは LLM のあの技術と同じだ」という対応関係を示していきます。

:::

それでは、[Evo 2 の概要](./evo2/01-overview.md)から始めましょう。
