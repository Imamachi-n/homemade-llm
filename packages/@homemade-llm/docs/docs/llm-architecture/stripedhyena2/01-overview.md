---
sidebar_position: 1
title: "01. 概要：なぜマルチハイブリッドか"
---

# 01. 概要：なぜマルチハイブリッドか

このセクションでは、論文 **「Systems and Algorithms for Convolutional Multi-Hybrid Language Models at Scale」**（Ku, Nguyen, Romero et al., 2025, [arXiv:2503.01868](https://arxiv.org/abs/2503.01868)）を詳しく読み解きます。これは、[生物学における LLM の応用事例](../../biology-llm-applications/evo2/01-overview.md)で扱った **Evo 2** を支える基盤アーキテクチャ **StripedHyena 2** の本体論文です。

:::tip[このドキュメントの位置づけ]

[Evo 2 のアーキテクチャ解説](../../biology-llm-applications/evo2/02-architecture.md)では、StripedHyena 2 を「$O(n^2)$ の壁を越えるための畳み込みマルチハイブリッド」として概観しました。本セクションはその **深掘り版** です。Evo 2 という応用から離れ、**アーキテクチャ・カーネル・分散学習** というシステム寄りの観点から、なぜこの設計が高速なのかを論文に沿って掘り下げます。

:::

## 1. この論文は何を解決するのか

大規模言語モデルのアーキテクチャ改善は、これまで主に次のように進んできました。

- **Attention の改良**（KV キャッシュ削減）：GQA・MQA・MLA・sliding window・linear attention
- **数値安定性**：pre-norm・SwiGLU・QK normalization
- **容量・長文脈での recall**：RoPE・MoE（Mixture of Experts）

しかし論文は、**「これら以外で、スケールにおいて一貫した改善をもたらした提案は驚くほど少ない」** と指摘します。その数少ない有望株が **ハイブリッド（hybrid）アーキテクチャ**——標準の層（self-attention ＋ FFN）に、新しい **入力依存の演算子（input-dependent operator）** を混ぜる設計です。

### 既存ハイブリッドの限界

ところが、linear attention や状態空間モデル（SSM, 例：Mamba）をベースにしたハイブリッドは、Transformer を置き換える標準にはなれていません。論文が挙げる理由は明快です。

- これら **固定状態（fixed-state）の演算子** は、**超長系列でしか効率の利点が出ない**。しかも、まさにその超長系列で full self-attention に **品質で大きく劣る**。
- 一般的な事前学習の設定（**短めの文脈・大きく広いモデル**）では、Transformer より **むしろ遅い**。
- そもそも「in-context recall を self-attention に匹敵させる」目的で作られたため、実用では結局 self-attention とのハイブリッドが必要になり、**複数の演算子が同じ能力（recall）を奪い合う冗長性** が生まれていた。

## 2. 2つの設計思想

この論文は、根本的に異なるアプローチを取ります。それが **hybridization-aware（ハイブリッド化を意識）** かつ **hardware-aware（ハードウェアを意識）** な設計です。基礎となるのは2つの観察です。

1. **演算子はトークン操作タスクに特化できる** — in-context recall・multi-token recall・compression といったサブタスクに対し、**入力依存の畳み込み（input-dependent convolution）と attention は相補的** に働く。たとえば入力依存の畳み込みは **ノイズ除去と multi-token recall**（バイト/文字レベルのデータ modeling に有用）に長け、attention は **長距離の精密な recall** に優れる。
2. **演算子とハードウェアアルゴリズムの co-design** — 演算子と GPU 向けアルゴリズムを一緒に設計することで、従来の代替アーキが Transformer を超えられなかった領域でも効率を得られる。

この2つから生まれるのが **マルチハイブリッド（multi-hybrid）**——**複数タイプの演算子の強みを組み合わせる** アーキテクチャです。

<figure style={{margin: '1.5rem auto', textAlign: 'center', maxWidth: '360px'}}>
  <svg viewBox="0 0 300 200" width="100%" role="img" aria-label="文脈長に対するスループット。Transformerは長文脈で急降下、SSMは中位、StripedHyena 2は全域で高い">
    <line x1="40" y1="165" x2="280" y2="165" stroke="currentColor" strokeOpacity="0.3" strokeWidth="1" />
    <line x1="40" y1="25" x2="40" y2="170" stroke="currentColor" strokeOpacity="0.3" strokeWidth="1" />
    <text x="270" y="182" fontSize="9.5" fill="currentColor" fillOpacity="0.7" textAnchor="end">文脈長 →</text>
    <text x="14" y="30" fontSize="9.5" fill="currentColor" fillOpacity="0.7">スループット</text>
    <polyline points="48,48 84,58 120,80 156,110 196,140 232,156" fill="none" stroke="#EF4444" strokeWidth="2.4" />
    <polyline points="48,128 92,124 140,118 192,112 232,106" fill="none" stroke="currentColor" strokeOpacity="0.55" strokeWidth="2.2" strokeDasharray="5 3" />
    <polyline points="48,44 92,52 140,62 192,74 232,86" fill="none" stroke="#3B82F6" strokeWidth="2.6" />
    <text x="150" y="135" fontSize="10" fill="#EF4444">Transformer  O(n²)</text>
    <text x="120" y="118" fontSize="9.5" fill="currentColor" fillOpacity="0.75">SSM / linear attn</text>
    <text x="120" y="56" fontSize="10" fill="#3B82F6" fontWeight="600">StripedHyena 2</text>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>狙いは「全域で速い」こと。短文脈で遅い SSM とも、長文脈で急降下する Transformer とも違い、マルチハイブリッドは広い文脈長で高スループットを保つ（論文 Fig. 1 の趣旨）</figcaption>
</figure>

## 3. StripedHyena 2 とは

**StripedHyena 2** は、論文が提案する **最初の畳み込みマルチハイブリッド（convolutional multi-hybrid）** アーキテクチャで、**40 億パラメータ・9 兆トークン** という大規模で検証されています。核となるのは、距離スケールの異なる **3 種類の入力依存畳み込み演算子** です。

| 演算子 | 正式名 | 役割 |
| --- | --- | --- |
| **SE** | Hyena-SE（short explicit） | 短い明示的フィルタ。局所的な multi-token recall。ハードウェア利用率を最大化 |
| **MR** | Hyena-MR（medium regularized） | 中程度（数百トークン）の正則化フィルタ |
| **LI** | Hyena-LI（long implicit） | 系列全体を集約する長い暗黙的フィルタ |

これらと attention を縞状（striped）に積み重ねます。詳細は [02. 3つの Hyena 演算子](./02-operators.md)で扱います。

### 主な成果

| 指標 | 結果 |
| --- | --- |
| end-to-end 学習速度（40B, 最適化 Transformer 比） | **1.2〜2.9 倍** 高速 |
| 前世代ハイブリッド（StripedHyena 1 等）比 | 1.1〜1.4 倍 高速 |
| 演算子レベル（H100・幅 4096, linear attention/SSM 比） | 約 **2 倍** のスループット |
| 検証スケール | **Evo 2 40B**（9 兆トークン・100 万文脈・塩基トークン化） |

注目すべきは、**短い系列長でも** Transformer と前世代 StripedHyena の両方より速い点です。従来のハイブリッドが「長文脈でしか速くない」という弱点を抱えていたのとは対照的です。

## 4. Evo 2 との関係

マルチハイブリッドは **バイト/文字レベルのデータ** の系列 modeling に特に優れます。これがゲノム（A・T・G・C の塩基トークン）を扱う **Evo 2** との相性の良さに直結します。論文は Evo 2 を一貫した **動機づけの実例（motivating example）** として用い、StripedHyena 2 上に構築された Evo 2 40B が **ゲノミクスの最先端基盤モデル** であることを示します。

:::note[Evo 2 を先に読むと理解が深まります]

本セクションはシステム・アルゴリズムに深く踏み込むため、まず [Evo 2 の概要](../../biology-llm-applications/evo2/01-overview.md)と[アーキテクチャ](../../biology-llm-applications/evo2/02-architecture.md)で「何のための、どんなスケールのモデルか」を押さえておくと、ここでの設計判断の意味が掴みやすくなります。

:::

## 5. 論文の貢献とこのあとの章

論文が議論する技術的基盤は、大きく3つです。

1. **アーキテクチャ設計**（演算子と block layout）→ [02](./02-operators.md)・[03](./03-architecture-scaling.md)
2. **テンソルコア向けの overlap-add ブロックカーネル** → [04](./04-hardware-algorithms.md)
3. **専用の context parallelism 戦略**（all-to-all・point-to-point）→ [05](./05-context-parallelism.md)

| ページ | 論文セクション | 主な内容 |
| --- | --- | --- |
| [02. 3つの Hyena 演算子](./02-operators.md) | §2.1 | Hyena 構造の数式・SE/MR/LI の設計 |
| [03. アーキテクチャ設計とスケーリング](./03-architecture-scaling.md) | §2.2–2.3 | block layout・filter grouping・文脈拡張・スループット |
| [04. ハードウェア対応カーネル](./04-hardware-algorithms.md) | §3 | Toeplitz・block convolution・two-stage・tensor core |
| [05. 長系列の分散学習](./05-context-parallelism.md) | §4・付録 | a2a/p2p・FFT 畳み込み・Radix-N・butterfly |

:::info[公開リソース]

- 学習インフラ **Savanna**: [github.com/Zymrael/savanna](https://github.com/Zymrael/savanna)
- 推論インフラ **Vortex**: [github.com/Zymrael/vortex](https://github.com/Zymrael/vortex)

:::

それでは、StripedHyena 2 の心臓部である [3つの Hyena 演算子](./02-operators.md)から見ていきましょう。
