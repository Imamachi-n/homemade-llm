---
sidebar_position: 7
title: "07. バイオセーフティと責任ある公開"
---

# 07. バイオセーフティと責任ある公開

[06](./06-chromatin-design.md) の Discussion で、Evo 2 の安全性への配慮に触れました。このページは、論文の **Extended Data Fig. 2** と Discussion をもとに、その **バイオセーフティ評価を詳しく** 掘り下げる付録です。Evo 2 は **完全オープンソース** で公開されたため、「誰でも使える」ことと「危険な使われ方を防ぐ」ことの両立が、設計段階から組み込まれています。

:::note[このページの位置づけ]

生物基盤モデルは強力なほど **デュアルユース（軍民両用／善悪両用）** のリスクを伴います。ここでは「Evo 2 がどんな対策を講じ、それが本当に効いているかをどう検証したか」を、データ・予測・生成の各面から見ます。LLM の安全対策（有害出力の抑制やレッドチーミング）と地続きの議論です。

:::

## 1. 生物基盤モデルとデュアルユース

DNA を自在に予測・生成できるモデルは、医療や基礎研究を加速する一方で、**病原体の設計・改変** に悪用される可能性も理論上はあります。とりわけ **ヒトに感染するウイルス** は懸念の中心です。

Evo 2 の開発チームは、**Responsible AI × Biodesign** のコミットメント（[responsiblebiodesign.ai](https://responsiblebiodesign.ai/)）に沿って、公開前にリスクを評価・緩和しました。中心となる対策が **学習データからの除外** です。

## 2. データ除外：危険なウイルスを学ばせない

[02. アーキテクチャ](./02-architecture.md)で触れたとおり、Evo 2 は（Evo 1 と同様に）**真核生物に感染するウイルスのゲノムを学習データから除外** しています。「学んでいないものは、うまく予測も生成もできない」——この当たり前の性質を、安全装置として積極的に使うアプローチです。

重要なのは、**この除外が本当に効いているかを実証した** 点です。論文は3つの角度から検証しました。

<figure style={{margin: '1.5rem auto', textAlign: 'center', maxWidth: '520px'}}>
  <svg viewBox="0 0 520 200" width="100%" role="img" aria-label="安全性の3段階検証：言語モデリング・予測・生成のすべてでヒトウイルスを扱えないことを確認">
    <rect x="20" y="40" width="150" height="120" rx="8" fill="#EF4444" fillOpacity="0.08" stroke="#EF4444" strokeWidth="1.4" />
    <text x="95" y="62" fontSize="11" fill="currentColor" textAnchor="middle" fontWeight="600">① 言語モデリング</text>
    <text x="95" y="84" fontSize="9.5" fill="currentColor" fillOpacity="0.85" textAnchor="middle">ヒトウイルスで</text>
    <text x="95" y="100" fontSize="9.5" fill="currentColor" fillOpacity="0.85" textAnchor="middle">高 perplexity</text>
    <text x="95" y="124" fontSize="9" fill="#EF4444" textAnchor="middle">= うまく</text>
    <text x="95" y="138" fontSize="9" fill="#EF4444" textAnchor="middle">モデル化できない</text>
    <rect x="185" y="40" width="150" height="120" rx="8" fill="#EF4444" fillOpacity="0.08" stroke="#EF4444" strokeWidth="1.4" />
    <text x="260" y="62" fontSize="11" fill="currentColor" textAnchor="middle" fontWeight="600">② 予測</text>
    <text x="260" y="84" fontSize="9.5" fill="currentColor" fillOpacity="0.85" textAnchor="middle">ウイルスタンパク質の</text>
    <text x="260" y="100" fontSize="9.5" fill="currentColor" fillOpacity="0.85" textAnchor="middle">DMS と相関なし</text>
    <text x="260" y="124" fontSize="9" fill="#EF4444" textAnchor="middle">= 変異効果を</text>
    <text x="260" y="138" fontSize="9" fill="#EF4444" textAnchor="middle">当てられない</text>
    <rect x="350" y="40" width="150" height="120" rx="8" fill="#EF4444" fillOpacity="0.08" stroke="#EF4444" strokeWidth="1.4" />
    <text x="425" y="62" fontSize="11" fill="currentColor" textAnchor="middle" fontWeight="600">③ 生成</text>
    <text x="425" y="84" fontSize="9.5" fill="currentColor" fillOpacity="0.85" textAnchor="middle">ウイルスタンパク質を</text>
    <text x="425" y="100" fontSize="9.5" fill="currentColor" fillOpacity="0.85" textAnchor="middle">生成しようとしても</text>
    <text x="425" y="124" fontSize="9" fill="#EF4444" textAnchor="middle">= ほぼランダム</text>
    <text x="425" y="138" fontSize="9" fill="#EF4444" textAnchor="middle">（レッドチーミング）</text>
    <text x="260" y="184" fontSize="10" fill="currentColor" fillOpacity="0.75" textAnchor="middle">3つの面すべてで「ヒトウイルスを扱えない」ことを確認</text>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>安全性の多層検証（論文 Extended Data Fig. 2a–c）。データ除外が「言語モデリング・予測・生成」のすべてで効いていることを実証</figcaption>
</figure>

### ① 言語モデリング：高い perplexity（Extended Data Fig. 2a）

**USDA Select Agents and Toxins List**（規制対象の危険な病原体・毒素のリスト）に載るウイルス配列に対し、Evo 2 は **一貫して高い perplexity**（言語モデルとしての「驚き」の大きさ）を示しました。非病原性ウイルスや原核生物ウイルス（ファージ）と比べて明確に高い——つまり **モデルがこれらの配列をうまく扱えない** ことを意味します。

<figure style={{margin: '1.5rem auto', textAlign: 'center', maxWidth: '420px'}}>
  <svg viewBox="0 0 400 190" width="100%" role="img" aria-label="perplexity比較。ヒト感染ウイルスは高く、ファージや非病原ウイルスは低い">
    <line x1="40" y1="155" x2="385" y2="155" stroke="currentColor" strokeOpacity="0.35" strokeWidth="1.2" />
    <text x="16" y="40" fontSize="9" fill="currentColor" fillOpacity="0.7">perplexity</text>
    <text x="16" y="52" fontSize="8" fill="currentColor" fillOpacity="0.6">（高=安全）</text>
    <rect x="70" y="40" width="60" height="115" fill="#EF4444" fillOpacity="0.55" /><text x="100" y="170" fontSize="9" fill="currentColor" textAnchor="middle">ヒト感染</text><text x="100" y="182" fontSize="9" fill="currentColor" textAnchor="middle">ウイルス</text>
    <rect x="175" y="110" width="60" height="45" fill="#10B981" fillOpacity="0.55" /><text x="205" y="170" fontSize="9" fill="currentColor" textAnchor="middle">ファージ</text>
    <rect x="280" y="118" width="60" height="37" fill="#10B981" fillOpacity="0.55" /><text x="310" y="170" fontSize="9" fill="currentColor" textAnchor="middle">非病原</text><text x="310" y="182" fontSize="9" fill="currentColor" textAnchor="middle">ウイルス</text>
    <text x="100" y="34" fontSize="9" fill="#EF4444" textAnchor="middle">扱えない</text>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>ヒト感染ウイルスは perplexity が高い（＝モデルが扱えない）。除外が意図どおり効いている（Extended Data Fig. 2a の概念図）</figcaption>
</figure>

### ② 予測：DMS と相関しない（Extended Data Fig. 2b）

[03. 変異効果予測](./03-variant-prediction.md)で見た変異効果予測を、ヒトウイルスタンパク質に対して試すと、**実験的な DMS フィットネス測定（ProteinGym）と相関しません**。Evo 2 も Evo 1 も、ウイルスタンパク質の変異効果は当てられない——予測能力も意図どおり弱められています。

### ③ 生成：レッドチーミングでもランダム（Extended Data Fig. 2c）

最も直接的な検証が **レッドチーミング**（攻撃者の視点で危険な出力を引き出そうとするテスト）です。ウイルスタンパク質を直接生成させようとしても、Evo 2 のアミノ酸配列回復率は **ランダムな生成と同程度** にとどまりました。意図的に引き出そうとしても、**ヒト病原体ウイルスのタンパク質は生成できない** のです。

:::tip[LLM とのつながり：レッドチーミングと安全性評価]

「危険な出力を引き出そうと攻撃的にテストし、防御が効いているか確認する」**レッドチーミング** は、自然言語 LLM の安全性評価（有害コンテンツ・脱獄プロンプトへの耐性チェック）とまったく同じ発想です。Evo 2 は、生物配列という別ドメインで、この安全性検証の方法論を取り入れています。

:::

## 3. 集団バイアス（ancestry bias）の評価（Extended Data Fig. 2d）

安全性に加え、**公平性** も評価されています。変異効果予測モデルには、**特定の祖先集団（ancestry）に対する偏り** が知られています——多くの予測器は、**非ヨーロッパ系の変異をより「病原性が高い」と過剰判定** する傾向があります（Pathak et al.）。

論文は、Evo 2 のこの **ancestry bias** を、各集団サブグループのスコアをヨーロッパ系と比較（min-max スケール後の比率・平均差）して定量しました。結果、Evo 2 は **集団遺伝データに依存しない（population-free）他の手法と同程度のバイアス** にとどまりました。集団頻度データを使う手法に比べ、配列だけから学ぶ Evo 2 は構造的に偏りを増幅しにくい、という位置づけです。

:::note[「バイアスがゼロ」ではない点に注意]

Evo 2 が他手法より極端に良い／悪いわけではなく、**population-free 手法として妥当な範囲** に収まっている、という結果です。変異効果予測の公平性は分野全体の課題であり、Evo 2 単独で解決されたわけではありません。

:::

## 4. 責任ある公開：予防とアクセスの両立

Evo 2 は、モデルパラメータ・学習コード・推論コード・データセットまで **完全公開** されました。これには明確なトレードオフがあります。

- **オープンの利点**：研究の **再現・検証・発展** を誰もが行える。科学の透明性と進歩を加速する。
- **オープンのリスク**：**想定外の使われ方**（事故・悪用）の可能性も開かれる。

論文は、データ除外・安全評価・集団バイアス評価という多面的な取り組みによって、**「予防（precaution）」と「アクセス（access）」を両立** させようとしています。著者らは、これを生物基盤モデルとして **最も包括的な評価のひとつ** と位置づけつつ、次の限界も率直に認めています。

:::warning[残された課題]

- **post-training で回避され得る**：タスク特化の追加学習（ファインチューニング）を施せば、データ除外による安全装置は **迂回される可能性** があります。公開モデルの再学習は慎重に扱うべきです。
- **評価の網羅性**：生物基盤モデルの実証的なリスク評価の前例は少なく、**評価手法とリスク緩和策のさらなる拡充** が必要だと述べられています。

:::

## 5. まとめ

- 生物基盤モデルの **デュアルユースリスク** に対し、Evo 2 は **真核感染ウイルスの学習データ除外** を安全装置として採用。
- 除外の効果を **言語モデリング（高 perplexity）・予測（DMS 無相関）・生成（レッドチーミングでランダム）** の3面で実証。
- **集団バイアス** を評価し、population-free 手法として妥当な範囲を確認。
- 完全オープン化にあたり **予防とアクセスを両立**。ただし post-training での迂回可能性など、課題も明示。

これで Evo 2 の解説は、本編（[01](./01-overview.md)〜[06](./06-chromatin-design.md)）に加え、安全性の付録まで揃いました。技術の力と、それを **責任を持って世に出す** 姿勢の両方が、この論文の重要なメッセージです。
