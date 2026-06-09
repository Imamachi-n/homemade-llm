---
sidebar_position: 2
title: "02. アーキテクチャと学習・データ"
---

# 02. アーキテクチャと学習・データ

このページでは、Evo 2 を支える3つの土台——**モデルアーキテクチャ（StripedHyena 2）**、**学習データ（OpenGenome2）**、**2段階の学習戦略**——を、LLM の技術と対応づけながら詳しく見ていきます。

論文では2つのサイズのモデルが学習されました。

| モデル | パラメータ数 | 学習トークン数 |
| --- | --- | --- |
| Evo 2 **7B** | 70 億 | 2.4 兆 |
| Evo 2 **40B** | 400 億 | **9.3 兆** |

いずれも、短い文脈での **事前学習（pretraining）** と、文脈を 100 万トークンまで伸ばす **中間学習（midtraining）** の2段階で訓練されます。

## 1. なぜ Transformer ではないのか：$O(n^2)$ の壁

Evo 2 の最大の特徴は **100 万トークンの文脈長** です。これは「DNA 上で遠く離れた要素（たとえばエンハンサーと遺伝子本体）の関係」を捉えるために不可欠ですが、標準的な Transformer では実現が困難です。理由は、Self-Attention の計算量にあります。

[Transformer の実装](../../llm-from-scratch/chapter2.md)で見たように、Self-Attention は次の式で全トークン間の関連度を計算します。

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V
$$

ここで $Q, K, V$ は系列長 $n$ ×次元 $d$ の行列です。問題は $QK^\top$ で、これは **$n \times n$ の行列** になります。つまり計算量・メモリともに **$O(n^2)$** で系列長に対して二次的に増えます。

$$
n = 10^6 \ \Rightarrow\ n^2 = 10^{12}
$$

文脈長が 100 万になると、注意行列の要素数は **1 兆** に達し、まともに計算できません。これが「長文脈の壁」です。

<figure style={{margin: '1.5rem auto', textAlign: 'center', maxWidth: '340px'}}>
  <svg viewBox="0 0 260 180" width="100%" role="img" aria-label="計算コストの比較。Transformer は n の二乗で急増、StripedHyena は緩やかに増加">
    <line x1="34" y1="150" x2="240" y2="150" stroke="currentColor" strokeOpacity="0.3" strokeWidth="1" />
    <line x1="40" y1="20" x2="40" y2="156" stroke="currentColor" strokeOpacity="0.3" strokeWidth="1" />
    <text x="232" y="166" fontSize="10" fill="currentColor" fillOpacity="0.7">文脈長 n</text>
    <text x="20" y="26" fontSize="10" fill="currentColor" fillOpacity="0.7">コスト</text>
    <polyline points="42,148 70,144 100,136 130,122 160,100 185,72 205,44 220,22" fill="none" stroke="#EF4444" strokeWidth="2.4" />
    <polyline points="42,148 80,142 120,135 160,127 200,119 230,113" fill="none" stroke="#10B981" strokeWidth="2.4" />
    <text x="150" y="48" fontSize="10.5" fill="#EF4444">Transformer  O(n²)</text>
    <text x="120" y="108" fontSize="10.5" fill="#10B981">StripedHyena（準二次）</text>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>Self-Attention は文脈長の二乗でコストが増える。長文脈には準二次（subquadratic）な演算子が必要</figcaption>
</figure>

## 2. StripedHyena 2：畳み込みマルチハイブリッド

そこで Evo 2 が採用したのが **StripedHyena 2** という **畳み込みマルチハイブリッド（convolutional multi-hybrid）** アーキテクチャです。Attention をすべて捨てるのではなく、**入力依存の畳み込み演算子（input-dependent convolution）** と Attention を組み合わせます。

:::note[Hyena 演算子とは]

通常の CNN のフィルタは固定ですが、**Hyena 演算子** は入力に応じてフィルタが変調される **長い畳み込み + データ依存ゲーティング** です。長い畳み込みは高速フーリエ変換（FFT）を使うと **$O(n \log n)$** で計算でき、$O(n^2)$ の Attention より圧倒的に軽くなります。これにより、Attention の「全ペア比較」に頼らずに長距離の依存関係を捉えられます。

:::

StripedHyena 2 は、距離スケールの異なる **3種類の Hyena 演算子** と Attention を層ごとに織り交ぜます（名前の "Striped"＝縞模様は、この交互配置を指します）。

| 演算子 | 略称 | 捉える距離 |
| --- | --- | --- |
| Short Explicit | **SE** | 短距離（局所パターン） |
| Medium Regularized | **MR** | 中距離 |
| Long Implicit | **LI** | 長距離（暗黙的に生成する長いフィルタ） |
| Rotary Attention | Attn | 必要箇所で全ペア比較を補完 |

<figure style={{margin: '1.5rem auto', textAlign: 'center', maxWidth: '600px'}}>
  <svg viewBox="0 0 580 150" width="100%" role="img" aria-label="StripedHyena 2 のブロック構成。SE・MR・LI の Hyena 演算子と Attention を交互に積む">
    <text x="20" y="60" fontSize="11" fill="currentColor" textAnchor="middle">入力</text>
    <line x1="34" y1="70" x2="56" y2="70" stroke="currentColor" strokeOpacity="0.5" strokeWidth="1.4" />
    <polygon points="60,70 52,66 52,74" fill="currentColor" fillOpacity="0.5" />
    <rect x="62" y="48" width="46" height="44" rx="4" fill="#3B82F6" fillOpacity="0.14" stroke="#3B82F6" strokeWidth="1.4" />
    <text x="85" y="74" fontSize="12" fill="currentColor" textAnchor="middle">SE</text>
    <rect x="114" y="48" width="46" height="44" rx="4" fill="#10B981" fillOpacity="0.14" stroke="#10B981" strokeWidth="1.4" />
    <text x="137" y="74" fontSize="12" fill="currentColor" textAnchor="middle">MR</text>
    <rect x="166" y="48" width="46" height="44" rx="4" fill="#8B5CF6" fillOpacity="0.14" stroke="#8B5CF6" strokeWidth="1.4" />
    <text x="189" y="74" fontSize="12" fill="currentColor" textAnchor="middle">LI</text>
    <rect x="218" y="48" width="56" height="44" rx="4" fill="#EF4444" fillOpacity="0.14" stroke="#EF4444" strokeWidth="1.4" />
    <text x="246" y="74" fontSize="11" fill="currentColor" textAnchor="middle">Attn</text>
    <rect x="280" y="48" width="46" height="44" rx="4" fill="#3B82F6" fillOpacity="0.14" stroke="#3B82F6" strokeWidth="1.4" />
    <text x="303" y="74" fontSize="12" fill="currentColor" textAnchor="middle">SE</text>
    <rect x="332" y="48" width="46" height="44" rx="4" fill="#10B981" fillOpacity="0.14" stroke="#10B981" strokeWidth="1.4" />
    <text x="355" y="74" fontSize="12" fill="currentColor" textAnchor="middle">MR</text>
    <rect x="384" y="48" width="46" height="44" rx="4" fill="#8B5CF6" fillOpacity="0.14" stroke="#8B5CF6" strokeWidth="1.4" />
    <text x="407" y="74" fontSize="12" fill="currentColor" textAnchor="middle">LI</text>
    <text x="452" y="74" fontSize="16" fill="currentColor" fillOpacity="0.6" textAnchor="middle">…</text>
    <line x1="470" y1="70" x2="492" y2="70" stroke="currentColor" strokeOpacity="0.5" strokeWidth="1.4" />
    <polygon points="496,70 488,66 488,74" fill="currentColor" fillOpacity="0.5" />
    <text x="540" y="60" fontSize="11" fill="currentColor" textAnchor="middle">出力</text>
    <text x="540" y="74" fontSize="9" fill="currentColor" fillOpacity="0.7" textAnchor="middle">（次塩基の確率）</text>
    <text x="290" y="128" fontSize="9.5" fill="currentColor" fillOpacity="0.75" textAnchor="middle">距離スケールの異なる演算子を縞状に積み重ね、各層が別々の距離の相互作用をモデル化する</text>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>StripedHyena 2 のブロック構成（論文 Fig. 1f に対応）。SE・MR・LI と Attention を層ごとに交互配置</figcaption>
</figure>

この設計の効果は明確です。論文によると、StripedHyena 2 は **40B・100 万文脈で、高度に最適化された Transformer に対して最大 3 倍のスループット** を達成し（Fig. 1g）、前世代の StripedHyena 1（リカレンスや長畳み込みベース）も上回ります。さらに DNA 上での **損失スケーリング（loss scaling）も改善** し、同じ学習データ量でより低い予測誤差に到達します。

:::tip[LLM とのつながり：Attention 一辺倒からの脱却]

「Transformer の Attention を別の演算子で置き換えて長文脈を効率化する」という流れは、自然言語 LLM でも State Space Model（Mamba など）やハイブリッドモデルとして活発に研究されています。Evo 2 はこの系譜の **Hyena 系ハイブリッド** を、ゲノムという超長文脈ドメインで大規模に実証した例と言えます。アーキテクチャの詳細は、本サイトの [StripedHyena 2 詳解](../../llm-architecture/stripedhyena2/01-overview.md)（姉妹論文 Ku et al., arXiv:2503.01868）で深掘りしています。

:::

## 3. 学習データ：OpenGenome2

Evo 2 は **OpenGenome2** という新しいデータセットで学習されました。**8.8 兆ヌクレオチド超**の、キュレーション済み・非冗長の配列からなり、生命のすべてのドメインを含みます。

| データソース | 内容 |
| --- | --- |
| **GTDB**（Genome Taxonomy Database） | 細菌・古細菌のゲノム |
| **NCBI** genomes | 真核生物ゲノム（動物・植物・菌類・原生生物） |
| **IMG/VR** | ウイルスゲノム（ただし真核宿主に感染するものは除外。後述） |
| **Metagenomics** | 環境メタゲノム |

真核ゲノムは長く複雑なので、論文では機能要素に注目した複数のサブセット——**genic（遺伝子）領域**、**プロモーター＋エクソン＋スプライス部位**、**mRNA**、**5 kb ウィンドウ**、**ncRNA**、**EPDnew（プロモーター DB）**、**オルガネラ**——を用意し、学習段階に応じて配合を変えています（Fig. 1d）。

:::note[データの重み付けという工夫]

すべての塩基を平等に学習させるのではなく、**事前学習では機能的な遺伝要素（genic window）を重点的に**、**中間学習では長い配列の構成を重点的に**サンプリングします。「重要な部分を厚く学習させる」というキュレーションが、限られた計算資源での性能を引き上げます。

:::

## 4. 2段階の学習戦略

Evo 2 は、生命の **分子スケールから個体スケールまで** の長さの情報を捉えるために、2段階で学習されます（Fig. 1c–e）。

<figure style={{margin: '1.5rem auto', textAlign: 'center', maxWidth: '600px'}}>
  <svg viewBox="0 0 580 180" width="100%" role="img" aria-label="2段階学習。事前学習は文脈長8192、中間学習で文脈長を100万まで段階的に拡張">
    <rect x="40" y="50" width="200" height="56" rx="6" fill="#3B82F6" fillOpacity="0.1" stroke="#3B82F6" strokeWidth="1.5" />
    <text x="140" y="74" fontSize="12.5" fill="currentColor" textAnchor="middle" fontWeight="600">1. 事前学習 Pretraining</text>
    <text x="140" y="93" fontSize="10" fill="currentColor" fillOpacity="0.8" textAnchor="middle">文脈長 8,192 / 遺伝子領域を重視</text>
    <line x1="244" y1="78" x2="296" y2="78" stroke="currentColor" strokeOpacity="0.55" strokeWidth="1.6" />
    <polygon points="300,78 291,73 291,83" fill="currentColor" fillOpacity="0.55" />
    <rect x="304" y="50" width="236" height="56" rx="6" fill="#10B981" fillOpacity="0.1" stroke="#10B981" strokeWidth="1.5" />
    <text x="422" y="74" fontSize="12.5" fill="currentColor" textAnchor="middle" fontWeight="600">2. 中間学習 Midtraining</text>
    <text x="422" y="93" fontSize="10" fill="currentColor" fillOpacity="0.8" textAnchor="middle">文脈長を 100 万まで段階的に拡張</text>
    <text x="140" y="34" fontSize="10" fill="currentColor" fillOpacity="0.75" textAnchor="middle">機能的な遺伝要素を学ぶ</text>
    <text x="422" y="34" fontSize="10" fill="currentColor" fillOpacity="0.75" textAnchor="middle">要素間の長距離の関係を学ぶ</text>
    <text x="40" y="138" fontSize="11" fill="currentColor" fillOpacity="0.6">文脈長:</text>
    <text x="140" y="138" fontSize="11" fill="#3B82F6" textAnchor="middle">8k</text>
    <text x="300" y="138" fontSize="11" fill="currentColor" fillOpacity="0.6" textAnchor="middle">→</text>
    <text x="350" y="138" fontSize="10.5" fill="currentColor" fillOpacity="0.7" textAnchor="middle">32k → 128k → 256k → 512k →</text>
    <text x="500" y="138" fontSize="12" fill="#10B981" textAnchor="middle" fontWeight="600">1M</text>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>2段階学習。短い文脈で基礎を学んでから、文脈長を100万まで伸ばして長距離依存を学習する</figcaption>
</figure>

- **第1段階：事前学習（pretraining）** — 文脈長 **8,192 トークン**。遺伝子領域（genic window）を重視した重み付けで、**機能的な遺伝要素** を学びます。
- **第2段階：中間学習（midtraining）** — 文脈長を段階的に **100 万トークン** まで拡張。エクソン・プロモーター・遺伝子といった **要素どうしの長距離の関係** を学びます。

:::tip[これは自然言語 LLM のベストプラクティスと同じ]

「まず短い文脈で効率よく事前学習し、あとから文脈長を拡張する」のは、Llama 3 など自然言語 LLM の長文脈化で標準的に使われる手法です。最初から長文脈で学習するより、**効率も最終品質も良い**ことが知られています。Evo 2 はこの知見をゲノムにそのまま適用しています。

:::

## 5. 長文脈は本当に機能するのか：Needle-in-a-Haystack

文脈長を 100 万にしても、モデルが実際にその全体を「読めて」いなければ意味がありません。これを検証するのが、自然言語 LLM でもおなじみの **Needle-in-a-Haystack（干し草の中の針）** テストです（Fig. 1i）。

- **干し草（haystack）**：100 万塩基のランダムな DNA
- **針（needle）**：その中のどこかに隠した 100 塩基の特定配列

Evo 2 はこの針を発見し、その値を予測できるか——というタスクです。結果、Evo 2 は **100 万トークンの文脈全体から針を正しく検索（recall）** できました。これは、文脈ウィンドウ全体の情報を実際に活用できていることの確認になります（retrieval スコア > 0.8 で有意、$P < 0.001$）。

## 6. トークン化とバイオセーフティ

最後に、入力の作り方と安全設計に触れます。

**単一ヌクレオチド・トークン化** — 自然言語 LLM は BPE などで単語を **サブワード** に分割しますが、Evo 2 は **1 塩基（A・T・G・C）を 1 トークン** とします。語彙がわずか4種類なので塩基単位が自然であり、なにより **1 塩基の違い（点変異）を捉える** には単一ヌクレオチド解像度が不可欠です。これが後の変異効果予測の精度を支えます。

**バイオセーフティのためのデータ除外** — Evo 1 と同様、**真核生物（ヒトを含む）に感染するウイルスのゲノムを学習データから除外** しています。これは、危険なヒト病原体ウイルスを設計・操作する能力を、公開モデルに持たせないための安全策です。実際この除外により、Evo 2 は真核ウイルス配列に対して **高いパープレキシティ（言語モデリング性能が低い）** を示し（Extended Data Fig. 2a）、意図通りに機能しています。安全性の詳細は [06. クロマチン設計と考察](./06-chromatin-design.md) で扱います。

## 7. まとめ

- 100 万文脈を実現するため、$O(n^2)$ の Attention に頼らない **StripedHyena 2**（SE/MR/LI の畳み込み＋Attention のハイブリッド）を採用。Transformer 比で最大 3 倍高速。
- **OpenGenome2**（8.8 兆 nt 超、全ドメイン）を、機能要素を重視して **キュレーション・重み付け**。
- **事前学習（8k）→ 中間学習（1M）** の2段階で、分子から個体までの長さスケールを学習。
- **単一ヌクレオチド解像度** のトークン化と、**真核ウイルス除外** による安全設計。

これらの土台の上で、Evo 2 は何を「学んだ」のでしょうか。次の [03. 変異効果予測](./03-variant-prediction.md) では、学習された尤度が **進化的制約** を捉え、ゼロショットで変異の影響を予測できることを見ていきます。
