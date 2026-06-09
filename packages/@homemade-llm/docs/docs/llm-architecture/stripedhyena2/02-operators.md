---
sidebar_position: 2
title: "02. 3つの Hyena 演算子"
---

# 02. 3つの Hyena 演算子

StripedHyena 2 の心臓部は、**入力依存の畳み込み（input-dependent convolution）** にもとづく **Hyena 演算子** です。このページでは、Hyena の基本構造を数式で押さえたうえで、距離スケールの異なる3つの変種——**SE・MR・LI**——を見ていきます。

## 1. 入力依存の畳み込みという発想

Self-Attention は、全トークン対の関連度を $QK^\top$ で計算します（[Transformer の実装](../../llm-from-scratch/chapter2.md)参照）。これは強力ですが $O(n^2)$ です。

Hyena の発想は、**「全対比較」の代わりに、長い畳み込みフィルタとゲーティングで系列を混ぜる** ことです。畳み込みは高速フーリエ変換（FFT）を使えば $O(n \log n)$、フィルタが短ければさらに軽く計算できます。鍵は、フィルタを **入力に応じて変調する（ゲーティングする）** ことで、固定フィルタの CNN にはない表現力を得る点です。

## 2. Hyena 演算子の構造

元の Hyena（Poli et al., 2023）の演算子は、**3つの「特徴量」$q, k, v$ を作り、ゲーティングと内部畳み込みで混ぜて、射影する** という構造です。直感的には次のように書けます（$*$ は畳み込み、$\odot$ は要素ごとの積）。

$$
q = h_T * (xW), \qquad k = h_H * (xU), \qquad v = h_K * (xP)
$$

$$
y = \Bigl[\, q \,\odot\, \bigl(h_G * (k \odot v)\bigr) \Bigr]\, M
$$

つまり、入力 $x$ を **dense 行列 $W, U, P$ で線形射影** し、それぞれ **短い畳み込み $h_T, h_H, h_K$ で特徴量化（featurization）** して $q, k, v$ を作ります。次に $k \odot v$ を **内部フィルタ $h_G$ で畳み込み**、$q$ で **ゲーティング** し、最後に **dense 行列 $M$ で射影** して出力 $y$ を得ます。

:::note[論文の正確な表記（添字付き）]

論文では、空間（チャネル）次元をギリシャ文字の上付き $\alpha, \beta$、時間次元を下付き $t, t'$ で表し、次のように書きます。

$$
\begin{aligned}
q^\alpha_t &= \mathsf{T}^\alpha_{tt'}\,(x^\beta_{t'} W^{\beta\alpha}) &
k^\alpha_t &= \mathsf{H}^\alpha_{tt'}\,(x^\beta_{t'} U^{\beta\alpha}) \\
v^\alpha_t &= \mathsf{K}^\alpha_{tt'}\,(x^\beta_{t'} P^{\beta\alpha}) &
y^\alpha_t &= \bigl(q^\beta_t\, \mathsf{G}^\beta_{tt'}\, k^\beta_{t'}\, v^\beta_{t'}\bigr)\, M^{\beta\alpha}
\end{aligned}
$$

ここで $\mathsf{T}, \mathsf{H}, \mathsf{K}, \mathsf{G}$ は **Toeplitz 行列**（それぞれフィルタ $h_T, h_H, h_K, h_G$ による畳み込みに対応）、$W, U, P, M$ は **dense 行列**です。畳み込みが Toeplitz 行列の積として書けることは、[04. ハードウェア対応カーネル](./04-hardware-algorithms.md)で効率化の鍵になります。

:::

<figure style={{margin: '1.5rem auto', textAlign: 'center', maxWidth: '620px'}}>
  <svg viewBox="0 0 600 150" width="100%" role="img" aria-label="Hyena 演算子のデータフロー：入力を射影・短畳み込みで q,k,v 化し、内部畳み込みとゲーティングで混ぜ、射影して出力">
    <rect x="14" y="58" width="70" height="36" rx="5" fill="#3B82F6" fillOpacity="0.1" stroke="#3B82F6" strokeWidth="1.3" />
    <text x="49" y="80" fontSize="12" fill="currentColor" textAnchor="middle">入力 x</text>
    <line x1="84" y1="76" x2="112" y2="76" stroke="currentColor" strokeOpacity="0.5" strokeWidth="1.3" />
    <polygon points="116,76 107,71 107,81" fill="currentColor" fillOpacity="0.5" />
    <rect x="118" y="44" width="120" height="64" rx="5" fill="#10B981" fillOpacity="0.1" stroke="#10B981" strokeWidth="1.3" />
    <text x="178" y="64" fontSize="10.5" fill="currentColor" textAnchor="middle">射影 + 短畳み込み</text>
    <text x="178" y="80" fontSize="10.5" fill="currentColor" textAnchor="middle">（featurization）</text>
    <text x="178" y="98" fontSize="11" fill="currentColor" textAnchor="middle" fontWeight="600">q, k, v</text>
    <line x1="238" y1="76" x2="266" y2="76" stroke="currentColor" strokeOpacity="0.5" strokeWidth="1.3" />
    <polygon points="270,76 261,71 261,81" fill="currentColor" fillOpacity="0.5" />
    <rect x="272" y="44" width="148" height="64" rx="5" fill="#8B5CF6" fillOpacity="0.1" stroke="#8B5CF6" strokeWidth="1.3" />
    <text x="346" y="64" fontSize="10.5" fill="currentColor" textAnchor="middle">内部畳み込み h_G</text>
    <text x="346" y="80" fontSize="10.5" fill="currentColor" textAnchor="middle">＋ ゲーティング</text>
    <text x="346" y="98" fontSize="11" fill="currentColor" textAnchor="middle">q ⊙ (h_G * (k⊙v))</text>
    <line x1="420" y1="76" x2="448" y2="76" stroke="currentColor" strokeOpacity="0.5" strokeWidth="1.3" />
    <polygon points="452,76 443,71 443,81" fill="currentColor" fillOpacity="0.5" />
    <rect x="454" y="58" width="84" height="36" rx="5" fill="#EF4444" fillOpacity="0.1" stroke="#EF4444" strokeWidth="1.3" />
    <text x="496" y="74" fontSize="10.5" fill="currentColor" textAnchor="middle">射影 M</text>
    <text x="496" y="89" fontSize="10.5" fill="currentColor" textAnchor="middle">→ 出力 y</text>
    <text x="300" y="132" fontSize="9.5" fill="currentColor" fillOpacity="0.7" textAnchor="middle">どの畳み込みフィルタを使うかで Hyena-SE / MR / LI が分かれる</text>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>Hyena 演算子のデータフロー。「内部フィルタ $h_G$ をどう作るか」が3つの変種を分ける（論文 Fig. 2.1 に対応）</figcaption>
</figure>

元の Hyena では、外側のフィルタ $h_T, h_H, h_K$ は **明示的（explicit）**——CNN のように各要素が学習パラメータ——で、内部フィルタ $h_G$ は **暗黙的（implicit）**——基底関数やニューラルネットの出力として生成——でした。このため、Hyena 系は **long convolution operator（長畳み込み演算子）** とも総称されます。

論文の主張は、**「すべての畳み込みが長く暗黙的である必要はない」「演算子はターゲットのハードウェアで高速に動くよう設計すべき」** というものです。ここから3つの変種が生まれます。

## 3. 3つの演算子：SE / MR / LI

3つの変種は、**内部フィルタの長さと作り方** が違います。

<figure style={{margin: '1.5rem auto', textAlign: 'center', maxWidth: '520px'}}>
  <svg viewBox="0 0 520 170" width="100%" role="img" aria-label="SE・MR・LIのフィルタ長比較。SEは短く、MRは中程度、LIは系列全体">
    <text x="14" y="40" fontSize="11" fill="currentColor" fontWeight="600">SE</text>
    <text x="40" y="40" fontSize="9.5" fill="currentColor" fillOpacity="0.7">短い明示的（長さ 4〜7）</text>
    <rect x="180" y="28" width="34" height="16" rx="2" fill="#3B82F6" fillOpacity="0.6" />
    <text x="14" y="86" fontSize="11" fill="currentColor" fontWeight="600">MR</text>
    <text x="40" y="86" fontSize="9.5" fill="currentColor" fillOpacity="0.7">中程度・正則化（長さ 〜128）</text>
    <rect x="180" y="74" width="150" height="16" rx="2" fill="#10B981" fillOpacity="0.6" />
    <text x="14" y="132" fontSize="11" fill="currentColor" fontWeight="600">LI</text>
    <text x="40" y="132" fontSize="9.5" fill="currentColor" fillOpacity="0.7">長い暗黙的（系列全体）</text>
    <rect x="180" y="120" width="326" height="16" rx="2" fill="#8B5CF6" fillOpacity="0.6" />
    <text x="343" y="40" fontSize="9" fill="currentColor" fillOpacity="0.6">局所</text>
    <text x="343" y="86" fontSize="9" fill="currentColor" fillOpacity="0.6">中距離</text>
    <text x="513" y="114" fontSize="9" fill="currentColor" fillOpacity="0.6" textAnchor="end">系列全体を集約</text>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>3つの Hyena 演算子のフィルタ長。捉える距離スケールが異なり、これらを組み合わせる（＝マルチハイブリッド）</figcaption>
</figure>

### Hyena-LI（long implicit）

元の設計に最も近い変種です。外側のフィルタ $h_T, h_H, h_K$ は短く明示的なまま、内部フィルタを **実指数の線形結合** として暗黙的に生成します。

$$
h_t = \sum_{n=1}^{d} R_n\, \lambda_n^{\,t-1}, \qquad R_n, \lambda_n \in \mathbb{R}
$$

実数値の簡略版で、これにより **再帰的（recurrent）なパラメータ化に切り替えて、生成時に定数メモリ** にできます（SSM のように、状態を持って1トークンずつ生成できる）。系列全体にわたる長距離の情報集約を担います。

### Hyena-SE（short explicit）

すべての畳み込みが **短く明示的なフィルタ** の変種です。フィルタが短ければ（論文の実験では 14 未満、最終的なスケール実行では **4〜7**）、単純な明示的パラメータ化で十分収束します。

- 短系列を含む幅広い入力域で **speedup の鍵**。
- **局所的な multi-token recall** に優れる。
- テンソルコア向けの実装で、**全系列混合演算子の中で最高スループット**。
- **FFN（フィードフォワード層）の代替** としても使える。

### Hyena-MR（medium regularized）

数百長の明示的フィルタをもつ変種です。長い明示的畳み込みは最適化が難しいのですが、単純な **指数減衰の正則化** で収束させます。

$$
h_t = \hat{h}_t\, \lambda^{-\alpha t}
$$

ここで $\hat{h}_t$ が学習パラメータ、$\alpha$ はチャネルごとに掃引（sweep）されます。filter grouping（[03](./03-architecture-scaling.md)）とテンソルコア実装により、**linear attention や状態空間モデルより大幅に高速** です。論文は「**Hyena-MR は Hyena-LI に対して、sliding window attention が通常の attention に対するのと同じ関係**」と表現しています——全域を見る代わりに、限られた窓を効率的に見る、という位置づけです。

## 4. FIR フィルタと効率的な生成

Hyena-SE と Hyena-MR のフィルタは **FIR（有限インパルス応答, finite-impulse response）** です。フィルタ長が有限なので、**自己回帰生成のときに過去の有限個の要素だけ保持すればよく、定数メモリ** で済みます（sliding window attention と同様）。Hyena-LI も、前述のとおり再帰的パラメータ化に切り替えれば定数メモリで生成できます。

:::tip[LLM とのつながり：効率化アーキの「いいとこ取り」]

- **SE**（短い FIR）＝ 局所パターンに強く、最速。FFN の代替にもなる。
- **MR**（中程度 FIR）＝ sliding window attention 的な中距離。
- **LI**（長い暗黙）＝ SSM/RNN 的な全系列集約・定数メモリ生成。
- **Attention**＝ 長距離の精密 recall。

自然言語 LLM で個別に研究されてきた「効率化の各路線」を、**1つのアーキ内で役割分担させて組み合わせる** のがマルチハイブリッドの本質です。どれをどう積むか（block layout）が次の論点になります。

:::

## 5. まとめ

- Hyena 演算子は、**射影 + 短畳み込みで $q,k,v$ を作り、内部フィルタ $h_G$ とゲーティングで混ぜて射影** する構造。畳み込みは Toeplitz 行列の積。
- **SE（短・明示的）/ MR（中・正則化）/ LI（長・暗黙的）** の3変種が、それぞれ局所・中距離・全系列をカバーする。
- SE/MR は FIR で定数メモリ生成、LI も再帰化で定数メモリ。

では、これらの演算子をどう積み重ね、どう高速化するのか。次の [03. アーキテクチャ設計とスケーリング](./03-architecture-scaling.md) で、block layout・filter grouping・スループットを見ていきます。
