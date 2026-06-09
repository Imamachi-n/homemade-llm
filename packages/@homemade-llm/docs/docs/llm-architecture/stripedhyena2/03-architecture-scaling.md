---
sidebar_position: 3
title: "03. アーキテクチャ設計とスケーリング"
---

# 03. アーキテクチャ設計とスケーリング

[02](./02-operators.md)で見た3つの Hyena 演算子を、**どう積み重ねるか（block layout）**、**どう効率化するか（filter grouping）**、**どこまで文脈を伸ばせるか（context extension）**、そして **実際どれだけ速いか（throughput）** を見ていきます。

## 1. ブロックレイアウト：演算子をどう積むか

マルチハイブリッドは、SE・MR・LI・MHA（multi-head attention）を **縞状（striped）に積み重ねて** 全体を構成します。論文は、7B のマルチハイブリッドを OpenGenome2 の 400B トークンで学習し、レイアウトを変えて事前学習の品質（perplexity, PPL）を比較しました。

| ブロックレイアウト | PPL@400B（低いほど良い） |
| --- | --- |
| MHA-MHA-MHA（純 Transformer 相当） | 3.09 |
| LI-LI-LI（純・長畳み込み） | 2.87 |
| SE-SE-LI | 2.88 |
| **SE-MR-LI** | **2.83** ✅ |

ここから2つの重要な知見が得られます。

- **SE-MR-LI が最良**。短・中・長の異なる距離スケールを組み合わせるのが効く。
- **純・長畳み込み（LI-LI-LI）は SE-SE-LI で置き換えても品質ほぼ同じ** で、スループットは大きく向上する。つまり「全部を長い暗黙フィルタにする」必要はない。

実際の StripedHyena 2 は、この **SE-MR-LI を基本ブロックとして深さ分だけ繰り返し**（7B なら深さ 32）、さらに **5 つの MHA を畳み込みブロックに interleave** します。

<figure style={{margin: '1.5rem auto', textAlign: 'center', maxWidth: '620px'}}>
  <svg viewBox="0 0 600 130" width="100%" role="img" aria-label="SE-MR-LIを基本単位として繰り返し、合間にMHAを挟むブロックレイアウト">
    <g fontSize="10" textAnchor="middle">
      <rect x="14" y="40" width="30" height="40" rx="3" fill="#3B82F6" fillOpacity="0.15" stroke="#3B82F6" strokeWidth="1.2" /><text x="29" y="64" fill="currentColor">SE</text>
      <rect x="46" y="40" width="30" height="40" rx="3" fill="#10B981" fillOpacity="0.15" stroke="#10B981" strokeWidth="1.2" /><text x="61" y="64" fill="currentColor">MR</text>
      <rect x="78" y="40" width="30" height="40" rx="3" fill="#8B5CF6" fillOpacity="0.15" stroke="#8B5CF6" strokeWidth="1.2" /><text x="93" y="64" fill="currentColor">LI</text>
      <rect x="110" y="40" width="30" height="40" rx="3" fill="#3B82F6" fillOpacity="0.15" stroke="#3B82F6" strokeWidth="1.2" /><text x="125" y="64" fill="currentColor">SE</text>
      <rect x="142" y="40" width="30" height="40" rx="3" fill="#10B981" fillOpacity="0.15" stroke="#10B981" strokeWidth="1.2" /><text x="157" y="64" fill="currentColor">MR</text>
      <rect x="174" y="40" width="30" height="40" rx="3" fill="#8B5CF6" fillOpacity="0.15" stroke="#8B5CF6" strokeWidth="1.2" /><text x="189" y="64" fill="currentColor">LI</text>
      <rect x="206" y="40" width="40" height="40" rx="3" fill="#EF4444" fillOpacity="0.15" stroke="#EF4444" strokeWidth="1.2" /><text x="226" y="64" fill="currentColor">MHA</text>
      <rect x="248" y="40" width="30" height="40" rx="3" fill="#3B82F6" fillOpacity="0.15" stroke="#3B82F6" strokeWidth="1.2" /><text x="263" y="64" fill="currentColor">SE</text>
      <rect x="280" y="40" width="30" height="40" rx="3" fill="#10B981" fillOpacity="0.15" stroke="#10B981" strokeWidth="1.2" /><text x="295" y="64" fill="currentColor">MR</text>
      <rect x="312" y="40" width="30" height="40" rx="3" fill="#8B5CF6" fillOpacity="0.15" stroke="#8B5CF6" strokeWidth="1.2" /><text x="327" y="64" fill="currentColor">LI</text>
      <text x="364" y="64" fill="currentColor" fillOpacity="0.6" fontSize="14">…</text>
      <rect x="386" y="40" width="40" height="40" rx="3" fill="#EF4444" fillOpacity="0.15" stroke="#EF4444" strokeWidth="1.2" /><text x="406" y="64" fill="currentColor">MHA</text>
    </g>
    <text x="220" y="104" fontSize="9.5" fill="currentColor" fillOpacity="0.7" textAnchor="middle">SE-MR-LI を深さ分（7B なら 32 層）繰り返し、合計 5 つの MHA を散りばめる</text>
    <g fontSize="9" fill="currentColor" fillOpacity="0.8">
      <rect x="446" y="34" width="11" height="11" fill="#3B82F6" fillOpacity="0.5" /><text x="461" y="43" textAnchor="start">SE</text>
      <rect x="446" y="48" width="11" height="11" fill="#10B981" fillOpacity="0.5" /><text x="461" y="57" textAnchor="start">MR</text>
      <rect x="446" y="62" width="11" height="11" fill="#8B5CF6" fillOpacity="0.5" /><text x="461" y="71" textAnchor="start">LI</text>
      <rect x="446" y="76" width="11" height="11" fill="#EF4444" fillOpacity="0.5" /><text x="461" y="85" textAnchor="start">MHA</text>
    </g>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>SE-MR-LI を基本単位に積み、要所に MHA を挟む縞状レイアウト（論文 Fig. 2.1・Table 2.1 に対応）</figcaption>
</figure>

:::note[新しいドメインでは ablation を]

SE-MR-LI は安定したベースラインですが、論文は「フィルタ長・減衰の強さ・初期化を変えるなら、新しいタスク/ドメインでは block layout を ablation（比較検証）すべき」と注意しています。万能の固定解ではなく、設計の出発点という位置づけです。

:::

## 2. フィルタのグループ化：GEMV から GEMM へ

StripedHyena 2 は、入力依存畳み込みに **グループ化（grouping）** という設計を採り入れます。フィルタを **チャネルのグループ全体で共有** するのです。グループ $\mathcal{G}$（サイズ $d_g$）に属するチャネル $\alpha$ は、同じフィルタ $h^{\mathcal{G}}$ で畳み込まれます。

$$
\forall \alpha \in \mathcal{G}: \quad y^\alpha_t = \sum_{j=0}^{t} h^{\mathcal{G}}_{t-j}\, x^\alpha_j
$$

なぜこれが効くのか。グループ化により、離散畳み込みを **GEMV（行列×ベクトル）の集まりではなく GEMM（行列×行列）の系列として表現** できるようになります。GEMM は GPU の **テンソルコア** が最も得意とする演算で、ここが高速化の土台です（詳細は [04](./04-hardware-algorithms.md)）。

<figure style={{margin: '1.5rem auto', textAlign: 'center', maxWidth: '560px'}}>
  <svg viewBox="0 0 560 170" width="100%" role="img" aria-label="グループ化なしは各チャネル個別フィルタでGEMV、グループ化ありは共有フィルタでGEMM">
    <text x="140" y="22" fontSize="11" fill="currentColor" textAnchor="middle">グループ化なし</text>
    <g strokeWidth="2.4">
      <line x1="40" y1="42" x2="40" y2="112" stroke="#3B82F6" /><line x1="70" y1="42" x2="70" y2="112" stroke="#10B981" />
      <line x1="100" y1="42" x2="100" y2="112" stroke="#EF4444" /><line x1="130" y1="42" x2="130" y2="112" stroke="#8B5CF6" />
      <line x1="160" y1="42" x2="160" y2="112" stroke="#F59E0B" /><line x1="190" y1="42" x2="190" y2="112" stroke="#06B6D4" />
      <line x1="220" y1="42" x2="220" y2="112" stroke="#EC4899" /><line x1="250" y1="42" x2="250" y2="112" stroke="#84CC16" />
    </g>
    <text x="145" y="132" fontSize="9.5" fill="currentColor" fillOpacity="0.7" textAnchor="middle">各チャネルが別フィルタ</text>
    <text x="145" y="150" fontSize="10.5" fill="#EF4444" textAnchor="middle" fontWeight="600">GEMV（テンソルコア活用しにくい）</text>
    <text x="416" y="22" fontSize="11" fill="currentColor" textAnchor="middle">グループ化あり</text>
    <g strokeWidth="2.4">
      <line x1="316" y1="42" x2="316" y2="112" stroke="#3B82F6" /><line x1="346" y1="42" x2="346" y2="112" stroke="#3B82F6" />
      <line x1="376" y1="42" x2="376" y2="112" stroke="#3B82F6" /><line x1="406" y1="42" x2="406" y2="112" stroke="#3B82F6" />
      <line x1="436" y1="42" x2="436" y2="112" stroke="#10B981" /><line x1="466" y1="42" x2="466" y2="112" stroke="#10B981" />
      <line x1="496" y1="42" x2="496" y2="112" stroke="#10B981" /><line x1="526" y1="42" x2="526" y2="112" stroke="#10B981" />
    </g>
    <text x="361" y="38" fontSize="8.5" fill="currentColor" fillOpacity="0.6" textAnchor="middle">グループ1（共有）</text>
    <text x="481" y="38" fontSize="8.5" fill="currentColor" fillOpacity="0.6" textAnchor="middle">グループ2（共有）</text>
    <text x="421" y="132" fontSize="9.5" fill="currentColor" fillOpacity="0.7" textAnchor="middle">グループ内で同じフィルタを共有</text>
    <text x="421" y="150" fontSize="10.5" fill="#10B981" textAnchor="middle" fontWeight="600">GEMM（テンソルコアをフル活用）</text>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>フィルタのグループ化。共有することで畳み込みを GEMM にまとめ、テンソルコアで高速化できる</figcaption>
</figure>

しかも **品質への影響は最小** です。グループサイズ 1（共有なし）と 16 で収束に差はなく、64 を超えると小さな劣化が出る程度。**ほぼ無料で効率を得られる** 設計です（なお、これは従来の grouped CNN とは別物で、グループ内でチャネルを混ぜるわけではありません）。

## 3. 文脈拡張：8k から 1M へ

StripedHyena 2 は、まず短い文脈で base モデルを学習し（7B なら 2T トークン・**8192 文脈**＝ [Evo 2 7B](../../biology-llm-applications/evo2/02-architecture.md)）、その後 **midtraining で 100 万トークンまで文脈を拡張** します。拡張には rotary attention 向けの技術——**position interpolation（PI）** と **adjusted base frequency（ABF）**、およびその組み合わせ——を流用します。

| 文脈長 | 検証 perplexity（PI + ABF, 7B） |
| --- | --- |
| 32k | 2.782 |
| 65k | 2.763 |
| 131k | 2.748 |
| 262k | 2.707 |
| 524k | 2.663 |
| **1M** | **2.597** |

文脈を伸ばしても perplexity の悪化はなく（むしろ低下）、**すべてのモデルが目標の最大文脈長で in-context recall を達成** しました（[needle-in-a-haystack](../../biology-llm-applications/evo2/02-architecture.md)）。

## 4. スケールでのスループット

[02](./02-operators.md)の FIR 演算子（SE・MR）のおかげで、StripedHyena 2 は前世代ハイブリッドや Transformer に対して一貫した高速化を示します。

- 7B・40B のいずれでも、H100 クラスタで最適化 Transformer 比 **1.2〜2.9 倍** 高速（dense 層・正規化層は FP8 精度）。
- **短い系列長でも** Transformer と前世代 StripedHyena の両方より高速。
- 40B でピーク MFU（Model FLOPs Utilization）は **16K 文脈で約 34%**。

:::note[長文脈で MFU が下がるのは「悪いこと」ではない]

長い系列長では、ハイブリッドの MFU は下がります。しかしこれは、**subquadratic（準二次）なスケーリングによってモデル全体の FLOPS 自体が減る** ためです。Transformer は $O(n^2)$ で「無駄に」FLOPS を積み上げて MFU を稼いでいるとも言えます。実際の壁時計時間（iteration time）では StripedHyena 2 が一貫して速く、分散設定をさらに調整すれば長文脈での高速化はもっと伸びる余地があります。

:::

## 5. まとめ

- **SE-MR-LI** ブロックを基本に MHA を散りばめるレイアウトが、品質・速度のバランスで最良（PPL 2.83）。
- **フィルタのグループ化** で畳み込みを GEMM にまとめ、品質をほぼ落とさずテンソルコアで高速化。
- 8k で事前学習 → **1M まで文脈拡張**（PI・ABF）。perplexity は悪化せず recall も維持。
- 最適化 Transformer 比 **1.2〜2.9 倍**、短文脈でも高速。

「グループ化で畳み込みを GEMM にする」と述べましたが、その GEMM を **GPU でどう実装するか** が次の核心です。[04. ハードウェア対応の畳み込みカーネル](./04-hardware-algorithms.md)へ進みましょう。
