---
sidebar_position: 4
title: "04. ハードウェア対応の畳み込みカーネル"
---

# 04. ハードウェア対応の畳み込みカーネル

[03](./03-architecture-scaling.md)で「フィルタをグループ化すると畳み込みを GEMM にまとめられる」と述べました。このページでは、その GEMM を **GPU のテンソルコアで実際にどう計算するか** ——論文の核心である **2段階ブロックアルゴリズム** を掘り下げます。

## 1. 畳み込みは Toeplitz 行列の積

まず出発点。**離散畳み込みは Toeplitz 行列との行列積と数学的に等価** です。長さ $\ell_h$ の因果 FIR フィルタ $h$ を、長さ $\ell$ の入力 $x$（通常 $\ell \gg \ell_h$）に適用した出力は、

$$
y_t = \sum_{k=0}^{t} h_{t-k}\, x_k \qquad (h_k = 0 \text{ for } k < 0 \text{ or } k \ge \ell_h)
$$

これを行列の形に書くと、フィルタ係数が斜めに並ぶ **下三角 Toeplitz 行列** $T$ との積になります。

$$
\mathbf{y} = T\mathbf{x}, \qquad
T = \begin{pmatrix}
h_0 & 0 & 0 & \cdots \\
h_1 & h_0 & 0 & \cdots \\
h_2 & h_1 & h_0 & \cdots \\
\vdots & \ddots & \ddots & \ddots
\end{pmatrix}
$$

StripedHyena 2 の演算子（SE/MR/LI）はいずれも **grouped depthwise convolution** ですが、従来 CNN 向けの実装（im2col・Winograd）はこれに最適化されていません。長いフィルタは FFT ベースになりますが、ハードウェア利用率が低い。そこで論文は、**filter grouping を活かした direct な multi-pass ブロックアルゴリズム** を採用します。

## 2. ブロック畳み込み：長い系列を分割する

古典的なデジタル信号処理の **ブロック畳み込み（block convolution）** を使います。入力と出力を **サイズ $\ell_b$ のチャンク** に分割し、$\ell_b \times \ell_b$ の部分ブロックごとに掛けます。

$$
\hat{y}_n = \sum_{k} H_{n-k}\, \hat{x}_k
$$

ここで $\hat{x}_k, \hat{y}_n$ は $k, n$ 番目の入力・出力チャンク、$H_k$ は Toeplitz $T$ を $\ell_b \times \ell_b$ に分割した部分行列です。ポイントは、**フィルタのサポート（長さ $\ell_h$）を超えた部分ブロックはすべてゼロになり、スキップできる** こと。$\ell_h \ll \ell$ のとき、これが大きな効率になります。

## 3. 2段階ブロックアルゴリズム

SE・MR では $\ell_h \ll \ell$ です。さらに **$\ell_h \le 2\ell_b$**（フィルタ長がブロックの2倍以内）なら、特に効率的な **2段階（two-stage）アルゴリズム** が使えます。このとき Toeplitz $T$ は、**対角ブロック部（$H_0$）** と **オフ対角部（$H_1$）** の2つだけに分解できます。

<figure style={{margin: '1.5rem auto', textAlign: 'center', maxWidth: '520px'}}>
  <svg viewBox="0 0 500 180" width="100%" role="img" aria-label="Toeplitz行列は対角ブロックH0とオフ対角H1の2段階に分解できる">
    <text x="70" y="20" fontSize="11" fill="currentColor" textAnchor="middle">T（全体）</text>
    <rect x="30" y="30" width="55" height="55" fill="#3B82F6" fillOpacity="0.5" stroke="#3B82F6" strokeWidth="1.2" /><text x="57" y="62" fontSize="12" fill="currentColor" textAnchor="middle">H₀</text>
    <rect x="85" y="30" width="55" height="55" fill="currentColor" fillOpacity="0.04" stroke="currentColor" strokeOpacity="0.3" strokeWidth="1" /><text x="112" y="62" fontSize="11" fill="currentColor" fillOpacity="0.5" textAnchor="middle">0</text>
    <rect x="30" y="85" width="55" height="55" fill="#10B981" fillOpacity="0.5" stroke="#10B981" strokeWidth="1.2" /><text x="57" y="117" fontSize="12" fill="currentColor" textAnchor="middle">H₁</text>
    <rect x="85" y="85" width="55" height="55" fill="#3B82F6" fillOpacity="0.5" stroke="#3B82F6" strokeWidth="1.2" /><text x="112" y="117" fontSize="12" fill="currentColor" textAnchor="middle">H₀</text>
    <text x="165" y="90" fontSize="20" fill="currentColor" textAnchor="middle">=</text>
    <text x="245" y="20" fontSize="10.5" fill="currentColor" textAnchor="middle">第1段（対角）</text>
    <rect x="205" y="30" width="55" height="55" fill="#3B82F6" fillOpacity="0.5" stroke="#3B82F6" strokeWidth="1.2" /><text x="232" y="62" fontSize="12" fill="currentColor" textAnchor="middle">H₀</text>
    <rect x="260" y="85" width="55" height="55" fill="#3B82F6" fillOpacity="0.5" stroke="#3B82F6" strokeWidth="1.2" /><text x="287" y="117" fontSize="12" fill="currentColor" textAnchor="middle">H₀</text>
    <rect x="205" y="30" width="110" height="110" fill="none" stroke="currentColor" strokeOpacity="0.2" strokeWidth="1" />
    <text x="340" y="90" fontSize="18" fill="currentColor" textAnchor="middle">+</text>
    <text x="425" y="20" fontSize="10.5" fill="currentColor" textAnchor="middle">第2段（オフ対角）</text>
    <rect x="385" y="85" width="55" height="55" fill="#10B981" fillOpacity="0.5" stroke="#10B981" strokeWidth="1.2" /><text x="412" y="117" fontSize="12" fill="currentColor" textAnchor="middle">H₁</text>
    <rect x="385" y="30" width="110" height="110" fill="none" stroke="currentColor" strokeOpacity="0.2" strokeWidth="1" />
    <text x="250" y="165" fontSize="9.5" fill="currentColor" fillOpacity="0.7" textAnchor="middle">H₀＝現在チャンク内のフィルタ点、H₁＝前チャンクからはみ出す（spillover）点</text>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>2段階分解。$T$ は対角の $H_0$（第1段）とオフ対角の $H_1$（第2段）だけになる（論文 式8）</figcaption>
</figure>

具体例として $\ell = 6$（系列長）、$\ell_h = 4$（フィルタ長）、$\ell_b = 3$（ブロックサイズ）を取ると、

$$
H_0 = \begin{pmatrix} h_0 & 0 & 0 \\ h_1 & h_0 & 0 \\ h_2 & h_1 & h_0 \end{pmatrix}, \qquad
H_1 = \begin{pmatrix} h_3 & h_2 & h_1 \\ 0 & h_3 & h_2 \\ 0 & 0 & h_3 \end{pmatrix}
$$

- **$H_0$** は「現在のチャンク $\hat{X}_n$ に整列するフィルタ点」を担当。
- **$H_1$** は「前のチャンク $\hat{X}_{n-1}$ からはみ出してくる（spillover）点」、つまりチャンク境界をまたぐ係数を担当。

各出力チャンクは、たった2つの行列積で計算できます。

$$
\hat{Y}_n = H_0\, \hat{X}_n + H_1\, \hat{X}_{n-1} \qquad (\hat{X}_{-1} = 0)
$$

$\ell_h \le 2\ell_b$ という条件が、「$H_1$ より先のオフ対角ブロックは出てこない」ことを保証しています。

<figure style={{margin: '1.5rem auto', textAlign: 'center', maxWidth: '560px'}}>
  <svg viewBox="0 0 560 140" width="100%" role="img" aria-label="出力チャンクYnは、現在チャンクにH0、前チャンクにH1を掛けて足す">
    <rect x="20" y="40" width="70" height="30" rx="4" fill="currentColor" fillOpacity="0.06" stroke="currentColor" strokeOpacity="0.5" strokeWidth="1.2" /><text x="55" y="60" fontSize="11" fill="currentColor" textAnchor="middle">X̂ₙ₋₁</text>
    <rect x="100" y="40" width="70" height="30" rx="4" fill="#3B82F6" fillOpacity="0.1" stroke="#3B82F6" strokeWidth="1.3" /><text x="135" y="60" fontSize="11" fill="currentColor" textAnchor="middle">X̂ₙ</text>
    <line x1="135" y1="70" x2="135" y2="92" stroke="#3B82F6" strokeOpacity="0.7" strokeWidth="1.3" /><polygon points="135,96 130,87 140,87" fill="#3B82F6" fillOpacity="0.7" />
    <text x="165" y="86" fontSize="10" fill="#3B82F6">× H₀</text>
    <path d="M 55 70 C 55 100, 110 100, 130 98" fill="none" stroke="#10B981" strokeOpacity="0.7" strokeWidth="1.3" /><polygon points="133,98 124,95 127,104" fill="#10B981" fillOpacity="0.7" />
    <text x="70" y="114" fontSize="10" fill="#10B981">× H₁（spillover）</text>
    <rect x="100" y="98" width="70" height="30" rx="4" fill="#EF4444" fillOpacity="0.1" stroke="#EF4444" strokeWidth="1.3" /><text x="135" y="118" fontSize="11" fill="currentColor" textAnchor="middle">Ŷₙ</text>
    <text x="220" y="86" fontSize="12" fill="currentColor">Ŷₙ = H₀ X̂ₙ + H₁ X̂ₙ₋₁</text>
    <text x="220" y="108" fontSize="9.5" fill="currentColor" fillOpacity="0.7">→ 2つの GEMM ＋ 加算だけ</text>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>各出力チャンクは「現在チャンク×H₀」と「前チャンク×H₁」の和。境界の spillover を H₁ が引き受ける</figcaption>
</figure>

## 4. テンソルコアへ：GEMM 化と再利用

この分解が GPU で速い理由は3つあります。

1. **$H_0, H_1$ は全チャンク・全チャネルで不変** — フィルタが決まれば $H_0, H_1$ は固定。一度オンチップメモリ（共有メモリ）にロードすれば、すべてのチャンクで再利用できます。
2. **[03](./03-architecture-scaling.md) のグループ化と組み合わさる** — グループ内の全チャネルが同じ $H_0, H_1$ を共有するので、小さな GEMV ではなく **大きな GEMM** にまとめられます。テンソルコアのサイズ $d_g$ と $\ell_b = d_g$ を揃えれば、$\hat{Y}_n = H_0\hat{X}_n + H_1\hat{X}_{n-1}$ は **2つのフル GEMM** として実行できます。
3. **「現在チャンク」と「前チャンク」を並列／パイプライン化** — 第1段（$H_0$）と第2段（$H_1$）は独立に走らせられます。

論文の前進カーネル（Algorithm 1）を擬似コードで示すと、こうなります。

```text
入力 v, q, k を ℓ_b × d_g のブロックに分割
for ブロック i = 0 .. ⌈ℓ/ℓ_b⌉ − 1:
    v_i, q_i, k_i, H_0, H_1 をオンチップメモリにロード
    y_i = H_0 · v_i              # 第1 GEMM（対角・block-diagonal）
    if i > 0:
        y_i += H_1 · v_{i-1}     # 第2 GEMM（オフ対角・spillover）
    y_i = q_i ⊙ y_i              # ゲーティング
return y
```

:::tip[なぜ GEMM 化がそんなに効くのか]

GPU のテンソルコアは **密な行列積（GEMM）** に特化したユニットで、桁違いのスループットを出します。depthwise 畳み込みを素朴に書くと小さな行列ベクトル積（GEMV）の山になり、テンソルコアを活かせません。**「グループ化 → GEMM 化 → テンソルコア」** という一連の co-design が、StripedHyena 2 の速さの正体です。これは [03](./03-architecture-scaling.md) の filter grouping と一体の工夫です。

:::

## 5. 実測：演算子レベルのスループット

この2段階アプローチの効果は実測でも明確です。Hyena-MR（フィルタ長 128）を、素朴な PyTorch 畳み込み（`F.conv1d`）と比べると、2段階ブロックカーネルは **レイテンシ・スループットともに大幅に改善** します。

さらに演算子レベルで他の主要演算子と比較すると、**Hyena-SE・Hyena-MR は、最適化された MHA（FlashAttention2・SDPA）や Mamba2・xLSTM・DeltaNet を上回るスループット** を、バッチサイズ1・幅 4096 の設定で達成します。

<figure style={{margin: '1.5rem auto', textAlign: 'center', maxWidth: '440px'}}>
  <svg viewBox="0 0 420 180" width="100%" role="img" aria-label="演算子別スループット。Hyena-SEとHyena-MRが他の演算子より高い">
    <line x1="30" y1="150" x2="410" y2="150" stroke="currentColor" strokeOpacity="0.35" strokeWidth="1.2" />
    <text x="14" y="40" fontSize="9" fill="currentColor" fillOpacity="0.7">TFLOPS/s</text>
    <rect x="40" y="30" width="34" height="120" fill="#3B82F6" fillOpacity="0.6" /><text x="57" y="164" fontSize="8.5" fill="currentColor" textAnchor="middle">Hyena-SE</text>
    <rect x="92" y="57" width="34" height="93" fill="#10B981" fillOpacity="0.6" /><text x="109" y="164" fontSize="8.5" fill="currentColor" textAnchor="middle">Hyena-MR</text>
    <rect x="150" y="99" width="34" height="51" fill="currentColor" fillOpacity="0.3" /><text x="167" y="164" fontSize="8.5" fill="currentColor" textAnchor="middle">SDPA</text>
    <rect x="202" y="116" width="34" height="34" fill="currentColor" fillOpacity="0.3" /><text x="219" y="164" fontSize="8.5" fill="currentColor" textAnchor="middle">FA2</text>
    <rect x="254" y="119" width="34" height="31" fill="currentColor" fillOpacity="0.3" /><text x="271" y="164" fontSize="8.5" fill="currentColor" textAnchor="middle">Mamba2</text>
    <rect x="306" y="126" width="34" height="24" fill="currentColor" fillOpacity="0.3" /><text x="323" y="164" fontSize="8.5" fill="currentColor" textAnchor="middle">xLSTM</text>
    <rect x="358" y="126" width="34" height="24" fill="currentColor" fillOpacity="0.3" /><text x="375" y="164" fontSize="8" fill="currentColor" textAnchor="middle">DeltaNet</text>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>演算子別スループットの概念図（H100・幅4096）。Hyena-SE/MR が attention・SSM 系を上回る（論文 Fig. 3.2・B.4 の趣旨）</figcaption>
</figure>

## 6. まとめ

- 畳み込みは **Toeplitz 行列の積**。SE/MR は $\ell_h \le 2\ell_b$ なら **2段階（$H_0$＋$H_1$）** に分解できる。
- $\hat{Y}_n = H_0\hat{X}_n + H_1\hat{X}_{n-1}$ という **2つの GEMM** に落ち、$H_0/H_1$ の再利用とグループ化で **テンソルコアをフル活用**。
- 結果、Hyena-SE/MR は MHA・Mamba2・xLSTM・DeltaNet を上回るスループット。

ここまでは1台の GPU 内の話でした。100 万トークンを **複数 GPU に分散** して学習するには、また別の工夫が要ります。最後の [05. 長系列の分散学習](./05-context-parallelism.md) で、context parallelism と FFT 畳み込みを見ていきます。
