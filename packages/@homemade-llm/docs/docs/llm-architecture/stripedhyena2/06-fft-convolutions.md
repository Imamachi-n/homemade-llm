---
sidebar_position: 6
title: "06. FFT 畳み込みの数学と分散実装"
---

# 06. FFT 畳み込みの数学と分散実装

[05](./05-context-parallelism.md)で、Hyena-LI の長い畳み込みを **FFT** で計算し、それを **複数 GPU に分散** する話に触れました。このページは、その数学的な背景を **論文の付録（A.2.4〜A.3）に沿って完全に導出** する深掘り編です。FFT の仕組みから、分散 p2p FFT がなぜ追加通信なしで成立するのかまでを追います。

:::note[このページの位置づけ]

ここは「もっと詳しく知りたい人向け」の付録的な章です。FFT の仕組みを知らなくても [05](./05-context-parallelism.md) までで StripedHyena 2 の全体像は掴めます。ここでは、信号処理と分散計算が交わる美しい部分を、数式で丁寧に追います。

:::

## 1. なぜ FFT 畳み込みを分散したいのか

Hyena-LI（[02](./02-operators.md)）の内部フィルタは **系列長と同じだけ長い** ため、素朴な畳み込みは重く、FFT で計算します。ところが FFT は原理上 **系列全体** を必要とします。一方、100 万トークンの系列は1台の GPU に収まらず、複数デバイスに分割（[05](./05-context-parallelism.md) のコンテキスト並列）されています。

> 「系列全体が必要な FFT」を「系列が分割された状態」でどう計算するか?

これが本ページの主題です。結論を先に言うと、**FFT の分割統治構造そのものが、デバイス分割と一致する** ため、うまく設計すれば追加の全交換（a2a）なしに分散計算できます。

## 2. 離散フーリエ変換（DFT）

まず土台の **離散フーリエ変換（DFT）** です。長さ $l$ の系列 $x$ に対し、

$$
y(k) = \mathrm{DFT}_l(x) = \sum_{j=0}^{l-1} x(j)\,\omega_l^{\,jk}, \qquad \omega_l = e^{-2\pi i / l},\ \ 0 \le k \le l-1
$$

ここで $\omega_l$ は **回転因子（twiddle factor）** と呼ばれる複素数で、複素平面上で単位円を $l$ 等分する点です。$i$ は虚数単位（$i = \sqrt{-1}$）。

DFT は **行列とベクトルの積** として書けます。$l = 4$ なら、

$$
\begin{pmatrix} y(0)\\ y(1)\\ y(2)\\ y(3) \end{pmatrix}
=
\begin{pmatrix}
1 & 1 & 1 & 1 \\
1 & \omega & \omega^2 & \omega^3 \\
1 & \omega^2 & 1 & \omega^2 \\
1 & \omega^3 & \omega^2 & \omega
\end{pmatrix}
\begin{pmatrix} x(0)\\ x(1)\\ x(2)\\ x(3) \end{pmatrix}
$$

（$\omega_l^{\,jk} = \omega_l^{\,jk \bmod l}$ を使って指数を簡約しています。）この行列をそのまま掛けると、要素数 $l^2$ の演算が必要で **$O(l^2)$** です。

## 3. FFT：分割統治の魔法

**高速フーリエ変換（FFT）** は、同じ DFT を **$O(l\log l)$** で計算します。鍵は、$l$ 点 DFT を **2つの $l/2$ 点 DFT に分解** できることです。入力を前半 $x(j)$ と後半 $x(j + l/2)$ に分けると、

$$
\begin{aligned}
y(k) &= \mathrm{DFT}_{l/2}\bigl(x(j) + x(j + l/2)\bigr) \\
y(k+1) &= \mathrm{DFT}_{l/2}\bigl(\omega_l^{\,j}\,(x(j) - x(j + l/2))\bigr)
\end{aligned}
$$

つまり「前半と後半を **足したもの**」「**引いて回転因子を掛けたもの**」、それぞれに半分のサイズの DFT を適用すればよい。半分の DFT をさらに半分に……と再帰すると、$l = 2$ まで分割でき、全体が $O(l\log l)$ になります。これが「速い」フーリエ変換の正体です。

<figure style={{margin: '1.5rem auto', textAlign: 'center', maxWidth: '460px'}}>
  <svg viewBox="0 0 440 180" width="100%" role="img" aria-label="l点DFTが2つのl/2点DFTに分割される分割統治の構造">
    <rect x="170" y="14" width="100" height="34" rx="5" fill="currentColor" fillOpacity="0.06" stroke="currentColor" strokeOpacity="0.6" strokeWidth="1.3" />
    <text x="220" y="35" fontSize="11" fill="currentColor" textAnchor="middle">l 点 DFT</text>
    <line x1="200" y1="48" x2="120" y2="78" stroke="currentColor" strokeOpacity="0.4" strokeWidth="1.2" />
    <line x1="240" y1="48" x2="320" y2="78" stroke="currentColor" strokeOpacity="0.4" strokeWidth="1.2" />
    <text x="150" y="68" fontSize="9" fill="currentColor" fillOpacity="0.7">前半+後半</text>
    <text x="290" y="68" fontSize="9" fill="#EF4444" fillOpacity="0.85">(前半−後半)×ω</text>
    <rect x="60" y="80" width="120" height="34" rx="5" fill="#3B82F6" fillOpacity="0.1" stroke="#3B82F6" strokeWidth="1.3" />
    <text x="120" y="101" fontSize="11" fill="currentColor" textAnchor="middle">l/2 点 DFT</text>
    <rect x="260" y="80" width="120" height="34" rx="5" fill="#10B981" fillOpacity="0.1" stroke="#10B981" strokeWidth="1.3" />
    <text x="320" y="101" fontSize="11" fill="currentColor" textAnchor="middle">l/2 点 DFT</text>
    <text x="120" y="138" fontSize="9.5" fill="currentColor" fillOpacity="0.7" textAnchor="middle">偶数番の出力 y(k)</text>
    <text x="320" y="138" fontSize="9.5" fill="currentColor" fillOpacity="0.7" textAnchor="middle">奇数番の出力 y(k+1)</text>
    <text x="220" y="166" fontSize="9.5" fill="currentColor" fillOpacity="0.7" textAnchor="middle">これを l=2 まで再帰 → 全体で O(l log l)</text>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>FFT の分割統治。l 点 DFT を 2 つの l/2 点 DFT に分け、足し引き＋回転因子で結合する</figcaption>
</figure>

## 4. バタフライと bit-reversal（DiF / DiT）

この「2分割の1ステップ」を可視化したのが **バタフライ（butterfly）** です。蝶のように2本の線が交差することからこう呼ばれます。分解の仕方で2種類あります。

**DiF（周波数間引き, Decimation-in-Frequency）**：

$$
X = x + y, \qquad Y = (x - y)\,\omega^j
$$

**DiT（時間間引き, Decimation-in-Time）**：

$$
X = x + \omega^j y, \qquad Y = x - \omega^j y
$$

<figure style={{margin: '1.5rem auto', textAlign: 'center', maxWidth: '320px'}}>
  <svg viewBox="0 0 300 140" width="100%" role="img" aria-label="DiFバタフライ。入力x,yからX=x+y、Y=(x-y)ωを計算する">
    <circle cx="46" cy="38" r="4" fill="#3B82F6" /><text x="30" y="42" fontSize="12" fill="currentColor">x</text>
    <circle cx="46" cy="104" r="4" fill="#10B981" /><text x="30" y="108" fontSize="12" fill="currentColor">y</text>
    <line x1="50" y1="38" x2="248" y2="38" stroke="#3B82F6" strokeOpacity="0.6" strokeWidth="1.3" />
    <line x1="50" y1="104" x2="248" y2="104" stroke="#10B981" strokeOpacity="0.6" strokeWidth="1.3" />
    <line x1="50" y1="38" x2="200" y2="104" stroke="currentColor" strokeOpacity="0.4" strokeWidth="1.1" />
    <line x1="50" y1="104" x2="150" y2="38" stroke="currentColor" strokeOpacity="0.4" strokeWidth="1.1" />
    <circle cx="150" cy="38" r="9" fill="none" stroke="currentColor" strokeOpacity="0.6" strokeWidth="1.1" /><text x="150" y="42" fontSize="11" fill="currentColor" textAnchor="middle">+</text>
    <circle cx="200" cy="104" r="9" fill="none" stroke="currentColor" strokeOpacity="0.6" strokeWidth="1.1" /><text x="200" y="108" fontSize="11" fill="currentColor" textAnchor="middle">−</text>
    <circle cx="228" cy="104" r="11" fill="none" stroke="#EF4444" strokeOpacity="0.7" strokeWidth="1.1" /><text x="228" y="108" fontSize="8.5" fill="#EF4444" textAnchor="middle">×ωʲ</text>
    <circle cx="270" cy="38" r="4" fill="#3B82F6" /><text x="252" y="30" fontSize="10" fill="currentColor">X</text>
    <circle cx="270" cy="104" r="4" fill="#10B981" /><text x="252" y="126" fontSize="10" fill="currentColor">Y</text>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>DiF バタフライ。2入力を足し引きし、片方に回転因子 $\omega^j$ を掛ける。FFT はこのバタフライの積み重ね</figcaption>
</figure>

重要な副作用として、**FFT の出力は順番が入れ替わります**。入力を順番どおりに並べると、出力は **ビット反転（bit-reversal）順** に並びます（たとえば 8 点なら $0,4,2,6,1,5,3,7$）。逆に入力をビット反転順にすれば出力が順番どおりになります。前者が **DiF**、後者が **DiT** です。この「並び替え」が、次の分散実装で効いてきます。

**逆 DFT（iDFT）** も、ほぼ同じ形で計算できます。

$$
x(j) = \mathrm{iDFT}_l(y) = \frac{1}{l}\sum_{k=0}^{l-1} y(k)\,\omega_l^{\,-jk}
$$

DFT との違いは、**回転因子の符号が逆（$\omega^{-jk}$）** で、**$1/l$ の正規化** が入るだけ。同じバタフライ構造が使えます。

## 5. FFT で畳み込みを計算する

畳み込みを FFT で計算する根拠は **畳み込み定理** です。時間領域の畳み込みは、周波数領域では **要素ごとの積** になります。

$$
x * h = \mathcal{F}^{-1}\!\bigl(\mathcal{F}(x) \odot \mathcal{F}(h)\bigr)
$$

つまり「$x$ と $h$ をそれぞれ FFT → 要素積 → 逆 FFT」で畳み込みが得られます。長いフィルタ（Hyena-LI）では、$O(l^2)$ の直接畳み込みより $O(l\log l)$ の FFT 畳み込みが圧倒的に有利です。

## 6. 分散 p2p FFT 畳み込み（CP=2）

ここが核心です。FFT は「**入力を2つに分けて独立に FFT し、バタフライで結合する**」構造でした。これは「**各分割を別デバイスが持つ p2p の状況**」とそっくりです。

2台（$N_{cp}=2$）で考えます。各デバイスが系列の半分 $x_0, x_1$ を持つとき、バタフライは

$$
a = x_0 + x_1, \qquad b = (x_0 - x_1)\,\omega^j
$$

を計算します（デバイス間で $x_0, x_1$ を1往復だけ通信）。あとは各デバイスが $a, b$ の半分サイズ FFT を **ローカルに独立計算** できます。

<figure style={{margin: '1.5rem auto', textAlign: 'center', maxWidth: '480px'}}>
  <svg viewBox="0 0 440 160" width="100%" role="img" aria-label="CP=2の分散FFT。2デバイスが半分ずつ持ち、バタフライで通信してから各自ローカルFFT">
    <text x="110" y="18" fontSize="10.5" fill="currentColor" textAnchor="middle">デバイス0</text>
    <text x="330" y="18" fontSize="10.5" fill="currentColor" textAnchor="middle">デバイス1</text>
    <rect x="60" y="26" width="100" height="28" rx="4" fill="#3B82F6" fillOpacity="0.12" stroke="#3B82F6" strokeWidth="1.2" /><text x="110" y="44" fontSize="10" fill="currentColor" textAnchor="middle">x₀（前半）</text>
    <rect x="280" y="26" width="100" height="28" rx="4" fill="#10B981" fillOpacity="0.12" stroke="#10B981" strokeWidth="1.2" /><text x="330" y="44" fontSize="10" fill="currentColor" textAnchor="middle">x₁（後半）</text>
    <path d="M 160 40 L 278 40" fill="none" stroke="currentColor" strokeOpacity="0.45" strokeWidth="1.1" strokeDasharray="4 3" /><polygon points="278,40 269,35 269,45" fill="currentColor" fillOpacity="0.45" /><polygon points="162,40 171,35 171,45" fill="currentColor" fillOpacity="0.45" />
    <text x="220" y="34" fontSize="8.5" fill="currentColor" fillOpacity="0.7" textAnchor="middle">p2p 通信</text>
    <text x="110" y="76" fontSize="10" fill="currentColor" textAnchor="middle">a = x₀+x₁</text>
    <text x="330" y="76" fontSize="10" fill="#EF4444" textAnchor="middle">b =(x₀−x₁)ω</text>
    <rect x="55" y="88" width="110" height="30" rx="4" fill="#3B82F6" fillOpacity="0.08" stroke="#3B82F6" strokeWidth="1.2" /><text x="110" y="107" fontSize="10" fill="currentColor" textAnchor="middle">ローカル FFT</text>
    <rect x="275" y="88" width="110" height="30" rx="4" fill="#10B981" fillOpacity="0.08" stroke="#10B981" strokeWidth="1.2" /><text x="330" y="107" fontSize="10" fill="currentColor" textAnchor="middle">ローカル FFT</text>
    <text x="220" y="142" fontSize="9.5" fill="currentColor" fillOpacity="0.7" textAnchor="middle">通信は1往復だけ。FFT 本体は各デバイスで並列</text>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>CP=2 の分散 FFT。バタフライの足し引きだけ通信し、半分サイズの FFT は各デバイスでローカルに走る</figcaption>
</figure>

問題は、前述の **bit-reversal** で出力のシャード配置が崩れることです。そのまま FFT すると、本来の系列分割（前半・後半）に戻すための **追加の a2a 通信** が要ります。

ところが——**畳み込みは FFT のあと必ず逆 FFT で戻す** ので、ここに救いがあります。**DiF の FFT（出力がビット反転）→ DiF の逆 FFT** と組み合わせると、ビット反転が往復で打ち消し合い、**最終的に入力と同じシャード配置に戻ります**。よって **追加の a2a なし** で分散 FFT 畳み込みが完結します。論文はこれを最小実装（CP=2）のコードで示しています。

```python
def dif_radix2_fft(x):
    # 入力を2分割（CP=2 をシミュレート：各分割が別デバイス相当）
    x0, x1 = split_halves(x)            # [..., N] -> 2 x [..., N/2]
    k = arange(N // 2)
    W = exp(-2j * pi * k / N)           # 回転因子（twiddle）
    a = x0 + x1                         # バタフライ（和）
    b = (x0 - x1) * W                   # バタフライ（差 × ω）
    return fft(a), fft(b)               # 各半分をローカルFFT（出力はビット反転）

def dif_radix2_ifft(fa, fb):
    x0 = ifft(fa); x1 = ifft(fb)        # 各半分をローカル逆FFT
    k = arange(N // 2)
    W = exp(+2j * pi * k / N)           # 逆向きの回転因子
    return 0.5 * concat([x0 + W * x1,   # 逆バタフライ＋正規化
                         x0 - W * x1])
# 恒等性: x == dif_radix2_ifft(*dif_radix2_fft(x))  ——配置が元に戻る
```

## 7. Radix-N と多デバイスへの拡張（CP=4 / 8）

2分割（Radix-2）を一般化したのが **Radix-$N$ FFT** です。$l$ 点 DFT を **$N$ 個の $l/N$ 点 DFT** に分割し、点ごとの演算（バタフライ）で結合します。$N = N_{cp}$（デバイス数）とすれば、そのまま多デバイスの分散 FFT になります。

Radix-2 では回転因子の組が2つでしたが、一般の $N$ では **$N$ 組の回転因子** $\{W_{nj}\}_{n=0}^{N-1}$ を使います。

$$
W_{nj} = \bigl[\omega^{0\cdot n},\ \omega^{1\cdot n},\ \dots,\ \omega^{(l/N - 1)\cdot n}\bigr]
$$

$N_{cp} = 4, 8$ でも同様に、DiF Radix-$N$ FFT と DiF 逆 FFT を組み合わせれば、系列全体を1台に集めることなく分散 FFT 畳み込みが計算できます（論文 Fig. A.4・A.5 にバタフライ図あり）。

:::tip[Radix-N は「分割の幅」を変えるだけ]

Radix-2 が「2分割」を再帰するのに対し、Radix-$N$ は「一度に $N$ 分割」します。デバイスが $N$ 台あるなら、最初の1段を $N$ 分割にして各デバイスに割り当て、以降は各デバイス内でローカル FFT——という対応が自然に取れます。FFT の分割構造とハードウェアの分割が一致する、という美しさがここにあります。

:::

## 8. それでも a2a が勝つことが多い（正直な評価）

ここまで p2p FFT のエレガントさを説明してきましたが、論文は誠実に **「さらなる最適化なしでは、Hyena-LI には a2a の方が速いことが多かった」** と報告しています。

理由は、分散 FFT は通信ステップが多段になり、各段の通信レイテンシが積み上がるためです。一方 a2a は一括交換で済みます（[05](./05-context-parallelism.md) 参照）。それでも p2p FFT の理論は、**系列全体を1台に集めずに FFT を完結できる** ことを示した点で重要で、将来の最適化の余地を残しています。

## 9. まとめ

- **DFT** は $O(l^2)$、**FFT** は分割統治で $O(l\log l)$。$l$ 点を2つの $l/2$ 点に分けて再帰する。
- 1ステップが **バタフライ**（$X=x+y,\ Y=(x-y)\omega^j$）。出力は **ビット反転順** に並ぶ。
- 畳み込み定理 $x*h = \mathcal{F}^{-1}(\mathcal{F}(x)\odot\mathcal{F}(h))$ で長フィルタを高速計算。
- **分散 p2p FFT** は、FFT の分割構造をデバイス分割に対応させる。DiF FFT＋DiF 逆 FFT でビット反転が往復で打ち消し、**追加 a2a なし** で完結。Radix-$N$ で多デバイスへ拡張。
- 実用上は a2a が速いことも多いが、理論的な見通しとして重要。

これで StripedHyena 2 の分散学習まわりを数学的に詰めました。[05 の本編](./05-context-parallelism.md)に戻るか、[01 の概要](./01-overview.md)から全体像を振り返ってみてください。
