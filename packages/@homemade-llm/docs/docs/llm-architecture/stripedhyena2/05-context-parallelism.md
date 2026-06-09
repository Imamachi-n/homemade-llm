---
sidebar_position: 5
title: "05. 長系列の分散学習（Context Parallelism）"
---

# 05. 長系列の分散学習（Context Parallelism）

[04](./04-hardware-algorithms.md)までは1台の GPU の中の話でした。しかし **100 万トークンの系列** は、1台の GPU メモリには到底収まりません。このページでは、長い系列を **複数の GPU に分散** して学習する **コンテキスト並列（context parallelism, CP）** と、StripedHyena 2 の畳み込みをどう分散するか——とりわけ美しい **FFT 畳み込み** ——を見ていきます。

## 1. なぜ分散学習が必要か

コンテキスト並列（CP）は、増大するモデルと入力次元に対応するため、**系列を分割して複数デバイスで処理** する技術です。data / tensor / sequence / pipeline parallelism といった他の分散手法を補完します。

具体的には、入力 $[D, L]$（次元 $D$ ×系列長 $L$）を $N_{cp}$ 台のデバイスに **系列次元で分割** し、各デバイスが $[D,\, L/N_{cp}]$ のシャードを持ちます。

問題は、Self-Attention や畳み込みのような **系列混合（sequence mixing）** が「他のデバイスにある部分」を必要とすることです。これをどう通信でやりくりするかが CP の核心です。

## 2. 2つの通信戦略：All-to-all と Point-to-point

論文は2つの通信戦略を整理します。

- **All-to-all（a2a）** — 各デバイスが **全デバイスとデータを交換** し、系列全体を再構成します。$[D, L/N_{cp}]$ を再配置して $[H/N_{cp}, L]$（チャネルを分割し、系列は全長）にする。これで各ランクが系列混合を独立に実行できます。DeepSpeed Ulysses で使われる方式です。
- **Point-to-point（p2p）** — 全デバイスにブロードキャストせず、**1度に1つのピアと直接交換** します。ブロック計算と通信を何ラウンドも繰り返す。Self-Attention 版が有名な **ring attention** です。

<figure style={{margin: '1.5rem auto', textAlign: 'center', maxWidth: '500px'}}>
  <svg viewBox="0 0 460 180" width="100%" role="img" aria-label="All-to-allは全デバイス間で交換、Point-to-pointは隣接ピアとリング状に交換">
    <text x="115" y="20" fontSize="11" fill="currentColor" textAnchor="middle">All-to-all（全交換）</text>
    <circle cx="60" cy="60" r="16" fill="#3B82F6" fillOpacity="0.12" stroke="#3B82F6" strokeWidth="1.3" /><text x="60" y="64" fontSize="9" fill="currentColor" textAnchor="middle">D0</text>
    <circle cx="170" cy="60" r="16" fill="#3B82F6" fillOpacity="0.12" stroke="#3B82F6" strokeWidth="1.3" /><text x="170" y="64" fontSize="9" fill="currentColor" textAnchor="middle">D1</text>
    <circle cx="60" cy="140" r="16" fill="#3B82F6" fillOpacity="0.12" stroke="#3B82F6" strokeWidth="1.3" /><text x="60" y="144" fontSize="9" fill="currentColor" textAnchor="middle">D2</text>
    <circle cx="170" cy="140" r="16" fill="#3B82F6" fillOpacity="0.12" stroke="#3B82F6" strokeWidth="1.3" /><text x="170" y="144" fontSize="9" fill="currentColor" textAnchor="middle">D3</text>
    <g stroke="#3B82F6" strokeOpacity="0.5" strokeWidth="1.1">
      <line x1="76" y1="60" x2="154" y2="60" /><line x1="60" y1="76" x2="60" y2="124" />
      <line x1="170" y1="76" x2="170" y2="124" /><line x1="76" y1="140" x2="154" y2="140" />
      <line x1="74" y1="72" x2="156" y2="128" /><line x1="156" y1="72" x2="74" y2="128" />
    </g>
    <text x="350" y="20" fontSize="11" fill="currentColor" textAnchor="middle">Point-to-point（リング）</text>
    <circle cx="295" cy="60" r="16" fill="#10B981" fillOpacity="0.12" stroke="#10B981" strokeWidth="1.3" /><text x="295" y="64" fontSize="9" fill="currentColor" textAnchor="middle">D0</text>
    <circle cx="405" cy="60" r="16" fill="#10B981" fillOpacity="0.12" stroke="#10B981" strokeWidth="1.3" /><text x="405" y="64" fontSize="9" fill="currentColor" textAnchor="middle">D1</text>
    <circle cx="405" cy="140" r="16" fill="#10B981" fillOpacity="0.12" stroke="#10B981" strokeWidth="1.3" /><text x="405" y="144" fontSize="9" fill="currentColor" textAnchor="middle">D2</text>
    <circle cx="295" cy="140" r="16" fill="#10B981" fillOpacity="0.12" stroke="#10B981" strokeWidth="1.3" /><text x="295" y="144" fontSize="9" fill="currentColor" textAnchor="middle">D3</text>
    <g stroke="#10B981" strokeOpacity="0.6" strokeWidth="1.3" fill="none">
      <path d="M 311 56 L 389 56" /><polygon points="393,56 384,51 384,61" fill="#10B981" fillOpacity="0.6" />
      <path d="M 409 76 L 409 124" /><polygon points="409,128 404,119 414,119" fill="#10B981" fillOpacity="0.6" />
      <path d="M 389 144 L 311 144" /><polygon points="307,144 316,139 316,149" fill="#10B981" fillOpacity="0.6" />
      <path d="M 291 124 L 291 76" /><polygon points="291,72 286,81 296,81" fill="#10B981" fillOpacity="0.6" />
    </g>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>2つの通信戦略。a2a は全デバイス間で一括交換、p2p は隣接ピアとリング状に逐次交換する</figcaption>
</figure>

## 3. Hyena 演算子をどう分散するか

Hyena の系列混合は、長さの異なる畳み込みです。CP 実装には演算子ごとの工夫が要ります。

- **a2a 畳み込み** — 入力を a2a で $[H/N_{cp}, L]$ に再配置し、各シャードを CP 領域内で畳み込み、再び a2a で戻します。フィルタは各 CP 領域内に保存・生成します（SE は各ランクが $H/N_{cp}$ 個のフィルタを保持、MR/LI はフィルタ計算を領域内で実行）。逆伝播では勾配を元のランクへ戻すため、追加の a2a が2回必要です。
- **p2p 畳み込み** — ここで **FIR フィルタの局所性** が効きます。因果畳み込みの大部分は **通信なしでローカルに計算でき**、各シャードのうち **最初の $\ell_h - 1$ 要素だけ** が「前のランクの末尾 $\ell_h - 1$ 要素」を必要とします。各ランクはフィルタのコピーを持ち、全 $D$ チャネルの畳み込みを担当します。

<figure style={{margin: '1.5rem auto', textAlign: 'center', maxWidth: '560px'}}>
  <svg viewBox="0 0 560 130" width="100%" role="img" aria-label="p2p畳み込みの局所性。各ランクは大部分をローカル計算し、境界のℓh-1要素だけ前ランクから受け取る">
    <rect x="30" y="45" width="150" height="34" rx="4" fill="#10B981" fillOpacity="0.1" stroke="#10B981" strokeWidth="1.3" /><text x="105" y="66" fontSize="10" fill="currentColor" textAnchor="middle">ランク0：ローカル計算</text>
    <rect x="190" y="45" width="14" height="34" fill="#EF4444" fillOpacity="0.35" stroke="#EF4444" strokeWidth="0.8" />
    <rect x="204" y="45" width="150" height="34" rx="4" fill="#10B981" fillOpacity="0.1" stroke="#10B981" strokeWidth="1.3" /><text x="279" y="66" fontSize="10" fill="currentColor" textAnchor="middle">ランク1：ローカル計算</text>
    <rect x="364" y="45" width="14" height="34" fill="#EF4444" fillOpacity="0.35" stroke="#EF4444" strokeWidth="0.8" />
    <rect x="378" y="45" width="150" height="34" rx="4" fill="#10B981" fillOpacity="0.1" stroke="#10B981" strokeWidth="1.3" /><text x="453" y="66" fontSize="10" fill="currentColor" textAnchor="middle">ランク2：ローカル計算</text>
    <path d="M 180 90 C 188 104, 196 104, 197 84" fill="none" stroke="#EF4444" strokeOpacity="0.7" strokeWidth="1.2" /><polygon points="197,86 192,95 202,93" fill="#EF4444" fillOpacity="0.7" />
    <path d="M 354 90 C 362 104, 370 104, 371 84" fill="none" stroke="#EF4444" strokeOpacity="0.7" strokeWidth="1.2" /><polygon points="371,86 366,95 376,93" fill="#EF4444" fillOpacity="0.7" />
    <text x="279" y="118" fontSize="9.5" fill="#EF4444" textAnchor="middle">境界の ℓₕ−1 要素だけ前ランクから受信（赤）</text>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>p2p 畳み込みの局所性。FIR なので大半はローカル計算で済み、通信は境界の $\ell_h-1$ 要素だけ</figcaption>
</figure>

論文はさらに、**通信と計算をオーバーラップ** する拡張も提案します（境界以外をゼロ埋めで先に計算し、通信完了後に境界分を足し込む。これは [04](./04-hardware-algorithms.md) の2段階分解と同じ発想です）。

## 4. FFT 畳み込み：周波数領域の魔法

長い暗黙フィルタをもつ **Hyena-LI** は、FFT で計算するのが一般的です。鍵は **畳み込み定理**——時間領域の畳み込みは、周波数領域では **単なる要素積** になる、という事実です。

$$
x * h = \mathcal{F}^{-1}\!\bigl(\mathcal{F}(x) \odot \mathcal{F}(h)\bigr)
$$

ここで $\mathcal{F}, \mathcal{F}^{-1}$ はフーリエ変換と逆変換です。離散フーリエ変換（DFT）は

$$
y(k) = \sum_{j=0}^{l-1} x(j)\,\omega_l^{jk}, \qquad \omega_l = e^{-2\pi i / l}
$$

で定義され、素朴に計算すると $O(l^2)$ ですが、**高速フーリエ変換（FFT）は分割統治で $O(l \log l)$** を達成します。$l$ 点 DFT を2つの $l/2$ 点 DFT に分解する、という再帰がその正体です。

この「2分割」の演算が **バタフライ（butterfly）** です。Decimation-in-Frequency（DiF）バタフライは次の2演算からなります。

$$
X = x + y, \qquad Y = (x - y)\,\omega^j
$$

<figure style={{margin: '1.5rem auto', textAlign: 'center', maxWidth: '320px'}}>
  <svg viewBox="0 0 300 150" width="100%" role="img" aria-label="DiFバタフライ。入力x,yからX=x+y、Y=(x-y)ωを計算する交差構造">
    <circle cx="50" cy="40" r="4" fill="#3B82F6" /><text x="34" y="44" fontSize="12" fill="currentColor">x</text>
    <circle cx="50" cy="110" r="4" fill="#10B981" /><text x="34" y="114" fontSize="12" fill="currentColor">y</text>
    <line x1="54" y1="40" x2="246" y2="40" stroke="#3B82F6" strokeOpacity="0.6" strokeWidth="1.3" />
    <line x1="54" y1="110" x2="246" y2="110" stroke="#10B981" strokeOpacity="0.6" strokeWidth="1.3" />
    <line x1="54" y1="40" x2="246" y2="110" stroke="currentColor" strokeOpacity="0.4" strokeWidth="1.1" />
    <line x1="54" y1="110" x2="246" y2="40" stroke="currentColor" strokeOpacity="0.4" strokeWidth="1.1" />
    <circle cx="150" cy="40" r="9" fill="none" stroke="currentColor" strokeOpacity="0.6" strokeWidth="1.1" /><text x="150" y="44" fontSize="11" fill="currentColor" textAnchor="middle">+</text>
    <circle cx="150" cy="110" r="9" fill="none" stroke="currentColor" strokeOpacity="0.6" strokeWidth="1.1" /><text x="150" y="44.5" fontSize="11" fill="currentColor" textAnchor="middle">+</text>
    <text x="150" y="114" fontSize="11" fill="currentColor" textAnchor="middle">−</text>
    <circle cx="205" cy="110" r="10" fill="none" stroke="#EF4444" strokeOpacity="0.7" strokeWidth="1.1" /><text x="205" y="114" fontSize="9" fill="#EF4444" textAnchor="middle">×ωʲ</text>
    <circle cx="250" cy="40" r="4" fill="#3B82F6" /><text x="258" y="44" fontSize="11" fill="currentColor">X = x+y</text>
    <circle cx="250" cy="110" r="4" fill="#10B981" /><text x="258" y="114" fontSize="11" fill="currentColor">Y =(x−y)ωʲ</text>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>DiF バタフライ。2入力を足し引きし、片方にひねり係数 $\omega^j$ を掛ける。この「蝶」の積み重ねが FFT</figcaption>
</figure>

### なぜこれが分散に向くのか

FFT は「**入力の独立な2分割に対して個別に FFT を行い、要素ごとの演算で結合する**」構造です。これは **各分割を別デバイスに置く p2p の状況とぴったり一致** します。素朴にやると FFT 後の出力が **ビット反転（bit-reversal）順** に並んでシャード配置が崩れますが、**畳み込みでは FFT → 逆 FFT と往復する** ので、DiF FFT と DiF 逆 FFT を組み合わせれば **入力と同じシャード配置に戻り、追加の a2a 通信が不要** になります。

$N_{cp} > 2$ への拡張には **Radix-$N$ FFT**（$l$ 点を $N$ 個の $l/N$ 点に分解）を使い、$N = N_{cp}$ として分散 FFT 畳み込みを実現します。ただし論文は、さらなる最適化なしでは **Hyena-LI には a2a の方が速いことが多かった** とも正直に報告しています。

:::tip[LLM とのつながり：長文脈学習の共通基盤]

ここで出てくる **ring attention（p2p）**・**DeepSpeed Ulysses（a2a）** は、自然言語 LLM の長文脈学習でも標準的な技術です。StripedHyena 2 は、これらの考え方を **畳み込み演算子向けに拡張** し、さらに FFT という信号処理の道具を分散学習に持ち込んだ点が新しいと言えます。

:::

## 5. 因果モデルの負荷分散

自己回帰（causal）モデルでは、計算が三角構造になり、単純に系列を分割すると **デバイス間で負荷が偏ります**（後ろのトークンほど計算量が多い）。これを避けるため、系列を工夫して割り当てます。

- **striped ordering**（Brandon et al.）：CP ランク数の2倍のシャードに分け、各ランクに2枚を縞状に配置。
- **zig-zag splitting**（Llama 3, Dubey et al.）：たとえば 8 シャード・$N_{cp}=4$ なら $[x_0, x_7], [x_1, x_6], [x_2, x_5], [x_3, x_4]$ のように、前半と後半をペアにして負荷を均等化。

**StripedHyena 2 の学習では、Llama 3 と同じ zig-zag 分割を採用** しています。

## 6. まとめ：StripedHyena 2 の全体像

このセクションでは、Evo 2 を支える **StripedHyena 2** を、論文に沿って深掘りしてきました。

| ページ | 要点 |
| --- | --- |
| [01 概要](./01-overview.md) | hybridization-aware ＋ hardware-aware な設計で、全域で速いマルチハイブリッドを実現 |
| [02 演算子](./02-operators.md) | Hyena 構造（射影＋畳み込み＋ゲーティング）と、SE/MR/LI の役割分担 |
| [03 設計](./03-architecture-scaling.md) | SE-MR-LI レイアウト・filter grouping・1M 文脈拡張・1.2〜2.9 倍高速 |
| [04 カーネル](./04-hardware-algorithms.md) | 畳み込み＝Toeplitz、2段階分解で GEMM 化しテンソルコアをフル活用 |
| 05 分散学習（本ページ） | a2a/p2p のコンテキスト並列、FFT 畳み込み、zig-zag 負荷分散 |

一貫しているのは、**「演算子の設計」と「ハードウェア・分散アルゴリズム」を一体で考える（co-design）** という思想です。Attention をただ別の演算子に置き換えるのではなく、**短・中・長の畳み込みと attention を役割分担させ、それぞれを GPU で最速に走らせる** ——この積み重ねが、[Evo 2](../../biology-llm-applications/evo2/01-overview.md) の **40B・9 兆トークン・100 万文脈** という規模を可能にしました。

:::tip[このセクションは今後も拡張予定]

「LLM アーキテクチャ」は、StripedHyena 2 を第1弾として、Transformer を補完・代替する効率的アーキ（Mamba などの状態空間モデル、各種ハイブリッド）を今後追加していく予定です。

:::

StripedHyena 2 が実際に何を成し遂げたかは、応用である [Evo 2](../../biology-llm-applications/evo2/01-overview.md) で確かめられます。アーキテクチャの土台（[Evo 2 アーキテクチャ編](../../biology-llm-applications/evo2/02-architecture.md)）と合わせて読むと、設計から応用までが一本につながります。

最後に原論文を：**Ku, Nguyen, Romero et al., "Systems and Algorithms for Convolutional Multi-Hybrid Language Models at Scale", arXiv:2503.01868 (2025).** [arxiv.org/abs/2503.01868](https://arxiv.org/abs/2503.01868)
