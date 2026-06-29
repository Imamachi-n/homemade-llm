---
sidebar_position: 3
title: "Chapter 3: Transformer を組み立てる残りの部品"
---

# Chapter 3: Transformer を組み立てる残りの部品

[前章](./chapter2.md)では、Transformer の心臓部 **Attention（注意機構）** を、概念から実装まで一気通貫でつかみました。ただし前章はずっと「**単語はすでにベクトルになっている**」ことを前提にしていました。

でも、よく考えると Attention だけでは Transformer は動きません。たとえば——

- そもそも **テキスト（文字列）をどうやってベクトルにする**の？
- Attention は全単語を一気に見るけど、**「何番目の単語か」という順番の情報**はどこから来るの？
- Attention の出力を、**もっと深く加工する**部品は要らないの？
- 層をたくさん積んだとき、**学習がちゃんと進む**ように支える仕組みは？

この章では、これらの疑問に答える **Attention 以外の5つの部品**を、初学者でも分かるように図解しながら厚めに解説します。

1. **トークン埋め込み**（埋め込みベクトル）… テキストを意味のあるベクトルに変える入り口
2. **位置エンコーディング** … 「何番目か」という順番の情報を足す
3. **フィードフォワード層**（FFN）… 各単語を個別に深く加工する
4. **スキップ接続**（残差接続）… 層を深く積んでも学習が進むようにする近道
5. **レイヤー正規化** … 値のスケールを整えて学習を安定させる

そして最後に、**Attention とこの5部品を全部組み合わせて、1つの Transformer ブロックを完成**させます。

:::tip[この章の読み方]

5つの部品は、どれも「**Transformer ブロックという1つの装置の、それぞれの部品**」です。バラバラに覚えるのではなく、「この部品はブロックのどこに、何のために付くのか」を意識すると一気に頭に入ります。各節の最後に、必ず全体像のどこに効くのかを示します。

:::

---

## 0. 全体像：5つの部品はブロックのどこにいる？

細かい話に入る前に、まず**完成形**（地図）を見ておきましょう。下の図が、これから組み立てる Transformer ブロック1個の中身です。前章の Attention（マルチヘッドアテンション）は中央にいて、この章で学ぶ5部品がそれを取り囲んでいます。

<figure style={{margin: '1.5rem auto', textAlign: 'center', maxWidth: '460px'}}>
  <svg viewBox="0 0 420 470" width="100%" role="img" aria-label="Transformer ブロックの全体像。5つの部品の位置">
    {/* 入力テキスト */}
    <rect x="135" y="10" width="150" height="30" rx="6" fill="currentColor" fillOpacity="0.08" stroke="currentColor" strokeOpacity="0.5" strokeWidth="1.2" />
    <text x="210" y="29" fontSize="12" fill="currentColor" textAnchor="middle">入力テキスト「猫が…」</text>
    {/* 1 トークン埋め込み */}
    <rect x="115" y="58" width="190" height="34" rx="6" fill="#3B82F6" fillOpacity="0.16" stroke="#3B82F6" strokeWidth="1.6" />
    <text x="210" y="79" fontSize="12.5" fill="currentColor" textAnchor="middle">① トークン埋め込み</text>
    {/* + 位置エンコーディング */}
    <rect x="115" y="104" width="190" height="34" rx="6" fill="#3B82F6" fillOpacity="0.16" stroke="#3B82F6" strokeWidth="1.6" />
    <text x="210" y="125" fontSize="12.5" fill="currentColor" textAnchor="middle">② ＋ 位置エンコーディング</text>
    {/* ブロックの枠 */}
    <rect x="40" y="158" width="340" height="262" rx="10" fill="currentColor" fillOpacity="0.03" stroke="currentColor" strokeOpacity="0.4" strokeWidth="1.4" strokeDasharray="6 4" />
    <text x="55" y="178" fontSize="11" fill="currentColor" fillOpacity="0.7">Transformer ブロック（×N 回くり返す）</text>
    {/* Multi-Head Attention */}
    <rect x="95" y="192" width="230" height="36" rx="6" fill="#EF4444" fillOpacity="0.14" stroke="#EF4444" strokeWidth="1.6" />
    <text x="210" y="214" fontSize="12.5" fill="currentColor" textAnchor="middle">マルチヘッドアテンション（前章）</text>
    {/* Add & Norm 1 */}
    <rect x="95" y="240" width="230" height="32" rx="6" fill="#10B981" fillOpacity="0.16" stroke="#10B981" strokeWidth="1.6" />
    <text x="210" y="260" fontSize="11.5" fill="currentColor" textAnchor="middle">④ スキップ接続 ＋ ⑤ レイヤー正規化</text>
    {/* FFN */}
    <rect x="95" y="300" width="230" height="36" rx="6" fill="#3B82F6" fillOpacity="0.16" stroke="#3B82F6" strokeWidth="1.6" />
    <text x="210" y="322" fontSize="12.5" fill="currentColor" textAnchor="middle">③ フィードフォワード層</text>
    {/* Add & Norm 2 */}
    <rect x="95" y="348" width="230" height="32" rx="6" fill="#10B981" fillOpacity="0.16" stroke="#10B981" strokeWidth="1.6" />
    <text x="210" y="368" fontSize="11.5" fill="currentColor" textAnchor="middle">④ スキップ接続 ＋ ⑤ レイヤー正規化</text>
    {/* 出力 */}
    <rect x="135" y="432" width="150" height="30" rx="6" fill="currentColor" fillOpacity="0.08" stroke="currentColor" strokeOpacity="0.5" strokeWidth="1.2" />
    <text x="210" y="451" fontSize="12" fill="currentColor" textAnchor="middle">出力（次のブロックへ）</text>
    {/* 矢印 */}
    <defs>
      <marker id="ovArrow" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto">
        <path d="M0,0 L6,3 L0,6 Z" fill="currentColor" fillOpacity="0.6" />
      </marker>
    </defs>
    <line x1="210" y1="40" x2="210" y2="56" stroke="currentColor" strokeOpacity="0.6" strokeWidth="1.4" markerEnd="url(#ovArrow)" />
    <line x1="210" y1="92" x2="210" y2="102" stroke="currentColor" strokeOpacity="0.6" strokeWidth="1.4" markerEnd="url(#ovArrow)" />
    <line x1="210" y1="138" x2="210" y2="190" stroke="currentColor" strokeOpacity="0.6" strokeWidth="1.4" markerEnd="url(#ovArrow)" />
    <line x1="210" y1="272" x2="210" y2="298" stroke="currentColor" strokeOpacity="0.6" strokeWidth="1.4" markerEnd="url(#ovArrow)" />
    <line x1="210" y1="380" x2="210" y2="430" stroke="currentColor" strokeOpacity="0.6" strokeWidth="1.4" markerEnd="url(#ovArrow)" />
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.4rem', opacity: 0.85}}>Transformer ブロックの全体像。中央の赤が前章の Attention、青と緑がこの章で学ぶ5部品</figcaption>
</figure>

データは上から下へ流れます。**テキスト → ①埋め込みでベクトル化 → ②位置情報を足す → ブロックの中で（Attention → ④⑤で整える → ③FFN → ④⑤で整える）→ 出力**、という流れです。この図を「地図」として、これから各部品を1つずつ訪ねていきましょう。

:::note[「ブロックを N 回くり返す」とは？]

Transformer は、上の点線で囲ったブロックを**何個も積み重ねて**できています（GPT 系では数十段）。1段目の出力が2段目の入力になり……とくり返すことで、単純な部品の組み合わせから深い表現力が生まれます。①②（埋め込み・位置）は最初に1回だけ、③〜⑤を含むブロックは N 回くり返す、という構造です。

:::

---

## 1. トークン埋め込み（埋め込みベクトル）

### 1.1 直感：単語を「意味を持つ数のリスト」に変える

コンピュータは「猫」という文字そのものを計算できません。足し算も内積もできるのは**数**だけです。そこで、まず各単語に**ベクトル**（数のリスト）を割り当てます。これが **トークン埋め込み（token embedding）** です。

では、どんなベクトルを割り当てればよいでしょう？ 一番素朴なのは、単語に番号を振って、その番号の場所だけ 1 にする方法です（**ワンホット表現**）。たとえば語彙が「猫・犬・走る・寝る」の4語なら、

- 猫 → $[1, 0, 0, 0]$
- 犬 → $[0, 1, 0, 0]$

ですが、これには2つの大問題があります。

1. **意味が入っていない。** 猫と犬は似た生き物なのに、ベクトルとしては全くの無関係（内積を取るとゼロ）。
2. **巨大すぎる。** 実際の語彙は数万語なので、1単語が数万次元のスカスカなベクトルになってしまう。

そこで Transformer では、各単語に **「短くて、中身がぎっしり詰まった（密な）、意味を反映したベクトル」** を割り当てます。たとえば次元を4にして、

- 猫 → $[0.8, 0.2, -0.5, 0.1]$
- 犬 → $[0.7, 0.3, -0.4, 0.2]$（猫と近い！）
- 走る → $[-0.6, 0.9, 0.1, 0.5]$（動詞なので離れている）

こうすると「猫と犬は近い」という意味が、ベクトルの近さ（前章の内積・コサイン類似度！）として表現できます。

:::note[「トークン」って単語のこと？]

ここでは分かりやすさのために「単語」と言っていますが、正確には **トークン（token）** という単位を使います。トークンは「単語より少し細かい、文字列の断片」で、たとえば "playing" が "play" + "ing" のように分割されることもあります。テキストをトークンに分ける仕組み（トークナイザー）は今後の章で扱います。この章では「トークン ≒ 単語」と思って大丈夫です。

:::

<figure style={{margin: '1.5rem auto', textAlign: 'center', maxWidth: '520px'}}>
  <svg viewBox="0 0 500 220" width="100%" role="img" aria-label="ワンホット表現と埋め込みの比較">
    {/* 左: one-hot */}
    <text x="120" y="22" fontSize="12.5" fill="currentColor" textAnchor="middle" fontWeight="bold">ワンホット表現</text>
    <text x="120" y="40" fontSize="10" fill="currentColor" fillOpacity="0.6" textAnchor="middle">長い・スカスカ・意味なし</text>
    <text x="22" y="72" fontSize="11" fill="currentColor">猫</text>
    <rect x="42" y="60" width="20" height="20" fill="#EF4444" fillOpacity="0.6" stroke="currentColor" strokeOpacity="0.4" />
    <rect x="62" y="60" width="20" height="20" fill="currentColor" fillOpacity="0.06" stroke="currentColor" strokeOpacity="0.4" />
    <rect x="82" y="60" width="20" height="20" fill="currentColor" fillOpacity="0.06" stroke="currentColor" strokeOpacity="0.4" />
    <rect x="102" y="60" width="20" height="20" fill="currentColor" fillOpacity="0.06" stroke="currentColor" strokeOpacity="0.4" />
    <rect x="122" y="60" width="20" height="20" fill="currentColor" fillOpacity="0.06" stroke="currentColor" strokeOpacity="0.4" />
    <text x="150" y="74" fontSize="11" fill="currentColor" fillOpacity="0.5">… 数万次元</text>
    <text x="22" y="112" fontSize="11" fill="currentColor">犬</text>
    <rect x="42" y="100" width="20" height="20" fill="currentColor" fillOpacity="0.06" stroke="currentColor" strokeOpacity="0.4" />
    <rect x="62" y="100" width="20" height="20" fill="#EF4444" fillOpacity="0.6" stroke="currentColor" strokeOpacity="0.4" />
    <rect x="82" y="100" width="20" height="20" fill="currentColor" fillOpacity="0.06" stroke="currentColor" strokeOpacity="0.4" />
    <rect x="102" y="100" width="20" height="20" fill="currentColor" fillOpacity="0.06" stroke="currentColor" strokeOpacity="0.4" />
    <rect x="122" y="100" width="20" height="20" fill="currentColor" fillOpacity="0.06" stroke="currentColor" strokeOpacity="0.4" />
    <text x="150" y="114" fontSize="11" fill="currentColor" fillOpacity="0.5">… 数万次元</text>
    {/* 右: embedding */}
    <text x="370" y="22" fontSize="12.5" fill="currentColor" textAnchor="middle" fontWeight="bold">埋め込みベクトル</text>
    <text x="370" y="40" fontSize="10" fill="currentColor" fillOpacity="0.6" textAnchor="middle">短い・ぎっしり・意味あり</text>
    <text x="282" y="72" fontSize="11" fill="currentColor">猫</text>
    <rect x="302" y="60" width="36" height="20" fill="#3B82F6" fillOpacity="0.7" stroke="currentColor" strokeOpacity="0.4" />
    <rect x="338" y="60" width="36" height="20" fill="#3B82F6" fillOpacity="0.3" stroke="currentColor" strokeOpacity="0.4" />
    <rect x="374" y="60" width="36" height="20" fill="#3B82F6" fillOpacity="0.5" stroke="currentColor" strokeOpacity="0.4" />
    <rect x="410" y="60" width="36" height="20" fill="#3B82F6" fillOpacity="0.2" stroke="currentColor" strokeOpacity="0.4" />
    <text x="282" y="112" fontSize="11" fill="currentColor">犬</text>
    <rect x="302" y="100" width="36" height="20" fill="#3B82F6" fillOpacity="0.65" stroke="currentColor" strokeOpacity="0.4" />
    <rect x="338" y="100" width="36" height="20" fill="#3B82F6" fillOpacity="0.35" stroke="currentColor" strokeOpacity="0.4" />
    <rect x="374" y="100" width="36" height="20" fill="#3B82F6" fillOpacity="0.45" stroke="currentColor" strokeOpacity="0.4" />
    <rect x="410" y="100" width="36" height="20" fill="#3B82F6" fillOpacity="0.25" stroke="currentColor" strokeOpacity="0.4" />
    <text x="370" y="150" fontSize="10.5" fill="currentColor" fillOpacity="0.7" textAnchor="middle">猫と犬の濃淡が似る ＝ 意味が近い</text>
    {/* 矢印 */}
    <text x="232" y="95" fontSize="22" fill="currentColor" fillOpacity="0.5" textAnchor="middle">→</text>
    <text x="232" y="180" fontSize="10.5" fill="currentColor" fillOpacity="0.6" textAnchor="middle">学習で獲得</text>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>ワンホット（左）は長くて意味がない。埋め込み（右）は短く、似た単語は似たベクトルになる</figcaption>
</figure>

### 1.2 定義：埋め込み行列から「行を引く」だけ

埋め込みの正体は、実はとてもシンプルです。**1枚の大きな表（行列）から、単語の番号で行を1本引いてくる**だけです。

語彙のサイズを $V$（語の総数）、埋め込みの次元を $d_{\text{model}}$ とすると、**埋め込み行列** $E$ は

$$
E \in \mathbb{R}^{V \times d_{\text{model}}}
$$

という大きさの行列です。$i$ 番目のトークン（トークン ID が $i$）の埋め込みベクトルは、この $E$ の **第 $i$ 行**そのものです。

$$
\text{embed}(i) = E_{i,:} \quad (\text{$E$ の $i$ 行目})
$$

つまり「埋め込み」という処理は、難しい計算ではなく**表引き**（ルックアップ）です。そして表 $E$ の中身（各単語のベクトル）は、最初はランダムな値で、**学習を通じて少しずつ意味のある値に調整されていきます**。ここがワンホットとの決定的な違いで、埋め込みは「学習で賢くなる」のです。

:::note[「埋め込みのもとはワンホット？」それとも別物？]

「1.1 で出てきたワンホットと、この埋め込みは別物なの？」と思うかもしれません。答えは **「計算のうえではつながっているが、意味を持つのは埋め込み側」** です。

実は、**ワンホット表現と埋め込み行列 $E$ の掛け算**は、**ルックアップ（行引き）とまったく同じ結果**になります。たとえば「猫」（ID = 3）のワンホット $[0,0,0,1]$ に $E$ を掛けると、他の行は $0$ が掛かって消え、$E$ の4行目だけが残ります。

$$
[\,0,\ 0,\ 0,\ 1\,]\, E = E_{3,:} = [\,0.8,\ 0.2,\ -0.5,\ 0.1\,]
$$

ここで**ワンホットの長さは語彙サイズ $V$ に等しく**、語彙が4語なら長さ4、数万語なら数万次元のスカスカなベクトルです。埋め込み行列 $E$ は $V \times d_{\text{model}}$ の大きさなので、掛け算は次元がぴったりつながります。

$$
\underbrace{(1 \times V)}_{\text{ワンホット}} \cdot \underbrace{(V \times d_{\text{model}})}_{E} = \underbrace{(1 \times d_{\text{model}})}_{\text{埋め込み}}
$$

埋め込みが「短くてぎっしり」なのは、この **$V$ 次元から $d_{\text{model}}$ 次元への圧縮**でもあるのです（$V$ は数万でも、$d_{\text{model}}$ は数百〜数千ほど）。

つまり「埋め込みはワンホットから作られる」とも言えます。ただし大事なのは、**意味のある数値を持っているのは $E$ の側**だということ。ワンホットは $0$ と $1$ しか持たず、「$E$ のどの行を選ぶか」を指すスイッチにすぎません。猫と犬が似たベクトルになるのも、学習で賢くなるのも、すべて $E$ の働きです。だから実装では掛け算を省いて、最初から行を引きます（→ 1.4 節）。

:::

<figure style={{margin: '1.5rem auto', textAlign: 'center', maxWidth: '440px'}}>
  <svg viewBox="0 0 420 240" width="100%" role="img" aria-label="埋め込み行列から行を引くルックアップ">
    <text x="115" y="20" fontSize="12" fill="currentColor" textAnchor="middle" fontWeight="bold">埋め込み行列 E（V×d_model）</text>
    {/* 行ラベルとグリッド */}
    {[
      {id: 0, w: '走る', y: 36, hl: false},
      {id: 1, w: '寝る', y: 66, hl: false},
      {id: 2, w: '犬', y: 96, hl: false},
      {id: 3, w: '猫', y: 126, hl: true},
      {id: 4, w: '…', y: 156, hl: false}
    ].map((r) => (
      <g key={r.id}>
        <text x="10" y={r.y + 18} fontSize="10" fill="currentColor" fillOpacity="0.7">{r.id}:{r.w}</text>
        <rect x="60" y={r.y} width="42" height="26" fill={r.hl ? '#3B82F6' : 'currentColor'} fillOpacity={r.hl ? 0.5 : 0.08} stroke="currentColor" strokeOpacity="0.4" />
        <rect x="102" y={r.y} width="42" height="26" fill={r.hl ? '#3B82F6' : 'currentColor'} fillOpacity={r.hl ? 0.5 : 0.08} stroke="currentColor" strokeOpacity="0.4" />
        <rect x="144" y={r.y} width="42" height="26" fill={r.hl ? '#3B82F6' : 'currentColor'} fillOpacity={r.hl ? 0.5 : 0.08} stroke="currentColor" strokeOpacity="0.4" />
        <rect x="186" y={r.y} width="42" height="26" fill={r.hl ? '#3B82F6' : 'currentColor'} fillOpacity={r.hl ? 0.5 : 0.08} stroke="currentColor" strokeOpacity="0.4" />
      </g>
    ))}
    {/* 矢印と結果 */}
    <text x="260" y="143" fontSize="20" fill="#3B82F6" fillOpacity="0.7">→</text>
    <text x="350" y="100" fontSize="11" fill="currentColor" textAnchor="middle">「猫」(ID=3)</text>
    <text x="350" y="116" fontSize="11" fill="currentColor" textAnchor="middle">のベクトル</text>
    <rect x="290" y="128" width="30" height="24" fill="#3B82F6" fillOpacity="0.5" stroke="currentColor" strokeOpacity="0.4" />
    <rect x="320" y="128" width="30" height="24" fill="#3B82F6" fillOpacity="0.5" stroke="currentColor" strokeOpacity="0.4" />
    <rect x="350" y="128" width="30" height="24" fill="#3B82F6" fillOpacity="0.5" stroke="currentColor" strokeOpacity="0.4" />
    <rect x="380" y="128" width="30" height="24" fill="#3B82F6" fillOpacity="0.5" stroke="currentColor" strokeOpacity="0.4" />
    <text x="115" y="210" fontSize="10.5" fill="currentColor" fillOpacity="0.7" textAnchor="middle">↑ 横が d_model 次元、縦が語彙 V 語ぶんの行</text>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>埋め込み＝「ID で行を1本引く」だけ。引いてきた行がその単語のベクトル</figcaption>
</figure>

### 1.3 具体例：実際に表を引いてみる

語彙4語、$d_{\text{model}} = 4$ の小さな例で考えます。埋め込み行列を

$$
E =
\begin{pmatrix}
-0.6 & 0.9 & 0.1 & 0.5 \\
-0.5 & 0.8 & 0.0 & 0.4 \\
\phantom{-}0.7 & 0.3 & -0.4 & 0.2 \\
\phantom{-}0.8 & 0.2 & -0.5 & 0.1
\end{pmatrix}
\begin{array}{l}
\leftarrow \text{0: 走る} \\
\leftarrow \text{1: 寝る} \\
\leftarrow \text{2: 犬} \\
\leftarrow \text{3: 猫}
\end{array}
$$

とすると、「猫」（ID = 3）の埋め込みは第3行（0始まり）なので $[0.8,\ 0.2,\ -0.5,\ 0.1]$、「犬」（ID = 2）は $[0.7,\ 0.3,\ -0.4,\ 0.2]$ です。猫と犬の行が似ていて、走る・寝るの行とは離れている——これが「意味をベクトルで表す」ということです。

### 1.4 コード：NumPy と PyTorch

NumPy なら、ルックアップは単なる**行のインデックス参照**です。

```python
import numpy as np

# 埋め込み行列 E（語彙4語 × d_model=4）。本来は学習で決まるが、ここでは固定値。
E = np.array([
    [-0.6,  0.9,  0.1,  0.5],  # 0: 走る
    [-0.5,  0.8,  0.0,  0.4],  # 1: 寝る
    [ 0.7,  0.3, -0.4,  0.2],  # 2: 犬
    [ 0.8,  0.2, -0.5,  0.1],  # 3: 猫
])

# 文「猫 が 走る」をトークン ID 列にしたもの（「が」は省略して 2語と仮定）
token_ids = np.array([3, 0])          # 猫=3, 走る=0

embedded = E[token_ids]               # ← これだけ！ 各 ID の行をまとめて引く
print(embedded)
# [[ 0.8  0.2 -0.5  0.1]   ← 猫
#  [-0.6  0.9  0.1  0.5]]  ← 走る
```

PyTorch には専用の層 `nn.Embedding` があります。中身は同じ「行を引く表」ですが、学習で表の中身が更新される点が大事です。

```python
import torch
import torch.nn as nn

vocab_size, d_model = 4, 4
embedding = nn.Embedding(vocab_size, d_model)   # V×d_model の表を内部に持つ（最初はランダム）

token_ids = torch.tensor([3, 0])                # 猫=3, 走る=0
embedded = embedding(token_ids)                 # 各 ID の行を引く
print(embedded.shape)                           # torch.Size([2, 4]) = 2トークン × 4次元
```

:::tip[なぜ `nn.Linear` ではなく `nn.Embedding`？]

ワンホットベクトルに重み行列を掛け算すると、結果は「掛けた行列の1行を取り出す」のと同じになります（他の成分は 0 が掛かって消えるため）。つまり埋め込みは**ワンホット × 重み行列**と数学的には等価です。でも、わざわざ巨大なワンホットを作って掛け算するのは無駄なので、`nn.Embedding` は「**掛け算をスキップして、最初から行を引く**」効率的なショートカットになっています。

:::

### 1.5 つながり：これが Transformer の入口

トークン埋め込みは、[全体像の図](#0-全体像5つの部品はブロックのどこにいる)の **①** にあたります。テキストをトークン ID の列に変え、各 ID を埋め込み行列で引くことで、**「トークン数 × $d_{\text{model}}$」の行列**が手に入ります。これがモデルへの入力で、前章で「すでにベクトルになっている」と仮定していたものの正体です。

ただし、この時点では**まだ「順番」の情報が入っていません**。「猫 が 走る」と「走る が 猫」を区別できないのです。それを解決するのが、次の位置エンコーディングです。

---

## 2. 位置エンコーディング

### 2.1 直感：Attention は「順番」を見ていない

前章で見たように、Attention は全単語ペアの関連度を**一気に**計算します。並列に計算できるのが Transformer の強みでした。でも、これには思わぬ落とし穴があります。**Attention の計算には「単語の順番」がどこにも入っていない**のです。

どういうことか、極端な例で確かめましょう。「猫 が 犬 を 追う」と「犬 が 猫 を 追う」は、まったく意味が逆です。でも Attention にとっては、入力は「猫・が・犬・を・追う という単語の集まり」でしかなく、**並び順をシャッフルしても結果が変わりません**（単語ベクトルの集合としては同じだから）。これでは「誰が誰を追うのか」が分からず、言語モデルとして致命的です。

:::warning[なぜ順番が消えるのか]

RNN（前章 2.1）は単語を1つずつ順番に読むので、順番が自然に入っていました。Transformer はその「順番に読む」のをやめて並列化した代わりに、**順番の情報を失ってしまった**のです。だから「順番を表す情報」を、人間が明示的に与えてあげる必要があります。これが位置エンコーディングの役目です。

:::

### 2.2 アイデア：埋め込みに「位置を表すベクトル」を足す

解決策はシンプルです。各単語の埋め込みベクトルに、**「その単語が何番目にあるか」を表すベクトル**を足し込みます。これを **位置エンコーディング（positional encoding, PE）** と呼びます。

$$
\text{入力}_{(pos)} = \underbrace{\text{embed}(\text{単語})}_{\text{意味}} + \underbrace{PE(pos)}_{\text{位置（何番目か）}}
$$

足し算で混ぜるだけ、というのがポイントです。こうすると、同じ「猫」でも 0 番目にある猫と 2 番目にある猫で少し違うベクトルになり、Attention が「順番」を区別できるようになります。

<figure style={{margin: '1.5rem auto', textAlign: 'center', maxWidth: '480px'}}>
  <svg viewBox="0 0 460 170" width="100%" role="img" aria-label="埋め込みに位置ベクトルを足して入力を作る">
    {/* embed */}
    <text x="60" y="22" fontSize="11" fill="currentColor" textAnchor="middle">単語の意味</text>
    <text x="60" y="37" fontSize="10" fill="currentColor" fillOpacity="0.6" textAnchor="middle">embed(猫)</text>
    {[0,1,2,3].map((i) => (
      <rect key={i} x={28 + i*16} y="48" width="14" height={[34,16,40,22][i]} fill="#3B82F6" fillOpacity="0.6" stroke="currentColor" strokeOpacity="0.3" />
    ))}
    <text x="150" y="80" fontSize="22" fill="currentColor" fillOpacity="0.6" textAnchor="middle">+</text>
    {/* PE */}
    <text x="240" y="22" fontSize="11" fill="currentColor" textAnchor="middle">位置（2番目）</text>
    <text x="240" y="37" fontSize="10" fill="currentColor" fillOpacity="0.6" textAnchor="middle">PE(2)</text>
    {[0,1,2,3].map((i) => (
      <rect key={i} x={208 + i*16} y="48" width="14" height={[12,30,8,26][i]} fill="#10B981" fillOpacity="0.6" stroke="currentColor" strokeOpacity="0.3" />
    ))}
    <text x="330" y="80" fontSize="22" fill="currentColor" fillOpacity="0.6" textAnchor="middle">=</text>
    {/* result */}
    <text x="410" y="22" fontSize="11" fill="currentColor" textAnchor="middle">入力ベクトル</text>
    <text x="410" y="37" fontSize="10" fill="currentColor" fillOpacity="0.6" textAnchor="middle">意味＋位置</text>
    {[0,1,2,3].map((i) => (
      <rect key={i} x={378 + i*16} y="48" width="14" height={[46,46,48,48][i]} fill="#EF4444" fillOpacity="0.45" stroke="currentColor" strokeOpacity="0.3" />
    ))}
    <text x="230" y="150" fontSize="10.5" fill="currentColor" fillOpacity="0.7" textAnchor="middle">同じ次元どうしを足すだけ。これで「意味」と「順番」を両方持つベクトルになる</text>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>埋め込み（意味）＋ 位置エンコーディング（順番）＝ ブロックへの入力</figcaption>
</figure>

### 2.3 定義：sin と cos の波で位置を表す

では、$PE(pos)$ の中身——「位置を表すベクトル」はどう作るのでしょう？ 元の論文では、**異なる波長の sin と cos の波**を使います。位置を $pos$、ベクトルの次元番号を $i$、ベクトル全体の長さを $d_{\text{model}}$ とすると、

$$
\begin{aligned}
PE_{(pos,\, 2i)}   &= \sin\!\left(\frac{pos}{10000^{\,2i / d_{\text{model}}}}\right) \\[4pt]
PE_{(pos,\, 2i+1)} &= \cos\!\left(\frac{pos}{10000^{\,2i / d_{\text{model}}}}\right)
\end{aligned}
$$

記号を1つずつ渡します。$pos$ は「何番目の単語か」（0, 1, 2, …）。$2i$ と $2i+1$ は「ベクトルの何番目の成分か」を偶数・奇数に分けたもので、**偶数番目には sin、奇数番目には cos**を入れます。分母の $10000^{2i/d_{\text{model}}}$ は、成分番号 $i$ が大きいほど大きくなる数で、これが**波の波長**（ゆっくり変化するか、速く変化するか）を決めます。

:::note[sin・cos は前章でやった三角関数]

ここで使う sin（サイン）と cos（コサイン）は、[Chapter 1](./chapter1.md) で学んだ三角関数そのものです。値が $-1$ から $1$ の間を波打つ関数、というイメージで十分です。難しい公式は要りません。

:::

### 2.4 なぜ sin・cos なのか：波長の違う「ものさし」を並べる

「位置を表すだけなら $pos$ をそのまま入れればいいのでは？」と思うかもしれません。でも、それだと長い文（$pos$ が何百にもなる）で値が際限なく大きくなり、学習が不安定になります。sin・cos は必ず $-1$〜$1$ に収まるので、その心配がありません。

そして本質は、**「波長の違う波を何本も束ねる」**ところにあります。下の図のように、ベクトルの先頭の成分は**速く変化する波**（短い波長）、後ろの成分ほど**ゆっくり変化する波**（長い波長）になっています。

<figure style={{margin: '1.5rem auto', textAlign: 'center', maxWidth: '520px'}}>
  <svg viewBox="0 0 500 230" width="100%" role="img" aria-label="次元ごとに波長の異なる sin cos の波">
    {/* 軸 */}
    <line x1="40" y1="40" x2="480" y2="40" stroke="currentColor" strokeOpacity="0.15" strokeWidth="1" />
    <line x1="40" y1="105" x2="480" y2="105" stroke="currentColor" strokeOpacity="0.15" strokeWidth="1" />
    <line x1="40" y1="170" x2="480" y2="170" stroke="currentColor" strokeOpacity="0.15" strokeWidth="1" />
    {/* 速い波 */}
    <text x="36" y="44" fontSize="10" fill="currentColor" fillOpacity="0.6" textAnchor="end">成分0</text>
    <polyline points="50,40 70,18 90,40 110,62 130,40 150,18 170,40 190,62 210,40 230,18 250,40 270,62 290,40 310,18 330,40 350,62 370,40 390,18 410,40 430,62 450,40" fill="none" stroke="#3B82F6" strokeWidth="2" />
    {/* 中くらい */}
    <text x="36" y="109" fontSize="10" fill="currentColor" fillOpacity="0.6" textAnchor="end">成分2</text>
    <polyline points="50,105 90,80 130,105 170,130 210,105 250,80 290,105 330,130 370,105 410,80 450,105" fill="none" stroke="#10B981" strokeWidth="2" />
    {/* ゆっくり */}
    <text x="36" y="174" fontSize="10" fill="currentColor" fillOpacity="0.6" textAnchor="end">成分4</text>
    <polyline points="50,170 130,142 210,170 290,198 370,170 450,150" fill="none" stroke="#EF4444" strokeWidth="2" />
    {/* pos 軸ラベル */}
    <text x="250" y="222" fontSize="10.5" fill="currentColor" fillOpacity="0.7" textAnchor="middle">→ 単語の位置 pos（右へ行くほど後ろの単語）</text>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>先頭の成分は速い波、後ろの成分ほど遅い波。波の「組み合わせ」で各位置に固有のパターンができる</figcaption>
</figure>

これは時計の **秒針・分針・時針**にそっくりです。秒針（速い波）だけでは1分後に同じ位置に戻ってしまいますが、分針・時針（遅い波）と**組み合わせれば**、どの瞬間も一意に表せます。同じように、波長の違う sin・cos を束ねると、**どの位置 $pos$ も世界に1つだけのパターン**になり、しかも「2つ隣」「5つ先」といった**相対的な位置関係**も波のズレとして自然に表現できます。

### 2.5 具体例と実装

$d_{\text{model}} = 4$ で、位置 $pos = 0, 1, 2$ の位置エンコーディングを計算してみます。NumPy で書くと次のとおりです。

```python
import numpy as np

def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, None]          # 位置 0,1,2,... を縦に並べる（列ベクトル）
    i   = np.arange(d_model)[None, :]          # 次元番号 0,1,2,...（横に並べる）
    # 分母 10000^(2i/d_model)。偶数・奇数で同じ波長を共有するため i//2 を使う
    denom = np.power(10000, (2 * (i // 2)) / d_model)
    angle = pos / denom                        # 角度（pos ÷ 波長）
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angle[:, 0::2])       # 偶数番目の成分 → sin
    pe[:, 1::2] = np.cos(angle[:, 1::2])       # 奇数番目の成分 → cos
    return pe

print(np.round(positional_encoding(3, 4), 3))
# [[ 0.     1.     0.     1.   ]   ← pos=0: sin0=0, cos0=1
#  [ 0.841  0.54   0.01   1.   ]   ← pos=1
#  [ 0.909 -0.416  0.02   1.   ]]  ← pos=2
```

`pos = 0` の行が `[0, 1, 0, 1]` になるのは、$\sin 0 = 0$、$\cos 0 = 1$ だからです。位置が進むにつれて各成分が波打って変化し、行（＝位置）ごとに違うパターンになっているのが分かります。

実際にモデルへ入れるときは、この位置エンコーディングを**埋め込みに足す**だけです。

```python
# 埋め込み（2.5節までの E[token_ids]）と同じ形にして足す
embedded = np.array([[ 0.8,  0.2, -0.5,  0.1],   # 猫（0番目）
                     [-0.6,  0.9,  0.1,  0.5]])  # 走る（1番目）
pe = positional_encoding(seq_len=2, d_model=4)
model_input = embedded + pe                       # ← 意味 ＋ 位置
```

PyTorch でも考え方は同じで、計算した位置エンコーディングを入力に加算します（実装では `register_buffer` で固定テーブルとして持つのが定番です）。

:::tip[学習する位置エンコーディングもある]

ここで紹介した sin・cos の式は**固定**（学習しない）方式です。これとは別に、位置ごとのベクトルを**埋め込みと同じように学習で獲得する**方式（learned positional embedding）もあり、GPT 系など多くのモデルで使われています。さらに新しいモデルでは、回転を使う **RoPE** など発展形も登場しています。いずれも「順番の情報を足す」という目的は同じです。

:::

### 2.6 つながり：これでブロックに入る準備が整う

位置エンコーディングは[全体像](#0-全体像5つの部品はブロックのどこにいる)の **②** です。①の埋め込み（意味）に②（位置）を足すことで、**「意味」と「順番」の両方を持った入力ベクトル**が完成し、ようやく Transformer ブロックの中（Attention）に送り込めます。

ここからは「ブロックの中身」の話に移ります。Attention は前章で学んだので、次はその出力を加工する **フィードフォワード層**です。

---

## 3. フィードフォワード層（FFN）

### 3.1 直感：Attention が「混ぜる」役、FFN が「考える」役

Attention の役割をひとことで言うと、「**単語どうしを見渡して、関連する情報を混ぜ合わせる**」ことでした。出力は Value の加重和——つまり、ほかの単語のベクトルを足し合わせたものです。

でも、よく考えると Attention の中心は**足し算**（加重和）です。足し算だけをいくら重ねても、表現できる関係には限界があります（数学的には「線形」の範囲を超えられません）。そこで、混ぜ合わせたあとの各単語ベクトルを、**1つずつ取り出して、もっと複雑に加工する**部品が必要になります。それが **フィードフォワード層（Feed-Forward Network, FFN）** です。

役割分担をイメージで言うと、こうです。

- **Attention** … 教室で「周りの人の意見を聞いて回る」（情報を集めて混ぜる）
- **FFN** … 集めた意見を持ち帰って「自分の頭の中で1人でじっくり考える」（各単語を個別に深く加工する）

ここで大事なのは、FFN は **各単語（各位置）に対して、まったく同じ処理を独立に適用する**という点です。0番目の単語も、5番目の単語も、同じ FFN を通ります。お隣の単語を覗き見ることはしません（混ぜるのは Attention の仕事だから）。

<figure style={{margin: '1.5rem auto', textAlign: 'center', maxWidth: '460px'}}>
  <svg viewBox="0 0 440 150" width="100%" role="img" aria-label="FFN は各位置に同じネットを独立に適用">
    {[
      {x: 30,  w: '猫'},
      {x: 150, w: 'が'},
      {x: 270, w: '走る'}
    ].map((t, idx) => (
      <g key={idx}>
        <rect x={t.x} y="14" width="90" height="22" rx="4" fill="#EF4444" fillOpacity="0.14" stroke="#EF4444" strokeOpacity="0.7" strokeWidth="1.2" />
        <text x={t.x + 45} y="29" fontSize="10.5" fill="currentColor" textAnchor="middle">{t.w}（Attn後）</text>
        <line x1={t.x + 45} y1="36" x2={t.x + 45} y2="56" stroke="currentColor" strokeOpacity="0.5" strokeWidth="1.3" />
        <rect x={t.x} y="58" width="90" height="26" rx="4" fill="#3B82F6" fillOpacity="0.16" stroke="#3B82F6" strokeWidth="1.4" />
        <text x={t.x + 45} y="75" fontSize="11" fill="currentColor" textAnchor="middle">FFN（同じ）</text>
        <line x1={t.x + 45} y1="84" x2={t.x + 45} y2="104" stroke="currentColor" strokeOpacity="0.5" strokeWidth="1.3" />
        <rect x={t.x} y="106" width="90" height="22" rx="4" fill="#10B981" fillOpacity="0.16" stroke="#10B981" strokeOpacity="0.7" strokeWidth="1.2" />
        <text x={t.x + 45} y="121" fontSize="10.5" fill="currentColor" textAnchor="middle">加工後</text>
      </g>
    ))}
    <text x="220" y="146" fontSize="10.5" fill="currentColor" fillOpacity="0.7" textAnchor="middle">各単語が、隣を見ずに「同じFFN」を独立に通る（重みは共有）</text>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>FFN は位置ごとに独立。すべての位置で同じ重みを使う（位置ごとに別物ではない）</figcaption>
</figure>

### 3.2 定義：広げて → 非線形 → 戻す

FFN の中身は、実はとてもシンプルな**2層のニューラルネット**です。式で書くと、

$$
\text{FFN}(x) = \max(0,\ x W_1 + b_1)\, W_2 + b_2
$$

です。順番に見ていきましょう。$x$ は1つの単語のベクトル（長さ $d_{\text{model}}$）。

1. **広げる**：$x W_1 + b_1$ で、いったん**大きな次元** $d_{\text{ff}}$ に引き伸ばす（典型的には $d_{\text{ff}} = 4 \times d_{\text{model}}$）。
2. **非線形をかける**：$\max(0,\ \cdot)$ は **ReLU** という関数で、「負の値を 0 にする」だけのシンプルな操作。これが「**非線形性**」を生み、足し算だけでは作れない複雑な変換を可能にします。
3. **戻す**：$W_2$ で元の次元 $d_{\text{model}}$ に縮める。

$W_1, W_2$ は重み行列、$b_1, b_2$ はバイアス（下駄ばき）で、すべて学習で決まります。

:::note[ReLU と GELU]

$\max(0, z)$ は **ReLU（Rectified Linear Unit）** と呼ばれ、「入力が負なら 0、正ならそのまま通す」という関数です。グラフにすると、原点で折れ曲がった「レ」の字の形になります。この**折れ曲がり**こそが非線形性の源で、これがあるおかげでネットワークは曲線的な複雑な関係を学べます。最近の LLM では、ReLU を滑らかにした **GELU** という関数がよく使われますが、「広げる → 非線形 → 戻す」という骨格は同じです。

:::

<figure style={{margin: '1.5rem auto', textAlign: 'center', maxWidth: '420px'}}>
  <svg viewBox="0 0 400 170" width="100%" role="img" aria-label="FFN の次元の広げて戻す形">
    {/* 入力 d_model */}
    <text x="45" y="20" fontSize="10.5" fill="currentColor" textAnchor="middle">入力</text>
    <text x="45" y="150" fontSize="10" fill="currentColor" fillOpacity="0.6" textAnchor="middle">d_model</text>
    {[0,1,2,3].map((i) => (
      <rect key={i} x="30" y={45 + i*16} width="30" height="14" fill="#3B82F6" fillOpacity="0.5" stroke="currentColor" strokeOpacity="0.3" />
    ))}
    {/* W1 矢印 */}
    <text x="115" y="80" fontSize="10" fill="currentColor" fillOpacity="0.7" textAnchor="middle">W₁</text>
    <text x="115" y="95" fontSize="9.5" fill="currentColor" fillOpacity="0.6" textAnchor="middle">広げる</text>
    <text x="130" y="105" fontSize="16" fill="currentColor" fillOpacity="0.5" textAnchor="middle">→</text>
    {/* 中間 d_ff (ReLU) */}
    <text x="200" y="20" fontSize="10.5" fill="currentColor" textAnchor="middle">中間（ReLU）</text>
    <text x="200" y="150" fontSize="10" fill="currentColor" fillOpacity="0.6" textAnchor="middle">d_ff = 4×d_model</text>
    {[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15].map((i) => (
      <rect key={i} x="170" y={28 + i*7} width="30" height="6" fill="#EF4444" fillOpacity="0.4" stroke="currentColor" strokeOpacity="0.25" />
    ))}
    {/* W2 矢印 */}
    <text x="285" y="80" fontSize="10" fill="currentColor" fillOpacity="0.7" textAnchor="middle">W₂</text>
    <text x="285" y="95" fontSize="9.5" fill="currentColor" fillOpacity="0.6" textAnchor="middle">戻す</text>
    <text x="300" y="105" fontSize="16" fill="currentColor" fillOpacity="0.5" textAnchor="middle">→</text>
    {/* 出力 d_model */}
    <text x="355" y="20" fontSize="10.5" fill="currentColor" textAnchor="middle">出力</text>
    <text x="355" y="150" fontSize="10" fill="currentColor" fillOpacity="0.6" textAnchor="middle">d_model</text>
    {[0,1,2,3].map((i) => (
      <rect key={i} x="340" y={45 + i*16} width="30" height="14" fill="#10B981" fillOpacity="0.55" stroke="currentColor" strokeOpacity="0.3" />
    ))}
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>いったん広い空間に持ち上げ、ReLU で非線形に加工してから、元の次元に戻す</figcaption>
</figure>

### 3.3 具体例：手で1ステップ追う

$d_{\text{model}} = 2$、$d_{\text{ff}} = 4$ の小さな例で、1単語ぶんを計算してみます。入力 $x = [1,\ -2]$、

$$
W_1 = \begin{pmatrix} 1 & 0 & -1 & 2 \\ 0 & 1 & 1 & -1 \end{pmatrix}, \quad b_1 = [0,0,0,0]
$$

とすると、$x W_1 = [1\cdot1 + (-2)\cdot0,\ \ 1\cdot0 + (-2)\cdot1,\ \ 1\cdot(-1)+(-2)\cdot1,\ \ 1\cdot2+(-2)\cdot(-1)] = [1,\ -2,\ -3,\ 4]$。

ここで ReLU（負を 0 に）を通すと $[1,\ 0,\ 0,\ 4]$。負だった2つの成分が消えました。これが「非線形に情報を選別する」イメージです。このあと $W_2$ で2次元に戻すと、加工済みの新しいベクトルが得られます。

### 3.4 コード：NumPy と PyTorch

```python
import numpy as np

def relu(z):
    return np.maximum(0, z)

def ffn(x, W1, b1, W2, b2):
    h = relu(x @ W1 + b1)     # ① 広げて ② ReLU で非線形
    return h @ W2 + b2        # ③ 元の次元に戻す

x  = np.array([[1.0, -2.0]])               # 1トークン（d_model=2）
W1 = np.array([[1, 0, -1, 2],
               [0, 1,  1, -1]], dtype=float) # d_model=2 → d_ff=4
b1 = np.zeros(4)
W2 = np.random.randn(4, 2) * 0.1           # d_ff=4 → d_model=2
b2 = np.zeros(2)

print(ffn(x, W1, b1, W2, b2).shape)        # (1, 2) ← 入力と同じ d_model
```

PyTorch では `nn.Linear` を2つ重ねるだけです。

```python
import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)   # 広げる
        self.fc2 = nn.Linear(d_ff, d_model)   # 戻す
        self.act = nn.ReLU()                  # 非線形（実際は nn.GELU() も多い）

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

ff = FeedForward(d_model=4, d_ff=16)
x  = torch.randn(2, 3, 4)                     # バッチ2 × 3トークン × 4次元
print(ff(x).shape)                            # torch.Size([2, 3, 4]) 形は変わらない
```

入力と出力で形（次元）が変わらないことに注目してください。だからブロックの中にそのまま差し込めます。

### 3.5 つながり：ブロックを「賢く」する非線形パート

FFN は[全体像](#0-全体像5つの部品はブロックのどこにいる)の **③** で、Attention のすぐ後ろに置かれます。「**Attention で混ぜる → FFN で深く加工する**」のワンセットが、Transformer ブロックの中核的な計算です。

ここまでで「混ぜる（Attention）」「加工する（FFN）」という2つの主処理がそろいました。ところが、こうした層を何十段も積み重ねると、**学習がうまく進まなくなる**という別の問題が出てきます。それを支えるのが、残る2部品——スキップ接続とレイヤー正規化です。

---

## 4. スキップ接続（残差接続）

### 4.1 直感：層を「飛び越える近道」を作る

Transformer は強くなるために、ブロックを何十段も積み重ねます。ところが、層を深くするほど学習が難しくなる、という根深い問題があります。理由は **勾配消失**です。

学習では、出力の誤差を**逆向きにたどって**各層の重みを調整します（この「たどる量」を勾配と呼びます）。層が深いと、この勾配が入口に届くまでに何度も掛け算され、だんだん小さくなって**ほぼ 0 になってしまう**ことがあります。勾配が 0 だと「どう直せばいいか」の信号が届かず、入口に近い層が学習できません。

これを解決するのが **スキップ接続（skip connection）**、別名 **残差接続（residual connection）** です。アイデアは拍子抜けするほど単純で、**層の入力 $x$ を、その層の出力にそのまま足す**だけです。

$$
y = x + F(x)
$$

ここで $F$ は層の処理（Attention や FFN）です。$x$ がそのまま出力に現れる「**近道（バイパス）**」ができるのがポイントです。

<figure style={{margin: '1.5rem auto', textAlign: 'center', maxWidth: '400px'}}>
  <svg viewBox="0 0 360 210" width="100%" role="img" aria-label="スキップ接続：入力を出力に足す近道">
    <defs>
      <marker id="resArrow" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto">
        <path d="M0,0 L6,3 L0,6 Z" fill="currentColor" fillOpacity="0.65" />
      </marker>
    </defs>
    {/* 入力 */}
    <circle cx="180" cy="24" r="5" fill="currentColor" fillOpacity="0.6" />
    <text x="195" y="28" fontSize="11" fill="currentColor">入力 x</text>
    {/* 分岐点から下へ */}
    <line x1="180" y1="29" x2="180" y2="60" stroke="currentColor" strokeOpacity="0.6" strokeWidth="1.4" markerEnd="url(#resArrow)" />
    {/* 層 F */}
    <rect x="120" y="62" width="120" height="40" rx="6" fill="#3B82F6" fillOpacity="0.16" stroke="#3B82F6" strokeWidth="1.6" />
    <text x="180" y="86" fontSize="12" fill="currentColor" textAnchor="middle">層 F（Attn / FFN）</text>
    <line x1="180" y1="102" x2="180" y2="132" stroke="currentColor" strokeOpacity="0.6" strokeWidth="1.4" markerEnd="url(#resArrow)" />
    {/* 近道（右に回り込む曲線） */}
    <path d="M 180 26 H 320 V 145 H 200" fill="none" stroke="#EF4444" strokeWidth="2" strokeDasharray="5 4" markerEnd="url(#resArrow)" />
    <text x="326" y="90" fontSize="10.5" fill="#EF4444" textAnchor="middle" transform="rotate(90 326 90)">近道（x をそのまま）</text>
    {/* 足し算ノード */}
    <circle cx="180" cy="145" r="13" fill="currentColor" fillOpacity="0.08" stroke="currentColor" strokeOpacity="0.6" strokeWidth="1.4" />
    <text x="180" y="150" fontSize="15" fill="currentColor" textAnchor="middle">＋</text>
    {/* 出力 */}
    <line x1="180" y1="158" x2="180" y2="184" stroke="currentColor" strokeOpacity="0.6" strokeWidth="1.4" markerEnd="url(#resArrow)" />
    <text x="180" y="202" fontSize="11" fill="currentColor" textAnchor="middle">出力 y = x + F(x)</text>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>入力 x は層 F を通る道と、何もせず足し算へ向かう「近道」に分かれ、最後に合流する</figcaption>
</figure>

### 4.2 なぜこれが効くのか：2つのうれしいこと

スキップ接続が効く理由は、大きく2つあります。

**① 勾配の「高速道路」ができる。** 出力に $x$ がそのまま含まれるので、逆向きにたどる勾配も、層 $F$ を通らずに**近道をまっすぐ入口まで届ける**ルートを持ちます。途中で 0 に薄まらないので、深いネットでも入口の層まで学習信号が届きます。

**② 「差分」だけ学べばよくなる。** 式を見ると、層 $F$ は「出力そのもの」ではなく「**入力からの変化分（残差）**」$F(x) = y - x$ を学べばよいことになります。もし「この層では何もしないのが最適」なら、$F(x) = 0$（つまり出力 $= x$）にすればよく、これは**簡単に学習できます**。何もしないことすら難しかった深いネットに、「とりあえず素通り」という安全なスタート地点を与えるイメージです。

:::tip[「残差」という呼び名の由来]

$F(x) = y - x$ は「出力から入力を引いた**残り**」なので、**残差（residual）** と呼ばれます。層に「ゼロから正解を作れ」と要求する代わりに、「**いまの $x$ に、どんな修正を加えるか**」だけを学ばせる——これが残差接続の発想です。画像認識の ResNet で大成功し、Transformer にも受け継がれました。

:::

### 4.3 具体例とコード

数値はとても簡単です。入力 $x = [1.0,\ 2.0]$、層の出力 $F(x) = [0.1,\ -0.3]$ なら、スキップ接続つきの出力は単に足すだけ：

$$
y = x + F(x) = [1.0 + 0.1,\ \ 2.0 + (-0.3)] = [1.1,\ 1.7]
$$

```python
import numpy as np

x  = np.array([1.0, 2.0])
Fx = np.array([0.1, -0.3])     # 層（Attention や FFN）の出力
y  = x + Fx                    # ← スキップ接続はこの足し算だけ
print(y)                       # [1.1 1.7]
```

PyTorch でも、層を呼んだ結果に入力を足すだけです。

```python
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, sublayer):
        super().__init__()
        self.sublayer = sublayer       # Attention や FFN を入れる

    def forward(self, x):
        return x + self.sublayer(x)    # y = x + F(x)
```

入力と出力の次元が同じでないと足せないことに注意してください。FFN（3節）が「入力と同じ $d_{\text{model}}$ で出力する」ように作られていたのは、まさにこのスキップ接続で足し合わせるためでもあります。

### 4.4 つながり：ブロックを深く積むための土台

スキップ接続は[全体像](#0-全体像5つの部品はブロックのどこにいる)の **④** で、Attention と FFN それぞれを「近道つき」で包みます。つまり実際のブロックでは、$\text{出力} = x + \text{Attention}(x)$、$\text{出力} = x + \text{FFN}(x)$ のように使われます。これがあるおかげで、Transformer は何十段も積み重ねても学習できるのです。

ただし、足し算をくり返すと値がどんどん大きくなったり、層によってスケールがバラバラになったりします。その暴れを抑える最後の部品が、レイヤー正規化です。

---

## 5. レイヤー正規化

### 5.1 直感：ベクトルの「目盛り」をそろえる

ここまでの部品を通ると、ベクトルの値は層ごとにバラバラなスケールになります。ある層では $[100, -98, 5]$ のように大きく暴れ、別の層では $[0.01, -0.02, 0.005]$ のように小さくしぼむ——こうしたムラがあると、学習がとても不安定になります（値が大きすぎて発散したり、小さすぎて進まなかったり）。

そこで、各ベクトルを通すたびに **「平均 0・ばらつき 1」に整え直す**のが **レイヤー正規化（Layer Normalization, LayerNorm）** です。身長と体重のように単位がバラバラな数値を「偏差値」に直して比べやすくするのと、まったく同じ発想です。値の**位置（平均）と広がり（ばらつき）の両方**をそろえることで、どの層でも計算が安定した範囲で進むようになります。

<figure style={{margin: '1.5rem auto', textAlign: 'center', maxWidth: '460px'}}>
  <svg viewBox="0 0 440 200" width="100%" role="img" aria-label="正規化前後でベクトルの値のばらつきがそろう">
    {/* baseline */}
    <line x1="20" y1="120" x2="200" y2="120" stroke="currentColor" strokeOpacity="0.3" strokeWidth="1.2" />
    <line x1="240" y1="120" x2="420" y2="120" stroke="currentColor" strokeOpacity="0.3" strokeWidth="1.2" />
    <text x="110" y="18" fontSize="11.5" fill="currentColor" textAnchor="middle" fontWeight="bold">正規化前</text>
    <text x="110" y="34" fontSize="10" fill="currentColor" fillOpacity="0.6" textAnchor="middle">バラバラに暴れている</text>
    {/* before bars (大きく暴れる) */}
    <rect x="35"  y="50"  width="22" height="70" fill="#EF4444" fillOpacity="0.5" />
    <rect x="70"  y="105" width="22" height="15" fill="#EF4444" fillOpacity="0.5" />
    <rect x="105" y="120" width="22" height="48" fill="#3B82F6" fillOpacity="0.5" />
    <rect x="140" y="78"  width="22" height="42" fill="#EF4444" fillOpacity="0.5" />
    {/* arrow */}
    <text x="220" y="100" fontSize="20" fill="currentColor" fillOpacity="0.55" textAnchor="middle">→</text>
    <text x="220" y="118" fontSize="9" fill="currentColor" fillOpacity="0.6" textAnchor="middle">LN</text>
    {/* after */}
    <text x="330" y="18" fontSize="11.5" fill="currentColor" textAnchor="middle" fontWeight="bold">正規化後</text>
    <text x="330" y="34" fontSize="10" fill="currentColor" fillOpacity="0.6" textAnchor="middle">平均0・ばらつき1にそろう</text>
    <rect x="255" y="92"  width="22" height="28" fill="#10B981" fillOpacity="0.6" />
    <rect x="290" y="120" width="22" height="26" fill="#10B981" fillOpacity="0.6" />
    <rect x="325" y="120" width="22" height="22" fill="#10B981" fillOpacity="0.6" />
    <rect x="360" y="98" width="22" height="22" fill="#10B981" fillOpacity="0.6" />
    <text x="220" y="190" fontSize="10.5" fill="currentColor" fillOpacity="0.7" textAnchor="middle">1本のベクトルの中の成分を、平均0・ばらつき1に整える</text>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>暴れた値（左）を、平均0・ばらつき1のそろった値（右）に整え直す</figcaption>
</figure>

### 5.2 定義：平均を引いて、ばらつきで割る

1本のベクトル $x = [x_1, x_2, \dots, x_d]$（長さ $d = d_{\text{model}}$）に対して、レイヤー正規化は次のように計算します。まず、そのベクトルの中の**平均** $\mu$ と**ばらつき**（分散 $\sigma^2$）を求めます。

$$
\mu = \frac{1}{d}\sum_{k=1}^{d} x_k, \qquad
\sigma^2 = \frac{1}{d}\sum_{k=1}^{d} (x_k - \mu)^2
$$

そして、各成分から平均を引き、ばらつきの平方根で割って整えます。

$$
\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

記号を渡します。$\dfrac{x - \mu}{\sqrt{\sigma^2}}$ の部分が正規化の本体で、「平均を 0 に、ばらつきを 1 に」する操作です。$\epsilon$（イプシロン）は $10^{-5}$ くらいの極小の数で、ばらつきが 0 のときに**ゼロ割りを防ぐ**ためのお守りです。$\odot$ は成分ごとの掛け算を表します。

最後の $\gamma$（ガンマ）と $\beta$（ベータ）は**学習で決まるパラメータ**で、「整えたあと、もう一度どれくらい伸縮・移動させるか」を調整します。これにより「正規化はするけれど、必要なら少し崩す」という柔軟さをモデルに残しています。

:::note[バッチ正規化との違い：どの方向に平均を取るか]

正規化には **バッチ正規化（BatchNorm）** という似た手法もありますが、**平均・ばらつきを取る方向**が違います。レイヤー正規化は「**1つのベクトルの中（特徴の方向）**」で計算するので、他のサンプルや文の長さに影響されません。だから、文ごとに長さが変わる言語データと相性がよく、Transformer ではレイヤー正規化が使われます。

:::

### 5.3 具体例：手で計算してみる

$x = [2,\ 4,\ 6]$ を正規化してみます（$\gamma = 1$、$\beta = 0$、$\epsilon$ は無視）。

- 平均：$\mu = (2 + 4 + 6)/3 = 4$
- 分散：$\sigma^2 = \dfrac{(2-4)^2 + (4-4)^2 + (6-4)^2}{3} = \dfrac{4 + 0 + 4}{3} = \dfrac{8}{3} \approx 2.67$、標準偏差 $\sigma \approx 1.63$
- 正規化：$\dfrac{[2,4,6] - 4}{1.63} = \dfrac{[-2,\ 0,\ 2]}{1.63} \approx [-1.22,\ 0,\ 1.22]$

平均が 0 になり（$-1.22 + 0 + 1.22 = 0$）、左右対称のそろった値になりました。元の「2, 4, 6」がどんなスケールでも、こうして決まった範囲に収まります。

### 5.4 コード：NumPy と PyTorch

```python
import numpy as np

def layer_norm(x, gamma, beta, eps=1e-5):
    mu  = x.mean(axis=-1, keepdims=True)        # 最後の次元（特徴方向）で平均
    var = x.var(axis=-1, keepdims=True)         # 同じ方向でばらつき
    x_hat = (x - mu) / np.sqrt(var + eps)       # 平均0・ばらつき1に整える
    return gamma * x_hat + beta                 # 学習パラメータで伸縮・移動

x = np.array([[2.0, 4.0, 6.0]])
print(np.round(layer_norm(x, gamma=1.0, beta=0.0), 2))   # [[-1.22  0.    1.22]]
```

PyTorch には `nn.LayerNorm` があり、`gamma`・`beta` を内部に持って自動で学習します。

```python
import torch
import torch.nn as nn

ln = nn.LayerNorm(4)              # d_model=4 の最後の次元を正規化
x  = torch.randn(2, 3, 4)         # バッチ2 × 3トークン × 4次元
print(ln(x).shape)                # torch.Size([2, 3, 4]) 形は変わらない
```

### 5.5 つながり：「Add & Norm」として効く

レイヤー正規化は[全体像](#0-全体像5つの部品はブロックのどこにいる)の **⑤** で、スキップ接続（④）とペアで使われます。Attention や FFN の直後に「**スキップ接続で足して（Add）、レイヤー正規化で整える（Norm）**」がワンセットで入り、合わせて **Add & Norm** と呼ばれます。これで値が暴れず、深いネットでも安定して学習できます。

:::tip[Post-LN と Pre-LN]

レイヤー正規化を「足し算の**後**」に置くか「層に入る**前**」に置くかで2流派あります。元論文は後（**Post-LN**：$\text{LN}(x + F(x))$）でしたが、深いモデルでは学習が不安定になりやすいため、最近の LLM の多くは前（**Pre-LN**：$x + F(\text{LN}(x))$）を採用しています。次節の組み立てでは、まず分かりやすい Post-LN の形で示します。

:::

これで5つの部品がすべてそろいました。いよいよ全部を組み合わせて、1つの Transformer ブロックを完成させましょう！

---

## 6. 全部組み合わせて Transformer ブロック完成！

### 6.1 データの旅を最初から最後までたどる

5つの部品と前章の Attention がそろったので、ついに **テキストが Transformer を通り抜ける流れ**を、最初から最後まで一気にたどれます。下の図が、その完成形です。

<figure style={{margin: '1.5rem auto', textAlign: 'center', maxWidth: '480px'}}>
  <svg viewBox="0 0 460 540" width="100%" role="img" aria-label="Transformer ブロックの完成形とデータの流れ">
    <defs>
      <marker id="fullArrow" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto">
        <path d="M0,0 L6,3 L0,6 Z" fill="currentColor" fillOpacity="0.6" />
      </marker>
    </defs>
    {/* テキスト */}
    <rect x="150" y="8" width="160" height="28" rx="6" fill="currentColor" fillOpacity="0.08" stroke="currentColor" strokeOpacity="0.5" strokeWidth="1.2" />
    <text x="230" y="26" fontSize="11" fill="currentColor" textAnchor="middle">「猫 が 走る」</text>
    {/* ① 埋め込み */}
    <rect x="130" y="52" width="200" height="30" rx="6" fill="#3B82F6" fillOpacity="0.16" stroke="#3B82F6" strokeWidth="1.5" />
    <text x="230" y="71" fontSize="11" fill="currentColor" textAnchor="middle">① トークン埋め込み</text>
    {/* ② 位置 */}
    <rect x="130" y="92" width="200" height="30" rx="6" fill="#3B82F6" fillOpacity="0.16" stroke="#3B82F6" strokeWidth="1.5" />
    <text x="230" y="111" fontSize="11" fill="currentColor" textAnchor="middle">② ＋ 位置エンコーディング</text>
    {/* ブロック枠 */}
    <rect x="35" y="140" width="390" height="350" rx="10" fill="currentColor" fillOpacity="0.03" stroke="currentColor" strokeOpacity="0.4" strokeWidth="1.4" strokeDasharray="6 4" />
    <text x="50" y="160" fontSize="10.5" fill="currentColor" fillOpacity="0.7">Transformer ブロック（この中を ×N 回くり返す）</text>
    {/* Attn */}
    <rect x="110" y="172" width="240" height="32" rx="6" fill="#EF4444" fillOpacity="0.14" stroke="#EF4444" strokeWidth="1.5" />
    <text x="230" y="192" fontSize="11" fill="currentColor" textAnchor="middle">マルチヘッドアテンション（前章）</text>
    {/* Add&Norm 1 */}
    <rect x="110" y="216" width="240" height="44" rx="6" fill="#10B981" fillOpacity="0.14" stroke="#10B981" strokeWidth="1.5" />
    <text x="230" y="234" fontSize="11" fill="currentColor" textAnchor="middle">Add ＆ Norm</text>
    <text x="230" y="250" fontSize="9.5" fill="currentColor" fillOpacity="0.7" textAnchor="middle">④ スキップ接続で足す ＋ ⑤ レイヤー正規化</text>
    {/* FFN */}
    <rect x="110" y="316" width="240" height="32" rx="6" fill="#3B82F6" fillOpacity="0.16" stroke="#3B82F6" strokeWidth="1.5" />
    <text x="230" y="336" fontSize="11" fill="currentColor" textAnchor="middle">③ フィードフォワード層</text>
    {/* Add&Norm 2 */}
    <rect x="110" y="360" width="240" height="44" rx="6" fill="#10B981" fillOpacity="0.14" stroke="#10B981" strokeWidth="1.5" />
    <text x="230" y="378" fontSize="11" fill="currentColor" textAnchor="middle">Add ＆ Norm</text>
    <text x="230" y="394" fontSize="9.5" fill="currentColor" fillOpacity="0.7" textAnchor="middle">④ スキップ接続で足す ＋ ⑤ レイヤー正規化</text>
    {/* 出力 */}
    <rect x="150" y="500" width="160" height="30" rx="6" fill="currentColor" fillOpacity="0.08" stroke="currentColor" strokeOpacity="0.5" strokeWidth="1.2" />
    <text x="230" y="519" fontSize="11" fill="currentColor" textAnchor="middle">次のブロック／最終出力へ</text>
    {/* 縦の矢印 */}
    <line x1="230" y1="36" x2="230" y2="50" stroke="currentColor" strokeOpacity="0.6" strokeWidth="1.4" markerEnd="url(#fullArrow)" />
    <line x1="230" y1="82" x2="230" y2="90" stroke="currentColor" strokeOpacity="0.6" strokeWidth="1.4" markerEnd="url(#fullArrow)" />
    <line x1="230" y1="122" x2="230" y2="170" stroke="currentColor" strokeOpacity="0.6" strokeWidth="1.4" markerEnd="url(#fullArrow)" />
    <line x1="230" y1="204" x2="230" y2="214" stroke="currentColor" strokeOpacity="0.6" strokeWidth="1.4" markerEnd="url(#fullArrow)" />
    <line x1="230" y1="260" x2="230" y2="314" stroke="currentColor" strokeOpacity="0.6" strokeWidth="1.4" markerEnd="url(#fullArrow)" />
    <line x1="230" y1="348" x2="230" y2="358" stroke="currentColor" strokeOpacity="0.6" strokeWidth="1.4" markerEnd="url(#fullArrow)" />
    <line x1="230" y1="404" x2="230" y2="498" stroke="currentColor" strokeOpacity="0.6" strokeWidth="1.4" markerEnd="url(#fullArrow)" />
    {/* スキップ接続の近道（Attn を迂回） */}
    <path d="M 95 156 V 238 H 108" fill="none" stroke="#EF4444" strokeOpacity="0.7" strokeWidth="1.6" strokeDasharray="5 4" markerEnd="url(#fullArrow)" />
    <path d="M 95 282 V 382 H 108" fill="none" stroke="#EF4444" strokeOpacity="0.7" strokeWidth="1.6" strokeDasharray="5 4" markerEnd="url(#fullArrow)" />
    <text x="78" y="200" fontSize="9" fill="#EF4444" textAnchor="middle" transform="rotate(90 78 200)">skip</text>
    <text x="78" y="332" fontSize="9" fill="#EF4444" textAnchor="middle" transform="rotate(90 78 332)">skip</text>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>完成した Transformer ブロック。赤い破線が「層を迂回する」スキップ接続の近道</figcaption>
</figure>

旅の流れを言葉でまとめると、こうです。

1. テキストをトークンに分け、**①埋め込み**で各トークンをベクトルにする
2. **②位置エンコーディング**を足して「順番」の情報を入れる
3. ここからブロックの中：**マルチヘッドアテンション**で単語どうしの情報を混ぜる
4. その出力を **④スキップ接続**で入力に足し、**⑤レイヤー正規化**で整える（Add & Norm）
5. **③FFN**で各単語を個別に深く加工する
6. また **④スキップ接続 ＋ ⑤レイヤー正規化**で整える
7. できあがった出力を、次の同じ形のブロックへ……これを N 回くり返す

### 6.2 式でまとめる

入力 $x$（埋め込み＋位置エンコーディング済み）に対して、1つの Transformer ブロックは次のように書けます（分かりやすい Post-LN の形）。

$$
\begin{aligned}
x' &= \text{LayerNorm}\bigl(x + \text{MultiHead}(x)\bigr) \\[4pt]
y  &= \text{LayerNorm}\bigl(x' + \text{FFN}(x')\bigr)
\end{aligned}
$$

1行目が「Attention → Add & Norm」、2行目が「FFN → Add & Norm」です。$x + (\dots)$ がスキップ接続、$\text{LayerNorm}(\dots)$ がレイヤー正規化。この章で学んだ部品が、そのまま式の中に並んでいるのが見て取れます。

### 6.3 コード：1ブロックを PyTorch で組む

前章の `nn.MultiheadAttention` と、この章で作った FFN・スキップ接続・レイヤー正規化を組み合わせると、1つのブロックがそのまま書けます。

```python
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)  # 前章
        self.ffn  = nn.Sequential(                                             # ③ FFN
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)   # ⑤ レイヤー正規化
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Attention → ④スキップ接続で足す → ⑤正規化
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)         # x + F(x) が ④ スキップ接続
        # FFN → ④スキップ接続で足す → ⑤正規化
        x = self.norm2(x + self.ffn(x))
        return x

block = TransformerBlock(d_model=4, n_heads=2, d_ff=16)
x = torch.randn(1, 3, 4)                     # バッチ1 × 3トークン × 4次元
print(block(x).shape)                        # torch.Size([1, 3, 4]) 入力と同じ形！
```

出力の形が入力とまったく同じ（`[1, 3, 4]`）なのが重要です。形が変わらないからこそ、**このブロックを何個でも積み重ねられる**のです。

```python
# ブロックを N 個積めば、それがもう Transformer の本体
n_layers = 6
transformer = nn.Sequential(*[
    TransformerBlock(d_model=4, n_heads=2, d_ff=16) for _ in range(n_layers)
])
print(transformer(x).shape)                  # torch.Size([1, 3, 4])
```

:::tip[これがほぼ GPT の中身]

ここで組んだブロックを数十段積み、入口に①②（埋め込み・位置）、出口に「次の単語を予測する層」を付ければ、構造としては GPT 系の言語モデルとほぼ同じです。残るのは「大量のテキストでどう学習するか」だけ——その学習の話は、今後の章で扱います。あなたはもう、Transformer の**部品も組み立て方も**理解できています。

:::

---

## 7. この章のまとめ

この章では、前章の Attention を取り囲む **5つの部品**を学び、最後に全部を組み立てて Transformer ブロックを完成させました。

### 5つの部品

- **① トークン埋め込み**：テキストを「意味を持つ密なベクトル」に変える入り口。正体は**埋め込み行列から ID で行を引くルックアップ**で、表の中身は学習で賢くなる。
- **② 位置エンコーディング**：並列計算で失われた「順番」の情報を、**波長の違う sin・cos の波**として埋め込みに足し込む。これで「猫が犬を追う」と「犬が猫を追う」を区別できる。
- **③ フィードフォワード層（FFN）**：Attention が「混ぜる」役なら、FFN は各単語を個別に「深く加工する」役。**広げて → ReLU で非線形 → 戻す**の2層ネットで、表現力を生む。
- **④ スキップ接続（残差接続）**：$y = x + F(x)$ という近道で、**勾配の高速道路**を作り「差分だけ学べばいい」状態にする。深く積んでも学習できる土台。
- **⑤ レイヤー正規化**：ベクトルを**平均0・ばらつき1**に整え、値の暴れを抑えて学習を安定させる。④とペアで **Add & Norm** として使う。

### 組み立て

- 1つの Transformer ブロックは $x' = \text{LayerNorm}(x + \text{MultiHead}(x))$、$y = \text{LayerNorm}(x' + \text{FFN}(x'))$。
- 入力と出力の形が同じなので、ブロックを **N 段積み重ねられる**。これが Transformer の本体。
- 入口に①②、出口に予測層を付け、大量のテキストで学習させれば——構造は GPT 系とほぼ同じ。

---

前章と本章で、**Transformer のアーキテクチャ**（部品と組み立て）は一通りつかめました。残る大きなピースは2つ——**テキストをどうトークン（数）に分けるか**（トークナイザー）と、**大量のデータでどう学習させるか**（学習ループ）です。これらは今後の章で扱っていきます。お疲れさまでした！🎉
