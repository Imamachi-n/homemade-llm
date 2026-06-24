---
sidebar_position: 2
title: "Chapter 2: Transformer"
---

# Chapter 2: Transformer

[前章](./chapter1.md)では、ベクトル・内積・行列積・softmax といった「LLM を支える数学」を積み上げました。この章では、いよいよその数学が組み上がってできる主役、**Transformer（トランスフォーマー）** に迫ります。

この章は **前半（概念）→ 後半（実装）** の2部構成です。まず「そもそも Transformer って何なの？」「なぜ必要だったのか」をつかみ、心臓部の **Attention（注意機構）** の仕組みを前章の数学とつなげて理解します。そのうえで後半では、その Attention を **実際に数値とコードで計算**し、手を動かしながら腑に落とします。

:::tip[この章の読み方]

数式や用語が出てきても身構えなくて大丈夫です。新しい記号や式は **使う直前に意味を渡します**。特に Attention は「前章で学んだ内積・softmax をくり返しているだけ」だと分かると、一気に見通しが良くなります。後半の実装も、前半で理解した3ステップをそのままコードにするだけです。

:::

---

## 1. ざっくり一言で言うと

Transformer をひとことで言うと、こうです。

> **文章の中で「どの単語が、どの単語に注目すべきか」を一気に計算する仕組み（ニューラルネットのアーキテクチャ）**

2017 年の論文 **"Attention Is All You Need"** で提案され、いまの GPT・Claude をはじめとする大規模言語モデルが、ほぼ例外なくこの Transformer をベースにしています。LLM を学ぶうえで「ここだけは外せない」という中心的な構造です。

そして大事なのは、Transformer は決して魔法の箱ではない、ということです。中身は前章で学んだ数学の組み合わせでできています。

| Transformer の部品 | 中身（前章で学んだ数学） |
| --- | --- |
| 単語を数で表す | ベクトル（単語埋め込み） |
| 単語どうしの関連度を測る | **内積・コサイン類似度** |
| 全単語ペアの関連度を一括計算 | **行列積 $QK^T$** |
| 関連度を「注目度（確率）」に変える | **softmax** |
| 表現を別空間へ変換する | **線形変換（重み行列）** |
| 「何番目の単語か」を表す | **三角関数（位置エンコーディング）** |

つまり Transformer の理解とは、**「前章の数学が、どういう順番で、何のために組み合わさるのか」を理解すること**にほかなりません。

:::note[アーキテクチャって何？]

「アーキテクチャ（architecture）」は、ここでは **ニューラルネットワークの設計図・構造の型**のことです。「どんな計算ブロックを、どんな順番でつなぐか」という骨組みを指します。Transformer は、その骨組みの一種の名前だと思ってください。

:::

---

## 2. なんで生まれたの？（背景がアツい🔥）

新しい技術は、たいてい「前のやり方の不満」から生まれます。Transformer も同じです。ここを押さえると、Attention という仕組みが「なぜその形なのか」が腑に落ちます。

### 2.1 むかしのやり方：1単語ずつ順番に読む（RNN）

Transformer 以前、文章を扱う主役は **RNN（再帰型ニューラルネット）** やその改良版の **LSTM** でした。これらは、人間が文章を読むように **単語を1つずつ、順番に**処理していきます。

前の単語を読んだ「記憶」を次の単語へ受け渡しながら、左から右へ進んでいくイメージです。一見、自然な方法に見えます。しかし、ここには大きな問題が2つありました。

#### 問題1：遅い（並列化できない）

順番に処理するということは、**2単語目は1単語目が終わるまで計算できない**ということです。100 単語あれば 100 回の処理を直列に待たなければなりません。

現代の GPU は「大量の計算を同時に（並列に）こなす」のが得意なのに、RNN はその強みを活かせません。順番待ちが発生してしまうからです。

#### 問題2：遠い単語を忘れる（長距離依存が苦手）

記憶を1単語ずつバケツリレーで受け渡していくと、**遠く離れた単語の情報は薄れていきます**。

> 「**昨日**、駅前の本屋で友達とばったり会って立ち話をして…（中略）…**それ**はとても楽しかった」

「それ」が「昨日の出来事」を指すと分かるには、ずっと前の情報を保持し続ける必要があります。ところが RNN ではリレーの途中で情報がぼやけてしまい、長い文ほど関係をとらえそこねます。

### 2.2 Transformer の発想：順番に読むのをやめる

そこで Transformer は、発想を逆転させました。

> **1単語ずつ順番に読むのをやめて、文章全体を一度に見渡し、すべての単語ペアの関連を一気に計算する。**

この転換が、上の2つの問題を同時に解決します。

- **並列化できる**：単語を順番待ちさせないので、全単語の計算を GPU でいっせいに走らせられる → **速い**。
- **遠い単語と直接つながる**：バケツリレーをやめ、どの単語ともダイレクトに関連度を計算する → **どんなに離れていても1ステップでつながる**。

下の図は、RNN（順番に渡す）と Transformer（全単語を直接つなぐ）の情報の流れ方の違いです。

<div style={{display: 'flex', flexWrap: 'wrap', gap: '1.5rem', justifyContent: 'center', alignItems: 'flex-start', margin: '1.25rem 0'}}>
  <figure style={{margin: 0, textAlign: 'center'}}>
    <svg viewBox="0 0 260 150" width="250" role="img" aria-label="RNN は単語を左から右へ順番に処理する">
      <circle cx="35" cy="60" r="18" fill="currentColor" fillOpacity="0.1" stroke="currentColor" strokeOpacity="0.55" strokeWidth="1.4" />
      <circle cx="105" cy="60" r="18" fill="currentColor" fillOpacity="0.1" stroke="currentColor" strokeOpacity="0.55" strokeWidth="1.4" />
      <circle cx="175" cy="60" r="18" fill="currentColor" fillOpacity="0.1" stroke="currentColor" strokeOpacity="0.55" strokeWidth="1.4" />
      <circle cx="245" cy="60" r="18" fill="currentColor" fillOpacity="0.1" stroke="currentColor" strokeOpacity="0.55" strokeWidth="1.4" />
      <line x1="53" y1="60" x2="87" y2="60" stroke="#EF4444" strokeWidth="2" markerEnd="url(#arrowR)" />
      <line x1="123" y1="60" x2="157" y2="60" stroke="#EF4444" strokeWidth="2" markerEnd="url(#arrowR)" />
      <line x1="193" y1="60" x2="227" y2="60" stroke="#EF4444" strokeWidth="2" markerEnd="url(#arrowR)" />
      <defs>
        <marker id="arrowR" markerWidth="7" markerHeight="7" refX="5.5" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill="#EF4444" />
        </marker>
      </defs>
      <text x="35" y="64" fontSize="11" fill="currentColor" textAnchor="middle">昨日</text>
      <text x="105" y="64" fontSize="11" fill="currentColor" textAnchor="middle">本屋</text>
      <text x="175" y="64" fontSize="11" fill="currentColor" textAnchor="middle">…</text>
      <text x="245" y="64" fontSize="11" fill="currentColor" textAnchor="middle">それ</text>
      <text x="130" y="120" fontSize="10" fill="currentColor" textAnchor="middle" fillOpacity="0.8">順番にリレー：遠い「昨日」は薄れる</text>
    </svg>
    <figcaption style={{fontSize: '0.82rem', marginTop: '0.4rem', opacity: 0.85}}>① RNN：左から右へ1単語ずつ。並列化できず、遠い単語の情報は薄れる</figcaption>
  </figure>
  <figure style={{margin: 0, textAlign: 'center'}}>
    <svg viewBox="0 0 260 150" width="250" role="img" aria-label="Transformer は全単語どうしを直接つなぐ">
      <line x1="35" y1="42" x2="105" y2="42" stroke="#3B82F6" strokeOpacity="0.55" strokeWidth="1.4" />
      <line x1="35" y1="42" x2="175" y2="42" stroke="#3B82F6" strokeOpacity="0.55" strokeWidth="1.4" />
      <line x1="35" y1="42" x2="245" y2="42" stroke="#3B82F6" strokeWidth="2.4" />
      <line x1="105" y1="42" x2="175" y2="42" stroke="#3B82F6" strokeOpacity="0.55" strokeWidth="1.4" />
      <line x1="105" y1="42" x2="245" y2="42" stroke="#3B82F6" strokeOpacity="0.55" strokeWidth="1.4" />
      <line x1="175" y1="42" x2="245" y2="42" stroke="#3B82F6" strokeOpacity="0.55" strokeWidth="1.4" />
      <circle cx="35" cy="42" r="16" fill="currentColor" fillOpacity="0.1" stroke="currentColor" strokeOpacity="0.55" strokeWidth="1.4" />
      <circle cx="105" cy="42" r="16" fill="currentColor" fillOpacity="0.1" stroke="currentColor" strokeOpacity="0.55" strokeWidth="1.4" />
      <circle cx="175" cy="42" r="16" fill="currentColor" fillOpacity="0.1" stroke="currentColor" strokeOpacity="0.55" strokeWidth="1.4" />
      <circle cx="245" cy="42" r="16" fill="currentColor" fillOpacity="0.1" stroke="currentColor" strokeOpacity="0.55" strokeWidth="1.4" />
      <text x="35" y="46" fontSize="10" fill="currentColor" textAnchor="middle">昨日</text>
      <text x="105" y="46" fontSize="10" fill="currentColor" textAnchor="middle">本屋</text>
      <text x="175" y="46" fontSize="10" fill="currentColor" textAnchor="middle">…</text>
      <text x="245" y="46" fontSize="10" fill="currentColor" textAnchor="middle">それ</text>
      <text x="130" y="120" fontSize="10" fill="currentColor" textAnchor="middle" fillOpacity="0.8">全ペアを直接計算：「それ」と「昨日」も1ステップ</text>
    </svg>
    <figcaption style={{fontSize: '0.82rem', marginTop: '0.4rem', opacity: 0.85}}>② Transformer：全単語どうしを直接つなぐ。並列化でき、遠い単語ともすぐつながる</figcaption>
  </figure>
</div>

:::note[論文タイトル "Attention Is All You Need" の意味]

「必要なのは Attention だけだ」という、やや挑発的なタイトルです。それまで主流だった RNN の「順番に読む」仕組みを **完全に取り払い**、後述する Attention だけで文章を処理してみせた——という宣言になっています。実際、それがうまくいったことが、その後の LLM 時代の幕開けになりました。

:::

:::warning[「全ペアを一気に」のコストには注意]

全単語ペアの関連を計算するということは、単語数が $n$ なら $n \times n$ 個の組み合わせを計算する、ということです。文章が長くなると計算量が急増する（おおよそ $n^2$ に比例する）という弱点もあり、これは後の章で扱う「効率化」の大きなテーマになります。まずは「全ペアを一気に見るからこそ速くて賢い」という利点を押さえておけば十分です。

:::

---

## 3. 一番の主役：Attention（注意機構）

ここからが Transformer の心臓部、**Attention（アテンション／注意機構）** です。名前のとおり「**どの単語に注意を向けるか**」を計算する仕組みで、Transformer の賢さはほぼここから来ています。

### 3.1 直感：その単語は、どの単語を見るべきか

次の文を読んでください。

> 「その**動物**は疲れていたので、**それ**は道を渡らなかった」

「それ」は何を指しているでしょうか。もちろん「動物」ですね。人間は無意識に分かりますが、機械にこの判断をさせるのが Attention です。

Attention のアイデアはこうです。

> **各単語が、文中の全単語に対して「私はあなたにどれくらい注目すべき？」という関連度を計算し、関連度の高い単語の情報をより多く受け取る。**

「それ」という単語を処理するとき、「動物」への関連度を高く、「疲れて」や「道」への関連度をそれなりに、関係ない単語への関連度を低く——というふうに **注目度の配分**を決め、その配分にしたがって情報を集めてくる、というわけです。

### 3.2 Query・Key・Value：検索にたとえる

Attention では、各単語から **3つのベクトル**を作ります。これがいちばんの肝です。図書館での「検索」にたとえると分かりやすいです。

| 名前 | 役割 | 検索のたとえ |
| --- | --- | --- |
| **Query（クエリ）** $q$ | 「私はこういう情報を探している」という問い合わせ | 検索キーワード |
| **Key（キー）** $k$ | 「私はこういう特徴を持つ単語だよ」という見出し | 本の背表紙ラベル |
| **Value（バリュー）** $v$ | 実際に持っている中身の情報 | 本の中身 |

#### この3つはどこから出てくるのか

「Query・Key・Value という3つのベクトルを作る」と言いましたが、ではこれらはどこから湧いてくるのでしょうか。答えは、**各単語のベクトルに、それぞれ専用の変換をかけて作る**です。

各単語はまず、前章で見たような数字の並び——**埋め込みベクトル（embedding）** $x$——で表されています。この同じ $x$ に、3種類の重み行列 $W_Q, W_K, W_V$ をかけることで、3つのベクトルを取り出します。

:::note[埋め込みベクトル（embedding）とは]

コンピュータは「猫」「犬」といった文字そのものを計算できません。そこで、各単語を **意味を表す数字の並び（ベクトル）** に変換します。これが **埋め込みベクトル（embedding）** です。

$$
\text{「猫」} \rightarrow (0.21,\ -0.84,\ 0.05,\ \dots,\ 0.63)
$$

ベクトルの長さ（数字の個数）は、たとえば 512 や 768 のように決められた **次元**で、1本が単語1個ぶんの「意味の座標」にあたります。

うれしいのは、**意味が近い単語ほどベクトルも近い向きになる**ように学習される点です。「猫」と「犬」はどちらも動物なので近く、「猫」と「車」は遠くなります。だからこそ、[Chapter 1](./chapter1.md) で学んだ「**内積が大きい＝似た方向＝関連が強い**」を使って、単語どうしの関連度を計算できるのです。Attention が動くのは、単語がこの埋め込みベクトルになっているおかげ、というわけです。

「単語」というバラバラな記号を、数字が連続する空間の中に **"埋め込む（置く）"** ——意味の似た単語が近所に集まる巨大な「意味の地図」に配置するイメージから、embedding と呼ばれます。

:::

$$
q = x W_Q, \quad k = x W_K, \quad v = x W_V
$$

| 記号 | 中身 |
| --- | --- |
| $x$ | その単語の埋め込みベクトル（Attention への入力） |
| $W_Q, W_K, W_V$ | Query・Key・Value 用の重み行列（変換の道具） |

つまり、**同じ1つの単語から、3つの違う「見方」で別々のベクトルを取り出している**わけです。これは [Chapter 1](./chapter1.md) で学んだ「**行列をかけることはベクトルを変換すること**」がそのまま効いている箇所です。

そして大事なのは、$W_Q, W_K, W_V$ が **学習で決まる**という点です。最初はでたらめな数字ですが、「どんな問い合わせ（Query）を投げ、どんな見出し（Key）を掲げれば、関連語をうまく見つけられるか」を、学習を通してモデル自身が獲得していきます。Query をどう作るか自体が、モデルの賢さの一部になっているのです。

#### Attention 1回の流れ

ここまでが準備です。実際の流れはこうです。ある単語の **Query** を、全単語の **Key** と照らし合わせて「相性（関連度）」を測り、相性の良い単語の **Value** をたくさん受け取る——これが Attention の1回の計算です。

そして「Query と Key の相性をどう測るか」こそ、前章で学んだ **内積**です。

:::tip[前章とのつながり：相性 ＝ 内積]

[Chapter 1](./chapter1.md) で「**内積が大きいほど、2つのベクトルは似た方向を向いている（＝関連が強い）**」と学びました。Attention はまさにこれを使います。Query と Key の内積を取れば、それがそのまま「その単語ペアの関連度スコア」になるのです。Attention は、内積による関連度の計算を全単語ペアにわたってくり返しているだけ、とも言えます。

:::

### 3.3 3ステップで見る Attention

Query・Key・Value がそろえば、Attention は3ステップで計算できます。

1. **関連度を測る（内積）**：注目元の単語の Query $q$ と、各単語の Key $k$ の内積を取る。値が大きいほど関連が強い。
2. **注目度に変える（softmax）**：内積で出たスコアを softmax にかけ、**すべて正・合計1**の「注目度の配分（重み）」にする。
3. **情報を集める（加重和）**：各単語の Value $v$ を、注目度を重みにして足し合わせる。注目した単語の情報が濃く混ざる。

<figure style={{margin: '1rem auto', textAlign: 'center', maxWidth: '460px'}}>
  <svg viewBox="0 0 420 220" width="420" role="img" aria-label="Attention の3ステップ：Query と各 Key の内積 → softmax で注目度 → Value の加重和">
    <text x="40" y="20" fontSize="11" fill="#3B82F6" textAnchor="middle" fontStyle="italic">Query</text>
    <rect x="20" y="28" width="40" height="20" rx="3" fill="#3B82F6" fillOpacity="0.18" stroke="#3B82F6" strokeOpacity="0.6" strokeWidth="1.2" />
    <text x="40" y="42" fontSize="10" fill="currentColor" textAnchor="middle">それ</text>
    <text x="150" y="20" fontSize="11" fill="currentColor" textAnchor="middle" fillOpacity="0.7">各単語の Key</text>
    <rect x="120" y="28" width="36" height="18" rx="3" fill="currentColor" fillOpacity="0.08" stroke="currentColor" strokeOpacity="0.45" strokeWidth="1" />
    <rect x="120" y="52" width="36" height="18" rx="3" fill="currentColor" fillOpacity="0.08" stroke="currentColor" strokeOpacity="0.45" strokeWidth="1" />
    <rect x="120" y="76" width="36" height="18" rx="3" fill="currentColor" fillOpacity="0.08" stroke="currentColor" strokeOpacity="0.45" strokeWidth="1" />
    <text x="138" y="41" fontSize="9" fill="currentColor" textAnchor="middle">動物</text>
    <text x="138" y="65" fontSize="9" fill="currentColor" textAnchor="middle">疲れ</text>
    <text x="138" y="89" fontSize="9" fill="currentColor" textAnchor="middle">道</text>
    <line x1="60" y1="38" x2="118" y2="37" stroke="currentColor" strokeOpacity="0.4" strokeWidth="1" />
    <line x1="60" y1="38" x2="118" y2="61" stroke="currentColor" strokeOpacity="0.4" strokeWidth="1" />
    <line x1="60" y1="38" x2="118" y2="85" stroke="currentColor" strokeOpacity="0.4" strokeWidth="1" />
    <text x="200" y="20" fontSize="10" fill="currentColor" textAnchor="middle" fillOpacity="0.7">内積</text>
    <text x="200" y="41" fontSize="9" fill="currentColor" textAnchor="middle">9.1</text>
    <text x="200" y="65" fontSize="9" fill="currentColor" textAnchor="middle">2.0</text>
    <text x="200" y="89" fontSize="9" fill="currentColor" textAnchor="middle">0.5</text>
    <text x="200" y="112" fontSize="14" fill="currentColor" textAnchor="middle">↓</text>
    <text x="200" y="128" fontSize="9" fill="#10B981" textAnchor="middle">softmax</text>
    <text x="270" y="20" fontSize="10" fill="#10B981" textAnchor="middle">注目度</text>
    <rect x="240" y="28" width="60" height="18" rx="3" fill="#10B981" fillOpacity="0.28" stroke="#10B981" strokeOpacity="0.6" strokeWidth="1" />
    <rect x="240" y="52" width="22" height="18" rx="3" fill="#10B981" fillOpacity="0.14" stroke="#10B981" strokeOpacity="0.4" strokeWidth="1" />
    <rect x="240" y="76" width="14" height="18" rx="3" fill="#10B981" fillOpacity="0.1" stroke="#10B981" strokeOpacity="0.4" strokeWidth="1" />
    <text x="305" y="41" fontSize="9" fill="currentColor">0.91</text>
    <text x="267" y="65" fontSize="9" fill="currentColor">0.07</text>
    <text x="259" y="89" fontSize="9" fill="currentColor">0.02</text>
    <text x="360" y="20" fontSize="11" fill="#EF4444" textAnchor="middle">出力</text>
    <text x="360" y="55" fontSize="13" fill="currentColor" textAnchor="middle">Σ</text>
    <text x="360" y="72" fontSize="8" fill="currentColor" textAnchor="middle" fillOpacity="0.7">Value の</text>
    <text x="360" y="83" fontSize="8" fill="currentColor" textAnchor="middle" fillOpacity="0.7">加重和</text>
    <rect x="335" y="92" width="50" height="20" rx="3" fill="#EF4444" fillOpacity="0.16" stroke="#EF4444" strokeOpacity="0.6" strokeWidth="1.2" />
    <text x="360" y="106" fontSize="9" fill="currentColor" textAnchor="middle">≈ 動物</text>
    <line x1="302" y1="37" x2="334" y2="98" stroke="#EF4444" strokeOpacity="0.5" strokeWidth="1.6" />
    <text x="210" y="175" fontSize="10" fill="currentColor" textAnchor="middle" fillOpacity="0.85">「それ」は「動物」へ最も強く注目し、その情報を濃く受け取る</text>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>Attention の3ステップ：内積で関連度 → softmax で注目度（合計1）→ Value の加重和</figcaption>
</figure>

この図では、「それ」の Query が「動物」の Key と最も相性が良く（内積 9.1）、softmax を通すと注目度が 0.91 に集中します。結果として出力は「動物」の Value がほぼそのまま——つまり「それ ＝ 動物」という対応を、計算だけで取り出せたことになります。

:::tip[前章とのつながり：注目度 ＝ softmax]

ステップ2で使う softmax も [Chapter 1](./chapter1.md) で学んだ関数です。内積で出たスコアはバラバラな大きさの数ですが、softmax を通すと **すべて正・合計1** の確率分布になります。これがそのまま「どの単語にどれだけ注目するか」の配分になる、というのが美しいところです。

:::

### 3.4 まとめて1つの式に：Scaled Dot-Product Attention

ここまでの3ステップを、全単語ぶんまとめて1つの式で書いたものが、Transformer の中核をなす式です。単語ごとの $q, k, v$ を縦に積んで行列 $Q, K, V$ にまとめると、

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d_k}}\right) V
$$

と書けます。記号が多いですが、やっていることは3ステップそのままです。

| 式の部分 | 対応するステップ | 中身 |
| --- | --- | --- |
| $Q K^{\top}$ | ① 関連度を測る | 全 Query と全 Key の **内積を一括計算**（前章の行列積！） |
| $\dfrac{\;\cdot\;}{\sqrt{d_k}}$ | （調整） | 値が大きくなりすぎないよう、Key の次元 $d_k$ の平方根で割る |
| $\text{softmax}(\cdots)$ | ② 注目度に変える | スコアを合計1の注目度へ |
| $(\cdots)\,V$ | ③ 情報を集める | 注目度を重みに Value を加重和 |

注目してほしいのは $Q K^{\top}$ の部分です。これは前章の「**行列積は内積の集まり**」がそのまま効いている箇所で、**全単語ペアの内積（関連度）を1回の行列積でまとめて計算**しています。RNN のような順番待ちなしに、全ペアを一気に求められる——これが「速い」の正体です。

:::note[なぜ $\sqrt{d_k}$ で割るのか（スケーリング）]

Query と Key の次元 $d_k$ が大きいほど、内積はたくさんの項を足すので値が大きくなりがちです。値が大きすぎると softmax の出力が一箇所に極端に偏り、学習がうまく進まなくなります。そこで $\sqrt{d_k}$ で割って大きさをならし、安定させています。この「割る」操作があるので **Scaled（スケール済み）Dot-Product Attention** と呼ばれます。詳しい理由は後の章でも触れます。

:::

### 3.5 Self-Attention と Multi-Head Attention

最後に、よく出てくる2つの言葉だけ押さえておきましょう。詳しくは後の章で扱いますが、名前の意味が分かるだけで理解がぐっと楽になります。

- **Self-Attention（自己注意）** … Query・Key・Value を **すべて同じ文章（同じ単語列）から作る** Attention のこと。文中の単語どうしが互いに注目し合い、「それ ＝ 動物」のような **文の内部の関係**をとらえます。Transformer の主役はこの Self-Attention です。
- **Multi-Head Attention（多頭注意）** … Attention を **複数セット並列に**行う仕組み。1つの「頭（head）」だけだと1種類の見方しかできませんが、頭を複数用意することで「文法的なつながりを見る頭」「意味的な近さを見る頭」のように **別々の観点から同時に注目**でき、最後にそれらを統合します。

<figure style={{margin: '1rem auto', textAlign: 'center', maxWidth: '380px'}}>
  <svg viewBox="0 0 340 130" width="330" role="img" aria-label="Multi-Head Attention：複数の頭が別々の観点で注目し、結果を統合する">
    <rect x="20" y="52" width="50" height="24" rx="4" fill="currentColor" fillOpacity="0.1" stroke="currentColor" strokeOpacity="0.5" strokeWidth="1.2" />
    <text x="45" y="68" fontSize="10" fill="currentColor" textAnchor="middle">入力</text>
    <line x1="70" y1="64" x2="110" y2="30" stroke="currentColor" strokeOpacity="0.4" strokeWidth="1.2" />
    <line x1="70" y1="64" x2="110" y2="64" stroke="currentColor" strokeOpacity="0.4" strokeWidth="1.2" />
    <line x1="70" y1="64" x2="110" y2="98" stroke="currentColor" strokeOpacity="0.4" strokeWidth="1.2" />
    <rect x="110" y="18" width="80" height="24" rx="4" fill="#3B82F6" fillOpacity="0.16" stroke="#3B82F6" strokeOpacity="0.6" strokeWidth="1.2" />
    <text x="150" y="34" fontSize="9" fill="currentColor" textAnchor="middle">頭1：文法を見る</text>
    <rect x="110" y="52" width="80" height="24" rx="4" fill="#10B981" fillOpacity="0.16" stroke="#10B981" strokeOpacity="0.6" strokeWidth="1.2" />
    <text x="150" y="68" fontSize="9" fill="currentColor" textAnchor="middle">頭2：意味を見る</text>
    <rect x="110" y="86" width="80" height="24" rx="4" fill="#EF4444" fillOpacity="0.16" stroke="#EF4444" strokeOpacity="0.6" strokeWidth="1.2" />
    <text x="150" y="102" fontSize="9" fill="currentColor" textAnchor="middle">頭3：…</text>
    <line x1="190" y1="30" x2="230" y2="64" stroke="currentColor" strokeOpacity="0.4" strokeWidth="1.2" />
    <line x1="190" y1="64" x2="230" y2="64" stroke="currentColor" strokeOpacity="0.4" strokeWidth="1.2" />
    <line x1="190" y1="98" x2="230" y2="64" stroke="currentColor" strokeOpacity="0.4" strokeWidth="1.2" />
    <rect x="230" y="52" width="50" height="24" rx="4" fill="currentColor" fillOpacity="0.1" stroke="currentColor" strokeOpacity="0.5" strokeWidth="1.2" />
    <text x="255" y="68" fontSize="10" fill="currentColor" textAnchor="middle">統合</text>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>Multi-Head Attention：複数の頭が別々の観点で注目し、最後に統合する</figcaption>
</figure>

:::tip[前半（概念）のまとめ]

- Transformer は「全単語ペアの注目度を一気に計算する」アーキテクチャで、いまの LLM の土台。
- RNN の「遅い・遠い単語を忘れる」を、「順番に読むのをやめる」ことで解決した。
- 心臓部の **Attention** は、**内積で関連度 → softmax で注目度 → Value の加重和** という、前章の数学そのもの。
- まとめると $\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V$。$QK^{\top}$ は全ペア内積の一括計算。

:::

---

ここまでで、Attention が「**内積 → softmax → 加重和**」の3ステップでできていることが分かりました。**ここからは後半（実装）パート**です。その3ステップを、実際の数値で手を動かし、最後は Python（NumPy）で実装して確かめます。ゴールは1つ。**アテンション機構の出力 $o$ を自分の手で計算できるようになること**です。

:::note[ここからは実践パート]

仕組みは前半でつかんだので、後半は「**結局どう計算するのか／コードでどう書くのか**」に集中します。手元で動かしたいときは [Google Colab](https://colab.research.google.com/?hl=ja) が手軽です（`pip install numpy` だけで動きます）。

:::

## 4. 注目度から出力 $o$ までを計算する

### 4.1 注目度 $a_i$ を「計算する形」に書き下す

前半の 3 節では、クエリ $q$ とキー $k_i$ の内積を softmax に通したものが注目度になる、と見ました。計算に移る前に、これを1つの式として書いておきます。キーが全部で $n$ 個あるとして、スコアは内積 $a'_i = q\,k_i^{\top}$、それを softmax に通した各成分を $a_i$ と書くと、

$$
a_i = \frac{\exp(q k_i^{\top})}{\sum_{j} \exp(q k_j^{\top})}
$$

これが **クエリ $q$ の、キー $k_i$ に対する注目度**です。各 $a_i$ は **スカラ（ただの1つの数）** で、$0 \le a_i \le 1$、かつ $a_1 + a_2 + \dots + a_n = 1$ を満たします。「どの位置をどれくらい重視するか」を表す割合だと思ってください。

:::note[なぜ $k_i$ を転置（$\top$）するの？]

スコアの式に出てくる $q\,k_i^{\top}$ の「$\top$（転置）」は、**横ベクトル同士を掛けて「1つの数（関連度スコア）」を取り出す**ために必要です。

このテキストでは $q$ も $k_i$ も**横に数を並べた行ベクトル**として扱います。次元を $d=3$ とすると、どちらも $1 \times 3$ の行列です。

$$
q = [\,q_1\ q_2\ q_3\,], \qquad k_i = [\,k_{i1}\ k_{i2}\ k_{i3}\,]
$$

行列のかけ算には「**左の列数と右の行数が一致しないと掛けられない**」というルールがあります。$q$（$1 \times 3$、列数 3）と $k_i$（$1 \times 3$、行数 1）はそのままでは $3 \ne 1$ で掛けられません。そこで $k_i$ を転置して $3 \times 1$ の**縦ベクトル**にします。

$$
k_i^{\top} = \begin{bmatrix} k_{i1} \\ k_{i2} \\ k_{i3} \end{bmatrix}
$$

すると $q\,k_i^{\top}$ は $(1 \times 3) \times (3 \times 1) = 1 \times 1$、つまり**スカラ**になり、これがちょうど内積（＝関連度）になります。

$$
q\,k_i^{\top} = q_1 k_{i1} + q_2 k_{i2} + q_3 k_{i3}
$$

上で「各 $a_i$ はスカラ」と書けるのは、この $\top$ のおかげです。前半まとめ（3.4 節）の $QK^{\top}$ も同じ理屈で、こちらは「全クエリ × 全キー」の内積を行列で一括計算した版です。

:::

### 4.2 バリューベクトル $v$ を用意する

出力を求めるには、前半 3.2 で出てきた **バリューベクトル** $v$（実際に受け取る中身の情報）を、いよいよ式に登場させます。本来は Key とは別物ですが、計算を追いやすくするため、この章の範囲では、

$$
\text{バリューベクトル } v = \text{キーベクトル } k
$$

とします。つまり $v_i = k_i$。いったんは「キーと同じものをもう一度使うだけ」と思っておけば十分です。

:::note[なぜ Value と Key を別々に用意するの？]

理由は、**「探すための見出し」と「実際に渡す中身」では役割がまったく違う**からです。Attention の中で各単語の情報は、2つの別の場面で使われます。

- **Key** $k_i$：クエリと内積を取って**関連度を測る**ための見出し（「これは関係ある？」）
- **Value** $v_i$：関連度が決まったあと、**実際に受け取る中身**（「で、結局どの情報をもらう？」）

前半 3.2 の図書館のたとえに乗せると、はっきりします。

- **Query** = 探したいキーワード（「猫の飼い方」）
- **Key** = 本の背表紙・**タイトル**（探すときにスキャンする部分）
- **Value** = 本の**中身そのもの**（実際に読む情報）

「タイトル」と「中身」が別物であるように、$k_i$ と $v_i$ も役割が違うので、一般には $v_i \ne k_i$ です。実際の Transformer では、同じ入力 $x_i$ から**別々の重み行列**で作ります。

$$
k_i = x_i W_K, \qquad v_i = x_i W_V
$$

こうして分けておくと、モデルは「**ある特徴でマッチさせて、でも渡すのは別の情報**」という使い分けを学習でき、表現力が上がります。

ただし計算を最初に追う段階では、両者を同じにしておくと式が見通しやすくなります。そこでこの章では $v_i = k_i$ で進め、両者を分ける一般形は後の章で扱います。

:::

### 4.3 出力 $o$ は「Value の加重和」

アテンション機構の出力 $o$ は、各バリューベクトル $v_i$ を注目度 $a_i$ で重みづけして、すべて足し合わせたものです。

$$
o = a_1 v_1 + a_2 v_2 + \dots + a_n v_n
$$

$o$ は $v$ と同じ次元数のベクトルになります（スカラ × ベクトルを足しているだけなので、次元は変わりません）。

やっていることを一言で言うと、これだけです。

> **$i$ 番目のバリューベクトルを、$i$ 番目の注目度倍して、全部足す。**

- 注目度が大きいもの ＝ クエリと強く関連しているもの。
- 注目度が大きい Value は $o$ に強く反映され、注目度がゼロに近い Value はほとんど無視される。

つまり $o$ とは、

> **クエリ $q$ が入力されたときの、$q$ に関連のある内容を強く反映した「新しいベクトル表現」**

なのです。前半 3.3 の「それ → 動物」の例で言えば、出力 $o$ は「動物」の情報を濃く受け取ったベクトルになります。

<figure style={{margin: '1.25rem auto', textAlign: 'center', maxWidth: '460px'}}>
  <svg viewBox="0 0 420 180" width="420" role="img" aria-label="出力 o は各バリューベクトルを注目度で重みづけして足し合わせたもの">
    <text x="60" y="22" fontSize="11" fill="currentColor" textAnchor="middle" fillOpacity="0.75">注目度 × Value</text>
    <rect x="20" y="34" width="80" height="22" rx="3" fill="#3B82F6" fillOpacity="0.30" stroke="#3B82F6" strokeOpacity="0.65" strokeWidth="1.2" />
    <text x="60" y="49" fontSize="10" fill="currentColor" textAnchor="middle">0.665 · v₁</text>
    <rect x="20" y="78" width="26" height="22" rx="3" fill="#10B981" fillOpacity="0.22" stroke="#10B981" strokeOpacity="0.55" strokeWidth="1.1" />
    <text x="60" y="93" fontSize="10" fill="currentColor" textAnchor="middle">0.090 · v₂</text>
    <rect x="20" y="122" width="42" height="22" rx="3" fill="#EF4444" fillOpacity="0.22" stroke="#EF4444" strokeOpacity="0.55" strokeWidth="1.1" />
    <text x="60" y="137" fontSize="10" fill="currentColor" textAnchor="middle">0.245 · v₃</text>
    <text x="150" y="92" fontSize="22" fill="currentColor" textAnchor="middle" fillOpacity="0.7">+</text>
    <line x1="175" y1="89" x2="245" y2="89" stroke="currentColor" strokeOpacity="0.5" strokeWidth="1.6" markerEnd="url(#arrowO)" />
    <defs>
      <marker id="arrowO" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
        <path d="M0,0 L6,3 L0,6 Z" fill="currentColor" fillOpacity="0.6" />
      </marker>
    </defs>
    <rect x="250" y="74" width="90" height="32" rx="4" fill="currentColor" fillOpacity="0.1" stroke="currentColor" strokeOpacity="0.55" strokeWidth="1.3" />
    <text x="295" y="95" fontSize="13" fill="currentColor" textAnchor="middle">出力 o</text>
    <text x="295" y="128" fontSize="9" fill="currentColor" textAnchor="middle" fillOpacity="0.8">≈ v₁ に近いベクトル</text>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>注目度が高い v₁ の成分が濃く混ざり、出力 o は v₁ に引き寄せられる（数値は次節 4.4 の計算例）</figcaption>
</figure>

:::note[結局 $o$ ってなに？ なぜこれがゴールなの？]

$o$ は **output（出力）の $o$**。アテンション機構が「内積 → softmax → 加重和」の3ステップの最後に吐き出す**答えのベクトル**で、この計算全体のゴールです。中身を3つの視点で整理すると：

- **作り方**：各 Value $v_i$ を注目度 $a_i$ で重みづけして全部足した「**加重和**」。注目度が大きい Value ほど濃く混ざり、ゼロに近い Value はほぼ無視される（要は関連度で重みづけした平均）。
- **意味**：クエリ $q$ に**関連する情報を濃く取り込んだ「新しいベクトル表現」**。「それ → 動物」の例なら、$o$ は「動物」の情報を強く受け取ったベクトルになる。
- **大きさ**：$a_i$ はスカラ、$v_i$ はベクトルなので、$o$ は **$v$ と同じ次元のベクトル**。入力と同じサイズで出てくるから、そのまま次の層へ渡していける。

ひとことで言えば、$o$ は「**$q$ の視点で文脈を読み直して作り直した、その単語の新しい意味ベクトル**」です。この $o$ を積み重ねていくのが Transformer の本体になります。

:::

### 4.4 ダミーの数値で計算してみる

言葉だけだとピンとこないので、小さな例で最後まで計算してみましょう。3 次元のベクトルで、キーは 3 個（$n=3$）とします。$v_i = k_i$ でしたね。

$$
q = (1,\ 0,\ 1), \quad
k_1 = (1,\ 0,\ 1), \quad
k_2 = (0,\ 1,\ 0), \quad
k_3 = (1,\ 1,\ 0)
$$

**ステップ① 内積でスコア $a'_i = q k_i^{\top}$ を出す**

$$
a'_1 = 1{\cdot}1 + 0{\cdot}0 + 1{\cdot}1 = 2, \quad
a'_2 = 1{\cdot}0 + 0{\cdot}1 + 1{\cdot}0 = 0, \quad
a'_3 = 1{\cdot}1 + 0{\cdot}1 + 1{\cdot}0 = 1
$$

$q$ と向きがそっくりな $k_1$ がいちばん高いスコア（2）になりました。

**ステップ② softmax で注目度 $a_i$ にする**

$\exp(2) \approx 7.389,\ \exp(0) = 1,\ \exp(1) \approx 2.718$、合計は $\approx 11.107$ なので、

$$
a_1 = \frac{7.389}{11.107} \approx 0.665, \quad
a_2 = \frac{1.000}{11.107} \approx 0.090, \quad
a_3 = \frac{2.718}{11.107} \approx 0.245
$$

合計はちゃんと 1（$0.665 + 0.090 + 0.245 = 1.000$）になっています。

**ステップ③ Value の加重和で出力 $o$ を作る**

$$
\begin{aligned}
o &= 0.665\,(1,0,1) + 0.090\,(0,1,0) + 0.245\,(1,1,0) \\
  &= (0.665 + 0 + 0.245,\quad 0 + 0.090 + 0.245,\quad 0.665 + 0 + 0) \\
  &\approx (0.910,\ 0.335,\ 0.665)
\end{aligned}
$$

できました！　出力 $o \approx (0.910,\ 0.335,\ 0.665)$ は、最も注目された $k_1 = (1,0,1)$ にかなり近いベクトルになっています。「$q$ に関連のある内容を強く反映した新しい表現」という説明が、数値でも確かめられました。

## 5. 行列で一気に書く

$n$ 個のキー・バリューを縦に積んで行列 $K, V$ にまとめると、ステップ①〜③の足し算をすべて **行列積1本ずつ**にまとめられます。

$$
o = \text{softmax}(q K^{\top})\, V
$$

足し算の記号 $\sum$ が消えているのがポイントです。$qK^{\top}$ が全キーとの内積（スコア $a'_1, \dots, a'_n$）を一括で計算し、最後の $V$ との積が加重和（$\sum_i a_i v_i$）をまとめて行ってくれます。

| 式の部分 | 対応するステップ | 中身 |
| --- | --- | --- |
| $q K^{\top}$ | ① スコア | 全キーとの内積をまとめて計算 |
| $\text{softmax}(\cdots)$ | ② 注目度 | スコアを合計1の注目度へ |
| $(\cdots)\,V$ | ③ 出力 | 注目度を重みに Value を加重和 |

:::note[前半の式とのちがい：$\sqrt{d_k}$]

前半 3.4 で出てきた式 $\text{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V$ には $\sqrt{d_k}$ で割る項がありました。ここではまず **割らない素朴な形**で計算の流れをつかみ、7 節でこのスケーリングを足します。

:::

:::tip[これが「速い」の正体]

ステップ①〜③をループで1個ずつ回す代わりに、行列積で一気に計算できる——これが前半で触れた「全単語ペアを並列に処理できる」という Transformer の強みそのものです。次の実装でも、ループ版と行列版の両方を書いて、結果が一致することを確かめます。

:::

## 6. NumPy で実装する

それでは、5 節までの計算をそのままコードにします。Colab や手元の Python で動かしてみてください。

**コード 2.1：入力ベクトルを用意する**

```python
import numpy as np

# クエリ（1個）とキー（3個）。本章では Value = Key とする。
q = np.array([1.0, 0.0, 1.0])
K = np.array([
    [1.0, 0.0, 1.0],   # k1
    [0.0, 1.0, 0.0],   # k2
    [1.0, 1.0, 0.0],   # k3
])
V = K  # バリューベクトル = キーベクトル
```

**コード 2.2：スコア $a' = qK^{\top}$ を計算する**

```python
scores = q @ K.T          # 内積をまとめて計算 → [2.0, 0.0, 1.0]
print(scores)
```

`@` は行列積（ベクトルどうしなら内積）の演算子です。手計算の $a'_1, a'_2, a'_3 = 2, 0, 1$ と一致します。

**コード 2.3：softmax で注目度にする**

```python
def softmax(x):
    e = np.exp(x - np.max(x))   # オーバーフロー防止に最大値を引く
    return e / e.sum()

a = softmax(scores)         # → [0.665, 0.090, 0.245]
print(a, a.sum())           # 合計は 1.0
```

:::note[なぜ最大値を引くの？]

数学的には $\text{softmax}(x)$ と $\text{softmax}(x - c)$ は完全に同じ値になります（分子・分母に同じ $\exp(-c)$ が掛かって打ち消し合うため）。一方コンピュータでは $\exp$ が大きな入力で簡単にオーバーフローするので、$c = \max(x)$ を引いて指数を $0$ 以下に抑えるのが定番のテクニックです。結果は変わらず、計算だけ安定します。

:::

**コード 2.4：出力 $o$ を加重和で求める**

```python
o = a @ V                   # softmax(qKᵀ)V を1行で
print(o)                    # → [0.910, 0.335, 0.665]
```

手計算で出した $o \approx (0.910,\ 0.335,\ 0.665)$ とぴたり一致します🎉

<details>
<summary>ループ版とまとめて答え合わせ（クリックで開く）</summary>

行列版が正しいことを、定義どおりのループ版と比べて確認します。

```python
# ループ版：o = Σ aᵢ vᵢ
o_loop = np.zeros_like(V[0])
for i in range(len(V)):
    o_loop += a[i] * V[i]

print(np.allclose(o_loop, a @ V))   # True
```

`np.allclose` が `True` を返せば、ループ版と行列版の出力が（浮動小数点の誤差の範囲で）一致しているということです。

</details>

## 7. （発展）スケール化内積アテンション

最後に、前半 3.4 で出てきた **$\sqrt{d_k}$ で割るスケーリング**を、ちゃんと理由づけしてから実装に足します。やること自体は「スコアをキーの次元数 $d_k$ の平方根で割るだけ」です。

$$
o = \text{softmax}\!\left(\frac{q K^{\top}}{\sqrt{d_k}}\right) V
$$

これが **スケール化内積アテンション（Scaled Dot-Product Attention）** で、実用上はこちらが標準形です。$d_k$ はキー（とクエリ）ベクトルの次元数で、たとえば $d_k = 64$ や $512$ のような値です。

なぜわざわざ割り算を1つ足すのか——順番に見ていきましょう。

### 7.1 なぜ割るのか：softmax の「偏りすぎ」を防ぐ

思い出してほしいのが softmax の性質です。softmax は入力の**差**を指数関数で強調します。入力の値が全体的に大きくなると、いちばん大きい成分だけが極端に勝ち、出力がほぼ「1, 0, 0, …」に張り付いてしまいます。

これが起きると困ることが2つあります。

- **注目がガチガチに偏る**：本当は「動物に強め、疲れに少し」と配分したいのに、「動物にほぼ全部」になってしまい、他の単語の情報を拾えない。
- **学習が進まなくなる**：出力が 1 や 0 に張り付いた softmax は、入力が少し変わっても出力がほとんど動きません。これは「**勾配がほぼ 0**（傾きが消える）」状態で、モデルが「どっちに修正すればいいか」を受け取れず、学習が止まってしまいます。

下の図の左（①）は、スコア（注目元と注目先の相性）を横軸に、softmax が返す注目度を縦軸にとったものです。**中央のスコアがほどよい範囲**ではカーブに傾きがあり、スコアが少し動けば注目度もちゃんと動きます（＝「どっちに修正すべきか」が伝わる＝学習できる）。ところが**スコアが大きい右側**はカーブが平らに寝てしまい、スコアが動いても注目度はほぼ 1 のまま変わりません。この「平らな領域」が、注目が偏りきって**勾配がほぼ 0 になる飽和ゾーン**です。

右（②）は、この2つの動作点が実際に生む**注目度の配分（棒グラフ）**です。飽和側（赤）は1か所に全振りして他をいっさい拾えず、ほどよい側（青）は強弱をつけつつ他の単語の情報も残しています。

<div style={{display: 'flex', flexWrap: 'wrap', gap: '1.5rem', justifyContent: 'center', alignItems: 'flex-start', margin: '1.25rem 0'}}>
  <figure style={{margin: 0, textAlign: 'center'}}>
  <svg viewBox="0 0 380 235" width="350" role="img" aria-label="注目度のカーブ：スコアが大きいほどカーブが平らになり勾配が消える">
    <line x1="50" y1="30" x2="50" y2="172" stroke="currentColor" strokeOpacity="0.5" strokeWidth="1" />
    <line x1="50" y1="172" x2="345" y2="172" stroke="currentColor" strokeOpacity="0.5" strokeWidth="1" markerEnd="url(#axR)" />
    <defs>
      <marker id="axR" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
        <path d="M0,0 L6,3 L0,6 Z" fill="currentColor" fillOpacity="0.5" />
      </marker>
      <marker id="pullA" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
        <path d="M0,0 L6,3 L0,6 Z" fill="#3B82F6" />
      </marker>
    </defs>
    <line x1="46" y1="100" x2="50" y2="100" stroke="currentColor" strokeOpacity="0.5" strokeWidth="1" />
    <line x1="46" y1="30" x2="50" y2="30" stroke="currentColor" strokeOpacity="0.5" strokeWidth="1" />
    <text x="43" y="174" fontSize="8" fill="currentColor" textAnchor="end" fillOpacity="0.7">0</text>
    <text x="43" y="103" fontSize="8" fill="currentColor" textAnchor="end" fillOpacity="0.7">0.5</text>
    <text x="43" y="33" fontSize="8" fill="currentColor" textAnchor="end" fillOpacity="0.7">1</text>
    <polyline points="50,169.6 73,169.1 97,167.5 120,163.4 143,153.3 167,132.3 190,100 213,67.7 237,46.7 260,36.6 283,32.5 307,31 330,30.4" fill="none" stroke="currentColor" strokeWidth="2" strokeOpacity="0.85" />
    <line x1="283" y1="32.5" x2="213" y2="67.7" stroke="#3B82F6" strokeOpacity="0.6" strokeWidth="1.6" strokeDasharray="4 3" markerEnd="url(#pullA)" />
    <circle cx="283" cy="32.5" r="5" fill="#EF4444" fillOpacity="0.85" stroke="#EF4444" strokeWidth="1" />
    <circle cx="213" cy="67.7" r="5" fill="#3B82F6" fillOpacity="0.85" stroke="#3B82F6" strokeWidth="1" />
    <text x="300" y="30" fontSize="9" fill="#EF4444" textAnchor="start">スコア大：平ら</text>
    <text x="300" y="42" fontSize="8" fill="#EF4444" textAnchor="start" fillOpacity="0.85">勾配ほぼ0</text>
    <text x="150" y="62" fontSize="9" fill="#3B82F6" textAnchor="end">ほどよい：傾きあり</text>
    <text x="150" y="74" fontSize="8" fill="#3B82F6" textAnchor="end" fillOpacity="0.85">学習できる</text>
    <text x="240" y="58" fontSize="8.5" fill="#3B82F6" textAnchor="middle">÷√dₖ で引き戻す</text>
    <text x="200" y="192" fontSize="10" fill="currentColor" textAnchor="middle" fillOpacity="0.8">スコア（Q と K の相性）大きい →</text>
    <text x="20" y="100" fontSize="10" fill="currentColor" textAnchor="middle" fillOpacity="0.8" transform="rotate(-90 20 100)">注目度</text>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>① 注目度のカーブ：スコアが大きいほど平らに飽和し、傾き（勾配）が消える。$\sqrt{d_k}$ で割るのは、動作点を平らな右側から傾きのある中央へ引き戻す操作にあたる</figcaption>
  </figure>
  <figure style={{margin: 0, textAlign: 'center'}}>
    <svg viewBox="0 0 250 175" width="240" role="img" aria-label="注目度の配分：飽和すると1か所に偏り、ほどよいと強弱をつけつつ他も残る">
      <line x1="15" y1="125" x2="235" y2="125" stroke="currentColor" strokeOpacity="0.5" strokeWidth="1" />
      <text x="66" y="18" fontSize="9" fill="#EF4444" textAnchor="middle">スコア大（飽和）</text>
      <rect x="35" y="45" width="18" height="80" fill="#EF4444" fillOpacity="0.30" stroke="#EF4444" strokeOpacity="0.65" strokeWidth="1.1" />
      <rect x="58" y="122" width="18" height="3" fill="#EF4444" fillOpacity="0.30" stroke="#EF4444" strokeOpacity="0.65" strokeWidth="1.1" />
      <rect x="81" y="122" width="18" height="3" fill="#EF4444" fillOpacity="0.30" stroke="#EF4444" strokeOpacity="0.65" strokeWidth="1.1" />
      <text x="44" y="138" fontSize="8" fill="currentColor" textAnchor="middle" fillOpacity="0.8">a₁</text>
      <text x="67" y="138" fontSize="8" fill="currentColor" textAnchor="middle" fillOpacity="0.8">a₂</text>
      <text x="90" y="138" fontSize="8" fill="currentColor" textAnchor="middle" fillOpacity="0.8">a₃</text>
      <text x="66" y="160" fontSize="9" fill="#EF4444" textAnchor="middle">1か所に全振り</text>
      <line x1="118" y1="35" x2="118" y2="125" stroke="currentColor" strokeOpacity="0.2" strokeWidth="1" strokeDasharray="3 3" />
      <text x="171" y="18" fontSize="9" fill="#3B82F6" textAnchor="middle">ほどよい</text>
      <rect x="140" y="72" width="18" height="53" fill="#3B82F6" fillOpacity="0.30" stroke="#3B82F6" strokeOpacity="0.65" strokeWidth="1.1" />
      <rect x="163" y="117" width="18" height="8" fill="#3B82F6" fillOpacity="0.30" stroke="#3B82F6" strokeOpacity="0.65" strokeWidth="1.1" />
      <rect x="186" y="105" width="18" height="20" fill="#3B82F6" fillOpacity="0.30" stroke="#3B82F6" strokeOpacity="0.65" strokeWidth="1.1" />
      <text x="149" y="138" fontSize="8" fill="currentColor" textAnchor="middle" fillOpacity="0.8">a₁</text>
      <text x="172" y="138" fontSize="8" fill="currentColor" textAnchor="middle" fillOpacity="0.8">a₂</text>
      <text x="195" y="138" fontSize="8" fill="currentColor" textAnchor="middle" fillOpacity="0.8">a₃</text>
      <text x="171" y="160" fontSize="9" fill="#3B82F6" textAnchor="middle">強弱つき・他も残る</text>
    </svg>
    <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>② 実際の注目度の配分：飽和（赤）は1か所に全振りして他を拾えない。ほどよい（青）は強弱をつけつつ他の情報も残る（具体的な数値は 7.3 で確認）</figcaption>
  </figure>
</div>

つまりスコアが大きくなりすぎると softmax が飽和して使いものにならない。だから **スコアの大きさをほどよく抑えたい**——これがスケーリングの目的です。後の 7.2 で見るように、その「ほどよい大きさ」に引き戻す割り算がちょうど $\sqrt{d_k}$ になります。

### 7.2 なぜ「平方根」なのか：スコアの“ばらつき”をそろえる

ここがいちばんの肝です。「大きさを抑えたい」だけなら適当な定数で割ってもよさそうなのに、なぜちょうど $\sqrt{d_k}$ なのでしょうか。理由は、**スコアのばらつき（標準偏差）がちょうど $\sqrt{d_k}$ に比例して大きくなる**からです。

ざっくり次のように考えます。クエリ $q$ とキー $k$ の各成分が、平均 $0$・ばらつき（分散）$1$ くらいでバラバラな値だとします。スコアは内積なので、$d_k$ 個の項の足し算です。

$$
q\,k^{\top} = q_1 k_1 + q_2 k_2 + \dots + q_{d_k} k_{d_k}
$$

内積の計算じたいは、**対応する成分どうしを掛けて、全部足すだけ**です。$1$ より大きい値が混じっても手順は変わりません。たとえば $3$ 次元（$d_k=3$）で

$$
q = (2,\ 1,\ 3), \qquad k = (1,\ 2,\ 2)
$$

なら、

$$
q\,k^{\top} = 2\cdot 1 + 1\cdot 2 + 3\cdot 2 = 2 + 2 + 6 = 10
$$

のように、位置ごとの積 $2{\cdot}1,\ 1{\cdot}2,\ 3{\cdot}2$ を足し合わせた **1つの数（スコア）** になります。

さて、ここからは「**次元 $d_k$ が大きいほどスコアが大きくなる**」様子を見やすくするため、各成分を $+1$ か $-1$ に単純化します（平均をちょうど $0$、分散をちょうど $1$ にそろえ、上の仮定にぴったり合わせるためです）。すると内積の各項 $q_i k_i$ も $+1$ か $-1$ のどちらかになり、**スコアは「$\pm 1$ を $d_k$ 個足し合わせたもの」**になります。

まず $d_k = 4$ で計算してみます。

| 位置 $i$ | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| クエリ $q_i$ | $+1$ | $+1$ | $-1$ | $+1$ |
| キー $k_i$ | $+1$ | $-1$ | $-1$ | $+1$ |
| 積 $q_i k_i$ | $+1$ | $-1$ | $+1$ | $+1$ |

スコアは各項の合計なので、

$$
q\,k^{\top} = (+1) + (-1) + (+1) + (+1) = 2
$$

目安どおり $\sqrt{4}=2$ くらいの大きさになりました。次に次元を $d_k = 16$ に増やすと、今度は $\pm 1$ を **16 個**足すことになります。たとえば $+1$ が 10 個・$-1$ が 6 個そろえば、スコア $= 10 - 6 = 4$。やはり目安の $\sqrt{16}=4$ くらいです。

**次元が $4 \to 16$ と 4 倍になると、スコアの大きさの目安は $2 \to 4$ と 2 倍（＝$\sqrt{4}$ 倍）に増えました。** 次元が増えるほど足し合わさる項が増えて、スコアが大きく振れるわけです。

この背景にあるのが「**独立な数をたくさん足すと、ばらつきは足した個数だけ大きくなる**」という統計の性質です。各項のばらつきが $1$ なら、$d_k$ 個足したスコアの分散は $d_k$、その平方根である **標準偏差（ばらつきの目安）は $\sqrt{d_k}$** になります。

:::note[コラム：分散と標準偏差ってなに？]

どちらも「**データが平均のまわりにどれくらい散らばっているか**」を表す数です。同じ平均でも、ぎゅっと固まっているか、広く散っているかを1つの数で言い表したいときに使います。

たとえば 3 人のテストの点を考えます。平均はどれも 50 点ですが、散らばり方が違います。

- $\{50,\ 50,\ 50\}$ … 全員ぴったり同じ。散らばり **ゼロ**。
- $\{40,\ 50,\ 60\}$ … 少し散らばっている。
- $\{0,\ 50,\ 100\}$ … 大きく散らばっている。

これを数値にする手順はこうです。

1. 各データの「平均からのズレ」を求める。
2. ズレを **2乗** して、その平均をとる。これが **分散**。
3. 分散の **平方根**（ルート）をとる。これが **標準偏差**。

$\{0,50,100\}$ なら、ズレは $-50,\ 0,\ +50$。2乗して平均すると分散は $\frac{2500+0+2500}{3} \approx 1667$、その平方根の標準偏差は $\sqrt{1667} \approx 41$ です。「だいたい平均 ±41 点くらいに散らばっている」と読めます。

3つを数直線に並べると、ひと目で違いが分かります。**平均（緑の線）はどれも 50 で同じ**なのに、散らばりを表す**青い帯の幅（＝標準偏差）**が大きく変わります。

<figure style={{margin: '1.25rem auto', textAlign: 'center', maxWidth: '460px'}}>
  <svg viewBox="0 0 440 235" width="430" role="img" aria-label="平均は同じ50でも、データの散らばり（標準偏差）が異なる3つの例を数直線で比較">
    <line x1="220" y1="40" x2="220" y2="206" stroke="#10B981" strokeWidth="1.4" strokeDasharray="4 3" strokeOpacity="0.9" />
    <text x="220" y="32" fontSize="10" fill="#10B981" textAnchor="middle">平均 50</text>
    <text x="70" y="56" fontSize="9" fill="currentColor" textAnchor="start" fillOpacity="0.85">{'{50, 50, 50}'}</text>
    <line x1="70" y1="70" x2="370" y2="70" stroke="currentColor" strokeOpacity="0.35" strokeWidth="1" />
    <circle cx="220" cy="70" r="4.5" fill="currentColor" fillOpacity="0.85" />
    <text x="232" y="65" fontSize="8" fill="currentColor" textAnchor="start" fillOpacity="0.6">×3（重なり）</text>
    <text x="380" y="73" fontSize="9" fill="#3B82F6" textAnchor="start">σ = 0</text>
    <text x="70" y="116" fontSize="9" fill="currentColor" textAnchor="start" fillOpacity="0.85">{'{40, 50, 60}'}</text>
    <rect x="195.5" y="120" width="49" height="20" rx="2" fill="#3B82F6" fillOpacity="0.18" stroke="#3B82F6" strokeOpacity="0.5" strokeWidth="1" />
    <line x1="70" y1="130" x2="370" y2="130" stroke="currentColor" strokeOpacity="0.35" strokeWidth="1" />
    <circle cx="190" cy="130" r="4.5" fill="currentColor" fillOpacity="0.85" />
    <circle cx="220" cy="130" r="4.5" fill="currentColor" fillOpacity="0.85" />
    <circle cx="250" cy="130" r="4.5" fill="currentColor" fillOpacity="0.85" />
    <text x="380" y="133" fontSize="9" fill="#3B82F6" textAnchor="start">σ ≈ 8.2</text>
    <text x="70" y="176" fontSize="9" fill="currentColor" textAnchor="start" fillOpacity="0.85">{'{0, 50, 100}'}</text>
    <rect x="97.5" y="180" width="245" height="20" rx="2" fill="#3B82F6" fillOpacity="0.18" stroke="#3B82F6" strokeOpacity="0.5" strokeWidth="1" />
    <line x1="70" y1="190" x2="370" y2="190" stroke="currentColor" strokeOpacity="0.35" strokeWidth="1" />
    <circle cx="70" cy="190" r="4.5" fill="currentColor" fillOpacity="0.85" />
    <circle cx="220" cy="190" r="4.5" fill="currentColor" fillOpacity="0.85" />
    <circle cx="370" cy="190" r="4.5" fill="currentColor" fillOpacity="0.85" />
    <text x="380" y="193" fontSize="9" fill="#3B82F6" textAnchor="start">σ ≈ 41</text>
    <text x="70" y="220" fontSize="8" fill="currentColor" textAnchor="middle" fillOpacity="0.6">0</text>
    <text x="220" y="220" fontSize="8" fill="currentColor" textAnchor="middle" fillOpacity="0.6">50</text>
    <text x="370" y="220" fontSize="8" fill="currentColor" textAnchor="middle" fillOpacity="0.6">100</text>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>平均（緑）はどれも 50 で同じでも、散らばりを表す青い帯の幅＝標準偏差 $\sigma$ は大きく違う。分散はこの幅をさらに2乗した量にあたる</figcaption>
</figure>

$$
\text{分散} = \overline{(\text{ズレ})^2}, \qquad \text{標準偏差} = \sqrt{\text{分散}}
$$

では、2つは結局それぞれ何を表しているのでしょうか。同じ「散らばり具合」を表す仲間ですが、役割を分けるとこうです。

- **分散** … 散らばりの大きさそのもの。ただしズレを **2乗** しているので、単位（スケール）が元と変わってしまう（点なら「点²」）。足し算と相性がよく、**数式や理論で扱いやすい**のが長所。
- **標準偏差** … その分散を **平方根で元のスケールに戻した** もの。「だいたい平均 ±これくらい散らばる」と、**人間が実感しやすい目安**になる。

ひとことで言えば、**分散は計算向き・標準偏差は読み取り向き**——同じ散らばりを別の単位で見ているだけ、と思えば十分です。なぜわざわざ分散を経由するかというと、次に出てくる「足すと散らばりがたまる」という性質が、標準偏差ではなく**分散でこそきれいに成り立つ**からです。

このコラムで覚えてほしいのは1点だけ。**独立なデータを足し合わせると、分散はそのまま足し算でたまっていく**（これを分散の加法性といいます）。だから「$1$ の項を $d_k$ 個足すと分散は $d_k$、標準偏差はその平方根の $\sqrt{d_k}$」という、本文の話につながるのです。

:::

| 次元 $d_k$ | スコアの典型的な大きさ（標準偏差 $\sqrt{d_k}$） |
| --- | --- |
| $4$ | $2$ |
| $64$ | $8$ |
| $512$ | $\approx 22.6$ |

つまり**次元が大きいモデルほど、何もしなければスコアが自動的に大きくなり**、softmax が飽和しやすくなります。そこでスコアを $\sqrt{d_k}$ で割ると、ばらつきが $1$ 前後に戻り、**$d_k$ がいくつでもスコアの大きさが一定**に保たれます。これがちょうど「平方根」を使う理由です。

:::note[コラム：なぜ「割る」とばらつきが 1 に戻るの？]

カギは、**データ全体を同じ数で割ると、散らばりの幅（標準偏差）も同じ数だけ縮む**という性質です。数直線に散らばった点を、まるごと $\frac{1}{8}$ に縮小コピーすると、点の間隔（＝散らばりの幅）も $\frac{1}{8}$ になる——それと同じイメージです。

<figure style={{margin: '1.25rem auto', textAlign: 'center', maxWidth: '460px'}}>
  <svg viewBox="0 0 440 215" width="430" role="img" aria-label="スコアを √dk で割ると、散らばりの幅が √dk から 1 に縮む様子">
    <line x1="220" y1="38" x2="220" y2="188" stroke="#10B981" strokeWidth="1.3" strokeDasharray="4 3" strokeOpacity="0.85" />
    <text x="220" y="30" fontSize="9" fill="#10B981" textAnchor="middle">平均（中心）</text>
    <text x="70" y="52" fontSize="9" fill="currentColor" textAnchor="start" fillOpacity="0.85">割る前（dₖ = 64）</text>
    <rect x="92" y="58" width="256" height="20" rx="2" fill="#3B82F6" fillOpacity="0.18" stroke="#3B82F6" strokeOpacity="0.5" strokeWidth="1" />
    <line x1="70" y1="68" x2="370" y2="68" stroke="currentColor" strokeOpacity="0.35" strokeWidth="1" />
    <circle cx="92" cy="68" r="3.5" fill="currentColor" fillOpacity="0.8" />
    <circle cx="160" cy="68" r="3.5" fill="currentColor" fillOpacity="0.8" />
    <circle cx="220" cy="68" r="3.5" fill="currentColor" fillOpacity="0.8" />
    <circle cx="290" cy="68" r="3.5" fill="currentColor" fillOpacity="0.8" />
    <circle cx="348" cy="68" r="3.5" fill="currentColor" fillOpacity="0.8" />
    <text x="92" y="92" fontSize="8" fill="currentColor" textAnchor="middle" fillOpacity="0.6">-8</text>
    <text x="220" y="92" fontSize="8" fill="currentColor" textAnchor="middle" fillOpacity="0.6">0</text>
    <text x="348" y="92" fontSize="8" fill="currentColor" textAnchor="middle" fillOpacity="0.6">+8</text>
    <text x="378" y="71" fontSize="9" fill="#3B82F6" textAnchor="start">幅 ≈ √dₖ = 8</text>
    <line x1="150" y1="100" x2="150" y2="138" stroke="currentColor" strokeOpacity="0.55" strokeWidth="1.4" markerEnd="url(#shrinkA)" />
    <defs>
      <marker id="shrinkA" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
        <path d="M0,0 L6,3 L0,6 Z" fill="currentColor" fillOpacity="0.6" />
      </marker>
    </defs>
    <text x="160" y="123" fontSize="9" fill="currentColor" textAnchor="start">÷√dₖ（=÷8）で縮小</text>
    <text x="70" y="152" fontSize="9" fill="currentColor" textAnchor="start" fillOpacity="0.85">割った後</text>
    <rect x="204" y="158" width="32" height="20" rx="2" fill="#3B82F6" fillOpacity="0.30" stroke="#3B82F6" strokeOpacity="0.6" strokeWidth="1" />
    <line x1="70" y1="168" x2="370" y2="168" stroke="currentColor" strokeOpacity="0.35" strokeWidth="1" />
    <circle cx="204" cy="168" r="3.5" fill="currentColor" fillOpacity="0.8" />
    <circle cx="220" cy="168" r="3.5" fill="currentColor" fillOpacity="0.8" />
    <circle cx="236" cy="168" r="3.5" fill="currentColor" fillOpacity="0.8" />
    <text x="204" y="192" fontSize="8" fill="currentColor" textAnchor="middle" fillOpacity="0.6">-1</text>
    <text x="236" y="192" fontSize="8" fill="currentColor" textAnchor="middle" fillOpacity="0.6">+1</text>
    <text x="378" y="171" fontSize="9" fill="#3B82F6" textAnchor="start">幅 = 1</text>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>$\sqrt{d_k}$ で割ると、散らばりの幅が $\sqrt{d_k}$ 倍ぶんだけ縮み、ちょうど 1 に揃う（縮小コピーのイメージ）</figcaption>
</figure>

式で書くと（$\sigma$ ＝標準偏差）こうなります。

$$
\sigma\!\left(\frac{\text{スコア}}{\sqrt{d_k}}\right) = \frac{\sigma(\text{スコア})}{\sqrt{d_k}} = \frac{\sqrt{d_k}}{\sqrt{d_k}} = 1
$$

割る前のばらつきが $\sqrt{d_k}$、それを $\sqrt{d_k}$ で割るので、ちょうど $1$ になります。

式だけだとイメージしにくいので、**小さな数で実際に計算**してみましょう。$d_k = 4$ のモデルで、あるクエリのスコアが2つ、$-2$ と $+2$ だったとします（このとき $\sqrt{d_k} = \sqrt{4} = 2$）。スコアを $\sqrt{d_k}=2$ で割る前と後で、標準偏差を計算して比べます。

| | 割る前のスコア | $\sqrt{d_k}=2$ で割った後 |
| --- | --- | --- |
| スコア | $-2,\ +2$ | $-1,\ +1$ |
| 平均 | $0$ | $0$ |
| 平均からのズレ | $-2,\ +2$ | $-1,\ +1$ |
| 分散（ズレ²の平均） | $\dfrac{(-2)^2+2^2}{2}=4$ | $\dfrac{(-1)^2+1^2}{2}=1$ |
| 標準偏差 $\sigma$ | $\sqrt{4}=2$ | $\sqrt{1}=1$ |

割る前の標準偏差は $2$（＝$\sqrt{d_k}$）、$2$ で割った後は $1$。たしかに **スコアを $2$ で割ったら、標準偏差も $2$ で割られて、ちょうど $1$ になりました**。式の $\frac{\sqrt{d_k}}{\sqrt{d_k}}=1$ は、これを文字に置きかえただけのものです。

次元が大きくても、まったく同じことが起きます。

- $d_k = 64$：ばらつき $8$ を $\sqrt{64}=8$ で割る → $1$
- $d_k = 512$：ばらつき $\approx 22.6$ を $\sqrt{512}\approx 22.6$ で割る → $\approx 1$

どの次元でも割ったあとのばらつきは $1$ に揃います。だから次元 $d_k$ を変えても、softmax に入るスコアの振れ幅はいつも同じ手頃なサイズ（およそ $\pm 1$）に保たれ、7.1 で見た飽和を避けられるのです。

:::

:::note[“分散が次元の数だけ足し合わさる”ってどういうこと？]

サイコロ1個の出目はばらつきますが、100 個振って合計すると、合計値はもっと大きな幅でばらつきます（だいたい「個数の平方根」倍に広がる）。内積も同じで、独立な項 $q_i k_i$ を $d_k$ 個足すほど、合計であるスコアの振れ幅が大きくなります。分散（ばらつきの2乗）が $d_k$ 倍、標準偏差が $\sqrt{d_k}$ 倍、というわけです。

なお、この「各成分が平均0・分散1で独立」という前提は厳密には成り立ちませんが、**スケーリングの動機**を理解するにはこの近似で十分です。元論文 "Attention Is All You Need" でも、この分散の議論を根拠に $\sqrt{d_k}$ で割ることが提案されています。

:::

### 7.3 スケーリングしないとどうなるか（数値で体感）

4.4 節で使ったスコア $a' = (2,\ 0,\ 1)$ は、実はすでに「ほどよい大きさ」になっていました。もし $d_k = 64$ のモデルで**スケーリングをサボった**場合、生のスコアは標準偏差 $\sqrt{64}=8$ くらいまで膨らみ、たとえば $(16,\ 0,\ 8)$ のような値になります。同じ「向きの相性」なのに、大きさだけが $8$ 倍になったイメージです。

この2つを softmax に通すと、結果はまるで違います。

$$
\text{softmax}(16,\ 0,\ 8) \approx (1.000,\ 0.000,\ 0.000), \qquad
\text{softmax}(2,\ 0,\ 1) \approx (0.665,\ 0.090,\ 0.245)
$$

スケーリングしない左は **1か所に全振り**（飽和）してしまい、他の単語の情報をいっさい拾えません。$\sqrt{d_k}=8$ で割った右（＝ $(2,0,1)$）は、強弱をつけつつ他もちゃんと残した、ほどよい配分になっています。

<div style={{display: 'flex', flexWrap: 'wrap', gap: '1.5rem', justifyContent: 'center', alignItems: 'flex-start', margin: '1.25rem 0'}}>
  <figure style={{margin: 0, textAlign: 'center'}}>
    <svg viewBox="0 0 220 165" width="215" role="img" aria-label="スケーリングなしの softmax は1か所に全振りして飽和する">
      <line x1="28" y1="115" x2="200" y2="115" stroke="currentColor" strokeOpacity="0.5" strokeWidth="1" />
      <rect x="45" y="32" width="34" height="83" fill="#EF4444" fillOpacity="0.30" stroke="#EF4444" strokeOpacity="0.65" strokeWidth="1.2" />
      <rect x="100" y="113" width="34" height="2" fill="#EF4444" fillOpacity="0.30" stroke="#EF4444" strokeOpacity="0.65" strokeWidth="1.2" />
      <rect x="155" y="113" width="34" height="2" fill="#EF4444" fillOpacity="0.30" stroke="#EF4444" strokeOpacity="0.65" strokeWidth="1.2" />
      <text x="62" y="26" fontSize="9" fill="currentColor" textAnchor="middle">1.000</text>
      <text x="117" y="108" fontSize="9" fill="currentColor" textAnchor="middle" fillOpacity="0.7">0.000</text>
      <text x="172" y="108" fontSize="9" fill="currentColor" textAnchor="middle" fillOpacity="0.7">0.000</text>
      <text x="62" y="128" fontSize="9" fill="currentColor" textAnchor="middle" fillOpacity="0.8">a₁</text>
      <text x="117" y="128" fontSize="9" fill="currentColor" textAnchor="middle" fillOpacity="0.8">a₂</text>
      <text x="172" y="128" fontSize="9" fill="currentColor" textAnchor="middle" fillOpacity="0.8">a₃</text>
      <text x="113" y="152" fontSize="10" fill="#EF4444" textAnchor="middle">スコア (16, 0, 8)：飽和</text>
    </svg>
    <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>① スケーリングなし：1か所に全振りし、他の情報を拾えない</figcaption>
  </figure>
  <figure style={{margin: 0, textAlign: 'center'}}>
    <svg viewBox="0 0 220 165" width="215" role="img" aria-label="√dk でスケーリングした softmax はほどよい配分になる">
      <line x1="28" y1="115" x2="200" y2="115" stroke="currentColor" strokeOpacity="0.5" strokeWidth="1" />
      <rect x="45" y="60" width="34" height="55" fill="#3B82F6" fillOpacity="0.30" stroke="#3B82F6" strokeOpacity="0.65" strokeWidth="1.2" />
      <rect x="100" y="108" width="34" height="7" fill="#3B82F6" fillOpacity="0.30" stroke="#3B82F6" strokeOpacity="0.65" strokeWidth="1.2" />
      <rect x="155" y="95" width="34" height="20" fill="#3B82F6" fillOpacity="0.30" stroke="#3B82F6" strokeOpacity="0.65" strokeWidth="1.2" />
      <text x="62" y="54" fontSize="9" fill="currentColor" textAnchor="middle">0.665</text>
      <text x="117" y="102" fontSize="9" fill="currentColor" textAnchor="middle" fillOpacity="0.8">0.090</text>
      <text x="172" y="89" fontSize="9" fill="currentColor" textAnchor="middle">0.245</text>
      <text x="62" y="128" fontSize="9" fill="currentColor" textAnchor="middle" fillOpacity="0.8">a₁</text>
      <text x="117" y="128" fontSize="9" fill="currentColor" textAnchor="middle" fillOpacity="0.8">a₂</text>
      <text x="172" y="128" fontSize="9" fill="currentColor" textAnchor="middle" fillOpacity="0.8">a₃</text>
      <text x="113" y="152" fontSize="10" fill="#3B82F6" textAnchor="middle">÷√64 → (2, 0, 1)：ほどよい</text>
    </svg>
    <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>② $\sqrt{d_k}$ でスケーリング：強弱をつけつつ他の情報も残る</figcaption>
  </figure>
</div>

同じ「向きの相性」から出発しても、スケーリングの有無で注目度がここまで変わります。**だから実用の Transformer は必ず $\sqrt{d_k}$ で割る**、というわけです。

### 7.4 実装

理由が分かったところで、コードにします。6 節にわり算を1つ足すだけです。Query が複数（行列 $Q$）になっても、同じ式がそのまま動きます。

```python
def softmax_rows(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
    d_k = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)          # スケール化したスコア
    weights = softmax_rows(scores)           # 行ごとに softmax
    return weights @ V

# 複数クエリ（2個）でも同じ関数でOK
Q = np.array([[1.0, 0.0, 1.0],
              [0.0, 1.0, 1.0]])
print(scaled_dot_product_attention(Q, K, V))
```

`softmax_rows` は「行ごとに合計1にする softmax」です。クエリが複数あるときは、各クエリ（各行）について独立に注目度を出す必要があるため、こうして行方向に正規化します。

:::tip[この章のまとめ]

**前半（概念）**

- Transformer は「全単語ペアの注目度を一気に計算する」アーキテクチャで、いまの LLM の土台。RNN の「遅い・遠い単語を忘れる」を、「順番に読むのをやめる」ことで解決した。
- 心臓部の **Attention** は、**内積で関連度 → softmax で注目度 → Value の加重和** という、前章の数学そのもの。

**後半（実装）**

- 注目度は $a_i = \dfrac{\exp(q k_i^{\top})}{\sum_j \exp(q k_j^{\top})}$。内積を softmax に通したスカラで、合計1。
- 出力は Value の加重和 $o = a_1 v_1 + \dots + a_n v_n$。注目した Value が濃く混ざった「新しい表現」。
- 行列でまとめると $o = \text{softmax}(qK^{\top})V$。NumPy では `softmax(q @ K.T) @ V` の数行で実装でき、手計算と一致する。
- 実用形は $\sqrt{d_k}$ で割る **スケール化内積アテンション** $\text{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V$。

:::

---

この章では、Transformer の心臓部 Attention を、**概念から実装まで一気通貫**でつかみました。ここまでは「単語はすでにベクトルになっている」ことを前提にしてきましたが、その入り口——**テキストをどうやってトークン（数）に変えるか**（トークナイザー）や、モデルの**学習**については、今後の章で扱っていく予定です。
