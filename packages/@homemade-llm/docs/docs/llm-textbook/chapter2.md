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

流れはこうです。ある単語の **Query** を、全単語の **Key** と照らし合わせて「相性（関連度）」を測り、相性の良い単語の **Value** をたくさん受け取る——これが Attention の1回の計算です。

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

### 4.2 バリューベクトル $v$ を用意する

出力を求めるには、前半 3.2 で出てきた **バリューベクトル** $v$（実際に受け取る中身の情報）を、いよいよ式に登場させます。本来は Key とは別物ですが、計算を追いやすくするため、この章の範囲では、

$$
\text{バリューベクトル } v = \text{キーベクトル } k
$$

とします。つまり $v_i = k_i$。いったんは「キーと同じものをもう一度使うだけ」と思っておけば十分です。

:::note[なぜ Value を別に呼ぶの？]

本来 Attention では Key（関連度を測る見出し）と Value（実際に渡す中身）は別々の役割を持ち、一般には $v_i \ne k_i$ です（前半 3.2 の図書館のたとえを思い出してください）。ただ計算を最初に追う段階では、両者を同じにしておくと式が見通しやすくなります。この章では $v_i = k_i$ で進め、両者を分ける一般形は後の章で扱います。

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

最後に、前半 3.4 で出てきた **$\sqrt{d_k}$ で割るスケーリング**を実装に足します。スコアをキーの次元数 $d_k$ の平方根で割るだけです。

$$
o = \text{softmax}\!\left(\frac{q K^{\top}}{\sqrt{d_k}}\right) V
$$

これが **スケール化内積アテンション（Scaled Dot-Product Attention）** で、実用上はこちらが標準形です（$d_k$ が大きいときに softmax が極端に偏るのを防ぐ、というのが理由でした）。

実装は 6 節にわり算を1つ足すだけです。Query が複数（行列 $Q$）になっても、同じ式がそのまま動きます。

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
