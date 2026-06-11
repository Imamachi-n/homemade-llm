---
sidebar_position: 2
title: "Chapter 2: Transformer とは何か"
---

# Chapter 2: Transformer とは何か

[前章](./chapter1.md)では、ベクトル・内積・行列積・softmax といった「LLM を支える数学」を積み上げました。この章では、いよいよその数学が組み上がってできる主役、**Transformer（トランスフォーマー）** の正体に迫ります。

まずは「そもそも Transformer って何なの？」を一言でつかみ、次に「なぜそんなものが必要だったのか」という背景、そして Transformer の心臓部である **Attention（注意機構）** を、前章の数学とつなげながら見ていきます。

:::tip[この章の読み方]

数式や用語が出てきても身構えなくて大丈夫です。新しい記号や式は **使う直前に意味を渡します**。特に Attention は「前章で学んだ内積・softmax をくり返しているだけ」だと分かると、一気に見通しが良くなります。

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

:::tip[ここまでのまとめ]

- Transformer は「全単語ペアの注目度を一気に計算する」アーキテクチャで、いまの LLM の土台。
- RNN の「遅い・遠い単語を忘れる」を、「順番に読むのをやめる」ことで解決した。
- 心臓部の **Attention** は、**内積で関連度 → softmax で注目度 → Value の加重和** という、前章の数学そのもの。
- まとめると $\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V$。$QK^{\top}$ は全ペア内積の一括計算。

:::

---

この章では Transformer の正体と、心臓部 Attention の仕組みをつかみました。次の節以降では、これらのブロックがどう積み重なって1つのモデルになるのか——位置エンコーディング、残差接続、フィードフォワード層、そしてエンコーダ／デコーダといった全体像——を見ていきます。
