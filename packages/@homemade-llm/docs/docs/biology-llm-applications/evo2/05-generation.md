---
sidebar_position: 5
title: "05. ゲノムスケール生成"
---

# 05. ゲノムスケール生成

ここまでは Evo 2 の **予測** と **解釈** を見てきました。このページでは、Evo 2 の **生成（generation）** 能力——学習した分布から **新しい DNA 配列を作り出す** 力を扱います。GPT がテキストを生成するのと同じ自己回帰の仕組みで、Evo 2 は **ミトコンドリアから細菌ゲノム規模** までの配列を生成します。

## 1. 生成モデルとしての Evo 2

Evo 2 は予測器であると同時に **生成モデル** です。仕組みは自然言語 LLM とまったく同じ自己回帰生成で、プロンプト（DNA の一部）を与えると、その続きを **1 塩基ずつ確率的にサンプリング** していきます。

$$
x_i \sim p(x_i \mid x_1, \dots, x_{i-1})
$$

<figure style={{margin: '1.5rem auto', textAlign: 'center', maxWidth: '560px'}}>
  <svg viewBox="0 0 560 170" width="100%" role="img" aria-label="自己回帰生成。プロンプトを与え、1塩基ずつ生成して配列を伸ばす">
    <rect x="20" y="58" width="130" height="44" rx="5" fill="#3B82F6" fillOpacity="0.1" stroke="#3B82F6" strokeWidth="1.4" />
    <text x="85" y="76" fontSize="10.5" fill="currentColor" textAnchor="middle">プロンプト</text>
    <text x="85" y="92" fontSize="11" fill="currentColor" textAnchor="middle">…A T G C</text>
    <line x1="150" y1="80" x2="184" y2="80" stroke="currentColor" strokeOpacity="0.5" strokeWidth="1.4" />
    <polygon points="188,80 179,75 179,85" fill="currentColor" fillOpacity="0.5" />
    <rect x="190" y="58" width="86" height="44" rx="5" fill="currentColor" fillOpacity="0.06" stroke="currentColor" strokeOpacity="0.7" strokeWidth="1.6" />
    <text x="233" y="84" fontSize="13" fill="currentColor" textAnchor="middle" fontWeight="700">Evo 2</text>
    <line x1="276" y1="80" x2="310" y2="80" stroke="currentColor" strokeOpacity="0.5" strokeWidth="1.4" />
    <polygon points="314,80 305,75 305,85" fill="currentColor" fillOpacity="0.5" />
    <rect x="316" y="58" width="150" height="44" rx="5" fill="#10B981" fillOpacity="0.1" stroke="#10B981" strokeWidth="1.4" />
    <text x="391" y="76" fontSize="10.5" fill="currentColor" textAnchor="middle">次の1塩基を予測</text>
    <text x="391" y="92" fontSize="11" fill="currentColor" textAnchor="middle">…A T G C <tspan fill="#10B981" fontWeight="700">G</tspan></text>
    <path d="M 391 104 C 391 140, 85 140, 85 106" fill="none" stroke="currentColor" strokeOpacity="0.45" strokeWidth="1.3" strokeDasharray="5 4" />
    <polygon points="85,108 80,118 90,118" fill="currentColor" fillOpacity="0.45" />
    <text x="238" y="152" fontSize="10" fill="currentColor" fillOpacity="0.75" textAnchor="middle">生成した塩基をプロンプトに足して繰り返す（自己回帰）</text>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>自己回帰生成。GPT のテキスト生成と同じく、生成した塩基を文脈に足しながら配列を伸ばす</figcaption>
</figure>

## 2. 遺伝子補完：プロンプトから機能タンパク質を作る

まず小さな単位として、**遺伝子の補完** を評価しました。古細菌・原核生物・4 系統の真核生物（菌類・原生生物・植物・動物）の計6種について、高度に保存された遺伝子を選び、**上流 1,000 bp ＋ 遺伝子の最初の 500〜1,000 bp** をプロンプトとして与え、続きを生成させます。

生成された遺伝子が本物とどれだけ一致するかを **アミノ酸配列回復率（amino acid sequence recovery）** で測ると、Evo 2 は高い回復率を示し、**モデルサイズが大きいほど向上**（40B / 7B が Evo 1 を上回る）、長文脈学習を通じても高い回復率を維持しました（Fig. 5b）。

## 3. バイオセーフティ：作れないものもある

[02](./02-architecture.md) で触れたデータ除外の効果が、生成にも表れます。ヒトに感染するウイルスのタンパク質については、Evo 2 は **ほぼランダムな回復率しか出せません**。直接ウイルスタンパク質を引き出そうとしても生成できず、**意図しない・偶発的なヒト病原体ウイルスタンパク質の生成が防がれています**（Extended Data Fig. 2c）。安全のために「あえて作れなくしてある」わけです。

## 4. ゲノムスケール生成

ここからが本領です。Evo 2 は **遺伝子単体ではなくゲノム規模** の配列を生成できます。論文では3つのスケールで検証しました。

<figure style={{margin: '1.5rem auto', textAlign: 'center', maxWidth: '560px'}}>
  <svg viewBox="0 0 560 170" width="100%" role="img" aria-label="生成したゲノムのスケール比較。ミトコンドリア16kb、酵母染色体III 330kb、M. genitalium 580kb">
    <text x="14" y="44" fontSize="10.5" fill="currentColor">ミトコンドリア</text>
    <rect x="180" y="32" width="10" height="18" rx="2" fill="#EF4444" fillOpacity="0.65" />
    <text x="200" y="46" fontSize="10" fill="currentColor" fillOpacity="0.85">16 kb</text>
    <text x="14" y="92" fontSize="10.5" fill="currentColor">酵母 第III染色体</text>
    <rect x="180" y="80" width="205" height="18" rx="2" fill="#10B981" fillOpacity="0.65" />
    <text x="395" y="94" fontSize="10" fill="currentColor" fillOpacity="0.85">330 kb</text>
    <text x="14" y="140" fontSize="10.5" fill="currentColor">M. genitalium</text>
    <rect x="180" y="128" width="360" height="18" rx="2" fill="#3B82F6" fillOpacity="0.65" />
    <text x="500" y="140" fontSize="10" fill="#fff" textAnchor="end" fontWeight="600">580 kb</text>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>生成したゲノムのスケール（横軸は配列長）。ミトコンドリアから細菌ゲノム規模まで（論文 Fig. 5a に対応）</figcaption>
</figure>

### ミトコンドリアゲノム（16 kb）

ヒトのミトコンドリア DNA の一部をプロンプトに、**16 kb の配列を 250 種類以上** 生成しました。**MitoZ** でアノテーションすると、ヒトミトコンドリアに期待される **CDS・tRNA 遺伝子・rRNA 遺伝子の正しい個数** を備えていました（Fig. 5c）。さらに、

- 自然遺伝子との **synteny（遺伝子の並び順）** を保持（Fig. 5e）。
- 生成タンパク質は **AlphaFold 3** 予測で、ヒトミトコンドリアタンパク質と一致する **多量体複合体** を形成（Fig. 5f）。
- **コドン使用頻度** がヒトミトコンドリアゲノムとよく一致（Extended Data Fig. 9d）。
- 各種類の tRNA を過不足なく生成（プロンプトに含めた2つを重複させずに）。

### 最小細菌ゲノム（M. genitalium, 580 kb）

100 万塩基の文脈を活かし、**最小ゲノムのモデル生物 *Mycoplasma genitalium*（約 580 kb）** の規模に挑戦しました。リファレンスの 10.5 kb をプロンプトに、**580 kb の配列を 10 本** 生成し **Prodigal** でアノテーション。生成遺伝子の **約 70% が有意な Pfam ヒット**（タンパク質ファミリーに一致）を示し、これは **Evo 1（131k 文脈）の 18% からの大幅な改善** です（Fig. 5g,h）。生成タンパク質の長さ・二次構造の分布は天然の *M. genitalium* に似ており、構造アラインメントも取れました（ただし構造予測の信頼度は天然遺伝子よりやや低め）。

### 真核染色体（酵母, 330 kb）

真核生物の生成能力を見るため、*Saccharomyces cerevisiae*（出芽酵母）の **第 III 染色体（約 316 kb）** の 10.5 kb をプロンプトに、**330 kb の配列を 20 本** 生成しました。生成配列には **tRNA・プロモーター・イントロン構造をもつ遺伝子** が含まれます（Fig. 5l）。tRNA や遺伝子の密度は天然より低いものの、遺伝子長分布は自然に近く、系統的近さの指標である **TUD（tetranucleotide usage deviation）** は天然の酵母と相関し、その一致は **40B が 7B より高い** 結果でした。

## 5. 評価と限界

これらの評価は、**MitoZ・Prodigal・GeneMark**（遺伝子予測）、**Pfam/hmmscan**（タンパク質ファミリー）、**AlphaFold 3・ESMFold**（構造予測の pLDDT・TM スコア）、**BLAST**（配列類似性）といった **in silico（計算上）の指標** に基づきます。

:::warning[「もっともらしさ」と「実際に機能すること」は別]

論文は限界を率直に述べています。これらの指標は配列が **ゲノムに似ている** ことを示しますが、**細胞内で機能する・複製できるゲノムであることを保証しません**。生成ゲノムには一部の必須遺伝子が欠けるなどの不足があり、ゲノム規模の設計を実際に検証するには **大規模で反復的な実験** が必要です。とはいえ Evo 2 は、Evo 1 より強力で **真核配列まで生成できる**、ゲノムスケール生成の確かな基盤を提供します。

:::

:::tip[LLM とのつながり：流暢さと正しさのギャップ]

「文章としては流暢だが事実として正しいとは限らない」という自然言語 LLM の課題（ハルシネーション）と、「配列としてはもっともらしいが機能するとは限らない」というゲノム生成の課題は、よく似た構造をしています。生成物の品質を **外部の検証器（ここでは構造予測や実験）で確かめる** という発想は、次ページの設計タスクでさらに重要になります。

:::

## 6. まとめ

- Evo 2 は GPT と同じ **自己回帰生成** で、遺伝子からゲノム規模までの DNA を生成できる。
- **ミトコンドリア（16 kb）・酵母染色体（330 kb）・M. genitalium（580 kb）** を生成し、正しい遺伝子構成・synteny・コドン使用・構造を再現。
- *M. genitalium* の Pfam ヒット率は **Evo 1 の 18% → Evo 2 の 70%** と大きく前進。
- ただし **in silico 指標は機能を保証しない**——実験的検証が次の課題。

次の [06. クロマチン設計と考察](./06-chromatin-design.md) では、ただ生成するのではなく **「狙った機能をもつ配列」を設計** する、推論時ガイダンスの手法を見ます。ここで「外部の検証器で導く」発想が主役になります。
