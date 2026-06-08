---
sidebar_position: 3
title: "03. 変異効果予測"
---

# 03. 変異効果予測

このページは、論文の2つのセクション——**「Evo 2 が進化的制約を学ぶ」** と **「ヒト変異効果予測」**——を統合して解説します。Evo 2 の予測能力の核心であり、**追加学習なし（ゼロショット）で変異の影響を当てる** 仕組みと、それが **BRCA1 のような臨床的に重要な遺伝子** にまで通用することを見ていきます。

## 1. ゼロショット予測の原理：尤度の変化を見る

Evo 2 は「次の塩基を予測する」ように学習されています。その副産物として、モデルは任意の配列に対して **尤度（likelihood）**＝「その配列が生命の世界でどれだけ"自然"か」を数値で出せます。

[02. アーキテクチャ](./02-architecture.md)の自己回帰の式を思い出すと、配列 $x_{1:n}$ の対数尤度は次のように書けます。

$$
\log p(x_{1:n}) = \sum_{i=1}^{n} \log p(x_i \mid x_1, \dots, x_{i-1})
$$

ここで重要な発想は、**変異（塩基の変化）の前後で尤度がどう変わるかを見る** ことです。野生型（wild-type, WT）配列と、変異を入れた配列の対数尤度の差を取ります。

$$
\Delta = \log p(x^{\text{mut}}) - \log p(x^{\text{WT}})
$$

- $\Delta$ が **大きく負**（尤度が下がる）→ その変異は「不自然」＝ **進化的に許されにくい＝有害** な可能性が高い。
- $\Delta$ が **ほぼ 0**（尤度が変わらない）→ 機能に影響しにくい中立的な変異。

これが **ゼロショット変異効果予測** です。タスク専用のラベル付きデータで学習し直す必要はなく、事前学習済みの尤度だけで予測できます。

<figure style={{margin: '1.5rem auto', textAlign: 'center', maxWidth: '560px'}}>
  <svg viewBox="0 0 560 200" width="100%" role="img" aria-label="変異前後で Evo 2 の尤度を比較し、その差で有害性を判定する">
    <text x="80" y="34" fontSize="11" fill="currentColor" textAnchor="middle">野生型 (WT)</text>
    <rect x="22" y="44" width="116" height="26" rx="4" fill="#10B981" fillOpacity="0.12" stroke="#10B981" strokeWidth="1.3" />
    <text x="80" y="62" fontSize="11" fill="currentColor" textAnchor="middle">…A T G C G T…</text>
    <text x="80" y="150" fontSize="11" fill="currentColor" textAnchor="middle">変異 (mutant)</text>
    <rect x="22" y="118" width="116" height="26" rx="4" fill="#EF4444" fillOpacity="0.12" stroke="#EF4444" strokeWidth="1.3" />
    <text x="80" y="136" fontSize="11" fill="currentColor" textAnchor="middle">…A T G <tspan fill="#EF4444" fontWeight="700">A</tspan> G T…</text>
    <line x1="142" y1="57" x2="206" y2="80" stroke="currentColor" strokeOpacity="0.5" strokeWidth="1.3" />
    <line x1="142" y1="131" x2="206" y2="108" stroke="currentColor" strokeOpacity="0.5" strokeWidth="1.3" />
    <rect x="208" y="68" width="96" height="52" rx="6" fill="currentColor" fillOpacity="0.06" stroke="currentColor" strokeOpacity="0.7" strokeWidth="1.6" />
    <text x="256" y="92" fontSize="14" fill="currentColor" textAnchor="middle" fontWeight="700">Evo 2</text>
    <text x="256" y="109" fontSize="9" fill="currentColor" fillOpacity="0.8" textAnchor="middle">対数尤度を計算</text>
    <line x1="304" y1="94" x2="350" y2="94" stroke="currentColor" strokeOpacity="0.5" strokeWidth="1.3" />
    <polygon points="354,94 345,89 345,99" fill="currentColor" fillOpacity="0.5" />
    <text x="360" y="60" fontSize="10.5" fill="#10B981">log p(WT)</text>
    <rect x="360" y="68" width="120" height="12" fill="#10B981" fillOpacity="0.6" />
    <text x="360" y="104" fontSize="10.5" fill="#EF4444">log p(mut)</text>
    <rect x="360" y="112" width="72" height="12" fill="#EF4444" fillOpacity="0.55" />
    <text x="490" y="98" fontSize="13" fill="currentColor" textAnchor="middle">Δ</text>
    <text x="445" y="152" fontSize="10.5" fill="currentColor" textAnchor="middle" fontStyle="italic">Δ が負に大きいほど「有害」と判定</text>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>ゼロショット変異効果予測。変異前後の尤度差 Δ で、その変異の有害性を見積もる（論文 Fig. 2a, 3a に対応）</figcaption>
</figure>

:::note[評価指標の読み方：AUROC と AUPRC]

このページには **AUROC** と **AUPRC** が頻出します。どちらも「有害 vs 無害」のような2クラス分類の良さを 0〜1 で測る指標です。

- **AUROC**（ROC 曲線の下の面積）：真陽性率（TPR）と偽陽性率（FPR）のトレードオフを表す。**0.5 がランダム、1.0 が完璧**。
- **AUPRC**（適合率–再現率曲線の下の面積）：陽性が少数の **不均衡データ** で特に有用。病的変異のように「当てたいクラスが希少」な場面で効く。

<figure style={{margin: '1rem auto', textAlign: 'center', maxWidth: '240px'}}>
  <svg viewBox="0 0 200 180" width="100%" role="img" aria-label="ROC 曲線。対角線がランダム、左上に膨らむほど良い分類器">
    <line x1="34" y1="150" x2="180" y2="150" stroke="currentColor" strokeOpacity="0.3" strokeWidth="1" />
    <line x1="40" y1="20" x2="40" y2="156" stroke="currentColor" strokeOpacity="0.3" strokeWidth="1" />
    <text x="170" y="166" fontSize="9.5" fill="currentColor" fillOpacity="0.7">FPR</text>
    <text x="14" y="28" fontSize="9.5" fill="currentColor" fillOpacity="0.7">TPR</text>
    <line x1="40" y1="150" x2="170" y2="20" stroke="currentColor" strokeOpacity="0.4" strokeWidth="1.2" strokeDasharray="4 3" />
    <path d="M 40 150 C 60 70, 110 38, 170 20" fill="none" stroke="#3B82F6" strokeWidth="2.4" />
    <text x="120" y="100" fontSize="9.5" fill="#3B82F6">良い分類器</text>
    <text x="92" y="128" fontSize="8.5" fill="currentColor" fillOpacity="0.6" transform="rotate(-32 92 128)">ランダム (0.5)</text>
  </svg>
  <figcaption style={{fontSize: '0.8rem', marginTop: '0.3rem', opacity: 0.85}}>ROC 曲線が左上に膨らむほど AUROC が 1 に近づく</figcaption>
</figure>

:::

## 2. Evo 2 が捉えた進化的制約

Evo 2 の尤度が本当に生物学的な制約を反映しているのか、論文は数々の証拠を示します（Fig. 2b–e）。

### コドンの3塩基周期性

タンパク質をコードする領域では、塩基3つ（コドン）が1つのアミノ酸に対応します。Evo 2 に各位置で変異を入れて尤度変化を測ると、**3塩基ごとの周期的なパターン** が現れました。コドンの1・2番目の塩基はアミノ酸を強く決めるので変異の影響が大きく、3番目（**ゆらぎ＝wobble 位置**）は同じアミノ酸になりやすいので影響が小さい——この周期性をモデルが学んでいます。

<figure style={{margin: '1.5rem auto', textAlign: 'center', maxWidth: '520px'}}>
  <svg viewBox="0 0 520 170" width="100%" role="img" aria-label="コドンの3塩基周期性。1・2番目の塩基変異は影響大、3番目は影響小">
    <line x1="40" y1="120" x2="500" y2="120" stroke="currentColor" strokeOpacity="0.3" strokeWidth="1" />
    <line x1="46" y1="25" x2="46" y2="126" stroke="currentColor" strokeOpacity="0.3" strokeWidth="1" />
    <text x="6" y="40" fontSize="9" fill="currentColor" fillOpacity="0.7">変異の</text>
    <text x="6" y="52" fontSize="9" fill="currentColor" fillOpacity="0.7">影響大</text>
    <text x="500" y="135" fontSize="9.5" fill="currentColor" fillOpacity="0.7" textAnchor="end">配列上の位置</text>
    <polyline points="56,45 78,48 100,108 122,46 144,50 166,108 188,44 210,49 232,110 254,47 276,51 298,109 320,45 342,50 364,108 386,46 408,52 430,110" fill="none" stroke="#3B82F6" strokeWidth="2.2" />
    <text x="56" y="142" fontSize="9" fill="currentColor" fillOpacity="0.6">1 2</text>
    <text x="96" y="142" fontSize="9" fill="#EF4444" fillOpacity="0.85">3</text>
    <text x="140" y="142" fontSize="9" fill="currentColor" fillOpacity="0.6">1 2</text>
    <text x="162" y="142" fontSize="9" fill="#EF4444" fillOpacity="0.85">3</text>
    <text x="350" y="40" fontSize="9.5" fill="#EF4444">谷＝3番目（wobble）は影響小</text>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>コドンの3塩基周期性。3番目（wobble 位置）の変異は影響が小さく、谷になる</figcaption>
</figure>

### 翻訳開始シグナル・RNA・遺伝暗号の違い

- **開始/終止コドンと翻訳シグナル** — 開始コドン周辺の変異は尤度を大きく下げます。さらに、コード領域の上流に、原核生物では **Shine–Dalgarno 配列**、真核生物では **Kozak 配列**（いずれも翻訳開始に関わる既知の配列）に対応するパターンが現れました。
- **コーディング vs 非コーディング** — 非同義置換・早期終止・フレームシフトは、同義置換よりはるかに大きく尤度を下げます。非コード領域でも、**tRNA・rRNA の欠失** は遺伝子間領域の欠失より大きな影響を受け、これらの RNA が必須であることと一致します。40B モデルは miRNA・snoRNA の欠失に対し 7B より高感度でした。
- **遺伝暗号の違いを文脈から判別** — Evo 2 は、標準コード（終止＝TAA/TAG/TGA）、**マイコプラズマのコード4**（終止＝TAA/TAG）、**繊毛虫のコード6**（終止＝TGA）の違いを学んでいました。繊毛虫ゲノムを人工的に標準コードに書き換えると、Evo 2 は標準の終止コドンへの変異を有害と予測する——つまり **配列の文脈から適切な遺伝暗号を判断** しています。

:::tip[LLM とのつながり：「文法」を教師なしで学ぶ]

自然言語 LLM が、品詞や構文を明示的に教わらずとも文法的な規則性を獲得するのと同じように、Evo 2 は **コドンの読み枠・翻訳シグナル・種ごとの遺伝暗号** といった「生命の文法」を、ラベルなしの配列だけから学びました。尤度がこれらの規則を反映しているからこそ、ゼロショット予測が成立します。

:::

## 3. 機能との相関：Deep Mutational Scanning

尤度が「進化的に許されるか」を反映するのは分かりました。では、実際の **分子機能（フィットネス）** とも相関するのでしょうか。これを確かめるのが **Deep Mutational Scanning（DMS）**——多数の変異体を実験的に作り、機能を網羅測定したデータです。

Evo 2 の尤度は、**9 種の原核タンパク質・6 種の真核タンパク質・7 種の RNA（rRNA, tRNA, リボザイム）** の DMS データと相関しました（Fig. 2f）。タンパク質 DMS では専用の **ProGen** 言語モデル、ncRNA DMS では RNA 言語モデルと **競争力ある性能** です。ただしタンパク質 DMS では最先端モデルには及ばず、また自然言語の傾向と同様、**最大スケールで性能が飽和・低下** する現象も観測されました。

なお、学習から除外したヒトウイルスのタンパク質では **相関が見られず**（Extended Data Fig. 2b）、これは [02](./02-architecture.md) のデータ除外が意図通りに効いている証拠です。

## 4. 埋め込みの活用：エクソン分類と遺伝子必須性

尤度（ゼロショット）だけでなく、Evo 2 の **埋め込み（embedding）** も強力な特徴量になります。

- **エクソン分類** — Evo 2 7B の埋め込みを入力に、**単一ヌクレオチド解像度のエクソン分類器**（軽量モデル）を訓練しました。学習に使っていない8種の生物で **AUROC 0.91–0.99** を達成し、Nucleotide Transformer や Evo 1 の埋め込み、保存度指標（GC 含量・PhyloP）を上回りました。ab initio 遺伝子予測の **AUGUSTUS** や、専用モデル **SegmentNT**（訓練外の種で）も上回ります。
- **遺伝子必須性** — 早期終止コドンを挿入したときの尤度低下を使って、遺伝子が必須かどうかを予測。細菌・古細菌・ファージで Evo 1 並み・他手法以上の性能でした。ヒト遺伝子必須性では Evo 2 40B が AUROC 0.66（他のゲノム LM の 0.50–0.59 を上回るが、絶対値は控えめ）。

:::note[ゼロショットと埋め込みベースの違い]

- **ゼロショット**：モデルの尤度をそのまま使う。追加学習ゼロ。
- **埋め込みベース**：Evo 2 の中間表現を特徴量として取り出し、その上に **軽量な分類器** を別途学習する。少量のラベルで高精度を狙える。

後者は「基盤モデルを特徴抽出器として使う」典型的な転移学習で、後述の BRCA1 でも効果を発揮します。

:::

## 5. ヒト臨床変異の予測

ここからが臨床応用です。**変異効果予測は遺伝病診断や創薬に直結する重要課題** ですが、従来のゲノム言語モデルは、複数種の配列アラインメント（MSA）を使う専用モデルに **大きく水をあけられていました**。Evo 2 はこの状況を変えます。

評価では多様なベースラインと比較されました。

| 種別 | モデル例 |
| --- | --- |
| 保存度スコア | PhyloP（100/241/447/470 種版） |
| タンパク質言語モデル | ESM-1b, ESM-2 |
| スプライシング予測（教師あり） | SpliceAI, Pangolin |
| 変異効果予測 | AlphaMissense, GPN-MSA, CADD |
| DNA/RNA 言語モデル | Nucleotide Transformer, RNA-FM, Evo 1 |

### ClinVar での結果

臨床変異データベース **ClinVar** の病的/良性ラベルで評価しました（Fig. 3b,c）。

| 変異クラス | サンプル数 | Evo 2 の立ち位置 |
| --- | --- | --- |
| コード SNV | 14,319 | 競争力あり（ESM-2 超え、ESM-1b/GPN-MSA には及ばず） |
| コード non-SNV（挿入・欠失） | 1,236 | **全手法を上回る**（AlphaMissense 等はそもそもスコア不能） |
| 非コード SNV | 34,761 | 教師なしモデルで **1 位**（教師ありにのみ劣る） |
| 非コード non-SNV | 3,894 | **全手法を上回る** |

特筆すべきは **non-SNV（挿入・欠失）** での圧勝です。AlphaMissense や GPN-MSA はアミノ酸置換しか扱えずスコアを出せませんが、Evo 2 は配列の尤度を見るだけなので挿入・欠失も自然に扱えます。

### スプライシング変異（SpliceVarDB）

実験検証済みのスプライシング変異 DB **SpliceVarDB** では、エクソン領域（1,181 変異）・イントロン領域（3,769 変異）ともに、Evo 2 40B/7B が **教師なしモデルで 1 位** でした（Fig. 3d）。イントロン変異では、教師あり専用モデル SpliceAI・CADD に僅差まで迫り、Pangolin は上回ります。

## 6. BRCA1 / BRCA2：がん関連遺伝子の変異判定

臨床的に最も重要なテストが、乳がん・卵巣がんに関わる **BRCA1** 遺伝子です。論文では、**飽和変異導入（saturation mutagenesis）**——ほぼ全ての可能な変異を実験的に作り機能を測定したデータ——を使って、機能喪失（loss-of-function, LOF）かどうかを当てさせました（Fig. 3e,f）。

- コード SNV（2,077 変異）で高性能、**非コード SNV（1,125 変異）では全モデルを上回り** ました。
- コード・非コードを合わせて評価しても他モデル超え（well-calibrated な予測）。
- スプライス部位の近く・遠く別に見ても、非コード変異で Evo 2 40B が教師ありスプライシング予測器すら上回ります。
- 関連遺伝子 **BRCA2** でも、GPN-MSA を超え、教師あり CADD に次ぐ2位。

Arc Institute の解説では「90% 以上の精度で良性と病的変異を区別できる」と紹介された能力です。

## 7. 教師あり埋め込みでさらに精度を上げる

ゼロショットは「専用の訓練データが無くても使える」点で価値がありますが、ラベルがあるなら **埋め込み＋軽量分類器** でさらに伸ばせます。論文では、BRCA1 変異だけで **リッジ回帰**（ridge regression）を Evo 2 埋め込みの上に学習しました（Fig. 3g–i）。

LLM の各層が異なる特徴を捉えることを踏まえ、40B の各ブロックから埋め込みを抽出して最適な層を探索。結果、LOF 変異とそれ以外を明確に分離し、テストセットで **AUROC 0.95・AUPRC 0.88** を達成、Evo 2 40B 単体のゼロショット予測を上回りました（ゼロショット $P = 2.2\times10^{-274}$、教師あり $P = 4.7\times10^{-69}$、いずれも Wilcoxon 検定）。

<figure style={{margin: '1.5rem auto', textAlign: 'center', maxWidth: '560px'}}>
  <svg viewBox="0 0 560 130" width="100%" role="img" aria-label="教師あり埋め込みのパイプライン。配列を Evo 2 に通し、埋め込みを抽出して軽量分類器を学習">
    <rect x="10" y="48" width="96" height="40" rx="5" fill="currentColor" fillOpacity="0.06" stroke="currentColor" strokeOpacity="0.6" strokeWidth="1.3" />
    <text x="58" y="66" fontSize="10.5" fill="currentColor" textAnchor="middle">変異を含む</text>
    <text x="58" y="80" fontSize="10.5" fill="currentColor" textAnchor="middle">DNA 配列</text>
    <line x1="106" y1="68" x2="134" y2="68" stroke="currentColor" strokeOpacity="0.5" strokeWidth="1.3" />
    <polygon points="138,68 129,63 129,73" fill="currentColor" fillOpacity="0.5" />
    <rect x="140" y="48" width="86" height="40" rx="5" fill="currentColor" fillOpacity="0.06" stroke="currentColor" strokeOpacity="0.7" strokeWidth="1.6" />
    <text x="183" y="72" fontSize="13" fill="currentColor" textAnchor="middle" fontWeight="700">Evo 2</text>
    <line x1="226" y1="68" x2="254" y2="68" stroke="currentColor" strokeOpacity="0.5" strokeWidth="1.3" />
    <polygon points="258,68 249,63 249,73" fill="currentColor" fillOpacity="0.5" />
    <rect x="260" y="48" width="110" height="40" rx="5" fill="#8B5CF6" fillOpacity="0.12" stroke="#8B5CF6" strokeWidth="1.4" />
    <text x="315" y="66" fontSize="10.5" fill="currentColor" textAnchor="middle">各層の埋め込み</text>
    <text x="315" y="80" fontSize="10.5" fill="currentColor" textAnchor="middle">を抽出・連結</text>
    <line x1="370" y1="68" x2="398" y2="68" stroke="currentColor" strokeOpacity="0.5" strokeWidth="1.3" />
    <polygon points="402,68 393,63 393,73" fill="currentColor" fillOpacity="0.5" />
    <rect x="404" y="48" width="104" height="40" rx="5" fill="#10B981" fillOpacity="0.12" stroke="#10B981" strokeWidth="1.4" />
    <text x="456" y="66" fontSize="10.5" fill="currentColor" textAnchor="middle">リッジ回帰</text>
    <text x="456" y="80" fontSize="10.5" fill="currentColor" textAnchor="middle">（軽量分類器）</text>
    <text x="456" y="112" fontSize="10" fill="currentColor" fillOpacity="0.75" textAnchor="middle">LOF? → AUROC 0.95</text>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>埋め込みベースの教師あり分類。基盤モデルを特徴抽出器として使い、軽量分類器で臨床タスクに特化させる</figcaption>
</figure>

:::info[非コード制御領域は今後の課題]

遺伝子から離れた制御領域（エンハンサーなど）は保存度が低く、予測が難しい領域です。**DART-eval** ベンチマークでは、Evo 2 40B は chromatin accessibility QTL（caQTL）で AUROC 0.58、DNase I QTL（dsQTL）で 0.66 と、Nucleotide Transformer（0.52 / 0.61）を上回るものの、配列→機能を直接学習した専用モデル **ChromBPNet**（0.77 / 0.89）には及びませんでした。配列だけから制御機能を捉える難しさが残ります。

:::

## 8. まとめ

- Evo 2 は **尤度の変化（Δ）** だけで、追加学習なしに変異の有害性を予測できる（ゼロショット）。
- その尤度は、**コドン周期性・翻訳シグナル・遺伝暗号の違い・必須 RNA** といった進化的制約を正しく反映している。
- **ClinVar・SpliceVarDB・BRCA1/BRCA2** で、従来のゲノム LM を大きく上回り、特に **挿入・欠失（non-SNV）** で全手法に勝つ。
- **埋め込み＋軽量分類器** でさらに精度を上げられ、BRCA1 で AUROC 0.95。
- しかも Evo 2 は **ヒトの遺伝的変異データを一切学習していない**——汎用の生命言語モデルとしての強さを示す。

予測ができるなら、次は「モデルは内部で何を理解しているのか？」が気になります。次の [04. 機構的解釈可能性](./04-interpretability.md) では、SAE を使って Evo 2 の"頭の中"を覗きます。
