---
sidebar_position: 4
title: "04. 機構的解釈可能性（SAE）"
---

# 04. 機構的解釈可能性（SAE）

[03. 変異効果予測](./03-variant-prediction.md)で、Evo 2 が変異の影響を正確に予測できることを見ました。では、モデルは内部で **何を「理解」しているのでしょうか**。このページでは、LLM 研究で発展した **機構的解釈可能性（mechanistic interpretability）** の手法を使って、Evo 2 の"頭の中"を覗きます。

## 1. なぜ解釈するのか：ブラックボックス批判への回答

大規模言語モデルはしばしば「ブラックボックス（中で何が起きているか分からない）」と批判されます。しかし近年、**スパースオートエンコーダ（sparse autoencoder, SAE）** を使うと、モデル内部に **意味のある特徴（feature）** が見つかることが分かってきました。

:::tip[LLM とのつながり：Anthropic の Claude 解釈研究と同じ手法]

「モデルの中間表現を SAE で分解して、人間が解釈できる特徴を取り出す」というアプローチは、Anthropic が **Claude 3 Sonnet** に対して行った研究（"Scaling Monosemanticity", Transformer Circuits, 2024）で大きく注目されました。自然言語の Claude で「金門橋の特徴」のような単義的な特徴が見つかったのと同じ手法を、Evo 2 は **ゲノム** に適用しています。論文も Cunningham et al. や Anthropic の monosemanticity 研究（参考文献29–31）を直接引いています。

:::

ポイントは **単義性（monosemanticity）** です。モデルの個々のニューロンは複数の概念に反応する **多義的（polysemantic）** で読みにくいのですが、SAE はこれを **「1つの特徴＝1つの概念」** に近い形へほどいてくれます。

## 2. スパースオートエンコーダ（SAE）の仕組み

SAE は、モデルの中間表現（活性, activation）を入力として、**疎な（ほとんどがゼロの）高次元ベクトル** に符号化し、それを元の表現に再構成するように学習されます。発火する少数の次元それぞれが、解釈可能な特徴に対応します。

<figure style={{margin: '1.5rem auto', textAlign: 'center', maxWidth: '620px'}}>
  <svg viewBox="0 0 600 210" width="100%" role="img" aria-label="SAE は Evo 2 の多義的な活性を、単義的で解釈可能な特徴に分解する">
    <rect x="18" y="60" width="150" height="84" rx="6" fill="currentColor" fillOpacity="0.06" stroke="currentColor" strokeOpacity="0.6" strokeWidth="1.4" />
    <text x="93" y="48" fontSize="11" fill="currentColor" textAnchor="middle">Evo 2 の活性（layer 26）</text>
    <text x="93" y="96" fontSize="10.5" fill="currentColor" fillOpacity="0.85" textAnchor="middle">多義的で</text>
    <text x="93" y="112" fontSize="10.5" fill="currentColor" fillOpacity="0.85" textAnchor="middle">そのままでは読めない</text>
    <line x1="168" y1="102" x2="206" y2="102" stroke="currentColor" strokeOpacity="0.5" strokeWidth="1.4" />
    <polygon points="210,102 201,97 201,107" fill="currentColor" fillOpacity="0.5" />
    <text x="188" y="92" fontSize="9" fill="currentColor" fillOpacity="0.7" textAnchor="middle">SAE</text>
    <rect x="214" y="60" width="120" height="84" rx="6" fill="#8B5CF6" fillOpacity="0.08" stroke="#8B5CF6" strokeWidth="1.4" />
    <text x="274" y="48" fontSize="11" fill="currentColor" textAnchor="middle">疎な特徴</text>
    <rect x="230" y="72" width="88" height="9" fill="#8B5CF6" fillOpacity="0.15" />
    <rect x="230" y="84" width="88" height="9" fill="#3B82F6" fillOpacity="0.7" />
    <rect x="230" y="96" width="88" height="9" fill="#8B5CF6" fillOpacity="0.15" />
    <rect x="230" y="108" width="88" height="9" fill="#10B981" fillOpacity="0.7" />
    <rect x="230" y="120" width="88" height="9" fill="#8B5CF6" fillOpacity="0.15" />
    <rect x="230" y="132" width="88" height="9" fill="#EF4444" fillOpacity="0.7" />
    <text x="274" y="158" fontSize="9" fill="currentColor" fillOpacity="0.7" textAnchor="middle">少数だけ発火（色付き）</text>
    <line x1="334" y1="102" x2="372" y2="102" stroke="currentColor" strokeOpacity="0.5" strokeWidth="1.4" />
    <polygon points="376,102 367,97 367,107" fill="currentColor" fillOpacity="0.5" />
    <rect x="380" y="44" width="202" height="120" rx="6" fill="currentColor" fillOpacity="0.04" stroke="currentColor" strokeOpacity="0.5" strokeWidth="1.3" />
    <text x="481" y="38" fontSize="11" fill="currentColor" textAnchor="middle">解釈可能な特徴の例</text>
    <circle cx="394" cy="62" r="4" fill="#3B82F6" /><text x="404" y="66" fontSize="10" fill="currentColor">プロファージ領域</text>
    <circle cx="394" cy="84" r="4" fill="#10B981" /><text x="404" y="88" fontSize="10" fill="currentColor">エクソン／イントロン境界</text>
    <circle cx="394" cy="106" r="4" fill="#EF4444" /><text x="404" y="110" fontSize="10" fill="currentColor">α-ヘリックス／β-シート</text>
    <circle cx="394" cy="128" r="4" fill="#8B5CF6" /><text x="404" y="132" fontSize="10" fill="currentColor">転写因子結合モチーフ</text>
    <circle cx="394" cy="150" r="4" fill="#3B82F6" /><text x="404" y="154" fontSize="10" fill="currentColor">tRNA／rRNA・フレームシフト</text>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>SAE は Evo 2 の多義的な活性を、単義的で解釈可能な特徴に分解する（論文 Fig. 4a に対応）</figcaption>
</figure>

論文では、Evo 2 の **layer 26** の表現に対して **Batch-TopK SAE** を学習しました。学習データは複数の真核・原核ゲノムから均等にサンプリングした **10 億トークン** です。

:::note[Batch-TopK SAE の中身]

SAE は活性 $\mathbf{h}$ を疎なコード $\mathbf{z}$ に符号化し、再構成 $\hat{\mathbf{h}}$ を出します。

$$
\mathbf{z} = \mathrm{TopK}\bigl(\mathrm{ReLU}(W_{\text{enc}}(\mathbf{h}-\mathbf{b}))\bigr), \qquad \hat{\mathbf{h}} = W_{\text{dec}}\,\mathbf{z} + \mathbf{b}
$$

学習目標は「再構成誤差を小さくしつつ、発火する特徴を少数に保つ（疎性）」ことです。**Batch-TopK** は、各トークン個別ではなく **バッチ全体で上位 K 個** の特徴だけを残す変種で、安定して単義的な特徴が得られます。$W_{\text{dec}}$ の各列が1つの特徴の「意味の方向」に対応します。

:::

## 3. 特徴を生物学的概念に対応づける

学習された特徴（SAE の各次元）が何を意味するのかは、**contrastive feature search（対照特徴探索）** で突き止めます。これは「ある注釈（例：プロファージ領域）を含む配列セグメントで、特に強く発火する特徴を探す」手法です（Fig. 4 / Extended Data Fig. 7a）。こうして見つかった特徴を、論文では `f/19746` のように **`f/` ＋通し番号** で表記します。

## 4. 発見された特徴のギャラリー

Evo 2 が **生物学的な注釈を一切与えられずに** 学んでいた特徴は、DNA・RNA・タンパク質・組織構造の各レベルにわたります。

### DNA レベル：可動性遺伝因子と遺伝子構造

- **プロファージ特徴（`f/19746`）** — 細菌ゲノムに潜むファージ由来領域で発火。E. coli K12 のプロファージ（cryptic prophage **CPZ-55** を含む）に反応します。興味深いのは、**CRISPR アレイのスペーサー配列**（過去に取り込んだファージ DNA の断片）でも発火する点です。しかも、スペーサーを **スクランブル（並べ替え）しても発火** したことから、モデルはファージ配列を **丸暗記しているのではなく**、「ファージらしさ」という概念を **関連付けて** いると分かります。geNomad（専用ツール）がファージと認識しない領域でも、インテグラーゼやインベルターゼを含む箇所で発火しました。
- **ORF・遺伝子間・tRNA・rRNA 特徴** — オープンリーディングフレーム（ORF）、遺伝子間領域、tRNA、rRNA に対応する特徴も見つかりました。

### タンパク質レベル：二次構造

DNA を学習しただけのモデルが、**タンパク質の立体構造**に対応する特徴まで持っていました。**α-ヘリックス**（`f/28741`）と **β-シート**（`f/22326`）に対応する特徴です。論文では、これらの特徴の活性を **AlphaFold 3** が予測した立体構造（EF-Tu と tRNA の複合体、RpoB–RpoC 複合体）に重ね合わせ、構造要素と一致することを示しています（Fig. 4d）。

:::tip[これは「マルチモーダル」な理解]

DNA 配列だけを入力に学習したのに、その内部表現は **RNA（tRNA/rRNA）やタンパク質の高次構造（α/β）** にまで及んでいました。セントラルドグマ全体を貫く情報が、DNA という単一のモダリティの学習から立ち上がっている——ゲノム言語モデリングの奥深さを示す結果です。

:::

### 変異感受性・制御・遺伝子構造（ヒトゲノム）

ヒトゲノムに解析を広げると、さらに多様な特徴が見つかりました。

- **フレームシフト特徴（`f/24278`）** — フレームシフトや早期終止のような **重篤な変異** で優先的に発火（影響の小さい変異タイプより強く反応）。
- **転写因子結合モチーフ** — ヒトプロモーター配列で、既知の転写因子（TF）結合部位に酷似した DNA モチーフに反応する特徴。ランダムに選んだヒトプロモーターで、Evo 2 の教師なし SAE 特徴は **HOCOMOCO v12 CORE** データベースの promoter-enriched モチーフの **70%** にヒットしました（TOMTOM ツール, q &lt; 0.01）。比較として、専用のモチーフ発見アルゴリズム **HOMER** は同じモチーフの **35%** しか recall しません。
- **エクソン／イントロン境界** — コード領域（`f/15680`）、イントロン（`f/28339`）、イントロン後のエクソンの最初の塩基（`f/1050`）、イントロン前のエクソンの最後の塩基（`f/25666`）に対応する特徴。コード領域特徴は **細菌の ORF でも発火** し、コード配列の普遍的な表現を学んでいることを示します。

:::note[高次の配列依存性を捉えている]

エクソン境界の特徴（`f/1050`, `f/25666`）は **複数塩基にまたがるスプライス部位** の信号を統合し、プロファージ特徴（`f/19746`）は **キロベース規模の文脈** を必要とします。これは、Evo 2 が単なる局所パターンではなく、**長距離の高次な配列依存性** を学んでいる証拠です（[02](./02-architecture.md) の長文脈設計が効いています）。

:::

### 種を超える転移：ウーリーマンモス

これらの特徴は、霊長類・マウス・アフリカツメガエル・ショウジョウバエといった **モデル生物だけ** で学習した SAE から得られたものです。それにもかかわらず、同じ特徴が **ウーリーマンモス** のゲノム領域にも転移して機能しました（Fig. 4g）。SAE が学んだ特徴が **種を超えて一般化** し、ゲノムアノテーションに使える可能性を示します。

## 5. まとめ

- Evo 2 は、**生物学的なラベルを与えられずに**、プロファージ・ORF・tRNA/rRNA・α/β 構造・転写因子モチーフ・エクソン境界といった **多彩な生物学的特徴** を内部に獲得していた。
- これらは LLM 研究と同じ **SAE × 単義性** の枠組みで取り出され、Anthropic の Claude 解釈研究と地続きの成果である。
- 特徴は **丸暗記ではなく概念の関連付け**（CRISPR スペーサー）であり、**長距離依存** を捉え、**種を超えて転移** する。
- 論文は **SAE モデルと可視化ツール（Evo Mech Interp Visualizer）** を公開し、コミュニティが特徴を探索できるようにしている。

「理解」しているモデルは、当然「創り出す」こともできます。次の [05. ゲノムスケール生成](./05-generation.md) では、Evo 2 が新しい DNA 配列を生成する能力を見ていきます。
