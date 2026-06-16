---
sidebar_position: 4
title: "04. スプライシング変異の予測"
---

# 04. スプライシング変異の予測

ここから変異効果予測に入ります。最初の主役は **スプライシング** です。論文の「Improved splicing variant predictions」と Fig. 3 を、AlphaGenome の新機構 **スプライスジャンクション予測** を軸に詳しく見ていきます。

## 1. なぜスプライシングが重要なのか

スプライシングは、**前駆 mRNA からイントロンを除去し、エクソンをつなぎ合わせて成熟 RNA を作る** 過程です。遺伝病の主要な原因の1つがこのスプライシングの乱れであり、変異の機能解釈で外せないモダリティです。

スプライシングの結果は、3つのレベルでモデル化できます。

<figure style={{margin: '1.5rem auto', textAlign: 'center', maxWidth: '600px'}}>
  <svg viewBox="0 0 580 220" width="100%" role="img" aria-label="スプライシングの3レベル。スプライス部位の確率、部位の競合的使用率、どのイントロンが除去されるかのジャンクション">
    <rect x="40" y="40" width="90" height="26" rx="4" fill="#3B82F6" fillOpacity="0.18" stroke="#3B82F6" strokeWidth="1.3" />
    <text x="85" y="58" fontSize="10" fill="currentColor" textAnchor="middle">エクソン1</text>
    <rect x="245" y="40" width="90" height="26" rx="4" fill="#3B82F6" fillOpacity="0.18" stroke="#3B82F6" strokeWidth="1.3" />
    <text x="290" y="58" fontSize="10" fill="currentColor" textAnchor="middle">エクソン2</text>
    <rect x="450" y="40" width="90" height="26" rx="4" fill="#3B82F6" fillOpacity="0.18" stroke="#3B82F6" strokeWidth="1.3" />
    <text x="495" y="58" fontSize="10" fill="currentColor" textAnchor="middle">エクソン3</text>
    <line x1="130" y1="53" x2="245" y2="53" stroke="currentColor" strokeOpacity="0.4" strokeWidth="1.2" />
    <line x1="335" y1="53" x2="450" y2="53" stroke="currentColor" strokeOpacity="0.4" strokeWidth="1.2" />
    <text x="187" y="44" fontSize="8.5" fill="currentColor" fillOpacity="0.7" textAnchor="middle">イントロン</text>
    <circle cx="130" cy="53" r="4" fill="#EF4444" /><circle cx="245" cy="53" r="4" fill="#10B981" />
    <text x="300" y="100" fontSize="10.5" fill="currentColor" textAnchor="middle" fontWeight="600">3つの予測レベル</text>
    <text x="60" y="130" fontSize="9.5" fill="#EF4444">① スプライス部位</text>
    <text x="60" y="145" fontSize="8.5" fill="currentColor" fillOpacity="0.8">各塩基がドナー/アクセプターである確率</text>
    <text x="60" y="168" fontSize="9.5" fill="#F59E0B">② スプライス部位使用率（SSU）</text>
    <text x="60" y="183" fontSize="8.5" fill="currentColor" fillOpacity="0.8">候補部位どうしの競合的な使われ方</text>
    <text x="60" y="206" fontSize="9.5" fill="#10B981">③ スプライスジャンクション（新規）</text>
    <text x="60" y="221" fontSize="8.5" fill="currentColor" fillOpacity="0.8">どのドナーとどのアクセプターがつながるか</text>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>スプライシングの3つの予測レベル（論文 Fig. 3a に対応）。AlphaGenome はこの3つに加え RNA-seq 読み取り深さも直接予測する</figcaption>
</figure>

1. **スプライス部位予測** — ある塩基がスプライスドナー/アクセプターとして働く確率（SpliceAI などが扱う）。
2. **スプライス部位使用率（SSU）予測** — 候補となる複数のスプライス部位の **競合的な選択**（Pangolin などが扱う）。
3. **スプライスジャンクション予測** — 具体的に **どのイントロンが除去されるか**（＝どのドナーとアクセプターがつながるか）。

**AlphaGenome はこの3つすべてを、RNA-seq 読み取り深さの予測と並行して出力します。** 特に③のジャンクション予測は新しい貢献です。

:::note[既存モデルは「部分的」だった]

SpliceAI はスプライス部位の予測に特化し、ジャンクション予測や部位間の競合は扱いません。Pangolin は使用率を扱いますが、やはりジャンクション予測は欠けています。Borzoi は RNA-seq 被覆から暗黙的にスプライス部位を推定するだけ（32 bp 解像度）。AlphaGenome は **明示的に・1 bp 解像度で・3レベルすべて** を出す点が新しいのです（Fig. 3a）。

:::

## 2. 既知の生物学を再現できるか（Fig. 3b–d）

新しいスコアラーを作る前に、論文はまず **既知の生物学的帰結を AlphaGenome が再現できるか** を確かめます。

- **エクソンスキッピング（Fig. 3b）** — 第3染色体の 4 bp 欠失（chr3:197081044: TACTC>T）は、GTEx の脛骨動脈組織でエクソンスキッピングを起こすことが知られています。AlphaGenome は **全レベルで** これを再現しました：当該エクソンのスプライス部位使用率の大幅低下、スキップされるエクソン端をつなぐジャンクションの消失、エクソンを迂回する新しいジャンクションの出現、エクソンの RNA-seq 被覆の強い減少。
- **新規スプライシングドナーの生成（Fig. 3c）** — COL6A2 の変異（大動脈組織）が新しいスプライスドナーを作り、既存のものを壊す様子を再現。
- **ISM によるモチーフの可視化（Fig. 3d）** — U2SURP 遺伝子のエクソン9とその周辺で **in silico mutagenesis（ISM）** を行うと、分岐点（branch point）・ポリピリミジン鎖・アクセプターモチーフ（AG）・ドナーモチーフ（GT）など、**既知のスプライシング関連モチーフ** が浮かび上がりました。

:::tip[ISM（in silico mutagenesis）とは]

ISM は「**配列のある領域の各塩基を、あらゆる別の塩基に置き換えてみて、予測がどれだけ変わるかを総当たりで調べる**」手法です。「ここを変えると予測が大きく動く」位置が、モデルが重要視している配列モチーフを表します。自然言語 LLM の「入力をマスク/摂動して帰属を測る」感度解析（saliency / occlusion）と発想は同じで、AlphaGenome では配列モチーフの可視化と解釈に多用されます（[06](./06-chromatin-variants.md)・[07](./07-multimodal-ablations.md) でも登場）。

:::

## 3. 複合スコアラー：4つの予測を1つの変異スコアに（Fig. 3e）

スプライス変異を体系的に検出するため、論文は **各モダリティごとに変異スコアを設計し、それらを足し合わせた複合スコア（composite score）** を作りました。

変異スコアの基本的な作り方は、**参照（REF）配列と変異（ALT）配列の予測を比べる** ことです。

1. REF 配列と ALT 配列をそれぞれモデルに通し、スプライス部位/ジャンクションの予測を得る。
2. その差分（ALT − REF）を取り、絶対値（ABS）の最大（MAX）を変異スコアとする。

つまり「変異によってスプライス予測が **最も大きく変わった量**」を、その変異の影響度とみなします。これを各モダリティで計算し、合算して複合スコアにします。

:::note[「差分の最大」で効果を測る発想]

配列モデルで変異効果を測る一般的なレシピは、**「ALT の予測 − REF の予測」** です。AlphaGenome は、スプライス部位やジャンクションの予測差のうち最大のものを取ることで、「どこか1か所でも大きくスプライシングが変われば影響あり」と捉えます。次ページ以降の発現変異（[05](./05-expression-variants.md)）・クロマチン変異（[06](./06-chromatin-variants.md)）でも、スコアの作り方は違えど「REF と ALT の予測差」という骨格は共通です。

:::

## 4. ベンチマーク結果（Fig. 3f–i）

複合スコアラーを、多様なスプライシング変異タスクで既存手法と比較します。比較対象は SpliceAI・Pangolin・DeltaSplice・AbSplice・Borzoi など。

| ベンチマーク | 内容 | AlphaGenome の結果 |
| --- | --- | --- |
| **sQTL 分類（Fig. 3f）** | fine-mapped スプライシング QTL の判別 | **最良**（10 kb 以内・200 bp 以内の両設定） |
| **スプライシング外れ値（Fig. 3g）** | GTEx の希少変異によるスプライシング異常 | ゼロショット・教師ありの **両方で最高** |
| **ClinVar 病原性分類（Fig. 3h）** | 病原性 vs 良性の判別 | 3カテゴリすべてで既存最良を上回る |
| **MFASS（Fig. 3i）** | MPRA で検証されたスプライス破壊変異 | Pangolin にやや劣るが SpliceAI/DeltaSplice を上回る |

ClinVar の3カテゴリの詳細（Fig. 3h、auPRC）：

- **深部イントロン＋同義変異**（スプライス部位から離れた変異）：**0.66**（Pangolin 0.64）。
- **スプライス領域**（イントロン側 6 bp・エクソン側 3 bp 以内）：**0.57**（Pangolin 0.55）。
- **ミスセンス**（AlphaMissense で「おそらく良性」とされたもの）：**0.18**（DeltaSplice/Pangolin 0.16）。

:::tip[auPRC という指標]

auPRC は「適合率–再現率曲線（Precision–Recall curve）の下の面積」です。陽性（病原性変異など）が少ない不均衡なデータで、分類器の良し悪しを測るのに適しています。1.0 が完璧で、ランダム分類器の値はデータ中の陽性割合に等しくなります。スプライス変異・QTL の評価で繰り返し登場します。

:::

### ジャンクション・スコアラー単独の強さ

注目すべきは、**スプライスジャンクション・スコアラー単独でも**、「深部イントロン＋同義変異」の ClinVar と MFASS を除く **すべてのベンチマークで既存手法を上回った** ことです。これは、**ジャンクションレベルでスプライシングをモデル化することの重要性** を裏づけます——「どの部位が使われるか」だけでなく「**どの部位とどの部位がつながるか**」まで見ることが効く、ということです。

総合すると、AlphaGenome は **7 ベンチマーク中 6 で SOTA** のスプライス変異効果予測を達成し、変化したスプライシング事象と転写産物構造を、より包括的に捉えられるようになりました。

## 5. まとめ

- スプライシングを **部位・使用率・ジャンクション** の3レベル＋RNA-seq で予測。特に **ジャンクション予測** が新規貢献。
- 既知の生物学（エクソンスキッピング・新規ドナー生成）を **全レベルで再現**。ISM でスプライシングモチーフを可視化。
- **「REF と ALT の予測差の最大」** を各モダリティで取り、合算した **複合スコア** で変異を評価。
- **7 ベンチマーク中 6 で SOTA**。ジャンクション・スコアラー単独でもほとんどのタスクで既存最良を上回る。

次の [05. 発現・遠位制御の変異](./05-expression-variants.md) では、遺伝子発現に効く変異（eQTL）、エンハンサー–遺伝子連結、ポリアデニル化変異（paQTL）を見ていきます。
