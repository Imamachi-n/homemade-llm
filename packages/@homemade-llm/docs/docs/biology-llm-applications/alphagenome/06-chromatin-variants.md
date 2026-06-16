---
sidebar_position: 6
title: "06. クロマチン変異と MPRA"
---

# 06. クロマチン変異と MPRA

このページでは、**クロマチン状態**（DNA の開き具合・転写因子の結合）に効く変異の予測を扱います。論文の Fig. 5 に対応し、(1) caQTL/dsQTL/bQTL の予測、(2) ISM によるモチーフ解釈、(3) MPRA（CAGI5）による配列活性の予測を見ていきます。

## 1. 3種類のクロマチン QTL

クロマチンに関わる変異効果は、3つの QTL タイプで評価されます。

| QTL タイプ | 略 | 何に効くか |
| --- | --- | --- |
| chromatin accessibility QTL | **caQTL** | クロマチンアクセシビリティ（ATAC で測る開き具合） |
| DNase sensitivity QTL | **dsQTL** | DNase 感受性（DNase で測る開き具合） |
| transcription factor binding QTL | **bQTL** | 転写因子の結合量（ChIP-seq で測る） |

:::tip[クロマチンの「開き具合」とは]

DNA はヒストンに巻きついて折りたたまれています。転写が活発な制御領域では DNA がほどけて **開いた（アクセシブルな）** 状態になり、転写因子が結合できます。ATAC-seq や DNase-seq は、この「開いている度合い」を測る実験手法です。変異がこの開き具合や転写因子結合を変えれば、下流の遺伝子発現にも波及します。

:::

## 2. スコアリング：センターマスク戦略（Fig. 5a）

クロマチン QTL のスコアリングには **センターマスク（centre-mask）戦略** を使います。変異の周囲の局所ウィンドウで、REF と ALT の予測（DNase や ChIP-seq のシグナル）を比べ、その差分を変異スコアとします（[04](./04-splicing-variants.md)・[05](./05-expression-variants.md) と同じ「ALT − REF」の骨格）。

## 3. ベンチマーク結果（Fig. 5b–d, g）

多様な祖先集団（ancestry）の fine-mapped QTL（ChromBPNet のベンチマーク）で、**Borzoi と専門特化モデル ChromBPNet の両方** と比較しました。

- **QTL causality（Fig. 5b、average precision）** と **効果量（Fig. 5c、Pearson r）** の両方で、QTL タイプ・祖先を横断して一貫して **SOTA**。
- 汎化性も確認：ヨーロッパ系 caQTL、ヨルバ系 DNase 感受性 QTL、ミクログリアや心臓平滑筋細胞など特定細胞型でも有効。
- 効果量の相関：caQTL（アフリカ系）で **Pearson r = 0.74**、SPI1 bQTL で **r = 0.55**（Fig. 5d,g）。

:::note[平均で +8.0%（対 ChromBPNet）]

[01](./01-overview.md) で触れた「アクセシビリティ QTL で ChromBPNet 比 +8.0%（5 データセット平均）」は、この一連の caQTL/dsQTL 評価をまとめた数字です。専門特化モデルである ChromBPNet を、その専門領域で上回っている点が重要です。

:::

## 4. ISM によるモチーフ解釈（Fig. 5e,f,h,i）

変異効果の **メカニズム** を、ISM（[04](./04-splicing-variants.md) 参照）で読み解きます。

- 高影響の変異に ISM を適用すると、予測されたアクセシビリティ/結合の変化は、**既知の転写因子モチーフの改変** に対応していました。
- 例：**NF-κB** などクロマチンアクセシビリティを調節する転写因子のモチーフ（Fig. 5e,f）。
- **SPI1 特異的な bQTL** では、ISM が局所配列中の **SPI1 モチーフの改変** を浮かび上がらせました（Fig. 5h,i）。JASPAR の行列 ID（MA0080.5 など）まで対応づけられています。

:::tip[配列ロゴ（sequence logo）]

ISM の結果は **配列ロゴ** として可視化されます。各位置で、予測に効く塩基ほど大きく描かれ、A/C/G/T の積み重なりがモチーフの「形」を表します。変異前後（REF/ALT）のロゴを比べると、「変異がどの転写因子モチーフを壊した/作った」かが視覚的にわかります。AlphaGenome が当てた効果が、**生物学的に意味のあるモチーフ** に裏づけられていることの確認になります。

:::

## 5. MPRA（CAGI5）：配列の活性そのものを予測（Fig. 5j）

最後に、局所配列の **制御活性そのもの** を測る **MPRA（massively parallel reporter assay, 大規模並列レポーターアッセイ）** での評価です。CAGI5（第5回 Critical Assessment of Genome Interpretation）の飽和変異導入 MPRA チャレンジを使います。

:::tip[MPRA とは]

MPRA は、短い DNA 配列をレポーター遺伝子につないで細胞に大量導入し、**各配列がどれだけ転写を駆動するか（活性）** を一度に測る実験です。変異を網羅的に入れた配列群（飽和変異導入）を測れば、「どの塩基変化が活性をどう変えるか」がわかります。配列の制御活性は、局所のクロマチンアクセシビリティや転写因子結合と密接に関係します。

:::

AlphaGenome を DNase・RNA-seq・ChIP 出力で評価し、Enformer・Borzoi・ChromBPNet と比較しました（Pearson r で評価）。

1. **ゼロショット（細胞型マッチ DNase）** — ChromBPNet・Borzoi Ensemble と同等（Pearson r = 0.57）。
2. **全細胞型の DNase 特徴を LASSO 回帰** — 細胞型マッチや Borzoi Ensemble を上回る（r = 0.63）。
3. **複数モダリティ × 全細胞型を LASSO 統合** — CAGI5 で **SOTA**（r = 0.65）。

:::note[LASSO 回帰での統合]

LASSO（least absolute shrinkage and selection operator）は、不要な特徴量の係数を 0 に潰しながら線形回帰する手法です。AlphaGenome の多数の出力（全細胞型・全モダリティの予測値）を特徴量として LASSO に入れると、有用なものだけが選ばれ、単一の細胞型マッチ予測より性能が上がります。**統合モデルが出す豊富な出力を、下流タスクの特徴量として再利用できる** ことを示しています。

:::

## 6. まとめ

- クロマチン変異は **caQTL/dsQTL/bQTL** の3タイプで評価。**センターマスク戦略**（ALT−REF の局所差）でスコアリング。
- causality・効果量とも、Borzoi と専門特化 ChromBPNet の両方を **QTL タイプ・祖先横断で上回る**。
- **ISM** が、予測された効果を **NF-κB・SPI1 など既知の転写因子モチーフ** の改変として裏づけ。
- **MPRA（CAGI5）** では、アクセシビリティ予測を特徴量に LASSO 統合して **SOTA**。

次の [07. 多モダリティ統合・アブレーション・考察](./07-multimodal-ablations.md) では、ここまでの全モダリティを束ねた **TAL1 がん変異の解釈**、設計選択の寄与を分解した **アブレーション**、そして論文の **Discussion**（限界と展望）を見ていきます。
