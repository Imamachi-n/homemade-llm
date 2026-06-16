---
sidebar_position: 2
title: "02. アーキテクチャと学習"
---

# 02. アーキテクチャと学習

このページでは、AlphaGenome を支える土台——**モデルアーキテクチャ（U-Net ＋ Transformer）**、**配列並列による塩基解像度学習**、**事前学習と蒸留の2段階学習**——を、LLM の技術と対応づけながら詳しく見ていきます。

## 1. 設計課題：1 Mb を 1 bp で出すには

[01](./01-overview.md) で見たように、AlphaGenome の挑戦は **「1 Mb の長い入力」と「1 bp の細かい出力」を同時に成立させる** ことです。これには2つの矛盾する要請を1つのネットワークで満たす必要があります。

- **局所的な細かいパターン** — スプライス部位・転写因子フットプリント・ポリアデニル化部位などは、わずか数塩基の並びで決まります。これを捉えるには **塩基レベルの解像度** が要ります。
- **長距離の依存関係** — エンハンサーとプロモーターは数十〜数百 kb 離れることがあります。これを捉えるには **広い文脈** が要ります。

AlphaGenome はこれを、**畳み込み（局所）と Transformer（長距離）を役割分担させる U-Net 風アーキテクチャ** で解決します。

## 2. U-Net 風バックボーン：縮めて、混ぜて、戻す

全体構造は、医用画像セグメンテーションで有名な **U-Net**（Ronneberger et al., 2015）に着想を得たエンコーダ–デコーダ型です（Fig. 1a, Extended Data Fig. 1a）。

<figure style={{margin: '1.5rem auto', textAlign: 'center', maxWidth: '620px'}}>
  <svg viewBox="0 0 600 340" width="100%" role="img" aria-label="U-Net 風アーキテクチャ。エンコーダで 1bp から 128bp へダウンサンプリング、中央の Transformer タワーで長距離依存を処理、デコーダで 1bp へ戻す。スキップ接続でエンコーダの解像度をデコーダに渡す">
    <text x="300" y="22" fontSize="12" fill="currentColor" textAnchor="middle" fontWeight="600">入力 DNA 1 Mb（1 bp 解像度）</text>
    <rect x="40" y="44" width="120" height="30" rx="4" fill="#3B82F6" fillOpacity="0.14" stroke="#3B82F6" strokeWidth="1.4" />
    <text x="100" y="64" fontSize="10.5" fill="currentColor" textAnchor="middle">Encoder（畳み込み）</text>
    <text x="100" y="92" fontSize="9" fill="currentColor" fillOpacity="0.75" textAnchor="middle">1 bp → 128 bp に縮小</text>
    <line x1="160" y1="59" x2="218" y2="120" stroke="currentColor" strokeOpacity="0.5" strokeWidth="1.4" />
    <polygon points="221,123 211,121 216,113" fill="currentColor" fillOpacity="0.5" />
    <rect x="210" y="110" width="180" height="60" rx="6" fill="#8B5CF6" fillOpacity="0.14" stroke="#8B5CF6" strokeWidth="1.6" />
    <text x="300" y="134" fontSize="11" fill="currentColor" textAnchor="middle" fontWeight="600">Transformer タワー</text>
    <text x="300" y="151" fontSize="9" fill="currentColor" fillOpacity="0.8" textAnchor="middle">128 bp 解像度・デバイス間通信</text>
    <text x="300" y="164" fontSize="9" fill="currentColor" fillOpacity="0.8" textAnchor="middle">長距離（エンハンサー–プロモーター）</text>
    <line x1="382" y1="120" x2="440" y2="59" stroke="currentColor" strokeOpacity="0.5" strokeWidth="1.4" />
    <polygon points="443,56 433,58 438,66" fill="currentColor" fillOpacity="0.5" />
    <rect x="440" y="44" width="120" height="30" rx="4" fill="#10B981" fillOpacity="0.14" stroke="#10B981" strokeWidth="1.4" />
    <text x="500" y="64" fontSize="10.5" fill="currentColor" textAnchor="middle">Decoder（畳み込み）</text>
    <text x="500" y="92" fontSize="9" fill="currentColor" fillOpacity="0.75" textAnchor="middle">128 bp → 1 bp に復元</text>
    <line x1="160" y1="50" x2="440" y2="50" stroke="currentColor" strokeOpacity="0.35" strokeWidth="1.2" strokeDasharray="5 4" />
    <text x="300" y="44" fontSize="8.5" fill="currentColor" fillOpacity="0.6" textAnchor="middle">スキップ接続</text>
    <line x1="500" y1="74" x2="500" y2="210" stroke="currentColor" strokeOpacity="0.5" strokeWidth="1.4" />
    <polygon points="500,214 496,205 504,205" fill="currentColor" fillOpacity="0.5" />
    <rect x="350" y="214" width="300" height="40" rx="5" fill="currentColor" fillOpacity="0.05" stroke="currentColor" strokeOpacity="0.4" strokeWidth="1.2" />
    <text x="500" y="231" fontSize="10" fill="currentColor" textAnchor="middle">1D 埋め込み（1 bp / 128 bp）</text>
    <text x="500" y="246" fontSize="8.5" fill="currentColor" fillOpacity="0.75" textAnchor="middle">→ 各トラックは線形変換で予測</text>
    <rect x="210" y="214" width="120" height="40" rx="5" fill="currentColor" fillOpacity="0.05" stroke="currentColor" strokeOpacity="0.4" strokeWidth="1.2" />
    <text x="270" y="231" fontSize="10" fill="currentColor" textAnchor="middle">2D 埋め込み</text>
    <text x="270" y="246" fontSize="8.5" fill="currentColor" fillOpacity="0.75" textAnchor="middle">→ 接触マップ（2,048 bp）</text>
    <line x1="300" y1="170" x2="270" y2="212" stroke="currentColor" strokeOpacity="0.45" strokeWidth="1.2" />
    <text x="300" y="290" fontSize="10.5" fill="currentColor" textAnchor="middle">出力ヘッド（モダリティごと・各解像度）</text>
    <text x="300" y="312" fontSize="9.5" fill="currentColor" fillOpacity="0.7" textAnchor="middle">RNA-seq / DNase / ヒストン ChIP / スプライス / 接触マップ …</text>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>U-Net 風バックボーン（論文 Fig. 1a / Extended Data Fig. 1 に対応）。畳み込みで縮小→Transformer で長距離を混ぜる→畳み込みで復元。スキップ接続が細かい解像度の情報をデコーダに渡す</figcaption>
</figure>

処理の流れは「縮めて、混ぜて、戻す」の3段です。

1. **エンコーダ（畳み込み＋max pooling）** — 入力配列を **1 bp から 128 bp へ段階的にダウンサンプリング** し、チャンネル数を増やします。畳み込みは **局所的な配列パターン** のモデル化を担います。
2. **Transformer タワー（128 bp 解像度）** — 縮小された表現に対して Self-Attention を適用し、**エンハンサー–プロモーター相互作用のような長距離依存** を捉えます。後述するデバイス間通信もここで効きます。
3. **デコーダ（畳み込み＋アップサンプリング）** — 解像度を **128 bp から 1 bp へ復元**。このとき **スキップ接続（skip connection）** でエンコーダの対応する段から細かい解像度の情報を引き込み、塩基レベルの精度を保ちます。

:::note[なぜ畳み込みと Transformer を分業させるのか]

Transformer の Self-Attention は計算量が系列長の二乗（$O(n^2)$）なので、1 Mb をそのまま 1 bp 解像度で Attention にかけるのは非現実的です。そこで AlphaGenome は、**重い Attention を「128 bp に縮小した後」だけに限定** します（系列長が約 1/128 になる）。塩基レベルの細かい処理は安価な畳み込みに任せ、Attention は長距離の混合に専念させる——この役割分担が、長文脈と高解像度の両立を可能にしています。Evo 2 が Attention 自体を Hyena 演算子で置き換えた（[Evo 2 のアーキテクチャ](../evo2/02-architecture.md)）のとは対照的なアプローチです。

:::

### 1D 埋め込みと 2D 埋め込み

エンコーダ/デコーダが生み出す表現は2種類あります。

- **1次元（1D）埋め込み**（1 bp / 128 bp 解像度）— 線形なゲノムの表現。**ほとんどのトラック予測の土台** で、各トラックの値は基本的にこの埋め込みの **線形変換** で得ます。
- **2次元（2D）埋め込み**（2,048 bp 解像度）— ゲノム上の2領域 $(i, j)$ の **空間的相互作用** を表す表現。**接触マップ（contact map）** の予測に使います。接触マップはペアワイズの行列なので、2次元の表現が必要になります。

### スプライスジャンクション専用ヘッド

トラック予測は原則「埋め込みの線形変換」ですが、**スプライスジャンクション数の予測だけは別の仕組み** を使います。これは、ドナー部位とアクセプター部位の **1D 埋め込みのペア間の相互作用** を捉える専用機構です（Extended Data Fig. 1）。「どのドナーとどのアクセプターがつながってイントロンが除去されるか」は本質的にペアの問題なので、専用設計になっています（詳細は [04](./04-splicing-variants.md)）。

## 3. 配列並列：8 台の TPU で 1 Mb を 1 bp 学習する

1 Mb の配列を 1 bp 解像度のまま扱うと、中間表現が巨大になり1台のアクセラレータに乗りません。AlphaGenome は **配列並列（sequence parallelism）** でこれを解決します。

- 1 Mb の配列を **131 kb のチャンク** に分割し、**相互接続された 8 台の TPU（v3）** に分散して処理します。
- Transformer タワーでは、**デバイス間通信** を行って各チャンクが他チャンクの情報を参照できるようにします。これにより、分割していても **1 Mb 全体にわたる Attention** が成立します。

:::tip[LLM とのつながり：シーケンス並列]

「長い系列を複数デバイスに分割し、Attention のときだけ通信して全体をつなぐ」のは、大規模 LLM の長文脈学習で使われる **sequence parallelism / context parallelism** とまさに同じ発想です。AlphaGenome はこの分散学習技術を、ゲノムという超長系列に適用して 1 bp 解像度の学習を可能にしています。

:::

## 4. 2段階学習：事前学習 → 蒸留

AlphaGenome は **事前学習（pretraining）** と **蒸留（distillation）** の2段階で訓練されます（Fig. 1b,c）。

<figure style={{margin: '1.5rem auto', textAlign: 'center', maxWidth: '620px'}}>
  <svg viewBox="0 0 600 250" width="100%" role="img" aria-label="2段階学習。事前学習で fold 別モデルと all-fold 教師モデルを作り、蒸留で all-fold 教師の予測を再現する単一の生徒モデルを得る">
    <rect x="30" y="40" width="250" height="150" rx="8" fill="#3B82F6" fillOpacity="0.07" stroke="#3B82F6" strokeWidth="1.5" />
    <text x="155" y="62" fontSize="12" fill="currentColor" textAnchor="middle" fontWeight="700">1. 事前学習 Pretraining</text>
    <rect x="50" y="78" width="210" height="42" rx="5" fill="#3B82F6" fillOpacity="0.12" stroke="#3B82F6" strokeWidth="1.2" />
    <text x="155" y="95" fontSize="10" fill="currentColor" textAnchor="middle">fold 別モデル（4分割交差検証）</text>
    <text x="155" y="110" fontSize="8.5" fill="currentColor" fillOpacity="0.75" textAnchor="middle">3/4 で学習・1/4 で検証 → 汎化を評価</text>
    <rect x="50" y="130" width="210" height="42" rx="5" fill="#3B82F6" fillOpacity="0.12" stroke="#3B82F6" strokeWidth="1.2" />
    <text x="155" y="147" fontSize="10" fill="currentColor" textAnchor="middle">all-fold 教師モデル</text>
    <text x="155" y="162" fontSize="8.5" fill="currentColor" fillOpacity="0.75" textAnchor="middle">全領域で学習 → 蒸留の教師に</text>
    <line x1="282" y1="151" x2="318" y2="151" stroke="currentColor" strokeOpacity="0.6" strokeWidth="1.8" />
    <polygon points="322,151 312,146 312,156" fill="currentColor" fillOpacity="0.6" />
    <rect x="324" y="40" width="250" height="150" rx="8" fill="#10B981" fillOpacity="0.07" stroke="#10B981" strokeWidth="1.5" />
    <text x="449" y="62" fontSize="12" fill="currentColor" textAnchor="middle" fontWeight="700">2. 蒸留 Distillation</text>
    <text x="449" y="86" fontSize="10" fill="currentColor" textAnchor="middle">教師アンサンブル（all-fold）</text>
    <line x1="449" y1="92" x2="449" y2="116" stroke="currentColor" strokeOpacity="0.55" strokeWidth="1.4" />
    <polygon points="449,120 444,111 454,111" fill="currentColor" fillOpacity="0.55" />
    <rect x="364" y="120" width="170" height="40" rx="5" fill="#10B981" fillOpacity="0.14" stroke="#10B981" strokeWidth="1.3" />
    <text x="449" y="137" fontSize="10.5" fill="currentColor" textAnchor="middle" fontWeight="600">単一の生徒モデル</text>
    <text x="449" y="151" fontSize="8.5" fill="currentColor" fillOpacity="0.8" textAnchor="middle">入力を増強・変異させて教師を再現</text>
    <text x="449" y="178" fontSize="9" fill="currentColor" fillOpacity="0.85" textAnchor="middle">→ 変異効果予測に使う本番モデル</text>
    <text x="300" y="218" fontSize="10" fill="currentColor" fillOpacity="0.75" textAnchor="middle">生徒モデルは H100 1 台で 1 変異あたり 1 秒未満（デバイス呼び出し1回）</text>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>2段階学習（論文 Fig. 1b,c に対応）。事前学習で fold 別／all-fold モデルを作り、all-fold 教師アンサンブルを単一の生徒モデルに蒸留する</figcaption>
</figure>

### 第1段階：事前学習

観測された実験データを教師信号に、**2種類のモデル** を作ります。

- **fold 別モデル（fold-specific）** — **4分割交差検証** で、参照ゲノムの 3/4 を学習に、残り 1/4 を検証・テストに使います。「学習時に見ていないゲノム領域」での **汎化性能を評価する** ために使われます（→ [03](./03-track-prediction.md) のトラック評価）。
- **all-fold 教師モデル** — 利用可能な全領域で学習したモデル。次の蒸留段階で **教師（teacher）** として働きます。

学習入力は、1 Mb 区間をサンプリングし、**ランダムシフト**・**逆相補鎖化（50%）** といった **データ増強（augmentation）** を施したものです。

### 第2段階：蒸留

事前学習と同じアーキテクチャをもつ **単一の生徒モデル（student）** を、**all-fold 教師アンサンブルの出力を再現するように** 訓練します。このとき入力には、**ランダムな増強と変異（mutation）を加えた配列** を使います。

蒸留の狙いと効果：

- **頑健性と変異効果予測精度の向上** — 先行研究（Zhou et al., 2024）と同様、蒸留により単一モデルでも頑健性と変異効果予測の精度が上がります。
- **推論の超高速化** — 生徒モデルは **全モダリティ・全細胞型を、1 変異あたりデバイス呼び出し1回** で予測します。**NVIDIA H100 1 台で 1 秒未満**。複数の独立学習モデルをアンサンブルする方式に比べ、大規模な変異スクリーニングで圧倒的に効率的です。

:::note[入力への変異が蒸留のカギ]

[07](./07-multimodal-ablations.md) のアブレーションで詳しく見ますが、**蒸留時に入力配列をランダムに変異させないと** 生徒モデルの性能が落ちます（eQTL sign で −0.06、causality で −0.01 など）。「教師が参照配列でどう振る舞うか」だけでなく「**変異させた配列で教師がどう振る舞うか**」まで生徒に教え込むことが、変異効果予測の精度に直結する、というわけです。変異効果予測が本番タスクなのだから、蒸留時にも変異を見せる——理にかなった設計です。

:::

:::tip[LLM とのつながり：知識蒸留とアンサンブル代替]

「複数の大きな教師モデルのアンサンブルを、1つの軽い生徒モデルに蒸留する（knowledge distillation）」のは、自然言語 LLM でも標準的な圧縮・高速化手法です。アンサンブルは強力ですが推論コストが重い——その性能を保ったまま1モデルに畳み込むのが蒸留です。AlphaGenome は、**1 変異 1 秒未満** という実用的な推論速度を、この蒸留で実現しています。

:::

## 5. 学習データ：ヒトとマウスの実験アトラス

AlphaGenome の教師信号は、**公開された実験データ** です。Evo 2 が生 DNA 配列だけで自己教師あり学習するのに対し、AlphaGenome は **「配列 → 実験測定値」の対応** を教師あり学習します。

| データソース | 提供するもの |
| --- | --- |
| **ENCODE** | DNase / ATAC / ヒストン・TF の ChIP-seq など |
| **GTEx** | 多組織の RNA-seq・スプライシング・QTL |
| **FANTOM5** | CAGE（転写開始） |
| **4D Nucleome** | Hi-C / Micro-C（接触マップ） |
| **ClinVar・gnomAD** | 変異の病原性アノテーション・頻度（評価用） |

出力トラック数は **ヒト 5,930 / マウス 1,128**。多様な組織・細胞型・細胞株にわたります（メタデータは論文 Supplementary Table 1, 2）。

:::tip[「種」を入力に与える]

AlphaGenome は DNA 配列に加えて **種の識別子（ヒット/マウス）** も入力に取ります。これにより、1つのモデルで両種のトラックを予測でき、種をまたいだ共有表現を学べます。

:::

## 6. まとめ

- **U-Net 風バックボーン**：畳み込み（局所・1 bp）で縮小 → **Transformer**（長距離・128 bp）で混合 → 畳み込みで 1 bp に復元。スキップ接続で解像度を維持。
- **1D 埋め込み**（トラック予測）と **2D 埋め込み**（接触マップ）、さらに **スプライスジャンクション専用ヘッド**。
- **配列並列**：1 Mb を 131 kb チャンクに分け **8 台の TPU** で塩基解像度学習。
- **2段階学習**：事前学習（fold 別＋all-fold 教師）→ **蒸留**（変異を加えた入力で教師を再現する単一生徒モデル）。H100 1 台で **1 変異 1 秒未満**。
- 教師信号は **ENCODE・GTEx・FANTOM5・4D Nucleome** などの実験アトラス。

次の [03. トラック予測性能](./03-track-prediction.md) では、このモデルが「見たことのないゲノム領域」でどれだけ正確にトラックを当てられるか——変異効果予測の前提となる基礎性能——を見ていきます。
