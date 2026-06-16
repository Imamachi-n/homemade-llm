---
sidebar_position: 4
title: "Evo 2 と AlphaGenome の使い分け"
---

# Evo 2 と AlphaGenome の使い分け

本セクションでは、生物学における LLM 応用の2大事例として [Evo 2](./evo2/01-overview.md) と [AlphaGenome](./alphagenome/01-overview.md) を見てきました。どちらも「生命の配列を計算で扱う」最先端モデルですが、**設計思想がまったく異なります**。

このページでは、「結局どっちが賢いの？」「どう使い分けるの？」という問いに、両論文の内容を踏まえて深く踏み込みます。

:::tip[先に結論]

「どちらが賢いか」を競わせるのは、**包丁とフライパンの優劣を競う** ようなもので、あまり意味がありません。両者は **狙うゴールが違う** からです。Evo 2 は配列を**生成**するモデル、AlphaGenome は配列の**機能を予測**するモデル。むしろ重要なのは、**この2つが相補的に協調しうる** という点で、それは両論文自身が示唆しています（後述）。

:::

## 1. そもそも別のゲームをやっている

まず、2つのモデルの本質的な違いを整理します。

| 観点 | Evo 2 | AlphaGenome |
| --- | --- | --- |
| パラダイム | 自己回帰ゲノム**言語モデル** | **sequence-to-function**（教師あり回帰） |
| 学習の仕方 | 自己教師あり（ラベル不要、生 DNA だけ） | 教師あり（実験測定データが必要） |
| 学習データ | 全生物ドメインの生ゲノム（9.3 兆塩基） | ヒト/マウスの**実験アトラス**（ENCODE・GTEx ほか） |
| 出力 | 配列の尤度・**新規配列の生成** | 機能ゲノミクス・トラック（発現・クロマチン等） |
| カバー範囲 | **全ドメイン**（細菌・古細菌・真核・ファージ） | ヒト・マウスのみ |
| アーキテクチャ | StripedHyena 2（畳み込みハイブリッド） | U-Net ＋ Transformer |
| 得意技 | 進化的制約・配列デザイン・汎用性 | 非コード変異の**機能的影響**を高精度・多モダリティで |

ポイントは、**学習信号（何を教師にするか）** が決定的に違うことです。

- **Evo 2** は「次の塩基を当てる」自己教師ありなので、**ラベルが一切要りません**。地球上のあらゆる生物の DNA をそのまま食べさせれば学習が進みます。
- **AlphaGenome** は「配列 → 実験測定値」の対応を学ぶ教師ありなので、**実験データが必要**です。だからこそ ENCODE や GTEx が整備されたヒト・マウスに範囲が限られます。

この一点が、以降のすべての「賢さの違い」を生み出します。

## 2. 「賢さ」を3つの軸で分解する

「どっちが賢い」を、3つの独立した軸に分けて考えると、それぞれ別の方向に優れていることが見えてきます。

<figure style={{margin: '1.5rem auto', textAlign: 'center', maxWidth: '600px'}}>
  <svg viewBox="0 0 600 280" width="100%" role="img" aria-label="3つの軸での比較。ラベル依存性・出力の直接性・カバレッジで Evo 2 と AlphaGenome がそれぞれ別方向に優れる">
    <text x="300" y="24" fontSize="12.5" fill="currentColor" textAnchor="middle" fontWeight="700">「賢さ」の3軸</text>
    <text x="150" y="58" fontSize="11" fill="currentColor" textAnchor="middle" fontWeight="600">① ラベル依存性</text>
    <rect x="60" y="68" width="180" height="44" rx="5" fill="#10B981" fillOpacity="0.12" stroke="#10B981" strokeWidth="1.4" />
    <text x="150" y="86" fontSize="9.5" fill="currentColor" textAnchor="middle" fontWeight="600">Evo 2 が有利</text>
    <text x="150" y="102" fontSize="8.5" fill="currentColor" fillOpacity="0.8" textAnchor="middle">ラベル不要・どんな生物でも</text>
    <text x="450" y="58" fontSize="11" fill="currentColor" textAnchor="middle" fontWeight="600">② 出力の直接性</text>
    <rect x="360" y="68" width="180" height="44" rx="5" fill="#3B82F6" fillOpacity="0.12" stroke="#3B82F6" strokeWidth="1.4" />
    <text x="450" y="86" fontSize="9.5" fill="currentColor" textAnchor="middle" fontWeight="600">AlphaGenome が有利</text>
    <text x="450" y="102" fontSize="8.5" fill="currentColor" fillOpacity="0.8" textAnchor="middle">機能値を直接・解釈できる</text>
    <text x="150" y="148" fontSize="11" fill="currentColor" textAnchor="middle" fontWeight="600">③ カバレッジ</text>
    <rect x="60" y="158" width="180" height="44" rx="5" fill="#10B981" fillOpacity="0.12" stroke="#10B981" strokeWidth="1.4" />
    <text x="150" y="176" fontSize="9.5" fill="currentColor" textAnchor="middle" fontWeight="600">Evo 2 が広い</text>
    <text x="150" y="192" fontSize="8.5" fill="currentColor" fillOpacity="0.8" textAnchor="middle">全生物ドメイン</text>
    <text x="450" y="148" fontSize="11" fill="currentColor" textAnchor="middle" fontWeight="600">③' 解像度・特異性</text>
    <rect x="360" y="158" width="180" height="44" rx="5" fill="#3B82F6" fillOpacity="0.12" stroke="#3B82F6" strokeWidth="1.4" />
    <text x="450" y="176" fontSize="9.5" fill="currentColor" textAnchor="middle" fontWeight="600">AlphaGenome が深い</text>
    <text x="450" y="192" fontSize="8.5" fill="currentColor" fillOpacity="0.8" textAnchor="middle">細胞型・モダリティ別</text>
    <text x="300" y="240" fontSize="10" fill="currentColor" fillOpacity="0.85" textAnchor="middle">「広く浅く・ラベルなし」の Evo 2 と「狭く深く・ラベルあり」の AlphaGenome</text>
    <text x="300" y="258" fontSize="9.5" fill="currentColor" fillOpacity="0.7" textAnchor="middle">＝ トレードオフの引き受け方が逆方向</text>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>「賢さ」を3軸に分解すると、Evo 2 と AlphaGenome はそれぞれ別の方向に優れている</figcaption>
</figure>

### 軸①：ラベル依存性 → Evo 2 の賢さ

Evo 2 の自己教師あり学習は、**ラベルが要らない** のが最大の武器です。実験で機能を測れていない生物——大半の細菌・古細菌・非モデル真核生物——にも、配列さえあれば適用できます。これは **スケーラビリティとエレガントさ** の賢さです。「人類がまだ機能を測れていない生命」にも手が届きます。

### 軸②：出力の直接性 → AlphaGenome の賢さ

一方 AlphaGenome は、出力が「H3K27ac が増える」「TAL1 の発現が上がる」といった **具体的で解釈可能な機能値** です。Evo 2 の出力（配列の尤度）は、機能的な問いに対して **間接的** です。「この変異は尤度を下げる＝進化的に保存されている＝たぶん重要」とは言えても、「**どの組織で・どの分子メカニズムで**効くか」までは直接言えません。AlphaGenome はそこを直接当てにいきます。これは **実用性・機構解釈** の賢さです。

### 軸③：カバレッジ vs 解像度 → 方向が逆

- **Evo 2** は **横に広い**：全生物ドメインを1つのモデルでカバー。
- **AlphaGenome** は **縦に深い**：ヒト/マウスに限られるが、**細胞型・組織・11 モダリティ** を単一塩基解像度で予測。

:::note[「賢さ」はトレードオフの引き受け方]

結局、両者は同じトレードオフを **逆方向に** 引き受けています。Evo 2 は「ラベルが要らない代わりに出力が間接的」、AlphaGenome は「出力が直接的な代わりにラベル（実験データ）に依存」。**実験データの天井が、そのまま AlphaGenome の天井** になり、**進化が刻んだ情報の限界が、そのまま Evo 2 の限界** になります。どちらが賢いかではなく、**どちらのトレードオフがあなたの問題に合うか** が本質です。

:::

## 3. 深掘り：変異の有害性予測における対比

両者の違いが最も面白く現れるのが、**変異が有害かどうかの予測** です。ここには「進化的制約ベース」と「機能ベース」という、2つの根本的に異なる考え方があります。

<figure style={{margin: '1.5rem auto', textAlign: 'center', maxWidth: '600px'}}>
  <svg viewBox="0 0 600 250" width="100%" role="img" aria-label="変異有害性予測の2つの考え方。進化的制約ベースは保存性で判断し機構には無関心、機能ベースは分子メカニズムを説明する">
    <rect x="30" y="40" width="250" height="170" rx="8" fill="#10B981" fillOpacity="0.07" stroke="#10B981" strokeWidth="1.5" />
    <text x="155" y="64" fontSize="11.5" fill="currentColor" textAnchor="middle" fontWeight="700">進化的制約ベース</text>
    <text x="155" y="80" fontSize="9" fill="currentColor" fillOpacity="0.7" textAnchor="middle">（Evo 2・保存性スコア）</text>
    <text x="50" y="106" fontSize="9.5" fill="currentColor">問い：この位置は進化的に</text>
    <text x="50" y="120" fontSize="9.5" fill="currentColor">　　　保存されているか？</text>
    <text x="50" y="144" fontSize="9.5" fill="#10B981">＋ どんな機構でも検出できる</text>
    <text x="50" y="160" fontSize="9.5" fill="#10B981">＋ ラベル不要・全生物</text>
    <text x="50" y="184" fontSize="9.5" fill="#EF4444">− なぜ有害かは説明しない</text>
    <text x="50" y="200" fontSize="9.5" fill="#EF4444">− 速進化領域の機能獲得に弱い</text>
    <rect x="320" y="40" width="250" height="170" rx="8" fill="#3B82F6" fillOpacity="0.07" stroke="#3B82F6" strokeWidth="1.5" />
    <text x="445" y="64" fontSize="11.5" fill="currentColor" textAnchor="middle" fontWeight="700">機能ベース</text>
    <text x="445" y="80" fontSize="9" fill="currentColor" fillOpacity="0.7" textAnchor="middle">（AlphaGenome）</text>
    <text x="340" y="106" fontSize="9.5" fill="currentColor">問い：この変異は分子機能を</text>
    <text x="340" y="120" fontSize="9.5" fill="currentColor">　　　どう変えるか？</text>
    <text x="340" y="144" fontSize="9.5" fill="#3B82F6">＋ 機構を説明できる</text>
    <text x="340" y="160" fontSize="9.5" fill="#3B82F6">＋ 細胞型・モダリティ別に</text>
    <text x="340" y="184" fontSize="9.5" fill="#EF4444">− 実験データのある範囲のみ</text>
    <text x="340" y="200" fontSize="9.5" fill="#EF4444">− 分子効果 ≠ 表現型</text>
    <text x="300" y="234" fontSize="10" fill="currentColor" fillOpacity="0.85" textAnchor="middle">両者は無関心な点が逆 → 組み合わせると相補的</text>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>変異有害性予測の2つの考え方。「保存されているか」と「機能をどう変えるか」は別の問い</figcaption>
</figure>

### 進化的制約ベース（Evo 2 寄り）

Evo 2 のゼロショット変異効果予測は、本質的に **「進化的にどれだけ保存されているか」** に根ざしています（[Evo 2 の変異効果予測](./evo2/03-variant-prediction.md)）。重要な配列は変異が淘汰されて保存される——だから尤度が下がる変異は有害だろう、という論理です。

- **長所**：**どんな分子メカニズムによる有害性でも** 検出できる（保存性は機構に無関心だから取りこぼしが少ない）。ラベルも要らない。
- **短所**：**なぜ有害か（どの機構で効くか）は説明しません**。また、がんの機能獲得型変異のように **速進化領域で起きる変異** には弱いことがあります。

### 機能ベース（AlphaGenome）

AlphaGenome は **「この変異が分子機能をどう変えるか」** を直接予測します。

- **長所**：**機構を説明できる**。「MYB モチーフが生成され、TAL1 の発現が上がる」というレベルまで踏み込めます（[AlphaGenome の TAL1 解析](./alphagenome/07-multimodal-ablations.md)）。細胞型・モダリティ別に効果を出せます。
- **短所**：**実験データのある範囲（ヒット/マウス・タンパク質コード偏重）でしか効きません**。さらに、分子効果は表現型（疾患そのもの）と同じではありません。

:::tip[論文自身が「相補的」と認めている]

AlphaGenome の論文は、自モデルを **「保存性ベースの有害性スコアを補完する道具」** と明確に位置づけています。保存性スコアは機構に無関心、AlphaGenome は機構を説明できる——だから **両方を併用すると相補的** だ、という主張です。つまり「進化的制約ベース」と「機能ベース」は、対立ではなく **組み合わせるべき2つのレンズ** なのです。

:::

## 4. 使い分けフローチャート

実務でどちらを選ぶか、決定の指針です。

| やりたいこと | 選ぶモデル | 理由 |
| --- | --- | --- |
| 新しい DNA 配列を**生成・設計**したい（合成ゲノム、人工エンハンサー） | **Evo 2** | 生成できるのは Evo 2 のみ |
| ヒトの非コード変異の**機能的影響**を知りたい（臨床・eQTL・スプライシング） | **AlphaGenome** | 機能を直接・多モダリティで予測 |
| **モデル生物でない／実験データが無い**生物を扱う | **Evo 2** | ラベル不要で全ドメイン対応 |
| 特定の**細胞型・組織の分子レベル**の読み出しが欲しい | **AlphaGenome** | 細胞型特異的トラックを出せる |
| 変異の有害性を**進化的保存性**で見たい | **Evo 2** | 進化的制約が本領 |
| 変異の有害性を**分子メカニズム**で説明したい | **AlphaGenome** | 機構解釈が本領 |
| タンパク質構造・プロファージなど**配列内在の特徴**を解釈したい | **Evo 2** | SAE による特徴抽出（[解釈可能性](./evo2/04-interpretability.md)） |
| エンハンサー–遺伝子の**遠位制御リンク**を当てたい | **AlphaGenome** | 1 Mb 文脈で連結を予測 |

:::note[判断の早道]

迷ったら、**「配列を作りたいのか、配列の機能を知りたいのか」** をまず問うてください。

- **作りたい（生成・設計）** → Evo 2
- **知りたい（機能・変異効果）** → AlphaGenome

そのうえで、**ヒト/マウス以外**を扱うなら自動的に Evo 2、**細胞型レベルの機能**が要るなら AlphaGenome、と絞り込めます。

:::

## 5. いちばん賢いのは「組み合わせ」

ここまで見たように、両者は競合ではなく **役割分担** です。最も強力な使い方は、**2つを直列につなぐ** ことです。

<figure style={{margin: '1.5rem auto', textAlign: 'center', maxWidth: '600px'}}>
  <svg viewBox="0 0 580 170" width="100%" role="img" aria-label="生成と評価のループ。Evo 2 で配列を生成し、AlphaGenome で機能を採点し、結果を次の生成にフィードバックする">
    <rect x="30" y="60" width="150" height="54" rx="8" fill="#10B981" fillOpacity="0.12" stroke="#10B981" strokeWidth="1.6" />
    <text x="105" y="84" fontSize="11.5" fill="currentColor" textAnchor="middle" fontWeight="700">Evo 2</text>
    <text x="105" y="101" fontSize="9" fill="currentColor" fillOpacity="0.8" textAnchor="middle">新しい配列を生成</text>
    <line x1="180" y1="87" x2="228" y2="87" stroke="currentColor" strokeOpacity="0.6" strokeWidth="1.8" />
    <polygon points="232,87 222,82 222,92" fill="currentColor" fillOpacity="0.6" />
    <text x="206" y="78" fontSize="8.5" fill="currentColor" fillOpacity="0.7" textAnchor="middle">配列</text>
    <rect x="234" y="60" width="160" height="54" rx="8" fill="#3B82F6" fillOpacity="0.12" stroke="#3B82F6" strokeWidth="1.6" />
    <text x="314" y="84" fontSize="11.5" fill="currentColor" textAnchor="middle" fontWeight="700">AlphaGenome</text>
    <text x="314" y="101" fontSize="9" fill="currentColor" fillOpacity="0.8" textAnchor="middle">機能を採点・評価</text>
    <line x1="394" y1="87" x2="442" y2="87" stroke="currentColor" strokeOpacity="0.6" strokeWidth="1.8" />
    <polygon points="446,87 436,82 436,92" fill="currentColor" fillOpacity="0.6" />
    <rect x="448" y="60" width="110" height="54" rx="8" fill="currentColor" fillOpacity="0.06" stroke="currentColor" strokeOpacity="0.5" strokeWidth="1.4" />
    <text x="503" y="84" fontSize="10" fill="currentColor" textAnchor="middle" fontWeight="600">採用 / 棄却</text>
    <text x="503" y="101" fontSize="8.5" fill="currentColor" fillOpacity="0.8" textAnchor="middle">設計の意思決定</text>
    <path d="M 314 114 Q 314 150 200 150 Q 105 150 105 116" fill="none" stroke="currentColor" strokeOpacity="0.45" strokeWidth="1.4" strokeDasharray="5 4" />
    <polygon points="105,116 100,126 110,126" fill="currentColor" fillOpacity="0.45" />
    <text x="210" y="165" fontSize="9" fill="currentColor" fillOpacity="0.7" textAnchor="middle">採点結果を次の生成にフィードバック</text>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>生成（Evo 2）× 評価（AlphaGenome）のループ。作って、採点して、改善する</figcaption>
</figure>

```text
Evo 2 で新しい配列を生成
   → AlphaGenome でその配列の機能を採点（発現・アクセシビリティ等）
   → 良い配列を採用 / 悪い配列を棄却
   → 結果を次の生成にフィードバック
```

実際、AlphaGenome の論文は Discussion で **「DNA 配列の生成モデルが新しく作った配列の機能的特性を AlphaGenome で予測することで補完しあえる」** と述べ、参考文献として Evo 2 を引いています（[AlphaGenome の考察](./alphagenome/07-multimodal-ablations.md)）。生成は Evo 2、評価は AlphaGenome——という分業は、論文著者自身の見立てでもあるわけです。

:::tip[これは LLM のエージェント設計と同じ構図]

「生成モデルが候補を出し、評価モデル（または報酬モデル）が採点して、良いものを選ぶ」というループは、自然言語 LLM の **生成 × 検証（generator–verifier）** や、強化学習の **方策 × 報酬モデル** とまったく同じ構図です。Evo 2（生成）と AlphaGenome（評価）の組み合わせは、この設計思想をゲノム工学に持ち込んだもの、と理解できます。

:::

## 6. LLM 全体から見た位置づけ

実は、この「自己教師あり汎用モデル vs 教師あり特化モデル」という対比は、**自然言語 LLM の歴史そのもの** と重なります。

| | 自己教師あり・汎用 | 教師あり・特化 |
| --- | --- | --- |
| 自然言語 | GPT 系の基盤モデル（次単語予測） | タスク特化のファインチューニング済みモデル |
| ゲノム | **Evo 2**（次塩基予測・生成） | **AlphaGenome**（機能トラックの教師あり予測） |

自然言語の世界では、**大きな自己教師あり基盤モデル** が汎用性で勝ち、特化タスクもファインチューニングで取り込む方向に進みました。一方ゲノムでは、**機能データが豊富なヒト/マウスの特定タスク** では、AlphaGenome のような教師あり特化アプローチが依然として強い精度を出します。**データの性質（ラベルの入手しやすさ）が、どちらのパラダイムが勝つかを決める** という、機械学習の普遍的な教訓がここにも現れています。

:::note[なぜゲノムでは特化型がまだ強いのか]

自然言語では「次の単語」を当てる自己教師あり学習が、翻訳・要約・QA などの下流タスクにそのまま転移しました。テキストには答えが言語の形で埋め込まれているからです。一方ゲノムでは、「次の塩基」を当てても「この組織でこの遺伝子がどれだけ発現するか」は**直接は出てきません**——それは実験で測るしかない量です。だから機能予測タスクでは、実験データを直接教師にする AlphaGenome が有利になります。ただし Evo 2 のような DNA 言語モデルを**特徴抽出器として活用**する方向（AlphaGenome 論文の将来展望でも言及）も進んでおり、両パラダイムの融合が次の焦点です。

:::

## 7. まとめ

- **「どちらが賢いか」は問いの立て方が惜しい**。両者は狙うゴールが違う（生成 vs 機能予測）。
- 賢さを3軸（ラベル依存性・出力の直接性・カバレッジ）で分解すると、**それぞれ逆方向に優れている**。同じトレードオフを逆向きに引き受けている。
- 変異有害性予測では、**進化的制約ベース（Evo 2）と機能ベース（AlphaGenome）** が相補的。論文自身が併用を推奨。
- 使い分けの早道は **「配列を作りたいか、機能を知りたいか」**。ヒト/マウス外なら Evo 2、細胞型レベルの機能なら AlphaGenome。
- 最強の使い方は **生成（Evo 2）× 評価（AlphaGenome）のループ**。これは LLM の generator–verifier と同じ構図。
- この対比は、**自己教師あり汎用 vs 教師あり特化** という LLM 全体の主題のゲノム版でもある。

2つのモデルを敵対させるのではなく、**それぞれの強みを理解して適材適所で使い、必要なら組み合わせる**——それが、生物学における配列モデル活用の最も賢い姿勢です。
