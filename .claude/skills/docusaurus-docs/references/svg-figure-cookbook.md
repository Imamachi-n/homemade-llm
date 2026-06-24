# インライン SVG 図テンプレート集（＝組み合わせて使う部品集）

Docusaurus（MDX）にそのまま貼れる SVG 図の**部品集**。座標と色・ラベルを調整して使う。
毎回ゼロから描くより速く、JSX 由来のエラーも避けられる。

**これは「当てはめるテンプレ」ではなく「組み合わせる部品」**として使うこと。図を描く前に、まず
SKILL.md「5. インライン SVG 図解」の手順で **「この図で何を1つ伝えるか」→「その核心に最適な形」** を決め、
その形を下の部品（座標軸・矢印マーカー・グリッド・棒・曲線・弧）で組み立てる。下の7種に当てはまらない図
——ボックス＋矢印のフロー、2列を線で結ぶ対応図、分布の並置比較、まったく新しい図——も、これらの部品を
組み合わせれば作れる。**発想は自由に、実装は実証済みの部品で。** 有用な新パターンを作ったらここに追記して育てる。

> ⚠️ **共通の落とし穴**：SVG の `<text>` の中に `$...$` の数式を書かないこと。remark-math が `<span>` に変換し、
> SVG 内に紛れてビルド警告（`span in a foreign namespace context` / `Stray end tag "text"`）になる。
> SVG 内はプレーン文字（`W_O`、`d_k`、`√dₖ` など）で書き、本物の数式ラベルは `<figcaption>` 側に出す。

## 目次

- [共通ルール（必読）](#共通ルール必読)
- [1. 座標軸 + 点 / ベクトル](#1-座標軸--点--ベクトル)
- [2. 単位円と角度 θ →（cosθ, sinθ）](#2-単位円と角度-θ-cosθ-sinθ)
- [3. 直角三角形（2点間の距離）](#3-直角三角形2点間の距離)
- [4. 2ベクトルと「なす角」（類似度・回転）](#4-2ベクトルとなす角類似度回転)
- [5. 関数の曲線（polyline）＋接線](#5-関数の曲線polyline接線)
- [6. 棒グラフ（変換 before → after）](#6-棒グラフ変換-before--after)
- [7. グリッド（行列・テンソル・行列積）](#7-グリッド行列テンソル行列積)

---

## 共通ルール（必読）

- **JSX 作法**：属性は camelCase（`strokeWidth` `strokeDasharray` `textAnchor` `fontSize`
  `fillOpacity` `strokeOpacity`）、`style` はオブジェクト `style={{...}}`、タグは必ず閉じる、
  HTML コメント不可。
- **座標系**：SVG の y は**下向き**。数学の「上が正」を描くときは `y = 中心 − 値` で反転する。
  各テンプレは反転済みの座標を入れてある。
- **テーマ追従**：軸・補助線・文字は `currentColor`（必要に応じ `strokeOpacity`/`fillOpacity` で淡く）。
  強調はアクセント色 青 `#3B82F6` / 赤 `#EF4444` / 緑 `#10B981`。
- **体裁**：`<figure style={{margin:'1rem auto', textAlign:'center', maxWidth:'...'}}>` で包み、
  `<figcaption style={{fontSize:'0.82rem', marginTop:'0.3rem', opacity:0.85}}>` を付ける。
- 横並びは `<div style={{display:'flex', flexWrap:'wrap', gap:'1.5rem', justifyContent:'center', alignItems:'flex-end'}}>` でラップ。
- 角度の弧は `<path d="M x1 y1 A r r 0 0 0 x2 y2" .../>`。曲がる向きが逆なら sweep フラグ（`0`/`1`）を入れ替える。

---

## 1. 座標軸 + 点 / ベクトル

原点 O から伸びるベクトルと点。`v` を別の座標に変えるだけで流用可。

```jsx
<figure style={{margin: '1rem auto', textAlign: 'center', maxWidth: '240px'}}>
  <svg viewBox="0 0 200 160" width="220" role="img" aria-label="ベクトル v">
    <line x1="20" y1="120" x2="185" y2="120" stroke="currentColor" strokeOpacity="0.25" strokeWidth="1" />
    <line x1="45" y1="20" x2="45" y2="140" stroke="currentColor" strokeOpacity="0.25" strokeWidth="1" />
    <text x="187" y="124" fontSize="10" fill="currentColor" fillOpacity="0.6">x</text>
    <line x1="45" y1="120" x2="150" y2="55" stroke="#3B82F6" strokeWidth="2.4" />
    <circle cx="150" cy="55" r="3.2" fill="#3B82F6" />
    <text x="154" y="52" fontSize="12" fill="#3B82F6" fontStyle="italic">v</text>
    <text x="33" y="132" fontSize="10" fill="currentColor">O</text>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>キャプション</figcaption>
</figure>
```

---

## 2. 単位円と角度 θ →（cosθ, sinθ）

第1象限の弧の上の点 P が `(cosθ, sinθ)`。横が cosθ（青）、縦が sinθ（緑）。
点 P の座標は中心 O=(45,165)・半径 R=110、角度 θ で `(45 + R·cosθ, 165 − R·sinθ)`。下は θ≈52°の例。

```jsx
<figure style={{margin: '1rem auto', textAlign: 'center', maxWidth: '280px'}}>
  <svg viewBox="0 0 220 200" width="250" role="img" aria-label="単位円。角度θの点は(cosθ, sinθ)">
    <line x1="20" y1="165" x2="205" y2="165" stroke="currentColor" strokeOpacity="0.25" strokeWidth="1" />
    <line x1="45" y1="30" x2="45" y2="180" stroke="currentColor" strokeOpacity="0.25" strokeWidth="1" />
    <text x="200" y="178" fontSize="10" fill="currentColor" fillOpacity="0.6">x（横）</text>
    <text x="50" y="38" fontSize="10" fill="currentColor" fillOpacity="0.6">y（縦）</text>
    <path d="M 155 165 A 110 110 0 0 0 45 55" fill="none" stroke="currentColor" strokeOpacity="0.4" strokeWidth="1.3" strokeDasharray="4 3" />
    <line x1="112.7" y1="78.3" x2="112.7" y2="165" stroke="currentColor" strokeOpacity="0.45" strokeWidth="1.1" strokeDasharray="4 3" />
    <line x1="112.7" y1="78.3" x2="45" y2="78.3" stroke="currentColor" strokeOpacity="0.45" strokeWidth="1.1" strokeDasharray="4 3" />
    <line x1="45" y1="165" x2="112.7" y2="165" stroke="#3B82F6" strokeWidth="3" />
    <line x1="45" y1="165" x2="45" y2="78.3" stroke="#10B981" strokeWidth="3" />
    <line x1="45" y1="165" x2="112.7" y2="78.3" stroke="#EF4444" strokeWidth="2.2" />
    <path d="M 73 165 A 28 28 0 0 0 62.2 142.9" fill="none" stroke="currentColor" strokeWidth="1.3" />
    <text x="69" y="159" fontSize="11" fill="currentColor">θ</text>
    <circle cx="112.7" cy="78.3" r="3.6" fill="#EF4444" />
    <text x="116" y="74" fontSize="11" fill="#EF4444">(cosθ, sinθ)</text>
    <text x="86" y="112" fontSize="11" fill="#EF4444">1</text>
    <text x="62" y="180" fontSize="11" fill="#3B82F6" textAnchor="middle">cosθ</text>
    <text x="8" y="125" fontSize="11" fill="#10B981">sinθ</text>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>角度θの点は (cosθ, sinθ)。横が cosθ、縦が sinθ</figcaption>
</figure>
```

---

## 3. 直角三角形（2点間の距離）

距離 = 横の差・縦の差を2辺とする斜辺、を示す。三平方の定理の説明に。

```jsx
<figure style={{margin: '1rem auto', textAlign: 'center', maxWidth: '260px'}}>
  <svg viewBox="0 0 240 165" width="230" role="img" aria-label="距離は直角三角形の斜辺">
    <line x1="55" y1="45" x2="185" y2="45" stroke="currentColor" strokeOpacity="0.55" strokeWidth="1.5" strokeDasharray="5 4" />
    <line x1="185" y1="45" x2="185" y2="140" stroke="currentColor" strokeOpacity="0.55" strokeWidth="1.5" strokeDasharray="5 4" />
    <line x1="55" y1="45" x2="185" y2="140" stroke="#10B981" strokeWidth="2.4" />
    <path d="M 174 45 L 174 56 L 185 56" fill="none" stroke="currentColor" strokeOpacity="0.55" strokeWidth="1.2" />
    <circle cx="55" cy="45" r="3.5" fill="#3B82F6" />
    <circle cx="185" cy="140" r="3.5" fill="#EF4444" />
    <text x="34" y="42" fontSize="12" fill="#3B82F6">P₁</text>
    <text x="189" y="143" fontSize="12" fill="#EF4444">P₂</text>
    <text x="120" y="38" fontSize="10" fill="currentColor" textAnchor="middle">横の差</text>
    <text x="191" y="96" fontSize="10" fill="currentColor">縦の差</text>
    <text x="104" y="100" fontSize="14" fill="#10B981" fontStyle="italic">d</text>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>距離 d は直角三角形の斜辺（三平方の定理）</figcaption>
</figure>
```

---

## 4. 2ベクトルと「なす角」（類似度・回転）

原点から2本のベクトル＋なす角の弧。コサイン類似度・回転（before/after）に使える。
回転を示すなら2本目を「回転後」として色を変え、弧に θ を添える。

```jsx
<figure style={{margin: '1rem auto', textAlign: 'center', maxWidth: '240px'}}>
  <svg viewBox="0 0 200 150" width="215" role="img" aria-label="2つのベクトルとなす角θ">
    <line x1="20" y1="110" x2="185" y2="110" stroke="currentColor" strokeOpacity="0.25" strokeWidth="1" />
    <line x1="90" y1="20" x2="90" y2="140" stroke="currentColor" strokeOpacity="0.25" strokeWidth="1" />
    <path d="M 127.6 96.3 A 40 40 0 0 0 110 75.4" fill="none" stroke="currentColor" strokeWidth="1.3" />
    <text x="114" y="98" fontSize="11" fill="currentColor">θ</text>
    <line x1="90" y1="110" x2="155.8" y2="86.1" stroke="#3B82F6" strokeWidth="2.4" />
    <line x1="90" y1="110" x2="125" y2="49.4" stroke="#10B981" strokeWidth="2.4" />
    <circle cx="155.8" cy="86.1" r="3.2" fill="#3B82F6" />
    <circle cx="125" cy="49.4" r="3.2" fill="#10B981" />
    <text x="159" y="84" fontSize="12" fill="#3B82F6" fontStyle="italic">a</text>
    <text x="128" y="46" fontSize="12" fill="#10B981" fontStyle="italic">b</text>
    <text x="78" y="123" fontSize="10" fill="currentColor">O</text>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>2ベクトルのなす角 θ</figcaption>
</figure>
```

---

## 5. 関数の曲線（polyline）＋接線

曲線は `<polyline points="x,y x,y ...">` で点を打つのが一番ラク（厳密な関数でなく形が伝わればよい）。
微分（接線の傾き）やシグモイドなどに。下は曲線＋ある点 P での接線と Δx, Δy。

```jsx
<figure style={{margin: '1rem auto', textAlign: 'center', maxWidth: '300px'}}>
  <svg viewBox="0 0 240 160" width="270" role="img" aria-label="曲線上の点での接線の傾き">
    <line x1="30" y1="135" x2="225" y2="135" stroke="currentColor" strokeOpacity="0.25" strokeWidth="1" />
    <line x1="40" y1="15" x2="40" y2="145" stroke="currentColor" strokeOpacity="0.25" strokeWidth="1" />
    <polyline points="48,128 72,118 96,103 120,84 140,66 160,49 180,34 200,22" fill="none" stroke="#3B82F6" strokeWidth="2.2" />
    <line x1="92" y1="106" x2="172" y2="46" stroke="#EF4444" strokeWidth="2" />
    <circle cx="120" cy="84" r="3.4" fill="#EF4444" />
    <text x="118" y="78" fontSize="11" fill="#EF4444">P</text>
    <line x1="120" y1="84" x2="160" y2="84" stroke="currentColor" strokeOpacity="0.5" strokeWidth="1.2" strokeDasharray="4 3" />
    <line x1="160" y1="84" x2="160" y2="54" stroke="currentColor" strokeOpacity="0.5" strokeWidth="1.2" strokeDasharray="4 3" />
    <text x="140" y="97" fontSize="10" fill="currentColor">Δx</text>
    <text x="164" y="72" fontSize="10" fill="currentColor">Δy</text>
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>接線の傾き ＝ Δy / Δx</figcaption>
</figure>
```

シグモイド等の S 字は同じ `polyline` で点列を `24,108 50,106 72,102 92,94 108,82 122,70 136,58 152,46 172,38 196,33 218,31`
のように打ち、`0` `0.5` `1` の補助線（破線）を添えると分かりやすい。

---

## 6. 棒グラフ（変換 before → after）

「入力 → 変換 → 出力」を2グループの棒で見せる。softmax（スコア → 確率）などに。
棒は `<rect>`。高さ＝値×スケール、`y = baseline − 高さ`。負の値は baseline より下に伸ばす。

```jsx
<figure style={{margin: '1rem auto', textAlign: 'center', maxWidth: '360px'}}>
  <svg viewBox="0 0 320 180" width="330" role="img" aria-label="変換前後の棒グラフ">
    <text x="75" y="16" fontSize="11" fill="currentColor" textAnchor="middle">入力</text>
    <line x1="20" y1="120" x2="135" y2="120" stroke="currentColor" strokeOpacity="0.35" strokeWidth="1.2" />
    <rect x="35" y="84" width="20" height="36" fill="#3B82F6" fillOpacity="0.7" />
    <rect x="70" y="108" width="20" height="12" fill="#3B82F6" fillOpacity="0.7" />
    <rect x="105" y="120" width="20" height="12" fill="#EF4444" fillOpacity="0.55" />
    <text x="160" y="100" fontSize="13" fill="currentColor" textAnchor="middle">変換</text>
    <text x="160" y="116" fontSize="15" fill="currentColor" textAnchor="middle">→</text>
    <text x="250" y="16" fontSize="11" fill="currentColor" textAnchor="middle">出力</text>
    <line x1="200" y1="120" x2="315" y2="120" stroke="currentColor" strokeOpacity="0.35" strokeWidth="1.2" />
    <rect x="215" y="53" width="20" height="67" fill="#10B981" fillOpacity="0.75" />
    <rect x="250" y="111" width="20" height="9" fill="#10B981" fillOpacity="0.75" />
    <rect x="285" y="116" width="20" height="4" fill="#10B981" fillOpacity="0.75" />
  </svg>
  <figcaption style={{fontSize: '0.82rem', marginTop: '0.3rem', opacity: 0.85}}>変換前のスコア（左）→ 変換後（右）</figcaption>
</figure>
```

---

## 7. グリッド（行列・テンソル・行列積）

セルは `<rect>` を等間隔に並べる。`fill="currentColor" fillOpacity="0.1"` で淡いマス目。
行や列を強調するときは下に色付き `<rect fillOpacity="0.18">` を敷く。

**階数（点 / 線 / 面 / 立体）の連図**：1マス → 1×3 → 3×3 → 3×3＋奥行きの枠線。
**行列積**：左行列の「行」と右行列の「列」を同じ色で塗り、結果のセルを同色で強調すると
「行×列 → そのセル」が直感的に伝わる。3×3 の例：

```jsx
{/* 3×3 グリッド（step 24, セル22）*/}
<svg viewBox="0 0 96 96" width="84" role="img" aria-label="行列（面）">
  <rect x="15" y="15" width="22" height="22" rx="2" fill="currentColor" fillOpacity="0.12" stroke="currentColor" strokeOpacity="0.55" strokeWidth="1.4" />
  <rect x="39" y="15" width="22" height="22" rx="2" fill="currentColor" fillOpacity="0.12" stroke="currentColor" strokeOpacity="0.55" strokeWidth="1.4" />
  <rect x="63" y="15" width="22" height="22" rx="2" fill="currentColor" fillOpacity="0.12" stroke="currentColor" strokeOpacity="0.55" strokeWidth="1.4" />
  <rect x="15" y="39" width="22" height="22" rx="2" fill="currentColor" fillOpacity="0.12" stroke="currentColor" strokeOpacity="0.55" strokeWidth="1.4" />
  <rect x="39" y="39" width="22" height="22" rx="2" fill="currentColor" fillOpacity="0.12" stroke="currentColor" strokeOpacity="0.55" strokeWidth="1.4" />
  <rect x="63" y="39" width="22" height="22" rx="2" fill="currentColor" fillOpacity="0.12" stroke="currentColor" strokeOpacity="0.55" strokeWidth="1.4" />
  <rect x="15" y="63" width="22" height="22" rx="2" fill="currentColor" fillOpacity="0.12" stroke="currentColor" strokeOpacity="0.55" strokeWidth="1.4" />
  <rect x="39" y="63" width="22" height="22" rx="2" fill="currentColor" fillOpacity="0.12" stroke="currentColor" strokeOpacity="0.55" strokeWidth="1.4" />
  <rect x="63" y="63" width="22" height="22" rx="2" fill="currentColor" fillOpacity="0.12" stroke="currentColor" strokeOpacity="0.55" strokeWidth="1.4" />
</svg>
```

奥行き（3階テンソル）は、上の前面グリッドに加えて、`+14,−14` ずらした枠線 `<rect fill="none">` と、
四隅をつなぐ4本の `<line>` を足すと立体に見える。
