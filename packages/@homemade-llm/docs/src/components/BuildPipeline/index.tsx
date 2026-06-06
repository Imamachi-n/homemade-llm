import Heading from '@theme/Heading';
import type { CSSProperties, ReactNode } from 'react';

import styles from './styles.module.css';

type Glyph = 'tokens' | 'vector' | 'attn' | 'logits' | 'loss';

type Stage = {
  i: string;
  name: string;
  en: string;
  shape: string;
  detail: string;
  glyph: Glyph;
};

const STAGES: Stage[] = [
  {
    i: '01',
    name: 'トークナイズ',
    en: 'Tokenizer',
    shape: 'text → [T]',
    detail: 'BPE で文字列をサブワード ID 列へ分解する。',
    glyph: 'tokens',
  },
  {
    i: '02',
    name: '埋め込み',
    en: 'Embedding',
    shape: '[T] → [T, d]',
    detail: 'トークン ID を意味を持つベクトルへ写像。',
    glyph: 'vector',
  },
  {
    i: '03',
    name: 'Transformer ×N',
    en: 'Self-Attention',
    shape: '[T, d] → [T, d]',
    detail: '注意機構で文脈を混ぜ、層を重ねて磨く。',
    glyph: 'attn',
  },
  {
    i: '04',
    name: '次トークン予測',
    en: 'LM Head',
    shape: '[T, d] → [V]',
    detail: '語彙全体の確率分布を出力する。',
    glyph: 'logits',
  },
  {
    i: '05',
    name: '学習ループ',
    en: 'Training',
    shape: 'loss → ∇ → θ',
    detail: '損失から勾配を求め、重みを更新し反復。',
    glyph: 'loss',
  },
];

function StageGlyph({ type }: { type: Glyph }): ReactNode {
  switch (type) {
    case 'tokens':
      return (
        <div className={styles.gTokens}>
          {['▁猫', 'が', '鳴'].map((t) => (
            <span key={t}>{t}</span>
          ))}
        </div>
      );
    case 'vector':
      return (
        <div className={styles.gVector}>
          {['0.21', '-1.3', '0.74', '0.08', '-0.5'].map((v) => (
            <span key={v}>{v}</span>
          ))}
        </div>
      );
    case 'attn':
      return (
        <div className={styles.gAttn} aria-hidden="true">
          {Array.from({ length: 16 }, (_, k) => (
            <span
              // biome-ignore lint/suspicious/noArrayIndexKey: 固定長の装飾グリッド
              key={k}
              style={{ '--d': `${(k * 53) % 16}` } as CSSProperties}
            />
          ))}
        </div>
      );
    case 'logits':
      return (
        <div className={styles.gLogits} aria-hidden="true">
          {[40, 72, 30, 95, 18].map((h, k) => (
            <span
              // biome-ignore lint/suspicious/noArrayIndexKey: 固定長の装飾バー
              key={k}
              className={h === 95 ? styles.gLogitsTop : undefined}
              style={{ height: `${h}%` }}
            />
          ))}
        </div>
      );
    case 'loss':
      return (
        <svg
          className={styles.gLoss}
          viewBox="0 0 64 30"
          fill="none"
          aria-hidden="true"
        >
          <title>loss curve</title>
          <polyline
            points="0,4 9,9 18,7 27,15 36,14 45,21 54,23 64,26"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      );
  }
}

export default function BuildPipeline(): ReactNode {
  return (
    <section className={styles.section}>
      <div className="container">
        <div className={styles.heading}>
          <span className={styles.kicker}>FORWARD PASS</span>
          <Heading as="h2" className={styles.title}>
            ゼロから組み立てる、LLM の 5 工程
          </Heading>
          <p className={styles.lead}>
            テキストが行列とベクトルに姿を変え、次の一語になるまで。1
            工程ずつ自分の手で。
          </p>
        </div>

        <ol className={styles.flow}>
          <span className={styles.rail} aria-hidden="true">
            <span className={styles.pulse} />
          </span>
          {STAGES.map((s, idx) => (
            <li
              className={styles.module}
              key={s.i}
              style={{ '--i': idx } as CSSProperties}
            >
              <span className={styles.port} aria-hidden="true" />
              <div className={styles.mod}>
                <div className={styles.modTop}>
                  <span className={styles.index}>{s.i}</span>
                  <span className={styles.shape}>{s.shape}</span>
                </div>
                <div className={styles.glyph}>
                  <StageGlyph type={s.glyph} />
                </div>
                <Heading as="h3" className={styles.stageTitle}>
                  {s.name}
                </Heading>
                <span className={styles.en}>{s.en}</span>
                <p className={styles.detail}>{s.detail}</p>
              </div>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}
