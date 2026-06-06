import Heading from '@theme/Heading';
import type { ReactNode } from 'react';

import styles from './styles.module.css';

type Stage = {
  index: string;
  emoji: string;
  title: string;
  subtitle: string;
  detail: string;
};

const STAGES: Stage[] = [
  {
    index: '01',
    emoji: '✂️',
    title: 'トークナイズ',
    subtitle: 'Tokenizer',
    detail: '生テキストを BPE でサブワードに分解し、ID 列へ。',
  },
  {
    index: '02',
    emoji: '🧮',
    title: '埋め込み',
    subtitle: 'Embedding',
    detail: 'トークン ID を意味を持つベクトルへ写像する。',
  },
  {
    index: '03',
    emoji: '🧠',
    title: 'Transformer ×N',
    subtitle: 'Self-Attention',
    detail: '注意機構で文脈を混ぜ合わせ、層を重ねて表現を磨く。',
  },
  {
    index: '04',
    emoji: '🎯',
    title: '次トークン予測',
    subtitle: 'LM Head',
    detail: '語彙全体の確率分布を出力し、次の1トークンを選ぶ。',
  },
  {
    index: '05',
    emoji: '🔁',
    title: '学習ループ',
    subtitle: 'Training',
    detail: '損失を計算し勾配で重みを更新。これを繰り返す。',
  },
];

export default function BuildPipeline(): ReactNode {
  return (
    <section className={styles.section}>
      <div className="container">
        <div className={styles.heading}>
          <span className={styles.kicker}>FROM SCRATCH</span>
          <Heading as="h2" className={styles.title}>
            ゼロから組み立てる、LLM の 5 工程
          </Heading>
          <p className={styles.lead}>
            ブラックボックスを開けて、1つずつ自分の手で組み立てていく。
          </p>
        </div>

        <ol className={styles.pipeline}>
          {STAGES.map((s, i) => (
            <li className={styles.stage} key={s.index}>
              <div className={styles.card}>
                <div className={styles.cardTop}>
                  <span className={styles.index}>{s.index}</span>
                  <span
                    className={styles.emoji}
                    role="img"
                    aria-label={s.title}
                  >
                    {s.emoji}
                  </span>
                </div>
                <Heading as="h3" className={styles.stageTitle}>
                  {s.title}
                </Heading>
                <span className={styles.subtitle}>{s.subtitle}</span>
                <p className={styles.detail}>{s.detail}</p>
              </div>
              {i < STAGES.length - 1 && (
                <span className={styles.connector} aria-hidden="true">
                  →
                </span>
              )}
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}
