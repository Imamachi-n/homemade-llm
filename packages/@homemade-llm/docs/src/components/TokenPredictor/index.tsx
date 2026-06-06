import { type ReactNode, useEffect, useState } from 'react';

import styles from './styles.module.css';

type Candidate = {
  token: string;
  prob: number;
};

type Step = {
  /** 採用される（argmax の）トークン。candidates の先頭と一致させる */
  candidates: Candidate[];
};

const SEED = '大規模言語モデルを';

/**
 * "大規模言語モデルを ゼロ から 作っ て 学ぶ 。" を1トークンずつ生成していくスクリプト。
 * 各ステップの candidates 先頭が argmax（採用トークン）。
 */
const SCRIPT: Step[] = [
  {
    candidates: [
      { token: 'ゼロ', prob: 0.71 },
      { token: 'API', prob: 0.13 },
      { token: '数式', prob: 0.1 },
      { token: '雰囲気', prob: 0.06 },
    ],
  },
  {
    candidates: [
      { token: 'から', prob: 0.83 },
      { token: 'を', prob: 0.09 },
      { token: 'だけ', prob: 0.05 },
      { token: '無しで', prob: 0.03 },
    ],
  },
  {
    candidates: [
      { token: '作っ', prob: 0.64 },
      { token: '理解し', prob: 0.2 },
      { token: '動かし', prob: 0.11 },
      { token: '眺め', prob: 0.05 },
    ],
  },
  {
    candidates: [
      { token: 'て', prob: 0.91 },
      { token: 'た', prob: 0.05 },
      { token: 'ては', prob: 0.03 },
      { token: 'つつ', prob: 0.01 },
    ],
  },
  {
    candidates: [
      { token: '学ぶ', prob: 0.74 },
      { token: '遊ぶ', prob: 0.12 },
      { token: '味わう', prob: 0.09 },
      { token: '極める', prob: 0.05 },
    ],
  },
  {
    candidates: [
      { token: '。', prob: 0.86 },
      { token: '！', prob: 0.09 },
      { token: '…', prob: 0.03 },
      { token: '🚀', prob: 0.02 },
    ],
  },
];

const STEP_MS = 1400;
const RESET_MS = 2600;
const fullSentence =
  SEED + SCRIPT.map((s) => s.candidates[0]?.token ?? '').join('');

export default function TokenPredictor(): ReactNode {
  // committed: すでに生成（採用）済みのトークン数
  const [committed, setCommitted] = useState(0);
  const [reducedMotion, setReducedMotion] = useState(false);

  useEffect(() => {
    const mql = window.matchMedia('(prefers-reduced-motion: reduce)');
    if (mql.matches) {
      setReducedMotion(true);
      setCommitted(SCRIPT.length);
      return;
    }

    let timer: ReturnType<typeof setTimeout>;
    const tick = (next: number) => {
      if (next > SCRIPT.length) {
        // 文が完成 → 少し見せてからリセット
        timer = setTimeout(() => {
          setCommitted(0);
          timer = setTimeout(() => tick(1), STEP_MS);
        }, RESET_MS);
        return;
      }
      setCommitted(next);
      timer = setTimeout(() => tick(next + 1), STEP_MS);
    };

    timer = setTimeout(() => tick(1), STEP_MS);
    return () => clearTimeout(timer);
  }, []);

  const generated = SCRIPT.slice(0, committed)
    .map((s) => s.candidates[0]?.token ?? '')
    .join('');
  const isComplete = committed >= SCRIPT.length;
  const currentStep = SCRIPT[committed];

  if (reducedMotion) {
    return (
      <div
        className={styles.console}
        role="img"
        aria-label="次トークン予測のデモ"
      >
        <ConsoleHeader />
        <div className={styles.stream}>
          <span className={styles.seed}>{SEED}</span>
          <span className={styles.generated}>
            {fullSentence.slice(SEED.length)}
          </span>
        </div>
      </div>
    );
  }

  return (
    <div className={styles.console} aria-hidden="true">
      <ConsoleHeader />

      <div className={styles.stream}>
        <span className={styles.seed}>{SEED}</span>
        <span className={styles.generated}>{generated}</span>
        <span className={styles.cursor} />
      </div>

      <div className={styles.predictLabel}>
        {isComplete ? 'sequence complete ✓' : 'P(next token | context)'}
      </div>

      <div className={styles.candidates} key={committed}>
        {!isComplete &&
          currentStep?.candidates.map((c, i) => (
            <div
              className={
                i === 0 ? `${styles.row} ${styles.rowTop}` : styles.row
              }
              key={c.token}
              style={{ animationDelay: `${i * 70}ms` }}
            >
              <span className={styles.token}>{c.token}</span>
              <span className={styles.barTrack}>
                <span
                  className={styles.barFill}
                  style={{ width: `${Math.round(c.prob * 100)}%` }}
                />
              </span>
              <span className={styles.prob}>{c.prob.toFixed(2)}</span>
              {i === 0 && <span className={styles.argmax}>argmax</span>}
            </div>
          ))}
        {isComplete && (
          <div className={styles.done}>
            <code>{fullSentence}</code>
          </div>
        )}
      </div>
    </div>
  );
}

function ConsoleHeader(): ReactNode {
  return (
    <div className={styles.header}>
      <span className={styles.dot} />
      <span className={styles.headerLabel}>homemade-llm · inference</span>
    </div>
  );
}
