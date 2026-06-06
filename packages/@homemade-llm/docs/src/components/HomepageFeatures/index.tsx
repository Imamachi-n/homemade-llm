import Heading from '@theme/Heading';
import type { CSSProperties, ReactNode } from 'react';

import styles from './styles.module.css';

/** アテンション重みのヒートマップ風アニメ（装飾） */
function AttentionMap(): ReactNode {
  const N = 8;
  return (
    <div className={styles.attn} aria-hidden="true">
      {Array.from({ length: N * N }, (_, k) => {
        const row = Math.floor(k / N);
        const col = k % N;
        // 下三角（causal mask）を強めに光らせる
        const causal = col <= row;
        const delay = ((row * 7 + col * 13) % 24) * 0.09;
        return (
          <span
            // biome-ignore lint/suspicious/noArrayIndexKey: 固定長の装飾グリッド
            key={k}
            className={causal ? styles.attnLive : undefined}
            style={{ '--d': `${delay}s` } as CSSProperties}
          />
        );
      })}
    </div>
  );
}

/** 自作レイヤーのコードトレース（装飾） */
function CodeTrace(): ReactNode {
  return (
    <pre className={styles.code} aria-hidden="true">
      <code>
        <span className={styles.cKw}>class</span> SelfAttention:{'\n'}
        {'  '}Q = x @ Wq{'   '}
        <span className={styles.cCom}># [T, d]</span>
        {'\n'}
        {'  '}K = x @ Wk{'\n'}
        {'  '}A = <span className={styles.cFn}>softmax</span>(Q @ K.ᵀ / √d)
        {'\n'}
        {'  '}
        <span className={styles.cKw}>return</span> A @ V
        <span className={styles.caret} />
      </code>
    </pre>
  );
}

/** 同じ処理を TS と Python で（将来の二言語対応を示す） */
function LangDuality(): ReactNode {
  return (
    <div className={styles.dual} aria-hidden="true">
      <div className={styles.dualRow}>
        <span className={`${styles.lang} ${styles.langTs}`}>TS</span>
        <code>const y = model.forward(ids)</code>
      </div>
      <div className={styles.dualRow}>
        <span className={`${styles.lang} ${styles.langPy}`}>PY</span>
        <code>y = model(ids)</code>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className={styles.heading}>
          <span className={styles.kicker}>WHY FROM SCRATCH</span>
          <Heading as="h2" className={styles.title}>
            なぜ、ゼロから作るのか
          </Heading>
        </div>

        <div className={styles.bento}>
          <article className={`${styles.tile} ${styles.tileLg}`}>
            <div className={styles.tileViz}>
              <AttentionMap />
            </div>
            <div className={styles.tileBody}>
              <span className={styles.tag}>01 / principle</span>
              <Heading as="h3" className={styles.tileTitle}>
                原理から理解する
              </Heading>
              <p className={styles.tileText}>
                Attention も勾配降下も、数式と動くコードの両輪で。
                右のヒートマップのような「中で起きていること」を、ブラックボックスにしない。
              </p>
            </div>
          </article>

          <article className={styles.tile}>
            <CodeTrace />
            <div className={styles.tileBody}>
              <span className={styles.tag}>02 / hands-on</span>
              <Heading as="h3" className={styles.tileTitle}>
                手を動かして作る
              </Heading>
              <p className={styles.tileText}>
                読むだけで終わらせない。1 行ずつ自分で実装し、動かして確かめる。
              </p>
            </div>
          </article>

          <article className={styles.tile}>
            <LangDuality />
            <div className={styles.tileBody}>
              <span className={styles.tag}>03 / polyglot</span>
              <Heading as="h3" className={styles.tileTitle}>
                TypeScript と Python
              </Heading>
              <p className={styles.tileText}>
                ドキュメントは TypeScript ベース。将来は Python
                のサンプルも扱えるよう設計している。
              </p>
            </div>
          </article>
        </div>
      </div>
    </section>
  );
}
