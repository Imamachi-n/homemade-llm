import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import BuildPipeline from '@site/src/components/BuildPipeline';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import LatentField from '@site/src/components/LatentField';
import TokenPredictor from '@site/src/components/TokenPredictor';
import Heading from '@theme/Heading';
import Layout from '@theme/Layout';
import { type ReactNode, useEffect, useState } from 'react';

import styles from './index.module.css';

const GLYPHS = 'アイウエオカキクサシスセソタチランダムХ＃＄％＆８０１▌▙▚∑∇∂λ';

/** マウント時に1回、左から順にデコードされるスクランブル表示。reduced-motion では即確定。 */
function Scramble({ text }: { text: string }): ReactNode {
  const [display, setDisplay] = useState(text);

  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    const chars = [...text];
    let frame = 0;
    const total = chars.length * 3 + 6;
    const id = setInterval(() => {
      frame++;
      const settled = Math.floor(frame / 3);
      setDisplay(
        chars
          .map((c, i) => {
            if (i < settled || c === '　' || c === ' ') return c;
            return GLYPHS[Math.floor(Math.random() * GLYPHS.length)] ?? c;
          })
          .join(''),
      );
      if (frame >= total) {
        setDisplay(text);
        clearInterval(id);
      }
    }, 45);
    return () => clearInterval(id);
  }, [text]);

  return <span className={styles.scramble}>{display}</span>;
}

function HomepageHero() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <header className={styles.hero}>
      <div className={styles.heroBg} aria-hidden="true">
        <LatentField />
        <div className={styles.plasma} />
        <div className={styles.grid} />
        <div className={styles.vignette} />
      </div>

      {/* 計器パネル風の角ティック */}
      <span className={`${styles.tick} ${styles.tickTL}`} aria-hidden="true" />
      <span className={`${styles.tick} ${styles.tickTR}`} aria-hidden="true" />
      <span className={`${styles.tick} ${styles.tickBL}`} aria-hidden="true" />
      <span className={`${styles.tick} ${styles.tickBR}`} aria-hidden="true" />

      <div className={`container ${styles.heroInner}`}>
        <div className={styles.heroCopy}>
          <span className={styles.badge}>
            <span className={styles.badgeDot} />
            build-your-own LLM · from scratch
          </span>
          <Heading as="h1" className={styles.title}>
            <span className={styles.titleDim}>大規模言語モデルを</span>
            <span className={styles.titleAccent}>
              <Scramble text="ゼロから作って学ぶ" />
            </span>
          </Heading>

          <div className={styles.readout} aria-hidden="true">
            <span>
              <i>params</i> 124M
            </span>
            <span>
              <i>layers</i> 12
            </span>
            <span>
              <i>ctx</i> 1024
            </span>
            <span>
              <i>vocab</i> 50257
            </span>
          </div>

          <p className={styles.subtitle}>
            {siteConfig.tagline}。トークナイザーから
            Transformer、学習ループまで。
            ブラックボックスを開けて、行列とベクトルの手触りごと理解する。
          </p>
          <div className={styles.buttons}>
            <Link className={styles.primaryBtn} to="/docs/intro">
              はじめる
              <span className={styles.btnArrow}>→</span>
            </Link>
            <Link
              className={styles.ghostBtn}
              to="/docs/llm-from-scratch/chapter1"
            >
              <span className={styles.prompt}>$</span> ゼロから作る
            </Link>
          </div>
        </div>

        <div className={styles.heroDemo}>
          <TokenPredictor />
        </div>
      </div>
    </header>
  );
}

export default function Home(): ReactNode {
  const { siteConfig } = useDocusaurusContext();

  // ホームページはテーマトグルに依らずダーク没入。navbar も合わせるため html にクラス付与。
  useEffect(() => {
    const el = document.documentElement;
    el.classList.add('homepage');
    return () => el.classList.remove('homepage');
  }, []);

  return (
    <Layout title={siteConfig.title} description={siteConfig.tagline}>
      <div className={styles.page}>
        <HomepageHero />
        <main>
          <BuildPipeline />
          <HomepageFeatures />
        </main>
      </div>
    </Layout>
  );
}
