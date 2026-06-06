import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import BuildPipeline from '@site/src/components/BuildPipeline';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import TokenPredictor from '@site/src/components/TokenPredictor';
import Heading from '@theme/Heading';
import Layout from '@theme/Layout';
import type { ReactNode } from 'react';

import styles from './index.module.css';

function HomepageHero() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <header className={styles.hero}>
      <div className={styles.heroBg} aria-hidden="true">
        <div className={styles.grid} />
        <div className={`${styles.blob} ${styles.blob1}`} />
        <div className={`${styles.blob} ${styles.blob2}`} />
        <div className={`${styles.blob} ${styles.blob3}`} />
        <div className={styles.scan} />
      </div>

      <div className={`container ${styles.heroInner}`}>
        <div className={styles.heroCopy}>
          <span className={styles.badge}>
            <span className={styles.badgeDot} />
            build-your-own LLM
          </span>
          <Heading as="h1" className={styles.title}>
            大規模言語モデルを
            <br />
            <span className={styles.titleAccent}>ゼロから作って学ぶ</span>
          </Heading>
          <p className={styles.subtitle}>
            {siteConfig.tagline}。トークナイザーから
            Transformer、学習ループまで。
            ライブラリの裏側で何が起きているのかを、手を動かしながら解き明かす。
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
  return (
    <Layout title={siteConfig.title} description={siteConfig.tagline}>
      <HomepageHero />
      <main>
        <BuildPipeline />
        <HomepageFeatures />
      </main>
    </Layout>
  );
}
