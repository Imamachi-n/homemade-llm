import Heading from '@theme/Heading';
import type { ReactNode } from 'react';

import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  emoji: string;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: '原理から理解する',
    emoji: '🔬',
    description: (
      <>
        Attention も勾配降下も、数式と動くコードの両輪で。
        「なぜそう動くのか」を腹落ちさせながら進める。
      </>
    ),
  },
  {
    title: '手を動かして作る',
    emoji: '🛠️',
    description: (
      <>
        読むだけで終わらせない。各章で実際に実装し、自分のモデルが文章を
        生成する瞬間まで辿り着く。
      </>
    ),
  },
  {
    title: 'TypeScript と Python',
    emoji: '🐍',
    description: (
      <>
        ドキュメントは TypeScript ベース。将来的には Python のサンプルコードも
        扱えるよう設計している。
      </>
    ),
  },
];

function Feature({ title, emoji, description }: FeatureItem) {
  return (
    <div className={styles.card}>
      <span className={styles.emoji} role="img" aria-label={title}>
        {emoji}
      </span>
      <Heading as="h3" className={styles.cardTitle}>
        {title}
      </Heading>
      <p className={styles.cardText}>{description}</p>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className={styles.heading}>
          <span className={styles.kicker}>WHY BUILD IT</span>
          <Heading as="h2" className={styles.title}>
            なぜ、ゼロから作るのか
          </Heading>
        </div>
        <div className={styles.cards}>
          {FeatureList.map((props) => (
            <Feature key={props.title} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
