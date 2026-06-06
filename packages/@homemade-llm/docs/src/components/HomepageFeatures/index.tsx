import Heading from '@theme/Heading';
import clsx from 'clsx';
import type { ReactNode } from 'react';

import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  emoji: string;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'ゼロから理解する',
    emoji: '🧱',
    description: (
      <>
        トークナイザーから
        Transformer、学習ループまで。ライブラリの内部で何が起きているかを、
        手を動かしながら理解できます。
      </>
    ),
  },
  {
    title: '実装と解説をセットで',
    emoji: '💡',
    description: (
      <>
        理論の解説だけでなく、動くコードと一緒に学べます。読んで終わりではなく、実際に動かして確かめられます。
      </>
    ),
  },
  {
    title: 'TypeScript と Python',
    emoji: '🐍',
    description: (
      <>
        ドキュメントは TypeScript ベースですが、将来的に Python
        のサンプルコードも扱えるよう設計しています。
      </>
    ),
  },
];

function Feature({ title, emoji, description }: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <span className={styles.featureEmoji} role="img" aria-label={title}>
          {emoji}
        </span>
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props) => (
            <Feature key={props.title} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
