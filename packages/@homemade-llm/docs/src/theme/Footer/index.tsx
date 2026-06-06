import Link from '@docusaurus/Link';
import { useThemeConfig } from '@docusaurus/theme-common';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import type { ReactNode } from 'react';

import styles from './styles.module.css';

type FooterItem = {
  label: string;
  to?: string;
  href?: string;
};

type FooterColumn = {
  title?: string;
  items: FooterItem[];
};

export default function Footer(): ReactNode {
  const { footer } = useThemeConfig();
  const { siteConfig } = useDocusaurusContext();

  const links = (footer?.links ?? []) as FooterColumn[];
  const copyright = footer?.copyright;
  const logoUrl = `${siteConfig.baseUrl}img/logo.svg`;

  return (
    <footer className={styles.footer}>
      <span className={styles.topline} aria-hidden="true" />
      <span className={styles.wordmark} aria-hidden="true">
        homemade-llm
      </span>

      <div className={`container ${styles.inner}`}>
        {/* CTA band */}
        <div className={styles.cta}>
          <div>
            <p className={styles.ctaKicker}>{'// ready to build?'}</p>
            <h2 className={styles.ctaTitle}>
              さあ、<span className={styles.ctaAccent}>ゼロから</span>作ろう。
            </h2>
          </div>
          <div className={styles.ctaActions}>
            <Link className={styles.primaryBtn} to="/docs/intro">
              はじめる<span className={styles.arrow}>→</span>
            </Link>
            <code className={styles.cmd}>
              <span className={styles.prompt}>$</span> pnpm dev:docs
              <span className={styles.caret} />
            </code>
          </div>
        </div>

        {/* Brand + sitemap */}
        <div className={styles.grid}>
          <div className={styles.brand}>
            <div className={styles.brandTop}>
              <img
                className={styles.logo}
                src={logoUrl}
                alt=""
                width={28}
                height={28}
              />
              <span className={styles.brandName}>{siteConfig.title}</span>
            </div>
            <p className={styles.tagline}>{siteConfig.tagline}</p>
            <div className={styles.meta}>
              <span>v1.0.0</span>
              <span className={styles.dot}>·</span>
              <span>MIT</span>
              <span className={styles.dot}>·</span>
              <span>built from scratch</span>
            </div>
          </div>

          {links.length > 0 && (
            <nav className={styles.cols} aria-label="フッター">
              {links.map((col) => (
                <div
                  className={styles.col}
                  key={col.title ?? col.items[0]?.label}
                >
                  {col.title && (
                    <h3 className={styles.colTitle}>{col.title}</h3>
                  )}
                  <ul className={styles.collist}>
                    {col.items.map((item) => (
                      <li key={item.label}>
                        <Link
                          className={styles.flink}
                          {...(item.href
                            ? { href: item.href }
                            : { to: item.to ?? '#' })}
                        >
                          {item.label}
                        </Link>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </nav>
          )}
        </div>

        {/* Bottom bar */}
        <div className={styles.bottom}>
          <div className={styles.status}>
            <span className={styles.statusDot} />
            status: <b>building</b> · 100% from scratch
          </div>
          {copyright && <div className={styles.copyright}>{copyright}</div>}
        </div>
      </div>
    </footer>
  );
}
