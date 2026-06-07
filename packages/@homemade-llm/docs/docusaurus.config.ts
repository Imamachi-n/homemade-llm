import type * as Preset from '@docusaurus/preset-classic';
import type { Config } from '@docusaurus/types';
import { themes as prismThemes } from 'prism-react-renderer';
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: '自作 LLM 入門',
  tagline: '大規模言語モデルをゼロから作って学ぶ',
  favicon: 'img/favicon.svg',

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
    faster: true, // Enable Rspack + SWC + Lightning CSS for faster builds
  },

  markdown: {
    mermaid: true,
  },
  themes: [
    '@docusaurus/theme-mermaid',
    [
      require.resolve('@easyops-cn/docusaurus-search-local'),
      {
        hashed: true,
        language: ['ja', 'en'],
        indexDocs: true,
        indexBlog: false,
        indexPages: true,
        docsRouteBasePath: '/docs',
        highlightSearchTermsOnTargetPage: true,
        explicitSearchResultPath: true,
      },
    ],
  ],

  // Set the production url of your site here
  url: 'https://Imamachi-n.github.io',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/homemade-llm/',

  // GitHub pages deployment config.
  organizationName: 'Imamachi-n',
  projectName: 'homemade-llm',

  onBrokenLinks: 'throw',

  // KaTeX の数式スタイルを読み込む
  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.16.47/dist/katex.min.css',
      type: 'text/css',
      integrity:
        'sha384-nH0MfJ44wi1dd7w6jinlyBgljjS8EJAh2JBoRad8a3VDw2K69vfaaqm4WnR+gXtA',
      crossorigin: 'anonymous',
    },
  ],

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'ja',
    locales: ['ja'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/Imamachi-n/homemade-llm/tree/main/packages/@homemade-llm/docs/',
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    // Replace with your project's social card
    image: 'img/logo.svg',
    colorMode: {
      respectPrefersColorScheme: true,
    },
    mermaid: {
      theme: {
        light: 'base',
        dark: 'dark',
      },
      options: {
        themeVariables: {
          primaryColor: '#DBEAFE',
          primaryTextColor: '#1E3A5F',
          primaryBorderColor: '#3B82F6',
          lineColor: '#94A3B8',
          secondaryColor: '#EDE9FE',
          tertiaryColor: '#F0F9FF',
          fontFamily: 'inherit',
        },
      },
    },
    navbar: {
      title: '自作 LLM 入門',
      logo: {
        alt: '自作 LLM 入門 Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'docsSidebar',
          position: 'left',
          label: 'Docs',
        },
        {
          type: 'search',
          position: 'right',
        },
        {
          href: 'https://github.com/Imamachi-n/homemade-llm',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'はじめに',
          items: [
            {
              label: 'イントロダクション',
              to: '/docs/intro',
            },
          ],
        },
        {
          title: '基礎知識',
          items: [
            {
              label: 'Chapter 1: LLM の全体像',
              to: '/docs/foundations/chapter1',
            },
          ],
        },
        {
          title: 'LLM をゼロから作る',
          items: [
            {
              label: 'Chapter 1: トークナイザー',
              to: '/docs/llm-from-scratch/chapter1',
            },
            {
              label: 'Chapter 2: Transformer の実装',
              to: '/docs/llm-from-scratch/chapter2',
            },
            {
              label: 'Chapter 3: 学習ループ',
              to: '/docs/llm-from-scratch/chapter3',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/Imamachi-n/homemade-llm',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Naoto Imamachi`,
    },
    prism: {
      theme: prismThemes.nightOwl,
      darkTheme: prismThemes.nightOwl,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
