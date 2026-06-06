# homemade-llm

> 自作 LLM（大規模言語モデル）を学ぶためのドキュメントとサンプルコードをまとめたモノレポです。

## ディレクトリ構成

本プロジェクトは pnpm ワークスペースによるモノレポ構成です。

```plaintext
/
├── packages/
│   └── @homemade-llm/
│       └── docs/                    # Docusaurus ドキュメント (@homemade-llm/docs)
│           ├── docs/                # ドキュメント本文（章立て）
│           ├── src/                 # トップページ・コンポーネント・CSS
│           ├── static/             # 画像などの静的ファイル
│           ├── docusaurus.config.ts
│           └── package.json
├── pnpm-workspace.yaml              # ワークスペース設定
├── package.json                     # ルート設定
├── tsconfig.json                    # 共通 TypeScript 設定
└── biome.json                       # Biome 設定
```

### Python コードの配置（将来）

将来的に Python コードを追加する場合は、`python/<name>/`（各パッケージが独自に `pyproject.toml` を持つ）に配置する想定です。Python パッケージは pnpm ワークスペースには含めず、別途 Python のツール（uv など）で管理します。`.gitignore` / `.secretlintignore` / `biome.json` は Python の生成物（`.venv`、`__pycache__` など）を無視するよう設定済みです。

## 開発環境

| カテゴリ | ツール | 説明 |
| --- | --- | --- |
| 共通 | [pnpm](https://pnpm.io/) | パッケージマネージャー（モノレポ対応） |
| | [Biome](https://biomejs.dev/) | リンター・フォーマッター |
| | [secretlint](https://github.com/secretlint/secretlint) | シークレット検出ツール（API キーの誤コミット防止） |
| | [husky](https://typicode.github.io/husky/) | Git hooks 管理（pre-commit で secretlint を自動実行） |
| docs | [Docusaurus](https://docusaurus.io/) | ドキュメントサイト |
| | [Rspack](https://rspack.rs/) | Rust 製の高速バンドラ（experimental_faster） |
| | [SWC](https://swc.rs/) | Rust 製の高速トランスパイラ・ミニファイア |
| | [Lightning CSS](https://lightningcss.dev/) | Rust 製の高速 CSS パーサー・ミニファイア |

## セットアップ

### pnpm のインストール

```zsh
# Homebrew
brew install pnpm

# または npm
npm install -g pnpm
```

### npm パッケージのインストール

```zsh
pnpm install
```

## 使用方法

### ドキュメントサイト

```zsh
# 開発サーバー起動（http://localhost:3000/homemade-llm/）
pnpm dev:docs

# ビルド
pnpm build:docs

# ビルド成果物をローカルでプレビュー
pnpm preview:docs
```

### 特定パッケージでのコマンド実行

```zsh
pnpm --filter @homemade-llm/docs <command>
```

## シークレット検出（secretlint）

API キーなどのシークレットが誤ってコミットされるのを防ぐため、[secretlint](https://github.com/secretlint/secretlint) を導入しています。

- `git commit` 時に husky の pre-commit hook 経由で自動実行される
- OpenAI / AWS / GCP / GitHub / Slack / npm など主要サービスの API キーパターンを検出

```zsh
# 手動でシークレットスキャンを実行
pnpm lint:secret
```

## ドキュメントの公開（GitHub Pages）

`main` ブランチへの push をトリガーに、GitHub Actions（`.github/workflows/deploy-docs.yml`）で Docusaurus をビルドし GitHub Pages へデプロイします。

- 公開 URL: `https://Imamachi-n.github.io/homemade-llm/`
- 初回のみ、リポジトリの **Settings > Pages > Build and deployment > Source** を **GitHub Actions** に設定してください。
