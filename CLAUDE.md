# プロジェクト設定

## プロジェクト概要

`homemade-llm` は自作 LLM（大規模言語モデル）を学ぶためのドキュメントとサンプルコードを管理するモノレポです。

- ドキュメントサイトは [Docusaurus](https://docusaurus.io/) で構築し、`packages/@homemade-llm/docs` に配置しています。
- 現状は TypeScript（Docusaurus）が中心ですが、将来的に Python コードが入る可能性があります。

## ディレクトリ規約

- TypeScript / JavaScript パッケージは pnpm ワークスペース配下（`packages/@homemade-llm/*`）に配置します。
- Python パッケージは `python/<name>/`（各自 `pyproject.toml` を持つ）に配置する想定です。Python は pnpm ワークスペースには含めません。

## プランファイルの命名規則

プランモードでプランを保存する際、ファイル名はプラン内容を反映した分かりやすいタイトルにすること。

- 形式: `YYYY-MM-DD-<内容を表す短い英語のスラッグ>.md`
- 例: `2026-02-20-add-authentication-feature.md`、`2026-02-20-refactor-api-endpoints.md`
- スラッグはケバブケース（kebab-case）で、簡潔にプランの目的を表す
