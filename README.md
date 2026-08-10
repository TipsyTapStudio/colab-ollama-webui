# colab-ollama-webui

Google Colab 上に Ollama + Open WebUI を立ち上げ、会話履歴を Google Drive に永続化する再現可能なノートブック。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GITHUB_USERNAME/colab-ollama-webui/blob/main/notebook.ipynb)

> GitHub へ push したら、上のリンクの `GITHUB_USERNAME` を自分のユーザー名に置き換えること。

## 現在のステータス

**フェーズ1**: トンネル URL の発行〜モデルとの会話（AC1〜AC2）まで。

会話履歴の Drive 永続化（AC3〜AC4、本プロジェクトの本丸）は未実装。ランタイムを削除すると履歴もアカウントも消える。計画の全体は [PRD.md](PRD.md)、試行の記録は [devlog/](devlog/) を参照。

## 使い方

1. 上の「Open in Colab」バッジからノートブックを開く（GPU ランタイムが既定で選ばれる）
2. 上から順にセルを実行する
3. セル6に表示される `https://xxx.trycloudflare.com` を別タブで開く

### 初回アクセスで必ずやること

トンネル URL には認証がなく、Open WebUI は最初にサインアップした人が管理者になる。

1. URL を開いたら、すぐに自分の管理者アカウントを作成する
2. 管理者パネル → 設定 → 一般 で「新規サインアップ」を無効化する

### 使い終わったら

ノートブック末尾の停止セルを実行してから、ランタイムを削除する。

## 構成

```
Colab VM (GPU ランタイム / 割り当ては可変)
├── Ollama          :11434   モデル推論
├── Open WebUI      :8080    Python 3.11 venv 上で起動
└── cloudflared             :8080 → https://<random>.trycloudflare.com
```

モデルは起動時に検出した VRAM 量から自動選択される（`MODEL` 変数で手動指定も可能）。

| VRAM | 自動選択されるモデル |
|---|---|
| 40GB〜 (A100) | qwen3:32b |
| 24GB (L4) | qwen3:14b |
| 16GB (T4) | qwen3:8b |
| GPU なし | qwen3:0.6b（動作確認用） |

## 開発

- [notebook.py](notebook.py)（jupytext percent 形式）が正。`notebook.ipynb` は生成物なので直接編集しない
- `.py` を編集したら `.ipynb` を再生成してコミットする:

```bash
python scripts/build_notebook.py
```

- jupytext を使う場合は `jupytext --to ipynb notebook.py` でも生成できる
