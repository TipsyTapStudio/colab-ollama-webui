# colab-ollama-webui

Google Colab 上に Ollama + Open WebUI を立ち上げ、会話履歴を Google Drive に永続化する再現可能なノートブック。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TipsyTapStudio/colab-ollama-webui/blob/main/notebook.ipynb)

## 現在のステータス

- **フェーズ1（AC1〜AC2）達成**: トンネル URL の発行〜モデルとの会話まで、実機で確認済み（2026-08-10）
- **フェーズ3 実装済み・検証中**: 会話履歴の Google Drive 永続化（AC3〜AC4、本プロジェクトの本丸）

計画の全体は [PRD.md](PRD.md)、試行の記録は [devlog/](devlog/) を参照。

## 使い方

1. 上の「Open in Colab」バッジからノートブックを開く（GPU ランタイムが既定で選ばれる）
2. 上から順にセルを実行する（Google Drive のアクセス許可を求められたら許可する）
3. セル7に表示される `https://xxx.trycloudflare.com` を別タブで開く

### 初回アクセスで必ずやること

トンネル URL には認証がなく、Open WebUI は最初にサインアップした人が管理者になる。

1. URL を開いたら、すぐに自分の管理者アカウントを作成する
2. 管理者パネル → 設定 → 一般 で「新規サインアップ」を無効化する

### 使い終わったら

ノートブック末尾の停止セルで `STOP` にチェックを入れて実行すると、会話履歴が Google Drive（`MyDrive/colab-ollama-webui/data`）へ書き戻される。その後にランタイムを削除する（チェックなしで実行しても何もしないので、「すべてのセルを実行」に巻き込まれても安全）。

## 構成

```
Colab VM (GPU ランタイム / 割り当ては可変)
├── Ollama          :11434   モデル推論
├── Open WebUI      :8081    Python 3.11 venv 上で起動（8080 は Colab 内部サービスと衝突する）
└── cloudflared             :8081 → https://<random>.trycloudflare.com

Google Drive (マウント)
└── MyDrive/colab-ollama-webui/
    └── data/    会話履歴・アカウント（起動時に復元、稼働中は5分ごと＋停止時に書き戻し）
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
