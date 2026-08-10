# PRD: colab-ollama-webui

Google Colab 上に Ollama + Open WebUI を立て、会話履歴を Google Drive に永続化する再現可能なノートブック。

最終更新: 2026-08-10

---

## 1. 背景と目的

ローカルLLMを試す環境が欲しいが、手元にGPUがない。Colab の GPU を使いたいが、Colab はセッションが切れるとすべて消えるため、会話の連続性が保てない。

**目的**: ノートブックを上から実行するだけで、前回の続きから会話できる ChatGPT ライクな環境が立ち上がる状態をつくる。

セットアップの手数を減らすことよりも、**セッションをまたいで会話履歴が生き残ること**を最優先とする。

### 前提

- **Colab Pro を前提とする。** GPU の選択肢が広がり、セッション長も伸びる
- **コンピューティングユニット（CU）は設計上の制約として扱わない。** 同じ Colab Pro 上で A1111 を運用した実績から、この程度の使い方では CU を使い切らないことが確認済み。また放置してもセッションが自然に切れるため、意図しない課金が発生する構造にもなっていない
- **Pro+ は前提としない。** バックグラウンド実行（最長24時間）はタブを閉じても走り続けるが、使うときに立ち上げる運用ではメリットが起動待ちの解消しかない。起動待ちが苦になった場合も、Pro+ への課金より先に起動時間の短縮（venv構築結果のキャッシュ、モデルのDrive配置）を試す。判断は8章の実測後

---

## 2. 要件

| ID | 要件 | 備考 |
|---|---|---|
| R1 | Chat UI として Open WebUI を使う | UIの自作はしない |
| R2 | 会話履歴が永続化され、次セッションで続きから話せる | **最重要**。ここが崩れたら他が満たされても失敗 |
| R3 | cloudflared のトンネル経由で、ブラウザの別タブから使う | A1111 の Colab ノートブックと同じ体験 |
| R4 | 起動はノートブックのセル実行だけで完結する | 手動のファイル操作を挟まない |
| R5 | ノートブックは GitHub で管理し、Colab から直接開ける | 履歴が読める形で保つ |
| R6 | 使い終わりに、履歴を確実に保存して停止できる | R2 を成立させるための操作。定期同期だけに頼らない |
| R7 | 割り当てられたGPUのVRAM量を起動時に把握し、載るモデルを判断できる | 有料プランではGPUの種類が固定されない |

---

## 3. 非目標（やらないこと）

初版のスコープを守るため、以下は明示的に対象外とする。

- 外部への恒久的な公開、他ユーザーとの共有
- スマホ・別端末からの利用
- RAG、ドキュメント取り込み、ナレッジベース機能の作り込み
- モデルの量子化、GGUF変換、自作モデルの取り込み
- 高可用性。Quick Tunnel に稼働保証がないことは前提として受け入れる
- PostgreSQL 等への DB 移行。SQLite のまま進める
- アイドル検知による自動停止などの省リソース機構。手動停止で十分とする
- 24時間常時稼働。使うときに立ち上げる運用とする

---

## 4. 構成

```
Colab VM (GPU ランタイム / 割り当ては可変)
├── Ollama          :11434   モデル推論
├── Open WebUI      :8081    Python 3.11 venv 上で起動（8080 は Colab 内部サービスと衝突するため不可）
└── cloudflared             :8081 → https://<random>.trycloudflare.com
                                    ↑ これを別タブで開く

Google Drive (マウント)
└── MyDrive/colab-ollama-webui/
    ├── data/        Open WebUI の DATA_DIR のバックアップ先
    └── models/      Ollama のモデル置き場
```

Open WebUI からは Ollama を `localhost:11434` で掴むため、Ollama 側にトンネルは不要。

### GPU の扱い

有料プランでは T4 / L4 / A100 などが割り当てられうるが、**どれが来るかは保証されない**。特定のGPUを前提とした設計にはせず、起動時に `nvidia-smi` で VRAM を確認し、その範囲で動くモデルを選ぶ（R7）。

VRAMごとの目安（4bit量子化）:

| VRAM | 想定 |
|---|---|
| 16GB (T4) | 7〜8Bクラス |
| 24GB (L4) | 14Bクラスまで |
| 40GB+ (A100) | 32Bクラスも視野に入る |

### 永続化の方針

Open WebUI は会話履歴・ユーザー・設定を `DATA_DIR` 配下の SQLite (`webui.db`) に持つ。したがって **DATA_DIR を保全すれば R2 は満たされる**。独自のログ出力機構は実装しない。

ただし SQLite を Drive の FUSE マウント上で直接開くのはファイルロックの都合で危険なため、以下の運用とする。

1. **起動時**: Drive → ローカル (`/content/owui-data`) へコピー
2. **稼働中**: 一定間隔でローカル → Drive へ rsync（バックグラウンド）
3. **終了時**: 明示的に書き戻して停止するセル（R6）

**3が主たる保存経路、2は保険**という位置づけ。普段は自分で止めるので3が働くが、セッションが予期せず切れた場合に備えて2を用意する。同期間隔の初期値は5分とし、実測で調整する。

### モデルの置き場

**未決。実測して決める。**

一度は「起動のたびの pull は CU の無駄だから Drive に置く」と結論づけたが、CU を制約から外した以上この根拠は消える。判断軸は純粋に起動時間へ戻る。

- Drive 配置: ダウンロード不要だが、Drive の I/O が遅く初回ロードで不利になりうる
- 毎回 pull: 数GBのダウンロード待ちが毎回発生するが、ローカルディスクからのロードは速い

両方を実測して短いほうを採る。

---

## 5. 既知のリスクと対処方針

| # | リスク | 対処 | 状態 |
|---|---|---|---|
| 1 | Open WebUI の Python バージョン制約。公式ドキュメント上は 3.11 / 3.12 対応だが、3.11 のみとする情報もあり錯綜している。Colab の標準は 3.12 系 | 3.11 の venv を別途作り、そこに導入する | 解消（2026-08-10 実機で導入・起動を確認） |
| 2 | SQLite × Drive(FUSE) のファイルロック問題 | 4章の同期方式で回避 | 実装済み・実機未検証 |
| 3 | Quick Tunnel の取得失敗。Cloudflare 自身がアカウントなしトンネルの稼働保証はないと明言 | リトライを入れる。恒常的に失敗するなら ngrok を検討（初版ではリトライのみ） | 対処済み（リトライ＋疎通確認を実装。実測では1回で取得。DNS 反映ラグの罠は devlog 参照） |
| 4 | WebSocket が通らない可能性 | cloudflared 経由なら通る見込み。ダメなら `ENABLE_WEBSOCKET_SUPPORT=false` でポーリングに落とす | 解消（2026-08-10 ストリーミング表示を実機確認） |
| 5 | トンネルURLに認証がない。Open WebUI は初回アクセス者が管理者になる | 起動直後に自分でアカウントを作り、その後 `ENABLE_SIGNUP=false` を設定する。**手順に組み込む** | 対処方針確定 |
| 6 | **GPUの割り当てが不定。** A100 を指定しても L4 に降格される事例が報告されている | 特定GPU前提の設計をしない。起動時にVRAMを検出する（R7） | 対処方針確定 |

「未検証」の項目は、実機で確認するまで解決したものとして扱わない。

---

## 6. 完了条件

| ID | 条件 |
|---|---|
| AC1 | ノートブックを上から順に実行するだけで、トンネルURLが出力される |
| AC2 | そのURLを別タブで開き、Open WebUI 上でモデルと会話できる |
| AC3 | **ランタイムを削除して再度実行したとき、前回の会話がチャット一覧に残っており、続きから話せる** |
| AC4 | 停止セルを実行すると履歴が Drive に書き戻され、安全にランタイムを落とせる |
| AC5 | 起動時に割り当てGPUとVRAMが表示され、載るモデルが判断できる |
| AC6 | 2回目以降の起動時間が許容範囲に収まる（目標値は初回実測後に確定） |
| AC7 | ノートブックが GitHub 上にあり、Colab から直接開いて実行できる |

AC3 が本丸。ここが通らなければ他がすべて通っても未達とする。

---

## 7. リポジトリ構成

```
colab-ollama-webui/
├── PRD.md          このファイル
├── README.md       使い方（Colab で開くリンクを含む）
├── notebook.py     jupytext 形式（# %% 区切り）— こちらを正とする
├── scripts/        Drive 同期、起動オーケストレーション
└── devlog/         試行と失敗の記録
```

`.ipynb` は JSON のため、出力セルを含んだ差分が読めない。`.py` を正とし、そこからノートブックを生成する。`nbstripout` の導入も検討する。

---

## 8. 進め方

1. まず AC1〜AC2 を通す（履歴の永続化は後回し）
2. 5章のリスク1〜4 を実機で潰す。結果は都度 devlog に残す
3. 永続化と停止セルを組み込み、AC3・AC4 を通す
4. 起動時間を実測し、モデル置き場の方針を確定する

リスク1〜4はGPUなしランタイムでも確認できる部分が多い。GPUランタイムは起動も遅いため、**GPUが要る検証と要らない検証は分けて進める**と反復が速い。

Claude Code 側では動作検証ができない。実行と結果のフィードバックは人力で行う。

---

## 9. 参考

- [Quick Start — Open WebUI](https://docs.openwebui.com/getting-started/quick-start/) — `DATA_DIR` の指定方法
- [Backups — Open WebUI](https://docs.openwebui.com/tutorials/maintenance/backups/) — データディレクトリの中身、rsync による差分バックアップ
- [Developing Open WebUI](https://docs.openwebui.com/getting-started/advanced-topics/development/) — Python バージョンの対応状況
- [Cloudflare Quick Tunnels](https://trycloudflare.com/) — アカウント不要のトンネル
- [Running LLM on google colab and accessing it from anywhere](https://medium.com/@debashishrambhola/running-llm-on-google-colab-and-accessing-it-from-anywhere-a-setup-guide-f55d2240b8a9)（2025-02-10） — Colab 上で Python 3.11 venv を作る手順
- [Running Ollama on Google Colab Through Pinggy](https://pinggy.io/blog/running_ollama_on_google_colab_with_pinggy/)（2025-09-23） — Ollama + Open WebUI の全体構成。トンネル部分は読み替えが必要
- [Colabでウェブアプリを実行する](https://zenn.dev/ohtaman/articles/run_webapp_on_colab) — Colab のポートプロキシ（今回は不採用だが、フォールバック候補）
