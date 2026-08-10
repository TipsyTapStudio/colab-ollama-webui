# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
#   accelerator: GPU
# ---

# %% [markdown]
# # colab-ollama-webui
#
# Colab 上に Ollama + Open WebUI を立ち上げ、cloudflared の Quick Tunnel 経由でブラウザの別タブから使う。
#
# 会話履歴は Google Drive（`MyDrive/colab-ollama-webui/data`）にバックアップされ、
# 次のセッションで復元される。ランタイムを削除しても続きから話せる。
#
# 使い方:
# 1. ランタイム → ランタイムのタイプを変更 → GPU を選択
# 2. 上から順にセルを実行（Google Drive のアクセス許可を求められたら許可する）
# 3. セル7に表示される URL を別タブで開き、「初回アクセス時にやること」に従う
# 4. 使い終わったらセル9で STOP にチェックを入れて実行 → ランタイムを削除

# %% [markdown]
# ## 0. 設定
#
# 使うモデルを右のプルダウンから選ぶ。一覧にないモデルは選択欄に直接入力もできる
# （Ollama や Hugging Face のモデル名。例: `llama3.1:8b`）。
# 変えたらこのセルを実行し直し、セル1とセル4も実行し直す。

# %%
#@title モデルと起動オプション { display-mode: "form" }
MODEL = "hf.co/HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive:Q4_K_M"  #@param ["auto", "qwen3:8b", "qwen3:4b", "qwen3:14b", "gemma3:12b", "hf.co/HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive:Q4_K_M", "hf.co/HauhauCS/Qwen3.5-27B-Uncensored-HauhauCS-Aggressive:Q4_K_M"] {allow-input: true}
#@markdown 既定は Uncensored 9B（T4 向け）。「auto」にすると割り当て GPU の VRAM からサイズを自動選択する。
#@markdown Uncensored 27B は L4(24GB)以上向け。タグ（:Q4_K_M）で失敗したらタグを消して試す。

#@markdown ---
#@markdown 画像を扱える gemma3:12b も一緒に入れる（qwen3 や Uncensored 版は画像非対応）:
LOAD_VISION_MODEL = True  #@param {type:"boolean"}

# %% [markdown]
# 以下は共通の定数とヘルパー（通常は編集不要）。

# %%
EXTRA_MODELS = ["gemma3:12b"] if LOAD_VISION_MODEL else []  # 追加で入れるモデル

WEBUI_PORT = 8081  # 8080 は Colab VM 自身の内部サービスが使っているため避ける
OLLAMA_URL = "http://127.0.0.1:11434"
DATA_DIR = "/content/owui-data"  # Open WebUI のデータ置き場（会話履歴の SQLite を含む）
VENV_DIR = "/content/owui-venv"
DRIVE_DATA = "/content/drive/MyDrive/colab-ollama-webui/data"  # Drive 上のバックアップ先
SYNC_INTERVAL_SEC = 300  # 稼働中の定期バックアップ間隔（PRD: 初期値5分、実測で調整）

import os
import re
import shutil
import subprocess
import threading
import time
import urllib.request

PROCS = {}


def run(cmd, check=True, env=None, heartbeat=False):
    """シェルコマンドを実行する。出力は Python 側で読み取ってセルに流す
    （fd 継承まかせにすると Colab でサブプロセスの出力が見えないことがあるため）。
    heartbeat=True にすると、出力のない長いコマンドでも30秒ごとに経過時間を表示する。"""
    print(f"$ {cmd}", flush=True)
    merged = {**os.environ, **(env or {})}
    start = time.time()
    p = subprocess.Popen(
        cmd, shell=True, env=merged,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, errors="replace",
    )
    hb_stop = None
    if heartbeat:
        hb_stop = threading.Event()

        def _hb():
            while not hb_stop.wait(30):
                print(f"  ...実行中（{time.time() - start:.0f}秒経過）", flush=True)

        threading.Thread(target=_hb, daemon=True).start()
    for line in p.stdout:
        print(line, end="", flush=True)
    p.wait()
    if hb_stop:
        hb_stop.set()
    if check and p.returncode != 0:
        raise RuntimeError(f"コマンドが失敗しました (exit {p.returncode}): {cmd}")
    return p.returncode


def spawn(name, cmd, env=None):
    """バックグラウンドで起動する。ログは /content/<name>.log へ。"""
    log = open(f"/content/{name}.log", "ab")
    merged = {**os.environ, **(env or {})}
    p = subprocess.Popen(
        cmd, shell=True, stdout=log, stderr=subprocess.STDOUT,
        env=merged, start_new_session=True,
    )
    PROCS[name] = p
    print(f"起動: {name} (pid {p.pid}, ログ: /content/{name}.log)")
    return p


def http_ok(url, timeout=3):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return 200 <= r.status < 300
    except Exception:
        return False


def wait_http(url, timeout, label, proc=None):
    """URL が応答するまで待つ。起動確認用。30秒ごとに経過を表示し、
    proc を渡された場合はプロセスが死んだ時点で即座に諦める。"""
    start = time.time()
    last_report = 0
    while time.time() - start < timeout:
        if http_ok(url):
            print(f"{label}: 起動を確認 ({time.time() - start:.0f}秒)")
            return True
        if proc is not None and proc.poll() is not None:
            print(f"{label}: プロセスが終了コード {proc.returncode} で落ちました")
            return False
        elapsed = time.time() - start
        if elapsed - last_report >= 30:
            print(f"{label}: 起動待ち... {elapsed:.0f}秒経過（初回は数分かかる）", flush=True)
            last_report = elapsed
        time.sleep(3)
    return False


def port_holder(port):
    """ポートを LISTEN しているプロセスの情報を返す（空文字列なら空きポート）。"""
    out = subprocess.run(
        f"ss -tlnp 'sport = :{port}'", shell=True, capture_output=True, text=True
    )
    lines = [l for l in out.stdout.strip().splitlines()[1:] if l.strip()]
    return "\n".join(lines)


def tail(name, n=40):
    """バックグラウンドプロセスのログ末尾を表示する。"""
    path = f"/content/{name}.log"
    if os.path.exists(path):
        with open(path, errors="replace") as f:
            print("".join(f.readlines()[-n:]))


def backup_to_drive(label="定期"):
    """DATA_DIR を Drive へバックアップする（R2 の心臓部）。
    SQLite は稼働中でも壊れないようオンラインバックアップ API でローカルに
    スナップショットを取り、それを Drive へコピーする（FUSE 上で SQLite を
    直接開かない）。その他のファイルは rsync で差分同期する。"""
    import sqlite3
    db = f"{DATA_DIR}/webui.db"
    if not os.path.exists(db):
        return False  # まだデータがない（初回起動前など）
    if not os.path.isdir("/content/drive/MyDrive"):
        return False  # Drive 未マウント
    os.makedirs(DRIVE_DATA, exist_ok=True)
    snap = "/content/webui.db.snapshot"
    src = sqlite3.connect(db)
    try:
        dst = sqlite3.connect(snap)
        with dst:
            src.backup(dst)
        dst.close()
    finally:
        src.close()
    shutil.copy2(snap, f"{DRIVE_DATA}/webui.db")
    os.remove(snap)
    subprocess.run(
        f"rsync -a --delete --exclude 'webui.db*' --exclude 'cache/' {DATA_DIR}/ {DRIVE_DATA}/",
        shell=True,
    )
    with open("/content/sync.log", "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {label}バックアップ完了\n")
    return True


print("設定完了")

# %% [markdown]
# ## 1. GPU / VRAM の確認（R7 / AC5）
#
# 有料プランでは GPU の種類が固定されないため、割り当てられた VRAM を見て載るモデルを決める。

# %%
cell_start = time.time()

VRAM_GB = 0
if shutil.which("nvidia-smi"):
    out = subprocess.run(
        "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits",
        shell=True, capture_output=True, text=True,
    )
    if out.returncode == 0 and out.stdout.strip():
        gpu_name, mem = [s.strip() for s in out.stdout.strip().splitlines()[0].split(",")]
        VRAM_GB = int(mem) / 1024
        print(f"GPU: {gpu_name} / VRAM: {VRAM_GB:.1f} GB")
if VRAM_GB == 0:
    print("GPU が見つかりません。CPU ランタイムです（トンネル等の検証は可能。推論は非常に遅い）")


def pick_model(vram_gb):
    if vram_gb >= 36:
        return "qwen3:32b"   # A100 40GB〜
    if vram_gb >= 20:
        return "qwen3:14b"   # L4 24GB
    if vram_gb >= 12:
        return "qwen3:8b"    # T4 16GB
    if vram_gb >= 6:
        return "qwen3:4b"
    return "qwen3:0.6b"      # CPU ランタイムでの動作確認用


RESOLVED_MODEL = pick_model(VRAM_GB) if MODEL == "auto" else MODEL
suffix = "（VRAM 量から自動選択）" if MODEL == "auto" else "（手動指定）"
print(f"使用モデル: {RESOLVED_MODEL} {suffix}")
print(f"所要時間: {time.time() - cell_start:.0f}秒")

# %% [markdown]
# ## 2. Google Drive のマウントと履歴の復元（R2）
#
# 会話履歴・アカウント・設定の実体は `DATA_DIR` 内の SQLite (`webui.db`)。
# Drive の FUSE マウント上で SQLite を直接開くのはファイルロックの都合で危険なため、
# ローカルにコピーして動かし、Drive へは定期的＋停止時に書き戻す（PRD 4章）。
# 初回は Drive へのアクセス許可のポップアップが出るので許可すること。

# %%
cell_start = time.time()

from google.colab import drive
drive.mount("/content/drive")

os.makedirs(DATA_DIR, exist_ok=True)
if os.path.exists(f"{DATA_DIR}/webui.db"):
    print("ローカルに稼働中のデータがあるため、Drive からの復元はスキップ（同一ランタイムでの再実行）")
elif os.path.exists(f"{DRIVE_DATA}/webui.db"):
    run(f"rsync -a {DRIVE_DATA}/ {DATA_DIR}/")
    print("Drive から前回の履歴を復元した")
else:
    print("Drive にバックアップなし（初回起動）。まっさらで始める")

print(f"所要時間: {time.time() - cell_start:.0f}秒")

# %% [markdown]
# ## 3. Ollama のインストールと起動

# %%
cell_start = time.time()

# 公式の install.sh は環境検査が多く Colab で失敗することがあるため、
# アーカイブを /usr/local に展開するだけの手動インストールにする。
# 配布形式は v0.32 時点で .tar.zst（.tgz は廃止済み）。展開に zstd が要る
if shutil.which("ollama") is None:
    if shutil.which("zstd") is None:
        run("apt-get install -y -qq zstd")
    run(
        "wget --progress=dot:giga -O /tmp/ollama.tar.zst "
        "https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tar.zst"
    )
    run("zstd -d -c /tmp/ollama.tar.zst | tar -xf - -C /usr/local", heartbeat=True)
    run("rm /tmp/ollama.tar.zst")
    run("ollama --version")
else:
    print("Ollama はインストール済み")

if http_ok(f"{OLLAMA_URL}/api/version"):
    print("Ollama は起動済み")
else:
    p = spawn("ollama", "ollama serve")
    if not wait_http(f"{OLLAMA_URL}/api/version", 60, "Ollama", proc=p):
        tail("ollama")
        raise RuntimeError("Ollama が起動しませんでした。上のログを確認してください")

print(f"所要時間: {time.time() - cell_start:.0f}秒")

# %% [markdown]
# ## 4. モデルの取得
#
# `MODEL` / `EXTRA_MODELS` を変えてこのセルを再実行すれば、モデルを追加できる。
# 会話ごとの切り替えは Open WebUI 画面上部のドロップダウンから。

# %%
cell_start = time.time()

run(f"ollama pull {RESOLVED_MODEL}")
for m in EXTRA_MODELS:
    run(f"ollama pull {m}")
run("ollama list")

print(f"所要時間: {time.time() - cell_start:.0f}秒")

# %% [markdown]
# ## 5. Open WebUI のインストール（Python 3.11 venv）
#
# Open WebUI の Python バージョン制約（リスク1）に対応するため、Colab 標準の Python とは
# 別に 3.11 の venv を作ってそこへ入れる。初回は数分かかる。

# %%
cell_start = time.time()

if shutil.which("python3.11") is None:
    run("apt-get update -qq")
    run("apt-get install -y -qq python3.11 python3.11-venv", check=False, heartbeat=True)
    if shutil.which("python3.11") is None:
        # 標準リポジトリに 3.11 がない場合（Ubuntu 24.04 など）は deadsnakes を使う
        run("add-apt-repository -y ppa:deadsnakes/ppa")
        run("apt-get update -qq")
        run("apt-get install -y -qq python3.11 python3.11-venv", heartbeat=True)
    run("apt-get install -y -qq python3.11-distutils", check=False)
run("python3.11 --version")

if os.path.exists(f"{VENV_DIR}/bin/open-webui"):
    print("Open WebUI はインストール済み")
else:
    run(f"python3.11 -m venv {VENV_DIR}")
    run(f"{VENV_DIR}/bin/pip install -q --upgrade pip", heartbeat=True)
    print("open-webui をインストール中（初回は5分ほどかかる。経過は30秒ごとに表示される）...")
    run(f"{VENV_DIR}/bin/pip install -q open-webui", heartbeat=True)

print(f"所要時間: {time.time() - cell_start:.0f}秒")

# %% [markdown]
# ## 6. Open WebUI の起動と定期バックアップの開始

# %%
cell_start = time.time()

WEBUI_ENV = {
    "DATA_DIR": DATA_DIR,
    "OLLAMA_BASE_URL": OLLAMA_URL,
    "ENABLE_OPENAI_API": "false",  # OpenAI API 連携は使わない
    "SCARF_NO_ANALYTICS": "true",
    "DO_NOT_TRACK": "true",
    "ANONYMIZED_TELEMETRY": "false",
    # "ENABLE_WEBSOCKET_SUPPORT": "false",  # リスク4: WebSocket が通らない場合はコメントを外して再起動
}

os.makedirs(DATA_DIR, exist_ok=True)

if http_ok(f"http://127.0.0.1:{WEBUI_PORT}/health"):
    print("Open WebUI は起動済み")
else:
    # 以前の実行の残骸（この venv から起動したプロセスとその子）を掃除してから起動する
    subprocess.run("pkill -f owui-venv", shell=True)
    time.sleep(3)
    holder = port_holder(WEBUI_PORT)
    if holder:
        print(f"ポート {WEBUI_PORT} を別のプロセスが使用中:\n{holder}")
        raise RuntimeError(
            f"ポート {WEBUI_PORT} が空いていません。上の表示を確認し、"
            "セル0の WEBUI_PORT を別の番号（例: 8082）に変えて上から実行し直してください"
        )
    p = spawn(
        "open-webui",
        f"{VENV_DIR}/bin/open-webui serve --host 127.0.0.1 --port {WEBUI_PORT}",
        env=WEBUI_ENV,
    )
    # 初回起動は内部 DB の初期化や埋め込みモデルの取得が走るため、長めに待つ
    if not wait_http(f"http://127.0.0.1:{WEBUI_PORT}/health", 600, "Open WebUI", proc=p):
        tail("open-webui")
        raise RuntimeError("Open WebUI が起動しませんでした。上のログを確認してください")

# 定期バックアップ（保険。主たる保存経路は停止セルの書き戻し — PRD 4章）
import threading

if "SYNC_THREAD" not in globals() or not SYNC_THREAD.is_alive():
    SYNC_STOP = threading.Event()

    def _sync_loop():
        while not SYNC_STOP.wait(SYNC_INTERVAL_SEC):
            try:
                backup_to_drive("定期")
            except Exception as e:
                with open("/content/sync.log", "a") as f:
                    f.write(f"{time.strftime('%H:%M:%S')} バックアップ失敗: {e}\n")

    SYNC_THREAD = threading.Thread(target=_sync_loop, daemon=True)
    SYNC_THREAD.start()
    print(f"定期バックアップを開始（{SYNC_INTERVAL_SEC // 60}分間隔、記録: /content/sync.log）")
else:
    print("定期バックアップは動作中")

print(f"所要時間: {time.time() - cell_start:.0f}秒")

# %% [markdown]
# ## 7. トンネルの発行（AC1）
#
# Quick Tunnel には稼働保証がない（リスク3）ため、URL が取れるまで最大5回リトライする。

# %%
cell_start = time.time()

# 前回実行分の cloudflared が残っていたら止める（URL は毎回変わるため取り直す）
subprocess.run("pkill -f 'cloudflared tunnel'", shell=True)

CLOUDFLARED = "/content/cloudflared"
if not os.path.exists(CLOUDFLARED):
    run(
        f"wget -q -O {CLOUDFLARED} "
        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    )
    run(f"chmod +x {CLOUDFLARED}")


def start_tunnel(port, attempts=5, wait_sec=30, live_sec=120):
    """トンネルを立て、URL が実際に外から疎通するまで確認してから返す。
    cloudflared は URL を予約した時点で表示するが、DNS 反映までラグがあるため、
    公開 URL 経由で /health が通るまで待つ。疎通しない URL は捨てて取り直す。"""
    log_path = "/content/cloudflared.log"
    for attempt in range(1, attempts + 1):
        open(log_path, "w").close()  # 前回のログが残っていると誤検出するためクリア
        proc = spawn("cloudflared", f"{CLOUDFLARED} tunnel --url http://127.0.0.1:{port} --no-autoupdate")
        url = None
        deadline = time.time() + wait_sec
        while time.time() < deadline:
            with open(log_path, errors="replace") as f:
                m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", f.read())
            if m:
                url = m.group(0)
                break
            if proc.poll() is not None:
                break  # プロセスが死んだら待たずに次の試行へ
            time.sleep(2)
        if url:
            print(f"URL を取得: {url} — 外からの疎通を確認中（DNS 反映に時間がかかることがある）...")
            if wait_http(f"{url}/health", live_sec, "トンネル", proc=proc):
                return url
            print(f"試行 {attempt}/{attempts}: URL は発行されたが疎通しません。トンネルを取り直します")
        else:
            print(f"試行 {attempt}/{attempts}: URL を取得できませんでした。リトライします")
        proc.kill()
    return None


TUNNEL_URL = start_tunnel(WEBUI_PORT)
if TUNNEL_URL:
    print("=" * 62)
    print(f"  Open WebUI: {TUNNEL_URL}")
    print("=" * 62)
    print("この URL を別タブで開く。初回アクセス時は次のセルの手順に従うこと。")
    print("開けない場合（DNS_PROBE_FINISHED_NXDOMAIN 等）は、1〜2分待ってから再読み込みする。")
else:
    tail("cloudflared")
    raise RuntimeError("トンネルを確立できませんでした。時間を置いて再実行を（恒常的に失敗するなら PRD リスク3の ngrok 検討へ）")

print(f"所要時間: {time.time() - cell_start:.0f}秒")

# %% [markdown]
# ## 初回アクセス時にやること
#
# **アカウント未作成（まっさらな Drive で始めた）ときだけ**の手順。
# 2回目以降のセッションでは、前回のアカウントでログインするだけでよい。
#
# - トンネル URL には認証がなく、**最初にサインアップした人が管理者になる**。
#   URL を開いたら、すぐに自分の管理者アカウントを作成する（メールアドレスは実在しなくてもよい）
# - 新規サインアップの無効化は不要（v0.11 は New Sign Ups が初期値オフ、新規登録者は保留承認制。
#   アカウントも設定も Drive に保存されて引き継がれる）

# %% [markdown]
# ## 8. ログを眺める（任意）
#
# サービスは裏で動いているため、通常はセルに何も流れない。A1111 のようにログを
# 流し見したいときは **WATCH にチェックを入れて実行**すると、3つのログを
# リアルタイムで表示し続ける（止めるにはセルの実行を停止 ■）。
# チェックなしで実行した場合は、各ログの末尾を1回表示するだけ。

# %%
WATCH = False  # @param {type:"boolean"}

if WATCH:
    print("ログを追跡中。止めるにはこのセルの実行を停止（■）する")
    _p = subprocess.Popen(
        "tail -n 5 -f /content/ollama.log /content/open-webui.log /content/cloudflared.log",
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, errors="replace",
    )
    try:
        for _line in _p.stdout:
            print(_line, end="", flush=True)
    finally:
        _p.kill()
else:
    for _name in ["ollama", "open-webui", "cloudflared"]:
        print(f"===== /content/{_name}.log の末尾 =====")
        tail(_name, 15)

# %% [markdown]
# ## 9. 停止（使い終わったら）— R6 / AC4
#
# **STOP にチェックを入れて（True にして）から実行**すると、履歴を Drive へ書き戻してから停止する。
# チェックなしでは何もしないため、「すべてのセルを実行」で誤ってサービスが止まることはない。
#
# ここが**主たる保存経路**（5分ごとの定期バックアップは保険）。
# カーネルを再起動した場合は、セル0とセル2を実行してからこのセルを使うこと。

# %%
STOP = False  # @param {type:"boolean"}

if not STOP:
    print("何もしていません。停止するには STOP にチェックを入れて（True にして）このセルを実行する")
else:
    # 先にトンネルと Open WebUI を止め、DB への書き込みが止まった状態で書き戻す
    subprocess.run("pkill -f 'cloudflared tunnel'", shell=True)
    subprocess.run("pkill -f owui-venv", shell=True)
    print("cloudflared / Open WebUI を停止した")
    time.sleep(5)
    if "SYNC_STOP" in globals():
        SYNC_STOP.set()
    if backup_to_drive("最終"):
        print(f"履歴を Drive へ書き戻した: {DRIVE_DATA}")
    else:
        print("警告: Drive への書き戻しができなかった（Drive 未マウント？ セル0とセル2を実行してから再試行を）")
    subprocess.run("pkill -f 'ollama serve'", shell=True)
    print("Ollama を停止した")
    print("停止完了。ランタイムを削除してよい（ランタイム → 接続解除してランタイムを削除）")
