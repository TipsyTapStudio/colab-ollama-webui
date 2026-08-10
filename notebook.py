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
# **現在はフェーズ1（AC1〜AC2）**: トンネル URL の発行とモデルとの会話まで。
# 会話履歴の Drive 永続化（AC3〜AC4）は未実装。ランタイムを削除すると履歴もアカウントも消える。
#
# 使い方:
# 1. ランタイム → ランタイムのタイプを変更 → GPU を選択
# 2. 上から順にセルを実行
# 3. セル6に表示される URL を別タブで開き、「初回アクセス時にやること」に従う

# %% [markdown]
# ## 0. 設定と共通ヘルパー

# %%
MODEL = "auto"  # "auto" なら VRAM 量から自動選択。例: "qwen3:8b", "gemma3:12b"

WEBUI_PORT = 8080
OLLAMA_URL = "http://127.0.0.1:11434"
DATA_DIR = "/content/owui-data"  # Open WebUI のデータ置き場（フェーズ3で Drive と同期する）
VENV_DIR = "/content/owui-venv"

import os
import re
import shutil
import subprocess
import time
import urllib.request

PROCS = {}


def run(cmd, check=True, env=None):
    """シェルコマンドを実行し、出力をそのままセルに流す。"""
    print(f"$ {cmd}", flush=True)
    merged = {**os.environ, **(env or {})}
    p = subprocess.run(cmd, shell=True, env=merged)
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


def wait_http(url, timeout, label):
    """URL が応答するまで待つ。起動確認用。"""
    start = time.time()
    while time.time() - start < timeout:
        if http_ok(url):
            print(f"{label}: 起動を確認 ({time.time() - start:.0f}秒)")
            return True
        time.sleep(3)
    return False


def tail(name, n=40):
    """バックグラウンドプロセスのログ末尾を表示する。"""
    path = f"/content/{name}.log"
    if os.path.exists(path):
        with open(path, errors="replace") as f:
            print("".join(f.readlines()[-n:]))


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
# ## 2. Ollama のインストールと起動

# %%
cell_start = time.time()

if shutil.which("ollama") is None:
    run("curl -fsSL https://ollama.com/install.sh | sh")
else:
    print("Ollama はインストール済み")

if http_ok(f"{OLLAMA_URL}/api/version"):
    print("Ollama は起動済み")
else:
    spawn("ollama", "ollama serve")
    if not wait_http(f"{OLLAMA_URL}/api/version", 60, "Ollama"):
        tail("ollama")
        raise RuntimeError("Ollama が起動しませんでした。上のログを確認してください")

print(f"所要時間: {time.time() - cell_start:.0f}秒")

# %% [markdown]
# ## 3. モデルの取得
#
# `MODEL` を変えてこのセルを再実行すれば、追加のモデルも取得できる。

# %%
cell_start = time.time()

run(f"ollama pull {RESOLVED_MODEL}")
run("ollama list")

print(f"所要時間: {time.time() - cell_start:.0f}秒")

# %% [markdown]
# ## 4. Open WebUI のインストール（Python 3.11 venv）
#
# Open WebUI の Python バージョン制約（リスク1）に対応するため、Colab 標準の Python とは
# 別に 3.11 の venv を作ってそこへ入れる。初回は数分かかる。

# %%
cell_start = time.time()

if shutil.which("python3.11") is None:
    run("apt-get update -qq")
    run("apt-get install -y -qq python3.11 python3.11-venv", check=False)
    if shutil.which("python3.11") is None:
        # 標準リポジトリに 3.11 がない場合（Ubuntu 24.04 など）は deadsnakes を使う
        run("add-apt-repository -y ppa:deadsnakes/ppa")
        run("apt-get update -qq")
        run("apt-get install -y -qq python3.11 python3.11-venv")
    run("apt-get install -y -qq python3.11-distutils", check=False)
run("python3.11 --version")

if os.path.exists(f"{VENV_DIR}/bin/open-webui"):
    print("Open WebUI はインストール済み")
else:
    run(f"python3.11 -m venv {VENV_DIR}")
    run(f"{VENV_DIR}/bin/pip install -q --upgrade pip")
    print("open-webui をインストール中（数分かかる）...")
    run(f"{VENV_DIR}/bin/pip install -q open-webui")

print(f"所要時間: {time.time() - cell_start:.0f}秒")

# %% [markdown]
# ## 5. Open WebUI の起動

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
    spawn(
        "open-webui",
        f"{VENV_DIR}/bin/open-webui serve --host 127.0.0.1 --port {WEBUI_PORT}",
        env=WEBUI_ENV,
    )
    # 初回起動は内部 DB の初期化や埋め込みモデルの取得が走るため、長めに待つ
    if not wait_http(f"http://127.0.0.1:{WEBUI_PORT}/health", 600, "Open WebUI"):
        tail("open-webui")
        raise RuntimeError("Open WebUI が起動しませんでした。上のログを確認してください")

print(f"所要時間: {time.time() - cell_start:.0f}秒")

# %% [markdown]
# ## 6. トンネルの発行（AC1）
#
# Quick Tunnel には稼働保証がない（リスク3）ため、URL が取れるまで最大5回リトライする。

# %%
cell_start = time.time()

CLOUDFLARED = "/content/cloudflared"
if not os.path.exists(CLOUDFLARED):
    run(
        f"wget -q -O {CLOUDFLARED} "
        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    )
    run(f"chmod +x {CLOUDFLARED}")


def start_tunnel(port, attempts=5, wait_sec=30):
    log_path = "/content/cloudflared.log"
    for attempt in range(1, attempts + 1):
        open(log_path, "w").close()  # 前回のログが残っていると誤検出するためクリア
        proc = spawn("cloudflared", f"{CLOUDFLARED} tunnel --url http://127.0.0.1:{port} --no-autoupdate")
        deadline = time.time() + wait_sec
        while time.time() < deadline:
            with open(log_path, errors="replace") as f:
                m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", f.read())
            if m:
                return m.group(0)
            if proc.poll() is not None:
                break  # プロセスが死んだら待たずに次の試行へ
            time.sleep(2)
        proc.kill()
        print(f"試行 {attempt}/{attempts}: URL を取得できませんでした。リトライします")
    return None


TUNNEL_URL = start_tunnel(WEBUI_PORT)
if TUNNEL_URL:
    print("=" * 62)
    print(f"  Open WebUI: {TUNNEL_URL}")
    print("=" * 62)
    print("この URL を別タブで開く。初回アクセス時は次のセルの手順に従うこと。")
else:
    tail("cloudflared")
    raise RuntimeError("トンネルを確立できませんでした。時間を置いて再実行を（恒常的に失敗するなら PRD リスク3の ngrok 検討へ）")

print(f"所要時間: {time.time() - cell_start:.0f}秒")

# %% [markdown]
# ## 初回アクセス時にやること（リスク5対応）
#
# トンネル URL には認証がなく、Open WebUI は**最初にサインアップした人が管理者になる**。
#
# 1. URL を開いたら、すぐに自分の管理者アカウントを作成する（メールアドレスは実在しなくてもよい）
# 2. 左下のユーザー名 → **管理者パネル → 設定 → 一般** で **新規サインアップを無効化** する
#
# ※ フェーズ1では履歴もアカウントも永続化されないため、ランタイムを削除するとやり直しになる。

# %% [markdown]
# ## 7. 停止
#
# フェーズ1版: プロセスを止めるだけ。Drive への履歴書き戻し（R6 / AC4）はフェーズ3でここに入る。
# このセルは単独で実行できる（再接続後などにセル0を実行し直す必要はない）。

# %%
import subprocess

for label, pattern in [
    ("cloudflared", "cloudflared tunnel"),
    ("Open WebUI", "open-webui serve"),
    ("Ollama", "ollama serve"),
]:
    subprocess.run(f"pkill -f '{pattern}'", shell=True)
    print(f"{label} を停止しました")

print("停止完了。ランタイムを削除してよい（ランタイム → セッションの管理）")
