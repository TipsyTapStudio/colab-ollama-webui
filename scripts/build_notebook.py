#!/usr/bin/env python3
"""notebook.py (jupytext percent 形式) から notebook.ipynb を生成する。

依存ライブラリなしで動く簡易コンバータ。notebook.py が正で、.ipynb は常に
このスクリプトで生成し直す（出力セルを含まないため diff が汚れない）。

使い方:
    python scripts/build_notebook.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "notebook.py"
DST = ROOT / "notebook.ipynb"

CELL_RE = re.compile(r"^# %%(?: \[(\w+)\])?\s*$")


def strip_header(lines):
    """先頭の jupytext YAML ヘッダ (# --- ... # ---) を読み飛ばす。"""
    if not lines or lines[0].rstrip() != "# ---":
        return lines
    for i in range(1, len(lines)):
        if lines[i].rstrip() == "# ---":
            return lines[i + 1:]
    return lines


def split_cells(lines):
    cells = []
    current_type = None
    current = []
    for line in lines:
        m = CELL_RE.match(line.rstrip("\n"))
        if m:
            if current_type is not None:
                cells.append((current_type, current))
            current_type = m.group(1) or "code"
            current = []
        elif current_type is not None:
            current.append(line.rstrip("\n"))
    if current_type is not None:
        cells.append((current_type, current))
    return cells


def clean(cell_type, lines):
    if cell_type == "markdown":
        # 各行の "# " プレフィックスを剥がす
        lines = [re.sub(r"^# ?", "", l) for l in lines]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def to_source(lines):
    """nbformat の source は「改行付き行のリスト、最終行のみ改行なし」。"""
    return [l + "\n" for l in lines[:-1]] + lines[-1:] if lines else []


def main():
    lines = SRC.read_text(encoding="utf-8").splitlines()
    cells = []
    for i, (cell_type, raw) in enumerate(split_cells(strip_header(lines))):
        body = clean(cell_type, raw)
        cell = {
            "cell_type": cell_type if cell_type == "markdown" else "code",
            "id": f"cell-{i:02d}",  # 決定的な id にして diff を安定させる
            "metadata": {},
            "source": to_source(body),
        }
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",  # Colab で開いたとき GPU ランタイムを既定にする
            "colab": {"provenance": [], "toc_visible": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    DST.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{DST.name} を生成しました ({len(cells)} セル)")


if __name__ == "__main__":
    main()
