#!/bin/sh
# Single entry point that needs only `sh` and Python 3.10+ (no make, no pre-installed deps).
#   sh run.sh eval       triage all tickets with the live model -> eval/results.json
#   sh run.sh run        API + review UI on http://127.0.0.1:8000
#   sh run.sh test       unit tests (no model needed)
#   sh run.sh dev-eval   same eval on the synthetic dev set -> eval/dev/
# Dependencies go into ./.venv; if the machine can't create a venv (e.g. Debian
# without python3-venv) they go into the current interpreter's user site instead.
set -eu
cd "$(dirname "$0")"

VENV=.venv

pick_python() {
    for c in python3 python; do
        if command -v "$c" >/dev/null 2>&1 &&
            "$c" -c 'import sys; sys.exit(sys.version_info < (3, 10))' >/dev/null 2>&1; then
            echo "$c"
            return 0
        fi
    done
    return 1
}

venv_python() {
    if [ -x "$VENV/bin/python" ]; then echo "$VENV/bin/python"
    elif [ -x "$VENV/Scripts/python.exe" ]; then echo "$VENV/Scripts/python.exe"
    fi
}

SYS_PY=$(pick_python) || { echo "ERROR: Python 3.10+ is required (python3 or python on PATH)." >&2; exit 1; }
PY=""

install() {
    stamp="$VENV/.installed"
    PY=$(venv_python)
    if [ -n "$PY" ] && [ -f "$stamp" ] && [ ! requirements.txt -nt "$stamp" ]; then
        return 0
    fi
    if [ -z "$PY" ]; then
        "$SYS_PY" -m venv "$VENV" >/dev/null 2>&1 || true
        PY=$(venv_python)
    fi
    if [ -n "$PY" ] && "$PY" -m pip --version >/dev/null 2>&1; then
        echo "Installing dependencies into $VENV ..." >&2
        "$PY" -m pip install --quiet --disable-pip-version-check -r requirements.txt
        touch "$stamp"
        return 0
    fi

    # Fallback: no working venv on this machine.
    rm -rf "$VENV"
    PY="$SYS_PY"
    if "$PY" -c 'import fastapi, uvicorn, pydantic, openai, pytest' >/dev/null 2>&1; then
        return 0
    fi
    echo "note: can't create a virtualenv here; installing dependencies for $PY (user site)" >&2
    pip_install="$PY -m pip install --quiet --disable-pip-version-check"
    $pip_install --user -r requirements.txt 2>/dev/null ||
        $pip_install --user --break-system-packages -r requirements.txt 2>/dev/null ||
        $pip_install -r requirements.txt
}

case "${1:-}" in
    install) install ;;
    eval) install; exec "$PY" -m triage.evaluate ;;
    dev-eval) install; exec "$PY" -m triage.evaluate --tickets data/dev/tickets.json --labels data/dev/labels.json --out eval/dev ;;
    run) install; exec "$PY" -m triage.api ;;
    test) install; exec "$PY" -m pytest -q ;;
    *) echo "usage: sh run.sh {install|run|eval|dev-eval|test}" >&2; exit 2 ;;
esac
