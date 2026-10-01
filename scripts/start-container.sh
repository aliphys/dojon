#!/bin/sh
set -eu

ollama_pid=""

cleanup() {
    if [ -n "${ollama_pid}" ]; then
        kill "${ollama_pid}" 2>/dev/null || true
        wait "${ollama_pid}" 2>/dev/null || true
    fi
}

trap cleanup EXIT INT TERM

case "${ENABLE_OLLAMA:-0}" in
    1|true|yes)
        mkdir -p "${OLLAMA_MODELS:-/root/.ollama}"
        ollama serve &
        ollama_pid=$!

        i=0
        until curl -fsS "http://127.0.0.1:${OLLAMA_PORT:-11434}/api/tags" >/dev/null 2>&1; do
            i=$((i + 1))
            if [ "$i" -ge 60 ]; then
                echo "Ollama did not become ready" >&2
                exit 1
            fi
            sleep 1
        done
        ;;
    0|false|no) ;;
    *) echo "ENABLE_OLLAMA must be 0/1, false/true, or no/yes" >&2; exit 2 ;;
esac

case "${ENABLE_JUPYTER:-0}" in
    1|true|yes) set -- python3 -m jupyterlab --ip=0.0.0.0 --port="${JUPYTER_PORT:-8888}" \
        --ServerApp.base_url="${JUPYTER_BASE_URL:-/}" --no-browser --allow-root ;;
    0|false|no) ;;
    *) echo "ENABLE_JUPYTER must be 0/1, false/true, or no/yes" >&2; exit 2 ;;
esac

# This is the standard entrypoint used by NVIDIA framework containers.
exec "$@"
