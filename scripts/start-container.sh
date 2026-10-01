#!/bin/sh
set -eu

case "${ENABLE_JUPYTER:-0}" in
    1|true|yes) set -- python3 -m jupyterlab --ip=0.0.0.0 --port="${JUPYTER_PORT:-8888}" --no-browser --allow-root ;;
    0|false|no) ;;
    *) echo "ENABLE_JUPYTER must be 0/1, false/true, or no/yes" >&2; exit 2 ;;
esac

# This is the standard entrypoint used by NVIDIA framework containers.
exec "$@"
