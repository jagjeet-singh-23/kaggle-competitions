#!/bin/zsh
# Local MLflow tracking server and UI at http://127.0.0.1:5050 (5000 is macOS AirPlay).
#
# Full MLflow runs in uvx's own environment. The project .venv keeps only mlflow-skinny,
# because the full package would downgrade pyarrow 26 -> 25 there, and skinny cannot
# write sqlite itself, so clients log through this server:
#
#     zsh mlflow_server.sh &                                  # once per boot
#     MLFLOW_TRACKING_URI=http://127.0.0.1:5050 python3 ranker.py --cv ...
#
# Runs live in $CASMI_DATA/mlflow (data/ by default), git-ignored like every other
# derived artifact. A worktree points CASMI_DATA at the main checkout's data/, so all
# worktrees share one experiment history.
D=${CASMI_DATA:-${0:a:h}/data}/mlflow
mkdir -p $D/artifacts || exit 1
export MLFLOW_DISABLE_AGENT_HINT=1
exec caffeinate -ims uvx --from 'mlflow==3.17.2' mlflow server \
  --host 127.0.0.1 --port 5050 \
  --backend-store-uri sqlite:///$D/mlflow.db \
  --artifacts-destination $D/artifacts
