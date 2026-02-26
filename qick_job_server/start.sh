#!/bin/bash
# ============================================================
# QICK Job Server — Startup Script
# ============================================================
# Run this from:  /Users/jay/Desktop/test/SQC_soc
#
# This starts both the API server and the worker.
# You can also run them in separate terminals:
#
#   Terminal 1 (Server):
#     python -m uvicorn qick_job_server.server:app --host 0.0.0.0 --port 8585
#
#   Terminal 2 (Worker):
#     python -m qick_job_server.worker --ns-host 192.168.10.179 --ns-port 8887
#
# ============================================================

set -e
cd "$(dirname "$0")/.."

echo "=== Starting QICK Job Server ==="
echo ""
echo "API docs will be at: http://localhost:8585/docs"
echo ""

# Start server in background
echo "[1/2] Starting FastAPI server on port 8585..."
python -m uvicorn qick_job_server.server:app --host 0.0.0.0 --port 8585 &
SERVER_PID=$!
echo "  Server PID: $SERVER_PID"

# Wait for server to be ready
sleep 2

# Start worker
echo "[2/2] Starting worker..."
echo "  Connecting to QICK via Pyro4..."
python -m qick_job_server.worker \
    --ns-host "${NS_HOST:-192.168.10.179}" \
    --ns-port "${NS_PORT:-8887}" \
    --proxy-name "${PROXY_NAME:-myqick}"

# If worker exits, kill the server too
echo "Worker exited. Stopping server..."
kill $SERVER_PID 2>/dev/null || true
