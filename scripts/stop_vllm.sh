#!/bin/bash
# Stop the vLLM server on $PORT (default 8000). Selects processes by comm=python with "vllm" and the port in their
# arguments, because `pkill -f vllm` would also match (and kill) the shell running this command.
PORT=${PORT:-8000}
pids=$(ps -eo pid=,comm=,args= | awk -v port="$PORT" '$2 ~ /^python/ && /vllm/ && /serve/ && $0 ~ "--port " port {print $1}')
if [ -z "$pids" ]; then
  echo "no vLLM server on port $PORT"
  exit 0
fi
echo "stopping vLLM (port $PORT): $pids"
kill $pids
for _ in $(seq 30); do
  kill -0 $pids 2>/dev/null || { echo stopped; exit 0; }
  sleep 1
done
echo "still running after 30s, sending SIGKILL"
kill -9 $pids
