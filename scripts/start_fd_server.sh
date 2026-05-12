#!/bin/bash
# Start FastDeploy Server for PaddleOCR-VL-1.5 on Metax C500
# Usage: bash /data/FastDeploy/scripts/start_fd_server.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# export FD_METAX_KVCACHE_MEM=32768  # KV Cache 显存上限(MB)，设此值则跳过 gpu_memory_utilization 自动计算
# export FD_WORKER_ALIVE_TIMEOUT=300
export TOKENIZERS_PARALLELISM=false

pkill -9 -f "fastdeploy" 2>/dev/null

# Clean stale FD-specific shared memory files from previous runs
# Must clean BEFORE waiting for old process, because old process may have
# left leaked shm objects that block new process initialization
rm -f /tmp/shm/worker_healthy_* /tmp/shm/loaded_model_* /tmp/shm/kv_cache_* /tmp/shm/model_weights_* /tmp/shm/cache_ready_* /tmp/shm/worker_ready_* /tmp/shm/exist_prefill_* /tmp/shm/exist_tasks_* /tmp/shm/exist_swapped_* /tmp/shm/get_profile_* 2>/dev/null

# Wait for processes to exit and GPU VRAM to be fully released
MAX_WAIT=120
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if ! pgrep -f "fastdeploy" > /dev/null 2>&1; then
        break
    fi
    sleep 2
    WAITED=$((WAITED + 2))
done

# Wait for GPU driver to reclaim VRAM after process exit
sleep 10

LD_PRELOAD=/tmp/shm_redirect.so python3 -m fastdeploy.entrypoints.openai.api_server \
    --model /mnt/moark-models/PaddleOCR-VL-1.5 \
    --port 8118 \
    --max-model-len 4096 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 64 \
    --gpu-memory-utilization 0.92 \
    --tensor-parallel-size 1 \
    --workers 2 \
    --graph-optimization-config '{"graph_opt_level":0, "use_cudagraph":true}'

# --graph-optimization-config '{"graph_opt_level":0, "use_cudagraph":true, "cudagraph_only_prefill":true}'
