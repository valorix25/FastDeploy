#!/bin/bash

pkill -9 -f "benchmark.py" 2>/dev/null
sleep 2

# Run PaddleOCR-VL-1.5 benchmark on OmniDocBench v1.5
# Prerequisite: FastDeploy server running on 127.0.0.1:8118
# Usage: bash /data/FastDeploy/scripts/run_benchmark.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

cd /data/FastDeploy/benchmarks/paddleocr_vl

LD_PRELOAD=/tmp/shm_redirect.so python3 benchmark.py \
    /data/OmniDocBench_v1_5/images_128 \
    -b 16 \
    -o output/br.images_128.NOW \
    --paddlex_config_path PaddleOCR-VL-1.5.yaml \
    --model_dir /mnt/moark-models/PaddleOCR-VL-1.5 \
    --device metax_gpu:0
