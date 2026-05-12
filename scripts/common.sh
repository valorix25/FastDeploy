#!/bin/bash
# Common setup for Metax C500 environment and shm_redirect shim
# Usage: source /data/FastDeploy/scripts/common.sh

# ===== Metax C500 环境初始化 =====
export MACA_PATH=/opt/maca

if [ ! -d ${HOME}/cu-bridge ]; then
  ${MACA_PATH}/tools/cu-bridge/tools/pre_make
fi

export CUCC_PATH=/opt/maca/tools/cu-bridge
export CUCC_CMAKE_ENTRY=2
export CUDA_PATH=${HOME}/cu-bridge/CUDA_DIR
export PATH=${CUDA_PATH}/bin:${MACA_PATH}/mxgpu_llvm/bin:${MACA_PATH}/bin:${CUCC_PATH}/tools:${CUCC_PATH}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_PATH}/lib64:${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:$LD_LIBRARY_PATH
export MACA_VISIBLE_DEVICES="0"
export PADDLE_XCCL_BACKEND=metax_gpu
export FLAGS_weight_only_linear_arch=80
export FD_MOE_BACKEND=cutlass
export ENABLE_V1_KVCACHE_SCHEDULER=1
export FD_ENC_DEC_BLOCK_NUM=2
export FD_SAMPLING_CLASS=rejection
# ===== End Metax C500 环境初始化 =====

# Build shm_redirect.so if it doesn't exist.
# Redirects shm_open to /tmp/shm to avoid /dev/shm 64MB limit (Docker default).
if [ ! -f /tmp/shm_redirect.so ]; then
    cat > /tmp/shm_redirect.c << 'CEOF'
#define _GNU_SOURCE
#include <dlfcn.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <errno.h>
static const char* REDIRECT_DIR = "/tmp/shm";
static char* redirect_path(const char* name) {
    static __thread char buf[512];
    if (name[0] == '/') name++;
    snprintf(buf, sizeof(buf), "%s/%s", REDIRECT_DIR, name);
    return buf;
}
int shm_open(const char *name, int oflag, mode_t mode) {
    static int created = 0;
    if (!created) { mkdir(REDIRECT_DIR, 01777); created = 1; }
    int fd = open(redirect_path(name), oflag, mode);
    if (fd >= 0 && (oflag & O_CREAT)) { fchmod(fd, mode); }
    return fd;
}
int shm_unlink(const char *name) {
    return unlink(redirect_path(name));
}
CEOF
    gcc -shared -fPIC -o /tmp/shm_redirect.so /tmp/shm_redirect.c -ldl
fi
mkdir -p /tmp/shm

# Clean stale shared memory files from previous runs
rm -f /dev/shm/triton_* /dev/shm/paddle_* /dev/shm/__KMP_REGISTERED_LIB_* 2>/dev/null
rm -f /tmp/shm/triton_* /tmp/shm/paddle_* /tmp/shm/__KMP_REGISTERED_LIB_* 2>/dev/null
