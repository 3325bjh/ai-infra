#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#SBATCH --job-name=mini-qwen36
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=00:30:00
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --partition=batch
#SBATCH --output=<SHARED_FS>/logs/mini_qwen36_%j.log
#SBATCH --exclusive

set -euo pipefail

CONTAINER_IMAGE="<PATH_TO_CONTAINER>.sqsh"
CONTAINER_MOUNTS="<SHARED_FS>:<SHARED_FS>,<PATH_TO_REPO>:/opt/Megatron-Bridge"
WORKDIR="/opt/Megatron-Bridge/mini-megatron"
LOGDIR="<SHARED_FS>/logs/mini_qwen36"

export MASTER_ADDR="$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)"
export MASTER_PORT="$((29500 + SLURM_JOB_ID % 1000))"
export OMP_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=0

mkdir -p "$LOGDIR"

srun --mpi=pmix \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts="$CONTAINER_MOUNTS" \
  --no-container-mount-home \
  bash -lc "cd $WORKDIR && uv run python scripts/train_qwen36.py \
    --recipe qwen36_35b_a3b_tiny_config \
    --train-iters 20 \
    --dtype bf16 \
    --save-dir $LOGDIR/checkpoints"
