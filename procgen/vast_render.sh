#!/usr/bin/env bash
# On a Vast.ai GPU instance: render procedural wrist scenes in parallel shards.
#   bash vast_render.sh FIRST_SHARD SHARDS SCENES_PER_SHARD JOBS_PER_GPU
# Expects /workspace/bundle.tar with assets_pack/, gripper_cad/ and procgen/ + synth/ sources.
set -euo pipefail
first=$1 shards=$2 scenes=$3 jobs=${4:-2}
cd /workspace
[ -d assets_pack ] || tar -xf bundle.tar
apt-get update -qq && apt-get install -y -qq libxrender1 libxi6 libxkbcommon0 libsm6 libgl1 libxxf86vm1 \
  libxfixes3 libegl1 curl ca-certificates > /dev/null
command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null
export PATH="$HOME/.local/bin:$PATH"
gpus=$(nvidia-smi -L | wc -l)
mkdir -p renders logs
# The first run downloads Blender 4.2.1; do it once before the shards start in parallel.
uvx --python 3.12 --from blenderproc blenderproc quickstart > logs/quickstart.log 2>&1 || true
run_shard() {
  local shard=$1 gpu=$2
  CUDA_VISIBLE_DEVICES=$gpu uvx --python 3.12 --from blenderproc blenderproc run procgen/render.py \
    --assets assets_pack --gripper gripper_cad --out renders --scenes "$scenes" --seed "$shard" \
    > logs/shard_$shard.log 2>&1
}
pids=()
for ((i = 0; i < shards; i++)); do
  run_shard $((first + i)) $((i % gpus)) &
  pids+=($!)
  # Keep at most gpus*jobs shards running.
  while [ "$(jobs -rp | wc -l)" -ge $((gpus * jobs)) ]; do sleep 5; done
done
wait
echo RENDER_DONE > logs/done
