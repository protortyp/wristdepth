#!/usr/bin/env bash
# Selected frames -> metric, robot-registered Gaussian splat.
#   research/wrist_depth/twin/run_twin.sh $WRIST_DEPTH_DATA/splat_v2 [viewer_port]
# Expects SPLAT_DIR/images (from select_frames.py). Writes poses, alignment, robot registration,
# and the trained splat (gs_out/ply) into SPLAT_DIR. Logs go to SPLAT_DIR/*.log.
set -euo pipefail
work=$(cd "${1:?splat dir with images/}" && pwd)
port=${2:-8092}
here=$(cd "$(dirname "$0")" && pwd)
uid=$(id -u)
gid=$(id -g)

echo "[1/5] COLMAP"
docker run --rm --device nvidia.com/gpu=all -v "$work":/w -w /w colmap/colmap:latest bash -c "
set -e
rm -rf database.db sparse dense && mkdir -p sparse
colmap feature_extractor --database_path database.db --image_path images \
  --ImageReader.single_camera 1 --ImageReader.camera_model OPENCV \
  --FeatureExtraction.use_gpu 1 --SiftExtraction.max_num_features 8192
colmap exhaustive_matcher --database_path database.db --FeatureMatching.use_gpu 1 \
  --FeatureMatching.guided_matching 1
# Video frames carry no focal length; calibrating the view graph first gives the global mapper
# the focal prior it needs (2.5 px -> 1.1 px reprojection error on the first capture).
colmap view_graph_calibrator --database_path database.db
colmap global_mapper --database_path database.db --image_path images --output_path sparse
colmap model_analyzer --path sparse/0
colmap image_undistorter --image_path images --input_path sparse/0 --output_path dense \
  --output_type COLMAP
mkdir -p dense/sparse_txt
colmap model_converter --input_path dense/sparse --output_path dense/sparse_txt --output_type TXT
chown -R $uid:$gid /w
" > "$work/sfm.log" 2>&1
grep -E "Registered images|Points:|Mean reprojection" "$work/sfm.log" | tail -3

cd "$here"
echo "[2/5] board alignment"
uv run --no-sync python align_board.py "$work"
echo "[3/5] robot registration"
uv run --no-sync python register_robot.py "$work" "${SO101_ROOT:-../../so101}"

echo "[4/5] gsplat data"
# Window views and other far-off frames get registered metres away or under the desk; they
# blow up the trainer's scene scale (2 km on the second capture) and MCMC diverges.
uv run --no-sync python filter_cameras.py "$work"
rm -rf "$work/gs_data/sparse/0"
mkdir -p "$work/gs_data/sparse/0"
ln -sfn ../dense/images "$work/gs_data/images"
docker run --rm -v "$work":/w -w /w --user "$uid:$gid" colmap/colmap:latest \
  colmap model_converter --input_path dense/sparse_metric --output_path gs_data/sparse/0 \
  --output_type BIN > /dev/null 2>&1

echo "[5/5] gsplat MCMC training, viewer on :$port"
export TORCH_CUDA_ARCH_LIST=8.6
uv run --no-sync python ../../../third_party/gsplat/examples/simple_trainer.py mcmc \
  --data-dir "$work/gs_data" --data-factor 1 --result-dir "$work/gs_out" \
  --no-normalize-world-space --port "$port" --save-ply --antialiased --use-bilateral-grid \
  --strategy.cap-max 3000000 --steps-scaler 1.5 > "$work/train.log" 2>&1
grep -E "PSNR" "$work/train.log" | tail -2
