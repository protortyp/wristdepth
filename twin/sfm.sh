#!/usr/bin/env bash
# Camera poses for the desk capture with COLMAP 4.3 (GPU SIFT, exhaustive matching, global mapper).
#   research/wrist_depth/twin/sfm.sh $WRIST_DEPTH_DATA/splat
# Both clips come from the same lens at the same resolution, so they share one OPENCV camera.
set -euo pipefail
work=$(cd "${1:?splat dir with images/}" && pwd)
docker run --rm --device nvidia.com/gpu=all -v "$work":/w -w /w colmap/colmap:latest bash -c '
set -e
rm -rf database.db sparse dense && mkdir -p sparse
colmap feature_extractor --database_path database.db --image_path images \
  --ImageReader.single_camera 1 --ImageReader.camera_model OPENCV \
  --FeatureExtraction.use_gpu 1 --SiftExtraction.max_num_features 8192
colmap exhaustive_matcher --database_path database.db --FeatureMatching.use_gpu 1 --FeatureMatching.guided_matching 1
colmap global_mapper --database_path database.db --image_path images --output_path sparse
colmap model_analyzer --path sparse/0
colmap image_undistorter --image_path images --input_path sparse/0 --output_path dense --output_type COLMAP
'
