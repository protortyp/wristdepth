#!/usr/bin/env bash
# On a Vast.ai instance (pytorch 2.8 cu128 devel image): gsplat 1.5.3 + example deps, patched trainers.
set -euo pipefail
cd /workspace
tar -xf bundle.tar
export TORCH_CUDA_ARCH_LIST="$(python -c 'import torch; c=torch.cuda.get_device_capability(); print(f"{c[0]}.{c[1]}")')"
echo "arch $TORCH_CUDA_ARCH_LIST" > setup.log
pip install -q ninja gsplat==1.5.3 "numpy<2" viser imageio[ffmpeg] scikit-learn tqdm "torchmetrics[image]" \
  opencv-python-headless "tyro>=0.8.8" pillow tensorboard tensorly pyyaml matplotlib splines \
  "pycolmap @ git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e" \
  "nerfview @ git+https://github.com/nerfstudio-project/nerfview@4538024fe0d15fd1a0e4d760f3695fc44ca72787" >> setup.log 2>&1
pip install -q --no-build-isolation "fused-ssim @ git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5" >> setup.log 2>&1
git clone -q --depth 1 --branch v1.5.3 https://github.com/nerfstudio-project/gsplat.git gsplat_src
python timelapse_patch.py gsplat_src/examples/simple_trainer.py
python timelapse_patch.py gsplat_src/examples/simple_trainer_2dgs.py
python -c "
import torch, gsplat
m=torch.randn(10,3,device='cuda'); q=torch.randn(10,4,device='cuda'); s=torch.rand(10,3,device='cuda')*.1
o=torch.rand(10,device='cuda'); c=torch.rand(10,3,device='cuda')
K=torch.tensor([[[100.,0,50],[0,100,50],[0,0,1]]],device='cuda'); V=torch.eye(4,device='cuda')[None]; V[0,2,3]=3
print('gsplat ok', gsplat.rasterization(m,q,s,o,c,V,K,100,100)[0].shape)" >> setup.log 2>&1
echo SETUP_DONE >> setup.log
