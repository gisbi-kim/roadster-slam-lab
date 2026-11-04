#!/usr/bin/env bash
# Run after torch 2.8.0 + torchvision 0.23.0 and CUDA 12.8 are installed.
set -euo pipefail
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="/opt/roadster-env/bin:$CUDA_HOME/bin:$PATH"
export MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.9
export PIP_FIND_LINKS=/opt/roadster-wheels
TASK_ROOT=/mnt/c/Users/gsk620/Projects/roadster-slam-runtime
python -m pip install setuptools wheel ninja 'numpy==1.26.4' 'opencv-python<4.12' scipy matplotlib einops huggingface_hub safetensors tqdm pillow pyyaml natsort plyfile roma scikit-learn tensorboard pyglet pyrealsense2 evo faiss-cpu cython open3d gtsam-develop pytorch_metric_learning pytorch-lightning termcolor 'viser==0.2.23' omegaconf requests trimesh lz4 ftfy regex
python - <<'PY'
import torch,gtsam
print(torch.__version__,torch.cuda.is_available(),gtsam.SL4)
PY
# PyTorch's pip CUDA libraries supply headers not present in the minimal nvcc
# package. Make those headers available to upstream extension builds.
export CPATH="$(python -c 'import site,glob; print(":".join(glob.glob(site.getsitepackages()[0]+"/nvidia/*/include")))'):${CPATH:-}"
cd "$TASK_ROOT/mast3r-slam"
python - <<'PY'
from pathlib import Path
import re
p=Path('setup.py');s=p.read_text();s=re.sub(r'        "-gencode=arch=compute_\d+,code=sm_\d+",\n','',s);s=s.replace('extra_compile_args["nvcc"] = [','extra_compile_args["nvcc"] = [\n        "-gencode=arch=compute_89,code=sm_89",');p.write_text(s)
p=Path('thirdparty/mast3r/mast3r/retrieval/processor.py');s=p.read_text().replace("torch.load(modelname, 'cpu')","torch.load(modelname, 'cpu', weights_only=False)");p.write_text(s)
p=Path('thirdparty/mast3r/dust3r/croco/models/curope/kernels.cu');s=p.read_text().replace('tokens.type()', 'tokens.scalar_type()');p.write_text(s)
p=Path('mast3r_slam/backend/src/matching_kernels.cu');s=p.read_text().replace('D11.type()', 'D11.scalar_type()');p.write_text(s)
p=Path('mast3r_slam/backend/src/gn_kernels.cu');s=p.read_text().replace('torch::linalg::linalg_norm', 'at::linalg_norm');p.write_text(s)
PY
python -m pip install --no-build-isolation --no-deps -e thirdparty/mast3r
python -m pip install --no-build-isolation --no-deps thirdparty/mast3r/asmk
python -m pip install --no-build-isolation --no-deps thirdparty/mast3r/dust3r/croco/models/curope
python -m pip install --no-build-isolation --no-deps git+https://github.com/princeton-vl/lietorch.git
python -m pip install --no-build-isolation --no-deps -e .
python - <<'PY'
import lietorch,mast3r_slam_backends
print('MASt3R CUDA extensions loaded')
PY
