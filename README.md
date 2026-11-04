# Roadster Spatial Lab

[Open the GitHub Pages viewer](https://gisbi-kim.github.io/roadster-slam-lab/)

Compare learned reconstruction and SLAM on the same Kickflip clip from [Flight POV | Roadster](https://www.youtube.com/watch?v=3NyyOZQzy_E). The large left view is cropped to 2628×2160; original timeline 4.833–13.900 seconds. Explore point clouds and camera trajectories, scrub frames, switch between per-frame/cumulative/all points, follow the camera, and download PLY / JSON.

## Experiments

| Method | Input frames | Model inference | Peak allocated GPU | Displayed points |
|---|---:|---:|---:|---:|
| DA3 Large, 5 FPS | 46 | 4.43 s | 8.76 GB | 362,321 |
| DA3 Large, 15 FPS | 136 | 27.59 s | 15.74 GB | 1,083,641 |
| VGGT-SLAM 2.0, 5 FPS | 46 | 9.49 s (tracking + graph) | 13.35 GB | 469,586 |
| LingBot-Map, common filter | 46 | 7.16 s | See manifest | 188,587 |

MASt3R-SLAM execution state is recorded in [the experiment catalog](src/experiments.json). Each completed run has a `public/data/<run>/manifest.json` with settings, upstream commit, runtime and output statistics. The old LingBot Kickflip and Ramp exports remain available; their absolute confidence threshold differs from the common filter.

The 5 FPS runs use byte-identical JPEG inputs; MASt3R reads losslessly converted PNG copies. Each model applies its own required resize/crop. DA3 15 FPS is a separate denser-input experiment. More points do not establish better reconstruction: DA3 still produces layered/overlapping road surfaces. Without a reference trajectory this is not an accuracy benchmark. Maximum/median translation step includes genuine abrupt motion. DA3/LingBot timing is model inference; SLAM timing includes tracking and graph optimization, excluding model loading and export.

## Viewer

```sh
npm ci
npm run dev
npm run build
node scripts/verify-viewer.mjs
```

Pushes to main build and deploy through `.github/workflows/pages.yml`. GitHub Pages serves precomputed results; inference runs locally on an RTX 4070 Ti SUPER (16 GB).

## Reproduce inference

Checkpoints, original video and raw predictions are excluded from Git. Clone official sources and download official weights:

- [LingBot-Map](https://github.com/Robbyant/lingbot-map) / [weights](https://huggingface.co/robbyant/lingbot-map)
- [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) / [DA3-LARGE](https://huggingface.co/depth-anything/DA3-LARGE)
- [MASt3R-SLAM](https://github.com/rmurai0610/MASt3R-SLAM), recursive clone, windows branch / [MASt3R weights](https://huggingface.co/naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric), plus upstream retrieval checkpoint and codebook.
- [VGGT-SLAM](https://github.com/MIT-SPARK/VGGT-SLAM), with VGGT_SPARK and SALAD dependencies / [VGGT weights](https://huggingface.co/facebook/VGGT-1B).

LingBot/DA3 use Python 3.11 on Windows. SLAM adapters use Ubuntu 24.04 in WSL, Python 3.12. PyTorch 2.8.0 + CUDA 12.8, torchvision 0.23.0. `scripts/setup_slam_wsl.sh` records the local extension build setup; adjust its local paths before reuse. CUDA builds target SM89. The curope dispatch uses `scalar_type()` instead of the removed `type()` API. The trusted upstream retrieval checkpoint loader uses `weights_only=False` for its Python metadata under PyTorch 2.8.

```sh
python scripts/reconstruct.py --source kickflip.mp4 --upstream /path/to/lingbot-map --checkpoint /path/to/lingbot-map.pt --title 'Roadster / Kickflip' --source-start 4.833333333 --output public/data/kickflip --work runtime/kickflip --fps 5
python scripts/run_da3.py --source kickflip.mp4 --checkpoint /path/to/da3-large --upstream /path/to/da3 --fps 5 --output public/data/da3-5fps --work runtime/da3-5fps
# Repeat DA3 with --fps 15 and separate output/work directories.
python scripts/run_mast3r.py --upstream /path/to/mast3r-slam --checkpoint /path/to/mast3r-hf --images /path/to/kickflip-png --output /path/to/public/data/mast3r --work /path/to/runtime/mast3r
python scripts/run_vggt_slam.py --upstream /path/to/vggt-slam --checkpoint /path/to/vggt-model.pt --images /path/to/runtime/da3-5fps --output /path/to/public/data/vggt-slam --work /path/to/runtime/vggt-slam
```

MASt3R uses upstream uncalibrated tracking and optimization. Only GUI imports are replaced with the same message defaults. Keyframe storage is sized to input count + 2 instead of 512; timestamps use actual input times. It exports optimized keyframes, with no interpolated poses. World points use Sim(3), camera matrices use the upstream SE(3) conversion. No invented pinhole calibration is assigned to its ray model (`intrinsics: null`).

VGGT-SLAM uses 17-frame submaps with one shared frame, place retrieval and SL(4) graph optimization. Only RAFT input subsampling and the local GUI are disabled, to process every supplied 5 FPS frame. Shared frames are deduplicated after graph optimization.

## Data and limitations

`scripts/export_common.py` exports new experiments with confidence at or above the per-frame 60th percentile (ties retained) inside a fixed ellipse, RGB maximum > 30, 3-pixel stride and exclusion of the furthest 2%. Confidence values are not comparable across models; ranks are within each model/frame. Raw predictions remain in local `runtime/<run>/predictions.npz`.

- Poses are camera-to-world, OpenCV axes (+X right, +Y down, +Z forward). Unknown camera-to-robot extrinsics mean these are not robot-body poses.
- Runs have independent origins and arbitrary scale, not calibrated meters. Clips are not aligned or merged.
- Fisheye distortion, darkness, smoke, reflections and camera-fixed bodywork can produce incorrect depth and motion. The ellipse is neither undistortion nor semantic masking.
- The old LingBot absolute filter removed all points in 32 of 46 Kickflip frames. The common relative filter retains points in every frame, but does not repair the underlying geometry or poses.
- Point count and valid rotation matrices do not measure geometric accuracy.
- Third-party video, source code and weights retain their respective ownership/licenses; this repository does not relicense them.

## Validation

`scripts/verify-data.py public/data/<run>` checks binary counts, finite coordinates, SE(3) matrices, timestamps and frame references. Browser verification checks every ready experiment, downloads, controls, and desktop/mobile rendering. No ground-truth accuracy is claimed.
