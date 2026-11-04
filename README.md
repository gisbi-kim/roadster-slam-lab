# Roadster Spatial Lab

A one-clip reconstruction experiment: official **LingBot-Map** depth and camera poses, displayed in an interactive Three.js viewer. The first sequence is the 12.33-second ramp segment of [Flight POV | Roadster](https://www.youtube.com/watch?v=3NyyOZQzy_E), cropped to the large left view.

The viewer now defaults to **Kickflip** (original 4.833–13.900 seconds): 46 frames at 5 FPS, 55,322 filtered points, 7.16 seconds of GPU inference. Select **Ramp** to compare with the original 62-frame run. Each sequence has an independent coordinate system and arbitrary scale; they are not aligned or merged. Kickflip data lives in `public/data/kickflip/`, with full raw predictions retained locally in `runtime/kickflip/predictions.npz`.

To reproduce Kickflip, add `--title "Roadster / Kickflip" --source-start 4.833333333 --output public/data/kickflip --work runtime/kickflip` to the inference command and supply `02_킥플립.mp4` as the source.

## Viewer

```sh
npm ci
npm run dev
npm run build
```

Orbit, zoom, scrub frames, display accumulated/per-frame/all points, follow the predicted camera, and download PLY / camera-to-world matrices. The deployed GitHub Pages site serves precomputed results; GPU inference runs locally.

Site: https://gisbi-kim.github.io/roadster-slam-lab/ . Pushes to `main` build and deploy through `.github/workflows/pages.yml`.

## Reproduce inference

Use Python 3.11, an NVIDIA CUDA GPU, PyTorch 2.8.0 + CUDA 12.8 and torchvision 0.23.0. Clone [official LingBot-Map](https://github.com/Robbyant/lingbot-map), install its dependencies with `pip install -e PATH_TO_UPSTREAM`, and download `lingbot-map.pt` from [the official model repository](https://huggingface.co/robbyant/lingbot-map). The exact upstream commit and inference settings are recorded in `public/data/manifest.json`.

```sh
python scripts/reconstruct.py --source /path/to/ramp.mp4 --upstream /path/to/lingbot-map --checkpoint /path/to/lingbot-map.pt --output public/data --work runtime --fps 5
```

The source clip has no right-hand panels: 2628×2160 pixels, original timeline 13.900–26.233 seconds. Model input retains this view, resizing to patch-compatible dimensions. No invented camera calibration or synthetic replacement geometry is used.

Outputs: confidence-filtered RGB point cloud (`cloud.ply`), binary viewer arrays, full camera-to-world matrices / estimated intrinsics (`trajectory.json`), input previews, and provenance (`manifest.json`). Unfiltered depth, confidence, RGB and poses are retained locally in `runtime/predictions.npz`, excluded from Git. Checkpoints and original full-resolution video are also excluded.

## Interpretation and limitations

- **Observed first-run quality:** the ramp surface appears in separated/overlapping pieces and the trajectory is jagged. This run demonstrates the pipeline, not a reliable recovered map. All 62 predicted poses are retained, including the difficult smoke/end frames.
- This is a learned monocular reconstruction baseline, not a validated SLAM system or a metrically calibrated survey.
- Pose is **camera-to-world**, OpenCV axes (+X right, +Y down, +Z forward). Camera-to-robot extrinsics are unknown, so these are not robot-body poses.
- Scale is in arbitrary model units, **not meters**. There is no reference trajectory or ground-truth accuracy measurement.
- Strong fisheye distortion, smoke, darkness, reflections and camera-fixed bodywork violate standard reconstruction assumptions. Estimated geometry and motion can be wrong.
- Viewer points exclude peripheral/body regions using a fixed elliptical mask, very dark pixels, low-confidence predictions, and the furthest 2% of remaining depth per frame. This mask is not lens undistortion or semantic smoke segmentation.
- Consecutive views are fused using model poses without bundle adjustment or verified loop closure. Independent clips must be aligned and checked before merging.
- Third-party video and model retain their respective ownership/licenses. See the linked sources. This repository does not license those materials.

## Validation

`node scripts/verify-viewer.mjs http://127.0.0.1:5173/roadster-slam-lab/` exercises real data loading, frame selection, display modes, camera following, playback, desktop/mobile layout, and browser runtime errors. `npm run build` produces the static site.

Kickflip quality: 32 of 46 frames have zero points after the display filters. The largest predicted translation step is between clip-relative 8.8 and 9.0 seconds, about 15.4× the median step. This is a diagnostic, not a ground-truth error metric. The result is sparse and not a reliable recovered map.
