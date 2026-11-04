"""Run official LingBot-Map on one clip; export measured predictions, never synthetic geometry."""
import argparse, json, sys, time, hashlib, subprocess
from pathlib import Path
import cv2
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument('--source', required=True)
p.add_argument('--upstream', required=True)
p.add_argument('--checkpoint', required=True)
p.add_argument('--output', default='public/data')
p.add_argument('--work', default='runtime')
p.add_argument('--fps', type=int, default=5)
p.add_argument('--image-size', type=int, default=518)
p.add_argument('--title', default='Roadster / Ramp')
p.add_argument('--source-start', type=float, default=13.9)
a = p.parse_args()
sys.path.insert(0, a.upstream)
from lingbot_map.models.gct_stream import GCTStream
from lingbot_map.utils.load_fn import load_and_preprocess_images
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri

out, work = Path(a.output), Path(a.work)
out.mkdir(parents=True, exist_ok=True)
(out/'frames').mkdir(exist_ok=True)
work.mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture(a.source)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
stride = max(1, round(fps/a.fps))
paths, times = [], []
for i in range(frame_count):
    ok, im = cap.read()
    if not ok: break
    if i % stride: continue
    path = work/f'frame_{i:06d}.jpg'
    cv2.imwrite(str(path), im)
    paths.append(str(path)); times.append(i/fps)
cap.release()
images = load_and_preprocess_images(paths, mode='crop', image_size=a.image_size, patch_size=14)
print('Input tensor:', tuple(images.shape), flush=True)
torch.set_num_threads(6)
model = GCTStream(img_size=a.image_size, patch_size=14, enable_3d_rope=True,
    max_frame_num=1024, kv_cache_sliding_window=16, kv_cache_scale_frames=2,
    kv_cache_cross_frame_special=True, kv_cache_include_scale_frames=True,
    use_sdpa=True, camera_num_iterations=4)
print('Loading official checkpoint on CPU', flush=True)
ckpt = torch.load(a.checkpoint, map_location='cpu', weights_only=True, mmap=True)
state = ckpt.get('model', ckpt)
missing, unexpected = model.load_state_dict(state, strict=False)
print('Missing / unexpected keys:', missing, unexpected, flush=True)
if missing: raise RuntimeError('Checkpoint missing model weights')
del ckpt, state
model = model.eval().cuda()
torch.cuda.reset_peak_memory_stats()
start = time.time()
with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
    pred = model.inference_streaming(images, num_scale_frames=2,
        keyframe_interval=2, output_device=torch.device('cpu'))
elapsed = time.time()-start
pose, intrinsic = pose_encoding_to_extri_intri(pred['pose_enc'].float(), images.shape[-2:])
poses, intrinsics = pose[0].numpy(), intrinsic[0].numpy()
depth = pred['depth'][0].float().numpy()[...,0]
conf = pred['depth_conf'][0].float().numpy()
rgb = (images.permute(0,2,3,1).numpy()*255).round().clip(0,255).astype(np.uint8)
np.savez_compressed(work/'predictions.npz', camera_to_world=poses, intrinsics=intrinsics,
    depth=depth, confidence=conf, rgb=rgb, times=np.array(times))
print('Raw predictions saved', flush=True)
n,h,w = depth.shape
yy,xx = np.mgrid[:h,:w]
# Exclude outer fisheye/body boundary; this is a visualization mask, not calibration.
interior = ((xx-w/2)/(w*.46))**2+((yy-h*.47)/(h*.44))**2 < 1
points, colors, frame_ids, scores, frames = [],[],[],[],[]
for i in range(n):
    d, c, k, pose = depth[i], conf[i], intrinsics[i], poses[i]
    valid = interior & np.isfinite(d) & (d>0) & np.isfinite(c) & (rgb[i].max(-1)>30)
    cutoff = max(1.5, float(np.quantile(c[valid], .6))) if valid.any() else 1.5
    valid &= c>=cutoff
    valid &= (xx%2==0)&(yy%2==0)
    if valid.any(): valid &= d <= np.quantile(d[valid], .98)
    camera = np.stack([(xx-k[0,2])*d/k[0,0],(yy-k[1,2])*d/k[1,1],d],-1)
    world = camera@pose[:3,:3].T+pose[:3,3]
    points.append(world[valid]); colors.append(rgb[i][valid]); scores.append(c[valid])
    frame_ids.append(np.full(valid.sum(),i,np.uint16))
    cv2.imwrite(str(out/'frames'/f'{i:04d}.jpg'),cv2.cvtColor(rgb[i],cv2.COLOR_RGB2BGR))
    matrix = np.eye(4); matrix[:3,:4] = pose
    frames.append(dict(index=i,time=times[i],cameraToWorld=matrix.tolist(),intrinsics=k.tolist(),
        image=f'frames/{i:04d}.jpg',pointCount=int(valid.sum()),confidenceThreshold=cutoff))
points=np.concatenate(points).astype('<f4'); colors=np.concatenate(colors)
frame_ids=np.concatenate(frame_ids).astype('<u2'); scores=np.concatenate(scores).astype('<f4')
if not len(points) or not np.isfinite(points).all() or not np.isfinite(poses).all():
    raise RuntimeError('Invalid reconstruction')
points.tofile(out/'positions.bin'); colors.tofile(out/'colors.bin'); frame_ids.tofile(out/'frame-ids.bin')
scores.tofile(out/'confidence.bin')
ply_dtype=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')])
verts=np.empty(len(points),ply_dtype)
for j,key in enumerate(['x','y','z']): verts[key]=points[:,j]
for j,key in enumerate(['red','green','blue']): verts[key]=colors[:,j]
with open(out/'cloud.ply','wb') as f:
    f.write((f'ply\nformat binary_little_endian 1.0\nelement vertex {len(points)}\n'
        'property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n').encode())
    f.write(verts.tobytes())
trajectory = dict(convention='camera-to-world, OpenCV camera axes (+x right,+y down,+z forward)',
    scale='arbitrary model units; not metric calibrated',frames=frames)
(out/'trajectory.json').write_text(json.dumps(trajectory),encoding='utf-8')
centers=poses[:,:3,3]
steps=np.linalg.norm(np.diff(centers,axis=0),axis=1)
meta=dict(title=a.title,model='LingBot-Map',modelRepo='https://github.com/Robbyant/lingbot-map',
    upstreamCommit=subprocess.check_output(['git','-C',a.upstream,'rev-parse','HEAD'],text=True).strip(),
    sourceUrl='https://www.youtube.com/watch?v=3NyyOZQzy_E',sourceSegment=[a.source_start,a.source_start+frame_count/fps],
    sourceSha256=hashlib.sha256(Path(a.source).read_bytes()).hexdigest(),
    duration=frame_count/fps,frames=n,width=w,height=h,pointCount=len(points),
    sampleFps=fps/stride,inferenceSeconds=elapsed,peakGpuGB=torch.cuda.max_memory_allocated()/1e9,
    settings=dict(imageSize=a.image_size,scaleFrames=2,keyframeInterval=2,cacheWindow=16,backend='SDPA',dtype='bfloat16'),
    bounds=[np.quantile(points,.01,axis=0).tolist(),np.quantile(points,.99,axis=0).tolist()],
    diagnostics=dict(pathLength=float(steps.sum()),maxStep=float(steps.max()),medianStep=float(np.median(steps))),
    limitations=['Experimental monocular reconstruction; no ground-truth accuracy evaluation.',
        'Uncalibrated fisheye projection, smoke, reflections and camera-fixed bodywork can corrupt geometry.',
        'Camera pose, not calibrated robot body pose. Scale is arbitrary, not meters.',
        'Point cloud display uses confidence, dark-pixel and peripheral masks; no geometric bundle adjustment.'])
(out/'manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
print(json.dumps(meta,indent=2),flush=True)
