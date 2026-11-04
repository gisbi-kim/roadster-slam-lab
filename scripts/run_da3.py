import argparse,sys,time,json,subprocess,hashlib
from pathlib import Path
import numpy as np
import cv2,torch
from depth_anything_3.api import DepthAnything3
from export_common import export_predictions
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--upstream',required=True);p.add_argument('--fps',type=int,default=5);p.add_argument('--resolution',type=int,default=504);p.add_argument('--output',required=True);p.add_argument('--work',required=True);a=p.parse_args()
work=Path(a.work);work.mkdir(parents=True,exist_ok=True)
cap=cv2.VideoCapture(a.source);fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));stride=round(fps/a.fps);paths=[];times=[]
for i in range(count):
    ok,im=cap.read()
    if not ok:break
    if i%stride:continue
    path=work/f'frame_{i:06d}.jpg';cv2.imwrite(str(path),im);paths.append(str(path));times.append(i/fps)
cap.release();torch.set_num_threads(6)
print(f'Loading DA3-Large, {len(paths)} frames',flush=True)
model=DepthAnything3.from_pretrained(a.checkpoint).to('cuda').eval()
torch.cuda.reset_peak_memory_stats();start=time.time()
pred=model.inference(paths,process_res=a.resolution,process_res_method='upper_bound_resize',use_ray_pose=False)
torch.cuda.synchronize();elapsed=time.time()-start
# Official exports treat prediction.extrinsics as world-to-camera.
ext=np.tile(np.eye(4),(len(paths),1,1));ext[:,:3,:4]=pred.extrinsics[:,:3,:4]
raw=dict(camera_to_world=np.linalg.inv(ext),intrinsics=pred.intrinsics,depth=pred.depth,confidence=pred.conf,rgb=pred.processed_images,times=np.array(times))
np.savez_compressed(work/'predictions.npz',**raw)
meta=dict(title='Roadster / Kickflip',model='Depth Anything 3 Large',modelRepo='https://github.com/ByteDance-Seed/Depth-Anything-3',upstreamCommit=subprocess.check_output(['git','-C',a.upstream,'rev-parse','HEAD'],text=True).strip(),sourceUrl='https://www.youtube.com/watch?v=3NyyOZQzy_E',sourceSegment=[4.833333333,13.9],sourceSha256=hashlib.sha256(Path(a.source).read_bytes()).hexdigest(),duration=count/fps,sampleFps=fps/stride,inferenceSeconds=elapsed,peakGpuGB=torch.cuda.max_memory_allocated()/1e9,settings=dict(checkpoint='depth-anything/DA3-LARGE',resolution=a.resolution,useRayPose=False,mode='joint multi-view inference'))
export_predictions(raw,a.output,meta)
