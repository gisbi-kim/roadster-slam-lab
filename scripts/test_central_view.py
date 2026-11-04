"""Controlled Ramp input ablation: full view, center crop, assumed fisheye remap.
This is numerical preprocessing, not learned image generation or calibration.
"""
import argparse,hashlib,json,subprocess,time,gc
from pathlib import Path
import cv2,numpy as np,torch
from depth_anything_3.api import DepthAnything3
from export_common import export_predictions

def transform(im,variant):
    h,w=im.shape[:2]
    if variant=='full':return im
    if variant=='crop':
        cw=int(w*.56)//2*2;ch=cw*3//4
        return cv2.resize(im[(h-ch)//2:(h-ch)//2+ch,(w-cw)//2:(w-cw)//2+cw],(1280,960),interpolation=cv2.INTER_AREA)
    # Equidistant r=f*theta, assumed 180 degree horizontal FOV; no fitted
    # distortion coefficients. Target is a 90-degree rectilinear virtual view.
    ow,oh=1280,960;f=ow/2;yy,xx=np.mgrid[:oh,:ow]
    x=(xx-(ow-1)/2)/f;y=(yy-(oh-1)/2)/f;r=np.hypot(x,y)
    scale=(w/np.pi)*np.arctan(r)/np.maximum(r,1e-10)
    mx=((w-1)/2+x*scale).astype('float32');my=((h-1)/2+y*scale).astype('float32')
    assert mx.min()>=0 and mx.max()<w-1 and my.min()>=0 and my.max()<h-1
    return cv2.remap(im,mx,my,cv2.INTER_LINEAR)

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--runtime',required=True);p.add_argument('--preview-only',action='store_true');a=p.parse_args()
    root=Path(a.runtime);work=Path('runtime/central-view');work.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(a.source);fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));frames=[];times=[]
    for i in range(count):
        ok,im=cap.read()
        if not ok:break
        if i%6==0:frames.append(im);times.append(i/fps)
    cap.release()
    variants=['full','crop','perspective'];rows=[]
    for idx in [0,35,len(frames)-1]:
        cells=[]
        for variant in variants:
            im=cv2.resize(transform(frames[idx],variant),(480,360));cv2.putText(im,f'{variant} | {times[idx]:.1f}s',(12,28),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,255),2);cells.append(im)
        rows.append(np.hstack(cells))
    cv2.imwrite(str(work/'input-comparison.jpg'),np.vstack(rows))
    if a.preview_only:return
    model=DepthAnything3.from_pretrained(str(root/'da3-large')).cuda().eval();torch.set_num_threads(6)
    for variant in variants:
        folder=work/variant;folder.mkdir(exist_ok=True);paths=[]
        for i,im in enumerate(frames):
            path=folder/f'frame_{i:06d}.jpg';cv2.imwrite(str(path),transform(im,variant));paths.append(str(path))
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();start=time.time()
        pred=model.inference(paths,process_res=504,process_res_method='upper_bound_resize',use_ray_pose=False)
        torch.cuda.synchronize();elapsed=time.time()-start;peak=torch.cuda.max_memory_allocated()/1e9
        ext=np.tile(np.eye(4),(len(paths),1,1));ext[:,:3,:4]=pred.extrinsics[:,:3,:4]
        raw=dict(camera_to_world=np.linalg.inv(ext),intrinsics=pred.intrinsics,depth=pred.depth,confidence=pred.conf,rgb=pred.processed_images,times=np.array(times))
        np.savez_compressed(folder/'predictions.npz',**raw)
        prep=dict(variant=variant,calibrated=False)
        if variant=='crop':prep.update(center=[.5,.5],cropWidthFraction=.56,cropAspectRatio=4/3,output=[1280,960])
        if variant=='perspective':prep.update(sourceModel='assumed equidistant',sourceHorizontalFovDegrees=180,targetHorizontalFovDegrees=90,principalPoint='image center',output=[1280,960],distortionCoefficients='not estimated')
        notes={'full':'램프 원본 입력 대조군. 같은 DA3·62프레임·5 FPS.','crop':'입력 단계에서 중앙 56% 폭, 4:3 크롭. 링 제거 목적이며 렌즈 왜곡 보정은 아님.','perspective':'입력 단계에서 가정한 어안→90° 원근 투영. 렌즈 캘리브레이션 없이 180° equidistant 모델을 가정한 실험.'}
        meta=dict(title='Roadster / Ramp',model='Depth Anything 3 Large',modelRepo='https://github.com/ByteDance-Seed/Depth-Anything-3',upstreamCommit=subprocess.check_output(['git','-C',str(root/'da3'),'rev-parse','HEAD'],text=True).strip(),sourceUrl='https://www.youtube.com/watch?v=3NyyOZQzy_E',sourceSegment=[13.9,26.233333333],sourceSha256=hashlib.sha256(Path(a.source).read_bytes()).hexdigest(),duration=count/fps,sampleFps=fps/6,inferenceSeconds=elapsed,peakGpuGB=peak,settings=dict(checkpoint='depth-anything/DA3-LARGE',resolution=504,useRayPose=False,preprocessing=prep),note=notes[variant])
        export_predictions(raw,f'public/data/ramp-{variant}',meta)
        del pred,raw;gc.collect()

if __name__=='__main__':main()
