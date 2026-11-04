"""Headless adapter for the upstream MASt3R-SLAM windows (WSL) branch."""
import argparse,os,sys,time,subprocess
from pathlib import Path
import numpy as np
import torch
from export_common import export_predictions

def export_result(keyframes,elapsed,output,work,root,input_count,fps,final_mode):
    from mast3r_slam.lietorch_utils import as_SE3
    records=[]
    for i in range(len(keyframes)):
        f=keyframes[i];rgb=(f.uimg.cpu().numpy()*255).clip(0,255).astype('uint8');h,w=rgb.shape[:2]
        records.append((f.T_WC.act(f.X_canon).cpu().numpy().reshape(h,w,3),rgb,f.get_average_conf().cpu().numpy().reshape(h,w),as_SE3(f.T_WC).matrix().cpu().numpy().reshape(4,4),float(f.frame_id)/fps))
    arrays=list(zip(*records));raw=dict(world_points=np.stack(arrays[0]),rgb=np.stack(arrays[1]),confidence=np.stack(arrays[2]),camera_to_world=np.stack(arrays[3]),times=np.array(arrays[4]))
    Path(work).mkdir(parents=True,exist_ok=True);np.savez_compressed(Path(work)/'predictions.npz',**raw)
    meta=dict(title='Roadster / Kickflip',model='MASt3R-SLAM',modelRepo='https://github.com/rmurai0610/MASt3R-SLAM',upstreamCommit=subprocess.check_output(['git','-C',root,'rev-parse','HEAD'],text=True).strip(),sourceUrl='https://www.youtube.com/watch?v=3NyyOZQzy_E',sourceSegment=[4.833333333,13.9],duration=9.066666667,sampleFps=fps,inferenceSeconds=elapsed,peakGpuGB=torch.cuda.max_memory_allocated()/1e9,settings=dict(inputFrames=input_count,branch='windows',config='base.yaml',useCalibration=False,export='optimized keyframes only',finalTrackingMode=final_mode,firstKeyframeTime=float(raw['times'][0]),lastKeyframeTime=float(raw['times'][-1])),note='최적화된 키프레임만 표시. 보간하지 않음. 비보정 ray 모델로 intrinsics는 null.')
    if final_mode=='RELOC':
        meta['note']=f'추적 상실 후 종료까지 재위치 추정 실패. 마지막 키프레임 {raw["times"][-1]:.2f}초. 전체 클립 복원 실패.'
    export_predictions(raw,output,meta)

def main():
    p=argparse.ArgumentParser();p.add_argument('--fps',type=float,default=5);p.add_argument('--upstream',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--images',required=True);p.add_argument('--output',required=True);p.add_argument('--work',required=True);a=p.parse_args()
    root=str(Path(a.upstream).resolve());sys.path.insert(0,root);os.chdir(root)
    source=(Path(root)/'main.py').read_text()
    # GUI message defaults are preserved without requiring an OpenGL window.
    source=source.replace('from mast3r_slam.visualization import WindowMsg, run_visualization', 'from dataclasses import dataclass\n@dataclass\nclass WindowMsg:\n    is_terminated: bool = False\n    is_paused: bool = False\n    next: bool = False\n    C_conf_threshold: float = 1.5')
    source=source.replace('model = load_mast3r(device=device)',f'model = load_mast3r(path={a.checkpoint!r}, device=device)')
    source=source.replace('dataset.subsample(config["dataset"]["subsample"])',f'dataset.subsample(config["dataset"]["subsample"])\n    dataset.timestamps = np.arange(len(dataset)) / {a.fps}\n    dataset.save_results = False')
    source=source.replace('fps_timer = time.time()','torch.cuda.reset_peak_memory_stats()\n    fps_timer = time.time()')
    source=source.replace('SharedKeyframes(manager, h, w)','SharedKeyframes(manager, h, w, buffer=len(dataset)+2)')
    source=source.replace('    print("done")',f'    torch.cuda.synchronize()\n    export_result(keyframes, time.time()-fps_timer, {a.output!r}, {a.work!r}, {root!r}, len(dataset), {a.fps}, mode.name)\n    print("done")')
    sys.argv=['main.py','--dataset',a.images,'--config','config/base.yaml','--no-viz']
    torch.set_num_threads(6)
    exec(compile(source,str(Path(root)/'main.py'),'exec'),dict(__name__='__main__',np=np,export_result=export_result))

if __name__=='__main__':main()
