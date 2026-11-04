"""Run upstream VGGT-SLAM 2.0 graph on every supplied 5 FPS frame."""
import argparse,sys,time,subprocess
from pathlib import Path
import numpy as np
import torch
from export_common import export_predictions

def main():
    p=argparse.ArgumentParser();p.add_argument('--upstream',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--images',required=True);p.add_argument('--output',required=True);p.add_argument('--work',required=True);a=p.parse_args()
    root=Path(a.upstream).resolve();sys.path[:0]=[str(root),str(root/'third_party/vggt'),str(root/'third_party/salad')]
    from vggt.models.vggt import VGGT
    import vggt_slam.solver as solver_module
    from vggt_slam.slam_utils import decompose_camera
    # Disable only the optional GUI and RAFT-based input subsampling. The input
    # is already sampled at 5 FPS; retrieval and graph optimization stay enabled.
    solver_module.Viewer=lambda:None
    solver_module.FrameTracker=lambda:None
    torch.set_num_threads(6);torch.set_grad_enabled(False)
    model=VGGT().cuda().eval()
    model.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=True))
    solver=solver_module.Solver(init_conf_threshold=25,lc_thres=.95)
    paths=sorted(Path(a.images).glob('frame_*.jpg'))
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.time()
    for first in range(0,len(paths)-1,16):
        batch=paths[first:first+17]
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            pred=solver.run_predictions([str(x) for x in batch],model,1,None,None)
        solver.add_points(pred);solver.graph.optimize()
        print(f'Optimized frames {first}..{first+len(batch)-1}',flush=True)
        del pred;torch.cuda.empty_cache()
    torch.cuda.synchronize();elapsed=time.time()-start;peak=torch.cuda.max_memory_allocated()/1e9
    records={}
    for submap in solver.map.get_submaps():
        if submap.get_lc_status():continue
        poses=submap.get_all_poses_world(solver.graph)
        projections=submap.get_all_poses_world(solver.graph,give_camera_mat=True)
        for i,fid in enumerate(submap.frame_ids):
            pts=submap.pointclouds[i];flat=pts.reshape(-1,3)
            ph=np.concatenate([flat,np.ones((len(flat),1))],axis=1)
            transformed=ph@solver.graph.get_homography(submap.get_id()+i).T
            world=(transformed[:,:3]/transformed[:,3:]).reshape(pts.shape)
            k,_,_,_=decompose_camera(projections[i][:3,:])
            records[fid]=(world,submap.colors[i],submap.conf[i],poses[i],k)
    order=sorted(records);arrays=list(zip(*(records[k] for k in order)))
    raw=dict(world_points=np.stack(arrays[0]),rgb=np.stack(arrays[1]).astype('uint8'),confidence=np.stack(arrays[2]),camera_to_world=np.stack(arrays[3]),intrinsics=np.stack(arrays[4]),times=np.array(order)/30)
    work=Path(a.work);work.mkdir(parents=True,exist_ok=True);np.savez_compressed(work/'predictions.npz',**raw)
    meta=dict(title='Roadster / Kickflip',model='VGGT-SLAM 2.0',modelRepo='https://github.com/MIT-SPARK/VGGT-SLAM',upstreamCommit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),sourceUrl='https://www.youtube.com/watch?v=3NyyOZQzy_E',sourceSegment=[4.833333333,13.9],duration=9.066666667,sampleFps=5,inferenceSeconds=elapsed,peakGpuGB=peak,settings=dict(inputFrames=len(paths),submapSize=17,overlap=1,maxLoops=1,lcThreshold=.95,graph='SL(4)',opticalFlowSubsampling=False),note='5 FPS 전체 입력; SL(4) 그래프 최적화. RAFT 입력 선별만 비활성화.')
    export_predictions(raw,a.output,meta)

if __name__=='__main__':main()
