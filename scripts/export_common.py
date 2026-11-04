"""Shared viewer export; confidence ranks are per-model, not comparable probabilities."""
import json,gzip
from pathlib import Path
import cv2
import numpy as np

def export_predictions(raw, output, meta):
    out=Path(output);(out/'frames').mkdir(parents=True,exist_ok=True)
    rgb=raw['rgb'];poses=raw['camera_to_world'];ks=raw.get('intrinsics')
    depth=raw.get('depth');world=raw.get('world_points')
    conf=raw['confidence'];times=raw['times'];n,h,w=conf.shape
    yy,xx=np.mgrid[:h,:w]
    mask=((xx-w/2)/(w*.46))**2+((yy-h*.47)/(h*.44))**2<1
    points=[];colors=[];ids=[];scores=[];frames=[]
    for i in range(n):
        if world is not None:
            xyz=world[i]
            d=np.linalg.norm(xyz-poses[i,:3,3],axis=-1)
        else:
            d=depth[i];k=ks[i]
            cam=np.stack([(xx-k[0,2])*d/k[0,0],(yy-k[1,2])*d/k[1,1],d],-1)
            xyz=cam@poses[i,:3,:3].T+poses[i,:3,3]
        valid=mask & np.isfinite(xyz).all(-1)&np.isfinite(conf[i])&(d>0)&(rgb[i].max(-1)>30)
        cutoff=float(np.quantile(conf[i][valid],.6)) if valid.any() else 0.
        valid &= conf[i]>=cutoff
        valid &= (xx%3==0)&(yy%3==0)
        if valid.any(): valid &= d<=np.quantile(d[valid],.98)
        points.append(xyz[valid]);colors.append(rgb[i][valid]);scores.append(conf[i][valid]);ids.append(np.full(valid.sum(),i,dtype='<u2'))
        cv2.imwrite(str(out/'frames'/f'{i:04d}.jpg'),cv2.cvtColor(rgb[i],cv2.COLOR_RGB2BGR))
        m=np.eye(4);m[:3,:4]=poses[i,:3,:4]
        frames.append(dict(index=i,time=float(times[i]),cameraToWorld=m.tolist(),intrinsics=ks[i].tolist() if ks is not None else None,image=f'frames/{i:04d}.jpg',pointCount=int(valid.sum()),confidenceThreshold=cutoff))
    xyz=np.concatenate(points).astype('<f4');rgbpts=np.concatenate(colors).astype('u1');fid=np.concatenate(ids);confidence=np.concatenate(scores).astype('<f4')
    if not len(xyz) or not np.isfinite(poses).all():raise ValueError('No valid reconstruction')
    xyz.tofile(out/'positions.bin');(out/'colors.bin.gz').write_bytes(gzip.compress(rgbpts.tobytes(),mtime=0));fid.tofile(out/'frame-ids.bin');confidence.tofile(out/'confidence.bin')
    dt=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')]);v=np.empty(len(xyz),dt)
    for j,k in enumerate(['x','y','z']):v[k]=xyz[:,j]
    for j,k in enumerate(['red','green','blue']):v[k]=rgbpts[:,j]
    with open(out/'cloud.ply','wb') as f:
        f.write((f'ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n').encode());f.write(v.tobytes())
    (out/'trajectory.json').write_text(json.dumps(dict(convention='camera-to-world; OpenCV +x right,+y down,+z forward',scale='arbitrary model units; not calibrated robot pose',frames=frames)),encoding='utf-8')
    steps=np.linalg.norm(np.diff(poses[:,:3,3],axis=0),axis=-1)
    meta.update(frames=n,width=w,height=h,pointCount=len(xyz),bounds=[np.quantile(xyz,.01,axis=0).tolist(),np.quantile(xyz,.99,axis=0).tolist()],
        diagnostics=dict(pathLength=float(steps.sum()),maxStep=float(steps.max()) if len(steps) else 0,medianStep=float(np.median(steps)) if len(steps) else 0,emptyFrames=sum(f['pointCount']==0 for f in frames)),
        displayFilter='Confidence >= per-frame 60th percentile (ties retained) among elliptical interior and RGB max>30; 3px stride; exclude furthest 2%. No absolute confidence cutoff.',
        limitations=['Uncalibrated fisheye video, smoke, reflections and fast rotations.','Relative camera pose, not metric robot pose. No ground truth.','Point count is a display statistic, not accuracy; sequences have independent coordinates.'])
    (out/'manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(meta,indent=2),flush=True)
    return meta
