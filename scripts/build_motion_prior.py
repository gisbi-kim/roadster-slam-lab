"""Manually observed external view + explicit virtual track, NOT ground truth.
Plane homography is set from four manually traced/assumed track correspondences.
Positions are prior-conditioned; flight roll, metric scale and extrinsics are not observed.
"""
import json,gzip,hashlib
from pathlib import Path
import cv2,numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation

OUT=Path('public/data/ramp-prior');OUT.mkdir(parents=True,exist_ok=True)
OBS=np.array([[0,464,205],[.5,464,205],[1,471,205],[1.5,485,198],[2,506,184],[2.5,530,164],[3,545,135],[3.5,555,109],[4,555,88],[4.5,550,65],[5,541,51],[5.5,528,39],[6,495,39],[6.5,452,38],[7,408,39],[7.5,357,63],[8,302,105],[8.5,249,174],[9,202,236],[9.5,153,262],[10,106,278],[10.5,57,304],[11,10,318]],float)
# Coordinates refer to the original top-right panel resized to 606 x 360.
PLANE=np.float32([[0,0],[0,2],[1,1],[-1.65,2]]) # forward Z, height H
IMAGE=np.float32([[470,210],[525,36],[560,110],[420,33]])
H=cv2.getPerspectiveTransform(PLANE,IMAGE);HI=np.linalg.inv(H)
def project(p):
    q=H@np.array([p[2],-p[1],1]);return q[:2]/q[2]
def surface(s,h=.06):
    if s<0:return np.array([0.,-h,s]),0.
    if s<=np.pi:return np.array([0.,-1+(1-h)*np.cos(s),(1-h)*np.sin(s)]),s
    return np.array([0.,-2+h,-(s-np.pi)]),np.pi
ss=[]
for t,u,v in OBS[:14]:
    grid=np.linspace(-.4,np.pi+1.65,800);err=[np.linalg.norm(project(surface(s)[0])-[u,v]) for s in grid];k=np.argmin(err)
    sol=minimize_scalar(lambda s:np.sum((project(surface(s)[0])-[u,v])**2),bounds=(grid[max(0,k-2)],grid[min(len(grid)-1,k+2)]),method='bounded');ss.append(sol.x)
ss=np.maximum.accumulate(ss)
TAKEOFF=6.85;LAND=9.5
route=PchipInterpolator(np.r_[OBS[:14,0],TAKEOFF],np.r_[ss,np.pi+1.65])
air_obs=OBS[(OBS[:,0]>=7)&(OBS[:,0]<=9)]
air_world=[]
for t,u,v in air_obs:
    q=HI@np.array([u,v,1]);q/=q[2];air_world.append([q[0],q[1]])
qland=HI@np.array([153,262,1]);qland/=qland[2]
flight_t=np.r_[TAKEOFF,air_obs[:,0],LAND]
flight_z=PchipInterpolator(flight_t,np.r_[-1.65,np.array(air_world)[:,0],qland[0]])
flight_h=PchipInterpolator(flight_t,np.r_[1.94,np.clip(np.array(air_world)[:,1],.06,1.94),.06])
ground_obs=OBS[OBS[:,0]>=9.5];ground_z=[]
for t,u,v in ground_obs:
    q=HI@np.array([u,v,1]);q/=q[2];ground_z.append(q[0])
ground=PchipInterpolator(ground_obs[:,0],ground_z)
def motion(t):
    if t<=TAKEOFF:
        pos,pitch=surface(float(route(t)));roll=0;phase='ramp contact' if route(t)<np.pi else 'upper overhang contact'
    elif t<LAND:
        pos=np.array([0.,-float(flight_h(t)),float(flight_z(t))]);pitch=np.pi
        q=np.clip((t-7.0)/(9.3-7.0),0,1);roll=np.pi*(6*q**5-15*q**4+10*q**3);phase='flight: planar position fit, assumed roll'
    else:
        z=float(ground(min(t,11)))
        if t>11:z+=float(ground.derivative()(11))*(t-11)
        pos=np.array([0.,-.06,z]);pitch=np.pi;roll=np.pi;phase='ground contact' if t<=11 else 'outside image: extrapolated'
    # OpenCV camera/body convention x right, y down, z forward.
    r=Rotation.from_euler('x',pitch).as_matrix()@Rotation.from_euler('z',roll).as_matrix()
    body=np.eye(4);body[:3,:3]=r;body[:3,3]=pos
    cam=body.copy();cam[:3,3]=pos+r@np.array([0,-.025,0])
    return body,cam,phase

times=np.arange(62)/5;frames=[]
for i,t in enumerate(times):
    b,c,phase=motion(t);frames.append(dict(index=i,time=float(t),bodyToWorld=b.tolist(),cameraToWorld=c.tolist(),phase=phase,projectedCenter=project(b[:3,3]).tolist(),externalImage=f'external/{i:04d}.jpg',povImage=f'../ramp-crop/frames/{i:04d}.jpg'))
res=[]
for t,u,v in OBS:res.append(float(np.linalg.norm(project(motion(t)[0][:3,3])-[u,v])))
observations=dict(method='Manual approximate body-centre annotations from contact sheets; not an automatic tracker',imageSize=[606,360],pixelUncertaintyApprox=5,observations=[dict(time=float(t),pixel=[float(u),float(v)]) for t,u,v in OBS],trackControlPoints=[dict(planeZH=p.tolist(),image=q.tolist()) for p,q in zip(PLANE,IMAGE)],homography=H.tolist(),reprojectionRMSE=float(np.sqrt(np.mean(np.square(res)))),contactRMSE=float(np.sqrt(np.mean(np.square(res[:14])))),note='Fitting residual against the same annotations, NOT independent pose accuracy. Plane dimensions and shape are assumed. Edited-panel timing is not verified sensor synchronization.')
(OUT/'observations.json').write_text(json.dumps(observations,indent=2))
(OUT/'trajectory.json').write_text(json.dumps(dict(convention='body/camera-to-world, +X right,+Y down,+Z initial forward; radius-normalized units',scale='one unit = assumed ramp radius, not calibrated metres',poseType='prior-conditioned virtual model, not measured ground truth',frames=frames)))

# Synthetic track samples for a clearly labelled geometry download.
pts=[];cols=[]
for s in np.linspace(-6,np.pi+1.65,700):
    p,_=surface(s,0)
    for x in np.linspace(-.34,.34,45):
        pts.append([x,p[1],p[2]]);cols.append([235,120,55] if abs(x)>.325 else ([230]*3 if abs(x)<.012 else [90,110,115]))
def ply(path,xyz,rgb):
    a=np.empty(len(xyz),dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')])
    for j,k in enumerate(['x','y','z']):a[k]=np.asarray(xyz)[:,j]
    for j,k in enumerate(['red','green','blue']):a[k]=np.asarray(rgb)[:,j]
    with open(path,'wb') as f:
        f.write(f'ply\nformat binary_little_endian 1.0\ncomment PRIOR MODEL - NOT MEASURED GEOMETRY\nelement vertex {len(a)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n'.encode());f.write(a.tobytes())
ply(OUT/'virtual-track.ply',pts,cols)

# Optional diagnostic: DA3's own depth (not projected onto the synthetic surface),
# placed using prior camera poses. A single sequence-wide scale is fitted to
# intersections with the assumed road during contact; no per-pixel snapping.
raw=np.load('runtime/central-view/crop/predictions.npz');depth=raw['depth'];conf=raw['confidence'];rgb=raw['rgb'];ks=raw['intrinsics'];n,h,w=depth.shape
yy,xx=np.mgrid[:h,:w];ratios=[];support=[]
def hits(o,d):
    vals=[]
    for height,zmin,zmax in [(0,-8,0),(-2,-1.65,0)]:
        l=(height-o[1])/np.where(abs(d[...,1])>1e-8,d[...,1],np.nan);z=o[2]+l*d[...,2];x=o[0]+l*d[...,0]
        vals.append(np.where((l>0)&(z>=zmin)&(z<=zmax)&(abs(x)<=.34),l,np.inf))
    aa=d[...,1]**2+d[...,2]**2;bb=2*((o[1]+1)*d[...,1]+o[2]*d[...,2]);cc=(o[1]+1)**2+o[2]**2-1;disc=bb*bb-4*aa*cc
    for sign in [-1,1]:
        l=(-bb+sign*np.sqrt(np.maximum(disc,0)))/(2*aa+1e-12);z=o[2]+l*d[...,2];y=o[1]+l*d[...,1];x=o[0]+l*d[...,0]
        vals.append(np.where((disc>=0)&(l>0)&(z>=0)&(y>=-2)&(y<=0)&(abs(x)<=.34),l,np.inf))
    return np.minimum.reduce(vals)
for i,t in enumerate(times):
    if t>6.8:continue
    k=ks[i];ray=np.stack([(xx-k[0,2])/k[0,0],(yy-k[1,2])/k[1,1],np.ones((h,w))],-1);cam=np.array(frames[i]['cameraToWorld']);d=ray@cam[:3,:3].T;l=hits(cam[:3,3],d)
    color=rgb[i].astype(float);gray=(color.max(-1)-color.min(-1)<55)&(color.mean(-1)>45)
    mask=np.isfinite(l)&(l<4)&(depth[i]>0)&np.isfinite(depth[i])&gray&(xx%5==0)&(yy%5==0)&(conf[i]>=np.quantile(conf[i],.6))
    r=l[mask]/depth[i][mask];r=r[np.isfinite(r)&(r>0)]
    if len(r)>30:ratios.extend(r.tolist());support.append(i)
scale=float(np.median(ratios));hp=[];hc=[];hi=[]
for i,t in enumerate(times):
    if t>6.8:continue
    k=ks[i];d=depth[i]*scale;cam=np.array(frames[i]['cameraToWorld'])
    xyz=np.stack([(xx-k[0,2])*d/k[0,0],(yy-k[1,2])*d/k[1,1],d],-1)@cam[:3,:3].T+cam[:3,3]
    mask=(xx%5==0)&(yy%5==0)&(rgb[i].max(-1)>40)&(conf[i]>=np.quantile(conf[i],.6))&np.isfinite(xyz).all(-1)&(d>0)&(d<3)
    hp.append(xyz[mask]);hc.append(rgb[i][mask]);hi.append(np.full(mask.sum(),i,dtype='<u2'))
hp=np.concatenate(hp).astype('<f4');hc=np.concatenate(hc).astype('u1');hi=np.concatenate(hi)
hp.tofile(OUT/'hybrid-positions.bin');(OUT/'hybrid-colors.bin.gz').write_bytes(gzip.compress(hc.tobytes(),mtime=0));hi.tofile(OUT/'hybrid-frame-ids.bin');ply(OUT/'prior-placed-da3.ply',hp,hc)

# Extract same-timeline top-right observations; no learned/generated imagery.
source=next(Path('C:/Users/gsk620/Downloads').glob('Flight POV*.mp4'))
cap=cv2.VideoCapture(str(source));cap.set(cv2.CAP_PROP_POS_FRAMES,417);(OUT/'external').mkdir(exist_ok=True)
for j in range(370):
    ok,im=cap.read()
    if not ok:break
    if j%6:continue
    i=j//6
    if i>=62:break
    panel=cv2.resize(im[:720,2628:],(606,360));cv2.imwrite(str(OUT/'external'/f'{i:04d}.jpg'),panel)
cap.release()
meta=dict(title='Ramp / external-view-conditioned virtual motion',radiusUnits=1,halfWidth=.34,overhangLength=1.65,bodyHeight=.06,cameraOffset=[0,-.025,0],takeoffTime=TAKEOFF,landingTime=LAND,frames=62,sampleFps=5,sourceSegment=[13.9,26.233333333],observationCount=len(OBS),fit=observations['reprojectionRMSE'],contactFit=observations['contactRMSE'],hybridPointCount=len(hp),hybridDepthScale=scale,hybridSupportFrames=support,hybridTimeRange=[0,6.8],sources=[dict(title='Flight POV | Roadster (Tesla)',url='https://www.youtube.com/watch?v=3NyyOZQzy_E',use='Top-right external panel: manual 2D position annotations; left panel: DA3 input'),dict(title='The Ramp | Roadster (Tesla)',url='https://www.youtube.com/watch?v=7p5LxFdc-Qg',use='Separately reviewed multiple external angles; supports ramp/overhang/flight/landing topology only, not metric dimensions or synchronization')],assumptions=['Circular ramp radius=1 and flat upper overhang length=1.65; not engineering measurements.','All motion is constrained to the centre vertical plane; cross-track depth is not observed.','Planar homography uses four manually traced/assumed correspondences, not calibrated camera intrinsics.','Body heading follows road tangent during contact; airborne half-roll timing is a smooth assumed schedule, not measured attitude.','Camera rigid transform is assumed; all distances remain radius-normalized.','Edited panels are compared at the same video time; sensor synchronization not established.','After 11 seconds the vehicle leaves the external image; final motion is extrapolated.','Optional DA3 points keep predicted depth and one prior-fitted scale, only during 0–6.8 s contact. They do not validate the virtual track.'])
(OUT/'manifest.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))
