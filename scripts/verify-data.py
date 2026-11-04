"""Check export integrity and transform conventions (not ground-truth accuracy)."""
import json
from pathlib import Path
import numpy as np
p=Path('public/data')
m=json.loads((p/'manifest.json').read_text())
t=json.loads((p/'trajectory.json').read_text())
xyz=np.fromfile(p/'positions.bin',dtype='<f4').reshape(-1,3)
ids=np.fromfile(p/'frame-ids.bin',dtype='<u2')
rgb=np.fromfile(p/'colors.bin',dtype='u1').reshape(-1,3)
assert len(xyz)==len(ids)==len(rgb)==m['pointCount']
assert len(t['frames'])==m['frames']
assert np.isfinite(xyz).all()
assert ids.min()==0 and ids.max()<m['frames']
times=[]
for f in t['frames']:
    matrix=np.array(f['cameraToWorld']);r=matrix[:3,:3]
    assert np.allclose(r.T@r,np.eye(3),atol=1e-4)
    assert np.isclose(np.linalg.det(r),1,atol=1e-4)
    assert np.allclose(matrix[3],[0,0,0,1])
    assert np.isfinite(matrix).all()
    assert (p/f['image']).exists()
    assert np.count_nonzero(ids==f['index'])==f['pointCount']
    times.append(f['time'])
assert np.all(np.diff(times)>0)
assert times[-1]<m['duration']
print(json.dumps({'status':'passed','points':len(xyz),'frames':len(times),'finite':True,'rotationsValid':True}))
