import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import './style.css';
import experiments from './experiments.json';
const $=id=>document.getElementById(id);
const base=import.meta.env.BASE_URL;
const params=new URLSearchParams(location.search);
const requested=params.get('run')||params.get('clip');
const experiment=experiments.find(e=>e.id===requested)||experiments[0];
const clip=experiment.clip;
const dataBase=base+'data/'+experiment.path;
$('clip').replaceChildren(...experiments.map(e=>{const option=document.createElement('option');option.value=e.id;option.disabled=e.status!=='ready';option.textContent=e.label+(e.status==='ready'?'':' · '+e.status);return option}));
$('clip').value=experiment.id;
$('clip').onchange=e=>{const url=new URL(location.href);url.searchParams.delete('clip');url.searchParams.set('run',e.target.value);location.assign(url)};
for(const [id,file] of [['download-cloud','cloud.ply'],['download-poses','trajectory.json'],['manifest','manifest.json']])$(id).href=dataBase+file;
$('source-name').textContent=clip==='kickflip'?'02 / KICKFLIP':'03 / RAMP';
let meta,frames,cloud,frameIds,positions,colors,selected=0,playing=false,following=false;
const scene=new THREE.Scene();scene.background=new THREE.Color('#121c1d');
const camera=new THREE.PerspectiveCamera(48,1,.001,10000);camera.up.set(0,-1,0);
const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));
$('canvas').appendChild(renderer.domElement);
const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;
const trackGroup=new THREE.Group();scene.add(trackGroup);
const marker=new THREE.Group();scene.add(marker);
let radius=1,center=new THREE.Vector3(),lastFrameTime=0;
function fit(){following=false;$('follow').classList.remove('active');camera.position.copy(center).add(new THREE.Vector3(.9,-.65,-1.25).multiplyScalar(radius));controls.target.copy(center);camera.near=Math.max(radius/10000,.0001);camera.far=radius*100;camera.updateProjectionMatrix();controls.update()}
function drawCloud(){const mode=$('mode').value;if(mode==='all'){cloud.geometry.setDrawRange(0,frameIds.length);cloud.geometry.setIndex(null);return}const indices=[];for(let i=0;i<frameIds.length;i++)if(mode==='frame'?frameIds[i]===selected:frameIds[i]<=selected)indices.push(i);cloud.geometry.setIndex(indices);cloud.geometry.setDrawRange(0,indices.length)}
function selectFrame(i){selected=Math.max(0,Math.min(i,frames.length-1));const f=frames[selected];$('scrub').value=selected;$('time').textContent=f.time.toFixed(2)+' s';$('frame-label').textContent=`${selected+1} / ${frames.length}`;$('frame').src=dataBase+f.image;const m=new THREE.Matrix4().set(...f.cameraToWorld.flat());marker.matrixAutoUpdate=false;marker.matrix.copy(m);if(cloud&&$('mode').value!=='all')drawCloud();if(following){const pos=new THREE.Vector3().setFromMatrixPosition(m);const direction=new THREE.Vector3(0,0,1).transformDirection(m);camera.position.copy(pos).addScaledVector(direction,-radius*.12);camera.up.set(0,-1,0).transformDirection(m);controls.target.copy(pos).addScaledVector(direction,radius*.18);controls.update()}}
async function unzip(response){const data=await response.arrayBuffer();const magic=new Uint8Array(data,0,Math.min(2,data.byteLength));return magic[0]===31&&magic[1]===139?new Response(new Blob([data]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer():data}
async function get(url,type='json'){const r=await fetch(dataBase+url);if(!r.ok)throw new Error(`${url}: HTTP ${r.status}`);return type==='json'?r.json():type==='gzip'?unzip(r):r.arrayBuffer()}
async function init(){if(experiment.status!=='ready'){document.querySelector('.downloads').hidden=true;$('manifest').hidden=true;throw new Error(experiment.note||'실험 준비 중')}[meta,{frames},positions,colors,frameIds]=await Promise.all([get('manifest.json'),get('trajectory.json'),get('positions.bin','bin'),get('colors.bin.gz','gzip'),get('frame-ids.bin','bin')]);positions=new Float32Array(positions);colors=new Uint8Array(colors);frameIds=new Uint16Array(frameIds);if(positions.length!==meta.pointCount*3||colors.length!==positions.length||frameIds.length!==meta.pointCount)throw new Error('Point cloud file sizes do not match manifest');const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.BufferAttribute(positions,3));geometry.setAttribute('color',new THREE.BufferAttribute(colors,3,true));cloud=new THREE.Points(geometry,new THREE.PointsMaterial({size:2,sizeAttenuation:false,vertexColors:true}));scene.add(cloud);
const box=new THREE.Box3(new THREE.Vector3(...meta.bounds[0]),new THREE.Vector3(...meta.bounds[1]));for(const f of frames)box.expandByPoint(new THREE.Vector3(f.cameraToWorld[0][3],f.cameraToWorld[1][3],f.cameraToWorld[2][3]));box.getCenter(center);radius=box.getSize(new THREE.Vector3()).length()*.65;
const centers=frames.map(f=>new THREE.Vector3(f.cameraToWorld[0][3],f.cameraToWorld[1][3],f.cameraToWorld[2][3]));trackGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(centers),new THREE.LineBasicMaterial({color:'#b7edca',depthTest:false,transparent:true,opacity:.9})));
const s=radius*.025;const corners=[[-s,-s*.7,s*1.6],[s,-s*.7,s*1.6],[s,s*.7,s*1.6],[-s,s*.7,s*1.6]];const vertices=[];corners.forEach((v,i)=>{vertices.push(new THREE.Vector3(),new THREE.Vector3(...v),new THREE.Vector3(...v),new THREE.Vector3(...corners[(i+1)%4]))});marker.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(vertices),new THREE.LineBasicMaterial({color:'#ffbc76',depthTest:false})));marker.add(new THREE.Mesh(new THREE.SphereGeometry(s*.15,12,8),new THREE.MeshBasicMaterial({color:'#ffbc76',depthTest:false})));
$('fps').textContent=meta.sampleFps+' FPS';$('filter-info').textContent=meta.displayFilter||'기존 필터: confidence ≥1.5, 상위 40%, 2px 간격. 새 실험과 표시 기준이 다릅니다.';$('quality').textContent=`${frames.length}프레임 중 ${frames.filter(f=>f.pointCount===0).length}프레임은 표시 필터를 통과한 점이 없습니다. 프레임별 모드에서 화면이 비어 보일 수 있습니다.`;$('source-time').textContent='원본 '+meta.sourceSegment.map(t=>t.toFixed(2)+'초').join('–');$('subtitle').textContent=(clip==='kickflip'?'킥플립':'램프')+' · '+meta.model+'의 공간 및 카메라 pose 추정 결과입니다.';$('points').textContent=new Intl.NumberFormat('en',{notation:'compact',maximumFractionDigits:1}).format(meta.pointCount);$('frames').textContent=meta.frames;$('resolution').textContent=`${meta.width} × ${meta.height}`;$('elapsed').textContent=meta.inferenceSeconds.toFixed(1)+' s';$('duration').textContent=meta.duration.toFixed(2)+' s';$('scrub').max=frames.length-1;$('method').textContent=`${meta.model} · 입력 ${meta.sampleFps} FPS · GPU ${meta.peakGpuGB.toFixed(1)} GB`+ (meta.note?' · '+meta.note:'');$('loading').hidden=true;fit();selectFrame(0);window.__reconstruction={clip,run:experiment.id,pointCount:meta.pointCount,frameCount:frames.length,ready:true};}
$('reset').onclick=()=>{camera.up.set(0,-1,0);fit()};$('follow').onclick=()=>{following=!following;$('follow').classList.toggle('active',following);selectFrame(selected)};$('scrub').oninput=e=>selectFrame(+e.target.value);$('mode').onchange=()=>cloud&&drawCloud();$('size').oninput=e=>{if(cloud)cloud.material.size=+e.target.value};$('trajectory').onchange=e=>trackGroup.visible=e.target.checked;$('play').onclick=()=>{if(!frames)return;playing=!playing;lastFrameTime=performance.now();$('play').textContent=playing?'Ⅱ':'▶';$('play').setAttribute('aria-label',playing?'일시 정지':'재생')};
new ResizeObserver(()=>{const {width,height}=$('canvas').getBoundingClientRect();renderer.setSize(width,height);camera.aspect=width/height;camera.updateProjectionMatrix()}).observe($('canvas'));
renderer.setAnimationLoop(t=>{if(playing&&frames&&t-lastFrameTime>1000*(frames[selected+1]?.time-frames[selected].time||1/meta.sampleFps)){selectFrame((selected+1)%frames.length);lastFrameTime=t}controls.update();renderer.render(scene,camera)});
init().catch(e=>{$('loading').textContent='데이터 로드 실패: '+e.message;console.error(e)});

async function comparison(){
 const rows=await Promise.all(experiments.map(async e=>{
  const tr=document.createElement('tr');
  if(e.status!=='ready'){for(const text of [e.label,e.status,'—','—',e.note||'준비 중']){const td=document.createElement('td');td.textContent=text;tr.append(td)}return tr}
  const r=await fetch(base+'data/'+e.path+'manifest.json');if(!r.ok)throw new Error(e.id+' manifest');const m=await r.json();
  const ratio=m.diagnostics.medianStep>0?(m.diagnostics.maxStep/m.diagnostics.medianStep).toFixed(1)+'×':'—';
  const td=document.createElement('td');const a=document.createElement('a');a.href='?run='+e.id;a.textContent=e.label;td.append(a);tr.append(td);
  for(const value of [m.frames.toLocaleString(),m.pointCount.toLocaleString(),m.inferenceSeconds.toFixed(1)+' s',ratio]){const cell=document.createElement('td');cell.textContent=value;tr.append(cell)}return tr;
 }));$('comparison-body').replaceChildren(...rows);
}
comparison().catch(e=>console.error(e));
