import { chromium } from '@playwright/test';
import fs from 'node:fs';
const base=process.argv[2]||'http://127.0.0.1:4174/roadster-slam-lab/';
const browser=await chromium.launch({headless:true});const page=await browser.newPage({viewport:{width:1400,height:1000}});const errors=[];page.on('pageerror',e=>errors.push(e.message));
await page.goto(base+'prior.html');await page.waitForFunction(()=>window.priorReady&&document.querySelector('#pov').naturalWidth>0);
await page.locator('#timeline').fill('35');await page.locator('#timeline').dispatchEvent('input');await page.waitForFunction(()=>document.querySelector('#time').textContent.startsWith('7.0'));
await page.locator('#hybrid').check();await page.waitForTimeout(1800);await page.locator('#road').uncheck();await page.locator('#road').check();await page.locator('#path').uncheck();await page.locator('#path').check();
await page.locator('#play').click();await page.waitForTimeout(500);await page.locator('#play').click();if(await page.locator('#timeline').inputValue()==='35')throw Error('playback stalled');
const t=await (await page.request.get(base+'data/ramp-prior/trajectory.json')).json();for(const f of t.frames){for(const key of ['externalImage','povImage']){const r=await page.request.get(new URL(f[key],base+'data/ramp-prior/').href);if(!r.ok())throw Error('missing '+f[key])}for(const key of ['bodyToWorld','cameraToWorld']){const m=f[key];if(m.flat().some(x=>!Number.isFinite(x)))throw Error('invalid pose');for(let a=0;a<3;a++)for(let b=0;b<3;b++){const dot=m[a].slice(0,3).reduce((s,x,k)=>s+x*m[b][k],0);if(Math.abs(dot-(a===b?1:0))>1e-6)throw Error('nonorthogonal pose')}}}
for(const a of await page.locator('#downloads a').all()){const r=await page.request.get(new URL(await a.getAttribute('href'),base).href);if(!r.ok())throw Error('missing download')}
fs.mkdirSync('runtime/prior',{recursive:true});await page.screenshot({path:'runtime/prior/view-desktop.png',fullPage:true});await page.setViewportSize({width:390,height:844});await page.screenshot({path:'runtime/prior/view-mobile.png',fullPage:true});if(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth))throw Error('mobile overflow');if(errors.length)throw Error(errors.join('\n'));console.log('PASS: 62 frames, poses, downloads, playback, toggles, desktop/mobile, no JS errors');await browser.close();

