import urllib.request,json,re,concurrent.futures
from pathlib import Path
def info(name,version):
    with urllib.request.urlopen(f'https://pypi.org/pypi/{name}/{version}/json',timeout=30) as r:return json.load(r)
data=info('torch','2.8.0')
packages=[('torch','2.8.0'),('torchvision','0.23.0')]
for requirement in data['info']['requires_dist']:
    match=re.match(r'((?:nvidia-[\w-]+|triton))==([\d.]+)',requirement)
    if match:packages.append(match.groups())
def url(pair):
    files=info(*pair)['urls']
    candidates=[f['url'] for f in files if ('manylinux' in f['filename'] and 'x86_64' in f['filename'] and ('cp312' in f['filename'] or 'py3-none' in f['filename']))]
    assert len(candidates)==1,(pair,candidates)
    return candidates[0]
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:urls=list(pool.map(url,packages))
Path('/tmp/roadster-wheel-urls.txt').write_text('\n'.join(urls)+'\n')
print(len(urls),'CUDA/PyTorch wheels')
