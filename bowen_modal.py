"""Train BOWEN variants on Modal GPUs in parallel (the PC is offline). Data lives in the Modal volume `bowen-world`
(traj_2026.npz, traj_2027.npz, desc.json, uploaded from results/world); each run writes results to its own folder there.

  A        live play only, single spot              WM_LIVE=1
  B        live play only, 3-option mixture         WM_LIVE=1 WM_MIX=3
  V3       B + hierarchical levels (team, team-season, defender) with learned shrinkage   WM_HIER=1
Same seed and game folds as the original BOWEN, so every model's predictions are out-of-fold on the same games.
  .venv/bin/modal run bowen_modal.py            (then: modal volume get bowen-world <dir>/pred.npz ...)
"""
import os, shutil, subprocess
import modal

app = modal.App('bowen-v2-v3')
vol = modal.Volume.from_name('bowen-world')
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('torch==2.6.0', 'numpy==2.1.3')
         .add_local_file('world_model.py', '/root/app/world_model.py'))
RUNS = {'bowen_live': {'WM_LIVE': '1', 'WM_MIX': '1'},
        'bowen_live_mix3': {'WM_LIVE': '1', 'WM_MIX': '3'},
        'bowen_v3_hier': {'WM_LIVE': '1', 'WM_MIX': '3', 'WM_HIER': '1'}}


@app.function(image=image, gpu='A10G', cpu=6, memory=32768, timeout=4 * 3600, volumes={'/root/app/results/world': vol})
def train(name: str, env: dict):
    w = '/root/app/results/world'
    os.makedirs(f'{w}/{name}', exist_ok=True)
    shutil.copy(f'{w}/desc.json', f'{w}/{name}/desc.json')
    e = dict(os.environ, WM_DIR=name, PYTHONUNBUFFERED='1', **env)
    r = subprocess.run(['python', '/root/app/world_model.py'], cwd='/root/app', env=e, capture_output=True, text=True)
    open(f'{w}/{name}/train.log', 'w').write(r.stdout + r.stderr)
    vol.commit()
    return name, r.returncode, (r.stdout + r.stderr)[-3000:]


@app.local_entrypoint()
def main():
    for name, code, tail in train.starmap(list(RUNS.items())):
        print(f'==== {name} exit {code}\n{tail}')
