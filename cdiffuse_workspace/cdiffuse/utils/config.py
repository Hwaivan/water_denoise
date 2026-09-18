from pathlib import Path
import yaml
def load_config(path):
    with open(path,encoding='utf-8') as f:c=yaml.safe_load(f)
    for s in ('project','data','conditioner','diffusion','model','training','sampler','evaluation'):
        if s not in c:raise KeyError(f'missing config section: {s}')
    c['_config_path']=str(Path(path).resolve());return c
