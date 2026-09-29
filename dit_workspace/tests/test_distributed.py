import torch
import pytest
from test_workflow import make_config,run_cli
from dit.utils.distributed import resolve_training_launch


def test_distributed_launch_resolution():
    c=resolve_training_launch(0,'cpu',environ={},cuda_available=False,cuda_device_count=0)
    assert c.mode=='cpu' and not c.spawn
    c=resolve_training_launch(2,'cuda',environ={},cuda_available=True,cuda_device_count=2)
    assert c.spawn and c.num_gpus==2
    c=resolve_training_launch(None,'cpu',environ={'WORLD_SIZE':'2','LOCAL_WORLD_SIZE':'2'},
                             cuda_available=False,cuda_device_count=0)
    assert c.mode=='external_ddp' and not c.spawn


def test_two_rank_cpu_training(tmp_path):
    if not torch.distributed.is_gloo_available():
        pytest.skip('This torch build has no Gloo backend')
    _,path=make_config(tmp_path)
    run_cli('tests/ddp_smoke.py',path,tmp_path/'rendezvous')
    state=torch.load(tmp_path/'run/checkpoints/last.pt',weights_only=False)
    assert state['global_step']==1 and len(state['rank_states'])==2
    assert (tmp_path/'run/checkpoints/best.pt').is_file()
