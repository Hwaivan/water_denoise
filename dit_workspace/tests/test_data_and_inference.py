import numpy as np
import pytest
import soundfile as sf
import torch
from dit.data.dataset import AudioDataset,_shared_limit,build_dataloader
from dit.data.audio_io import save_audio,load_audio
from dit.factory import build_components
from dit.samplers.base import enhance_long_waveform,SampleResult
from test_workflow import make_config


def test_data_pair_shared_gain_and_float_output(tmp_path):
    c,_=make_config(tmp_path)
    ds=AudioDataset(c['data'],'train',7)
    a=ds[0]
    assert torch.equal(a['noise'],a['noisy']-a['clean'])
    clean,noisy=torch.tensor([1.,-2.]),torch.tensor([3.,-4.])
    x,y=_shared_limit(clean,noisy,.5)
    torch.testing.assert_close(x,clean/8)
    torch.testing.assert_close(y,noisy/8)
    save_audio(str(tmp_path/'float.wav'),noisy,16000)
    torch.testing.assert_close(load_audio(str(tmp_path/'float.wav'),16000),noisy)


@pytest.mark.parametrize('mode',['real','white','mixed'])
def test_online_noise_power_and_epoch(tmp_path,mode):
    c,_=make_config(tmp_path)
    clean_list=tmp_path/'clean.txt'
    clean_list.write_text(str(tmp_path/'clean.wav')+'\n')
    noise_list=tmp_path/'noise.txt'
    noise_list.write_text(str(tmp_path/'noisy.wav')+'\n')
    c['data'].update(data_mode='on_the_fly',noise_mode=mode,train_manifest=str(clean_list),
                     noise_manifest=str(noise_list),white_noise_probability=0.,white_noise_ratio=.3,
                     snr_min=-3.,snr_max=-3.)
    ds=AudioDataset(c['data'],'train',42)
    item=ds[0]
    n=item['length']
    actual=10*torch.log10(item['clean'][:n].square().mean()/item['noise'][:n].square().mean())
    assert actual.item()==pytest.approx(-3.,abs=1e-4)
    torch.testing.assert_close(item['noisy'],ds[0]['noisy'],atol=0,rtol=0)
    ds.set_epoch(1)
    assert not torch.equal(item['noisy'],ds[0]['noisy'])


def test_short_input_and_chunk_reconstruction(tmp_path):
    c,_=make_config(tmp_path)
    model,_,_,_,sampler=build_components(c)
    short=sampler.sample_waveform(model,torch.randn(1,7))
    assert short.waveform.shape==(1,7)
    class Identity:
        representation=sampler.representation
        def sample_waveform(self,model,noisy,lengths,sample_rate,generator):
            return SampleResult(torch.empty(0),noisy,1,0.,0.)
    x=torch.randn(531)
    result=enhance_long_waveform(Identity(),model,x,16000,chunk_samples=128,overlap_samples=32)
    torch.testing.assert_close(result.waveform[0],x)
    assert result.nfe==6


def test_silent_evaluation_export_and_variable_lengths(tmp_path):
    import json
    from dit.engine.evaluator import evaluate
    from dit.utils.logging import create_logger
    c,_=make_config(tmp_path)
    sf.write(tmp_path/'silent.wav',np.zeros(179,dtype=np.float32),16000,subtype='FLOAT')
    with open(c['data']['test_manifest'],'a') as f:
        f.write(str(tmp_path/'silent.wav')+'\t'+str(tmp_path/'silent.wav')+'\n')
    c['training']['batch_size']=2
    ds=AudioDataset(c['data'],'test',1)
    model,_,_,_,sampler=build_components(c)
    summary=evaluate(model,sampler,build_dataloader(ds,c,False,1),torch.device('cpu'),c,
        'unused.pt',str(tmp_path/'eval'),1,torch.Generator().manual_seed(1),
        create_logger('test_variable_lengths',str(tmp_path/'eval'),filename='evaluation.log'))
    assert summary['count']==2 and summary['valid_count']==1 and summary['invalid_count']==1
    assert sf.info(tmp_path/'eval/enhanced_wavs/silent.wav').frames==179


def test_oom_fallback_reduces_chunks(tmp_path):
    c,_=make_config(tmp_path)
    model,rep,_,_,_=build_components(c)
    class SimulatedOOM:
        representation=rep
        def sample_waveform(self,model,noisy,lengths,sample_rate,generator):
            if noisy.shape[-1]>128:
                raise torch.cuda.OutOfMemoryError('synthetic OOM for fallback test')
            return SampleResult(torch.empty(0),noisy,1,0.,0.)
    x=torch.randn(200)
    result=enhance_long_waveform(SimulatedOOM(),model,x,16000,chunk_samples=256,overlap_samples=32)
    torch.testing.assert_close(result.waveform[0],x)
    assert result.nfe==2
