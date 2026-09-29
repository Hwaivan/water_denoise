"""Real CLI CPU integration tests; tiny generated WAVs, never production data."""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import soundfile as sf
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def make_config(directory, kind="score", representation="complex_ri"):
    directory.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((ROOT / "configs/dit_tiny.yaml").read_text())
    t = np.arange(320, dtype=np.float32)/16000
    clean = .2*np.sin(2*np.pi*800*t)
    noisy = clean + np.random.default_rng(4).normal(0,.07,clean.shape).astype(np.float32)
    sf.write(str(directory/"clean.wav"),clean,16000,subtype="FLOAT")
    sf.write(str(directory/"noisy.wav"),noisy,16000,subtype="FLOAT")
    manifest = directory/"pairs.txt"
    manifest.write_text(str(directory/"noisy.wav")+"\t"+str(directory/"clean.wav")+"\n")
    for split in ("train","valid","test"):
        config["data"][split+"_manifest"] = str(manifest)
    config["experiment"]["output_dir"] = str(directory/"run")
    config["experiment"]["name"] = "test_"+kind+"_"+representation
    config["process"]["type"] = kind
    config["representation"]["type"] = representation
    if representation == "magnitude":
        config["process"]["noise_convention"] = "real_unit_variance"
    if kind != "score":
        config["sampler"] = dict(name="ddpm" if kind=="ddpm" else "euler",num_steps=3,use_ema=True)
        config["process"]["ddpm"]["num_steps"] = 3
    path = directory/"config.yaml"
    path.write_text(yaml.safe_dump(config),encoding="utf-8")
    return config,path


def run_cli(script, *args):
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run([sys.executable,str(ROOT/script),*map(str,args)],cwd=ROOT,
                            env=env,text=True,capture_output=True,timeout=90)
    assert result.returncode == 0, result.stdout+result.stderr
    return result


@pytest.mark.parametrize("representation", ["complex_ri","magnitude"])
def test_score_cli_closed_loop(tmp_path, representation):
    config,path = make_config(tmp_path,representation=representation)
    run_cli("train.py","--config",path,"--device","cpu")
    checkpoint = tmp_path/"run/checkpoints/last.pt"
    state = torch.load(checkpoint,weights_only=False)
    assert set(("model","ema_model","optimizer","scheduler","scaler","epoch","global_step",
                "best_metric","config","rng_state","rank_states")) <= set(state)
    assert state["global_step"] == 1
    record = json.loads((tmp_path/"run/metrics.jsonl").read_text().splitlines()[-1])
    assert np.isfinite(record["train/loss"])
    assert "val/loss" in record and "val/SI-SNRi" in record
    assert (tmp_path/"run/checkpoints/best.pt").is_file()
    assert list((tmp_path/"run/tensorboard").glob("events.*"))
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    events=EventAccumulator(str(tmp_path/"run/tensorboard")).Reload()
    assert {'train/loss','val/loss','val/score_loss','val/SI-SNRi','val/SDRi'} <= set(events.Tags()['scalars'])
    runtime = tmp_path/"run/runtime_config.yaml"
    run_cli("evaluate.py","--config",runtime,"--checkpoint",checkpoint,"--output-dir",tmp_path/"evaluation","--device","cpu")
    out = tmp_path/"evaluation"
    for name in ("per_file_metrics.csv","summary_metrics.json","enhanced_wavs/noisy.wav","evaluation.log"):
        assert (out/name).is_file()
    import csv
    with (out/'per_file_metrics.csv').open(encoding='utf-8-sig') as stream:
        reader=csv.DictReader(stream)
        assert reader.fieldnames==['file_id','input_sdr','output_sdr','sdri','input_si_snr','output_si_snr',
                                   'si_snri','duration','inference_time','rtf','nfe','valid','error']
    summary=json.loads((out/"summary_metrics.json").read_text())
    assert summary["count"] == summary["valid_count"] == 1 and summary["invalid_count"] == 0
    assert summary["NFE"] == 8  # Two overlapping chunks * 2 predictors * (1+1).
    assert set(("mean","std","median","p25","p75")) <= set(summary["si_snri"])
    run_cli("infer.py","--config",runtime,"--checkpoint",checkpoint,"--input",tmp_path/"noisy.wav",
            "--output",tmp_path/"enhanced.wav","--device","cpu")
    audio, rate = sf.read(tmp_path/"enhanced.wav")
    assert rate == 16000 and len(audio) == 320 and np.isfinite(audio).all()
    # Full restore then one further tiny batch, not a formal training run.
    config["training"]["epochs"] = 2
    path.write_text(yaml.safe_dump(config))
    run_cli("train.py","--config",path,"--resume",checkpoint,"--device","cpu")
    resumed = torch.load(checkpoint,weights_only=False)
    assert resumed["global_step"] == 2 and resumed["epoch"] == 1
    config["experiment"]["output_dir"] = str(tmp_path/"uninterrupted")
    path.write_text(yaml.safe_dump(config))
    run_cli("train.py","--config",path,"--device","cpu")
    full = torch.load(tmp_path/"uninterrupted/checkpoints/last.pt",weights_only=False)
    for name,value in resumed["model"].items():
        torch.testing.assert_close(value, full["model"][name],atol=0,rtol=0)


def test_ema_and_metric_definitions():
    from dit.utils.ema import ExponentialMovingAverage
    from dit.metrics.audio_metrics import si_snr,scale_dependent_sdr,compute_audio_metrics
    model = torch.nn.Linear(2,1)
    ema = ExponentialMovingAverage(model,.5)
    old = model.weight.detach().clone()
    with torch.no_grad():
        model.weight.add_(2)
    ema.update(model)
    torch.testing.assert_close(ema.shadow["weight"],old+1)
    with ema.average_parameters(model):
        torch.testing.assert_close(model.weight,old+1)
    torch.testing.assert_close(model.weight,old+2)
    ref = torch.tensor([1.,-1.,1.,-1.])
    torch.testing.assert_close(si_snr(2*ref+3,ref),si_snr(ref,ref))
    assert abs(scale_dependent_sdr(2*ref,ref).item()) < 1e-6
    row = compute_audio_metrics(ref,ref,ref+torch.tensor([.1,.3,-.2,.1]))[0]
    assert row["si_snri"] == pytest.approx(row["output_si_snr"]-row["input_si_snr"])
    assert row["sdri"] == pytest.approx(row["output_sdr"]-row["input_sdr"])
    assert not compute_audio_metrics(ref,torch.zeros_like(ref),ref)[0]["valid"]
    assert not compute_audio_metrics(ref*float('nan'),ref,ref)[0]["valid"]


@pytest.mark.parametrize("kind", ["ddpm","flow"])
def test_other_process_cli(tmp_path,kind):
    config,path=make_config(tmp_path,kind=kind)
    run_cli("train.py","--config",path,"--device","cpu")
    ckpt=tmp_path/"run/checkpoints/last.pt"
    run_cli("evaluate.py","--config",path,"--checkpoint",ckpt,"--output-dir",tmp_path/"eval","--device","cpu")
    summary=json.loads((tmp_path/"eval/summary_metrics.json").read_text())
    assert summary["valid_count"]==1 and summary["NFE"]==6
    row=json.loads((tmp_path/"run/metrics.jsonl").read_text().splitlines()[-1])
    assert ("val/noise_loss" if kind=="ddpm" else "val/flow_loss") in row
    assert "train/sigma_mean" not in row
