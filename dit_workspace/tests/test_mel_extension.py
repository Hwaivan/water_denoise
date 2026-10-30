"""CPU unit tests for both Mel routes, Score and Flow, and WAV reconstruction."""
from pathlib import Path
import hashlib
import pytest
import torch
import yaml

from dit.factory import build_components
from dit.representations.mel import LogMelRepresentation, slaney_mel_bank
from dit.utils.config import validate_config


def _config(tmp_path, mode, process):
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "configs/dit_tiny.yaml").read_text(encoding="utf-8"))
    cfg["stft"].update(n_fft=64, win_length=48, hop_length=16, onesided=True)
    cfg["data"].update(sample_rate=16000, segment_seconds=0.032)
    cfg["compression"]["crop_frames"] = 33
    cfg["model"].update(name="dit", tokenization="frame1d", position_embedding="sincos_1d",
                        depth=1, hidden_size=32, num_heads=4, mlp_ratio=2, max_tokens=128)
    cfg["model"].pop("patch_size", None)
    cfg["process"]["type"] = process
    cfg["training"]["score_loss_weighting"] = "sigma2"
    cfg["sampler"] = {"name": "euler", "num_steps": 2, "use_ema": True} if process == "flow" else {
        "name": "pc", "num_steps": 2, "use_ema": True,
        "predictor": "euler_maruyama", "corrector": "none", "corrector_steps": 0,
        "corrector_step_size": 0.5}
    mel = dict(n_mels=16, f_min=0., f_max=8000., norm="slaney", power=2.,
               log_eps=1e-6, normalization="dataset", stats_path=str(tmp_path / "mel_stats.pt"))
    torch.save({"mean": torch.zeros(16), "std": torch.ones(16)}, mel["stats_path"])
    mel["stats_sha256"] = hashlib.sha256(Path(mel["stats_path"]).read_bytes()).hexdigest()
    if mode == "log_mel":
        cfg["representation"] = dict(type="log_mel", mel=mel)
        cfg["conditioning"] = dict(type="none")
        cfg["reconstruction"] = dict(type="mel_gain_noisy_phase", gain_min=.05, gain_max=4.)
        cfg["process"]["noise_convention"] = "real_unit_variance"
    else:
        cfg["representation"] = dict(type="complex_ri", phase_source="noisy")
        cfg["conditioning"] = dict(type="complex_plus_log_mel", mel=mel)
        cfg["process"]["noise_convention"] = "circular_complex_unit_energy"
    return cfg


@pytest.mark.parametrize("mel_bands", [16, 32])
def test_mel_filterbank_covers_bands(mel_bands):
    w = slaney_mel_bank(16000, 512, mel_bands, 0, 8000)
    assert w.shape == (mel_bands, 257)
    assert torch.isfinite(w).all() and torch.all(w >= 0)
    assert torch.all(w.sum(-1) > 0)


def test_pure_mel_identity_gain_and_hash(tmp_path):
    cfg = _config(tmp_path, "log_mel", "flow")
    rep = build_components(cfg)[1]
    noisy = torch.randn(1, 512) * .05
    encoded = rep.encode(noisy)
    assert encoded.shape == (1, 1, 16, 33)
    recovered = rep.decode(encoded, noisy.size(-1), noisy_waveform=noisy)
    torch.testing.assert_close(recovered, noisy, atol=1e-5, rtol=1e-4)
    original = cfg["representation"]["mel"]["stats_sha256"]
    cfg["representation"]["mel"]["stats_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256"):
        build_components(cfg)
    cfg["representation"]["mel"]["stats_sha256"] = original


@pytest.mark.parametrize("mode", ["log_mel", "complex_plus_log_mel"])
@pytest.mark.parametrize("process", ["score", "flow"])
def test_train_and_sample(mode, process, tmp_path):
    cfg = _config(tmp_path, mode, process)
    validate_config(cfg)
    model, representation, flow_process, objective, sampler = build_components(cfg)
    clean = torch.randn(1, 512) * .04
    noisy = clean + torch.randn_like(clean) * .03
    state = representation.encode(noisy)
    assert state.shape == (1, 1 if mode == "log_mel" else 2,
                           16 if mode == "log_mel" else 33, 33)
    loss = objective(model, clean, noisy, torch.Generator().manual_seed(4))["total"]
    assert torch.isfinite(loss)
    loss.backward()
    assert model.final_layer.projection.weight.grad is not None
    assert torch.isfinite(model.final_layer.projection.weight.grad).all()
    # This confirms score/velocity integrators accept optional Mel context.
    model.eval()
    result = sampler.sample_waveform(model, noisy, generator=torch.Generator().manual_seed(6))
    assert result.waveform.shape == noisy.shape
    assert torch.isfinite(result.waveform).all()
    assert result.nfe == 2
    if mode == "complex_plus_log_mel":
        assert model.frame_embed.projection.in_features == 4 * 33 + 16
    else:
        assert model.frame_embed.projection.in_features == 2 * 16


def test_mel_aux_invalid_ddpm(tmp_path):
    cfg = _config(tmp_path, "complex_plus_log_mel", "flow")
    cfg["process"]["type"] = "ddpm"
    with pytest.raises(ValueError, match="supports Score and Flow"):
        validate_config(cfg)
