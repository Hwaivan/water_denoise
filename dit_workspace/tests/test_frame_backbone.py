"""Minimal frame geometry, factory, score backward and sampling checks."""
from pathlib import Path
import pytest
import torch
from dit.factory import build_components, build_model, build_representation
from dit.models.dit import ConditionalDiT
from dit.models.frame_dit import FrameConditionalDiT, FrameEmbed
from dit.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("channels", [1, 2])
def test_frame_backbone(channels):
    model = FrameConditionalDiT(257, channels, depth=1, hidden_size=32, num_heads=4)
    for frames in (17, 161):
        state = torch.randn(1, channels, 257, frames)
        output = model(state, state, torch.tensor([.3]))
        assert output.shape == state.shape and output.count_nonzero() == 0
        geometry = model.geometry(257, frames)
        assert geometry["token_count"] == geometry["frame_count"] == frames
        assert geometry["tokenization"] == "frame1d" and "patch_grid" not in geometry
        assert model.frame_embed(model.input_adapter(state, state)).shape == (1, frames, 32)
    assert model.frame_embed.projection.in_features == 2 * channels * 257
    assert model.final_layer.projection.out_features == channels * 257
    assert model.final_layer.modulation[-1].weight.count_nonzero() == 0
    tokens, time = torch.randn(1, 7, 32), torch.randn(1, 32)
    torch.testing.assert_close(model.blocks[0](tokens, time), tokens)
    # Nonzero, distinguishable values verify channel/frequency/time packing.
    embed = FrameEmbed(2 * channels, 3, 2 * channels * 3)
    with torch.no_grad():
        embed.projection.weight.copy_(torch.eye(2 * channels * 3))
        embed.projection.bias.zero_()
    x = torch.arange(2 * channels * 3 * 5.).reshape(1, 2 * channels, 3, 5)
    torch.testing.assert_close(embed(x), x.permute(0, 3, 1, 2).flatten(2))
    with torch.no_grad():
        model.final_layer.projection.bias.copy_(torch.arange(channels * 257.))
    output = model(state, state, torch.tensor([.3]))
    expected = torch.arange(channels * 257.).reshape(1, channels, 257, 1).expand_as(state)
    torch.testing.assert_close(output, expected)
    with pytest.raises(ValueError):
        model(torch.zeros(1, channels, 256, 17), torch.zeros(1, channels, 256, 17), torch.zeros(1))
    model.max_tokens = 16
    with pytest.raises(ValueError):
        model(state, state, torch.tensor([.3]))


def test_factory_and_legacy_checkpoint():
    config = load_config(ROOT / "configs/dit_score_complex_small.yaml")
    config["model"].update(depth=1, hidden_size=32, num_heads=4)
    rep = build_representation(config)
    torch.manual_seed(9)
    reference = ConditionalDiT(channels=rep.channels, **config["model"])
    torch.manual_seed(9)
    legacy = build_model(config, rep)
    assert type(legacy) is ConditionalDiT
    legacy.load_state_dict(reference.state_dict(), strict=True)
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(value, legacy.state_dict()[name], rtol=0, atol=0)
    config["model"]["tokenization"] = "patch2d"
    assert type(build_model(config, rep)) is ConditionalDiT
    for duration, frames in ((1, 161), (4, 641)):
        frame = load_config(ROOT / f"configs/dit_score_complex_frame_{duration}s.yaml")
        frame["model"].update(depth=1, hidden_size=32, num_heads=4)
        r = build_representation(frame)
        model = build_model(frame, r)
        assert type(model) is FrameConditionalDiT
        assert r.encode(torch.zeros(1, duration * frame["data"]["sample_rate"])).shape[-2:] == (257, frames)
        assert model.geometry(257, frames)["token_count"] == frames
    frame["stft"].update(n_fft=64, win_length=48, hop_length=16, onesided=False)
    r = build_representation(frame)
    assert build_model(frame, r).frequency_bins == 64


def test_frame_score_backward_and_sampler():
    config = load_config(ROOT / "configs/dit_tiny.yaml")
    config["model"].update(tokenization="frame1d", position_embedding="sincos_1d")
    config["sampler"]["num_steps"] = 1
    model, rep, process, objective, sampler = build_components(config)
    assert process.kind == "score"
    clean, noisy = torch.randn(1, 320), torch.randn(1, 320)
    loss = objective(model, clean, noisy, torch.Generator().manual_seed(3))["total"]
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert model.final_layer.projection.weight.grad.norm() > 0
    result = sampler.sample_waveform(model.eval(), noisy, generator=torch.Generator().manual_seed(4))
    assert result.waveform.shape == noisy.shape and torch.isfinite(result.waveform).all()
    assert result.representation.shape == rep.encode(noisy).shape
    assert result.nfe == 1 + config["sampler"]["corrector_steps"]
