"""End-to-end model and STFT smoke tests."""

import torch

from losses import LossManager
from models.dccrn import DCCRN
from utils.audio import STFTFrontend


def _small_model() -> DCCRN:
    return DCCRN(
        stft_config={
            "n_fft": 64,
            "win_length": 64,
            "hop_length": 16,
            "window": "hann",
            "center": True,
        },
        encoder_channels=[2, 4],
        kernel_size=[5, 2],
        stride=[2, 1],
        lstm_layers=1,
        lstm_hidden_size=8,
        reconstruction_mode="dccrn_e",
    )


def test_stft_inverse_reconstructs_waveform() -> None:
    frontend = STFTFrontend(n_fft=64, win_length=64, hop_length=16)
    waveform = torch.randn(2, 1024)
    reconstructed = frontend.inverse(frontend.transform(waveform), length=waveform.shape[-1])
    assert reconstructed.shape == waveform.shape
    assert torch.allclose(reconstructed, waveform, atol=1.0e-5, rtol=1.0e-4)


def test_dccrn_forward_has_finite_length_preserving_outputs() -> None:
    model = _small_model().eval()
    mixture = torch.randn(2, 1024)
    lengths = torch.tensor([1024, 800])
    with torch.inference_mode():
        outputs = model(mixture, lengths)
    assert outputs["waveform"].shape == mixture.shape
    assert outputs["mask"].shape == outputs["spectrum"].shape
    assert torch.is_complex(outputs["mask"])
    assert torch.isfinite(outputs["waveform"]).all()
    assert torch.count_nonzero(outputs["waveform"][1, 800:]) == 0


def test_single_batch_can_optimize() -> None:
    model = _small_model().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    mixture = torch.randn(2, 1024)
    target = torch.randn(2, 1024)
    lengths = torch.tensor([1024, 900])
    outputs = model(mixture, lengths)
    loss = LossManager({"si_snr": {"weight": 1.0}})(outputs, target, lengths)["total"]
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()

