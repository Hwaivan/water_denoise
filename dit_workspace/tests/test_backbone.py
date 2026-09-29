import pytest
import torch
from dit.models.dit import ConditionalDiT, PatchEmbed
from dit.models.embeddings import TimestepEmbedder, position_embedding
from dit.representations.spectral import SpectralRepresentation
from dit.representations.stft import complex_to_channels, channels_to_complex


def representation(kind="complex_ri"):
    return SpectralRepresentation(type=kind, n_fft=64, win_length=48, hop_length=16)


def test_stft_and_compression_roundtrip():
    r = representation()
    wave = torch.randn(2, 519)
    spec = r.stft(wave)
    torch.testing.assert_close(r.istft(spec, 519), wave, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(r.decompress(r.compress(spec)), spec)
    torch.testing.assert_close(channels_to_complex(complex_to_channels(spec)), spec)
    torch.testing.assert_close(r.decode(r.encode(wave), 519), wave, atol=1e-6, rtol=1e-5)


def test_magnitude_uses_noisy_phase_and_final_clamp():
    r = representation("magnitude")
    y = torch.randn(2, 519)
    state = r.encode(y)
    phase = r.stft(y).angle()
    torch.testing.assert_close(r.decode(state, 519, phase), y, atol=1e-6, rtol=1e-5)
    assert r.to_spectrum(-state, phase).abs().max() == 0
    other_phase = r.stft(torch.randn_like(y)).angle()
    assert not torch.allclose(r.decode(state, 519, other_phase), y)
    with pytest.raises(ValueError):
        r.decode(state, 519)


@pytest.mark.parametrize("bins", [256, 257])
@pytest.mark.parametrize("patch", [(4,4), (4,8), (2,4), (2,8), (8,8)])
@pytest.mark.parametrize("channels", [1, 2])
def test_shapes(bins, patch, channels):
    m = ConditionalDiT(channels, depth=2, hidden_size=32, num_heads=4, patch_size=patch)
    x = torch.randn(1, channels, bins, 17)
    out = m(state=x, condition=x, time=torch.tensor([.37]))
    assert out.shape == x.shape
    assert torch.count_nonzero(out) == 0
    tokens, grid, original = m.patch_embed(m.input_adapter(x, x))
    assert tokens.shape[1] == m.geometry(bins, 17)["token_count"]
    # Verify unpatchify preserves order and the last odd bin, independently of zeros.
    padded = torch.nn.functional.pad(x, (0, -17 % patch[1], 0, -bins % patch[0]))
    packed = padded.reshape(1, channels, grid[0], patch[0], grid[1], patch[1]).permute(0,2,4,3,5,1).flatten(3).flatten(1,2)
    torch.testing.assert_close(m.unpatchify(packed, grid, original), x)


def test_adaln_identity_and_time():
    m = ConditionalDiT(depth=2, hidden_size=32, num_heads=4)
    x, c = torch.randn(2, 7, 32), torch.randn(2, 32)
    torch.testing.assert_close(m.blocks[0](x, c), x)
    for block in m.blocks:
        assert block.modulation[-1].weight.count_nonzero() == 0
    e = TimestepEmbedder(32)
    torch.testing.assert_close(e(torch.tensor([0,1])), e(torch.tensor([0.,1.])))
    assert not torch.allclose(e(torch.tensor([.2])), e(torch.tensor([.8])))
    p = position_embedding((3, 7), 32, torch.device("cpu"))
    assert p.shape == (1,21,32)
    assert not torch.equal(p[:,0], p[:,1])
