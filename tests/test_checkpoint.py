"""Checkpoint save/restore smoke test."""

from pathlib import Path

import torch

from utils.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters())
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        {"epoch": 1, "model": model.state_dict(), "optimizer": optimizer.state_dict()},
        str(path),
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    restored = load_checkpoint(str(path), model, optimizer)
    assert restored["epoch"] == 1
    for name, value in model.state_dict().items():
        assert torch.equal(value, original[name])

