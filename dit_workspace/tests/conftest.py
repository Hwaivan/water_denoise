import sys
from pathlib import Path
import torch
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_num_threads(1)


def pytest_configure(config):
    if not config.option.basetemp:
        # Keep generated WAVs/checkpoints inside this standalone workspace.
        directory = Path(__file__).parent / "_artifacts"
        directory.mkdir(exist_ok=True)
        config.option.basetemp = str(directory / uuid.uuid4().hex)
