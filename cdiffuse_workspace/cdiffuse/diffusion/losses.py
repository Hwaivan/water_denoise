import torch


def masked_mse(prediction, target, lengths):
    """MSE over valid samples only; prediction/target ``[B,1,L]``, lengths ``[B]``."""
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("loss inputs must be equal [B,1,L]")
    mask = torch.arange(prediction.shape[-1], device=prediction.device)[None, None] < lengths[:, None, None]
    loss = ((prediction-target).square()*mask).sum()/mask.sum().clamp_min(1)
    if not torch.isfinite(loss): raise FloatingPointError("non-finite masked MSE")
    return loss
