"""Shared waveform preparation; process chooses targets and error semantics."""
import torch
from torch import nn
from dit.representations.stft import fit_spectrogram_frames


# class RegressionObjective(nn.Module):
#     def __init__(self, representation, process, crop_frames=None):
#         super().__init__()
#         self.representation, self.process = representation, process
#         self.crop_frames = crop_frames

class RegressionObjective(nn.Module):
    def __init__(
        self,
        representation,
        process,
        crop_frames=None,
        score_loss_weighting="none",
    ):
        super().__init__()

        if score_loss_weighting not in ("none", "sigma2"):
            raise ValueError(
                f"Unknown score_loss_weighting: {score_loss_weighting}"
            )

        self.representation = representation
        self.process = process
        self.crop_frames = crop_frames
        self.score_loss_weighting = score_loss_weighting

    def forward(self, model, clean_waveform, noisy_waveform, generator=None, deterministic=False):
        if clean_waveform.shape != noisy_waveform.shape:
            raise ValueError("Clean/noisy waveforms must be aligned")
        with torch.autocast(device_type=clean_waveform.device.type, enabled=False):
            clean = self.representation.encode(clean_waveform.float())
            condition = self.representation.encode(noisy_waveform.float())
            if self.crop_frames:
                maximum = max(0, clean.shape[-1] - self.crop_frames)
                start = 0 if deterministic or not maximum else int(torch.randint(
                    maximum+1, (1,), device=clean.device, generator=generator).item())
                clean, condition = fit_spectrogram_frames(clean, condition, self.crop_frames, start)
            pair = self.process.sample_training_pair(clean, condition, generator, deterministic)
        # predicted = model(state=pair.state, condition=condition, time=pair.time)
        # error = (predicted.float() - pair.target).square()
        # # Legacy complex score loss sums both channel errors; DDPM/flow use MSE.
        # loss = error.sum(1).mean() if self.process.kind == "score" else error.mean()
        predicted = model(
            state=pair.state,
            condition=condition,
            time=pair.time,
        )

        error = (
            predicted.float()
            - pair.target
        ).square()

        if self.process.kind == "score":

            if self.score_loss_weighting == "none":

                # 原始 score loss
                loss = error.sum(dim=1).mean()

            elif self.score_loss_weighting == "sigma2":

                # sigma^2-weighted score matching
                sigma = self.process.sde.std(
                    pair.time
                )

                sigma2 = sigma.square().reshape(
                    sigma.shape[0],
                    *([1] * (error.ndim - 1)),
                )

                loss = (
                    sigma2 * error
                ).sum(dim=1).mean()
 
            else:
                raise ValueError(
                    self.score_loss_weighting
                )

        else:
            loss = error.mean()
        return {"total": loss, self.process.loss_name: loss.detach(),
                **{k: v.detach() for k, v in pair.diagnostics.items()}}
