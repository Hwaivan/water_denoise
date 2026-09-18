from typing import Any, Dict, Optional

import torch
from torch import nn

from .si_snr import SISNRLoss, si_snr
from .smse import SMSELoss
from .waveform import WaveformL1Loss, WaveformMSELoss


class LossManager(nn.Module):
    def __init__(
        self,
        config: Dict[str, Any],
        stft_config: Dict[str, Any],
    ) -> None:
        super().__init__()

        components = config.get("components", {})

        self.losses = nn.ModuleDict()
        self.weights: Dict[str, float] = {}

        for name, loss_config in components.items():
            enabled = bool(loss_config.get("enabled", True))
            weight = float(loss_config.get("weight", 1.0))

            if not enabled or weight == 0.0:
                continue

            if name == "si_snr":
                loss_module = SISNRLoss()
            elif name == "waveform_l1":
                loss_module = WaveformL1Loss()
            elif name == "waveform_mse":
                loss_module = WaveformMSELoss()
            elif name == "smse":
                loss_module = SMSELoss(stft_config)
            else:
                raise ValueError(
                    f"Unknown loss '{name}'. "
                    "Available losses: "
                    "si_snr, waveform_l1, waveform_mse, smse"
                )

            self.losses[name] = loss_module
            self.weights[name] = weight

        if len(self.losses) == 0:
            raise ValueError("At least one loss must be enabled")

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        target: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        results: Dict[str, torch.Tensor] = {}
        total = target.new_zeros(())

        for name, loss_module in self.losses.items():
            value = loss_module(outputs, target, lengths)
            weighted_value = self.weights[name] * value

            results[f"{name}_loss"] = value
            results[f"{name}_weighted"] = weighted_value

            total = total + weighted_value

        # 始终单独计算正向SI-SNR指标
        results["si_snr"] = si_snr(
            outputs["waveform"],
            target,
            lengths,
        ).mean().detach()

        results["total"] = total

        return results

# class LossManager(nn.Module):
#     def __init__(self, config: Dict) -> None:
#         super().__init__()

#         components = config.get("components", {})

#         self.losses = nn.ModuleDict()
#         self.weights = {}

#         for name, loss_config in components.items():
#             enabled = bool(loss_config.get("enabled", True))
#             weight = float(loss_config.get("weight", 1.0))

#             if not enabled or weight == 0.0:
#                 continue

#             if name not in LOSS_REGISTRY:
#                 raise ValueError(
#                     f"Unknown loss '{name}'. "
#                     f"Available losses: {list(LOSS_REGISTRY)}"
#                 )

#             self.losses[name] = LOSS_REGISTRY[name]()
#             self.weights[name] = weight

#         if not self.losses:
#             raise ValueError("At least one loss must be enabled")

#     def forward(
#         self,
#         outputs: Dict[str, torch.Tensor],
#         target: torch.Tensor,
#         lengths: Optional[torch.Tensor] = None,
#     ) -> Dict[str, torch.Tensor]:

#         results = {}
#         total = target.new_zeros(())

#         for name, loss_module in self.losses.items():
#             value = loss_module(outputs, target, lengths)
#             weighted_value = self.weights[name] * value

#             results[name] = value
#             results[f"{name}_weighted"] = weighted_value
#             total = total + weighted_value

#         # SI-SNR始终作为监控指标，不一定参与训练
#         results["si_snr"] = si_snr(
#             outputs["waveform"],
#             target,
#             lengths,
#         ).mean().detach()

#         results["total"] = total
#         return results