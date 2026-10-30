"""Only component assembly entry point used by train, evaluate and infer."""
from dit.models import ConditionalDiT
from dit.models.frame_dit import FrameConditionalDiT
from dit.representations import SpectralRepresentation
from dit.processes.score_sde import ScoreProcess
from dit.processes.ddpm import DDPMProcess
from dit.processes.flow_matching import FlowProcess
from dit.objectives import RegressionObjective
from dit.samplers.score_pc import ScorePC
from dit.samplers.base import WaveformSampler
from dit.samplers.ddpm import DDPMSampler
from dit.samplers.flow_ode import FlowODE


def build_representation(config):
    compression = {k: v for k, v in config["compression"].items() if k != "crop_frames"}
    return SpectralRepresentation(**config["representation"], **config["stft"], **compression)


def build_model(config, representation):
    values = dict(config["model"])
    tokenization = values.pop("tokenization", "patch2d")
    if tokenization == "patch2d":
        return ConditionalDiT(channels=representation.channels, **values)
    if tokenization == "frame1d":
        # A baseline-derived config may still contain patch-only geometry.
        values.pop("patch_size", None)
        stft = config["stft"]
        bins = stft["n_fft"] // 2 + 1 if stft.get("onesided", True) else stft["n_fft"]
        return FrameConditionalDiT(frequency_bins=bins, channels=representation.channels, **values)
    raise ValueError("Unknown model tokenization: " + str(tokenization))


def build_process(config, representation):
    values = config["process"]
    convention = values.get("noise_convention") or (
        "circular_complex_unit_energy" if representation.channels == 2 else "real_unit_variance")
    if representation.channels == 1 and convention != "real_unit_variance":
        raise ValueError("Magnitude needs real_unit_variance")
    if values["type"] == "score":
        if representation.channels == 2 and convention != "circular_complex_unit_energy":
            raise ValueError("Score complex baseline requires circular unit energy noise")
        return ScoreProcess(convention, **values["score"])
    if values["type"] == "ddpm":
        return DDPMProcess(convention, **values["ddpm"])
    if values["type"] == "flow":
        return FlowProcess(convention, **values["flow"])
    raise ValueError(values["type"])


# def build_objective(config, process, representation):
#     return RegressionObjective(representation, process, config["compression"].get("crop_frames"))

def build_objective(config, process, representation):
    return RegressionObjective(
        representation,
        process,
        config["compression"].get("crop_frames"),
        score_loss_weighting=config["training"].get(
            "score_loss_weighting",
            "none",
        ),
    )

def build_sampler(config, process, representation):
    values = dict(config["sampler"])
    values.pop("use_ema", None)
    name = values.pop("name", "pc")
    if process.kind == "score" and name == "pc":
        solver = ScorePC(process, **values)
    elif process.kind == "ddpm" and name == "ddpm":
        solver = DDPMSampler(process, **values)
    elif process.kind == "flow" and name in ("euler", "heun"):
        solver = FlowODE(process, solver=name, **values)
    else:
        raise ValueError("Sampler does not match configured process")
    return WaveformSampler(representation, solver)


def build_components(config):
    representation = build_representation(config)
    model = build_model(config, representation)
    process = build_process(config, representation)
    return (model, representation, process, build_objective(config, process, representation),
            build_sampler(config, process, representation))
