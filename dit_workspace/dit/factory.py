"""Unified component assembly; support log-Mel state and log-Mel auxiliary condition."""
from dit.models import ConditionalDiT
from dit.models.frame_dit import FrameConditionalDiT
from dit.representations import SpectralRepresentation, LogMelRepresentation
from dit.processes.score_sde import ScoreProcess
from dit.processes.ddpm import DDPMProcess
from dit.processes.flow_matching import FlowProcess
from dit.objectives import RegressionObjective
from dit.samplers.score_pc import ScorePC
from dit.samplers.base import WaveformSampler
from dit.samplers.ddpm import DDPMSampler
from dit.samplers.flow_ode import FlowODE


def _mel_codec(config, mel, reconstruction=None):
    compression = {k: v for k, v in config["compression"].items() if k != "crop_frames"}
    return LogMelRepresentation(
        mel=mel, reconstruction=reconstruction,
        sample_rate=int(config["data"]["sample_rate"]),
        **config["stft"], **compression,
    )


def build_representation(config):
    if config["representation"]["type"] == "log_mel":
        return _mel_codec(config, config["representation"]["mel"],
                          config.get("reconstruction"))
    compression = {k: v for k, v in config["compression"].items() if k != "crop_frames"}
    return SpectralRepresentation(**config["representation"], **config["stft"], **compression)


def build_auxiliary_encoder(config):
    cond = config.get("conditioning", {})
    if cond.get("type", "none") == "none":
        return None
    if cond["type"] == "complex_plus_log_mel":
        return _mel_codec(config, cond["mel"])
    raise ValueError("Unknown conditioning type: " + str(cond.get("type")))


def build_model(config, representation):
    values = dict(config["model"])
    tokenization = values.pop("tokenization", "patch2d")
    auxiliary_bins = int(config.get("conditioning", {}).get("mel", {}).get("n_mels", 0)) \
        if config.get("conditioning", {}).get("type") == "complex_plus_log_mel" else 0
    if tokenization == "patch2d":
        if auxiliary_bins:
            raise ValueError("Mel auxiliary conditioning currently requires frame1d")
        return ConditionalDiT(channels=representation.channels, **values)
    if tokenization == "frame1d":
        values.pop("patch_size", None)
        bins = getattr(representation, "bins", None)
        if bins is None:
            stft = config["stft"]
            bins = stft["n_fft"] // 2 + 1 if stft.get("onesided", True) else stft["n_fft"]
        return FrameConditionalDiT(frequency_bins=bins, channels=representation.channels,
                                   aux_condition_bins=auxiliary_bins, **values)
    raise ValueError("Unknown model tokenization: " + str(tokenization))


def build_process(config, representation):
    values = config["process"]
    convention = values.get("noise_convention") or (
        "circular_complex_unit_energy" if representation.channels == 2 else "real_unit_variance")
    if representation.channels == 1 and convention != "real_unit_variance":
        raise ValueError("Single-channel magnitude/Mel requires real_unit_variance")
    if values["type"] == "score":
        if representation.channels == 2 and convention != "circular_complex_unit_energy":
            raise ValueError("Score complex baseline requires circular unit energy noise")
        return ScoreProcess(convention, **values["score"])
    if values["type"] == "ddpm":
        return DDPMProcess(convention, **values["ddpm"])
    if values["type"] == "flow":
        return FlowProcess(convention, **values["flow"])
    raise ValueError(values["type"])


def build_objective(config, process, representation, auxiliary_encoder=None):
    return RegressionObjective(
        representation, process, config["compression"].get("crop_frames"),
        auxiliary_encoder=auxiliary_encoder,
        score_loss_weighting=config["training"].get("score_loss_weighting", "none"),
    )


def build_sampler(config, process, representation, auxiliary_encoder=None):
    values = dict(config["sampler"])
    values.pop("use_ema", None)
    name = values.pop("name", "pc")
    if process.kind == "score" and name == "pc":
        solver = ScorePC(process, **values)
    elif process.kind == "ddpm" and name == "ddpm":
        if auxiliary_encoder is not None:
            raise ValueError("DDPM + auxiliary Mel condition is not yet supported")
        solver = DDPMSampler(process, **values)
    elif process.kind == "flow" and name in ("euler", "heun"):
        solver = FlowODE(process, solver=name, **values)
    else:
        raise ValueError("Sampler does not match configured process")
    return WaveformSampler(representation, solver, auxiliary_encoder=auxiliary_encoder)


def build_components(config):
    representation = build_representation(config)
    auxiliary = build_auxiliary_encoder(config)
    model = build_model(config, representation)
    process = build_process(config, representation)
    return (model, representation, process,
            build_objective(config, process, representation, auxiliary),
            build_sampler(config, process, representation, auxiliary))
