#!/usr/bin/env python3
"""Create a complete trainable runtime YAML from an existing DiT YAML."""
import argparse
from pathlib import Path
import yaml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--mode", choices=("log_mel", "complex_plus_log_mel"), required=True)
    p.add_argument("--process", choices=("flow", "score"), default="flow")
    p.add_argument("--data-mode", choices=("paired", "on_the_fly"), default="paired")
    for key in ("train", "valid", "test", "noise"):
        p.add_argument(f"--{key}-manifest", default="")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--segment-seconds", type=float, default=5.0)
    p.add_argument("--n-fft", type=int, default=512)
    p.add_argument("--win-length", type=int, default=400)
    p.add_argument("--hop-length", type=int, default=100)
    p.add_argument("--mel-bands", type=int, default=128)
    p.add_argument("--mel-normalization", choices=("dataset", "utterance"), default="dataset")
    p.add_argument("--hidden", type=int, default=768)
    p.add_argument("--depth", type=int, default=10)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--mlp-ratio", type=float, default=2.0)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.0001)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--amp", choices=("true", "false"), default="true")
    p.add_argument("--sampling-valid-batches", type=int, default=4)
    p.add_argument("--eval-interval", type=int, default=5)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--snr-min", type=float, default=-10)
    p.add_argument("--snr-max", type=float, default=-5)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    base_path, output = Path(a.base).resolve(), Path(a.output).resolve()
    cfg = yaml.safe_load(base_path.read_text(encoding="utf-8-sig"))
    run_dir = output.parent
    if a.data_mode == "paired" and not a.train_manifest:
        p.error("paired mode requires --train-manifest")
    if a.data_mode == "on_the_fly" and (not a.train_manifest or not a.noise_manifest):
        p.error("on_the_fly mode requires --train-manifest and --noise-manifest")
    if not a.valid_manifest or not a.test_manifest:
        p.error("--valid-manifest and --test-manifest are required")
    if a.hidden % a.heads:
        p.error("hidden dimension must be divisible by attention heads")

    cfg["experiment"].update(name=a.name, output_dir=str(run_dir), seed=a.seed)
    cfg["data"].update(
        data_mode=a.data_mode, sample_rate=a.sample_rate, segment_seconds=a.segment_seconds,
        train_manifest=str(Path(a.train_manifest).resolve()),
        valid_manifest=str(Path(a.valid_manifest).resolve()),
        test_manifest=str(Path(a.test_manifest).resolve()),
        noise_manifest=(str(Path(a.noise_manifest).resolve()) if a.data_mode == "on_the_fly" else None),
        snr_min=a.snr_min, snr_max=a.snr_max, mono=True, num_workers=a.num_workers,
    )
    cfg["stft"].update(n_fft=a.n_fft, win_length=a.win_length, hop_length=a.hop_length,
                       center=True, onesided=True)
    cfg["compression"]["crop_frames"] = round(a.segment_seconds * a.sample_rate) // a.hop_length + 1
    cfg["model"].update(tokenization="frame1d", name="dit", position_embedding="sincos_1d",
                        hidden_size=a.hidden, depth=a.depth, num_heads=a.heads, mlp_ratio=a.mlp_ratio)
    cfg["model"].pop("patch_size", None)
    cfg["training"].update(batch_size=a.batch, epochs=a.epochs, learning_rate=a.lr,
                           amp=(a.amp == "true"), score_loss_weighting="sigma2" if a.process == "score" else "none")
    cfg["validation"].update(eval_interval=a.eval_interval, max_sampling_batches=a.sampling_valid_batches)
    cfg["inference"].update(chunk_seconds=a.segment_seconds, overlap_seconds=0.)
    mel = dict(n_mels=a.mel_bands, f_min=0., f_max=a.sample_rate / 2,
               norm="slaney", power=2., log_eps=1e-6, normalization=a.mel_normalization,
               stats_path=(str(run_dir / "mel_stats.pt") if a.mel_normalization == "dataset" else None))
    if a.mode == "log_mel":
        cfg["representation"] = {"type": "log_mel", "mel": mel}
        cfg["conditioning"] = {"type": "none"}
        cfg["reconstruction"] = {"type": "mel_gain_noisy_phase", "gain_min": 0.05, "gain_max": 4.0}
        cfg["process"]["noise_convention"] = "real_unit_variance"
    else:
        cfg["representation"] = {"type": "complex_ri", "phase_source": "noisy"}
        cfg["conditioning"] = {"type": "complex_plus_log_mel", "mel": mel}
        cfg.pop("reconstruction", None)
        cfg["process"]["noise_convention"] = "circular_complex_unit_energy"
    cfg["process"]["type"] = a.process
    if a.process == "flow":
        cfg["process"]["flow"] = {"source": "gaussian"}
        cfg["sampler"] = {"name": "euler", "num_steps": a.steps or 10, "use_ema": True}
    else:
        cfg["sampler"] = {"name": "pc", "num_steps": a.steps or 30, "use_ema": True,
                          "predictor": "euler_maruyama", "corrector": "none", "corrector_steps": 0,
                          "corrector_step_size": 0.5}
    run_dir.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"Created {output} ({a.mode}, {a.process}, {a.data_mode})")
    if a.mel_normalization == "dataset":
        print("Mel stats to prepare:", mel["stats_path"])
    else:
        print("Utterance normalization: no dataset stats file required")


if __name__ == "__main__":
    main()
