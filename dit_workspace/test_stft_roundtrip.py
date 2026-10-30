#!/usr/bin/env python3
"""
测试与当前 DiT 相同的波形频谱往返变换：

waveform
  -> complex STFT
  -> power-law compression
  -> inverse power-law compression
  -> iSTFT
  -> reconstructed waveform

metrics 在保存 WAV 之前计算，因此主要反映 STFT/压缩/逆压缩/iSTFT
本身引入的数值误差，而不是 WAV 文件量化误差。

list 支持：
1. 单列：
   /path/to/audio.wav

2. 配对：
   noisy.wav<TAB>clean.wav

通过 LIST_COLUMN 选择测试哪一列。
"""

import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from dit.representations.stft import ComplexSTFTTransform


# ============================================================
# 全局参数配置区
# 只需要修改这里
# ============================================================

# 输入 list
LIST_FILE = (
    "/data/huayifan/water_denoise/"
    "data/ShipsEar-12class_W5H1/train_16k_signal.list"
)

# 如果 list 是 noisy<TAB>clean：
#   0 = noisy
#   1 = clean
# 如果是单列 list，保持 0 即可。
LIST_COLUMN = 0

# 建议直接读取对应实验的 runtime_config.yaml，
# 这样 STFT / compression 参数与模型严格一致。
CONFIG_FILE = (
    "/data/huayifan/water_denoise/dit_workspace/"
    "runs/dit_score_complex_frame_token_5s_onthefly/runtime_config.yaml"
)

# 输出目录
OUTPUT_DIR = (
    "/data/huayifan/water_denoise/dit_workspace/"
    "runs/stft_roundtrip"
)

# 0 = 处理全部文件
LIMIT = 0

# 保存格式：
#   "FLOAT"  ：推荐，用于尽量隔离 STFT round-trip 误差
#   "PCM_16" ：会额外引入 16-bit 量化误差
#   "PCM_24"
SAVE_SUBTYPE = "FLOAT"

# 当 CONFIG_FILE = "" 或 None 时使用下面默认参数
DEFAULT_SAMPLE_RATE = 16000

DEFAULT_STFT = {
    "n_fft": 512,
    "win_length": 400,
    "hop_length": 100,
    "window": "hann",
    "center": True,
    "normalized": False,
    "onesided": True,
}

DEFAULT_COMPRESSION = {
    "alpha": 0.5,
    "beta": 0.15,
}

# 指标数值稳定项
METRIC_EPS = 1.0e-12


# ============================================================
# 一般不需要修改下面内容
# ============================================================


def load_config(path):
    if not path:
        return None

    import yaml

    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def parse_manifest(path, column):
    items = []

    with open(path, "r", encoding="utf-8-sig") as stream:
        for lineno, raw in enumerate(stream, 1):
            line = raw.strip()

            if not line or line.startswith("#"):
                continue

            fields = line.split("\t") if "\t" in line else line.split()

            if column >= len(fields):
                raise ValueError(
                    f"{path}:{lineno}: LIST_COLUMN={column}, "
                    f"but line has only {len(fields)} field(s): {line}"
                )

            items.append(fields[column])

    return items


def load_audio(path, expected_sr):
    waveform, sample_rate = sf.read(
        path,
        dtype="float32",
        always_2d=True,
    )

    if sample_rate != expected_sr:
        raise ValueError(
            f"{path}: sample rate {sample_rate} != expected {expected_sr}. "
            "Please resample explicitly before this test."
        )

    # 当前实验按单通道处理。
    waveform = waveform.mean(axis=1)

    return torch.from_numpy(
        np.ascontiguousarray(waveform)
    )


def compute_metrics(reference, estimate, eps=METRIC_EPS):
    reference = reference.double()
    estimate = estimate.double()

    error = estimate - reference

    signal_energy = reference.square().sum()
    error_energy = error.square().sum()

    # 当前 dit/metrics/audio_metrics.py 中 scale-dependent SDR 的定义：
    #
    # 10 log10( ||x||^2 / ||x_hat - x||^2 )
    #
    # 对这个“原信号 -> 重建信号”的 round-trip 测试，
    # 常规定义下的 SNR 与这里的 SDR 数值相同。
    sdr = 10.0 * torch.log10(
        (signal_energy + eps)
        / (error_energy + eps)
    )

    snr = sdr.clone()

    # SI-SNR
    estimate_zm = estimate - estimate.mean()
    reference_zm = reference - reference.mean()

    reference_energy = (
        reference_zm.square().sum().clamp_min(eps)
    )

    scale = (
        (estimate_zm * reference_zm).sum()
        / reference_energy
    )

    target = scale * reference_zm
    residual = estimate_zm - target

    si_snr = 10.0 * torch.log10(
        target.square().sum().clamp_min(eps)
        / residual.square().sum().clamp_min(eps)
    )

    relative_l2 = (
        torch.linalg.vector_norm(error)
        / torch.linalg.vector_norm(reference).clamp_min(eps)
    )

    return {
        "snr_db": float(snr),
        "sdr_db": float(sdr),
        "si_snr_db": float(si_snr),
        "mse": float(error.square().mean()),
        "mae": float(error.abs().mean()),
        "max_abs_error": float(error.abs().max()),
        "relative_l2_error": float(relative_l2),
    }


def make_output_name(path, index):
    stem = Path(path).stem
    return f"{index:05d}_{stem}_roundtrip.wav"


def main():
    config = load_config(CONFIG_FILE)

    if config is None:
        sample_rate = DEFAULT_SAMPLE_RATE
        stft_config = dict(DEFAULT_STFT)
        compression_config = dict(DEFAULT_COMPRESSION)

    else:
        sample_rate = int(
            config["data"]["sample_rate"]
        )

        stft_config = dict(
            config["stft"]
        )

        compression_config = {
            "alpha": float(
                config["compression"]["alpha"]
            ),
            "beta": float(
                config["compression"]["beta"]
            ),
        }

    print("============================================================")
    print("STFT round-trip test")
    print("============================================================")
    print("List file       :", LIST_FILE)
    print("List column     :", LIST_COLUMN)
    print("Config file     :", CONFIG_FILE)
    print("Output dir      :", OUTPUT_DIR)
    print("Sample rate     :", sample_rate)
    print("STFT            :", stft_config)
    print("Compression     :", compression_config)
    print("Save subtype    :", SAVE_SUBTYPE)
    print("============================================================")
    print()

    transform = ComplexSTFTTransform(
        **stft_config,
        **compression_config,
    ).cpu().eval()

    paths = parse_manifest(
        LIST_FILE,
        LIST_COLUMN,
    )

    if LIMIT > 0:
        paths = paths[:LIMIT]

    if not paths:
        raise ValueError(
            "No audio paths found in LIST_FILE."
        )

    output_dir = Path(OUTPUT_DIR)
    wav_dir = output_dir / "roundtrip_wavs"

    wav_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for index, path in enumerate(paths):
        waveform = load_audio(
            path,
            sample_rate,
        )[None]

        length = waveform.shape[-1]

        with torch.inference_mode():

            # ------------------------------------------------
            # 与模型一致的频谱变换
            # ------------------------------------------------

            compressed = transform.encode(
                waveform
            )

            # ------------------------------------------------
            # 与模型一致的逆变换
            # ------------------------------------------------

            reconstructed = transform.decode(
                compressed,
                length,
            )

        reference = waveform[0]
        estimate = reconstructed[0, :length]

        if not torch.isfinite(estimate).all():
            raise FloatingPointError(
                f"Non-finite reconstruction: {path}"
            )

        metrics = compute_metrics(
            reference,
            estimate,
        )

        output_path = (
            wav_dir
            / make_output_name(path, index)
        )

        sf.write(
            output_path,
            estimate.cpu().numpy(),
            sample_rate,
            subtype=SAVE_SUBTYPE,
        )

        row = {
            "index": index,
            "source": str(
                Path(path).resolve()
            ),
            "output": str(
                output_path.resolve()
            ),
            "sample_rate": sample_rate,
            "samples": length,
            "duration": length / sample_rate,
            "freq_bins": compressed.shape[-2],
            "frames": compressed.shape[-1],
            **metrics,
        }

        rows.append(row)

        print(
            f"[{index + 1}/{len(paths)}] "
            f"{Path(path).name} | "
            f"SDR={metrics['sdr_db']:.3f} dB | "
            f"SNR={metrics['snr_db']:.3f} dB | "
            f"SI-SNR={metrics['si_snr_db']:.3f} dB | "
            f"max|e|={metrics['max_abs_error']:.3e}"
        )

    # ========================================================
    # 保存逐文件结果
    # ========================================================

    csv_path = (
        output_dir
        / "roundtrip_metrics.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as stream:

        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    # ========================================================
    # 汇总
    # ========================================================

    summary = {
        "count": len(rows),
        "list": str(
            Path(LIST_FILE).resolve()
        ),
        "column": LIST_COLUMN,
        "sample_rate": sample_rate,
        "stft": stft_config,
        "compression": compression_config,
        "save_subtype": SAVE_SUBTYPE,
        "metrics_computed_before_wav_save": True,
    }

    metric_names = (
        "snr_db",
        "sdr_db",
        "si_snr_db",
        "mse",
        "mae",
        "max_abs_error",
        "relative_l2_error",
    )

    for name in metric_names:
        values = np.asarray(
            [row[name] for row in rows],
            dtype=np.float64,
        )

        summary[name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "median": float(
                np.median(values)
            ),
        }

    summary_path = (
        output_dir
        / "roundtrip_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as stream:

        json.dump(
            summary,
            stream,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("============================================================")
    print("Finished")
    print("============================================================")
    print("Metrics CSV :", csv_path)
    print("Summary JSON:", summary_path)
    print("WAVs        :", wav_dir)
    print("============================================================")


if __name__ == "__main__":
    main()
