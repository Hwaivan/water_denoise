import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 输入文件
# csv_path = Path(sys.argv[1] if len(sys.argv) > 1 else "per_file_metrics.csv")
csv_path = "/data/huayifan/water_denoise/dit_workspace/runs/dit_score_logmel_frame_token_5s_hidSz=768_numhead=12_smallMLP_loss=sigma2_zys_paired/evaluation/zys_test_all_no_corrector_30steps/per_file_metrics.csv"
csv_path = "/data/huayifan/water_denoise/dit_workspace/runs/dit_flow_complex_frame_token_5s_hidSz=1024_numhead=8_smallMLP_loss=sigma2_zys_paired/evaluation/zys_test_all_no_corrector_30steps/per_file_metrics.csv"
csv_path = "/data/huayifan/water_denoise/dit_workspace/runs/dit_score_complex_frame_token_5s_hidSz=1024_smallMLP_depth=8_real_loss=sigma2_zys_paired/evaluation/zys_test_all_no_corrector/per_file_metrics.csv"
csv_path = "/data/huayifan/water_denoise/dit_workspace/runs/dit_flow_logmel_frame_token_5s_hidSz=768_numhead=12_smallMLP_loss=sigma2_zys_paired/evaluation/zys_test_all_no_corrector_30steps/per_file_metrics.csv"
csv_path = "/data/huayifan/water_denoise/dit_workspace/runs/dit_score_complex_frame_token_5s_hidSz=1024_real_loss=sigma2_onthefly_-10~0dB/evaluation/zys_test_all_no_corrector_30steps/per_file_metrics.csv"
# 已知输入 SNR
snr_levels = np.array([-10, -9, -8, -7, -6, -5])

df = pd.read_csv(csv_path)
total = len(df)

# 排除无效样本
df = df[
    df["valid"].astype(str).str.strip().str.lower().isin(["true", "1"])
].copy()

metrics = [
    "input_sdr", "output_sdr", "sdri",
    "input_si_snr", "output_si_snr", "si_snri",
    "inference_time", "rtf", "nfe"
]

for col in metrics:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["input_sdr", "sdri", "si_snri"])

# 根据 input_sdr 匹配最近的预设 SNR
distances = np.abs(
    df["input_sdr"].to_numpy()[:, None] - snr_levels[None, :]
)

nearest = distances.argmin(axis=1)
matched = distances[np.arange(len(df)), nearest] < 0.5

excluded = len(df) - matched.sum()
df = df.loc[matched].copy()
df["snr_db"] = snr_levels[nearest[matched]]

# 各 SNR 组分别统计
summary = df.groupby("snr_db").agg(
    count=("file_id", "size"),
    input_sdr=("input_sdr", "mean"),
    output_sdr=("output_sdr", "mean"),
    sdri_mean=("sdri", "mean"),
    sdri_std=("sdri", "std"),
    input_si_snr=("input_si_snr", "mean"),
    output_si_snr=("output_si_snr", "mean"),
    si_snri_mean=("si_snri", "mean"),
    si_snri_std=("si_snri", "std"),
    inference_time=("inference_time", "mean"),
    rtf=("rtf", "mean"),
    nfe=("nfe", "mean"),
)

summary = summary.reindex(snr_levels).round(4)

# 保存结果
output_path = Path(csv_path).with_name("metrics_by_snr.csv")
summary.to_csv(output_path, encoding="utf-8-sig")

print(f"总样本数: {total}")
print(f"有效匹配样本数: {len(df)}")
print(f"SNR 未匹配样本数: {excluded}")
print("\n各 SNR 统计结果：")
print(summary.to_string())
print(f"\n结果已保存至: {output_path}")