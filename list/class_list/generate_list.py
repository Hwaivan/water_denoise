from pathlib import Path
import os
import re

# =========================
# 配置
# =========================
# wave_dir = Path("/data/huayifan/water_denoise/data/ShipsEar-12class_W5H1/test/16k")

# wave_dir = Path("/data/huayifan/water_denoise/data/ShipsEar-12class_W5H1/valid/16k")
# wave_dir = Path("/data/huayifan/water_denoise/data/ShipsEar-12class_W5H1/train")
# output_list = Path("shipsear_train_16k.list")
# output_list = Path("shipsear_valid_16k.list")
# output_list = Path("shipsear_test_16k.list")

# wave_dir = Path("/data/huayifan/water_denoise/data/ShipsEar-12class_W5H1/test/snr_-15dB")
# output_list = Path("./noisy/shipsear_test_16k_snr=-15dB.list")

base_model = "dit_score_complex_frame_token_5s_hidSz=1024_real_loss=sigma2_onthefly"
enhanced_dir = 'snr_0dB_no_corrector'
wave_dir = Path(f"/data/huayifan/water_denoise/dit_workspace/runs/{base_model}/evaluation/{enhanced_dir}/enhanced_wavs")
output_dir = Path(f"./enhanced/{base_model}")
os.makedirs(output_dir, exist_ok=True)
output_list = os.path.join(output_dir,f'{enhanced_dir}.list')

from pathlib import Path

# 船只类别映射
category_to_label = {
    "Passengers": 0,
    "Tugboat": 1,
    "Oceanliner": 2,
    "RORO": 3,
    "Motorboat": 4,
    "Trawler": 5,
    "Pilotship": 6,
    "Sailboat": 7,
    "Musselboat": 8,
    "Fishboat": 9,
    "Dredger": 10,
    "Naturalambientnoise": 11,
}

# 生成 list
with open(output_list, "w", encoding="utf-8") as f:
    for wave_path in sorted(wave_dir.rglob("*.wav")):

        # 按 '_' 分割文件名，查找匹配的类别
        category = next(
            (s for s in wave_path.stem.split("_") if s in category_to_label),
            None
        )

        if category is None:
            print(f"[WARNING] 未知类别: {wave_path.name}")
            continue

        label = category_to_label[category]
        f.write(f"{wave_path.resolve()}\t{label}\n")

print(f'{output_list} write over.')