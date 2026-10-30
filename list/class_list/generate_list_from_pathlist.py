
from pathlib import Path

# =========================
# 配置
# =========================

# 输入：每行一个音频路径
input_list = Path("shipsear_test_16k_only_signal_path.list")

# 输出：每行 path\tlabel
output_list = Path("shipsear_test_16k_only_signal.list")

# 船只类别映射（保持原脚本不变）
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

# =========================
# 生成新 list
# =========================

output_list.parent.mkdir(parents=True, exist_ok=True)

total = 0
written = 0
unknown = 0

with input_list.open("r", encoding="utf-8") as fin, \
     output_list.open("w", encoding="utf-8") as fout:

    for line_num, line in enumerate(fin, start=1):
        line = line.strip()

        if not line:
            continue

        total += 1

        # 输入 list 每行只包含音频路径
        wave_path = Path(line)

        # 按 '_' 分割文件名，查找类别
        category = next(
            (
                s for s in wave_path.stem.split("_")
                if s in category_to_label
            ),
            None
        )

        if category is None:
            print(
                f"[WARNING] 第 {line_num} 行未知类别: "
                f"{wave_path.name}"
            )
            unknown += 1
            continue

        label = category_to_label[category]

        # 保留原始路径，不改变输入顺序
        fout.write(f"{wave_path}\t{label}\n")
        written += 1

print(f"\n{output_list} write over.")
print(f"输入音频数: {total}")
print(f"成功写入数: {written}")
print(f"未知类别数: {unknown}")
