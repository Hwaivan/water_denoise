from pathlib import Path
import random


# ===== 直接修改这里 =====
input_list = "./ShipsEar_train_noise.list"

train_list = "./ShipsEar_train_noise_train.list"
valid_list = "./ShipsEar_train_noise_valid.list"

# 训练集比例，例如0.8表示80%训练、20%验证
train_ratio = 0.8

# 随机种子，固定后每次划分结果一致
random_seed = 42

# 是否打乱
shuffle = True


input_path = Path(input_list)

if not input_path.is_file():
    raise FileNotFoundError(f"列表文件不存在：{input_path}")

if not 0 < train_ratio < 1:
    raise ValueError("train_ratio必须位于0和1之间")

with input_path.open("r", encoding="utf-8-sig") as f:
    lines = [
        line.strip()
        for line in f
        if line.strip() and not line.lstrip().startswith("#")
    ]

if not lines:
    raise ValueError("输入列表为空")

# 可选：去重
lines = list(dict.fromkeys(lines))

if shuffle:
    random.seed(random_seed)
    random.shuffle(lines)

split_index = int(len(lines) * train_ratio)

train_lines = lines[:split_index]
valid_lines = lines[split_index:]

train_path = Path(train_list)
valid_path = Path(valid_list)

train_path.parent.mkdir(parents=True, exist_ok=True)
valid_path.parent.mkdir(parents=True, exist_ok=True)

with train_path.open("w", encoding="utf-8") as f:
    for line in train_lines:
        f.write(line + "\n")

with valid_path.open("w", encoding="utf-8") as f:
    for line in valid_lines:
        f.write(line + "\n")

print(f"总样本数：{len(lines)}")
print(f"训练集数量：{len(train_lines)}")
print(f"验证集数量：{len(valid_lines)}")
print(f"训练集保存到：{train_path.resolve()}")
print(f"验证集保存到：{valid_path.resolve()}")