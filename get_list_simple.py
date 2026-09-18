from pathlib import Path

from pathlib import Path


# ===== 直接修改这里 =====
audio_dir = "/data/huayifan/data_open/shipsear/ShipsEar-12class_W5h1/test"
output_file = "./list/ShipsEar_test_noise.list"

keyword = "noise"

# 可选："with" 或 "without"
mode = "with"

# 是否递归扫描子文件夹
recursive = True

audio_extensions = {
    ".wav",
    ".flac",
    ".mp3",
    ".ogg",
    ".m4a",
    ".aac",
}


root = Path(audio_dir).expanduser().resolve()

if not root.is_dir():
    raise FileNotFoundError(f"文件夹不存在：{root}")

if mode not in {"with", "without"}:
    raise ValueError("mode只能设置为 'with' 或 'without'")

files = root.rglob("*") if recursive else root.glob("*")

audio_paths = sorted(
    path.resolve()
    for path in files
    if path.is_file() and path.suffix.lower() in audio_extensions
)

selected_paths = []

for path in audio_paths:
    filename = path.stem.lower()
    key = keyword.strip().lower()

    # keyword为空时，不进行筛选，全部写入
    if not key:
        selected_paths.append(path)

    elif mode == "with" and key in filename:
        selected_paths.append(path)

    elif mode == "without" and key not in filename:
        selected_paths.append(path)


output_path = Path(output_file)
output_path.parent.mkdir(parents=True, exist_ok=True)

with output_path.open("w", encoding="utf-8") as f:
    for path in selected_paths:
        f.write(f"{path}\n")


for path in selected_paths:
    print(path)

print(f"筛选模式：{mode}")
print(f"关键词：{keyword if keyword else '空，未进行筛选'}")
print(f"全部音频数量：{len(audio_paths)}")
print(f"写入音频数量：{len(selected_paths)}")
print(f"列表保存位置：{output_path.resolve()}")


# ##############原始简单版#######################
# # ===== 直接修改这里 =====
# audio_dir = "/data/huayifan/data_open/shipsear/ShipsEar-12class/train"
# output_file = "./list/ShipsEar_train.list"

# # 是否递归扫描子文件夹
# recursive = True

# # 支持的音频格式
# audio_extensions = {
#     ".wav",
#     ".flac",
#     ".mp3",
#     ".ogg",
#     ".m4a",
#     ".aac",
# }


# root = Path(audio_dir).expanduser().resolve()

# if not root.is_dir():
#     raise FileNotFoundError(f"文件夹不存在：{root}")

# files = root.rglob("*") if recursive else root.glob("*")

# audio_paths = sorted(
#     path.resolve()
#     for path in files
#     if path.is_file() and path.suffix.lower() in audio_extensions
# )

# output_path = Path(output_file)
# output_path.parent.mkdir(parents=True, exist_ok=True)

# with output_path.open("w", encoding="utf-8") as f:
#     for path in audio_paths:
#         f.write(f"{path}\n")

# # print(f"{audio_paths}")
# print("\n".join(str(path) for path in audio_paths))
# print(f"共找到 {len(audio_paths)} 个音频文件")
# print(f"列表已保存到：{output_path.resolve()}")