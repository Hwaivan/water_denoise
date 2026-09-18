from pathlib import Path

import numpy as np
import soundfile as sf


# ==================== 参数配置 ====================
INPUT_DIR = Path(r"/data/huayifan/data_open/shipsear/ShipsEar-12class/train")       # 原始音频文件夹
OUTPUT_DIR = Path(r"/data/huayifan/data_open/shipsear/ShipsEar-12class_W5h1/train")      # 切割结果文件夹

WINDOW_SECONDS = 5.0                         # 窗长：5秒
HOP_SECONDS = 1.0                            # 窗移：例如1秒

# 是否保留末尾不足5秒的片段
KEEP_LAST_INCOMPLETE = False

# 支持的音频格式
AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg"}
# ================================================


def split_audio(
    audio_path: Path,
    output_dir: Path,
    window_seconds: float,
    hop_seconds: float,
    keep_last_incomplete: bool = False,
) -> int:
    """
    对单个音频进行滑窗切割。

    返回：
        保存的音频片段数量。
    """
    audio, sample_rate = sf.read(audio_path, always_2d=False)

    window_samples = int(round(window_seconds * sample_rate))
    hop_samples = int(round(hop_seconds * sample_rate))

    if window_samples <= 0:
        raise ValueError("窗长必须大于0。")

    if hop_samples <= 0:
        raise ValueError("窗移必须大于0。")

    total_samples = len(audio)
    segment_index = 0
    start_sample = 0

    while start_sample < total_samples:
        end_sample = start_sample + window_samples

        # 末尾不足一个完整窗
        if end_sample > total_samples:
            if not keep_last_incomplete:
                break

            segment = audio[start_sample:total_samples]

            # 补零到5秒
            pad_length = window_samples - len(segment)

            if audio.ndim == 1:
                segment = np.pad(segment, (0, pad_length))
            else:
                segment = np.pad(
                    segment,
                    ((0, pad_length), (0, 0)),
                )
        else:
            segment = audio[start_sample:end_sample]

        start_time = start_sample / sample_rate
        end_time = min(end_sample, total_samples) / sample_rate

        output_name = (
            f"{audio_path.stem}"
            f"_{segment_index:04d}"
            f"_{start_time:.2f}s-{end_time:.2f}s.wav"
        )
        output_path = output_dir / output_name

        sf.write(output_path, segment, sample_rate)

        segment_index += 1
        start_sample += hop_samples

    return segment_index


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(
        path
        for path in INPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )

    if not audio_files:
        print(f"未在以下文件夹中找到音频：{INPUT_DIR}")
        return

    total_segments = 0

    for audio_path in audio_files:
        try:
            segment_count = split_audio(
                audio_path=audio_path,
                output_dir=OUTPUT_DIR,
                window_seconds=WINDOW_SECONDS,
                hop_seconds=HOP_SECONDS,
                keep_last_incomplete=KEEP_LAST_INCOMPLETE,
            )

            total_segments += segment_count
            print(f"{audio_path.name}：生成{segment_count}个片段")

        except Exception as error:
            print(f"{audio_path.name}处理失败：{error}")

    print("-" * 50)
    print(f"处理完成，共生成{total_segments}个音频片段。")
    print(f"保存位置：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()