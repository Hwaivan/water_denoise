import argparse
import csv
import json
import time
from pathlib import Path

import torch

from cdiffuse.utils.config import load_config
from cdiffuse.utils.checkpoint import load_checkpoint
from cdiffuse.data import CDiffuSEDataset, build_dataloader, save_audio
from cdiffuse.factory import build_components
from cdiffuse.metrics.audio_metrics import metric_row


def _within_limit(index, limit):
    return limit < 0 or index < limit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()

    c = load_config(a.config)
    device = torch.device(a.device)

    model, schedule, sampler = build_components(c)
    model.to(device)
    ck = load_checkpoint(a.checkpoint, model, device)
    model.load_state_dict(ck.get("ema_model", ck["model"]))
    model.eval()

    out = Path(a.output_dir)
    raw_dir = out / "raw_enhanced_wavs"
    final_dir = out / "enhanced_wavs"
    step_root = out / "diffusion_steps"
    raw_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    ev = c.get("evaluation", {})
    audio_limit = int(ev.get("save_audio_limit", -1))
    ratio = float(ev.get("noisy_mix_ratio", 0.2))
    save_steps = bool(ev.get("save_diffusion_steps", False))
    step_example_limit = int(ev.get("diffusion_example_limit", 2))
    step_interval = int(ev.get("diffusion_step_interval", 5))

    if step_interval < 1:
        raise ValueError("evaluation.diffusion_step_interval must be >= 1")

    if save_steps and step_example_limit != 0:
        step_root.mkdir(parents=True, exist_ok=True)

    rows = []
    global_index = 0

    dataset = CDiffuSEDataset(c["data"], a.split, c["sampler"]["seed"])
    loader = build_dataloader(dataset, c, False)

    for batch in loader:
        noisy = batch["noisy"].to(device)
        batch_size = noisy.shape[0]

        # 若本 batch 中仍有需要保存扩散过程的样本，则让 sampler 返回 states
        need_steps = (
            save_steps
            and step_example_limit != 0
            and (step_example_limit < 0 or global_index < step_example_limit)
        )
        save_interval = step_interval if need_steps else 0

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        result = sampler.sample(model, noisy, save_interval=save_interval)

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        # sampler.states 没有显式记录 step 编号，这里按 sampler 的保存条件恢复
        saved_step_numbers = []
        if save_interval:
            saved_step_numbers = [
                step
                for step in range(schedule.num_steps, 0, -1)
                if step % save_interval == 0 or step == 1
            ]
            if len(saved_step_numbers) != len(result.states):
                raise RuntimeError(
                    "Saved sampler states do not match inferred step numbers: "
                    f"{len(result.states)} vs {len(saved_step_numbers)}"
                )

        for i, L in enumerate(batch["lengths"]):
            sample_index = global_index
            global_index += 1

            n = int(L)
            raw = result.waveform[i, :n]
            clean = batch["clean"][i, :n].to(device)
            inp = noisy[i, :n]
            final = (1.0 - ratio) * raw + ratio * inp

            row = metric_row(raw, final, clean, inp)
            duration = n / c["data"]["sample_rate"]
            row.update(
                file_id=batch["file_id"][i],
                duration_seconds=duration,
                inference_seconds=elapsed / batch_size,
                rtf=(elapsed / batch_size) / duration,
                nfe=result.nfe,
                sampling_steps=result.nfe,
            )
            rows.append(row)

            # 保存最终结果
            if _within_limit(sample_index, audio_limit):
                prefix = f"{sample_index + 1:06d}_{batch['file_id'][i]}"
                save_audio(
                    raw_dir / f"{prefix}.wav",
                    raw,
                    c["data"]["sample_rate"],
                )
                save_audio(
                    final_dir / f"{prefix}.wav",
                    final,
                    c["data"]["sample_rate"],
                )

            # 保存反向扩散中间时间步
            if (
                save_steps
                and save_interval
                and _within_limit(sample_index, step_example_limit)
            ):
                prefix = f"{sample_index + 1:06d}_{batch['file_id'][i]}"
                sample_dir = step_root / prefix
                sample_dir.mkdir(parents=True, exist_ok=True)

                # 输入 noisy 作为参考
                save_audio(
                    sample_dir / "noisy_input.wav",
                    inp,
                    c["data"]["sample_rate"],
                )

                for step, state in zip(saved_step_numbers, result.states):
                    # state shape: [B, 1, L]
                    wav = state[i, 0, :n]
                    save_audio(
                        sample_dir / f"step_{step:03d}.wav",
                        wav,
                        c["data"]["sample_rate"],
                    )

    fields = [
        "file_id",
        "input_sdr",
        "raw_output_sdr",
        "final_output_sdr",
        "raw_sdri",
        "final_sdri",
        "input_si_snr",
        "raw_output_si_snr",
        "final_output_si_snr",
        "raw_si_snri",
        "final_si_snri",
        "duration_seconds",
        "inference_seconds",
        "rtf",
        "nfe",
        "sampling_steps",
        "valid",
        "error",
    ]

    with open(out / "per_file_metrics.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)

    valid = [r for r in rows if r["valid"]]
    summary = {
        "count": len(rows),
        "valid_count": len(valid),
        "nfe": rows[0]["nfe"] if rows else 0,
        "diffusion_num_steps": schedule.num_steps,
        "save_diffusion_steps": save_steps,
        "diffusion_step_interval": step_interval,
    }
    summary.update(
        {
            k: sum(r[k] for r in valid) / len(valid)
            for k in fields
            if valid and isinstance(valid[0].get(k), float)
        }
    )

    with open(out / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()