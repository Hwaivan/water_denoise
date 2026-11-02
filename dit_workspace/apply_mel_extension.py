#!/usr/bin/env python3
"""Install the Mel extension into an existing dit_workspace (safe backups, dry run).

Designed for the verified 2026-10-30 public main source layout. If local files
have diverged, it stops instead of silently overwriting incompatible code.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent


def exactly_once(text, before, after, label):
    count = text.count(before)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one source anchor, found {count}; merge local changes manually")
    return text.replace(before, after, 1)


EXPECTED_BLOBS = {
    "dit/factory.py": "d8eb125fcc1a05a2845829e088275bf227b4c241",
    "dit/models/frame_dit.py": "88c13632cf84edae6f711ead2fd13df30bde10e4",
    "dit/objectives/regression.py": "42cb5d40266d65dd83e2882c77cf4cf302591793",
    "dit/samplers/score_pc.py": "49f32aab14359ccfa58ab8fc1fa3122a99450436",
    "dit/samplers/flow_ode.py": "ae924d733992528aed66499bd3231b130521940c",
}

def transformed(root, allow_source_drift=False):
    out = {}
    def get(path):
        p = root / path
        if not p.is_file():
            raise RuntimeError(f"Missing repository source: {p}")
        return p.read_text(encoding="utf-8")

    path = "dit/representations/spectral.py"
    x = get(path)
    x = exactly_once(x, "def decode(self, representation, length, noisy_phase=None):",
                       "def decode(self, representation, length, noisy_phase=None, noisy_waveform=None):", path)
    out[path] = x

    path = "dit/representations/__init__.py"
    x = get(path)
    if "LogMelRepresentation" not in x:
        x += "\nfrom .mel import LogMelRepresentation\n"
    out[path] = x

    path = "dit/samplers/base.py"
    x = get(path)
    x = exactly_once(x,
        "    def __init__(self, representation, solver):\n        self.representation, self.solver = representation, solver\n\n    def sample_spectrum(self, model, condition, generator=None):\n        return self.solver.sample(model, condition, generator)",
        "    def __init__(self, representation, solver, auxiliary_encoder=None):\n"
        "        self.representation, self.solver = representation, solver\n"
        "        self.auxiliary_encoder = auxiliary_encoder\n\n"
        "    def sample_spectrum(self, model, condition, generator=None, aux_condition=None):\n"
        "        if aux_condition is None:\n"
        "            return self.solver.sample(model, condition, generator)\n"
        "        return self.solver.sample(model, condition, generator, aux_condition=aux_condition)", path)
    x = exactly_once(x,
        "        self.representation.to(noisy.device)\n        self.solver.process.to(noisy.device)",
        "        self.representation.to(noisy.device)\n"
        "        self.solver.process.to(noisy.device)\n"
        "        if self.auxiliary_encoder is not None:\n"
        "            self.auxiliary_encoder.to(noisy.device)", path)
    x = exactly_once(x,
        "            phase = self.representation.stft(noisy.float()).angle() if self.representation.channels == 1 else None\n"
        "            estimate, nfe = self.sample_spectrum(model, condition, generator)\n"
        "            waveform = self.representation.decode(estimate, noisy.shape[-1], phase)[..., :original_length]",
        "            phase = (self.representation.stft(noisy.float()).angle()\n"
        "                     if self.representation.type == 'magnitude' else None)\n"
        "            aux = (self.auxiliary_encoder.encode(noisy.float())\n"
        "                   if self.auxiliary_encoder is not None else None)\n"
        "            if aux is not None and aux.shape[-1] != condition.shape[-1]:\n"
        "                raise ValueError('Mel and state frame count mismatch during inference')\n"
        "            estimate, nfe = self.sample_spectrum(model, condition, generator, aux)\n"
        "            waveform = self.representation.decode(\n"
        "                estimate, noisy.shape[-1], phase, noisy_waveform=noisy)[..., :original_length]", path)
    out[path] = x

    path = "train.py"
    x = get(path)
    x = exactly_once(x,
        '        bins = config["stft"]["n_fft"] // 2 + 1 if config["stft"]["onesided"] else config["stft"]["n_fft"]',
        '        bins = getattr(representation, "bins", None)\n'
        '        if bins is None:\n'
        '            bins = (config["stft"]["n_fft"] // 2 + 1 if config["stft"]["onesided"]\n'
        '                    else config["stft"]["n_fft"])', path)
    out[path] = x

    path = "dit/utils/checkpoint.py"
    x = get(path)
    x = exactly_once(x,
        '    for key in ("model", "representation", "stft", "compression", "process"):',
        '    for key in ("model", "representation", "stft", "compression", "process",\n'
        '                "conditioning", "reconstruction"):', path)
    out[path] = x

    path = "dit/utils/config.py"
    x = get(path)
    anchor = "\n\ndef load_config(path):"
    checker = '''\n\ndef _validate_mel(config):
    rep = config["representation"]
    kind = rep.get("type")
    conditioning = config.get("conditioning", {"type": "none"})
    cond_type = conditioning.get("type", "none")
    if kind not in ("complex_ri", "magnitude", "log_mel"):
        raise ValueError("Unknown representation type: " + str(kind))
    if cond_type not in ("none", "complex_plus_log_mel"):
        raise ValueError("Unknown conditioning type: " + str(cond_type))

    def check_mel(mel):
        if int(mel.get("n_mels", 0)) < 1:
            raise ValueError("mel.n_mels must be positive")
        if mel.get("normalization", "dataset") == "dataset" and not mel.get("stats_path"):
            raise ValueError("Dataset-normalized Mel requires stats_path")
        if mel.get("normalization", "dataset") not in ("none", "dataset"):
            raise ValueError("Mel normalization must be none or dataset")
        f_max = mel.get("f_max")
        if f_max is not None and float(f_max) > config["data"]["sample_rate"] / 2:
            raise ValueError("Mel f_max above Nyquist")
        if mel.get("norm", "slaney") not in ("slaney", "none"):
            raise ValueError("Unsupported Mel filter norm")
        if float(mel.get("power", 2.0)) <= 0:
            raise ValueError("Mel power must be positive")

    if kind == "log_mel":
        if cond_type != "none":
            raise ValueError("For pure Mel state use conditioning.type=none")
        if not config["stft"].get("onesided", True):
            raise ValueError("Mel requires onesided STFT")
        if config["process"].get("noise_convention") != "real_unit_variance":
            raise ValueError("log_mel requires real_unit_variance diffusion noise")
        check_mel(rep.get("mel", {}))
        if config.get("reconstruction", {}).get("type", "mel_gain_noisy_phase") != "mel_gain_noisy_phase":
            raise ValueError("Only mel_gain_noisy_phase is implemented")
    if cond_type == "complex_plus_log_mel":
        if kind != "complex_ri" or config["model"].get("tokenization") != "frame1d":
            raise ValueError("Complex + Mel condition requires complex_ri and frame1d")
        if config["process"]["type"] not in ("score", "flow"):
            raise ValueError("Complex + Mel conditioning supports Score and Flow")
        if not config["stft"].get("onesided", True):
            raise ValueError("Mel requires onesided STFT")
        check_mel(conditioning.get("mel", {}))
    if config["training"].get("score_loss_weighting", "none") not in ("none", "sigma2"):
        raise ValueError("score_loss_weighting must be none or sigma2")
'''
    x = exactly_once(x, anchor, checker + anchor, path)
    x = exactly_once(x,
        '    if not inf.get(\'chunk_seconds\') or not 0 <= inf[\'overlap_seconds\'] < inf[\'chunk_seconds\']:\n'
        '        raise ValueError(\'Require finite positive chunk_seconds and smaller nonnegative overlap\')',
        '    if not inf.get(\'chunk_seconds\') or not 0 <= inf[\'overlap_seconds\'] < inf[\'chunk_seconds\']:\n'
        '        raise ValueError(\'Require finite positive chunk_seconds and smaller nonnegative overlap\')\n'
        '    _validate_mel(config)', path)
    out[path] = x

    # Drop-in replacements: copies are retained and never overwritten without backup.
    for rel in ("dit/factory.py", "dit/models/frame_dit.py", "dit/objectives/regression.py",
                "dit/samplers/score_pc.py", "dit/samplers/flow_ode.py"):
        old = get(rel)
        if not old.strip():
            raise RuntimeError("Empty source file: " + rel)
        data = (root / rel).read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\x00" + data).hexdigest()
        if not allow_source_drift and blob != EXPECTED_BLOBS[rel]:
            raise RuntimeError(f"{rel} differs from verified GitHub main (blob {blob}). "
                               "Do not overwrite local fixes; manually merge or pass --allow-source-drift")
        out[rel] = (HERE / "modified_files" / rel).read_text(encoding="utf-8")
    for rel in ("dit/representations/mel.py",):
        if (root / rel).exists():
            raise RuntimeError(f"{rel} already exists. Inspect before installing")
        out[rel] = (HERE / rel).read_text(encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="path of dit_workspace containing train.py")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-source-drift", action="store_true", help="explicitly allow replacing locally modified core sources")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    if not (root / "train.py").is_file():
        ap.error("--root must refer to dit_workspace")
    new = transformed(root, allow_source_drift=args.allow_source_drift)  # complete preflight before changes
    print(f"Validated {len(new)} source files for installation")
    for rel, content in new.items():
        print(f"  {rel}: {len(content)} chars")
    support = []
    for src_folder, dest_folder in (("scripts", "mel_scripts"), ("configs", "configs"), ("tests", "tests")):
        for src in (HERE / src_folder).rglob("*"):
            if src.is_file() and src.suffix != ".pyc" and "__pycache__" not in src.parts:
                target = root / dest_folder / src.relative_to(HERE / src_folder)
                if target.exists():
                    raise RuntimeError(f"Support target already exists: {target}")
                support.append((src, target))
    if args.dry_run:
        print("Dry run OK; nothing changed")
        return
    backup = root / (".mel_extension_backup_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    for rel in new:
        old = root / rel
        if old.exists():
            (backup / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old, backup / rel)
    for rel, content in new.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    for src, target in support:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
    print(f"Installed successfully. Source backup: {backup}")
    if not (root / "dit/data").exists():
        print("WARNING: dit/data is missing on this checkout; original training cannot run until restored")


if __name__ == "__main__":
    main()
