# CDiffuSE for underwater acoustic denoising

Independent discrete, time-domain CDiffuSE implementation. It does not import
SGMSE and cannot load an SGMSE checkpoint. STFT magnitude is used only as the
noisy conditioner; the diffusion state and network output remain real waveform
tensors `[B,1,L]`.

## What was reused and replaced

The project follows the existing SGMSE manifest (`noisy<TAB>clean`), audio I/O,
online mixing, CLI, EMA/checkpoint, metric, and evaluation conventions. OUVE,
continuous-time score matching, NCSN++, complex-STFT state, and predictor-
corrector sampling were replaced by discrete CDiffuSE formulas, DiffWave, and a
conditional DDPM sampler.

The paper equations are implemented in `cdiffuse/diffusion/schedule.py`
(`beta`, `alpha`, `alpha_bar`, `m`, `delta`, reverse coefficients),
`process.py` (conditional forward distribution and combined-noise target), and
`sampler.py` (initial distribution and full reverse chain). Arrays explicitly
contain state zero.

## Commands

```sh
python train.py --config configs/cdiffuse_water_base.yaml --device cuda
python train.py --config configs/cdiffuse_water_large.yaml --device cuda
python evaluate.py --config configs/cdiffuse_water_base.yaml --checkpoint runs/cdiffuse_water_base/checkpoints/best.pt --split test --output-dir runs/cdiffuse_water_base/evaluation --device cuda
python infer.py --config configs/cdiffuse_water_base.yaml --checkpoint runs/cdiffuse_water_base/checkpoints/best.pt --input input.wav --output enhanced.wav --device cuda
pytest -q
```

Full sampling performs exactly `T` model evaluations (`50` Base, `200` Large).
`fast_sampler.py` implements and tests DiffWave log-alpha schedule mapping for
`[0.0001, 0.001, 0.01, 0.05, 0.2, 0.35]`; skipped conditional posterior
sampling is deliberately not exposed as a replacement for the validated full
sampler until its CDiffuSE posterior is independently validated.

Clean-Mel pretraining has an explicit separate entry point and transfer contract,
but is not claimed as validated. Core CDiffuSE trains from scratch when
`training.pretrained_diffwave_checkpoint` is null.
