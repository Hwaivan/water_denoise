# Phase A: local compatibility contract (2026-09-29)

Audited actual local sources, before implementation:

- `sgmse_workspace/{train,evaluate,infer}.py`, configs/sgmse_water_small.yaml,
  sgmse package data/{dataset,audio_io}, utils/{stft,checkpoint,ema,seed,
  distributed,logging,config}, diffusion/{ouve,losses,sampler},
  engine/{trainer,evaluator}, metrics/audio_metrics and THIRD_PARTY_NOTICES.
- Root train/evaluate/infer, datasets/{audio_dataset,mixing}, metrics/enhancement_metrics.
- cdiffuse_workspace/train.py and cdiffuse/{data/dataset,diffusion/process}.

Findings and frozen choices:

1. Local `sgmse/data` **exists**. Vendor its waveform implementation into this
   workspace, with provenance, rather than importing a sibling workspace.
   Paired rows are noisy<TAB>clean, paths relative to execution directory.
   Sample-rate/length disagreement fails fast. train/valid/test are separate.
2. Online clean/noise mixing uses valid-region power and random SNR, no DC
   removal, zero padding, deterministic seed + epoch + index; real/white/mixed
   environmental noise supported. Shared peak gain only, no independent
   normalization. Root DCCRN instead removes DC/repeats short inputs; do not use
   that different recipe for the NCSN++ comparison. CDiffuSE also differs.
3. Freeze 16 kHz, n_fft=512, win=400, hop=100, Hann, centered, unnormalized,
   one-sided, alpha=.5, beta=.15, training frame crop=128.
   Complex spectra use real/imag channels. Magnitude uses noisy phase only.
4. OUVE mean = exp(-gamma*t)*x0 + (1-exp(-gamma*t))*y.
   Local variance = sigma_min^2*(r^(2t)-exp(-2*gamma*t))/(gamma+log(r)).
   Local diffusion g = sigma_min*r^t*sqrt(2*log(r)). This pair has the known
   log(r) inconsistency; deliberately preserve both for the default baseline.
   Circular Gaussian real/imag variances=.5; target=-z/std; complex error is
   sum of Re/Im squared errors, then mean. No sigma weighting.
5. PC: prior y+std(T)*z; 30 descending Euler-Maruyama steps, one Langevin
   correction/step, correction h=.5*std^2; last predictor omits noise. NFE=60.
6. Existing trainer has AMP, EMA, gradient clipping, ReduceLROnPlateau on loss,
   torchrun DDP and optional spawned GPUs. EMA generative validation selects
   best by SI-SNRi. JSONL/TensorBoard tags retain train/loss, grad_norm, lr,
   sigma_mean, t_mean, val/score_loss, SI-SNRi, SDRi, epoch_seconds.
7. Checkpoint contract: model, ema_model (decay/shadow), optimizer, scheduler,
   scaler, epoch, global_step, best_metric, config, rng_state. DiT weights are
   architecturally different and cannot load an NCSN++ checkpoint.
8. Metrics: independently mean-centered projection SI-SNR; scale-dependent
   signal-to-error SDR, improvements = output-input. CSV retains file_id,
   input_sdr, output_sdr, sdri, input_si_snr, output_si_snr, si_snri, duration,
   inference_time, rtf, nfe, valid, error. JSON retains counts, population std,
   median/quartiles, NFE, timing, seed, checkpoint, git_commit, config.
   Export enhanced_wavs and evaluation.log; use float WAV without normalization.
9. Local evaluator includes padded tails in metrics; new evaluation slices each
   file to its true length. Explicitly match segmentation when comparing runs.
   Bound long-audio global attention with configured chunks in validation too.
10. Local checkpoint loader needs explicit trusted weights_only=False on modern
    PyTorch. Epoch resume also needs loader generator/per-rank RNG restoration.
    Persistent dataset workers would retain stale epoch; disable persistence.
11. No run_train*.sh/run_test*.sh or AGENTS.md found in this checkout. `.git`
    is empty and git status fails, so use source hashes to verify preservation;
    report git_commit=unknown rather than invent a commit.

Only dit_workspace is written. No formal training or push is authorized.
