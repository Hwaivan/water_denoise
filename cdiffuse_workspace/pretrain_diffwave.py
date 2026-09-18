"""Clean-conditioned DDPM pretraining entry point placeholder.

The backbone transfer contract is intentionally separate from CDiffuSE. A
future clean-Mel trainer may save the shared input, embedding, residual, skip,
and output keys; CDiffuSE must reinitialize its noisy-spectrum conditioner.
"""
raise SystemExit("Clean-Mel pretraining interface is declared but not yet validated; train CDiffuSE from scratch with pretrained_diffwave_checkpoint: null")
