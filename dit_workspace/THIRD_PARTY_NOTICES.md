# Third-party and local source provenance

## DiT structural reference

- Repository: https://github.com/facebookresearch/DiT
- Source: https://github.com/facebookresearch/DiT/blob/main/models.py
- License: Creative Commons Attribution-NonCommercial 4.0 International
  (https://github.com/facebookresearch/DiT/blob/main/LICENSE.txt,
  https://creativecommons.org/licenses/by-nc/4.0/).
- Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
- Inspected main on 2026-09-29; an immutable upstream commit was not available
  in this local checkout and is not asserted here.
- `dit/models/{embeddings,blocks,dit}.py` implements the DiT design with
  rectangular spectral patches, dynamic positions, state/condition channels,
  ordinary PyTorch SDPA, no class labels, no learned variance, no timm,
  pad/unpad, and a token budget. The upstream design/reference is attributed;
  no upstream whole source file or pretrained weights were imported.
- Research reference: William Peebles and Saining Xie, *Scalable Diffusion
  Models with Transformers*, https://arxiv.org/abs/2212.09748.

## Local water_denoise compatibility sources

Repository identity supplied by the project: https://github.com/Hwaivan/water_denoise.
The local `.git` is empty, so its commit and remote provenance cannot be
verified. `legacy_hashes.json` records the actual source SHA-256 values.
No root LICENSE was present in the supplied checkout; no new license grant
for the project's existing code is assumed.

Vendored/adapted local files from `sgmse_workspace/sgmse`:

- data/dataset.py -> dit/data/dataset.py: rename/import isolation; preserve
  mixing/cropping/gain; allow silent evaluation rows; recreate workers per epoch.
- utils/stft.py -> dit/representations/stft.py: move window to input device.
- diffusion/ouve.py -> dit/processes/ouve.py: preserve exact local equations.
- utils/{checkpoint,ema,seed,distributed,logging}.py -> dit/utils/: retain
  contracts, modern trusted checkpoint loading, CPU RNG tensor restore.
- metrics/audio_metrics.py -> dit/metrics/audio_metrics.py: same metric math.
- engine/{trainer,evaluator}.py and train/evaluate/infer.py: generic process,
  true-length evaluation, bounded sampling, process config validation,
  per-rank epoch resume, full runtime configuration and additional metadata.

Data audio IO is reimplemented with mandatory float soundfile output; its
optional torchaudio/linear resampling order follows the local fallback behavior.
Configuration parsing is workspace-specific. Legacy workspaces are never
imported by application code. Tests optionally import their math for parity.

## SGMSE mathematical ancestry

Local THIRD_PARTY_NOTICES records https://github.com/sp-uhh/sgmse under MIT,
`sgmse/sdes.py` blob `14600fbc2f72a87dd30919da048481674c559ab3`, and LICENSE
blob `44f44000468b7389f9230f3f5aff564466c68b05` (local audit dated 2026-07-24).
These identifiers are inherited provenance, not independently reverified here.
Only the local OUVE math/engineering is reused, not its NCSN++ or FIR code.
The local variance intentionally omits the upstream log(r) factor.

MIT License

Copyright (c) 2022 Signal Processing (SP), Universität Hamburg

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
