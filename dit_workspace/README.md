# Complex-STFT Conditional DiT for underwater denoising

本工作区建立一个统一生成式 backbone，首先比较 **NCSN++ + 本地 OUVE**
与 **DiT + 同一 OUVE**，随后固定 DiT 比较 Score、DDPM、Rectified Flow。
应用代码不依赖相邻 SGMSE/CDiffuSE 或根目录 DCCRN。审计见 [AUDIT.md](AUDIT.md)。
这里不包含训练好的模型，也不以 smoke 的输出证明降噪效果。

## Architecture

```
waveform -> STFT -> representation -> process training state
                                     |             |
                                     state       noisy condition
                                     +---- channel concat ----+
                                      Conv2d PatchEmbed [pf,pt]
                                      dynamic 2D sin/cos positions
time -> sinusoidal -> MLP ---------> DiT blocks + adaLN-Zero
                                      FinalLayer -> unpatchify/crop
                                      score / epsilon / velocity
                                      process sampler -> representation
                                      inverse compression -> iSTFT -> waveform
```

`ConditionalDiT.forward(state, condition, time)` 始终接收实数 `[B,C,F,T]`、
`[B,C,F,T]`、`[B]`，返回 `[B,C,F,T]`。没有复数 Transformer：只有 STFT 和
谱边界使用 complex tensor；过程在其等价 Re/Im 坐标上计算。
空间条件保留逐时频对应关系，只有时间通过 adaLN 注入。

DiT 设计遵循 [Peebles & Xie](https://arxiv.org/abs/2212.09748) 及
[官方参考](https://github.com/facebookresearch/DiT/blob/main/models.py)：
attention/MLP 前的无仿射 LayerNorm、六组 shift/scale/gate、adaLN-Zero，
以及零初始化 FinalLayer。使用 PyTorch SDPA，不依赖 timm。
没有类别/文本条件、VAE、Mel 或 vocoder。

- small: depth=12, hidden=384, heads=6, mlp_ratio=4, patch=[4,8]。
- tiny: depth=2, hidden=64, heads=4；缩小 STFT 与音频，只用于测试。
- 奇数 F=257 和任意时间长度：右侧/高频端补零到 patch 整数网格，输出裁回。
  不丢弃任何频点。位置编码分别编码频率、时间网格，不假定正方形。
- 默认 small 训练 F=257,T=128：grid=[65,16]，1040 tokens。
  启动时打印参数量、实际网格和 token 数，并写入 runtime_config.yaml。
- `max_tokens` 在创建注意力矩阵前拒绝超预算输入。长音频默认 1 秒分块、
  0.25 秒交叠；这项推理设置要与比较对象显式对齐。

## Spectral representations

默认 16 kHz，Hann，n_fft=512、win_length=400、hop_length=100，center=true，
onesided=true，normalized=false，与本地 SGMSE small 一致。

压缩：`Xc = beta * |X|^alpha * exp(j*angle(X))`，alpha=.5、beta=.15。

| representation.type | state/condition | concat | output | reconstruction |
|---|---|---|---|---|
| complex_ri | 每个 `[B,2,F,T]` | `[B,4,F,T]` | `[B,2,F,T]` | 预测 Re/Im，逆压缩 |
| magnitude | 每个 `[B,1,F,T]` | `[B,2,F,T]` | `[B,1,F,T]` | 最终幅度 clamp≥0 + noisy phase |

Magnitude 的中间状态不 clamp；只有物理谱重建时才限制非负。必须设置
`phase_source: noisy`，不允许 clean phase。测试覆盖两种模式的训练、采样与解码。

## Unified mathematical processes

`factory.py` 装配 representation、model、process、objective、sampler。
`Process.sample_training_pair` 返回 state、time、target、diagnostics。
`RegressionObjective` 共用 STFT/crop/forward，按 process 选择误差语义。
`WaveformSampler` 共用解码、计时和长度处理。

### Score (default)

`mu(t)=exp(-gamma*t)*x0+(1-exp(-gamma*t))*y`，`xt=mu(t)+sigma(t)*z`，
target=`-z/sigma(t)`。complex loss=`mean(dr²+di²)`，magnitude loss 为实数 MSE。
不加入 sigma loss weighting。

`ouve_variant: water_denoise` 显式保留本地公式：

```
r = sigma_max / sigma_min
sigma(t)^2 = sigma_min^2 * (r^(2t) - exp(-2*gamma*t)) / (gamma + log(r))
g(t) = sigma_min * r^t * sqrt(2*log(r))
```

这对 marginal/diffusion 沿用了旧实现已知的 log(r) 不一致；当前只支持此
variant，不静默替换成官方形式。PC 默认 30 步、每步 1 次 Langevin corrector，
corrector 步长=.5*sigma²，逆向 Euler-Maruyama，最后一步不加 predictor 噪声。
本地数值和完整 PC 链与旧实现的 seeded parity 测试均保留。

### DDPM

`xt=sqrt(alpha_bar[t])*x0+sqrt(1-alpha_bar[t])*epsilon`；y 只进入网络。
训练 epsilon MSE，反向均值采用标准 epsilon 参数化，噪声方差为 posterior
`beta[t]*(1-alpha_bar[t-1])/(1-alpha_bar[t])`，t=0 不加噪声。
支持 linear/cosine beta schedule；cosine 使用 s=.008、beta≤.999，beta_start/end
只用于 linear。离散 t=0,...,N-1 原样送入统一时间 embedding。
必须完整执行 N 步；设置较少 sampler steps 会明确报错，不能伪装 DDIM。

### Rectified Flow

source `z0~Gaussian`，`xt=(1-t)*z0+t*x1`，t~Uniform(0,1)，target=`x1-z0`。
y 只作为条件；从 t=0 到 1 积分。提供 Euler 与 Heun；不使用 noisy-start。

### Noise conventions

生成噪声集中在 `dit/utils/noise.py`。
默认 complex 三种过程都使用 `circular_complex_unit_energy`：Re/Im 方差各 .5，
复数能量期望=1。DDPM 因而是具有同一圆对称噪声尺度的 channel formulation，
其前向和反向噪声始终一致。若研究标准每实通道方差=1 的 DDPM，可显式设
`real_unit_variance`，并把这一尺度变动作为实验变量记录。Score complex baseline
只允许圆对称单位能量。magnitude 统一使用 `real_unit_variance`。

## Data and configuration

统一 YAML：experiment/data/stft/compression/representation/model/process/
sampler/training/validation/metrics/logging/distributed/inference。
五个完整 preset 置于 configs/。修改 manifest 后才可启动正式实验。

Paired: `noisy.wav<TAB>clean.wav`。相对路径相对于执行目录，不相对于 YAML。
train、valid、test 分离。online 时 train_manifest 是 clean 单列路径，
noise_manifest 是 noise 单列路径，valid/test 仍为 paired。
保留 real/white/mixed 环境噪声与 snr_min/max、有效长度功率及 shared_peak_limit。
不分别归一化、不重复短样本。noise_mode 与生成过程 Gaussian 是不同概念。

默认 clean/noisy 采样率和长度须一致；数据加载可一起重采样到配置频率。
不同环境可能使用 torchaudio 或线性插值后备，公平比较推荐预先统一采样率。
单文件 infer 对采样率不符直接报错，以保证输出点数与输入严格一致。

## Run

Python≥3.10、PyTorch≥2.2。先在所选环境安装 requirements.txt。

```bash
cd dit_workspace
python -m pip install -r requirements.txt
python train.py --config configs/dit_score_complex_small.yaml --device cpu
python train.py --config configs/dit_score_complex_small.yaml --num-gpus 1
torchrun --standalone --nproc_per_node=2 train.py --config configs/dit_score_complex_small.yaml
python train.py --config runs/dit_score_complex_small/runtime_config.yaml --resume runs/dit_score_complex_small/checkpoints/last.pt
python evaluate.py --config runs/dit_score_complex_small/runtime_config.yaml --checkpoint runs/dit_score_complex_small/checkpoints/best.pt --test-manifest data/test_pairs.txt --output-dir runs/dit_score_complex_small/evaluation/test --seed 1234 --num-steps 30
python infer.py --config runs/dit_score_complex_small/runtime_config.yaml --checkpoint runs/dit_score_complex_small/checkpoints/best.pt --input noisy.wav --output enhanced.wav
pytest -q
python smoke.py
```

GPU 使用 CUDA AMP、梯度裁剪、EMA；CPU 自动关闭 AMP。支持 native DDP、
内部多卡 spawn、Adam、ReduceLROnPlateau 和确定性 seed。保存 last/best/周期
checkpoint，raw model、EMA、optimizer、scheduler、scaler、epoch/global_step、
best_metric、full config、Python/NumPy/Torch RNG，另存各 rank RNG 与 loader RNG。
恢复在 epoch 边界；要求同样 world size。best 只按 validation 的 EMA SI-SNRi。
checkpoint 不兼容 NCSN++ 参数，但容器字段兼容旧工作流。完整 checkpoint
包含 Python RNG，按可信本地文件加载。

train 与 eval/infer 使用同一 factory，checkpoint 校验 model/representation/
STFT/compression/process，避免同形状权重语义混淆。允许改变采样步数（DDPM
仍须完整 schedule）、Euler/Heun、测试 manifest 和 batch_size。
shell 脚本有参数配置区，测试优先读取实验 runtime_config.yaml。

## Logs, validation and evaluation contract

`metrics.jsonl` / TensorBoard 保留 train/loss、grad_norm、lr、t_mean、
epoch_seconds，score 额外 sigma_mean；统一 val/loss，并分别记录
val/score_loss、val/noise_loss 或 val/flow_loss，及 val/SI-SNRi、val/SDRi。
TensorBoard 若启用但缺失会报错，不静默丢日志。

评估导出：per_file_metrics.csv、summary_metrics.json、enhanced_wavs/、evaluation.log。
CSV：file_id,input_sdr,output_sdr,sdri,input_si_snr,output_si_snr,si_snri,
duration,inference_time,rtf,nfe,valid,error。重复 file_id 明确报错，避免覆盖 WAV。
静音参考标为 invalid，仍输出音频；损坏文件/采样失败会明确报错，不伪造指标。

SDR 为 scale-dependent signal-to-error；SI-SNR 独立去均值后投影；improvement
为增强减原始。按真实长度逐文件计算，避免 batch padding 影响，指标计算前
不 clip；soundfile FLOAT WAV 保留越界幅度。没有 DNSMOS/WER/BSS-Eval 替换。

JSON 包括 count/valid_count/invalid_count，各指标 mean/std/median/p25/p75，
全无有效值时为 null；NFE、total_nfe、mean_inference_time、mean_rtf、seed、
checkpoint、git_commit、config、process/sampler/num_steps。
此本地快照没有有效 Git 元数据，git_commit 明确为 unknown。

NFE = 成功生成路径上的网络调用次数：PC=N*(1+corrector_steps)，DDPM=N，
Euler=N，Heun=2N；长文件按实际 chunk 累加，JSON 的 NFE 为逐文件均值，
total_nfe 为总和。OOM 重试失败路径不计入 NFE，但耗时计入 inference_time。
CUDA 计时前后同步，RTF=文件推理总耗时/文件时长。
测试 batch_size 控制读取/补齐；模型逐文件分块采样以保证准确长度与逐文件计时。
长音频 SampleResult.representation 为最后一块表示，waveform 是完整拼接结果。

## Validation scope and remaining extensions

已实现：两种表示、统一 DiT、三种过程、PC/ancestral DDPM/Euler/Heun、
waveform 闭环、日志和恢复训练、CPU/GPU/DDP 代码路径。
实际测试结果及硬件限制见 IMPLEMENTATION_REPORT.md 和 SMOKE_RESULTS.json。
smoke 只运行有限合成 batch，输出质量不作为科研结论。

尚未实现：official OUVE variant、DDIM、x0/v prediction、noisy-start/source
扩展、cross-attention、probability-flow Score ODE、辅助波形损失、mid-epoch resume。
没有优化 global attention 的长序列复杂度，因此分块边界会影响生成结果。

首次实验应固定 SGMSE/DiT 的 manifest、STFT、compression、OUVE、30+30 NFE、
数据 seed、验证子集、best criterion 和分块设置，只更换 backbone。旧 SGMSE
默认整段采样，必须显式与这里对齐；旧 batch padding 指标问题建议评估 batch=1。
随后固定 DiT 规模/表征/数据，再比较 Score/DDPM/Flow；报告不同采样预算与 RTF，
不能仅凭相同步数声称相同计算量。
