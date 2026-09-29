# 实现验收报告 — 2026-09-29

## 1. Repository audit

写代码前审阅了本地 SGMSE 数据、STFT、OUVE/loss/PC、训练器、评估器、
EMA/checkpoint/DDP/logging、三个入口及配置，也检查了根目录 DCCRN 的
datasets/mixing/metrics 和入口、CDiffuSE 的数据与条件扩散实现。
详细兼容合同在 [AUDIT.md](AUDIT.md)。

- 本地存在完整 `sgmse/data`，无需修复旧工作区。
- 根目录混合会去 DC/重复短样本，SGMSE 使用有效功率/零补齐；新工作区
  选择自包含移植 SGMSE 数据语义，保留 shared gain。
- 旧 OUVE 方差与 g(t) 的已知 log(r) 不一致，显式保留 water_denoise variant。
- 旧评估的 batch padding 与整段采样默认不适合直接用于 global attention；
  新评估按有效长度和配置分块处理，比较实验须显式对齐分块。
- `.git` 为空，无法给出真实 commit；git_commit=unknown。
- 复核 legacy_hashes.json 中 **96 个既有源文件：0 个变化**。

## 2. Final architecture

waveform → STFT → representation → process state → state+noisy condition
→ rectangular PatchEmbed → 2D sin/cos + DiT/adaLN-Zero → FinalLayer
→ unpatchify/crop → process sampler → inverse compression → iSTFT。

complex_ri 每路 2 channels，concat=4，输出 2；magnitude 每路 1，concat=2，
输出 1。幅度仅最终重建 clamp，始终使用 noisy phase。全部 Transformer
层为普通实数算子，F=257 的尾频点通过 pad/unpad 保留。

## 3. Unified process design

共享 `dit/models/dit.py`、representation、RegressionObjective、factory、
trainer、WaveformSampler、waveform evaluator。模型只暴露
`forward(state, condition, time)`，不识别 score/epsilon/velocity。

| Process | Training pair/target | Sampler |
|---|---|---|
| Score | 本地 OUVE mean/std，target=-z/std，复数误差按 channel 求和 | PC，末步无 predictor 噪声 |
| DDPM | 标准 q_sample，epsilon MSE；y 仅作为条件 | 完整 ancestral posterior chain |
| Flow | (1-t)z+t*x，target=x-z，velocity MSE | Euler / Heun |

噪声集中管理，complex 默认 circular unit energy；magnitude 为实数单位方差。
DDPM 可显式改为每实通道单位方差；不得把该改动隐藏在公平比较中。

## 4. Files added/modified

全部产品代码与文档新增在 `dit_workspace/`；没有修改旧工作流源文件。

- `dit/models/{dit,blocks,embeddings}.py`：唯一 DiT backbone。
- `dit/representations/{stft,spectral}.py`：两种谱接口与可逆压缩。
- `dit/processes/{base,ouve,score_sde,ddpm,flow_matching}.py`。
- `dit/objectives/regression.py`、`dit/samplers/{base,score_pc,ddpm,flow_ode}.py`。
- `dit/data/`、`dit/engine/`、`dit/metrics/`、`dit/utils/`：隔离移植与适配。
- `train.py`、`evaluate.py`、`infer.py`、`factory.py`（位于 dit/）。
- 五套 YAML、两个 shell 脚本、README、第三方来源记录、测试与 smoke.py。
- `TEST_RESULTS.xml`、`SMOKE_RESULTS.json`：本轮实际验收证据。

## 5. Tests

执行环境：Windows，Python 3.10.19，PyTorch 2.2.2，CPU；CUDA unavailable。
现有 PyTorch 环境缺 pytest/soundfile/TensorBoard，依赖安装到工作区
`.test_dependencies`，没有改动 Anaconda 环境。由于 Windows ACL，实际
测试经授权在沙箱外运行。

实际最终命令（仓库根目录 PowerShell）：

```powershell
$env:PYTHONPATH=(Resolve-Path dit_workspace/.test_dependencies).Path
$env:PYTHONDONTWRITEBYTECODE='1'
& D:\Anaconda3\envs\Pytorch\python.exe -m pytest -q dit_workspace/tests --junitxml=dit_workspace/TEST_RESULTS.xml
```

**48 passed，0 failed，0 skipped，77.86 秒。**

验收覆盖：STFT/compression/channel round trip、noisy phase、矩形 patch、
256/257 bins、adaLN identity/zero output、连续/离散时间、Score target/OUVE
及完整 seeded PC 与旧实现一致、DDPM q/target/posterior、Flow 两端点和
速度、Euler/Heun 精确积分常速度与 NFE、同一模型三过程、一步反传、
数据混合 SNR/共享增益/epoch 变化、EMA、checkpoint、精确 epoch resume、
两个真实 CPU DDP rank、评估 schema/静音/不同长度、TensorBoard event tags、
单文件 CLI、短音频、chunk cross-fade 和模拟 OOM 回退。

Score 阶段先获得 30 passed 后才实现 DDPM/Flow。早期两次集成运行因
Windows 临时目录权限/父目录缺失失败，修复 fixture 目录后全部通过，
未删除或弱化断言。测试产物保留在 tests/_artifacts 下。

两个 shell 脚本分别通过 `bash -n`；没有执行正式训练脚本。

## 6. Smoke results

实际执行 `python smoke.py`（使用上述 Python 与 PYTHONPATH）。seed=42。
每种组合仅一个 optimizer step，然后短采样。完整数值在 SMOKE_RESULTS.json。

| Representation | Process | Parameters | One-step loss | NFE |
|---|---|---:|---:|---:|
| complex_ri | score | 190720 | 20.005764 | 4 |
| complex_ri | ddpm | 190720 | 0.485652 | 3 |
| complex_ri | flow | 190720 | 0.487034 | 3 |
| magnitude | score | 184544 | 146.293152 | 4 |
| magnitude | ddpm | 184544 | 0.978560 | 3 |
| magnitude | flow | 184544 | 0.973729 | 3 |

这些 loss 的时间/target/尺度不同，不可直接当作效果排名。

- Tiny complex state/output `[1,2,33,17]`，内部 concat `[1,4,33,17]`；
  magnitude state/output `[1,1,33,17]`，concat `[1,2,33,17]`。
- 原始推理 grid `[9,3]`，27 tokens；训练 crop=16 后 grid `[9,2]`，18 tokens。
- 六组输出波形都是 `[1,256]` 且有限；CLI 另验证 320 点文件长度严格保持。
- 正式 small complex 配置已装配、统计：**32,536,000 参数**，F=257/T=128
  对应 grid `[65,16]`，**1040 tokens**；未启动正式训练。
- GPU peak memory：不可测（无可用 CUDA）；未声称已验证 GPU 性能。

## 7. Compatibility

| Contract | Result |
|---|---|
| CSV / JSON / WAV / evaluation.log | 实际导出并验证字段、统计、长度 |
| checkpoint | 保存/恢复全部公共字段，另存 per-rank 和 loader RNG；epoch resume 参数逐项等于不中断训练 |
| TensorBoard / JSONL | 实际解析 event tags；统一 val/loss 和过程专用 loss |
| SI-SNR / SDR / improvements | 保留旧公式；按有效长度计算，无预先 clip/独立 normalize |
| RTF | 逐文件计时除时长，分块/拼接计入；CUDA 同步代码已实现但未实测 |
| NFE | PC=N*(1+c)，DDPM/Euler=N，Heun=2N；分块累加，summary 报均值和总量 |
| DDP / EMA / AMP | CPU 两 rank 训练通过；EMA round trip 通过；AMP/NCCL 需有 CUDA 再验证 |

兼容的是数据语义和工件字段，**不是 NCSN++ checkpoint 权重**。
测试 batch_size 是数据读取批量；采样逐文件执行，确保准确计时和长度。

## 8. Known limitations

- 无 CUDA：GPU forward/backward、AMP、NCCL 多卡和显存尚未实测；torchrun
  环境解析与两个 CPU rank 的实际训练已测，rendezvous 使用 Windows FileStore。
- 未做真实水声数据长训，不对增强质量或三种方法优劣作结论。
- 未实现 DDIM、x0/v prediction、official OUVE variant、Score probability-flow
  ODE、noisy-start flow、cross-attention、辅助波形损失、mid-epoch resume。
- 默认 global attention 需分块，拼接边界可能影响生成；公平比较须匹配分块。
- OOM 重试的失败路径不计 NFE，重试耗时计入 RTF。
- 没有可用 Git commit。原始 DiT 参考代码许可为 CC BY-NC 4.0，SGMSE
  数学来源与本地移植清单记录在 THIRD_PARTY_NOTICES.md。

## 9. Recommended first experiment

先准备并固定与 SGMSE 同一套 train/valid/test manifests，将分块和种子设置
对齐，比较 **existing NCSN++ + local OUVE** 与 **DiT + same local OUVE**。

```bash
cd dit_workspace
python train.py --config configs/dit_score_complex_small.yaml --num-gpus 1
python evaluate.py --config runs/dit_score_complex_small/runtime_config.yaml --checkpoint runs/dit_score_complex_small/checkpoints/best.pt --test-manifest data/test_pairs.txt --output-dir runs/dit_score_complex_small/evaluation/test --seed 1234 --num-steps 30 --device cuda
```

上述为建议命令，**未执行正式长训练，未 commit，未 push**。
