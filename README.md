# DCCRN 单通道语音降噪训练框架

本项目实现论文 *DCCRN: Deep Complex Convolution Recurrent Network for
Phase-Aware Speech Enhancement* 的工程化单通道监督训练流程：干净语音与噪声在线按
SNR 混合，经 STFT、复数编码器、LSTM 瓶颈和复数解码器预测掩膜，再以负 SI-SNR
训练。代码只依赖原生 PyTorch 训练循环，支持 CPU 与单 GPU。

## 环境

代码以参考环境中的 Python 3.8.5、PyTorch 1.11、torchaudio 0.11 为兼容基线，同时支持
Python 3.10+ 和较新 PyTorch。若当前环境已经包含参考 `requirements.txt` 中的包，
无需重新安装。新环境可安装本项目的最小依赖：

```bash
pip install -r requirements.txt
```

音频优先由 torchaudio 读取；torchaudio 不可用时自动回退到 soundfile，并提供纯
PyTorch 线性重采样兜底。TensorBoard 不可用时只关闭事件写入，不影响训练。

## 数据列表

训练默认使用分离列表并在线随机配对：

```text
# data/train_clean.txt
/absolute/path/clean_001.wav
/absolute/path/clean_002.wav

# data/train_noise.txt
/absolute/path/noise_001.wav
/absolute/path/noise_002.wav
```

验证、测试使用固定配对，每行是“带噪音频、制表符、干净音频”：

```text
/absolute/path/noisy_001.wav<TAB>/absolute/path/clean_001.wav
```

数据集统一返回 `mixture/target/noise: [T]`、`length`、路径和 `input_snr`；批处理函数
将变长波形补零为 `[B,T]`，损失和指标会忽略补零区域。验证/测试固定配对，训练随机性
由 `seed + epoch + index` 决定，因此可复现。

## 配置与运行

所有路径、模型、STFT 和训练参数都在 `configs/dccrn_base.yaml`。先修改其中的数据
列表路径，再执行：

```bash
python train.py --config configs/dccrn_base.yaml
```

断点恢复（恢复模型、优化器、调度器、AMP scaler、epoch 和最佳指标）：

```bash
python train.py --config configs/dccrn_base.yaml \
  --resume runs/dccrn_base/checkpoints/last.pt
```

测试集评估会生成 `per_sample.csv` 与 `summary.json`：

```bash
python evaluate.py --config configs/dccrn_base.yaml \
  --checkpoint runs/dccrn_base/checkpoints/best.pt
```

当 `evaluation.save_audio_examples: true` 时，还会保存配置数量上限内的 mixture、target
和 enhanced WAV，便于主观试听。

单文件/长音频推理：

```bash
python infer.py --config configs/dccrn_base.yaml \
  --checkpoint runs/dccrn_base/checkpoints/best.pt \
  --input input.wav --output enhanced.wav
```

长音频默认按 10 秒分块、1 秒重叠并加权 overlap-add；可用
`--chunk-seconds` 和 `--overlap-seconds` 覆盖。

运行测试：

```bash
pytest -q
```

## 模型接口与设计

`DCCRN.forward(mixture, lengths)` 接受 `[B,T]`，返回：

```text
waveform: [B,T]            增强波形
spectrum: [B,F,Frames]     原生 PyTorch complex 增强频谱
mask:     [B,F,Frames]     原生 PyTorch complex 掩膜
```

STFT 频谱在模块间统一使用原生 complex tensor；复数卷积内部临时转换为
`[B,2C,F,Frames]`，前半通道为实部、后半为虚部。U-Net skip connection 使用专门的
复数拼接函数，防止实虚部通道错位。模型支持任意 batch size 和时间长度（长度需足以
通过配置的编码器层数）。

重建策略由 `model.reconstruction_mode` 选择：

- `dccrn_e`：掩膜转换为幅度/相位，限制幅度后修正带噪相位（默认）。
- `dccrn_c`：有界复数掩膜与带噪频谱做标准复数乘法。
- `dccrn_r`：分别掩蔽带噪频谱的实部和虚部。

新增重建方式只需继承 `MaskReconstruction` 并注册到
`models/mask_reconstruction.py`；新增指标放入 `metrics/` 并由 evaluator 汇总；新增损失
放入 `losses/`，在 `LossManager` 中读取权重即可；新模型通过 `models.build_model` 工厂接入。

## 损失与指标

训练目标是负 SI-SNR。目标和估计分别按有效长度去均值，估计投影到目标方向后计算
投影能量与残差能量之比；投影系数不 detach。测试报告包含 SI-SNR、SI-SNRi、简单
signal-to-error SDR、SDRi、波形 L1 和相关系数。这里的 SDR 明确不是完整 Vincent
BSS Eval SDR。

## 与论文实现的差异和扩展点

当前版本优先保证清晰、稳定和可验证：默认使用实值 LSTM 瓶颈；复数 BatchNorm 对
实部和虚部分别归一化，而不是完整协方差白化；尚未实现 DCCRN-CL、Complex LSTM、
复数谱组合损失和多通道输入。DCCRN-C/R 重建已实现但未作为论文逐项复现实验校准。

后续增加 Complex LSTM 时替换 `models/complex_layers.py` 的 `RecurrentBottleneck`；增加
DCCRN-CL 或新掩膜公式时扩展 `models/mask_reconstruction.py`；增加复数谱损失时扩展
`losses/si_snr.py` 中的 `LossManager`；多通道输入还需同步调整 STFT 输入、模型首层复数
通道数、数据集返回格式和推理音频 I/O。

## 项目结构

```text
configs/                    实验 YAML
datasets/                   音频 I/O、在线混合、Dataset 与 DataLoader
models/                     复数层、DCCRN、掩膜重建策略
losses/                     SI-SNR 与 LossManager
metrics/                    评估指标
engine/                     Trainer 与 evaluator
utils/                      配置、日志、随机种子、checkpoint、STFT
tests/                      数值、shape、反向传播、数据与 checkpoint 测试
train.py                    训练入口
evaluate.py                 测试集评估入口
infer.py                    单文件及分块推理入口
```
