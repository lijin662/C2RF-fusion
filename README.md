# C2RF Fusion：RoadScene / OSDataset

面向服务器训练的双模态图像融合实验项目，基于 [原作者 C2RF](https://github.com/QinglongYan-hub/C2RF)。主要实验为 **RoadScene 可见光–红外融合** 和 **OSDataset 光学–SAR 融合**。保留原模型作为 baseline，提供数据准备、两阶段训练、断点续训、融合推理和指标统计，方便继续修改网络与损失、做消融对比。

本仓库只上传代码、脚本和说明。数据、预训练权重、结果、本地虚拟环境均不提交。当前本地验证是小样本 CPU 流程测试，尚无完整数据训练后的效果结论。

## 1. 服务器安装

```bash
git clone https://github.com/<你的GitHub用户名>/C2RF-fusion.git
cd C2RF-fusion
# 建议用 Python 3.11/3.12；PYTHON_BIN 可指定已安装的解释器。
PYTHON_BIN=python3 bash scripts/setup_env.sh
source .venv/bin/activate
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

安装脚本默认使用 pip 的 PyTorch 包。如果服务器需要特定 CUDA wheel，先根据驱动选择 PyTorch 官方提供的索引，再设置 `TORCH_INDEX_URL` 运行安装脚本；例如 `TORCH_INDEX_URL=<对应的官方索引URL> bash scripts/setup_env.sh`。正式训练前确认上面的 CUDA 检查输出 `True`。

本机 CPU 已验证的核心版本为 torch 2.14.1、torchvision 0.29.1、kornia 0.9.0。依赖范围见 [requirements.txt](requirements.txt)，Linux CUDA 环境需在服务器实际验证。

## 2. 数据放置

脚本使用 `DATA_ROOT` 指定数据根目录，默认是项目下的 `dataset`。推荐把正式数据存到服务器独立数据盘，例如 `/data/c2rf`。

```text
/data/c2rf/
├── train/
│   ├── RoadScene/{vi,ir}/同名图片.png
│   └── OSDataset/{vi,ir}/同名图片.png
├── val/
│   └── OSDataset/{vi,ir}/同名图片.png
└── test/
    ├── RoadScene/{vi,ir}/同名图片.png
    └── OSDataset/{vi,ir}/同名图片.png
```

RoadScene 的 `vi` 是可见光、`ir` 是红外；OSDataset 的 `vi` 是光学、`ir` 是 SAR。训练/融合使用已配准图像对，纯融合不需要 `vi_warp`、flow 或配准网络权重。两个模态必须同名且尺寸相同。目录名不同时可额外传 `--vis_dir xxx --ir_dir xxx`。配对检查会提示未配对文件；正式实验应补齐数据再运行。

### OSDataset 下载与转换

来源：[作者官方仓库](https://github.com/xym2009/OSdataset)，原始 ZIP 约 1.87 GB，许可为 CC-BY-NC，使用请遵守作者要求并引用其论文。

| 尺寸 | train 对数 | val 对数 | test 对数 |
|---|---:|---:|---:|
| 256×256 | 8,044 | 952 | 1,696 |
| 512×512 | 2,011 | 238 | 424 |

原始结构是 `OSdataset/256/train/opt1.png`、`sar1.png` 等。转换脚本按数字编号配对，保留作者 train/val/test 划分，输出可追溯的 `manifest.json` 和样例 `preview.png`。

```bash
bash scripts/download_osdataset.sh
DATA_ROOT=/data/c2rf bash scripts/prepare_osdataset.sh
# 默认全量 256 版本；512 版本请使用另一个数据集名字以免混淆：
DATA_ROOT=/data/c2rf RESOLUTION=512 DATASET_NAME=OSDataset512 bash scripts/prepare_osdataset.sh
```

只验证代码时可设置 `LIMIT=4`。小样本与全量数据请用不同 `DATASET_NAME` 或 `DATA_ROOT`；转换脚本拒绝覆盖已经准备好的目录。256/512 来自相关原始影像，不能把不同版本混作独立训练/测试数据。val 划分保留用于后续调参；当前训练脚本不自动挑选验证集最优模型。

### RoadScene 数据

请把你已有的完整 RoadScene 图像对放到上述目录。原作者仓库仅附带少量示例，不能用它们代表完整训练/测试集。作者的 [RoadScene 预训练权重](https://drive.google.com/drive/folders/1wOSVg9CsqZBJkHWYMGD1kCER9tThSxYk) 可选；从头做纯融合训练无需预下载权重。

## 3. 数据检查和小样本测试

```bash
export DATA_ROOT=/data/c2rf
export DEVICE=cuda
bash scripts/run.sh check RoadScene
bash scripts/run.sh check OSDataset
bash scripts/run.sh smoke RoadScene
bash scripts/run.sh smoke OSDataset
```

CPU 环境用 `DEVICE=cpu`。`smoke` 验证真实数据加载和随机权重前向，不能当作模型效果评估。报告在 `result/<数据集>/<实验名>/smoke/report.json`。

完整的自动流程测试使用临时生成的图像，验证两个训练阶段、冻结参数、断点续训、权重加载、融合图片导出和评估：

```bash
python -m unittest discover -s tests -v
```

## 4. 分别训练两个融合模型

```bash
export DATA_ROOT=/data/c2rf
export DEVICE=cuda
export BATCH_SIZE=2
export WORKERS=4
export EXPERIMENT=baseline
bash scripts/run.sh train RoadScene
bash scripts/run.sh train OSDataset
```

上面两条命令顺序执行；单张卡建议分别训练。默认总共 150 epoch：前 50 epoch 预训练 Encoder/Decoder，后 100 epoch 冻结它们、训练 Fusion_layer。AFF 中的全局 BatchNorm 要求训练 batch size 至少为 2，训练会丢弃最后一个不足 batch 的批次。训练随机裁剪为 256；推理保留原始尺寸。

快速检查两个训练阶段，可以缩短为：

```bash
EXPERIMENT=quickcheck bash scripts/run.sh train OSDataset \
  --n_ep_Fu 2 --warmup_epochs 1 --max_steps 1 --crop_size 32
```

该命令只检验执行，不能用于正式结果。正式训练请去掉 `--max_steps` 和小裁剪设置。

每个实验单独保存：

```text
result/OSDataset/baseline/
├── config.json              # 超参数和数据路径
├── loss.csv                 # 每轮平均训练损失
├── latest.pt                # 每轮保存，可续训
├── epoch_0150.pt            # 定期/最后的完整快照
├── Encoder.pth              # 进入融合阶段后导出的独立组件
├── Decoder.pth
└── Fusion_layer.pth
```

续训必须保持原有总 epoch、warmup、batch size、crop size、学习率和种子等参数：

```bash
bash scripts/run.sh train OSDataset --resume result/OSDataset/baseline/latest.pt
```

已存在实验不会被无意覆盖。完整 checkpoint 包含模型、优化器、学习率调度器和随机状态，按 epoch 边界恢复。生产运行可使用 tmux，或：

```bash
mkdir -p logs
nohup bash scripts/run.sh train OSDataset > logs/osdataset-baseline.log 2>&1 &
tail -f logs/osdataset-baseline.log
```

## 5. 融合推理与对比

```bash
bash scripts/run.sh fuse RoadScene
bash scripts/run.sh evaluate RoadScene
bash scripts/run.sh fuse OSDataset
bash scripts/run.sh evaluate OSDataset
```

`fuse` 默认读取 `result/<数据集>/<实验名>/latest.pt`；也可以传 `--fusion_checkpoint /path/epoch_xxxx.pt`。融合图在实验目录的 `fused/`，可见光的色度会保留；OSDataset 的输入为灰度，输出也不包含额外颜色信息。

`metrics.csv` 保存逐图结果，`metrics.json` 保存均值。当前提供 EN、SD、SF、MI 作为可重复的基础诊断：统一转换为 8 位 PIL L 灰度，EN/MI 单位为 bit，SD/SF 按 0–255 计算，MI 为 `MI(F,VI)+MI(F,IR)`。这些指标没有融合真值，也可能偏好噪声或高对比度，尤其 SAR 场景不能仅凭数值升高认定改进。请结合视觉细节、伪影和下游任务判断。

## 6. 后续改进怎么做

先分别完成两个数据集的 `baseline`。网络主要在 `modules/FusionNet.py`、`modules/layers_fusion.py`，融合损失在 `modules/losses.py::Fusionloss`，两阶段训练在 `train_Fu.py`。保持测试集、预处理、种子、训练预算一致，每次改动使用新的实验名：

```bash
EXPERIMENT=loss_v1 bash scripts/run.sh train OSDataset
EXPERIMENT=loss_v1 bash scripts/run.sh fuse OSDataset
EXPERIMENT=loss_v1 bash scripts/run.sh evaluate OSDataset
```

建议先在 val 上筛选结构/损失，再固定方案评估 test；一次只改一个因素，保留 baseline 的 checkpoint、config 和指标。本仓库没有宣称某个改动已提高融合质量。

## 7. 可选：原 C2RF 配准后融合

`model.py`、`train_Reg.py` 和 `test.py` 保留为原论文配准路线。`test.py` 需要 `test/<数据集>/vi_warp` 和 `checkpoint/<数据集>/` 下的 `Encoder.pth`、`Decoder.pth`、`Fusion_layer.pth`、`RegNet.pth`：

```bash
bash scripts/run.sh reg-test RoadScene
# 缺少合成 warp 时可显式生成；这会写入数据目录：
bash scripts/run.sh reg-test RoadScene --generate_test_warp
```

RoadScene 权重不等同于训练好的光学–SAR 权重。当前主要验证和推荐运行路径是上面的纯融合流程。

## 来源与引用

本项目是原始实现的服务器实验适配版，原始网络代码归原作者所有；上游未在本仓库基线中附带独立 LICENSE 文件，本项目不替上游授予额外许可。

- C2RF：[原始实现](https://github.com/QinglongYan-hub/C2RF)，[论文](https://link.springer.com/article/10.1007/s11263-025-02427-1)，*C2RF: Bridging Multi-modal Image Registration and Fusion via Commonality Mining and Contrastive Learning*, IJCV 2025。
- OSDataset：Xiang et al., *Automatic Registration of Optical and SAR Images Via Improved Phase Congruency Model*, IEEE JSTARS 2020, 13:5847–5861；[作者数据发布页](https://github.com/xym2009/OSdataset)。
