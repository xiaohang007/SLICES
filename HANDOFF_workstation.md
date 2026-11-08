# 交接：在 GPU 工作站上测试 `refactor-uv` 分支

这份文档给在工作站上接手测试的人（或 Claude Code 会话）。笔记本（纯 CPU、WSL2、16 GB 内存）上能测的都已经测过，剩下的是**只能在 GPU 上验证的部分**。

## 1. 分支里有什么
| 提交 | 内容 |
|---|---|
| `eb1f6c9` | 新增 `relax_model="auto"`（`src/slices/relax_backends.py`）：有 CUDA GPU 时用 NequIP-OAM-L 弛豫；没有 GPU 时用 NequIP-OAM-S 弛豫，再用 NequIP-OAM-L 算一次能量。模型在本机首次使用时编译并缓存到 `~/.cache/slices/nequip`；GPU 上的编译方式依次尝试 AOT + OpenEquivariance → AOT → TorchScript，失败自动降级 |
| `119afaf` | 改用 uv 管理环境，Python 3.10，`uv.lock` 锁定版本；去掉 m3gnet、TensorFlow；MatterGPT 的注意力改用 PyTorch 自带的 `scaled_dot_product_attention`，不再需要 flash-attn；并行脚本在导入 numpy 前把数学库线程设为 1；修复 `generate.py` 里 `sys` 被覆盖的问题 |

## 2. 笔记本上已经验证过的
- 21 项力场选择/降级测试全部通过（包括强制 GPU 但不可用时自动降级到 CPU）。
- 用新环境（numpy 2.2、pymatgen 2025.10）重新编码 MatterGPT 全部 2383 个结构：与仓库数据 0 个不同。
- 新注意力实现与旧实现结果一致：模型输出差 < 1e-6，同一随机种子生成的字符串完全相同，旧的 `.pt` 权重可以照常加载。
- MatterGPT 全流程在 CPU 上跑通：数据集 → 训练（batch 8）→ 生成 → 解码 + 形成能 + 新颖性（32 条真实 SLICES 中 31 条成功，形成能误差中位数 0.016 eV/atom）。
- 图形界面能启动。

## 3. 需要在工作站上确认的
1. `uv sync --extra cu128 --extra oeq` 能否装上，特别是 OpenEquivariance（`oeq`）。
2. NequIP 在 GPU 上实际用哪种编译方式：文件名里是 `aotinductor-oeq`、`aotinductor` 还是 `torchscript`。
3. GPU 上解码的速度，以及多个解码进程共用一块 GPU 时的显存（每个进程都会加载一份 OAM-L）。
4. GPU 上训练 MatterGPT 的速度和显存：现在没有 flash-attn，用的是 PyTorch 自带的注意力。
5. GPU 结果和 CPU 方案（`SLICES_DEVICE=cpu`）对比。

## 4. 怎么跑
前提：`nvidia-smi` 能正常显示显卡，驱动支持 CUDA 12.8（50 系显卡必须）。

```bash
# 没装 uv 的话（任选一种）
curl -LsSf https://astral.sh/uv/install.sh | sh
# pip install uv -i https://pypi.tuna.tsinghua.edu.cn/simple

cd <SLICES 仓库目录>
git fetch origin
git switch refactor-uv
git pull

bash tools/workstation_check.sh
```

脚本会依次做：装环境 → 打印 torch/GPU 信息 → 编译模型（GPU 和 CPU 各一次）→ 力场对比测试（32 条真实 SLICES）→ 训练 MatterGPT 1 轮（教程参数，batch 36）→ 生成 10 条 → 用 4 个和 8 个进程解码 32 条真实 SLICES。全程每 5 秒记录一次显存和内存。

- 所有结果写到 `~/slices_ws_test/<日期_时间>/`，**把整个文件夹打包发回来**；最重要的是 `summary.log`。
- MatterGPT 在这个文件夹里的副本中运行，不会在仓库里留下文件。环境装在仓库下的 `.venv`，不影响已有的 conda 环境或 LeMat-GenBench 的环境。
- 整个过程大约 0.5–1.5 小时，主要看训练速度。

可调的选项（环境变量）：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `EXTRAS` | `--extra cu128 --extra oeq` | 驱动较老时改成 `--extra cu126`；`oeq` 装不上时脚本会自动去掉后重试 |
| `TRAIN_EPOCHS` | `1` | `0` 跳过训练和生成 |
| `DECODE_THREADS` | `4 8` | 要测试的解码进程数 |

例如只测力场、不训练：`TRAIN_EPOCHS=0 bash tools/workstation_check.sh`

## 5. 可能遇到的问题
| 现象 | 处理 |
|---|---|
| `uv sync` 下载 torch 很慢 | torch 从 download.pytorch.org 下载（其他包走清华镜像），第一次需要耐心；之后有缓存 |
| `oeq` 装不上或编译失败 | 脚本会自动降级。把 `uv_sync.log` 和 `prepare_gpu.log` 发回来 |
| 某种编译方式失败后不想再重试 | 失败记录在 `~/.cache/slices/nequip/*.failed`，删掉对应文件后会重新尝试 |
| 训练报 CUDA out of memory | 记下显卡型号和显存；可以先用 `TRAIN_EPOCHS=0` 完成其余测试 |
| 解码时显存不够 | 把 `DECODE_THREADS` 调小，例如 `DECODE_THREADS="2 4"` |

## 6. 已知问题和待定事项
- **CPU 上训练很耗内存**：PyTorch 的省内存注意力在 CPU 上不支持 dropout（模型的注意力 dropout 是 0.1），batch 36 约需 13 GB 内存，曾两次把笔记本的 WSL 撑崩。要不要把注意力 dropout 改为 0，需要作者决定。GPU 上没有这个限制，这次的测试会顺便确认。
- flash-attn 没有适配 torch 2.9 + Python 3.10 的预编译包，所以不再安装。
- Docker 镜像和 Dockerfile 还是旧环境（M3GNet），没有更新。
- PyPI 上的 slices 仍是 2.0.12（旧代码）；合并后需要升版本号再发布。
- HTS/ 和 benchmark/ 下的旧流程仍然写着 `relax_model="m3gnet"`，在新环境里会报错；RNN 相关脚本按约定不改动。

## 7. 手动单独测某一项
```bash
source .venv/bin/activate
python -m slices.relax_backends                 # 只看会选哪个力场
python -m slices.relax_backends --prepare       # 编译并试跑
python examples/3.benchmark_relax_models.py --n 32 --models auto nequip-s+l nequip-m+l
SLICES_DEVICE=cpu python examples/3.benchmark_relax_models.py --n 32 --models auto
```
