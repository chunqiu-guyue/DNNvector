# DNNvector — G10 向量生命周期分析的测量与适配

本科科研轮转课题的代码、数据与产物。工作分两部分：在 ResNet50 上测量并统计训练负载的
向量非活跃窗口，然后把 G10 的离线分析流程从普通卷积扩展到 ResNeXt50_32x4d 的 32 组分组卷积。

结论与论证以报告为准，本仓库提供支撑材料：代码、原始测量与统计产物。

## 目录

| 路径 | 内容 |
|---|---|
| `G10/` | G10（MICRO'23）artifact 副本，及本次对 `ast / analysis / codegen / parser / scanner / cudnn` 的适配改动 |
| `G10/results/` | 适配后的运行产物：kernel 时间表、workspace 尺寸表、`interval.config`、仿真结果 |
| `task1/ResNet50/` | G10 侧测量产物（449 行 kernel 时间表、697 个窗口的 `interval.config` 等） |
| `task1/task1_tools/` | 解析、统计与采样脚本 |
| `task1/task1_out/` | 脚本输出：窗口主表、`stats.json`、`tables.md`、稳定性判定 |
| `traces/` | PyTorch Profiler 采集的 kernel trace 与显存快照 |
| `figs/` | 报告插图的绘图脚本与输出 |
| `kernel_trace.py`、`step*.py` | 采样、逐层归属与派发比对脚本 |


## 环境

- WSL2 / Ubuntu，NVIDIA RTX 5060 Laptop（sm_120）
- CUDA 12.8 + cuDNN；G10 编译依赖 g++、flex、bison
- 统计与绘图：Python 3、numpy、matplotlib（仅出图需要）

