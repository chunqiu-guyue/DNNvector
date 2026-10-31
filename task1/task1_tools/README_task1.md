# 任务一执行手册：ResNet50 非活跃窗口长尾形态与跨迭代稳定性

> 配套脚本：`task1_tools/t1_multistep_trace.py`、`t1_extract.py`、`t1_analyze.py`、`t1_stability.py`、`t1_peak_mem.py`
> 全部在真实环境执行，本目录只做归档。

---

## 0 先厘清：两条测量途径谁负责什么

这是任务一最容易含混的地方，必须先钉死，否则报告里"算子时序"会写两套互相打架的数。

### 0.1 两者测的根本不是同一个东西

| | PyTorch Profiler | G10 cuDNN profiling（profile config） |
|---|---|---|
| 跑什么 | **端到端训练 step**（整个 forward+backward 图） | codegen 生成的**单个 kernel 独立程序**，一次一个算子 |
| 测什么 | 每个 GPU kernel 的真实 `ts/dur`，含 launch 间隔、并发、多 stream | 该 kernel **孤立执行**的耗时（micro-benchmark） |
| 有"第几步"的概念吗 | **有**：`schedule(wait, warmup, active)` 选第几个训练步 | **没有**：它不跑训练迭代，只跑一个算子 |
| 稳态怎么保证 | warmup 步 + active 步 | `cudnn/src/main.cu:712-719`：`iterations=1, mean_iterations=1`（**hardcoded，无 warmup**） |
| 能给出张量语义吗 | ❌ 只到 kernel，不知道读写哪个张量 | ✅ `interval.config` 给到"哪个张量的第几个窗口" |
| 产物 | `kernel_trace.json` + `mem_snapshot.pickle` | `cudnn_kernel_times.txt` + `interval.config` + `TensorPeriodLog.py` |

### 0.2 所以 `active=1` 在 G10 里没有对应开关

- **PyTorch 的 `active`** 选的是"训练迭代"。`active=1` = 只记录 1 个稳态训练步。
- **G10 侧不存在等价物**，因为它的测量单位是"单个 kernel 的一次独立执行"，压根没有迭代维度。
- 若一定要找 G10 里最接近的东西，是 `cudnn/src/main.cu` 的 `iterations / mean_iterations`（跑 N 次、只对最后 M 次取平均），但它被写死成 `1 / 1` —— 等于**每个 kernel 只测一次，且没有预热**。
- 另一个容易误认的是 sim config 的 `num_iteration`，那是**仿真**重复迭代数，与分析阶段无关。

> 这条 hardcoded `1/1` 有实际后果：`cudnn_kernel_times.txt` 里 `001 = 11478.94`
> 明显是首次 cuDNN 调用开销污染（其余层多在数百 µs）。这也正好说明为什么需要
> PyTorch 的端到端测量来校核——449 行求和 162671 µs vs trace 端到端 152940 µs，
> 差 6.4%，量级自洽但存在个别异常值。

### 0.3 谁才是任务一的"最终结果"

不是二选一，而是**分工**，题目三个维度各自只能由一个来源支撑：

| 题目要求 | 唯一可行的来源 | 为什么 |
|---|---|---|
| 统计**向量**非活跃窗口分布 | **G10**（`interval.config` / `TensorPeriodLog.py`） | PyTorch Profiler 没有张量级语义，给不出"哪个张量空闲多久" |
| 长尾形态 | **G10** | 同上，分布的对象是张量窗口 |
| 提取算子执行时序 | 两者都要，但用途不同 | G10 的逐 kernel 时间**驱动生命周期时间轴**；PyTorch 的端到端时序用于**校核与并发分析** |
| **跨迭代稳定性** | **PyTorch**（必须多步） | G10 测的是单 kernel 孤立执行，没有迭代概念；它只能给"结构层"确定性 |

一句话：**结论主体来自 G10，证据校核与稳定性来自 PyTorch**。报告里必须分栏写，不能混着说"算子时序"。

### 0.4 一句话结论：只缺一件数据

你搬进 `forresnet50/` 的东西已经覆盖 **Q1 算子时序、Q2 窗口分布、Q3 长尾形态**，
以及 Q4 的**结构层**证据，现在还补上了 `TensorPeriodLog.py`（可直接复现论文同源图）。
**唯一缺口仍是"多步 trace"**——现有 `resnet50_bs16_kernel_trace.json` 是 `active=1` 采的单步，
无法支撑"跨迭代稳定性"的时长层结论。

因此整个任务一 = **1 次新实验（多步采样）+ 5 个脚本的离线计算**。

---

## 1 现有数据盘点

### 1.1 G10 侧（`forresnet50/ResNet50/`）——完整，无需重跑

| 文件 | 行数 | 用途 | 任务一角色 |
|---|---|---|---|
| `cudnn_kernel_times.txt` | 449 | 逐调度单元实测耗时（µs 量级） | **Q1 数据源**、窗口时长的计算基础 |
| `cudnn_workspace_sizes.txt` | 449 | 逐 kernel workspace | 旁证（workspace 不进窗口统计） |
| `cudnn_input_data.txt` | 449 | 输入规模 | 备用 |
| `statistics/interval.config` | 7508 | 1019 个张量的生灭 + 697 个非活跃窗口 | **Q2/Q3 数据源** |
| `statistics/kernels.config` | 5283 | 449 kernel ↔ 层 ↔ 输入输出张量 | 前反向分界、访问序列、张量角色 |
| `statistics/layers.config` | 2526 | 每层 Input/Output/d_Input/d_Output/Weight/d_Weight | **张量分类**（activation/grad/weight） |
| `statistics/tensors.config` | 1019 | 张量清单 + 地址范围 | 显存口径 |
| `statistics/real_mem.config` | 449 | 逐 kernel 显存占用 | 峰值对账（可选） |
| `statistics/kernel_time_table.config` | 449 | 时间表数值版 | 校核 `cudnn_kernel_times` |
| `statistics/pre_dealloc.config` | 3157 | 预释放曲线 | 备用 |
| `statistics/prefetch_guide.config` | 0 | 空（`use_prefetch=0`） | 不用 |
| `sim_result.*` | — | 仿真统计 | 任务二用，任务一不用 |
| `Makefile` / `src/` / `scripts/` | — | profile 阶段编译脚手架 | 不参与分析 |

### 1.2 实测侧（`forresnet50/`）——够用，但只有 1 步

| 文件 | 内容 | 限制 |
|---|---|---|
| `resnet50_bs16_kernel_trace.json` | 单步 Chrome trace，1015 kernel（窗内 921） | **只有 1 个稳态步** |
| `resnet50_bs16_mem_snapshot.pickle` | 单步显存 alloc/free 快照 | 单步 |
| `resnet50_bs16_meta.json` | 采样元信息 | active_step=2 |
| `ResNet50_layer_attribution*.txt` | trace kernel → 层归属 | 已完成 |
| `kernel_trace.py` | 采样脚本（`active=1` 版） | 需改造为多步 |

### 1.2b G10 sim 模式产物（新补）——可直接复现论文图

| 文件 | 内容 | 用法 |
|---|---|---|
| `ResNet50_TensorPeriodLog.py` | `sd_size` / `sd_time` 两个各 697 项的数组，即论文 `figure_drawing/` 脚本的输入 | `t1_paper_repro.py` 直接读它 |
| `ResNet50_NNMemConsumptionLog.py` | 显存消耗日志 | 论文 `plot_mem_consumption.py` 的输入，任务一可选 |

**重要：`sd_time` 的单位是 ns，不是 µs。** 证据：

```
sd_time max          = 1.6267e8
cudnn_kernel_times∑  = 162671（µs）
1.6267e8 / 1000      = 162670 µs ≈ 整步 162671 µs   ← 正好等于"整步"
```

论文脚本 `plot_tensor_period_distribution.py` 把 `sd_time` 直接画在标着
`Inactive Time (µs)` 的轴上，等于把 ns 当 µs 标，**会差 1000 倍**。
`t1_paper_repro.py --unit-scale 1e-3` 默认做了换算，与 trace 口径对齐。

另一个必须注意的口径差：`TensorPeriodLog.py` 的 697 项**含全局权重窗口**
（median ≈ 50.65 ms），而清洗后（剔 global + looped）375 项的 median ≈ 1.20 ms。
两者不矛盾，是分母不同 —— 报告里两张图要分别标注口径。

### 1.3 缺口

| 缺什么 | 为什么需要 | 怎么补 |
|---|---|---|
| **多步 trace（≥3 步，建议 5 步）** | 跨迭代稳定性的时长层证据 | `t1_multistep_trace.py` |
| （可选）`sim_result` 多迭代对比 | 若想把仿真侧的迭代稳定性也纳入 | `num_iteration` 调大后重跑 sim config |

### 1.4 关于 G10 自带的分析工具（`G10/src/resources/`）

论文的图有对应脚本；现在 `TensorPeriodLog.py` 已补齐，**技术上可以改路径直接用**，
但有两个坑，所以本手册用 `t1_paper_repro.py` 替代：

| 论文脚本 | 作用 | 现状 | 建议 |
|---|---|---|---|
| `figure_drawing/plot_tensor_period_distribution.py` | size vs inactive time 双对数散点 | 可改路径跑 | **别直接用**：①它 `exec` 硬编码 `../../../results/BERT_Base/...`，路径依赖重；②`sd_time` 是 ns，图上标 µs 会差 1000× |
| `figure_drawing/plot_tensor_time_cdf.py` | 非活跃时长 CDF（横轴为百分位） | 同上 | 同上；`t1_paper_repro.py --tag paper` 输出的 `figB` 即等效 |
| `figure_drawing/plot_mem_consumption.py` | 显存曲线 | 可用 `NNMemConsumptionLog.py` | 可选，用 `t1_peak_mem.py` 替代更直接 |
| `gatherKernelInfo.py` | 汇总 `sim_result.*` → `data.json` | ✗ | 面向多模型批量实验，任务一用不上 |
| `figureDrawingDataPrepKernelCDF.sh` | 拷 `sim_result.kernel` 到绘图目录 | ✗ | 面向论文 Figure 13（slowdown CDF），与任务一无关 |

**唯一的正面价值**：这些脚本的轴标签写 `Inactive Time ($\mu$s)`，说明论文认为
"非活跃时长应以 µs 表达" —— 这与 `t1_extract.py` 把 kernel 时间统一到 µs 的做法一致，
只是论文脚本在取数时漏了 ÷1000。报告里这样写即可：

> 「按 G10 artifact `src/resources/figure_drawing/plot_tensor_period_distribution.py` 的绘图口径
> （横轴 inactive time/µs、纵轴 size/byte、双对数）绘制 ResNet50 的张量周期分布；
> 由于 `TensorPeriodLog.py` 输出的 `sd_time` 为 ns 量级（其最大值为 1.6267e8，
> 与实测整步 162671 µs 符合 1000× 关系），绘图前统一换算为 µs。」

---

## 2 执行顺序（共 5 步，唯一需要 GPU 的是步骤 1）

```bash
cd forresnet50

# ── 步骤 0：论文同源图（纯 CPU，秒级，可选但推荐）───────────
#     用 TensorPeriodLog.py 复现论文样式的两张图，证明"我们和论文看的是同一个量"
python3 task1_tools/t1_paper_repro.py \
    --period-log ./ResNet50_TensorPeriodLog.py \
    --outdir ./task1_out --unit-scale 1e-3

# ── 步骤 1：唯一的新实验（需 GPU，约 3–5 分钟）──────────────
python3 task1_tools/t1_multistep_trace.py --output-dir ./task1_out --active 5

# ── 步骤 2：解析 G10 产物，生成主表（纯 CPU，秒级）──────────
python3 task1_tools/t1_extract.py \
    --g10-dir ./ResNet50 \
    --outdir  ./task1_out \
    --ref-step-us 152940

# ── 步骤 3：长尾与分布分析 + 出图（纯 CPU，秒级）───────────
python3 task1_tools/t1_analyze.py --indir ./task1_out

# ── 步骤 4：跨迭代稳定性（纯 CPU，秒级）───────────────────
python3 task1_tools/t1_stability.py \
    --trace ./task1_out/resnet50_bs16_multi5_kernel_trace.json

# ── 可选：显存峰值对账 ────────────────────────────────────
python3 task1_tools/t1_peak_mem.py \
    --real-mem ./ResNet50/statistics/real_mem.config \
    --snapshot ./resnet50_bs16_mem_snapshot.pickle
```

依赖：不需要 numpy/pandas；`matplotlib` 只在步骤 0 和 3 出图时用（`pip3 install matplotlib`）。
所有图标签都用英文，避免缺字体出现方块。

### 2.1 两张图的分工（别搞混）

| 图 | 来源 | 分母 | 用途 |
|---|---|---|---|
| `paper_figA/figB` | `TensorPeriodLog.py`（697 项，**含**全局权重） | 论文口径 | 证明我们复现了论文的同源图；与论文 Figure 可直接对照 |
| `fig1..fig5` | `interval.config`（清洗后 375 项，**不含** global/looped） | 本项目分析口径 | 长尾统计、分层、覆盖度、稳定性 —— **这些才是报告的结论主体** |

报告里两张图都要放，并各标一句口径。用量化差异（median 50.65 ms vs 1.20 ms）
正好可以说明"全局权重窗口必须剔除"，这是一个有说服力的方法学细节。

---

## 3 每步的产物与验收

### 步骤 0 产物

```
paper_figA_size_vs_time.pdf/.png   论文同款 size–time 散点（x=µs, y=byte, 双对数）
paper_figB_time_cdf.pdf/.png       论文同款 CDF（横轴为窗口百分位）
paper_repro.json                   论文口径的统计摘要（697 项）
```

验收：`paper_repro.json` 里 `max` 应 ≈ **162670**（µs）。若显示 1.6267e8，说明
`--unit-scale` 忘了设（默认 1e-3，不会错，除非手动改）。

### 步骤 2 产物（`task1_out/`）

```
extract_report.txt   口径报告（单位判定、分界、各类计数）
kernels.csv          449 行
tensors.csv          1019 行
intervals.csv        697 行（含 t_start_us/t_end_us/dur_us/role/cross_phase）
axis.json            kernel 序号→绝对时间的前缀和（覆盖度扫描线用）
```

**必须核对的 5 个数**（与本空间一致即口径正确）：

| 量 | 期望值 | 若不符说明 |
|---|---|---|
| 单位判定 | `us`（sum≈162671） | 若判成 `ms`，检查 `--ref-step-us` |
| kernel 数 | 449 | 文件不完整 |
| 前反向分界 kernel | 应为第一个含 `Backward` 的 ID | — |
| 张量数（含全局） | 1019 | — |
| 窗口记录 / looped / global | 697 / 213 / 322 | 记录数口径 |

### 步骤 3 产物

```
stats.json          全部数值
tables.md           6 张 Markdown 表，直接贴进报告
fig1_size_vs_time   论文同款散点
fig2_ccdf            CCDF + 尾部拟合（形状判定）
fig3_size_bucket     按 size 分桶箱线
fig4_idle_bytes      空闲字节随时间（扫描线）
fig5_kernel_cdf      逐 kernel 耗时 CDF
```

### 步骤 4 产物

```
stability.json      每步原始值 + CV + 逐 op CV + Spearman + 判定 + 诊断
stability.md        表 7，直接贴进报告
```

两条对齐路径：默认按 `External id` 把 cpu_op 关联到 kernel（A 路径，粒度=一次 cuDNN 调用）；
若一个都匹配不到，自动退化为按 kernel 名字聚合（B 路径），并打印诊断
（窗内 cpu_op 数 / 带 External id 数 / 成功关联数 / 最常见 op 名）。
强制指定用 `--mode op` 或 `--mode kernel`。

---

## 4 判据与预期（写报告时照此下结论）

| 指标 | 判据 | 你要填的 |
|---|---|---|
| 窗口时长跨度 | max/median 是否 > 10× | 预期 ~125×（150163/1199）→ 跨数量级 |
| Top-10% 贡献率 | >60% 才算典型长尾 | 预期 ~28.7% → **不是幂律长尾** |
| 形状判定 | log10 直方图峰谷 | 预期 **双峰**（~1 ms 与 ~150 ms 两簇） |
| 墙钟 CV | <3% 稳定 | 需实测 |
| kernel 数 | 各步完全一致 | 需实测 |
| 逐 op CV 中位数 | <5% | 需实测 |
| 前向排名 Spearman | >0.9 | 需实测 |

**报告里最容易被追问的一点**：如果你只报 Top-10% = 28.7% 就说"长尾"，会被反驳。
正确表述是：

> 非活跃窗口呈**双峰分布**而非单峰长尾：一类是相位内相邻 kernel 之间的短空隙
> （中位数约 1.2 ms），另一类是激活跨前向到反向的长等待（接近整步 150 ms）。
> 前者的总量由大量小窗口累积，后者由少数长窗口主导；按 Top-k 贡献率看，
> 前 20% 窗口贡献 52.9% 的总空闲时长，尾部并不满足幂律（α 与 R² 见附表）。

---

## 5 报告骨架（与脚本产物一一对应）

```
1 测量方法与口径
  1.1 设备（RTX 5060 8GB）/ torch 版本 / bs16 / 5 步稳态
  1.2 两条测量途径的分工
      ← 必写：G10 逐 kernel 孤立测量（无 warmup，iterations=1）
              vs PyTorch 端到端 step 测量（active=5）
              → 前者给张量语义，后者给端到端校核与迭代稳定性
  1.3 三重时间口径：trace 实测 µs ｜ cudnn_kernel_times µs ｜ sd_time/Estimated ns
      ← 必须给出 ns→µs 的 1000× 换算证据（max(sd_time)/1000 ≈ 整步）

2 算子执行时序（Q1）           → kernels.csv + fig5_kernel_cdf + 表 6 Top-15
                                 附：G10 逐 kernel 和 162671µs vs trace 152940µs（差 6.4%）
3 非活跃窗口分布（Q2）         → 表 1 + 清洗口径（697→N）+ paper_figA/figB（论文口径对照）
4 长尾形态（Q3）               → 表 2 三指标 + fig2_ccdf + fig3_size_bucket + 形状判定
                                 表 3 可卸载潜力（扫描线）+ fig4_idle_bytes
5 跨迭代稳定性（Q4）           → 表 7 + stability.json 判定
                                 5.1 结构层：静态图唯一确定（构造性）
                                 5.2 时长层：5 步实测 CV（PyTorch，G10 无此能力）
6 与 G10 离线分析的交叉验证     → 表 5（est vs 实测）+ 表 4（跨相位分层）
7 局限                         → 单设备/单精度/单 bs；5 步；kernel 时间为累加非墙钟；
                                 G10 无 warmup 导致个别 kernel 异常（001=11478µs）；
                                 cudaMalloc 盲区；双峰判定依赖分界点定义
8 结论                          → 三条可引用结论
```

---

## 6 三条必须避免的口径错误

1. **`physical_ms` / `dur_us` 是 µs**。`interval.config` 的 `Estimated Time` 是 ns（×1000），
   论文图 x 轴写 µs，三者不可混。
2. **必须剔除 `is_looped=1`（213 条，占时间 65.5%）与 `is_global=1`（322 条）**。
   脚本已自动过滤，别在报告里又用原始 697 当分母。
3. **`sum(窗口时长)` 不能除以步长**。375 个窗口在时间上大量重叠，
   必须用 `fig4` 的扫描线（峰值 / 均值 / 积分）表达可卸载潜力。

4. **G10 的 `Estimated Time` 少算末端 kernel**（读源码得到，报告里可写成实现细节发现）。

   `analysis.cc:2065` 写的是：

   ```cpp
   time_estimated = kernel_time_table[end] - kernel_time_table[start];
   ```

   而 `kernel_time_table` 是**按 ns 累计的前缀和**（`table[i] = sum(kernel[0..i-1]) × 1000`，
   已被 `statistics/kernel_time_table.config` 逐项核对：`table[1]=2251.23`、`table[2]=1.14812e7`，
   与 `kt 前缀和×1000` 完全吻合）。因此：

   ```
   est = sum(kernel[start .. end-1]) × 1000 (ns)     ← 不含 kernel[end]
   物理空闲期 = sum(kernel[start .. end])  (µs)      ← 含两端
   ```

   `ast.h:364` 那行 `double time_estimated; //us` 的注释是错的，实际是 ns。

   实测后果（`t1_analyze` 表 5 按跨度分层）：

   | 窗口跨度 `end-start` | N | median(est/1000 ÷ dur) |
   |---|---|---|
   | 1 | 157 | **0.0459** ← G10 显著低估 |
   | 2–10 | 27 | 0.9669 |
   | >10 | 191 | **0.9997** ← 完全吻合 |
   | 全部 | 375 | 0.9959 |

   即：**长窗口上 G10 的估算与物理空闲期几乎完全一致（0.9997），
   短窗口（跨度=1）因未计入末端 kernel 而系统性低估到 4.6%**。
   报告里这样写，既解释了差异，又证明了复算的正确性。

---

## 7 如果时间只剩一天，最小可行路径

1. 跑 `t1_multistep_trace.py`（3–5 分钟）；
2. 跑 `t1_extract.py` + `t1_analyze.py`（秒级），拿到 6 张表和 5 张图；
3. 跑 `t1_stability.py`，拿到表 7；
4. 想加一张"和论文对得上"的图，再跑 `t1_paper_repro.py`（秒级）；
5. 按 §5 骨架填报告，§4 的判据直接抄进结论段。

挑战任务（工具流 / 注意力模型）不必做，报告"后续工作"里写一句
`decide_mode` 改走 `cudnnGetConvolution*Algorithm_v7` 离线查询即可。
