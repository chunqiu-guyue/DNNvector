### 表 1 非活跃窗口描述统计（清洗后）

| N | 总时长(µs) | 均值 | 中位数 | P90 | P95 | P99 | 最大 | max/median |
|---|---|---|---|---|---|---|---|---|
| 375 | 9451211 | 25203.2 | 1198.9 | 63308.5 | 66359.5 | 91381.5 | 150162.6 | 125.3× |

### 表 2 长尾三指标

| 指标 | 值 | 判读 |
|---|---|---|
| top1pct 贡献率 | 5.1% （4 个窗口） | |
| top5pct 贡献率 | 16.3% （19 个窗口） | |
| top10pct 贡献率 | 29.4% （38 个窗口） | |
| top20pct 贡献率 | 52.9% （75 个窗口） | |
| Gini | 0.596 | 0=均匀，1=极度集中 |
| CCDF 尾指数 α | 4.85 （R²=0.784） | 越小尾越重 |
| semi-log 拟合 R² | 0.754 | 高于 log-log 则更接近指数尾 |
| 形状判定 | **双峰** | 由 log10 直方图峰谷判定 |

### 表 3 可卸载潜力（扫描线）

| 量 | 值 |
|---|---|
| 峰值同时空闲 | 1631.29 MB |
| 平均同时空闲 | 624.60 MB |
| 平均/峰值 | 38.3% |

### 表 4 分层（跨前后向 vs 相内）

| 类别 | N | 中位数(µs) | P90(µs) | 最大(µs) |
|---|---|---|---|---|
| 跨前后向 | 174 | 50336.7 | 66979.3 | 150162.6 |
| 相位内 | 201 | 90.9 | 1018.4 | 29641.0 |

### 表 5 G10 估算 vs 物理窗口时长（按窗口跨度分层）

G10 的 `time_estimated = table[end] - table[start]`（table 为 ns 前缀和），
等于 `sum(kernel[start..end-1])`，**不含末端 kernel[end]**；
本表的 `dur_us` 为物理空闲期 `sum(kernel[start..end])`。

| 窗口跨度 (end-start) | N | median(est/1000 ÷ dur) | 判读 |
|---|---|---|---|
| span=1 | 157 | 0.0459 | G10 少算末端 kernel，显著低估 |
| span=2-10 | 27 | 0.9669 | 两者口径一致 |
| span>10 | 191 | 0.9997 | 两者口径一致 |
| **全部** | 375 | 0.9959 | 长窗口主导，整体接近 1 |

### 表 6 Top-15 慢 kernel

| kid | 名称 | 层 | 耗时(µs) | 占比 |
|---|---|---|---|---|
| 2 | BatchNorm2d_Forward | 1 BatchNorm2d | 30873.4 | 18.98% |
| 7 | Conv2d_Forward | 6 Conv2d | 23166.6 | 14.24% |
| 3 | ReLU_Forward | 2 ReLU | 20829.8 | 12.80% |
| 151 | BatchNorm2d_Backward | 137 BatchNorm2d | 16018.1 | 9.85% |
| 1 | Conv2d_Forward | 0 Conv2d | 11478.9 | 7.06% |
| 149 | AdaptiveAvgPool2d_Backward | 140 AdaptiveAvgPool2d | 8225.1 | 5.06% |
| 159 | Conv2d_Backward_Input | 134 Conv2d | 6932.9 | 4.26% |
| 4 | MaxPool2d_Forward | 3 MaxPool2d | 6706.6 | 4.12% |
| 321 | Conv2d_Backward_Weight | 58 Conv2d | 6583.7 | 4.05% |
| 153 | Conv2d_Backward_Weight | 136 Conv2d | 3662.0 | 2.25% |
| 158 | Conv2d_Backward_Weight | 134 Conv2d | 2791.7 | 1.72% |
| 214 | Conv2d_Backward_Weight | 108 Conv2d | 1714.4 | 1.05% |
| 175 | Conv2d_Backward_Weight | 126 Conv2d | 1491.1 | 0.92% |
| 186 | Conv2d_Backward_Weight | 121 Conv2d | 1372.6 | 0.84% |
| 447 | Conv2d_Backward_Input | 0 Conv2d | 1026.7 | 0.63% |
