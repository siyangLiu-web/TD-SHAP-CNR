# 🕸️ TD-SHAP-CNR (Time-Delay SHAP Causal Network Reconstruction)

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-Active-success)

`tdshap-cnr` 是一个高鲁棒性的 Python 工具包，专为多维连续时间序列（如多模态生理信号、脑电、金融指标等）设计，用于**非线性动态因果网络的重构与全景拓扑评估**。

本工具包基于**时间延迟嵌入（Time-Delay Embedding）**、**随机森林（Random Forest）**、**SHAP 贡献度分解**以及**自适应 K-Means 网络修剪**算法，能够精准追踪时间序列系统在演化过程中的网络拓扑突变与单通路极值爆发机制。

---

## ✨ 核心特性

- **🌌 节点与领域无关性 (Node-Agnostic)**：不对输入特征做任何生硬限制或命名预设。只要是格式为 `(Samples, Features)` 的时序数据表，工具包均能自动解析节点并构建网络。
- **🛡️ 高鲁棒性 (Crash-Proof)**：内置防御性数据清洗流水线，自动处理 `NaN`、`Inf`，并自动过滤静息或零方差的时序片段，确保动态滑动窗口不崩溃。
- **📊 全景拓扑特征提取**：自动计算每一个动态时间窗口的 10 项复杂网络图论特征（如净信息流、全局效率、富人俱乐部等），并额外输出序列全局的**时间复杂度（时间熵）**。
- **⚡ 单状态通路极值与频次追踪**：创新性提出单状态摘要评估算法，无需外部独立基线对比，即可准确提取通路的绝对点亮率（活跃频次）与 Top-K 爆发极值。

---

## 📦 安装 (Installation)

推荐使用虚拟环境进行安装（要求 Python 3.8+）。

### 从源码本地安装 (开发模式)

```bash
# 1. 克隆本仓库
git clone [https://github.com/yourusername/tdshap_cnr.git](https://github.com/yourusername/tdshap_cnr.git)
cd tdshap_cnr

# 2. 安装依赖并以可编辑模式安装包
pip install -e .

```

*(注：安装完成后，您可以在系统的任何 Python 脚本中直接 `import tdshap_cnr`)*

---

## 🚀 快速入门 (Quick Start)

这是一个最简的端到端调用示例，无需任何复杂配置即可体验全套因果网络重构。

```python
import pandas as pd
import numpy as np
from tdshap_cnr import PipelineConfig, CausalNetworkPipeline

# 1. 准备您的时序数据 (示例: 生成包含5个节点的随机游走时序数据)
np.random.seed(42)
data = np.cumsum(np.random.randn(1000, 5), axis=0) 
df_features = pd.DataFrame(data, columns=['Node_A', 'Node_B', 'Node_C', 'Node_D', 'Node_E'])

# 2. 初始化流水线 (默认开启所有功能)
pipeline = CausalNetworkPipeline()

# 3. 运行完整评估
results = pipeline.run_pipeline(df_features)

# 4. 获取核心输出
print("动态网络张量维度:", results['tensor_continuous'].shape) 
print("\n时序拓扑演化 (前3个窗口):\n", results['topology_features_series'].head(3))
print("\n特定通路的统计摘要:\n", results['pathway_summaries'].head())

```

---

## ⚙️ 进阶配置与使用 (Advanced Usage)

对于工程与研究应用，您可以通过配置系统（`PipelineConfig`）对流水线的每一个细节进行微调。

### 1. 配置参数详情

```python
from tdshap_cnr import PipelineConfig, ModelConfig, DynamicConfig, EvalConfig

# 实例化配置总控
config = PipelineConfig()

# === 核心模型配置 ===
config.model.max_lag = 20                # 扩大最大滞后阶数
config.model.rf_n_estimators = 200       # 增加随机森林树的数量以提升稳定性
config.model.shap_background_samples = 300 # 控制 SHAP 背景样本数，平衡精度与内存消耗

# === 动态窗口追踪配置 ===
config.dynamic.window_size = 150         # 每个滑动窗口包含 150 个时间步
config.dynamic.step_size = 30            # 步长设为 30

# === 拓扑特征与评估配置 ===
# 自定义需要计算的图论特征 (不需要的可以删去以节省算力)
config.eval.topology_metrics = [
    'density', 'net_flow', 'global_efficiency', 'modularity'
]
# 修改通路强度摘要的极值提取比例 (默认提取 Top-10% 的极值求均值)
config.eval.top_k_ratio = 0.15           

```

### 2. 执行单通路定向分析

当网络节点极多（如脑电 128 通道）时，计算所有 $N \times (N-1)$ 条边的统计摘要非常耗时。您可以通过传入 `specific_pathways` 仅对关心的通路进行计算。

```python
pipeline = CausalNetworkPipeline(config)

# 仅关心 Node_A 对 Node_B 的驱动，以及 Node_C 对 Node_B 的驱动
results = pipeline.run_pipeline(
    df_features=df_features,
    specific_pathways=['Node_A->Node_B', 'Node_C->Node_B'] 
)

```

---

## 📂 核心输出结果解析 (Outputs Structure)

`run_pipeline` 返回一个字典，包含 5 大核心结构：

### 1. `results['tensor_continuous']` & `results['tensor_binary']`

* **类型**: `numpy.ndarray`
* **形状**: `(Time_Windows, N_Nodes, N_Nodes)`
* **说明**: 包含时间轴上所有窗口的因果邻接矩阵（连续强度与经过自适应 K-Means 修剪后的二值图）。

### 2. `results['topology_features_series']`

* **类型**: `pandas.DataFrame`
* **说明**: 记录了系统随时间演化的拓扑结构。包含如下字段（视配置而定）：
* `Window_Idx`: 时间窗口索引
* `<NodeName>_InDegree` / `OutDegree` / `NetFlow`: 各节点的加权入度/出度/净信息流
* `Density`: 整体连通率
* `Global_Efficiency`, `Clustering`, `Modularity`, `Max_Participation`, `Rich_Club`, `Spectral_Radius` 等高级宏观图论指标。



### 3. `results['topology_temporal_entropy']`

* **类型**: `pandas.DataFrame` (形状为 `1 x N_Metrics`)
* **说明**: 将上述时序拓扑结构的波动性压缩为一个二阶特征（即**时间熵**），反映各个指标在整个单状态观测期内的复杂度和动态不稳定性。数值越大，代表系统重组与波动越剧烈。

### 4. `results['pathway_summaries']`

* **类型**: `pandas.DataFrame`
* **说明**: 用于评估特定因果通路强度的统计摘要。字段包括：
* `Pathway`: 边名称 (如 `Node_A->Node_B`)
* `Mean_Strength`: 全局时间平均因果强度
* `TopK_Peak_Strength`: Top-K% 爆发上限极值（用于反映脉冲生理信号的驱动上限）
* `Strength_Std`: 强度波动率（替代传统的基于双状态对比的方差）
* `Active_Windows_Ratio`: 绝对点亮频率（反映该通路在整个过程中存活且大于0的窗口比例）



---

## 📝 算法原理解析与参考文献

1. **时延嵌入特征空间**: 将序列转化为具备自回归性质的监督学习滞后矩阵。
2. **随机森林拟合**: 在不施加严格线性假设的条件下，利用树集成模型捕获节点间的非线性交互动力学。
3. **SHAP 分解**: 利用博弈论 Shapley 值将模型的预测贡献反向拆解给每一个滞后驱动特征，累加构建连续因果张量。
4. **K-Means 自适应修剪**: 将矩阵元素展开为一维分布，利用无监督聚类自适应寻找有效边的截断下界阈值。
5. **动态拓扑降维**: 将三维时变网络张量映射为随时间演化的低维拓扑特征集。
