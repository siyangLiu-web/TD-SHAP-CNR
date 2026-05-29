from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass
class ModelConfig:
    max_lag: int = 15  # 最大滞后阶数
    rf_n_estimators: int = 150  # 随机森林树的数量
    rf_max_depth: Optional[int] = 10  # 树的最大深度
    rf_max_features: str = 'sqrt'  # 最大特征选择策略
    rf_n_jobs: int = -1  # 多核并行加速
    rf_random_state: int = 42
    shap_background_samples: int = 500  # SHAP 期望值计算的背景样本数


@dataclass
class PruneConfig:
    method: Literal['kmeans', 'percentile', 'none'] = 'kmeans'
    kmeans_tolerance: float = 1.4  # 聚类下界的松弛因子
    percentile_threshold: float = 90.0


@dataclass
class DynamicConfig:
    window_size: int = 120  # 滑动窗口大小 (样本点)
    step_size: int = 40  # 滑动步长


@dataclass
class EvalConfig:
    topology_metrics: List[str] = field(default_factory=lambda: [
        'in_degree', 'out_degree', 'net_flow',
        'density', 'clustering', 'global_efficiency',
        'modularity', 'participation_coefficient',
        'rich_club', 'spectral_radius'
    ])

    # 是否计算序列的二阶特征：时间复杂度 (时间熵)
    calculate_temporal_entropy: bool = True

    top_k_ratio: float = 0.10  # 提取 Top-10% 作为极端爆发上限


@dataclass
class PipelineConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    prune: PruneConfig = field(default_factory=PruneConfig)
    dynamic: DynamicConfig = field(default_factory=DynamicConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)