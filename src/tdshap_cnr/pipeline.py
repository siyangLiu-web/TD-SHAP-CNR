import pandas as pd
from tdshap_cnr.config import PipelineConfig
from tdshap_cnr.models.core import TDShapCausalModel
from tdshap_cnr.dynamics.builder import DynamicNetworkBuilder
from tdshap_cnr.metrics.evaluator import NetworkAnalyzer


class CausalNetworkPipeline:
    def __init__(self, config: PipelineConfig = PipelineConfig()):
        self.cfg = config
        self.model = TDShapCausalModel(self.cfg.model, self.cfg.prune)
        self.builder = DynamicNetworkBuilder(self.model, self.cfg.dynamic.window_size, self.cfg.dynamic.step_size)
        self.analyzer = NetworkAnalyzer(self.cfg.eval)

    def run_pipeline(self, df_features: pd.DataFrame, specific_pathways: list = None) -> dict:
        """
        核心执行入口：对单一状态的特征时序表执行完整的建模与评估
        """
        # 1. 动态窗口滑动，重构张量
        dyn_results = self.builder.build_timeline(df_features)

        # 2. 提取 10项 完整时序拓扑结构与时间熵
        df_topology, df_temporal_entropy = self.analyzer.extract_temporal_topology(
            dyn_results['tensors_binary'],
            dyn_results['tensors_continuous'],
            dyn_results['nodes']
        )

        # 3. 基于方案A 提取各通路的统计摘要
        nodes = dyn_results['nodes']
        n_nodes = len(nodes)
        pathway_records = []

        for i in range(n_nodes):
            for j in range(n_nodes):
                if i == j: continue
                pathway_name = f"{nodes[i]}->{nodes[j]}"

                # 若用户指定了特定通路，则过滤
                if specific_pathways and pathway_name not in specific_pathways:
                    continue

                seq = dyn_results['tensors_continuous'][:, i, j]
                metrics = self.analyzer.summarize_causal_pathway(seq)

                if metrics:
                    metrics['Pathway'] = pathway_name
                    pathway_records.append(metrics)

        # 格式化通路评估表
        df_pathways = pd.DataFrame(pathway_records)
        if not df_pathways.empty:
            cols = ['Pathway'] + [c for c in df_pathways.columns if c != 'Pathway']
            df_pathways = df_pathways[cols]

        return {
            'nodes': nodes,
            'tensor_continuous': dyn_results['tensors_continuous'],
            'tensor_binary': dyn_results['tensors_binary'],
            'topology_features_series': df_topology,
            'topology_temporal_entropy': df_temporal_entropy,
            'pathway_summaries': df_pathways
        }