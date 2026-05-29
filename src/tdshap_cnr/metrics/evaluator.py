import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms import community as nx_comm
import logging

logger = logging.getLogger(__name__)


class NetworkAnalyzer:
    def __init__(self, eval_config):
        self.cfg = eval_config

    def extract_temporal_topology(self, tensors_binary: np.ndarray, tensors_continuous: np.ndarray, nodes: list):
        n_windows, n_nodes, _ = tensors_binary.shape
        records = []

        for t in range(n_windows):
            A_bin, A_cont = tensors_binary[t], tensors_continuous[t]
            row = {'Window_Idx': t}
            if any(m in self.cfg.topology_metrics for m in ['in_degree', 'out_degree', 'net_flow']):
                in_deg = np.sum(A_cont, axis=0)
                out_deg = np.sum(A_cont, axis=1)
                for i, node in enumerate(nodes):
                    if 'in_degree' in self.cfg.topology_metrics: row[f'{node}_InDegree'] = in_deg[i]
                    if 'out_degree' in self.cfg.topology_metrics: row[f'{node}_OutDegree'] = out_deg[i]
                    if 'net_flow' in self.cfg.topology_metrics: row[f'{node}_NetFlow'] = out_deg[i] - in_deg[i]
            if 'spectral_radius' in self.cfg.topology_metrics:
                try:
                    row['Spectral_Radius'] = max(abs(np.linalg.eigvals(A_cont)))
                except Exception:
                    row['Spectral_Radius'] = 0.0

            G_dir = nx.from_numpy_array(A_bin, create_using=nx.DiGraph)
            G_und = G_dir.to_undirected()

            if 'density' in self.cfg.topology_metrics:
                row['Density'] = np.sum(A_bin) / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0
            if 'clustering' in self.cfg.topology_metrics:
                row['Clustering'] = nx.average_clustering(G_dir)
            if 'global_efficiency' in self.cfg.topology_metrics:
                try:
                    row['Global_Efficiency'] = nx.global_efficiency(G_und)
                except Exception:
                    row['Global_Efficiency'] = 0.0

            communities = None
            if any(m in self.cfg.topology_metrics for m in ['modularity', 'participation_coefficient']):
                if G_und.number_of_edges() > 0:
                    communities = list(nx_comm.greedy_modularity_communities(G_und))
                    if 'modularity' in self.cfg.topology_metrics:
                        row['Modularity'] = nx_comm.modularity(G_und, communities)
                else:
                    row['Modularity'] = 0.0
                if 'participation_coefficient' in self.cfg.topology_metrics:
                    row['Max_Participation'] = self._calc_max_participation(G_und, communities)

            if 'rich_club' in self.cfg.topology_metrics:
                if G_und.number_of_edges() > 0:
                    rc = nx.rich_club_coefficient(G_und, normalized=False)
                    row['Rich_Club'] = max(rc.values()) if rc else 0.0
                else:
                    row['Rich_Club'] = 0.0

            records.append(row)

        df_topology = pd.DataFrame(records)

        df_temporal_entropy = pd.DataFrame()
        if self.cfg.calculate_temporal_entropy and not df_topology.empty:
            df_temporal_entropy = self._calculate_temporal_entropy(df_topology)

        return df_topology, df_temporal_entropy

    def _calc_max_participation(self, G, communities) -> float:
        if not communities or len(communities) < 2: return 0.0
        node_to_comm = {node: i for i, comm in enumerate(communities) for node in comm}
        max_P = 0.0
        for node in G.nodes():
            deg = G.degree(node)
            if deg == 0: continue
            comm_degree = {}
            for neighbor in G.neighbors(node):
                c = node_to_comm.get(neighbor, -1)
                if c != -1: comm_degree[c] = comm_degree.get(c, 0) + 1
            P_i = 1.0 - sum((k_is / deg) ** 2 for k_is in comm_degree.values())
            if P_i > max_P: max_P = P_i
        return max_P

    def _calculate_temporal_entropy(self, df_topology: pd.DataFrame) -> pd.DataFrame:
        entropy_records = {}
        for col in df_topology.columns:
            if col == 'Window_Idx': continue
            ts = df_topology[col].dropna().values
            if len(ts) < 2 or np.std(ts) < 1e-6:
                entropy_records[col + "_Temporal_Entropy"] = 0.0
            else:
                counts, _ = np.histogram(ts, bins='fd')
                probs = counts[counts > 0] / len(ts)
                entropy_records[col + "_Temporal_Entropy"] = -np.sum(probs * np.log2(probs))
        return pd.DataFrame([entropy_records])

    def summarize_causal_pathway(self, seq: np.ndarray) -> dict:

        if len(seq) == 0: return {}

        k = max(1, int(self.cfg.top_k_ratio * len(seq)))
        top_k_mean = np.mean(np.sort(seq)[-k:])

        return {
            'Mean_Strength': np.mean(seq),  # 全局平均强度
            'TopK_Peak_Strength': top_k_mean,  # Top-K 爆发上限
            'Strength_Std': np.std(seq),  # 反映演化波动率
            'Max_Strength': np.max(seq),  # 绝对极值
            'Active_Windows_Ratio': np.mean(seq > 0)  # 绝对点亮频率
        }