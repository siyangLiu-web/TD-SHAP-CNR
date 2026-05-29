import logging
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from tdshap_cnr.config import ModelConfig, PruneConfig

logger = logging.getLogger(__name__)


class TDShapCausalModel:
    def __init__(self, model_cfg: ModelConfig, prune_cfg: PruneConfig):
        self.m_cfg = model_cfg
        self.p_cfg = prune_cfg
        self.scaler = RobustScaler() 

    def _validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        df.fillna(0, inplace=True)
        valid_cols = [c for c in df.columns if np.var(df[c]) > 1e-12]
        return df[valid_cols]

    def fit_predict(self, df_features: pd.DataFrame):
        df_clean = self._validate_and_clean(df_features)
        nodes = df_clean.columns.tolist()
        n_nodes = len(nodes)

        if n_nodes < 2:
            raise ValueError("有效节点数不足，无法构建网络。")

        A_pred = np.zeros((n_nodes, n_nodes))
        norm_data = self.scaler.fit_transform(df_clean)
        df_norm = pd.DataFrame(norm_data, columns=nodes)

        for i, target_node in enumerate(nodes):
            drivers = [n for n in nodes if n != target_node]
            X_matrix, Y_target = self._build_delay_embedding(df_norm, target_node, drivers)

            if len(X_matrix) < 5:
                return np.zeros((n_nodes, n_nodes)), np.zeros((n_nodes, n_nodes)), 0.0, nodes

            rf = RandomForestRegressor(
                n_estimators=self.m_cfg.rf_n_estimators,
                max_depth=self.m_cfg.rf_max_depth,
                max_features=self.m_cfg.rf_max_features,
                n_jobs=self.m_cfg.rf_n_jobs,
                random_state=self.m_cfg.rf_random_state
            ).fit(X_matrix, Y_target)

            explainer = shap.TreeExplainer(rf)
            bg_size = min(self.m_cfg.shap_background_samples, X_matrix.shape[0])
            bg_data = shap.sample(X_matrix, bg_size, random_state=self.m_cfg.rf_random_state)
            shap_values = explainer.shap_values(bg_data)
            shap_importance = np.mean(np.abs(shap_values), axis=0)

            idx = self.m_cfg.max_lag
            for driver_node in drivers:
                driver_idx = nodes.index(driver_node)
                A_pred[driver_idx, i] = np.sum(shap_importance[idx: idx + self.m_cfg.max_lag])
                idx += self.m_cfg.max_lag

        A_binary, threshold = self._prune_network(A_pred)
        return A_pred, A_binary, threshold, nodes

    def _build_delay_embedding(self, df: pd.DataFrame, target: str, drivers: list):
        N = len(df)
        max_l = self.m_cfg.max_lag
        X_matrix = [df[target].values[max_l - lag: N - lag] for lag in range(1, max_l + 1)]
        for driver in drivers:
            sig = df[driver].values
            X_matrix.extend([sig[max_l - lag: N - lag] for lag in range(1, max_l + 1)])
        return np.column_stack(X_matrix), df[target].values[max_l:]

    def _prune_network(self, A_pred: np.ndarray):
        n_nodes = A_pred.shape[0]
        np.fill_diagonal(A_pred, 0)
        edge_weights = A_pred[~np.eye(n_nodes, dtype=bool)].reshape(-1, 1)

        if self.p_cfg.method == 'kmeans':
            if np.std(edge_weights) < 1e-6:
                return np.zeros_like(A_pred), 0.0
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(edge_weights)
            centers = kmeans.cluster_centers_.flatten()
            sig_weights = edge_weights[kmeans.labels_ == np.argmax(centers)]
            thresh = np.min(sig_weights) / self.p_cfg.kmeans_tolerance if len(sig_weights) > 0 else np.median(
                edge_weights)
        elif self.p_cfg.method == 'percentile':
            thresh = np.percentile(edge_weights, self.p_cfg.percentile_threshold)
        else:
            thresh = 0.0

        A_binary = (A_pred >= thresh).astype(int)
        np.fill_diagonal(A_binary, 0)
        return A_binary, thresh