import numpy as np
import pandas as pd
import logging
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class DynamicNetworkBuilder:
    def __init__(self, model, window_size: int, step_size: int):
        self.model = model
        self.window_size = window_size
        self.step_size = step_size

    def build_timeline(self, df_features: pd.DataFrame, show_progress: bool = True):
        n_samples = len(df_features)
        matrices_continuous, matrices_binary, valid_time_indices = [], [], []

        starts = range(0, n_samples - self.window_size + 1, self.step_size)
        iterator = tqdm(starts, desc="Dynamic Tracking") if show_progress else starts
        last_valid_A, last_valid_Bin = None, None

        for start in iterator:
            end = start + self.window_size
            df_window = df_features.iloc[start:end]

            try:
                A_pred, A_bin, _, current_nodes = self.model.fit_predict(df_window)
                full_A_pred = self._align_matrices(A_pred, current_nodes, df_features.columns.tolist())
                full_A_bin = self._align_matrices(A_bin, current_nodes, df_features.columns.tolist())

                matrices_continuous.append(full_A_pred)
                matrices_binary.append(full_A_bin)
                last_valid_A, last_valid_Bin = full_A_pred, full_A_bin
                valid_time_indices.append((start, end))

            except Exception as e:
                logger.warning(f"Window [{start}:{end}] failed. Using fallback. Error: {e}")
                if last_valid_A is not None:
                    matrices_continuous.append(last_valid_A)
                    matrices_binary.append(last_valid_Bin)
                else:
                    n_all = len(df_features.columns)
                    matrices_continuous.append(np.zeros((n_all, n_all)))
                    matrices_binary.append(np.zeros((n_all, n_all)))
                valid_time_indices.append((start, end))

        return {
            'nodes': df_features.columns.tolist(),
            'tensors_continuous': np.array(matrices_continuous),
            'tensors_binary': np.array(matrices_binary),
            'time_windows': valid_time_indices
        }

    def _align_matrices(self, small_mat, current_nodes, all_nodes):
        n_all = len(all_nodes)
        full_mat = np.zeros((n_all, n_all))
        for i, src in enumerate(current_nodes):
            for j, tgt in enumerate(current_nodes):
                full_mat[all_nodes.index(src), all_nodes.index(tgt)] = small_mat[i, j]
        return full_mat