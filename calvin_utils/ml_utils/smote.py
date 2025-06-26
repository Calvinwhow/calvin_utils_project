from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np

class SmoteOversampler:
    """
    Balance classes in a pandas DataFrame with SMOTE while tagging
    synthetic rows.

    Parameters
    ----------
    outcome_col : str
        Name of the target column.
    smote_kwargs : dict, optional
        Keyword arguments forwarded to `imblearn.over_sampling.SMOTE`.
        Example: {'sampling_strategy': 'auto', 'k_neighbors': 5, 'random_state': 0}
    """

    def __init__(self, outcome_col: str, **smote_kwargs):
        self.outcome_col = outcome_col
        self.smote_kwargs = smote_kwargs
        self.smote_ = SMOTE(**smote_kwargs)

    def fit_resample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame with balanced classes and an `is_synthetic` flag."""
        if self.outcome_col not in df.columns:
            raise KeyError(f"Outcome column '{self.outcome_col}' not found in DataFrame.")

        X = df.drop(columns=[self.outcome_col])
        y = df[self.outcome_col]

        # SMOTE generates new samples and appends them after the originals
        X_res, y_res = self.smote_.fit_resample(X, y)

        n_original = len(df)
        n_total = len(X_res)
        is_synthetic = np.concatenate(
            [np.zeros(n_original, dtype=bool), np.ones(n_total - n_original, dtype=bool)]
        )

        df_resampled = pd.DataFrame(X_res, columns=X.columns)
        df_resampled[self.outcome_col] = y_res
        df_resampled["is_synthetic"] = is_synthetic

        return df_resampled