import numpy as np
import plotly.graph_objects as go
from typing import Callable, Optional, List
class GeneralModelPlot:
    def __init__(self, model_predict: Callable, X_data: np.ndarray, Y_data: np.ndarray, feature_names: Optional[List[str]] = None):
        """
        Generalized plotter for any predictive model.

        Parameters
        ----------
        model_predict : Callable
            Prediction function accepting an array of shape (n_samples, n_features).
        X_data : np.ndarray
            Input features array (n_samples, n_features).
        Y_data : np.ndarray
            Outcome array (n_samples,).
        feature_names : list[str], optional
            Feature names for axis labels. If None, defaults to generic labels.
        """
        self.model_predict = model_predict
        self.X_data = X_data
        self.Y_data = Y_data
        self.feature_names = feature_names if feature_names else [f"X{i}" for i in range(X_data.shape[1])]

    def prepare_grid(self, num_slices: int, var_idx1: int, var_idx2: int, fixed_values: Optional[np.ndarray] = None):
        """Prepare a grid of predictor values for plotting."""
        x1_range = np.linspace(self.X_data[:, var_idx1].min(), self.X_data[:, var_idx1].max(), num_slices)
        x2_range = np.linspace(self.X_data[:, var_idx2].min(), self.X_data[:, var_idx2].max(), num_slices)
        x1_vals, x2_vals = np.meshgrid(x1_range, x2_range)

        if fixed_values is None:
            fixed_values = np.median(self.X_data, axis=0)

        X_grid = np.tile(fixed_values, (num_slices*num_slices, 1))
        X_grid[:, var_idx1] = x1_vals.ravel()
        X_grid[:, var_idx2] = x2_vals.ravel()

        return X_grid, x1_vals, x2_vals

    def _create_sliders(self, make_frame, fixed_values, var_idx1, var_idx2, slider_steps=20):
        sliders = []
        total_slider_space = 0.08
        slider_start = 0.01
        slider_step = total_slider_space / (self.X_data.shape[1] - 2)

        for idx, i in enumerate(range(self.X_data.shape[1])):
            if i not in [var_idx1, var_idx2]:
                min_val, max_val = self.X_data[:, i].min(), self.X_data[:, i].max()
                step_vals = np.linspace(min_val, max_val, slider_steps)

                sliders.append(dict(
                    active=slider_steps // 2,
                    currentvalue={"prefix": f"{self.feature_names[i]}: "},
                    pad={"t": 0},
                    y=slider_start + idx * slider_step,
                    len=0.9,
                    steps=[dict(
                        method='restyle',
                        args=[{"z": [make_frame(np.array([val if j == i else fixed_values[j] for j in range(len(fixed_values))]))[0]]}],
                        label=f"{val:.2f}"
                    ) for val in step_vals]
                ))
        return sliders

    def plot_with_sliders(self, num_slices=100, var_idx1=0, var_idx2=1, plot_residuals=True, limit_z=False):
        """Create an interactive 3D surface plot with sliders for additional variables."""
        assert self.X_data.shape[1] > 2, "Need more than two variables for sliders."

        fixed_values = np.median(self.X_data, axis=0)

        def make_frame(slider_vals):
            X_grid, x1_vals, x2_vals = self.prepare_grid(num_slices, var_idx1, var_idx2, fixed_values=slider_vals)
            Y_pred = self.model_predict(X_grid).reshape(num_slices, num_slices)
            return Y_pred, x1_vals, x2_vals

        y_pred_initial, x1_vals, x2_vals = make_frame(fixed_values)

        fig = go.Figure()
        fig.add_trace(go.Surface(z=y_pred_initial, x=x1_vals, y=x2_vals, colorscale='Greys', opacity=0.7))

        if plot_residuals:
            fig.add_trace(go.Scatter3d(
                x=self.X_data[:, var_idx1],
                y=self.X_data[:, var_idx2],
                z=self.Y_data,
                mode='markers',
                marker=dict(color='red', size=5),
                name='Actual Data'
            ))

            Y_pred_points = self.model_predict(self.X_data)
            for xi1, xi2, yi, yi_pred in zip(self.X_data[:, var_idx1], self.X_data[:, var_idx2], self.Y_data, Y_pred_points):
                fig.add_trace(go.Scatter3d(
                    x=[xi1, xi1],
                    y=[xi2, xi2],
                    z=[yi, yi_pred],
                    mode='lines',
                    line=dict(color='blue', dash='dash'),
                    showlegend=False
                ))

        sliders = self._create_sliders(make_frame, fixed_values, var_idx1, var_idx2)

        if limit_z:
            range = [4*np.min(self.Y_data), 4*np.max(self.Y_data)]
        else:
            range = [yi, yi_pred]
        fig.update_layout(
            scene=dict(
            xaxis_title=self.feature_names[var_idx1],
            yaxis_title=self.feature_names[var_idx2],
            zaxis_title="Outcome",
            aspectmode='cube',
            zaxis=dict(range=range)
            ),
            sliders=sliders,
            width=1200,
            height=800 + len(sliders) * 40,
            margin=dict(l=20, r=20, t=20, b=40)
        )

        return fig
