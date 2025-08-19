import os 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

class ResampleVisualizer:
    def __init__(self, stat_array_1, stat_array_2, model1_name="X", model2_name="Y", stat="R²", out_dir=None):
        """
        Initializes the resampling plot object for visualizing paired delta statistics.
        This class is used to plot the paired delta between a statistic observed for 
        each resampling, and can visualize either bootstraps or permutations.
        Args:
            stat_array_1 (float): The statistical values 
            stat_array_2 (float): The R-squared value for the second region of interest (ROI).
            model1_name (str, optional): Name of the first model. Defaults to "X".
            model2_name (str, optional): Name of the second model. Defaults to "Y".
            stat (str, optional): The statistic name to be plotted. Defaults to "R²".
            out_dir (str, optional): Where to save. 
        """
        self.stat_array_1 = stat_array_1
        self.stat_array_2 = stat_array_2 
        self.delta_r2 = self._stat_array_1 - self._stat_array_2
        self.model1_name = model1_name
        self.model2_name = model2_name
        self.out_dir = out_dir
        self.stat = stat
        self.BLACK = '#211D1E'
        self.GREY = '#8E8E8E'
        self.colors = self.compute_line_colors_abrupt()
        self.prepare_out_dir()

    @property
    def stat_array_1(self):
        return self._stat_array_1

    @stat_array_1.setter
    def stat_array_1(self, value):
        value = np.array(value)  # Convert to numpy array
        if hasattr(self, '_stat_array_2') and len(value) != len(self._stat_array_2):
            raise ValueError("stat_array_1 and stat_array_2 must have the same length.")
        self._stat_array_1 = value

    @property
    def stat_array_2(self):
        return self._stat_array_2

    @stat_array_2.setter
    def stat_array_2(self, value):
        value = np.array(value)
        if hasattr(self, '_stat_array_1') and len(value) != len(self._stat_array_1):
            raise ValueError("stat_array_1 and stat_array_2 must have the same length.")
        self._stat_array_2 = value

    def prepare_out_dir(self):
        if self.out_dir is not None:
            os.makedirs(self.out_dir, exist_ok=True)
            
    def get_black_to_grey_colormap_abrupt(self):
        return LinearSegmentedColormap.from_list("black_grey_abrupt", [
            (0.0, self.GREY),
            (0.5, self.GREY),
            (0.50001, self.BLACK),
            (1.0, self.BLACK)
        ])

    def compute_line_colors_abrupt(self):
        slope_diff = self.stat_array_2 - self.stat_array_1
        max_abs = np.max(np.abs(slope_diff))
        normed = (slope_diff + max_abs) / (2 * max_abs)  # scale to [0, 1] with zero at 0.5
        cmap = self.get_black_to_grey_colormap_abrupt()
        return cmap(normed)

    def compute_stats(self, alpha=0.05, analytic=False):
        differences = np.array(self.delta_r2)
        p_val = np.mean(differences > 0)
        if analytic:
            print("Confidence Interval Method: Analytic Formula")
            mean_delta = np.mean(differences)
            std_error = np.std(differences, ddof=1) / np.sqrt(len(differences))
            z_score = abs(np.percentile(np.random.normal(0, 1, 100000), 100 * (1 - alpha / 2)))
            ci_lower = mean_delta - z_score * std_error
            ci_upper = mean_delta + z_score * std_error
        else:
            print("Confidence Interval Method: Empiric Percentile")
            ci_lower = np.percentile(differences, 100 * (alpha / 2))
            ci_upper = np.percentile(differences, 100 * (1 - alpha / 2))
        mean_delta = np.mean(differences)
        return {"p": p_val, "ci_lower": ci_lower, "ci_upper": ci_upper, "mean_delta": mean_delta}

    def plot_paired_slopes(self, ax):
        self.plot_resample_dots(ax, self.stat_array_1, 1, self.GREY)
        self.plot_resample_dots(ax, self.stat_array_2, 0, self.BLACK)
        for i in range(len(self.stat_array_1)):
            ax.plot([0, 1], [self.stat_array_2[i], self.stat_array_1[i]], color=self.colors[i], linewidth=2, alpha=0.8)
        self.setup_slope_subplot(ax)

    def plot_resample_dots(self, ax, y_vals, x_coord, color, size=150):
        ax.scatter(np.full(len(y_vals), x_coord), y_vals, color=color,
                   edgecolors="white", linewidth=1.2, alpha=0.9, s=size, marker='o', zorder=3)

    def setup_slope_subplot(self, ax):
        ax.set_xticks([0, 1])
        ax.set_xticklabels([self.model2_name, self.model1_name])
        ax.set_title(f"Paired {self.stat}", fontsize=20)
        ax.set_ylabel(f"{self.stat}", fontsize=20)
        ax.set_xlim(-0.5, 1.5)
        ax.tick_params(labelsize=16)
        sns.despine(ax=ax)

    def setup_histogram(self, ax, abs_limit, ymax_padded):
        sns.histplot(self.delta_r2, bins=40, kde=True, color=self.BLACK, ax=ax, edgecolor='white')
        ax.axvline(0, color=self.GREY, linestyle='--')
        ax.set_xlim([-abs_limit, abs_limit])
        ax.set_ylim([0, ymax_padded])
        ax.set_title(f"Δ{self.stat} Distribution", fontsize=20)
        ax.set_xlabel(f"Additional {self.stat} (Δ{self.stat})", fontsize=20)
        ax.set_ylabel("Number of Resamples", fontsize=20)
        ax.tick_params(labelsize=16)
        ax.text(abs_limit * -0.95, ymax_padded * 0.95, f"Favours {self.model2_name}", ha='left', va='top', fontsize=16, color=self.BLACK)
        ax.text(abs_limit * 0.95, ymax_padded * 0.95, f"Favours {self.model1_name}", ha='right', va='top', fontsize=16, color=self.BLACK)
        sns.despine(ax=ax)

    def annotate_histogram_ci(self, ax, mean_delta, ci_lower, ci_upper, p, abs_limit, ymax_padded):
        left_density = np.sum(self.delta_r2 < 0)
        right_density = np.sum(self.delta_r2 >= 0)
        x_text = -0.9 * abs_limit if left_density < right_density else 0.9 * abs_limit
        ha_text = 'left' if left_density < right_density else 'right'
        p_val = 1 - p if p > 0.5 else p
        ci_text = f"Δ{self.stat} = {mean_delta:.4f}\nCI: [{ci_lower:.4f}, {ci_upper:.4f}]\np = {p_val:.4f}"
        ax.text(x_text, ymax_padded * 0.9, ci_text, ha=ha_text, va='top', fontsize=14, color=self.BLACK)

    def draw(self, name='delta_plot.svg', verbose=True):
        hist_counts, _ = np.histogram(self.delta_r2, bins=40)
        ymax_padded = np.max(hist_counts) * 1.10
        abs_limit = np.max(np.abs(self.delta_r2))

        stats = self.compute_stats()
        fig, axes = plt.subplots(1, 2, figsize=(10, 6), gridspec_kw={'width_ratios': [0.7, 1.3]})

        self.plot_paired_slopes(axes[0])
        # Increase the width of the axis lines
        for spine in axes[0].spines.values():
            spine.set_linewidth(2)
        for spine in axes[1].spines.values():
            spine.set_linewidth(2)

        self.setup_histogram(axes[1], abs_limit, ymax_padded)
        self.annotate_histogram_ci(
            axes[1],
            stats['mean_delta'],
            stats['ci_lower'],
            stats['ci_upper'],
            stats['p'],
            abs_limit,
            ymax_padded
        )
        # Save the figure
        if self.out_dir is not None:
            save_path = os.path.join(self.out_dir, name)
            fig.savefig(save_path, format='svg')
        plt.tight_layout()
        if verbose: plt.show()