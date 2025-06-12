import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, mannwhitneyu


class DynamitePlotter:
    """
    Create bar-and-error (“dynamite”) plots comparing groups split by sign of an effect
    and annotate Welch t-test and Mann–Whitney U statistics on the figure.
    """

    def __init__(self,
                 figsize: tuple[int, int] = (6, 6),
                 bar_palette: str = "Set2",
                 title_fontsize: int = 20,
                 label_fontsize: int = 20,
                 tick_fontsize: int = 16,
                 spine_width: int = 2):
        self.figsize = figsize
        self.bar_palette = bar_palette
        self.title_fontsize = title_fontsize
        self.label_fontsize = label_fontsize
        self.tick_fontsize = tick_fontsize
        self.spine_width = spine_width

    def plot(self,
             df: pd.DataFrame,
             group_col: str,
             y_col: str,
             dataset_name: str,
             out_dir: str | None = None,
             pos_label: str = "Positive Effect",
             neg_label: str = "Negative Effect",
             xlabel: str=None, 
             ylabel: str=None) -> None:
        """s
        Parameters
        ----------
        df          : DataFrame containing data.
        group_col   : Column with categorical labels.
        y_col       : Column to plot (Y-axis).
        dataset_name: Used for title and file naming.
        out_dir     : Directory to save SVG; if None, nothing is saved.
        """
        unique_groups = df[group_col].unique()
        if len(unique_groups) != 2:
            raise ValueError("Must be two groups in group_col. Detected: ", len(df[group_col].unique()) )        
        group1, group2 = unique_groups
        pos_group = df.loc[df[group_col] == group2, y_col].dropna()
        neg_group = df.loc[df[group_col] == group1, y_col].dropna()

        # Stats
        t_stat, t_p = ttest_ind(pos_group, neg_group, equal_var=False)
        u_stat, u_p = mannwhitneyu(pos_group, neg_group, alternative="two-sided")

        # Prep plotting data
        plot_df = pd.concat(
            [
                pos_group.to_frame(name=y_col).assign(Group=pos_label),
                neg_group.to_frame(name=y_col).assign(Group=neg_label),
            ],
            ignore_index=True,
        )
        means = plot_df.groupby("Group")[y_col].mean()
        sems = plot_df.groupby("Group")[y_col].sem()

        # Plot
        plt.figure(figsize=self.figsize)
        ax = sns.barplot(x=means.index, y=means.values, palette=self.bar_palette, ci=None)
        ax.errorbar(
            x=np.arange(len(means)),
            y=means.values,
            yerr=sems.values,
            fmt="none",
            ecolor="black",
            capsize=4,
            lw=1,
            zorder=3,
        )

        ax.set_title(dataset_name, fontsize=self.title_fontsize)
        ax.set_ylabel(ylabel if ylabel is not None else y_col, fontsize=self.label_fontsize)
        ax.set_xlabel(xlabel if xlabel is not None else group_col, fontsize=self.label_fontsize)

        ax.text(
            0.05,
            0.95,
            f"t = {t_stat:.2f}, p = {t_p:.2e}\nU = {u_stat:.2f}, p = {u_p:.2e}",
            transform=ax.transAxes,
            fontsize=self.tick_fontsize,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0, edgecolor="none"),
        )

        ax.tick_params(axis="both", labelsize=self.tick_fontsize)
        for spine in ax.spines.values():
            spine.set_linewidth(self.spine_width)

        plt.tight_layout()

        if out_dir:
            os.makedirs(os.path.join(out_dir, "dynamite_plots"), exist_ok=True)
            plt.savefig(
                os.path.join(out_dir, f"dynamite_plots/{dataset_name}_dynamite.svg"),
                bbox_inches="tight",
            )

        plt.show()
