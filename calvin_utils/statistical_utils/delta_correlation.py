# Redefining the class and its methods
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
import seaborn as sns
from tqdm import tqdm
import pandas as pd

class ScatterWithConfidence:
    def __init__(self, data_df):
        self.data_df = data_df

    @staticmethod
    def compute_analytic_confidence_interval(x, y, x_vals):
        slope, intercept = np.polyfit(x, y, 1)
        y_fit = slope * x_vals + intercept
        
        residuals = y - (slope * x + intercept)
        stderr = np.sqrt(np.sum(residuals**2) / (len(y) - 2))
        
        t_value = t.ppf(0.975, df=len(x)-2)
        ci = t_value * stderr * np.sqrt(1/len(x) + (x_vals - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
        
        upper_bound = y_fit + ci
        lower_bound = y_fit - ci
        
        return y_fit, lower_bound, upper_bound

    def permute_data_and_difference_in_pearson_r(self, x_one, x_two, split_by, split_value, n_permutations=1000, permute_columns=[]):
        original_diff = self.data_df[self.data_df[split_by] < split_value][x_one].corr(self.data_df[self.data_df[split_by] < split_value][x_two]) - \
                       self.data_df[self.data_df[split_by] >= split_value][x_one].corr(self.data_df[self.data_df[split_by] >= split_value][x_two])

        permuted_diffs = []

        for _ in range(n_permutations):
            permuted_df = self.data_df.copy()
            for column in permute_columns:
                permuted_df[column] = np.random.permutation(permuted_df[column].values)
            
            diff = permuted_df[permuted_df[split_by] < split_value][x_one].corr(permuted_df[permuted_df[split_by] < split_value][x_two]) - \
                   permuted_df[permuted_df[split_by] >= split_value][x_one].corr(permuted_df[permuted_df[split_by] >= split_value][x_two])
            
            permuted_diffs.append(diff)

        p_value = np.mean([diff <= original_diff for diff in permuted_diffs])
        return original_diff, p_value

    def plot_with_analytic_ci_manual_pvalue(self, x_one, x_two, 
                                            split_by, split_value, 
                                            x_label='X1', y_label='X2', 
                                            upper_split_legend='Above Split', lower_split_legend='Below Split',
                                            alpha=0.3, manual_p_value=None, permute_column=None, 
                                            save=False, out_dir=None,
                                            colour1='red', colour2='blue'):
        fig, ax = plt.subplots(figsize=(4.75, 4))
        
        group1 = self.data_df[self.data_df[split_by] < split_value]
        group2 = self.data_df[self.data_df[split_by] >= split_value]
        print(len(group1), len(group2))

        ax.scatter(group1[x_one], group1[x_two], color=colour1, label=lower_split_legend, s=40, alpha=alpha, marker='o')
        ax.scatter(group2[x_one], group2[x_two], color=colour2, label=upper_split_legend, s=40, alpha=alpha, marker='o')
        
        x_vals = np.linspace(self.data_df[x_one].min(), self.data_df[x_one].max(), 400)
        
        for group, color in [(group1, colour1), (group2, colour2)]:
            y_fit, lower_bound, upper_bound = self.compute_analytic_confidence_interval(group[x_one], group[x_two], x_vals)
            ax.plot(x_vals, y_fit, color=color)
            ax.fill_between(x_vals, lower_bound, upper_bound, color=color, alpha=alpha/4)
        
        if manual_p_value is None:
            if permute_column:
                rho, manual_p_value = self.permute_data_and_difference_in_pearson_r(x_one, x_two, split_by, split_value, n_permutations=10000, permute_columns=[permute_column])
            else:
                rho, manual_p_value = self.permute_data_and_difference_in_pearson_r(x_one, x_two, split_by, split_value, n_permutations=10000, permute_columns=[x_one, x_two, split_by])
        
        ax.set_title(f"\u0394 r = {rho:.2f} , p = {manual_p_value:.4f}")
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        ax.legend(loc='best', frameon=False)
        ax.grid(False)
        sns.despine(ax=ax)
        
        if save and out_dir is not None:
            plt.savefig(f"{out_dir}/delta_corr_scatterplot.png", bbox_inches='tight')
            plt.savefig(f"{out_dir}/delta_corr_scatterplot.svg", bbox_inches='tight')
            print(f'Saved to {out_dir}/delta_corr_scatterplot.svg')
        return fig
    
class DeltaCorrelation(ScatterWithConfidence):
    def __init__(self, data_df):
        super().__init__(data_df)

    def plot_histogram_of_delta_r(self, x_one, x_two, split_by, split_value, n_permutations=1000, 
                                permute_columns=[], bins=50, one_tail=False, color_palette='dark',
                                out_dir=None):
        # Generate the empirical distribution of delta_r
        delta_rs = []
        for _ in range(n_permutations):
            permuted_df = self.data_df.copy()
            for column in permute_columns:
                permuted_df[column] = np.random.permutation(permuted_df[column].values)

            delta_r = permuted_df[permuted_df[split_by] < split_value][x_one].corr(permuted_df[permuted_df[split_by] < split_value][x_two]) - \
                    permuted_df[permuted_df[split_by] >= split_value][x_one].corr(permuted_df[permuted_df[split_by] >= split_value][x_two])
            delta_rs.append(delta_r)

        # Calculate the observed delta_r
        observed_delta_r, _ = self.permute_data_and_difference_in_pearson_r(x_one, x_two, split_by, split_value, permute_columns=permute_columns)

        if one_tail:
            observed_delta_r = np.abs(observed_delta_r)
            delta_rs = np.abs(delta_rs)

        # Calculate p-value
        if one_tail:
            p_value = np.mean([delta_r >= observed_delta_r for delta_r in delta_rs])
        else:
            p_value = np.mean([delta_r <= observed_delta_r for delta_r in delta_rs])

        # Generate the displot (KDE + Histogram) using Seaborn
        sns.set_palette(color_palette)
        current_palette = sns.color_palette(color_palette)
        chosen_color = current_palette[4]
        plt.figure(figsize=(4.75, 4))
        g = sns.displot(delta_rs, kde=True, bins=bins, label="Empirical $\\Delta r$ Distribution", element="step", color='blue', alpha=.6)
        plt.axvline(x=observed_delta_r, color='red', linestyle='-', linewidth=1.5, label=f"Observed $\\Delta r$", alpha=0.6)
        plt.title(f"$\\Delta r$ = {observed_delta_r}, p = {p_value}")
        plt.xlabel("$\\Delta r$")
        plt.ylabel("Count")
        plt.legend()
        
        fig = g.fig
        if out_dir is not None:
            fig.savefig(f"{out_dir}/delta_corr_perm_dist.png", bbox_inches='tight')
            fig.savefig(f"{out_dir}/delta_corr_perm_dist.svg", bbox_inches='tight')
            print(f'Saved to {out_dir}/hist_kde.svg')
            
    def plot_group_correlations(self, x_one, x_two, split_by, split_value,
                                n_bootstraps=1000, permute_columns=None,
                                transpose=True, xlim=(-1, 1), out_dir=None):
        if permute_columns is None:
            permute_columns = [x_one, x_two, split_by]

        g1 = self.data_df[self.data_df[split_by] < split_value]
        g2 = self.data_df[self.data_df[split_by] >= split_value]

        corr_g1, corr_g2 = [], []
        for _ in range(n_bootstraps):
            boot_g1 = g1.sample(len(g1), replace=True)
            boot_g2 = g2.sample(len(g2), replace=True)
            corr_g1.append(boot_g1[x_one].corr(boot_g1[x_two]))
            corr_g2.append(boot_g2[x_one].corr(boot_g2[x_two]))


        df_plot = pd.DataFrame({
            'Correlation': corr_g1 + corr_g2,
            'Group': [f'<{split_value}'] * len(corr_g1) +
                     [f'>={split_value}'] * len(corr_g2)
        })

        plt.figure(figsize=(4.75, 4))
        orient = 'h' if transpose else 'v'

        # ------- key change: NO hue="Correlation" -------
        if transpose:
            sns.boxplot(data=df_plot, x='Correlation', y='Group',
                        orient='h', palette='coolwarm')
            plt.xlim(*xlim)
            plt.xlabel('Pearson r')
            plt.ylabel('Group')
        else:
            sns.boxplot(data=df_plot, x='Group', y='Correlation',
                        orient='v', palette='coolwarm')
            plt.ylim(*xlim)
            plt.ylabel('Pearson r')
            plt.xlabel('Group')

        # overlay observed correlations
        obs_g1 = self.data_df[self.data_df[split_by] < split_value][x_one] \
                     .corr(self.data_df[self.data_df[split_by] < split_value][x_two])
        obs_g2 = self.data_df[self.data_df[split_by] >= split_value][x_one] \
                     .corr(self.data_df[self.data_df[split_by] >= split_value][x_two])

        if transpose:
            plt.scatter([obs_g1, obs_g2], [0, 1], c='k', s=60, zorder=5, label='Observed r')
        else:
            plt.scatter([0, 1], [obs_g1, obs_g2], c='k', s=60, zorder=5, label='Observed r')

        plt.legend(frameon=False)
        sns.despine()

        if out_dir:
            plt.savefig(out_dir + '/delta_corr_boxplot.png', bbox_inches='tight')
            plt.savefig(out_dir + '/delta_corr_boxplot.svg', bbox_inches='tight')
        return plt.gcf()