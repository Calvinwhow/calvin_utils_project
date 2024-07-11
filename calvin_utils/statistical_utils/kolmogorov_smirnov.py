import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from itertools import combinations
import math

def ks_test_and_plot_cdf(sample1, sample2, outdir):
    # Calculate the Kolmogorov-Smirnov statistic and p-value
    ks_stat, p_value = ks_2samp(sample1, sample2)
    print(f"KS Statistic: {ks_stat}")
    print(f"P-value: {p_value}")

    # Sort the samples for CDF plotting
    sorted_sample1 = np.sort(sample1)
    sorted_sample2 = np.sort(sample2)
    cdf1 = np.arange(1, len(sorted_sample1) + 1) / len(sorted_sample1)
    cdf2 = np.arange(1, len(sorted_sample2) + 1) / len(sorted_sample2)

    # Plotting the CDFs
    plt.figure(figsize=(10, 6))
    plt.step(sorted_sample1, cdf1, label='CDF Sample 1', where='post')
    plt.step(sorted_sample2, cdf2, label='CDF Sample 2', where='post')
    plt.title('CDF Comparison of Two Samples')
    plt.xlabel('Sample Values')
    plt.ylabel('CDF')
    plt.legend()
    sns.despine()
    plt.grid(False)
    plt.savefig(outdir + '/KS.svg')
    plt.show()

def pivot_dataframe(df, concat_col, category_col):
    # Create a new DataFrame where each unique category becomes a column
    # and the values from concat_col are listed under these category columns
    # First, ensure that the index is reset for the DataFrame to avoid issues during pivoting
    df.reset_index(drop=True, inplace=True)

    # Create a new DataFrame where each row will have the category as a column and the corresponding values
    # from concat_col under that category
    pivoted_df = df.pivot(columns=category_col, values=concat_col)
    
    return pivoted_df

class CalvinKolmogorovSmirnov:
    
    def __init__(self, xlim=None, xlabel='Sample Values', ylabel='CDF'):
        """
        Initialize the DataAnalysis class with optional x-axis limits for plots.
        
        Parameters:
        xlim (tuple): Optional. A tuple specifying the x-axis limits (min, max) for the plots.
        """
        self.xlim = xlim
        self.xlabel=xlabel
    
    def ks_test(self, sample1, sample2):
        """
        Perform the Kolmogorov-Smirnov test for two samples.
        
        Parameters:
        sample1 (array-like): First sample data.
        sample2 (array-like): Second sample data.
        
        Returns:
        tuple: KS statistic and p-value.
        """
        ks_stat, p_value = ks_2samp(sample1, sample2)
        return ks_stat, p_value
    
    def plot_cdf(self, pivoted_df, ax, title, ks_results, bonferroni_corr=True):
        """
        Plot the Cumulative Distribution Functions (CDFs) for columns in the pivoted DataFrame.
        
        Parameters:
        pivoted_df (DataFrame): Pivoted DataFrame with samples as columns.
        ax (matplotlib.axes.Axes): Axes object to plot on.
        title (str): Title of the plot.
        ks_results (dict): Dictionary containing KS test results for each pair of columns.
        bonferroni_corr (bool) : Performs bonferroni correction for multiple comparisons.
        """
        for column in pivoted_df.columns:
            sorted_sample = np.sort(pivoted_df[column].dropna())
            cdf = np.arange(1, len(sorted_sample) + 1) / len(sorted_sample)
            ax.step(sorted_sample, cdf, label=f'CDF {column}', where='post')
        
        ax.set_title(title,fontsize='x-large')
        ax.set_xlabel(self.xlabel, fontsize='x-large')
        ax.set_ylabel('CDF', fontsize='x-large')
        if self.xlim is not None:
            ax.set_xlim(self.xlim[0], self.xlim[1])
        
        ax.legend(loc='upper left')
        sns.despine()
        ax.grid(False)
        
        
        # Prepare Bonferroni correction
        if bonferroni_corr:
            n_comparisons = len(ks_results.keys())
            corrected_ks_results = {}
            for pair, result in ks_results.items():
                if result[1] < 0.05: # Perform bonferroni adjustment. 
                    pval_corr = result[1]*n_comparisons
                    print(f"{pair} has been Bonferroni corrected by factor {n_comparisons} from {result[1]} to {pval_corr}")
                else: # No need to correct. Is already insignificant. 
                    pval_corr = result[1]
                corrected_ks_results[pair] = (result[0], pval_corr)
            text = "\n".join([f"{pair[0]}-{pair[1]}: p = {result[1]:.3e}" for pair, result in corrected_ks_results.items()])
        else:
            text = "\n".join([f"{pair[0]}-{pair[1]}: p = {result[1]:.3e}" for pair, result in ks_results.items()])
        # Add KS test results at the bottom right-hand side
        ax.text(0.95, 0.05, text, ha='right', va='bottom', transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.0))

    def ks_test_and_plot_cdf(self, df, concat_col, category_cols, ax, verbose=False):
        """
        Perform KS test and plot CDF for a given concatenation column.
        
        Parameters:
        df (DataFrame): Original DataFrame.
        concat_col (str): Column name to be concatenated for pivoting.
        category_cols (list): List of columns to categorize by.
        ax (matplotlib.axes.Axes): Axes object to plot on.
        verbose (bool): Optional. If True, print detailed KS test results.
        
        Returns:
        dict: Dictionary of KS test results for each pair of columns in the pivoted DataFrame.
        """
        pivoted_df = self.pivot_dataframe(df, concat_col, category_cols)
        
        ks_results = {}
        
        for col1, col2 in combinations(pivoted_df.columns, 2):
            sample1 = pivoted_df[col1].dropna()
            sample2 = pivoted_df[col2].dropna()
            ks_stat, p_value = self.ks_test(sample1, sample2)
            ks_results[(col1, col2)] = (ks_stat, p_value)
            if verbose:
                print(f"KS Statistic for {col1} vs {col2}: {ks_stat}")
                print(f"P-value for {col1} vs {col2}: {p_value}")

        self.plot_cdf(pivoted_df, ax, f'CDF Comparison for {concat_col}', ks_results)
        
        return ks_results

    def pivot_dataframe(self, df, concat_col, category_cols):
        """
        Pivot the DataFrame based on the concatenation column and category columns.
        
        Parameters:
        df (DataFrame): Original DataFrame.
        concat_col (str): Column name to be concatenated for pivoting.
        category_cols (list): List of columns to categorize by.
        
        Returns:
        DataFrame: Pivoted DataFrame.
        """
        df.reset_index(drop=True, inplace=True)
        pivoted_df = df.pivot_table(index=df.index, columns=category_cols, values=concat_col)
        
        return pivoted_df
    
    def orchestrate_ks_tests(self, df, concat_cols, category_cols, outdir, filetype='svg'):
        """
        Orchestrate multiple iterations of KS tests and plot CDFs for a list of concatenation columns.
        
        Parameters:
        df (DataFrame): Original DataFrame.
        concat_cols (list): List of concatenation columns.
        category_cols (list): List of columns to categorize by.
        outdir (str): Output directory for saving plots.
        filetype (str): File type for saving plots (e.g., 'svg', 'png').
        
        Returns:
        dict: Dictionary containing KS test results for each concatenation column.
        """
        n = len(concat_cols)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*5))

        # Handle the case where there is only one plot
        if n == 1:
            axes = np.array([axes])

        axes = axes.flatten()

        all_ks_results = {}
        
        for i, concat_col in enumerate(concat_cols):
            ks_results = self.ks_test_and_plot_cdf(df, concat_col, category_cols, axes[i])
            all_ks_results[concat_col] = ks_results
        
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.savefig(f"{outdir}/CDF_plots.{filetype}")
        plt.show()
        
        return all_ks_results


class CalvinKolmogorovSmirnovKDE:
    
    def __init__(self, xlim=None, xlabel='Sample Values', ylabel='Density'):
        """
        Initialize the CalvinKolmogorovSmirnov class with optional x-axis limits for plots.
        
        Parameters:
        xlim (tuple): Optional. A tuple specifying the x-axis limits (min, max) for the plots.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        """
        self.xlim = xlim
        self.xlabel = xlabel
        self.ylabel = ylabel
    
    def ks_test(self, sample1, sample2):
        """
        Perform the Kolmogorov-Smirnov test for two samples.
        
        Parameters:
        sample1 (array-like): First sample data.
        sample2 (array-like): Second sample data.
        
        Returns:
        tuple: KS statistic and p-value.
        """
        ks_stat, p_value = ks_2samp(sample1, sample2)
        return ks_stat, p_value
    
    def plot_kde(self, pivoted_df, ax, title, ks_results, bonferroni_corr=True):
        """
        Plot KDE histograms for columns in the pivoted DataFrame.
        
        Parameters:
        pivoted_df (DataFrame): Pivoted DataFrame with samples as columns.
        ax (matplotlib.axes.Axes): Axes object to plot on.
        title (str): Title of the plot.
        ks_results (dict): Dictionary containing KS test results for each pair of columns.
        bonferroni_corr (bool): Optional. If True, apply Bonferroni correction to p-values.
        """
        for column in pivoted_df.columns:
            sns.kdeplot(pivoted_df[column].dropna(), ax=ax, label=f'KDE {column}')
        
        ax.set_title(title, fontsize='x-large')
        ax.set_xlabel(self.xlabel, fontsize='x-large')
        ax.set_ylabel(self.ylabel, fontsize='x-large')
        if self.xlim is not None:
            ax.set_xlim(self.xlim[0], self.xlim[1])
        
        ax.legend(loc='upper left')
        sns.despine()
        ax.grid(False)
        
        # Prepare Bonferroni correction
        if bonferroni_corr:
            n_comparisons = len(ks_results.keys())
            corrected_ks_results = {}
            for pair, result in ks_results.items():
                if result[1] < 0.05:  # Perform Bonferroni adjustment
                    pval_corr = result[1] * n_comparisons
                    pval_corr = min(pval_corr, 1.0)  # Ensure p-value does not exceed 1
                    print(f"{pair} has been Bonferroni corrected by factor {n_comparisons} from {result[1]} to {pval_corr}")
                else:  # No need to correct. It is already insignificant
                    pval_corr = result[1]
                corrected_ks_results[pair] = (result[0], pval_corr)
            text = "\n".join([f"{pair[0]}-{pair[1]}: p = {result[1]:.3e}" for pair, result in corrected_ks_results.items()])
        else:
            text = "\n".join([f"{pair[0]}-{pair[1]}: p = {result[1]:.3e}" for pair, result in ks_results.items()])
        
        # Add KS test results at the bottom right-hand side
        ax.text(0.95, 0.05, text, ha='right', va='bottom', transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.0))

    def ks_test_and_plot_kde(self, df, concat_col, category_cols, ax, verbose=False):
        """
        Perform KS test and plot KDE histograms for a given concatenation column.
        
        Parameters:
        df (DataFrame): Original DataFrame.
        concat_col (str): Column name to be concatenated for pivoting.
        category_cols (list): List of columns to categorize by.
        ax (matplotlib.axes.Axes): Axes object to plot on.
        verbose (bool): Optional. If True, print detailed KS test results.
        
        Returns:
        dict: Dictionary of KS test results for each pair of columns in the pivoted DataFrame.
        """
        pivoted_df = self.pivot_dataframe(df, concat_col, category_cols)
        
        ks_results = {}
        
        for col1, col2 in combinations(pivoted_df.columns, 2):
            sample1 = pivoted_df[col1].dropna()
            sample2 = pivoted_df[col2].dropna()
            ks_stat, p_value = self.ks_test(sample1, sample2)
            ks_results[(col1, col2)] = (ks_stat, p_value)
            if verbose:
                print(f"KS Statistic for {col1} vs {col2}: {ks_stat}")
                print(f"P-value for {col1} vs {col2}: {p_value}")

        self.plot_kde(pivoted_df, ax, f'KDE Comparison for {concat_col}', ks_results)
        
        return ks_results

    def pivot_dataframe(self, df, concat_col, category_cols):
        """
        Pivot the DataFrame based on the concatenation column and category columns.
        
        Parameters:
        df (DataFrame): Original DataFrame.
        concat_col (str): Column name to be concatenated for pivoting.
        category_cols (list): List of columns to categorize by.
        
        Returns:
        DataFrame: Pivoted DataFrame.
        """
        df.reset_index(drop=True, inplace=True)
        pivoted_df = df.pivot_table(index=df.index, columns=category_cols, values=concat_col)
        
        return pivoted_df
    
    def orchestrate_ks_tests(self, df, concat_cols, category_cols, outdir, filetype='svg'):
        """
        Orchestrate multiple iterations of KS tests and plot KDE histograms for a list of concatenation columns.
        
        Parameters:
        df (DataFrame): Original DataFrame.
        concat_cols (list): List of concatenation columns.
        category_cols (list): List of columns to categorize by.
        outdir (str): Output directory for saving plots.
        filetype (str): File type for saving plots (e.g., 'svg', 'png').
        
        Returns:
        dict: Dictionary containing KS test results for each concatenation column.
        """
        n = len(concat_cols)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*5))
        
        # Handle the case where there is only one plot
        if n == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        all_ks_results = {}
        
        for i, concat_col in enumerate(concat_cols):
            ks_results = self.ks_test_and_plot_kde(df, concat_col, category_cols, axes[i])
            all_ks_results[concat_col] = ks_results
        
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.savefig(f"{outdir}/KDE_plots.{filetype}")
        plt.show()
        
        return all_ks_results