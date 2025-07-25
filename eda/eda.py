#! /usr/bin/env python
# coding: utf-8
from typing import List
import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from aquarel import load_theme
from matplotlib import cm
from matplotlib.colors import ListedColormap


RAW_DATA_PATH = './data/raw/'
EDA_PATH = './data/eda/'
raw_data_name = 'ObesityDataSet_raw_and_data_sinthetic.csv'
data_filepath = os.path.join(RAW_DATA_PATH, raw_data_name)


def load_csv(
        path: str
        ) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    return pd.read_csv(path)


def plot_hist(
        df: pd.DataFrame,
        feature: str
        ) -> plt.Figure:
    """Plot a histogram for a given feature in the DataFrame.

    :param df: The DataFrame containing the data.
    :param feature: The feature to plot.
    :return: The matplotlib Figure object.
    """
    fig, ax = plt.subplots()
    ax.hist(df[feature])
    plt.title('Feature: {}'.format(feature))
    plt.ylabel("Frequency")
    return fig


def plot_bar(
        df: pd.DataFrame,
        feature: str
        ) -> plt.Figure:
    """Plot a bar chart for a given categorical feature in the DataFrame

    :param df: The DataFrame containing the data.
    :param feature: The categorical feature to plot.
    :return: The matplotlib Figure object."""

    # Get counts of unique values in the feature
    counts = df[feature].value_counts().to_dict()

    fig, ax = plt.subplots()
    ax.bar(counts.keys(), counts.values())
    
    if len(counts.keys()) > 3:
        plt.xticks(rotation=90)
    else:
        plt.xticks(rotation=45)
    
    plt.ylabel("Frequency")
    plt.title('Feature: {}'.format(feature))
    return fig


def plot_corr_matrix(
    df: pd.DataFrame,
    features: list = None,
    title: str = "Correlation Matrix" 
    ) -> plt.Figure:
    """Plot a correlation matrix for a given set of features in the DataFrame using matplotlib.

    :param df: The DataFrame containing the data.
    :param features: The features to include in the correlation matrix, defaults to None
    :param title: The title of the plot, defaults to "Correlation Matrix"
    :return: The matplotlib Figure object.
    """
    if features is None:
        features = df.columns.tolist()
    corr = df[features].corr().to_numpy()
    fig, ax = plt.subplots(figsize=(10, 8))
    # Define your own custom colormap for the correlation matrix
    custom_colors = ["#75d6bc", "#b799e8", "#414546", "#fca964"]
    custom_cmap = ListedColormap(custom_colors)
    cax = ax.matshow(corr, cmap=custom_cmap)
    fig.colorbar(cax)
    ax.set_xticks(range(len(features)))
    ax.set_yticks(range(len(features)))
    ax.set_xticklabels(features, rotation=45, ha='left')
    ax.set_yticklabels(features)
    plt.title(title, pad=20)
    # Annotate correlation values
    for (i, j), val in np.ndenumerate(corr):
        ax.text(j, i, f"{val:.2f}", ha='center', va='center', color='black')
    return fig


def save_fig(
        fig: plt.Figure,
        path: str
        ) -> None:
    """Save a matplotlib Figure to a file

    :param fig: The matplotlib Figure object to save
    :param path: The file path where the figure will be saved
    """
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)


def get_unique_values(
        df: pd.DataFrame,
        feature: str
        ) -> np.ndarray:
    """Obtain unique values for a given feature in the DataFrame
    :param df: The DataFrame containing the data.
    :param feature: The feature for which to obtain unique values.
    :return: An array of unique values for the specified feature.
    """
    series_feat = df[feature]
    return series_feat.unique()


def plot_feature_pairs(
        df: pd.DataFrame,
        feature_names: List,
        color_labels: list = None,
        title_prefix: str = ''
        ) -> None:
    """Helper function to create scatter plots for all possible pairs of features.
    
    :param df: DataFrame containing the features to be plotted.
    :param feature_names: List of feature names to be used in plotting.
    :param color_labels: Optional. Cluster or class labels to color the scatter plots.
    :param title_prefix: Optional. Prefix for plot titles to distinguish between different sets of plots.
    """
    # Create a figure for the scatter plots
    plt.figure(figsize=(60, 60))
    
    # Counter for subplot index
    plot_number = 1
    
    # fig, ax = plt.subplots(figsize=(20, 20))
    # Loop through each pair of features
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            plt.subplot(len(feature_names)-1, len(feature_names)-1, plot_number)

            # Scatter plot colored by labels if provided
            if color_labels is not None:
                plt.scatter(df[feature_names[i]], df[feature_names[j]],
                           c=color_labels, cmap='viridis', alpha=0.7)
            else:
                plt.scatter(df[feature_names[i]], df[feature_names[j]], alpha=0.7)

            # Set labels and title
            plt.xlabel(feature_names[i])
            plt.ylabel(feature_names[j])
            plt.title(f'{title_prefix}{feature_names[i]} vs {feature_names[j]}')

            # Increment the plot number
            plot_number += 1

    # # Adjust layout to prevent overlap
    plt.tight_layout()

    return plt.gcf()


def _get_binary_columns(
        df: pd.DataFrame
        ) -> List[str]:
    """Identify binary columns in the DataFrame.
    """
    binary_columns = []
    print("Checking for binary columns...")
    for col in df.columns:
        unique_vals = sorted(df[col].dropna().unique())
        # Check if column has exactly 2 unique values and they are 0 and 1
        # if len(unique_vals) == 2 and set(unique_vals) == {0, 1}:
        if len(unique_vals) == 2:
            binary_columns.append(col)
            # print(f"✓ Found binary column: {col} with values {unique_vals}")
            print(f"✓ Found binary column: {col}")
        else:
            # print(f"✗ Skipping {col}: values = {unique_vals}")
            print(f"✗ Skipping {col}")
    
    return binary_columns


def plot_binary_contingency_heatmap(
        df: pd.DataFrame,
        binary_columns=None
        ) -> plt.Figure:
    """Create a contingency table heatmap for binary columns
    
    :param df: DataFrame containing the data.
    :param binary_columns: Optional list of binary columns to analyze. If None, will find them automatically.
    """
    
    # If no binary columns provided, find them automatically
    if binary_columns is None:
        binary_columns = _get_binary_columns(df)
    
    if len(binary_columns) < 2:
        print(f"Error: Need at least 2 binary columns. Found {len(binary_columns)}: {binary_columns}")
        return None
    
    print(f"\nAnalyzing {len(binary_columns)} binary columns: {binary_columns}")
    
    # Calculate subplot dimensions
    n_pairs = len(binary_columns) * (len(binary_columns) - 1) // 2
    n_cols = min(3, int(np.ceil(np.sqrt(n_pairs))))  # Max 3 columns
    n_rows = int(np.ceil(n_pairs / n_cols))
    
    # Create a figure for the heatmaps
    fig, ax = plt.subplots(figsize=(5 * n_cols, 4 * n_rows))
    
    # Counter for subplot index
    plot_number = 1
    for i in range(len(binary_columns)):
        for j in range(i + 1, len(binary_columns)):
            col1 = binary_columns[i]
            col2 = binary_columns[j]

            # Create contingency table
            contingency = pd.crosstab(df[col1], df[col2], margins=True)

            ax = plt.subplot(n_rows, n_cols, plot_number)
            
            # Plot heatmap (exclude margins)
            sns.heatmap(contingency.iloc[:-1, :-1], annot=True, fmt='d', 
                       cmap='Blues', cbar_kws={'label': 'Count'})
            
            # Add percentage annotations
            total = contingency.iloc[-1, -1]
            for row_idx in range(2):
                for col_idx in range(2):
                    count = contingency.iloc[row_idx, col_idx]
                    percentage = (count / total) * 100
                    ax.text(col_idx + 0.5, row_idx + 0.7, f'({percentage:.1f}%)', 
                            ha='center', va='center', fontsize=8, color='red')

            ax.set_xlabel(col2)
            ax.set_ylabel(col1)
            ax.set_title(f'{col1} vs {col2}')
            
            # Increment the plot number
            plot_number += 1
    
    plt.tight_layout()
    
    return fig


def main():
    start_time = time.time()
    print(f"Script started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    
    eda_folderpath = os.path.join(EDA_PATH, str(start_time))
    if not os.path.exists(eda_folderpath):
        os.makedirs(eda_folderpath)
        print(f"EDA directory created at {eda_folderpath}")
    else:
        print(f"EDA directory already exists at {eda_folderpath}")

    
    # Load data
    if not os.path.exists(data_filepath):
        raise FileNotFoundError(f"Data file not found at {data_filepath}. Please check the path.")
    df = load_csv(data_filepath)
    
    print(f"Data loaded successfully from {data_filepath}. Shape: {df.shape}") 

    # Split features by type
    categ_features = []
    num_features = []

    for feature in df.columns.to_list():
        series_feat = df[feature]
        if series_feat.dtype == 'object':
            categ_features.append(feature)
        else:
            num_features.append(feature)

   # Start EDA report
    report_filepath = os.path.join(eda_folderpath, 'eda_report.txt')
    with open(report_filepath, 'a') as report_file:
        print(f"EDA started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}", file=report_file)
        print(f"Data loaded from {data_filepath} with shape {df.shape}", file=report_file)
        
        print("\nData overview:", file=report_file)
        print(df.describe(), file=report_file)

        print("Data types in the dataset:", file=report_file)
        print(df.dtypes.to_string(), file=report_file)

        print("\nFeatures in the dataset:", file=report_file)
        print(df.columns.tolist(), file=report_file)
        
        print("Categorical features:", file=report_file)
        print(categ_features, file=report_file)
        print("\nNumerical features:", file=report_file)
        print(num_features, file=report_file)

        # Display unique values for categorical features
        report_file.write("\nUnique values for categorical features:\n")
        for c_feature in categ_features:
           unique = get_unique_values(df, c_feature)
           print(f'Feature: {c_feature:<30} no.: {len(unique):<5} values: {unique}', file=report_file)

        
    # Plot histograms for numerical features
    print("Plotting histograms for numerical features")
    for feature in num_features:
        fig_hist = plot_hist(df, feature)
        save_fig(fig_hist, os.path.join(eda_folderpath, f"histogram_{feature}.png"))
    print("Histograms saved successfully.")

    # Plot correlation matrix
    print("Plotting correlation matrix")
    fig_corr = plot_corr_matrix(df, features=num_features, title="Correlation Matrix of Numerical Features")
    save_fig(fig_corr, os.path.join(eda_folderpath, "correlation_matrix.png"))
    print("Correlation matrix saved successfully.")

    # Plot bar charts for categorical features
    print("Plotting bar charts for categorical features")
    for feature in categ_features:
        fig_bar = plot_bar(df, feature)
        save_fig(fig_bar, os.path.join(eda_folderpath, f"bar_chart_{feature}.png"))
    print("Bar charts saved successfully.")
    

    # Plot feature pairs
    print("Plotting feature pairs")
    feature_names = num_features + categ_features
    fig_pairs = plot_feature_pairs(df, feature_names=feature_names, title_prefix='Feature Pair: ')
    save_fig(fig_pairs, os.path.join(eda_folderpath, "feature_pairs.png"))
    print("Feature pairs plot saved successfully.")


    # Plot binary contingency heatmaps
    print("Plotting binary contingency heatmaps")
    fig_binary_heatmaps = plot_binary_contingency_heatmap(df, binary_columns=None)
    save_fig(fig_binary_heatmaps, os.path.join(eda_folderpath, "binary_contingency_heatmaps.png"))
    print("Binary contingency heatmaps saved successfully.")

    

    end_time = time.time()
    print(f"EDA finished at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")

    
if __name__ == "__main__":
    theme = load_theme("scientific")
    theme.params['colors']['palette'] = ["#75d6bc", "#b799e8", "#414546", "#fca964"]
    theme.params['ticks']['width_minor'] = 0.0
    theme = load_theme("boxy_light").set_overrides({
    "axes.grid": False
    })
    theme.apply()
    main()
    theme.apply_transforms()