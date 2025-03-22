#!/usr/bin/env python
# =============================================================================
# File: cluster.py
# Purpose: Cluster S&P 500 stocks using time-series & fundamental data, then
#          repeat the analysis after removing outliers.
#
# Steps:
#   1) Original Analysis (unchanged logic)
#       - Load data, aggregate, standardize, run K-means for k=2..10
#       - Plot Elbow & Silhouette, finalize clusters
#       - Summaries, bar chart, heatmap, cluster_assignments.csv
#
#   2) No-Outlier Analysis (new section)
#       - Remove outliers from the aggregated data based on z-scores
#       - Repeat the entire clustering pipeline
#       - Save separate results, cluster_assignments_no_outliers.csv
#
# Dependencies:
#   - pandas, numpy, matplotlib, seaborn, scikit-learn
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

def main():
    # =========================================================================
    # SECTION A: ORIGINAL ANALYSIS (UNCHANGED)
    # =========================================================================
    print("\n--- Original Analysis ---\n")

    # -----------------------------
    # STEP 1: Data Collection and Preprocessing
    # -----------------------------
    data = pd.read_csv("timeseries_cluster_ready.csv", parse_dates=["date"])
    print("Loaded timeseries data (sample):")
    print(data.head(), "\n")

    # Aggregate by ticker
    agg_data = data.groupby("ticker").agg({
        "daily_return": "mean",
        "volatility": "mean",
        "avg_volume": "mean",
        "pe_ratio": "mean",
        "ps_ratio": "mean",
        "pb_ratio": "mean",
        "sector": "first"
    }).reset_index()

    print("Aggregated data (one row per ticker):")
    print(agg_data.head(), "\n")

    # -----------------------------
    # STEP 2: Clustering Analysis Implementation
    # -----------------------------
    features = agg_data[["daily_return", "volatility", "avg_volume", "pe_ratio", "ps_ratio", "pb_ratio"]]
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    k_values = range(2, 11)
    inertias = []
    sil_scores = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)
        sil = silhouette_score(X, labels)
        sil_scores.append(sil)
        print(f"k={k}: Inertia = {kmeans.inertia_:.2f}, Silhouette Score = {sil:.2f}")

    # Elbow Plot
    plt.figure(figsize=(8, 4))
    plt.plot(k_values, inertias, marker='o', linestyle='-')
    plt.title("Elbow Method (Original): Inertia vs. Number of Clusters")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.grid(True)
    plt.show()

    # Silhouette Plot
    plt.figure(figsize=(8, 4))
    plt.plot(k_values, sil_scores, marker='o', linestyle='-', color='orange')
    plt.title("Silhouette Scores (Original) vs. Number of Clusters")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.show()

    # Choose best k from silhouette
    best_k = k_values[sil_scores.index(max(sil_scores))]
    print(f"Optimal number of clusters (original) = {best_k}\n")

    # Final K-means
    kmeans_final = KMeans(n_clusters=best_k, random_state=42)
    agg_data["cluster"] = kmeans_final.fit_predict(X)

    # Summaries
    cluster_summary = agg_data.groupby("cluster")[["daily_return", "volatility", "avg_volume",
                                                  "pe_ratio", "ps_ratio", "pb_ratio"]].mean()
    print("Cluster Summary (original):")
    print(cluster_summary, "\n")

    # Number of companies in each cluster
    cluster_counts = agg_data["cluster"].value_counts().sort_index()
    print("Number of companies in each cluster (original):")
    for c in cluster_counts.index:
        print(f"  Cluster {c}: {cluster_counts[c]} companies")
    print()

    # Sector comparison
    sector_tab = pd.crosstab(agg_data["cluster"], agg_data["sector"])
    print("Clusters vs. Sector (original):")
    print(sector_tab, "\n")

    # Risk-adjusted return
    cluster_perf = agg_data.groupby("cluster")[["daily_return", "volatility"]].mean()
    cluster_perf["risk_adjusted_return"] = cluster_perf["daily_return"] / cluster_perf["volatility"]
    print("Cluster Performance (original):")
    print(cluster_perf, "\n")

    median_rar = cluster_perf["risk_adjusted_return"].median()
    best_clusters = cluster_perf[cluster_perf["risk_adjusted_return"] > median_rar].index.tolist()
    print(f"Clusters with above-median RAR (original): {best_clusters}\n")

    # Grouped bar chart (raw means)
    features_list = ["daily_return", "volatility", "avg_volume", "pe_ratio", "ps_ratio", "pb_ratio"]
    cluster_ids = cluster_summary.index
    bar_width = 0.12
    plt.figure(figsize=(10, 5))
    x_indices = np.arange(len(features_list))

    for i, cid in enumerate(cluster_ids):
        cluster_vals = cluster_summary.loc[cid, features_list].values
        plt.bar(x_indices + i * bar_width, cluster_vals, width=bar_width, label=f"Cluster {cid}")

    plt.xticks(x_indices + bar_width*(len(cluster_ids)-1)/2, features_list, rotation=45)
    plt.title("Comparison of Feature Means by Cluster (Original)")
    plt.ylabel("Mean Value")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Z-score Heatmap
    zscore_summary = (cluster_summary - cluster_summary.mean()) / cluster_summary.std()
    plt.figure(figsize=(10, 6))
    sns.heatmap(zscore_summary, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Z-score Heatmap (Original) of Avg Feature Values per Cluster")
    plt.xlabel("Features")
    plt.ylabel("Cluster")
    plt.show()

    # Save cluster assignments
    agg_data.to_csv("cluster_assignments.csv", index=False)
    print("Saved original cluster assignments to 'cluster_assignments.csv'\n")

    # =========================================================================
    # SECTION B: ANALYSIS AFTER REMOVING OUTLIERS
    # =========================================================================
    print("\n--- No-Outlier Analysis ---\n")

    # We'll define outliers in the standardized space:
    # any row with absolute z-score > 3 in at least one feature.
    # Let's reuse X from the standard scaler above, ensuring the same scale.

    # For each row, find the max absolute z-score across all features
    max_abs_z = np.abs(X).max(axis=1)
    # We'll keep rows that have max_abs_z <= 3
    mask_no_outliers = (max_abs_z <= 3)

    # Create a new subset of agg_data and X
    agg_data_no_outliers = agg_data[mask_no_outliers].copy().reset_index(drop=True)
    X_no_outliers = X[mask_no_outliers]

    print(f"Removed {len(agg_data) - len(agg_data_no_outliers)} outliers. "
          f"Remaining data points: {len(agg_data_no_outliers)}\n")

    # Rerun the clustering analysis with no outliers
    k_values_2 = range(2, 11)
    inertias_2 = []
    sil_scores_2 = []

    for k in k_values_2:
        kmeans_2 = KMeans(n_clusters=k, random_state=42)
        labels_2 = kmeans_2.fit_predict(X_no_outliers)
        inertias_2.append(kmeans_2.inertia_)
        sil_2 = silhouette_score(X_no_outliers, labels_2)
        sil_scores_2.append(sil_2)
        print(f"[No Outliers] k={k}: Inertia = {kmeans_2.inertia_:.2f}, Silhouette Score = {sil_2:.2f}")

    # Elbow Plot (No Outliers)
    plt.figure(figsize=(8, 4))
    plt.plot(k_values_2, inertias_2, marker='o', linestyle='-')
    plt.title("Elbow Method (No Outliers): Inertia vs. Number of Clusters")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.grid(True)
    plt.show()

    # Silhouette Plot (No Outliers)
    plt.figure(figsize=(8, 4))
    plt.plot(k_values_2, sil_scores_2, marker='o', linestyle='-', color='orange')
    plt.title("Silhouette Scores (No Outliers) vs. Number of Clusters")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.show()

    best_k_2 = k_values_2[sil_scores_2.index(max(sil_scores_2))]
    print(f"Optimal number of clusters (no outliers) = {best_k_2}\n")

    kmeans_final_2 = KMeans(n_clusters=best_k_2, random_state=42)
    agg_data_no_outliers["cluster_no_outliers"] = kmeans_final_2.fit_predict(X_no_outliers)

    # Summaries
    cluster_summary_2 = agg_data_no_outliers.groupby("cluster_no_outliers")[[
        "daily_return", "volatility", "avg_volume", "pe_ratio", "ps_ratio", "pb_ratio"
    ]].mean()
    print("Cluster Summary (no outliers):")
    print(cluster_summary_2, "\n")

    # Number of companies in each cluster
    cluster_counts_2 = agg_data_no_outliers["cluster_no_outliers"].value_counts().sort_index()
    print("Number of companies in each cluster (no outliers):")
    for c in cluster_counts_2.index:
        print(f"  Cluster {c}: {cluster_counts_2[c]} companies")
    print()

    # Sector comparison
    sector_tab_2 = pd.crosstab(agg_data_no_outliers["cluster_no_outliers"], agg_data_no_outliers["sector"])
    print("Clusters vs. Sector (no outliers):")
    print(sector_tab_2, "\n")

    # Risk-adjusted return
    cluster_perf_2 = agg_data_no_outliers.groupby("cluster_no_outliers")[["daily_return", "volatility"]].mean()
    cluster_perf_2["risk_adjusted_return"] = cluster_perf_2["daily_return"] / cluster_perf_2["volatility"]
    print("Cluster Performance (no outliers):")
    print(cluster_perf_2, "\n")

    median_rar_2 = cluster_perf_2["risk_adjusted_return"].median()
    best_clusters_2 = cluster_perf_2[cluster_perf_2["risk_adjusted_return"] > median_rar_2].index.tolist()
    print(f"Clusters with above-median RAR (no outliers): {best_clusters_2}\n")

    # Grouped bar chart (raw means, no outliers)
    features_list_2 = ["daily_return", "volatility", "avg_volume", "pe_ratio", "ps_ratio", "pb_ratio"]
    cluster_ids_2 = cluster_summary_2.index
    bar_width_2 = 0.12
    plt.figure(figsize=(10, 5))
    x_indices_2 = np.arange(len(features_list_2))

    for i, cid in enumerate(cluster_ids_2):
        cluster_vals_2 = cluster_summary_2.loc[cid, features_list_2].values
        plt.bar(x_indices_2 + i * bar_width_2, cluster_vals_2, width=bar_width_2, label=f"Cluster {cid}")

    plt.xticks(x_indices_2 + bar_width_2*(len(cluster_ids_2)-1)/2, features_list_2, rotation=45)
    plt.title("Comparison of Feature Means by Cluster (No Outliers)")
    plt.ylabel("Mean Value")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Z-score Heatmap (no outliers)
    zscore_summary_2 = (cluster_summary_2 - cluster_summary_2.mean()) / cluster_summary_2.std()
    plt.figure(figsize=(10, 6))
    sns.heatmap(zscore_summary_2, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Z-score Heatmap (No Outliers) of Avg Feature Values per Cluster")
    plt.xlabel("Features")
    plt.ylabel("Cluster")
    plt.show()

    # Save cluster assignments (no outliers)
    agg_data_no_outliers.to_csv("cluster_assignments_no_outliers.csv", index=False)
    print("Saved no-outlier cluster assignments to 'cluster_assignments_no_outliers.csv'")

    print("\nAll analyses complete!\n")


if __name__ == "__main__":
    main()
