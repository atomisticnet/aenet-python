#!/usr/bin/env python

"""
Perform dimension reduction and cluster analysis of a reference data
set.

"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import argparse

__author__ = "Alexander Urban"
__email__ = "aurban@atomistic.net"
__date__ = "2020-11-01"
__version__ = "0.1"


def clustering(data_file, num_dimensions, num_clusters):
    print("Reading data.")
    df = pd.read_csv(data_file)
    # Don't use the last two columns, which are the energy and the path
    dim = len(df.columns) - 3
    x = df.iloc[:, :dim].values

    print("Standardizing features.")
    x = StandardScaler().fit_transform(x)

    print("Performing PCA.")
    pca = PCA(n_components=num_dimensions)
    components = pca.fit_transform(x)
    df_pca = pd.DataFrame(
        data=np.concatenate((components, df.iloc[:, dim:].values), axis=1),
        columns=["PC{}".format(i+1) for i in range(num_dimensions)] + [
            "num_atoms", "energy", "path"])
    with open("pca.csv", "w") as fp:
        df_pca.to_csv(fp)
    print("Explained variance ratio per PC: \n", pca.explained_variance_ratio_)
    print("Total explained variance       : ",
          np.sum(pca.explained_variance_ratio_))

    print("Performing k-means clustering.")
    kmeans = KMeans(n_clusters=num_clusters).fit(components)
    df_kmeans = pd.DataFrame(
        data=np.c_[kmeans.labels_, df.iloc[:, dim:].values],
        columns=["cluster", "num_atoms", "energy", "path"])
    with open("kmeans.csv", "w") as fp:
        df_kmeans.to_csv(fp)

    clusters, counts = np.unique(kmeans.labels_, return_counts=True)
    counts = dict(zip(clusters, counts))
    energies = df.iloc[:, dim].values
    E_mean = [np.mean(energies[kmeans.labels_ == c]) for c in clusters]
    E_std = [np.std(energies[kmeans.labels_ == c]) for c in clusters]
    idx = np.argsort(E_mean)
    with open("cluster_stats.txt", "w") as fp:
        fp.write("# Cluster statistics:\n")
        fp.write("  #   count E_mean       E_std\n")
        for i in idx:
            c = clusters[i]
            fp.write("  {:3d} {:5d} {:12.6e} {:12.6e}\n".format(
                c, counts[c], E_mean[i], E_std[i]))


if (__name__ == "__main__"):

    parser = argparse.ArgumentParser(
        description=__doc__+"\n{} {}".format(__date__, __author__),
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        "data_file",
        help="File with structure fingerprints.")

    parser.add_argument(
        "-d", "--num-dimensions",
        help="Number of pricipal components (default: 20).",
        type=int,
        default=20)

    parser.add_argument(
        "-c", "--num-clusters",
        help="Number of clusters (default: 100).",
        type=int,
        default=100)

    args = parser.parse_args()

    clustering(args.data_file, args.num_dimensions, args.num_clusters)
