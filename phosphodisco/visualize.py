from .classes import Clusters
from pandas import DataFrame
import pandas as pd
import numpy as np
from typing import Optional
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from .catheat import heatmap as catheat
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist


def compute_order(
        df,
        optimal=True,
        dist_method="euclidean",
        cluster_method="average"
):
    dist_mat = pdist(df, metric=dist_method)
    link_mat = hierarchy.linkage(dist_mat, method=cluster_method)

    if optimal==True:
        return hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(link_mat, dist_mat))
    else:
        return hierarchy.leaves_list(link_mat)


def visualize_cluster_heatmaps(
        clusters: Clusters,
        annotations: Optional[DataFrame] = None,
        col_cluster=True,
        row_cluster=True,
        cluster_kws: dict = {},
        annot_kws: dict = {},
        heatmap_kws: dict = {},
        file_prefix: str = 'heatmap'
):
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    sns.set(font='arial', style='white', color_codes=True, font_scale=1.3)
    matplotlib.rcParams.update({'savefig.bbox': 'tight'})

    cluster_sets = clusters.cluster_labels
    cluster_sets = {
        cluster_name: cluster_sets.index[cluster_sets==cluster_name] for cluster_name in
        clusters.nmembers_per_cluster.keys() if cluster_name != -1
    }
    values = clusters.abundances

    for cluster_name, sites in cluster_sets.items():
        df = values.loc[sites, :]

        if row_cluster:
            row_order = compute_order(df, **cluster_kws)
            row_order = [df.index[i] for i in row_order]
        else:
            row_order = df.index

        if col_cluster:
            col_order = compute_order(df.transpose(), **cluster_kws)
            col_order = [df.columns[i] for i in col_order]
        else:
            col_order = df.columns
        df = df.reindex(col_order, axis=1).reindex(row_order, axis=0)
        if annotations is None:
            header = pd.DataFrame(np.empty(len(col_order)), columns=[''])
        else:
            header = annotations.reindex(col_order).transpose()

        fig_len = 0.25*(len(df) + len(header))
        fig_width = 0.1*len(col_order)

        fig = plt.figure(figsize=(fig_width, fig_len))
        gs = plt.GridSpec(
            nrows=3, ncols=2,
            height_ratios=[len(header)]+2*[len(df)/2],
            width_ratios=[len(col_order), 5],
            hspace=0, wspace=0
        )

        heat_ax = plt.subplot(gs[1:, 0])
        cbar_ax = plt.subplot(gs[-1, -1])
        if annotations is not None:
            header_ax = plt.subplot(gs[0, 0])
            leg_ax = plt.subplot(gs[1, 1])
            leg_ax.axis('off')
            catheat(
                header,
                xticklabels=False, yticklabels=header.index,
                ax=header_ax, leg_ax=leg_ax, leg_kws=dict(loc=(0,0), labelspacing=0),
                **annot_kws
            )
            header_ax.set_yticklabels(header.index, rotation=0)
            header_ax.set_title('Cluster %s' % cluster_name)
            header_ax.set_xlabel('')
            header_ax.set_ylabel('')

        sns.heatmap(
            df,
            xticklabels=df.columns, yticklabels=['-'.join(i) for i in df.index],
            ax=heat_ax,
            cbar_ax=cbar_ax,
            **heatmap_kws
        )
        heat_ax.set_yticklabels(heat_ax.get_yticklabels(), rotation=0)
        heat_ax.set_xlabel('')
        heat_ax.set_ylabel('')

        plt.savefig('%s.cluster%s.pdf' % (file_prefix, cluster_name))
        plt.show()
        plt.close()
