from .classes import ProteomicsData
from pandas import DataFrame
import pandas as pd
import numpy as np
from typing import Optional
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from .catheat import heatmap as catheat
import logomaker
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist


matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
sns.set(font='arial', style='white', color_codes=True, font_scale=1.3)
matplotlib.rcParams.update({'savefig.bbox': 'tight'})


def compute_order(
        df,
        optimal=True,
        dist_method="euclidean",
        cluster_method="average"
):
    """

    Args:
        df:
        optimal:
        dist_method:
        cluster_method:

    Returns:

    """
    dist_mat = pdist(df, metric=dist_method)
    link_mat = hierarchy.linkage(dist_mat, method=cluster_method)

    if optimal==True:
        return hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(link_mat, dist_mat))
    else:
        return hierarchy.leaves_list(link_mat)


def visualize_modules(
        data: ProteomicsData,
        annotations: Optional[DataFrame] = None,
        col_cluster=True,
        row_cluster=True,
        cluster_kws: dict = {},
        annot_kws: dict = {},
        heatmap_kws: dict = {},
        file_prefix: str = 'heatmap'
):

    cluster_sets = data.modules
    cluster_sets = {
        cluster_name: cluster_sets.index[cluster_sets==cluster_name] for cluster_name in
        data.modules.unique() if cluster_name != -1
    }
    values = data.normed_phospho

    for cluster_name, sites in cluster_sets.items():
        if int(cluster_name) == -1:
            continue

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


def visualize_regulator_coefficients(
        data: ProteomicsData,
        percentile_cutoff: float = 95,
        savefig_prefix: Optional[str] = None,
        **heatmap_kwargs
):
    if data.regulator_coefficients is None:
        raise KeyError(
            'Must calculate regulator coefficients using '
            'ProteomicsData.calculate_regulator_coefficients before visualizing. '
        )
    cut_off = np.nanpercentile(
        data.regulator_coefficients.abs().values.flatten(), percentile_cutoff
    )
    subset = data.regulator_coefficients[(data.regulator_coefficients > cut_off).any(axis=1)]
    ax = sns.heatmap(subset, **heatmap_kwargs)
    if savefig_prefix:
        plt.savefig('%s.pdf' % savefig_prefix)
    return ax


def visualize_annotation_associations(
        data: ProteomicsData,
        percentile_cutoff: float = 0,
        savefig_prefix: Optional[str] = None,
        **heatmap_kwargs
):
    if data.annotation_association_FDR is None:
        raise KeyError(
            'Must calculate regulator coefficients using '
            'ProteomicsData.calculate_regulator_coefficients before visualizing. '
        )
    temp = -np.log10(data.annotation_association_FDR)
    cut_off = np.nanpercentile(temp.values.flatten(), percentile_cutoff)
    subset = temp[(temp > cut_off).any(axis=1)]
    ax = sns.heatmap(subset, **heatmap_kwargs)
    if savefig_prefix:
        plt.savefig('%s.pdf' % savefig_prefix)
    return ax


def visualize_aa(seq_dfs, save_prefix: Optional[str] = None, **logo_kws):
    for module, ps in seq_dfs.items():
        logo_kws['color_scheme'] = logo_kws.get('color_scheme', 'NajafabadiEtAl2017')
        logo = logomaker.Logo(ps, **logo_kws)
        logo.ax.set_title('Module %s motif enrichment' % module)
        if save_prefix:
            plt.savefig('%s.logo.motif_enrichment.module%s.pdf')


def visualize_set_enrichment():
    pass