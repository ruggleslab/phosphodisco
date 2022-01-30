from .classes import ProteomicsData
from pathlib import Path
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
        df: DataFrame,
        optimal: bool = True,
        dist_method: str="euclidean",
        cluster_method: str="average"
):
    """Computes order of samples for clustered heatmaps.

    Args:
        df: Data with rows to cluster.
        optimal: Whether to return optimal ordering. Slows stuff down.
        dist_method: Which distance calculation to use.
        cluster_method: Which hierarchical clustering method to use.

    Returns: Clustered order of rows.

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
    """Makes heatmap figures of sites vs samples for each module.

    Args:
        data: ProteomicsData object containing AT LEAST normed_phospho data with no missing
        values (maybe imputed), and modules assigned.
        annotations: A DataFrame with samples as rows and categorical annotations to visualize as
        columns. This isn't taken directly from ProteomicsData because that table may have many
        more columns than anyone wants to visualize.
        col_cluster: Whether to cluster samples in the heatmaps.
        row_cluster: Whether to cluster rows in the heatmaps.
        cluster_kws: Additional keyword args to pass to visualize.compute_order
        annot_kws: Additional keyword args to pass to catheat.heatmap
        heatmap_kws: Additional keyword args to pass to sns.heatmap
        file_prefix: File prefix for each figure. Suffix will be .clusterX.pdf

    Returns: None

    """

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
        fig_width = 0.15*len(col_order)

        _ = plt.figure(figsize=(fig_width, fig_len))
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
    """Visualizes the associations between putative regulators and modules.

    Args:
        data: ProteomicsData object with regulator_coefficients assigned.
        percentile_cutoff: Heatmaps are filtered to show high associations only. What threshold
        should be used.
        savefig_prefix: If prefix is provided, a figure will be saved with this prefix.
        **heatmap_kwargs: Additional keyword args for sns.heatmap

    Returns: matplotlib ax with heatmap of coefficients

    """
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
    """Visualizes the associations between sample annotations and modules.

    Args:
        data: ProteomicsData object with annotation_association_FDR assigned.
        percentile_cutoff: Heatmaps are filtered to show high associations only. What
        percentile threshold should be used.
        savefig_prefix: If prefix is provided, a figure will be saved with this prefix.
        **heatmap_kwargs: Additional keyword args for sns.heatmap

    Returns: matplotlib ax with heatmap of associations

    """
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
    """Draws logos of each amino acid sequence motif for each module. Can pass in either
    module_aa_freqs or module_aa_enrichment from ProteomicsData objects.

    Args:
        seq_dfs: Either module_aa_freqs or module_aa_enrichment from ProteomicsData objects.
        save_prefix: If prefix is provided, a figure will be saved with this prefix.
        **logo_kws: Additional keyword args for logomaker.Logo

    Returns: None

    """
    for module, ps in seq_dfs.items():
        logo_kws['color_scheme'] = logo_kws.get('color_scheme', 'NajafabadiEtAl2017')
        logo = logomaker.Logo(ps, **logo_kws)
        logo.ax.set_title('Module %s motif enrichment' % module)
        if save_prefix:
            plt.savefig('%s.logo.motif_enrichment.module%s.pdf' % module)


def visualize_set_enrichment(
        module_enrichment_dict,
        pval_cutoff: float = 0.05,
        save_prefix: Optional[str] = None,
        **barplot_kws
):
    """Draws barplots for either enrichr or ptm-ssGSEA set enrichments per module.

    Args:
        module_enrichment_dict: go_enrichment or ptm_enrichment from ProteomicsData objects.
        pval_cutoff: p-val cut off to filter for significant enrichments.
        save_prefix: If prefix is provided, a figure will be saved with this prefix.
        **barplot_kws: Additional keyword args for sns.barplot

    Returns: None

    """
    barplot_kws['color'] = barplot_kws.get('color', '#BDBDBD')
    for module, df in module_enrichment_dict.items():
        temp = df[df['Adjusted P-value'] < pval_cutoff]
        temp = temp.sort_values('Adjusted P-value')
        if len(temp) > 0:
            fig, axs = plt.subplots(figsize=(6, 0.3 * len(temp)))
            sns.barplot(
                x=-np.log10(temp['Adjusted P-value']),
                y=temp.index,
                ax=axs,
                **barplot_kws
            )
            for i, row in enumerate(temp.iterrows()):
                row = row[1]
                plt.text(0, i, row['Genes'], va='center', fontsize=10)
            plt.xlabel('log$_{10}$(adjusted p-value)')
            plt.ylabel('Terms')
            plt.title('Cluster %s GO terms, FDR %s' % (module, pval_cutoff))
            if save_prefix:
                plt.savefig('%s.barplot.enrichment.pdf')

def visualize_aa_overlap(
        module_overlap_df_dict: dict,
        save_path: Optional[str]=None
        ):
    """
    Plots aa_overlap heatmap for each module.
    Args:
        module_overlap_df_dict: dictionary, output of motif_analysis.aa_overlap_from_df
                                which can be called via ProteomicsData.analyze_aa_overlap
                                keys are modules, values are DataFrames of 
                                aa_overlap scores for each pair of phosphosites
                                within the module
        save_fig:               path to folder where pdfs of plots should be saved.
                                saves no plots if None.
    Returns:
        None
    """
    for module, module_df in module_overlap_df_dict.items():  
        fig_len = 0.5*module_df.shape[0]
        fig_width = 0.4*module_df.shape[1]

        fig = plt.figure(figsize = (fig_len, fig_width))
        sns.heatmap(module_df, xticklabels = module_df.columns, yticklabels = module_df.index)
        plt.title(f'Module {module}')
        if save_path is not None:
            plt.savefig(Path(save_path) / Path(f'heatmap.aa_overlap.module{module}.pdf'))
    plt.show()
    plt.close()
