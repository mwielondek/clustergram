import matplotlib.pyplot as plt
import numpy as np

from .labeling import opt_cluster_labeling

def draw(x, clusters, targets=None, legend=True, cluster_padding=0.45, one_indexed=False,
         xlabel=None, fig_dpi=200, fig_size=(8,7), fig_facecolor='xkcd:mint green'):
    n_xvals, n_samples = clusters.shape
    padding = np.linspace(-cluster_padding, cluster_padding, n_samples)

    fig, ax = plt.subplots()
    ax.plot(x, clusters + padding + 1, 'o-', markerfacecolor=(1,1,1,0.9), markersize=3, drawstyle='steps-mid')

    if targets is not None:
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        assert max(targets) < len(colors), "not enough colors for 1-1 mapping with targets"
        for i, line in enumerate(ax.lines[:-1]):
            line.set_color(colors[targets[i]])

    n_clusters = list(map(lambda row: np.unique(row).size, clusters))
    ax.plot(x, n_clusters, ':', color='#CCCCCC')

    ax.grid(which='both', axis='x', color='#CCCCCC', linestyle=(0, (1, 10)))
    ax.set_xticks(x, minor=True)
    ax.set_xticks(x[::n_xvals//4], minor=False)

    yticks = np.arange(1, max(n_clusters)+1, 1)
    ax.set_yticks(yticks, minor=False)
    yticks = map(lambda y: [y - padding[0], y + padding[0]], yticks)
    yticks = np.array(list(yticks)).flatten().tolist()
    ax.set_yticks(yticks, minor=True)

    ax.set_ylabel('Cluster ID')
    ax.set_xlabel(xlabel)
    if legend:
        if targets is None:
            ax.legend((np.arange(n_samples) + one_indexed).tolist() + ['# of clu.'], loc=0)
        else:
            vals, idx = np.unique(targets, return_index=True)
            nplines = np.array(ax.lines)
            ax.legend(nplines[idx], vals + one_indexed, loc=0)

    for idx, _ in list(enumerate(yticks))[::2]:
        ax.axhspan(*yticks[idx:idx+2], alpha=0.1)

    fig.set_dpi(fig_dpi)
    fig.set_size_inches(fig_size)
    fig.patch.set_facecolor(fig_facecolor)

    return fig, ax
