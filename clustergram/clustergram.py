import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from dataclasses import dataclass

from .labeling import opt_cluster_labeling

@dataclass(init=True, eq=False)
class Clustergram:

    cluster_padding: float = 0.45
    optimize_labeling: bool = True
    legend: bool = True
    fig_dpi: int = 200
    fig_size: tuple = (8,7)
    fig_facecolor: str = 'xkcd:mint green'


    def draw(self, x, clusters, targets=None, sort=True, one_indexed=False, xlabel=None,
                linewidth=1, scoring=False, scoring_X=None, scoring_annotate=True, scoring_beta=1):

        n_xvals, n_samples = clusters.shape
        padding = np.linspace(-self.cluster_padding, self.cluster_padding, n_samples)

        if self.optimize_labeling:
            for i in range(1, len(clusters)):
                clusters[i] = opt_cluster_labeling(clusters[i - 1], clusters[i])

        if sort and targets is not None:
            idx = targets.argsort()
            targets = targets[idx]
            clusters = clusters[:, idx]

        fig, ax = plt.subplots()
        size_param = dict(markersize=linewidth, linewidth=linewidth)
        ax.plot(x, clusters + padding + 1, 'o-', drawstyle='steps-mid', markerfacecolor=(1,1,1,0.9), **size_param)

        if targets is not None:
            # colors = plt.rcParams['axes.prop_cycle']
            # colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
            colors = plt.cm.tab10.colors
            line_cycler     = cycler(linestyle=['-', '--'])
            color_cycler    = cycler(color=colors)
            styles = list(line_cycler * color_cycler)
            for i, line in enumerate(ax.lines):
                line.set_color(styles[targets[i]]['color'])
                line.set_linestyle(styles[targets[i]]['linestyle'])

        n_clusters = list(map(lambda row: np.unique(row).size, clusters))
        ax.plot(x, n_clusters, ':', color='#CCCCCC')

        ax.grid(which='both', axis='x', color='#CCCCCC', linestyle=(0, (1, 10)))
        ax.set_xticks(x, minor=True)
        ax.set_xticks(x[::max(1, n_xvals//4)], minor=False)

        yticks = np.arange(1, max(n_clusters)+1, 1)
        ax.set_yticks(yticks, minor=False)
        yticks = map(lambda y: [y - padding[0], y + padding[0]], yticks)
        yticks = np.array(list(yticks)).flatten().tolist()
        ax.set_yticks(yticks, minor=True)

        ax.set_ylabel('Cluster ID')
        ax.set_xlabel(xlabel)
        if self.legend:
            if targets is None:
                ax.legend((np.arange(n_samples) + one_indexed).tolist() + ['# of clu.'], loc=0)
            else:
                vals, idx = np.unique(targets, return_index=True)
                nplines = np.array(ax.lines)
                ax.legend(nplines[idx], vals + one_indexed, loc=0)

        for idx, _ in list(enumerate(yticks))[::2]:
            ax.axhspan(*yticks[idx:idx+2], alpha=0.1)

        fig.set_dpi(self.fig_dpi)
        fig.set_size_inches(self.fig_size)
        fig.patch.set_facecolor(self.fig_facecolor)

        if scoring:
            from .scoring import append
            append(ax, x, clusters, targets, scoring_X, scoring_annotate, beta=scoring_beta)

        return fig, ax
