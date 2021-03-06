import numpy as np
from sklearn import metrics
from mpl_toolkits.axes_grid1 import make_axes_locatable

def append(ax, xvals, clusters, y, X=None, annotate=True, beta=1):
    scores = [
         [metrics.adjusted_mutual_info_score(y, c) for c in clusters]
        ,[metrics.adjusted_rand_score(y, c) for c in clusters]
        ,[metrics.v_measure_score(y, c, beta=beta) for c in clusters]
        ,[metrics.homogeneity_score(y, c) for c in clusters]
        ,[metrics.completeness_score(y, c) for c in clusters]
    ]
    if X is not None:
        scores.append([metrics.silhouette_score(X, c) if np.unique(c).size > 1 else 0 for c in clusters])
    scores_lbl = ['Adj. MI', 'Adj. Rand', 'V measure\nbeta=%.1f' % beta, 'Homogen.', 'Completen.', 'Silhouette']

    plt_sz = .4
    divider = make_axes_locatable(ax)

    for score, name in zip(scores, scores_lbl):
        ax_new = divider.append_axes("bottom", plt_sz, sharex=ax, pad=0)
        ax_new.plot(xvals, score)
        ax_new.set_ylabel(name, rotation='horizontal', ha='right', va='center')
        ax_new.set_yticks([])
        ax_new.set_ylim([0,1])
        # find max
        mx, my = xvals[np.argmax(score)], max(score)
        # print("Max {}: {} at {}".format(name, my, mx))
        ax_new.plot(mx, my, 'ro', markersize=3)
        if annotate:
            y_delta = 0.3
            y_delta = y_delta * -1 if my > 0.5 else y_delta
            ax_new.annotate('{:.2f}'.format(my).lstrip('0'), (mx, my + y_delta), color='0.3', fontsize='x-small', va='center')

    fig = ax.figure
    fig_sz = fig.get_size_inches()
    fig.set_size_inches(np.add(fig_sz, (0, len(scores) * plt_sz)))

    return fig, ax_new
