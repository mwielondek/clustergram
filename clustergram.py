import matplotlib.pyplot as plt
import numpy as np

def draw(x, clusters, targets=None, legend=True, cluster_padding=0.45, one_indexed=False):
    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['figure.figsize'] = (8,7)

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
    ax.set_xlabel('g-values')
    if legend:
        if targets is None:
            ax.legend((np.arange(n_samples) + one_indexed).tolist() + ['# of clu.'], loc=0)
        else:
            vals, idx = np.unique(targets, return_index=True)
            nplines = np.array(ax.lines)
            ax.legend(nplines[idx], vals + one_indexed, loc=0)

    fig.patch.set_facecolor('xkcd:mint green')

    for idx, _ in list(enumerate(yticks))[::2]:
        ax.axhspan(*yticks[idx:idx+2], alpha=0.1)

    return fig, ax

def opt_cluster_labeling(prev, current):
    """Try to maximize number of elements that don't change clusters."""

    # if already matched, return
    if (current == prev).sum() == current.size:
        return current

    unique_curr = np.unique(current)  # unique clusters
    n_unique_curr = unique_curr.size
    target_clusters = np.unique(prev) # only interested in translating into cluster id from prev
    n_unique_prev = target_clusters.size

    # check if the same but translated
    if n_unique_prev == n_unique_curr:
        idx1, idx2 = np.unique(current, return_index=True)  # idx1 = unique cls, idx2 = at pos
        idx3 = prev[idx2]                                   # cls in prev at pos of unique els in current
        if np.unique(idx3).size == idx3.size:               # all must be unique, otherwise no 1-1 mapping
            # translate and compare
            proto = np.zeros(idx1.max() + 1)
            proto[idx1] = idx3
            if (proto[current] == prev).all():
                return prev

    # start with clusters that has the most corresponding entries
    mappings = get_mapping(current, prev)[:, 0]         # already sorted, remove counts
    mappings = np.array(mappings.tolist())
    placeholder = -1                                    # for unassigned targets
    translations = np.ones(n_unique_curr).astype(int) * placeholder
    while mappings.size:
        ffrom, to = mappings[0]
        translations[unique_curr == ffrom] = to
        # remove from mappings all where from or to are the same
        idx1, = (mappings[:,0] == ffrom).nonzero()
        idx2, = (mappings[:,1] == to).nonzero()
        mappings = np.delete(mappings, np.append(idx1, idx2), axis=0)

    # check which from nodes we don't have a translation for and update with lowest available id
    idx = (translations == placeholder)
    used_ids = translations[(translations >= 0) & (translations < n_unique_curr)]
    lowest = np.delete(list(range(n_unique_curr)), used_ids)
    translations[idx] = lowest[:np.flatnonzero(idx).size]

    # map current using unique elements table
    translated_current = list(map(lambda x: np.where(unique_curr == x)[0][0], current))
    return translations[translated_current]

def get_mapping(x, y):
    """Returns list of mappings from x to y sorted by counts."""
    mappings = np.vstack((x, y)).T
    mappings = np.unique(mappings, axis=0, return_counts=True)
    mappings = map(np.ndarray.tolist, mappings)
    mappings = list(zip(*mappings))
    return np.array(sorted(mappings, key=itemgetter(1), reverse=True))
