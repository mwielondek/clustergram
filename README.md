# clustergram.py ፨
Visualize clustering over some parameter

## Installation
Download and either place inside your working directory, or append to your `sys.path`. Then import using `import clustergram`.

In case you're missing some of the dependencies run:
```$ pip install -r requirements.txt```

## Reference
Clustergram exposes only one function, `draw`, for which short parameter reference follows below:
```
  x        -  x values
  clusters -  array of the shape (n_xvals, n_samples) of cluster IDs to which each sample belongs
  targets  -  if true labels are known, colors the plot accordingly
  scoring  -  adds subplots using `sklearn.metrics` for scoring the clustering
  X        -  original feature array of the shape (n_samples, n_features) required for `sklearn.metrics.silhouette_score`
```
For remaining parameters (mostly self-explanatory) see source code.

## Example
Clustergram with scoring plots, using data from the [zoo dataset](https://archive.ics.uci.edu/ml/datasets/Zoo), and the [BCPNN](https://github.com/mwielondek/BCPNN) classifier (which uses the x values as the softmax probability distribution parameter G in the transfer function).

![Example screenshot](example_zoo.png?raw=true)
