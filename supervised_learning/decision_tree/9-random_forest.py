#!/usr/bin/env python3
"""Random Forest Classifier"""

import numpy as np
Decision_Tree = __import__('8-build_decision_tree').Decision_Tree


class Random_Forest:
    """Random Forest Classifier"""

    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed
        self.trees = []

    def predict(self, explanatory):
        """Predict the class labels for the input data"""
        tree_predictions = np.array([tree.predict(explanatory)
                                     for tree in self.trees]).astype(int)
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(),
                                   axis=0, arr=tree_predictions)

    def fit(self, explanatory, target, n_trees=100, verbose=0):
        """Fit the random forest to the training data"""
        self.target = target
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        accuracies = []
        for i in range(n_trees):
            T = Decision_Tree(max_depth=self.max_depth,
                              min_pop=self.min_pop, seed=self.seed + i)
            T.fit(explanatory, target)
            self.trees.append(T)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
            accuracies.append(T.accuracy(T.explanatory, T.target))
        if verbose == 1:
            forest_acc = self.accuracy(self.explanatory, self.target)
            print(f"""  Training finished.
    - Mean depth                     : {np.array(depths).mean()}
    - Mean number of nodes           : {np.array(nodes).mean()}
    - Mean number of leaves          : {np.array(leaves).mean()}
    - Mean accuracy on training data : {np.array(accuracies).mean()}
    - Accuracy of the forest on td   : {forest_acc}""")

    def accuracy(self, explanatory, target):
        """Calculates the accuracy of the forest"""
        return np.sum(self.predict(explanatory) == target) / target.shape[0]

    def __str__(self):
        """Visual representation of the forest"""
        return '\n'.join([f'Tree {i}:\n{str(tree)}'
                          for i, tree in enumerate(self.trees)])
