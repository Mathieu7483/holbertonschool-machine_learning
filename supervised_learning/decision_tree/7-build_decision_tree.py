#!/usr/bin/env python3
"""7 Build a decision tree"""

import numpy as np


class Node:
    """Class Node that represents a node in a decision tree"""

    def __init__(self, feature=None, threshold=None,
                 left_child=None, right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth
        self.lower = None
        self.upper = None

    def max_depth_below(self):
        """Returns the maximum depth of the tree below the node"""
        if self.is_leaf:
            return self.depth
        return max(self.left_child.max_depth_below(),
                   self.right_child.max_depth_below())

    def count_nodes_below(self, only_leaves=False):
        """Returns the number of nodes/leaves below the node"""
        if self.is_leaf:
            return 1
        count = 0 if only_leaves else 1
        if self.left_child:
            count += self.left_child.count_nodes_below(only_leaves)
        if self.right_child:
            count += self.right_child.count_nodes_below(only_leaves)
        return count

    def get_leaves_below(self):
        """Returns a list of all the leaves below the node"""
        if self.is_leaf:
            return [self]
        leaves = []
        if self.left_child:
            leaves += self.left_child.get_leaves_below()
        if self.right_child:
            leaves += self.right_child.get_leaves_below()
        return leaves

    def update_bounds_below(self):
        """Recursively computes lower and upper bounds for each node"""
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -np.inf}

        for child in [self.left_child, self.right_child]:
            if child:
                child.lower = self.lower.copy()
                child.upper = self.upper.copy()
                if child == self.left_child:
                    child.lower[self.feature] = self.threshold
                else:
                    child.upper[self.feature] = self.threshold
                child.update_bounds_below()

    def update_indicator(self):
        """Calculates the indicator function for the node"""
        def is_large_enough(x):
            return np.all([x[:, f] > self.lower.get(f, -np.inf)
                           for f in self.lower], axis=0)

        def is_small_enough(x):
            return np.all([x[:, f] <= self.upper.get(f, np.inf)
                           for f in self.upper], axis=0)

        self.indicator = lambda x: np.all([is_large_enough(x),
                                           is_small_enough(x)], axis=0)

    def pred(self, x):
        """Returns the prediction of the node for a single input x"""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        return self.right_child.pred(x)

    def __str__(self):
        """Visual representation of the node"""
        out = ("root" if self.is_root else "-> node")
        out += f" [feature={self.feature}, threshold={self.threshold}]\n"
        if self.left_child:
            out += self.left_child_add_prefix(str(self.left_child))
        if self.right_child:
            out += self.right_child_add_prefix(str(self.right_child))
        return out

    def left_child_add_prefix(self, text):
        """Adds a prefix to the left child for visualization purposes"""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """Adds a prefix to the right child for visualization purposes"""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += ("       " + x) + "\n"
        return new_text


class Leaf(Node):
    """Class Leaf that represents a leaf in a decision tree"""

    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Returns the maximum depth of the tree
        below the leaf (which is just the depth of the leaf)"""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Returns the number of nodes/leaves below the leaf
        (which is just 1 if only_leaves is True, and 0 otherwise)"""
        return 1

    def get_leaves_below(self):
        """Returns a list of all the leaves below the leaf
        (which is just a list containing the leaf itself)"""
        return [self]

    def update_bounds_below(self):
        """Leaves do not have bounds"""
        pass

    def pred(self, x):
        """Returns the prediction of the leaf for a single input x"""
        return self.value

    def __str__(self):
        """Visual representation of the leaf"""
        return f"-> leaf [value={self.value}]\n"


class Decision_Tree():
    """Class Decision_Tree that represents a decision tree"""

    def __init__(self, max_depth=10, min_pop=1,
                 seed=0, split_criterion="random", root=None):
        self.rng = np.random.default_rng(seed)
        self.root = root if root else Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """Returns the depth of the tree"""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Returns the number of nodes/leaves in the tree"""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def update_bounds(self):
        """Updates the bounds of all the nodes in the tree"""
        self.root.update_bounds_below()

    def update_predict(self):
        """Updates the predict function of the tree
        based on the current structure of the tree"""
        self.update_bounds()
        leaves = self.root.get_leaves_below()
        for leaf in leaves:
            leaf.update_indicator()

        def predict_all(A):
            """Predicts the class for each input in A
            by checking which leaf's indicator function is satisfied"""
            results = np.empty(A.shape[0], dtype=int)
            for leaf in leaves:
                results[leaf.indicator(A)] = leaf.value
            return results
        self.predict = predict_all

    def accuracy(self, test_explanatory, test_target):
        """Returns the accuracy of the tree on the given test dataset"""
        return np.sum(np.equal(self.predict(test_explanatory),
                               test_target)) / test_target.size

    def np_extrema(self, arr):
        """Returns the minimum and maximum of a numpy array"""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Randomly selects a feature and a threshold for splitting the node"""
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[node.sub_population, feature])
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def fit_node(self, node):
        """Recursively fits the tree by splitting nodes"""
        node.feature, node.threshold = self.split_criterion(node)
        left_population = node.sub_population & (
            self.explanatory[:, node.feature] > node.threshold)
        right_population = node.sub_population & (
            self.explanatory[:, node.feature] <= node.threshold)

        # leaf conditions
        def is_leaf(pop, depth):
            if np.sum(pop) < self.min_pop or depth >= self.max_depth:
                return True
            if np.unique(self.target[pop]).size == 1:
                return True
            return False

        # left child
        if is_leaf(left_population, node.depth + 1):
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # right child
        if is_leaf(right_population, node.depth + 1):
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """Creates and returns a leaf child"""
        value = np.bincount(self.target[sub_population]).argmax()
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Creates and returns a node child"""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit(self, explanatory, target, verbose=0):
        """Trains the decision tree"""
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion

        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(
                f"  Training finished.\n"
                f"    - Depth                     : {self.depth()}\n"
                f"    - Number of nodes           : {self.count_nodes()}\n"
                f"    - Number of leaves          : "
                f"{self.count_nodes(only_leaves=True)}\n"
                f"    - Accuracy on training data : "
                f"{self.accuracy(self.explanatory, self.target)}"
            )

    def __str__(self):
        """Visual representation of the tree"""
        return str(self.root)
