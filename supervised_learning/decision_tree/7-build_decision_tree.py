#!/usr/bin/env python3
"""7 Build a decision tree"""

from platform import node

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
        else:
            return max(self.left_child.max_depth_below(),
                       self.right_child.max_depth_below())

    def count_nodes_below(self, only_leaves=False):
        """Returns the number of nodes/leaves below the node"""
        count = 0 if only_leaves else 1
        if self.left_child:
            count += self.left_child.count_nodes_below(only_leaves)
        if self.right_child:
            count += self.right_child.count_nodes_below(only_leaves)
        return count

    def get_leaves_below(self):
        """Returns a list of all the leaves below the node"""
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
            self.lower = {0: -1 * np.inf}

        if self.left_child:
            self.left_child.lower = self.lower.copy()
            self.left_child.upper = self.upper.copy()
            self.left_child.lower[self.feature] = self.threshold

        if self.right_child:
            self.right_child.lower = self.lower.copy()
            self.right_child.upper = self.upper.copy()
            self.right_child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            if child:
                child.update_bounds_below()

    def __str__(self):
        """Visual representation of the node"""
        if self.is_root:
            out = (f"root [feature={self.feature}, "
                   f"threshold={self.threshold}]\n")
        else:
            out = (f"-> node [feature={self.feature}, "
                   f"threshold={self.threshold}]\n")

        if self.left_child:
            out += self.left_child_add_prefix(str(self.left_child))
        if self.right_child:
            out += self.right_child_add_prefix(str(self.right_child))
        return out

    def left_child_add_prefix(self, text):
        """Adds prefix for left child visualization"""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """Adds prefix for right child visualization"""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += ("       " + x) + "\n"
        return new_text

    def update_indicator(self):
        """Calculates the indicator function for the node"""
        def is_large_enough(x):
            """Checks if individuals are above lower bounds"""
            return np.all(np.array([np.greater(x[:, key], self.lower[key])
                                    for key in self.lower.keys()]), axis=0)

        def is_small_enough(x):
            """Checks if individuals are below or equal to upper bounds"""
            return np.all(np.array([np.less_equal(x[:, key], self.upper[key])
                                    for key in self.upper.keys()]), axis=0)

        self.indicator = lambda x: np.all(np.array([is_large_enough(x),
                                                    is_small_enough(x)]),
                                          axis=0)

    def pred(self, x):
        """Returns the prediction of the node for a single input x"""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """Class Leaf that represents a leaf in a decision tree"""

    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Returns the depth of the leaf"""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """A leaf always counts as 1"""
        return 1

    def get_leaves_below(self):
        """Returns the leaf itself in a list"""
        return [self]

    def update_bounds_below(self):
        """Leaves do not have children to update"""
        pass

    def __str__(self):
        """String representation of a leaf"""
        return f"-> leaf [value={self.value}]"

    def pred(self, x):
        """"Returns the value of the leaf for any input x"""
        return self.value


class Decision_Tree():
    """Class Decision_Tree that represents a decision tree"""

    def __init__(self, max_depth=10, min_pop=1,
                 seed=0, split_criterion="random", root=None):
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """Returns the maximum depth of the tree"""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Returns the number of nodes in the tree"""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def update_bounds(self):
        """Starts the update bounds process from the root"""
        self.root.update_bounds_below()

    def get_leaves(self):
        """Returns a list of all the leaves in the tree"""
        return self.root.get_leaves_below()

    def __str__(self):
        """Returns the string representation of the tree"""
        return self.root.__str__()

    def update_predict(self):
        """Updates the predict function of the tree
        based on the indicators of the leaves"""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.array([
            next(leaf.value for leaf in leaves
                 if leaf.indicator(x.reshape(1, -1))[0])
            for x in A
        ])

    def pred(self, x):
        """Returns the prediction of the tree for a single input x"""
        return self.root.pred(x)

    def fit_node(self, node):
        node.feature, node.threshold = self.split_criterion(node)

        left_population = np.where(self.explanatory[node.sub_population, node.feature] > node.threshold)[0]
        right_population = np.where(self.explanatory[node.sub_population, node.feature] <= node.threshold)[0]

        # Is left node a leaf ?
        is_left_leaf = (node.depth + 1 >= self.max_depth or len(left_population) < self.min_pop)

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # Is right node a leaf ?
        is_right_leaf = (node.depth + 1 >= self.max_depth or len(right_population) < self.min_pop)

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)


    def get_leaf_child(self, node, sub_population):
        value = np.bincount(self.target[sub_population]).argmax()
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth+1
        leaf_child.sub_population = sub_population
        return leaf_child


    def get_node_child(self, node, sub_population):
        n = Node()
        n.depth = node.depth+1
        n.sub_population = sub_population
        return n


    def accuracy(self, test_explanatory, test_target):
        return np.sum(np.equal(self.predict(test_explanatory), test_target))/test_target.size
