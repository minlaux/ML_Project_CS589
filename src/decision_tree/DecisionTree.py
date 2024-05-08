# Polina Petrova
# Implementation of Decision Tree algorithm


import numpy as np
import math


class DecisionNode:
    """implements single node from decision tree"""
    def __init__(self, feature_index = None, threshold = None, left_child = None, right_child = None, value = None):
        # feature index to split on
        self.feature_index = feature_index
        # threshold for binary split
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        # value of leaf node (class/label)
        self.value = value 


class DecisionTree:
    """builds a decision tree"""
    def __init__(self, max_depth = None):
        self.max_depth = max_depth

    def entropy(self, labels: list) -> float:
        """calculates the entropy of a list of labels"""
        total_instances = len(labels)
        label_counts = {}
        entropy = 0.0

        # count occurrences of each label and input into dictionary
        for label in labels:
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1

        # calculate entropy
        for count in label_counts.values():
            prob = count / total_instances
            entropy -= prob * math.log2(prob)

        return entropy

    def information_gain(self, data: list, labels: list, feature_index: int, threshold) -> float:
        """calculates information gain given a feature and a threshold"""
        total_instances = len(labels)
        #original entropy
        parent_entropy = self.entropy(labels)

        # split into left and right branches based on threshold
        left_indices = []
        right_indices = []

        for i, val in enumerate(data[:, feature_index]):
            if val <= threshold:
                left_indices.append(i)
            else:
                right_indices.append(i)

        # find labels of indices
        left_labels = labels[left_indices]
        right_labels = labels[right_indices]

        # find entropy on labels
        entropy_left = self.entropy(left_labels)
        entropy_right = self.entropy(right_labels)

        # find probabilities of left and right branches
        prob_left = len(left_indices) / total_instances
        prob_right = len(right_indices) / total_instances

        information_gain = parent_entropy - (prob_left * entropy_left + prob_right * entropy_right)

        return information_gain

    def split(self, data: list, labels: list):
        """finds the best feature and threshold to split data on"""
        best_feature = None
        best_threshold = None
        max_info_gain = 0.0

        for feature_index in range(data.shape[1]):
            unique_vals = np.unique(data[:, feature_index])
            
            for threshold in unique_vals:
                info_gain = self.information_gain(data, labels, feature_index, threshold)
                if info_gain > max_info_gain:
                    max_info_gain = info_gain
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold

    def build_tree(self, data: list, labels: list, depth = 0):
        """recursively builds decision tree"""
        # stop splitting if max depth reached or all node data belongs to same class/label
        if depth == self.max_depth or len(np.unique(labels)) == 1:
            return DecisionNode(value = labels[0])

        best_feature, best_threshold = self.split(data, labels)

        # if unable to find split,
        # create leaf node based on majority class of dataset
        if best_feature is None:
            # return DecisionNode(value = max(set(labels), key = labels.count))
            return DecisionNode(value = max(set(labels.tolist()), key = labels.tolist().count))


        # split into left and right branches based on best threshold value
        left_indices = []
        right_indices = []

        for i, val in enumerate(data[:, best_feature]):
            if val <= best_threshold:
                left_indices.append(i)
            else:
                right_indices.append(i)

        left_child = self.build_tree(data[left_indices], labels[left_indices], depth + 1)
        right_child = self.build_tree(data[right_indices], labels[right_indices], depth + 1)

        return DecisionNode(feature_index = best_feature, threshold = best_threshold,
                             left_child = left_child, right_child = right_child)

    def fit(self, data, labels):
        """fits decision tree to given data"""
        self.root = self.build_tree(data, labels)

    def predict_one(self, node: DecisionNode, instance: list):
        """predicts class label for a single instance"""
        # if leaf node, return class label
        if node.value is not None:
            return node.value

        if instance[node.feature_index] <= node.threshold:
            return self.predict_one(node.left_child, instance)
        else:
            return self.predict_one(node.right_child, instance)

    def predict(self, data):
        """predicts class labels for each instance"""
        predictions = []

        for instance in data:
            predictions.append(self.predict_one(self.root, instance))

        return predictions
    
    def train(self, x, y):
        """trains decision tree"""
        # fit decision tree to data
        self.fit(x, y)
        predictions = self.predict(x)

        return predictions