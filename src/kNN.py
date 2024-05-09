# Polina Petrova
# Implementation of k-NN algorithm


import numpy as np
import math


class KNN:
    """trains k-NN algorithm"""
    def __init__(self, k: int):
        self.k = k
        self.x = None
        self.y = None

    def euc_dist(self, x1, x2):
        """calculates euclidean distance between points x1 and x2"""
        return math.sqrt(sum((x1 - x2) ** 2))
    
    def fit(self, x, y):
        """fits algorithm to given data"""
        self.x = x
        self.y = y

    def predict(self, data):
        """predicts class labels for instances in x"""
        predictions = []

        for point1 in data:
            # calculate distances between the current instance and all other training instances
            distances = [self.euc_dist(point1, point2) for point2 in self.x]

            # get the indices of the k nearest neighbors
            nn = np.argsort(distances)[: self.k]

            # get labels of k nearest neighbors
            n_labels = self.y[nn]

            counts = {}

            # loop through k-nn labels 
            for label in n_labels:
                # increment count by 1 if label is in counts dictionary
                if label in counts:
                    counts[label] += 1
                else:
                    counts[label] = 1

            # make prediction based on majority class/label
            p_label = max(counts, key = counts.get)
            predictions.append(p_label)

        return predictions