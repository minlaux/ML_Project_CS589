# Evaluating Neural Network and Decision Tree algorithms on titanic dataset

import sys
sys.path.append("./src/neural_net")
sys.path.append("./src/decision_tree")

from NeuralNet import *
from DecisionTree import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
from tabulate import tabulate
import random

# load titanic dataset
import_data = pd.read_csv("src/datasets/titanic.csv")
