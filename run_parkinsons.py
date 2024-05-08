# Evaluating k-NN and Neural Network algorithms on parkinsons dataset

import sys
sys.path.append("./src/neural_net")
sys.path.append("./src/k-NN")

from NeuralNet import *
from kNN import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
from tabulate import tabulate
import random

# load parkinsons dataset
import_data = pd.read_csv("src/datasets/parkinsons.csv")