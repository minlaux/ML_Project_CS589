# Evaluating Decision Tree and Random Forrest algorithms on loan dataset

#from NeuralNet import * random forrest file
from DecisionTree import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
from tabulate import tabulate
import random

# load parkinsons dataset
import_data = pd.read_csv("src/datasets/loan.csv")