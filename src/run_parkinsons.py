# Evaluating k-NN and Neural Network algorithms on parkinsons dataset


#from NeuralNet import *
from kNN import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle


def f_score(predictions, true_labels):
    tp = 0
    fp = 0
    fn = 0
    
    # calculate true positives, false positives, false negatives
    for pred, true in zip(predictions, true_labels):
        if pred == true:
            tp += 1
        elif pred != true:
            if pred == 1:
                fp += 1
            else:
                fn += 1
    
    # calculate precision, recall, and F1-score
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    
    return f1_score

def normalise(x):
    """normalises values in instance to [0, 1] using min/max method"""
    # access each column
    for column_index in range(x.shape[1]):
        column_vals = x[:, column_index]
        min_x = min(column_vals)
        max_x = max(column_vals)

        # check if range is zero
        if max_x == min_x:
            # return original values
            x[:, column_index] = column_vals
        else:
            # normalise current column
            x[:, column_index] = (column_vals - min_x) / (max_x - min_x)
    return x

def kfold_indices(x: list, k: int):
    """
    performs k-fold split on x
    returns indices of each fold
    """
    # determine approximate size of the fold
    fold_size = len(x) // k
    index_intervals = np.arange(len(x))
    fold_indices = []

    for i in range(k):
        fold_indices.append(index_intervals[(i * fold_size) : ((i + 1) * fold_size)])

    return fold_indices

def split_folds(fold_indices: list, i: int):
    """splits folds into training and testing sets"""
    train_indices = np.concatenate([fold_indices[j] for j in range(len(fold_indices)) if j != i])
    test_indices = fold_indices[i]
    
    return train_indices, test_indices

def k_fold_cross_validation(x, y, n_folds: int, k: int):
    """
    n_folds: number of folds for cross-validation
    """
    fold_indices = kfold_indices(x, n_folds)

    # record evaluation metrics
    training_accuracies = []
    testing_accuracies = []

    training_f_score = []
    testing_f_score = []

    all_predictions = []
    
    for i in range(n_folds):
        # split data into training and testing sets
        train_indices, test_indices = split_folds(fold_indices, i)
        x_train, y_train = x[train_indices], y[train_indices]
        x_test, y_test = x[test_indices], y[test_indices]

        knn = KNN(k)
        # fit k-NN algorithm to training data
        knn.fit(normalise(x_train), y_train)
        
        # calculate accuracy of predicting training set
        train_predictions = knn.predict(x_train)
        train_accuracy = np.mean(train_predictions == y_train)
        training_accuracies.append(train_accuracy)

        # calculate accuracy of predicting testing set 
        test_predictions = knn.predict(x_test)
        test_accuracy = np.mean(test_predictions == y_test)
        testing_accuracies.append(test_accuracy)

        # calculate F-score of predicting testing set 
        testing_f = f_score(test_predictions, y_test)
        testing_f_score.append(testing_f)

    return training_accuracies, testing_accuracies, testing_f_score

def main():
    # load parkinsons dataset
    import_data = pd.read_csv("src/datasets/parkinsons.csv")

    # shuffle data
    park_data = shuffle(import_data)
    
    # extract x and y from data
    park_x = park_data.iloc[:, :-1].values
    park_y = park_data.iloc[:, -1].values
    
    print("Number of attributes:", len(park_x[0]))
    print("Number of instances:", len(park_x))

    # number of folds to use for cross-validation
    num_folds = 10

    # number of neighbours to use for k-NN
    # k = 25

    # range of k values to evaluate
    k_values = range(1, 105, 5)

    # list to store result of each k value
    results = []

    # perform cross-validation for each value of k
    for k in k_values:
        train_acc, test_acc, test_f = k_fold_cross_validation(park_x, park_y, num_folds, k)
        
        # calculate mean accuracy and F-score for testing data
        mean_test_accuracy = np.mean(test_acc)
        mean_test_f_score = np.mean(test_f)

        results.append({'k': k, 'test_accuracy': mean_test_accuracy, 'test_f_score': mean_test_f_score})

    # create dataframe from the results
    results_df = pd.DataFrame(results)
    print(results_df)
    print(results_df.to_latex(float_format="{%.4f}"))

    # train_acc, test_acc, test_f = k_fold_cross_validation(park_x, park_y, num_folds, k)
    # # #print("Predictions:", predictions)
    # print("Training accuracy:", train_acc)
    # # print("Testing accuracy:", test_acc)
    # # print("Testing F-Score:", test_f)
    # print("For", k, "nearest neighbours \n")
    # print("Test accuracy:", np.mean(test_acc))
    # print("Test F-Score:", np.mean(test_f))

if __name__ == "__main__":
    main()