# Evaluating Neural Network and k-NN algorithms on digits dataset

# from NeuralNet import *
from kNN import *
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import statistics

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
    
    # Calculate precision, recall, and F1-score
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
    # import digits dataset
    # 10 classes: 0-9
    # 8x8=64 numerical attributes (greyscale values)
    # 1797 instances
    digits = datasets.load_digits()

    digits = datasets.load_digits(return_X_y=True)
    digits_x = digits[0]
    digits_y = digits[1]
    N = len(digits_x)

    num_folds = 10

    k = 15

    train_acc, test_acc, test_f = k_fold_cross_validation(digits_x, digits_y, num_folds, k)
    #print("Predictions:", predictions)
    # print("Training accuracy:", train_acc)
    # print("Testing accuracy:", test_acc)
    # print("Testing F-Score:", test_f)
    print("For", k, "nearest neighbours \n")
    print("Test accuracy:", np.mean(test_acc))
    print("Test F-Score:", np.mean(test_f))


    # prints 64 attributes of a random digit and its class
    # shows the digit on the screen
    digit_to_show = np.random.choice(range(N), 1)[0]
    print("Attributes:", digits_x[digit_to_show])
    print("Class:", digits_y[digit_to_show])

    # plt.imshow(np.reshape(digits_x[digit_to_show], (8,8)))
    # plt.show()

if __name__ == "__main__":
    main()




# # import digits dataset
# digits = datasets.load_digits()
# # 10 classes: 0-9
# # 8x8=64 numerical attributes (greyscale values)
# # 1797 instances


# ##### EXAMPLE FROM SCIKIT-LEARN WEBSITE #####
# digits = datasets.load_digits(return_X_y=True)
# digits_dataset_X = digits[0]
# digits_dataset_y = digits[1]
# N = len(digits_dataset_X)

# # prints 64 attributes of a random digit and its class
# # shows the digit on the screen
# digit_to_show = np.random.choice(range(N), 1)[0]
# print("Attributes:", digits_dataset_X[digit_to_show])
# print("Class:", digits_dataset_y[digit_to_show])

# plt.imshow(np.reshape(digits_dataset_X[digit_to_show], (8,8)))
# plt.show()



# _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
# for ax, image, label in zip(axes, digits.images, digits.target):
#     ax.set_axis_off()
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
#     ax.set_title("Training: %i" % label)

# # flatten the images
# n_samples = len(digits.images)
# data = digits.images.reshape((n_samples, -1))

# # Create a classifier: a support vector classifier
# clf = svm.SVC(gamma=0.001)

# # Split data into 50% train and 50% test subsets
# X_train, X_test, y_train, y_test = train_test_split(
#     data, digits.target, test_size=0.5, shuffle=False
# )

# # Learn the digits on the train subset
# clf.fit(X_train, y_train)

# # Predict the value of the digit on the test subset
# predicted = clf.predict(X_test)