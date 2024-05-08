# %% [markdown]
# Imports:

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

# %% [markdown]
# Loading datasets. For some reason (maybe lack of memory) these are just forgotten by the environment. So reports may be needed later

# %%
wine = pd.read_csv('datasets/wine.csv', sep='\s+')
house = pd.read_csv('datasets/house_votes_84.csv')
cancer = pd.read_csv('datasets/cancer.csv', sep='\s+')
contraceptive = pd.read_csv('datasets/contraceptive+method+choice/cmc.data', header=None)

#renames class column so my network can find output:
cancer.rename(columns={'Class': 'class'}, inplace=True)
contraceptive.rename(columns={9: 'class'}, inplace=True)

# %% [markdown]
# Globals are used as hyper-parameters in very specific places. Each dataset/example test should reset these

# %%
FINAL_MODEL = False
def reset_hyperparameters():
    global LAMBDA, ALPHA
    LAMBDA = 0
    ALPHA = 1
    
reset_hyperparameters()

# %% [markdown]
# Preparing data
# - cross-validation with stratified sampling
# - normalization of data
# - one-hot encoding of classes

# %%
def normalize_training_data(dataset):
    """
    Normalizes non-class columns to the [0,1] range
    returns non-class columns
    """
    for j in range(dataset.shape[1]):
        max = np.max(dataset[:,j])
        min = np.min(dataset[:,j])
        
        dataset[:,j] = (dataset[:,j] - min)/(max-min)
    
    return dataset

def vectorize_classes(class_labels):
    """
    one-hot encoding of class labels
    - returns: list of vectors representing y
    """
    classes = np.unique(class_labels)
    classes = np.sort(classes)
    
    #first index is index of second class in codified data
    num_classes = len(classes)
    index_col = np.arange(0, num_classes)
    classes = np.c_[index_col, classes]
    
    y = []
    
    for i in range(class_labels.shape[0]):
        curr_class = class_labels[i]
        class_index = np.where(classes[:, 1] == curr_class)[0][0]
        
        # one hot encoding
        
        #reshapes into vector
        y_i = np.zeros((num_classes), dtype=np.int64)
        
        y_i[class_index] = 1
        y.append(y_i)
    
    #converts to n by num_classes matrix
    y = np.vstack(y)

    return y

def create_folds(x, y, num_folds):
    """
    creates arrays of sub datasets used for cross-validation
    - x and y have to be treated differently
    """
    x_folds = []
    y_folds = []
    
    #roughly the number of observations per class
    fold_size = x.shape[0] / num_folds
    
    #folds have correct distribution of classes
    for i in range(num_folds):
        #prevents uneven distribution of fold-sizes
        lower = i * fold_size
        if (i < num_folds - 1):
            upper = (i + 1) * fold_size
        else:
            upper = x.shape[0]
        
        x_fold = x[int(lower):int(upper)]
        y_fold = y[int(lower):int(upper)]
        
        x_folds.append(x_fold)
        y_folds.append(y_fold)
        
    return x_folds, y_folds
    
def prepare_data(dataset, num_folds):
    #shuffling data - this is the only time we do so
    dataset = dataset.sample(frac=1)
    
    #separates class_labels
    class_labels = dataset.pop('class')
    
    #converts to numpy matrix
    x = dataset.to_numpy()
    class_labels = class_labels.to_numpy()
    
    #normalizes x data to [0,1] range
    #this prevents overflow errors and helps processing
    x = normalize_training_data(x)
    
    #encodes classes into vectors (one hot encoding)
    y = vectorize_classes(class_labels)
    
    #creates an array of sub-datasets used later for cross-validation
    x_folds, y_folds = create_folds(x, y, num_folds)

    return x_folds, y_folds
    
def create_sets(x, y, curr_fold):
    """
    Returns train and test sets for cross-validation
    - Converts x data frames into numpy matrix
    """
    x = copy.deepcopy(x)
    y = copy.deepcopy(y)
    
    #test data, we pop one fold
    test_x = x.pop(curr_fold)
    
    test_y = y.pop(curr_fold)
    
    #train data
    train_x = np.vstack(x)
    train_y = np.vstack(y)
    
    return train_x, train_y, test_x, test_y

# %%
def initialize_weights(dimensions, gradient:bool):
    """
    Initializes weights
    """
    matrices = []
    for i in range(len(dimensions) - 1):

        if gradient:
            weights = np.zeros((dimensions[i+1], dimensions[i]+1))
        else:
            #samples initial theta matrices from a standard normal distribution
            weights = np.random.randn(dimensions[i+1], dimensions[i]+1)
            
        matrices.append(weights)
    return matrices

def forward_propagation(instance, weights_list):
    """
    Forward propagates a training instance 
        returns final layer of network
                and all layers of neural network
    """
    activated_layers = []
    
    #activates input layer
    # act_layer = sigmoid(instance)
    act_layer = np.array(instance)
    
    #reshapes into a vector
    act_layer = act_layer.reshape(-1,1)
    
    #adds a bias term
    act_layer = np.insert(act_layer, 0, values=1)
    
    #reshapes again after insertion
    act_layer = act_layer.reshape(-1,1)
    
    #appends
    activated_layers.append(act_layer)
    
    for weights in weights_list:
        layer = np.dot(weights, act_layer)
        
        #sigmoid function layer
        act_layer = sigmoid(layer)
        
        #adding bias for next phase
        act_layer = np.insert(act_layer, 0, values=1)   
        
        #reshapes after insertion
        act_layer = act_layer.reshape(-1,1)
        
        #store the activated layer for back-propagation
        activated_layers.append(act_layer)
        

    y_hat = act_layer[1:]
    activated_layers[len(activated_layers)-1] = activated_layers[len(activated_layers)-1][1:]
    
    return y_hat, activated_layers

def sigmoid(vector):
    return 1 / (1 + np.exp(-vector))

def compute_cost(y_hat, y_true):
    #dimensions must match
    cost_arr = - y_true * np.log(y_hat) - (1-y_true) * np.log(1-y_hat)
    #common overflow error
    return np.sum(cost_arr)

def regularize_cost(weights_list, batch_size):
    """
    Adds cost for overfitting
        Computes squared sum of all non-bias weights
    """
    sum = 0
    for weights in weights_list:
        weights = weights[:, 1:]
        weights = np.square(weights)
        sum += np.sum(weights)
    
    sum = LAMBDA * sum / (2 * batch_size) 
    return sum

def compute_delta(weights_list, activation_layers, y_true):
    """
    Computes delta vectors for each hidden layer
    """
    #we remove from the end and exclude the first layer
    #we also pop to exclude the final pred. layer
    
    #delta of prediction layer
    delta_next = activation_layers.pop() 
    delta_next = delta_next - y_true.reshape(-1,1)
    
    #stores deltas in the same order as activation layers
    deltas = [delta_next]
    
    while len(activation_layers) > 1:
        thetas = weights_list.pop()
        activation = activation_layers.pop()
        activation_2 = 1 - activation
        
        delta_curr = np.dot(np.transpose(thetas), delta_next)
        delta_curr = delta_curr * activation * activation_2
        
        #removes delta associated with bias node
        delta_curr = delta_curr[1:]
        
        #reshapes into a vector
        # delta_curr = delta_curr.reshape(-1,1)
        
        #stores and resets next delta layer
        deltas.insert(0,delta_curr)
        delta_next = delta_curr
        
    return deltas

def compute_gradients(activation_layers, deltas, previous_gradients):
    """
    - computes gradients based on activations of neurons and next layer deltas
    - adds these to the previous gradients
    """
    #deltas list starts at first hidden layer (second layer), 
    for i in range(len(activation_layers) - 1):
        activation = np.transpose(activation_layers[i])
        delta_t = deltas[i]
        
        gradient_layer = np.dot(delta_t, activation)

        #accumulates to previous gradients
        previous_gradients[i] += gradient_layer
        
    return previous_gradients

def regularize_gradients(weights_list, prev_gradients, batch_size):
    """
    Computes a regularized gradient which punishes large weights
    - adds it to the previous gradient and normalizes it for final updates
    """
    # regularization if any
    reg_gradients = []
    for weights in weights_list:
        reg_gradient = LAMBDA * copy.deepcopy(weights)
        reg_gradient[:,0] = 0
        reg_gradients.append(reg_gradient)
    
    for i in range(len(prev_gradients)):
        reg_gradients[i] += prev_gradients[i]
        
    for i in range(len(reg_gradients)):
        reg_gradients[i] = reg_gradients[i] / batch_size
    
    return reg_gradients

def update_weights(weights_list, reg_gradients):
    """
    Updates weights based on regularized gradients
    """
    for i in range(len(weights_list)):
        weights_list[i] = weights_list[i] - ALPHA * reg_gradients[i]
    
    return weights_list

def record_confusion(output):
    """
    Compiles accuracy, recall, precision, and f1 score based on output
    """
    acc = 0
    prec = 0
    rec = 0
    num_classes = output.shape[0]
    n = np.sum(output[:,1])
    
    #for each class compute performance metrics
    for c in range(num_classes):
        tp, tn, fp, fn = 0,0,0,0
        
        pred_pos = output[c,1]
        act_pos = output[c,2]
        
        pred_neg = np.sum(output[:,1]) - pred_pos
        act_neg = np.sum(output[:,2]) - act_pos
        
        tp = pred_pos if pred_pos <= act_pos else act_pos
        tn = pred_neg if pred_neg <= act_neg else act_neg
        fn = act_pos - tp
        fp = act_neg - tn
        
        acc += (tp + tn) / n
        
        #precision causes errors: division by zero
        if tp + fp == 0:
            prec += 0
        elif tp != 0:
            prec += tp / (tp + fp)
        else:
            prec += 0
            
        #found out later that recall also causes errors
        rec += tp / (tp + fn) if tp != 0 else 0
    
    acc = acc / num_classes
    prec = prec / num_classes
    rec = rec / num_classes
    
    f1 = 2 * prec * rec / (prec + rec) if prec + rec != 0 else 0
    
    ret = {
        'accuracy':acc,
        'precision':prec,
        'recall':rec,
        'f1-score':f1
    } 
    return ret
    
def neural_network(dataset, architecture, num_folds, batch_size, num_iterations):
    """
    Runs a neural network based on a dataset, architecture
    - Cross validation is used for training with num_folds
    """
    x,y = prepare_data(dataset, num_folds)
    metrics_sum = []

    for curr_fold in range(num_folds):
        train_x, train_y, test_x, test_y = create_sets(
            x, y, curr_fold)
        
        weights_list = initialize_weights(architecture, gradient=False)
        empty_gradient = initialize_weights(architecture, gradient=True)
        
        gradients = copy.deepcopy(empty_gradient)
        
        cost_list = np.array([])
        cost = 0
        
        instances_seen = 0
        
        #only for the final model
        if FINAL_MODEL:
            test_costs = []
        
        #stopping criteria:
            #loops num_iterations times over the dataset
        for _ in range(num_iterations):
            
            #loops over every iteration of dataset, updates weights when i is a multiple of batch_size
            for i in range(train_x.shape[0]): 
                x_i = train_x[i].reshape(-1,1)
                y_i = train_y[i].reshape(-1,1)
                
                #forward propagation
                y_hat, activation_layers = forward_propagation(x_i, weights_list)
                
                #adds cost for this training instance
                cost += compute_cost(y_hat, y_i)

                #backward pass
                deltas = compute_delta(weights_list.copy(), activation_layers.copy(), y_i)
                
                gradients = compute_gradients(activation_layers, deltas, gradients)
                
                #updates for the following conditional
                instances_seen += 1
                
                #updates weights based on regularized gradients
                if (instances_seen % batch_size) == 0:
                    #regularizes cost and records:
                    cost = cost / batch_size
                    cost += regularize_cost(weights_list, batch_size)
                    cost_list = np.append(cost_list, cost)
                    #resets cost
                    cost = 0
                    
                    #regularizes gradients
                    reg_gradients = regularize_gradients(weights_list, gradients, batch_size)
                    gradients = copy.deepcopy(empty_gradient)
                
                    #updates weights
                    weights_list = update_weights(weights_list, reg_gradients)
                    
                    #observes and records cost for test data - same as below
                    if FINAL_MODEL:
                        test_cost = 0
                        #first column is class, second is y_hat, third is true
                        output = np.c_[np.arange(0, test_y.shape[1]), np.zeros(test_y.shape[1]), np.zeros(test_y.shape[1])]
                        
                        #forward propagation of test set
                        for i in range(test_x.shape[0]):
                            x_i = test_x[i].reshape(-1,1)
                            y_i = test_y[i].reshape(-1,1)
                            
                            y_hat, activation_layers = forward_propagation(x_i, weights_list)
                            test_cost += compute_cost(y_hat, y_i)
                            
                            y_hat_index = np.argmax(y_hat)
                            y_true_index = np.argmax(y_i)
                            output[y_hat_index, 1] += 1
                            output[y_true_index, 2] += 1 
                            
                        test_cost += regularize_cost(weights_list, test_x.shape[0])
                        test_costs.append(test_cost)
                    
        
        #POST-TRAINING evaluation: 
        test_cost = 0
        
        #first column is class, second is y_hat, third is true
        output = np.c_[np.arange(0, test_y.shape[1]), np.zeros(test_y.shape[1]), np.zeros(test_y.shape[1])]
        
        #forward propagation of test set
        for i in range(test_x.shape[0]):
            x_i = test_x[i].reshape(-1,1)
            y_i = test_y[i].reshape(-1,1)
            
            y_hat, activation_layers = forward_propagation(x_i, weights_list)
            test_cost += compute_cost(y_hat, y_i)
            
            y_hat_index = np.argmax(y_hat)
            y_true_index = np.argmax(y_i)
            output[y_hat_index, 1] += 1
            output[y_true_index, 2] += 1 
            
        test_cost += regularize_cost(weights_list, test_x.shape[0])

        #evaluates and records metrics
        scores = record_confusion(output)

        #metrics for one fold
        metrics = {
            'training-costs': cost_list,
            'test-cost-final':test_cost,
            'instances-seen':(instances_seen / num_folds)
        }
        metrics.update(scores)
        
        if FINAL_MODEL:
            temp = {
                'test-costs': test_costs
            }
            metrics.update(temp)
            
        metrics_sum.append(metrics)
        
    #compiles metrics and model hyperparameters
    test = metrics_sum[0]
    stats = {
        'accuracy': [m['accuracy'] for m in metrics_sum],
        'precision': [m['precision'] for m in metrics_sum],
        'recall': [m['recall'] for m in metrics_sum],
        'f1-score': [m['f1-score'] for m in metrics_sum],
        'training-costs': [m['training-costs'] for m in metrics_sum],
        'test-cost-final': [m['test-cost-final'] for m in metrics_sum],
        'instances-seen': np.min([m['instances-seen'] for m in metrics_sum]),
        'batch-size':batch_size,
        'num-iterations': num_iterations,
        'alpha': ALPHA,
        'lambda': LAMBDA,
        'architecture': architecture
    }
    if FINAL_MODEL:
        temp = {
            'test-costs': [m['test-costs'] for m in metrics_sum]
        }
        stats.update(temp)

    return stats

# %% [markdown]
# Creates graphs/records for analysis

# %%
def plot_test_cost(stats, batch_size, title):
    fig, ax = plt.subplots()
    costs = stats['test-costs']
    
    num_obs = stats['instances-seen']
    
    for i in range(len(costs)):
        costs[i] = np.array(costs[i])
        costs[i].resize(int(num_obs), refcheck=False)
    
    
    costs = np.stack(costs, axis=0)
    y = np.mean(costs, axis=0)

    # mean_costs = np.mean(costs, axis=0)
    
    x = np.arange(0, int(num_obs))
    
    plt.plot(x,y, color='orange')
    
    plt.xlabel('Number of training iterations')
    plt.ylabel('(Regularized) Training Cost')
    
    p_title = 'Training Cost for ' + title
    plt.title(p_title)
    
    file_name = '../figures/' + title + '-test-cost.png'
    plt.savefig(file_name)
    
    plt.show()
    return


def vis_convergence(stats, num_obs=-1):
    """
    Helps visualize the cost function
    """
    if num_obs == -1:
        num_obs = stats['instances-seen']
    
    costs = stats['training-costs']
    
    for i in range(len(costs)):
        costs[i] = np.array(costs[i])
        costs[i].resize(int(num_obs), refcheck=False)
    
    # costs = stats['training-costs']
    
    costs = np.stack(costs, axis=0)
    y = np.mean(costs, axis=0)
    x = np.arange(0, y.shape[0])
    
    fig, ax = plt.subplots()
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()
    ax.plot(x,y)
    
    return fig

#same thing as above but for tuning hyperparameters
def vis_convergence_tune(stats, num_obs=-1):
    """
    Helps visualize the cost function
    """
    if num_obs == -1:
        num_obs = stats['instances-seen']
    
    costs = stats['training-costs']
    
    for i in range(len(costs)):
        costs[i] = np.array(costs[i])
        costs[i].resize(int(num_obs), refcheck=False)
    
    # costs = stats['training-costs']
    
    costs = np.stack(costs, axis=0)
    y = np.mean(costs, axis=0)
    x = np.arange(0, y.shape[0])
    
    fig, ax = plt.subplots()

    plt.tight_layout()
    ax.plot(x,y)
    
    plt.show()

def compile_convergence_plots(convergence_plots, title):
    fig, axes = plt.subplots(1, 6, figsize=(18, 6))
    fig.subplots_adjust(bottom=0.2, wspace=0)
    
    for i in range(6):
        axes[i].imshow(convergence_plots[i].get_figure().canvas.renderer.buffer_rgba())        
        axes[i].set_xlabel(f'{i+1}')
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    
    file_name = '../figures/' + title + '_converge.png'
    
    plt.savefig(file_name, bbox_inches='tight')
    
    plt.show()
    
def compress_metrics(stats, log=False):
    acc = np.mean(stats['accuracy'])
    rec = np.mean(stats['recall'])
    prec = np.mean(stats['precision'])
    f1 = np.mean(stats['f1-score'])
    
    mean_test_cost = np.mean(stats['test-cost-final'])
    
    if log:
        print(f'accuracy: {acc:.4f}')
        print(f'recall: {rec:.4f}')
        print(f'precision: {prec:.4f}')
        print(f'F1-Score: {f1:.4f}')
        print(f'mean final test cost: {mean_test_cost:.4f}')
    
    ret = {
        'm_accuracy':acc,
        'm_recall':rec,
        'm_precision':prec,
        'm_f1-score':f1,
        'm_test-cost-final':mean_test_cost
    }
    stats.update(ret)
    return

def accuracy_spread(model_data, title=''):
    accuracies = []
    for mod in model_data:
        #accuracy box plot
        accuracies.append(mod['accuracy'])
    
    fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(6, 3), sharey=True)
    
    for i, ax in enumerate(axes):
        ax.boxplot(accuracies[i])
        ax.set_xlabel(f'Model {i+1}')
        
        ax.set_xticklabels([])
        ax.tick_params(axis='both', which='both', length=0)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        ax.grid(True)
    
    axes[0].set_ylabel('Accuracy Spread')
        
    plt.suptitle('Accuracy Spread')
    
    plt.tight_layout()
    
    file_name = '../figures/' + title + '_acc_spread.png'
    
    plt.savefig(file_name)
    plt.show()

def summarize_to_latex(model_data):    
    data = {}
    
    for i, m in zip(range(6), model_data):
        one = round(m['alpha'], 3)
        two = round(m['lambda'], 4)
        three = m['architecture']
        four = round(m['m_accuracy'], 4)
        five = round(m['m_f1-score'], 4)
        six = round(m['m_test-cost-final'],2)
        seven = 0
        
        data[f'Model {i+1}'] = [one, two, three, four, five, six, seven]
    
    #seven columns
    metrics = ['Learning Rate $\\alpha$', 'Regularization $\\lambda$', 'Architecture', 'Mean Accuracy',
               'Mean F1-score', 'Mean Test Cost', 'Converges Around']
    
    df = pd.DataFrame(data, index=metrics)

    latex_table = df.to_latex(float_format="%.4f")

    print(latex_table)

# %% [markdown]
# For each dataset, I visually confirm that the training cost flatlines
# - I train each model individually and by hand to try and achieve the highest results, recording what I've done

# %% [markdown]
# Wine dataset:

# %%
def wine():
    global ALPHA, LAMBDA
    
    wine = pd.read_csv('datasets/wine.csv', sep='\s+')
    
    #models:
    #arch 4     a = 0.1, l = 0.1
    #arch 4,4   a = 0.5, l = .02 
    #arch 2     a = 0.5, l = 0.01
    #arch 1     a = 0.7, l = 0.01
    #arch 1,1,1 a = 0.1, l = 0.03
    #arch 64,32 a = 0.5, l = 0.01

    arch_list = [
        [13,4,3],
        [13,4,4,3],
        [13,2,3],
        [13,1,3],
        [13,32,3],
        [13,64,3]
    ]
    alpha_list = [0.1, 0.5, 0.5, 0.7, 0.1, 0.5]
    lambda_list = [0.1, 0.02, 0.01, 0.01, 0.03, 0.01]

    model_data = []
    convergence_plots = []
        
    for i in range(6):
        ALPHA = alpha_list[i]
        LAMBDA = lambda_list[i]
        model = neural_network(wine, arch_list[i], num_folds=10, batch_size=10, num_iterations=100)
        compress_metrics(model, log=False)
        
        convergence_plots.append(vis_convergence(model))
        
        model_data.append(model)

    #for the report
    accuracy_spread(model_data, title='wine')
    summarize_to_latex(model_data)
    compile_convergence_plots(convergence_plots, title='wine')
    
    reset_hyperparameters()

def wine_final():
    global FINAL_MODEL, LAMBDA, ALPHA
    FINAL_MODEL = True

    #can't find the global for some reason
    wine = pd.read_csv('datasets/wine.csv', sep='\s+')

    ALPHA = 0.5
    LAMBDA = 0.01
    model = neural_network(wine, [13,64,3], num_folds=10, batch_size=10, num_iterations=100)
    compress_metrics(model, log=False)
    plot_test_cost(model, 10, 'wine')

    FINAL_MODEL = False
    reset_hyperparameters()
    return

# wine()
# wine_final()

# %%
def house():
    global LAMBDA, ALPHA
    house = pd.read_csv('datasets/house_votes_84.csv')
    #models:
    #arch 8     a = 0.2, l = 0.25
    #arch 2,2   a = 0.5, l = 0
    #arch 16,8  a = 0.3, l = 0.02
    #arch 1,8   a = 0.5, l = 0.01
    #arch 4,4,4,4  a = 0.25, l = 0.01   #very big jump l=0.01 to 0.02
    #arch 32,32 a = 0.5, l = 0.05

    arch_list = [
        [16,8,2],
        [16,2,2,2],
        [16,16,8,2],
        [16,1,8,2],
        [16,4,4,4,4,2],
        [16,32,32,2]
    ]
    alpha_list = [0.2, 0.5, 0.3, 0.5, 0.25, 0.5]
    lambda_list = [0.25, 0, 0.02, 0.01, 0.01, 0.05]

    model_data = []
    convergence_plots = []
        
    for i in range(6):
        ALPHA = alpha_list[i]
        LAMBDA = lambda_list[i]
        model = neural_network(house, arch_list[i], num_folds=10, batch_size=10, num_iterations=100)
        compress_metrics(model, log=False)
        
        convergence_plots.append(vis_convergence(model))
        
        model_data.append(model)

    #for the report
    accuracy_spread(model_data, title='house')
    summarize_to_latex(model_data)
    compile_convergence_plots(convergence_plots, title='house')
    
    reset_hyperparameters()
    return
    
def house_final():
    global FINAL_MODEL, LAMBDA, ALPHA
    FINAL_MODEL = True

    #can't find the global for some reason
    house = pd.read_csv('datasets/house_votes_84.csv')

    ALPHA = 0.5
    LAMBDA = 0
    model = neural_network(house, [16,2,2,2], num_folds=10, batch_size=10, num_iterations=100)
    compress_metrics(model, log=False)
    plot_test_cost(model, 10, 'house')

    FINAL_MODEL = False
    reset_hyperparameters()
    return

# house()
# house_final()

# %% [markdown]
# Cancer Dataset

# %%
def cancer():
    global LAMBDA, ALPHA

#models:
#arch 8,4   a = 0.2, l = 0.02
#arch 2     a = 0.5, l = 0
#arch 4,4   a = 0.3, l = 0.001
#arch 32,32 a = 0.2, l = 0.05
#arch 128  a = 0.25, l = 0      #very big jump l=0.01 to 0.02
#arch 1, 32 a = 0.5, l = 0.05
    arch_list = [
        [9,8,4,2],
        [9,2,2],
        [9,4,4,2],
        [9,32,32,2],
        [9,128,2],
        [9,1,32,2]
    ]
    alpha_list = [0.2, 0.5, 0.3, 0.2, 0.25, 0.5]
    lambda_list = [0.02, 0, 0.001, 0.05, 0, 0.05]

    model_data = []
    convergence_plots = []
        
    for i in range(6):
        ALPHA = alpha_list[i]
        LAMBDA = lambda_list[i]
        model = neural_network(cancer, arch_list[i], num_folds=10, batch_size=10, num_iterations=100)
        compress_metrics(model, log=False)
        
        convergence_plots.append(vis_convergence(model))
        
        model_data.append(model)

    #for the report
    accuracy_spread(model_data, title='cancer')
    summarize_to_latex(model_data)
    compile_convergence_plots(convergence_plots, title='cancer')
    
    reset_hyperparameters()

def cancer_final():
    global FINAL_MODEL, LAMBDA, ALPHA
    FINAL_MODEL = True

    #can't find the global for some reason
    cancer = pd.read_csv('datasets/cancer.csv', sep='\s+')

    #renames class column so my network can find output:
    cancer.rename(columns={'Class': 'class'}, inplace=True)

    ALPHA = 0.2
    LAMBDA = 0.02
    model = neural_network(cancer, [9,8,4,2], num_folds=10, batch_size=10, num_iterations=100)
    compress_metrics(model, log=False)
    plot_test_cost(model, 10, 'cancer')

    FINAL_MODEL = False
    reset_hyperparameters()
    return

# cancer()
# cancer_final()

# %% [markdown]
# Contraceptive-Use Dataset

# %%
def contraceptive():
    global LAMBDA, ALPHA

    #models:
    #arch 8,8   a = 0.025, l = 0.001
    #arch 32,32 a = 0.015, l = 0
    #arch 2     a = 0.03, l = 0.001
    #arch 64    a = 0.0005, l = 0.01
    #arch 64,8  a = 0.005, l = 0.01      #very big jump l=0.01 to 0.02
    #arch 8,4,4 a = 0.001, l = 0.2
    arch_list = [
        [9,8,8,3],
        [9,32,32,3],
        [9,2,3],
        [9,64,3],
        [9,64,8,3],
        [9,8,4,4,3]
    ]
    alpha_list = [0.2, 0.5, 0.3, 0.2, 0.25, 0.5]
    lambda_list = [0.02, 0, 0.001, 0.05, 0, 0.05]

    model_data = []
    convergence_plots = []
        
    for i in range(6):
        ALPHA = alpha_list[i]
        LAMBDA = lambda_list[i]
        model = neural_network(contraceptive, arch_list[i], num_folds=10, batch_size=10, num_iterations=100)
        compress_metrics(model, log=False)
        
        convergence_plots.append(vis_convergence(model))
        
        model_data.append(model)

    #for the report
    accuracy_spread(model_data, title='contraceptive')
    summarize_to_latex(model_data)
    compile_convergence_plots(convergence_plots, title='contraceptive')

    reset_hyperparameters()

def contraceptive_final():
    global FINAL_MODEL, LAMBDA, ALPHA
    FINAL_MODEL = True

    #can't find the global for some reason
    contraceptive = pd.read_csv('datasets/contraceptive+method+choice/cmc.data', header=None)

    #renames class column so my network can find output:
    contraceptive.rename(columns={9: 'class'}, inplace=True)

    ALPHA = 0.2
    LAMBDA = 0.02
    model = neural_network(contraceptive, [9,8,8,3], num_folds=10, batch_size=10, num_iterations=100)
    compress_metrics(model, log=False)
    plot_test_cost(model, 10, 'Contraceptive Method Choice')

    FINAL_MODEL = False
    reset_hyperparameters()
    return

# contraceptive()
# contraceptive_final()

# %% [markdown]
# Testing Backpropagation

# %%
def array_to_string(array, rank):
    # if type(array) == np.float64:
    #     return f'{array:.5f}'
    
    out = ''
    if rank == 1:
        for num in array:
            out += f'{num[0]:.5f}' + '   '
    elif rank == 2:
        for arr in array:
            for num in arr:
                out += f'{num:.5f}' + '   '
            out += '\n'
    return out

def log_out(title, expected, received, tail=False):
    if isinstance(received, np.float64):
        received = f'{received:.5f}'
    else:
        received = array_to_string(received, len(received.shape))
        
    if tail:
        tail = '\n'
    else:
        tail = ''

    print(title + '\n\n' + 'expected'.rjust(15).ljust(55) + expected + '\n' +
            'received'.rjust(15).ljust(55) + tail + received)
    
def run_ex1():
    x = [0.13, 0.42]
    y = [0.90, 0.23]
    
    # network_architecture = [1,2,1]
    
    weights1 = [
        [0.4, 0.1],
        [0.3, 0.2]
    ]
    weights2 = [
        [0.7, 0.5, 0.6]
    ]
    
    weights_list = [np.array(weights1), np.array(weights2)]
    
    gradients = [np.zeros((2,2)), np.zeros((1,3))]
    
    print('STEP'.ljust(55) + 'RESULT'.ljust(35))
    
    y_hat_1, act_layers1 = forward_propagation(x[0], weights_list)
    
    log_out('forward propagation instance 0', '0.79403', y_hat_1.reshape(1,-1))
    
    #example logs
    # y_hat_1, actlist1 = forward_propagation(x[0], weights_list)
    # log_out('forward propagation instance 0', '0.83318   0.84132', y_hat_1.reshape(1,-1))

    # cost_1 = compute_cost(y_hat_1, y[0])
    # log_out('cost for instance 0', '0.791', cost_1)
    
    # y_hat_2, actlist2 = forward_propagation(x[1], weights_list)
    # log_out('forward propagation instance 1', '0.82953   0.83832', y_hat_2.reshape(1,-1))
    
    # cost_2 = compute_cost(y_hat_2, y[1])
    # log_out('cost for instance 1', '1.944', cost_2)
    
    # cost = (cost_1 + cost_2) / 2
    # cost += regularize_cost(weights_list, 2)
    # log_out('cumulative cost for step', '1.90351', cost)
    
    y_hat_2, act_layers2 = forward_propagation(x[1], weights_list)
    # print(f'instance 1, my_result: {y_hat_2[0]}, true: {.79597}')
    
    log_out('forward propagation instance 1', '0.79597', y_hat_2.reshape(1,-1))
    
    cost_1 = compute_cost(y_hat_1, y[0])
    # print(f'expected cost 1, 0.366, received: {cost_1}')
    
    log_out('cost for instance 0', '0.366', cost_1)
    
    cost_2 = compute_cost(y_hat_2, y[1])
    # print(f'expected cost 2, 1.276, received: {cost_2}')
    
    log_out('cost for instance 1', '1.276', cost_2)
    
    sum_cost = (cost_1 + cost_2) / 2
    # print(f'expected J: 0.82098, received: {sum_cost}')
    
    log_out('cumulative cost for step', '0.82098', sum_cost)
    
    #backward pass
    print('\nbackpass instance 1 -----------------')
    
    deltas = compute_delta(weights_list.copy(), act_layers1.copy(), np.array(y[0]))
    print(f'expected delta layer3: [-0.10597], received: {deltas[1]}')
    print(f'expected delta layer2: [-0.01270   -0.01548], \n received{deltas[0]}\n')
    
    gradients1 = compute_gradients(act_layers1, deltas, copy.deepcopy(gradients))
    print(f'expected gradient theta 2: -0.10597  -0.06378  -0.06155  \nreceived {gradients1[1]}')
    
    print(f'\nexpected gradient theta 1:\n -0.01270  -0.00165 \n -0.01548  -0.00201  \nreceived {gradients1[0]}')
    
    
    print('\nbackpass instance 2 -----------------')
    
    deltas = compute_delta(weights_list.copy(), act_layers2.copy(), np.array(y[1]))
    print(f'expected delta layer3: [0.56597], received: {deltas[1]}')
    print(f'expected delta layer2: [0.06740   0.08184], \n received{deltas[0]}\n')
    
    gradients2= compute_gradients(act_layers2, deltas, copy.deepcopy(gradients))
    print(f'expected gradient theta 2: 0.56597  0.34452  0.33666  \nreceived {gradients2[1]}')
    
    print(f'\nexpected gradient theta 1:\n 0.06740  0.02831 \n 0.08184  0.03437  \nreceived {gradients2[0]}')
    
    
    print('\ncummulative gradients -----------------')
    deltas = compute_delta(weights_list.copy(), act_layers2.copy(), np.array(y[1]))
    gradients = compute_gradients(act_layers2, deltas, gradients1)
    
    for i in range(len(gradients)):
        gradients[i] = gradients[i]/2
    
    print(f'expected gradient theta 2: 0.23000  0.14037  0.13756  \nreceived {gradients[1]}')
    
    print(f'\nexpected gradient theta 1:\n 0.02735  0.01333 \n 0.03318  0.01618  \nreceived {gradients[0]}')
   
# my numbers match for example 1 
run_ex1()

# %%
def run_ex2():
    global LAMBDA
    LAMBDA = 0.250
    
    weights1 = [
        [0.42000, 0.15000, 0.40000],
        [0.72000, 0.10000, 0.54000],
        [0.01000, 0.19000, 0.42000],
        [0.30000, 0.35000, 0.68000]
    ]
    weights2 = [
        [0.21000, 0.67000, 0.14000, 0.96000, 0.87000],
        [0.87000, 0.42000, 0.20000, 0.32000, 0.89000],
        [0.03000, 0.56000, 0.80000, 0.69000, 0.09000]
    ]
    weights3 = [
        [0.04000, 0.87000, 0.42000, 0.53000],
        [0.17000, 0.10000, 0.95000, 0.69000]
    ]
    weights_list = [np.array(weights1),np.array(weights2),np.array(weights3)]
    
    empty_gradients = initialize_weights(dimensions=[2,4,3,2], gradient=True)
    
    x = [np.array([0.32000, 0.68000]), np.array([0.83000, 0.02000])]
    y = [np.array([0.75000, 0.98000]), np.array([0.75000, 0.28000])]
    
    x = [x_i.reshape(-1,1) for x_i in x]
    y = [y_i.reshape(-1,1) for y_i in y]
    
    print('STEP'.ljust(55) + 'RESULT'.ljust(35))
    
    y_hat_1, actlist1 = forward_propagation(x[0], weights_list)
    log_out('forward propagation instance 0', '0.83318   0.84132', y_hat_1.reshape(1,-1))

    cost_1 = compute_cost(y_hat_1, y[0])
    log_out('cost for instance 0', '0.791', cost_1)
    
    y_hat_2, actlist2 = forward_propagation(x[1], weights_list)
    log_out('forward propagation instance 1', '0.82953   0.83832', y_hat_2.reshape(1,-1))
    
    cost_2 = compute_cost(y_hat_2, y[1])
    log_out('cost for instance 1', '1.944', cost_2)
    
    cost = (cost_1 + cost_2) / 2
    cost += regularize_cost(weights_list, 2)
    log_out('cumulative cost for step', '1.90351', cost)
    
    print('Back Pass instance 0------------------\n')
    deltas = compute_delta(copy.deepcopy(weights_list), copy.deepcopy(actlist1), y[0])
    log_out('delta layer 4', '0.08318   -0.13868', deltas[2].reshape(1,-1))
    log_out('delta layer 3', '0.00639   -0.00925   -0.00779', deltas[1].reshape(1,-1))
    log_out('delta layer 2', '-0.00087   -0.00133   -0.00053   -0.00070', deltas[0].reshape(1,-1))
    
    gradients = compute_gradients(actlist1, deltas, copy.deepcopy(empty_gradients))
    log_out('gradient theta 3', '\n0.08318  0.07280  0.07427  0.06777\n-0.13868  -0.12138  -0.12384  -0.11300', gradients[2], True)
    log_out('gradient theta 2', '\n0.00639  0.00433  0.00482  0.00376  0.00451\n-0.00925  -0.00626  -0.00698  -0.00544  -0.00653\n-0.00779  -0.00527  -0.00587  -0.00458  -0.00550', gradients[1], True)
    log_out('gradient theta 1', '\n-0.00087  -0.00028  -0.00059\n-0.00133  -0.00043  -0.00091\n-0.00053  -0.00017  -0.00036\n-0.00070  -0.00022  -0.00048', gradients[0], True)
    
    print('Back Pass instance 1------------------\n')
    deltas = compute_delta(copy.deepcopy(weights_list), copy.deepcopy(actlist2), y[1])
    log_out('delta layer 4', '0.07953   0.55832', copy.deepcopy(deltas[2]).reshape(1,-1))
    log_out('delta layer 3', '0.01503   0.05809   0.06892', copy.deepcopy(deltas[1]).reshape(1,-1))
    log_out('delta layer 2', '0.01694   0.01465   0.01999   0.01622', copy.deepcopy(deltas[0]).reshape(1,-1))
    
    #somethings going wrong in the [1:] obs of each gradient TODO
    gradients2 = compute_gradients(actlist2, deltas, copy.deepcopy(empty_gradients))
    log_out('gradient theta 3', '\n0.07953  0.06841  0.07025  0.06346\n0.55832  0.48027  0.49320  0.44549', gradients2[2], True)
    log_out('gradient theta 2', '\n0.01503  0.00954  0.01042  0.00818  0.00972\n0.05809  0.03687  0.04025  0.03160  0.03756\n0.06892  0.04374  0.04775  0.03748  0.04456\n', gradients2[1], True)
    log_out('gradient theta 1', '\n0.01694  0.01406  0.00034\n0.01465  0.01216  0.00029\n0.01999  0.01659  0.00040\n0.01622  0.01346  0.00032', gradients2[0], True)
    
    print('Regularized Gradients------------------\n')
    gradients = compute_gradients(actlist2, deltas, gradients)

    regularized_gradients = regularize_gradients(weights_list, gradients, 2)
    log_out('reg gradient theta 3', '\n0.08135  0.17935  0.12476  0.13186\n0.20982  0.19195  0.30343  0.25249', regularized_gradients[2], True)
    log_out('reg gradient theta 2', '\n0.01071  0.09068  0.02512  0.12597  0.11586\n0.02442  0.06780  0.04164  0.05308  0.12677\n0.03056  0.08924  0.12094  0.10270  0.03078 ', regularized_gradients[1], True)
    log_out('reg gradient theta 1', '\n0.00804  0.02564  0.04987\n0.00666  0.01837  0.06719\n0.00973  0.03196  0.05252\n0.00776  0.05037  0.08492', regularized_gradients[0], True)
    
# my numbers match for example 2
run_ex2()


