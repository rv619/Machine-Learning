import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import copy, deepcopy
from math import e

def split (dataset, training_ratio):
    n = len (dataset) # Size of dataset
    r = int (training_ratio * n) # No. of training examples
    np.random.shuffle (dataset) # Shuffling the dataset
    return dataset[:r,:], dataset[r:,:]

def phi (x, a = 1.0): # Activation function
    return (1. / (1. + np.exp((-a) * x))) #sigmoid
    
def phi_dash (x, a = 1.0): # Derivative of activation function
    phi_x = phi (x, a)
    return a * phi_x * (1 - phi_x)

def feed_forward (x, w, h, bias):
    u = np.array ([None]*(h+2))
    phi_u = np.array ([None]*(h+2))
    phi_dash_u = np.array ([None]*(h+2))
    u[0] = np.insert (deepcopy (x), 0, bias, axis = 1)
    phi_u[0] = u[0]
    for i in range (h+1):
        u[i+1] = np.dot (phi_u[i], np.transpose (w[i]))
        phi_dash_u[i+1] = phi_dash (u[i+1]) # No bias yet
        # Add bias
        if i != h:
            u[i+1] = np.insert (u[i+1], 0, bias, axis = 1)
        phi_u[i+1] = phi (u[i+1])
    return phi_u, phi_dash_u
    
def back_propagate (error, w, phi_dash_u, h):
    delta = np.array ([None]*(h+1))
    delta[h] = np.multiply (error, phi_dash_u[h+1])
    for i in range (h-1,-1,-1):
        del_cur = np.dot (delta[i+1], w[i+1])
        del_cur = np.delete (del_cur, 0, axis = 1)
        delta[i] = np.multiply (phi_dash_u[i+1], del_cur)
    return delta
    
def update_weights (w, phi_u, delta, h, learning_rate):
    for i in range (h+1):
        del_w = np.dot (np.transpose (delta[i]), phi_u[i])
        w[i] += learning_rate * del_w
    return w
    
def train_and_test (train_data, test_data, hidden, h, d, c, class_mapping, bias, iterations, learning_rate = 0.1, eps = 0.001, epoch = 10000):
    n_train = len (train_data) # Size of training dataset
    n_test = len (test_data) # Size of testing dataset
    # Split the data
    x_train, y_train = np.array (train_data[:,:d], dtype = np.float), train_data[:,d:]
    x_test, y_test = np.array (test_data[:,:d], dtype = np.float), test_data[:,d:]
    w = np.array ([None]*(h+1))
    dimensions = hidden.copy()
    dimensions.insert (0, d)
    dimensions.append (c)
    w = np.array ([None]*(h+1))
    for i in range (h+1):
        w[i] = np.random.random((dimensions[i+1], dimensions[i]+1))
    in_sample = []
    out_of_sample = []
    it = 0
    while it < iterations:
        E_train = 0
        for i in range (n_train):
            x_cur = x_train[i:i+1,:]
            phi_u, phi_dash_u = feed_forward (x_cur, w, h, bias)
            y_mask = np.array([0.0] * c) # one-hot representation
            y_mask[class_mapping[y_train[i,0]]] = 1.0
            error = y_mask - phi_u[h+1]
            delta = back_propagate (error, w, phi_dash_u, h)
            w = update_weights (w, phi_u, delta, h, learning_rate)
            e = 0.5 * np.sum (np.multiply (error, error))
            E_train += e / n_train
            
        # Calculating in sample error
        phi_u, phi_dash_u = feed_forward (x_train, w, h, bias)
        y_mask = np.zeros((n_train,c),dtype=np.float)
        for i in range (n_train):
            y_mask[class_mapping[y_train[i,0]]] = 1.0
        error = y_mask - phi_u[h+1] 
        e = 0.5 * np.sum (np.multiply (error, error))
        E_train = e / n_train
        
        # Calculating out of sample error
        phi_u, phi_dash_u = feed_forward (x_test, w, h, bias)
        y_mask = np.zeros((n_test,c),dtype=np.float)
        for i in range (n_test):
            y_mask[class_mapping[y_test[i,0]]] = 1.0
        error = y_mask - phi_u[h+1] 
        e = 0.5 * np.sum (np.multiply (error, error))
        E_test = e / n_test
        
        in_sample.append (E_train)
        out_of_sample.append (E_test)
        
        it += 1
        
    n_iter = np.linspace(1, it, it)
    
    plt.plot(n_iter, in_sample)
    plt.plot(n_iter, out_of_sample)
    
    return w
    
def run (filename, hidden, h, iterations = 1000, ratio = 0.8, bias = 1.0):
    # Read the csv file
    dataset = pd.read_csv (filename).to_numpy()
    dataset = dataset[1:,:]
    # Get no. of examples and features
    n = dataset.shape[0]
    d = dataset.shape[1] - 1
    # Get the classes and its count
    classes = []
    for i in range (len (dataset)):
        classes.append (dataset[i,d])
    classes = list (set (classes))
    c = len(classes)
    # Split the data set
    train_data, test_data = split (dataset, ratio)
    # Map classes
    class_mapping = {}
    for i in range (len(classes)):
        class_mapping [classes[i]] = i
    # Train the data
    w = train_and_test (train_data, test_data, hidden, h, d, c, class_mapping, bias, iterations)

def main ():
    file = 'iris.csv'
    h = 2 # no. of hidden layers
    hidden = [20,20] # no. of nodes in hidden layers
    ratio = 0.2
    times = 1
    iterations = 1000
    for _ in range (times):
        run (file, hidden, h, iterations, ratio)
    plt.savefig ('graphhhhh.png', dpi=2000)
    plt.show()

main()