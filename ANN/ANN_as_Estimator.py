import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import copy, deepcopy

def split (dataset, training_ratio):
    n = len (dataset) # Size of dataset
    r = int (training_ratio * n) # No. of training examples
    np.random.shuffle (dataset) # Shuffling the dataset
    return dataset[:r,:], dataset[r:,:]

def phi (x, a = 1): # Activation function
    return (1 / (1 + np.exp((-a) * x))) #sigmoid
    
def phi_dash (x, a = 1): # Derivative of activation function
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
        else:
            phi_u[i+1] = u[i+1]
        
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
    
def train (train_data, hidden, h, d, bias, learning_rate = 0.01, eps = 0.001, epoch = 1000):
    n = len (train_data) # Size of training dataset
    # Split the data
    x, y = train_data[:,:d], train_data[:,d:]
    it = 0
    w = np.array ([None]*(h+1))
    dimensions = hidden.copy()
    dimensions.insert (0, d)
    dimensions.append (1) # c = 1 for regression
    w = np.array ([None]*(h+1))
    for i in range (h+1):
        w[i] = np.random.random((dimensions[i+1], dimensions[i]+1))
    while it < epoch:
        E = 0
        for i in range (n):
            x_cur = x[i:i+1,:]
            phi_u, phi_dash_u = feed_forward (x_cur, w, h, bias)
            y_n = np.array([y[i,0]])
            error = y_n - phi_u[h+1]
            delta = back_propagate (error, w, phi_dash_u, h)
            w = update_weights (w, phi_u, delta, h, learning_rate)
            e = 0.5 * np.sum (np.multiply (error, error))
            E += e / n
        if E < eps:
            break
        it += 1
    return w
    
def test (test_data, w, h, d, bias):
    n = len (test_data) # Size of testing dataset
    # Split the data
    x, y = test_data[:,:d], test_data[:,d:]
    phi_u, phi_dash_u = feed_forward (x, w, h, bias)
    sum_sq_r, sum_sq_t = 0, 0
    y_avg = 0
    for i in range (n):
        y_avg += phi_u[h+1][i][0]
    y_avg /= n
    for i in range (n):
        sum_sq_r += (y[i,0] - phi_u[h+1][i][0]) ** 2
        sum_sq_t += (y[i,0] - y_avg) ** 2
    r_sq = sum_sq_r / sum_sq_t
    # Display the test results
    print ("R_Square = " + str(round(r_sq, 4)))
    
def run (filename, hidden, h, d, bias = 1.0):
    # Read the csv file
    dataset = pd.read_csv (filename).to_numpy()
    # Split the data set
    train_data, test_data = split (dataset, 0.8)
    # Train the data
    w = train (train_data, hidden, h, d, bias)
    # Test the data
    test (test_data, w, h, d, bias)

def call (h, hidden):
    file1 = 'DatasetMulti-Variate1.csv'
    file2 = 'DatasetMulti-Variate2.csv'
    d = 2 # no. of features
    print ("No. of hidden layers = ", h)
    print ("----------------------------------------------")
    print ("Results (" + file1 + ")")
    print ("----------------------------------------------\n")
    run (file1, hidden, h, d)
    print ("\n----------------------------------------------")
    print ("Results (" + file2 + ")")
    print ("----------------------------------------------\n")
    run (file2, hidden, h, d)
    print ("\n----------------------------------------------")
    
def main ():
    h = 1 # no. of hidden layers
    hidden = [2] # no. of nodes in hidden layers
    call (h, hidden)
    h = 2 # no. of hidden layers
    hidden = [2,2] # no. of nodes in hidden layers
    call (h, hidden)
    h = 3 # no. of hidden layers
    hidden = [2,2,2] # no. of nodes in hidden layers
    call (h, hidden)

main()