import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import copy, deepcopy

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
    
def train (x, y, hidden, h, d, c, bias, learning_rate = 0.1, eps = 0.001, epoch = 10000):
    n = len (x) # Size of training dataset
    it = 0
    w = np.array ([None]*(h+1))
    dimensions = hidden.copy()
    dimensions.insert (0, d)
    dimensions.append (c)
    w = np.array ([None]*(h+1))
    for i in range (h+1):
        w[i] = np.random.random((dimensions[i+1], dimensions[i]+1))
    while it < 10000:
        E = 0
        for i in range (n):
            x_cur = x[i:i+1,:]
            phi_u, phi_dash_u = feed_forward (x_cur, w, h, bias)
            y_mask = np.array ([0] * c) # one-hot representation
            y_mask [y[i,0]] = 1
            error = y_mask - phi_u[h+1]
            delta = back_propagate (error, w, phi_dash_u, h)
            w = update_weights (w, phi_u, delta, h, learning_rate)
            e = 0.5 * np.sum (np.multiply (error, error))
            E += e / n
        if E < eps:
            break
        it += 1
    return w
    
def test (x, y, w, h, bias):
    n = len (x) # Size of testing dataset
    phi_u, phi_dash_u = feed_forward (x, w, h, bias)
    y_pred = phi_u[h+1].argmax (axis = 1)
    print ("x1\t\tx2\t\ty\t\ty (predicted)")
    print ("---------------------------------------------------------------")
    mismatched = 0
    for i in range (n):
        print (str(x[i,0]) + "\t\t" + str(x[i,1]) + "\t\t" + str(y[i,0]) + "\t\t" + str(y_pred[i]))
        mismatched += abs (y[i,0] - y_pred[i])
    accuracy = ((n - mismatched) / n) * 100
    print ("---------------------------------------------------------------")
    print ("Accuracy = " + str(round(accuracy, 2)))
    
def run (data, hidden, h, d, c, bias = 1.0):
    x, y = data [:,:d], data[:,d:]
    # Train the data
    w = train (x, y, hidden, h, d, c, bias)
    # Test the data
    test (x, y, w, h, bias)
    
def main ():
    d = 2 # no. of features
    c = 2 # no. of classes
    h = 1 # no. of hidden layers
    hidden = [4] # no. of nodes in hidden layers 
    print ("---------------------------------------------------------------")
    print ("Results for AND gate")
    print ("---------------------------------------------------------------")
    data = np.array ([[0,0,0],[0,1,0],[1,0,0],[1,1,1]])
    run (data, hidden, h, d, c)
    print ("\n---------------------------------------------------------------")
    print ("Results for OR gate")
    print ("---------------------------------------------------------------")
    data = np.array ([[0,0,0],[0,1,1],[1,0,1],[1,1,1]])
    run (data, hidden, h, d, c)
    print ("\n---------------------------------------------------------------")
    print ("Results for XOR gate")
    print ("---------------------------------------------------------------")
    data = np.array ([[0,0,0],[0,1,1],[1,0,1],[1,1,0]])
    run (data, hidden, h, d, c)
    print ("\n---------------------------------------------------------------")

main()