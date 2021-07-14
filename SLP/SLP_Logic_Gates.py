import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def phi (x, a = 1): # Activation function
    return (1 / (1 + np.exp((-a) * x))) #sigmoid
    
def phi_dash (x, a = 1): # Derivative of activation function
    phi_x = phi (x, a)
    return a * phi_x * (1 - phi_x)

def train (x, y, d, c):
    n = len (x) # Size of training dataset
    x = np.insert (x, 0, 1, axis = 1)
    epoch = 0
    eps = 0.01
    learning_rate = 0.01
    w = np.random.random((c, d + 1))
    while epoch < 10000:
        E = 0
        for i in range (n):
            x_cur = x[i:i+1,:]
            u = np.dot(x_cur, np.transpose (w))
            v = phi (u)
            y_mask = np.array ([0] * c) # one-hot representation
            y_mask [y[i,0]] = 1
            del_y = y_mask - v
            del_w = np.dot (np.transpose (np.multiply (del_y, phi_dash (u))), x_cur)
            w += learning_rate * del_w
            e = 0.5 * np.sum (np.multiply (del_y, del_y))
            E += e / n
        if E < eps:
            break
        epoch += 1
    return w
    
def test (x, y, w, d, c):
    n = len (x) # Size of testing dataset
    x = np.insert (x, 0, 1, axis = 1)
    y_pred = np.dot (x, np.transpose (w)).argmax (axis = 1)
    print ("x1\t\tx2\t\ty\t\ty (predicted)")
    print ("---------------------------------------------------------------")
    mismatched = 0
    for i in range (n):
        print (str(x[i,1]) + "\t\t" + str(x[i,2]) + "\t\t" + str(y[i,0]) + "\t\t" + str(y_pred[i]))
        mismatched += abs (y[i,0] - y_pred[i])
    accuracy = ((n - mismatched) / n) * 100
    print ("---------------------------------------------------------------")
    
    print ("Accuracy = " + str(round(accuracy, 2)))
    
def run (data, d, c):
    x, y = data [:,:d], data[:,d:]
    # Train the data
    w = train (x, y, d, c)
    # Test the data
    test (x, y, w, d, c)
    
def main ():
    d = 2 # no. of features
    c = 2 # no. of classes
    print ("---------------------------------------------------------------")
    print ("Results for AND gate")
    print ("---------------------------------------------------------------")
    data = np.array ([[0,0,0],[0,1,0],[1,0,0],[1,1,1]])
    run (data, d, c)
    print ("\n---------------------------------------------------------------")
    print ("Results for OR gate")
    print ("---------------------------------------------------------------")
    data = np.array ([[0,0,0],[0,1,1],[1,0,1],[1,1,1]])
    run (data, d, c)
    print ("\n---------------------------------------------------------------")
    print ("Results for XOR gate")
    print ("---------------------------------------------------------------")
    data = np.array ([[0,0,0],[0,1,1],[1,0,1],[1,1,0]])
    run (data, d, c)
    print ("\n---------------------------------------------------------------")

main()