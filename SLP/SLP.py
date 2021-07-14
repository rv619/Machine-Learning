import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

def train (train_data, d, c, class_mapping):
    n = len (train_data) # Size of training dataset
    # Split the data
    x, y = train_data[:,:d], train_data[:,d:]
    x = np.insert (x, 0, 1, axis = 1)
    eps = 0.1
    learning_rate = 0.01
    w = np.random.random((c, d + 1))
    while True:
        E = 0
        for i in range (n):
            x_cur = x[i:i+1,:]
            u = np.dot(x_cur, np.transpose (w))
            v = phi (u)
            y_mask = np.array ([0] * c) # one-hot representation
            y_mask [class_mapping [y[i,0]]] = 1
            del_y = y_mask - v
            del_w = np.dot (np.transpose (np.multiply (del_y, phi_dash (u))), x_cur)
            w += learning_rate * del_w
            e = 0.5 * np.sum (np.multiply (del_y, del_y))
            E += e / n
        if E < eps:
            break
    return w
    
def test (test_data, w, d, c, class_mapping):
    n = len (test_data) # Size of testing dataset
    # Split the data
    x, y = test_data[:,:d], test_data[:,d:]
    x = np.insert (x, 0, 1, axis = 1)
    y_pred = np.dot (x, np.transpose (w)).argmax (axis = 1)
    mismatched = 0
    for i in range (n):
        mismatched += abs (class_mapping [y[i,0]] - y_pred[i])
    accuracy = ((n - mismatched) / n) * 100
    print ("\nAccuracy = " + str(round(accuracy, 2)))
    
def run (filename, d, c, classes):
    # Read the csv file
    dataset = pd.read_csv (filename).to_numpy()
    # Split the data set
    train_data, test_data = split (dataset, 0.8)
    # Map classes
    class_mapping = {}
    for i in range (len(classes)):
        class_mapping [classes[i]] = i
    # Train the data
    w = train (train_data, d, c, class_mapping)
    print ("w = \n")
    print (w)
    # Test the data
    test (test_data, w, d, c, class_mapping)
    
def main ():
    file1 = 'SLP1.csv'
    file2 = 'SLP2.csv'
    d = 2 # no. of features
    c = 2 # no. of classes
    print ("----------------------------------------------")
    print ("Results (" + file1 + ")")
    print ("----------------------------------------------\n")
    classes = [1, 2]
    run (file1, d, c, classes)
    print ("\n----------------------------------------------")
    print ("Results (" + file2 + ")")
    print ("----------------------------------------------\n")
    classes = [-1, 1]
    run (file2, d, c, classes)
    print ("\n----------------------------------------------")

main()