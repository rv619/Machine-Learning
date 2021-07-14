import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def split (dataset, training_ratio):
    n = len (dataset) # Size of dataset
    r = int (training_ratio * n) # No. of training examples
    np.random.shuffle (dataset) # Shuffling the dataset
    return dataset[:r,:], dataset[r:,:]

def getpowersutil (pos, k, d, powers, power):
    if pos == k:
        powers.append (power.copy())
        return
    for i in range (d + 1):
        power[pos] = i
        getpowersutil (pos + 1, k, d, powers, power)

def getpowers (k, d):
    power = [0] * k
    powers = []
    getpowersutil (0, k, d, powers, power)
    return powers

def getcoefficients (data, powers):
    n = len (data)
    coef = []
    factor = len (powers)
    k = len (powers[0])
    for i in range (n):
        c_row = [1] * factor
        for c in range (factor):
            for j in range (k):
                c_row [c] *= (data [i][j] ** powers [c][j])
        coef.append (c_row)
    return np.array (coef)

def train (train_data, powers):
    n = len (train_data) # Size of training dataset
    # Split the data
    x, y = train_data[:,:2], train_data[:,2:]
    # Transforming 'd' degree data to 1-D
    x_coef = getcoefficients (x, powers)
    # W = pseudo-inverse(X'.X) . X' . Y
    x_coef_tr = x_coef.transpose()
    w = np.dot (np.dot (np.linalg.pinv (np.dot (x_coef_tr, x_coef)), x_coef_tr), y)
    return w
    
def test (test_data, w, powers):
    n = len (test_data) # Size of testing dataset
    # Split the data
    x, y = test_data[:,:2], test_data[:,2:]
    # Transforming 'd' degree data to 1-D
    x_coef = getcoefficients (x, powers)
    # Y = X.W
    y_pred = np.dot (x_coef, w)
    sum_sq_r, sum_sq_t = 0, 0
    y_avg = np.mean (y, axis = 0) [0]
    for i in range (n):
        sum_sq_r += (y[i][0] - y_pred[i][0]) ** 2
        sum_sq_t += (y[i][0] - y_avg) ** 2
    r_sq = 1 - sum_sq_r / sum_sq_t
    # Display the test results
    print ("R_Square = " + str(round(r_sq, 4)))
    
def run (filename, k, d):
    # Read the csv file
    dataset = pd.read_csv (filename).to_numpy()
    print ("\nFeatures = " + str(k) + ", Polynomial Degree = " + str(d))
    # Split the data set
    train_data, test_data = split (dataset, 0.8)
    # Get polynomial to linear transformation powers
    powers = getpowers (k, d)
    # Train the data
    w = train (train_data, powers)
    # Test the data
    test (test_data, w, powers)
    
def main ():
    file1 = 'DatasetMulti-Variate1.csv'
    file2 = 'DatasetMulti-Variate2.csv'
    k = 2 # no. of features
    print ("----------------------------------------------");
    print ("Results (" + file1 + ")")
    print ("----------------------------------------------");
    run (file1, k, 1)
    run (file1, k, 2)
    run (file1, k, 3)
    print ("\n----------------------------------------------");
    print ("Results (" + file2 + ")")
    print ("----------------------------------------------");
    run (file2, k, 1)
    run (file2, k, 2)
    run (file2, k, 3)
    print ("\n----------------------------------------------");

main()