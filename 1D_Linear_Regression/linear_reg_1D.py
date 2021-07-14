import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def split (dataset, training_ratio):
    n = len (dataset) # Size of dataset
    r = int (training_ratio * n) # No. of training examples
    np.random.shuffle (dataset) # Shuffling the dataset
    return dataset[:r,:], dataset[r:,:]

def train (train_data):
    n = len (train_data) # Size of training dataset
    # Split the data and flatten the numpy arrays
    x, y = np.ravel (train_data[:,:1]), np.ravel (train_data[:,1:])
    # Plot the training dataset
    plt.scatter (x, y, color = "blue", s = 1, label = 'Train data')
    # Calculate mean, variance and covariance
    mean_x = np.sum(x) / n
    mean_y = np.sum(y) / n
    covariance_x_y = (1/n) * np.sum(x*y) - mean_x * mean_y 
    var_x = (1/n) * np.sum(x*x) - mean_x * mean_x 
    # y(bar) = a * x(bar) + b
    a = covariance_x_y / var_x
    b = mean_y - a * mean_x
    return a, b
    
def test (test_data, coefficients):
    n = len (test_data) # Size of testing dataset
    a, b = coefficients[0], coefficients[1]
    # Split the data and flatten the numpy arrays
    x, y = np.ravel (test_data[:,:1]), np.ravel (test_data[:,1:])
    # Plot the testing dataset
    plt.scatter (x, y, color = "red", s = 1, label = 'Test data')
    # Display the test results
    p = 10; # No. of decimal places
    print ("X\t\tY\t\tY(Predicted)");
    print ("______________________________________________");
    for i in range (n):
        y_pred = a * x[i] + b
        print (str(round(x[i],p)) + "\t" + str(round(y[i],p)) + "\t" + str(round(y_pred,p)))
    print ("______________________________________________");
    
def plot_regression_line (dataset, coefficients):
    a, b = coefficients[0], coefficients[1]
    # Split the data and flatten the numpy arrays
    x, y = np.ravel (dataset[:,:1]), np.ravel (dataset[:,1:])
    y_pred = a * x + b
    # Plot the regression line
    plt.plot (x, y_pred, color = "green")
    plt.xlabel ('X')
    plt.ylabel ('Y')
    plt.legend (loc = 'lower right')
    plt.title ("Linear Regression Model")
    plt.savefig ('output_plot.png', dpi=400)
    
def main ():
    # Read the csv file
    dataset = pd.read_csv ('Linear_Regression_Data_1D.csv').to_numpy()
    # Split the data set
    train_data, test_data = split (dataset, 0.8)
    # Train the data
    coefficients = train (train_data)
    # Test the data
    test (test_data, coefficients)
    # Plot the data
    plot_regression_line (dataset, coefficients)
    
main()