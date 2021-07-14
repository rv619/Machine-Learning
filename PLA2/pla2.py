import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def split (dataset, training_ratio):
    n = len (dataset) # Size of dataset
    r = int (training_ratio * n) # No. of training examples
    np.random.shuffle (dataset) # Shuffling the dataset
    return dataset[:r,:], dataset[r:,:]

def get_break (dataset):
    dataset = dataset[dataset[:, -1].argsort()]
    i = 0
    while dataset[i,-1] == -1:
        i += 1
    return dataset, i

# Works only on linearly separable dataset.
def train (train_data):
    n = len (train_data) # Size of training dataset
    # Get features and classes
    features = train_data[:, :-1]
    classes = np.array(train_data[:,-1])
    n_features = features.shape[1] # No. of features
    w = np.zeros(shape=(1, n_features + 1))
    # Make way for bias
    features = np.insert(features, 0, 1, axis = 1)
    #The data may or may not be linearly separable, we will iterate till we get a 100% accuracy on training dataset or complete 1000 iterations
    itr = 1
    n_mispredicted = 0 # No. of mispredictions
    while True:
        for i in range (n):
            predicted_class = 1 if np.dot(w, features[i].transpose()) > 0 else -1
            difference = (classes[i] - predicted_class) / (1 - (-1)) # Normalize the difference
            if difference != 0:
                w += difference * features[i]
                n_mispredicted += 1
        itr += 1
        if n_mispredicted == 0 or itr > 1000:
            break
        n_mispredicted = 0
    n_correct = n - n_mispredicted
    accuracy = n_correct / n * 100
    accuracy = "{:.2f}".format(accuracy)
    print("Training Accuracy after 1000 iterations = " + str(n_correct) + " | " + str(n) + " = " + str(accuracy) + "%")
    return w
    
def test (test_data, w):
    n = len (test_data) # Size of testing dataset
    # Get features and classes
    features = test_data[:, :-1]
    classes = np.array(test_data[:,-1])
    # Make way for bias
    features = np.insert(features, 0, 1, axis = 1)
    predicted_classes = np.ravel(np.dot(w, features.transpose()))
    predicted_classes = np.where(predicted_classes > 0, 1, -1)
    difference = classes - predicted_classes
    n_correctly_predicted = (difference == 0).sum() # No. of correct predictions
    return n_correctly_predicted
    
def plotPLA (w, train_data, test_data):
    # Clear previous figure, if any
    plt.clf()
    x = np.arange (6)
    y = (- w.item(0,1) * x - w.item(0,0)) / w.item(0,2)
    # Plot the decision boundary
    plt.plot(x, y, color = "green", linewidth = 1)
    # Sort the training dataset and get the index where Class 2 starts
    train_data, b = get_break (train_data)
    plt.scatter (np.array(train_data[:b, 0]), np.array(train_data[:b, 1]), marker = ",", color = "blue", s = 1, label = 'Training Data (Class 1)')
    plt.scatter (np.array(train_data[b:, 0]), np.array(train_data[b:, 1]), marker = ",", color = "red", s = 1, label = 'Training Data (Class 2)')
    # Sort the testing dataset and get the index where Class 2 starts
    test_data, b = get_break (test_data)
    plt.scatter (np.array(test_data[:b, 0]), np.array(test_data[:b, 1]), marker = "o", color = "blue", s = 2, label = 'Testing Data (Class 1)')
    plt.scatter (np.array(test_data[b:, 0]), np.array(test_data[b:, 1]), marker = "o", color = "red", s = 2, label = 'Testing Data (Class 2)')
    plt.xlabel ('A')
    plt.ylabel ('B')
    plt.legend (loc = 'lower right')
    plt.title ("Perceptron Learning Algorithm")
    plt.savefig ('output_plot.png', dpi=2000)
    
def run_algorithm (highest_accuracy):
    # Read the csv file
    dataset = pd.read_csv('PLA2.csv').to_numpy()
    # Split the data set
    train_data, test_data = split (dataset, 0.8)
    # Train the data
    w = train (train_data)
    # Test the data
    n_correctly_predicted = test (test_data, w)
    accuracy = n_correctly_predicted / len(test_data) * 100
    if accuracy > highest_accuracy:
        # Plot the data and update highest_accuracy
        plotPLA (w, train_data, test_data)
        highest_accuracy = accuracy
    # Display current runtime results
    print ("No. of correct test data predictions (Total no. of test data examples) = " + str(n_correctly_predicted) + " | (" + str(len(test_data)) + ")") 
    accuracy = "{:.2f}".format(accuracy)
    print ("Test Accuracy = " + str(accuracy) + "%")
    return highest_accuracy

def main():
    times = 5
    highest_accuracy = 0
    print ("Running the algorithm " + str(times) + " times...")
    print ("-------------------------------------------------------------------")
    for t in range (times):
        print ("For runtime no. " + str(t + 1) + "  ------------------------------------------------")
        highest_accuracy = run_algorithm (highest_accuracy)
        print ("-------------------------------------------------------------------")
    
main()