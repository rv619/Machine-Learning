This is a single layer perceptron (SLP) machine learning model.
 
It has been tested and trained on the datasets present in 
"SLP1.csv" and "SLP2.csv" files.

The main python code is in "SLP.py" file.
Test-Train split is randomized.

The code is generalized for 'd' features and 'c' classes.

Pandas, Mathplotlib, and Numpy libraries are essential 
for the code to run.

A screenshot of the output (test results) can be found in 
'SLP_Test_Data_Output.png' file.

There is another python file - 'SLP_Logic_Gates.py' that tests
the model on AND, OR, XOR logic gates.

A screenshot of the output of the above python code can be
found in 'Logic_GateS_Output_Screenshot.png' file.

As evident from the output: -
  -> SLP works fine on 'AND' and 'OR' gates (100% accuracy, each). 
     But it fails on 'XOR' gate (25% accuracy).