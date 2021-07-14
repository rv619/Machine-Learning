This is an ANN estimator machine learning model.
 
It has been tested and trained on the datasets present in 
"DatasetMulti-Variate1.csv" and "DatasetMulti-Variate2.csv" files.

The main python code is in "ANN_as_Estimator.py" file.
Test-Train split is randomized.

The code is generalized for "h" hidden layers.
 
Pandas, Mathplotlib, and Numpy libraries are essential 
for the code to run.

A screenshot of the output (test results) can be found in 
'test_results.png' file.

As evident from the output: -
  -> ANN estimator gives better results than Multivariate 
     Linear Regression.
  -> As we increase the no. of hidden layers, the R-square value 
     also seem to increase. 

The corresponding Multivariate Linear Regression R-square values
can be found in 'comparison_with_multi_variate.png' file.

The test results are evaluated on R-square parameter.