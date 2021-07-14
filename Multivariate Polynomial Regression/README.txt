This is a multivariate polynomial regression machine learning model.
 
It has been tested and trained on the datasets present in 
"DatasetMulti-Variate1.csv" and "DatasetMulti-Variate2.csv" files.

The main python code is in "poly_reg_nD.py" file.
Test-Train split is randomized.

The code is generalized and a hypothesis polynomial of 'd' degree
can be used on a dataset having 'k' features.
 
Pandas, Mathplotlib, and Numpy libraries are essential 
for the code to run.

A screenshot of the output (test results) can be found in 
'test_results.png' file.

As evident from the output: -
  -> Linear, quadratic, and cubic hypothesis polynomials give 
     good results on the first dataset.
  -> Linear polynomial gives poor result, while quadratic and 
     cubic gives satisfactory results on the second dataset.

The test results are evaluated on R-square parameter.