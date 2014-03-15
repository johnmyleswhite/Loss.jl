Loss.jl
=======

General functions for estimating loss functions inspired by Kaggle's release
of code for many common metrics. This package implements the full Cartesian
product of two ways of classifying loss functions:

* Single element loss function definitions:
	* Absolute deviation
	* Squared error
	* 0/1 loss
	* Hinge loss
	* Log loss
* Aggregation mechanism across elements:
	* Mean
	* Root Mean
	* Median
	* Minimum
	* Maximum

# To Do

* Add other metrics
	* AUC
* Add convenient abbreviations
	* rmse
	* ...
