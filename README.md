A sample dataset is generated with data points that are arranged in concentric circles and therefore are unable to be effectively separated linearly.
We define a Quantum Embedding Kernel using Pennylane’s kernels modules, with random parameters. Using this output, an SVC is applied and the accuracy is measured.
The next step is to apply standard kernels to the original dataset. These kernels are of linear, polynomial, and radial basis function form. As before, they are passed to an SVC and the accuracy is measured.

The data generation and processing was conducted in Python.
Quantum kernelling used Pennylane. 
The SVC algorithm was from scikit-learn. 
NumPy was used for data processing and Matplotlib was used for plotting the data.
Code for Quantum kernelling was adapted from a Pennylane tutorial.
