# FermiClassifier
Train a neural network to identify the reduced temperature of a Fermi distribution for real-time analysis in image analysis

First need to generate training data...hard without a good implementation of polylog in python so I generate the data with Mathematica
and read in with python. Reduced temperatures are randomly chosen from {0.1,1.0} in 0.01 increments (similar to data). The distribution is
normalized such that the atoms number is fixed (currently 400 atoms), the integral is 1, and the imaging efficiency gives a a maximum
amplitude of 1. This is a good approximation of the situation in lab as this normalization can be enforced and the image can be rescaled to
have a maximum amplitude of 1. All images are centered (i.e., middle is the maximum amplitude) which can be approximated in lab with 
pre-analysis (use Gaussian fit to identify center and rescale ROI).

After the data has been loaded (with options for noise and fringes), train the neural network and run the analysis to see how the net does. Currently, the analysis has the test set loaded (and hardcoded in) but can be changed to the analysis set for diagnostics and improvement of hyper parameters.
