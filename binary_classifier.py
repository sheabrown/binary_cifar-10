# ====================================================================================
# This is a simple neural network (really just a binary linear logistic classifier) 
# with no frills, meant to train a vector of weights (no bias in this one!),
# used in the Astrophysical Machine Learning course at the University of Iowa 
# https://astrophysicalmachinelearning.wordpress.com/ taught by Shea Brown
# Written by Shea Brown, shea-brown@uiowa.edu, https://sheabrownastro.wordpress.com/
# =====================================================================================
import numpy as np
import sys

# sigmoid function
# -----------------------
def sigmoid(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
    
# Function to train the neural net
# --------------------------------
def train(X,y,N=10000,alpha=0.01):
	np.random.seed(1)
	# initialize weights randomly with mean 0
	W = 2*np.random.random((len(X[0]),1)) - 1

	# Start gradient descent
	# ---------------------------------------------------
	for iter in xrange(N):
		sys.stdout.write("\rProcessing epoch %i" % iter)
                sys.stdout.flush()

    		# Forward Propagation (make a guess)
		# We do this for all our training examples
		# in one go.  
	    	l1 = sigmoid(np.dot(X,W))

		# how much did we miss?
    		l1_error = y - l1 

		# multiply how much we missed by the 
    		# slope of the sigmoid at the values in l1
		# times a learning rate
		# -------------------------------------------------
    		l1_delta = alpha * l1_error * sigmoid(l1,True)

    		# update weights by multiplying l1_delta by the
		# transpose of the data (this is what you get 
		# if you take the gradient of a loss function
		# that is Loss=0.5*(y-l1)**2, which is an L2 
		# norm loss function, or least squares error 
		# -------------------------------------------
    		W += np.dot(X.T,l1_delta)
	return W 

# Function to make a prediction with user-defined weights W
# ----------------------------------------------------------
def predict(data,W):
	out=sigmoid(np.dot(data,W))
	return out

