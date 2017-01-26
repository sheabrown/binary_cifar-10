# ==================================================================================
# Written by Shea Brown (shea-brown@uiowa.edu, https://sheabrownastro.wordpress.com/)
# for the Astrophysical Machine Learning course at the University of Iowa.
# This is a very basic binary linear classifier (binary_classifier.py) using
# batch gradient descent, applied to the cifar-10 dataset. The images have been 
# separated into "man-made" [0] or "animal" [1] labels to allow a simple binary
# classification (load_binary_images.py). This should work for any binary problem. 
# ==================================================================================
import matplotlib.pyplot as plt
from binary_classifier import train, predict
import numpy as np
from load_binary_images import loadTrainingSet, loadValidationSet, loadTestSet, getNames, rgbImage
import sys
plt.rcParams['figure.figsize'] = (7,7) # Make the figures a bit bigger

# Get and print the names of the original 10 classes
# ---------------------------------------------------
names=getNames()
print names

# Get the data, shuffled and split between train and validation sets
# -------------------------------------------------------------------
X_train, y_train = loadTrainingSet()
X_testo, y_test = loadValidationSet()

# Train the classifier on the training set. Here we are using a basic linear 
# classifier with a L2 norm loss function (i.e., least squares error), batch
# gradient decent, and a default learning rate of alpha=0.01 (you change this
# if you want below, e.g., W=train(data, target,4000,alpha=0.005)). We first 
# normalize the images to avoid overflow (large numbers) in our sigmoid function 
# -------------------------------------------------------------------------------
X_train=(X_train-np.mean(X_train))/700.0
X_test=(X_testo-np.mean(X_testo))/700.0
y_train=y_train.reshape(len(y_train),1)
y_test=y_test.reshape(len(y_test),1)

# Feel free to modify the number of epochs and learning rate
W=train(X_train, y_train,100)

# Try on the unknown validation images and store in an array of predictions
# If the prediction is greater then 0.5, consider that the classifier guessed "animal",
# and "man-made" otherwise. 
# -----------------------------------------------------------------------------------
i=0
guesses=np.zeros(len(y_test))
p=np.zeros(len(y_test))
for i in range(0,len(y_test)-1):
	pred=predict(X_test[i,:].reshape(1,-1),W)
	p[i]=pred
	if pred >= 0.5:
		guesses[i]=1.0
	else:
		guesses[i]=0.0


# Print out some statistics of the guesses
print('The mean of p() is ',np.mean(p))
print('The rms of p() is ',np.sqrt(np.var(p)))

# Compare this to the 'true' classifications. How many did we get right?
# -------------------------------------------------------------------------
y_test=y_test.reshape(len(y_test))
index=y_test==1.0
print('len index',len(index))
cor=np.sum(guesses[index] == y_test[index])
tot=len(y_test[index])
print 'The estimator gave '+str(cor)+' animals the correct classification, out of '+str(tot)+' total.'  
print 'That is a true positive rate of '+str(100.0*cor/tot)+'%'

index=y_test==0.0
cor=np.sum(guesses[index] == y_test[index])
tot=len(y_test[index])
print 'The estimator gave '+str(cor)+' man-maid sources the correct classification, out of '+str(tot)+' total.'
print 'That is '+str(100.0*cor/tot)+'%'
print 'The false positive rate is '+str(100.0*(tot-cor)/tot)+'%'

# Plot a histogram of the prediction values
# -----------------------------------------
#plt.hist(p,bins=30)
#plt.show()

# Check which items we got right / wrong
correct_indices = np.nonzero(guesses == y_test)[0]
incorrect_indices = np.nonzero(guesses != y_test)[0]

s=(20000,3,32,32)
X_testo=X_testo.reshape(s)
X_testo=np.moveaxis(X_testo,1,3)
plt.figure()
for i, correct in enumerate(correct_indices[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_testo[correct])
    plt.title("Predicted {}, Class {}".format(guesses[correct], y_test[correct]))

plt.figure()
for i, incorrect in enumerate(incorrect_indices[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_testo[incorrect])
    plt.title("Predicted {}, Class {}".format(guesses[incorrect], y_test[incorrect]))
