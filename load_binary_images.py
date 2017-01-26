# =======================================
# These scripts will load the cifar-10
# image set in binary classification mode
# where "animals" are [1] and "man-made"
# objects are [0]. Written by Shea Brown
# shea-brown@uiowa.com, 
# https://sheabrownastro.wordpress.com/
# ========================================
import numpy as np

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def getNames():
    labels=unpickle('batches.meta')
    names=labels['label_names']
    return names

def getImages(dict):
    data=dict['data']
    # Uncomment this for conv. neural networks
    #s=(10000,3,32,32)
    images=data #data.reshape(s)    
    return images

def getLabels(dict):
    y=dict['labels']
    # Switch labes to binary classes
    # ------------------------------
    y=np.asarray(y)
    y[y==1]=0
    y[y==2]=1
    y[y==3]=1
    y[y==4]=1
    y[y==5]=1
    y[y==6]=1
    y[y==7]=1
    y[y==8]=0
    y[y==9]=0
    return y

def rgbImage(images, i):
    im=images[i]
    rgb=np.zeros((32,32,3))
    for j in range(0,2):
    	rgb[:,:,j]=im[j]
    return rgb

def loadTrainingSet():
    batch1=unpickle('data_batch_1')
    batch2=unpickle('data_batch_2')
    images1=getImages(batch1)
    labels1=getLabels(batch1)
    images2=getImages(batch2)
    labels2=getLabels(batch2)
    images=np.concatenate((images1,images2),axis=0)
    labels=np.concatenate((labels1,labels2),axis=0)
    return images, labels

def loadValidationSet():
    batch1=unpickle('data_batch_3')
    batch2=unpickle('data_batch_4')
    images1=getImages(batch1)
    labels1=getLabels(batch1)
    images2=getImages(batch2)
    labels2=getLabels(batch2)
    images=np.concatenate((images1,images2),axis=0)
    labels=np.concatenate((labels1,labels2),axis=0)
    return images, labels

def loadTestSet():
    batch1=unpickle('data_batch_5')
    images1=getImages(batch1)
    labels1=getLabels(batch1)
    fimages1, flabels1 = flipAndShuffle(images1,labels1)
    images=np.concatenate((images1,fimages1),axis=0)
    labels=np.concatenate((labels1,flabels1),axis=0)
    return images, labels

