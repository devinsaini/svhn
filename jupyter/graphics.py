import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from keras import backend as K

def displaySamples(X, ylabels=None, ylabelspred=None, ycounttrue=None, ycountpred=None, index=None):
    # sample random 8 images
    samples = np.random.randint(0,X.shape[0], size=8)
    
    plt.figure(figsize=(12,4))
    for i, idx in enumerate(samples):
        ax = plt.subplot(2,4,i+1)
        ax.imshow(X[idx])
        
        if index is not None:
            assert (len(ylabels)==1), "ylabels, ylabelspred can have only one element if index is given"
        ax.text(0, -12, "count: {} [{}]".format("-" if ycountpred is None else ycountpred[idx], ycounttrue[idx]), ha="left", va="bottom", size="medium",color="green" if ycountpred is not None and ycountpred[idx]==ycounttrue[idx] else "red")
        
        ax.text(0, -4, "label: {} [{}]".format("-" if ylabelspred is None else ylabelspred[idx], ylabels[idx]), ha="left", va="bottom", size="medium",color="green" if ylabelspred is not None and np.array_equal(ylabelspred[idx],ylabels[idx]) else "red")
        

def showTrainingHistory(history):
	print(history.history.keys())
	# summarize history for accuracy
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	# summarize history for loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()

def showCNNConv(model, n, X_sample):
    layeroutfunc = K.function([model.layers[0].input, K.learning_phase()],[model.layers[n].output])
    layerout = np.array(layeroutfunc([X_sample[np.newaxis,:], False]))[0]
    l = np.rollaxis(layerout, 3, 1)[0]
    print l.shape
    
    plt.figure(figsize=(12,12))
    plt.subplots_adjust(hspace=0.01)
    for i in range(0,l.shape[0]):
        plt.subplot(8,8,i+1)
        plt.imshow(l[i], cmap='gray', interpolation='none')
        plt.axis('off')

def showDenseConv(model, n, X_sample):
    layeroutfunc = K.function([model.layers[0].input, K.learning_phase()],[model.layers[n].output])
    layerout = np.array(layeroutfunc([X_sample, False]))[0]
    plt.bar(np.arange(len(layerout[0])),layerout[0])