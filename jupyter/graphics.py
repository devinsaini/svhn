import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from keras import backend as K

def displaySamples(X, ytrue=None, ypred=None, ycounttrue=None, ycountpred=None, ycoordtrue=None):
    # sample random 8 images
    indices = np.random.randint(0,X.shape[0], size=8)
    
    plt.figure(figsize=(12,4))
    for i, idx in enumerate(indices):
        ax = plt.subplot(2,4,i+1)
        ax.imshow(X[idx])
        
        if ycoordtrue is not None:
            ax.add_patch(patches.Rectangle((ycoordtrue[idx][1],ycoordtrue[idx][0]),ycoordtrue[idx][3],ycoordtrue[idx][2], fill=False, edgecolor='red'))
            
        ax.text(0, -12, "count: {}[{}]".format("X" if ycountpred is None else ycountpred[idx], ycounttrue[idx]), ha="left", va="bottom", size="medium",color="green" if ycountpred is not None and ycountpred[idx]==ycounttrue[idx] else "red")
        
        ax.text(0, -4, "count: {}[{}]".format("X" if ypred is None else ypred[idx], ytrue[idx]), ha="left", va="bottom", size="medium",color="green" if ypred is not None and ypred[idx]==ytrue[idx] else "red")
        

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