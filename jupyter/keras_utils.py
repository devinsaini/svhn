import keras
import numpy as np
import matplotlib.pyplot as plt
                                  
class DynamicPlot(keras.callbacks.Callback):
    def __init__(self, metrics=['loss'], axes=None):
        '''
        keras callback to plot training metrics dynamically. Set verbose=0 in call to fit
        '''
        self.metrics = metrics
        if axes is not None:
            self.ax = axes
            self.fig = axes.figure
        else:
            self.fig, self.ax = plt.subplots(1,1)

    def on_train_begin(self, logs={}):
        self.epochs = []
        self.metric_vals = {key:[] for key in self.metrics}
        return
        
    def on_epoch_end(self, epoch, logs={}):
        self.epochs = np.append(self.epochs, epoch)

        for metric in self.metrics:
            self.metric_vals[metric] = np.append(self.metric_vals[metric], logs.get(metric))
        if self.ax.lines:
            for metric in self.metric_vals.iterkeys():
                line = [line for line in self.ax.lines if line.get_label()==metric][0]
                line.set_xdata(self.epochs)
                line.set_ydata(self.metric_vals[metric])
            self.ax.set_title("epoch : {}/{}".format(epoch+1, self.params['nb_epoch']))
            self.ax.relim()
            self.ax.autoscale_view()
        else:
            for metric in self.metric_vals.iterkeys():
                self.ax.plot(self.epochs, self.metric_vals[metric], label=metric)
                self.ax.legend(loc='upper left')
                self.ax.set_xlabel("epoch")
        self.fig.canvas.draw()
        return