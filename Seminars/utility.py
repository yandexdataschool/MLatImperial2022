from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import sklearn as skl
from sklearn import tree
from sklearn.metrics import accuracy_score
from pylab import *
from itertools import chain
from collections import deque
import sys
from datetime import datetime

from pylab import *


def row(x):
    '''Given sequence x returns numpy array x as row-vector with shape (1,len(x))
    Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.'''    
    if not isinstance(x,np.ndarray):
        x=np.array(x)
    assert len(x.shape)==1,'x should contain only one axis!'
    return array(x)[np.newaxis,:]

def vec(x):
    '''Given sequence x returns numpy array x as vector-column with shape (len(x),1)
    Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.'''    
    if not isinstance(x,np.ndarray):
        x=np.array(x)
    assert len(x.shape)==1,'x should contain only one axis!'
    return x[:,np.newaxis]
    
    
def normalize(z):
    '''Feature normalization
    Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.'''
    return (z-min(z))/(max(z)-min(z))
    
    

def piter(x, percent_period=1,period=None,end="| ", show=True):
    '''Iterates through x (any iterable object, having len) returning iteratively elements from x and printing progress.
    Progress is printed every <period> iterations or after every <percent_period> percent of total was complete.
    Useful for controlling how much of the long computations were completed.

    Example:
        for i in piter([10,11,12,13,14,15],2):
            print(i)
    Output:
        0.00% done
        10
        11
        33.33% done
        12
        13
        66.67% done
        14
        15
    Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.'''

    if show==False: # do nothing
        for element in x:
            yield element
    else:
    
        if hasattr(x,'__len__'):
            total = len(x)
            if period==None:
                period=max(1,total//(100/percent_period))
            for i,element in enumerate(x):
                if i % period==0:
                    print('%.0f' % (100*i/total), end=end)
                yield element
            print('100.0',end=end+'\n')
        else: # no len
            for i,element in enumerate(x):
                if i % period==0:
                    print('%d' % i, end=end)
                yield element
            print('%d'%i,end=end+'\n')




def show_performance(iters, train_err_rates, val_err_rates, train_losses, val_losses, figsize=(14, 5), title_str='',
                     verticals=[], verticals2=[]):
    '''Plots a graph of performance measures of converging algorithm. Can plot dynamic graph if called many times.
    '''

    from IPython import display

    clf()
    gcf().set_size_inches(*figsize)

    ax1 = gcf().add_subplot(111)
    ax1.plot(iters, train_losses, 'g--', label='train loss')
    ax1.plot(iters, val_losses, 'r--', label='val loss')
    xlabel('iterations')
    legend(loc='upper left')
    ylabel('loss')

    ax2 = gcf().add_subplot(111, sharex=ax1, frameon=False)
    ax2.plot(iters, train_err_rates, 'g-', lw=3, label='train err.rate')
    ax2.plot(iters, val_err_rates, 'r-', lw=3, label='val err.rate')
    ax2.yaxis.tick_right();
    ax2.yaxis.set_label_position("right")
    ylabel('err.rate')

    for vert in verticals:
        axvline(vert, linestyle=':', color='g')

    for vert in verticals2:
        axvline(vert, linestyle=':', color='b')

    legend(loc='upper right')

    title(title_str)
    display.display(gcf())
    display.clear_output(wait=True)


'''    
#Demo of show_performance:
iters=[]
train_losses = []
val_losses = []
train_err_rates = []
val_err_rates = []
m=0.01

for i in range(3,16):
    iters.append(i)
    train_losses.append( 100+1/(i*i)+m*randn(1)[0] )
    val_losses.append( 100+1/i+m*randn(1)[0] )
    train_err_rates.append( 1/(i*i)+10*m*randn(1)[0] )
    val_err_rates.append( 1/i+10*m*randn(1)[0] )

    show_performance(iters, train_err_rates, val_err_rates, train_losses, val_losses)    
'''


def plot_predictions_2D(model, X_train, Y_train, task, feature_names=None,
                         train=True, n=50, cmap=None, point_size=15,
                         offset=0.05, alpha=1,title=None):
    '''Plots decision regions for classifier clf trained on design matrix X=[x1,x2] with classes y.
    n is the number of ticks along each direction
    Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.

    model: prediction model, supporting sklearn interface
    X_train: design matrix [n_objects x n_features]
    Y_train: vector of outputs [n_objects]
    task: either "regression" or "classification"
    feature_names: list of feature names [n_features]
    n: how many bins to use along each dimension
    cmap: matplotlib colormap
    point_size: size of points for scatterplot
    offset: margin size around training data distribution
    alpha: visibility of predictions (0=invisible, 1=fully visible)
    '''
    plt.figure()
    x1, x2 = X_train[:, 0], X_train[:, 1]
    if train:
        model.fit(X_train, Y_train)

    margin1 = offset * (x1.max() - x1.min())
    margin2 = offset * (x2.max() - x2.min())

    # create a mesh to plot in
    x1_min, x1_max = x1.min() - margin1, x1.max() + margin1
    x2_min, x2_max = x2.min() - margin2, x2.max() + margin2

    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, n),
                           np.linspace(x2_min, x2_max, n))

    X_test = hstack([vec(xx1.ravel()), vec(xx2.ravel())])
    Y_test = model.predict(X_test)
    yy = Y_test.reshape(n, n)

    if task == 'regression':

        vmin = minimum(min(Y_train), min(Y_test))
        vmax = maximum(max(Y_train), max(Y_test))

        import matplotlib.cm as cm
        from matplotlib.colors import Normalize

        if cmap == None:
            cmap = cm.hot
        norm = Normalize(vmin=vmin, vmax=vmax)

        Y_train = cmap(norm(Y_train))
        yy = cmap(norm(yy))

        img = imshow(yy, extent=(x1_min, x1_max, x2_min, x2_max), interpolation='nearest', origin='lower', alpha=alpha)
        scatter(x1, x2, facecolor=Y_train, lw=1, edgecolor='k', s=point_size)

    elif task == 'classification':
        # this is a set of well distinguishable colors. Useful for visualizing many graphs on one plot.
                 
        COLORS=[[0,0.5,1],[1,0,0],[0.2,1,0],[1,0.5,0],[1,0,1],[0.5,0.5,0.5],[0.5,0,1],[1,1,0],[0,1,1],[ 0.25   ,0.58,  0.50],[0,0,1]]
        if __name__=='__main__':  # colors demonstration
            for i in range(len(COLORS)):
                plot([i,i],c=COLORS[i],linewidth=3)

            ylim([-1,len(COLORS)])


        classes = unique(Y_train)
        assert len(classes) <= len(COLORS), 'Classes count should be <=%s' % len(COLORS)

        y2color = lambda y: COLORS[[c for c in classes].index(y)]
        Z = zeros([n, n, 3])
        for i in arange(n):
            for j in arange(n):
                Z[i, j, :] = y2color(yy[i, j])

        img = imshow(Z, extent=(x1_min, x1_max, x2_min, x2_max), interpolation='nearest', origin='lower', alpha=alpha)
        scatter(x1, x2, c=[COLORS[[c for c in classes].index(y)] for y in Y_train], lw=1, edgecolor='k', s=point_size)

    else:
        raise Exception('task should be either "regression" or "classification"!')

    plt.axis([x1_min, x1_max, x2_min, x2_max])

    if feature_names != None:
        xlabel(feature_names[0])
        ylabel(feature_names[1])

    if title != None:
        plt.title(title)


def show_param_dependency(m, X_train, Y_train, X_test, Y_test, param_name, param_vals, loss_fun, x_label=None):
    '''score_fun='accuracy',
       Show plot, showing dependency of score_fun (estimated using CV on X_train, Y_train) 
       on parameter param_name taking values in param_vals.
       Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.'''    
    
       
    
    if x_label is None:
        x_label = param_name
        
    losses = zeros(len(param_vals))

    for i, param_val in enumerate(piter(param_vals)):
        m.set_params(**{param_name:param_val})
        m.fit(X_train, Y_train)
        Y_hat = m.predict(X_test)
        if loss_fun=='error_rate':
                losses[i] = mean(Y_hat!=Y_test)
        elif loss_fun=='MAE':
                losses[i] = mean(abs(Y_hat-Y_test))                
        elif loss_fun=='MSE':
                losses[i] = mean((Y_hat-Y_test)**2)
        elif loss_fun=='RMSE':
                losses[i] = sqrt(mean((Y_hat-Y_test)**2))
        else:
            raise ValueError('Unknown loss %s!'%loss_fun)
            
    xlabel(x_label)
    ylabel(loss_fun)
    plot(param_vals, losses)
    print('Min %s = %.4f for %s=%s' % (loss_fun, min(losses), param_name, param_vals[argmin(losses)]) ) 
	


def plot_classifier_decision(classifier, X, y, plot_scatter=True, margin=0.1):
    x_range = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_range = X[:, 1].min() - margin, X[:, 1].max() + margin
    xx, yy = numpy.meshgrid(numpy.linspace(*x_range, num=200),
                            numpy.linspace(*y_range, num=200))
    data = numpy.vstack([xx.flatten(), yy.flatten()]).T

    p = classifier.predict_proba(data)[:, 1]
    plt.contourf(xx, yy, p.reshape(xx.shape), cmap='bwr', alpha=.5)
    if plot_scatter:
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', s=30)

    plt.xlim(*x_range)
    plt.ylim(*y_range)

    