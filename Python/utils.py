'''
Created on Dec 24, 2014

@author: morgan
'''

import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt 
from matplotlib.path import Path
import matplotlib.patches as patches


#===============================================================================
# Basic function x -> \Phi(x)
# Define your own basic functions
#===============================================================================
def feature_mapping(x):
    if x.ndim is 1:
        # this is a single point
        if x.size > 2:
            print 'Only 2 dimensional is allowed for our demo!'
            return None
        else:
            # mapping here (x,y,xy,x^2,y^2,xy^2,yx^2)
            return np.array([x[0], x[1], x[0]*x[1], x[0]**2, x[1]**2, x[0]*x[1]**2, x[1]*x[0]**2])
            # TODO: test other mapping
    else:
        # x contains multiple points
        if x.shape[1] > 2:
            print 'Only 2 dimensional is allowed for our demo!'
            return None
        else:
            # mapping here (x,y,xy,x^2,y^2,xy^2,yx^2)
            X = np.vstack((x[:,0], x[:,1], x[:,0]*x[:,1], x[:,0]**2, x[:,1]**2, x[:,0]*x[:,1]**2, x[:,1]*x[:,0]**2))
            return X.T
            # TODO: test other mapping
            

#===============================================================================
# Input a 2-d dataset, return a p-dimensional dataset by a feature mapping
#===============================================================================
def gen_lm_data(X2d, w, b):
    Xtr = feature_mapping(X2d) 
    if Xtr.shape[1] != w.size:
        print 'Feature size does not equal the weight!'
        return None
    if Xtr is not None:
        weights = w.reshape(w.size,1)
        ytr = Xtr.dot(weights) + b    
        return (Xtr, ytr)
    
    
#===========================================================================
# Generate two 2-d gaussian dataset 
#===========================================================================
def gen_2gauss(nsamples=[50,50], 
               mu=np.array([[1.5,0],[-1.5,0]]), 
               c1=[[1,0], [0,1]], c2=[[1,0], [0,1]]):    
    X0 = np.random.multivariate_normal(mean=mu[0,:], cov=c1, size=nsamples[0])
    X1 = np.random.multivariate_normal(mean=mu[1,:], cov=c2, size=nsamples[1])
    y0 = np.ones((nsamples[0], ) )
    y1 = -1*np.ones((nsamples[1], ) )
    X = np.concatenate((X0,X1), 0)
    y = np.concatenate((y0,y1), 0)
    return (X, y)       


#===============================================================================
# Generate a two-class multivariate (d features) gaussian dataset
# Each feature has mean [+m, -m] for both classes, and variance is set the same
# There are n_informative features 
#===============================================================================
def gen2_mvn(n_samples_each=100, n_features=2, n_informative=0, max_mean=10, min_mean=1, var=1):
    cov = np.diag(np.ones(n_features)*var)
    if n_informative == 0:
        means_positive = np.linspace(max_mean, min_mean, n_features)
        means_negative = -1*means_positive
    else:
        informative_means = np.linspace(max_mean, min_mean, n_informative)
#         noninformative_means = min_mean + max_mean*np.random.rand(n_features - n_informative)
        noninformative_means = np.zeros(n_features - n_informative)
        means_positive = np.append(informative_means, noninformative_means)
        means_negative= np.append(-1*informative_means, noninformative_means)
    X0 = np.random.multivariate_normal(means_positive, cov, n_samples_each)
    X1 = np.random.multivariate_normal(means_negative, cov, n_samples_each)
    X = np.vstack((X0,X1)) 
    y =np.concatenate((np.ones(n_samples_each), -1*np.ones(n_samples_each)))  
    return (X, y)


#===============================================================================
# Generates 3 circle data with different densities, r, dev
#===============================================================================
def gen3_circles(c, r, n, dev):
    def generate_circle_data(center, r, n, dev):
        phi = 2 * np.pi *  np.random.randn(n)
        c1 = np.empty(n)
        c1.fill(center[0])
        c2 = np.empty(n)
        c2.fill(center[1])
        c = np.array([c1, c2])
        h = r * np.array([np.cos(phi), np.sin(phi)])
        gnd_X = c + h
        #add noise
        X = gnd_X + np.random.randn(2, n) * dev
        X = X.transpose()
        return X

    data1 = generate_circle_data(c[0], r[0], n[0], dev[0])
    data2 = generate_circle_data(c[1], r[1], n[1], dev[1])
    data3 = generate_circle_data(c[2], r[2], n[2], dev[2])
    X = np.vstack((np.vstack((data1, data2)), data3))
    y1 = np.empty(n[0])
    y1.fill(1)
    y2 = np.empty(n[1])
    y2.fill(2)
    y3 = np.empty(n[2])
    y3.fill(3)
    y = np.vstack((np.vstack((np.matrix(y1).transpose(), np.matrix(y2).transpose())), np.matrix(y3).transpose()))
    #plt.scatter(np.array(X)[:, 0], np.array(X)[:, 1], c=np.array(y))
    #plt.show()
    yield X
    yield y 


#===============================================================================
# gen svc synthetic data
# one gauss in center, 3 small cluster on circle
#===============================================================================
def gen_svc_data(c, r, cluster_sizes=[50,10,10,10]):
    """
    generate a dataset with gauss center as normal samples, outliers are surrouding
    it with smaller sample size
    :param c: gaussian mean
    :param r: radius of the circle
    :param cluster_sizes: 4x1 list comprising sample sizes
    :return: input X, input labels y
    """
    X = np.empty((0,2))
    cov = np.diag(np.ones(2)*0.1)
    gauss = np.random.multivariate_normal(np.array(c), cov, cluster_sizes[0])
    y = -1*np.ones(gauss.shape[0])
    X = np.vstack((X, gauss))
    outlier_classes = len(cluster_sizes)-1
    angle_seeds = np.arange(0,outlier_classes)*(2*np.pi/outlier_classes)
    for i in range(outlier_classes):
        angles = angle_seeds[i]+(np.pi/18)*np.random.randn(cluster_sizes[i+1])
        x = np.concatenate((c[0]+(r+0.2*np.random.randn())*np.cos(angles).reshape(cluster_sizes[i+1],1),\
                            c[1]+(r+0.2*np.random.randn())*np.sin(angles).reshape(cluster_sizes[i+1],1)), 1)
        yc = np.ones(x.shape[0])
        X = np.vstack((X, x))
        y = np.concatenate((y,yc))
    return X,y


def gen_noise_gauss(n_samples=100, noise_level=.1, u=[0,0], cov=0.5):
    """
    Generate gaussian dataset with uniform noises
    :param n_samples: number of samples
    :param noise_level: ratio of noises
    :param u: gaussain mean
    :param cov: gaussian covariance
    :return: dataset X => y in [+1, -1]
    """
   # np.random.seed(0)
    neg_size = int(n_samples*(1-noise_level))
    pos_size = n_samples - neg_size
    sigma = np.diag(np.ones(2)*cov)
    gauss = np.random.multivariate_normal(np.array(u), sigma, neg_size)
    # TODO: min,max computed from gaussian
    outliers = np.random.uniform(low=-4, high=4, size=(pos_size, 2))
    print "outliers"
    print outliers
    X = np.r_[gauss, outliers]
    y = np.r_[-np.ones(neg_size), np.ones(pos_size)]
    return X, y


#===============================================================================
# Plot SVC boundary
#===============================================================================
def plot_svc(X, y, mysvc, bounds=None, grid=50):
    if bounds is None:
        xmin = np.min(X[:, 0], 0)
        xmax = np.max(X[:, 0], 0)
        ymin = np.min(X[:, 1], 0)
        ymax = np.max(X[:, 1], 0)
    else:
        xmin, ymin = bounds[0], bounds[0]
        xmax, ymax = bounds[1], bounds[1]
    aspect_ratio = (xmax - xmin) / (ymax - ymin)
    xgrid, ygrid = np.meshgrid(np.linspace(xmin, xmax, grid),
                              np.linspace(ymin, ymax, grid))
    plt.gca(aspect=aspect_ratio)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xticks([])
    plt.yticks([])
    plt.hold(True)
    plt.plot(X[y == 1, 0], X[y == 1, 1], 'bo')
    plt.plot(X[y == -1, 0], X[y == -1, 1], 'ro')
    
    box_xy = np.append(xgrid.reshape(xgrid.size, 1), ygrid.reshape(ygrid.size, 1), 1)
    if mysvc is not None:
        scores = mysvc.decision_function(box_xy)
    else:
        print 'You must have a valid SVC object.'
        return None;
    
    CS=plt.contourf(xgrid, ygrid, scores.reshape(xgrid.shape), alpha=0.5, cmap='jet_r')
    plt.contour(xgrid, ygrid, scores.reshape(xgrid.shape), levels=[0], colors='k', linestyles='solid', linewidths=1.5)
    plt.contour(xgrid, ygrid, scores.reshape(xgrid.shape), levels=[-1,1], colors='k', linestyles='dashed', linewidths=1)
    plt.plot(mysvc.support_vectors_[:,0], mysvc.support_vectors_[:,1], 'ko', markerfacecolor='none', markersize=10)
    CB = plt.colorbar(CS)
    
    
#===============================================================================
# Plot the bound box for a given bounds
# bds is a list of tuples with each (min,max) of each dimension
#===============================================================================
def plot_bounds(bds, ax = plt.gca()):
    # plot the bounds
    if type(bds) is not list:
        print 'you need input a list for the bounds'
    
    if len(bds) > 2:
        print 'We can only plot 2-d bounds, sorry!'
        
    verts = [
        (bds[0][0], bds[1][0]), # left, bottom
        (bds[0][0], bds[1][1]), # left, top
        (bds[0][1], bds[1][1]), # right, top
        (bds[0][1], bds[1][0]), # right, bottom
        (bds[0][0], bds[1][0]), # ignored
        ]
    codes = [Path.MOVETO,
             Path.LINETO,
             Path.LINETO,
             Path.LINETO,
             Path.CLOSEPOLY,
             ]
    path = Path(verts, codes)
    patch = patches.PathPatch(path,  ls='dashed', lw=1, facecolor='none',)
    ax.add_patch(patch)   
    
    
#===============================================================================
# Get the grid X,Y by given data range
# X, Y are matrix of size gridxgrid (50*50 by default)
#===============================================================================
def getBoxbyX(X, grid=50, padding=True):
    '''
    Get the meshgrid X,Y given data set X
    :param X: dataset
    :param grid: meshgrid step size
    :param padding: if add extra padding around
    :return: X,Y
    '''
    if X.shape[1] > 2:
        print 'We can only get the grid in 2-d dimension!'
        return None
    else:
        minx = min(X[:,0])
        maxx = max(X[:,0])
        miny = min(X[:,1])
        maxy = max(X[:,1])
        padding_x = 0.05*(maxx-minx)
        padding_y = 0.05*(maxy-miny)
        if padding:
            X,Y = np.meshgrid(np.linspace(minx-padding_x, maxx+padding_x, grid), 
                              np.linspace(miny-padding_y, maxy+padding_y, grid))
        else:
            X,Y = np.meshgrid(np.linspace(minx, maxx, grid), 
                              np.linspace(miny, maxy, grid))
    return (X, Y)


#===============================================================================
# generate a list of square subplots by cols,rows
#===============================================================================
def getSquareSubplots(rows=1, cols=1, 
                      fig_w=4, fig_h=4, 
                      pad_x=.05, pad_y=.05, 
                      gutter_x=.05, gutter_y=.05):
    # define the figsize to fit the cols and rows of subplots
    fig = plt.figure(figsize=(fig_w, fig_h))
    axes_height = (1-2*pad_y-(rows-1)*gutter_y)/rows
    axes_width = axes_height*fig_h/fig_w
    for i in range(rows):
        for j in range(cols):
            x = pad_x + j*(axes_width+gutter_x)
            y = (1 - pad_y) - i*(axes_height+gutter_y) - axes_height
            fig.add_axes([x, y, axes_width, axes_height])
    return fig

#===============================================================================
# set axis square
#===============================================================================
def setAxSquare(ax):
    xlim0, xlim1 = ax.get_xlim()    
    ylim0, ylim1 = ax.get_ylim()    
    ax.set_aspect((xlim1-xlim0)/(ylim1-ylim0))

#===============================================================================
# Kurcheva (2007) consistency index: correction for a chance
# A and B are both arrays indicating indicies of features 
# n is the full length of features
#===============================================================================
def kurcheva_ConsistencyIndex(A, B, n):
    if len(A) != len(B):
        print 'Two sets of features must have the same length!'
        return None
    r = len(A) - len(np.setdiff1d(A, B))
    k = float(A.size)
    if k >= n:
        print 'The subset must be less than full feature size n!'
        return None
    else:
        return (r*n - k**2)/(k*n - k**2) 


#===============================================================================
# compute feature stability of two sets of feature subsets
#===============================================================================
def feature_stability(subset1, subset2, k):
    d = subset1.shape[1]
    if subset2 is None:
        n = subset1.shape[0]
        res = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                fea1 = subset1[i].argsort()
                fea2 = subset1[j].argsort()
                res[i,j]=kurcheva_ConsistencyIndex(fea1[-1*k:], fea2[-1*k:], d)
        list_i = np.empty(shape=(0,))
        list_j = np.empty(shape=(0,))
        for i in range(1,n):
            list_i = np.concatenate((list_i, i*np.ones(i))) 
        for j in range(1,n):
            list_j = np.concatenate((list_j, np.arange(j))) 
        ki_pairs = res[list_i.tolist(), list_j.tolist()].ravel()                 
        mean = ki_pairs.mean()
        std = np.std(ki_pairs)
        return (mean, std)
    else:
        if subset1.shape[1] != subset2.shape[1]:
            print 'feature sizes are not equal, exit!'
            return None
        n = subset1.shape[0]
        m = subset2.shape[0]
        res = np.zeros((n,m))
        for i in range(n):
            for j in range(m):
                fea1 = subset1[i].argsort()
                fea2 = subset2[j].argsort()
                res[i,j]=kurcheva_ConsistencyIndex(fea1[-1*k:], fea2[-1*k:], d)
        mean = res.mean()
        std = res.std()        
        return (mean, std)
        



if __name__ == "__main__":
    # unit test here
    pass
            