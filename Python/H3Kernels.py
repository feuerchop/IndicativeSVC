__author__ = 'Huang Xiao'
'''
Kernel functions support RBF,linear,polynomial kernels
'''
import numpy as np

def kernel(x1, x2=np.array([]), metric='linear', filter_params=True, coef0=1.0, degree=2, gamma=0.5):
    '''
    kernel methods support rbf, linear and polynomial
    :param x1: N*d ndarray
    :param x2: M*d ndarray
    :param metric: string type in {'rbf', 'linear', 'polynomial'}
    :param coef0:
    :param degree:
    :param gamma:
    :return:
    '''
    if x1.ndim == 1:
        x1 = x1.reshape(1, x1.size)
    if x2.size > 0 and x2.ndim == 1:
        x2 = x2.reshape(1, x2.size)
    if x2.size > 0 and x2.shape[1] != x1.shape[1]:
        exit("dims are not matched. exit!")
    if metric not in ['linear','rbf','polynomial']:
        exit('kernel type is not supported!')

    if metric == 'linear':
        if x2.size == 0:
            return coef0*np.dot(x1, x1.T)
        else:
            return coef0*np.dot(x1, x2.T)

    if metric == 'rbf':
        # TODO: this is still very slow
        if x2.size == 0:
            n,d = x1.shape
            x1_norm = np.einsum('ij,ij->i', x1, x1)
            eu_dists = x1_norm.reshape(n,1) - 2*np.dot(x1,x1.T) + x1_norm.reshape(1,n)
        else:
            n1,d1 = x1.shape
            n2,d2 = x2.shape
            x1_norm = np.einsum('ij,ij->i', x1, x1)
            x2_norm = np.einsum('ij,ij->i', x2, x2)
            eu_dists = x1_norm.reshape(n1,1) - 2*np.dot(x1,x2.T) + x2_norm.reshape(1,n2)
        return np.exp(-gamma*eu_dists)




