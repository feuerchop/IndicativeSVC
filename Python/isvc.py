from sklearn.metrics.pairwise import pairwise_kernels as kernel
from cvxopt import matrix as cvxmatrix
from cvxopt import solvers
import numpy as np

class iSVC:
    """
    Class indicative SVC (Support vector clustering) using Ben-Hur's hypersphere solution
    given partially labelled y
    """
    def __init__(self, kernel='rbf', method='qp',
                 C=1.0, gamma=0.5, coef0=1, degree=3, eps=1e-3,
                 labeling=False, cached=False, display=False):
            self.eps = eps
            self.method = method            # optimization methods, default use qp-solver
            self.kernel = kernel            # kernel string, default is rbf, only 'rbf', 'linear', 'poly' are supported
            self.alpha = None               # dual variables
            self.sv_ind = None              # indices of SVs
            self.bsv_ind = None             # indices of BSVs
            self.inside_ind = None          # indices of insides+SVs
            self.b = None                   # squared norm of center a
            self.sv = None                  # SVs + BSVs     
            self.sv_inx = None              # indices of SVs+BSVs
            self.nsv = None                 # number of SVs+BSVs
            self.r = None                   # radius of the hypersphere
            self.rid = None                 # Index of the SV for R
            self.K = None                   # cache of the kernel matrix on X
            self.C = C                      # regularizer coef, default 1, no outlier is allowed
            self.c_weighted = C             # regularizer coef reweighted by indicative labels
            self.gamma = gamma              # param for rbf/poly kernel, default 1
            self.coef0 = coef0              # param for polynomial kernel, default 1
            self.degree = degree            # param for polynomial kernel, default 3
            self.fval = None                # objective function value after converge
            self.cluster_labels = None      # cluster labels
            self.y = None                   # class labels, +1 normal, -1 outlier
            self.labeling = labeling        # call labeling process, default False
            self.cached = cached            # wether store K in cache
            self.display = display          # show details of solver
            
    #===========================================================================
    # fit function: 
    # X: nxd data set
    #===========================================================================
    def fit(self, X, y=None):
        if self.display:
            solvers.options['show_progress'] = True
        else:
            solvers.options['show_progress'] = False
        
        n = X.shape[0]
        
        # Step 1: Optimizing the SVDD dual problem.....
        K = kernel(X, metric=self.kernel, n_jobs=1,
                   filter_params=True, gamma=self.gamma,
                   coef0=self.coef0, degree=self.degree)
        q = cvxmatrix(-K.diagonal(), tc='d')

        #TODO: make it a separate function call

        influence = np.zeros(n) # f_i in the paper
        influence_anomaly = np.zeros(n)
        influence_normal = np.zeros(n)
        c_weighted = np.zeros(n) # c_i in the paper

        anomaly_indices = np.where(y==1)[0]
        normal_indices = np.where(y==-1)[0]

        for i in range(0, n):
            sum_kernels_anomaly = 0
            sum_kernels_normal=0
            num_samples_anomaly=0
            num_samples_normal=0
            for j in range(0, n):
                if(y[j]==1): # anomaly
                    if (K[i,j]>np.exp(-2)):
                        sum_kernels_anomaly+=K[i,j]
                        num_samples_anomaly+=1

                elif(y[j]==-1): # normal
                    if (K[i,j]>np.exp(-2)):
                        sum_kernels_normal+=K[i,j]
                        num_samples_normal-=1
                else:
                    pass # label unknown

            if (num_samples_anomaly>0):
                influence_anomaly[i] = sum_kernels_anomaly/float(num_samples_anomaly)
                influence[i] += influence_anomaly[i]

            if (num_samples_normal<0):
                influence_normal[i] = sum_kernels_normal/float(num_samples_normal)
                influence[i] += influence_normal[i]


            if (influence[i]>np.exp(-2)):
                c_weighted[i] = self.C*(1-influence[i])/(1-np.exp(-2))+(1/np.size(X,0))*(influence[i]-np.exp(-2))/(1-np.exp(-2))
            elif (influence[i]<(-(np.exp(-2)))):
                c_weighted[i] = self.C*(1-np.abs(influence[i]))/(1-np.exp(-2))+(np.abs(influence[i])-np.exp(-2))/(1-np.exp(-2))
            else:
                c_weighted[i]=0

            if (c_weighted[i]<0):
                print "Negative Impact!!!"
                exit(-1)

        # solver
        if self.method is 'qp':

            P = cvxmatrix(2*K, tc='d')
            G = cvxmatrix(np.vstack((-np.eye(n), np.eye(n))), tc='d')                   # lhs box constraints
            #h = cvxmatrix(np.concatenate((np.zeros(n), self.C*np.ones(n))), tc='d')     # rhs box constraints
            # TODO: c values for normal samples need to be minused by 2*self.eps to make it strong inequality
            h = cvxmatrix(np.concatenate((np.zeros(n), c_weighted)), tc='d') # zeros for >=0, c_weighted for <=c_i
            # optimize using cvx solver

            Aeq = np.zeros((len(anomaly_indices),n))
            beq = np.zeros(len(anomaly_indices))
            i=0
            for ind in anomaly_indices:
                Aeq[i,ind]=1
                beq[i]=c_weighted[ind]
                i+=1

            A = cvxmatrix(Aeq, tc='d')
            b = cvxmatrix(beq, tc='d')
            # optimize using cvx solver
            sol = solvers.qp(P,q,G,h,A,b,initvals=cvxmatrix(np.zeros(n), tc='d'))

        if self.method is 'smo':
            #TODO: apply SMO algorithm
            alpha = None
            
        # setup SVC model
        alpha = np.asarray(sol['x'])
        inx = np.where(alpha > self.eps)[0]  
        self.sv_inx = inx       
        self.alpha = alpha[inx]
        self.nsv = inx.size
        self.sv_ind = np.where((np.ravel(alpha) > self.eps) & (np.ravel(alpha) < c_weighted-self.eps))[0]
        self.bsv_ind= np.where(np.ravel(alpha) >= c_weighted-self.eps)[0]
        self.inside_ind = np.where(np.ravel(alpha) < c_weighted-self.eps)[0]
        k_inx = K[inx[:,None], inx]                                                     # submatrix of K(sv+bsv, sv+bsv)
        k_sv = K[self.sv_ind[:,None], self.sv_ind]                                      # submatrix of K(sv,sv)
        k_bsv = K[self.bsv_ind[:,None], self.bsv_ind]                                   # submatrix of K(bsv,bsv)
        # 2-norm of center a^2
        self.b = self.alpha.reshape(1,self.nsv).dot(k_inx).reshape(1,self.nsv).dot(self.alpha.reshape(self.nsv,1))
        #including both of SV and BSV (bounded support vectors)
        self.sv= X[inx, :]
        d = k_sv.diagonal() - 2*self.alpha.reshape(1,self.nsv).dot(K[inx[:,None], self.sv_ind]) + self.b * np.ones(self.sv_ind.size)
        self.r = d.max()
        self.rid = self.sv_ind[np.argmax(d.ravel())]
        d_bsv = k_bsv.diagonal() - 2*self.alpha.reshape(1,self.nsv).dot(K[inx[:,None], self.bsv_ind]) + self.b * np.ones(self.bsv_ind.size)
        self.fval = self.r+self.C*(d_bsv - self.r).sum()
        if self.cached: self.K = K
        self.y = -1*np.ones(n)
        self.y[self.bsv_ind] = 1

        if self.labeling:
            #Step 2: Labeling cluster index by using CG
            self.predict(X)

        self.c_weighted = c_weighted
        # self.influence = influence
        # self.res_alpha = alpha


    def incremental(self, x_, y_=None):
        """
        Incremental SVC involving supervised labels
        :param x_: new coming data stream
        :param y_: new coming data label stream, can be none
        :return: update classifier
        """
        # TODO: implement incremental version of iSVC
        pass

    def predict(self, X):
        """
        Predict the cluster labels using CG algorithm
        :param X: input samples X
        """
        def find_adjacent(X):
            """
            The Adjacency matrix between pairs of points whose images lie in
            or on the sphere in feature space.
            (i.e. points that belongs to one of the clusters in the data space
            given a pair of data points that belong to different clusters,
            any path that connects them must exit from the sphere in feature
            space. Such a path contains a line segment of points y, such that:
            kdist2(y,model)>model.r.
            Checking the line segment is implemented by sampling a number of
            points (10 points).
            BSVs are unclassfied by this procedure, since their feature space
            images lie outside the enclosing sphere.( adjcent(bsv,others)=-1 )

            :param X: input samples X
            :return: an adjacent matrix of X
            """
            N = X.shape[0]
            adjacent = np.zeros((N, N))
            R = self.r + self.eps  # Squared radius of the minimal enclosing ball
            for i in xrange(0, N-1):
                for j in xrange(0, N-1):
                    # if the j is adjacent to i - then all j adjacent's are also adjacent to i
                    if j<i:
                        if adjacent[i, j] == 1:
                            adjacent[i, np.where(adjacent[j, :] == 1)[0]] = 1
                    else:
                        # if adajecancy already found - no point in checking again
                        if adjacent[i, j] != 1:
                            # goes over 10 points in the interval between these 2 Sample points
                            # unless a point on the path exits the shpere - the points are adjacnet
                            adj_flag = 1
                            for interval in np.arange(0,1,0.1):
                                z = X[i] + interval * (X[j] - X[i])
                                # calculates the sub-point distance from the sphere's center
                                d = self.kdist2(z.reshape(1,z.size))
                                if d > R:
                                    adj_flag = 0
                                    break
                            if adj_flag == 1:
                                adjacent[i,j] = 1
                                adjacent[j,i] = 1
            return adjacent

        def find_connected_components(adjacent):
            """
            clusters_assignments - label each point with its cluster assignement.
            Finds Connected Components in the graph represented by
            the adjacency matrix, each component represents a cluster.
            :param adjacent: adjacent matrix
            :return: cluster labels
            """
            N = adjacent.shape[0]
            clusters_assignments = np.zeros(N)
            cluster_index = 0
            done = 0
            while done != 1:
                root = 1
                while clusters_assignments[root] != 0:
                    root = root + 1
                    if root > N: #all nodes are clustered
                        done = 1
                        break
                if done != 1: #an unclustered node was found - start DFS
                    cluster_index = cluster_index + 1
                    stack = np.zeros(N)
                    stack_index = 0
                    while stack_index != 0:
                        node = stack[stack_index]
                        stack_index = stack_index - 1
                        clusters_assignments[node] = cluster_index
                        for i in xrange(0, N-1):
                            #check that this node is a neighbor and not clustered yet
                            if (adjacent[node,i] == 1 & clusters_assignments[i] == 0 & i != node):
                                stack_index = stack_index + 1
                                stack[stack_index] = i
            return clusters_assignments

        N = X.shape[0]
        adjacent = find_adjacent(X[self.inside_ind, :])
        clusters = find_connected_components(adjacent)
        self.cluster_labels = np.zeros(N)
        self.cluster_labels[self.inside_ind] = np.double(clusters)

    # predict labels on Xtt 
    def predict_y(self, Xtt):
        dist_tt = self.kdist2(Xtt)
        y = -1*np.ones(Xtt.shape[0])
        y[dist_tt.ravel() > self.r] = 1
        return y

    def decision_function(self, X):
        """
        Compute the distances of X with the ball center
        :param X: input samples X
        :return: Squared distance between vectors in the feature space and the center hyperball
        """
        n = X.shape[0]
        K = kernel(X, metric=self.kernel, filter_params=True, gamma=self.gamma, coef0=self.coef0, degree=self.degree)
        f = K.diagonal()
        K_sv = kernel(X, self.sv, metric=self.kernel, filter_params=True, gamma=self.gamma, coef0=self.coef0, degree=self.degree)
        d = f - 2*self.alpha.reshape(1,self.nsv).dot(K_sv.T) + self.b * np.ones(n)
        return d

