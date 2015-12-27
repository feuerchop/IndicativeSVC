# from sklearn.metrics.pairwise import pairwise_kernels as kernel
from H3Kernels import kernel
import numpy as np


class H3iSVC(object):
    """
    Class indicative SVC (Support vector clustering) using Ben-Hur's hypersphere solution
    given partially labelled y.
    """

    def __init__(self, kernel='rbf', similarity=0.9,
                 C=1.0, gamma=0.5, coef0=1, degree=3, eps=1e-4,
                 labeling=False, display=False):
        self.eps = eps                          # numerical tolerance level
        self.kernel = kernel                    # kernel string, default is rbf, only 'rbf', 'linear', 'poly' are supported
        self.alpha = None                       # dual variables
        self.sv_ind = None                      # indices of SVs
        self.bsv_ind = None                     # indices of BSVs
        self.a2 = None                          # squared norm of center a
        self.sv = None                          # store SVs + BSVs
        self.sv_inx = None                      # indices of SVs+BSVs
        self.nsv = None                         # number of SVs+BSVs
        self.r = None                           # radius of the hypersphere
        self.C = C                              # regularizer coef, default 1, no outlier is allowed
        self.c_weighted = C                     # regularizer coef reweighted by indicative labels
        self.k_diagonal = None                  # cache of diagonal kernel entries
        self.impact_abnormal_indices = []       # indices of potentially abnormal points
        self.impact_normal_indices = []         # indices of potentially normal points
        self.similarity = similarity            # threshold to control the impact of given labels
        self.gamma = gamma                      # param for rbf/poly kernel, default 1
        self.coef0 = coef0                      # param for polynomial kernel, default 1
        self.degree = degree                    # param for polynomial kernel, default 3
        self.cluster_labels = None              # cluster labels
        self.y = None                           # class labels, +1 abnormal, -1 normal
        self.labeling = labeling                # call labeling process, default False
        self.display = display                  # show details of solver

    def reweight_c(self, X, y, h):
        """
        reweight regularization parameter C using given labels
        :param K: computed kernel matrix
        :param y: partially given labels
        :param h: bandwidth for impact area, normally <= 1
        :return: reweigth c parameters
        """
        N = y.size
        anomaly_idx = np.where(y == 1)[0]
        normal_idx = np.where(y == -1)[0]
        self.c_weighted = self.C*np.ones(N)

        # we use rbf kernel matrix for similarity measure
        # similarity from 0 to 1, if two points are identical, they are set as 1 else 0
        K_abnormal = kernel(X[anomaly_idx], X, metric='rbf',  filter_params=True, gamma=self.gamma,
                                coef0=self.coef0, degree=self.degree)
        K_normal = kernel(X[normal_idx], X, metric='rbf',  filter_params=True, gamma=self.gamma,
                                coef0=self.coef0, degree=self.degree)
        # filter out far points
        K_abnormal[K_abnormal < h] = 0
        K_normal[K_normal < h] = 0
        impact_abnormal_indices = np.where(K_abnormal >= h)
        impact_normal_indices = np.where(K_normal >= h)
        K_impact = np.r_[K_abnormal, -K_normal]
        # positive impact score means the sample is more likely to be outlier
        def compute_f(K, id):
            nonzeros = K[:, id].nonzero()[0].size
            if nonzeros == 0:
                return 0
            else:
                return K[:, id].sum()/float(nonzeros)

        impact_scores = np.array([compute_f(K_impact, idx) for idx in np.arange(K_impact.shape[1])])
        # pos_idx = np.where(impact_score > 0)[0]     # likely anomaly
        # neg_idx = np.where(impact_score < 0)[0]     # likely normal
        for idx, score in enumerate(impact_scores):
            if score > 0:
                # more likely abnormal
                self.c_weighted[idx] = self.C*(1-score)/(1-h)+(1.0/N)*(score-h)/(1-h)
            elif score < 0:
                # more likely normal
                self.c_weighted[idx] = self.C*(1+score)/(1-h)-(score+h)/(1-h)
            else:
                pass
        return (impact_abnormal_indices, impact_normal_indices)

    def fit(self, X, y=None):

        '''
        Main training function of SVC
        :param X: Nxd dataset
        :param y: Input supervised labels or none
        :return: trained SVC model
        '''

        n, d = X.shape
        iter = 0

        def setup_model(sol):
            """
            Embedded method to wrap up model properties after training
            """
            alpha = np.asarray(sol)
            inx = np.where(alpha > self.eps)[0]
            self.sv_inx = inx
            self.alpha = alpha[inx]
            self.nsv = inx.size
            self.sv_ind = np.where((np.ravel(alpha) > self.eps) & (np.ravel(alpha) < self.c_weighted-self.eps))[0]
            self.bsv_ind = np.where(np.ravel(alpha) >= self.c_weighted-self.eps)[0]
            self.sv = X[inx, :]
            self.y = -1*np.ones(n)
            self.y[self.bsv_ind] = 1

        # ----- main fit loop ------
        # setup supervised labels if they exist
        if y is not None:
            anomaly_idx = np.where(y == 1)[0]
            normal_idx = np.where(y == -1)[0]
            # reweigh C values
            impact_abnormal_indices, impact_normal_indices = self.reweight_c(X, y, self.similarity)
            self.impact_abnormal_indices = [anomaly_idx[impact_abnormal_indices[0]], impact_abnormal_indices[1]]
            self.impact_normal_indices = [normal_idx[impact_normal_indices[0]], impact_normal_indices[1]]
        else:
            anomaly_idx = normal_idx = np.array([], dtype=int)
            self.c_weighted = self.C*np.ones(n)

        def take_step(i1, i2, a):
            """
            SMO-type solver see Fan et al. (JMLR 2005) paper
            Ref.: Fan et al., Working set selection using second order information for training
                  support vector machines. JMLR 2005.10.
            Update alpha1 and alpha2 in SMO
            i1, i2: indices of 1st, 2nd candidate variables
            We denote set B as indices set [i1,i2] and N as the rest of indices
            """

            # if x1 and x2 is identical, update fails
            # TODO: for high-dimensional data, this could be slow.
            if np.all(X[i1] == X[i2]):
                if self.display:
                    print 'identical samples found!'
                return 0
            alph1, alph2 = a[i1], a[i2]
            delta = alph1+alph2
            # determine lower/upper bound for alpha_2
            L = np.array([0, delta-self.C]).max()
            H = np.array([self.C, delta]).min()
            # if L/H are identical, exit update
            # if np.abs(L-H) < 2*self.eps:
            #     if self.display:
            #         print 'identical bounds found!'
            #     return 0

            # compute kernel submatrix K_BN which is the only requirement to update solution
            # TODO: replace with own kernel function
            K_BN = kernel(X[(i1,i2),:], X, metric=self.kernel,  filter_params=True, gamma=self.gamma,
                         coef0=self.coef0, degree=self.degree)
            k11 = K_BN[0, i1]
            k22 = K_BN[1, i2]
            k12 = K_BN[0, i2]

            # compute 2nd derivative
            eta = k11+k22-2*k12
            dist1 = (k11 - 2*np.dot(a[self.sv_inx].reshape(1,self.nsv),
                                   K_BN[0, self.sv_inx].reshape(self.nsv,1)) + self.a2)[0,0]
            dist2 = (k22 - 2*np.dot(a[self.sv_inx].reshape(1,self.nsv),
                                   K_BN[1, self.sv_inx].reshape(self.nsv,1)) + self.a2)[0,0]
            if eta > 0:
                # 2nd derivative posiitve means that minimum is along the line
                alph2_new = alph2 + 0.5*(dist2-dist1)/eta
                # keep it within bound
                if alph2_new < L+self.eps:
                    alph2_new = L
                elif alph2_new > H-self.eps:
                    alph2_new = H
            else:
                # 2nd derivative negative means minimum is on the bounds
                # see my working notes
                v1_v2 = (0.5-alph1)*k11 - (0.5-alph2)*k22 - (alph2-alph1)*k12 -\
                        0.5*(dist1-dist2)
                if L == 0:
                    delta_obj = (delta**2 - delta)*(k11-k22) + 2*delta*v1_v2
                else:
                    # L = delta - C
                    delta_obj = (-2*self.C+delta**2-2*delta*self.C+delta)*k11 -\
                                (delta**2-2*delta*self.C-delta+2*self.C)*k22 +\
                                (4*self.C-2*delta)*v1_v2
                if delta_obj < -self.eps:
                    alph2_new = L    # Lobj
                elif delta_obj > self.eps:
                    alph2_new = H    # Hobj
                else:
                    alph2_new = alph2
            alph1_new = delta-alph2_new
            a[i1], a[i2] = alph1_new, alph2_new
            delta_a1 = alph1_new - alph1
            delta_a2 = alph2_new - alph2
            delta_aB = np.array([[delta_a1], [delta_a2]])

            # setup model properties
            setup_model(a)

            # update a2, r, fprime
            # see my working notes
            N_inx = np.setdiff1d(self.sv_inx, [i1,i2])
            self.a2 = (self.a2 + (alph1_new**2 - alph1**2)*k11 + (alph2_new**2 - alph2**2)*k22 + \
                      (2*alph1_new*alph2_new-2*alph1*alph2)*k12 + \
                      2*delta_aB.T.dot(K_BN[:, N_inx]).dot(a[N_inx].reshape(N_inx.size,1)))[0,0]

            if i2 in self.sv_ind:
                self.r = k11 - 2*np.dot(a[self.sv_inx].reshape(1,self.nsv),
                                K_BN[0, self.sv_inx].reshape(self.nsv,1)) + self.a2
            elif i1 in self.sv_ind:
                self.r = k22 - 2*np.dot(a[self.sv_inx].reshape(1,self.nsv),
                                   K_BN[1, self.sv_inx].reshape(self.nsv,1)) + self.a2
            else:
                self.r = 0.5*(k11 - 2*np.dot(a[self.sv_inx].reshape(1,self.nsv),
                                                K_BN[0, self.sv_inx].reshape(self.nsv,1)) + self.a2 + \
                              k22 - 2*np.dot(a[self.sv_inx].reshape(1,self.nsv),
                                                K_BN[1, self.sv_inx].reshape(self.nsv,1)) + self.a2)
            self.r = self.r[0,0]
            self.__fprime = self.__fprime + 2*K_BN.T.dot(delta_aB).ravel()

            if self.display:
                fval = self.a2 - a.reshape(1,n).dot(self.k_diagonal.reshape(n,1))[0,0]
                print 'iter.%d > [%d] %.3f -> %.3f | [%d] %.3f -> %.3f | r: %.3f | fval: %.3f |[eta] %.2f | SVs: %d | BSVs: %d ' \
                      %(iter, i2, alph2, alph2_new,
                        i1, alph1, alph1_new,
                        self.r, fval, eta,
                        self.sv_ind.size, self.bsv_ind.size)
            return 1

        # ----- smo main loop -------
        a = np.zeros(n)
        # 1. Initialize: set n*C-1 points to be 1/(n*C)
        init_set = np.random.choice(n, int(1.0/self.C)-1)
        a[init_set] = self.C
        # Make sure the sum of alpha is 1
        a[np.random.choice(np.setdiff1d(np.arange(n), init_set), 2)] = 0.5*(1-a[init_set].sum())
        # Setup initial model
        setup_model(a)
        #TODO: random start may save time at beginning for large dataset
        k_inx = kernel(X[self.sv_inx], metric=self.kernel,  filter_params=True, gamma=self.gamma,
                           coef0=self.coef0, degree=self.degree)
        self.a2 = (self.alpha.reshape(1, self.nsv).dot(k_inx).reshape(1, self.nsv).dot(self.alpha.reshape(self.nsv, 1)))[0,0]
        # cache the kernel diagonal
        self.k_diagonal = np.array([kernel(X[i], metric=self.kernel,  filter_params=True, gamma=self.gamma,
                           coef0=self.coef0, degree=self.degree)[0,0] for i in np.arange(n)])
        K_sv = kernel(X, self.sv, metric=self.kernel,  filter_params=True,
                     gamma=self.gamma, coef0=self.coef0, degree=self.degree)
        dists = (self.k_diagonal-2*self.alpha.reshape(1, self.nsv).dot(K_sv.T)+self.a2*np.ones(n)).ravel()
        self.r = dists.max()
        # initial gradients
        self.__fprime = self.a2 - dists
        I_up = np.where(a < self.C - self.eps)[0]
        I_low = np.where(a > self.eps)[0]
        m_lower = (-self.__fprime[I_up]).max()
        m_upper = (-self.__fprime[I_low]).min()

        tau = 1
        while m_lower - m_upper > 1e-3:
            # stopping criteria: m(a) - M(a) < eps
            # print m_lower - m_upper
            fprime_up_max = np.max(-self.__fprime[I_up])
            idx_i = I_up[np.argmax(-self.__fprime[I_up])]
            idx_j_candidates = I_low[np.where(-self.__fprime[I_low] < fprime_up_max)[0]]
            # TODO: handle non-psd case
            eta_valid = self.k_diagonal[idx_i]+self.k_diagonal[idx_j_candidates]- \
                                        2*kernel(X[idx_i], X[idx_j_candidates],
                                        metric=self.kernel,  filter_params=True,
                                        gamma=self.gamma, coef0=self.coef0, degree=self.degree).ravel()
            # negative 2nd derivative should have a positive tau, see Ref. paper
            eta_valid[np.where(eta_valid <=0)[0]] = tau
            idx_j = idx_j_candidates[np.argmin([-(-self.__fprime[idx_i] + self.__fprime[j])**2/eta_valid[i] \
                                       for i,j in enumerate(idx_j_candidates)])]
            # iterate main routine
            take_step(idx_i, idx_j, a)
            # after take step, we need to update optimal conditions
            I_up = np.where(a < self.C - self.eps)[0]
            I_low = np.where(a > self.eps)[0]
            m_lower = (-self.__fprime[I_up]).max()
            m_upper = (-self.__fprime[I_low]).min()
            iter += 1

        # cluster labeling process
        if self.labeling:
            self.assign_clusters(X)

    def assign_clusters(self, X):
        '''
        Assign samples to different
        :param X: dataset Nxd
        :return: class labels of X
        '''
        if self.sv is None:
            print 'Model is not trained yet!'
            exit(1)

        def is_same_cluster(x, y):
            # sampling 10 points along x-y line segment
            flag = True
            for point in [x+r*(y-x) for r in np.arange(0.1, 1.1, 0.1)]:
                if self.decision_function(point.reshape(1, x.size)) > self.r+self.eps:
                    # some point lies outside the ball
                    flag = False
                    break
            return flag

        # initialize with one class with first point
        y_labels = -np.ones_like(self.y)
        normal_idx = np.where(self.y == -1)[0]
        y_labels[normal_idx] = 0
        # first normal point
        seeds = {1:[X[normal_idx[0], :]]}
        for id in np.arange(1, X.shape[0]):
            if id in normal_idx:
                point = X[id, :]
                is_new_class = True
                for label in seeds.keys():
                    if is_same_cluster(point, seeds[label][0]):
                        seeds[label].append(point)
                        y_labels[id] = label
                        is_new_class = False
                        break
                if is_new_class:
                    # we found a new class
                    seeds[label+1] = [point]
                    y_labels[id] = label+1
                    if self.display:
                        print 'New cluster found:'
                        print "\n".join(['Cluster[{:d}]: {:d} samples.'.format(k, len(l)) for k, l in seeds.items()])
        self.cluster_labels = y_labels

    def predict_y(self, Xtt):
        """
        Predict labels for testing dataset
        :param Xtt: M*D ndarray
        :return: labels for Xtt, +1 for outliers, -1 for normal points
        """
        dist_tt = self.decision_function(Xtt)
        y = -1*np.ones(Xtt.shape[0])
        y[dist_tt.ravel() > self.r] = 1
        return y

    def decision_function(self, X):
        """
        Compute the distances of X to the ball center
        :param X: input samples X
        :return: scalar or 1-dim array for squared distance
                 between X in the feature space and the center hyperball
        """
        if X.ndim == 1:
            X = X.reshape(1, X.size)
        n = X.shape[0]
        K = kernel(X, metric=self.kernel,  filter_params=True, gamma=self.gamma, coef0=self.coef0, degree=self.degree)
        f = K.diagonal()
        # Note that self.sv and self.a2 are needed to be cached in the model to compute the distance
        K_sv = kernel(X, self.sv, metric=self.kernel,  filter_params=True,
                     gamma=self.gamma, coef0=self.coef0, degree=self.degree)
        d = f-2*self.alpha.reshape(1, self.nsv).dot(K_sv.T)+self.a2*np.ones(n)
        if n == 1:
            d = d[0, 0]
        return d.ravel()

    def draw_model(self, X, ax=None):
        """
        Plot clustering boundary after training
        :param X: training dataset
        :param ax: plot axes handle, defautl current axes as plt.gca()
        Note: need to import matplotlib.pyplot as plt
        """
        if self.sv is None:
            print 'Model is not trained yet!'
            pass
        else:
            minx = min(X[:, 0])
            maxx = max(X[:, 0])
            miny = min(X[:, 1])
            maxy = max(X[:, 1])
            padding_x = 0.05*(maxx-minx)
            padding_y = 0.05*(maxy-miny)
            XX, YY = np.meshgrid(np.linspace(minx-padding_x, maxx+padding_x, 50),
                                 np.linspace(miny-padding_y, maxy+padding_y, 50))
            dist = self.decision_function(np.c_[XX.reshape((XX.size, 1)), YY.reshape((YY.size, 1))])

            if ax is None: ax = plt.gca()
            ax.contour(XX, YY, dist.reshape(XX.shape), levels=[self.r], colors='k', linewidths=[1])
            ax.contourf(XX, YY, dist.reshape(XX.shape), 5, cmap='bone', alpha=0.2)
            if self.cluster_labels is None:
                ax.plot(X[:, 0], X[:, 1], 'k.', ms=4)
                ax.plot(X[self.bsv_ind, 0], X[self.bsv_ind, 1], 'r.', ms=4)
            else:
                ax.scatter(X[self.cluster_labels > 0 , 0], X[self.cluster_labels > 0 , 1],
                           c=100*self.cluster_labels[self.cluster_labels>0], alpha=0.7)
                ax.plot(X[self.bsv_ind, 0], X[self.bsv_ind, 1], 'k.', ms=4)
            ax.plot(X[self.sv_ind, 0], X[self.sv_ind, 1], 'ko', ms=12, mfc='none', mec='k')
            a_all = np.zeros(X.shape[0])
            a_all[self.sv_inx] = self.alpha
            ax.set(title='SVC using SMO')
            # for i in np.arange(X.shape[0]):
            #     plt.annotate('{:.3f}'.format(a_all[i]), X[i])
            # set squared axis
            xlim0, xlim1 = ax.get_xlim()
            ylim0, ylim1 = ax.get_ylim()
            ax.set_aspect((xlim1-xlim0)/(ylim1-ylim0))



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from scipy.io import loadmat,savemat
    import cProfile
    # X = np.random.multivariate_normal([0, 0], [[0.5, 0], [0, 0.5]], 100)
    # y = -np.ones(X.shape[0])
    # Xa = np.random.rand(5, 2)
    # ya = np.ones(Xa.shape[0])
    # Xtr, ytr = np.r_[X, Xa], np.r_[y, ya]
    # data = savemat("syndata.mat", mdict={'Xtr': Xtr, 'ytr': ytr})
    data = loadmat('../datasets/syndata.mat')
    Xtr, ytr = data['X'], data['Y']
    clf1 = H3iSVC(C=0.005, gamma=0.5, kernel='linear', display=False, labeling=False)
    cProfile.run("clf1.fit(Xtr)")
    # from H3iSVC_QP import H3iSVC_QP as svcqp
    # clf2 = svcqp(C=0.05, gamma=1.5, kernel='rbf')
    # clf2.fit(Xtr)
    print '--- done! ---'
    plt.subplot(1,1,1)
    clf1.draw_model(Xtr)
    # # plt.subplot(1,2,2)
    # # clf2.draw_model(Xtr)
    plt.show()

