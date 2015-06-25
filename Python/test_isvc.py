__author__ = 'morgan'

# main function to test the SVC Model
from matplotlib import pyplot as plt
import utils
from isvc import iSVC
import numpy as np

clf = iSVC(display=True, gamma=1)
#generate data
X, y = utils.gen_noise_gauss(100)
big_x, big_y = utils.getBoxbyX(X, grid=30)
big_xy = np.c_[big_x.reshape(big_x.size, 1), big_y.reshape(big_x.size, 1)]
#build SVC model
clf.fit(X)
# distance to center a
dist_all = clf.decision_function(big_xy)
# plot
outliers = X[clf.bsv_ind, :]
plt.plot(X[clf.inside_ind, 0], X[clf.inside_ind, 1], 'ko', mfc='None')
# plt.plot(X[clf.sv_ind, 0], X[clf.sv_ind, 1], 'bo')
# plt.plot(X[clf.sv_ind, 0], X[clf.sv_ind, 1], 'o', markerfacecolor='None', ms=12, markeredgewidth=1.5, markeredgecolor='k')
plt.plot(X[clf.bsv_ind, 0], X[clf.bsv_ind, 1], 'ks', mfc='None')
plt.plot(X[clf.bsv_ind, 0], X[clf.bsv_ind, 1], 'kx', mfc='None')
plt.contourf(big_x,big_y,dist_all.reshape(big_x.shape), 100, cmap='bone_r', alpha=0.2)
plt.contour(big_x,big_y,dist_all.reshape(big_x.shape), [clf.r], colors='k')
utils.setAxSquare(plt.gca())
plt.show()