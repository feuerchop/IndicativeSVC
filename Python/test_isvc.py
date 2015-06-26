__author__ = 'morgan'

# main function to test the SVC Model
from matplotlib import pyplot as plt
import utils
from isvc import iSVC
import numpy as np
import isvc_gui as gui

clf = iSVC(display=True, gamma=1)
#generate data
X, y = utils.gen_noise_gauss(100)
n = np.size(X,0)
big_x, big_y = utils.getBoxbyX(X, grid=30)
big_xy = np.c_[big_x.reshape(big_x.size, 1), big_y.reshape(big_x.size, 1)]
# add supervised labels
y_semisupervised= np.zeros((n,1))
n_perm = np.random.permutation(n) # random partition for labeling

labeled_fraction = 0.1

num_labels = int(np.ceil(labeled_fraction*n)) # number of labeled examples based on the fraction of dataset size
print "Number of labels used: " + str(num_labels)
y_semisupervised[n_perm[0:num_labels]] = y[n_perm[0:num_labels]].reshape(num_labels,1) # label selected examples
#x_labeled = X[n_perm[0:num_labels], :]
#y_labeled = y[n_perm[0:num_labels]]
x_labeled_normal = X[np.where(y_semisupervised==-1)[0], :]
x_labeled_anomaly = X[np.where(y_semisupervised==1)[0],:]
x_labeled_unknown = X[np.where(y_semisupervised==0)[0],:]

#build SVC model
#clf.fit(X, y_semisupervised)
# distance to center a
#dist_all = clf.decision_function(big_xy)
# plot
#outliers = X[clf.bsv_ind, :]


#print "Number of support vectors: " + str(np.size(clf.sv_ind,0))
#print "Number of inside points: " + str(np.size(clf.inside_ind,0))
#print "Number of outside points: " + str(np.size(clf.bsv_ind,0))

# plt.plot(X[clf.bsv_ind, 0], X[clf.bsv_ind, 1], 'bo')
# plt.plot(X[clf.inside_ind, 0], X[clf.inside_ind, 1], 'go')
# plt.plot(X[clf.sv_ind, 0], X[clf.sv_ind, 1], 'ro')
#
# plt.plot(x_labeled_normal[:,0], x_labeled_normal[:,1], 'g*', markersize=12)
# plt.plot(x_labeled_anomaly[:,0], x_labeled_anomaly[:,1], 'b*', markersize=12)
# plt.contourf(big_x,big_y,dist_all.reshape(big_x.shape), 100, cmap='bone_r', alpha=0.2)
# plt.contour(big_x,big_y,dist_all.reshape(big_x.shape), [clf.r], colors='k')
# utils.setAxSquare(plt.gca())
# plt.show()

model = gui.Model()
ctrl = gui.Controller(model)
# View
root = gui.Tk()
root.title("iSVC Demo")
view = gui.View(root, ctrl)
model.views.append(view)


for anomaly in x_labeled_anomaly:
    model.add_sample(anomaly[0], anomaly[1], 1)


for normal in x_labeled_normal:
    model.add_sample(normal[0], normal[1], -1)


for unknown in x_labeled_unknown:
    model.add_sample(unknown[0], unknown[1], 0)


model_params = {'C':0.01, 'gamma':0.5, 'coef0':1, 'degree':3, 'kernel':'rbf'}
model.fit(model_params)



root.mainloop()


