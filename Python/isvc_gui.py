# let's build a SVM demo

from Tkinter import *
import numpy as np
from sklearn.svm import SVC, OneClassSVM
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pylab as plt
from matplotlib.contour import ContourSet
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure

class Model(object):

    def __init__(self):
        self.data = []
        self.clf = None
        self.views = []
        self.DEBUG_INFO = ""
        self.is_fitted = False

    def add_sample(self, x,y,label):
        self.data.append((x,y,label))
        self.changed("sample_added")

    def clear_samples(self):
        self.data = []
        self.clf = None
        self.is_fitted = False
        self.changed("sample_cleared")

    def fit(self, params):
        if len(self.data) <= 1:
            self.DEBUG_INFO = "No samples are there!"
            self.changed("alert_generated")
            return
        X = np.asarray(self.data)[:,0:2]
        y = np.asarray(self.data)[:,2]
        if np.unique(y).size == 1:
            # only one-class
            self.clf = OneClassSVM(nu=params['nu'],
                             gamma=params['gamma'],
                             degree=params['degree'],
                             coef0=params['coef0'],
                             kernel=params['kernel'])
            self.clf.fit(X)
        else:
            self.clf = SVC(C=params['C'],
                           gamma=params['gamma'],
                           degree=params['degree'],
                           coef0=params['coef0'],
                           kernel=params['kernel'])
            self.clf.fit(X, y)
        self.is_fitted = True
        self.changed("model_fitted")

    def changed(self, status):
        for view in self.views:
            view.update(status, self)

class View(object):

    def __init__(self, parent, controller):
        self.controller = controller
        self.fig = fig = Figure(figsize=(6,5), dpi=100)
        self.ax = fig.add_subplot(111)
        self.contours = []              # used to store objects in axes except the samples themselves
        self.init_ax()
        self.canvas = c = FigureCanvasTkAgg(fig, parent)
        c.get_tk_widget().config(bd=2, cursor="plus")
        c.get_tk_widget().pack(side=TOP)
        self.toolbar = tbar = Toolbar(parent, bd=2)
        tbar.pack(side=TOP, expand=True, fill=BOTH)
        self.console = console = Text(parent, bd=2, relief=GROOVE)
        # console.pack(side=LEFT, expand=True)

        # canvas event binding
        c.mpl_connect("button_press_event", self.pick)
        c.mpl_connect("motion_notify_event", self.showpos)
        c.mpl_connect("axes_leave_event", self.clearpos)
        c.show()

        # widget event binding
        self.toolbar.clearbtn.bind("<Button-1>", self.controller.clear_all)
        self.toolbar.fitbtn.bind("<Button-1>", self.controller.fitmodel)
        self.toolbar.console_clearbtn.bind("<Button-1>", self.clear_console)
        self.toolbar.kernel.trace('w', self.set_kernel)
        self.toolbar.C.trace('w', self.set_c)
        self.toolbar.gamma.trace('w', self.set_gamma)
        self.toolbar.nu.trace('w', self.set_nu)
        self.toolbar.degree.trace('w', self.set_degree)

    def set_kernel(self, *args):
        self.controller.params['kernel'] = self.toolbar.kernel.get()

    def set_c(self, *args):
        self.controller.params['C'] = self.toolbar.C.get()
        # new parameter setted, update model and fit again
        self.controller.fitmodel(event=None)

    def set_nu(self, *args):
        self.controller.params['nu'] = self.toolbar.nu.get()
        # new parameter setted, update model and fit again
        self.controller.fitmodel(event=None)

    def set_gamma(self, *args):
        self.controller.params['gamma'] = self.toolbar.gamma.get()
        # new parameter setted, update model and fit again
        self.controller.fitmodel(event=None)

    def set_degree(self, *args):
        self.controller.params['degree'] = self.toolbar.degree.get()

    def pick(self, event):
        if event.xdata and event.ydata:
            if event.button == 1:
                self.controller.add_sample(event.xdata, event.ydata, -1)
            elif event.button == 3:
                self.controller.add_sample(event.xdata, event.ydata, 1)

    def showpos(self, event):
        if event.xdata and event.ydata:
            self.toolbar.xylabel.config(text="X:%.2f, Y:%.2f" % (event.xdata, event.ydata))

    def clearpos(self, event):
        self.toolbar.xylabel.config(text="X:-, Y:-")

    # View receive events from model or controller
    def update(self, status, model):
        if status == "sample_added":
            if model.data[-1][2] == 1:
                color = 'none'
            else:
                color = 'k'
            self.ax.plot(model.data[-1][0], model.data[-1][1], 'o', mfc=color)
            if model.is_fitted is True:
                # new sample coming, update model and fit again
                self.controller.fitmodel(event=None)

        if status == "sample_cleared":
            self.remove_surface()
            self.ax.cla()
            self.init_ax()
            self.console.delete(1.0, END)

        if status == "alert_generated":
            self.console.insert(END, model.DEBUG_INFO+"\n")

        if status == "model_fitted":
            self.remove_surface()
            self.plot_contour(model)
            self.console.insert(END, "%s fitted on %d samples.\n" % (type(model.clf).__name__, len(model.data)))

        self.canvas.draw()

    def init_ax(self):
        xlim, ylim = [0,4], [0,4]
        self.ax.set(xticks=[], yticks=[])
        self.ax.set(xlim=xlim, ylim=ylim)

    def clear_console(self, event):
        self.console.delete(1.0, END)

    def remove_surface(self):
        """Remove old decision surface."""
        if len(self.contours)>0:
            for contour in self.contours:
                if isinstance(contour,ContourSet):
                    for lineset in contour.collections:
                        lineset.remove()
                else:
                    contour.remove()
            self.contours=[]

    def plot_contour(self, model):
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        Ax,Ay = np.meshgrid(np.linspace(xlim[0], xlim[1], 30), np.linspace(ylim[0], ylim[1], 30))
        all_points_x = np.concatenate( (Ax.ravel().reshape(Ax.size, 1),
                                  Ay.ravel().reshape(Ay.size, 1)), 1)
        N  = all_points_x.shape[0]
        dists = model.clf.decision_function(all_points_x)
        c1 = self.ax.contour(Ax,Ay,dists.reshape(Ax.shape), levels=[0], colors='k', linestyles='solid', linewidths=[2])
        c2 = self.ax.contour(Ax,Ay,dists.reshape(Ax.shape), levels=[1,-1], colors='k', linestyles='dotted', linewidths=[2])
        # plot support vectors
        c3 = self.ax.scatter(model.clf.support_vectors_[:,0], model.clf.support_vectors_[:,1],
                             s=200, edgecolors="k", facecolors="none")
        c4 = self.ax.contourf(Ax,Ay,dists.reshape(Ax.shape), 10, cmap='bone', alpha=0.5)
        self.contours.append(c1)
        self.contours.append(c2)
        self.contours.append(c3)
        self.contours.append(c4)



class Controller(object):

    def __init__(self, model):
        self.params = {'C':1.0, 'gamma':0.5, 'degree':2, 'kernel':"rbf", 'coef0':0.0, 'nu':0.5}
        self.model = model

    def add_sample(self, x, y, label):
        self.model.add_sample(x, y, label)

    def clear_all(self, event):
        self.model.clear_samples()

    def fitmodel(self, event):
        self.model.fit(self.params)


class Toolbar(Frame):

    def __init__(self, parent, *args, **kwargs):
        Frame.__init__(self, parent, *args, **kwargs)
        self.cframe = Frame(self)
        self.nuframe = Frame(self)
        self.gframe = Frame(self)
        self.dframe = Frame(self)
        # Var
        self.kernel = StringVar()
        self.C = DoubleVar()
        self.gamma = DoubleVar()
        self.coef0 = DoubleVar()
        self.degree = IntVar()
        self.nu = DoubleVar()
        self.is_surface = BooleanVar()
        self.esf = IntVar()

        # Constants
        kernel_list = ("rbf", "linear", "poly")
        degree_list = [1,2,3,4,5]
        self.kernel.set(kernel_list[0])
        self.C.set(1.0)
        self.gamma.set(0.5)
        self.coef0.set(0.0)
        self.nu.set(0.5)
        self.degree.set(2)
        self.is_surface.set(True)
        self.esf = 0

        # widgets
        self.xylabel = Label(self, width=10, text="X:-, Y:-", relief=GROOVE)
        self.fitbtn = Button(self, text="fit", width=5)
        self.clearbtn = Button(self, text="reset", width=5)
        self.console_clearbtn = Button(self, text="clear log", width=5)
        self.erfbtn = Checkbutton(self, text="Error surface", variable=self.esf, width=12)
        self.c_label = Label(self.cframe, text="C")
        self.c_slider = Scale(self.cframe, from_=0.1, to=10, resolution=0.5, orient=HORIZONTAL,
                              relief=FLAT, sliderlength=20, sliderrelief=GROOVE,
                              tickinterval=5, variable=self.C)
        self.nu_label = Label(self.nuframe, text="Nu")
        self.nu_slider = Scale(self.nuframe, from_=0, to=1, resolution=0.05, orient=HORIZONTAL,
                              relief=FLAT, sliderlength=20, sliderrelief=GROOVE,
                              tickinterval=0.5, variable=self.nu)
        self.g_label = Label(self.gframe, text="gamma")
        self.g_slider = Scale(self.gframe, from_=0.1, to=10, resolution=0.5, orient=HORIZONTAL,
                              relief=FLAT, sliderlength=20, sliderrelief=GROOVE,
                              tickinterval=5, variable=self.gamma)
        self.d_label = Label(self.dframe, text="degree")
        self.degree_list = OptionMenu(self.dframe, self.degree, *degree_list)
        self.degree_list.config(width=4)
        self.kernel_list = OptionMenu(self, self.kernel, *kernel_list)
        self.kernel_list.config(width=10)

        # layout
        # row 0
        self.kernel_list.grid(row=0, column=0, sticky=E+W)
        self.fitbtn.grid(row=0, column=1, sticky=E+W)
        self.clearbtn.grid(row=0, column=2, sticky=E+W)
        self.console_clearbtn.grid(row=0, column=3, sticky=E+W)
        self.erfbtn.grid(row=0, column=4, stick=E+W)

        # row 1
        self.c_label.pack(side=LEFT)
        self.c_slider.pack(side=LEFT)
        self.nu_label.pack(side=LEFT)
        self.nu_slider.pack(side=LEFT)
        self.g_label.pack(side=LEFT)
        self.g_slider.pack(side=LEFT)
        self.d_label.pack(side=LEFT)
        self.degree_list.pack(side=LEFT)
        self.cframe.grid(row=1, column=0)
        self.nuframe.grid(row=1, column=1)
        self.gframe.grid(row=1, column=2)
        self.dframe.grid(row=1, column=3)
        self.xylabel.grid(row=1, column=4)


if __name__ == '__main__':
    # Model
    m = Model()
    # Controller
    ctrl = Controller(m)
    # View
    root = Tk()
    root.title("SVM Demo")
    v = View(root, ctrl)
    m.views.append(v)
    root.mainloop()
