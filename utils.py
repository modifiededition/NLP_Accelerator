import inspect
import collections
from IPython import display
from matplotlib_inline import backend_inline
import matplotlib.pyplot as plt


def add_to_class(Class):
    """Register funtion as method in the original class"""

    def wrapper(obj):
        setattr(Class,obj.__name__, obj)

    return wrapper


class HyperParameters:
    """The base class of hyperparameters."""
    def save_hyperparameters(self, ignore=[]):
        """Defined in :numref:`sec_oo-design`"""
        raise NotImplemented

    def save_hyperparameters(self, ignore=[]):
        """Save function arguments into class attributes.

        Defined in :numref:`sec_utils`"""
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k:v for k, v in local_vars.items()
                        if k not in set(ignore+['self']) and not k.startswith('_')}
        for k, v in self.hparams.items():
            setattr(self, k, v)


def use_svg_display():
    """Use the svg format to display a plot in Jupyter.

    Defined in :numref:`sec_calculus`"""
    backend_inline.set_matplotlib_formats('svg')


class ProgressBoard(HyperParameters):

    """ Board to display the line chart."""

    def __init__(self, xlabel=None,ylabel = None, xlim=None, ylim=None,
                  xscale = 'linear', yscale = 'linear',
                  ls=['-', '--', '-.', ':'], colors=['C0', 'C1', 'C2', 'C3'],
                  fig=None, axes=None, figsize=(3.5, 2.5), display=True):

         self.save_hyperparameters()

    def draw(self, x,y,label, every_n = 1):

        Point = collections.namedtuple('Point',['x','y'])

        if not hasattr(self, 'raw_points'):
            self.raw_points = collections.OrderedDict()
            self.data = collections.OrderedDict()

        if label not in self.raw_points:
            self.raw_points[label] = []
            self.data[label] = []

        points = self.raw_points[label]
        line = self.data[label]

        points.append(Point(x, y))
        if len(points) != every_n:
            return

        mean = lambda x: sum(x) / len(x)

        line.append( Point(mean([p.x for p in points]),
                        mean([p.y for p in points]))
                )

        points.clear()
        if not self.display:
            return

        use_svg_display()

        if self.fig is None:
            self.fig = plt.figure(figsize=self.figsize)

        plt_lines, labels = [], []

        for (k, v), ls, color in zip(self.data.items(), self.ls, self.colors):
                plt_lines.append(plt.plot([p.x for p in v], [p.y for p in v],
                                            linestyle=ls, color=color)[0])
                labels.append(k)

        axes = self.axes if self.axes else plt.gca()

        if self.xlim:
            axes.set_xlim(self.xlim)

        if self.ylim:
            axes.set_ylim(self.ylim)

        if not self.xlabel:
            self.xlabel = self.x

        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)
        axes.set_xscale(self.xscale)
        axes.set_yscale(self.yscale)
        axes.legend(plt_lines, labels)

        display.display(self.fig)
        display.clear_output(wait=True)


