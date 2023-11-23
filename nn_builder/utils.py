import inspect
import collections
from IPython import display
from matplotlib_inline import backend_inline
import matplotlib.pyplot as plt
import torch
import os
import requests
import hashlib

import zipfile
import tarfile


DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'


def cpu():  #@save
    """Get the CPU device."""
    return torch.device('cpu')

def gpu(i=0):  #@save
    """Get a GPU device."""
    return torch.device(f'cuda:{i}')

def get_num_gpus():  #@save
    """Get the number of available GPUs."""
    return torch.cuda.device_count()

def try_gpu(i=0):  #@save
    """Return gpu(i) if exists, otherwise return cpu()."""
    if get_num_gpus() >= i + 1:
        return gpu(i)
    return cpu()

def try_all_gpus():  #@save
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    return [gpu(i) for i in range(get_num_gpus())]


def add_to_class(Class):
    """Register funtion as method in the original class"""

    def wrapper(obj):
        setattr(Class,obj.__name__, obj)

    return wrapper


class HyperParameters:
    """The base class of hyperparameters."""
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


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images.

    Defined in :numref:`sec_utils`"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        try:
            img = img.to('cpu').detach().numpy()
        except:
            pass
        ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

def download(url, folder='../data', sha1_hash=None):
    """Download a file to folder and return the local filepath.

    Defined in :numref:`sec_utils`"""
    # if not url.startswith('http'):
    #     # For back compatability
    #     url, sha1_hash = DATA_HUB[url]

    os.makedirs(folder, exist_ok=True)
    fname = os.path.join(folder, url.split('/')[-1])
    # Check if hit cache
    if os.path.exists(fname) and sha1_hash:
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname
    # Download
    print(f'Downloading {fname} from {url}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

def extract(filename, folder=None):
    """Extract a zip/tar file into folder.

    Defined in :numref:`sec_utils`"""
    base_dir = os.path.dirname(filename)
    _, ext = os.path.splitext(filename)
    assert ext in ('.zip', '.tar', '.gz'), 'Only support zip/tar files.'
    if ext == '.zip':
        fp = zipfile.ZipFile(filename, 'r')
    else:
        fp = tarfile.open(filename, 'r')
    if folder is None:
        folder = base_dir
    fp.extractall(folder)


