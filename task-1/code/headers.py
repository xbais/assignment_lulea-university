##########################
#        Headers         #
##########################

# BASIC IMPORTS
#--------------
import os 
import time
import threading
from rich.console import Console
from rich.markdown import Markdown # For printing pretty markdown : to be used instead of the usual Python print fn via a custom print_md fn
console = Console()
#from rich.traceback import install
#install(show_locals=True) # Add rich as the default traceback handler
from rich.progress import track # For pretty progress bars
from rich.progress import Progress # For advanced pretty progress bars
import rich
# for now()
import datetime 
# for timezone()
import pytz
import atexit
import signal
import traceback
#import multiprocessing
import torch.multiprocessing as multiprocessing
#multiprocessing.set_start_method('spawn')
import os
import numpy as np
from tqdm import tqdm

# ML / DL IMPORTS
#----------------
import scipy
import torch, torchvision
# Device configuration
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device
from torch.autograd import Variable
from torch import optim
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.models import vit_b_32, vit_b_16
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data import Subset
import torch.nn as nn
import torch.nn.functional as F

#from functools import lru_cache
#from scalene import scalene_profiler

# Turn profiling on
#scalene_profiler.start()
# import glob
import sys
from profgraph import core as profiler_graph # local_module
from PIL import Image # pillow==latest
#import h5py
#h5py._errors.unsilence_errors()
import matplotlib.pyplot as plt
import math
import pickle
import random
import multiprocessing
import copy
import pandas as pd
import gc # garbage collector for python
from numba import njit
from collections import Counter
from torch.utils.data import DataLoader
import hashlib
import shutil
import marko


#from tqdm import trange
from sklearn.model_selection import train_test_split # scikit-learn==latest
from sklearn.metrics import confusion_matrix # scikit-learn==latest
import seaborn
import pandas as pd
# Get Confusion Matrix use : confusion_matrix(y_true, y_pred)
from datetime import date
import pickle 

import importlib
import json
import glob

############
from colorama import init
init()
from colorama import Fore, Back, Style

import pdb #; pdb.Pdb().set_trace()
import math
import argparse
import networkx as nx # Used for profiling
from typeguard import typechecked
from rich.progress import track
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn
from functools import wraps
from scipy import stats as st

import asyncio
import functools
import copy
import imageio
import cv2 # opencv-python==latest
