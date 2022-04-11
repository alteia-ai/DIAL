from itertools import filterfalse

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


def CrossEntropy2d(input, target, weight=None, size_average=True):
    """ Cross entropy loss """
    return nn.CrossEntropyLoss(weight, size_average)(input, target.long())