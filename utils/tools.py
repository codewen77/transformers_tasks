# -*- coding: utf-8 -*-
# @Time    : 2024/9/7 21:32
# @Author  : codewen77
import os
import random

import numpy as np
import torch


def seed_everything(seed=1024):
    """
    设置整个开发环境的seed
    :param seed:
    :return:
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
