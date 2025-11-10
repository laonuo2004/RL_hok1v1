#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from agent_diy.conf.conf import Config


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # User-defined network
        # 用户自定义网络
