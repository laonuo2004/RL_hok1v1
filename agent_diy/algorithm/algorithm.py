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
import os
from agent_diy.conf.conf import Config


class Algorithm:
    def __init__(self, model, device=None, logger=None, monitor=None):
        self.device = device
        self.model = model
        self.train_step = 0

        self.logger = logger
        self.monitor = monitor

    def learn(self, list_sample_data):
        # Code to implement model train
        # 实现模型训练的代码
        pass
