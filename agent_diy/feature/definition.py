#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from kaiwu_agent.utils.common_func import create_cls, Frame, attached
from agent_diy.conf.conf import Config
import numpy as np
import collections
import random
import itertools
import os
import json

SampleData = create_cls("SampleData", npdata=None)

ObsData = create_cls("ObsData", feature=None, legal_action=None, lstm_cell=None, lstm_hidden=None)

ActData = create_cls(
    "ActData",
    action=None,
    d_action=None,
    prob=None,
    value=None,
    lstm_cell=None,
    lstm_hidden=None,
)

NONE_ACTION = [0, 15, 15, 15, 15, 0]


@attached
def sample_process(collector):
    return collector.sample_process()


class FrameCollector:
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.rl_data_map = [collections.OrderedDict() for _ in range(num_agents)]
        self.m_replay_buffer = [[] for _ in range(num_agents)]

    def reset(self, num_agents):
        self.num_agents = num_agents
        self.rl_data_map = [collections.OrderedDict() for _ in range(self.num_agents)]
        self.m_replay_buffer = [[] for _ in range(self.num_agents)]

    def sample_process(self):
        return


@attached
def SampleData2NumpyData(g_data):
    return g_data.npdata


@attached
def NumpyData2SampleData(s_data):
    return SampleData(npdata=s_data)
