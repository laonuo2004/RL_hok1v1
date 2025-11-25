#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from kaiwu_agent.utils.common_func import Frame, attached
import random

from agent_diy.feature.definition import (
    sample_process,
    FrameCollector,
    NONE_ACTION,
)
from agent_diy.conf.conf import GameConfig
from tools.model_pool_utils import get_valid_model_pool
from kaiwudrl.common.checkpoint.model_file_sync import ModelFileSync
from tools.train_env_conf_validate import read_usr_conf, check_usr_conf
from tools.metrics_utils import get_training_metrics


@attached
def workflow(envs, agents, logger=None, monitor=None):
    # hok1v1 environment
    # hok1v1环境
    env = envs[0]
    # Number of agents, in hok1v1 the value is 2
    # 智能体数量，在hok1v1中值为2
    agent_num = len(agents)
    # Frame Collector
    # 帧收集器
    frame_collector = FrameCollector(agent_num)

    # Directly pull and load the model file from the model pool.
    # 直接从modelpool里拉取model文件加载
    model_file_sync_wrapper = ModelFileSync()
    model_file_sync_wrapper.make_local_model_dirs(logger)

    # Read and validate configuration file
    # 配置文件读取和校验
    usr_conf = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error(f"usr_conf is None, please check agent_diy/conf/train_env_conf.toml")
        return
    # check_usr_conf is a tool to check whether the environment configuration is correct
    # It is recommended to perform a check before calling reset.env
    # check_usr_conf会检查环境配置是否正确，建议调用reset.env前先检查一下
    valid = check_usr_conf(usr_conf, logger)
    if not valid:
        logger.error(f"check_usr_conf return False, please check agent_diy/conf/train_env_conf.toml")
        return

    # Make eval matches as evenly distributed as possible
    # 引入随机因子，让eval对局尽可能平均分布
    random_eval_start = random.randint(0, usr_conf["env_conf"]["episode"]["eval_interval"])

    # Please implement your DIY algorithm flow
    # 请实现你DIY的算法流程
    # ......

    # Single environment process (30 frame/s)
    # 单局流程 (30 frame/s)
    """
    while True:
        pass
    """

    return
