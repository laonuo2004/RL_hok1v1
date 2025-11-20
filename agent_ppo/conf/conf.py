#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

################################################################
# 游戏配置 - 可以修改，这里配置奖励权重和模型保存间隔
################################################################
#NOTE: 可以修改奖励权重，调整智能体的行为倾向
class GameConfig:
    # Set the weight of each reward item and use it in reward_manager
    # 设置各个回报项的权重，在reward_manager中使用
    REWARD_WEIGHT_DICT = {
        
        "hp_point": 2.0,        # 敌我生命值比值？
        "tower_hp_point": 5.0,  # 敌我防御塔塔生命值比值？
        "money": 0.006,         # 金钱
        "exp": 0.006,           # 经验
        # ep_rate奖励权重-------这个似乎是不释放技能的关键，消耗蓝量会被惩罚？
        # 这很匪夷所思，受到cd的限制，蓝量很难被消耗完。
        # 我们需要考虑的点是如何让鲁班对英雄释放更多的技能，而不是匆匆忙忙的把技能释放给小兵，甚至空放技能
        # 为此可能要新增释放技能未命中目标的惩罚，或者把技能释放的位置或方向写死（直接获取target的位置），那每个技能都需要单独设计
        # 这也解释了为什么鲁班only会释放闪现、恢复技能。但是这并没有解释鲁班为什么不选择回城
        # 连招设计可以之后再考虑，先让鲁班学会较为合理的释放技能
        "ep_rate": 0.75,        # 能量值
        "death": -1.0,          # 死亡
        "kill": -0.6,           # 击杀
        "last_hit": 0.5,        # 补刀
        "forward": 0.01,        # 推进
        "frontline_follow": 8.0,  # 跟随己方最前线单位
    }
    # Time decay factor, used in reward_manager
    # 时间衰减因子，在reward_manager中使用
    TIME_SCALE_ARG = 0
    # Model save interval configuration, used in workflow
    # 模型保存间隔配置，在workflow中使用
    MODEL_SAVE_INTERVAL = 1800

################################################################
# 维度配置 - 不用改，特征维度配置（与环境返回的特征维度对应）
################################################################
#NOTE: 不用改，除非修改了特征处理方式，否则不要改这些维度
# Dimension configuration, used when building the model
# 维度配置，构建模型时使用
class DimConfig:
    DIM_OF_SOLDIER_1_10 = [18, 18, 18, 18]
    DIM_OF_SOLDIER_11_20 = [18, 18, 18, 18]
    DIM_OF_ORGAN_1_2 = [18, 18]
    DIM_OF_ORGAN_3_4 = [18, 18]
    DIM_OF_HERO_FRD = [235]
    DIM_OF_HERO_EMY = [235]
    DIM_OF_HERO_MAIN = [14]
    DIM_OF_GLOBAL_INFO = [25]


################################################################
# 算法配置 - 可以修改，这里配置学习率、PPO参数、网络结构等
################################################################
#NOTE: 可以修改这些超参数来调优算法（学习率、clip范围、LSTM层数等）
# Configuration related to model and algorithms used
# 模型和算法使用的相关配置
class Config:
    NETWORK_NAME = "network"
    LSTM_TIME_STEPS = 16
    LSTM_UNIT_SIZE = 512
    DATA_SPLIT_SHAPE = [
        810,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        12,
        16,
        16,
        16,
        16,
        9,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        LSTM_UNIT_SIZE,
        LSTM_UNIT_SIZE,
    ]
    SERI_VEC_SPLIT_SHAPE = [(725,), (85,)]
    INIT_LEARNING_RATE_START = 1e-3
    TARGET_LR = 1e-4
    TARGET_STEP = 5000
    BETA_START = 0.025
    LOG_EPSILON = 1e-6
    LABEL_SIZE_LIST = [12, 16, 16, 16, 16, 9]
    IS_REINFORCE_TASK_LIST = [
        True,
        True,
        True,
        True,
        True,
        True,
    ]

    CLIP_PARAM = 0.2

    MIN_POLICY = 0.00001

    TARGET_EMBED_DIM = 32

    data_shapes = [
        [(725 + 85) * 16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [192],
        [256],
        [256],
        [256],
        [256],
        [144],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [512],
        [512],
    ]

    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()
    LEGAL_ACTION_SIZE_LIST[-1] = LEGAL_ACTION_SIZE_LIST[-1] * LEGAL_ACTION_SIZE_LIST[0]

    GAMMA = 0.995
    LAMDA = 0.95

    USE_GRAD_CLIP = True
    GRAD_CLIP_RANGE = 0.5

    # The input dimension of samples on the learner from Reverb varies depending on the algorithm used.
    # For instance, the dimension for ppo is 15584,
    # learner上reverb样本的输入维度, 注意不同的算法维度不一样, 比如示例代码中ppo的维度是15584
    # **注意**，此项必须正确配置，应该与definition.py中的NumpyData2SampleData函数数据对齐，否则可能报样本维度错误
    SAMPLE_DIM = sum(DATA_SPLIT_SHAPE[:-2]) * LSTM_TIME_STEPS + sum(DATA_SPLIT_SHAPE[-2:])
    
    # 可配置的技能按钮集合（默认包含技能1/2/3的按钮索引）
    # 修改此列表以包含你认为应被视为“技能释放”的 button 索引
    SKILL_BUTTON_IDS = [4, 5, 6]

    # 注意：不再提供全局固定技能方向配置，建议使用基于 target 的动态规则或奖励引导
