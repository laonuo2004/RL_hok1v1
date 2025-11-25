#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch
import torch.nn as nn
from torch.nn import ModuleDict
import torch.nn.functional as F

import numpy as np
from math import ceil, floor
from collections import OrderedDict
from typing import Dict, List, Tuple

from agent_ppo.conf.conf import DimConfig, Config


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # feature configure parameter
        # 特征配置参数
        self.model_name = Config.NETWORK_NAME
        self.data_split_shape = Config.DATA_SPLIT_SHAPE
        self.lstm_time_steps = Config.LSTM_TIME_STEPS
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE
        self.m_learning_rate = Config.INIT_LEARNING_RATE_START
        self.m_var_beta = Config.BETA_START
        self.log_epsilon = Config.LOG_EPSILON
        self.label_size_list = Config.LABEL_SIZE_LIST
        self.is_reinforce_task_list = Config.IS_REINFORCE_TASK_LIST
        self.min_policy = Config.MIN_POLICY
        self.clip_param = Config.CLIP_PARAM
        self.restore_list = []
        self.var_beta = self.m_var_beta
        self.learning_rate = self.m_learning_rate
        self.target_embed_dim = Config.TARGET_EMBED_DIM
        self.cut_points = [value[0] for value in Config.data_shapes]
        self.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST

        self.feature_dim = Config.SERI_VEC_SPLIT_SHAPE[0][0]
        self.legal_action_dim = np.sum(Config.LEGAL_ACTION_SIZE_LIST)
        self.lstm_hidden_dim = Config.LSTM_UNIT_SIZE

        # NETWORK DIM
        # 网络维度
        self.hero_data_len = sum(Config.data_shapes[0])
        self.single_hero_feature_dim = int(DimConfig.DIM_OF_HERO_EMY[0])
        self.single_soldier_feature_dim = int(DimConfig.DIM_OF_SOLDIER_1_10[0])
        self.single_organ_feature_dim = int(DimConfig.DIM_OF_ORGAN_1_2[0])
        self.hero_main_feature_dim = int(DimConfig.DIM_OF_HERO_MAIN[0])
        self.global_feature_dim = int(np.sum(DimConfig.DIM_OF_GLOBAL_INFO))

        self.all_hero_feature_dim = (
            int(np.sum(DimConfig.DIM_OF_HERO_FRD))
            + int(np.sum(DimConfig.DIM_OF_HERO_EMY))
            + int(np.sum(DimConfig.DIM_OF_HERO_MAIN))
        )
        self.all_soldier_feature_dim = int(np.sum(DimConfig.DIM_OF_SOLDIER_1_10)) + int(
            np.sum(DimConfig.DIM_OF_SOLDIER_11_20)
        )
        self.all_organ_feature_dim = int(np.sum(DimConfig.DIM_OF_ORGAN_1_2)) + int(np.sum(DimConfig.DIM_OF_ORGAN_3_4))

        # build network
        # 构建网络
        fc_hero_main_dim_list = [self.hero_main_feature_dim, 64, 32, 16]
        self.hero_main_mlp = MLP(fc_hero_main_dim_list, "hero_main_mlp")

        fc_hero_dim_list = [self.single_hero_feature_dim, 512, 256, 96 + self.target_embed_dim]
        self.hero_mlp = MLP(fc_hero_dim_list[:-1], "hero_mlp", non_linearity_last=True)
        self.hero_frd_fc = nn.Sequential(
            OrderedDict(
                [
                    (
                        "hero_frd_fc",
                        make_fc_layer(fc_hero_dim_list[-2], fc_hero_dim_list[-1]),
                    )
                ]
            )
        )
        self.hero_emy_fc = nn.Sequential(
            OrderedDict(
                [
                    (
                        "hero_emy_fc",
                        make_fc_layer(fc_hero_dim_list[-2], fc_hero_dim_list[-1]),
                    )
                ]
            )
        )

        # The first two fully connected layers are shared by soldier vectors from different camps
        # 前两个全连接层由不同阵营小兵向量共享
        fc_soldier_dim_list = [self.single_soldier_feature_dim, 64, 64, self.target_embed_dim]
        self.soldier_mlp = MLP(fc_soldier_dim_list[:-1], "soldier_mlp", non_linearity_last=True)

        self.soldier_frd_fc = nn.Sequential(
            OrderedDict(
                [
                    (
                        "soldier_frd_fc",
                        make_fc_layer(fc_soldier_dim_list[-2], fc_soldier_dim_list[-1]),
                    )
                ]
            )
        )
        self.soldier_emy_fc = nn.Sequential(
            OrderedDict(
                [
                    (
                        "soldier_emy_fc",
                        make_fc_layer(fc_soldier_dim_list[-2], fc_soldier_dim_list[-1]),
                    )
                ]
            )
        )

        fc_organ_dim_list = [self.single_organ_feature_dim, 64, 64, self.target_embed_dim]
        self.organ_mlp = MLP(fc_organ_dim_list[:-1], "organ_mlp", non_linearity_last=True)
        self.organ_frd_fc = nn.Sequential(
            OrderedDict(
                [
                    (
                        "organ_frd_fc",
                        make_fc_layer(fc_organ_dim_list[-2], fc_organ_dim_list[-1]),
                    )
                ]
            )
        )
        self.organ_emy_fc = nn.Sequential(
            OrderedDict(
                [
                    (
                        "organ_emy_fc",
                        make_fc_layer(fc_organ_dim_list[-2], fc_organ_dim_list[-1]),
                    )
                ]
            )
        )

        concat_dim = (
            fc_hero_main_dim_list[-1]
            + 2 * fc_hero_dim_list[-1]
            + 2 * fc_soldier_dim_list[-1]
            + 2 * fc_organ_dim_list[-1]
            + self.global_feature_dim
        )
        fc_concat_dim_list = [concat_dim, 512]
        self.concat_mlp = MLP(fc_concat_dim_list, "concat_mlp", non_linearity_last=True)

        self.lstm = torch.nn.LSTM(
            input_size=self.lstm_unit_size,
            hidden_size=self.lstm_unit_size,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=False,
        )

        self.label_mlp = ModuleDict(
            {
                "hero_label{0}_mlp".format(label_index): MLP(
                    [self.lstm_unit_size, self.label_size_list[label_index]],
                    "hero_label{0}_mlp".format(label_index),
                )
                for label_index in range(len(self.label_size_list) - 1)
            }
        )
        self.lstm_tar_embed_mlp = make_fc_layer(self.lstm_unit_size, self.target_embed_dim)

        self.value_mlp = MLP([self.lstm_unit_size, 64, 1], "hero_value_mlp")

        self.target_embed_mlp = make_fc_layer(self.target_embed_dim, self.target_embed_dim, use_bias=False)

    def forward(self, data_list, inference=False):
        feature_vec, lstm_hidden_init, lstm_cell_init = data_list

        result_list = []

        # feature_vec_split
        # 特征向量分割
        feature_vec_split_list = feature_vec.split(
            [
                self.all_hero_feature_dim,
                self.all_soldier_feature_dim,
                self.all_organ_feature_dim,
                self.global_feature_dim,
            ],
            dim=1,
        )
        hero_vec_list = feature_vec_split_list[0].split(
            [
                int(np.sum(DimConfig.DIM_OF_HERO_FRD)),
                int(np.sum(DimConfig.DIM_OF_HERO_EMY)),
                int(np.sum(DimConfig.DIM_OF_HERO_MAIN)),
            ],
            dim=1,
        )
        soldier_vec_list = feature_vec_split_list[1].split(
            [
                int(np.sum(DimConfig.DIM_OF_SOLDIER_1_10)),
                int(np.sum(DimConfig.DIM_OF_SOLDIER_11_20)),
            ],
            dim=1,
        )
        organ_vec_list = feature_vec_split_list[2].split(
            [
                int(np.sum(DimConfig.DIM_OF_ORGAN_1_2)),
                int(np.sum(DimConfig.DIM_OF_ORGAN_3_4)),
            ],
            dim=1,
        )
        global_info_list = feature_vec_split_list[3]

        _soldier_1_10 = soldier_vec_list[0].split(DimConfig.DIM_OF_SOLDIER_1_10, dim=1)
        _soldier_11_20 = soldier_vec_list[1].split(DimConfig.DIM_OF_SOLDIER_11_20, dim=1)

        _organ_1_2 = organ_vec_list[0].split(DimConfig.DIM_OF_ORGAN_1_2, dim=1)
        _organ_3_4 = organ_vec_list[1].split(DimConfig.DIM_OF_ORGAN_3_4, dim=1)
        _hero_frd = hero_vec_list[0].split(DimConfig.DIM_OF_HERO_FRD, dim=1)
        _hero_emy = hero_vec_list[1].split(DimConfig.DIM_OF_HERO_EMY, dim=1)
        _hero_main = hero_vec_list[2].split(DimConfig.DIM_OF_HERO_MAIN, dim=1)
        _global_info = global_info_list

        tar_embed_list = []

        # hero_main
        # 英雄主向量
        for index in range(len(_hero_main)):
            main_hero = self.hero_main_mlp(_hero_main[index])
        hero_main_result = main_hero

        hero_emy_result_list = []
        for index in range(len(_hero_emy)):
            hero_emy_mlp_out = self.hero_mlp(_hero_emy[index])
            hero_emy_fc_out = self.hero_emy_fc(hero_emy_mlp_out)
            _, split_1 = hero_emy_fc_out.split([96, self.target_embed_dim], dim=1)
            tar_embed_list.append(split_1)
            hero_emy_result_list.append(hero_emy_fc_out)

        hero_emy_concat_result = torch.cat(hero_emy_result_list, dim=1)
        reshape_hero_emy = hero_emy_concat_result.reshape(-1, 1, 1, 96 + self.target_embed_dim)
        pool_hero_emy, _ = reshape_hero_emy.max(dim=2)
        output_dim = int(np.prod(pool_hero_emy.shape[1:]))
        reshape_pool_hero_emy = pool_hero_emy.reshape(-1, output_dim)

        # hero_share
        # 英雄共享向量
        hero_frd_result_list = []
        for index in range(len(_hero_frd)):
            hero_frd_mlp_out = self.hero_mlp(_hero_frd[index])
            hero_frd_fc_out = self.hero_frd_fc(hero_frd_mlp_out)
            _, split_1 = hero_frd_fc_out.split([96, self.target_embed_dim], dim=1)
            tar_embed_list.append(split_1)
            hero_frd_result_list.append(hero_frd_fc_out)

        hero_frd_concat_result = torch.cat(hero_frd_result_list, dim=1)
        reshape_hero_frd = hero_frd_concat_result.reshape(-1, 1, 1, 96 + self.target_embed_dim)
        pool_hero_frd, _ = reshape_hero_frd.max(dim=2)
        output_dim = int(np.prod(pool_hero_frd.shape[1:]))
        reshape_pool_hero_frd = pool_hero_frd.reshape(-1, output_dim)

        # soldier_share
        # 士兵共享向量
        soldier_frd_result_list = []
        for index in range(len(_soldier_1_10)):
            soldier_frd_mlp_out = self.soldier_mlp(_soldier_1_10[index])
            soldier_frd_fc_out = self.soldier_frd_fc(soldier_frd_mlp_out)
            soldier_frd_result_list.append(soldier_frd_fc_out)

        soldier_frd_concat_result = torch.cat(soldier_frd_result_list, dim=1)
        reshape_frd_soldier = soldier_frd_concat_result.reshape(-1, 1, 4, self.target_embed_dim)
        pool_frd_soldier, _ = reshape_frd_soldier.max(dim=2)
        output_dim = int(np.prod(pool_frd_soldier.shape[1:]))
        reshape_pool_frd_soldier = pool_frd_soldier.reshape(-1, output_dim)

        soldier_emy_result_list = []
        for index in range(len(_soldier_11_20)):
            soldier_emy_mlp_out = self.soldier_mlp(_soldier_11_20[index])
            soldier_emy_fc_out = self.soldier_emy_fc(soldier_emy_mlp_out)
            soldier_emy_result_list.append(soldier_emy_fc_out)
            tar_embed_list.append(soldier_emy_fc_out)

        soldier_emy_concat_result = torch.cat(soldier_emy_result_list, dim=1)
        reshape_emy_soldier = soldier_emy_concat_result.reshape(-1, 1, 4, self.target_embed_dim)
        pool_emy_soldier, _ = reshape_emy_soldier.max(dim=2)
        output_dim = int(np.prod(pool_emy_soldier.shape[1:]))
        reshape_pool_emy_soldier = pool_emy_soldier.reshape(-1, output_dim)

        organ_frd_result_list = []
        for index in range(len(_organ_1_2)):
            organ_frd_mlp_out = self.organ_mlp(_organ_1_2[index])
            organ_frd_fc_out = self.organ_frd_fc(organ_frd_mlp_out)
            organ_frd_result_list.append(organ_frd_fc_out)

        organ_1_concat_result = torch.cat(organ_frd_result_list, dim=1)
        reshape_frd_organ = organ_1_concat_result.reshape(-1, 1, 2, self.target_embed_dim)
        pool_frd_organ, _ = reshape_frd_organ.max(dim=2)
        output_dim = int(np.prod(pool_frd_organ.shape[1:]))
        reshape_pool_frd_organ = pool_frd_organ.reshape(-1, output_dim)

        organ_emy_result_list = []
        for index in range(len(_organ_3_4)):
            organ_emy_mlp_out = self.organ_mlp(_organ_3_4[index])
            organ_emy_fc_out = self.organ_emy_fc(organ_emy_mlp_out)
            organ_emy_result_list.append(organ_emy_fc_out)

        organ_emy_concat_result = torch.cat(organ_emy_result_list, dim=1)
        reshape_emy_organ = organ_emy_concat_result.reshape(-1, 1, 2, self.target_embed_dim)
        pool_emy_organ, _ = reshape_emy_organ.max(dim=2)
        output_dim = int(np.prod(pool_emy_organ.shape[1:]))
        reshape_pool_emy_organ = pool_emy_organ.reshape(-1, output_dim)
        tar_embed_list.append(reshape_pool_emy_organ)

        tar_embed_0 = 0.1 * torch.ones_like(tar_embed_list[-1]).to(feature_vec.device)
        tar_embed_list.insert(0, tar_embed_0)

        tar_embed_8 = 0.1 * torch.ones_like(tar_embed_list[-1]).to(feature_vec.device)
        tar_embed_list.append(tar_embed_8)

        concat_result = torch.cat(
            [
                reshape_pool_frd_soldier,
                reshape_pool_emy_soldier,
                reshape_pool_frd_organ,
                reshape_pool_emy_organ,
                hero_main_result,
                reshape_pool_hero_frd,
                reshape_pool_hero_emy,
                _global_info,
            ],
            dim=1,
        )

        # public concat
        # 公共连接层
        fc_public_result = self.concat_mlp(concat_result)

        reshape_fc_public_result = fc_public_result.reshape(-1, self.lstm_time_steps, 512)

        # public lstm: first hidden-state h0，second cell-state c0
        # 公共LSTM：第一个隐藏状态 h0，第二个细胞状态 c0
        lstm_initial_state_in = [
            lstm_hidden_init.unsqueeze(0),
            lstm_cell_init.unsqueeze(0),
        ]
        lstm_outputs, state = self.lstm(reshape_fc_public_result, lstm_initial_state_in)

        lstm_outputs = torch.cat([lstm_outputs[:, idx, :] for idx in range(lstm_outputs.size(1))], dim=1)

        self.lstm_cell_output = state[1]
        self.lstm_hidden_output = state[0]

        reshape_lstm_outputs_result = lstm_outputs.reshape(-1, self.lstm_unit_size)

        # output label
        # 输出标签
        for label_index, label_dim in enumerate(self.label_size_list[:-1]):
            label_mlp_out = self.label_mlp["hero_label{0}_mlp".format(label_index)](reshape_lstm_outputs_result)
            result_list.append(label_mlp_out)

        lstm_tar_embed_result = self.lstm_tar_embed_mlp(reshape_lstm_outputs_result)

        tar_embedding = torch.stack(tar_embed_list, dim=1)

        ulti_tar_embedding = self.target_embed_mlp(tar_embedding)
        reshape_label_result = lstm_tar_embed_result.reshape(-1, self.target_embed_dim, 1)

        label_result = torch.matmul(ulti_tar_embedding, reshape_label_result)
        target_output_dim = int(np.prod(label_result.shape[1:]))

        reshape_label_result = label_result.reshape(-1, target_output_dim)
        result_list.append(reshape_label_result)

        # output value
        # 输出价值
        value_result = self.value_mlp(reshape_lstm_outputs_result)
        result_list.append(value_result)

        # prepare for infer graph
        # 准备推理图
        logits = torch.flatten(torch.cat(result_list[:-1], 1), start_dim=1)
        value = result_list[-1]

        if inference:
            return [logits, value, self.lstm_cell_output, self.lstm_hidden_output]
        else:
            return result_list

    def compute_loss(self, data_list, rst_list):
        seri_vec = data_list[0].reshape(-1, self.data_split_shape[0])
        usq_reward = data_list[1].reshape(-1, self.data_split_shape[1])
        usq_advantage = data_list[2].reshape(-1, self.data_split_shape[2])
        usq_is_train = data_list[-3].reshape(-1, self.data_split_shape[-3])

        usq_label_list = data_list[3 : 3 + len(self.label_size_list)]
        for shape_index in range(len(self.label_size_list)):
            usq_label_list[shape_index] = (
                usq_label_list[shape_index].reshape(-1, self.data_split_shape[3 + shape_index]).long()
            )

        old_label_probability_list = data_list[3 + len(self.label_size_list) : 3 + 2 * len(self.label_size_list)]
        for shape_index in range(len(self.label_size_list)):
            old_label_probability_list[shape_index] = old_label_probability_list[shape_index].reshape(
                -1, self.data_split_shape[3 + len(self.label_size_list) + shape_index]
            )

        usq_weight_list = data_list[3 + 2 * len(self.label_size_list) : 3 + 3 * len(self.label_size_list)]
        for shape_index in range(len(self.label_size_list)):
            usq_weight_list[shape_index] = usq_weight_list[shape_index].reshape(
                -1,
                self.data_split_shape[3 + 2 * len(self.label_size_list) + shape_index],
            )

        # squeeze tensor
        # 压缩张量
        reward = usq_reward.squeeze(dim=1)
        advantage = usq_advantage.squeeze(dim=1)
        label_list = []
        for ele in usq_label_list:
            label_list.append(ele.squeeze(dim=1))
        weight_list = []
        for weight in usq_weight_list:
            weight_list.append(weight.squeeze(dim=1))
        frame_is_train = usq_is_train.squeeze(dim=1)

        label_result = rst_list[:-1]

        value_result = rst_list[-1]

        _, split_feature_legal_action = torch.split(
            seri_vec,
            [
                np.prod(self.seri_vec_split_shape[0]),
                np.prod(self.seri_vec_split_shape[1]),
            ],
            dim=1,
        )
        feature_legal_action_shape = list(self.seri_vec_split_shape[1])
        feature_legal_action_shape.insert(0, -1)
        feature_legal_action = split_feature_legal_action.reshape(feature_legal_action_shape)

        legal_action_flag_list = torch.split(feature_legal_action, self.label_size_list, dim=1)

        # loss of value net
        # 值网络的损失
        fc2_value_result_squeezed = value_result.squeeze(dim=1)
        self.value_cost = 0.5 * torch.mean(torch.square(reward - fc2_value_result_squeezed), dim=0)
        new_advantage = reward - fc2_value_result_squeezed
        self.value_cost = 0.5 * torch.mean(torch.square(new_advantage), dim=0)

        # for entropy loss calculate
        # 用于熵损失计算
        label_logits_subtract_max_list = []
        label_sum_exp_logits_list = []
        label_probability_list = []

        epsilon = 1e-5  # 0.00001

        # policy loss: ppo clip loss
        # 策略损失：PPO剪辑损失
        self.policy_cost = torch.tensor(0.0)
        for task_index in range(len(self.is_reinforce_task_list)):
            if self.is_reinforce_task_list[task_index]:
                final_log_p = torch.tensor(0.0)
                boundary = torch.pow(torch.tensor(10.0), torch.tensor(20.0))
                one_hot_actions = nn.functional.one_hot(label_list[task_index].long(), self.label_size_list[task_index])

                legal_action_flag_list_max_mask = (1 - legal_action_flag_list[task_index]) * boundary

                label_logits_subtract_max = torch.clamp(
                    label_result[task_index]
                    - torch.max(
                        label_result[task_index] - legal_action_flag_list_max_mask,
                        dim=1,
                        keepdim=True,
                    ).values,
                    -boundary,
                    1,
                )

                label_logits_subtract_max_list.append(label_logits_subtract_max)

                label_exp_logits = (
                    legal_action_flag_list[task_index] * torch.exp(label_logits_subtract_max) + self.min_policy
                )

                label_sum_exp_logits = label_exp_logits.sum(1, keepdim=True)
                label_sum_exp_logits_list.append(label_sum_exp_logits)

                label_probability = 1.0 * label_exp_logits / label_sum_exp_logits
                label_probability_list.append(label_probability)

                policy_p = (one_hot_actions * label_probability).sum(1)
                policy_log_p = torch.log(policy_p + epsilon)
                old_policy_p = (one_hot_actions * old_label_probability_list[task_index] + epsilon).sum(1)
                old_policy_log_p = torch.log(old_policy_p)
                final_log_p = final_log_p + policy_log_p - old_policy_log_p
                ratio = torch.exp(final_log_p)
                clip_ratio = ratio.clamp(0.0, 3.0)

                surr1 = clip_ratio * advantage
                surr2 = ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param) * advantage
                temp_policy_loss = -torch.sum(
                    torch.minimum(surr1, surr2) * (weight_list[task_index].float()) * 1
                ) / torch.maximum(torch.sum((weight_list[task_index].float()) * 1), torch.tensor(1.0))

                self.policy_cost = self.policy_cost + temp_policy_loss

        # cross entropy loss
        # 交叉熵损失
        current_entropy_loss_index = 0
        entropy_loss_list = []
        for task_index in range(len(self.is_reinforce_task_list)):
            if self.is_reinforce_task_list[task_index]:
                temp_entropy_loss = -torch.sum(
                    label_probability_list[current_entropy_loss_index]
                    * legal_action_flag_list[task_index]
                    * torch.log(label_probability_list[current_entropy_loss_index] + epsilon),
                    dim=1,
                )

                temp_entropy_loss = -torch.sum(
                    (temp_entropy_loss * weight_list[task_index].float() * 1)
                ) / torch.maximum(torch.sum(weight_list[task_index].float() * 1), torch.tensor(1.0))

                entropy_loss_list.append(temp_entropy_loss)
                current_entropy_loss_index = current_entropy_loss_index + 1
            else:
                temp_entropy_loss = torch.tensor(0.0)
                entropy_loss_list.append(temp_entropy_loss)

        self.entropy_cost = torch.tensor(0.0)
        for entropy_element in entropy_loss_list:
            self.entropy_cost = self.entropy_cost + entropy_element

        self.entropy_cost_list = entropy_loss_list

        self.loss = self.value_cost + self.policy_cost + self.var_beta * self.entropy_cost

        return self.loss, [
            self.loss,
            [self.value_cost, self.policy_cost, self.entropy_cost],
        ]

    def set_train_mode(self):
        self.lstm_time_steps = Config.LSTM_TIME_STEPS
        self.train()

    def set_eval_mode(self):
        self.lstm_time_steps = 1
        self.eval()


def make_fc_layer(in_features: int, out_features: int, use_bias=True):
    """Wrapper function to create and initialize a linear layer

    Args:
        in_features (int): ``in_features``
        out_features (int): ``out_features``

    Returns:
        nn.Linear: the initialized linear layer
    """
    """ 创建并初始化线性层的包装函数

    参数:
        in_features (int): 输入特征数
        out_features (int): 输出特征数

    返回:
        nn.Linear: 初始化的线性层
    """
    fc_layer = nn.Linear(in_features, out_features, bias=use_bias)

    nn.init.orthogonal(fc_layer.weight)
    if use_bias:
        nn.init.zeros_(fc_layer.bias)

    return fc_layer


class MLP(nn.Module):
    def __init__(
        self,
        fc_feat_dim_list: List[int],
        name: str,
        non_linearity: nn.Module = nn.ReLU,
        non_linearity_last: bool = False,
    ):
        """Create a MLP object

        Args:
            fc_feat_dim_list (List[int]): ``in_features`` of the first linear layer followed by
                ``out_features`` of each linear layer
            name (str): human-friendly name, serving as prefix of each comprising layers
            non_linearity (nn.Module, optional): the activation function to use. Defaults to nn.ReLU.
            non_linearity_last (bool, optional): whether to append a activation function in the end.
                Defaults to False.
        """
        """ 创建一个MLP对象

        参数:
            fc_feat_dim_list (List[int]): 第一个线性层的输入特征数，后续每个线性层的输出特征数
            name (str): 人类友好的名称，作为每个组成层的前缀
            non_linearity (nn.Module, optional): 要使用的激活函数。默认为 nn.ReLU。
            non_linearity_last (bool, optional): 是否在最后附加一个激活函数。默认为 False。
        """
        super(MLP, self).__init__()
        self.fc_layers = nn.Sequential()
        for i in range(len(fc_feat_dim_list) - 1):
            fc_layer = make_fc_layer(fc_feat_dim_list[i], fc_feat_dim_list[i + 1])
            self.fc_layers.add_module("{0}_fc{1}".format(name, i + 1), fc_layer)
            if i + 1 < len(fc_feat_dim_list) - 1 or non_linearity_last:
                self.fc_layers.add_module("{0}_non_linear{1}".format(name, i + 1), non_linearity())

    def forward(self, data):
        return self.fc_layers(data)
