#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from kaiwu_agent.utils.common_func import create_cls, Frame, attached
import numpy as np
import collections
from agent_ppo.conf.conf import Config


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


# Create the sample for the current frame
# 创建当前帧的样本
def build_frame(agent, state_dict):
    obs_data, act_data = agent.obs_data, agent.act_data
    is_train = False
    frame_state = state_dict["frame_state"]
    hero_list = frame_state["hero_states"]
    frame_no = frame_state["frameNo"]
    for hero in hero_list:
        hero_camp = hero["actor_state"]["camp"]
        hero_hp = hero["actor_state"]["hp"]
        if hero_camp == agent.hero_camp:
            is_train = True if hero_hp > 0 else False

    if obs_data.feature is not None:
        feature_vec = np.array(obs_data.feature)
    else:
        feature_vec = np.array(state_dict["observation"])

    reward = state_dict["reward"]["reward_sum"]

    sub_action_mask = state_dict["sub_action_mask"]

    prob, value, action = act_data.prob, act_data.value, act_data.action
    lstm_cell, lstm_hidden = act_data.lstm_cell, act_data.lstm_hidden

    legal_action = _update_legal_action(state_dict["legal_action"], action)
    frame = Frame(
        frame_no=frame_no,
        feature=feature_vec.reshape([-1]),
        legal_action=legal_action.reshape([-1]),
        action=action,
        reward=reward,
        reward_sum=0,
        value=value.flatten()[0],
        next_value=0,
        advantage=0,
        prob=prob,
        sub_action=sub_action_mask[action[0]],
        lstm_info=np.concatenate([lstm_cell.flatten(), lstm_hidden.flatten()]).reshape([-1]),
        is_train=False if action[0] < 0 else is_train,
    )
    return frame


# Construct legal_action based on the actual action taken
# 根据实际采用的action，构建legal_action
def _update_legal_action(original_la, action):
    target_size = Config.LABEL_SIZE_LIST[-1]
    top_size = Config.LABEL_SIZE_LIST[0]
    original_la = np.array(original_la)
    fix_part = original_la[: -target_size * top_size]
    target_la = original_la[-target_size * top_size :]
    target_la = target_la.reshape([top_size, target_size])[action[0]]
    return np.concatenate([fix_part, target_la], axis=0)


class FrameCollector:
    def __init__(self, num_agents):
        self._data_shapes = Config.data_shapes
        self._LSTM_FRAME = Config.LSTM_TIME_STEPS

        self.num_agents = num_agents
        self.rl_data_map = [collections.OrderedDict() for _ in range(num_agents)]
        self.m_replay_buffer = [[] for _ in range(num_agents)]

        self.gamma = Config.GAMMA
        self.lamda = Config.LAMDA

    def reset(self, num_agents):
        self.num_agents = num_agents
        self.rl_data_map = [collections.OrderedDict() for _ in range(self.num_agents)]
        self.m_replay_buffer = [[] for _ in range(self.num_agents)]

    def save_frame(self, rl_data_info, agent_id):
        # samples must saved by frame_no order
        # 样本必须按帧号顺序保存
        reward = self._clip_reward(rl_data_info.reward)

        if len(self.rl_data_map[agent_id]) > 0:
            last_key = list(self.rl_data_map[agent_id].keys())[-1]
            last_rl_data_info = self.rl_data_map[agent_id][last_key]
            last_rl_data_info.next_value = rl_data_info.value
            last_rl_data_info.reward = reward

        rl_data_info.reward = 0
        self.rl_data_map[agent_id][rl_data_info.frame_no] = rl_data_info

    def save_last_frame(self, reward, agent_id):
        if len(self.rl_data_map[agent_id]) > 0:
            last_key = list(self.rl_data_map[agent_id].keys())[-1]
            last_rl_data_info = self.rl_data_map[agent_id][last_key]
            last_rl_data_info.next_value = 0
            last_rl_data_info.reward = reward

    def sample_process(self):
        self._calc_reward()
        self._format_data()
        return self.m_replay_buffer

    def _calc_reward(self):
        """
        Calculate cumulated reward and advantage with GAE.
        reward_sum: used for value loss
        advantage: used for policy loss
        V(s) here is a approximation of target network
        """
        """
        计算累积奖励和优势函数（GAE）。
        reward_sum: 用于值损失
        advantage: 用于策略损失
        V(s) 这里是目标网络的近似值
        """
        for i in range(self.num_agents):
            reversed_keys = list(self.rl_data_map[i].keys())
            reversed_keys.reverse()
            gae, last_gae = 0.0, 0.0
            for j in reversed_keys:
                rl_info = self.rl_data_map[i][j]
                delta = -rl_info.value + rl_info.reward + self.gamma * rl_info.next_value
                gae = gae * self.gamma * self.lamda + delta
                rl_info.advantage = gae
                rl_info.reward_sum = gae + rl_info.value

    # For every LSTM_TIME_STEPS samples, concatenate 1 LSTM state
    # 每LSTM_TIME_STEPS个样本，需要拼接1个lstm状态
    def _reshape_lstm_batch_sample(self, sample_batch, sample_lstm):
        sample = np.zeros([np.prod(sample_batch.shape) + np.prod(sample_lstm.shape)])
        idx, s_idx = 0, 0

        sample[-sample_lstm.shape[0] :] = sample_lstm
        for split_shape in self._data_shapes[:-2]:
            one_shape = split_shape[0] // self._LSTM_FRAME
            sample[s_idx : s_idx + split_shape[0]] = sample_batch[:, idx : idx + one_shape].reshape([-1])
            idx += one_shape
            s_idx += split_shape[0]
        return SampleData(npdata=sample.astype(np.float32))

    # Create the sample for the current frame
    # 根据LSTM_TIME_STEPS，组合送入样本池的样本
    def _format_data(self):
        sample_one_size = np.sum(self._data_shapes[:-2]) // self._LSTM_FRAME
        sample_lstm_size = np.sum(self._data_shapes[-2:])
        sample_batch = np.zeros([self._LSTM_FRAME, sample_one_size])
        first_frame_no = -1

        for i in range(self.num_agents):
            sample_lstm = np.zeros([sample_lstm_size])
            cnt = 0
            for j in self.rl_data_map[i]:
                rl_info = self.rl_data_map[i][j]
                if cnt == 0:
                    first_frame_no = rl_info.frame_no

                idx, dlen = 0, 0

                dlen = rl_info.feature.shape[0]
                sample_batch[cnt, idx : idx + dlen] = rl_info.feature
                idx += dlen

                dlen = rl_info.legal_action.shape[0]
                sample_batch[cnt, idx : idx + dlen] = rl_info.legal_action
                idx += dlen

                sample_batch[cnt, idx] = rl_info.reward_sum
                idx += 1
                sample_batch[cnt, idx] = rl_info.advantage
                idx += 1

                dlen = 6
                sample_batch[cnt, idx : idx + dlen] = rl_info.action
                idx += dlen

                for p in rl_info.prob:
                    dlen = len(p)
                    sample_batch[cnt, idx : idx + dlen] = p
                    idx += dlen

                dlen = 6
                sample_batch[cnt, idx : idx + dlen] = rl_info.sub_action
                idx += dlen

                sample_batch[cnt, idx] = rl_info.is_train
                idx += 1

                assert idx == sample_one_size, "Sample check failed, {}/{}".format(idx, sample_one_size)

                cnt += 1
                if cnt == self._LSTM_FRAME:
                    cnt = 0
                    sample = self._reshape_lstm_batch_sample(sample_batch, sample_lstm)
                    self.m_replay_buffer[i].append(sample)
                    sample_lstm = rl_info.lstm_info

    def _clip_reward(self, reward, max=100, min=-100):
        if reward > max:
            reward = max
        elif reward < min:
            reward = min
        return reward

    def __len__(self):
        return max([len(agent_samples) for agent_samples in self.rl_data_map])


@attached
def SampleData2NumpyData(g_data):
    return g_data.npdata


@attached
def NumpyData2SampleData(s_data):
    return SampleData(npdata=s_data)
