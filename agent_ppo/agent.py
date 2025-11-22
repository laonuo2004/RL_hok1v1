#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch
import math

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import os
from agent_ppo.model.model import Model
from agent_ppo.feature.definition import *
import numpy as np
from kaiwu_agent.agent.base_agent import (
    BaseAgent,
    predict_wrapper,
    exploit_wrapper,
    learn_wrapper,
    save_model_wrapper,
    load_model_wrapper,
)

from agent_ppo.conf.conf import Config
from kaiwu_agent.utils.common_func import attached
from agent_ppo.feature.reward_process import GameRewardManager
from torch.optim.lr_scheduler import LambdaLR
from agent_ppo.algorithm.algorithm import Algorithm


@attached
class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        self.cur_model_name = ""
        self.device = device
        # Create Model and convert the model to achannel-last memory format to achieve better performance.
        # 创建模型, 将模型转换为通道后内存格式，以获得更好的性能。
        self.model = Model().to(self.device)
        self.model = self.model.to(memory_format=torch.channels_last)

        # config info
        # 配置信息
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.lstm_hidden = np.zeros([self.lstm_unit_size])
        self.lstm_cell = np.zeros([self.lstm_unit_size])
        self.label_size_list = Config.LABEL_SIZE_LIST
        self.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE

        # env info
        # 环境信息
        self.hero_camp = 0
        self.player_id = 0
        self.game_id = None

        # learning info
        # 学习信息
        self.train_step = 0
        self.lr = Config.INIT_LEARNING_RATE_START
        parameters = self.model.parameters()
        self.optimizer = torch.optim.Adam(params=parameters, lr=self.lr, betas=(0.9, 0.999), eps=1e-8)
        self.parameters = [p for param_group in self.optimizer.param_groups for p in param_group["params"]]
        self.target_lr = Config.TARGET_LR
        self.target_step = Config.TARGET_STEP
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)

        # tools
        # 工具
        self.reward_manager = None
        self.logger = logger
        self.monitor = monitor

        # predict local or remote
        # 本地预测或远程预测
        self.is_predict_remote = True

        self.algorithm = Algorithm(self.model, self.optimizer, self.scheduler, self.device, self.logger, self.monitor)

        super().__init__(agent_type, device, logger, monitor)

    #NOTE: 不用改，学习率衰减函数已经实现好了（可根据需求修改衰减策略）
    def lr_lambda(self, step):
        # Define learning rate decay function
        # 定义学习率衰减函数
        if step > self.target_step:
            return self.target_lr / self.lr
        else:
            return 1.0 - ((1.0 - self.target_lr / self.lr) * step / self.target_step)

    ################################################################
    # 内部推理方法 - 不用改，模型推理逻辑已实现好
    ################################################################
    #NOTE: 不用改，这个方法完成模型推理的完整流程（从ObsData到ActData）
    def _model_inference(self, list_obs_data):
        # Using the network for inference
        # 使用网络进行推理
        feature = [obs_data.feature for obs_data in list_obs_data]
        legal_action = [obs_data.legal_action for obs_data in list_obs_data]
        lstm_cell = [obs_data.lstm_cell for obs_data in list_obs_data]
        lstm_hidden = [obs_data.lstm_hidden for obs_data in list_obs_data]

        input_list = [np.array(feature), np.array(lstm_cell), np.array(lstm_hidden)]
        torch_inputs = [torch.from_numpy(nparr).to(torch.float32) for nparr in input_list]
        for i, data in enumerate(torch_inputs):
            data = data.reshape(-1)
            torch_inputs[i] = data.float()

        feature, lstm_cell, lstm_hidden = torch_inputs
        feature_vec = feature.reshape(-1, self.seri_vec_split_shape[0][0])
        lstm_hidden_state = lstm_hidden.reshape(-1, self.lstm_unit_size)
        lstm_cell_state = lstm_cell.reshape(-1, self.lstm_unit_size)

        format_inputs = [feature_vec, lstm_hidden_state, lstm_cell_state]

        self.model.set_eval_mode()
        with torch.no_grad():
            output_list = self.model(format_inputs, inference=True)

        np_output = []
        for output in output_list:
            np_output.append(output.numpy())

        logits, value, _lstm_cell, _lstm_hidden = np_output[:4]

        _lstm_cell = _lstm_cell.squeeze(axis=0)
        _lstm_hidden = _lstm_hidden.squeeze(axis=0)

        list_act_data = list()
        for i in range(len(legal_action)):
            prob, action, d_action = self._sample_masked_action(logits[i], legal_action[i])
            list_act_data.append(
                ActData(
                    action=action,
                    d_action=d_action,
                    prob=prob,
                    value=value,
                    lstm_cell=_lstm_cell[i],
                    lstm_hidden=_lstm_hidden[i],
                )
            )
        return list_act_data

    ################################################################
    # 平台接口方法 - 不能改方法签名和返回值类型，这些是框架要求的接口
    ################################################################
    #NOTE: 不能改方法签名 predict(self, observation)，平台在分布式训练时会调用此接口
    #NOTE: 必须返回 [ActData(...)] 格式
    @predict_wrapper
    def predict(self, observation):
        # The remote prediction will not call agent.reset in the workflow. Users can use the game_id to determine whether a new environment
        # 远程预测不会在workflow中重置agent，用户可以通过game_id判断是否是新的对局，并根据新对局对agent进行重置
        game_id = observation["game_id"]
        if self.game_id != game_id:
            player_id = observation["player_id"]
            camp = observation["player_camp"]
            self.reset(camp, player_id)
            self.game_id = game_id

        # exploit is automatically called when submitting an evaluation task.
        # The parameter is the observation returned by env, and it returns the action used by env.step.
        # exploit在提交评估任务时自动调用，参数为env返回的state_dict, 返回env.step使用的action
        obs_data = self.observation_process(observation)
        # Call _model_inference for model inference, executing local model inference
        # 模型推理调用_model_inference, 执行本地模型推理
        act_data = self._model_inference([obs_data])[0]
        self.update_status(obs_data, act_data)
        action = self.action_process(observation, act_data, False)
        return [ActData(action=action)]

    #NOTE: 不能改方法签名 exploit(self, observation)，平台在评估任务时会调用此接口
    #NOTE: 必须返回 [ActData(...)] 格式
    @exploit_wrapper
    def exploit(self, observation):
        # Evaluation task will not call agent.reset in the workflow. Users can use the game_id to determine whether a new environment
        # 评估任务不会在workflow中重置agent，用户可以通过game_id判断是否是新的对局，并根据新对局对agent进行重置
        game_id = observation["game_id"]
        if self.game_id != game_id:
            player_id = observation["player_id"]
            camp = observation["player_camp"]
            self.reset(camp, player_id)
            self.game_id = game_id

        # exploit is automatically called when submitting an evaluation task.
        # The parameter is the observation returned by env, and it returns the action used by env.step.
        # exploit在提交评估任务时自动调用，参数为env返回的state_dict, 返回env.step使用的action
        obs_data = self.observation_process(observation)
        # Call _model_inference for model inference, executing local model inference
        # 模型推理调用_model_inference, 执行本地模型推理
        act_data = self._model_inference([obs_data])[0]
        self.update_status(obs_data, act_data)
        d_action = self.action_process(observation, act_data, False)
        return [ActData(d_action=d_action)]

    ################################################################
    # workflow 调用的辅助方法 - 不用改，已经处理好了分布式/单机的逻辑
    ################################################################
    #NOTE: 不用改，workflow中训练时调用此方法，已处理好分布式/单机的区别
    def train_predict(self, observation):
        # Call agent.predict for distributed model inference
        # 调用agent.predict，执行分布式模型推理
        if self.is_predict_remote:
            act_data, model_version = self.predict(observation)
            return act_data[0].action

        obs_data = self.observation_process(observation)
        act_data = self._model_inference([obs_data])[0]
        self.update_status(obs_data, act_data)
        return self.action_process(observation, act_data, True)

    #NOTE: 不用改，workflow中评估时调用此方法，已处理好分布式/单机的区别
    def eval_predict(self, observation):
        # Call agent.predict for distributed model inference
        # 调用agent.predict，执行分布式模型推理
        if self.is_predict_remote:
            act_data, model_version = self.exploit(observation)
            return act_data[0].d_action

        obs_data = self.observation_process(observation)
        act_data = self._model_inference([obs_data])[0]
        self.update_status(obs_data, act_data)
        return self.action_process(observation, act_data, False)

    ################################################################
    # 数据处理方法 - 可以修改，根据需求自定义数据处理逻辑
    ################################################################
    MOVE_BUTTON_INDEX = 2
    MOVE_GRID_SIZE = 16
    MOVE_CENTER_INDEX = MOVE_GRID_SIZE // 2
    TOWER_PROTECT_RADIUS = 8800
    ENEMY_HERO_SAFE_RADIUS = 15000
    DEFAULT_ATTACK_RANGE = 8000
    MAX_RULE_TRIGGER_RANGE = 15000
    DEFAULT_MINION_COUNT = 2

    #NOTE: 可以修改，将ActData转换为环境可用的动作格式，可以添加规则后处理
    def action_process(self, observation, act_data, is_stochastic):
        base_action = act_data.action if is_stochastic else act_data.d_action
        if base_action is None:
            return base_action

        if not isinstance(base_action, list):
            base_action = list(base_action)

        override_action = self._tower_push_override(observation, base_action)
        if override_action is not None:
            return override_action

        return base_action

    #NOTE: 可以修改，将环境observation转换为ObsData，可以自定义特征处理方式
    def observation_process(self, observation):
        feature_vec, legal_action = (
            observation["observation"],
            observation["legal_action"],
        )
        return ObsData(
            feature=feature_vec, legal_action=legal_action, lstm_cell=self.lstm_cell, lstm_hidden=self.lstm_hidden
        )

    #NOTE: 不能改方法签名 learn(self, list_sample_data)，平台会调用此接口进行训练
    @learn_wrapper
    def learn(self, list_sample_data):
        return self.algorithm.learn(list_sample_data)

    #NOTE: 不能改方法签名 save_model(self, path=None, id="1")，平台会调用此接口保存模型
    #NOTE: 文件名必须包含 "model.ckpt-{id}" 格式
    @save_model_wrapper
    def save_model(self, path=None, id="1"):
        # To save the model, it can consist of multiple files, and it is important to ensure that
        #  each filename includes the "model.ckpt-id" field.
        # 保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        torch.save(self.model.state_dict(), model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    #NOTE: 不能改方法签名 load_model(self, path=None, id="1")，平台会调用此接口加载模型
    @load_model_wrapper
    def load_model(self, path=None, id="1"):
        # When loading the model, you can load multiple files, and it is important to ensure that
        # each filename matches the one used during the save_model process.
        # 加载模型, 可以加载多个文件, 注意每个文件名需要和save_model时保持一致
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        if self.cur_model_name == model_file_path:
            self.logger.info(f"current model is {model_file_path}, so skip load model")
        else:
            self.model.load_state_dict(
                torch.load(
                    model_file_path,
                    map_location=self.device,
                )
            )
            self.cur_model_name = model_file_path
            self.logger.info(f"load model {model_file_path} successfully")

    #NOTE: 不用改，每局游戏开始时重置agent状态
    def reset(self, hero_camp, player_id):
        self.hero_camp = hero_camp
        self.hero_camp_str = f"PLAYERCAMP_{hero_camp}"
        self.player_id = player_id
        self.lstm_hidden = np.zeros([self.lstm_unit_size])
        self.lstm_cell = np.zeros([self.lstm_unit_size])
        self.reward_manager = GameRewardManager(player_id)

    #NOTE: 不用改，更新agent的内部状态（用于保存当前帧数据）
    def update_status(self, obs_data, act_data):
        self.obs_data = obs_data
        self.act_data = act_data
        self.lstm_cell = act_data.lstm_cell
        self.lstm_hidden = act_data.lstm_hidden

    ################################################################
    # 推塔规则相关辅助方法
    ################################################################
    def _tower_push_override(self, observation, current_action):
        if observation is None or len(current_action) < 6:
            return None

        frame_state = observation.get("frame_state") or {}
        hero_states = frame_state.get("hero_states") or []
        npc_states = frame_state.get("npc_states") or []

        main_hero = None
        enemy_hero = None
        for hero in hero_states:
            actor_state = hero.get("actor_state") or {}
            if hero.get("player_id") == self.player_id:
                main_hero = hero
            elif actor_state.get("camp") != self.hero_camp_str:
                enemy_hero = hero

        if main_hero is None:
            return None

        enemy_tower = None
        for npc in npc_states:
            if npc.get("sub_type") == "ACTOR_SUB_TOWER" and npc.get("camp") != self.hero_camp_str:
                enemy_tower = npc
                break

        if enemy_tower is None:
            return None

        tower_pos = self._extract_position(enemy_tower)
        hero_pos = self._extract_position(main_hero, is_hero=True)
        if tower_pos is None or hero_pos is None:
            return None

        dist_tower = self._distance(hero_pos, tower_pos)
        if dist_tower is None or dist_tower > self.MAX_RULE_TRIGGER_RANGE:
            return None

        actor_state = main_hero.get("actor_state") or {}
        attack_range = actor_state.get("attack_range")
        if attack_range is None or attack_range <= 0:
            attack_range = self.DEFAULT_ATTACK_RANGE
        required_minion_cnt = 0 if attack_range > self.TOWER_PROTECT_RADIUS else self.DEFAULT_MINION_COUNT

        if not self._has_ally_minion_near_tower(npc_states, tower_pos, required_minion_cnt):
            return None

        enemy_pos = self._extract_position(enemy_hero, is_hero=True) if enemy_hero else None
        if enemy_pos is not None:
            dist_enemy = self._distance(hero_pos, enemy_pos)
            if dist_enemy is not None and dist_enemy <= self.ENEMY_HERO_SAFE_RADIUS:
                return None

        if self._has_enemy_minion_in_range(npc_states, hero_pos, attack_range):
            return None

        if dist_tower <= attack_range:
            attack_action = current_action.copy()
            attack_action[0] = 3
            attack_action[5] = 7
            return attack_action

        move_x, move_z = self._encode_direction(hero_pos, tower_pos)
        override_action = current_action.copy()
        override_action[0] = self.MOVE_BUTTON_INDEX
        override_action[1] = move_x
        override_action[2] = move_z
        override_action[3] = 4
        override_action[4] = 2
        override_action[5] = 0
        return override_action

    def _has_ally_minion_near_tower(self, npc_states, tower_pos, required_cnt):
        if required_cnt <= 0:
            return True
        minion_cnt = 0
        for npc in npc_states:
            if npc.get("camp") != self.hero_camp_str or npc.get("sub_type") != "ACTOR_SUB_SOLDIER":
                continue
            minion_pos = self._extract_position(npc)
            if minion_pos is None:
                continue
            dist = self._distance(minion_pos, tower_pos)
            if dist is not None and dist <= self.TOWER_PROTECT_RADIUS:
                minion_cnt += 1
                if minion_cnt >= required_cnt:
                    return True
        return False

    def _has_enemy_minion_in_range(self, npc_states, hero_pos, attack_range):
        if attack_range is None or attack_range <= 0:
            return False
        for npc in npc_states:
            if npc.get("camp") == self.hero_camp_str or npc.get("sub_type") != "ACTOR_SUB_SOLDIER":
                continue
            pos = self._extract_position(npc)
            if pos is None:
                continue
            dist = self._distance(hero_pos, pos)
            if dist is not None and dist <= attack_range:
                return True
        return False

    def _extract_position(self, entity, is_hero=False):
        if entity is None:
            return None
        if is_hero:
            actor_state = entity.get("actor_state") or {}
            location = actor_state.get("location") or {}
        else:
            location = entity.get("location") or {}
        x = location.get("x")
        z = location.get("z")
        if x is None or z is None:
            return None
        return float(x), float(z)

    def _distance(self, pos_a, pos_b):
        if pos_a is None or pos_b is None:
            return None
        return math.hypot(pos_a[0] - pos_b[0], pos_a[1] - pos_b[1])

    def _encode_direction(self, origin, target):
        dx = target[0] - origin[0] if self.hero_camp == 1 else origin[0] - target[0]
        dz = target[1] - origin[1] if self.hero_camp == 1 else origin[1] - target[1]
        norm = math.hypot(dx, dz)
        if norm < 1e-5:
            return self.MOVE_CENTER_INDEX, self.MOVE_CENTER_INDEX
        dx /= norm
        dz /= norm
        scale = self.MOVE_GRID_SIZE - 1
        idx = int(round(((dx + 1) / 2) * scale))
        idz = int(round(((dz + 1) / 2) * scale))
        idx = max(0, min(scale, idx))
        idz = max(0, min(scale, idz))
        return idx, idz

    ################################################################
    # 动作采样相关方法 - 可以修改采样策略，但注意保持Action Mask机制
    ################################################################
    #NOTE: 可以修改采样策略，但必须保持Action Mask机制（legal_action过滤）
    # get final executable actions
    def _sample_masked_action(self, logits, legal_action):
        """
        Sample actions from predicted logits and legal actions
        return: probability, stochastic and deterministic actions with additional []
        """
        """
        从预测的logits和合法动作中采样动作
        返回：以列表形式概率、随机和确定性动作
        """

        prob_list = []
        action_list = []
        d_action_list = []
        label_split_size = [sum(self.label_size_list[: index + 1]) for index in range(len(self.label_size_list))]
        legal_actions = np.split(legal_action, label_split_size[:-1])
        logits_split = np.split(logits, label_split_size[:-1])
        for index in range(0, len(self.label_size_list) - 1):
            probs = self._legal_soft_max(logits_split[index], legal_actions[index])
            prob_list += list(probs)
            sample_action = self._legal_sample(probs, use_max=False)
            action_list.append(sample_action)
            d_action = self._legal_sample(probs, use_max=True)
            d_action_list.append(d_action)

        # deals with the last prediction, target
        # 处理最后的预测，目标
        index = len(self.label_size_list) - 1
        target_legal_action_o = np.reshape(
            legal_actions[index],  # [12, 8]
            [
                self.legal_action_size[0],
                self.legal_action_size[-1] // self.legal_action_size[0],
            ],
        )
        one_hot_actions = np.eye(self.label_size_list[0])[action_list[0]]  # [12]
        one_hot_actions = np.reshape(one_hot_actions, [self.label_size_list[0], 1])  # [12, 1]
        target_legal_action = np.sum(target_legal_action_o * one_hot_actions, axis=0)

        legal_actions[index] = target_legal_action  # [12]
        probs = self._legal_soft_max(logits_split[-1], target_legal_action)
        prob_list += list(probs)
        sample_action = self._legal_sample(probs, use_max=False)
        action_list.append(sample_action)

        one_hot_actions = np.eye(self.label_size_list[0])[d_action_list[0]]
        one_hot_actions = np.reshape(one_hot_actions, [self.label_size_list[0], 1])
        target_legal_action_d = np.sum(target_legal_action_o * one_hot_actions, axis=0)

        probs = self._legal_soft_max(logits_split[-1], target_legal_action_d)

        d_action = self._legal_sample(probs, use_max=True)
        d_action_list.append(d_action)

        return [prob_list], action_list, d_action_list

    #NOTE: 不用改，已实现好的Legal Action Mask的softmax，将非法动作概率设为极小值
    def _legal_soft_max(self, input_hidden, legal_action):
        _lsm_const_w, _lsm_const_e = 1e20, 1e-5
        _lsm_const_e = 0.00001

        tmp = input_hidden - _lsm_const_w * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True)
        tmp = np.clip(tmp - tmp_max, -_lsm_const_w, 1)
        tmp = (np.exp(tmp) + _lsm_const_e) * legal_action
        probs = tmp / np.sum(tmp, keepdims=True)
        return probs

    #NOTE: 不用改，已实现好的动作采样方法（随机采样或确定性采样）
    def _legal_sample(self, probs, legal_action=None, use_max=False):
        # Sample with probability, input probs should be 1D array
        # 根据概率采样，输入的probs应该是一维数组
        if use_max:
            return np.argmax(probs)

        return np.argmax(np.random.multinomial(1, probs, size=1))

    #NOTE: 不用改，workflow中本地模式加载模型时调用此方法
    def load_model_local(self, model_file_path, idx):
        # When loading the local model, you can load multiple files, and it is important to ensure that
        # each filename matches the one used during the save_model process.
        # 加载本地模型, 可以加载多个文件, 注意每个文件名需要和save_model时保持一致
        self.is_predict_remote = False
        self.model.load_state_dict(
            torch.load(
                model_file_path,
                map_location=torch.device("cpu"),
            )
        )
        self.logger.info(f"agent {idx} load model {model_file_path} successfully")
