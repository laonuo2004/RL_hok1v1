#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from kaiwu_agent.agent.base_agent import (
    predict_wrapper,
    exploit_wrapper,
    learn_wrapper,
    save_model_wrapper,
    load_model_wrapper,
    BaseAgent,
)
from agent_diy.model.model import Model
from kaiwu_agent.utils.common_func import attached
from agent_diy.feature.definition import *
from diy.conf.conf import Config
from agent_diy.algorithm.algorithm import Algorithm


@attached
class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        super().__init__(agent_type, device, logger, monitor)
        self.cur_model_name = ""
        self.device = device
        # Create Model and convert the model to a channel-last memory format to achieve better performance.
        # 创建模型, 将模型转换为通道后内存格式，以获得更好的性能。
        self.model = Model().to(self.device)
        self.model = self.model.to(memory_format=torch.channels_last)

        # env info
        # 环境信息
        self.hero_camp = 0
        self.player_id = 0
        self.game_id = None

        # tools
        # 工具
        self.reward_manager = None
        self.logger = logger
        self.monitor = monitor
        self.algorithm = Algorithm(self.model, self.device, self.logger, self.monitor)

    def _model_inference(self, list_obs_data):
        # Code to implement model inference
        # 实现模型推理的代码
        list_act_data = [ActData()]
        return list_act_data

    @predict_wrapper
    def predict(self, observation):
        # The remote prediction will not call agent.reset in the workflow. Users can use the game_id to determine whether a new environment
        # 远程预测不会在workflow中重置agent，用户可以通过game_id判断是否是新的对局，并根据新对局对agent进行重置
        game_id = observation["game_id"]
        if self.game_id != game_id:
            self.reset()
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

    @exploit_wrapper
    def exploit(self, observation):
        # Evaluation task will not call agent.reset in the workflow. Users can use the game_id to determine whether a new environment
        # 评估任务不会在workflow中重置agent，用户可以通过game_id判断是否是新的对局，并根据新对局对agent进行重置
        game_id = observation["game_id"]
        if self.game_id != game_id:
            self.reset()
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

    @learn_wrapper
    def learn(self, list_sample_data):
        return self.algorithm.learn(list_sample_data)

    def action_process(self, observation, act_data, is_stochastic):
        # Implement the conversion from ActData to action
        # 实现ActData到action的转换
        return act_data.action

    def observation_process(self, observation):
        # Implement the conversion from State to ObsData
        # 实现State到ObsData的转换
        return ObsData(feature=[], legal_action=[], lstm_cell=self.lstm_cell, lstm_hidden=self.lstm_hidden)

    def reset(self):
        pass

    def update_status(self, obs_data, act_data):
        self.obs_data = obs_data
        self.act_data = act_data

    @save_model_wrapper
    def save_model(self, path=None, id="1"):
        # To save the model, it can consist of multiple files, and it is important to ensure that
        #  each filename includes the "model.ckpt-id" field.
        # 保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        torch.save(self.model.state_dict(), model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

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
