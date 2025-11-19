#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch
import math     # 新增！！ 计算距离等简单几何量，用于初始固定化策略

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
    #NOTE: 可以修改，将ActData转换为环境可用的动作格式，可以添加规则后处理

    '''尝试加入固定策略：开局若干秒后强制往己方塔移动'''
    '''实际测试影响不大，PPO训练时间达到20h之后就可以基本开局向塔走，但是此改动是可以保留的，另外下面将75帧改成5帧，但是没做测评，预计可以正常执行'''
    def action_process(self, observation, act_data, is_stochastic):
        """
        决策后统一处理动作的函数。
        - 这里先检查是否需要启用“固定走塔策略”（例如开局若干秒后强制往己方塔移动）
        - 如果不触发固定策略，就按原来的 PPO 动作输出逻辑返回：
          * 训练时使用随机采样动作 action（is_stochastic=True）
          * 评估/测试时使用最大概率动作 d_action（is_stochastic=False）
        """

        # 从环境返回的 observation 中尝试取出帧状态 frame_state
        # frame_state 是 env.step 返回的原始帧信息（包含 hero_states、npc_states 等）
        frame_state = observation.get("frame_state", None)

        # 如果当前观测中确实包含 frame_state，才有可能根据帧号等信息做“固定策略”
        if frame_state is not None:
            # 从 frame_state 中获取当前帧号，若不存在则默认 -1
            frame_no = frame_state.get("frameNo", -1)

            # 当帧号达到或超过 75 时（约等于开局 5 秒后），尝试触发“固定走塔策略”
            # 这里的 75 是一个经验阈值：1 step = 6 frame，75 frame ≈ 12~13 step，对应现实时间若干秒
            #这里为了突出效果改成5帧后开始执行固定策略，即约等于开局0.3秒后
            if frame_no >= 5:
                # 调用自定义的固定策略函数，传入完整 observation
                # 该函数内部根据英雄位置、己方塔位置等计算一个“朝塔移动”的动作
                fixed_action = self._fixed_go_tower_action(observation)

                # 如果固定策略返回了一个非 None 的合法动作，
                # 说明本帧要“强制执行固定策略动作”，直接返回该动作并覆盖 PPO 的输出
                if fixed_action is not None:
                    return fixed_action

        # 如果没有触发固定策略（frame_state 不存在 / 帧号未达阈值 / 固定策略返回 None），
        # 则走原有逻辑：
        #   训练时采用随机采样动作 action（策略分布上采样，便于探索）
        #   评估时采用最大概率动作 d_action（贪心选取当前策略认为最优的动作）
        if is_stochastic:
            # 训练阶段，保持原有随机采样动作，不做任何指引或覆盖
            return act_data.action
        else:
            # 评估阶段（is_stochastic=False）默认返回模型的确定性动作
            # 如果你希望在评估时启用“技能方向指引”（仅改变技能方向，不改变是否释放技能/何时释放），
            # 请按需将下面整个注释块解注释并启用。下面实现：
            # - 仅在 button == 4（技能1）且 target != 0 时尝试指引方向
            # - 先尝试将方向指向最近的敌方英雄（如果有）
            # - 如果没有敌方英雄，则寻找最近的小兵（soldier），并将方向指向最近小兵
            # - 否则回退到模型原始决定
            # - 注意：所有变更仅修改 skill_x, skill_z 两个维度（方向），不会修改 button/target
            #
            # 注：此段代码为注释示例，目的在于给出一个可直接复制启用的实现；
            # 启用前请确保 frame_state 的字段格式与注释中的访问方式一致。

            # ---------- 开始：评估期技能方向指引（全部为注释，默认不生效） ----------
            # out_action = list(act_data.d_action)
            # try:
            #     # 取出帧信息（包含 hero_states, npc_states 等）
            #     frame_state = observation.get("frame_state", None)
            #     # 仅在有帧信息且按钮为技能1（4）时才尝试指引
            #     if frame_state is not None and len(out_action) >= 6 and out_action[0] == 4:
            #         # 仅在 target 非 0 时才尝试使用指引（表示有明确目标类型）
            #         if out_action[5] != 0:
            #             hero_states = frame_state.get("hero_states", [])
            #             npc_states = frame_state.get("npc_states", [])
            #
            #             # 找到当前 Agent 控制的英雄（用于计算相对位置）
            #             main_hero = None
            #             for h in hero_states:
            #                 if h.get("player_id") == self.player_id:
            #                     main_hero = h
            #                     break
            #
            #             # 如果没找到 main_hero，就不做指引
            #             if main_hero is None:
            #                 raise RuntimeError("main_hero not found in frame_state")
            #
            #             mh_loc = main_hero["actor_state"]["location"]
            #             mh_x, mh_z = mh_loc["x"], mh_loc["z"]
            #
            #             # 1) 优先寻找最近的敌方英雄（如果存在）
            #             enemy_heroes = [h for h in hero_states if h.get("camp") != self.hero_camp]
            #             chosen = None
            #             if len(enemy_heroes) > 0:
            #                 min_d = None
            #                 for eh in enemy_heroes:
            #                     loc = eh["actor_state"]["location"]
            #                     d = math.hypot(loc["x"] - mh_x, loc["z"] - mh_z)
            #                     if min_d is None or d < min_d:
            #                         min_d = d
            #                         chosen = eh
            #             else:
            #                 # 2) 若无敌方英雄，再寻找最近的小兵（NPC 中 sub_type == 'ACTOR_SUB_SOLDIER'）
            #                 soldiers = [n for n in npc_states if n.get("sub_type") == "ACTOR_SUB_SOLDIER"]
            #                 if len(soldiers) > 0:
            #                     min_d = None
            #                     for s in soldiers:
            #                         loc = s.get("location")
            #                         if loc is None:
            #                             continue
            #                         d = math.hypot(loc["x"] - mh_x, loc["z"] - mh_z)
            #                         if min_d is None or d < min_d:
            #                             min_d = d
            #                             chosen = s
            #
            #             # 3) 如果选中了目标（enemy hero 或最近小兵），把 skill_x/skill_z 指向该目标
            #             if chosen is not None:
            #                 # 有两种数据结构：英雄使用 actor_state.location，小兵可能直接在 location 字段
            #                 if "actor_state" in chosen:
            #                     tgt_loc = chosen["actor_state"]["location"]
            #                     tgt_x, tgt_z = tgt_loc["x"], tgt_loc["z"]
            #                 else:
            #                     tgt_loc = chosen.get("location")
            #                     tgt_x, tgt_z = tgt_loc["x"], tgt_loc["z"]
            #                 dx = tgt_x - mh_x
            #                 dz = tgt_z - mh_z
            #                 # 将连续角度映射到 16 个离散方向
            #                 ang = math.atan2(dz, dx)
            #                 idx = int(round(ang / (2 * math.pi) * 16)) % 16
            #                 # 仅覆盖方向分量，不改变 button/target
            #                 out_action[3] = int(idx)
            #                 out_action[4] = int(idx)
            #
            # except Exception:
            #     # 为了安全起见，任何异常都不应该影响评估流程，回退到模型原始决定
            #     out_action = list(act_data.d_action)
            #
            # # 启用时返回被指引后的动作
            # return out_action
            # ---------- 结束：评估期技能方向指引（全部为注释，默认不生效） ----------

            # ---------- 开始：评估期技能二（button==5）方向指引（注释示例） ----------
            # 说明：
            # - 技能二（button==5）只对英雄生效，且施法方向作用于整条路径。
            # - 本示例仅修改方向（skill_x, skill_z），不改变 button/target/是否释放，由智能体决定。
            # - 当且仅当：评估阶段（此处位于评估分支）、button==5、且能在 frame_state 中看到敌方英雄坐标时，覆盖方向为指向最近可见敌方英雄。
            # - 若视野中没有敌方英雄坐标，则回退到模型原始决定（不做任何修改）。
            #
            # 使用方式：将下面注释段解注释即可启用；建议先在离线评估/仿真中验证效果后再批量使用。
            #
            # out_action = list(act_data.d_action)
            # try:
            #     frame_state = observation.get("frame_state", None)
            #     # 只有在有帧信息且动作长度满足时才尝试
            #     if frame_state is not None and len(out_action) >= 6 and out_action[0] == 5:
            #         # 技能二仅对英雄生效：检查视野中是否有敌方英雄位置
            #         hero_states = frame_state.get("hero_states", [])
            #         # 找到当前 Agent 控制的英雄
            #         main_hero = None
            #         for h in hero_states:
            #             if h.get("player_id") == self.player_id:
            #                 main_hero = h
            #                 break
            #         if main_hero is None:
            #             raise RuntimeError("main_hero not found in frame_state")
            #         mh_loc = main_hero["actor_state"]["location"]
            #         mh_x, mh_z = mh_loc["x"], mh_loc["z"]
            #
            #         # 收集所有可见的敌方英雄（含坐标）
            #         enemy_heroes = [h for h in hero_states if h.get("camp") != self.hero_camp]
            #         # 如果有可见敌方英雄，则取最近的一个，计算方向并覆盖 skill_x/skill_z
            #         if len(enemy_heroes) > 0:
            #             min_d = None
            #             chosen = None
            #             for eh in enemy_heroes:
            #                 # 英雄位置通常在 actor_state.location
            #                 if "actor_state" not in eh:
            #                     continue
            #                 loc = eh["actor_state"]["location"]
            #                 if loc is None:
            #                     continue
            #                 d = math.hypot(loc["x"] - mh_x, loc["z"] - mh_z)
            #                 if min_d is None or d < min_d:
            #                     min_d = d
            #                     chosen = loc
            #             if chosen is not None:
            #                 tgt_x, tgt_z = chosen["x"], chosen["z"]
            #                 dx = tgt_x - mh_x
            #                 dz = tgt_z - mh_z
            #                 ang = math.atan2(dz, dx)
            #                 idx = int(round(ang / (2 * math.pi) * 16)) % 16
            #                 out_action[3] = int(idx)
            #                 out_action[4] = int(idx)
            # except Exception:
            #     # 任何异常都回退到模型原始决定
            #     out_action = list(act_data.d_action)
            #
            # # 启用后返回被指引的动作（只修改方向）
            # return out_action
            # ---------- 结束：评估期技能二（button==5）方向指引（注释示例） ----------

            # 默认仍然返回模型的确定性动作，不启用任何指引

            # ---------- 开始：评估期技能三（button==6）方向指引（注释示例） ----------
            # 说明：
            # - 技能三（button==6）与技能二类似，只与方向向量的方向有关，与模的大小无关。
            # - 当智能体决定释放技能三时：
            #     1) 若能观测到敌方英雄坐标，向最近敌方英雄方向施法（按技能二的方向计算方法）。
            #     2) 若视野中无敌方英雄，则检测小兵：若存在多个小兵（1/2/3），则按向量和方向施法：
            #           - 对每个小兵计算向量 v_i = 小兵坐标 - 我方英雄坐标
            #           - 求和 V = sum_i v_i（向量和）
            #           - V 的方向即为施法方向（如果 V 很大，可按比例缩放到合法范围，但方向不变）
            #     3) 若无小兵，回退到模型原始决定（不修改方向）
            # - 本实现仅覆盖方向分量 out_action[3], out_action[4]（skill_x, skill_z），不改变 button/target
            # - 为安全起见，本段为注释示例；任何异常都应回退到模型决定
            #
            # 使用方式：将下面注释段解注释并启用；推荐配合配置开关控制（例如 Config.FORCE_SKILL_DIR_IN_EVAL）
            #
            # out_action = list(act_data.d_action)
            # try:
            #     frame_state = observation.get("frame_state", None)
            #     if frame_state is not None and len(out_action) >= 6 and out_action[0] == 6:
            #         # 找到我方英雄位置
            #         hero_states = frame_state.get("hero_states", [])
            #         main_hero = None
            #         for h in hero_states:
            #             if h.get("player_id") == self.player_id:
            #                 main_hero = h
            #                 break
            #         if main_hero is None:
            #             raise RuntimeError("main_hero not found in frame_state")
            #         mh_loc = main_hero["actor_state"]["location"]
            #         mh_x, mh_z = mh_loc["x"], mh_loc["z"]
            #
            #         # 1) 优先寻找最近的敌方英雄
            #         enemy_heroes = [h for h in hero_states if h.get("camp") != self.hero_camp]
            #         chosen = None
            #         if len(enemy_heroes) > 0:
            #             min_d = None
            #             for eh in enemy_heroes:
            #                 if "actor_state" not in eh:
            #                     continue
            #                 loc = eh["actor_state"]["location"]
            #                 if loc is None:
            #                     continue
            #                 d = math.hypot(loc["x"] - mh_x, loc["z"] - mh_z)
            #                 if min_d is None or d < min_d:
            #                     min_d = d
            #                     chosen = loc
            #             if chosen is not None:
            #                 tgt_x, tgt_z = chosen["x"], chosen["z"]
            #                 dx = tgt_x - mh_x
            #                 dz = tgt_z - mh_z
            #                 ang = math.atan2(dz, dx)
            #                 idx = int(round(ang / (2 * math.pi) * 16)) % 16
            #                 out_action[3] = int(idx)
            #                 out_action[4] = int(idx)
            #         else:
            #             # 2) 没有敌方英雄，则收集小兵并求向量和
            #             npc_states = frame_state.get("npc_states", [])
            #             soldiers = [n for n in npc_states if n.get("sub_type") == "ACTOR_SUB_SOLDIER"]
            #             if len(soldiers) > 0:
            #                 sum_dx, sum_dz = 0.0, 0.0
            #                 for s in soldiers:
            #                     loc = s.get("location")
            #                     if loc is None:
            #                         continue
            #                     sum_dx += (loc["x"] - mh_x)
            #                     sum_dz += (loc["z"] - mh_z)
            #                 # 如果向量和近似为零（数值不稳定），回退到模型决定
            #                 if abs(sum_dx) < 1e-6 and abs(sum_dz) < 1e-6:
            #                     raise RuntimeError("sum vector is too small")
            #                 ang = math.atan2(sum_dz, sum_dx)
            #                 idx = int(round(ang / (2 * math.pi) * 16)) % 16
            #                 out_action[3] = int(idx)
            #                 out_action[4] = int(idx)
            # except Exception:
            #     out_action = list(act_data.d_action)
            #
            # # 启用后返回被指引的动作（只修改方向）
            # return out_action
            # ---------- 结束：评估期技能三（button==6）方向指引（注释示例） ----------

            return act_data.d_action

    '''以下为固定策略的具体实现函数'''
    def _fixed_go_tower_action(self, observation):
        """
        固定策略：根据当前帧信息，构造一个“朝己方防御塔移动”的动作。
        - 如果无法拿到必要信息（例如没有 frame_state / 找不到己方英雄 / 防御塔），返回 None，不干预 PPO。
        - 如果英雄已经在己方塔附近（距离阈值之内），返回 None，不再强制往塔走。
        - 否则，根据英雄位置和己方塔位置计算一个 16 方向中的离散角度，
          并构造一个合法的 6 维动作：[button, move_x, move_z, skill_x, skill_z, target]。
        """

        # 从环境观测中取出原始帧状态 frame_state（包含英雄、NPC、防御塔等原始信息）
        frame_state = observation.get("frame_state", None)
        if frame_state is None:
            # 如果当前 observation 中没有 frame_state，就没法做几何计算 → 不干预 PPO，返回 None
            return None

        # 所有英雄状态列表（包括我方与敌方英雄）
        hero_states = frame_state.get("hero_states", [])
        # 所有 NPC 状态列表（包括小兵、防御塔、水晶等）
        npc_states = frame_state.get("npc_states", [])

        # main_hero：当前这个 Agent 所控制的英雄
        # main_tower：己方防御塔
        # enemy_tower：敌方防御塔（仅用于估计整体地图尺度和“塔附近”阈值）
        main_hero = None
        main_tower = None
        enemy_tower = None

        # 在 hero_states 中找到“我自己”控制的那一个英雄
        # 对比条件：player_id == self.player_id（在 Agent.reset 时保存）
        for hero in hero_states:
            if hero.get("player_id") == self.player_id:
                main_hero = hero
                break

        # 如果找不到自己英雄，说明当前帧信息不完整或异常，不做固定策略
        if main_hero is None:
            return None

        # 在 npc_states 中查找双方的防御塔：
        # sub_type == "ACTOR_SUB_TOWER" 意味着这是一个“塔”单位
        for organ in npc_states:
            if organ.get("sub_type") == "ACTOR_SUB_TOWER":
                # 如果阵营 == self.hero_camp，则视为己方塔
                if organ.get("camp") == self.hero_camp:
                    # 只记录第一个符合条件的己方塔（1v1 只有一座主塔）
                    if main_tower is None:
                        main_tower = organ
                else:
                    # 阵营不等于当前英雄阵营 → 敌方塔，同样只记录一个
                    if enemy_tower is None:
                        enemy_tower = organ

        # 如果没能找到己方防御塔，则无法计算“朝塔移动”的方向 → 不干预 PPO
        if main_tower is None:
            return None

        # 取出英雄当前位置（世界坐标 / 地图坐标）
        hero_loc = main_hero["actor_state"]["location"]
        # 取出己方防御塔位置
        tower_loc = main_tower["location"]

        # 英雄坐标 (x, z)
        hero_pos = (hero_loc["x"], hero_loc["z"])
        # 己方塔坐标 (x, z)
        tower_pos = (tower_loc["x"], tower_loc["z"])

        # 计算从英雄指向己方塔的向量 (dx, dz)
        dx = tower_pos[0] - hero_pos[0]
        dz = tower_pos[1] - hero_pos[1]

        # 英雄到己方塔的欧氏距离，用于判断“是否已经够近”
        dist_hero2tower = math.hypot(dx, dz)

        # 下面估计一个“塔到塔的距离”，用来决定“塔附近”的尺度
        if enemy_tower is not None:
            # 若能获取敌方塔，则用“己方塔到敌方塔”的距离作为地图尺度
            emy_loc = enemy_tower["location"]
            emy_pos = (emy_loc["x"], emy_loc["z"])
            # 己方塔 → 敌方塔的距离
            dist_main2emy = math.hypot(emy_pos[0] - tower_pos[0], emy_pos[1] - tower_pos[1])
            # 把“塔到塔距离的 20%”作为“己方塔附近”的阈值
            near_threshold = dist_main2emy * 0.2
        else:
            # 如果拿不到敌方塔，就退化成一个固定的距离阈值（单位为坐标系的长度单位）
            near_threshold = 300.0

        # 如果当前英雄距离己方塔已经“足够近”（小于阈值），说明已经在塔附近，不再强制移动
        if dist_hero2tower <= near_threshold:
            return None

        # 计算从英雄指向己方塔的角度（弧度制），atan2(dz, dx) 范围在 [-pi, pi]
        angle = math.atan2(dz, dx)
        # 将连续角度映射到 16 个离散方向（0~15）：
        # 1. angle / (2*pi) → [-0.5, 0.5) 区间
        # 2. 乘以 16，四舍五入得到离散方向索引
        # 3. 取模 16，确保在 [0, 15] 范围内
        idx = int(round(angle / (2 * math.pi) * 16)) % 16

        # 按动作空间定义构造一个合法的 6 维动作：
        # LABEL_SIZE_LIST = [12, 16, 16, 16, 16, 9]
        # 其中：
        #   action[0]：button（12 类动作：Move / 普攻 / 技能1 / 技能2 / 技能3 / 闪现等）
        #   action[1]：Move_X（16 个方向）
        #   action[2]：Move_Z（16 个方向）
        #   action[3]：Skill_X（这里不用技能，塞一个无意义的默认值 15）
        #   action[4]：Skill_Z（同上）
        #   action[5]：Target（9 个预定义目标，这里选择 0 表示“无目标 / 默认”）
        button_move = 1        # 假设 1 对应“移动”这个 button（与环境约定一致）
        move_x = idx           # X 方向上使用我们计算出的离散方向 idx
        move_z = idx           # Z 方向同样使用 idx，让移动方向指向塔
        skill_x = 15           # 不释放技能，用一个默认的最大索引占位（通常 15 表示“无意义方向”）
        skill_z = 15           # 同上
        target_none = 0        # 目标设为 0，通常表示“无特定目标”

        # 返回一个完整的多离散动作列表 [button, move_x, move_z, skill_x, skill_z, target]
        # 这个动作会被 env.step 直接使用，覆盖 PPO 原始输出
        return [button_move, move_x, move_z, skill_x, skill_z, target_none]

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
