#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import os
import time
import random
from agent_ppo.feature.definition import (
    sample_process,
    build_frame,
    FrameCollector,
    NONE_ACTION,
)
from kaiwu_agent.utils.common_func import attached
from agent_ppo.conf.conf import GameConfig
from tools.model_pool_utils import get_valid_model_pool
from kaiwudrl.common.checkpoint.model_file_sync import ModelFileSync
from tools.train_env_conf_validate import read_usr_conf, check_usr_conf
from tools.metrics_utils import get_training_metrics


@attached
def workflow(envs, agents, logger=None, monitor=None):
    # Whether the agent is training, corresponding to do_predicts
    # 智能体是否进行训练
    do_learns = [True, True]
    last_save_model_time = time.time()

    # Directly pull and load the model file from the model pool.
    # 直接从modelpool里拉取model文件加载
    model_file_sync_wrapper = ModelFileSync()
    model_file_sync_wrapper.make_local_model_dirs(logger)

    # Read and validate configuration file
    # 配置文件读取和校验
    usr_conf = read_usr_conf("agent_ppo/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error(f"usr_conf is None, please check agent_ppo/conf/train_env_conf.toml")
        return
    # check_usr_conf is a tool to check whether the environment configuration is correct
    # It is recommended to perform a check before calling reset.env
    # check_usr_conf会检查环境配置是否正确，建议调用reset.env前先检查一下
    valid = check_usr_conf(usr_conf, logger)
    if not valid:
        logger.error(f"check_usr_conf return False, please check agent_ppo/conf/train_env_conf.toml")
        return

    while True:
        for g_data in run_episodes(envs, agents, logger, monitor, model_file_sync_wrapper, usr_conf):
            for index, (d_learn, agent) in enumerate(zip(do_learns, agents)):
                if d_learn and len(g_data[index]) > 0:
                    # The learner trains in a while true loop, here learn actually sends samples
                    # learner 采用 while true 训练，此处 learn 实际为发送样本
                    agent.learn(g_data[index])
            g_data.clear()

            now = time.time()
            if now - last_save_model_time > GameConfig.MODEL_SAVE_INTERVAL:
                agents[0].save_model()
                last_save_model_time = now


def run_episodes(envs, agents, logger, monitor, model_file_sync_wrapper, usr_conf):
    # hok1v1 environment
    # hok1v1环境
    env = envs[0]
    # Number of agents, in hok1v1 the value is 2
    # 智能体数量，在hok1v1中值为2
    agent_num = len(agents)
    # Episode counter
    # 对局数量计数器
    episode_cnt = 0
    # ID of Agent to training
    # 每一局要训练的智能体的id
    default_opponent_agent = usr_conf["env_conf"]["episode"]["opponent_agent"]
    if usr_conf["env_conf"]["monitor"]["monitor_side"] == -1:
        auto_switch_monitor_side = True
        train_agent_id = 0
    else:
        auto_switch_monitor_side = False
        train_agent_id = usr_conf["env_conf"]["monitor"]["monitor_side"]

    # Frame Collector
    # 帧收集器
    frame_collector = FrameCollector(agent_num)
    # Make eval matches as evenly distributed as possible
    # 引入随机因子，让eval对局尽可能平均分布
    random_eval_start = random.randint(0, usr_conf["env_conf"]["episode"]["eval_interval"])

    predict_success_count = 0
    load_model_success_count = 0
    last_report_monitor_time = 0

    # Single environment process (30 frame/s)
    # 单局流程 (30 frame/s)
    while True:
        # Settings before starting a new environment
        # 以下是启动一个新对局前的设置
        (
            pull_model_success,
            current_available_model_files,
        ) = model_file_sync_wrapper.get_last_model_file(logger)
        if not pull_model_success or not current_available_model_files:
            continue
        model_file_path = current_available_model_files[0]

        # Retrieving training metrics
        # 获取训练中的指标
        training_metrics = get_training_metrics()
        if training_metrics:
            for key, value in training_metrics.items():
                if key == "env":
                    for env_key, env_value in value.items():
                        logger.info(f"training_metrics {key} {env_key} is {env_value}")
                else:
                    logger.info(f"training_metrics {key} is {value}")

        # Set the id of the agent to be trained. id=0 means the blue side, id=1 means the red side.
        # 设置要训练的智能体的id，id=0表示蓝方，id=1表示红方，每一局都切换一次阵营。默认对手智能体是selfplay即自己
        if auto_switch_monitor_side:
            train_agent_id = 1 - train_agent_id
            if train_agent_id not in [0, 1]:
                raise Exception("monitor_side is not valid, valid monitor_side list is [0, 1], please check")
        usr_conf["env_conf"]["monitor"]["monitor_side"] = train_agent_id

        # Evaluate at a certain frequency during training to reflect the improvement of the agent during training
        # 智能体支持边训练边评估，训练中按一定的频率进行评估，反映智能体在训练中的水平
        opponent_agent = default_opponent_agent
        eval_interval = usr_conf["env_conf"]["episode"]["eval_interval"]
        if eval_interval == 0:
            is_eval = True
        else:
            is_eval = (episode_cnt + random_eval_start) % eval_interval == 0
        if is_eval:
            opponent_agent = usr_conf["env_conf"]["episode"]["eval_opponent_type"]
        usr_conf["env_conf"]["episode"]["opponent_agent"] = opponent_agent

        # Start a new environment
        # 启动新对局，返回初始环境状态
        observation, state = env.reset(usr_conf=usr_conf)
        if observation is None:
            logger.info(f"episode {episode_cnt}, reset is None happened!")
            continue

        # Game variables
        # 对局变量
        episode_cnt += 1
        frame_no = 0
        step = 0
        # Record the cumulative rewards of the agent in the environment
        # 记录对局中智能体的累积回报，用于上报监控
        total_reward_dicts = [{}, {}]
        logger.info(f"Episode {episode_cnt} start, usr_conf is {usr_conf}")

        # The 'do_predicts' specifies which agents are to perform model predictions.
        # Since the default opponent model is 'selfplay', it is set to [True, True] by default.
        # do_predicts指定哪些智能体要进行模型预测，由于默认对手模型是selfplay，默认设置[True, True]
        do_predicts = [True, True]

        # Reset agent
        # 重置agent
        for i, agent in enumerate(agents):
            player_id = observation[i]["player_id"]
            camp = observation[i]["player_camp"]
            agent.reset(camp, player_id)

            # The agent to be trained should load the latest model
            # 要训练的智能体应加载最新的模型
            if i == train_agent_id:
                # train_agent_id uses the latest model
                # train_agent_id 使用最新模型
                agent.load_model_local(model_file_path, i)
                load_model_success_count += 1
            else:
                if opponent_agent == "common_ai":
                    # common_ai does not need to load a model, no need to predict
                    # 如果对手是 common_ai 则不需要加载模型, 也不需要进行预测
                    do_predicts[i] = False
                elif opponent_agent == "selfplay":
                    # Training model, "latest" - latest model, "random" - random model from the model pool
                    # 加载训练过的模型，可以选择最新模型，也可以选择随机模型 "latest" - 最新模型, "random" - 模型池中随机模型
                    # agent.load_model(id="latest")
                    agent.load_model_local(model_file_path, i)
                    load_model_success_count += 1
                else:
                    # Opponent model, model_id is checked from kaiwu.json
                    # 选择kaiwu.json中设置的对手模型, model_id 即 opponent_agent，必须设置正确否则报错
                    eval_candidate_model = get_valid_model_pool(logger)
                    if int(opponent_agent) not in eval_candidate_model:
                        raise Exception(f"model_id {opponent_agent} not in {eval_candidate_model}")
                    else:
                        agent.load_model(id=opponent_agent)
                        agent.is_predict_remote = True

            logger.info(f"agent_{i} reset playerid:{player_id} camp:{camp}")

        # Reward initialization
        # 回报初始化
        for i in range(agent_num):
            reward = agents[i].reward_manager.result(observation[i]["frame_state"])
            observation[i]["reward"] = reward
            for key, value in reward.items():
                if key in total_reward_dicts[i]:
                    total_reward_dicts[i][key] += value
                else:
                    total_reward_dicts[i][key] = value

        # Reset environment frame collector
        # 重置环境帧收集器
        frame_collector.reset(num_agents=agent_num)

        max_frame_no = int(os.environ.get("max_frame_no", "0"))

        while True:
            # Initialize the default actions. If the agent does not make a decision, env.step uses the default action.
            # 初始化默认的actions，如果智能体不进行决策，则env.step使用默认action
            actions = [
                NONE_ACTION,
            ] * agent_num

            for index, (d_predict, agent) in enumerate(zip(do_predicts, agents)):
                if d_predict:
                    if not is_eval:
                        actions[index] = agent.train_predict(observation[index])
                        predict_success_count += 1
                    else:
                        actions[index] = agent.eval_predict(observation[index])
                        predict_success_count += 1

                    # Only when do_predict=True and is_eval=False, the agent's environment data is saved.
                    # 仅do_predict=True且is_eval=False时，智能体的对局数据保存。即评估对局数据不训练，不是最新模型产生的数据不训练
                    if not is_eval and not agent.is_predict_remote:
                        frame = build_frame(agent, observation[index])
                        frame_collector.save_frame(frame, agent_id=index)

            """
            The format of action is like [[2, 10, 1, 14, 8, 0], [1, 3, 10, 10, 9, 0]]
            There are 2 agents, so the length of the array is 2, and the order of values in
            each element is: button, move (2), skill (2), target
            action格式形如[[2, 10, 1, 14, 8, 0], [1, 3, 10, 10, 9, 0]]
            2个agent, 故数组的长度为2, 每个元素里面的值的顺序是:button, move(2个), skill(2个), target
            """

            # Step forward
            # 推进环境到下一帧，得到新的状态
            frame_no, observation, score, terminated, truncated, state = env.step(actions)

            # Disaster recovery
            # 容灾
            if observation is None:
                logger.info(f"episode {episode_cnt}, step({step}) is None happened!")
                break

            # Reward generation
            # 计算回报，作为当前环境状态observation的一部分
            for i in range(agent_num):
                reward = agents[i].reward_manager.result(observation[i]["frame_state"])
                observation[i]["reward"] = reward
                for key, value in reward.items():
                    if key in total_reward_dicts[i]:
                        total_reward_dicts[i][key] += value
                    else:
                        total_reward_dicts[i][key] = value

            step += 1

            now = time.time()
            if now - last_report_monitor_time >= 60:
                monitor_data = {
                    "actor_predict_succ_cnt": predict_success_count,
                    "actor_load_last_model_succ_cnt": load_model_success_count,
                }

                monitor.put_data({os.getpid(): monitor_data})
                last_report_monitor_time = now

            # Normal end or timeout exit
            # 正常结束或超时退出
            is_gameover = terminated or truncated or (max_frame_no > 0 and frame_no >= max_frame_no)
            if is_gameover:
                logger.info(
                    f"episode_{episode_cnt} terminated in fno_{frame_no}, truncated:{truncated}, eval:{is_eval}, train_agent_rewards:{total_reward_dicts[train_agent_id]}"
                )
                # Reward for saving the last state of the environment
                # 保存环境最后状态的reward
                for index, (d_predict, agent) in enumerate(zip(do_predicts, agents)):
                    if d_predict and not is_eval and not agent.is_predict_remote:
                        frame_collector.save_last_frame(
                            agent_id=index,
                            reward=observation[index]["reward"]["reward_sum"],
                        )

                monitor_data = {}
                if monitor and is_eval:
                    monitor_data["reward"] = round(total_reward_dicts[train_agent_id]["reward_sum"], 2)
                    monitor.put_data({os.getpid(): monitor_data})

                # Sample process
                # 进行样本处理，准备训练
                if len(frame_collector) > 0 and not is_eval:
                    list_agents_samples = sample_process(frame_collector)
                    yield list_agents_samples
                break
