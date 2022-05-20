import os
import torch
import numpy as np
import tkinter as tk
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import style
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from tqdm import tqdm
import gym
from Intersection_generate_weighted_fair import TandemEnv, close_traci, LAMBDA_FAIR
from DeepQL_agent import DQNAgent

style.use("ggplot")
customer_color = '#191970'
line_color = '#00BFFF'
# def _take_action(self, action):
#     phase = action
#     for n in range(self.N_tl):
#         if action[n] != self.last_action[n]:
#             phase[n] = (self.last_action[n] + 1) % 4
#     self.last_action = action
#     for tl_ind, ph in enumerate(phase):
#         traci.trafficlight.setPhase(str(tl_ind+1), ph)
#     for _ in range(6):
#         traci.simulationStep()
#     traci.simulationStep()
# QL_MAX = 8
Q_TH = 15
EPISODE_NO_UPDATE_TARGET = 40
DIR = ['NS', 'EW', 'SN', 'WE']
COLOR = ['r', 'b', 'g', 'k']
LAMBDA_FAIR = 0.5
scenario = "Intersection_LAMBDA_FAIR_%3.2f" % LAMBDA_FAIR
FIG_DIRECTORY = '/Users/majidraeis/Desktop/SUMO/figs/' + scenario
FILE_DIRECTORY = '/Users/majidraeis/Desktop/SUMO/files/' + scenario
if not os.path.exists(FIG_DIRECTORY):
    os.makedirs(FIG_DIRECTORY)
if not os.path.exists(FILE_DIRECTORY):
    os.makedirs(FILE_DIRECTORY)

# def binary_rep(st):
#     st_bin = []
#     for j in range(len(st)):
#         st_bin = np.append(st_bin, np.array(list(np.binary_repr(int(st[j])).zfill(QL_MAX))).astype(np.int8)[
#                                    :QL_MAX])
#     return st_bin


def run_train(gui, N_tl, N_time, ep_num):
    """execute the TraCI control loop"""
    hidden = 32
    gamma = 0.99
    epsilon = 0.5
    replay_buffer_size = 10 ** 6
    batch_size = 25
    env = TandemEnv(gui, N_tl, N_time)
    n_actions = env.action_space.n
    state_space_dim = env.observation_space.shape[0]
    agent = DQNAgent(state_space_dim, n_actions, replay_buffer_size, batch_size,
                     hidden, gamma)
    tot_reward = np.zeros(ep_num)
    violation_prob = np.zeros(ep_num)
    for ep in tqdm(range(ep_num)):
        state = env.reset()
        done = False
        epsilon *= 0.99
        epsilon = max(epsilon, 0.01)
        k = 1
        while not done:
            action = agent.get_action(state[0], epsilon)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state[0], action, next_state[0], reward, done)
            agent.update_network()
            state = next_state
            tot_reward[ep] = reward + gamma * tot_reward[ep]
            violation = np.sum(state[0, [0, 2]]) > Q_TH
            violation_prob[ep] += (violation - violation_prob[ep])/k
            k += 1
        close_traci()

        if ep % EPISODE_NO_UPDATE_TARGET == 0:
            agent.update_target_network()

    torch.save(agent.policy_net.state_dict(), FILE_DIRECTORY + '/policy_net.pth')
    print("====================================")
    print("Model has been saved...")
    print("====================================")

    plt.figure()
    plt.plot(tot_reward)
    plt.xlabel('Episodes')
    plt.ylabel('Total reward')
    plt.savefig(FIG_DIRECTORY + '/Reward_training.pdf', bbox_inches='tight')

    plt.figure()
    plt.plot(violation_prob)
    plt.xlabel('Episodes')
    plt.ylabel('Violation Prob')
    plt.savefig(FIG_DIRECTORY + '/Prob_violation_training.pdf', bbox_inches='tight')


def run_test(N_tl, N_time):
    """execute the TraCI control loop"""
    hidden = 32
    gamma = 0.99
    epsilon = 0.0
    ns_throughput_tot = 0
    we_throughput_tot = 0
    replay_buffer_size = 10 ** 6
    batch_size = 25
    env = TandemEnv(gui, N_tl, N_time)
    n_actions = env.action_space.n
    state_space_dim = env.observation_space.shape[0]
    print("state_space_dim=", state_space_dim)
    agent = DQNAgent(state_space_dim, n_actions, replay_buffer_size, batch_size,
                     hidden, gamma)
    state = env.reset()
    done = False
    agent.policy_net.load_state_dict(torch.load(FILE_DIRECTORY + '/policy_net.pth'))

    sum_wait = [[], [], [], []]
    qls = [[], [], [], []]
    time_interval = []
    we_th = []
    ns_th = []
    while not done:
        tem_state = state.squeeze()
        action = agent.get_action(tem_state, epsilon)
        next_state, reward, done, info = env.step(action)
        time_interval.append(env.time)
        we_th.append(env.tot_we_throughput)
        ns_th.append(env.tot_ns_throughput)


        i = 2
        sum_wait[i].append(env.sum_wait[i, 0])
        qls[i].append(env.qls[i])

        # throughput = np.array(cal_throughput(env.ids, env.old_ids))
        # print("throughput", throughput)
        # ns_throughput = sum(throughput[[0, 2]])
        # print("ns_th", ns_throughput)
        # we_throughput = sum(throughput[[1, 3]])
        # print("we_th", we_throughput)
        # print('----------')
        # backlog_flag = np.sum(env.qls) > 0
        # if backlog_flag:
        #     ns_throughput_tot += ns_throughput
        #     we_throughput_tot += we_throughput
        # print("fair_dev", env.fair_dev)
        state = next_state
    close_traci()
    print("we/ns", env.tot_we_throughput/env.tot_ns_throughput)
    plt.plot(time_interval, np.array(we_th)/np.array(time_interval), 'r', label="WE")
    plt.plot(time_interval, np.array(ns_th)/np.array(time_interval), 'b', label="NS")
    plt.xlabel("Time")
    plt.ylabel("Throughput")
    plt.legend()
    plt.savefig(FIG_DIRECTORY + '/proportional_fairness.pdf', bbox_inches='tight')

# this is the main entry point of this script


# def cal_throughput(ids, old_ids):
#     passed_veh_ids = [[], [], [], []]
#     passed_veh_nums = []
#     for dir_index in range(len(ids)):
#         passed_veh_nums.append(0)
#         for i in old_ids[dir_index]:
#             if i not in ids[dir_index]:
#                 passed_veh_ids[dir_index].append(i)
#                 passed_veh_nums[dir_index] += 1
#     return passed_veh_nums

if __name__ == "__main__":
    # options = get_options()
    gui = 1
    N_tl = 1
    N_time = 8000  # number of time steps
    ep_num = 300
    if gui:
        run_test(N_tl, N_time)
    else:
        run_train(gui, N_tl, N_time, ep_num)
