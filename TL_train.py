import os
import torch
import numpy as np
import tkinter as tk
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import style
import xml.etree.ElementTree as ET
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
import gym
from Intersection_generate import RL_Env, RL_Env_WeightedFair, close_traci
from DeepQL_agent import DQNAgent

# style.use/("ggplot")
customer_color = '#191970'
line_color = '#00BFFF'
Q_TH = 15
PHASE_NS = 0
PHASE_WE = 1
X_NUM_LANE = 3
Y_NUM_LANE = 2
EW_TOT_RATIO = 3 / 5
CYCLE_LENGTH = 60
GAMMA = 2
EPISODE_NO_UPDATE_TARGET = 40
DIR = ['NS', 'EW', 'SN', 'WE']
COLOR = ['r', 'b', 'g', 'k']
DIR = ['u', 'r', 'd', 'l']
LANE_NUM = [Y_NUM_LANE, X_NUM_LANE, Y_NUM_LANE, X_NUM_LANE]
LAMBDA_FAIR = 0.01
NS_WEIGHT = 1.0
WE_WEIGHT = 1.5
NS_WEIGHT /= LAMBDA_FAIR
WE_WEIGHT /= LAMBDA_FAIR
WIN = 10*60  # 15 min
seed = 1000 #10 55 110
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True


def window_average_rate(time_list, win):
    rate = []
    for i in range(len(time_list)):
        count = 0
        while i+count < len(time_list) and time_list[i+count] < time_list[i] + win:
            count += 1
        rate.append(count/win)
    return rate


def window_average_phase(input_list, win):
    ave = []
    len_input = len(input_list)
    for i in range(len(input_list)):
        count = Counter(input_list[i:min(i+win, len_input)])
        ave.append(count[1]/(count[0] + count[1]))
    return ave


def window_average(arrival_list, waiting_time_list, win):
    ave = []
    for i in range(len(arrival_list)):
        count = 0
        average = 0
        while i+count < len(arrival_list) and arrival_list[i+count] < arrival_list[i] + win:
            average += waiting_time_list[i+count]
            count += 1
        ave.append(average/count)
    return ave


def window_max(arrival_list, waiting_time_list, win):
    maximums = []
    for i in range(len(arrival_list)):
        count = 0
        max_waiting = 0
        while i+count < len(arrival_list) and arrival_list[i+count] < arrival_list[i] + win:

            max_waiting = max(max_waiting, waiting_time_list[i+count])
            count += 1
        maximums.append(max_waiting)
    return maximums


def jain_index(input_list):

    return (np.sum(input_list)) ** 2 / (len(input_list) * np.sum(input_list**2))


def run_train_RL(gui, N_tl, N_time, ep_num, arrival_distributions):
    """execute the TraCI control loop"""
    scenario = "RL_%3.2f_%s_%s" \
               % (GAMMA, arrival_distributions['we'], arrival_distributions['ns'])
    FIG_DIRECTORY = '/Users/majidraeis/OneDrive/SUMO/figs/' + scenario
    FILE_DIRECTORY = '/Users/majidraeis/OneDrive/SUMO/files/' + scenario
    if not os.path.exists(FIG_DIRECTORY):
        os.makedirs(FIG_DIRECTORY)
    if not os.path.exists(FILE_DIRECTORY):
        os.makedirs(FILE_DIRECTORY)

    hidden = 32
    gamma = 0.99
    epsilon = 0.5
    replay_buffer_size = 10 ** 6
    batch_size = 25
    env = RL_Env(gui, N_tl, N_time, 5, 10, GAMMA, arrival_distributions)
    n_actions = env.action_space.n
    state_space_dim = env.observation_space.shape[0]
    agent = DQNAgent(state_space_dim, n_actions, replay_buffer_size, batch_size,
                     hidden, gamma)
    tot_reward = np.zeros(ep_num)
    waiting_time_max = np.zeros(ep_num)
    waiting_time_quantile = np.zeros(ep_num)
    waiting_time_mean = np.zeros(ep_num)
    for ep in tqdm(range(ep_num)):
        # print('ep=', ep)
        state = env.reset()
        done = False
        epsilon *= 0.99
        epsilon = max(epsilon, 0.01)
        k = 1
        we_th = []
        ns_th = []
        while not done:
            action = agent.get_action(state[0], epsilon)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state[0], action, next_state[0], reward, done)
            we_th.append(env.temp_we_throughput)
            ns_th.append(env.temp_ns_throughput)
            agent.update_network()
            state = next_state
            tot_reward[ep] = reward + gamma * tot_reward[ep]
            k += 1
        close_traci()
        tree = ET.parse('tripinfo.xml')
        root = tree.getroot()
        waiting_times = []
        for child in root:
            waiting_times.append(float(child.attrib['waitingTime']))
        waiting_time_max[ep] = np.max(waiting_times)
        waiting_time_quantile[ep] = np.quantile(waiting_times, 0.95)
        waiting_time_mean[ep] = np.mean(waiting_times)

        if ep % EPISODE_NO_UPDATE_TARGET == 0:
            agent.update_target_network()
    name = '/reward_%d.npy' % seed
    np.save(FILE_DIRECTORY + name, tot_reward)
    name = '/waiting_time_max_%d.npy' %seed
    np.save(FILE_DIRECTORY + name, waiting_time_max)
    name = '/waiting_time_quantile_%d.npy' % seed
    np.save(FILE_DIRECTORY + name, waiting_time_quantile)
    name = '/waiting_time_mean_%d.npy' % seed
    np.save(FILE_DIRECTORY + name, waiting_time_mean)
    name = '/policy_net_%d.pth' % seed
    torch.save(agent.policy_net.state_dict(), FILE_DIRECTORY + name)
    print("====================================")
    print("Model has been saved...")
    print("====================================")

    plt.figure()
    plt.plot(tot_reward)
    plt.xlabel('Episodes')
    plt.ylabel('Total reward')
    plt.savefig(FIG_DIRECTORY + '/Reward_training.pdf', bbox_inches='tight')

    plt.figure()
    plt.plot(waiting_time_mean)
    plt.xlabel('Episodes')
    plt.ylabel('Average waiting time')
    plt.savefig(FIG_DIRECTORY + '/waiting_time_mean_training.pdf', bbox_inches='tight')

    plt.figure()
    plt.plot(waiting_time_max)
    plt.xlabel('Episodes')
    plt.ylabel('Maximum waiting time')
    plt.savefig(FIG_DIRECTORY + '/waiting_time_max_training.pdf', bbox_inches='tight')

    plt.figure()
    plt.plot(waiting_time_quantile)
    plt.xlabel('Episodes')
    plt.ylabel('0.95-Quantile')
    plt.savefig(FIG_DIRECTORY + '/waiting_time_quantile_training.pdf', bbox_inches='tight')



def run_train_RL_Weighted(gui, N_tl, N_time, ep_num, arrival_distributions):
    """execute the TraCI control loop"""
    scenario = "RL_WEIGHTED_%3.2f_%s_%s" \
               % (LAMBDA_FAIR, arrival_distributions['we'], arrival_distributions['ns'])
    FIG_DIRECTORY = '/Users/majidraeis/OneDrive/SUMO/figs/' + scenario
    FILE_DIRECTORY = '/Users/majidraeis/OneDrive/SUMO/files/' + scenario
    if not os.path.exists(FIG_DIRECTORY):
        os.makedirs(FIG_DIRECTORY)
    if not os.path.exists(FILE_DIRECTORY):
        os.makedirs(FILE_DIRECTORY)

    hidden = 32
    gamma = 0.99
    epsilon = 0.5
    replay_buffer_size = 10 ** 6
    batch_size = 25
    env = RL_Env_WeightedFair(gui, N_tl, N_time, 5, 10, GAMMA, arrival_distributions, NS_WEIGHT, WE_WEIGHT)
    n_actions = env.action_space.n
    state_space_dim = env.observation_space.shape[0]
    agent = DQNAgent(state_space_dim, n_actions, replay_buffer_size, batch_size,
                     hidden, gamma)
    tot_reward = np.zeros(ep_num)
    for ep in tqdm(range(ep_num)):
        state = env.reset()

        done = False
        epsilon *= 0.99
        epsilon = max(epsilon, 0.01)
        while not done:
            action = agent.get_action(state[0], epsilon)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state[0], action, next_state[0], reward, done)
            agent.update_network()
            state = next_state
            tot_reward[ep] = reward + gamma * tot_reward[ep]
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


def run_test(env, agent, arrival_distributions, controller_name):
    """execute the TraCI control loop"""
    if controller_name == "RL":
        scenario = "%s_%3.2f_%s_%s" \
                   % (controller_name, GAMMA, arrival_distributions['we'], arrival_distributions['ns'])
    elif controller_name == "RL_WEIGHTED":
        scenario = "%s_%3.2f_%s_%s" \
                   % (controller_name, LAMBDA_FAIR, arrival_distributions['we'], arrival_distributions['ns'])
    else:
        scenario = "%s_%s_%s" \
                   % (controller_name, arrival_distributions['we'], arrival_distributions['ns'])
    FIG_DIRECTORY = '/Users/majidraeis/OneDrive/SUMO/figs/' + scenario
    FILE_DIRECTORY = '/Users/majidraeis/OneDrive/SUMO/files/' + scenario
    if not os.path.exists(FIG_DIRECTORY):
        os.makedirs(FIG_DIRECTORY)
    if not os.path.exists(FILE_DIRECTORY):
        os.makedirs(FILE_DIRECTORY)

    state = env.reset()
    done = False
    sum_wait = [[], [], [], []]
    qls = [[], [], [], []]
    time_interval = []
    we_th = []
    ns_th = []
    fairness_dev = []
    actions = []
    if controller_name in ["RL", "RL_WEIGHTED"]:
        if controller_name == "RL":
            name = '/policy_net_%d.pth' % seed
        else:
            name = '/policy_net.pth'
        agent.policy_net.load_state_dict(torch.load(FILE_DIRECTORY + name))
    while not done:
        if controller_name in ["RL", "RL_WEIGHTED"]:
            temp_state = state.squeeze()
            action = agent.get_action(temp_state, epsilon)
        elif controller_name == "SOTL":
            action = agent.get_action(env.phase)
        elif controller_name == "FixedTime":
            action = FixedTime(env.time, EW_TOT_RATIO, CYCLE_LENGTH)
        elif controller_name == "MaxPressure":
            action = MaxPressure(env.qls)
        actions.append(action)
        next_state, reward, done, info = env.step(action)
        time_interval.append(env.time)
        we_th.append(env.temp_we_throughput)
        ns_th.append(env.temp_ns_throughput)
        fairness_dev.append(env.fair_dev)
        i = 2
        sum_wait[i].append(env.sum_wait[i, 0])
        qls[i].append(env.qls[i])
        state = next_state
    close_traci()
    tree = ET.parse('tripinfo.xml')
    root = tree.getroot()
    waiting_times_ns = []
    waiting_times_we = []
    arrival_times_ns = []
    arrival_times_we = []
    for child in root:
        if child.attrib['id'][:2] in ['we', 'ew']:
            waiting_times_we.append(float(child.attrib['waitingTime']))
            arrival_times_we.append(float(child.attrib['depart']))

        if child.attrib['id'][:2] in ['ns', 'sn']:
            waiting_times_ns.append(float(child.attrib['waitingTime']))
            arrival_times_ns.append(float(child.attrib['depart']))
    np.save(FILE_DIRECTORY+'/waiting_times_ns.npy', waiting_times_ns)
    np.save(FILE_DIRECTORY + '/waiting_times_we.npy', waiting_times_we)
    index_we = np.argsort(arrival_times_we)
    index_ns = np.argsort(arrival_times_ns)
    arrival_times_we = np.array(arrival_times_we)
    arrival_times_ns = np.array(arrival_times_ns)
    waiting_times_we = np.array(waiting_times_we)
    waiting_times_ns = np.array(waiting_times_ns)
    arrival_times_we = arrival_times_we[index_we]
    arrival_times_ns = arrival_times_ns[index_ns]
    waiting_times_we = waiting_times_we[index_we]
    waiting_times_ns = waiting_times_ns[index_ns]
    average_arrival_rate_we = window_average_rate(arrival_times_we, WIN)
    average_arrival_rate_ns = window_average_rate(arrival_times_ns, WIN)

    plt.ioff()
    FONT_SIZE = 13
    FONT_SIZE_LEG = 11
    FIG_SIZE = (5, 3)
    plt.figure(figsize=FIG_SIZE)
    plt.plot(arrival_times_we, average_arrival_rate_we, 'b', label="WE", lw=2)
    plt.plot(arrival_times_ns, average_arrival_rate_ns, 'orange', label="NS", lw=2)
    plt.xlabel("Time (s)", fontsize=FONT_SIZE)
    plt.ylabel("Arrival rate", fontsize=FONT_SIZE)
    plt.xlim(0, 9000)
    plt.locator_params(axis="x", nbins=5)
    plt.legend(loc=3, fontsize=FONT_SIZE_LEG)
    plt.savefig(FIG_DIRECTORY + '/average_arrival_rate.pdf', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=FIG_SIZE)
    plt.plot(time_interval, window_average(time_interval, we_th, WIN), 'b', label="WE", lw=2)
    plt.plot(time_interval, window_average(time_interval, ns_th, WIN), 'orange', label="NS", lw=2)
    plt.xlabel("Time (s)", fontsize=FONT_SIZE)
    plt.ylabel("Throughput", fontsize=FONT_SIZE)
    plt.legend(loc=1, fontsize=FONT_SIZE_LEG)
    plt.xlim(0, 9000)
    plt.ylim(0, 8)
    plt.locator_params(axis="x", nbins=5)
    name = "/throughput_%s.pdf" % controller_name
    plt.savefig(FIG_DIRECTORY + name, bbox_inches='tight')
    plt.close()

    # plt.figure()
    # plt.plot(time_interval, np.array(fairness_dev), 'b')
    # plt.xlabel("Time", fontsize=FONT_SIZE)
    # plt.ylabel("Fairness deviation", fontsize=FONT_SIZE)
    # plt.savefig(FIG_DIRECTORY + '/fairness_dev.pdf', bbox_inches='tight')
    # plt.close()

    # average_phase_ratio = np.array(window_average_phase(actions, WIN))
    # plt.figure()
    # plt.plot(time_interval, average_phase_ratio, label='WE')
    # plt.plot(time_interval, 1 - average_phase_ratio, label='NS')
    # plt.xlabel("Time", fontsize=FONT_SIZE)
    # plt.ylabel("Phase ratio", fontsize=FONT_SIZE)
    # plt.legend()
    # plt.savefig(FIG_DIRECTORY + '/phase_ratio.pdf', bbox_inches='tight')
    # plt.close()

    waiting_times_ave_we = np.array(window_average(arrival_times_we, waiting_times_we, WIN))
    waiting_times_ave_ns = np.array(window_average(arrival_times_ns, waiting_times_ns, WIN))
    plt.figure(figsize=FIG_SIZE)
    plt.plot(arrival_times_we, waiting_times_ave_we, 'b', label='WE', lw=2)
    plt.plot(arrival_times_ns, waiting_times_ave_ns, 'orange', label='NS', lw=2)
    plt.xlabel("Time (s)", fontsize=FONT_SIZE)
    plt.ylabel("Average waiting times", fontsize=FONT_SIZE)
    plt.xlim(0, 9000)
    plt.ylim(0, 50)
    plt.locator_params(axis="x", nbins=5)
    plt.legend(loc=2, fontsize=FONT_SIZE_LEG)
    name = "/average_waiting_time_%s.pdf" % controller_name
    plt.savefig(FIG_DIRECTORY + name, bbox_inches='tight')
    plt.close()

    waiting_times_max_we = np.array(window_max(arrival_times_we, waiting_times_we, WIN))
    waiting_times_max_ns = np.array(window_max(arrival_times_ns, waiting_times_ns, WIN))
    plt.figure(figsize=FIG_SIZE)
    plt.plot(arrival_times_we, waiting_times_max_we, 'b', label='WE', lw=2)
    plt.plot(arrival_times_ns, waiting_times_max_ns, 'orange', label='NS', lw=2)
    plt.xlabel("Time (s)", fontsize=FONT_SIZE)
    plt.ylabel("Maximum waiting times", fontsize=FONT_SIZE)
    plt.xlim(0, 9000)
    plt.locator_params(axis="x", nbins=5)
    plt.legend(loc=3, fontsize=FONT_SIZE_LEG)
    name = "/maximum_waiting_time_%s.pdf" % controller_name
    plt.savefig(FIG_DIRECTORY + name, bbox_inches='tight')
    plt.close()

    waiting_times_total = np.append(waiting_times_we, waiting_times_ns)
    mean_waiting_time = np.mean(waiting_times_total)
    std_waiting_time = np.std(waiting_times_total)
    max_waiting_time = np.max(waiting_times_total)
    quantile_waiting_time = np.quantile(waiting_times_total, 0.95)
    table.write(controller_name.ljust(15))
    table.write("Ave waiting = %3.2f, Max waiting = %3.2f, Jain Index = %3.2f, 0.95 quantile = %3.2f \n \n"
                % (mean_waiting_time, max_waiting_time, jain_index(waiting_times_total), quantile_waiting_time))


def FixedTime(time, ew_tot_ratio, cycle_length):
    if time % cycle_length < ew_tot_ratio * cycle_length:
        return PHASE_WE
    else:
        return PHASE_NS


def MaxPressure(queue_list):

    NS_pressure = queue_list[0] + queue_list[2]
    WE_pressure = queue_list[1] + queue_list[3]
    if WE_pressure > NS_pressure:
        return PHASE_WE
    else:
        return PHASE_NS


class SOTL():
    def __init__(self, theta, omega, mu, rho, env):
        self.theta = theta
        self.omega = omega
        self.mu = mu
        self.rho = rho
        self.kappa = 0
        self.loop_ids_rho = []
        self.loop_ids_omega = []
        self.env = env

    def get_action(self, phase):
        action = phase
        self.loop_ids_rho = []
        self.loop_ids_omega = []
        for edge_index, edge in enumerate(DIR):
            for lane in range(LANE_NUM[edge_index]):
                if edge_index % 2 == phase:
                    loop_id_green = "%s_%d_omega" % (edge, lane)
                    self.loop_ids_omega.append(loop_id_green)
                else:
                    loop_id_red = "%s_%d_rho" % (edge, lane)
                    self.loop_ids_rho.append(loop_id_red)

        self.kappa = sum([self.env.loop_count(loop_id)
                          for loop_id in self.loop_ids_rho])
        cars_approaching_green = sum([self.env.loop_count(loop_id)
                                      for loop_id in self.loop_ids_omega])
        if not (0 < cars_approaching_green < self.mu):
            if self.kappa >= self.theta:
                action = not phase

        return action

    def create_loop_detectors(self):
        with open("data/tandem.add.xml", "w") as additional:
            print("""<additional>""", file=additional)
            for edge_index, edge in enumerate(DIR):
                for lane in range(LANE_NUM[edge_index]):
                    lane_name = "1%s_1_%d" % (edge, lane)
                    loop_id_green = "%s_%d_omega" % (edge, lane)
                    loop_id_red = "%s_%d_rho" % (edge, lane)
                    loop_id = "%s_%d" % (edge, lane)
                    print('    <laneAreaDetector id="%s" lane="%s" pos="%d" freq="1" file="tandem.out" friendlyPos="True"/>' % (
                        loop_id_green, lane_name, -self.omega), file=additional)
                    print('    <laneAreaDetector id="%s" lane="%s" pos="%d" freq="1" file="tandem.out" friendlyPos="True"/>' % (
                        loop_id_red, lane_name, -self.rho), file=additional)
                    print('    <laneAreaDetector id="%s" lane="%s" pos="-40" freq="1" file="tandem.out" friendlyPos="True"/>' % (
                        loop_id, lane_name), file=additional)
            print("""    <tlLogic id="1" type="static" programID="myProgram" offset="0">
            <phase duration="1000" state="rrrrrGGGGggrrrrrGGGGgg"/>
            <phase duration="3"  state="rrrrryyyyyyrrrrryyyyyy"/>
            <phase duration="1000" state="GGGggrrrrrrGGGggrrrrrr"/>
            <phase duration="3"  state="yyyyyrrrrrryyyyyrrrrrr"/>
            </tlLogic>""", file=additional)
            print("</additional>", file=additional)





if __name__ == "__main__":
    # options = get_options()
    gui = 1
    test = 1
    N_tl = 1
    N_time = 10000 #600*15  # number of time steps
    ep_num = 620
    # Poisson, TimeVarying, Markov
    # arrival_distributions = {'ew': 'Poisson', 'we': 'Poisson',
    #                          'ns': 'Markov', 'sn': 'Markov'}
    scenarios = {
                 # 1: {'ew': 'Poisson', 'we': 'Poisson',
                 #     'ns': 'Poisson', 'sn': 'Poisson'},
                 2: {'ew': 'Poisson', 'we': 'Poisson',
                     'ns': 'Markov', 'sn': 'Markov'}
                 # 3: {'ew': 'Poisson', 'we': 'Poisson',
                 #     'ns': 'TimeVarying', 'sn': 'TimeVarying'}
                 }

    # scenarios = {
    #              1: {'ew': 'Poisson', 'we': 'Poisson',
    #                  'ns': 'Poisson', 'sn': 'Poisson'},
    #              2: {'ew': 'Markov', 'we': 'Markov',
    #                  'ns': 'Markov', 'sn': 'Markov'}
    #              3: {'ew': 'Poisson', 'we': 'Poisson',
    #                  'ns': 'TimeVarying', 'sn': 'TimeVarying'}
    #              }

    if test:
        table = open("Table.txt", "w")
        hidden = 32
        gamma = 0.99
        epsilon = 0.5
        replay_buffer_size = 10 ** 6
        batch_size = 25
        for i, arrival_distributions in enumerate(scenarios.values()):
            # table.write("=======================  %s - %s  =========================\n"
            #             % (scenarios[i+1]['we'], scenarios[i+1]['ns']))
            # ======================= RL ==============================
            controller_name = "RL"
            env = RL_Env(gui, N_tl, N_time, 5, 10, GAMMA, arrival_distributions)
            n_actions = env.action_space.n
            state_space_dim = env.observation_space.shape[0]
            agent = DQNAgent(state_space_dim, n_actions, replay_buffer_size, batch_size,
                             hidden, gamma)
            run_test(env, agent, arrival_distributions, controller_name)
            # =========================================================
            # ======================= Wighted RL ==============================
            # controller_name = "RL_WEIGHTED"
            # env = RL_Env_WeightedFair(gui, N_tl, N_time, 5, 10, GAMMA,
            #                           arrival_distributions, NS_WEIGHT, WE_WEIGHT)
            # n_actions = env.action_space.n
            # state_space_dim = env.observation_space.shape[0]
            # agent = DQNAgent(state_space_dim, n_actions, replay_buffer_size, batch_size,
            #                  hidden, gamma)
            # run_test(env, agent, arrival_distributions, controller_name)
            # =========================================================
            # ======================= SOTL ==============================
            controller_name = "SOTL"
            theta = 5
            omega = 25
            rho = 25
            mu = 3
            env = RL_Env(gui, N_tl, N_time, 1, 10, GAMMA, arrival_distributions)
            agent = SOTL(theta, omega, mu, rho, env)
            agent.create_loop_detectors()
            run_test(env, agent, arrival_distributions, controller_name)
            # =========================================================
            # ======================= Max Pressure ==============================
            controller_name = "MaxPressure"
            env = RL_Env(gui, N_tl, N_time, 1, 10, GAMMA, arrival_distributions)
            run_test(env, agent, arrival_distributions, controller_name)
            # =========================================================
            # ======================= Fixed Time ==============================
            controller_name = "FixedTime"
            env = RL_Env(gui, N_tl, N_time, 1, 10, GAMMA, arrival_distributions)
            run_test(env, agent, arrival_distributions, controller_name)
            # =========================================================

            # run_test_RL(N_tl, N_time, arrival_distributions)
            # run_test_RL_Weighted(N_tl, N_time, arrival_distributions)
            # run_test_fixed_time(N_tl, N_time, arrival_distributions)
            # run_test_max_pressure(N_tl, N_time, arrival_distributions)
            # run_test_sotl(N_tl, N_time, arrival_distributions)
            table.write("\n")
        table.close()

    else:

        # for arrival_distributions in scenarios.values():
        arrival_distributions = scenarios[2]
        run_train_RL(gui, N_tl, N_time, ep_num, arrival_distributions)
        # run_test_RL(N_tl, N_time, arrival_distributions)
        # run_train_RL_Weighted(gui, N_tl, N_time, ep_num, arrival_distributions)
        # run_test_RL_Weighted(N_tl, N_time, arrival_distributions)
        # run_test_sotl(N_tl, N_time, arrival_distributions)