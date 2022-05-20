from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import random
import gym
import copy
from gym import spaces
import numpy as np
# we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import traci  # noqa

PHASE_NS = 0
PHASE_WE = 1
QL_MAX = 40
DIR_IN_PHASE = [['u', 'd'], ['l', 'r']]
DIR = ['u', 'r', 'd', 'l']
LOOP_NUM_EDGE = 3
X_NUM_LANE = 3
Y_NUM_LANE = 2



def generate_node(N_tl):
    x_dist = 250.0
    y_dist = 200.0
    with open("data/tandem.nod.xml", "w") as nodes:

        print("""<nodes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
        xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/nodes_file.xsd">
""", file=nodes)
        for i in range(1, N_tl+1):
            print('     <node id="%i" x="%3.2f" y="0"  type="traffic_light"/>' % (
                i, i*x_dist), file=nodes)
            print('     <node id="%iu" x="%3.2f" y="%3.2f"  type="priority"/>' % (
                i, i*x_dist, y_dist), file=nodes)
            print('     <node id="%id" x="%3.2f" y="%3.2f"  type="priority"/>' % (
                i, i*x_dist, -y_dist), file=nodes)
        print('     <node id="1l" x="0" y="0"  type="priority"/>', file=nodes)
        print('     <node id="%ir" x="%3.2f" y="0"  type="priority"/>' % (
            i, (i+1) * x_dist), file=nodes)

        print("</nodes>", file=nodes)


def generate_edge(N_tl):
    max_speed_we = 13.89 #(m/s) 50km/h
    max_speed_ew = 13.89 #(m/s) 50km/h
    max_speed_ns = 8.33 #(m/s) 30km/h
    max_speed_sn = 8.33 #(m/s) 30km/h
    N_lanes_we = X_NUM_LANE
    N_lanes_ew = X_NUM_LANE
    N_lanes_ns = Y_NUM_LANE
    N_lanes_sn = Y_NUM_LANE
    high_priority = 78
    low_priority = 46

    with open("data/tandem.edg.xml", "w") as edg:

        print("""<edges xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/edges_file.xsd">
""", file=edg)
        for i in range(1, N_tl+1):
            if i < N_tl:
                # WE
                print('     <edge id="%i_%i" from="%i" to="%i" priority="%i" numLanes="%i" speed="%3.2f" />' % (
                    i, i+1, i, i+1, high_priority, N_lanes_we, max_speed_we), file=edg)
                # EW
                print('     <edge id="%i_%i" from="%i" to="%i" priority="%i" numLanes="%i" speed="%3.2f" />' % (
                    i+1, i, i+1, i, high_priority, N_lanes_ew, max_speed_ew), file=edg)
            # up_NS
            print('     <edge id="%iu_%i" from="%iu" to="%i" priority="%i" numLanes="%i" speed="%3.2f" />' % (
                i, i, i, i, low_priority, N_lanes_ns, max_speed_ns), file=edg)
            # up_SN
            print('     <edge id="%i_%iu" from="%i" to="%iu" priority="%i" numLanes="%i" speed="%3.2f" />' % (
                i, i, i, i, low_priority, N_lanes_sn, max_speed_sn), file=edg)
            # down_NS
            print('     <edge id="%i_%id" from="%i" to="%id" priority="%i" numLanes="%i" speed="%3.2f" />' % (
                i, i, i, i, low_priority, N_lanes_ns, max_speed_ns), file=edg)
            # down_SN
            print('     <edge id="%id_%i" from="%id" to="%i" priority="%i" numLanes="%i" speed="%3.2f" />' % (
                i, i, i, i, low_priority, N_lanes_sn, max_speed_sn), file=edg)

        # 1_l to/from 1
        print('     <edge id="1l_1" from="1l" to="1" priority="%i" numLanes="%i" speed="%3.2f" />' % (
            high_priority, N_lanes_we, max_speed_we), file=edg)
        print('     <edge id="1_1l" from="1" to="1l" priority="%i" numLanes="%i" speed="%3.2f" />' % (
            high_priority, N_lanes_ew, max_speed_ew), file=edg)
        # N_tl_r to/from N_tl
        print('     <edge id="%i_%ir" from="%i" to="%ir" priority="%i" numLanes="%i" speed="%3.2f" />' % (
            i, i, i, i, high_priority, N_lanes_we, max_speed_we), file=edg)
        print('     <edge id="%ir_%i" from="%ir" to="%i" priority="%i" numLanes="%i" speed="%3.2f" />' % (
            i, i, i, i, high_priority, N_lanes_ew, max_speed_ew), file=edg)

        print("</edges>", file=edg)


def generate_con(N_tl):
        # N_tl = 5  # number of traffic lights
        with open("data/tandem.con.xml", "w") as con:
            print("""<connections xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
             xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/connections_file.xsd">""", file=con)
            for i in range(1, N_tl+1):
                if i < N_tl-1:
                    # WE
                    print('     <connection from="%i_%i" to="%i_%i"/>' % (
                            i, i+1, i+1, i+2), file=con)
                    #WE (Right turn)
                    # print('     <connection from="<FROM_EDGE_ID>" to="<T0_EDGE_ID>" fromLane="<INT_1>"'
                    #       ' toLane="<INT_2>"/>' % (i, i + 1, i + 1, i + 2), file=con)

                    # EW
                    print('     <connection from="%i_%i" to="%i_%i"/>' % (
                            i+2, i+1, i+1, i), file=con)
                # NS
                print('     <connection from="%iu_%i" to="%i_%id"/>' % (
                    i, i, i, i), file=con)
                # SN
                print('     <connection from="%id_%i" to="%i_%iu"/>' % (
                    i, i, i, i), file=con)
            # WE
            print('     <connection from="1l_1" to="1_2"/>', file=con)
            print('     <connection from="%i_%i" to="%i_%ir"/>' % (N_tl-1, N_tl, N_tl, N_tl), file=con)
            # EW
            print('     <connection from="2_1" to="1_1l"/>', file=con)
            print('     <connection from="%ir_%i" to="%i_%i"/>' % (N_tl, N_tl, N_tl, N_tl-1), file=con)

            print("</connections>", file=con)



def generate_routefile(N_tl, N_time, arrival_distributions):
    random.seed(42)  # make tests reproducible
    # demand per second from different directions
    # t_step = 15*60
    # we_rate_list = np.array([0.05, 0.1, 0.15, 0.3, 0.5, 0.55, 0.4, 0.25, 0.1, 0.02])
    # ew_rate_list = np.array([0.05, 0.1, 0.15, 0.3, 0.5, 0.55, 0.4, 0.25, 0.1, 0.02])
    # ns_rate_list = np.array([0.2, 0.25, 0.27, 0.27, 0.3, 0.35, 0.25, 0.2, 0.18, 0.15])
    # sn_rate_list = np.array([0.2, 0.25, 0.27, 0.27, 0.3, 0.35, 0.25, 0.2, 0.18, 0.15])
    # we_rate_list = we_rate_list / 3
    # ew_rate_list = ew_rate_list / 3
    # ns_rate_list = ns_rate_list / 3
    # sn_rate_list = sn_rate_list / 3

    PERIOD = 50*10 # period equals PERIOD*off_on_ratio
    P_MAX = 1/4
    P_MIN = 1/10

    p_ns_on_off = 0.28
    p_ns_off_on = 0.02
    p_sn_on_off = 0.28
    p_sn_off_on = 0.02

    p_we_on_off = 0.24
    p_we_off_on = 0.06
    p_ew_on_off = 0.24
    p_ew_off_on = 0.06

    p_we = 1/5 #1/5
    p_ew = 1/5
    p_ns = 1/15
    p_sn = 1/15

    vehNr = 0
    s_we = 0
    s_ew = 0
    s_ns = 0
    s_sn = 0
    # BURST_LEN = 2
    # inter_arrival = 40

    with open("data/tandem.rou.xml", "w") as routes:
        print("""<routes>
        <vType id="car" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.67" \
        guiShape="passenger"/>
        <vType id="bus" accel="0.8" decel="4.5" sigma="0.5" length="7" minGap="3" maxSpeed="25" \
        guiShape="bus"/>""", file=routes)

        # WE
        route_we = "1l_1 "
        # EW
        route_ew = "%ir_%i " % (N_tl, N_tl)
        for j in range(1, N_tl + 1):
            if j < N_tl:
                # WE
                route_we += "%i_%i " % (j, j+1)
                # EW
                route_ew += "%i_%i " % (N_tl-j+1, N_tl-j)
            # NS
            print('    <route edges = "%iu_%i %i_%id" color = "yellow" id = "route_ns%i" />' % (j, j, j, j, j),
                  file=routes)
            # SN
            print('    <route edges = "%id_%i %i_%iu" color = "yellow" id = "route_sn%i" />' % (j, j, j, j, j),
                  file=routes)
        # WE
        route_we += "%i_%ir " % (N_tl, N_tl)
        print('    <route edges = "%s" color = "red" id = "route_we" />' % route_we,
              file=routes)
        # EW
        route_ew += "1_1l "
        print('    <route edges = "%s" color = "red" id = "route_ew" />' % route_ew,
              file=routes)

        for i in range(N_time):
            # -------------------West-East------------------------
            if arrival_distributions['we'] == "Poisson":
                vehNr = arrival_generate_poisson("we", i, p_we, vehNr, 0, routes)

            elif arrival_distributions['we'] == "Markov":
                vehNr, s_we = arrival_generate_markov("we", s_we, p_we_on_off, p_we_off_on, i, vehNr, 0, routes)

            elif arrival_distributions['we'] == "TimeVarying":
                vehNr = arrival_generate_time_varying_poisson("we", i, PERIOD, P_MIN, P_MAX, vehNr, 0, routes)
                # vehNr = arrival_generate_time_varying_poisson("we", i, t_step, we_rate_list, vehNr, 0,
                #                                       routes)
            # -------------------East-West------------------------
            if arrival_distributions['ew'] == "Poisson":
                vehNr = arrival_generate_poisson("ew", i, p_ew, vehNr, 0, routes)

            elif arrival_distributions['ew'] == "Markov":
                vehNr, s_ew = arrival_generate_markov("ew", s_ew, p_ew_on_off, p_ew_off_on, i, vehNr, 0, routes)

            elif arrival_distributions['ew'] == "TimeVarying":
                vehNr = arrival_generate_time_varying_poisson("ew", i, PERIOD, P_MIN, P_MAX, vehNr, 0, routes)
                # vehNr = arrival_generate_time_varying_poisson("ew", i, t_step, ew_rate_list, vehNr, 0,
                #                                               routes)

            for j in range(1, N_tl+1):
                # -------------------North-South------------------------
                if arrival_distributions['ns'] == "Poisson":
                    vehNr = arrival_generate_poisson("ns", i, p_ns, vehNr, j, routes)

                elif arrival_distributions['ns'] == "Markov":
                    vehNr, s_ns = arrival_generate_markov("ns", s_ns, p_ns_on_off, p_ns_off_on, i, vehNr, j, routes)

                elif arrival_distributions['ns'] == "TimeVarying":
                    vehNr = arrival_generate_time_varying_poisson("ns", i, PERIOD, P_MIN, P_MAX, vehNr, j, routes)
                    # vehNr = arrival_generate_time_varying_poisson("ns", i, t_step, ns_rate_list, vehNr, j,
                    #                                               routes)
                # -------------------South-North------------------------
                if arrival_distributions['sn'] == "Poisson":
                    vehNr = arrival_generate_poisson("sn", i, p_sn, vehNr, j, routes)

                elif arrival_distributions['sn'] == "Markov":
                    vehNr, s_sn = arrival_generate_markov("sn", s_sn, p_sn_on_off, p_sn_off_on, i, vehNr, j, routes)

                elif arrival_distributions['sn'] == "TimeVarying":
                    vehNr = arrival_generate_time_varying_poisson("sn", i, PERIOD, P_MIN, P_MAX, vehNr, j, routes)
                    # vehNr = arrival_generate_time_varying_poisson("sn", i, t_step, sn_rate_list, vehNr, j,
                    #                                               routes)
        print("</routes>", file=routes)


def arrival_generate_poisson(direction, time, prob, veh_nr, tl_nr, routes_file):
    if random.uniform(0, 1) < prob:
        if direction in ['ew', 'we']:
            print('    <vehicle id="%s_%i" type="car" route="route_%s" depart="%i" departLane="free" '
                  'departSpeed="speedLimit" />' % (direction, veh_nr, direction, time), file=routes_file)
        else:
            print('    <vehicle id="%s%i_%i" type="car" route="route_%s%i" depart="%i" departLane="free" />'
                  % (direction, tl_nr, veh_nr, direction, tl_nr, time), file=routes_file)
        return veh_nr + 1
    return veh_nr


def arrival_generate_time_varying_poisson(direction, time, period, p_min, p_max, veh_nr, tl_nr, routes_file):
    temp = arrival_generate_poisson(direction, time, get_prob(time, period, p_min, p_max), veh_nr, tl_nr, routes_file)
    return temp


def arrival_generate_markov(direction, state, p_on_off, p_off_on, time, veh_nr, tl_nr, routes_file):

    if state:
        if direction in ['ew', 'we']:
            print('    <vehicle id="%s_%i" type="car" route="route_%s" depart="%i" departLane="free" '
                  'departSpeed="speedLimit" />' % (direction, veh_nr, direction, time), file=routes_file)
        else:
            print('    <vehicle id="%s%i_%i" type="car" route="route_%s%i" depart="%i" departLane="free" />'
                  % (direction, tl_nr, veh_nr, direction, tl_nr, time), file=routes_file)

        next_state = 0 if random.uniform(0, 1) < p_on_off else 1
        return veh_nr + 1, next_state
    else:
        next_state = 1 if random.uniform(0, 1) < p_off_on else 0
        return veh_nr, next_state


def get_prob(time, period, p_min, p_max):
    #  rate is in veh/s
    off_on_ratio = 4
    return p_max if not (time//period) % off_on_ratio else p_min


# def get_prob(time, t_step, rate_list):
#     #  rate is in veh/s
#     return rate_list[time//t_step]


def close_traci():
    traci.close()
    sys.stdout.flush()


class RL_Env(gym.Env):

    def __init__(self, gui, N_tl, N_time, slot, min_time, gamma, arrival_distributions):
        #         self.__version__ = "0.1.0"
        # General variables defining the environment
        self.gamma = gamma
        self.N_tl = N_tl
        self.state_size = 9
        self.gui = gui
        self.slot = slot
        self.min_time = min_time
        self.action_space = spaces.Discrete(2)
        # self.B_max = 8
        self.lam = 1
        self.time = 0
        self.phase = 0
        self.ids = [[], [], [], []]
        self.old_ids = [[], [], [], []]
        self.flag_phase_change = 0
        state_min = np.zeros(self.state_size)
        state_max = np.append(100 * np.ones(8), 1)
        self.observation_space = spaces.Box(state_min, state_max)
        self.MAX_STEPS = N_time
        generate_node(self.N_tl)
        generate_edge(self.N_tl)
        os.system("netconvert data/tandem.netccfg")
        generate_routefile(self.N_tl, self.MAX_STEPS, arrival_distributions)
        # this script has been called from the command line. It will start sumo as a
        # server, then connect and run
        self.qls = np.zeros(4)
        self.ql_prev_slot = np.zeros(4)
        self.wait_times = {0: [], 1: [], 2: [], 3: []}
        self.sum_wait = np.zeros((4, 2))
        self.loop_counts = np.zeros((4, LOOP_NUM_EDGE))
        self.loop_speeds = np.zeros((4, LOOP_NUM_EDGE))
        self.tot_ns_throughput = 0
        self.tot_we_throughput = 0
        self.temp_ns_throughput = 0
        self.temp_we_throughput = 0
        self.fair_dev = 0


    def step(self, action):

        self._take_action(action)
        self.time = traci.simulation.getTime()
        for i, j in enumerate(DIR):
            edge = "1%s_1" % j
            ql_old = 0 if self.flag_phase_change else self.qls[i]
            self.qls[i] = traci.edge.getLastStepHaltingNumber(edge)
            delta_ql = self.qls[i] - ql_old
            self._calculate_delays_per_edge(action, delta_ql, i)
            if j in DIR_IN_PHASE[action]:
                self.sum_wait[i] = np.zeros(2)
            else:
                new_sum_wait = 2 * self.sum_wait[i, 0] - self.sum_wait[i, 1] + delta_ql
                self.sum_wait[i, 1] = self.sum_wait[i, 0]
                self.sum_wait[i, 0] = new_sum_wait
        state = np.append(self.qls, self.sum_wait[:, 0])
        state = np.append(state, self.phase)
        state = np.reshape(state, (1, -1))
        reward = self._get_reward(action)
        self.old_ids = copy.copy(self.ids)
        done = True
        if self.time < self.MAX_STEPS and np.sum(self.qls < QL_MAX) > 3:
            done = False
        return state, reward, done, {}


    def _calculate_delays_per_edge(self, action, delta_ql, edge_index):
        if DIR[edge_index] in DIR_IN_PHASE[action]:
            self.wait_times[edge_index] = []
        else:
            if delta_ql > 0:
                self.wait_times[edge_index] = np.append(self.wait_times[edge_index],
                                                        np.zeros(int(delta_ql)))
            if self.qls[edge_index] > 0:
                self.wait_times[edge_index] += 1


    def _take_action(self, action):
        time_inc = self.slot
        self.flag_phase_change = action != self.phase
        self.phase = action
        if action == PHASE_NS:
            tl_phase = 2
            if self.flag_phase_change:
                tl_phase = 1
                time_inc = self.min_time
        else:
            tl_phase = 0
            if self.flag_phase_change:
                tl_phase = 3
                time_inc = self.min_time
        traci.trafficlight.setPhase('1', tl_phase)
        # reward = 0
        for _ in range(time_inc):
            traci.simulationStep()
        for i, j in enumerate(DIR):
            edge = "1%s_1" % j
            self.ids[i] = traci.edge.getLastStepVehicleIDs(edge)

        throughputs = np.array(self._cal_throughput(self.ids, self.old_ids))
        self.temp_ns_throughput = sum(throughputs[[0, 2]])
        self.temp_we_throughput = sum(throughputs[[1, 3]])
        backlog = []
        for j in DIR:
            edge = "1%s_1" % j
            backlog.append(self.loop_count_cal(edge) > 0)

        backlog = np.array(backlog)
        backlog_flag = np.sum(backlog[[0, 2]]) > 0 and np.sum(backlog[[1, 3]]) > 0
        # print(backlog_flag)
        if backlog_flag:
            self.tot_ns_throughput += self.temp_ns_throughput
            self.tot_we_throughput += self.temp_we_throughput
            self.fair_dev += self.tot_ns_throughput - self.tot_we_throughput


    def _cal_throughput(self, ids, old_ids):
        passed_veh_ids = [[], [], [], []]
        passed_veh_nums = []
        for dir_index in range(len(ids)):
            passed_veh_nums.append(0)
            for i in old_ids[dir_index]:
                if i not in ids[dir_index]:
                    passed_veh_ids[dir_index].append(i)
                    passed_veh_nums[dir_index] += 1
        return passed_veh_nums

    def loop_count_cal(self, edge):

        loop_count_per_edge = 0
        direction = edge[1]
        num_lines = {"u": Y_NUM_LANE, "r": X_NUM_LANE, "d": Y_NUM_LANE, "l": X_NUM_LANE}
        for line in range(num_lines[direction]):
            loop_id = "%s_%i" % (direction, line)
            loop_count_per_edge += traci.lanearea.getLastStepVehicleNumber(loop_id)
        return loop_count_per_edge

    def _get_reward(self, action):
        cost = 0
        for edge in range(len(DIR)):
            for d in self.wait_times[edge]:
                cost += 1 + self.gamma * (2 * d - 1)
                # cost += 1 + self.priority[edge] * (2 * d - 1)
        return - cost

    def loop_count(self, loop_id):
        return traci.lanearea.getLastStepVehicleNumber(loop_id)

    def reset(self):
        # options.nogui = options
        if not self.gui:
            sumoBinary = checkBinary('sumo')
        else:
            sumoBinary = checkBinary('sumo-gui')

        traci.start([sumoBinary, "-c", "data/tandem.sumocfg",
                     "--tripinfo-output", "tripinfo.xml", "--duration-log.disable", "true",
                     "--no-step-log", "true", "--time-to-teleport", "-2"])
        self.qls = np.zeros(4)
        self.ql_prev_slot = np.zeros(4)
        self.wait_times = {0: [], 1: [], 2: [], 3: []}
        self.ids = [[], [], [], []]
        self.old_ids = [[], [], [], []]
        self.phase = 0
        self.flag_phase_change = 0
        self.time = 0
        self.sum_wait = np.zeros((4, 2))
        self.tot_ns_throughput = 0
        self.tot_we_throughput = 0
        self.fair_dev = 0
        state = np.zeros((1, self.state_size))
        return state


class RL_Env_WeightedFair(RL_Env):

    def __init__(self, gui, N_tl, N_time, slot, min_time, gamma, arrival_distributions, ns_weight, we_weight):
        super().__init__(gui, N_tl, N_time, slot, min_time, gamma, arrival_distributions)
        self.ns_weight = ns_weight
        self.we_weight = we_weight
        self.state_size = 6
        state_min = np.append(np.zeros(5), -100)
        state_max = np.append(100 * np.ones(4), [1, 100])
        self.observation_space = spaces.Box(state_min, state_max)


    def step(self, action):

        self._take_action(action)
        self.time = traci.simulation.getTime()
        veh_num = []
        for i, j in enumerate(DIR):
            edge = "1%s_1" % j
            self.qls[i] = traci.edge.getLastStepHaltingNumber(edge)
            veh_num.append(self.loop_count_cal(edge))
        state = np.append(veh_num, self.phase)
        state = np.append(state, self.fair_dev)
        state = np.reshape(state, (1, -1))
        reward = self._get_reward(action)
        self.old_ids = copy.copy(self.ids)
        done = True
        if self.time < self.MAX_STEPS and np.sum(self.qls < QL_MAX) > 3:
            done = False
        return state, reward, done, {}

    def _take_action(self, action):
        time_inc = self.slot
        self.flag_phase_change = action != self.phase
        self.phase = action
        if action == PHASE_NS:
            tl_phase = 2
            if self.flag_phase_change:
                tl_phase = 1
                time_inc = self.min_time
        else:
            tl_phase = 0
            if self.flag_phase_change:
                tl_phase = 3
                time_inc = self.min_time

        traci.trafficlight.setPhase('1', tl_phase)
        for _ in range(time_inc):
            traci.simulationStep()
        for i, j in enumerate(DIR):
            edge = "1%s_1" % j
            self.ids[i] = traci.edge.getLastStepVehicleIDs(edge)

    def _get_reward(self, action):
        throughputs = np.array(self._cal_throughput(self.ids, self.old_ids))
        self.temp_ns_throughput = sum(throughputs[[0, 2]])
        self.temp_we_throughput = sum(throughputs[[1, 3]])
        backlog = []
        for j in DIR:
            edge = "1%s_1" % j
            backlog.append(self.loop_count_cal(edge) > 0)

        backlog = np.array(backlog)
        backlog_flag = np.sum(backlog[[0, 2]]) > 0 and np.sum(backlog[[1, 3]]) > 0
        reward = - np.sum(self.qls)
        if backlog_flag:

            self.tot_ns_throughput += self.temp_ns_throughput
            self.tot_we_throughput += self.temp_we_throughput
            self.fair_dev += self.temp_ns_throughput / self.ns_weight - self.temp_we_throughput / self.we_weight
            reward = - np.sum(self.qls) - abs(self.fair_dev) * backlog_flag
        return reward
